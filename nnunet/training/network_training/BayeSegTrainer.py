from typing import OrderedDict
import torch
import torch.nn as nn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader2D, load_dataset
from nnunet.utilities.tensor_utilities import sum_tensor
import torch.nn.functional as F
import math
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.cuda.amp import autocast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
import numpy as np
import cv2


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
                    

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def default_conv(in_channels, out_channels, kernel_size, strides=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, strides,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv=default_conv, n_feats=64, kernel_size=3,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

    
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
        
        
class ResNet(nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=10,
                 bn = False):
        super(ResNet, self).__init__()
        if num_in_ch != 1:
            self.pseudo_3d = True
        else:
            self.pseudo_3d = False
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResBlock, num_block, n_feats=num_feat, bn = bn)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_last], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        out = self.conv_last(self.lrelu(out))
        if self.pseudo_3d:
            out += x[:, self.pseudo_3d//2].unsqueeze(1)
        else:
            out += x
        return out

def sample_normal_jit(mu, log_var):
    sigma = torch.exp(log_var / 2)
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


class BayeSeg(SegmentationNetwork):
    def __init__(self, args, unet, freeze_whst=False):
        super(BayeSeg,self).__init__()
        
        self.args = args

        if freeze_whst:
            for p in self.parameters():
                p.requires_grad_(False)
                
        self.num_classes = args.num_classes
        self.conv_op = nn.Conv2d
        
        # reconstruct clean image x and infer noise
        self.res_clean = ResNet(num_in_ch=self.args.pseudo_3d_slices, num_out_ch = 2)
        self.res_noise = ResNet(num_in_ch=self.args.pseudo_3d_slices, num_out_ch = 2, num_block=6, bn=True)  # 推断噪声的话，也许加BN的效果会更好一些？
        # pred mu and log var unit for seg_masks: B x K x W x H
        self.unet = unet
        self.input_shape_must_be_divisible_by = self.unet.input_shape_must_be_divisible_by
        
        # postprecess
        self.softmax = nn.Softmax(dim=1)
        
        # TODO: modify Dx & Dz
        Dx = torch.zeros([1,1,3,3],dtype=torch.float)
        Dx[:,:,1,1] = 1
        Dx[:,:,1,2] = -1
        self.Dx = nn.Parameter(data=Dx, requires_grad=False)
        Dy = torch.zeros([1,1,3,3],dtype=torch.float)
        Dy[:,:,1,1] = 1
        Dy[:,:,2,1] = -1
        self.Dy = nn.Parameter(data=Dy, requires_grad=False)
        
    def generate_m(self, samples):
        # m : mean of noise
        feature = self.res_noise(samples)
        mu_m, log_var_m = torch.chunk(feature, 2, dim=1)
        log_var_m = torch.clamp(log_var_m, -20, 0)
        m, _ = sample_normal_jit(mu_m, log_var_m)
        return m, mu_m, log_var_m
    
    def generate_x(self, samples):
        # x : clean image
        feature = self.res_clean(samples)
        mu_x, log_var_x = torch.chunk(feature, 2, dim=1)
        log_var_x = torch.clamp(log_var_x, -20, 0)
        x, _ = sample_normal_jit(mu_x, log_var_x)
        return x, mu_x, log_var_x
    
    def generate_z(self, x):
        # z : Seg logit
        feature = self.unet(x)
        mu_z, log_var_z = torch.chunk(feature, 2, dim=1)
        log_var_z = torch.clamp(log_var_z, -20, 0)
        z, _ = sample_normal_jit(mu_z, log_var_z)
        return self.softmax(z), self.softmax(mu_z), log_var_z

    def forward(self, samples: torch.Tensor):
        x, mu_x, log_var_x = self.generate_x(samples)
        m, mu_m, log_var_m = self.generate_m(samples)
        z, mu_z, log_var_z = self.generate_z(x)

        K = self.num_classes
        
        # compute VB params
        ###################################
        # noise std rho
        if self.args.pseudo_3d_slices != 1:
            samples = samples[:, self.args.pseudo_3d_slices//2].unsqueeze(1)

        if self.args.clip_x:
            residual = samples - (torch.clip(x, -0.1, 0.1) + m)
        else:
            residual = samples - (x + m)

        mu_rho_hat = (2*self.args.gamma_rho + 1) / (torch.clip(residual, -0.01, 0.01) * torch.clip(residual, -0.01, 0.01) + 2*self.args.phi_rho)
        normalization = torch.sum(mu_rho_hat).detach()
        n, _ = sample_normal_jit(m, torch.log(1 / mu_rho_hat))
        
        # Image line upsilon
        alpha_upsilon_hat = 2*self.args.gamma_upsilon + K
        difference_x_x = F.conv2d(mu_x, self.Dx, padding=1)
        difference_x_y = F.conv2d(mu_x, self.Dy, padding=1)
        difference_x_2 = difference_x_x**2 + difference_x_y**2
        beta_upsilon_hat = torch.sum(mu_z*(difference_x_2 + 4*torch.exp(log_var_x)),
                                     dim = 1, keepdim = True) + 2*self.args.phi_upsilon  # B x 1 x W x H
        mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
       
        # Seg boundary omega
        difference_z_x = F.conv2d(mu_z, self.Dx.expand(K,1,3,3), padding=1, groups=K)  # B x K x W x H
        difference_z_y = F.conv2d(mu_z, self.Dy.expand(K,1,3,3), padding=1, groups=K)  # B x K x W x H
        difference_z_2 = difference_z_x**2 + difference_z_y**2
        alpha_omega_hat = 2*self.args.gamma_omega + 1
        pseudo_pi = torch.mean(mu_z, dim=(2,3), keepdim=True)
        beta_omega_hat = pseudo_pi*(difference_z_2 + 4*torch.exp(log_var_z)) + 2*self.args.phi_omega
        mu_omega_hat = alpha_omega_hat / beta_omega_hat
 
        # Seg category probability pi
        _, _, W, H = samples.shape
        alpha_pi_hat = self.args.alpha_pi + W*H/2
        beta_pi_hat = torch.sum(mu_omega_hat*(difference_z_2 + 4*torch.exp(log_var_z)), dim=(2,3), keepdim=True)/2 + self.args.beta_pi
        digamma_pi = torch.special.digamma(alpha_pi_hat + beta_pi_hat) - torch.special.digamma(beta_pi_hat)
        
        # compute loss-related
        kl_y = residual*mu_rho_hat.detach()*residual

        kl_mu_z = torch.sum(digamma_pi.detach()*difference_z_2*mu_omega_hat.detach(), dim=1)
        kl_sigma_z = torch.sum(digamma_pi.detach()*(4*torch.exp(log_var_z)*mu_omega_hat.detach() - log_var_z), dim=1)
        
        kl_mu_x = torch.sum(difference_x_2*mu_upsilon_hat.detach()*mu_z.detach(), dim=1)
        kl_sigma_x = torch.sum(4*torch.exp(log_var_x)*mu_upsilon_hat.detach()*mu_z.detach(), dim=1) - log_var_x
        
        kl_mu_m = self.args.sigma_0*mu_m*mu_m
        kl_sigma_m = self.args.sigma_0*torch.exp(log_var_m) - log_var_m

        visualize = {
                     'recon':torch.concat([x, mu_x, torch.exp(log_var_x/2)]),
                     'noise':torch.concat([n, m, 1/mu_rho_hat.sqrt()]),
                     'logit':torch.concat([z[:,2:3,...], mu_z[:,2:3,...], torch.exp(log_var_z/2)[:,2:3,...]]),
                     'lines':mu_upsilon_hat, 'contour': mu_omega_hat[:,2:3,...],
                    }

        # visualize = {'y': samples, 'n': n, 'm': m, 'rho': mu_rho_hat, 'x': x, 'upsilon': mu_upsilon_hat, 'z': z, 'omega': mu_omega_hat}
        pred = z if self.training else mu_z
        out = {
               'pred_masks': pred, 'kl_y':kl_y,
               'kl_mu_z':kl_mu_z, 'kl_sigma_z':kl_sigma_z,
               'kl_mu_x':kl_mu_x, 'kl_sigma_x':kl_sigma_x,
               'kl_mu_m':kl_mu_m, 'kl_sigma_m':kl_sigma_m,
               'normalization': normalization,
               'rho':mu_rho_hat,
               'omega':mu_omega_hat*digamma_pi,
               'upsilon':mu_upsilon_hat*mu_z,
               'visualize':visualize,
              }
        
        # self.save_image(mu_omega_hat, 'contour')
        # self.save_image(mu_upsilon_hat, 'line')
        # self.save_image(samples, 'input')

        return out
    
    def save_image(self, image, tag):
        if type(image) == np.ndarray:
            image = image[0]
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            visual = 255 * image
        else:
            if image.requires_grad:
                image = image.detach().cpu().numpy()
            else:
                image = image.cpu().numpy()

            if image.shape[1] != 1:
                ncol = 2
                nrow = image.shape[1] // ncol
                height, width = image.shape[2:]
                visual = np.zeros((nrow * height, ncol * width))
                for i in range(nrow):
                    for j in range(ncol):
                        ub, db = i * height, (i+1) * height
                        lb, rb = j * width, (j+1) * width
                        part_image = image[0, ncol*i+j, ...]
                        part_image = (part_image - part_image.min()) / (part_image.max() - part_image.min() + 1e-6)
                        visual[ub: db, lb: rb] = 255 * part_image
            else:
                image = image[0, 0, ...]
                image = (image - image.min()) / (image.max() - image.min() + 1e-6)
                visual = 255 * image

        cv2.imwrite(tag + '.png', visual)
        print('yes')
    
    def _internal_maybe_mirror_and_pred_2D(self, x, mirror_axes, do_mirroring=True, mult=None):

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x)['pred_masks'])
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )))['pred_masks'])
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )))['pred_masks'])
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)))['pred_masks'])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_3D_2Dconv_tiled(self, case_image, patch_size, do_mirroring, mirror_axes=(0, 1), step_size=0.5, regions_class_order=None,
                                          use_gaussian=False, pad_border_mode="edge", pad_kwargs=None, all_in_gpu=False, verbose=True):
        if all_in_gpu:
            raise NotImplementedError

        case_shp = case_image.shape
        assert len(case_shp) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(case_shp[1]):
            if self.args.pseudo_3d_slices == 1:
                x = case_image[:, s]
            else:
                mn = s - (self.args.pseudo_3d_slices - 1) // 2
                mx = s + (self.args.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_shp[1])
                x = case_image[:, valid_mn:valid_mx]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(x.shape)
                    shp_for_pad[1] = need_to_pad_below
                    x = np.concatenate((np.zeros(shp_for_pad), x), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(x.shape)
                    shp_for_pad[1] = need_to_pad_above
                    x = np.concatenate((x, np.zeros(shp_for_pad)), 1)
                x = x.reshape((-1, case_shp[-2], case_shp[-1]))
            
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(
                x, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

class BayeSeg_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, total_epoch, weight_bayes, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        super().__init__(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice, log_dice, ignore_label)
        self.total_epoch = total_epoch
        self.weight_bayes = weight_bayes
    
    def forward(self, net_output, target, epoch):

        pred = net_output['pred_masks']

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(pred, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(pred, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        
        N = net_output['normalization']
        loss_y = torch.sum(net_output['kl_y']) / N
        loss_mu_m = torch.sum(net_output['kl_mu_m']) / N
        loss_sigma_m = torch.sum(net_output['kl_sigma_m']) / N
        loss_mu_x = torch.sum(net_output['kl_mu_x']) / N
        loss_sigma_x = torch.sum(net_output['kl_sigma_x']) / N
        loss_mu_z = torch.sum(net_output['kl_mu_z']) / N
        loss_sigma_z = torch.sum(net_output['kl_sigma_z']) / N
        loss_Bayes = loss_y + loss_mu_m + loss_sigma_m + loss_mu_x + loss_sigma_x + loss_mu_z + loss_sigma_z

        print('Bayes loss: ', loss_Bayes.item())
        # if loss_Bayes.item() > 10000:
        #    loss_Bayes = 0.00001 * loss_Bayes
        # elif loss_Bayes.item() > 1000 and loss_Bayes.item() < 10000:
        #    loss_Bayes = 0.0001 * loss_Bayes
        # elif loss_Bayes.item() > 100 and loss_Bayes.item() < 1000:
        #    loss_Bayes = 0.001 * loss_Bayes
        # elif loss_Bayes.item() > 10 and loss_Bayes.item() < 100:
        #    loss_Bayes = 0.01 * loss_Bayes
            
        # result += 100 * loss_Bayes * (1 - math.cos(math.pi * epoch / self.total_epoch)) / 2

        result += self.weight_bayes * loss_Bayes

        return result


class BayeSegTrainer(nnUNetTrainer):
    def __init__(self, plans_file, fold, args, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.args = args
        self.initial_lr = args.lr
        self.max_num_epochs = args.epochs
        self.cutmix_prob = args.cutmix_prob
        self.init_args = (plans_file, fold, args, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
    
    def initialize_unet(self):
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        unet = Generic_UNet(self.num_input_channels, self.base_num_features, 2*self.num_classes, net_numpool,
                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        unet.inference_apply_nonlin = softmax_helper
        return unet
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_tr = DataLoader_pseudo3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent, pad_mode="constant",
                                    pad_sides=self.pad_all_sides, memmap_mode='r', pseudo_3d_slices=self.args.pseudo_3d_slices)
        dl_val = DataLoader_pseudo3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                     oversample_foreground_percent=self.oversample_foreground_percent, pad_mode="constant",
                                     pad_sides=self.pad_all_sides, memmap_mode='r', pseudo_3d_slices=self.args.pseudo_3d_slices)
        return dl_tr, dl_val
    
    def get_aux_generator(self):

        aux_base_folder = "/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_preprocessed/ACDC"
        self.folder_with_ACDC = join(aux_base_folder, "nnUNetData_plans_v2.1_2D_stage0")
        aux_dataset = load_dataset(self.folder_with_ACDC)
        basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                    self.data_aug_params['rotation_y'],
                                                    self.data_aug_params['rotation_z'],
                                                    self.data_aug_params['scale_range'])
        dl_aux = DataLoader_pseudo3D(aux_dataset, basic_generator_patch_size, self.patch_size, self.batch_size, pad_mode="constant",
                                     pad_sides=self.pad_all_sides, memmap_mode='r', pseudo_3d_slices=self.args.pseudo_3d_slices)
        gen_aux, _ = get_default_augmentation(dl_aux, None, self.patch_size, self.data_aug_params)
        return gen_aux

    def initialize_network(self):
        unet = self.initialize_unet()
        self.args.num_classes = self.num_classes
        self.network = BayeSeg(self.args, unet)
        self.network.cuda()

    def setup_DA_params(self):

        self.do_dummy_2D_aug = False
        if max(self.patch_size) / min(self.patch_size) > 1.5:
            default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def initialize(self, training=True, force_load_plans=False):

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        # also initialize the loss of BayeSeg here
        self.loss = BayeSeg_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, self.max_num_epochs, self.args.weight_bayes)
        self.batch_size = self.args.batch_size

        self.setup_DA_params()
        if self.args.pseudo_3d_slices != 1:
            use_mask_for_norm = OrderedDict()
            for i in range(self.args.pseudo_3d_slices):
                use_mask_for_norm[i] = True
            self.data_aug_params["mask_was_used_for_normalization"] = use_mask_for_norm

        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier']+"_stage%d" % self.stage)

        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            self.dl_aux = self.get_aux_generator()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                unpack_dataset(self.folder_with_ACDC)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen = get_moreDA_augmentation(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                deep_supervision_scales=None,
                pin_memory=True,
                use_nondetMultiThreadedAugmenter=False
            )
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        r = np.random.rand(1)
        if r < self.cutmix_prob and self.network.training:
            aux_image = next(self.dl_aux)['data']
            aux_image = maybe_to_torch(aux_image)
            aux_image = to_cuda(aux_image)
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), self.args.cut_ratio)
            aux_image[:, :, bbx1:bbx2, bby1:bby2] = data[:, :, bbx1:bbx2, bby1:bby2]
            data = aux_image

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target, self.epoch)
                print(l.item())

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target, self.epoch)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        output = output['pred_masks']
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))


def rand_bbox(size, cut_ratio):
    lam = np.random.uniform(cut_ratio[0], cut_ratio[1])
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    # cx = np.random.randint(W)
    # cy = np.random.randint(H)

    # center
    cx = W // 2
    cy = H // 2

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class DataLoader_pseudo3D(DataLoader2D):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge", pad_kwargs_data=None, pad_sides=None):
        super(DataLoader_pseudo3D, self).__init__(data, patch_size, final_patch_size, batch_size, oversample_foreground_percent,
                                                  memmap_mode, pseudo_3d_slices)
    
    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros((self.batch_size, self.pseudo_3d_slices, self.data_shape[-2], self.data_shape[-1]), dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb: valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}
