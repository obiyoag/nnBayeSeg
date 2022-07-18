from collections import OrderedDict
import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import glob
from sklearn.model_selection import KFold
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--institution", type=str, required=True, choices=['both', 'FeTA2022_Institution1', 'FeTA2022_Institution2'])
    args = parser.parse_args()

    out_folder = "/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_raw_data_base/nnUNet_raw_data/"
    out_folder = join(out_folder, args.task_name)

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    maybe_mkdir_p(join(out_folder, "predictions"))

    folder = "/home/gaoyibo/Datasets/FeTA"
    all_train_files = []

    if args.institution != "both":
        folder = join(folder, args.institution)
        assert os.path.exists(folder), f'provided FeTA path {folder} does not exist'
        image_list = glob.glob(os.path.join(folder, 'sub*', '*', '*T2w.nii.gz'))
        label_list = glob.glob(os.path.join(folder, 'sub*', '*', '*dseg.nii.gz'))
        image_list, label_list = sorted(image_list), sorted(label_list)
    else:
        assert os.path.exists(folder), f'provided FeTA path {folder} does not exist'
        image_list = glob.glob(os.path.join(folder, '*', 'sub*', '*', '*T2w.nii.gz'))
        label_list = glob.glob(os.path.join(folder, '*', 'sub*', '*', '*dseg.nii.gz'))
        image_list, label_list = sorted(image_list), sorted(label_list)

    for image_path, label_path in zip(image_list, label_list):
        patient_identifier = image_path.split("/")[-1][4:7]
        file_name = "FeTA_" + patient_identifier
        all_train_files.append(file_name + "_0000.nii.gz")
        shutil.copy(image_path, join(out_folder, "imagesTr", file_name + "_0000.nii.gz"))
        shutil.copy(label_path, join(out_folder, "labelsTr", file_name + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "FeTA"
    json_dict['description'] = "Fetal Tissue Annotation and Segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see FeTA challenge"
    json_dict['licence'] = "see FeTA challenge"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "external cerebrospinal fluid",
        "2": "grey matter",
        "3": "white matter",
        "4": "ventricles",
        "5": "cerebellum",
        "6": "deep grey matter",
        "7": "brainstem",
    }
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in all_train_files]
    json_dict['test'] = []

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

    splits = []
    patients = np.unique([i[:-12] for i in all_train_files])

    kf = KFold(5, shuffle=True, random_state=12345)
    for tr, val in kf.split(patients):
        splits.append(OrderedDict())
        tr_patients = patients[tr]
        splits[-1]['train'] = [i[:-12] for i in all_train_files if i[:-12] in tr_patients]
        val_patients = patients[val]
        splits[-1]['val'] = [i[:-12] for i in all_train_files if i[:-12] in val_patients]
    
    save_pickle(splits, join(out_folder, "splits_final.pkl"))
