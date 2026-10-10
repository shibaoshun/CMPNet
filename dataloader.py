import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from utils.ElasticDeform import ElasticDeform2D_rand
import os
from torch.utils.data import DataLoader
import utils.transforms as T
from scipy.io import loadmat
from utils.subsample import MaskFunc
# %%
def cobmine_all_coils(image, sensitivity):
    """return sensitivity combined images from all coils"""
    combined = T.complex_multiply(sensitivity[..., 0], -sensitivity[..., 1], image[..., 0], image[..., 1])
    return combined.sum(dim=0)

def load_train_data_path(dataset_dir, train_kspace_name, train_maps_name, train_sam, val_kspace_name, val_maps_name, val_sam):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    data_list = {}
    data_list['train'] = []
    data_list['val'] = []

    path_train_kspace = os.path.join(dataset_dir, train_kspace_name)
    path_train_maps = os.path.join(dataset_dir, train_maps_name)
    path_train_sam = os.path.join(dataset_dir, train_sam)
    path_val_kspace = os.path.join(dataset_dir, val_kspace_name)
    path_val_maps = os.path.join(dataset_dir, val_maps_name)
    path_val_sam = os.path.join(dataset_dir, val_sam)

    path_train_kspace_list = os.listdir(path_train_kspace)
    path_train_maps_list = os.listdir(path_train_maps)
    path_train_sam2 = os.listdir(path_train_sam)
    path_val_kspace_list = os.listdir(path_val_kspace)
    path_val_maps_list = os.listdir(path_val_maps)
    path_val_sam2 = os.listdir(path_val_sam)

    for train_kspace_filename, train_maps_filename, sam_filename in zip(path_train_kspace_list, path_train_maps_list, path_train_sam2):
        raw = os.path.join(path_train_kspace, train_kspace_filename)
        sen = os.path.join(path_train_maps, train_maps_filename)
        sam = os.path.join(path_train_sam, sam_filename)
        data_list['train'] += [[raw, sen, sam]]

    for val_kspace_filename, val_maps_filename, sam_filename_val in zip(path_val_kspace_list, path_val_maps_list, path_val_sam2):
        raw = os.path.join(path_val_kspace, val_kspace_filename)
        sen = os.path.join(path_val_maps, val_maps_filename)
        sam = os.path.join(path_val_sam, sam_filename_val)
        data_list['val'] += [[raw, sen, sam]]

    return data_list

def load_train_data(dataset_dir, train_kspace_name, train_maps_name, val_kspace_name, val_maps_name):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    data_list = {}
    data_list['train'] = []
    data_list['val'] = []

    path_train_kspace = os.path.join(dataset_dir, train_kspace_name)
    path_train_maps = os.path.join(dataset_dir, train_maps_name)
    path_val_kspace = os.path.join(dataset_dir, val_kspace_name)
    path_val_maps = os.path.join(dataset_dir, val_maps_name)

    path_train_kspace_list = os.listdir(path_train_kspace)
    path_train_maps_list = os.listdir(path_train_maps)
    path_val_kspace_list = os.listdir(path_val_kspace)
    path_val_maps_list = os.listdir(path_val_maps)
    for train_kspace_filename, train_maps_filename in zip(path_train_kspace_list, path_train_maps_list):
        raw = os.path.join(path_train_kspace, train_kspace_filename)
        sen = os.path.join(path_train_maps, train_maps_filename)
        data_list['train'] += [[raw, sen]]

    for val_kspace_filename, val_maps_filename in zip(path_val_kspace_list, path_val_maps_list):
        raw = os.path.join(path_val_kspace, val_kspace_filename)
        sen = os.path.join(path_val_maps, val_maps_filename)
        data_list['val'] += [[raw, sen]]

    return data_list



class MRIDataset_b(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)
        # return 10


    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch_brain(subject_id, self.acceleration, self.center_fraction)

class MRIDataset_c(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)
        # return 10


    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch_cmr(subject_id, self.acceleration, self.center_fraction)

class MRIDataset_k(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)
        # return 10


    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch_knee(subject_id, self.acceleration, self.center_fraction)



def get_epoch_batch_brain(subject_id, acc, center_fract):
    ''' get training data '''
    rawdata_name, coil_name = subject_id
    rawdata = np.complex64(loadmat(rawdata_name)['kspace_n']).transpose(2, 0, 1)

    sensitivity = np.complex64(loadmat(coil_name)['gt_cms'])
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])

    rawdata2 = T.to_tensor(rawdata)
    sensitivity2 = T.to_tensor(sensitivity.transpose(2, 0, 1))
    return data_for_training(rawdata2, sensitivity2, mask_func, rawdata_name)

def get_epoch_batch_cmr(subject_id, acc, center_fract):
    ''' get training data '''
    rawdata_name, coil_name = subject_id
    rawdata = np.complex64(loadmat(rawdata_name)['kspace']).transpose(2, 0, 1)

    sensitivity = np.complex64(loadmat(coil_name)['maps'])
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])

    rawdata2 = T.to_tensor(rawdata)
    sensitivity2 = T.to_tensor(sensitivity.transpose(2, 0, 1))
    return data_for_training(rawdata2, sensitivity2, mask_func, rawdata_name)

def get_epoch_batch_knee(subject_id, acc, center_fract):
    ''' get training data '''
    rawdata_name, coil_name = subject_id
    rawdata = np.complex64(loadmat(rawdata_name)['kspace']).transpose(2, 0, 1)

    sensitivity = np.complex64(loadmat(coil_name)['maps'])
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])

    rawdata2 = T.to_tensor(rawdata)
    sensitivity2 = T.to_tensor(sensitivity.transpose(2, 0, 1))
    return data_for_training(rawdata2, sensitivity2, mask_func, rawdata_name)


def data_for_training(rawdata, sensitivity, mask_func, rawdata_name, norm=True):
    ''' normalize each slice using complex absolute max value'''
    coils, Ny, Nx, ps = rawdata.shape

    # shift data
    shift_kspace = rawdata
    x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    adjust = (-1) ** (x + y)
    shift_kspace = T.ifftshift(shift_kspace, dim=(-3, -2)) * torch.from_numpy(adjust).view(1, Ny, Nx, 1).float()

    # apply masks
    shape = np.array(shift_kspace.shape)
    shape[:-3] = 1
    mask = mask_func(shape)
    mask = T.ifftshift(mask)

    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]),shift_kspace)


    img_gt, img_und = T.ifft2n(shift_kspace), T.ifft2n(masked_kspace)

    masks = mask.repeat(coils, Ny, 1, ps)
    if norm:
        norm = T.complex_abs(img_und).max()

        if norm < 1e-6: norm = 1e-6
        # normalized recon
    else:
        norm = 1

    # normalize data to learn more effectively
    img_gt, img_und, rawdata_und = img_gt / norm, img_und / norm, masked_kspace / norm
    sense_gt = cobmine_all_coils(img_gt,sensitivity)
    sense_und = cobmine_all_coils(img_und, sensitivity)
    return sense_und, sense_gt, rawdata_und, masks, sensitivity, str(rawdata_name)



def load_testdata_path(dataset_dir, train_kspace_name, train_maps_name):
    """ Go through each subset (training, validation) under the data directory
    and list the file names and landmarks of the subjects
    """
    data_list = {}
    data_list['test'] = []

    path_train_kspace = os.path.join(dataset_dir, train_kspace_name)
    path_train_maps = os.path.join(dataset_dir, train_maps_name)

    path_train_kspace_list = os.listdir(path_train_kspace)  # yuan
    path_train_maps_list = os.listdir(path_train_maps)

    for train_kspace_filename, train_maps_filename in zip(path_train_kspace_list, path_train_maps_list):
        raw = os.path.join(path_train_kspace, train_kspace_filename)
        sen = os.path.join(path_train_maps, train_maps_filename)
        data_list['test'] += [[raw, sen]]
    return data_list

class MRIDataset_test(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch_test(subject_id, self.acceleration, self.center_fraction)


def get_epoch_batch_test(subject_id, acc, center_fract):
    ''' get training data '''
    rawdata_name, coil_name = subject_id
    rawdata = np.complex64(loadmat(rawdata_name)['k_space']).transpose(2, 0, 1)
    sensitivity = np.complex64(loadmat(coil_name)['maps'])
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])

    filename=os.path.basename(rawdata_name.split('/')[-1])
    coilname=os.path.basename(coil_name.split('/')[-1])

    rawdata2 = T.to_tensor(rawdata)
    sensitivity2 = T.to_tensor(sensitivity.transpose(2, 0, 1))
    return data_for_testing(rawdata2, sensitivity2, mask_func, filename)



def data_for_testing(rawdata, sensitivity, mask_func, rawdata_name, norm=True):
    ''' normalize each slice using complex absolute max value'''
    coils, Ny, Nx, ps = rawdata.shape

    # shift data
    shift_kspace = rawdata
    x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
    adjust = (-1) ** (x + y)
    shift_kspace = T.ifftshift(shift_kspace, dim=(-3, -2)) * torch.from_numpy(adjust).view(1, Ny, Nx,
                                                                                           1).float()

    # apply masks
    shape = np.array(shift_kspace.shape)
    shape[:-3] = 1
    mask = mask_func(shape)
    mask = T.ifftshift(mask)
    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), shift_kspace)
    img_gt, img_und = T.ifft2n(shift_kspace), T.ifft2n(
        masked_kspace)

    masks = mask.repeat(coils, Ny, 1, ps)

    if norm:
        norm = T.complex_abs(img_und).max()

        if norm < 1e-6: norm = 1e-6
        # normalized recon
    else:
        norm = 1

    # normalize data to learn more effectively
    img_gt, img_und, rawdata_und = img_gt / norm, img_und / norm, masked_kspace / norm  # 归一化
    sense_gt = cobmine_all_coils(img_gt,
                                 sensitivity)
    sense_und = cobmine_all_coils(img_und, sensitivity)
    return sense_und, sense_gt, rawdata_und, masks, sensitivity, rawdata_name
    
    # ==========================================
# 前列腺微调专属 Dataset
# ==========================================
class MRIDataset_p(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch_prostate(subject_id, self.acceleration, self.center_fraction)

def get_epoch_batch_prostate(subject_id, acc, center_fract):
    import os
    rawdata_name, coil_name = subject_id
    
    # 提取脚本存下来的 kspace 和 maps 已经是 [线圈, 高, 宽] 格式了
    rawdata = np.complex64(loadmat(rawdata_name)['kspace'])
    sensitivity = np.complex64(loadmat(coil_name)['maps'])
    
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    
    rawdata2 = T.to_tensor(rawdata)
    sensitivity2 = T.to_tensor(sensitivity)
    
    # 提取一下文件名字 (比如 prostate_slice_000.mat)
    filename = os.path.basename(rawdata_name.split('/')[-1])
    
    # 🚨 核心修复：把 filename 作为第 4 个参数乖乖交上去！
    return data_for_training(rawdata2, sensitivity2, mask_func, filename)

