import h5py
import torch
import os
from pathlib import Path
import numpy as np
from Utils.Util import save_tensor_img
from Model.model_SiPRNet import MyNet
import random

# Seed everything
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == '__main__':
    dataset_name = 'RAF'
    modelname = 'SiPRNet_'

    if dataset_name == 'RAF':
        test_set = h5py.File('./Data/RAF_data.h5', 'r')
        m_model = MyNet()
        m_model.load_state_dict(torch.load("./Model/SiPRNet_RAF.pth"))
    elif dataset_name == 'Fashion':
        test_set = h5py.File('./Data/Fashion_data.h5', 'r')
        m_model = MyNet()
        m_model.load_state_dict(torch.load("./Model/SiPRNet_Fashion.pth"))

    else:
        raise Exception('No dataset')

    m_model = m_model.eval().to(device)
    save_dir = os.path.join('Result', modelname + dataset_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # create a directory if not exist

    for idx in list(test_set.keys()):

        data = test_set[idx]
        in_tensor, tar_tensor = torch.from_numpy(data[:, 0:1, :, :]).to(device), \
                                torch.from_numpy(data[:, 1:, :, :]).to(device)

        pred = m_model(in_tensor)

        save_file_prefix = os.path.join(save_dir, modelname + "img_%04d" % (int(idx)))
        save_tensor_img(pred, save_file_prefix, isGT=False)
        save_tensor_img(tar_tensor, save_file_prefix, isGT=True)
