# 2023/10/31 Re-arrange code for generating Noises (Perturbation)

import torch
import os
import random
import numpy
from pathlib import Path
from datasets.augmentations1 import train_transform
from datasets.split_data_set_combined import SplitDatasetCombined_BDD
from attack.uap_phantom_sponge import UAPPhantomSponge


def set_random_seed(seed_value, use_cuda=True):
    numpy.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
        
        
# settings
models_vers = [5] # for example: models_vers = [5] or models_vers = [3, 4, 5]
epsilon = 70
lambda_1 = 1
lambda_2 = 10
seed = 42
patch_size=(640,640)
img_size=(640,640)
batch_size = 8
num_workers = 4
max_labels_per_img = 65
BDD_IMG_DIR = r'C:\Users\Scott Moran\Documents\Research\NMSProject-master\PhantomSponges\BDD_Dir\BDD_IMG_DIR'
BDD_LAB_DIR = r'C:\Users\Scott Moran\Documents\Research\NMSProject-master\PhantomSponges\BDD_Dir\BDD_LAB_DIR'


# Load Dataset
def collate_fn(batch):
    return tuple(zip(*batch))


split_dataset = SplitDatasetCombined_BDD(
            img_dir= BDD_IMG_DIR,
            lab_dir= BDD_LAB_DIR,
            max_lab=max_labels_per_img,
            img_size=img_size,
            transform=train_transform,
            collate_fn=collate_fn)


train_loader, val_loader, test_loader = split_dataset(val_split=0.1,
                                                      shuffle_dataset=True,
                                                      random_seed=seed,
                                                      batch_size=batch_size,
                                                      ordered=False,
                                                      collate_fn=collate_fn)

# Creat UAP
torch.cuda.empty_cache()
patch_name = r"yolov"
for ver in models_vers:
  patch_name += f"_{ver}"
patch_name += f"_epsilon={epsilon}_lambda1={lambda_1}_lambda2={lambda_2}"

uap_phantom_sponge_attack = UAPPhantomSponge(patch_folder=patch_name, train_loader=train_loader, val_loader=val_loader, epsilon = epsilon, lambda_1=lambda_1, lambda_2=lambda_2, patch_size=patch_size, models_vers=models_vers)
adv_img = uap_phantom_sponge_attack.run_attack()



