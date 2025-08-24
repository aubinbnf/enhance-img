import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CelebASuperResDataset(Dataset):
    def __init__(self, root_dir, hr_size=128, scale=4):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)  # liste des fichiers d’images
        self.hr_size = hr_size
        self.scale = scale
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # chemin de l’image
        img_path = os.path.join(self.root_dir, self.files[idx])
        
        # charger image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # redimensionner en HR (ex. 128x128)
        HR = cv2.resize(img, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)

        # créer LR (ex. 32x32 puis remonter en 128x128)
        lr_size = self.hr_size // self.scale
        LR = cv2.resize(HR, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        LR = cv2.resize(LR, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)

        # convertir en tenseurs [0,1]
        HR = self.to_tensor(HR)
        LR = self.to_tensor(LR)

        return LR, HR
