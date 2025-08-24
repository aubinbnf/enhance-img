import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSRCNN(nn.Module):
    def __init__(self):
        super(SimpleSRCNN, self).__init__()
        # 1ère couche : extraction de features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        # 2ème couche : non-linéarité + transformation
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        # 3ème couche : reconstruction de l'image
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # sortie, pas de ReLU pour conserver les valeurs
        return x
