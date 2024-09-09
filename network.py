import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models

from losses import *

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.vision_backbone= nn.Sequential(
            *list(models.resnet18(weights=None).children())[:-1])
        self.touch_backbone= nn.Sequential(
            *list(models.resnet18(weights=None).children())[:-1])
        
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True)

        self.vision_fc = nn.Linear(512, 128)
        self.touch_fc = nn.Linear(512, 128)

        self.disc_loss = DiscLoss()
        self.cos_loss = nn.CosineSimilarity(dim=-1)
        self.ploss = nn.PairwiseDistance(p=2)
        

    def forward(self, touch_input, vision_input, label):
        x_touch = self.touch_backbone(touch_input)
        x_vision = self.vision_backbone(vision_input)

        x_touch = torch.flatten(x_touch,start_dim=1)
        x_vision = torch.flatten(x_vision,start_dim=1)

        x1 = torch.cat([x_vision, x_touch], dim=1)
        x2 = torch.cat([x_touch, x_vision], dim=1)

        y1, _ = self.lstm(x1)
        y2, _ = self.lstm(x2)

        # y1, _ = self.lstm(x_touch)
        # y2, _ = self.lstm(x_vision)

        y1 = self.touch_fc(y1)
        y2 = self.vision_fc(y2)
        
        output = {}
        output['y1'] = y1
        output['y2'] = y2

        # output['loss'] = 0.1*self.disc_loss(x_touch) - self.cos_loss(y1[label == 1], y2[label == 1]).mean() \
        #     + self.cos_loss(y1[label == 0], y2[label == 0]).mean()

        output['loss'] = 0.05 * self.disc_loss(x_touch) \
              + self.ploss(y1[label == 1], y2[label == 1]).mean() \
              - 0.1 * self.ploss(y1[label == 0], y2[label == 0]).mean()

        
        return output