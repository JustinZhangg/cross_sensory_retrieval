import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = sys.float_info.epsilon

class DiscLoss(nn.Module):
    def __init__(self, ):
        super(DiscLoss, self).__init__()
    
    def forward(self, x):
        var = x.var(dim=0).mean()
        loss = torch.log(1+x.size(0)/(var+eps))
            
        return loss
    
