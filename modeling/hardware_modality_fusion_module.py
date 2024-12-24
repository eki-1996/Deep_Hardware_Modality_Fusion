import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchvision import transforms

class Exposuref(nn.Module):

    def __init__(self, in_channel=16, out_channel=1, kernal_size=8, binarize_type='full', kwargs={}):
        #5 dimension. batch x channel x time x width x height
        super().__init__()
        self.in_channel = in_channel  
        self.out_channel = out_channel 
        self.kernal_size = kernal_size  
        self.noise_count = 100
        self.kwargs = kwargs
        self.binarize_type = binarize_type
        self.weight = Parameter(torch.Tensor(self.out_channel, self.in_channel, self.kernal_size, self.kernal_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.stdv = math.sqrt(1.5 / (self.kernal_size * self.kernal_size * self.in_channel))
        self.weight.data.uniform_(-self.stdv, self.stdv)
        self.weight.lr_scale = 1. / self.stdv

    def forward(self, input):
        if self.training and self.noise_count > 0:
            p = max(self.noise_count, 0) / 100
            self.noise_count -= 1
            contious_weight = (self.weight * (1-p) + 
                               torch.rand(self.weight.data.size()).uniform_(-self.stdv, self.stdv).to('cuda').mul(p)
                               ).repeat(1, 1, int(input.shape[-2] / self.kernal_size), int(input.shape[-1] / self.kernal_size))
        else:
            contious_weight = self.weight.repeat(1, 1, int(input.shape[-2] / self.kernal_size), int(input.shape[-1] / self.kernal_size))
        out = input * torch.sigmoid(contious_weight)
        # print(f"input: min: {torch.min(input)}, max: {torch.max(input)}.  contious weight: min: {torch.min(torch.sigmoid(contious_weight))}, max: {torch.max(torch.sigmoid(contious_weight))}. out: min: {torch.min(out.mean(dim=2))}, max: {torch.max(out.mean(dim=2))}")
        # out = input * torch.softmax(contious_weight, dim=1)
        # out = input * contious_weight
        return out.mean(dim=2)
