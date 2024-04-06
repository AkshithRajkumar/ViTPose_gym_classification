import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from torch.utils.data import DataLoader

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.num_layers = len(num_channels)

        # Modified here
        in_channels = input_size[-1]

        for i in range(self.num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2

            out_channels = num_channels[i]
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
            self.conv_layers.append(conv_layer)
            batch_norm_layer = nn.BatchNorm1d(out_channels)
            self.batch_norm_layers.append(batch_norm_layer)

            in_channels = out_channels

        self.linear = nn.Linear(num_channels[-1], 3)
        self.dropout_layer = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_poses, num_joints, num_coords = x.size()
        x = x.reshape(batch_size, num_poses * num_joints, num_coords).transpose(1, 2)
        for i in range(self.num_layers):
            conv_layer = self.conv_layers[i]
            batch_norm_layer = self.batch_norm_layers[i]
            x = conv_layer(x)
            x = batch_norm_layer(x)
            x = nn.functional.relu(x)
            x = self.dropout_layer(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
