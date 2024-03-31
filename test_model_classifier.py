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
from model_arch import TCN

def predict():

    torch.manual_seed(1)
    use_cuda = True
    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = TCN(input_size=(150, 17 ,2), num_channels=[8, 8, 8, 8], kernel_size=3, dropout=0.2).to(device)

    npz_dir = 'snatch\snatch_4'

    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_path = os.path.join(root, file)
                
                # Load .npz file
                npz_data = np.load(npz_path)
                npz_data = npz_data['reconstruction'][:, :, :2]
                # Pad the data to a fixed length
                last_row = np.expand_dims(npz_data[-1], 0)
                fixed_length = 150
                if len(npz_data) < fixed_length:
                    npz_data = np.concatenate([npz_data, np.tile(last_row, (fixed_length - len(npz_data), 1, 1))])
                else:
                    npz_data = npz_data[:fixed_length]

    pred_dl = DataLoader(npz_data, batch_size=1, shuffle=False)

    model.load_state_dict(torch.load('saved_models_150\model_e39_l0.7096.pth'))

    model.eval()

    with torch.no_grad():
        for inputs in pred_dl:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Convert outputs to predicted labels (e.g., using argmax)
            _, predicted = torch.max(outputs, 1)

    return predicted

if __name__ == "__main__":
    
    predicted_labels = predict()
    print("Predicted labels:", predicted_labels)        