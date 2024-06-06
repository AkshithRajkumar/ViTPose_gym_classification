Classification of Gym exercises using keypoints
The purpose is to develop a model that can analyze the form of people doing different gym exercises. The objective is to implement vision transformers for pose estimation using videos. The focus is to estimate pose using vision transformers, and then apply a classifier to determine the exercise they are doing based on the keypoints and their temporal evolution.

The vision transformer model being used is an adaptation of VitPose (Xu et al., 2022).

We used a Temporal Convolution Network to classify the keypoints that were generated from the Pose estimator.

Section A of the report below describes this work, and has the following parts:

Data pre-processing
Pose Estimation
Classification
Section B describes a holistic discussion of the overall results.

Shown below is the Model architecture we used DLPROJECT.png

Links
# Colab Link
https://colab.research.google.com/drive/1rudcGUj16VXiqVZGi0Lpn1IeXBfuXeR9?usp=sharing
# Dataset Link
https://utoronto-my.sharepoint.com/:f:/g/personal/timoteo_frelau_mail_utoronto_ca/EhZ0vtU69VZIis0kp9hFRgQBegF_YGBKzzmJ0-KbyQQAOg?e=NqswQU
SECTION A##
Part A: Data Collection and Preprocessing
Our Dataset comprises of videos that were gathared from popular social media platforms such as YouTube and Instagram. Additionally, we recorded our own videos (as test data). Our focus was on three key exercises: Snatch, Deadlift and Squat resulting in a balanced dataset comprising of 183 videos.

image.png

While choosing the videos various aspects have been considered based on our learnings in applying the models on videos. Videos with multiple people, objects have not been chosen. Videos with similar angle of recording were chosen. The start and end pose of the each exercises were determined and the videos are cropped to start and end at those poses.

After downloading the videos, we cropped them around the person so that only the human was left on the frames. We also cropped the video length to one repetition.
Sample of pre-processed data:

image.png

Part B: Pose Estimation###
This model was pre-existing.
The video in mp4 format is loaded into a pre-trained ViTPose Transformer model that was implemented by us using pytorch. The processed frames from Pose estimator are then used to visualize the poses of the humans in the data. This is how we knew if a video was good to fit for the classifier, or if we had to reject it because of poor keypoints estimation of the model.
We also collect the keypoints from the estimator as a numpy array and these are saved as a .npz file for each video.

Cloning our Pose Estimation repo
The pose estimation is achieved by running the preprocesssed dataset into a custom pretrained ViTPose model to extract the keypoints. The output is also visualized for evaluation. The work done to implement our Pose estimation can be found at the link here.

# Clone the GitHub repository into Google Colab
!git clone https://github.com/AkshithRajkumar/ViTPose_gym_classification.git
Cloning into 'ViTPose_gym_classification'...
remote: Enumerating objects: 1199, done.
remote: Counting objects: 100% (258/258), done.
remote: Compressing objects: 100% (241/241), done.
remote: Total 1199 (delta 13), reused 256 (delta 12), pack-reused 941
Receiving objects: 100% (1199/1199), 166.52 MiB | 28.41 MiB/s, done.
Resolving deltas: 100% (16/16), done.
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
Mounted at /content/drive
#load vitpose-b-multi-coco.pth model from drive

#Alternatively you can download the model and load into the colab as well

model_path = '/content/drive/MyDrive/Colab Notebooks/MIE1517 Project/vitpose-b-multi-coco.pth'
!pip install ffmpeg
Collecting ffmpeg
  Downloading ffmpeg-1.4.tar.gz (5.1 kB)
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: ffmpeg
  Building wheel for ffmpeg (setup.py) ... done
  Created wheel for ffmpeg: filename=ffmpeg-1.4-py3-none-any.whl size=6080 sha256=63b95aefb709e613b21dcb52b9046bc273efef5a69850277893ea12975e7666f
  Stored in directory: /root/.cache/pip/wheels/8e/7a/69/cd6aeb83b126a7f04cbe7c9d929028dc52a6e7d525ff56003a
Successfully built ffmpeg
Installing collected packages: ffmpeg
Successfully installed ffmpeg-1.4
%cd /content/ViTPose_gym_classification
!python inference1.py
/content/ViTPose_gym_classification
!!!!
363it [14:36,  2.42s/it]
This shows the way to run it for one video. After cloning the repo, we have multiple inference codes that inplements ViTPose on different structured data such as folders, singular files, either by passing the folder location, or just the video location, or by updating the code to the location prior to running.

Part C: Classifier###
The classifier is a temporal convolutional network that takes as input the keypoints, and classify the video as one of the three exercises.

First, we import the necessary libraries.

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
We had to create a custom implementation of the PyTorch Dataset class to load the keypoints that were saved as a numpy .npz file by VitPose.

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        file_paths = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            files = os.listdir(class_dir)
            for file in files:
                file_paths.append((os.path.join(class_dir, file), label))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx]
        data = np.load(file_path)
        data = data['reconstruction'][:, :, :2]
        # Pad the data to a fixed length
        last_row = np.expand_dims(data[-1], 0)
        fixed_length = 150
        if len(data) < fixed_length:
            data = np.concatenate([data, np.tile(last_row, (fixed_length - len(data), 1, 1))])
        else:
            data = data[:fixed_length]
        features = torch.from_numpy(data).float()  # Assuming features are stored under the key 'features'
        return features, label
We then carry out some sanity checks, to confirm first whether all the .npz files have been included in CustomDataset, and then to check the dimensions of the resulting tensors.

data = CustomDataset(r"new_keypoints_30_03")
print(len(data))
182
tmp = next(iter(data))
print(tmp[0].shape, tmp[1])
torch.Size([150, 17, 2]) 0
next(iter(data))[0].shape, next(iter(data))[1]
(torch.Size([150, 17, 2]), 0)
# use the DataLoader class to load the data
from torch.utils.data import DataLoader

batch_size = 16
shuffle = True

squats_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

# Example usage
for batch in squats_loader:
    features, labels = batch
    print(features.shape, labels.shape)
    break

print(labels)
torch.Size([16, 150, 17, 2]) torch.Size([16])
tensor([1, 1, 0, 1, 0, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 0])
Then we build our temporal convolutional network.

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
use_cuda = True
torch.manual_seed(1)
if torch.cuda.is_available() and use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
cpu
# Train, validation, and test splits
train_size = 100
val_size = 50
test_size = 32

seed = 20

train_data, test_data = train_test_split(data, test_size=test_size, train_size=150, shuffle=True, random_state=seed)
train_data, val_data = train_test_split(train_data, test_size=val_size, train_size=train_size, shuffle=True, random_state=seed)
print(len(train_data), len(val_data))
print(len(test_data))
100 50
32
Some more sanity checks, this time to see if the train test split has introduced significant imbalances into our data. It has introduced some imbalance, but it's minor.

# write a code to print the number of data in each label of 1 and 0
for i in range(3):
    print(f"Number of data in label {i}: {len([label for _, label in train_data if label == i])}")
Number of data in label 0: 35
Number of data in label 1: 31
Number of data in label 2: 34
for i in range(3):
    print(f"Number of data in label {i}: {len([label for _, label in val_data if label == i])}")
Number of data in label 0: 18
Number of data in label 1: 18
Number of data in label 2: 14
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size=4, shuffle=True)
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    return correct / total
Here is our train function.

import os
import torch.nn.utils as utils

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, n_epochs=30):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    model.to(device)

    prev_loss = 10000
    save_models = './saved_models_150'  # Define the path to save models
    os.makedirs(save_models, exist_ok=True)  # Create the directory if it doesn't exist

    for epoch in range(n_epochs):
        model.train()

        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        train_acc = get_accuracy(model, train_loader, device)
        train_acc_history.append(train_acc)


        val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc = get_accuracy(model, val_loader, device)
        val_acc_history.append(val_acc)

        scheduler.step(val_loss)

        if val_loss < prev_loss:
            prev_loss = val_loss
            path2save = os.path.join(save_models, f"model_e{epoch+1}_l{val_loss:.4f}.pth")
            torch.save(model.state_dict(), path2save)
            print(f"Model saved in {path2save}")

        print(f'Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        print(f'Train Accuracy = {train_acc:.4f}, Val Accuracy = {val_acc:.4f}')
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history
from torch.optim.lr_scheduler import ReduceLROnPlateau
model = TCN(input_size=(150, 17 ,2), num_channels=[8, 8, 8, 8], kernel_size=3, dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, amsgrad=True)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.90)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()
c:\Users\Akshith\anaconda3\envs\vitpose\lib\site-packages\torch\optim\lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
Part D: Quantitative analysis
Part(i): Training and validation accuracy
train_loss_history, val_loss_history, train_acc_history, val_acc_history = train(model, train_dl, val_dl, optimizer, scheduler, criterion, device, 100)
Model saved in ./saved_models_150\model_e1_l1.1001.pth
Epoch 1/100: Train Loss = 1.1024, Val Loss = 1.1001
Train Accuracy = 0.3100, Val Accuracy = 0.3200
Model saved in ./saved_models_150\model_e2_l1.0972.pth
Epoch 2/100: Train Loss = 1.0911, Val Loss = 1.0972
Train Accuracy = 0.4100, Val Accuracy = 0.4000
Epoch 3/100: Train Loss = 1.0787, Val Loss = 1.0981
Train Accuracy = 0.4300, Val Accuracy = 0.3400
Model saved in ./saved_models_150\model_e4_l1.0933.pth
Epoch 4/100: Train Loss = 1.0723, Val Loss = 1.0933
Train Accuracy = 0.5200, Val Accuracy = 0.3600
Model saved in ./saved_models_150\model_e5_l1.0847.pth
Epoch 5/100: Train Loss = 1.0645, Val Loss = 1.0847
Train Accuracy = 0.5500, Val Accuracy = 0.4200
Model saved in ./saved_models_150\model_e6_l1.0740.pth
Epoch 6/100: Train Loss = 1.0540, Val Loss = 1.0740
Train Accuracy = 0.5700, Val Accuracy = 0.4800
Model saved in ./saved_models_150\model_e7_l1.0662.pth
Epoch 7/100: Train Loss = 1.0368, Val Loss = 1.0662
Train Accuracy = 0.6000, Val Accuracy = 0.4200
Model saved in ./saved_models_150\model_e8_l1.0421.pth
Epoch 8/100: Train Loss = 1.0282, Val Loss = 1.0421
Train Accuracy = 0.5900, Val Accuracy = 0.4400
Model saved in ./saved_models_150\model_e9_l1.0263.pth
Epoch 9/100: Train Loss = 1.0090, Val Loss = 1.0263
Train Accuracy = 0.7100, Val Accuracy = 0.6200
Model saved in ./saved_models_150\model_e10_l1.0138.pth
Epoch 10/100: Train Loss = 0.9953, Val Loss = 1.0138
Train Accuracy = 0.6600, Val Accuracy = 0.5000
Model saved in ./saved_models_150\model_e11_l0.9996.pth
Epoch 11/100: Train Loss = 0.9819, Val Loss = 0.9996
Train Accuracy = 0.7100, Val Accuracy = 0.5800
Epoch 12/100: Train Loss = 0.9577, Val Loss = 1.0162
Train Accuracy = 0.6500, Val Accuracy = 0.6000
Model saved in ./saved_models_150\model_e13_l0.9898.pth
Epoch 13/100: Train Loss = 0.9464, Val Loss = 0.9898
Train Accuracy = 0.6700, Val Accuracy = 0.6000
Model saved in ./saved_models_150\model_e14_l0.9796.pth
Epoch 14/100: Train Loss = 0.9498, Val Loss = 0.9796
Train Accuracy = 0.7600, Val Accuracy = 0.5200
Model saved in ./saved_models_150\model_e15_l0.9630.pth
Epoch 15/100: Train Loss = 0.9274, Val Loss = 0.9630
Train Accuracy = 0.7300, Val Accuracy = 0.6800
Model saved in ./saved_models_150\model_e16_l0.9351.pth
Epoch 16/100: Train Loss = 0.8992, Val Loss = 0.9351
Train Accuracy = 0.7900, Val Accuracy = 0.6400
Model saved in ./saved_models_150\model_e17_l0.9205.pth
Epoch 17/100: Train Loss = 0.8872, Val Loss = 0.9205
Train Accuracy = 0.7600, Val Accuracy = 0.6200
Model saved in ./saved_models_150\model_e18_l0.8908.pth
Epoch 18/100: Train Loss = 0.8614, Val Loss = 0.8908
Train Accuracy = 0.8500, Val Accuracy = 0.7000
Epoch 19/100: Train Loss = 0.8558, Val Loss = 0.9375
Train Accuracy = 0.7900, Val Accuracy = 0.6400
Model saved in ./saved_models_150\model_e20_l0.8705.pth
Epoch 20/100: Train Loss = 0.8402, Val Loss = 0.8705
Train Accuracy = 0.8200, Val Accuracy = 0.6800
Model saved in ./saved_models_150\model_e21_l0.8519.pth
Epoch 21/100: Train Loss = 0.8170, Val Loss = 0.8519
Train Accuracy = 0.8600, Val Accuracy = 0.7200
Model saved in ./saved_models_150\model_e22_l0.8381.pth
Epoch 22/100: Train Loss = 0.8280, Val Loss = 0.8381
Train Accuracy = 0.8900, Val Accuracy = 0.8000
Model saved in ./saved_models_150\model_e23_l0.8373.pth
Epoch 23/100: Train Loss = 0.8031, Val Loss = 0.8373
Train Accuracy = 0.8700, Val Accuracy = 0.7800
Epoch 24/100: Train Loss = 0.7671, Val Loss = 0.8558
Train Accuracy = 0.8400, Val Accuracy = 0.7000
Model saved in ./saved_models_150\model_e25_l0.8255.pth
Epoch 25/100: Train Loss = 0.7713, Val Loss = 0.8255
Train Accuracy = 0.8700, Val Accuracy = 0.7600
Model saved in ./saved_models_150\model_e26_l0.8193.pth
Epoch 26/100: Train Loss = 0.7931, Val Loss = 0.8193
Train Accuracy = 0.8700, Val Accuracy = 0.7200
Model saved in ./saved_models_150\model_e27_l0.7955.pth
Epoch 27/100: Train Loss = 0.7822, Val Loss = 0.7955
Train Accuracy = 0.9000, Val Accuracy = 0.7800
Epoch 28/100: Train Loss = 0.7470, Val Loss = 0.8034
Train Accuracy = 0.8900, Val Accuracy = 0.7600
Model saved in ./saved_models_150\model_e29_l0.7882.pth
Epoch 29/100: Train Loss = 0.7523, Val Loss = 0.7882
Train Accuracy = 0.8900, Val Accuracy = 0.8200
Epoch 30/100: Train Loss = 0.7268, Val Loss = 0.8273
Train Accuracy = 0.8400, Val Accuracy = 0.7200
Model saved in ./saved_models_150\model_e31_l0.7837.pth
Epoch 31/100: Train Loss = 0.7318, Val Loss = 0.7837
Train Accuracy = 0.8800, Val Accuracy = 0.8000
Model saved in ./saved_models_150\model_e32_l0.7528.pth
Epoch 32/100: Train Loss = 0.7483, Val Loss = 0.7528
Train Accuracy = 0.9200, Val Accuracy = 0.8600
Epoch 33/100: Train Loss = 0.7171, Val Loss = 0.7787
Train Accuracy = 0.9000, Val Accuracy = 0.7400
Model saved in ./saved_models_150\model_e34_l0.7171.pth
Epoch 34/100: Train Loss = 0.6991, Val Loss = 0.7171
Train Accuracy = 0.9600, Val Accuracy = 0.8800
Epoch 35/100: Train Loss = 0.7075, Val Loss = 0.7892
Train Accuracy = 0.8900, Val Accuracy = 0.8200
Epoch 36/100: Train Loss = 0.7301, Val Loss = 0.7596
Train Accuracy = 0.8900, Val Accuracy = 0.8000
Epoch 37/100: Train Loss = 0.7136, Val Loss = 0.7621
Train Accuracy = 0.9100, Val Accuracy = 0.7800
Epoch 38/100: Train Loss = 0.6762, Val Loss = 0.8332
Train Accuracy = 0.8400, Val Accuracy = 0.7400
Model saved in ./saved_models_150\model_e39_l0.7096.pth
Epoch 39/100: Train Loss = 0.6960, Val Loss = 0.7096
Train Accuracy = 0.9400, Val Accuracy = 0.8800
Epoch 40/100: Train Loss = 0.6845, Val Loss = 0.7366
Train Accuracy = 0.9200, Val Accuracy = 0.8000
Epoch 41/100: Train Loss = 0.6637, Val Loss = 0.7630
Train Accuracy = 0.8900, Val Accuracy = 0.7800
Epoch 42/100: Train Loss = 0.6694, Val Loss = 0.7333
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 43/100: Train Loss = 0.6643, Val Loss = 0.7205
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 44/100: Train Loss = 0.6793, Val Loss = 0.7258
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 45/100: Train Loss = 0.6874, Val Loss = 0.7307
Train Accuracy = 0.9300, Val Accuracy = 0.8400
Epoch 46/100: Train Loss = 0.6711, Val Loss = 0.7226
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 47/100: Train Loss = 0.6815, Val Loss = 0.7348
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 48/100: Train Loss = 0.6919, Val Loss = 0.7273
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 49/100: Train Loss = 0.6806, Val Loss = 0.7199
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 50/100: Train Loss = 0.6849, Val Loss = 0.7262
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 51/100: Train Loss = 0.6500, Val Loss = 0.7198
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 52/100: Train Loss = 0.6849, Val Loss = 0.7140
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 53/100: Train Loss = 0.6590, Val Loss = 0.7158
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 54/100: Train Loss = 0.6569, Val Loss = 0.7154
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 55/100: Train Loss = 0.6767, Val Loss = 0.7140
Train Accuracy = 0.9600, Val Accuracy = 0.8600
Epoch 56/100: Train Loss = 0.6678, Val Loss = 0.7123
Train Accuracy = 0.9600, Val Accuracy = 0.8600
Epoch 57/100: Train Loss = 0.6730, Val Loss = 0.7128
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 58/100: Train Loss = 0.6783, Val Loss = 0.7222
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 59/100: Train Loss = 0.6780, Val Loss = 0.7259
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 60/100: Train Loss = 0.6833, Val Loss = 0.7228
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 61/100: Train Loss = 0.6833, Val Loss = 0.7251
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 62/100: Train Loss = 0.6631, Val Loss = 0.7182
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 63/100: Train Loss = 0.6687, Val Loss = 0.7150
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 64/100: Train Loss = 0.6575, Val Loss = 0.7204
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 65/100: Train Loss = 0.6742, Val Loss = 0.7330
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 66/100: Train Loss = 0.6690, Val Loss = 0.7355
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 67/100: Train Loss = 0.6750, Val Loss = 0.7235
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 68/100: Train Loss = 0.6861, Val Loss = 0.7234
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 69/100: Train Loss = 0.6936, Val Loss = 0.7205
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 70/100: Train Loss = 0.6704, Val Loss = 0.7277
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 71/100: Train Loss = 0.6536, Val Loss = 0.7246
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 72/100: Train Loss = 0.6658, Val Loss = 0.7186
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 73/100: Train Loss = 0.6965, Val Loss = 0.7307
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 74/100: Train Loss = 0.6731, Val Loss = 0.7203
Train Accuracy = 0.9400, Val Accuracy = 0.8400
Epoch 75/100: Train Loss = 0.6762, Val Loss = 0.7230
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 76/100: Train Loss = 0.6545, Val Loss = 0.7207
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 77/100: Train Loss = 0.6581, Val Loss = 0.7122
Train Accuracy = 0.9600, Val Accuracy = 0.8400
Epoch 78/100: Train Loss = 0.6657, Val Loss = 0.7201
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 79/100: Train Loss = 0.6786, Val Loss = 0.7179
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 80/100: Train Loss = 0.6800, Val Loss = 0.7200
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 81/100: Train Loss = 0.6599, Val Loss = 0.7176
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 82/100: Train Loss = 0.6667, Val Loss = 0.7152
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 83/100: Train Loss = 0.6816, Val Loss = 0.7205
Train Accuracy = 0.9400, Val Accuracy = 0.8400
Epoch 84/100: Train Loss = 0.6620, Val Loss = 0.7278
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 85/100: Train Loss = 0.6594, Val Loss = 0.7265
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 86/100: Train Loss = 0.6767, Val Loss = 0.7270
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 87/100: Train Loss = 0.6645, Val Loss = 0.7166
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 88/100: Train Loss = 0.6767, Val Loss = 0.7220
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 89/100: Train Loss = 0.6792, Val Loss = 0.7177
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 90/100: Train Loss = 0.6838, Val Loss = 0.7233
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 91/100: Train Loss = 0.6869, Val Loss = 0.7175
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 92/100: Train Loss = 0.6623, Val Loss = 0.7264
Train Accuracy = 0.9500, Val Accuracy = 0.8600
Epoch 93/100: Train Loss = 0.6882, Val Loss = 0.7162
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 94/100: Train Loss = 0.6832, Val Loss = 0.7343
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 95/100: Train Loss = 0.6613, Val Loss = 0.7292
Train Accuracy = 0.9300, Val Accuracy = 0.8600
Epoch 96/100: Train Loss = 0.6680, Val Loss = 0.7174
Train Accuracy = 0.9400, Val Accuracy = 0.8400
Epoch 97/100: Train Loss = 0.6504, Val Loss = 0.7165
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 98/100: Train Loss = 0.6585, Val Loss = 0.7230
Train Accuracy = 0.9500, Val Accuracy = 0.8400
Epoch 99/100: Train Loss = 0.6527, Val Loss = 0.7240
Train Accuracy = 0.9400, Val Accuracy = 0.8600
Epoch 100/100: Train Loss = 0.6774, Val Loss = 0.7257
Train Accuracy = 0.9500, Val Accuracy = 0.8600
# Plot the training and validation loss
import matplotlib.pyplot as plt

plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
No description has been provided for this image
No description has been provided for this image
Both training and validation loss and accuracy show improvement, indicating effective learning. However, around the 40th epoch, we observe a stabilization in performance, where further training does not significantly improve the model's accuracy.
Part (ii): Inference
def inference(model, data_loader, device):
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
    return predictions, labels
from sklearn.metrics import classification_report
train_predictions, train_labels = inference(model, train_dl, device)
val_predictions, val_labels = inference(model, val_dl, device)

# Compute classification report for train and val datasets
train_report = classification_report(train_labels, train_predictions)
val_report = classification_report(val_labels, val_predictions)

print("Classification Report for Train Data:")
print(train_report)
print("Classification Report for Validation Data:")
print(val_report)
Classification Report for Train Data:
              precision    recall  f1-score   support

           0       0.92      1.00      0.96        35
           1       1.00      0.94      0.97        31
           2       0.94      0.91      0.93        34

    accuracy                           0.95       100
   macro avg       0.95      0.95      0.95       100
weighted avg       0.95      0.95      0.95       100

Classification Report for Validation Data:
              precision    recall  f1-score   support

           0       0.74      1.00      0.85        17
           1       0.94      1.00      0.97        16
           2       1.00      0.59      0.74        17

    accuracy                           0.86        50
   macro avg       0.89      0.86      0.85        50
weighted avg       0.89      0.86      0.85        50

# Save the model checkpoint
torch.save(model.state_dict(), 'tcn_model_ex3_150.pth')
# testdata = CustomDataset(r"test_npz_files")

# # Create DataLoader for test data
# test_dl = DataLoader(testdata, batch_size=4, shuffle=True)
for i in range(3):
    print(f"Number of data in label {i}: {len([label for _, label in test_data if label == i])}")
Number of data in label 0: 8
Number of data in label 1: 11
Number of data in label 2: 13
test_dl = DataLoader(test_data, batch_size=4, shuffle=True)
# test accuracy
model.load_state_dict(torch.load('saved_models_150\model_e39_l0.7096.pth'))

test_acc = get_accuracy(model, test_dl, device)
print(f'Test Accuracy = {test_acc:.4f}')
Test Accuracy = 0.9375
from sklearn.metrics import classification_report

# Assuming your model is named 'model' and you have a test DataLoader named 'test_dl'
# Iterate over test DataLoader to make predictions
model.eval()
all_predictions = []
all_labels = []
misclassified_samples = []  # Initialize list to store misclassified samples
misclassified_labels = []
misclassified_predictions = []

with torch.no_grad():
    for inputs, labels in test_dl:
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(inputs.shape)
        # Forward pass
        outputs = model(inputs)

        # Convert outputs to predicted labels (e.g., using argmax)
        _, predicted = torch.max(outputs, 1)


        # Append predicted labels and ground truth labels to lists
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                misclassified_samples.append(inputs[i].cpu().numpy())
                misclassified_labels.append(labels[i].cpu().item())
                misclassified_predictions.append(predicted[i].cpu().item())

# Compute classification report
report = classification_report(all_labels, all_predictions)
print(report)
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
torch.Size([4, 150, 17, 2])
              precision    recall  f1-score   support

           0       0.89      1.00      0.94         8
           1       0.92      1.00      0.96        11
           2       1.00      0.85      0.92        13

    accuracy                           0.94        32
   macro avg       0.94      0.95      0.94        32
weighted avg       0.94      0.94      0.94        32

Looking at the classification report: The averages metrics for the training data are all around 95%, indicating balanced performance across classes. While precision and recall for Deadlift and Snatch classes are relatively high in the validation data, there is a notable drop in recall for the Squat class, which is reflected in the lower overall accuracy. The overall accuracy of test data is 94%, indicating that the model performs well on unseen data.
Part(iii): Confusion matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(all_labels)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()
print(conf_matrix)
No description has been provided for this image
[[ 8  0  0]
 [ 0 11  0]
 [ 1  1 11]]
The confusion matrix shows that the model made very few misclassifications, with only one misclassification each between Deadlift and Squat, and between Snatch and Squat.
Part E: Qualitative analysis
The following code identifies missclassified examples.

for sample, true_label, pred in zip(misclassified_samples, misclassified_labels, misclassified_predictions):
    print("True Label:", true_label)
    print("Predicted Label:", pred)
    print("Misclassified Sample Data:", sample)
    print("-" * 50)  # Separator for clarity
True Label: 2
Predicted Label: 0
Misclassified Sample Data: [[[163.04504 247.69788]
  [150.63751 247.08813]
  [148.08594 243.51233]
  ...
  [512.04663 236.25818]
  [619.9007  141.52075]
  [690.65796 229.93787]]

 [[167.45667 248.56903]
  [154.38525 247.70868]
  [152.05103 243.80548]
  ...
  [511.8587  235.70786]
  [619.2172  141.8844 ]
  [691.8187  230.76877]]

 [[167.61322 248.82315]
  [154.43506 248.72937]
  [151.69696 243.13748]
  ...
  [514.2989  236.63312]
  [620.2706  140.5285 ]
  [692.77734 230.45447]]

 ...

 [[162.51459 255.49597]
  [150.3169  256.43896]
  [148.19415 252.41568]
  ...
  [518.69916 224.49457]
  [625.7087  146.04117]
  [698.16284 236.8689 ]]

 [[168.0498  253.41895]
  [154.55255 254.74023]
  [153.02356 250.82687]
  ...
  [518.127   222.57867]
  [626.5542  145.90308]
  [700.8999  237.24112]]

 [[165.03467 252.67194]
  [151.57104 253.08728]
  [150.35736 249.75555]
  ...
  [516.67554 222.57416]
  [628.8982  147.12186]
  [699.29004 236.53476]]]
--------------------------------------------------
True Label: 2
Predicted Label: 1
Misclassified Sample Data: [[[336.35303 239.77917]
  [328.2533  247.48834]
  [327.53558 234.30634]
  ...
  [580.0405  203.82379]
  [669.0034  280.0421 ]
  [665.0515  198.03244]]

 [[337.58264 239.74304]
  [331.44165 246.96698]
  [330.3036  234.85297]
  ...
  [578.664   203.52563]
  [669.28394 278.60815]
  [664.4302  197.21906]]

 [[340.74536 237.41382]
  [333.04492 244.77539]
  [332.24744 232.67432]
  ...
  [576.2775  203.88733]
  [670.91284 278.47888]
  [664.01807 197.4635 ]]

 ...

 [[336.4026  235.0831 ]
  [330.78943 243.10886]
  [328.9685  230.31613]
  ...
  [577.27484 197.9248 ]
  [669.4238  278.46655]
  [664.89014 196.82068]]

 [[336.4026  235.0831 ]
  [330.78943 243.10886]
  [328.9685  230.31613]
  ...
  [577.27484 197.9248 ]
  [669.4238  278.46655]
  [664.89014 196.82068]]

 [[336.4026  235.0831 ]
  [330.78943 243.10886]
  [328.9685  230.31613]
  ...
  [577.27484 197.9248 ]
  [669.4238  278.46655]
  [664.89014 196.82068]]]
--------------------------------------------------
import os

# Directory containing .npz files
npz_dir = 'new_keypoints_30_03'

# List to store the paths of matched .npz files
matched_files = []

# Iterate over directories containing .npz files
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

            # Compare npz_data with misclassified data
            # Assuming misclassified_data is a list of misclassified data
            for data, lab, pre in zip(misclassified_samples, misclassified_labels, misclassified_predictions):
                # Check if data matches with npz_data
                if np.all(npz_data == data):
                    matched_files.append(npz_path)
                    print("True Label:", lab, "Predicted Label:", pre)
                    break  # Exit loop if match is found

# Print paths of matched .npz files
for file in matched_files:
    print("Matched .npz file:", file)
True Label: 2 Predicted Label: 1
True Label: 2 Predicted Label: 0
Matched .npz file: new_keypoints_30_03\squats\s_12.npz
Matched .npz file: new_keypoints_30_03\squats\s_42.npz
The two following squat videos are missclassified:
Picture2.gif Picture1.gif

1. The first video has an actual Label of Squat and a predicted Label of Snatch. This could be because the observed temporal dynamics closely resemble the characteristic motion pattern of a snatch exercise, wherein the arms move from a lower to an upper position in a similar manner. This similarity in movement trajectory likely led to the misclassification.
The second video has an actual Label of Squat as well but a predicted Label of Deadlift. A possible explanation of missclassification is the movement captured in this data closely resembles the mechanics of a deadlift, characterized by a consistent downward arm position throughout the action. This similarity in motion likely led to the misclassification.
Part F: Performance on New Data##
The model interface follows the same flow as the pipeline. Initially, a video is chosen via file browsing.

The video is processed through VitPose to generate keypoints, which can be overlaid on the original video.
This video is correctly classified as a deadlift and all keypoints are accurately inferred by VitPose, shot from the front with a steady camera at nearly eye level.

PHOTO

WhatsApp Image 2024-04-05 at 18.45.30_86f4a668.jpg

Now let’s look at another video taken from online.
As we can see, this is a video of a squat exercise.

We notice that this video has several obscured keypoints and a few seem to frequently switch.
We see that the video is misclassified as a result of this. This could be due to lack of such angled data in training set.

PHOTO

WhatsApp Image 2024-04-05 at 18.51.41_90a992ec.jpg

Section B
Discussion of the overall results:
Performance of the model and learnings from the project

Overall our model is working very well.

The general trend seen with each class in terms of number of frames to cover a single repetition of the exercise varies. The squat has the least number(80), while deadlift and snatch are close to 150).
We ended up padding the squat videos with last frame to match the length of deadlifts and snatches effectively extended the length of the squat videos without necessarily adding meaningful temporal information.
The addition of padded frames to the squat videos may increase the risk of overfitting, especially if the model learns to exploit patterns specific to the padding rather thanexercise movements. This might be a reason why the model is performing relatively bad with unseen squat data.

The model seems to focus on Distinct Hand Movements. Deadlifts, squats, and snatches are distinct exercises that often involve different hand movements or positions. For example, during a deadlift, the hands typically grip a weights, whereas during a squat, the hands may be free or holding onto a support. while, snatches involve rapid hand movements to lift a weight overhead. Hand movements could be key discriminative features for the model to differentiate between the different exercises .The model might be leveraging these temporal patterns to make accurate predictions.

And finally, we see that data with moving camera seemed to drastically affect model performance.
In videos with a fixed camera, consecutive frames often exhibit high temporal consistency, making it easier for the model to track the subject's pose over time. However, in videos with a moving camera, the subject's appearance may change from one frame to the next due to camera motion, making it more challenging for the model to maintain temporal consistency. This variability can make it harder for the pose estimator to accurately detect key points.

Summary of related work
There are many different methods that can be used for action classification. These include using applying CNNs on the individual images (Karpathy et al., 2014), and recurrent neural networks (Ng et al., 2015). The recurrent approaches are however more difficult to train, while the approach applying CNNs to individual frames doesn't make use of the full available spatiotemporal information.

One approach that we applied to our data to provide us with a baseline model was the use of (2+1)D CNNs, similar to the work of Tran et al. (2018). This ResNet is inspired by 3D convolutional neural networks, which have a single kernel that traverses the spatial and temporal dimensions.

The study that most closely ressembles the work that we are doing here is that by Singh et al. (2022), who applied vision transformers to violence detection, in the process developing the Video Vision Transformers (ViViT) model. This model however adapts the vision transformers directly to do the classification task, whereas we use them merely to extract keypoints that a separate model then uses to carry out classification.

Another model that is similar to ours is the Spatio-Temporal Graph Convolutional Network (ST-GCN), which is based on graph convolutional networks and is well suited for modeling actions as a sequence of poses. But due to the inherent complexity of training graph neural networks, it is not the approach we adopted for our study.

For the above mentioned reasons, we adopted the vision transformer model being used is an adaptation of VitPose (Xu et al., 2022) and build a temporal convolutional neural network on top of it.
