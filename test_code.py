import os
from datasets.JointsDataset import BaseballDataset
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':

    json_dir = "/home/jbright/jb_ws/PitcherNet/dataset/pose-estimation/data_json_with2d/"
    img_dir =  "/home/jbright/jb_ws/PitcherNet/dataset/pose-estimation/data_img/"
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.406, 0.456], std=[0.229, 0.224, 0.225]),
    ])
    dataset = BaseballDataset(img_dir, json_dir, transform)
    # print(len(x))

    train_len = int(0.75 * len(dataset))
    val_len = int(0.20 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    datasets_train, datasets_valid, datasets_test = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    cnt = 0
    for i in datasets_valid:
        img = cv2.imread(i[3]['image'])
        plt.imshow(i[0].permute(1,2,0))
        for [x,y,_] in i[3]['joints'][5:]:
            plt.scatter(x,y,color='red')
        plt.show()
        break