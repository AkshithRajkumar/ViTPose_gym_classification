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

While choosing the videos various aspects have been considered based on our learnings in applying the models on videos. Videos with multiple people, objects have not been chosen. Videos with similar angle of recording were chosen. The start and end pose of the each exercises were determined and the videos are cropped to start and end at those poses.

After downloading the videos, we cropped them around the person so that only the human was left on the frames. We also cropped the video length to one repetition.


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
