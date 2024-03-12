import argparse
import os.path as osp
import os

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm 

from time import time
from PIL import Image
from torchvision.transforms import transforms

from models.model import ViTPose
from utils.top_down_eval import keypoints_from_heatmaps

from configs.ViTPose_base_coco_256x192 import model as model_cfg
from configs.ViTPose_base_coco_256x192 import data_cfg

__all__ = ['inference']
            
            
@torch.no_grad()
def inference(img_path: Path, model_cfg: dict, ckpt_path: Path, device: torch.device) -> np.ndarray:
    
    # Prepare model
    vit_pose = ViTPose(model_cfg)
    
   
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)
    # print(f">>> Model loaded: {ckpt_path}")
    
    # Prepare input data
    img = Image.open(img_path)
    org_w, org_h = img.size
    # print(f">>> Original image size: {org_h} X {org_w} (height X width)")
    # print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
    # print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")
    img_tensor = transforms.Compose (
        [transforms.Resize((img_size[1], img_size[0])),
         transforms.ToTensor()]
    )(img).unsqueeze(0).to(device)
    
    # Feed to model
    # tic = time()
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
    # elapsed_time = time()-tic
    # print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
    
    # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                           unbiased=True, use_udp=True)
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    
    # Visualization 
    # for pid, point in enumerate(points):
        
    #     img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
    #     img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
    #                                     points_color_palette='gist_rainbow', skeleton_color_palette='jet',
    #                                     points_palette_samples=10, confidence_threshold=0.4)
    #     save_name = img_path.replace(".jpg", "_result.jpg")
    #     cv2.imwrite(save_name, img)
    #     print("Saved at", save_name)

    return points
    

# if __name__ == "__main__":
#     from configs.ViTPose_base_coco_256x192 import model as model_cfg
#     from configs.ViTPose_base_coco_256x192 import data_cfg
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image-path', nargs='+', type=str, default='examples/sample.jpg', help='image path(s)')
#     args = parser.parse_args()
    
#     CUR_DIR = osp.dirname(__file__)
#     # CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
#     CKPT_PATH = "vitpose-b-multi-coco.pth"
    
#     img_size = data_cfg['image_size']
#     if type(args.image_path) != list:
#          args.image_path = [args.image_path]
#     for img_path in args.image_path:
#         print(img_path)
#         keypoints = inference(img_path=img_path, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
#                               device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
#                               save_result=True)

def main(image_folder, output_folder):
    # CUR_DIR = os.path.dirname(__file__)
    CKPT_PATH = "vitpose-b-multi-coco.pth"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # for folder_name in tqdm(sorted(os.listdir(image_folder))):

    #     # print("hi")

    #     folder_path = os.path.join(image_folder, folder_name)
    #     if os.path.isdir(folder_path):
    #         print(f"Processing folder: {folder_name}")
    
    kp_f = []
    for idx, filename in tqdm(enumerate(sorted(os.listdir(image_folder)))):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            keypoints = inference(img_path=image_path, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                                    device=torch.device("cuda") if torch.cuda.is_available()) else torch.device('cpu'),)
            
            keypoints = keypoints.squeeze()

            # kp_neck = (keypoints[5]+keypoints[6])/2

            # keypoints = np.append(keypoints, kp_neck)

            # keypoints = keypoints.reshape(18,3)
            
            kp_f.append(keypoints)
    
    output_path = os.path.join(output_folder, "output.npz")
    print(output_path)

    # print("Saving in {}".format(output_path))
    np.savez(output_path, reconstruction=kp_f)


# def main(video_path, output_folder):
#     CKPT_PATH = "vitpose-b-multi-coco.pth"
    
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # cap = cv2.VideoCapture(video_path)
#     # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # # fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # kp_f = []
#     # for i in tqdm(range(frame_count)):
#     for folder_name in tqdm(sorted(os.listdir(image_folder))):
#         print("hi")
#         ret, frame = cap.read()
#         if not ret:
#             break

#         keypoints = inference(img=frame, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
#                               device=torch.device("cuda"))
        
#         keypoints = keypoints.squeeze()

#         # kp_neck = (keypoints[5]+keypoints[6])/2
#         # keypoints = np.append(keypoints, kp_neck)
#         # keypoints = keypoints.reshape(18, 3)

#         # kp_f.append(keypoints)

#         output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")

#         np.savez(output_path, reconstruction=kp_f)
    
    # return keypoints
    # output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}.npz")

    # np.savez(output_path, reconstruction=kp_f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True, help='path to folder containing images')
    parser.add_argument('--output-folder', type=str, required=True, help='path to folder for saving keypoints')
    args = parser.parse_args()
    
    img_size = data_cfg['image_size']

    main(args.image_folder, args.output_folder)