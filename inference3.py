import os

import torch

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm 

from time import time
from PIL import Image
from torchvision.transforms import transforms

from models.model import ViTPose
from utils.top_down_eval import keypoints_from_heatmaps
from utils.visualization import draw_points_and_skeleton, joints_dict

from configs.ViTPose_base_coco_256x192 import model as model_cfg
from configs.ViTPose_base_coco_256x192 import data_cfg

import re 

__all__ = ['inference']
            
save_dir = "save_dir"

def vid_to_img(input_folder, save_folder_name):

    image_folder = os.path.join(input_folder.split(".")[0], "tracklet/")
    os.makedirs(image_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_folder)
    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("!!!!")
            break

        filename = os.path.join(image_folder, f"frame_{frame_counter}.jpg")
        cv2.imwrite(filename, frame)

        frame_counter += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return image_folder

@torch.no_grad()
def inference(img_path: Path, model_cfg: dict, ckpt_path: Path, device: torch.device, save_folder_name:Path) -> np.ndarray:
    
    vit_pose = ViTPose(model_cfg)
    
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)

    img = Image.open(img_path)
    org_w, org_h = img.size
    img_tensor = transforms.Compose (
        [transforms.Resize((img_size[1], img_size[0])),
         transforms.ToTensor()]
    )(img).unsqueeze(0).to(device)
    
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy() 
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                           unbiased=True, use_udp=True)
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    
    for pid, point in enumerate(points):
        
        img = np.array(img)[:, :, ::-1] 
        img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                        points_palette_samples=10, confidence_threshold=0.0)
        
        save_name = img_path.replace(".jpg", "_result.jpg").split("tracklet/")[1]

        output_folder = os.path.join(save_folder_name, save_dir)
        os.makedirs(output_folder, exist_ok=True)
        
        save_file = os.path.join(output_folder, save_name)
        # print(save_file, save_name, img_path)
        cv2.imwrite(save_file, img)

    return points

def main(input_folder, i):
    # CKPT_PATH = "/content/drive/MyDrive/Colab Notebooks/MIE1517 Project/vitpose-b-multi-coco.pth"
    CKPT_PATH = r"C:\Users\Akshith\Documents\GitHub\ViTPose\vitpose-b-multi-coco.pth"

    save_folder_name = input_folder.split(".")[0]
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)


    image_folder = vid_to_img(input_folder, save_folder_name)

    kp_f = []
    sorted_images = sorted(os.listdir(image_folder), key=lambda f: int(re.findall(r'\d+', f)[-1]))

    for _, filename in tqdm(enumerate(sorted_images)):
        image_path = os.path.join(image_folder, filename)
        keypoints = inference(img_path=image_path, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                                device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),save_folder_name=save_folder_name)
        
        keypoints = keypoints.squeeze()
        
        kp_f.append(keypoints)
    
    output_path = os.path.join(save_folder_name, f"d_{i}.npz")

    np.savez(output_path, reconstruction=kp_f)


if __name__ == "__main__":

    for i in range(1,52):
        input_folder = f"deadlift\d_{i}.mp4"
        img_size = data_cfg['image_size']
        print("video:",i)
        main(input_folder, i)
    # input_folder = "v140.mp4"
    # img_size = data_cfg['image_size']

    # main(input_folder)