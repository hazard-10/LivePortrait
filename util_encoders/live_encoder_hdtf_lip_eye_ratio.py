'''
For parasing training set with LivePortrait latent
'''

import json
import os

import time

import os
import contextlib
import os.path as osp
import numpy as np
import cv2
import torch
import yaml
import tyro
import subprocess
from rich.progress import track
import torchvision
import cv2
import threading
import queue
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
import numpy as np
import time
import torch
import imageio
from tqdm import tqdm

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

args = ArgumentConfig()
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)
# print("inference_cfg: ", inference_cfg)
# print("crop_cfg: ", crop_cfg)
device = 'cuda'
print("Compile complete")

'''
Common modules
'''

from src.utils.helper import load_model, concat_feat
from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from src.config.inference_config import InferenceConfig
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from src.utils.crop import _transform_img, prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from src.utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image
from src.utils.filter import smooth


'''
Util functions
'''

import numpy as np

def calculate_distance_ratio(lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6) -> np.ndarray:
    return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
            (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))


def calc_eye_close_ratio(lmk: np.ndarray, target_eye_ratio: np.ndarray = None) -> np.ndarray:
    lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12)
    righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36)
    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)


def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
    return calculate_distance_ratio(lmk, 90, 102, 48, 66)

def calc_ratio(lmk_lst):
    input_eye_ratio_lst = []
    input_lip_ratio_lst = []
    for lmk in lmk_lst:
        # for eyes retargeting
        input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
        # for lip retargeting
        input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
    return input_eye_ratio_lst, input_lip_ratio_lst

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))  # Resize to 256x256
        frames.append(frame)

    cap.release()
    return video_path, frames

def read_multiple_videos(video_paths, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(read_video_frames, video_paths))
    return results

'''
Main module for inference
'''
model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)
# init F
appearance_feature_extractor = load_model(inference_cfg.checkpoint_F, model_config, device, 'appearance_feature_extractor')
# init M
motion_extractor = load_model(inference_cfg.checkpoint_M, model_config, device, 'motion_extractor')
# init W
warping_module = load_model(inference_cfg.checkpoint_W, model_config, device, 'warping_module')
# init G
spade_generator = load_model(inference_cfg.checkpoint_G, model_config, device, 'spade_generator')
# init S and R
if inference_cfg.checkpoint_S is not None and os.path.exists(inference_cfg.checkpoint_S):
    stitching_retargeting_module = load_model(inference_cfg.checkpoint_S, model_config, device, 'stitching_retargeting_module')
else:
    stitching_retargeting_module = None

cropper = Cropper(crop_cfg=crop_cfg, device=device)




def load_video(root_dir):
    video_paths = {}

    # Walk through the root directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                # Get the full path of the video
                video_path = os.path.join(root, file)

                # Extract the first level key (user ID) from the path
                first_level_key = os.path.relpath(root, root_dir).split(os.sep)[0]

                # Initialize the list for this first_level_key if it doesn't exist
                if first_level_key not in video_paths:
                    video_paths[first_level_key] = []

                # Add the video path to the list
                video_paths[first_level_key].append(video_path)

    # Sort the video paths for each first_level_key
    for first_level_key in video_paths:
        video_paths[first_level_key].sort()

    # Print the results
    print(f"Generated {len(video_paths)} video paths.")
    return video_paths

def process_single_video(frames):
    driving_rgb_lst = frames
    driving_lmk_crop_lst = cropper.calc_lmks_from_cropped_video(driving_rgb_lst, skip_target=1)
    c_d_eyes_lst, c_d_lip_lst = calc_ratio(driving_lmk_crop_lst)
    return c_d_eyes_lst, c_d_lip_lst

def process_videos(uid, video_paths, output_dir):
    for video_path in tqdm(video_paths, desc=f"Processing videos for {uid}"):
        video_frames = read_video_frames(video_path)[1]  # Get only the frames

        c_d_eyes, c_d_lip = process_single_video(video_frames)

        # Reshape and combine the data
        c_d_eyes = np.squeeze(c_d_eyes)  # Shape: (L, 2)
        c_d_lip = np.squeeze(c_d_lip)    # Shape: (L, 1)

        # Prepare the data for saving (L, 3)
        video_data = np.column_stack((c_d_eyes, c_d_lip))

        # Create the output filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{video_name}.npy")

        # Save the processed data
        np.save(output_file, video_data)
        print(f"Saved {output_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process videos for live encoding')
    parser.add_argument('--root_dir', type=str, default='/mnt/e/data/diffposetalk_data/TFHP_raw/crop/',
                        help='Root directory containing the video files')
    parser.add_argument('--output_dir', type=str, default='/mnt/e/data/diffposetalk_data/TFHP_raw/lip_eye_ratio/',
                        help='Output directory for processed files')

    args = parser.parse_args()

    video_paths = load_video(args.root_dir)
    print(video_paths)

    for uid, paths in tqdm(video_paths.items(), desc="Processing users"):
        uid_output_dir = os.path.join(args.output_dir, uid)
        os.makedirs(uid_output_dir, exist_ok=True)
        process_videos(uid, paths, uid_output_dir)
