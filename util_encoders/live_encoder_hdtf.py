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

def prepare_videos_(imgs, device):
    """ construct the input as standard
    imgs: NxHxWx3, uint8
    """
    if isinstance(imgs, list):
        _imgs = np.array(imgs)
    elif isinstance(imgs, np.ndarray):
        _imgs = imgs
    else:
        raise ValueError(f'imgs type error: {type(imgs)}')

    # y = _imgs.astype(np.float32) / 255.
    y = _imgs
    y = torch.from_numpy(y).permute(0, 3, 1, 2)  # NxHxWx3 -> Nx3xHxW
    y = y.to(device)
    y = y / 255.
    y = torch.clamp(y, 0, 1)

    return y

def get_kp_info(x: torch.Tensor, **kwargs) -> dict:
    """ get the implicit keypoint information
    x: Bx3xHxW, normalized to 0~1
    flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
    return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
    """
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16,
                                 enabled=inference_cfg.flag_use_half_precision):
        kp_info = motion_extractor(x)

        if inference_cfg.flag_use_half_precision:
            # float the dict
            for k, v in kp_info.items():
                if isinstance(v, torch.Tensor):
                    kp_info[k] = v.float()

    flag_refine_info: bool = kwargs.get('flag_refine_info', True)
    if flag_refine_info:
        bs = kp_info['kp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
        kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
        kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
        # kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        # kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

    return kp_info

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

    # video_paths = {}

    # # Iterate through the JSON structure
    # for first_level_key in sorted(data.keys()):
    #     video_paths[first_level_key] = []
    #     for second_level_key in sorted(data[first_level_key].keys()):
    #         for third_level_key in sorted(data[first_level_key][second_level_key]):
    #             third_level_key = third_level_key.split('.')[0]
    #             # Construct the video path
    #             video_path = os.path.join(root_dir, first_level_key, second_level_key, f"{third_level_key}.mp4")
    #             video_paths[first_level_key].append(video_path)

    #     # Sort the video paths for each first_level_key
    #     video_paths[first_level_key].sort()

    # Clear existing video_paths
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

def process_videos(uid, video_paths, output_dir, param_list, param_dim_list):
    read_2_gpu_batch_size = 2048
    gpu_batch_size = 16
    process_queue = torch.Tensor().to(device)

    video_frames = read_multiple_videos(video_paths, num_threads=4)
    all_frames = []
    total_frames = 0
    video_lengths = []
    vid_keys = []

    for video_path, frames in video_frames:
        all_frames.extend(frames)
        frame_count = len(frames)
        total_frames += frame_count
        video_lengths.append(frame_count)
        clip_id = video_path
        vid_keys.append(f"{clip_id.split('/')[-1].split('.')[0]}")

    all_frames = np.array(all_frames)

    read_data_2_gpu_pointer = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {uid}")

    while read_data_2_gpu_pointer < total_frames:
        current_batch_size = min(read_2_gpu_batch_size, total_frames - read_data_2_gpu_pointer)

        batch_input = all_frames[read_data_2_gpu_pointer:read_data_2_gpu_pointer + current_batch_size]
        batch_input = prepare_videos_(batch_input, device)

        mini_batch_start = 0
        all_info = []
        while mini_batch_start < batch_input.shape[0]:
            mini_batch_end = min(mini_batch_start + gpu_batch_size, batch_input.shape[0])
            mini_batch = batch_input[mini_batch_start:mini_batch_end]

            x_info = get_kp_info(mini_batch)

            concat_tensor = torch.cat([
                x_info['kp'], # 63
                x_info['exp'], # 63, .reshape(mini_batch_end - mini_batch_start, -1),
                x_info['t'], # 3
                x_info['pitch'], # 1
                x_info['yaw'], # 1
                x_info['roll'], # 1
                x_info['scale'], # 1
            ], dim=1)

            all_info.append(concat_tensor)

            mini_batch_start = mini_batch_end
        all_info_tensor = torch.cat(all_info, dim=0)

        process_queue = torch.cat((process_queue, all_info_tensor), dim=0)

        while len(vid_keys) > 0 and len(process_queue) >= video_lengths[0]:
            current_vid_key = vid_keys[0]
            current_frame_count = video_lengths[0]

            video_tensor = process_queue[:current_frame_count]
            save_path = os.path.join(output_dir, f"{current_vid_key}.npy")
            print(f"Saving {save_path}")
            np.save(save_path, video_tensor.cpu().numpy())

            process_queue = process_queue[current_frame_count:]
            vid_keys.pop(0)
            video_lengths.pop(0)

        read_data_2_gpu_pointer += current_batch_size
        pbar.update(current_batch_size)

    pbar.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process videos for live encoding')
    parser.add_argument('--root_dir', type=str, default='/mnt/e/data/diffposetalk_data/TFHP_raw/crop/',
                        help='Root directory containing the video files')
    parser.add_argument('--output_dir', type=str, default='/mnt/e/data/diffposetalk_data/TFHP_raw/live_latent/',
                        help='Output directory for processed files')

    args = parser.parse_args()

    video_paths = load_video(args.root_dir)
    print(video_paths)

    param_list = ['kp', 'exp', 't', 'pitch', 'yaw', 'roll', 'scale']
    param_dim_list = [63, 63, 3, 1, 1, 1, 1]
    # for p in param_list:
    #     os.makedirs(os.path.join(args.output_dir, p), exist_ok=True)

    for uid, paths in tqdm(video_paths.items(), desc="Processing users"):
        uid_output_dir = os.path.join(args.output_dir, uid)
        os.makedirs(uid_output_dir, exist_ok=True)
        process_videos(uid, paths, uid_output_dir, param_list, param_dim_list)
