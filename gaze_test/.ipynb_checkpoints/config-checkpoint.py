""" This script is a pipeline for generating and processing motion-driven portrait videos using SyncNet and motion keypoint processing. It includes configuration setup, SyncNet video processing, image preparation, motion transformation, and rendering the final output. """

import os
import sys
import cv2
import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_image_rgb, prepare_source, get_kp_info, transform_keypoint, 
    extract_feature_3d, warp_decode, get_rotation_matrix
)


def get_configurations():
    """Define and return configuration values."""

    script_dir = os.path.dirname(os.path.abspath(__file__))  
    img_path = os.path.join(script_dir, "examples", "gaze_face_example2.png")
    audio_path = os.path.join(script_dir, "examples", "366_1733367920.wav")
    model_path = os.path.join(script_dir, "model_checkpoint", "norm_no_vel_ep_60.pth")
    config_path = os.path.join(script_dir, "model_checkpoint", "config.json")
    
    config = {
        "run_syncnet_only": False,
        "run_syncnet_with_motion_gen": False,
        "plot_motion": False,
        "write_vid": True,
        "sync_only_vid_root_dir": '/mnt/e/wsl_projects/LivePortrait/sync_output/2024-12-01-20-37_null_weight_1_ep_55_len_5/',
        "portrait_imgs": [
            img_path
        ],
        "audio_paths": [
            audio_path
        ],
        "model_weights_pairs": [
            (config_path, model_path),
        ],
        "cfg_scale_opts": [0.65],
        "mouth_open_ratio_opts": [0.25],
        "subtract_avg_motion_opts": [False],
        "motion_mean_exp_path": '/mnt/kostas-graid/datasets/yiliu7/liveProtrait/hdtf/train_split/audio_latent/TH_00192/000.npy',
        "offset_std": 0.002,
        "max_val": [0.018, 0.018, 0.018],
        "rest_val": [0.006, -0.002, -0.002],
        "min_val": [-0.005, -0.015, -0.015]
    }
    return config

'hdtf/train_split/audio_latent/TH_00192/000.npy'



def process_syncnet(sync_only_vid_root_dir):
    """Process SyncNet on all videos in the specified directory."""
    sync_tmp_dir = './sync_output/tmp'
    sync_tmp_dir_abs = os.path.abspath(sync_tmp_dir)
    os.makedirs(sync_tmp_dir_abs, exist_ok=True)
    print(f'Sync temporary directory: {sync_tmp_dir_abs}')

    result_dict = {}
    json_output_path = os.path.join(sync_only_vid_root_dir, 'syncnet_results.json')

    # Walk through all files in the directory
    for root, dirs, files in tqdm(os.walk(sync_only_vid_root_dir), desc="Processing directories"):
        for file in files:
            if file.endswith('.mp4'):
                vid_path = os.path.join(root, file)
                vid_path_abs = os.path.abspath(vid_path)
                print(f'Processing {vid_path_abs}')

                # Get relative path from root directory
                rel_path = os.path.relpath(vid_path, sync_only_vid_root_dir)

                # Mock SyncNet call (replace with actual function)
                s_c, s_d = call_syncnet(vid_path_abs, sync_tmp_dir_abs)
                result_dict[rel_path] = [s_c, s_d]
                print(f'{rel_path} done, confidence: {s_c}, min_dist: {s_d}')

                # Save results
                json.dump(result_dict, open(json_output_path, 'w'), indent=4)
                print(f'Results saved to {json_output_path}')




def process_image_and_motion(config, eye_exp_increnment = 0):
    """Load image, extract keypoints, and process motion."""
    image_path = config["portrait_imgs"][0]
    motion_mean_exp_path = config["motion_mean_exp_path"]
    offset_std = config["offset_std"]

    # Load and prepare image
    img_rgb = load_image_rgb(image_path)
    img_crop_256x256 = cv2.resize(img_rgb, (256, 256))
    I_s = prepare_source(img_crop_256x256)

    # keypoint info
    x_s_info = get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    x_exp = x_s_info['exp']
    x_scale = x_s_info['scale']
    x_s = transform_keypoint(x_s_info)

    # identity transforms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_identity = torch.zeros((1, 3), dtype=torch.float32, device=device)
    pitch_identity = yaw_identity = roll_identity = torch.zeros((1), dtype=torch.float32, device=device)
    scale_identity = torch.ones((1), dtype=torch.float32, device=device) * 1.5

    # Extract features
    f_s = extract_feature_3d(I_s)
    x_exp = torch.zeros_like(x_exp).reshape(-1)


    index_interval = 100

    #x_exp[4] = -0.03 + eye_exp_increnment * (0.03 - (-0.03)) / (1.0 * index4_interval)

    x_exp[33] = -0.03 + eye_exp_increnment * (0.03 - (-0.03)) / (1.0 * index_interval)
    x_exp[45] = -0.03 + eye_exp_increnment * (0.03 - (-0.03)) / (1.0 * index_interval)
    x_exp[48] = -0.03 + eye_exp_increnment * (0.03 - (-0.03)) / (1.0 * index_interval)
    
    x_exp = x_exp.reshape(1, 21, 3)

    # Transform keypoints
    x_d_i = x_scale * (x_c_s @ get_rotation_matrix(pitch_identity, yaw_identity, roll_identity) + x_exp) + t_identity

    # Add random offset to final dimension
    random_offset = torch.randn(1, device=device) * offset_std
    x_s[:, :, -1] += random_offset

    return f_s, x_s, x_d_i




def render_output(f_s, x_s, x_d_i, output_path):
    """Render the output using warp and decode, then save it."""
    out = warp_decode(f_s, x_s, x_d_i)
    out_img = (out['out'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    print(f"Output image shape: {out_img.shape}")

    cv2.imwrite(output_path, out_img[0])
    print(f"Output image saved to {output_path}")
    return out_img




if __name__ == "__main__":

    config = get_configurations()

    if config["run_syncnet_only"]:
        process_syncnet(config["sync_only_vid_root_dir"])

    script_dir = os.path.dirname(os.path.abspath(__file__))  
        
    for i in range(0, 101):
        f_s, x_s, x_d_i = process_image_and_motion(config, i)

        output_path = os.path.join(script_dir, "outputs", "img_rendered", "index_33_45_48", f"img_exp_{i:03d}.png")
        out_img = render_output(f_s, x_s, x_d_i, output_path)
