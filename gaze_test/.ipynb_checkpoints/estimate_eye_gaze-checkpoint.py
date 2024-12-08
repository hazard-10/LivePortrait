""" 
This script implements a pipeline for estimating human eye gaze, processing motion, 
and visualizing motion data over time. It includes functionalities for model initialization, 
configuration management, and gaze estimation using the L2CSNet model. 
"""
import os
import sys
import cv2
import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Third-party libraries
from l2cs import Pipeline, render

# Project-specific imports
from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from audio_dit.dataset import process_motion_tensor

from utils import partial_fields, reverse_normalize_pose
from utils import load_and_process_audio, load_model, prepare_source


def load_configs():
    """Load and return configurations."""
    args = ArgumentConfig()
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    return args, inference_cfg, crop_cfg


def initialize_models(inference_cfg, device):
    """Initialize models using configurations."""
    model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)

    # Initialize models
    appearance_feature_extractor = load_model(inference_cfg.checkpoint_F, model_config, device, 'appearance_feature_extractor')
    motion_extractor = load_model(inference_cfg.checkpoint_M, model_config, device, 'motion_extractor')
    warping_module = load_model(inference_cfg.checkpoint_W, model_config, device, 'warping_module')
    spade_generator = load_model(inference_cfg.checkpoint_G, model_config, device, 'spade_generator')

    # Optional stitching/retargeting module
    if inference_cfg.checkpoint_S is not None and os.path.exists(inference_cfg.checkpoint_S):
        stitching_retargeting_module = load_model(inference_cfg.checkpoint_S, model_config, device, 'stitching_retargeting_module')
    else:
        stitching_retargeting_module = None

    return appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module


def process_motion(motion_mean_exp_path, audio_model_config, headpose_bound_list, device, normalize_pose_flag=True):
    """Process motion tensor and return processed motion."""
    motion_mean_exp_tensor = torch.from_numpy(np.load(motion_mean_exp_path)).to(device=device)
    motion_mean_exp_tensor = motion_mean_exp_tensor.unsqueeze(0)

    motion_tensor, _, _, _ = process_motion_tensor(
        motion_mean_exp_tensor,
        None,
        latent_type=audio_model_config['motion_latent_type'],
        latent_mask_1=audio_model_config['latent_mask_1'],
        latent_bound=torch.tensor(audio_model_config['latent_bound'], device=device),
        use_headpose=True,
        headpose_bound=torch.tensor(headpose_bound_list, device=device)
    )

    if normalize_pose_flag:
        motion_tensor = reverse_normalize_pose(motion_tensor, headpose_bound=torch.tensor(headpose_bound_list, device=device))

    return motion_tensor


def plot_dimensions_over_time(motion_tensor, generated_motion_to_plot, dims_range, title, ylabel_prefix, output_path):
    """Plot specified dimensions over time."""
    last_dims = motion_tensor[0, :, dims_range].cpu().numpy()
    generated_last_dims = generated_motion_to_plot[0, :, dims_range].cpu().numpy()

    time_steps = np.arange(last_dims.shape[0])
    fig, axes = plt.subplots(len(dims_range), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title)

    for i, dim_idx in enumerate(dims_range):
        axes[i].plot(time_steps, last_dims[:, i], label='Ground Truth')
        axes[i].plot(time_steps, generated_last_dims[:, i], alpha=0.5, label='Generated')
        axes[i].set_ylabel(f'{ylabel_prefix} {dim_idx}')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()


def estimate_gaze(out_img, pipeline_weights_path, device='cuda'):
    """Estimate gaze pitch and yaw from an image."""
    gaze_pipeline = Pipeline(
        weights=pipeline_weights_path,
        arch='ResNet50',
        device=device
    )
    frame = out_img[0]
    results = gaze_pipeline.step(frame)

    gaze_pitch = results.pitch * 180 / np.pi  # Convert radians to degrees
    gaze_yaw = results.yaw * 180 / np.pi      # Convert radians to degrees

    return gaze_pitch, gaze_yaw


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args, inference_cfg, crop_cfg = load_configs()
    models = initialize_models(inference_cfg, device)

    #config_path = inference_cfg.models_config
    config_path = os.path.join(script_dir, "model_checkpoint", "config.json")

    weight_path = inference_cfg.checkpoint_F
    with open(config_path, 'r') as f:
        audio_model_config = yaml.safe_load(f)

    # Process motion
    headpose_bound_list = [-21, 25, -30, 30, -23, 23, -0.3, 0.3, -0.3, 0.28]
    motion_mean_exp_path = "/mnt/kostas-graid/datasets/yiliu7/liveProtrait/hdtf/train_split/audio_latent/TH_00192/000.npy"
    motion_tensor = process_motion(motion_mean_exp_path, audio_model_config, headpose_bound_list, device)

    # Example: Plot motion dimensions
    dims_to_plot = [-5, -4, -3, -2, -1]
    
    output_path = os.path.join(script_dir, "outputs", "gaze_extraction", "last5dim_vs_t.png")
    
    plot_dimensions_over_time(
        motion_tensor,
        generated_motion_to_plot=motion_tensor,  # Replace with actual generated motion
        dims_range=dims_to_plot,
        title="Last 5 Dimensions Over Time",
        ylabel_prefix="Dim", 
        output_path=output_path
    )

    
    # Gaze estimation example
    img_path = os.path.join(script_dir, "examples", "gaze_face_example1.png")
    
    out_img = [cv2.imread(img_path)]  
    pipeline_weights_path = os.path.join(script_dir, "L2CS-Net", "models", "L2CSNet_gaze360.pkl")

    gaze_pitch, gaze_yaw = estimate_gaze(out_img, pipeline_weights_path, device)
    print(f"Gaze Pitch: {gaze_pitch}, Gaze Yaw: {gaze_yaw}")
