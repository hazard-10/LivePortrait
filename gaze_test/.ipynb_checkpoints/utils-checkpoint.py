# standard
import contextlib
import glob
import json
import os
import sys
import os.path as osp
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# thirdparty
import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchaudio
import torchvision
import torchvision.transforms as transforms
import tyro
import yaml
from rich.progress import track
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from scipy.signal import savgol_filter

# project-specific
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
from src.utils.cropper import Cropper
from src.utils.helper import load_model
from src.utils.io import load_image_rgb
from audio_dit.dataset import load_and_process_pair
from audio_dit.dataset import process_motion_tensor
from audio_dit.inference import InferenceManager, get_model

# syncnet inference function
from syncnet.syncnet import syncnet_inference


from gaze_extraction import initialize_pipeline, process_img


# constants
MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000
FRAME_RATE = 25
SECTION_LENGTH = 3
OVERLAP = 10

DB_ROOT = 'vox2-audio-tx'
LOG = 'log'
AUDIO = 'audio/audio'
OUTPUT_DIR = 'audio_encoder_output'
BATCH_SIZE = 16

prev_context_len = 67
gen_len_per_window = 8


run_syncnet_with_motion_gen = False
plot_motion = False
write_vid = True


headpose_bound_list = [
    -21,                   25,                   -30,                   30,                  -23,                     23,
    -0.3,                  0.3,                  -0.3,                 0.28,                ]
headpose_loss_weight = [0.01, 0.01, 0.01, 0.01, 0.1, 0.1]


index4_interval = 10
index33_interval = 10


max_val = [0.02, 0.018, 0.018, 0.018] 
rest_val = [0, 0.006, -0.002, -0.002]
min_val = [-0.02, -0.005, -0.015, -0.015]


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})
    

def load_configs():
    """Load and return configurations."""
    args = ArgumentConfig()
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    return args, inference_cfg, crop_cfg

device = 'cuda'
args = ArgumentConfig()
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)
cropper = Cropper(crop_cfg=crop_cfg, device=device)



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


# Integrate the initialization of configs and models into the main script
args, inference_cfg, crop_cfg = load_configs()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module = initialize_models(inference_cfg, device)

# Load Wav2Vec2 models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


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
        kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

    return kp_info


def prepare_source(img: np.ndarray) -> torch.Tensor:
    """ construct the input as standard
    img: HxWx3, uint8, 256x256
    """
    h, w = img.shape[:2]
    x = img.copy()

    if x.ndim == 3:
        x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
    elif x.ndim == 4:
        x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
    else:
        raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
    x = np.clip(x, 0, 1)  # clip to 0~1
    x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
    x = x.to(device)
    return x



def warp_decode(feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """ get the image after the warping of the implicit keypoints
    feature_3d: Bx32x16x64x64, feature volume
    kp_source: BxNx3
    kp_driving: BxNx3
    """
    # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16,
                                 enabled=inference_cfg.flag_use_half_precision):
        # get decoder input
        ret_dct = warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
        # decode
        ret_dct['out'] = spade_generator(feature=ret_dct['out'])

    return ret_dct



def extract_feature_3d( x: torch.Tensor) -> torch.Tensor:
    """ get the appearance feature of the image by F
    x: Bx3xHxW, normalized to 0~1
    """
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16,
                                 enabled=inference_cfg.flag_use_half_precision):
        feature_3d = appearance_feature_extractor(x)

    return feature_3d.float()



def transform_keypoint(kp_info: dict):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = kp_info['kp']    # (bs, k, 3)
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    t, exp = kp_info['t'], kp_info['exp']
    scale = kp_info['scale']

    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)

    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
    else:
        num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed




# Move model and processor to global scope
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

def load_and_process_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    original_sample_rate = sample_rate

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # print(file_path," waveform.shape ",waveform.shape)

    # Calculate section length and overlap in samples
    section_samples = SECTION_LENGTH * 16027
    overlap_samples = int(OVERLAP / FRAME_RATE * TARGET_SAMPLE_RATE)
    # print('section_samples',section_samples,'overlap_samples',overlap_samples)

    # pad 10 overlap at the beginning
    waveform = torch.nn.functional.pad(waveform, (overlap_samples, 0))
    # Pad if shorter than 3 seconds
    if waveform.shape[1] < section_samples:
        waveform = torch.nn.functional.pad(waveform, (0, section_samples - waveform.shape[1]))
        return [waveform.squeeze(0)], original_sample_rate

    # Split into sections with overlap
    sections = []
    start = 0

    # print('starting to segment', file_path)
    while start < waveform.shape[1]:
        end = start + section_samples
        if end >= waveform.shape[1]:
            tmp=waveform[:, start:min(end, waveform.shape[1])]
            tmp = torch.nn.functional.pad(tmp, (0, section_samples - tmp.shape[1]))
            sections.append(tmp.squeeze(0))
            # print(tmp.shape)
            break
        else:
            sections.append(waveform[:, start:min(end, waveform.shape[1])].squeeze(0))

        start = int(end - overlap_samples)


    return file_path, sections



def inference_process_wav_file(path):
    audio_path, segments = load_and_process_audio(path)
    # print(audio_path,segments)
    segments = np.array(segments)

    inputs = wav2vec_processor(segments, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(device)

    with torch.no_grad():
        outputs = wav2vec_model(inputs)
        latent = outputs.last_hidden_state

        seq_length = latent.shape[1]
        new_seq_length = int(seq_length * (FRAME_RATE / 50))

        latent_features_interpolated = F.interpolate(latent.transpose(1,2),
                                                     size=new_seq_length,
                                                     mode='linear',
                                                     align_corners=True).transpose(1,2)
    return latent_features_interpolated



def autoregress_load_and_process_audio(file_path):
    first_segment_prev_length = 10
    first_segment_main_length = 65
    remaining_segment_prev_length = prev_context_len
    remaining_segment_main_length = gen_len_per_window

    # below is the same as load_and_process_audio
    waveform, og_sample_rate = torchaudio.load(file_path)

    if og_sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, og_sample_rate, TARGET_SAMPLE_RATE)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # define sample count
    per_window_samples = SECTION_LENGTH * 16027
    first_prev_samples = int(first_segment_prev_length * 16027 / FRAME_RATE)
    remaining_overlap_samples = int(remaining_segment_prev_length / FRAME_RATE * TARGET_SAMPLE_RATE)
    # pad 10 overlap at the beginning
    waveform = torch.nn.functional.pad(waveform, (first_prev_samples, 0))

    # split into windows with overlap
    windows = []
    start = 0

    total_sample_count = waveform.shape[1]
    while start < total_sample_count:
        end = start + per_window_samples
        if end >= total_sample_count: # need to pad since last exceeds total sample count
            tmp = waveform[:, start:min(end, total_sample_count)]
            tmp = torch.nn.functional.pad(tmp, (0, per_window_samples - tmp.shape[1]))
            windows.append(tmp.squeeze(0))
            break
        else:
            windows.append(waveform[:, start:min(end, total_sample_count)].squeeze(0))
        start = int(end - remaining_overlap_samples)

    return windows



def autoregress_inference_process_wav_file(path):
    windows = autoregress_load_and_process_audio(path)
    windows = np.array(windows)

    inputs = wav2vec_processor(windows, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(device)
    with torch.no_grad():
        outputs = wav2vec_model(inputs)
        latent = outputs.last_hidden_state

        seq_length = latent.shape[1]
        new_seq_length = int(seq_length * (FRAME_RATE / 50))

        latent_features_interpolated = F.interpolate(latent.transpose(1,2),
                                                     size=new_seq_length,
                                                     mode='linear',
                                                     align_corners=True).transpose(1,2)
    return latent_features_interpolated



def process_motion_batch(gen_motion_batch, f_s, x_s, x_c_s, x_s_info, audio_model_config, warp_decode_func):
    frames = []
    B, feat_count = gen_motion_batch.shape
    full_motion = gen_motion_batch.reshape(B, feat_count)
    # full_motion = torch.cat([motion_prev[0], full_motion], dim=0)
    if audio_model_config['use_headpose']:
        pose = full_motion[:, -5:]
        exp = full_motion[:, :-5]
        
    t_identity = torch.zeros((1, 3), dtype=torch.float32, device=device)
    pitch_identity = torch.zeros((1), dtype=torch.float32, device=device)
    yaw_identity = torch.zeros((1), dtype=torch.float32, device=device)
    roll_identity = torch.zeros((1), dtype=torch.float32, device=device)
    scale_identity = torch.ones((1), dtype=torch.float32, device=device) * 1.5

    if not audio_model_config['use_headpose']:
        t_s = x_s_info['t']
        pitch_s = x_s_info['pitch'] - 10
        yaw_s = yaw_identity
        roll_s = roll_identity
        scale_s = x_s_info['scale']

    full_63_exp = torch.zeros(full_motion.shape[0], 63, device=device)
    for i, dim in enumerate(audio_model_config['latent_mask_1']):
        # print(i, dim)
        full_63_exp[:, dim] = exp[:, i]
    full_motion = full_63_exp.reshape(-1, 63)

    x_d_list = []

    for i in tqdm(range(full_motion.shape[0]), desc="Generating x_d"):
        motion = full_motion[i].reshape(21, 3)
        pose_i = pose[i].unsqueeze(0)

        # # Extract values from motion
        exp = motion
        if audio_model_config['use_headpose']:
            pitch, yaw, roll, t_x, t_y = pose_i.unbind(-1)  # or pose_i.tolist()
            t = torch.tensor([t_x, t_y, 0], device=device)
            scale = x_s_info['scale']
            # print("pitch, yaw, roll shape", pitch.shape, yaw.shape, roll.shape, pose_i.shape)
        else:
            t = t_s
            pitch = pitch_s
            yaw = yaw_s
            roll = roll_s
            scale = scale_s
            t = torch.tensor(t, device=device)

    
        x_d_i = scale * (x_c_s @ get_rotation_matrix(pitch, yaw, roll) + exp) + t

        curr_frame = (warp_decode_func(f_s, x_s, x_d_i)['out'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        curr_frame_image = Image.fromarray(curr_frame[0])
        curr_frame_image.save('curr_frame.png')
        
        gaze_pipeline = initialize_pipeline()
        gaze_yaw, gaze_pitch = process_img('curr_frame.png', gaze_pipeline)

        print("gaze yaw: ", gaze_yaw)
        print("gaze pitch: ", gaze_pitch)

        degree_pitch = pitch[0].cpu().numpy() 
        degree_yaw = yaw[0].cpu().numpy() 
        rad_pitch = degree_pitch * 3.1415926 / 180.
        rad_yaw = degree_yaw * 3.1415926 / 180.
        # print("head pose yaw: ", rad_yaw)
        # print("head pose pitch: ", rad_pitch)

        exp_idx4 = (rad_pitch - 0.04) / 1.14
        exp_idx33_45_48 = (rad_yaw - 0.02) / -2.2

        pitch_data = pd.DataFrame({
            "Degree Pitch": [degree_pitch],
            "exp_idx4": [exp_idx4]
        })


        yaw_data = pd.DataFrame({
            "Degree Yaw": [degree_yaw],
            "exp_idx33_45_48": [exp_idx33_45_48]
        })

        pitch_file = "degree_pitch_vs_exp_idx4.csv"
        pitch_data.to_csv(pitch_file, mode='a', header=not pd.io.common.file_exists(pitch_file), index=False)

        yaw_file = "degree_yaw_vs_exp_idx33_45_48.csv"
        yaw_data.to_csv(yaw_file, mode='a', header=not pd.io.common.file_exists(yaw_file), index=False)

        # exp_idx4 = (gaze_pitch - 0.04) / 1.14
        # exp_idx33_45_48 = (gaze_pitch - 0.02) / -2.2

        # print("print(exp_idx4): ", exp_idx4)
        # print("print(exp_idx33_45_48): ", exp_idx33_45_48)

        # exp[1, 1] = exp_idx4 * 0.1
        exp[11, 0] = -exp_idx33_45_48
        exp[15, 0] = -exp_idx33_45_48
        exp[16, 0] = -exp_idx33_45_48

        indices = [(1, 1), (11, 0), (15, 0), (16, 0)]
        for i, (row, col) in enumerate(indices):
            exp[row, col] = np.clip(exp[row, col].cpu().numpy(), min_val[i], max_val[i])

        x_d_i = scale * (x_c_s @ get_rotation_matrix(pitch, yaw, roll) + exp) + t
        
        x_d_list.append(x_d_i.squeeze(0))

    x_d_batch = torch.stack(x_d_list, dim=0)
    f_s_batch = f_s.expand(x_d_batch.shape[0], -1, -1, -1, -1)
    x_s_batch = x_s.expand(x_d_batch.shape[0], -1, -1)

    inference_batch_size = 4
    num_batches = (x_d_batch.shape[0] + inference_batch_size - 1) // inference_batch_size

    frames = []
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * inference_batch_size
        end_idx = min((i + 1) * inference_batch_size, x_d_batch.shape[0])

        batch_f_s = f_s_batch[start_idx:end_idx]
        batch_x_s = x_s_batch[start_idx:end_idx]
        batch_x_d = x_d_batch[start_idx:end_idx]

        out = warp_decode_func(batch_f_s, batch_x_s, batch_x_d)

        # Convert to numpy array
        batch_frames = (out['out'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        frames.extend(list(batch_frames))

    print("list shape: ", len(frames))
    print("image shape: ", frames[0].shape)
    
    return frames



def write_video(all_frames, audio_path, output_path):
    output_no_audio_path = 'outputs/audio_driven_video_no_audio.mp4'
    output_video = output_path

    # Remove the files if they exist
    if os.path.exists(output_no_audio_path):
        os.remove(output_no_audio_path)
    if os.path.exists(output_video):
        os.remove(output_video)
    fps = 25  # Adjust as needed

    height, width, layers = all_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_no_audio_path, fourcc, fps, (width, height))

    for frame in all_frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()

    # Add audio to the video using ffmpeg
    input_video = output_no_audio_path
    input_audio = audio_path  # Use the path to your audio file

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_video,
        '-i', input_audio,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_video
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(output_no_audio_path)
        # print(f"Video with audio saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio to video: {e}")



# blink noise
def fixed_blink_noise(motion_tensor, motion_gt):
    eye_indices = torch.tensor([5, 9], device=motion_tensor.device)
    rest_latent = torch.tensor([-0.005], device=motion_tensor.device)
    spikes = torch.tensor([0.0015, 0.0085, 0.011, 0.0075, 0.002], device=motion_tensor.device)
    freeze_index = [0, 1, 2, 4, 6, 8, 10]
    freeze_index = torch.tensor(freeze_index, device=motion_tensor.device)
    period = 12
    period_counter = 0
    reset_flag = False
    for i in range(motion_tensor.shape[0]):
        in_period_index = i % period
        if in_period_index < 5 and period_counter >= period:
            for eye_index in eye_indices:
                motion_tensor[i, eye_index] = spikes[in_period_index]
                motion_tensor[i, eye_index] += torch.randn_like(motion_tensor[i, eye_index]) * 0.002
            if in_period_index == 4:
                reset_flag = True
                period_counter = 0
        else:
            for eye_index in eye_indices:
                motion_tensor[i, eye_index] = rest_latent
                motion_tensor[i, eye_index] += torch.randn_like(motion_tensor[i, eye_index]) * 0.0002
            period_counter += 1
        if reset_flag:
            reset_flag = False
            period = torch.randint(15, 18, (1,)).item()
        for f in freeze_index:
            # motion_tensor[i, f] = motion_gt[i, f]
            motion_tensor[i, f] = 0
    return motion_tensor




# normalize headpose
def normalize_pose(full_motion, headpose_bound):
    assert headpose_bound is not None and len(headpose_bound) % 2 == 0
    headpose_bound = torch.tensor(headpose_bound)
    headpose_bound = headpose_bound.reshape(headpose_bound.shape[0] // 2, 2)

    # Assuming full_motion is a tensor of shape (batch_size, sequence_length, num_features)
    # and the last 5 features are the ones to be normalized
    last_5_features = full_motion[:, :, -5:]

    # Normalize each of the last 5 features
    for i in range(5):
        lower_bound = headpose_bound[i][0]
        upper_bound = headpose_bound[i][1]

        # Clamp the values within the specified bounds
        clamped = torch.clamp(last_5_features[:, :, i], min=lower_bound, max=upper_bound)

        # Normalize to the range [-0.05, 0.05]
        normalized = (clamped - lower_bound) / (upper_bound - lower_bound) * 0.1 - 0.05

        # Update the last 5 features with the normalized values
        last_5_features[:, :, i] = normalized

    # Update the full_motion tensor with the normalized last 5 features
    full_motion[:, :, -5:] = last_5_features

    return full_motion



def reverse_normalize_pose(normalized_motion, headpose_bound):
    assert headpose_bound is not None and len(headpose_bound) % 2 == 0
    headpose_bound = torch.tensor(headpose_bound)
    headpose_bound = headpose_bound.reshape(headpose_bound.shape[0] // 2, 2)

    # Assuming normalized_motion is a tensor of shape (batch_size, sequence_length, num_features)
    # and the last 5 features are the ones to be reversed
    last_5_features = normalized_motion[:, :, -5:]

    # Reverse normalization for each of the last 5 features
    for i in range(5):
        lower_bound = headpose_bound[i][0]
        upper_bound = headpose_bound[i][1]

        # Reverse the normalization from [-0.05, 0.05] to the original range
        original = (last_5_features[:, :, i] + 0.05) / 0.1 * (upper_bound - lower_bound) + lower_bound

        # Update the last 5 features with the original values
        last_5_features[:, :, i] = original

    # Update the normalized_motion tensor with the original last 5 features
    normalized_motion[:, :, -5:] = last_5_features

    return normalized_motion




def inference_one_input(audio_path, portrait_path, output_vid_path, inference_manager, audio_model_config, cfg_s, mouth_ratio, subtract_avg_motion):
    
    motion_mean_exp_path = '/mnt/kostas-graid/datasets/yiliu7/liveProtrait/hdtf/train_split/audio_latent/TH_00192/000.npy'
    motion_mean_exp_tensor = torch.from_numpy(np.load(motion_mean_exp_path)).to(device='cuda')
    motion_mean_exp_tensor = motion_mean_exp_tensor.unsqueeze(0).to(device='cuda')

    motion_tensor, _, _, _ = process_motion_tensor(motion_mean_exp_tensor, None, \
                                latent_type=audio_model_config['motion_latent_type'],
                                latent_mask_1=audio_model_config['latent_mask_1'],
                                latent_bound=torch.tensor(audio_model_config['latent_bound'], device='cuda'),
                                use_headpose=True, headpose_bound=torch.tensor(headpose_bound_list, device='cuda'))

    mean_exp = torch.mean(motion_tensor.reshape(-1, motion_tensor.shape[-1]), dim=0)
    
    audio_latent = autoregress_inference_process_wav_file(audio_path)
    window_count = audio_latent.shape[0]
    # load portrait
    img_rgb = load_image_rgb(portrait_path)
    source_rgb_lst = [img_rgb]
    source_lmk = cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
    img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
    I_s = prepare_source(img_crop_256x256)
    x_s_info = get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    x_s = transform_keypoint(x_s_info)
    f_s = extract_feature_3d(I_s)

    # inference
    mouth_open_ratio_val = mouth_ratio
    mouth_open_ratio_input = torch.tensor([mouth_open_ratio_val], device=device).unsqueeze(0)
    out_motion = torch.tensor([], device=device)
    out_null_motion = torch.tensor([], device=device)
    motion_dim = audio_model_config['x_dim']
    shape_in = x_c_s.reshape(1, -1).to(device)
    for batch_index in range(0, window_count):
        # print(f'batch_index: {batch_index}')
        if batch_index == 0:
            this_audio_prev = audio_latent[0:1, 0:10, :]
            audio_seq = audio_latent[0:1, 10:, :]
            this_motion_prev = torch.zeros(1, 10, motion_dim , device=device)
            gen_length = 65
        else:
            this_audio_prev = audio_latent[batch_index:batch_index+1, 0:prev_context_len, :]
            audio_seq = audio_latent[batch_index:batch_index+1, prev_context_len:, :]
            gen_length = gen_len_per_window

        mean_exp_expanded = mean_exp.expand(1, -1)
        generated_motion, null_motion = inference_manager.inference(audio_seq,
                                                    shape_in, this_motion_prev, this_audio_prev, #seq_mask=seq_mask,
                                                    cfg_scale=cfg_s,
                                                    mouth_open_ratio = mouth_open_ratio_input,
                                                    denoising_steps=10,
                                                    gen_length=gen_length,
                                                    mean_exp=mean_exp_expanded)
        full_window_motion = torch.cat((this_motion_prev, generated_motion), dim=1)
        this_motion_prev = full_window_motion[:, -prev_context_len:, :]
        # generated_motion = generated_motion.reshape(-1, 6)

        if subtract_avg_motion:
            generated_motion = generated_motion - torch.mean(generated_motion, dim=-1, keepdim=True)
        generated_motion = generated_motion - torch.mean(null_motion, dim=-2, keepdim=True)
        out_motion = torch.cat((out_motion, generated_motion.reshape(-1, motion_dim)), dim=0)
        out_null_motion = torch.cat((out_null_motion, null_motion.reshape(-1, motion_dim)), dim=0)

    # out_motion = out_motion - out_null_motion
    # Then modify the filtering line to:
    out_motion_filtered = savgol_filter(out_motion.cpu().numpy(), window_length=5, polyorder=2, axis=0)
    out_motion = torch.tensor(out_motion_filtered, device=device)
    # out_null_motion = out_null_motion - out_null_motion

    motion_gt = motion_tensor[:, :out_motion.shape[0], :].squeeze(0)
    print(f'motion_gt shape: {motion_gt.shape}, out_motion shape: {out_motion.shape}')
    out_motion = fixed_blink_noise(out_motion, motion_gt)
    out_pose_motion = out_motion[:, -5:]
    out_pose_smoothed = savgol_filter(out_pose_motion.cpu().numpy(), window_length=15, polyorder=2, axis=0)
    out_motion[:, -5:-2] *= 2
    out_motion[:, -2:] *= 0.5
    out_motion[:, -5:] = torch.tensor(out_pose_smoothed, device=device)
    plot_gt = False
    if plot_motion:
        # Get total number of dimensions
        n_dims = out_motion.shape[1]
        vel = out_motion[1:] - out_motion[:-1]
        acc = vel[1:] - vel[:-1]

        null_vel = out_null_motion[1:] - out_null_motion[:-1]
        null_acc = null_vel[1:] - null_vel[:-1]

        gt_vel = motion_gt[1:] - motion_gt[:-1]
        gt_acc = gt_vel[1:] - gt_vel[:-1]

        # Create figure with subplots - three rows per dimension
        fig, axs = plt.subplots(n_dims*3, 1, figsize=(24, 5*n_dims*3))

        # Plot each dimension
        for i in tqdm(range(n_dims)):
            # Plot position
            position = out_motion[:, i].cpu().detach().numpy()
            null_position = out_null_motion[:, i].cpu().detach().numpy()
            gt_position = motion_gt[:, i].cpu().detach().numpy()
            axs[i*3].plot(position, label='Motion')
            if plot_gt:
                axs[i*3].plot(gt_position, label='Ground Truth', alpha=0.5)
            # else:
            #     axs[i*3].plot(null_position, label='Null Motion', alpha=0.5)
            axs[i*3].set_ylabel('Position')
            axs[i*3].set_xlabel('Time step')
            axs[i*3].legend()

            # Plot velocity
            velocity = vel[:, i].cpu().detach().numpy()
            null_velocity = null_vel[:, i].cpu().detach().numpy()
            gt_velocity = gt_vel[:, i].cpu().detach().numpy()
            axs[i*3 + 1].plot(velocity, label='Motion')
            # axs[i*3 + 1].plot(null_velocity, label='Null Motion', alpha=0.5)
            if plot_gt:
                axs[i*3 + 1].plot(gt_velocity, label='Ground Truth', alpha=0.5)
            axs[i*3 + 1].set_title(f'Dimension {i} Velocity')
            axs[i*3 + 1].set_ylabel('Velocity')
            axs[i*3 + 1].set_xlabel('Time step')
            axs[i*3 + 1].legend()

            # Plot acceleration
            acceleration = acc[:, i].cpu().detach().numpy()
            null_acceleration = null_acc[:, i].cpu().detach().numpy()
            gt_acceleration = gt_acc[:, i].cpu().detach().numpy()
            axs[i*3 + 2].plot(acceleration, label='Motion')
            # axs[i*3 + 2].plot(null_acceleration, label='Null Motion', alpha=0.5)
            if plot_gt:
                axs[i*3 + 2].plot(gt_acceleration, label='Ground Truth', alpha=0.5)
            axs[i*3 + 2].set_title(f'Dimension {i} Acceleration')
            axs[i*3 + 2].set_ylabel('Acceleration')
            axs[i*3 + 2].set_xlabel('Time step')
            axs[i*3 + 2].legend()

        plt.tight_layout()
        plt.show()

    generated_motion = out_motion
    generated_motion = reverse_normalize_pose(generated_motion.unsqueeze(0), headpose_bound=torch.tensor(headpose_bound_list, device='cuda'))
    generated_motion = generated_motion.squeeze(0)
    if not write_vid:
        return generated_motion


    all_frames = process_motion_batch(generated_motion, f_s, x_s, x_c_s, x_s_info, audio_model_config, warp_decode)
    write_video(all_frames, audio_path, output_vid_path)

    return generated_motion



def call_syncnet(output_vid_path, tmp_dir):

    # Extract the reference from the video filename
    video_basename = os.path.basename(output_vid_path)
    reference = os.path.splitext(video_basename)[0]

    results, activesd = syncnet_inference(output_vid_path, reference, tmp_dir, keep_output=False)
    if results:
        return results['confidence'], results['min_dist']

    
def change_working_dir_to_script_location():
    os.chdir('/mnt/e/wsl_projects/LivePortrait/')
