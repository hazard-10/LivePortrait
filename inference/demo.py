import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import json
import yaml
from time import sleep
import subprocess
from threading import Thread
from queue import Queue
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter
import cv2
import torch
import torchaudio
import torch.nn.functional as F
torchaudio.set_audio_backend("soundfile")

# wave2vec
from transformers import Wav2Vec2Model, Wav2Vec2Processor
#LivePortrait
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.helper import load_model
from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix
from src.utils.cropper import Cropper
from src.utils.io import load_image_rgb
# DiT
from audio_dit.inference import get_model
from audio_dit.dataset import process_motion_tensor
print("Finishing importing")

# config
# wave2vec
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
prev_context_len = 10
gen_len_per_window = 65

# DiT model
config_path = 'D:/Projects/Upenn_CIS_5650/final-project/config/config.json'
weight_path = 'D:/Projects/Upenn_CIS_5650/final-project/config/model.pth'
motion_mean_exp_path = 'D:/Projects/Upenn_CIS_5650/final-project/config/000.npy'

cfg_s = 0.65
mouth_ratio = 0.25
subtract_avg_motion = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# LivePortrait Pipeline
load_lp_start = datetime.now()
def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

args = ArgumentConfig()
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)

'''
Main function for inference
'''

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
print(f"Finish loading LivePortrait Pipeline in {(datetime.now() - load_lp_start).total_seconds()}s\n")

# wav2vec Pipeline
load_w2v_start = datetime.now()
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
    total_frame = int(waveform.shape[1] / TARGET_SAMPLE_RATE * FRAME_RATE) + 1
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

    return windows, total_frame

def autoregress_inference_process_wav_file(path):
    windows, total_frame = autoregress_load_and_process_audio(path)
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
    return latent_features_interpolated, total_frame

# Move model and processor to global scope
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
print(f"Finish loading wave2vec Pipeline in {(datetime.now() - load_w2v_start).total_seconds()}s\n")

# DiT Pipeline
load_dit_start = datetime.now()
audio_model_config = json.load(open(config_path))
inference_manager = get_model(config_path, weight_path, device)
print(f"Finish loading DiT Pipeline in {(datetime.now() - load_dit_start).total_seconds()}s\n")

# Motion tuning
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
            # motion_tensor[i, f] = motion_gt[i, f]|
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

headpose_bound = torch.tensor([-21, 25, -30, 30, -23, 23, -0.3, 0.3, -0.3, 0.28], device = device)
motion_mean_exp_tensor = torch.from_numpy(np.load(motion_mean_exp_path)).to(device = device)
motion_mean_exp_tensor = motion_mean_exp_tensor.unsqueeze(0).to(device = device)
motion_tensor, _, _, _ = process_motion_tensor(motion_mean_exp_tensor, None, \
                            latent_type=audio_model_config['motion_latent_type'],
                            latent_mask_1=audio_model_config['latent_mask_1'],
                            latent_bound=torch.tensor(audio_model_config['latent_bound'], device=device),
                            use_headpose=True, headpose_bound=headpose_bound)
mean_exp = torch.mean(motion_tensor.reshape(-1, motion_tensor.shape[-1]), dim=0)
mouth_open_ratio_input = torch.tensor([mouth_ratio], device=device).unsqueeze(0)
motion_dim = audio_model_config['x_dim']

is_processing = False
print("Ready to Animate\n")

def inference_thread(portrait_path, audio_path, frame_queue):
    t1 = datetime.now()
    # Process image
    img_rgb = load_image_rgb(portrait_path)
    img_crop_256x256 = cv2.resize(img_rgb, (256, 256))  # force to resize to 256x256
    I_s = prepare_source(img_crop_256x256)
    x_s_info = get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    x_s = transform_keypoint(x_s_info)
    f_s = extract_feature_3d(I_s)
    shape_in = x_c_s.reshape(1, -1).to(device)
    t2 = datetime.now()
    # Process audio
    audio_latent, total_frame = autoregress_inference_process_wav_file(audio_path)
    t3 = datetime.now()
    # Generate Motion
    out_motion = torch.tensor([], device=device)
    window_count = audio_latent.shape[0]
    for batch_index in range(0, window_count):

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

        if subtract_avg_motion:
            generated_motion = generated_motion - torch.mean(generated_motion, dim=-1, keepdim=True)

        generated_motion = generated_motion - torch.mean(null_motion, dim=-2, keepdim=True)
        out_motion = torch.cat((out_motion, generated_motion.reshape(-1, motion_dim)), dim=0)
    t4 = datetime.now()
    # Smooth and tune motion
    out_motion_filtered = savgol_filter(out_motion[:total_frame].cpu().numpy(), window_length=5, polyorder=2, axis=0)
    out_motion = torch.tensor(out_motion_filtered, device=device)

    motion_gt = motion_tensor[:, :out_motion.shape[0], :].squeeze(0)
    out_motion = fixed_blink_noise(out_motion, motion_gt)
    out_pose_motion = out_motion[:, -5:]
    out_pose_smoothed = savgol_filter(out_pose_motion.cpu().numpy(), window_length=15, polyorder=2, axis=0)
    out_motion[:, -5:] = torch.tensor(out_pose_smoothed, device=device)

    full_motion = reverse_normalize_pose(out_motion.unsqueeze(0), headpose_bound=headpose_bound).squeeze(0)
    t5 = datetime.now()
    # Generate frame
    pose = full_motion[:, -5:]
    exp = full_motion[:, :-5]

    full_63_exp = torch.zeros(full_motion.shape[0], 63, device=device)

    full_63_exp[:, audio_model_config['latent_mask_1']] = exp
    full_motion = full_63_exp.reshape(-1, 63)

    x_d_list = []
    scale = x_s_info['scale']

    for i in range(full_motion.shape[0]):
        exp = full_motion[i].reshape(21, 3)
        pitch, yaw, roll, t_x, t_y = pose[i].unsqueeze(0).unbind(-1)
        t = torch.tensor([t_x, t_y, 0], device=device)

        x_d_i = scale * (x_c_s @ get_rotation_matrix(pitch, yaw, roll) + exp) + t
        x_d_list.append(x_d_i.squeeze(0))

    x_d_batch = torch.stack(x_d_list, dim=0)
    t6 = datetime.now()
    print(f"image processing time {(t2 - t1).total_seconds()}s")
    print(f"audio processing time {(t3 - t2).total_seconds()}s")
    print(f"motion generation time {(t4 - t3).total_seconds()}s")
    print(f"motion smoothing time {(t5 - t4).total_seconds()}s")
    print(f"x_d generation time {(t6 - t5).total_seconds()}s")
    print(f"total time before frame generation {(t6 - t1).total_seconds()}s")

    for i in range(full_motion.shape[0]):
        out = warp_decode(f_s, x_s, x_d_batch[i])
        batch_frame = out['out'].permute(0, 2, 3, 1).mul_(255).to(torch.uint8).squeeze(0)
        frame_queue.put(batch_frame)

    frame_queue.put(None)

def generate_talking_head(portrait_path, audio_path):

    pre_time = datetime.now()

    global is_processing

    if is_processing:
        return None

    is_processing = True
    process_btn.interactive = False
    frame_queue = Queue()

    thread = Thread(
        target=inference_thread,
        args=(portrait_path, audio_path, frame_queue)
    )
    thread.start()

    first = True

    while True:

        frame_tensor = frame_queue.get()
        if frame_tensor is None:
            break
        frame = frame_tensor.cpu().numpy()

        # Ensure 25 fps
        if (datetime.now() - pre_time).total_seconds() < 0.035:
            sleep(0.035 - (datetime.now() - pre_time).total_seconds())

        #print(f"time to display {datetime.now() - pre_time}")

        if first:
            # Start audio playback and show first frame
            first = False
            print(f"initial latency {datetime.now() - pre_time}")
            yield frame, gr.update(value=audio_path, autoplay=True)
        else:
            yield frame, gr.update()

        pre_time = datetime.now()

    is_processing = False
    process_btn.interactive = True

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Real Time Talking Head")

    with gr.Row():
        # Left column for inputs and button
        with gr.Column(scale=1):
            portrait_input = gr.Image(
                label="Input Portrait 🖼️",
                type="filepath"
            )
            audio_input = gr.Audio(
                label="Input Audio 🎞️",
                type="filepath"
            )
            process_btn = gr.Button("Animate 🚀")

        # Right column for video output
        with gr.Column(scale=1):
            video_output = gr.Image(label="Generated Animation", streaming=True)

    process_btn.click(
        fn=generate_talking_head,
        inputs=[portrait_input, audio_input],
        outputs=[video_output, audio_input]
    )

if __name__ == "__main__":
    demo.launch(server_port=9995, share=False)
