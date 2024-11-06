import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

'''
How data loading works in this project:
* We assume the npy will be sufficient to load into memory
* latent composition:
    concat_tensor = torch.cat([
        x_info['kp'], # 63
        x_info['exp'], # 63,
        x_info['t'], # 3
        x_info['pitch'], # 1
        x_info['yaw'], # 1
        x_info['roll'], # 1
        x_info['scale'], # 1
    ], dim=1)

    for person_feat
     - kp single batch avg used for v1. exp keep the same, use global avg scale. t and rot will be subtracted in subsequent frames,
     - so we can ignore t and rot for now
    for training data
     - use individual kp, exp, subtract dominant t and rot, use global scale



1. Load both audio and motion latent from the npy files with load_npy_files
    2.  Motion latent is processed to have a window of 65 frames with 10 frames overlap
    2.1 The rotation and translation are frontalized
3.  The processed data is then loaded into a custom dataset class for random sampling
'''

'''
Part 1: Load and process the npy files
'''
def load_and_process_pair(audio_file, motion_file, latent_type='exp', latent_mask_1=None, latent_bound=None):
    # Load audio file
    # Check the type of audio_file
    if isinstance(audio_file, (str, os.PathLike)):
        # If it's a string (file path), load the audio data
        audio_data = np.load(audio_file)
    elif isinstance(audio_file, (np.ndarray, torch.Tensor)):
        # If it's already a numpy array or torch tensor, use it as is
        audio_data = audio_file
    else:
        # If it's neither a string nor a numpy array/torch tensor, raise an error
        raise ValueError("audio_file must be a file path string, numpy array, or torch tensor")

    # Load and process motion file
    motion_data = np.load(motion_file)
    pad_length = (65 - (motion_data.shape[0] - 10) % 65) % 65
    padded_data = np.pad(motion_data, ((0, pad_length), (0, 0)), mode='constant')

    data_without_first_10 = padded_data[10:]
    N = data_without_first_10.shape[0] // 65
    reshaped_data = data_without_first_10[:N*65].reshape(N, 65, 136)
    last_10 = reshaped_data[:, -10:, :]
    prev_10 = np.concatenate([padded_data[:10][None, :, :], last_10[:-1]], axis=0)
    motion_data = np.concatenate([prev_10, reshaped_data], axis=1)

    end_indices = torch.ones(N, dtype=torch.int32) * (motion_data.shape[1] - 1)
    end_indices[-1] = motion_data.shape[1] - 1 - pad_length

    # Ensure audio and motion data have the same number of frames.
    # Prev lookup show 1 frame mismatch is common. In this case we only fix batch size mismatch
    min_frames = min(audio_data.shape[0], motion_data.shape[0])
    audio_data = audio_data[:min_frames]
    motion_data = motion_data[:min_frames]
    end_indices = end_indices[:min_frames]

    motion_tensor = torch.from_numpy(motion_data)
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.from_numpy(audio_data)
    elif isinstance(audio_data, torch.Tensor):
        audio_tensor = audio_data

    motion_tensor, audio_tensor, shape_tensor, mouth_tensor = process_motion_tensor(motion_tensor, audio_tensor, latent_type, latent_mask_1, latent_bound)

    return motion_tensor, audio_tensor, shape_tensor, mouth_tensor, end_indices

def process_directory(uid, audio_root, motion_root, latent_type='exp', latent_mask_1=None, latent_bound=None):
    audio_dir = os.path.join(audio_root, uid)
    motion_dir = os.path.join(motion_root, uid)

    if not (os.path.isdir(audio_dir) and os.path.isdir(motion_dir)):
        print(f"Directory not found for ", audio_dir, motion_dir)
        return None, None

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
    motion_files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])

    motion_tensor_list = []
    audio_tensor_list = []
    shape_tensor_list = []
    mouth_tensor_list = []
    end_indices_list = []
    for audio_file, motion_file in zip(audio_files, motion_files):
        if audio_file != motion_file and audio_file.split('+')[0] != motion_file.split('+')[0]:
            print(f"Mismatch in {uid}: {audio_file} and {motion_file}")
            continue

        audio_path = os.path.join(audio_dir, audio_file)
        motion_path = os.path.join(motion_dir, motion_file)

        motion_tensor, audio_tensor, shape_tensor, mouth_tensor, end_indices = load_and_process_pair(audio_path, motion_path, latent_type, latent_mask_1, latent_bound)
        motion_tensor_list.append(motion_tensor)
        audio_tensor_list.append(audio_tensor)
        shape_tensor_list.append(shape_tensor)
        mouth_tensor_list.append(mouth_tensor)
        end_indices_list.append(end_indices)

    return torch.cat(motion_tensor_list, dim=0), torch.cat(audio_tensor_list, dim=0), torch.cat(shape_tensor_list, dim=0), \
            torch.cat(mouth_tensor_list, dim=0), torch.cat(end_indices_list, dim=0)

def load_npy_files(audio_root, motion_root, start_idx=None, end_idx=None, latent_type='exp', latent_mask_1=None, latent_bound=None):
    all_dir_list = sorted(os.listdir(audio_root))
    if start_idx is not None and end_idx is not None:
        dir_list = all_dir_list[start_idx:end_idx]
    else:
        dir_list = all_dir_list

    all_motion_data = []
    all_audio_data = []
    all_shape_data = []
    all_mouth_data = []
    all_end_indices = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_uid = {executor.submit(process_directory, uid, audio_root, motion_root, latent_type, latent_mask_1, latent_bound): uid for uid in dir_list}

        for future in tqdm(as_completed(future_to_uid), total=len(dir_list), desc="Processing directories"):
            uid = future_to_uid[future]
            try:
                motion_data, audio_data, shape_data, mouth_data, end_indices = future.result()
                if audio_data is not None and motion_data is not None:
                    all_motion_data.append(motion_data)
                    all_audio_data.append(audio_data)
                    all_shape_data.append(shape_data)
                    all_mouth_data.append(mouth_data)
                    all_end_indices.append(end_indices)
            except Exception as exc:
                print(f'{uid} generated an exception: {exc}')

    motion_tensor = torch.concat(all_motion_data, dim=0)
    audio_tensor = torch.concat(all_audio_data, dim=0)
    shape_tensor = torch.concat(all_shape_data, dim=0)
    mouth_tensor = torch.concat(all_mouth_data, dim=0)
    end_indices = torch.concat(all_end_indices, dim=0)

    # print(f"audio loaded from disk. tensor shape: {audio_tensor.shape}")
    # print(f"motion loaded from disk. tensor shape: {motion_tensor.shape}")

    # motion_tensor = normalize_motion_tensor(motion_tensor, latent_bound, latent_mask_1)
    return motion_tensor, audio_tensor, shape_tensor, mouth_tensor, end_indices

def normalize_motion_tensor(motion_tensor, latent_bound=None, latent_mask_1=None):
    motion_tensor = motion_tensor[:, :, :-1] # Remove the last feature which is always 0
    print(f"motion_tensor shape: {motion_tensor.shape}")
    if latent_bound is not None:
        assert len(latent_bound) % 2 == 0
        assert latent_mask_1 is not None

        latent_bound = torch.tensor(latent_bound).reshape(-1, 2)
        min_vals = torch.zeros(len(latent_mask_1))
        max_vals = torch.zeros(len(latent_mask_1))

        for i, mask_index in enumerate(latent_mask_1):
            min_vals[i] = latent_bound[mask_index][0]
            max_vals[i] = latent_bound[mask_index][1]
    else:
        min_vals, _ = torch.min(motion_tensor.reshape(-1, motion_tensor.shape[-1]), dim=0)
        max_vals, _ = torch.max(motion_tensor.reshape(-1, motion_tensor.shape[-1]), dim=0)

    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1.0  # Set to 1 where max and min are the same

    min_bound, max_bound = -0.05, 0.05
    normalized_tensor = motion_tensor.clone()
    normalized_tensor = (normalized_tensor - min_vals) / denominator
    normalized_tensor = normalized_tensor * (max_bound - min_bound) + min_bound
    normalized_tensor = torch.clamp(normalized_tensor, min=min_bound, max=max_bound)

    normalized_tensor = torch.cat([motion_tensor[:, :, :-5], normalized_tensor[:, :, -5:]], dim=2)

    # # Calculate quantiles for each feature dimension
    # quantiles = torch.tensor([0.0, 0.005, 0.1, 0.25, 0.5, 0.75, 0.9, 0.995, 1.0])
    # num_features = normalized_tensor.shape[-1]

    # print("Quantiles for each feature dimension:")
    # for i in range(num_features):
    #     feat_quantiles = torch.quantile(normalized_tensor[:, i], quantiles)
    #     print(f"Feature {i}: {feat_quantiles.tolist()}")

    # return normalized_tensor
    return motion_tensor

def reverse_normalize_motion_tensor(motion_tensor, latent_bound, latent_mask_1):
    assert latent_bound is not None and latent_mask_1 is not None
    latent_bound = torch.tensor(latent_bound).reshape(-1, 2)
    full_63_exp = torch.zeros(motion_tensor.shape[0], 68)

    a, b =  -0.05, 0.05
    for i, dim in enumerate(latent_mask_1):
        min_bound = latent_bound[dim][0]
        max_bound = latent_bound[dim][1]

        # Reverse the normalization process
        denormalized = (motion_tensor[:, i] - a) / (b - a)  # First, map back to [0, 1]
        denormalized = denormalized * (max_bound - min_bound) + min_bound  # Then, map to original range

        full_63_exp[:, dim] = denormalized

    # Assuming the last 5 dimensions were not normalized and should be kept as is
    ret_tensor = torch.cat([motion_tensor[:, :-5], full_63_exp[:, -5:]], dim=1)
    return ret_tensor

'''
Part 2: Frontalize the motion data, swtich euler angles to quaternions
'''

def euler_to_quaternion(pitch, yaw, roll):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)

def angular_distance_vectorized(poses1, poses2):
    diff = torch.abs(poses1.unsqueeze(1) - poses2.unsqueeze(0))
    diff = torch.min(diff, 2*torch.pi - diff)
    return torch.norm(diff, dim=2)

def find_dominant_pose(poses):
    distances = angular_distance_vectorized(poses, poses)
    total_distances = torch.sum(distances, dim=1)
    min_distance_index = torch.argmin(total_distances)
    return poses[min_distance_index], min_distance_index

def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * np.pi
    yaw = yaw_ / 180 * np.pi
    roll = roll_ / 180 * np.pi

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose

@torch.no_grad()
def process_motion_tensor(motion_tensor, audio_tensor, latent_type='exp', latent_mask_1=None, latent_bound=None):
    device = motion_tensor.device
    n_batches, seq_len, _ = motion_tensor.shape
    all_in_bound = torch.ones(n_batches, seq_len, dtype=torch.bool)

    return_dominant_headpose = True
    # Extract each component
    kp = motion_tensor[:, :, :63]
    exp = motion_tensor[:, :, 63:126]
    translation = motion_tensor[:, :, 126:129]
    orientation = motion_tensor[:, :, 129:132]
    scale = motion_tensor[:, :, 132:133]
    eye_open_ratio = motion_tensor[:, :, 133:135]
    mouth_open_ratio = motion_tensor[:, :, 135:136]

    if return_dominant_headpose:
        '''
        x_d_i_new = scale_tensor * (x_c_s @ R_tensor + exp_tensor) + t_tensor

        In version one:
        1. we compute the batch avg of kp used as shape feat
        2. we keep the exp the same
        3. we use the global avg of scale, so constant
        4. we subtract the dominant t and rot from each frame
        '''
        # Find dominant orientation and translation for each frame
        dominant_orientations = torch.zeros((n_batches, 3), device=device)
        dominant_translations = torch.zeros((n_batches, 3), device=device)

        for i in range(n_batches):
            dominant_orientations[i], _ = find_dominant_pose(orientation[i])
            dominant_translations[i] = torch.median(translation[i], dim=0).values

        # Subtract dominant orientation and translation
        orientation_adjusted = orientation - dominant_orientations.unsqueeze(1)
        translation_adjusted = translation - dominant_translations.unsqueeze(1)

    if True:
        '''
        Use exp for motion representation
        '''

        motion_tensor = torch.tensor([])
        exp = exp.reshape(n_batches, seq_len, -1)
        if latent_mask_1 is not None:
            for i, d in enumerate(latent_mask_1):
                if d >= 63:
                    continue # skip headpose features
                if i == 0:
                    motion_tensor = exp[:, :, d:d+1]
                else:
                    motion_tensor = torch.cat([motion_tensor, exp[:, :, d:d+1]], dim=2)
            motion_tensor = motion_tensor.reshape(n_batches, seq_len, -1)
        # compute canonical shape kp, using the average of first 5 frames
        first_frame_kp = torch.mean(kp[:, :5, :], dim=1)

        # Compute the median of mouth_open_ratio
        median_mouth_open_ratio = torch.median(mouth_open_ratio)
        mouth_open_ratio = median_mouth_open_ratio.expand(n_batches, 1)

        if return_dominant_headpose:
            translation_adjusted = translation_adjusted[:, :, :2] # z is always 0
            motion_tensor = torch.cat([motion_tensor, orientation_adjusted, translation_adjusted], dim=2)
        return motion_tensor, audio_tensor, first_frame_kp, mouth_open_ratio


'''
Part 3: Create a custom dataset class for random sampling
'''
class MotionAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        motion_latents, audio_latents, shape_latents, mouth_latents, end_indices = data
        assert len(motion_latents) == len(audio_latents), "Motion and audio latents must have the same length"
        assert len(motion_latents) == len(shape_latents), "Motion and shape latents must have the same length"
        assert len(motion_latents) == len(mouth_latents), "Motion and mouth latents must have the same length"
        assert len(motion_latents) == len(end_indices), "Motion and end indices must have the same length"
        self.motion_latents = motion_latents
        self.audio_latents = audio_latents
        self.shape_latents = shape_latents
        self.mouth_latents = mouth_latents
        self.end_indices = end_indices

    def __len__(self):
        return len(self.motion_latents)

    def __getitem__(self, idx):
        return {
            "motion_latent": self.motion_latents[idx],
            "audio_latent": self.audio_latents[idx],
            "shape_latent": self.shape_latents[idx],
            "mouth_latent": self.mouth_latents[idx],
            "end_indices": self.end_indices[idx]
        }
