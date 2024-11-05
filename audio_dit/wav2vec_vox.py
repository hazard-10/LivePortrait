import json
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
import numpy as np
from typing import List, Tuple
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Constants
MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000
FRAME_RATE = 25  # This is the target frame rate for video
SECTION_LENGTH = 3  # seconds of sequence
OVERLAP = 10  # frame of context

DB_ROOT='vox2-audio-tx'
LOG='log'
AUDIO='audio/audio'
OUTPUT_DIR='audio_encoder_output'
BATCH_SIZE=16

def read_multiple_audios(paths, num_threads=12):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(load_and_process_audio, paths))
    return results


def read_json_and_form_paths(data,id_key):
    filenames=[]
    file_paths = []
    
    # Iterate through the nested structure
    for id_key, id_value in data.items():
        os.makedirs(os.path.join(DB_ROOT,OUTPUT_DIR,id_key), exist_ok=True)
        for url_key, url_value in id_value.items():
            for clip_id in url_value.keys():
                # Form the file path
                file_path = os.path.join(DB_ROOT,AUDIO,id_key, url_key, clip_id.replace('.txt', '.wav'))
                file_name = os.path.join(DB_ROOT,OUTPUT_DIR,id_key, url_key+'+'+clip_id.replace('.txt', ''))
                filenames.append(file_name)
                file_paths.append(file_path)
    
    return file_paths, filenames

def load_and_process_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    
    original_sample_rate = sample_rate
    
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    print(file_path," waveform.shape ",waveform.shape)
    
    # Calculate section length and overlap in samples
    section_samples = SECTION_LENGTH * 16027
    overlap_samples = int(OVERLAP / FRAME_RATE * TARGET_SAMPLE_RATE)
    print('section_samples',section_samples,'overlap_samples',overlap_samples)
    
    # Pad if shorter than 3 seconds
    if waveform.shape[1] < section_samples:
        waveform = torch.nn.functional.pad(waveform, (0, section_samples - waveform.shape[1]))
        return [waveform.squeeze(0)], original_sample_rate
    
    # Split into sections with overlap
    sections = []
    start = 0

    print('starting to segment', file_path)
    while start < waveform.shape[1]:
        end = start + section_samples
        if end >= waveform.shape[1]:
            tmp=waveform[:, start:min(end, waveform.shape[1])]
            tmp = torch.nn.functional.pad(tmp, (0, section_samples - tmp.shape[1]))
            sections.append(tmp.squeeze(0))
            print(tmp.shape)
            break
        else:
            sections.append(waveform[:, start:min(end, waveform.shape[1])].squeeze(0))
        
        start = int(end - overlap_samples)
        
    
    return file_path, sections

def inference_process_wav_file(path, output_path):
    device = torch.device(f"cuda")
    
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    audio_path, segments = load_and_process_audio(path)
    segments = np.array(segments)
    
    inputs = processor(segments, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(device)
            
    with torch.no_grad():
        outputs = model(inputs)
        latent = outputs.last_hidden_state
        
        seq_length = latent.shape[1]
        new_seq_length = int(seq_length * (FRAME_RATE / 50))  # Assuming Wav2Vec2 outputs at ~50Hz
            
        latent_features_interpolated = F.interpolate(latent.transpose(1,2), 
                                                        size=new_seq_length, 
                                                        mode='linear', 
                                                        align_corners=True).transpose(1,2)
        
        np.save(output_path, latent_features_interpolated.cpu().numpy())


def process_wav_file(paths, output_paths,uid):
    device = torch.device(f"cuda")
    
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    read_2_gpu_batch_size = 2048
    gpu_batch_size = BATCH_SIZE
    process_queue = torch.Tensor().to(device)

    audio_segments = read_multiple_audios(paths, num_threads=4)
    all_segments = []
    total_segments = 0

    audio_lengths = []
    output_fns = []

    for (audio_path, segments), output_fn in zip(audio_segments,output_paths):
        all_segments.extend(segments)
        segment_count = len(segments)
        total_segments += segment_count
        audio_lengths.append(segment_count)

        output_fns.append(output_fn)

    all_segments = np.array(all_segments)
    print(all_segments.size)

    read_data_2_gpu_pointer = 0
    pbar = tqdm(total=total_segments, desc=f"Processing {uid}")

    while read_data_2_gpu_pointer < total_segments:
        current_batch_size = min(read_2_gpu_batch_size, total_segments - read_data_2_gpu_pointer)

        batch_input = all_segments[read_data_2_gpu_pointer:read_data_2_gpu_pointer + current_batch_size]

        mini_batch_start = 0
        all_info = []
        while mini_batch_start < batch_input.shape[0]:
            mini_batch_end = min(mini_batch_start + gpu_batch_size, batch_input.shape[0])
            mini_batch = batch_input[mini_batch_start:mini_batch_end]

            inputs = processor(mini_batch, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
            
            latent = outputs.last_hidden_state
            print('latent',latent.shape)
            seq_length = latent.shape[1]
            new_seq_length = int(seq_length * (FRAME_RATE / 50))  # Assuming Wav2Vec2 outputs at ~50Hz
            
            latent_features_interpolated = F.interpolate(latent.transpose(1,2), 
                                                            size=new_seq_length, 
                                                            mode='linear', 
                                                            align_corners=True).transpose(1,2)
            print('latent_features_interpolated',latent_features_interpolated.shape)
            all_info.append(latent_features_interpolated)

            mini_batch_start = mini_batch_end
        all_info_tensor = torch.cat(all_info, dim=0)

        process_queue = torch.cat((process_queue, all_info_tensor), dim=0)

        print(audio_lengths)
        while len(output_fns) > 0 and len(process_queue) >= audio_lengths[0]:
            current_output_fn = output_fns[0]
            current_segment_count = audio_lengths[0]

            audio_tensor = process_queue[:current_segment_count]
            np.save(current_output_fn, audio_tensor.cpu().numpy())
            print('save',current_output_fn)
            process_queue = process_queue[current_segment_count:]
            output_fns.pop(0)
            audio_lengths.pop(0)
           

        read_data_2_gpu_pointer += current_batch_size
        pbar.update(current_batch_size)

    pbar.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process videos for live encoding')
    parser.add_argument('-m','--mode', type=str, default='train',
                        help='train or inference')
    parser.add_argument('-i','--inference_file_path', type=str, default=None,
                        help='path of wav file you want to run inference on')
    parser.add_argument('-o','--inference_output_path', type=str, default=None,
                        help='path of wav file you want to run inference on')

    args = parser.parse_args()

    mode = args.mode

    if mode == 'train':
        json_file = 'output_union_512.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
        for uid in data.keys():
            wav_file_paths, file_names = read_json_and_form_paths(data[uid],uid)
            process_wav_file(wav_file_paths, file_names, uid)
        print("finish wav2vec2")
    elif mode == 'inference':
        if args.inference_file_path and args.inference_output_path:
            inference_process_wav_file(args.inference_file_path, args.inference_output_path)
        else:
            raise ValueError("You should enter the path of the file you want to run inference on.")
