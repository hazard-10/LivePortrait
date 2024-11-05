'''
For parsing training set with LivePortrait latent
'''

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_video(json_path):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    video_paths = {}
    total_vid = 0

    # Iterate through the JSON structure
    for uid_key in tqdm(sorted(data.keys()), desc="Loading video paths"):
        video_paths[uid_key] = []
        for second_level_key in sorted(data[uid_key].keys()):
            for third_level_key in sorted(data[uid_key][second_level_key]):
                third_level_key = third_level_key.split('.')[0]
                # Construct the video path
                video_path = os.path.join(uid_key, second_level_key, f"{third_level_key}.mp4")
                video_paths[uid_key].append(video_path)
                total_vid += 1

        # Sort the video paths for each first_level_key
        video_paths[uid_key].sort()

    # Print the results
    print(f"Generated {len(video_paths)} uids. Total videos: {total_vid}, Sample video path: {video_paths[uid_key][0]}")
    return video_paths

def download_file(s3, bucket_name, file_key, local_file_path):
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    s3.download_file(bucket_name, file_key, local_file_path)

def download_all(video_paths, bucket_name, bucket_prefix, output_dir, max_workers=10):
    s3 = boto3.client(
        's3',
        aws_access_key_id='09b8eba18e372016bdde427aa3061da5',
        aws_secret_access_key='e19dbd7f7887f03c83ae601936a9e48417ba1dd9699867694d2763b8d9069d00',
        region_name='wnam',
        endpoint_url='https://61bd92b4c0d2a2dc249b7c78aeaafda7.r2.cloudflarestorage.com'
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for uid, paths in video_paths.items():
            for video_path in paths:
                file_key = os.path.join(bucket_prefix, video_path)
                local_file_path = os.path.join(output_dir, video_path)
                futures.append(executor.submit(download_file, s3, bucket_name, file_key, local_file_path))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            future.result()  # This will raise any exceptions that occurred during download

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download videos from S3-compatible storage')
    parser.add_argument('--json_path', type=str, default='/mnt/c/Users/mjh/Downloads/output_union_512.json',
                        help='Path to the JSON file containing video information')
    parser.add_argument('--bucket_name', type=str, default='vox2-full',
                        help='Name of the S3 bucket')
    parser.add_argument('--bucket_prefix', type=str, default='videos/512/',
                        help='Prefix path in the bucket where videos are stored')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for downloaded files')

    args = parser.parse_args()

    video_paths = load_video(args.json_path)
    download_all(video_paths, args.bucket_name, args.bucket_prefix, args.output_dir)
