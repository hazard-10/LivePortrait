# python syncnet.py --videofile mp4_filename --data_dir output_dir --keep_output (flag arg if you want to keep the data)
import os
import json
import pickle
import numpy as np
import subprocess
import shutil
import argparse
import re

# Change working directory to script's location

def run_command(command, capture_output=False):
    if capture_output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        output, error = process.communicate()
        if process.returncode != 0:
            print(f"Error executing command: {command}")
            print(error)
            return False, None
        return True, output
    else:
        process = subprocess.Popen(command, shell=True)
        process.wait()
        if process.returncode != 0:
            print(f"Error executing command: {command}")
            return False
        return True

def run_syncnet_pipeline(videofile, reference, data_dir, parent_dir):
    # Run pipeline
    if not run_command(f"python {parent_dir}/run_pipeline.py --videofile {videofile} --reference {reference} --data_dir {data_dir}"):
        return None, None

    # Run syncnet and capture output
    success, output = run_command(f"python {parent_dir}/run_syncnet.py --videofile {videofile} --reference {reference} --data_dir {data_dir}", capture_output=True)
    if not success:
        return None, None

    # Run visualise
    # if not run_command(f"python run_visualise.py --videofile {videofile} --reference {reference} --data_dir {data_dir}"):
    #     return None, None

    # Parse the output to extract required values
    lines = output.split('\n')
    av_offset = None
    min_dist = None
    confidence = None
    framewise_conf = None

    for line in lines:
        if line.startswith("Framewise conf:"):
            # Use regex to extract the array of numbers
            match = re.search(r'\[(.*?)\]', line, re.DOTALL)
            if match:
                numbers_str = match.group(1)
                framewise_conf = np.array([float(x) for x in numbers_str.split()])
        elif line.startswith("AV offset:"):
            av_offset = int(line.split(':')[1].strip())
        elif line.startswith("Min dist:"):
            min_dist = float(line.split(':')[1].strip())
        elif line.startswith("Confidence:"):
            confidence = float(line.split(':')[1].strip())

    # Load activesd.pckl
    activesd_file = os.path.join(data_dir, 'pywork', reference, 'activesd.pckl')
    with open(activesd_file, 'rb') as f:
        activesd = pickle.load(f)

    # Store the extracted values
    results = {
        'av_offset': av_offset,
        'min_dist': min_dist,
        'confidence': confidence,
        'framewise_conf': framewise_conf.tolist() if framewise_conf is not None else None,
    }

    # Save results to a file
    results_file = os.path.join(data_dir, str(reference)+'syncnet_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"SyncNet results saved to: {results_file}")

    # Copy activesd.pckl to data_dir
    new_activesd_file = os.path.join(data_dir, f"{reference}_activesd.pckl")
    shutil.copy2(activesd_file, new_activesd_file)
    print(f"ActiveSD file copied to: {new_activesd_file}")

    return results, activesd

def syncnet_inference(videofile, reference, data_dir, keep_output):
    # Ensure data_dir exists
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)


    print(f"Creating new directory: {data_dir}")
    os.makedirs(data_dir)

    # Run the entire pipeline and get results
    syncnet_results, activesd = run_syncnet_pipeline(videofile, reference, data_dir, "./")
    if syncnet_results is None:
        print("SyncNet processing failed.")
        return None, None

    if not keep_output:
        # Remove the output directory, except for the results and activesd files
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if name != f"{reference}_syncnet_results.json" and name != f"{reference}_activesd.pckl":
                    os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"Cleaned up output directory, keeping only results and ActiveSD files.")
    else:
        print(f"Output directory kept: {data_dir}")

    return syncnet_results, activesd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SyncNet pipeline")
    parser.add_argument("--videofile", required=True, help="Path to input video file")
    parser.add_argument("--data_dir", required=True, help="Path to output directory")
    parser.add_argument("--keep_output", action="store_true", help="Keep the output directory after processing")
    args = parser.parse_args()

    # Extract the reference from the video filename
    video_basename = os.path.basename(args.videofile)
    reference = os.path.splitext(video_basename)[0]

    print(f"Using reference: {reference}")

    results, activesd = syncnet_inference(args.videofile, reference, args.data_dir, args.keep_output)
    if results:
        print("\nSyncNet Results:")
        print(f"AV Offset: {results['av_offset']}")
        print(f"Confidence: {results['confidence']}")
        print(f"Min Dist: {results['min_dist']}")
        print(f"Framewise Conf Shape: {np.array(results['framewise_conf']).shape}")
        print(f"ActiveSD Shape: {np.array(activesd).shape}")
