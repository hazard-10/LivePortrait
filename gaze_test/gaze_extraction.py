import os
import cv2
import torch
from l2cs import Pipeline, render
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import linregress

def initialize_pipeline():
    """Initialize the gaze estimation pipeline."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    weight_path = os.path.join(script_dir, "L2CS-Net", "models", "L2CSNet_gaze360.pkl")

    return Pipeline(
        weights=weight_path,
        arch='ResNet50',
        device=torch.device('cuda') 
    )


def check_video_path(vid_path):
    """Check if the video path exists."""
    if not os.path.exists(vid_path):
        raise FileNotFoundError(f"Error: Video path does not exist: {vid_path}")



def process_img(vid_path, gaze_pipeline):
    """Process the video to extract pitch and yaw values."""
    frame = cv2.imread(vid_path)
    if frame is None:
        raise FileNotFoundError(f"Error: Cannot open image file: {vid_path}")

    result = gaze_pipeline.step(frame)
    render(frame, result)

    pitch_value = result.pitch[0]
    yaw_value = result.yaw[0]

    return pitch_value, yaw_value
    

def process_video(vid_path, gaze_pipeline):
    """Process the video to extract pitch and yaw values."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot open video file.")

    pitch_values = []
    yaw_values = []
    frame_numbers = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_count in tqdm(range(total_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame.")
            break

        results = gaze_pipeline.step(frame)
        render(frame, results)

        pitch_values.append(results.pitch[0])
        yaw_values.append(results.yaw[0])
        frame_numbers.append(frame_count)

    cap.release()
    return frame_numbers, pitch_values, yaw_values




def plot_results(frame_numbers, pitch_values, yaw_values, output_path):
    """Plot pitch and yaw values over time and save the image."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(frame_numbers, pitch_values, 'b-')
    ax1.set_title('Pitch over Time')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Pitch (radians)')
    ax1.grid(True)

    ax2.plot(frame_numbers, yaw_values, 'r-')
    ax2.set_title('Yaw over Time')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Yaw (radians)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()



def plot_yaw_vs_exp(yaw_values, exp_range, output_path):
    """
    Plot yaw values against experiment values.

    Args:
        yaw_values (list): List of yaw values.
        exp_range (list or np.array): Range of experiment values.
        output_path (str): Path to save the plot.
    """
    if len(yaw_values) != len(exp_range):
        raise ValueError("Yaw values and experiment range must have the same length.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(exp_range, yaw_values, marker='o', linestyle='-', color='r')
    plt.title("Yaw Values vs. Exp")
    plt.xlabel("Exp Values")
    plt.ylabel("Yaw (radians)")
    plt.grid(True)

    plt.savefig(output_path)
    print(f"Yaw vs. Experiment plot saved to {output_path}")
    plt.close()



def plot_pitch_vs_exp(pitch_values, exp_range, output_path):
    if len(pitch_values) != len(exp_range):
        raise ValueError("Pitch values and experiment range must have the same length.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(exp_range, pitch_values, marker='o', linestyle='-', color='r')
    plt.title("Pitch Values vs. Exp")
    plt.xlabel("Exp Values")
    plt.ylabel("Pitch (radians)")
    plt.grid(True)

    plt.savefig(output_path)
    print(f"Pitch vs. Exp plot saved to {output_path}")
    plt.close()




def fit_linear_relationship_with_scipy(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err


def plot_linear_fit(x, y, slope, intercept, title, xlabel, ylabel, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, slope * np.array(x) + intercept, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()



def main():
    """Main function to run the gaze extraction pipeline."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))  
        #vid_path = os.path.join(script_dir, "outputs", "inference_audio", "output_video.mp4")

        yaw_values = []
        pitch_values = []

        index_interval = 100
        
        for exp_i in range(0, index_interval + 1):
            # output_path_idx4 = os.path.join(script_dir, "outputs", "gaze_extraction", "index4", f"pitch_yaw_vs_t_{exp_i:03d}.png")
            # output_path_idx33_45_48 = os.path.join(script_dir, "outputs", "gaze_extraction", "index_33_45_48", f"pitch_yaw_vs_t_{exp_i:03d}.png")
            vid_path_index4 = os.path.join(script_dir, "outputs", "img_rendered", "index4", f"img_exp_{exp_i:03d}.png")
            vid_path_index33_45_48 = os.path.join(script_dir, "outputs", "img_rendered", "index33_45_48", f"img_exp_{exp_i:03d}.png")

            gaze_pipeline = initialize_pipeline()

            _, pitch_value = process_img(vid_path_index4, gaze_pipeline) ### yaw and pitch value are opposite

            yaw_value, _ = process_img(vid_path_index33_45_48, gaze_pipeline) ### yaw and pitch value are opposite
            
            yaw_values.append(yaw_value)
            pitch_values.append(pitch_value)

        output_path_idx4 = os.path.join(script_dir, "outputs", "gaze_exp_relation", "index4", "pitch_vs_exp.png")
        output_path_idx33_45_48 = os.path.join(script_dir, "outputs", "gaze_exp_relation", "index33_45_48", "yaw_vs_exp.png")

        exp_range = np.linspace(-0.3, 0.3, index_interval + 1)
        plot_pitch_vs_exp(pitch_values, exp_range, output_path_idx4)
        plot_yaw_vs_exp(yaw_values, exp_range, output_path_idx33_45_48)


        # Fit and plot pitch vs exp
        pitch_slope, pitch_intercept, pitch_r_value, pitch_p_value, pitch_std_err = fit_linear_relationship_with_scipy(exp_range, pitch_values)
        output_path_pitch_fit = os.path.join(script_dir, "outputs", "gaze_exp_relation", "index4", "pitch_vs_exp_fit.png")
        plot_linear_fit(
            exp_range, pitch_values, pitch_slope, pitch_intercept,
            title="Pitch vs. Exp (Linear Fit)",
            xlabel="Exp Values",
            ylabel="Pitch (radians)",
            output_path=output_path_pitch_fit
        )
        print(f"Pitch Linear Fit: y = {pitch_slope:.2f}x + {pitch_intercept:.2f}, R^2 = {pitch_r_value**2:.2f}")

        # Fit and plot yaw vs exp
        yaw_slope, yaw_intercept, yaw_r_value, yaw_p_value, yaw_std_err = fit_linear_relationship_with_scipy(exp_range, yaw_values)
        output_path_yaw_fit = os.path.join(script_dir, "outputs", "gaze_exp_relation", "index33_45_48", "yaw_vs_exp_fit.png")
        plot_linear_fit(
            exp_range, yaw_values, yaw_slope, yaw_intercept,
            title="Yaw vs. Exp (Linear Fit)",
            xlabel="Exp Values",
            ylabel="Yaw (radians)",
            output_path=output_path_yaw_fit
        )
        print(f"Yaw Linear Fit: y = {yaw_slope:.2f}x + {yaw_intercept:.2f}, R^2 = {yaw_r_value**2:.2f}")


        print("Processing complete.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()



