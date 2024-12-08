from utils import *


def test_inference(image_path, audio_path, output_video_path, inference_manager, audio_model_config, cfg_s=1.0, mouth_ratio=0.3, subtract_avg_motion=False):
    try:
        # Call inference_one_input to process the inputs and generate the video
        generated_motion = inference_one_input(
            audio_path=audio_path,
            portrait_path=image_path,
            output_vid_path=output_video_path,
            inference_manager=inference_manager,
            audio_model_config=audio_model_config,
            cfg_s=cfg_s,
            mouth_ratio=mouth_ratio,
            subtract_avg_motion=subtract_avg_motion,
        )
        print(f"Video generated successfully and saved at: {output_video_path}")

    except Exception as e:
        print(f"Error during inference: {e}")



if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    img_path = os.path.join(script_dir, "examples", "gaze_face_example1.png")
    audio_path = os.path.join(script_dir, "examples", "366_1733367920.wav")
    output_video_path = os.path.join(script_dir, "outputs", "inference_audio", "output.mp4")
    config_path = os.path.join(script_dir, "model_checkpoint", "config.json")
    weight_path = os.path.join(script_dir, "model_checkpoint", "headpsoe_full_norm_with_vel_ep_160.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference_manager = get_model(config_path, weight_path, device)
    with open(config_path, 'r') as f:
        audio_model_config = yaml.safe_load(f)
    

    # Test the inference pipeline
    test_inference(
        image_path=img_path,
        audio_path=audio_path,
        output_video_path=output_video_path,
        inference_manager=inference_manager,
        audio_model_config=audio_model_config
    )
