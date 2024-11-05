import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import json

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    # Try relative import (for when used as a package)
    from .models.dit_model import DiffLiveHead
    from .models.vanilla_transformer import VanillaTransformer
except ImportError:
    # If relative import fails, try absolute import (for when run directly)
    from models.dit_model import DiffLiveHead
    from models.vanilla_transformer import VanillaTransformer

class InferenceManager:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        self.config = self.load_config(config_path)
        self.model_type = self.config["model_type"]
        self.model = self.init_model()
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def init_model(self):
        if self.model_type == 'dit':
            model = DiffLiveHead(
                fps=25,
                x_dim=self.config["x_dim"],
                person_dim=self.config["person_dim"],
                a_dim=self.config["a_dim"],
                use_indicator=False,
                feature_dim=self.config["hidden_size"],
                n_heads=self.config["num_attention_heads"],
                n_layers=self.config["num_layers"],
                use_shape_feat=self.config["use_shape_feat"],
                use_mouth_open_ratio=self.config["use_mouth_open_ratio"],
                device=self.device
            ).to(self.device)
        elif self.model_type == "vanilla":
            model = VanillaTransformer(
                x_dim=self.config["x_dim"],
                a_dim=self.config["a_dim"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                num_heads=self.config["num_attention_heads"],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
            
        return model
    
    def load_checkpoint(self, checkpoint_path):
        # Load the state dictionary
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Remove 'module.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v

        # Load the state dictionary into the model
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print(f"Model checkpoint loaded from {checkpoint_path}")

    @torch.no_grad()
    def inference(self, audio_latent, shape_tensor, prev_motion_feat=None, prev_audio_feat=None, cfg_scale=1.0, mouth_open_ratio=None, denoising_steps=None):
        """
        Perform inference on a batch of input audio and shape tensors.
        
        Args:
        - audio_latent (torch.Tensor): Audio latent tensor of shape (B, T, a_dim)
        - shape_tensor (torch.Tensor): Shape tensor of shape (B, x_dim)
        - prev_motion_feat (torch.Tensor, optional): Previous motion features of shape (B, prev_seq_length, x_dim)
        - prev_audio_feat (torch.Tensor, optional): Previous audio features of shape (B, prev_seq_length, a_dim)
        
        Returns:
        - torch.Tensor: Generated motion of shape (B, T, x_dim)
        """
        audio_latent = audio_latent.to(self.device)
        shape_tensor = shape_tensor.to(self.device)
        
        assert prev_audio_feat is not None and prev_motion_feat is not None, "Previous motion and audio features are required for inference"
        
        if prev_motion_feat is not None:
            prev_motion_feat = prev_motion_feat.to(self.device)
        if prev_audio_feat is not None:
            prev_audio_feat = prev_audio_feat.to(self.device)
        
        if self.model_type == 'dit':
            # Call the sample method of the model
            generated_motion = self.model.sample_new(
                audio_or_feat=audio_latent,
                shape_feat=shape_tensor,
                mouth_open_ratio=mouth_open_ratio,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                cfg_scale=cfg_scale,
                total_denoising_steps=denoising_steps
            )
        elif self.model_type == 'vanilla':
            x_input = torch.zeros(shape_tensor.shape[0], audio_latent.shape[1], prev_motion_feat.shape[-1]).to(self.device)
            generated_motion = self.model.inference(x_input, prev_motion_feat, audio_latent, prev_audio_feat, shape_tensor)
        
        return generated_motion

# Example usage:
# def example_inference(inference_manager, audio_latent=None, shape_tensor=None, prev_motion_feat=None, prev_audio_feat=None, cfg_scale=1.0, total_denoising_steps=None):
#     assert audio_latent is not None and shape_tensor is not None, "Audio and shape tensors are required for inference"
    
#     generated_motion = inference_manager.inference(audio_latent, shape_tensor, prev_motion_feat, prev_audio_feat, cfg_scale, total_denoising_steps)
#     print(f"Generated motion shape: {generated_motion.shape}")
#     return generated_motion

def get_model(config_path, checkpoint_path, device='cuda'):
    if not os.path.exists(config_path):
        assert False, f"Config file not found at {config_path}"
    if not os.path.exists(checkpoint_path):
        assert False, f"Checkpoint file not found at {checkpoint_path}"
    
    return InferenceManager(config_path, checkpoint_path, device)

if __name__ == "__main__":
    batch_size = 2
    random_seq_length = 65
    prev_seq_length = 10
    audio_dim = 768
    shape_dim = 63
    motion_dim = 63
    random_audio_latent = torch.randn(batch_size, random_seq_length, audio_dim)
    random_shape_tensor = torch.randn(batch_size, shape_dim)
    random_prev_motion_feat = torch.randn(batch_size, prev_seq_length, motion_dim)
    random_prev_audio_feat = torch.randn(batch_size, prev_seq_length, audio_dim)
    ckp_dir = "/mnt/e/wsl_projects/audio_live/output/0911-15-21-33"
    config_sample_path = ckp_dir + "/config.json"
    checkpoint_sample_path = ckp_dir + "/checkpoint_epoch_500/model.pth"
    example_inference(config_sample_path, checkpoint_sample_path, random_audio_latent, random_shape_tensor, random_prev_motion_feat, random_prev_audio_feat)
