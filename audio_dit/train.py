import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import reduce
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from diffusers import DDIMScheduler
import argparse
import matplotlib.pyplot as plt
import os
import time
import json

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
torch.manual_seed(3)

from models.dit_model import DiffLiveHead
from models.vanilla_transformer import VanillaTransformer
from models.faceformer import FaceFormer
from dataset import load_npy_files, process_motion_tensor, MotionAudioDataset

def setup(rank, world_size):
    print(f"Initializing process group on rank {rank}")
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

class TrainingManager:
    def __init__(self, args, config, rank, world_size, 
                 train_data, val_data):
        self.args = args
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
        self.model_type = config["model_type"]
        self.model = None
        self.noise_scheduler = None
        self.optimizer = None
        self.scaler = None
        self.lr_scheduler = None
        self.dataloader = None
        self.val_dataloaders = None
        
        self.epoch_losses = []
        self.iteration_losses = []
        self.val_loss = []
        self.val_feature_loss = []
        self.lr_during_training = []
        
        self.output_dir = os.path.join(args.output_dir, time.strftime("%m%d-%H-%M-%S"))
        self.checkpoint_dir = args.checkpoint_dir

        self.train_data = train_data # motion_latents, audio_latents, shape_latents, mouth_latents, end_indices
        self.val_data = val_data # N datasets of (motion_latents, audio_latents, shape_latents, mouth_latents, end_indices)

        self.prev_seq_length = 10
        self.data_regulator = None
        
        self.loss_weight = torch.tensor(self.config["loss_weight"]).to(self.device)
        
        if self.rank == 0 and not self.config["validate_only"]:
            os.makedirs(self.output_dir, exist_ok=True)
            
            config_path = os.path.join(self.output_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Config saved to {config_path}")

    def init_model(self):
        # Initialize model based on model_type
        if self.model_type == "dit":
            self.model = DiffLiveHead(
                fps=25,
                x_dim=self.config["x_dim"],
                person_dim=self.config["person_dim"],
                a_dim=self.config["a_dim"],
                use_indicator=False,
                feature_dim=self.config["hidden_size"],
                n_heads = self.config["num_attention_heads"],  
                n_layers=self.config["num_layers"],
                n_diff_steps = self.config["n_diff_steps"],
                use_shape_feat=self.config["use_shape_feat"],
                use_mouth_open_ratio=self.config["use_mouth_open_ratio"],
                device=self.device
            ).to(self.device)
        elif self.model_type == "vanilla":
            self.model = VanillaTransformer(
                x_dim=self.config["x_dim"],
                a_dim=self.config["a_dim"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                num_heads=self.config["num_attention_heads"],
            ).to(self.device)
        elif self.model_type == "faceformer":
            self.model = FaceFormer(
                x_dim=self.config["x_dim"],
                a_dim=self.config["a_dim"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                num_heads=self.config["num_attention_heads"],
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # self.model = self.model.bfloat16()
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

    def load_checkpoint(self):
        model_path = os.path.join(self.checkpoint_dir, "model.pth")
        optimizer_path = os.path.join(self.checkpoint_dir, "optimizer.pth")
        scheduler_path = os.path.join(self.checkpoint_dir, "scheduler.pth")

        if os.path.exists(model_path) or os.path.exists(optimizer_path):
            if os.path.exists(model_path): 
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                if self.rank == 0:
                    print(f"Model checkpoint loaded from {model_path}")
            if os.path.exists(optimizer_path) and self.optimizer is not None:
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                if self.rank == 0:
                    print(f"Checkpoint loaded from {optimizer_path}")
            if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
                if self.rank == 0:
                    print(f"Scheduler checkpoint loaded from {scheduler_path}")
        elif self.rank == 0:
            print(f"No checkpoint found in {self.checkpoint_dir}")
            
    def get_scheduler(self):
        # if self.config["model_type"] == "dit" or self.config["model_type"] == "faceformer":
        if self.config["scheduler"] == "warmup_cosine":
            from scheduler import GradualWarmupScheduler
            after_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config["lr_max_iters"],
                                                                self.config["learning_rate"] * self.config["lr_min_scale"])
            return GradualWarmupScheduler(self.optimizer, 1,self.config["warmup_iters"], after_scheduler)
        elif self.config["scheduler"] == "warmup":
            # use ordinary warmup
            from scheduler import GradualWarmupScheduler
            return GradualWarmupScheduler(self.optimizer, 1, self.config["warmup_iters"])
        else:
            return None
        
        # elif self.config["model_type"] == "vanilla":
        #     if self.config["scheduler"] == "none":
        #         return None
        #     elif self.config["scheduler"] == "cosine":
        #         return torch.optim.lr_scheduler.CosineAnnealingLR(
        #             self.optimizer,
        #             T_max=self.config["lr_max_iters"],
        #             eta_min=self.config["learning_rate"] * self.config["lr_min_scale"]
        #         )
        #     elif self.config["scheduler"] == "linear":
        #         return torch.optim.lr_scheduler.LinearLR(
        #             self.optimizer,
        #             start_factor=1.0,
        #             total_iters=self.config["lr_max_iters"],
        #             end_factor=self.config["lr_min_scale"],
        #         )
        #     else:
        #         raise ValueError(f"Unknown scheduler type: {self.config['scheduler']}")

    # def normalize_features(self, validate=False):
    #     # Standardize (mean=0, std=1)
    #     if self.data_regulator is None:
    #         motion_mean = torch.mean(self.motion_latents, dim=0, keepdim=True)
    #         motion_std = torch.std(self.motion_latents, dim=0, keepdim=True)
    #         self.data_regulator = {"motion_mean": motion_mean, "motion_std": motion_std}
    #         torch.save(self.data_regulator, os.path.join(self.output_dir, "data_regularization.pt"))
            
    #     if validate:
    #         self.val_motion_latents = (self.val_motion_latents - self.data_regulator["motion_mean"]) / (self.data_regulator["motion_std"] + 1e-8)
    #     else:
    #         self.motion_latents = (self.motion_latents - self.data_regulator["motion_mean"]) / (self.data_regulator["motion_std"] + 1e-8)      
        
    
    def run(self):
        self.model.train()
        if self.checkpoint_dir:
            self.load_checkpoint()
        # self.normalize_features(validate=False)
        # self.normalize_features(validate=True)
        
        dataset = MotionAudioDataset(self.train_data)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], sampler=sampler, pin_memory=True)
        if len(self.val_data) > 0:
            valid_datasets_list = [MotionAudioDataset(val_data)
                                   for val_data in self.val_data]
            valid_samplers = [DistributedSampler(valid_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False) for valid_dataset in valid_datasets_list]
            self.val_dataloaders = [DataLoader(valid_dataset, batch_size=self.config["valid_batch_size"], sampler=valid_sampler, pin_memory=True) for valid_dataset, valid_sampler in zip(valid_datasets_list, valid_samplers)]
        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        self.lr_scheduler = self.get_scheduler()
        self.scaler = GradScaler()
        
        data_size_per_rank = len(self.dataloader.dataset) // self.world_size + len(self.dataloader.dataset) % self.world_size
        iter_per_epoch = data_size_per_rank // self.config["batch_size"] + data_size_per_rank % self.config["batch_size"]
        self.config["num_epochs"] = self.config["num_iterations"] // iter_per_epoch
        epoch_pbar = tqdm(range(self.config["num_epochs"]), desc="Training Epochs", disable=self.rank != 0)
        # epoch_pbar = range(self.config["num_epochs"])
        self.iteration = 0
        for epoch in epoch_pbar:
            self.dataloader.sampler.set_epoch(epoch)
            batch_loss = torch.zeros(1).to(self.device)
            mini_batch_pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", leave=False, disable=self.rank != 0)
            # mini_batch_pbar = self.dataloader
            for mini_batch in mini_batch_pbar:
                loss = self.train_step(mini_batch)
                # Sum the total loss across all processes
                reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                loss /= self.world_size
                batch_loss += loss
                current_lr = self.optimizer.param_groups[0]['lr']
                
                if self.rank == 0:
                    self.iteration_losses.append(loss.item())
                    self.lr_during_training.append(current_lr)
                self.iteration += 1
                if self.iteration % 10 == 0:
                    mini_batch_pbar.set_postfix({"Loss": f"{loss.item():.2e}", "LR": f"{current_lr:.6e}"})
                
            if self.rank == 0:
                avg_loss = batch_loss.item() / (len(self.dataloader) * self.world_size)
                self.epoch_losses.append(avg_loss)
                epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.2e}", "LR": f"{current_lr:.6e}"})

            if (epoch + 1) % self.config["save_interval"] == 0 and self.rank == 0:
                val_loss, val_feature_loss = self.validate()
                self.val_loss.append(val_loss)
                self.val_feature_loss.append([f'{i:.5e}' for i in np.mean(val_feature_loss, axis=0)])
            if (epoch + 1) % self.config["save_interval"] == 0 and self.rank == 0:
                self.plot_and_save_loss()
            if (epoch + 1) % (self.config["save_interval"] * 5) == 0 and self.rank == 0:
                self.save_checkpoint(epoch + 1)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        t0 = time.time()

        x = batch['motion_latent'].to(self.device)
        a = batch['audio_latent'].to(self.device)
        s = batch['shape_latent'].to(self.device)
        m = batch['mouth_latent'].to(self.device)
        end_indices = batch['end_indices'].to(self.device)
        # batch_size = x.shape[0]

        x_prev, x_gt = x[:, :self.prev_seq_length], x[:, self.prev_seq_length:]
        a_prev, a_train = a[:, :self.prev_seq_length], a[:, self.prev_seq_length:]
        x_shape = s
        # t1 = time.time()
        # Convert inputs to BF16
        # x_gt = x_gt.bfloat16()
        # x_prev = x_prev.bfloat16()
        # a_train = a_train.bfloat16()
        # a_prev = a_prev.bfloat16()
        if self.model_type == "dit":
            # by default not use indicator  
            
            noise, x_out, prev_motion_coef, prev_audio_feat = \
                self.model(motion_feat = x_gt, audio_or_feat = a_train, 
                           shape_feat=x_shape, style_feat=None, mouth_open_ratio = m,
                           prev_motion_feat = x_prev, prev_audio_feat = a_prev)
            x_pred = x_out
            
            # non_zero_mask = (x_gt != 0)
            # squared_diff = (x_pred - x_gt) ** 2
            # masked_squared_diff = squared_diff * non_zero_mask
            # loss = masked_squared_diff.sum() / non_zero_mask.sum()
            x_pred = self.loss_weight * x_pred
            x_gt = self.loss_weight * x

            batch_size, seq_length, _ = x_gt.shape
            mask = torch.arange(seq_length, device=self.device).expand(batch_size, seq_length) < end_indices.unsqueeze(1)
            # Apply the mask to both x_pred and x_gt
            x_pred = x_pred[mask]
            x_gt = x_gt[mask]

            loss = F.mse_loss(x_pred, x_gt)
            
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler and self.iteration < self.config["lr_max_iters"] + self.config["warmup_iters"]: 
                self.lr_scheduler.step()
            
        elif self.model_type == "vanilla":
            # with autocast(dtype=torch.bfloat16):
            x_input = torch.zeros_like(x_gt)  # Zero out the motion data for now
            x_pred = self.model(x_input, x_prev, a_train, a_prev, x_shape)
            loss = F.mse_loss(x_pred, x_gt)
            
            loss.backward()
            self.optimizer.step()
        
            if self.lr_scheduler:
                # self.scaler.step(self.optimizer)
                self.lr_scheduler.step()
            # self.scaler.update()
        elif self.model_type == "faceformer":
            x_pred = self.model(x_gt, x_prev, a_train, a_prev, x_shape)
            loss = F.mse_loss(x_pred, x[:, :-1, :])
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        # torch.cuda.synchronize()
        # t2 = time.time()
        # if self.rank == 0:
        #     print(f"Data prep time: {(t1 - t0) * 1000:.2f}ms, Forward pass time: {(t2 - t1) * 1000:.2f}ms")
        return loss.detach()

    def save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pth"))
        if self.lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pth"))
        print(f"Checkpoint saved at epoch {epoch}")
    
    def plot_and_save_loss(self):
        marks = [5, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 1e-6]
        fig, axes = plt.subplots(len(marks), 2, figsize=(40, 5 * len(marks)))
        val_loss_to_plot = [val_loss[-1] for val_loss in self.val_loss]
        
        for i, mark in enumerate(marks):
            # Iteration losses
            filtered_losses = [(j, loss) for j, loss in enumerate(self.iteration_losses) if loss <= mark]
            
            if filtered_losses:
                iterations, losses = zip(*filtered_losses)
                axes[i, 0].plot(iterations, losses)
                axes[i, 0].set_title(f"Iteration Loss (up to {mark:.0e})")
                axes[i, 0].set_xlabel("Iteration")
                axes[i, 0].set_ylabel("Loss")
                axes[i, 0].set_ylim(0, mark)
                
                if i == len(marks) - 1:  # For the smallest mark, use log scale
                    axes[i, 0].set_yscale('log')
            
            # Validation losses
            filtered_val_losses = [(j, loss) for j, loss in enumerate(val_loss_to_plot) if loss <= mark]
            
            if filtered_val_losses:
                val_iterations, val_losses = zip(*filtered_val_losses)
                axes[i, 1].plot(val_iterations, val_losses)
                axes[i, 1].set_title(f"Validation Loss (up to {mark:.0e})")
                axes[i, 1].set_xlabel("Validation Iteration")
                axes[i, 1].set_ylabel("Loss")
                axes[i, 1].set_ylim(0, mark)
                
                if i == len(marks) - 1:  # For the smallest mark, use log scale
                    axes[i, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss_plot.png"))
        plt.close()

        # Save losses to files
        with open(os.path.join(self.output_dir, "epoch_losses.txt"), "a") as f:
            for loss in self.epoch_losses:
                f.write(f"{loss}\n")
            self.epoch_losses = []
        
        with open(os.path.join(self.output_dir, "iteration_losses.txt"), "a") as f:
            for loss in self.iteration_losses:
                f.write(f"{loss}\n")
            self.iteration_losses = []
            
        with open(os.path.join(self.output_dir, "validation_feature_losses.txt"), "a") as f:
            for loss in self.val_feature_loss:
                f.write(f"{loss}\n")
            self.val_feature_loss = []
            
        if self.rank == 0:
            print("Loss plot and data saved")


    def validate(self):
        self.model.eval()
        if self.checkpoint_dir and self.config["validate_only"]:
            self.load_checkpoint()
        
        # if self.config["validate_only"]:
        #     assert self.data_regulator is not None, "Data regulator must be initialized before validation"
        #     self.normalize_features(validate=True)
        
        total_loss = []
        num_batches = []
        feature_losses = []
        all_val_info = []
        all_gen_info = []

        # assert len(self.val_motion_latents) == len(self.val_audio_latents), f"Validation motion and audio latents must have the same length, \
        #     currently have {len(self.val_motion_latents)} and {len(self.val_audio_latents)}"
        assert self.val_data is not None
        valid_datasets_list = [MotionAudioDataset(val_data)
                                for val_data in self.val_data]
        valid_samplers = [DistributedSampler(valid_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False) for valid_dataset in valid_datasets_list]
        self.val_dataloaders = [DataLoader(valid_dataset, batch_size=self.config["valid_batch_size"], sampler=valid_sampler, pin_memory=True) for valid_dataset, valid_sampler in zip(valid_datasets_list, valid_samplers)]


        with torch.no_grad():
            total_loss = [0 for _ in range(len(self.val_dataloaders) + 1)]
            num_batches = [0 for _ in range(len(self.val_dataloaders) + 1)]
            for val_dataloader_idx, val_dataloader in enumerate(self.val_dataloaders):
                for batch in val_dataloader:
                    x = batch['motion_latent'].to(self.device)
                    a = batch['audio_latent'].to(self.device)
                    s = batch['shape_latent'].to(self.device)
                    m = batch['mouth_latent'].to(self.device)

                    x_prev, x_gt = x[:, :self.prev_seq_length], x[:, self.prev_seq_length:]
                    a_prev, a_train = a[:, :self.prev_seq_length], a[:, self.prev_seq_length:]
                    x_shape = s
                    if self.model_type == "dit":
                        noise, x_out, prev_motion_coef, prev_audio_feat = \
                            self.model(x_gt, a_train, shape_feat=x_shape, style_feat=None, mouth_open_ratio = m,
                            prev_motion_feat = x_prev, prev_audio_feat = a_prev)
                        x_pred = x_out
                        mse_loss = F.mse_loss(x_pred, x)
                        l1_abs_loss = torch.abs(x_pred - x)
                        
                    elif self.model_type == "vanilla":
                        x_input = torch.zeros_like(x_gt)
                        x_pred = self.model.module.inference(x_input, x_prev, a_train, a_prev, x_shape)

                        # Calculate loss for each feature dimension
                        mse_loss = F.mse_loss(x_pred, x_gt)
                        l1_abs_loss = torch.abs(x_pred - x_gt)
                        
                    elif self.model_type == "faceformer":
                        x_pred = self.model(x_gt, x_prev, a_train, a_prev, x_shape)
                        mse_loss = F.mse_loss(x_pred, x[:, :-1, :])
                        l1_abs_loss = torch.abs(x_pred - x_gt)
                    
                    l1_abs_loss = l1_abs_loss.reshape(l1_abs_loss.shape[0] * l1_abs_loss.shape[1], -1) # (batch_size * seq_len, kp * 3)
                    x_gt_reshaped = x_gt.reshape(x_gt.shape[0] * x_gt.shape[1], -1) # (batch_size * seq_len, kp * 3)
                    x_pred_reshaped = x_pred.reshape(x_pred.shape[0] * x_pred.shape[1], -1) # (batch_size * seq_len, kp * 3)    
                    feature_losses.append(l1_abs_loss.cpu().numpy())
                    all_val_info.append(x_gt_reshaped.cpu().numpy())
                    all_gen_info.append(x_pred_reshaped.cpu().numpy())
                    
                    loss = mse_loss.mean()
                    total_loss[val_dataloader_idx] += loss.item()
                    total_loss[-1] += loss.item()
                    num_batches[val_dataloader_idx] += 1
                    num_batches[-1] += 1

        # Gather losses from all processes
        # total_loss = [total_loss[i] for i in range(len(self.val_dataloaders) + 1)]
        # torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        # num_batches = [num_batches[i] for i in range(len(self.val_dataloaders) + 1)]
        # torch.distributed.all_reduce(num_batches, op=torch.distributed.ReduceOp.SUM)

        avg_loss = [total_loss[i] / num_batches[i] for i in range(len(self.val_dataloaders) + 1)]

        feature_losses = np.concatenate(feature_losses, axis=0) # (all_frames, kp * 3)
        all_val_info = np.concatenate(all_val_info, axis=0) # (all_frames, kp * 3)
        all_gen_info = np.concatenate(all_gen_info, axis=0) # (all_frames, kp * 3)
        if self.rank == 0 :
            val_d_names = ["vox2", "hdtf"]
            full_output = ""
            for i in range(len(val_d_names)):
                if val_d_names[i] in self.config["dataset"]:
                    full_output += f"{val_d_names[i]}: {avg_loss[i]:.5e}; "
            full_output += f"Total: {avg_loss[-1]:.5e}"
            print(f"Avg validation Loss for {full_output}")
            print(f"Mean loss per feature: {[f'{i:.5e}' for i in np.mean(feature_losses, axis=0)]}")
            
        if self.rank == 0 and self.config["validate_only"]:
            print("\nFeature-wise loss statistics:")
            print(f"Mean loss per feature: {[f'{i:.5e}' for i in np.mean(feature_losses, axis=0)]}")
            print(f"Std dev of loss per feature: {[f'{i:.5e}' for i in np.std(feature_losses, axis=0)]}")
            print(f"Min loss per feature: {[f'{i:.5e}' for i in np.min(feature_losses, axis=0)]}")
            print(f"Max loss per feature: {[f'{i:.5e}' for i in np.max(feature_losses, axis=0)]}")
            
            print("\nGround truth features:")
            print(f"Mean: {[f'{i:.5e}' for i in np.mean(all_val_info, axis=0)]}")
            print(f"Std dev: {[f'{i:.5e}' for i in np.std(all_val_info, axis=0)]}")
            print(f"Min: {[f'{i:.5e}' for i in np.min(all_val_info, axis=0)]}")
            print(f"Max: {[f'{i:.5e}' for i in np.max(all_val_info, axis=0)]}")
            
            print("\nGenerated features:")
            print(f"Mean: {[f'{i:.5e}' for i in np.mean(all_gen_info, axis=0)]}")
            print(f"Std dev: {[f'{i:.5e}' for i in np.std(all_gen_info, axis=0)]}")
            print(f"Min: {[f'{i:.5e}' for i in np.min(all_gen_info, axis=0)]}")
            print(f"Max: {[f'{i:.5e}' for i in np.max(all_gen_info, axis=0)]}")
            # print("\nLoss for each feature:")
            # print("Feature\tLoss\t\tZ-Score")
            # mean_loss = np.mean(feature_losses, axis=0)
            # std_loss = np.std(feature_losses, axis=0)
            # for idx, loss in enumerate(feature_losses):
            #     z_score = (loss - mean_loss) / std_loss
            #     print(f"{idx}\t{loss:.5e}\t{z_score:.5f}")
                
            # ranked_indices = np.argsort(feature_losses)
            
            # print("\nFull ranked list of features by loss:")
            # for rank, idx in enumerate(reversed(ranked_indices)):
            #     print(f"{idx} with loss {feature_losses[idx]:.5e}")
            
            # print("\nTop 5 features with highest loss:")
            # top_5_indices = ranked_indices[-5:][::-1]
            # for idx in top_5_indices:
            #     print(f"Feature {idx}: {feature_losses[idx]:.5e}")
            
            # print("\nBottom 5 features with lowest loss:")
            # bottom_5_indices = ranked_indices[:5]
            # for idx in bottom_5_indices:
            #     print(f"Feature {idx}: {feature_losses[idx]:.5e}")

        return avg_loss, feature_losses

def rank_entrance(rank, world_size, args, config, 
                  train_data, val_data):
    setup(rank, world_size)
    manager = TrainingManager(args, config, rank, world_size, 
                              train_data, val_data)
    manager.init_model()
    if config["validate_only"]:
        manager.validate()
    else:
        manager.run()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the motion-audio model")
    parser.add_argument("-c", "--checkpoint_dir", type=str, default=None, 
                        help="Directory containing checkpoint files (model.pth and optimizer.pth)")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                        help="Number of GPUs to use")
    parser.add_argument("--model_type", type=str, choices=["dit", "vanilla", "faceformer"], default="dit",
                        help="Type of model to use: 'dit' or 'vanilla'")
    parser.add_argument("-val", "--validate_only", action="store_true",
                        help="Run validation only, without training") # by default training would include validation
    on_remote = True
    if not on_remote:
        parser.add_argument("-va", "--vox2_audio_root", type=str, default='/mnt/e/data/live_latent/audio_latent/')
        parser.add_argument("-vm", "--vox2_motion_root", type=str, default='/mnt/e/data/live_latent/motion_temp/')
        parser.add_argument("-htn", "--hdtf_train_root", type=str, default="/mnt/e/data/diffposetalk_data/TFHP_raw/train_split/")
        parser.add_argument("-htt", "--hdtf_test_root", type=str, default="/mnt/e/data/diffposetalk_data/TFHP_raw/test_split/")
        used_dataset = "vox2"
        parser.add_argument("--vox2_train_end_idx", type=int, default=10)
    else:
        parser.add_argument("-va", "--vox2_audio_root", type=str, default='/home/ubuntu/vox2-az/audio_latent/')
        parser.add_argument("-vm", "--vox2_motion_root", type=str, default='/home/ubuntu/vox2-az/new_live_latent/')
        parser.add_argument("-htn", "--hdtf_train_root", type=str, default="/home/ubuntu/vox2-az/hdtf/train_split/")
        parser.add_argument("-htt", "--hdtf_test_root", type=str, default="/home/ubuntu/vox2-az/hdtf/test_split/")
        used_dataset = "vox2/hdtf"
        parser.add_argument("--vox2_train_end_idx", type=int, default=4200)
    parser.add_argument("-o", "--output_dir", type=str, default="output")
    parser.add_argument("--vox2_train_start_idx", type=int, default=0)
    parser.add_argument("--hdtf_train_start_idx", type=int, default=0)
    parser.add_argument("--hdtf_train_end_idx", type=int, default=337)
    parser.add_argument("--vox2_validate_start_idx", type=int, default=4200) # 4200
    parser.add_argument("--dataset", type=str, default=used_dataset) # vox2, hdtf, or both
    parser.add_argument("--vox2_validate_end_idx", type=int, default=4250) # 4617
    args = parser.parse_args()
    
     # Scheduler type. Accept ["none", "warmup_cosine", "warmup", "warmup_decay", "cosine", "linear"]
    param_combo = {
        1: {
            "save_interval": 500,
            "lr_min_scale": 0.2,
            "warmup_iters": 600,
            "lr_max_iters": 500000,
            "learning_rate": 1e-4,
            "scheduler": "none",
        },
        10: {
            "save_interval": 1,
            "lr_min_scale": 0.2,
            "warmup_iters": 600,
            "lr_max_iters": 500000,
            "learning_rate": 1e-4,
            "scheduler": "none",
        },
        80: {
            "save_interval": 1,
            "lr_min_scale": 0.2,
            "warmup_iters": 600,
            "lr_max_iters": 500000,
            "learning_rate": 1e-4,
            "scheduler": "none",
        },
        387: {
            "save_interval": 3,
            "lr_min_scale": 0.05,
            "warmup_iters": 5000,
            "lr_max_iters": 200000,
            "learning_rate": 2e-5,
            "scheduler": "warmup",
        },
        400: {
            "save_interval": 5,
            "lr_min_scale": 0.02,
            "warmup_iters": 30000,
            "lr_max_iters": 200000,
            "learning_rate": 5e-5,
            "scheduler": "warmup",
        },
        1137: {
            "save_interval": 1,
            "lr_min_scale": 0.5,
            "warmup_iters": 10000,
            "lr_max_iters": 200000,
            "learning_rate": 2e-5,
            "scheduler": "warmup_cosine",
        },
        1000: {
            "save_interval": 1,
            "lr_min_scale": 0.5,
            "warmup_iters": 10000,
            "lr_max_iters": 200000,
            "learning_rate": 2e-5,
            "scheduler": "warmup_cosine",
        },
        4200: {
            "save_interval": 1,
            "lr_min_scale": 0.2,
            "warmup_iters": 10000,
            "lr_max_iters": 1000000,
            "learning_rate": 5e-5,
            "scheduler": "warmup_cosine",
        },
    }
    count_key = args.vox2_train_end_idx - args.vox2_train_start_idx

    latent_mask_1 = [ 4, 6, 7, 22, 33, 34, 40, 43, 45, 46, 48, 51, 52, 53, 57, 58, 59, 60, 61, 62 ] # deleted 49,
    loss_weight   = [ 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, ]
    latent_mask_1 += [i for i in range(63, 63 + 5)] # add 5 headpose features
    loss_weight +=  [0.01, 0.01, 0.01, 0.1, 0.1] # relative scale is [0.003, 0.002, 0.003, 0.2, 0.2]
    latent_bound_list =[
        -0.05029296875,        0.0857086181640625,   -0.07587742805480957,  0.058624267578125,   -0.0004341602325439453,  0.00019466876983642578, 
        -0.038482666015625,    0.0345458984375,      -0.030120849609375,    0.038360595703125,   -3.0279159545898438e-05, 1.3887882232666016e-05,
        -0.0364990234375,      0.036102294921875,    -0.043212890625,       0.046844482421875,   -4.3332576751708984e-05, 1.8775463104248047e-05, 
        -0.03326416015625,     0.057373046875,       -0.03460693359375,     0.031707763671875,   -0.0001958608627319336,  0.0005192756652832031,
        -0.0728759765625,      0.0587158203125,      -0.04840087890625,     0.039642333984375,   -0.00025916099548339844, 0.00048089027404785156, 
        -0.09722900390625,     0.12469482421875,     -0.1556396484375,      0.09326171875,       -0.00018024444580078125, 0.00037860870361328125,
        -0.0279384758323431,   0.010650634765625,    -0.039306640625,       0.03802490234375,    -1.049041748046875e-05,  3.6954879760742188e-06, 
        -0.032989501953125,    0.044281005859375,    -0.037261962890625,    0.0433349609375,     -0.00022792529489379376, 0.0003247261047363281,
        -0.0288234855979681,   0.006015777587890625, -0.0108795166015625,   0.0134124755859375,  -7.784366607666016e-05,  5.2034854888916016e-05, 
        -0.01531982421875,     0.027801513671875,    -0.036041259765625,    0.0242156982421875,  -8.83340835571289e-05,   2.6464462280273438e-05,
        -0.06463623046875,     0.0303802490234375,   -0.0446159653365612,   0.03619384765625,    -0.02947998046875,       0.030792236328125, 
        -0.0159145500510931,   0.018890380859375,    -0.01898193359375,     0.0264739990234375,  -6.103515625e-05,        3.266334533691406e-05,
        -0.0094450069591403,   0.00604248046875,     -0.005710510071367025, 0.00557708740234375, -2.866983413696289e-05,  1.4543533325195312e-05, 
        -0.0265350341796875,   0.01186370849609375,  -0.0227047111839056,   0.01386260986328125, -0.000133514404296875,   6.687641143798828e-05, 
        -0.01129150390625,     0.01331329345703125,  -0.0251922607421875,   0.0195465087890625,  -8.285045623779297e-06,  6.079673767089844e-06, 
        -0.0141599727794528,   0.018341064453125,    -0.0189971923828125,   0.029296875,         -6.049728108337149e-05,  3.057718276977539e-05, 
        -0.01216888427734375,  0.02069091796875,     -0.016754150390625,    0.017974853515625,   -0.00014078617095947266, 6.842613220214844e-05, 
        -0.01910400390625,     0.016204833984375,    -0.025634765625,       0.04150390625,       -0.0100250244140625,     0.00991058349609375, 
        -0.005596160888671875, 0.01132965087890625,  -0.0269775390625,      0.02166748046875,    -0.000362396240234375,   9.059906005859375e-05,
        -0.0325927734375,      0.038818359375,       -0.05877685546875,     0.076416015625,      -0.02215576171875,       0.019775390625,
        -0.0219573974609375,   0.0247344970703125,   -0.039764404296875,    0.045,               -0.01512908935546875,    0.017730712890625,
        -21,                   25,                   -30,                   30,                  -23,                     23,
        -0.3,                  0.3,                  -0.06,                 0.28,
        ]

    '''
    0    -0.05029296875,        0.0857086181640625,   -0.07587742805480957,  0.058624267578125,   -0.0004341602325439453,  0.00019466876983642578, 
    3    -0.038482666015625,    0.0345458984375,      -0.030120849609375,    0.038360595703125,   -3.0279159545898438e-05, 1.3887882232666016e-05,
    6    -0.0364990234375,      0.036102294921875,    -0.043212890625,       0.046844482421875,   -4.3332576751708984e-05, 1.8775463104248047e-05, 
    9    -0.03326416015625,     0.057373046875,       -0.03460693359375,     0.031707763671875,   -0.0001958608627319336,  0.0005192756652832031,
    12    -0.0728759765625,      0.0587158203125,      -0.04840087890625,     0.039642333984375,   -0.00025916099548339844, 0.00048089027404785156, 
    15    -0.09722900390625,     0.12469482421875,     -0.1556396484375,      0.09326171875,       -0.00018024444580078125, 0.00037860870361328125,
    18    -0.0279384758323431,   0.010650634765625,    -0.039306640625,       0.03802490234375,    -1.049041748046875e-05,  3.6954879760742188e-06, 
    21    -0.032989501953125,    0.044281005859375,    -0.037261962890625,    0.0433349609375,     -0.00022792529489379376, 0.0003247261047363281,
    24    -0.0288234855979681,   0.006015777587890625, -0.0108795166015625,   0.0134124755859375,  -7.784366607666016e-05,  5.2034854888916016e-05, 
    27    -0.01531982421875,     0.027801513671875,    -0.036041259765625,    0.0242156982421875,  -8.83340835571289e-05,   2.6464462280273438e-05,
    30    -0.06463623046875,     0.0303802490234375,   -0.0446159653365612,   0.03619384765625,    -0.02947998046875,       0.030792236328125, 
    33    -0.0159145500510931,   0.018890380859375,    -0.01898193359375,     0.0264739990234375,  -6.103515625e-05,        3.266334533691406e-05,
    36    -0.0094450069591403,   0.00604248046875,     -0.005710510071367025, 0.00557708740234375, -2.866983413696289e-05,  1.4543533325195312e-05, 
    39    -0.0265350341796875,   0.01186370849609375,  -0.0227047111839056,   0.01386260986328125, -0.000133514404296875,   6.687641143798828e-05, 
    42    -0.01129150390625,     0.01331329345703125,  -0.0251922607421875,   0.0195465087890625,  -8.285045623779297e-06,  6.079673767089844e-06, 
    45    -0.0141599727794528,   0.018341064453125,    -0.0189971923828125,   0.029296875,         -6.049728108337149e-05,  3.057718276977539e-05, 
    48    -0.01216888427734375,  0.02069091796875,     -0.016754150390625,    0.017974853515625,   -0.00014078617095947266, 6.842613220214844e-05, 
    51    -0.01910400390625,     0.016204833984375,    -0.025634765625,       0.04150390625,       -0.0100250244140625,     0.00991058349609375, 
    54    -0.005596160888671875, 0.01132965087890625,  -0.0269775390625,      0.02166748046875,    -0.000362396240234375,   9.059906005859375e-05,
    57    -0.0325927734375,      0.038818359375,       -   -0.02215576171875,       0.019775390625, 
    60     -0.0219573974609375,  0.0247344970703125,   -0.039764404296875,    0.045,               -0.01512908935546875,    0.017730712890625,    ]
    '''
    
    
    config = {
        # Model parameters
        "model_type" : args.model_type,
        "x_dim": len(latent_mask_1), # add 5, last 3 for rot, 2 for xy translation
        "person_dim": 63,  # Dimension for person latent
        "a_dim": 768,  # Dimension for audio latent
        "hidden_size": 512,  # Hidden size for the transformer
        "num_layers": 8,  # Number of transformer layers
        "num_attention_heads": 8,  # Number of attention heads
        # Training parameters
        "n_diff_steps" : 50, # Number of diffusion steps
        "batch_size": 32,  # Batch size for training
        "learning_rate": param_combo[count_key]["learning_rate"],  # Learning rate for the optimizer
        "num_epochs": -1,  # Update based on the number of iterations
        "save_interval": param_combo[count_key]["save_interval"],  # Interval for saving checkpoints
        "num_iterations": 4000000,  # Number of training iterations
        "scheduler": param_combo[count_key]["scheduler"],
        "warmup_iters": param_combo[count_key]["warmup_iters"],  # Number of warmup iterations
        "lr_min_scale": param_combo[count_key]["lr_min_scale"],  # Minimum learning rate scale
        "lr_max_iters": param_combo[count_key]["lr_max_iters"],  # Number of iterations for cosine scheduler
        "weight_decay": 1e-4,  # Weight decay for the optimizer
        # condition parameter
        "use_shape_feat": True, # whether to use condition
        "use_mouth_open_ratio": True, # condition type, concat or add
        # dataset parameters
        "dataset": args.dataset,
        "motion_latent_type": "exp",  # Motion latent type. Accept ["exp", "x_d"]
        "latent_mask_1": latent_mask_1,
        "latent_bound": latent_bound_list,
        "loss_weight": loss_weight,
        # training data parameters
        "vox2_train_start_idx": args.vox2_train_start_idx,  # Starting index for training data
        "vox2_train_end_idx": args.vox2_train_end_idx,  # Ending index for training data
        "hdtf_train_start_idx": args.hdtf_train_start_idx,  # Starting index for training data
        "hdtf_train_end_idx": args.hdtf_train_end_idx,  # Ending index for training data
        # Validation parameters
        "validate_only":args.validate_only, # Whether to validate the model, no matter if in training or testing mode
                          # full dataset used 4617 uid. use last 17 uid for validation by default
        # "validate_start_idx": 4600,  # Starting index for validation
        "valid_batch_size": 512,  # Batch size for validation
        "validate_interval": 1000,  # Interval for validation
        "vox2_validate_start_idx": args.vox2_validate_start_idx,
        "vox2_validate_end_idx": args.vox2_validate_end_idx,
    }

    # Load data once in the main process
    train_data = [torch.Tensor([]) for _ in range(5)]   # motion, audio, shape, mouth, end_indices
    val_data = []  # motion, audio, shape, mouth, end_indices
    # train_audio_latents, train_motion_latents, train_shape_latents, train_mouth_latents, train_flag_latents, \
    #     val_audio_latents, val_motion_latents, val_shape_latents, val_mouth_latents, val_flag_latents = \
    #     torch.Tensor([]), torch.Tensor([]), torch.Tensor([]),torch.Tensor([]), torch.Tensor([]), [], [], [], [], []
    
    if "vox2" in args.dataset:
        vox2_val_data = load_npy_files(args.vox2_audio_root, args.vox2_motion_root,
                                                                         args.vox2_validate_start_idx, args.vox2_validate_end_idx, 
                                                                        config["motion_latent_type"], config["latent_mask_1"], config["latent_bound"])
        val_data.append(vox2_val_data)
        print(f"Vox2 validation Data loaded. audio shape: {val_data[-1][0].shape}, \
                motion shape: {val_data[-1][1].shape}, shape shape: {val_data[-1][2].shape}, mouth shape: {val_data[-1][3].shape}, end_indices shape: {val_data[-1][4].shape}")
        if not config["validate_only"]:
            # Normal Training mode
            vox2_train_data = load_npy_files(args.vox2_audio_root, args.vox2_motion_root, 
                                             args.vox2_train_start_idx, args.vox2_train_end_idx, 
                                             config["motion_latent_type"], config["latent_mask_1"], config["latent_bound"])
            for i, d in enumerate(vox2_train_data):
                train_data[i] = torch.cat([train_data[i], d], dim=0)
            print(f"Vox2 training Data loaded. audio shape: {train_data[0].shape}, motion shape: {train_data[1].shape}, \
                                                shape shape: {train_data[2].shape}, mouth shape: {train_data[3].shape}, end_indices shape: {train_data[4].shape}")

    if "hdtf" in args.dataset:
        train_audio_root = args.hdtf_train_root + "audio_latent/"
        train_motion_root = args.hdtf_train_root + "live_latent/"
        test_audio_root = args.hdtf_test_root + "audio_latent/"
        test_motion_root = args.hdtf_test_root + "live_latent/"
        hdtf_val_data = load_npy_files(test_audio_root, test_motion_root, 0, 8,   
                                        config["motion_latent_type"], config["latent_mask_1"], config["latent_bound"]) # load all validation data
        val_data.append(hdtf_val_data)
        print(f"HDTF validation Data loaded. audio shape: {val_data[-1][0].shape}, motion shape: {val_data[-1][1].shape}, shape shape: {val_data[-1][2].shape}, mouth shape: {val_data[-1][3].shape}, end_indices shape: {val_data[-1][4].shape}")
        if not config["validate_only"]:
            hdtf_train_data = load_npy_files(train_audio_root, train_motion_root, 
                                             args.hdtf_train_start_idx, args.hdtf_train_end_idx, 
                                             config["motion_latent_type"], config["latent_mask_1"], config["latent_bound"])
            print(f"HDTF training Data loaded. audio shape: {hdtf_train_data[0].shape}, motion shape: {hdtf_train_data[1].shape}, \
                                                shape shape: {hdtf_train_data[2].shape}, mouth shape: {hdtf_train_data[3].shape}, end_indices shape: {hdtf_train_data[4].shape}")
            for i, d in enumerate(hdtf_train_data):
                train_data[i] = torch.cat([train_data[i], d], dim=0)
        
    
    torch.multiprocessing.spawn(rank_entrance, args=(args.world_size, args, config, 
                                                     train_data, val_data ), 
                                nprocs=args.world_size)
    