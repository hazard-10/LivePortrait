import torch
from model import TransformerBackbone
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device
# batch pa
batch_size = 2
# Model parameters
x_dim = 78  # Example dimension for motion data
a_dim = 768  # Example dimension for audio data
prev_seq_length = 10
max_seq_length = 1024  # Example max sequence length
hidden_size = 512
num_layers = 8
num_attention_heads = 8

# Instantiate the model
model = TransformerBackbone(
    x_dim=x_dim,
    a_dim=a_dim,
    max_seq_length=max_seq_length,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_attention_heads=num_attention_heads,
    norm_type="ada_norm_zero",
    device=device
).to(device)

# Create dummy input data
x = torch.randn(batch_size, max_seq_length - prev_seq_length, x_dim).to(device)
x_prev = torch.randn(batch_size, prev_seq_length, x_dim).to(device)
a = torch.randn(batch_size, max_seq_length - prev_seq_length, a_dim).to(device)
a_prev = torch.randn(batch_size, prev_seq_length, a_dim).to(device)
t = torch.randint(0, 1000, (batch_size,)).to(device)



for i in range(2):
# Forward pass
# try:
    start_time = time.time()
    with torch.no_grad():
        output = model(x, x_prev, a, a_prev, t)
    print(f"Forward pass successful. total time: {time.time() - start_time}")
    print(f"Forward pass successful. Output shape: {output.shape}")
# except Exception as e:
#     print(f"Error during forward pass: {str(e)}")
