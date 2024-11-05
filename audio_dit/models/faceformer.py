import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def enc_dec_mask(T, S, frame_width=1, expansion=0, device='cuda'):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)

class FaceFormer(nn.Module):
    def __init__(
        self,
        x_dim,
        a_dim,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        train_dropout=0.1,
        prev_dropout=0.5,
        max_seq_length=76 # 10 prev + 65 current + 1 for shape
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        
        # Embedders
        self.x_embedder = nn.Linear(x_dim, hidden_size)
        self.a_embedder = nn.Linear(a_dim, hidden_size)
        self.shape_embedder = nn.Linear(63, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.1)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, activation='gelu', 
                                                   dim_feedforward=hidden_size * 4, dropout=train_dropout, 
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer
        self.final_layer =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.x_dim)
        )
        
        # dropout
        self.prev_dropout = nn.Dropout(prev_dropout)
        self.train_dropout = nn.Dropout(train_dropout)

    def forward(self, x_train, x_prev, a_train, a_prev, shape_feat, teacher_forcing = False):
        # Concatenate current and previous inputs
        x_full = torch.cat([x_prev, x_train], dim=1)
        a_full = torch.cat([a_prev, a_train], dim=1)

        # Embed inputs
        # x_emb = self.x_embedder(x_full)
        a_emb = self.a_embedder(a_full)
        shape_emb = self.shape_embedder(shape_feat).unsqueeze(1) # from B,S to B,1,S
        
        B, T, D = x_full.shape
        if teacher_forcing:
            x_input = x_full[:, :-1, :] # shift by 1
            x_emb = self.x_embedder(x_input)
            input_full = torch.cat([shape_emb, x_emb], dim=1) # B, 1 + T - 1, D
            # Apply positional encoding
            input_emb = self.pos_encoder(input_full)
            # get mask
            memory_mask = enc_dec_mask(input_full.shape[1] - 1, a_full.shape[1], device=x_full.device)
            memory_mask = F.pad(memory_mask, (0, 0, 1, 0), value=False)
            # Transformer decoder
            out = self.transformer_decoder(
                tgt=input_emb, 
                memory=a_emb,
                memory_mask=memory_mask
            )
            x_out = self.final_layer(out)
        else:
            for i in range(T):
                if i == 0:
                    input_full = shape_emb
                    input_emb = self.pos_encoder(input_full)
                else:
                    input_emb = self.pos_encoder(input_full)                    
                # get mask
                memory_mask = enc_dec_mask(input_full.shape[1] - 1, a_full.shape[1], device=x_full.device)
                memory_mask = F.pad(memory_mask, (0, 0, 1, 0), value=False)
                out = self.transformer_decoder(
                    tgt=input_emb, 
                    memory=a_emb,
                    memory_mask=memory_mask
                )
                x_out = self.final_layer(out)
                x_out_last = x_out[:, -1, :].unsqueeze(1)
                x_out_last_emb = self.x_embedder(x_out_last)
                input_full = torch.cat([input_full, x_out_last_emb], dim=1)
        return x_out
