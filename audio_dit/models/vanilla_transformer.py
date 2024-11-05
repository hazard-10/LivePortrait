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

class VanillaTransformer(nn.Module):
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
        # Mask
        motion_len = max_seq_length - 1
        alignment_mask = enc_dec_mask(motion_len, motion_len)
        alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
        self.register_buffer('alignment_mask', alignment_mask)

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
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_train, x_prev, a_train, a_prev, shape_feat):
        B, T_prev, D = x_prev.shape
        x_prev = self.prev_dropout(x_prev)
        a_prev = self.prev_dropout(a_prev)
        x_train = self.train_dropout(x_train)
        a_train = self.train_dropout(a_train)
        # Concatenate current and previous inputs
        x_full = torch.cat([x_prev, x_train], dim=1)
        a_full = torch.cat([a_prev, a_train], dim=1)

        # Embed inputs
        x_emb = self.x_embedder(x_full)
        a_emb = self.a_embedder(a_full)
        shape_emb = self.shape_embedder(shape_feat).unsqueeze(1) # from B,S to B,1,S
        input_full = torch.cat([shape_emb, x_emb], dim=1) # B, 1 + T, D
        # Apply positional encoding
        input_emb = self.pos_encoder(input_full)
        # a_emb = self.pos_encoder(a_emb.transpose(0, 1)).transpose(0, 1) Audio doesn't need positional encoding

        # memory_mask = enc_dec_mask(x_full.device, T, S)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=input_emb, 
            memory=a_emb, 
            # tgt_mask=tgt_mask,
            memory_mask=self.alignment_mask
        )

        # Final layer
        output = self.final_layer(output)

        # Return only the newly generated part (excluding x_prev)
        return output[:, T_prev + 1:]
    
    def inference(self, x_inference, x_prev, a, a_prev, shape_feat, init_method='zero'):
        device = x_prev.device
        B, T_prev, D = x_prev.shape
        _, S, _ = a.shape

        # Concatenate a_prev and a
        a_full = torch.cat([a_prev, a], dim=1)
        
        # Embed and apply positional encoding to a_full
        a_emb = self.a_embedder(a_full)
        # a_emb = self.pos_encoder(a_emb.transpose(0, 1)).transpose(0, 1) Audio doesn't need positional encoding

        # Initialize x based on the specified method
        if init_method == 'zero':
            x_inference = torch.zeros_like(x_inference)
        elif init_method == 'random':
            x_inference = torch.randn_like(x_inference)
        else:
            raise ValueError("init_method must be 'zero' or 'random'")
        
        # Concatenate x_prev and x
        x_full = torch.cat([x_prev, x_inference], dim=1)

        # Embed and apply positional encoding to x_full
        x_emb = self.x_embedder(x_full)
        a_emb = self.a_embedder(a_full)
        shape_emb = self.shape_embedder(shape_feat).unsqueeze(1) # from B,S to B,1,S
        input_full = torch.cat([shape_emb, x_emb], dim=1) # B, 1 + T, D
        # Apply positional encoding
        input_emb = self.pos_encoder(input_full)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=input_emb,
            memory=a_emb,
            # tgt_mask=tgt_mask,
            memory_mask=self.alignment_mask
        )

        # Final layer
        output = self.final_layer(output)

        # Return only the newly generated part (excluding x_prev)
        return output[:, T_prev + 1:]