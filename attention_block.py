import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
import numpy as np
import random
from rotary_embedding_torch import RotaryEmbedding
from torchvision.transforms import Resize
from mamba import Mamba, MambaConfig

class Block(nn.Module):
    def __init__(self, embed_dim, max_len, attn_type):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.attn_type = attn_type
        self.linear_q = nn.Linear(embed_dim, embed_dim)  # Linear layer for queries
        self.linear_k = nn.Linear(embed_dim, embed_dim)  # Linear layer for keys
        self.linear_v = nn.Linear(embed_dim, embed_dim)  # Linear layer for values
        self.register_buffer('mask', torch.tril(torch.ones(int(max_len), int(max_len))))
    def forward(self, x, original_img=None):
        q = self.linear_q(x)
        if self.attn_type == 'cross_attn':
            k = self.linear_k(original_img)
            v = self.linear_v(original_img)
        elif self.attn_type == 'self_attn':
            k = self.linear_k(x)
            v = self.linear_v(x)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        return y

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class BaseNet(nn.Module):
    def __init__(self, embed_dim, max_len, 
                 layer_type, in_channels,
                 pe_type=None, **kwargs):
        super(BaseNet, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pe_type = pe_type
        self.pe, self.rotary_emb = None, None
        self.layer_type = layer_type
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_context = nn.Conv2d(3, embed_dim, kernel_size=1, stride=1, padding=0)

        if  self.layer_type == 'cross_attn' or  self.layer_type == 'self_attn':
            if self.pe_type == 'learnedPE':
                self.pe = nn.Embedding(int(max_len), embed_dim)
            elif self.pe_type == 'RoPE':
                self.rotary_emb = RotaryEmbedding(dim=embed_dim)
            else:
                assert self.pe_type == 'NoPE'

        self.layer = None

        if layer_type == 'cross_attn' or layer_type == 'self_attn':
            self.layer = Block(embed_dim, max_len, layer_type)

        elif layer_type == 'mamba':
            config = MambaConfig(d_model=embed_dim, n_layers=1)
            self.layer = Mamba(config)

        self.ln = LayerNorm(embed_dim, True)

        print(f"1 {layer_type} block")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: {pe_type}")
        print(f"Context length (# of pixels): {max_len}")
    

    def resize_orig_image(self, x, original_img):
        assert(x.shape[2] == x.shape[3])
        new_height, new_width = x.shape[2], x.shape[3]
        reshaped_img = Resize((new_height, new_width))(original_img)
        return reshaped_img

    def forward(self, x, original_img):
        spatial_len = x.shape[2]
        context = self.resize_orig_image(x, original_img)
        x = self.proj_in(x)
        x = x.flatten(start_dim=2, end_dim=3)
        x = x.permute((0, 2, 1))

        encoded_img = self.proj_context(context)
        encoded_img = encoded_img.flatten(start_dim=2, end_dim=3)
        encoded_img = encoded_img.permute((0, 2, 1))

        b, t, _ = x.size()

        if self.pe_type == 'learnedPE':
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)
            pe_emb = self.pe(pos)
            x = x + pe_emb
            encoded_img = encoded_img + pe_emb
        
        elif self.pe_type == 'RoPE':
            x = self.rotary_emb.rotate_queries_or_keys(x)
            encoded_img = self.rotary_emb.rotate_queries_or_keys(encoded_img)
        
        if self.layer_type == 'cross_attn' or self.layer_type == 'self_attn':
            x = self.ln(self.layer(x, encoded_img)) + x
        
        elif self.layer_type == 'mamba':
            x = self.ln(self.layer(x)) + x
        

        x = x.permute((0, 2, 1)).reshape((b, self.embed_dim, spatial_len, spatial_len))
        x = self.proj_out(x)
        return x


    