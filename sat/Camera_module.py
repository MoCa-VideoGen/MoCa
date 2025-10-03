
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.nn.attention import sdpa_kernel
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision
from torch.cuda.amp import autocast
from torch import amp

from einops import rearrange, repeat
import math

####################################################Plucker ####################################################


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False): Whether to add bias to linear layers.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,  
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.heads = heads
        self.dropout = dropout
        self.scale = dim_head**-0.5  # Scaling factor for attention scores
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        Forward pass for cross-attention.
        """

        batch_size, seq_len, _ = hidden_states.shape 
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
     

        # Compute query, key, value projections
        query = self.to_q(hidden_states) 
        key = self.to_k(encoder_hidden_states) 
        value = self.to_v(encoder_hidden_states) 
        query = query.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3).contiguous()
        key = key.view(batch_size, -1, self.heads, query.size(-1)).permute(0, 2, 1, 3).contiguous()
        value = value.view(batch_size, -1, self.heads, query.size(-1)).permute(0, 2, 1, 3).contiguous()
        
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            attention_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )

        # Combine heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.to_out(attention_output)
        

class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        

    def forward(self, x):   
        x = x.permute(0, 2, 1).contiguous()  
        x = self.conv(x) 
        x = x.permute(0, 2, 1).contiguous() 
        return x



class CameraInsert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        RT_dim: int,   
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.cross_attention = CrossAttention(
            query_dim=hidden_dim,
            cross_attention_dim=RT_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.norm_hidden_states = nn.LayerNorm(hidden_dim)
       
        
        for name, param in self.named_parameters():
            param.register_hook(lambda grad, name=name: self._gradient_hook(grad, name))
        
    def _gradient_hook(self, grad, name):
        #print(f"Computing gradient for {name}")
        return grad
    def forward(self, hidden_states, RT, video_length):
       
        with amp.autocast("cuda", enabled=False):
            assert not torch.isnan(hidden_states).any(), "NaN before LayerNorm"
            norm_hidden_states = self.norm_hidden_states(hidden_states)
            assert not torch.isnan(norm_hidden_states).any(), "NaN after LayerNorm"
            hidden_states = self.cross_attention(
                hidden_states=norm_hidden_states, 
                encoder_hidden_states=RT,         
            ) + hidden_states 
           
            return hidden_states  


