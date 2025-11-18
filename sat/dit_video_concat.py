from functools import partial
import os
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from einops import rearrange, repeat
from sgm.modules.diffusionmodules.openaimodel import Timestep
from sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
from sgm.modules.traj_module import MGF, TrajExtractor
from sgm.util import instantiate_from_config
from torch import nn
from Camera_module import CameraInsert
from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.mpu.layers import ColumnParallelLinear
from sat.ops.layernorm import LayerNorm, RMSNorm
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from sat.helpers import logger, print_all, print_rank0
from sklearn.decomposition import PCA
from scipy.stats import entropy, kurtosis
import pywt
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import imageio
from pathlib import Path

from timm.models.layers import Mlp
from timm.layers import DropPath

import matplotlib.pyplot as plt
import numpy as np
import os

def save_attention_map(attn_vector, save_path, T=13, H=30, W=45, cmap='viridis', step_id=None, layer_id=None):
   
    attn_np = attn_vector.detach().cpu().float().numpy()

   
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

    fig, axes = plt.subplots(1, T, figsize=(T*2, 2))
    for t in range(T):
        axes[t].imshow(attn_np[t], cmap=cmap)
        axes[t].axis('off')
        axes[t].set_title(f"t={t}")
    filename = ""
    if step_id is not None:
        filename += f"{step_id}step_"
    if layer_id is not None:
        filename += f"{layer_id}layer"
    filename += ".jpg"
    filepath = os.path.join(save_path, filename)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close(fig)

def visualize_features(tensor, save_dir, subfolder_name, step_id, layer_id):
    # Convert tensor to numpy array on CPU
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Get tensor dimensions
    B, T, H, W, C = tensor.shape
    batch_tensor = tensor[0]  # Take first sample
    
    # Create output directories
    subfolder_path = os.path.join(save_dir, subfolder_name)
    Path(subfolder_path).mkdir(parents=True, exist_ok=True)
    
    # Generate 40 random (t, channel) pairs
    random_pairs = [(np.random.randint(0, T), np.random.randint(0, C)) 
                   for _ in range(15)]
    
    # Save each frame with unique combination
    for idx, (t, channel) in enumerate(random_pairs):
        # Extract frame data
        frame = batch_tensor[t, :, :, channel]  
        
        # Normalization
        frame_min, frame_max = frame.min(), frame.max()
        if frame_min != frame_max:
            frame = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame = np.zeros_like(frame)
        
        # Visualization and saving
        plt.figure(figsize=(10, 10))
        plt.imshow(frame, cmap='gray', vmin=0, vmax=1, interpolation='none')
        plt.axis('off')
        plt.title(f"t={t}, channel={channel}", fontsize=8)
        plt.savefig(
            os.path.join(subfolder_path, f"{step_id}_{layer_id}_sample{idx:02d}_t{t}_c{channel}.png"),
            bbox_inches='tight',
            pad_inches=0,
            dpi=100
        )
        plt.close()
    

class SimpleTemporalAttention(nn.Module):
    def __init__(self, dim=3072, heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        
    def forward(self, x, B, T, mask=None):
        BT, HW, C = x.shape
       
        x_reshaped = x.view(B, T, HW, C).permute(0, 2, 1, 3).contiguous()
        x_flat = x_reshaped.reshape(B*HW, T, C)
        attended = self.attn(
            self.norm(x_flat),
            x_flat,
            x_flat,
            key_padding_mask=mask
        )[0]
        
        # 恢复形状
        attended = attended.view(B, HW, T, C).permute(0, 2, 1, 3).contiguous()
        return attended.reshape(B, T*HW, C)


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        bias=True,
        text_hidden_size=None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"]  # (b,t,c,h,w)
        B, T = images.shape[:2]
        emb = images.view(-1, *images.shape[2:])
        emb = self.proj(emb)  # ((b t),d,h/2,w/2) 
        emb = emb.view(B, T, *emb.shape[1:])
        emb = emb.flatten(3).transpose(2, 3) 
        emb = rearrange(emb, "b t n d -> b (t n) d")

        images_reference = kwargs["images_reference"]  # (b,t,c,h,w)
        B, T = images_reference.shape[:2]
        emb_reference = images_reference.view(-1, *images_reference.shape[2:]) 
        emb_reference = self.proj(emb_reference) 
        emb_reference = emb_reference.view(B, T, * emb_reference.shape[1:])
        emb_reference = emb_reference.flatten(3).transpose(2, 3)  # (b,t,n,d)
        emb_reference = rearrange(emb_reference, "b t n d -> b (t n) d")

        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs["encoder_outputs"])
            emb = torch.cat((text_emb, emb), dim=1)  # (b,n_t+t*n_i,d) 
            emb_reference = torch.cat((text_emb, emb_reference), dim=1) 

        emb = emb.contiguous()
        emb_reference = emb_reference.contiguous()
        return emb, emb_reference  # (b,n_t+t*n_i,d)

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_height * grid_width, axis=1)  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim]) 

    return pos_embed  # [T, H*W, D]


def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Basic2DPositionEmbeddingMixin(BaseMixin):
    def __init__(self, height, width, compressed_num_frames, hidden_size, text_length=0):
        super().__init__()
        self.height = height
        self.width = width
        self.spatial_length = height * width
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)), requires_grad=False
        )

    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width)
        self.pos_embedding.data[:, -self.spatial_length :].copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


class Basic3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        text_length=0,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.text_length = text_length
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.num_patches = height * width * compressed_num_frames
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)), requires_grad=False
        )
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation

    def position_embedding_forward(self, position_ids, **kwargs):
        if kwargs["images"].shape[1] == 1:
            return self.pos_embedding[:, : self.text_length + self.spatial_length]

        return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class Rotary3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        rot_v=False,
        learnable_pos_embed=False,
    ):
        super().__init__()
        self.rot_v = rot_v

        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        freqs = rearrange(freqs, "t h w d -> (t h w) d")

        freqs = freqs.contiguous()
        freqs_sin = freqs.sin()
        freqs_cos = freqs.cos()
        self.register_buffer("freqs_sin", freqs_sin)
        self.register_buffer("freqs_cos", freqs_cos)

        self.text_length = text_length
        if learnable_pos_embed:
            num_patches = height * width * compressed_num_frames + text_length
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True)
        else:
            self.pos_embedding = None

    def rotary(self, t, **kwargs):
        seq_len = t.shape[2]
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

        return t * freqs_cos + rotate_half(t) * freqs_sin

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        attention_fn_default = HOOKS_DEFAULT["attention_fn"]

        query_layer[:, :, self.text_length :] = self.rotary(query_layer[:, :, self.text_length :])
        key_layer[:, :, self.text_length :] = self.rotary(key_layer[:, :, self.text_length :])
        if self.rot_v:
            value_layer[:, :, self.text_length :] = self.rotary(value_layer[:, :, self.text_length :])

        return attention_fn_default( 
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """
    if rope_position_ids is not None:
        assert NotImplementedError
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum("nlpqc->ncplq", x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        b = x.shape[0]
        imgs = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p)

    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        latent_width,
        latent_height,
        elementwise_affine,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True))

        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width
        self.latent_height = latent_height

    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, kwargs["text_length"] :, :], kwargs["emb"]  # x:(b,(t n),d)
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return unpatchify(
            x,
            c=self.out_channels,
            p=self.patch_size,
            w=self.latent_width // self.patch_size,
            h=self.latent_height // self.patch_size,
            rope_position_ids=kwargs.get("rope_position_ids", None),
            **kwargs,
        )

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()
        self.w2 = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features,
                    hidden_features,
                    gather_output=False,
                    bias=bias,
                    module=self,
                    name="dense_h_to_4h_gate",
                )
                for i in range(num_layers)
            ]
        )

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args["layer_id"]].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args["layer_id"]](x)
        hidden = origin.activation_func(x2) * x1
        x = origin.dense_4h_to_h(hidden)
        return x

class WaveletTransform(nn.Module):
    def __init__(self, levels=1, wavelet='haar'):
        super().__init__()
        self.levels = levels
       
        self.register_buffer('dec_lo_h', torch.tensor([1.0, 1.0]).view(1, 1, -1) / 2**0.5) 
        self.register_buffer('dec_hi_h', torch.tensor([-1.0, 1.0]).view(1, 1, -1) / 2**0.5)
        self.register_buffer('dec_lo_v', torch.tensor([1.0, 1.0]).view(1, 1, 1, 2) / 2**0.5) 
        self.register_buffer('dec_hi_v', torch.tensor([-1.0, 1.0]).view(1, 1, 1, 2) / 2**0.5) 
        
    def _pad_to_even(self, x: torch.Tensor) -> torch.Tensor:
        h_pad = x.size(2) % 2
        w_pad = x.size(3) % 2
        return F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')

    def _dwt_2d(self, x: torch.Tensor) -> tuple:
        lo_row = F.conv2d(x, self.dec_lo_h.unsqueeze(1), stride=(1, 2), padding=(0, 1))
        hi_row = F.conv2d(x, self.dec_hi_h.unsqueeze(1), stride=(1, 2), padding=(0, 1))
        
        
        ll = F.conv2d(lo_row.permute(0, 1, 3, 2).contiguous(), self.dec_lo_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() 
        lh = F.conv2d(lo_row.permute(0, 1, 3, 2).contiguous(), self.dec_hi_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() 
        hl = F.conv2d(hi_row.permute(0, 1, 3, 2).contiguous(), self.dec_lo_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() 
        hh = F.conv2d(hi_row.permute(0, 1, 3, 2).contiguous(), self.dec_hi_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() 
        
        return ll, (lh, hl, hh)
    
    def _idwt_2d(self, coeffs: tuple) -> torch.Tensor:
        lh, hl, hh = coeffs
        lo_row = F.conv_transpose2d(lh.permute(0, 1, 3, 2).contiguous(), self.dec_hi_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() + \
                 F.conv_transpose2d(hl.permute(0, 1, 3, 2).contiguous(), self.dec_lo_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous()
        hi_row = F.conv_transpose2d(hh.permute(0, 1, 3, 2).contiguous(), self.dec_hi_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() + \
                 F.conv_transpose2d(hl.permute(0, 1, 3, 2).contiguous(), self.dec_lo_v, stride=(1, 2), padding=(0, 1)).permute(0, 1, 3, 2).contiguous() 
        
        x = (F.conv_transpose2d(lo_row, self.dec_lo_h.unsqueeze(1), stride=(1, 2), padding=(0, 1)) + \
            F.conv_transpose2d(hi_row, self.dec_hi_h.unsqueeze(1), stride=(1, 2), padding=(0, 1))).contiguous() 
        return x

   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.float()  

            max_val = torch.max(torch.abs(x))
            scale_factor = max_val if max_val > 1.0 else 1.0
            x = x / scale_factor

          
            B, THW, C = x.shape
            T, H, W = 13, 30, 45
            x = x.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
            x = x.reshape(B * C * T, 1, H, W).contiguous()
            x = self._pad_to_even(x)

            high_freq = torch.zeros_like(x)
            ll = x
            for _ in range(self.levels):
                ll, (lh, hl, hh) = self._dwt_2d(ll)
                high_freq += self._idwt_2d((lh, hl, hh))

            high_freq = high_freq[:, :, :H, :W]
            high_freq = high_freq.view(B, C, T, H, W).permute(0, 2, 3, 4, 1).contiguous()
            result = high_freq.reshape(B, THW, C)
            return result * scale_factor
    
class Modulation(nn.Module):
    def __init__(self, in_channels, out_channels, height, width):
        super(Modulation, self).__init__()
        self.height = height
        self.width = width
        
      
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
       

        

    def forward(self, Zraw, Zcondition):
        B, HW, C = Zraw.shape
        assert Zcondition.shape == Zraw.shape, "Error: Zraw and Zcondition must have the same shape"
        
        T = HW // (self.height * self.width)
        print(f"Zraw shape is {Zraw.shape},Zcondition shape is {Zcondition.shape}")
        
        Zconcat = torch.cat((Zraw, Zcondition), dim=-1)  # [B, T*H*W, 2*C]
        print(f"Zconcat shape is {Zconcat.shape}")
       
        Zconcat = Zconcat.view(B, T, self.height, self.width, 2*C)
        print(f"Zconcat shape is {Zconcat.shape}")
        Zconcat = Zconcat.permute(0, 4, 1, 2, 3).contiguous()

        alpha = self.conv(Zconcat.view(B * T, 2*C, self.height, self.width).contiguous())  # [B*T, C, H, W]
        alpha = self.sigmoid(alpha)  # [B*T, C, H, W]
   
        alpha = alpha.view(B, T, self.height, self.width, C).contiguous() 
        alpha = alpha.permute(0, 2, 3, 1, 4).contiguous().reshape(B, HW, C)

        zblend = alpha * Zraw  + (1 - alpha) * Zcondition 

        return zblend

class High_Freq_CrossAttn(nn.Module): 
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 10,
        dim_head: int = 256,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        
        self.norm_hidden_states = nn.LayerNorm(query_dim)
        self.norm_encoder_hidden_states = nn.LayerNorm(cross_attention_dim)
        
        self.dropout = dropout
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        nn.init.xavier_uniform_(self.to_q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.to_k.weight, gain=0.1)
        nn.init.xavier_uniform_(self.to_v.weight, gain=0.1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
     
        dtype = hidden_states.dtype
    
        hidden_states = self.norm_hidden_states(hidden_states)
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states) if encoder_hidden_states is not None else None

        hidden_states = hidden_states.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Project to query/key/value
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states) #[]

        # Reshape for multi-head attention
        B, T, _ = q.shape
        q = q.view(B, T, self.heads, self.dim_head).contiguous().transpose(1, 2)  # [B, heads, T, dim_head]
        k = k.view(B, -1, self.heads, self.dim_head).contiguous().transpose(1, 2)
        v = v.view(B, -1, self.heads, self.dim_head).contiguous().transpose(1, 2)

        # Flash Attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )
       
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.to_out(attn_output)

class AdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
        
    ):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.camera_insert = nn.ModuleList([CameraInsert(hidden_dim=3072, RT_dim=32) for _ in range(num_layers)]).to(self.device)
        self.High_Freq_CrossAttn = nn.ModuleList([High_Freq_CrossAttn( query_dim=3072,
            cross_attention_dim=3072,
            heads= 10,
            dim_head=256,
            dropout=0.1) for _ in range(num_layers)]).to(self.device)
        
        self.WaveletTransform = WaveletTransform(levels=1, wavelet='haar').to(self.device)
        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
        self.temporal_selfattn = SimpleTemporalAttention()

    def layer_forward(
        self,
        hidden_states_input,
        hidden_states_reference,
        mask,
        *args, #None
        **kwargs,
    ):
       
            step_id = kwargs["step_id"]
            hidden_states = hidden_states_input
            hidden_states_reference = hidden_states_reference
            text_length = kwargs["text_length"] 
           
            layer = self.transformer.layers[kwargs["layer_id"]]
            adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]
            H = 30
            W = 45
            T = 13

            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                text_shift_msa,
                text_scale_msa,
                text_gate_msa,
                text_shift_mlp,
                text_scale_mlp,
                text_gate_mlp,
            ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
            gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
                gate_msa.unsqueeze(1),
                gate_mlp.unsqueeze(1),
                text_gate_msa.unsqueeze(1),
                text_gate_mlp.unsqueeze(1),
            )
            
            ############################ReferenceNet Start######################
           
            text_hidden_states_reference = hidden_states_reference[:, :text_length]  # (b,n,d)
            Batch =  text_hidden_states_reference.shape[0]
            img_hidden_states_reference = hidden_states_reference[:, text_length:]  # (b,(t n),d)
           
        
            # self full attention (b,(t n),d)
            img_attention_input_reference = layer.input_layernorm(img_hidden_states_reference)
            text_attention_input_reference = layer.input_layernorm(text_hidden_states_reference)
            img_attention_input_reference = modulate(img_attention_input_reference, shift_msa, scale_msa) #[1, 17550, 3072]
            text_attention_input_reference = modulate(text_attention_input_reference, text_shift_msa, text_scale_msa) #[1,226,3072]
            C = text_attention_input_reference.shape[-1]  # C：3072
            img_hidden_states_reference_guidance = img_attention_input_reference

            attention_input_reference = torch.cat((text_attention_input_reference, img_attention_input_reference), dim=1)  # (b,n_t+t*n_i,d) [1,17776,3072]
            save_cfg={"text_length": 226, "token_idx": 6, "avg_heads": True}  
            save_cfg = None
            kwargs["save_cfg"] = save_cfg
            
            vis_attn , attention_output_reference = layer.attention(attention_input_reference, mask, **kwargs) 
            
           
           
            
            text_attention_output_reference = attention_output_reference[:, :text_length]  # (b,n,d) [B,226,3072]
            img_attention_output_reference = attention_output_reference[:, text_length:]  # (b,(t n),d) []
           
        
          

            if self.transformer.layernorm_order == "sandwich":
                text_attention_output_reference = layer.third_layernorm(text_attention_output_reference)
                img_attention_output_reference = layer.third_layernorm(img_attention_output_reference)
            
            
            img_hidden_states_reference = img_hidden_states_reference + gate_msa * img_attention_output_reference  # (b,(t n),d)
            text_hidden_states_reference = text_hidden_states_reference + text_gate_msa * text_attention_output_reference  # (b,n,d)

            # mlp (b,(t n),d)
            img_mlp_input_reference = layer.post_attention_layernorm(img_hidden_states_reference)  # vision (b,(t n),d)
            text_mlp_input_reference = layer.post_attention_layernorm(text_hidden_states_reference)  # language (b,n,d)
            img_mlp_input_reference = modulate(img_mlp_input_reference, shift_mlp, scale_mlp)
            text_mlp_input_reference = modulate(text_mlp_input_reference, text_shift_mlp, text_scale_mlp)
            mlp_input_reference = torch.cat((text_mlp_input_reference, img_mlp_input_reference), dim=1)  # (b,(n_t+t*n_i),d
            mlp_output_reference = layer.mlp(mlp_input_reference, **kwargs)
            img_mlp_output_reference = mlp_output_reference[:, text_length:]  # vision (b,(t n),d)
            text_mlp_output_reference = mlp_output_reference[:, :text_length]  # language (b,n,d)
            if self.transformer.layernorm_order == "sandwich":
                text_mlp_output_reference = layer.fourth_layernorm(text_mlp_output_reference)
                img_mlp_output_reference = layer.fourth_layernorm(img_mlp_output_reference)

            img_hidden_states_reference = img_hidden_states_reference + gate_mlp * img_mlp_output_reference  # vision (b,(t n),d)
            text_hidden_states_reference = text_hidden_states_reference + text_gate_mlp * text_mlp_output_reference  # language (b,n,d)
            
            
          
            hidden_states_reference = torch.cat((text_hidden_states_reference, img_hidden_states_reference), dim=1)  # (b,(n_t+t*n_i),d)
            
            ############################ReferenceNet END######################
        
           
            text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
            img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)
          
            assert not torch.isnan(img_hidden_states).any(), "NaN 11"
        
            # self full attention (b,(t n),d)
            img_attention_input = layer.input_layernorm(img_hidden_states)
            text_attention_input = layer.input_layernorm(text_hidden_states)
            img_attention_input = modulate(img_attention_input, shift_msa, scale_msa) 
            text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa) 
            RT = kwargs["camera_traj"] 
           
            
           
            C = img_attention_input.shape[-1]  
            if RT is not None:
                device = img_attention_input.device
                RT = RT.to(device)
                camera_insert = self.camera_insert[kwargs["layer_id"]]
                img_attention_input = rearrange(img_attention_input, "B (T H W) C -> (B T) (H W) C", H=H, W=W).contiguous() #[13,1350,3072]
                RT =  rearrange(RT, "B C T H W -> (B T) (H W) C").contiguous() 
              
                assert not torch.isnan(img_attention_input).any(), "NaN before camera_insert"
                img_attention_input = camera_insert(img_attention_input, RT, video_length = 13) 
                assert not torch.isnan(img_attention_input).any(), "NaN after camera_insert"
                img_attention_input = rearrange(img_attention_input, "(B T) (H W) C ->  B (T H W) C", T=T, H=H,W=W).contiguous()
              
                
            ############################ Modulation Fusion Start ######################

            with  torch.amp.autocast('cuda'):
                reference_high_freq = self.WaveletTransform(img_hidden_states_reference_guidance)
            
           
            assert not torch.isnan(reference_high_freq).any(), "NaN after WaveletTransform"
           
            reference_high_freq = rearrange(reference_high_freq, "B (T H W) C -> (B T) (H W) C", H=H, W=W).contiguous()
            img_hidden_states_rt = rearrange(img_attention_input, "B (T H W) C -> (B T) (H W) C", H=H, W=W).contiguous()
          

            reference_high_freq = reference_high_freq.to(img_hidden_states_reference_guidance.dtype)
            img_hidden_states_rt = img_hidden_states_rt.to(img_hidden_states_reference_guidance.dtype)
            High_Freq_CrossAttn =self.High_Freq_CrossAttn[kwargs["layer_id"]]
            img_hidden_states_fusion = High_Freq_CrossAttn(img_hidden_states_rt, reference_high_freq) + img_hidden_states_rt 
          
            img_hidden_states_fusion = self.temporal_selfattn(img_hidden_states_fusion, Batch, T, None) 
         
        ############################ Modulation Fusion End ######################
           
         
            img_attention_input=img_attention_input.to(text_attention_input.dtype)
            #attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
            attention_input = torch.cat((text_attention_input, img_hidden_states_fusion), dim=1)  # (b,n_t+t*n_i,d)
        
            _ , attention_output = layer.attention(attention_input, mask, **kwargs) 
            text_attention_output = attention_output[:, :text_length]  # (b,n,d)
            img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)
         

            ###############################reference latents###############################
          
               
            ############################### MSF END ###############################
            img_attention_output = img_attention_output.to(text_attention_input.dtype)
           
            if self.transformer.layernorm_order == "sandwich":
                text_attention_output = layer.third_layernorm(text_attention_output)
                img_attention_output = layer.third_layernorm(img_attention_output)
            img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
            text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)

            # mlp (b,(t n),d)
            img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
            text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
            img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
            text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
            mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
            mlp_output = layer.mlp(mlp_input, **kwargs)
            img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
            text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)


            if self.transformer.layernorm_order == "sandwich":
                text_mlp_output = layer.fourth_layernorm(text_mlp_output)
                img_mlp_output = layer.fourth_layernorm(img_mlp_output)

            img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
            text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)

            hidden_states = torch.cat((text_hidden_states, text_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)
        
        
            return hidden_states,hidden_states_reference

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        return old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class LightweightResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=2):
        super().__init__()
        hidden_ch = in_ch * expansion
        
        self.conv = nn.Sequential(
           
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.GroupNorm(4, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=hidden_ch),
            nn.GroupNorm(4, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.GroupNorm(4, out_ch),
        )
        
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x) + self.shortcut(x))

class EfficientTemporalConv(nn.Module):
  
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, 
                     padding=kernel_size//2, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool1d(kernel_size, stride=stride, padding=kernel_size//2),
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
        ) if in_ch != out_ch or stride > 1 else nn.Identity()
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, out_ch // 4, 1),
            nn.ReLU(),
            nn.Conv1d(out_ch // 4, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, T]
        x = x.reshape(B * H * W, C, T)
        x_main = self.conv(x)
        x_shortcut = self.shortcut(x)
        
       
        attention = self.temporal_attention(x_main)
        x_main = x_main * attention
        

        x_out = x_main + x_shortcut
        
        T_new = x_out.shape[-1]
        x_out = x_out.view(B, H, W, -1, T_new).permute(0, 4, 3, 1, 2)
        return x_out

class LightweightCameraEncoder(nn.Module):
    def __init__(self, in_channels=6, out_channels=32, downscale_coef=2):
        super().__init__()
        self.initial_unshuffle = nn.PixelUnshuffle(downscale_coef)
        in_ch = in_channels * (downscale_coef ** 2) 

       
        self.encoder = nn.Sequential(
          
            nn.Conv2d(in_ch, 48, kernel_size=3, stride=2, padding=1), 
            LightweightResBlock(48, 48),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),   
            LightweightResBlock(96, 96),

            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  
            LightweightResBlock(128, 128),
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

        
        self.time_compress = nn.Sequential(
            EfficientTemporalConv(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            EfficientTemporalConv(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )
        
        self.final_proj = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, num_frames):
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.initial_unshuffle(x)
        x = self.encoder(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        x = self.time_compress(x)
        x = x.permute(0, 2, 1, 3, 4) 
        return x


class DiffusionTransformer(BaseModel):
    def __init__(
        self,
        transformer_args,
        num_frames,
        time_compressed_rate,
        latent_width,
        latent_height,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_layers,
        num_attention_heads,
        elementwise_affine,
        time_embed_dim=None,
        num_classes=None,
        modules={},
        input_time="adaln",
        adm_in_channels=None,
        parallel_output=True,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
        use_SwiGLU=False,
        use_RMSNorm=False,
        zero_init_y_embed=False,
        **kwargs,
    ):
        
        
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.time_compressed_rate = time_compressed_rate
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.elementwise_affine = elementwise_affine
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation
        self.inner_hidden_size = hidden_size * 4
        self.zero_init_y_embed = zero_init_y_embed
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            self.dtype = torch.float32

        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu

        if use_RMSNorm:
            kwargs["layernorm"] = RMSNorm
        else:
            kwargs["layernorm"] = partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6)

        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        super().__init__(args=transformer_args, transformer=None, **kwargs)
        self.camera_encoder = LightweightCameraEncoder()
    
       
        module_configs = modules
        self._build_modules(module_configs)

        if use_SwiGLU:
            self.add_mixin(
                "swiglu", SwiGLUMixin(num_layers, hidden_size, self.inner_hidden_size, bias=False), reinit=True
            )
        

    def _build_modules(self, module_configs):
        model_channels = self.hidden_size
      
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert self.adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(self.adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
                if self.zero_init_y_embed:
                    nn.init.constant_(self.label_emb[0][2].weight, 0)
                    nn.init.constant_(self.label_emb[0][2].bias, 0)
            else:
                raise ValueError()

        pos_embed_config = module_configs["pos_embed_config"]
        self.add_mixin(
            "pos_embed",
            instantiate_from_config(
                pos_embed_config,
                height=self.latent_height // self.patch_size,
                width=self.latent_width // self.patch_size,
                compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
                hidden_size=self.hidden_size,
            ),
            reinit=True,
        )

        patch_embed_config = module_configs["patch_embed_config"]
        self.add_mixin(
            "patch_embed",
            instantiate_from_config(
                patch_embed_config,
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                in_channels=self.in_channels,
            ),
            reinit=True,
        )
        if self.input_time == "adaln":
            adaln_layer_config = module_configs["adaln_layer_config"]
            self.add_mixin(
                "adaln_layer",
                instantiate_from_config(
                    adaln_layer_config,
                    height=self.latent_height // self.patch_size,
                    width=self.latent_width // self.patch_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
                    hidden_size_head=self.hidden_size // self.num_attention_heads,
                    time_embed_dim=self.time_embed_dim,
                    elementwise_affine=self.elementwise_affine,
                ),
            )
        else:
            raise NotImplementedError

        final_layer_config = module_configs["final_layer_config"]
        self.add_mixin(
            "final_layer",
            instantiate_from_config(
                final_layer_config,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                time_embed_dim=self.time_embed_dim,
                latent_width=self.latent_width,
                latent_height=self.latent_height,
                elementwise_affine=self.elementwise_affine,
            ),
            reinit=True,
        )

        if "lora_config" in module_configs:
            lora_config = module_configs["lora_config"]
            self.add_mixin("lora", instantiate_from_config(lora_config, layer_num=self.num_layers), reinit=True)

        return

    def forward(self, x, x_reference, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape 
        
        


        # traj extractor
        video_flow = kwargs.get("video_flow", None)
        
        kwargs["video_flow_features"] = None

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if x_reference.dtype != self.dtype:
            x_reference = x_reference.to(self.dtype)
        RT = kwargs["camera_traj"].to(self.dtype)
        self.camera_encoder = self.camera_encoder.to(self.dtype)
       
        RT = RT.to(torch.bfloat16)
        RT = self.camera_encoder(RT,49).to(torch.bfloat16)  #[B, 32, 13, 30, 45]
      
        kwargs["camera_traj"] = RT  
        # This is not use in inference
        if "concat_images" in kwargs and kwargs["concat_images"] is not None:
            if kwargs["concat_images"].shape[0] != x.shape[0]:
                concat_images = kwargs["concat_images"].repeat(2, 1, 1, 1, 1)
            else:
                concat_images = kwargs["concat_images"]
            x = torch.cat([x, concat_images], dim=2)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        kwargs["seq_length"] = t * h * w // (self.patch_size**2)
        kwargs["images"] = x  
        kwargs['images_reference'] = x_reference
        kwargs["emb"] = emb   
        kwargs["encoder_outputs"] = context
        kwargs["text_length"] = context.shape[1] 

        kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
        
        tem = super().forward(**kwargs)
        output = tem[0][0]
        output_references = tem[1][0]
        return output, output_references
