# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
from torch import Tensor
from functools import partial
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from einops import rearrange
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn
from .gated_delta_rule_ops import chunk_gated_delta_rule

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KSCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

class StateSpaceAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int , n_embd: int, head_size = None) -> None:
        super().__init__()
        self.local = layer_idx % config.full_per_layer < config.full_per_layer-1
        if head_size is not None:
            self.head_size = head_size
            self.n_head =  n_embd // head_size
            self.n_query_groups = self.n_head 
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        self.layer_idx = layer_idx
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels= self.kv_dim,
                out_channels= self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups= self.kv_dim,
                padding=d_conv - 1,
            ) 

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        ks_cache: Optional[KSCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KSCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        last_state = ks_cache[-1] if use_cache else None
        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1 )  
        v = v.reshape(B,  T, -1 )  
        if self.sc:
            q = causal_conv1d_fn(
                        x = q.transpose(-1,-2),
                        weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                        bias=self.q_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            k = causal_conv1d_fn(
                        x = k.transpose(-1,-2),
                        weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                        bias=self.k_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            v = causal_conv1d_fn(
                        x = v.transpose(-1,-2),
                        weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                        bias=self.v_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2) 

        q = q.reshape(B,  T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.head_size)  
        v = v.reshape(B,  T, -1, self.head_size)

        if not self.config.nope:         
            cos, sin = rope
            # apply rope in fp32 significanly stabalize training
            # fused rope expect (batch_size, seqlen, nheads, headdim)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        if ks_cache is not None:
            cache_k, _ = ks_cache
            cache_k = cache_k.to(dtype=k.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            
        
        v, recurrent_state = chunk_gated_delta_rule(q, k, v, beta, gk, initial_state=recurrent_state, output_final_state=use_cache)
        ks_cache = k, recurrent_state
        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y, ks_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func
            if self.local and self.config.local_window > -1:
                win_tuple = (self.config.local_window-1, 0)
            else:
                win_tuple = (-1,-1)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True, window_size=win_tuple)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)
