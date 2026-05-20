from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from math import floor, ceil, exp, log
import string
import nltk
import fnmatch
from nltk.corpus import stopwords
import math
from ptp.utils_img_processing import *
import os
from torchvision.transforms import functional

class FeatandAttnProcessor:
    def __init__(self, controller, place_in_unet, before_softmax, mode="sdxl", attn_uncond=False):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.before_softmax = before_softmax
        self.attn_uncond = attn_uncond
        
        if mode == "sdxl":
            self.call_impl = self._call_sdxl
        elif mode == "sd3":
            self.call_impl = self._call_sd3
        elif mode == "flux":
            self.call_impl = self._call_flux
        elif mode == "pixart":
            self.call_impl = self._call_pixart
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        return self.call_impl(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
    
    def _call_flux(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # 1. Image (Sample) Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        print(query.shape, key.shape, value.shape)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 2. Text (Context) Projections - Only exists in DoubleStreamBlocks
        is_cross = encoder_hidden_states is not None
        if is_cross:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate text and image tokens (Joint Attention)
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # 3. Apply RoPE (Rotary Positional Embeddings)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Controller Integration (Replaced SDXL logic)
        # Note: In FLUX, 'query' and 'key' already include context tokens if is_cross=True
        scale = 1 / math.sqrt(query.shape[-1])
        if not self.before_softmax:
            # Calculate attention scores
            # shape: [batch, heads, seq_len, seq_len]
            attention_probs = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
            attention_probs = attention_probs.softmax(dim=-1)

            # Controller call (mirrors your SDXL logic)
            attention_probs, value, query, key, recomp = self.controller(
                attention_probs, value, query, key, is_cross, self.place_in_unet, self.attn_uncond
            )
            
            if recomp: # If controller modified Q or K, recalculate
                attention_probs = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
                attention_probs = attention_probs.softmax(dim=-1)
        else:
            # Manual scores logic
            attention_scores = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
            attention_scores, value, query, key, recomp = self.controller(
                attention_scores, value, query, key, is_cross, self.place_in_unet, self.attn_uncond
            )
            if recomp:
                attention_scores = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
            attention_probs = attention_scores.softmax(dim=-1)

        # 5. Output Computation
        hidden_states = torch.einsum("b h i j, b h j d -> b h i d", attention_probs, value)
        
        # Reshape back from heads
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Split and Project
        if is_cross:
            # Split text (context) and image (sample) tokens
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # Image output projection
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            # Text output projection
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            # SingleStream path
            return hidden_states
        
    def _call_sd3(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # 1. Image (Sample) Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        print(query.shape, key.shape, value.shape)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 2. Text (Context) Projections - Only exists in DoubleStreamBlocks
        is_cross = encoder_hidden_states is not None
        if is_cross:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate text and image tokens (Joint Attention)
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # 3. Apply RoPE (Rotary Positional Embeddings)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Controller Integration (Replaced SDXL logic)
        # Note: In FLUX, 'query' and 'key' already include context tokens if is_cross=True
        scale = 1 / math.sqrt(query.shape[-1])
        if not self.before_softmax:
            # Calculate attention scores
            # shape: [batch, heads, seq_len, seq_len]
            attention_probs = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
            attention_probs = attention_probs.softmax(dim=-1)

            # Controller call (mirrors your SDXL logic)
            attention_probs, value, query, key, recomp = self.controller(
                attention_probs, value, query, key, is_cross, self.place_in_unet, self.attn_uncond
            )
            
            if recomp: # If controller modified Q or K, recalculate
                attention_probs = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
                attention_probs = attention_probs.softmax(dim=-1)
        else:
            # Manual scores logic
            attention_scores = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
            attention_scores, value, query, key, recomp = self.controller(
                attention_scores, value, query, key, is_cross, self.place_in_unet, self.attn_uncond
            )
            if recomp:
                attention_scores = torch.einsum("b h i d, b h j d -> b h i j", query, key) * scale
            attention_probs = attention_scores.softmax(dim=-1)

        # 5. Output Computation
        hidden_states = torch.einsum("b h i j, b h j d -> b h i d", attention_probs, value)
        
        # Reshape back from heads
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Split and Project
        if is_cross:
            # Split text (context) and image (sample) tokens
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # Image output projection
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            # Text output projection
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            # SingleStream path
            return hidden_states
    
    def _call_pixart(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Standard query projection
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # HEADS CONVERSION
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # --- SLICE MASK TO MATCH KEY DIMENSION ---
        if attention_mask is not None:
            # attention_mask shape is [batch*heads, 1, total_seq_len]
            # query tokens = sequence_length (usually 4096)
            # key tokens = key.shape[1] (4096 for self-attn, 300 for cross-attn)
            if attention_mask.shape[-1] != key.shape[1]:
                if is_cross:
                    # For cross-attn, take the text part of the mask (usually the end)
                    # In PixArt, text tokens are usually at the end of the mask
                    attention_mask = attention_mask[:, :, -key.shape[1]:]
                else:
                    # For self-attn, take the image part (the beginning)
                    attention_mask = attention_mask[:, :, :key.shape[1]]
            # Ensure batch size matches [batch*heads]
            # target_batch = 64, mask_batch = 4
            if attention_mask.shape[0] < query.shape[0]:
                num_heads = query.shape[0] // attention_mask.shape[0]
                # Repeat the mask for each head
                attention_mask = attention_mask.repeat_interleave(num_heads, dim=0)
        
        # print("ooooo----",query.shape,key.shape,value.shape,is_cross,self.place_in_unet)
        if not self.before_softmax:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # Controller integration
            attention_probs, value, query, key, recomp = self.controller(
                attention_probs, value, query, key, is_cross, self.place_in_unet, self.attn_uncond
            )
            if recomp:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
        else:
            # Use your custom get_attention_scores which now receives the sliced mask
            attention_probs, attention_scores = self.get_attention_scores(attn, query, key, attention_mask)
            attention_scores, value, query, key, recomp = self.controller(
                attention_scores, value, query, key, is_cross, self.place_in_unet, self.attn_uncond
            )
            if recomp:
                attention_probs, attention_scores = self.get_attention_scores(attn, query, key, attention_mask)
            attention_probs = attention_scores.softmax(dim=-1)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear and Dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    
    def _call_sdxl(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # print(query.shape,key.shape,value.shape,is_cross,self.place_in_unet)
        if not self.before_softmax:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # one line change
            attention_probs, value, query, key, recomp = self.controller(attention_probs, value, query, key, is_cross, self.place_in_unet, self.attn_uncond)
            if recomp:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
        else:
            attention_probs, attention_scores = self.get_attention_scores(attn, query, key, attention_mask)
            attention_scores, value, query, key, recomp = self.controller(attention_scores, value, query, key, is_cross, self.place_in_unet, self.attn_uncond)
            if recomp:
                attention_probs, attention_scores = self.get_attention_scores(attn, query, key, attention_mask)
            attention_probs = attention_scores.softmax(dim=-1)
        
        # print(attention_probs.shape,query.shape,key.shape,value.shape)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
    def get_attention_scores(
        self, attn: Attention, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        # del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs, attention_scores

def create_controller(
    prompts: List[str], feat_attn_kwargs: Dict, num_inference_steps: int, tokenizer, device, attn_res
) -> AttentionControl:
    edit_type = feat_attn_kwargs.get("edit_type", None)
    path = feat_attn_kwargs.get("path", None)
    visualize = feat_attn_kwargs.get("visualize", None)
    print_blk = feat_attn_kwargs.get("print_blk", None)
    local_blend_words = feat_attn_kwargs.get("local_blend_words", None)
    edit_blocks = feat_attn_kwargs.get("edit_blocks", None)
    per_word = feat_attn_kwargs.get("per_word_cross_masks", None)
    per_word_cross_masks = per_word[0] if per_word is not None else None
    per_word_strengths = per_word[1] if per_word is not None else None
    word_inds = per_word[2] if per_word is not None else None
    word_cross_coords = per_word[3] if per_word is not None else None

    delta_s = feat_attn_kwargs.get("delta", None)
    place_in_unet = feat_attn_kwargs.get("place_in_unet", None)
    relative_factor = feat_attn_kwargs.get("relative_factor", None)
    blending_alphas = feat_attn_kwargs.get("blending_alphas", None)
    pad_crop = feat_attn_kwargs.get("pad_crop", None)
    pad_kwargs = feat_attn_kwargs.get("pad_kwargs", None)
    sr = feat_attn_kwargs.get("sr", None)
    comp_words = feat_attn_kwargs.get("comp_words", None)
    edit_start_step = feat_attn_kwargs.get("edit_start_step", None)
    extra_kwargs = feat_attn_kwargs.get("extra_kwargs", None)
    
    edit_start_step = int(feat_attn_kwargs["edit_start_step"] * num_inference_steps) if edit_start_step is not None else int(num_inference_steps)
    e_s = {}
    d_s = {}
    if edit_blocks is not None:
        for k in edit_blocks.keys():
            e_s[k] = edit_start_step
            d_s[k] = delta_s
    
    if extra_kwargs is not None:
        if "t_edit" in extra_kwargs.keys():
            for k,v in extra_kwargs["t_edit"].items():
                e_s[k] = int(v * num_inference_steps)
        elif "delta" in extra_kwargs.keys():
            for k,v in extra_kwargs["delta"].items():
                d_s[k] = int(v * num_inference_steps)
    # edit_start_step = e_s
    # delta_s = d_s
    operators = feat_attn_kwargs.get("operators", None)
    op_params = feat_attn_kwargs.get("op_params", None)

    if visualize:
        os.makedirs(os.path.join(path,"viz"),exist_ok=True)
        
    # shift → scale → reweight
    if edit_type == "composite":
        if local_blend_words is not None:
            lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        else:
            lb = None
        return FeatandAttnComposite(
            prompts,
            num_inference_steps,
            operators=operators,
            op_params=op_params,
            tokenizer=tokenizer,
            device=device,
            edit_start_step=edit_start_step,
            comp_words=comp_words,
            attn_res=attn_res,
            local_blend=lb,
            edit_blocks=edit_blocks,
            per_word_cross_masks=per_word_cross_masks,
            per_word_strengths=per_word_strengths,
            word_inds=word_inds,
            word_cross_coords=word_cross_coords,
            delta=delta_s,
            place_in_unet=place_in_unet,
            relative_factor=relative_factor,
            blending_alphas=blending_alphas,
            pad_crop=pad_crop,
            pad_kwargs=pad_kwargs,
            sr=sr,
            path=path,
            visualize=visualize,
            print_blk=print_blk,
        )
    if edit_type == "affine":
        if local_blend_words is not None:
            lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        else:
            lb = None
        return FeatandAttnAffine(
            prompts,
            num_inference_steps,
            operators=operators,
            op_params=op_params,
            tokenizer=tokenizer,
            device=device,
            edit_start_step=edit_start_step,
            comp_words=comp_words,
            attn_res=attn_res,
            local_blend=lb,
            edit_blocks=edit_blocks,
            per_word_cross_masks=per_word_cross_masks,
            per_word_strengths=per_word_strengths,
            word_inds=word_inds,
            word_cross_coords=word_cross_coords,
            delta=delta_s,
            place_in_unet=place_in_unet,
            relative_factor=relative_factor,
            blending_alphas=blending_alphas,
            pad_crop=pad_crop,
            pad_kwargs=pad_kwargs,
            sr=sr,
            path=path,
            visualize=visualize,
            print_blk=print_blk,
        )
    if edit_type == "geometric":
        if local_blend_words is not None:
            lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=attn_res)
        else:
            lb = None
        return FeatandAttnGeometric(
            prompts,
            num_inference_steps,
            operators=operators,
            op_params=op_params,
            tokenizer=tokenizer,
            device=device,
            edit_start_step=edit_start_step,
            comp_words=comp_words,
            attn_res=attn_res,
            local_blend=lb,
            edit_blocks=edit_blocks,
            per_word_cross_masks=per_word_cross_masks,
            per_word_strengths=per_word_strengths,
            word_inds=word_inds,
            word_cross_coords=word_cross_coords,
            delta=delta_s,
            place_in_unet=place_in_unet,
            relative_factor=relative_factor,
            blending_alphas=blending_alphas,
            pad_crop=pad_crop,
            pad_kwargs=pad_kwargs,
            sr=sr,
            path=path,
            visualize=visualize,
            print_blk=print_blk,
        )
    if edit_type == "no-edit":
        return AttentionStore(save=True)

    raise ValueError(f"Edit type {edit_type} not recognized. Use one of: replace, refine, reweight.")


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, val, query, key, is_cross: bool, place_in_unet: str, attn_uncond: bool):
        recomp = False
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if attn.dim() == 4:
                # FLUX MM-DiT: tensor is [batch, heads, seq, seq] (4D); batch=2
                # holds (source, target). No CFG split (schnell, guidance_scale=0),
                # so pass the whole pair to forward — its 4D-aware branch edits
                # attn[1:] in place using attn[0] as source.
                attn, val, query, key, recomp = self.forward(attn, val, query, key, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                # attn[2,3] - im0_cond, im1_cond
                attn[h // 2 :], val[h // 2 :], query[h // 2 :], key[h // 2 :], recomp = self.forward(attn[h // 2 :], val[h // 2 :], query[h // 2 :], key[h // 2 :], is_cross, place_in_unet)
                if attn_uncond:
                    # attn[0,1] - im0_uncond, im1_uncond
                    attn[: h // 2], val[: h // 2], query[: h // 2], key[: h // 2], recomp = self.forward(attn[: h // 2], val[: h // 2], query[: h // 2], key[: h // 2], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn, val, query, key, recomp

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, attn_res=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.attn_res = attn_res

class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, val, query, k, is_cross: bool, place_in_unet: str):
        place_in_unet = place_in_unet[:-1] if place_in_unet != "mid" and place_in_unet[-1] in ["0","1", "2"]  else place_in_unet
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] == 32**2 and self.save:  # avoid memory overhead
            # [32, 4096, T=4096] --> [4,8,4096,T=4096] --> [4,4096,T=4096]
            num_heads = attn.shape[0] // (self.batch_size)
            self.step_store[key].append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1)) # storing cross-attn at res=32 lowest down/up
        return attn, val, query, k, False
    
    def store_resnet_features(self, feats, place_in_unet):
        key = f"{place_in_unet}_resnet"
        if key not in self.resnet_step_store:
            self.resnet_step_store[key] = feats.clone()
        else:
            self.resnet_step_store[key] += feats

        # accumulate into running sum
        if key not in self.resnet_store:
            self.resnet_store[key] = feats.clone()
        else:
            self.resnet_store[key] += feats

        return feats

    def between_steps(self):
        if len(self.attention_store) == 0:
            # self.attention_store = {k: v.copy() for k, v in self.step_store.items()}
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
        self.resnet_step_store = {k: [] for k in self.resnet_step_store}

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
        }
        return average_attention
    
    def get_average_resnet(self):
        average_resnet = {
            key: [item / self.cur_step for item in self.resnet_store[key]] for key in self.resnet_store
        }
        return average_resnet

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=None,save=False):
        super(AttentionStore, self).__init__(attn_res)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.resnet_step_store = {}   # temporary per-step storage
        self.resnet_store = {}        # accumulated features across steps
        self.save=save

class LocalBlend:
    def __call__(self, x_t, attention_store):
        # note that this code works on the latent level!
        k = 1
        # maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]  # These are the numbers because we want to take layers that are 256 x 256, I think this can be changed to something smarter...like, get all attentions where thesecond dim is self.attn_res[0] * self.attn_res[1] in up and down cross.
        maps = [m for m in attention_store["down_cross"] + attention_store["mid_cross"] +  attention_store["up_cross"] if m.shape[1] == self.attn_res[0] * self.attn_res[1]]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, self.attn_res[0], self.attn_res[1], self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1) # since alpha_layers is all 0s except where we edit, the product zeroes out all but what we change. Then, the sum adds the values of the original and what we edit. Then, we average across dim=1, which is the number of layers.
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)

        mask = mask[:1] + mask[1:]
        mask = mask.to(torch.float16)
        x_t = x_t[:1] + mask * (x_t - x_t[:1]) # x_t[:1] is the original image. mask*(x_t - x_t[:1]) zeroes out the original image and removes the difference between the original and each image we are generating (mostly just one). Then, it applies the mask on the image. That is, it's only keeping the cells we want to generate.
        return x_t

    def __init__(
        self, prompts: List[str], words: [List[List[str]]], tokenizer, device, threshold=0.3, attn_res=None
    ):
        self.max_num_words = 77
        self.attn_res = attn_res

        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device) # a one-hot vector where the 1s are the words we modify (source and target)
        self.threshold = threshold
        self.prompts = prompts
        self.words = words
        self.tokenizer = tokenizer 


class FeatandAttnControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    @abc.abstractmethod
    def edit_self_attention(self, attn_base, mod):
        raise NotImplementedError

    @abc.abstractmethod
    def edit_resnet_features(self, feats, place_in_unet):
        raise NotImplementedError
    
    @abc.abstractmethod
    def edit_cross_attention(self, attn_base):
        raise NotImplementedError
    
    @abc.abstractmethod
    def edit_q_k_v(self, val_base, place_in_unet, is_cross):
        raise NotImplementedError

    def forward(self, attn, val, query, key, is_cross: bool, place_in_unet: str):
        super(FeatandAttnControlEdit, self).forward(attn, val, query, key, is_cross, place_in_unet[:-1] if place_in_unet != "mid" else place_in_unet) #AttentionStore
        # attn,q,k,v [i0, i1] [2*num_heads, res*res, T] -- [0,1] or [2,3]
        recomp = False
        if is_cross or (self.edit_start_step + self.delta > self.cur_step >= self.edit_start_step):
            # FLUX MM-DiT tensors arrive as [batch, heads, ...] (4D) — already
            # in the per-image layout this function operates on. SDXL tensors
            # arrive flat as [batch*heads, ...] (3D) and need the reshape.
            is_4d = (attn.dim() == 4)
            if not is_4d:
                h = attn.shape[0] // (self.batch_size)  #num_heads
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:]) # [2, num_heads, res*res, T]
                val = val.reshape(self.batch_size, h, *val.shape[1:])
                query = query.reshape(self.batch_size, h, *query.shape[1:])
                key = key.reshape(self.batch_size, h, *key.shape[1:])

            val_src = val[0]
            query_src = query[0]
            key_src = key[0]
            attn_src = attn[0]
            
            if is_cross:
                if self.should_edit_feat(place_in_unet,"cross"):
                    attn[1:] = self.edit_cross_attention(attn_src, place_in_unet)
                    recomp = False
                if self.should_edit_feat(place_in_unet,"c-query"):
                    query[1:] = self.edit_q_k_v(query_src, place_in_unet, feat_type="query",is_cross=True)
                    recomp = True
                if self.should_edit_feat(place_in_unet,"c-key"):
                    key[1:] = self.edit_q_k_v(key_src, place_in_unet, feat_type="key", is_cross=True)
                    recomp = True
                if self.should_edit_feat(place_in_unet,"c-value"):
                    val[1:] = self.edit_q_k_v(val_src, place_in_unet, feat_type="value", is_cross=True)
                    recomp = True
            else:
                if self.should_edit_feat(place_in_unet,"self"):
                    attn[1:] = self.edit_self_attention(attn_src, place_in_unet)
                    recomp = False
                if self.should_edit_feat(place_in_unet,"s-query"):
                    query[1:] = self.edit_q_k_v(query_src, place_in_unet, feat_type="query",is_cross=False)
                    recomp = True
                if self.should_edit_feat(place_in_unet,"s-key"):
                    key[1:] = self.edit_q_k_v(key_src, place_in_unet, feat_type="key", is_cross=False)
                    recomp = True
                if self.should_edit_feat(place_in_unet,"s-value"):
                    val[1:] = self.edit_q_k_v(val_src, place_in_unet, feat_type="value", is_cross=False)
                    recomp = True
            if not is_4d:
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                val = val.reshape(self.batch_size * h, *val.shape[2:])
                query = query.reshape(self.batch_size * h, *query.shape[2:])
                key = key.reshape(self.batch_size * h, *key.shape[2:])
        # if self.cur_step > (self.edit_start_step + self.delta):
        #     if not is_cross:
                # if self.should_edit_feat(place_in_unet,"s-value"):
                #     h = attn.shape[0] // (self.batch_size)
                #     val = val.reshape(self.batch_size, h, *val.shape[1:])
                #     val[1] = val[0]
                #     val = val.reshape(self.batch_size * h, *val.shape[2:])
                # if self.should_edit_feat(place_in_unet,"s-key"):
                #     h = attn.shape[0] // (self.batch_size)
                #     key = key.reshape(self.batch_size, h, *key.shape[1:])
                #     key[1] = key[0]
                #     key = key.reshape(self.batch_size * h, *key.shape[2:])
        return attn, val, query, key, recomp

    def should_edit_feat(self, place_in_unet,inp_type="cross"):
        return place_in_unet in self.edit_blocks[inp_type]

    def mask_input_(self, inp: torch.Tensor, mask=None, inp_type: str = "resnet", mode: str = "bilinear",
        padding_mode: str = "zeros",):
        dtype, device, B = inp.dtype, inp.device, inp.shape[0]
        if inp_type in [ "cross","s-value","c-value","s-query","c-query","s-key","c-key"]: #, "value"
            H = W = int(inp.shape[1]**0.5) 
            mask = mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask.float(), size=(H, W), mode='bilinear', align_corners=False).view(-1).unsqueeze(-1).repeat(
                B,1,1).to(dtype).to(device)
            return mask
        elif inp_type == "self":
            B, HW, HW = inp.shape
            H = W = int(HW**0.5)
            mask = mask.permute(2, 0, 1).unsqueeze(0)
            mask = F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False
                    ).view(1, -1).to(dtype).to(device)
            # #only Q
            # mask = mask.repeat(B, HW, 1)
            # #only K
            # mask = mask.repeat(B, HW, 1).transpose(1, 2)
            ##both Q and K
            mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
            mask = mask.repeat(B, 1, 1)
            return mask
        else:
            B, C, H, W = inp.shape
            if mask.shape[0] != H:
                mask = mask.permute(2,0,1).unsqueeze(0)
                mask = F.interpolate(mask.float(), mode='bilinear', align_corners=True,  size=(H, W),).squeeze(0).permute(1,2,0)
            mask = mask.unsqueeze(0).unsqueeze(1).squeeze(-1).repeat(B,C,1,1).to(dtype).to(device)
            return mask
                    
    def normalized_blend(self, base_feats, edited_dict, mask_dict, alphas, inp_type="cross"):
        if inp_type =="cross":
            return edited_dict * alphas[0] + alphas[1] * base_feats 
        else:
            feats_accum = torch.zeros_like(base_feats)
            weight_accum = torch.zeros_like(base_feats)  # one weight per pixel/query-key
            for word, edited_word in edited_dict.items():
                if word not in mask_dict:
                    continue
                mask = mask_dict[word]
                feats_accum += alphas[0] * edited_word #/(len(self.comp_words))
                weight_accum += alphas[0] * mask
            feats_accum += alphas[1] * base_feats
            weight_accum += alphas[1]
            return feats_accum / (weight_accum + 1e-8)

    def get_alpha(self,place_in_unet,type_feat="resnet",schedule="lin_w",kwargs=None):
        base_map = self.base_alphas
        kwargs=kwargs[schedule] if kwargs is not None else None
        if type_feat not in base_map:
            raise ValueError(f"Unknown type_feat: {type_feat}")

        base_edit, _ = base_map[type_feat]
        w = self.weight_schedule(schedule=schedule, **kwargs)
        w = 1
        alpha_edit = base_edit * w
        alpha_base = 1.0 - alpha_edit

        return [alpha_edit, alpha_base]
    
    def weight_schedule(self, schedule="lin_w", **kwargs):
        x = self.cur_step / self.num_steps   # normalize to [0,1]

        if schedule == None:
            return 1.0
        if schedule == "lin":
            return max(0.0, 1.0 - x)
        if schedule == "lin_w":
            t1 = kwargs.get("t1", 0.0)  # fraction of T
            t2   = kwargs.get("t2", 0.5)    # fraction of T
            if t1 >= t2:
                raise ValueError("t1 must be < t2 in linear_window schedule")
            if x < t1:
                return 1.0
            if x <= t2:
                return 1.0 - (x - t1) / (t2 - t1)

            # After window: off
            return 0.0
        if schedule == "exp":
            k = kwargs.get("k", 5.0)   # stronger default decay
            return math.exp(-k * x)

        if schedule == "cos":
            return 0.5 * (1 + math.cos(math.pi * x))

        if schedule == "sig":
            k = kwargs.get("k", 10.0)
            t_mid = kwargs.get("t_mid", 0.4)
            return 1 - (1 / (1 + math.exp(-k * (x - t_mid))))

        if schedule == "piece":
            t1 = kwargs.get("t1", 0.3)   # strong region end
            t2 = kwargs.get("t2", 0.7)   # soft region end

            if x < t1:
                return 1.0
            elif x < t2:
                # linear drop from 1→0.5 between t1 and t2
                return 1.0 - 0.5 * ((x - t1) / (t2 - t1))
            else:
                # late stage: strongly reduced edits
                return 0.0
        raise ValueError(f"Unknown schedule: {schedule}")

    def reweight_input_(self, inp: torch.Tensor, weights=None, inp_type: str = "resnet", mode: str = "bilinear",
        padding_mode: str = "zeros",):
        if weights is None:
            print("No editing parameters given")
        ndim = inp.ndim
        if ndim == 3:
            B, HW, T = inp.shape
            H = W = int(HW ** 0.5)
            inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)  # (B, T, H, W)
        elif ndim == 4:
            B, T, H, W = inp.shape
        else:
            raise ValueError(f"Invalid cross-attn shape: {inp.shape}")
        # ---- Case 1: Cross-attention ----
        reweighted = inp.clone()
        if inp_type == "cross":
            for i in range(T):
                reweighted[:,i:i+1] = inp[:, i:i+1] * weights[i]
        # ---- Case 2: ResNet / Self-attn ----
        else:
            reweighted = inp * weights
        if ndim == 3:
                reweighted = reweighted.permute(0, 2, 3, 1).reshape(B, H * W, T)
        return reweighted
    
    def shift_input_(self, inp: torch.Tensor, shifts=None, inp_type: str = "resnet", mode: str = "bilinear",
        padding_mode: str = "zeros",):
        if shifts is None:
            print("No editing parameters given")
        ndim = inp.ndim
        if ndim == 3:
            B, HW, T = inp.shape
            H = W = int(HW ** 0.5)
            inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)  # (B, T, H, W)
        elif ndim == 4:
            B, T, H, W = inp.shape
        else:
            raise ValueError(f"Invalid cross-attn shape: {inp.shape}")
        # Base grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=inp.device, dtype=inp.dtype),
            torch.linspace(-1, 1, W, device=inp.device, dtype=inp.dtype),
            indexing="ij"
        )
        base_grid = torch.stack((x, y), dim=-1)

        # ---- Case 1: Cross-attention ----
        if inp_type == "cross":
            norm_shifts_x = shifts[:, 0].to(inp.dtype) * 2 / (W - 1)
            norm_shifts_y = shifts[:, 1].to(inp.dtype) * 2 / (H - 1)

            shifted_list = []
            for i in range(T):
                shifted_grid = base_grid.clone()
                shifted_grid[..., 0] -= norm_shifts_x[i]
                shifted_grid[..., 1] -= norm_shifts_y[i]
                shifted_grid = shifted_grid.unsqueeze(0).expand(B, H, W, 2).to(inp.dtype)
                shifted = F.grid_sample(inp[:, i:i+1],shifted_grid,mode=mode, padding_mode=padding_mode,align_corners=True,)
                shifted_list.append(shifted)

            x_shifted = torch.cat(shifted_list, dim=1)  # (B, T, H, W)
        # ---- Case 2: ResNet / Self-attn ----
        else:
            dx, dy = shifts
            norm_dx = dx * 2 / (W - 1)
            norm_dy = dy * 2 / (H - 1)

            grid = torch.stack((x - norm_dx, y - norm_dy), dim=-1)
            grid = grid.unsqueeze(0).expand(B, H, W, 2)

            x_shifted = F.grid_sample(inp,grid,mode=mode,padding_mode=padding_mode,align_corners=True,)
        if ndim == 3:
                x_shifted = x_shifted.permute(0, 2, 3, 1).reshape(B, H * W, T)
        return x_shifted
    
    def scale_input_(self, inp: torch.Tensor, scales=None, centers=None, inp_type: str = "resnet", mode: str = "bilinear",
        padding_mode: str = "zeros",):
        if scales is None or centers is None:
            print("No editing parameters given")
        ndim = inp.ndim
        if ndim == 3:
            B, HW, T = inp.shape
            H = W = int(HW ** 0.5)
            inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)  # (B, T, H, W)
        elif ndim == 4:
            B, T, H, W = inp.shape
        else:
            raise ValueError(f"Invalid tensor shape: {inp.shape}")
        # Base grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=inp.device, dtype=inp.dtype),
            torch.linspace(-1, 1, W, device=inp.device, dtype=inp.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1)  # (H, W, 2)

        # ---- Case 1: Cross-attention ----
        if inp_type == "cross":
            if centers is None:
                # Estimate center by max activation per token
                centers_xy = []
                for i in range(T):
                    flat_idx = inp[:, i].abs().mean(0).argmax()
                    cy, cx = divmod(flat_idx.item(), W)
                    centers_xy.append([cx, cy])
                centers = torch.tensor(centers_xy, device=inp.device, dtype=inp.dtype)
            

            scaled_list = []
            for i in range(T):
                s = scales[i]
                cx, cy = centers[i]

                # Normalize centers to [-1,1]
                cx = cx * 2 / (W - 1) - 1
                cy = cy * 2 / (H - 1) - 1

                grid = base_grid.clone()
                grid[..., 0] -= cx
                grid[..., 1] -= cy
                grid /= s  # zoom in/out
                grid[..., 0] += cx
                grid[..., 1] += cy
                grid = grid.unsqueeze(0).expand(B, H, W, 2)

                scaled = F.grid_sample(
                    inp[:, i:i+1],
                    grid,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=True,
                )
                scaled_list.append(scaled)
            out = torch.cat(scaled_list, dim=1)

        # ---- Case 2: ResNet / Self-attn ----
        else:
            if centers is None:
                cx = cy = 0.0
            else:
                cx, cy = centers
                cx = cx * 2 / (W - 1) - 1
                cy = cy * 2 / (H - 1) - 1
            grid = base_grid.clone()
            grid[..., 0] -= cx
            grid[..., 1] -= cy
            grid /= scales
            grid[..., 0] += cx
            grid[..., 1] += cy
            grid = grid.unsqueeze(0).expand(B, H, W, 2)

            out = F.grid_sample(
                inp, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

        if ndim == 3:
            out = out.permute(0, 2, 3, 1).reshape(B, H * W, T)
        return out
    
    def rotate_input_(self, inp: torch.Tensor, angles=None, centers=None, inp_type: str = "resnet", mode: str = "bilinear", padding_mode: str = "zeros",):
        ndim = inp.ndim
        if angles is None or centers is None:
            print("No editing parameters given")
        if ndim == 3:
            B, HW, T = inp.shape
            H = W = int(HW ** 0.5)
            inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)  # (B, T, H, W)
        elif ndim == 4:
            B, T, H, W = inp.shape
        else:
            raise ValueError(f"Invalid tensor shape: {inp.shape}")

        # Base coordinate grid in [-1,1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=inp.device, dtype=inp.dtype),
            torch.linspace(-1, 1, W, device=inp.device, dtype=inp.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1)  # (H, W, 2)

        # --- Case 1: Cross-attention ---
        if inp_type == "cross":
            if centers is None:
                # Find max activation center per token
                centers_xy = []
                for i in range(T):
                    flat_idx = inp[:, i].abs().mean(0).argmax()
                    cy, cx = divmod(flat_idx.item(), W)
                    centers_xy.append([cx, cy])
                centers = torch.tensor(centers_xy, device=inp.device, dtype=inp.dtype)

            rotated_list = []
            for i in range(T):
                θ = angles[i] * torch.pi / 180.0
                cos_t, sin_t = torch.cos(θ), torch.sin(θ)
                rot_matrix = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=inp.device, dtype=inp.dtype)

                cx, cy = centers[i]
                cx = cx * 2 / (W - 1) - 1
                cy = cy * 2 / (H - 1) - 1

                grid = base_grid.clone()
                grid[..., 0] -= cx
                grid[..., 1] -= cy
                grid = grid @ rot_matrix.T
                grid[..., 0] += cx
                grid[..., 1] += cy

                grid = grid.unsqueeze(0).expand(B, H, W, 2)
                rotated = F.grid_sample(
                    inp[:, i:i+1],
                    grid,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=True,
                )
                rotated_list.append(rotated)

            out = torch.cat(rotated_list, dim=1)

        # --- Case 2: ResNet / Self-attn ---
        else:
            θ = angles if torch.is_tensor(angles) else torch.tensor(angles, device=inp.device, dtype=inp.dtype)
            θ = θ * torch.pi / 180.0
            cos_t, sin_t = torch.cos(θ), torch.sin(θ)
            rot_matrix = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=inp.device, dtype=inp.dtype)

            if centers is None:
                cx = cy = 0.0
            else:
                cx, cy = centers
                cx = cx * 2 / (W - 1) - 1
                cy = cy * 2 / (H - 1) - 1

            grid = base_grid.clone()
            grid[..., 0] -= cx
            grid[..., 1] -= cy
            grid = grid @ rot_matrix.T
            grid[..., 0] += cx
            grid[..., 1] += cy
            grid = grid.unsqueeze(0).expand(B, H, W, 2)

            out = F.grid_sample(
                inp, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

        if ndim == 3:
            out = out.permute(0, 2, 3, 1).reshape(B, H * W, T)
        return out
    
    def get_affine_matrix(self,params,H=None, W=None):
        # params: (..., 7)
        tx, ty, sx, sy, theta, shx, shy = params
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        if H is not None and W is not None:
            tx = 2 * tx / W
            ty = 2 * ty / H
        A = torch.stack([
            sx * cos_t + shy * sin_t,
            -sy * sin_t + shx * cos_t,
            tx,
            sx * sin_t + shy * cos_t,
            sy * cos_t + shx * sin_t,
            ty,
        ], dim=-1).reshape(-1, 2, 3)
        return A
    
    def affine_warp_input_(self, inp: torch.Tensor, params=None, inp_type: str = "resnet", mode: str = "bilinear",
        padding_mode: str = "zeros",):
        ndim = inp.ndim
        if params is None:
            print("No editing parameters given")
        if ndim == 3:  # (B, HW, T) or (B, H*W, H*W)
            B, HW, T = inp.shape
            H = W = int(HW ** 0.5)
            inp = inp.reshape(B, H, W, T).permute(0, 3, 1, 2)  # (B, T, H, W)
        elif ndim == 4:
            B, C, H, W = inp.shape
        else:
            raise ValueError(f"Invalid tensor shape {inp.shape}")

        # ---- CASE A: RESNET / SELF-ATTN ----
        if inp_type in ["resnet", "self","s-value","c-value","s-query","c-query","s-key","c-key"]:
            affine_mats = self.get_affine_matrix(torch.tensor(params),H,W).expand(B, -1, -1)
            # Expect affine_mats of shape (B, 2, 3)
            grid = F.affine_grid(affine_mats, size=inp.size(), align_corners=False).to(inp.device).to(inp.dtype)
            warped = F.grid_sample(inp, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

        # ---- CASE B: CROSS-ATTN ----
        elif inp_type == "cross":
            # Expect affine_mats of shape (T, 2, 3)
            warped_list = []
            for i in range(T): 
                A = self.get_affine_matrix(torch.tensor(params[i]),H,W).expand(B, -1, -1)  # (B, 2, 3)
                grid = F.affine_grid(A, (B, 1, H, W), align_corners=False).to(inp.device).to(inp.dtype)
                warped_i = F.grid_sample(inp[:, i:i+1], grid, mode=mode, padding_mode=padding_mode, align_corners=False,)
                warped_list.append(warped_i)
            warped = torch.cat(warped_list, dim=1)

        # ---- RESHAPE BACK IF NEEDED ----
        if ndim == 3:
            warped = warped.permute(0, 2, 3, 1).reshape(B, H * W, T)
        return warped

    def __init__(
        self,
        prompts,
        num_steps: int,
        local_blend: Optional[LocalBlend],
        tokenizer,
        device,
        attn_res=None,
        edit_blocks=None,
        per_word_cross_masks=None,
        edit_start_step=None,
        per_word_strengths=None,
        word_inds=None,
        word_cross_coords=None,
        comp_words=None,
        delta=None,
        place_in_unet=None,
        relative_factor=None,
        blending_alphas=None,
        pad_crop=False,
        pad_kwargs=None,
        sr=256,
        path=None,
        visualize=None,
        print_blk=None,
        
    ):
        super(FeatandAttnControlEdit, self).__init__(attn_res=attn_res)
        # add tokenizer and device here
        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.local_blend = local_blend
        self.edit_start_step = edit_start_step
        self.edit_blocks = edit_blocks
        self.per_word_cross_masks=per_word_cross_masks
        self.per_word_strengths=per_word_strengths
        self.word_cross_coords = word_cross_coords
        self.comp_words=comp_words
        self.num_steps = num_steps
        self.delta = int(delta * num_steps)
        self.place_in_unet = place_in_unet
        self.relative_factor=relative_factor
        self.blending_alphas=blending_alphas
        self.w_schedule = blending_alphas["schedule"]
        self.w_kwargs = blending_alphas["weight_alphas_kwargs"]
        self.base_alphas = blending_alphas["base_alphas"]
        self.pad_crop = pad_crop
        self.pad_kwargs = pad_kwargs
        self.sr=sr
        self.path=path
        self.visualize=visualize
        self.print_blk=print_blk
        
        if word_inds is None:
            idd = {}
            for word in self.comp_words:
                inds = get_word_inds(prompts[0], word, tokenizer)
                idd[word] = inds
            self.word_inds = idd

class FeatandAttnComposite(FeatandAttnControlEdit):
    def composite_apply(self, feat, word, inp_type):
        # pad up to higher size
        ndim = feat.ndim
        if ndim == 3:
            B, HW, T = feat.shape
            H = W = int(HW ** 0.5)
        else:
            B, T, H, W = feat.shape
        if self.pad_crop:
            feat_pad = center_pad(feat,(self.sr,self.sr),self.pad_kwargs)
        else:
            feat_pad = feat.clone()
        edited = feat_pad.clone()
        for op_name in self.operators:
            if inp_type != "cross":
                params = self.op_params[op_name][inp_type].get(word, {})
            else:
                params = self.op_params[op_name].get(inp_type, {})
            if op_name == "reweight":
                edited = self.reweight_input_(inp=edited, inp_type=inp_type, **params)
            elif op_name == "rotate":
                edited = self.rotate_input_(inp=edited, inp_type=inp_type, **params)
            elif op_name == "scale":
                edited = self.scale_input_(inp=edited, inp_type=inp_type, **params)
            elif op_name == "shift":
                edited = self.shift_input_(inp=edited, inp_type=inp_type, **params)
        # central crop back
        edited_crop = center_crop(edited,(H,W)) if self.pad_crop else edited.clone()
        return edited_crop

    def edit_resnet_features(self, feats, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="resnet"):
            return feats
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return feats
        feats_base = feats.clone()
        edited = {}
        mask = {}
        alphas = self.get_alpha(place_in_unet,type_feat="resnet",schedule=self.w_schedule["resnet"], kwargs=self.w_kwargs["resnet"])
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                mask[word] = self.mask_input_(feats_base,self.per_word_cross_masks[word],inp_type="resnet")
                feats_word = feats_base * mask[word]
                if self.cur_step == self.edit_start_step and self.visualize:
                    visualize_resnet_edit(feats_base,feats_word,title=self.path+"/viz/"+f"resnet-masked-{word}")
                    print(f"resnets={self.cur_step}-{word}-{place_in_unet}")
                edited[word] = self.composite_apply(feats_word, word, inp_type="resnet")
                if self.cur_step == self.edit_start_step and self.visualize:
                    visualize_resnet_edit(feats_word,edited[word],title=self.path+"/viz/"+f"resnet-masked-edited-{word}")
        feats_edited = self.normalized_blend(feats_base,edited,mask,alphas,inp_type="resnet")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_resnet_edit(feats_base,feats_edited,title=self.path+"/viz/"+f"resnet-final-edited")
        return feats_edited

    def edit_q_k_v(self, val_base, place_in_unet, feat_type="value", is_cross=False):
        inp_type = f"c-{feat_type}" if is_cross else f"s-{feat_type}"
        if not self.should_edit_feat(place_in_unet, inp_type=inp_type):
            return val_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return val_base
        if val_base.ndim > 3:
            val_base = val_base.squeeze(0)
            
        # B, HW, C = val_base.shape
        # H = W = int(HW**0.5)
        # val_base = val_base.reshape(B,H,W,C).permute(0,3,1,2)
        edited = {}
        mask = {}
        alphas = self.get_alpha(place_in_unet,type_feat=feat_type,schedule=self.w_schedule[feat_type], kwargs=self.w_kwargs[feat_type])
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                mask[word] = self.mask_input_(val_base,self.per_word_cross_masks[word],inp_type=feat_type)
                val_word = val_base * mask[word]
                if self.cur_step == self.edit_start_step and self.visualize:
                    print(f"val={self.cur_step}-{word}-{place_in_unet}")
                edited[word] = self.composite_apply(val_word, word, inp_type=feat_type)
        val_edited = self.normalized_blend(val_base,edited,mask,alphas,inp_type=feat_type)
        # print(val_edited.min(),val_edited.max(),val_edited.mean())
        return val_edited #.permute(0,3,1,2).reshape(B, HW, C)
    
    def edit_cross_attention(self, attn_base, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="cross"):
            return attn_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return attn_base
        if attn_base.ndim > 3:
            attn_base = attn_base.squeeze(0)
        edited = {}
        alphas = self.get_alpha(place_in_unet,type_feat="cross",schedule=self.w_schedule["cross"], kwargs=self.w_kwargs["cross"])
        mask = torch.ones_like(attn_base)
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                inds = self.word_inds[word]
                mask[:,:,inds] = self.mask_input_(attn_base,self.per_word_cross_masks[word],inp_type="cross")
        attn_word = attn_base * mask
        edited = self.composite_apply(attn_word, None, inp_type="cross")
        attn_edited = self.normalized_blend(attn_base,edited,mask,alphas,inp_type="cross")
        if self.cur_step == self.edit_start_step and self.visualize:
            print(f"cross={self.cur_step}-{place_in_unet}")
            visualize_cross_attention_edit(attn_base,attn_edited,token_idx=[self.word_inds[word] for word in self.comp_words],title=self.path+"/viz/"+f"cross-final-edited", word=self.comp_words)
        return attn_edited

    def edit_self_attention(self, attn_base, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="self"):
            return attn_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return attn_base
        if attn_base.ndim > 3:
            attn_base = attn_base.squeeze(0)
        edited = {}
        mask = {}
        alphas = self.get_alpha(place_in_unet,type_feat="self",schedule=self.w_schedule["self"], kwargs=self.w_kwargs["self"])
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                mask[word] = self.mask_input_(attn_base,self.per_word_cross_masks[word],inp_type="self")
                attn_word = attn_base * mask[word]
                if self.cur_step == self.edit_start_step and self.visualize:
                    print(f"self={self.cur_step}-{word}-{place_in_unet}")
                    visualize_self_attention_edit(attn_base,attn_word,title=self.path+"/viz/"+f"self-masked-{word}",normalize=False)   
                edited[word] = self.composite_apply(attn_word, word, inp_type="self")
                if self.cur_step == self.edit_start_step and self.visualize:
                    visualize_self_attention_edit(attn_word,edited[word],title=self.path+"/viz/"+f"self-masked-edited-{word}",normalize=False)
        attn_edited = self.normalized_blend(attn_base,edited,mask,alphas,inp_type="self")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_self_attention_edit(attn_base,attn_edited,title=self.path+"/viz/"+f"self-final-edited",normalize=False)
        return attn_edited

    def make_params(self,operators, op_params, comp_words, prompts, tokenizer, rel_fac):
        params = {"reweight":{},"rotate":{},"shift":{},"scale":{}}
        for k in ["self","resnet","cross","value","query","key"]:
            for key in params.keys():
                params[key][k] = {}
        for op_name in operators:
            if op_name == "reweight":
                temp = get_equalizer(prompts[1], comp_words, op_params[op_name]["equalizer_strengths"], tokenizer=tokenizer, rel_fac=rel_fac)
                for word in temp[1]:
                    for k in ["self","resnet","value","query","key"]:
                        if word not in params["reweight"][k]:
                            params["reweight"][k][word] = {}
                        params["reweight"][k][word]["weights"] = self.relative_factor[k] * temp[2][word]
                params["reweight"]["cross"]["weights"] = temp[0]
            elif op_name == "rotate":
                temp = get_rotater(prompts[1], comp_words, op_params[op_name]["rotate_angles"], tokenizer=tokenizer, rel_fac=rel_fac)
                # coords = torch.zeros((77,2))
                for word in temp[1]:
                    inds= self.word_inds[word]
                    for k in ["self","resnet","value","query","key"]:
                        if word not in params["rotate"][k]:
                            params["rotate"][k][word] = {}
                        params["rotate"][k][word]["angles"] = self.relative_factor[k] * temp[2][word]
                        params["rotate"][k][word]["centers"] = (0.0,0.0) #self.word_cross_coords[word]
                        # coords[inds,0] = self.word_cross_coords[word][0]
                        # coords[inds,1] = self.word_cross_coords[word][1]
                params["rotate"]["cross"]["angles"] = temp[0]
                params["rotate"]["cross"]["centers"] = torch.zeros((77,2)) #coords
            elif op_name == "scale":
                temp = get_scaler(prompts[1], comp_words, op_params[op_name]["scale_strengths"], tokenizer=tokenizer, rel_fac=rel_fac)
                coords = torch.zeros((77,2))
                for word in temp[1]:
                    inds= self.word_inds[word]
                    for k in ["self","resnet","value","query","key"]:
                        if word not in params["scale"][k]:
                            params["scale"][k][word] = {}
                        params["scale"][k][word]["scales"] = self.relative_factor[k] * temp[2][word]
                        params["scale"][k][word]["centers"] = self.word_cross_coords[word]
                        coords[inds,0] = self.word_cross_coords[word][0]
                        coords[inds,1] = self.word_cross_coords[word][1]
                params["scale"]["cross"]["scales"] = temp[0]
                params["scale"]["cross"]["centers"] = coords
            elif op_name == "shift":
                temp = get_shifter(prompts[1], comp_words, op_params[op_name]["shift_pos"], tokenizer=tokenizer, rel_fac=rel_fac, max_pixels=int(0.25*self.sr))
                for word in temp[1]:
                    for k in ["self","resnet","value","query","key"]:
                        if word not in params["shift"][k]:
                            params["shift"][k][word] = {}
                        params["shift"][k][word]["shifts"] = (self.relative_factor[k] * temp[2][word][0], self.relative_factor[k] * temp[2][word][1])
                params["shift"]["cross"]["shifts"] = temp[0]
        return params

    def __init__(
        self,
        prompts,
        num_steps,
        operators,  # list of ops, e.g. ["reweight", "rotate", "scale", "shift"]
        op_params,  # dict: { "reweight": {...}, "rotate": {...}, ... }
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[FeatandAttnControlEdit] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        edit_start_step=None,
        comp_words=None,
        edit_blocks=None,
        per_word_cross_masks=None,
        per_word_strengths=None,
        word_inds=None,
        word_cross_coords=None,
        delta=None,
        place_in_unet=None,
        relative_factor=None,
        blending_alphas=None,
        pad_crop=False,
        pad_kwargs = None,
        sr=256,
        path=None,
        visualize=None,
        print_blk=None,
    ):
        super(FeatandAttnComposite, self).__init__(
            prompts, num_steps,
            local_blend, tokenizer, device, attn_res, edit_blocks, per_word_cross_masks, edit_start_step,
            per_word_strengths, word_inds, word_cross_coords, comp_words, delta, place_in_unet, relative_factor, blending_alphas, 
            pad_crop, pad_kwargs, sr, path, visualize, print_blk,
        )
        self.operators = operators
        self.op_params = self.make_params(operators, op_params, comp_words, prompts, tokenizer, relative_factor["cross"])
        self.comp_words = comp_words
        self.prev_controller = controller

class FeatandAttnGeometric(FeatandAttnControlEdit):
    def geo_trans_apply(self, feat, word, inp_type='cross'):
        # pad up to higher size
        # print(inp_type)
        ndim = feat.ndim
        if ndim == 3:
            B, HW, T = feat.shape
            H_o = W_o = int(HW ** 0.5)
        else:
            B, T, H_o, W_o = feat.shape

        if inp_type != 'cross':
            params = self.op_params["geometric"].get(inp_type).get(word, {})
            if params.numel() == 0:
                return feat
            else:
                feat_pad = center_pad(feat, (self.sr, self.sr),self.pad_kwargs) if self.pad_crop else feat
            if inp_type in ["self", "value", "query","key"]:
                B, HW, C = feat_pad.shape
                H = W = int(HW**0.5)
                feat_pad = feat_pad.reshape(B,H,W,C).permute(0,3,1,2) # [B,C,H,W]
                warped_feat, warped_mask = self.trans(feat_pad,params)
                warped_crop = center_crop(warped_feat,(H_o,W_o)) if self.pad_crop else warped_feat.clone()
                warped_crop = warped_crop.permute(0,2,3,1).reshape(B,H_o*W_o,C)
            if inp_type == "resnet":
                warped_feat, warped_mask_ = self.trans(feat_pad,params)
                warped_crop = center_crop(warped_feat,(H_o,W_o)) if self.pad_crop else warped_feat.clone()
                # visualize_mask_edit(warped_mask_.float(),title=self.path+"/viz/"+f"mask-edited")                
        elif inp_type == "cross":
            warped_crop = feat.clone()
            # for word in self.comp_words:
            params = self.op_params["geometric"].get(inp_type).get(word, {})
            if params.numel() == 0:
                return feat
            else:
                feat_pad = center_pad(feat, (self.sr, self.sr),self.pad_kwargs) if self.pad_crop else feat
            inds = self.word_inds[word][0]
            B, HW, C = feat_pad.shape
            H = W = int(HW**0.5)
            feat_pad = feat_pad.reshape(B,H,W,C).permute(0,3,1,2)
            warped_feat, warped_mask = self.trans(feat_pad,params)
            warped_crop_ = center_crop(warped_feat,(H_o,W_o)) if self.pad_crop else warped_feat.clone()
            warped_crop_ = warped_crop_.permute(0,2,3,1).reshape(B,H_o*W_o,C)
            # warped_crop[:,:,inds:inds+1] = warped_crop_[:,:,inds:inds+1]
            warped_crop = warped_crop_
        return warped_crop
################################ one mask #####################################
    def edit_resnet_features(self, feats, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="resnet"):
            return feats
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return feats
        if self.cur_step > 5:
            return feats
        feats_base = feats.clone()
        edited = self.geo_trans_apply(feats_base, self.comp_words[0], inp_type="resnet")
        if self.cur_step == self.edit_start_step and self.print_blk:
            print(f"resnets={self.cur_step}-{place_in_unet}")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_resnet_edit(feats_base,edited,title=self.path+"/viz/"+f"resnet-edited")
        indices = torch.tensor([0, 0, 2, 2], dtype=torch.long,device=feats.device)
        edited_ = torch.index_select(edited, dim=0, index=indices)
        return edited_
    
    def edit_q_k_v(self, val_base, place_in_unet, feat_type="value", is_cross=False):
        inp_type = f"c-{feat_type}" if is_cross else f"s-{feat_type}"
        if not self.should_edit_feat(place_in_unet, inp_type=inp_type):
            return val_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return val_base
        if val_base.ndim > 3:
            val_base = val_base.squeeze(0)
        edited = self.geo_trans_apply(val_base, self.comp_words[0], inp_type=feat_type)
        if self.cur_step == self.edit_start_step and self.print_blk:
            print(f"{inp_type}={self.cur_step}-{place_in_unet}")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_qkv_edit(val_base,edited,title=self.path+"/viz/"+f"{inp_type}-edited",normalize=False)
        return edited
    
    def edit_self_attention(self, attn_base, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="self"):
            return attn_base
        # if attn_base.shape[2] <= 32 ** 2:
        #     attn_base = attn_base.unsqueeze(0).expand(attn_base.shape[0], *attn_base.shape)
        #     return attn_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return attn_base
        if attn_base.ndim > 3:
            attn_base = attn_base.squeeze(0)
        edited = self.geo_trans_apply(attn_base, self.comp_words[0], inp_type="self")
        if self.cur_step == self.edit_start_step and self.print_blk:
            print(f"self={self.cur_step}-{place_in_unet}")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_self_attention_edit(attn_base,edited,title=self.path+"/viz/"+f"self-edited",normalize=False)
        return edited
    
    def edit_cross_attention(self, attn_base, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="cross"):
            return attn_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return attn_base
        if attn_base.ndim > 3:
            attn_base = attn_base.squeeze(0)
        edited = self.geo_trans_apply(attn_base, self.comp_words[0], inp_type="cross")
        if self.cur_step == self.edit_start_step and self.print_blk:
            print(f"cross={self.cur_step}-{place_in_unet}")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_cross_attention_edit(attn_base,edited,token_idx=[self.word_inds[word] for word in self.comp_words[0:1]],title=self.path+"/viz/"+f"cross-edited", word=self.comp_words[0])
        return edited
################################ no mask ######################################
    # def edit_resnet_features(self, feats, place_in_unet):
    #     if not self.should_edit_feat(place_in_unet, inp_type="resnet"):
    #         return feats
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return feats

    #     feats_base = feats.clone()
    #     edited = {}
    #     mask = {}
    #     alphas = self.get_alpha(place_in_unet, type_feat="resnet",schedule=self.w_schedule["resnet"], kwargs=self.w_kwargs["resnet"])
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             mask[word] = torch.ones_like(feats_base)
    #             feats_word = feats_base * mask[word]
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 visualize_resnet_edit(feats_base,feats_word,title=self.path+"/viz/"+f"resnet-masked-{word}")
    #                 print(f"resnets={self.cur_step}-{word}-{place_in_unet}")
    #             edited[word] = self.geo_trans_apply(feats_word, word, inp_type="resnet")
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 visualize_resnet_edit(feats_word,edited[word],title=self.path+"/viz/"+f"resnet-masked-edited-{word}")
    #     feats_edited = self.normalized_blend(feats_base, edited, mask, alphas, inp_type="resnet")
    #     if self.cur_step == self.edit_start_step and self.visualize:
    #         visualize_resnet_edit(feats_base,feats_edited,title=self.path+"/viz/"+f"resnet-final-edited")
    #     return feats_edited

    # def edit_q_k_v(self, val_base, place_in_unet, feat_type="value", is_cross=False):
        # inp_type = f"c-{feat_type}" if is_cross else f"s-{feat_type}"
        # if not self.should_edit_feat(place_in_unet, inp_type=inp_type):
    #         return val_base
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return val_base
    #     if val_base.ndim > 3:
    #         val_base = val_base.squeeze(0)
    #     edited = {}
    #     mask = {}
    #     alphas = self.get_alpha(place_in_unet,type_feat=feat_type,schedule=self.w_schedule[feat_type], kwargs=self.w_kwargs[feat_type])
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             mask[word] = torch.ones_like(val_base)
    #             val_word = val_base * mask[word]
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 print(f"val={self.cur_step}-{word}-{place_in_unet}")
    #             edited[word] = self.geo_trans_apply(val_word, word, inp_type=feat_type)
    #     val_edited = self.normalized_blend(val_base,edited,mask,alphas,inp_type=feat_type)
    #     return val_edited
    
    # def edit_cross_attention(self, attn_base, place_in_unet):
    #     if not self.should_edit_feat(place_in_unet, inp_type="cross"):
    #         return attn_base
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return attn_base
    #     if attn_base.ndim > 3:
    #         attn_base = attn_base.squeeze(0)
    #     alphas = self.get_alpha(place_in_unet, type_feat="cross",schedule=self.w_schedule["cross"], kwargs=self.w_kwargs["cross"])
    #     mask = torch.ones_like(attn_base)
    #     edited = {}
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             inds = self.word_inds[word]
    #             mask[:, :, inds] = torch.ones_like(attn_base[:, :, inds])
    #     attn_word = attn_base * mask
    #     edited = self.geo_trans_apply(attn_word, self.comp_words, inp_type="cross")
    #     attn_edited = self.normalized_blend(attn_base, edited, mask, alphas, inp_type="cross")
    #     if self.cur_step == self.edit_start_step and self.visualize:
    #         print(f"cross={self.cur_step}-{place_in_unet}")
    #         visualize_cross_attention_edit(attn_base,attn_edited,token_idx=[self.word_inds[word] for word in self.comp_words],title=self.path+"/viz/"+f"cross-final-edited", word=self.comp_words)
    #     return attn_edited
    
    # def edit_self_attention(self, attn_base, place_in_unet):
    #     if not self.should_edit_feat(place_in_unet, inp_type="self"):
    #         return attn_base
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return attn_base
    #     if attn_base.ndim > 3:
    #         attn_base = attn_base.squeeze(0)
    #     alphas = self.get_alpha(place_in_unet, type_feat="self",schedule=self.w_schedule["self"], kwargs=self.w_kwargs["self"])
    #     edited = {}
    #     mask = {}
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             mask[word] = torch.ones_like(attn_base)
    #             attn_word = attn_base * mask[word]
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 print(f"self={self.cur_step}-{word}-{place_in_unet}")
    #                 visualize_self_attention_edit(attn_base,attn_word,title=self.path+"/viz/"+f"self-masked-{word}",normalize=False)
    #             edited[word] = self.geo_trans_apply(attn_word, word, inp_type="self")
    #         if self.cur_step == self.edit_start_step and self.visualize:
    #             visualize_self_attention_edit(attn_word,edited[word],title=self.path+"/viz/"+f"self-masked-edited-{word}",normalize=False)
    #     attn_edited = self.normalized_blend(attn_base, edited, mask, alphas, inp_type="self")
    #     if self.cur_step == self.edit_start_step and self.visualize:
    #         visualize_self_attention_edit(attn_base,attn_edited,title=self.path+"/viz/"+f"self-final-edited",normalize=False)
    #     return attn_edited
    ################################## mask ##########################
    # def edit_resnet_features(self, feats, place_in_unet):
    #     if not self.should_edit_feat(place_in_unet, inp_type="resnet"):
    #         return feats
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return feats
    #     feats_base = feats.clone()
    #     edited = {}
    #     mask = {}
    #     alphas = self.get_alpha(place_in_unet, type_feat="resnet",schedule=self.w_schedule["resnet"], kwargs=self.w_kwargs["resnet"])
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             mask[word] = self.mask_input_(feats_base, self.per_word_cross_masks[word], inp_type="resnet")
    #             feats_word = feats_base * mask[word]
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 visualize_resnet_edit(feats_base,feats_word,title=self.path+"/viz/"+f"resnet-masked-{word}")
    #                 print(f"resnets={self.cur_step}-{word}-{place_in_unet}")
    #             edited[word] = self.geo_trans_apply(feats_word, word, inp_type="resnet")
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 visualize_resnet_edit(feats_word,edited[word],title=self.path+"/viz/"+f"resnet-masked-edited-{word}")
    #     feats_edited = self.normalized_blend(feats_base, edited, mask, alphas, inp_type="resnet")
    #     if self.cur_step == self.edit_start_step and self.visualize:
    #         visualize_resnet_edit(feats_base,feats_edited,title=self.path+"/viz/"+f"resnet-final-edited")
    #     return feats_edited

    # def edit_q_k_v(self, val_base, place_in_unet, feat_type="value", is_cross=False):
        # inp_type = f"c-{feat_type}" if is_cross else f"s-{feat_type}"
        # if not self.should_edit_feat(place_in_unet, inp_type=inp_type):
    #         return val_base
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return val_base
    #     if val_base.ndim > 3:
    #         val_base = val_base.squeeze(0)
    #     edited = {}
    #     mask = {}
    #     alphas = self.get_alpha(place_in_unet,type_feat=feat_type,schedule=self.w_schedule[feat_type], kwargs=self.w_kwargs[feat_type])
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             mask[word] = self.mask_input_(val_base,self.per_word_cross_masks[word],inp_type=feat_type)
    #             val_word = val_base * mask[word]
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 print(f"val={self.cur_step}-{word}-{place_in_unet}")
    #             edited[word] = self.geo_trans_apply(val_word, word, inp_type=feat_type)
    #     val_edited = self.normalized_blend(val_base,edited,mask,alphas,inp_type=feat_type)
    #     return val_edited
    
    # def edit_cross_attention(self, attn_base, place_in_unet):
    #     if not self.should_edit_feat(place_in_unet, inp_type="cross"):
    #         return attn_base
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return attn_base
    #     if attn_base.ndim > 3:
    #         attn_base = attn_base.squeeze(0)
    #     alphas = self.get_alpha(place_in_unet, type_feat="cross",schedule=self.w_schedule["cross"], kwargs=self.w_kwargs["cross"])
    #     mask = torch.ones_like(attn_base)
    #     edited = {}
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             inds = self.word_inds[word]
    #             mask[:, :, inds] = self.mask_input_(attn_base, self.per_word_cross_masks[word], inp_type="cross")
    #     attn_word = attn_base * mask
    #     edited = self.geo_trans_apply(attn_word, self.comp_words, inp_type="cross")
    #     attn_edited = self.normalized_blend(attn_base, edited, mask, alphas, inp_type="cross")
    #     if self.cur_step == self.edit_start_step and self.visualize:
    #         print(f"cross={self.cur_step}-{place_in_unet}")
    #         visualize_cross_attention_edit(attn_base,attn_edited,token_idx=[self.word_inds[word] for word in self.comp_words],title=self.path+"/viz/"+f"cross-final-edited", word=self.comp_words)
    #     return attn_edited

    # def edit_self_attention(self, attn_base, place_in_unet):
    #     if not self.should_edit_feat(place_in_unet, inp_type="self"):
    #         return attn_base
    #     if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
    #         return attn_base
    #     if attn_base.ndim > 3:
    #         attn_base = attn_base.squeeze(0)
    #     alphas = self.get_alpha(place_in_unet, type_feat="self",schedule=self.w_schedule["self"], kwargs=self.w_kwargs["self"])
    #     edited = {}
    #     mask = {}
    #     for word in self.comp_words:
    #         if word in self.per_word_cross_masks:
    #             mask[word] = self.mask_input_(attn_base, self.per_word_cross_masks[word], inp_type="self")
    #             attn_word = attn_base * mask[word]
    #             if self.cur_step == self.edit_start_step and self.visualize:
    #                 print(f"self={self.cur_step}-{word}-{place_in_unet}")
    #                 visualize_self_attention_edit(attn_base,attn_word,title=self.path+"/viz/"+f"self-masked-{word}",normalize=False)
    #             edited[word] = self.geo_trans_apply(attn_word, word, inp_type="self")
    #         if self.cur_step == self.edit_start_step and self.visualize:
    #             visualize_self_attention_edit(attn_word,edited[word],title=self.path+"/viz/"+f"self-masked-edited-{word}",normalize=False)
    #     attn_edited = self.normalized_blend(attn_base, edited, mask, alphas, inp_type="self")
    #     if self.cur_step == self.edit_start_step and self.visualize:
    #         visualize_self_attention_edit(attn_base,attn_edited,title=self.path+"/viz/"+f"self-final-edited",normalize=False)
    #     return attn_edited
    
    def __init__(
        self,
        prompts,
        num_steps,
        operators,
        op_params,  # dict: {"affine": {"params": per-word param list or dict}}
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[FeatandAttnControlEdit] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        edit_start_step=None,
        comp_words=None,
        edit_blocks=None,
        per_word_cross_masks=None,
        per_word_strengths=None,
        word_inds=None,
        word_cross_coords=None,
        delta=None,
        place_in_unet=None,
        relative_factor=None,
        blending_alphas=None,
        pad_crop=False,
        pad_kwargs=None,
        sr=256,
        path=None,
        visualize=None,
        print_blk=None,
    ):
        super().__init__(
            prompts, num_steps, local_blend, tokenizer, device, attn_res, edit_blocks, per_word_cross_masks,
            edit_start_step, per_word_strengths, word_inds, word_cross_coords, comp_words, delta, place_in_unet, relative_factor, blending_alphas, 
            pad_crop, pad_kwargs, sr, path, visualize, print_blk,
        )
        self.comp_words = comp_words
        self.prev_controller = controller
        temp = get_geo_transformer(comp_words, op_params["geometric"]["params"])
        params = {"geometric":{}}
        for k in ["self","resnet","cross","value","query","key"]:
            for key in params.keys():
                params[key][k] = {}
        for word in temp[0]:
            for k in ["self","resnet","value","cross","query","key"]:
                if word not in params["geometric"][k]:
                    params["geometric"][k][word] = {}
                params["geometric"][k][word] = torch.tensor(list(element * self.relative_factor[k] for element in temp[1][word]),device=self.device)
        self.op_params = params
        self.trans = op_params["geometric"]["trans"]


class FeatandAttnAffine(FeatandAttnControlEdit):
    def affine_apply(self, feat, word, inp_type):
        # pad up to higher size
        ndim = feat.ndim
        if ndim == 3:
            B, HW, T = feat.shape
            H = W = int(HW ** 0.5)
        else:
            B, T, H, W = feat.shape
        if self.pad_crop:
            feat_pad = center_pad(feat,(self.sr,self.sr),self.pad_kwargs)
        else:
            feat_pad = feat.clone()
        if inp_type != "cross":
            params = self.op_params["affine"].get(inp_type).get(word, {})
        else:
            params = self.op_params["affine"].get(inp_type, {})
        if params is None:
            return feat
        warped = self.affine_warp_input_(inp=feat_pad, inp_type=inp_type, **params)
        # central crop back
        warped_crop = center_crop(warped,(H,W)) if self.pad_crop else warped.clone()
        return warped_crop

    def edit_resnet_features(self, feats, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="resnet"):
            return feats
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return feats

        feats_base = feats.clone()
        edited = {}
        mask = {}
        alphas = self.get_alpha(place_in_unet, type_feat="resnet",schedule=self.w_schedule["resnet"], kwargs=self.w_kwargs["resnet"])
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                mask[word] = self.mask_input_(feats_base, self.per_word_cross_masks[word], inp_type="resnet")
                feats_word = feats_base * mask[word]
                if self.cur_step == self.edit_start_step and self.visualize:
                    visualize_resnet_edit(feats_base,feats_word,title=self.path+"/viz/"+f"resnet-masked-{word}")
                    print(f"resnets={self.cur_step}-{word}-{place_in_unet}")
                edited[word] = self.affine_apply(feats_word, word, inp_type="resnet")
                if self.cur_step == self.edit_start_step and self.visualize:
                    visualize_resnet_edit(feats_word,edited[word],title=self.path+"/viz/"+f"resnet-masked-edited-{word}")
        feats_edited = self.normalized_blend(feats_base, edited, mask, alphas, inp_type="resnet")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_resnet_edit(feats_base,feats_edited,title=self.path+"/viz/"+f"resnet-final-edited")
        return feats_edited

    def edit_q_k_v(self, val_base, place_in_unet, feat_type="value", is_cross=False):
        inp_type = f"c-{feat_type}" if is_cross else f"s-{feat_type}"
        if not self.should_edit_feat(place_in_unet, inp_type=inp_type):
            return val_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return val_base
        if val_base.ndim > 3:
            val_base = val_base.squeeze(0)
        edited = {}
        mask = {}
        alphas = self.get_alpha(place_in_unet,type_feat=feat_type,schedule=self.w_schedule[feat_type], kwargs=self.w_kwargs[feat_type])
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                mask[word] = self.mask_input_(val_base,self.per_word_cross_masks[word],inp_type=feat_type)
                val_word = val_base * mask[word]
                if self.cur_step == self.edit_start_step and self.visualize:
                    print(f"val={self.cur_step}-{word}-{place_in_unet}")
                edited[word] = self.affine_apply(val_word, word, inp_type=feat_type)
        val_edited = self.normalized_blend(val_base,edited,mask,alphas,inp_type=feat_type)
        return val_edited
    
    def edit_cross_attention(self, attn_base, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="cross"):
            return attn_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return attn_base
        if attn_base.ndim > 3:
            attn_base = attn_base.squeeze(0)
        alphas = self.get_alpha(place_in_unet, type_feat="cross",schedule=self.w_schedule["cross"], kwargs=self.w_kwargs["cross"])
        mask = torch.ones_like(attn_base)
        edited = {}
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                inds = self.word_inds[word]
                mask[:, :, inds] = self.mask_input_(attn_base, self.per_word_cross_masks[word], inp_type="cross")
        attn_word = attn_base * mask
        edited = self.affine_apply(attn_word, None, inp_type="cross")
        attn_edited = self.normalized_blend(attn_base, edited, mask, alphas, inp_type="cross")
        if self.cur_step == self.edit_start_step and self.visualize:
            print(f"cross={self.cur_step}-{place_in_unet}")
            visualize_cross_attention_edit(attn_base,attn_edited,token_idx=[self.word_inds[word] for word in self.comp_words],title=self.path+"/viz/"+f"cross-final-edited", word=self.comp_words)
        return attn_edited

    def edit_self_attention(self, attn_base, place_in_unet):
        if not self.should_edit_feat(place_in_unet, inp_type="self"):
            return attn_base
        if self.edit_start_step + self.delta <= self.cur_step < self.edit_start_step:
            return attn_base
        if attn_base.ndim > 3:
            attn_base = attn_base.squeeze(0)
        alphas = self.get_alpha(place_in_unet, type_feat="self",schedule=self.w_schedule["self"], kwargs=self.w_kwargs["self"])
        edited = {}
        mask = {}
        for word in self.comp_words:
            if word in self.per_word_cross_masks:
                mask[word] = self.mask_input_(attn_base, self.per_word_cross_masks[word], inp_type="self")
                attn_word = attn_base * mask[word]
                if self.cur_step == self.edit_start_step and self.visualize:
                    print(f"self={self.cur_step}-{word}-{place_in_unet}")
                    visualize_self_attention_edit(attn_base,attn_word,title=self.path+"/viz/"+f"self-masked-{word}",normalize=False)
                edited[word] = self.affine_apply(attn_word, word, inp_type="self")
            if self.cur_step == self.edit_start_step and self.visualize:
                visualize_self_attention_edit(attn_word,edited[word],title=self.path+"/viz/"+f"self-masked-edited-{word}",normalize=False)
        attn_edited = self.normalized_blend(attn_base, edited, mask, alphas, inp_type="self")
        if self.cur_step == self.edit_start_step and self.visualize:
            visualize_self_attention_edit(attn_base,attn_edited,title=self.path+"/viz/"+f"self-final-edited",normalize=False)
        return attn_edited
    
    def __init__(
        self,
        prompts,
        num_steps,
        operators,
        op_params,  # dict: {"affine": {"params": per-word param list or dict}}
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[FeatandAttnControlEdit] = None,
        tokenizer=None,
        device=None,
        attn_res=None,
        edit_start_step=None,
        comp_words=None,
        edit_blocks=None,
        per_word_cross_masks=None,
        per_word_strengths=None,
        word_inds=None,
        word_cross_coords=None,
        delta=None,
        place_in_unet=None,
        relative_factor=None,
        blending_alphas=None,
        pad_crop=False,
        pad_kwargs=None,
        sr=256,
        path=None,
        visualize=None,
        print_blk=None,
    ):
        super().__init__(
            prompts, num_steps, local_blend, tokenizer, device, attn_res, edit_blocks, per_word_cross_masks,
            edit_start_step, per_word_strengths, word_inds, word_cross_coords, comp_words, delta, place_in_unet, relative_factor, blending_alphas, 
            pad_crop, pad_kwargs, sr, path, visualize, print_blk,
        )
        self.comp_words = comp_words
        self.prev_controller = controller
        temp = get_affiner(prompts[1], comp_words, op_params["affine"]["params"], tokenizer=tokenizer, rel_fac=relative_factor["cross"], max_pixels=int(0.3*next(iter(self.per_word_cross_masks.values())).shape[0]))
        params = {"affine":{}}
        for k in ["self","resnet","cross","value","query","key"]:
            for key in params.keys():
                params[key][k] = {}
        for word in temp[1]:
            for k in ["self","resnet","value","query","key"]:
                if word not in params["affine"][k]:
                    params["affine"][k][word] = {}
                params["affine"][k][word]["params"] = tuple(element * self.relative_factor[k] for element in temp[2][word])
        params["affine"]["cross"]["params"] = temp[0]
        self.op_params = params

    
### util functions for all Edits
def update_alpha_time_word(
    alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor] = None
):
    if isinstance(bounds, float):
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha



### util function for all edits
def get_valid_tokens(prompt, tokenizer):
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger_eng")

    tokens = tokenizer(prompt)["input_ids"]
    decoder = tokenizer.decode

    words = [decoder(token) for token in tokens]
    tags = nltk.pos_tag(words)

    selected_indices = []
    selected_words = []
    stop_words = set(stopwords.words('english'))

    for i, (word, pos) in enumerate(tags):
        if i == 0:
            continue
        if word in stop_words or word in string.punctuation or pos.startswith("VB"):
            continue
        selected_indices.append(i)
        selected_words.append(word)
    return selected_indices, selected_words[:-1]

### util functions for LocalBlend and ReplacementEdit
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def view_images(images, num_rows=1, offset_ratio=0.02, name="a"):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(f"{name}.png")
    return pil_img

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def aggregate_attention(attention_store, text: str, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(text), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention(attention_store, tokenizer, text: str, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(text[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, text, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    view_images(np.stack(images, axis=0),name=f"./cross_attn_at_{select}")

def show_self_attention_comp(attention_store, text: str, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, text, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    attention_maps = attention_maps.astype('float32')
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1),name=f"./cross_attn_comp_{select}_{max_com}")

def show_resnet_feature_components(attention_store, res, from_where: list, max_com=4, select=0, name=None):
    """
    Visualize spatial co-activation structure from ResNet feature maps using SVD
    over the spatial correlation matrix (not raw channels).
    """

    resnet_map = attention_store.get_average_resnet()
    num_pixels = res
    out_feats = []

    # Collect ResNet features
    for location in from_where:
        key = f"{location}_resnet"
        if key not in resnet_map or len(resnet_map[key]) == 0:
            print(f"No features found for {key}.")
            continue

        for item in resnet_map[key]:
            if item.shape[2] == num_pixels:  # match resolution
                maps = item.reshape(item.shape[0], -1, res, res)[select]
                out_feats.append(maps)

    if not out_feats:
        print("No matching features found at this resolution.")
        return

    out_feats = torch.stack(out_feats, dim=0).mean(0)  # [C, H, W]
    out_feats = out_feats.to(torch.float32)#numpy().astype('float32')

    C, H, W = out_feats.shape
    feat = out_feats.reshape(C, -1)  # [C, H*W]
    feat = feat - feat.mean(axis=1, keepdims=True)
    corr = torch.matmul(feat.T, feat)  # spatial similarity
    corr /= (torch.linalg.norm(feat, axis=0, keepdims=True).T @ torch.linalg.norm(feat, axis=0, keepdims=True) + 1e-8)
    corr = torch.nan_to_num(corr)
    u, s, vh = torch.linalg.svd(corr - corr.mean(axis=1, keepdims=True), full_matrices=False)

    images = []
    for i in range(min(max_com, u.shape[1])):
        comp = u[:, i].reshape(res, res).cpu().numpy()
        comp = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
        comp = (comp * 255).astype(np.uint8)
        comp = np.repeat(comp[:, :, None], 3, axis=2)
        img = Image.fromarray(comp).resize((256, 256))
        images.append(np.array(img))

    concatenated = np.concatenate(images, axis=1)
    view_images(concatenated, name=name if name is not None else f"./resnet_spatial_modes_{from_where[0]}_{res}")

def show_resnet_feature_overlay(attention_store, res, from_where: list, max_com=4, select=0, alpha=0.6, name=None):
    """
    Overlay top spatial SVD components from ResNet feature maps with distinct colors.
    """

    resnet_map = attention_store.get_average_resnet()
    out_feats = []
    # Collect ResNet features
    for location in from_where:
        key = f"{location}_resnet"
        if key not in resnet_map or len(resnet_map[key]) == 0:
            print(f"No features found for {key}.")
            continue

        for item in resnet_map[key]:
            if item.shape[2] == res:  # match resolution
                maps = item.reshape(item.shape[0], -1, res, res)[select]
                out_feats.append(maps)

    if not out_feats:
        print("No matching features found.")
        return

    # Aggregate and reshape
    out_feats = torch.stack(out_feats, dim=0).mean(0)  # [C, H, W]
    # out_feats = out_feats.cpu().numpy().astype('float32')
    out_feats = out_feats.to(torch.float32)
    C, H, W = out_feats.shape
    feat = out_feats.reshape(C, -1)
    feat -= feat.mean(axis=1, keepdims=True)
    corr = torch.matmul(feat.T, feat)
    corr /= (torch.linalg.norm(feat, axis=0, keepdims=True).T @ torch.linalg.norm(feat, axis=0, keepdims=True) + 1e-8)
    corr = torch.nan_to_num(corr)
    u, s, vh = torch.linalg.svd(corr - corr.mean(axis=1, keepdims=True), full_matrices=False)

    # --- Create RGB overlay of top spatial components ---
    color_map = np.array([
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.0, 1.0],   # magenta
        [0.0, 1.0, 1.0],   # cyan
    ])

    num_com = min(max_com, color_map.shape[0])
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(num_com):
        comp = u[:, i].reshape(H, W).cpu().numpy()
        comp = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
        comp = np.clip(comp, 0, 1)
        overlay += alpha * comp[..., None] * color_map[i]

    overlay = np.clip(overlay, 0, 1)

    img = (overlay * 255).astype(np.uint8)
    img = Image.fromarray(img).resize((256, 256))

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"ResNet Spatial Components Overlay ({res}x{res})")
    plt.show()
    plt.savefig(name if name is not None else f"./resnet_svd_{from_where[0]}_{res}")
    plt.close()

def create_cross_attention_masks(attention_store, prompts, words, tokenizer, from_where, out_res, ratio=0.6):
    attn = aggregate_cross_attention(attention_store, from_where, is_cross=True, out_res=out_res)
    # output = []
    output = {}
    images = []
    weights = {}
    idx = {}
    cord = {}
    for i, (prompt, words_) in enumerate(zip(prompts, words)):
        if isinstance(words_, str):
            words_ = [words_]
        for word in words_:
            inds = get_word_inds(prompt, word, tokenizer)
            # image = attn[:, :, inds]
            # image = 255 * image / image.max()
            # image = image.unsqueeze(-1).expand(*image.shape, 3)
            # image = image.squeeze(2).detach().cpu().numpy().astype(np.uint8)
            # image = np.array(Image.fromarray(image).resize((256, 256)))
            # image = text_under_image(image, word)
            # images.append(image)
            mask, coord, mean_val = get_energy_attn_map(attn[:, :, inds],ratio,inds,word)
            # output.append({'word':word,'index':inds,'mask':mask,'coord':coord})
            output[word] = torch.from_numpy(mask)
            weights[word] = mean_val
            idx[word] = inds
            cord[word] = coord
            # view_images(np.stack(images, axis=0),name=f"./cross_attn_at_{word}")
    total_sum = sum(weights.values())
    return [output, {key: value / total_sum for key, value in weights.items()}, idx, cord]

def aggregate_cross_attention(attention_store, from_where: List[str], is_cross: bool, out_res: int) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        cat_dim = 0
        attention_maps = attention_store.get_average_attention()
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                res = int(np.sqrt(item.shape[1]))
                cross_maps = item.reshape(
                    -1, res, res, item.shape[-1]
                )
                cross_maps = cross_maps.permute(0, 3, 1, 2)
                cross_maps = F.interpolate(
                    cross_maps,
                    mode='bilinear',
                    align_corners=True,
                    size=(out_res, out_res),
                )
                cross_maps = cross_maps.permute(0, 2, 3, 1)
                out.append(cross_maps)
        out = torch.cat(out, dim=cat_dim)
        out = out.sum(0) / out.shape[0]
        return out

def get_energy_attn_map(attn, ratio, idx, title):
    max_val = torch.max(attn)
    threshold = ratio * max_val
    binary_map = (attn >= threshold).detach().cpu().numpy().astype(np.uint8)
    flat_idx = attn.argmax()
    h, w, _ = attn.shape
    row, col = divmod(flat_idx.item(), w)
    norm_x = col * 2 / (w - 1) - 1  # [-1, 1]
    norm_y = row * 2 / (h - 1) - 1
    cord = (norm_x, norm_y)  # for F.grid_sample grids
    return binary_map, cord, attn.mean().item()

### util functions for ReweightEdit
def get_equalizer(
    text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer, rel_fac
):
    def normalizer(s_list, max_logit=10):
        res = []
        for s in s_list:
            s = float(s)
            s = max(-1.0, min(1.0, s))
            res.append(s * max_logit)
        return res
    
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    equalizer = torch.ones(77)
    values = torch.tensor(normalizer(values), dtype=torch.float32)
    # values = torch.tensor(values, dtype=torch.float32)
    dict_equalize = {}
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[inds] = torch.FloatTensor(rel_fac * values[i])
        dict_equalize[word] = values[i]
    return equalizer, word_select, dict_equalize

def get_geo_transformer(
    word_select: Union[int, Tuple[int, ...]],
    affines: Any,
):
    # Normalize input format
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    if isinstance(affines[0], (int, float)):
        affines = [affines]

    # --- Assign to words ---
    dict_affine = {}
    for i, word in enumerate(word_select):
        dict_affine[word] = affines[i]

    return word_select, dict_affine
    
def get_affiner(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    affines: Union[
        List[Tuple[float, float, float, float, float]],
        Tuple[Tuple[float, float, float, float, float], ...],
    ],
    tokenizer,
    rel_fac,
    max_pixels: int = 50,      # max translation in pixels
    max_scale: float = 3,    # ±50% scaling change
    max_angle_deg: float = 30,  # ±45° rotation
    max_shear_deg: float = 10.0,
):
    def normalizer(a_list):
        res = []
        for a in a_list:
            tx, ty, sx, sy, theta, shx, shy = a

            # --- Translation ---
            tx = max(-1.0, min(1.0, float(tx)))
            ty = max(-1.0, min(1.0, float(ty)))
            tx_pix = tx * max_pixels
            ty_pix = ty * max_pixels

            # --- Scale (log-exp mapping) ---
            sx = max(-1.0, min(1.0, float(sx)))
            sy = max(-1.0, min(1.0, float(sy)))
            sx_real = torch.exp(torch.tensor(sx) * torch.log(torch.tensor(max_scale)))
            sy_real = torch.exp(torch.tensor(sy) * torch.log(torch.tensor(max_scale)))

            # --- Rotation (degrees → radians) ---
            theta = float(theta)
            if abs(theta) <= 1.0:
                theta_deg = theta * max_angle_deg
            theta_deg = max(-max_angle_deg, min(max_angle_deg, theta))
            theta_rad = theta_deg * torch.pi / 180.0

            # --- Shear (degrees → radians) ---
            shx = float(shx)
            shy = float(shy)
            if abs(shx) <= 1.0:
                shx_deg = shx * max_shear_deg
            shx_deg = max(-max_shear_deg, min(max_shear_deg, shx))
            if abs(shy) <= 1.0:
                shy_deg = shy * max_shear_deg
            shy_deg = max(-max_shear_deg, min(max_shear_deg, shy))
            shx_rad = shx_deg * torch.pi / 180.0
            shy_rad = shy_deg * torch.pi / 180.0

            res.append((tx_pix, ty_pix, sx_real.item(), sy_real.item(),
                        theta_rad, shx_rad, shy_rad))
        return res

    # Normalize input format
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    if isinstance(affines[0], (int, float)):
        affines = [affines]

    affines = normalizer(affines)
    print(affines)
    # --- Assign to words ---
    affine_tensor = torch.zeros(77, 7)
    dict_affine = {}
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        for ind in inds:
            affine_tensor[ind, :] = rel_fac * torch.FloatTensor(affines[i])
        dict_affine[word] = affines[i]

    return affine_tensor, word_select, dict_affine

def get_shifter(
    text: str,  word_select: Union[int, Tuple[int, ...]], shifts: Union[List[Tuple[float, float]], Tuple[Tuple[float, float], ...]],
    tokenizer, rel_fac, max_pixels
):
    def normalizer(s_list):
        res = []
        for s in s_list:
            sx, sy = s
            sx = max(-1.0, min(1.0, float(sx)))
            sy = max(-1.0, min(1.0, float(sy)))
            dx = sx * max_pixels
            dy = sy * max_pixels
            res.append((dx,dy))
        return res
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    if isinstance(shifts[0], (int, float)):
        shifts = [shifts]  # allow single (dx, dy)

    shifts = normalizer(shifts)
    shifter = torch.zeros(77, 2)
    dict_shift = {}
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        for ind in inds:
            shifter[ind, :] = rel_fac * torch.FloatTensor(shifts[i])
        dict_shift[word] = shifts[i]
    return shifter, word_select, dict_shift

### util functions for FeatandAttnScale
def get_scaler(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    scales: Union[List[float], Tuple[float, ...]],
    tokenizer, rel_fac, max_scale=1.5
):
    def normalizer(s_list, max_scale=max_scale):
        res = []
        for s in s_list:
            s = float(s)
            s = max(-1.0, min(1.0, s))
            res.append(exp(s * log(max_scale)))
        return res

    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    if isinstance(scales, (int, float)):
        scales = [scales]

    scales = normalizer(scales)
    scaler = torch.ones(77)
    dict_scale = {}
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        for ind in inds:
            scaler[ind] = float(rel_fac * scales[i])
        dict_scale[word] = scales[i]
    return scaler, word_select, dict_scale

### util functions for FeatandAttnRotate
def get_rotater(
    text: str,
    word_select: Union[int, Tuple[int, ...]],
    angles: Union[List[float], Tuple[float, ...]],
    tokenizer, rel_fac, max_angle=15.0
):
    def normalizer(angle_list, max_angle=max_angle):
        res = []
        for a in angle_list:
            a = float(a)
            # interpret normalized [-1,1] values as proportion of max_angle
            if abs(a) <= 1.0:
                a = a * max_angle
            a = max(-max_angle, min(max_angle, a))
            res.append(a)
        return res

    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    if isinstance(angles, (int, float)):
        angles = [angles]

    angles = normalizer(angles)
    rotater = torch.ones(77)
    dict_rotate = {}
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        for ind in inds:
            rotater[ind] = float(rel_fac * angles[i])
        dict_rotate[word] = angles[i]
    return rotater, word_select, dict_rotate


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def visualize_cross_attention_edit(attn_b, attn_a, token_idx=[0], title="cross_attn_edit", word="", normalize=True):
    """
    Visualizes how shifting changes cross-attention spatially for a specific token.
    attn_*: shape (B, H*W, T)
    """
    B, HW, T = attn_b.shape
    H = W = int(HW ** 0.5)
    fig, axs = plt.subplots(len(token_idx), 2, figsize=(8, 4))
    for i, tk in enumerate(token_idx):
        attn_before = attn_b[0, :, tk].reshape(H, W)
        attn_after = attn_a[0, :, tk].reshape(H, W)
        
        if normalize:
            attn_before = attn_before / attn_before.max()
            attn_after = attn_after / attn_after.max()
        
        vmax = attn_before.max()
        axs[i,0].imshow(attn_before.detach().cpu(), cmap="magma",vmin=0,vmax=vmax)
        axs[i,0].set_title(f"Attention - {word[i]} Unedited")
        axs[i,0].axis("off")
        
        axs[i,1].imshow(attn_after.detach().cpu(), cmap="magma",vmin=0,vmax=vmax)
        axs[i,1].set_title(f"Attention - {word[i]} Edited")
        axs[i,1].axis("off")
    output_path = "/".join((title.split("/")[:-1]))
    file_n = len(fnmatch.filter(os.listdir(output_path), '*.png'))
    plt.savefig(f"./{title}_{file_n+1}.png")
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_resnet_edit(inp, edited, title="resnet-edit"):
        
        inp_vis_uncond = inp[0].mean(0).detach().cpu()
        edited_vis_uncond = edited[1].mean(0).detach().cpu()
        inp_vis_cond = inp[2].mean(0).detach().cpu()
        edited_vis_cond = edited[3].mean(0).detach().cpu()

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(8, 4))
        fig.suptitle("Edit visualization")

        axes[0,0].imshow(inp_vis_uncond, cmap="magma")
        axes[0,0].set_title("Original uncond")
        axes[0,0].axis("off")

        axes[0,1].imshow(edited_vis_uncond, cmap="magma")
        axes[0,1].set_title("Edited uncond")
        axes[0,1].axis("off")

        axes[1,0].imshow(inp_vis_cond, cmap="magma")
        axes[1,0].set_title("Original cond")
        axes[1,0].axis("off")

        axes[1,1].imshow(edited_vis_cond, cmap="magma")
        axes[1,1].set_title("Edited cond")
        axes[1,1].axis("off")
        output_path = "/".join((title.split("/")[:-1]))
        if not os.path.exists(output_path): os.makedirs(output_path)
        file_n = len(fnmatch.filter(os.listdir(output_path), '*.png'))
        plt.savefig(f"./{title}_{file_n+1}.png")
        plt.tight_layout()
        plt.show()
        plt.close()

def visualize_mask_edit(inp, title="mask-edit"):
        
        inp_vis_uncond = inp[0].mean(0).detach().cpu()
        edited_vis_uncond = inp[1].mean(0).detach().cpu()
        inp_vis_cond = inp[2].mean(0).detach().cpu()
        edited_vis_cond = inp[3].mean(0).detach().cpu()

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(8, 4))
        fig.suptitle("Edit visualization")

        axes[0,0].imshow(inp_vis_uncond, cmap="gray")
        axes[0,0].set_title("Tranformation Mask")
        axes[0,0].axis("off")

        axes[0,1].imshow(edited_vis_uncond, cmap="gray")
        axes[0,1].set_title("Tranformation Mask")
        axes[0,1].axis("off")

        axes[1,0].imshow(inp_vis_cond, cmap="gray")
        axes[1,0].set_title("Tranformation Mask")
        axes[1,0].axis("off")

        axes[1,1].imshow(edited_vis_cond, cmap="gray")
        axes[1,1].set_title("Tranformation Mask")
        axes[1,1].axis("off")
        output_path = "/".join((title.split("/")[:-1]))
        if not os.path.exists(output_path): os.makedirs(output_path)
        file_n = len(fnmatch.filter(os.listdir(output_path), '*.png'))
        plt.savefig(f"./{title}_{file_n+1}.png")
        plt.tight_layout()
        plt.show()
        plt.close()

def visualize_self_attention_edit(attn_b, attn_a, title="self-edit", normalize=False, query_index=None):
    """
    Visualizes self-attention for a specific query location.
    attn_*: shape (Batch, Heads, Query_HW, Key_HW) or (B, HW, HW)
    """
    # Ensure shape is (HW, HW) by removing Batch and Head dims (averaging heads)
    if attn_b.ndim == 4:
        attn_b = attn_b.mean(dim=1)[0]
        attn_a = attn_a.mean(dim=1)[0]
    elif attn_b.ndim == 3:
        attn_b = attn_b[0]
        attn_a = attn_a[0]

    HW = attn_b.shape[0]
    H = W = int(HW ** 0.5)

    # Instead of mean(dim=-1), pick a specific pixel to see what it attends to.
    # If no index is provided, we pick the center pixel.
    if query_index is None:
        query_index = (H // 2) * W + (W // 2)

    # Extract the attention map for that specific query pixel
    map_before = attn_b[query_index].reshape(H, W).detach().cpu()
    map_after = attn_a[query_index].reshape(H, W).detach().cpu()

    # Min-Max Normalization for visibility
    def norm(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    map_before_norm = norm(map_before)
    map_after_norm = norm(map_after)
    
    diff = map_after_norm - map_before_norm

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Before
    im0 = axs[0].imshow(map_before_norm, cmap="magma")
    axs[0].set_title(f"Unedited (Query @ {query_index})")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Plot After
    im1 = axs[1].imshow(map_after_norm, cmap="magma")
    axs[1].set_title("Warped Attention")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Plot Difference
    im2 = axs[2].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    axs[2].set_title("Delta (After - Before)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs: ax.axis("off")

    # Save logic
    output_dir = os.path.dirname(title) if "/" in title else "."
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_n = len(fnmatch.filter(os.listdir(output_dir), '*.png'))
    save_path = f"{title}_{file_n+1}.png"
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


def visualize_qkv_edit(feat_b, feat_a, title="QKV-Edit", normalize=False, norm_type="l2"):
    """
    Visualizes Query, Key, or Value features.
    feat_*: shape (B, HW, C)
    norm_type: "l2" to see spatial energy, "max" for max-activation
    """
    # 1. Setup dimensions
    B, HW, C = feat_b.shape
    H = W = int(HW ** 0.5)

    # 2. Convert to spatial maps
    # We reduce the C dimension to 1D to make it plottable
    if norm_type == "l2":
        # Calculate L2 Norm across the channel dimension
        map_b = torch.norm(feat_b[0], dim=-1).reshape(H, W).detach().cpu()
        map_a = torch.norm(feat_a[0], dim=-1).reshape(H, W).detach().cpu()
    else:
        # Take the mean across channels
        map_b = feat_b[0].mean(dim=-1).reshape(H, W).detach().cpu()
        map_a = feat_a[0].mean(dim=-1).reshape(H, W).detach().cpu()

    # 3. Calculate Difference
    diff = map_a - map_b

    # 4. Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use global Vmax so the comparison is fair
    vmax = max(map_b.max(), map_a.max())
    vmin = min(map_b.min(), map_a.min())

    im0 = axs[0].imshow(map_b, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Original")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(map_a, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Edited")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Difference plot uses a diverging map
    diff_lim = diff.abs().max()
    im2 = axs[2].imshow(diff, cmap="coolwarm", vmin=-diff_lim, vmax=diff_lim)
    axs[2].set_title("Difference (A - B)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.axis("off")
        
    output_dir = os.path.dirname(title) if "/" in title else "."
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_n = len(fnmatch.filter(os.listdir(output_dir), '*.png'))
    save_path = f"{title}_{file_n+1}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

# def visualize_q_k_v_edit(attn_b, attn_a, title="self-edit",normalize=False):
#         """
#         Visualizes how shifting changes self-attention spatially.
#         attn_*: shape (B, H*W, T)
#         """
#         B, HW, _ = attn_b.shape
#         H = W = int(HW ** 0.5)
#         # Collapse over key dimension → per query location
#         attn_before = attn_b.mean(dim=-1)[0].reshape(H, W)
#         attn_after = attn_a.mean(dim=-1)[0].reshape(H, W)

#         if normalize:
#             attn_before = attn_before / attn_before.max()
#             attn_after = attn_after / attn_after.max()

#         diff = attn_after - attn_before
#         diff = diff / (diff.abs().max() + 1e-8)

#         fig, axs = plt.subplots(1, 3, figsize=(12, 4))

#         axs[0].imshow(attn_before.detach().cpu(), cmap="magma")
#         axs[0].set_title("Self Attention Unedited")
#         axs[0].axis("off")

#         axs[1].imshow(attn_after.detach().cpu(), cmap="magma")
#         axs[1].set_title("Self Attention Edited")
#         axs[1].axis("off")

#         axs[2].imshow(diff.detach().cpu(), cmap="coolwarm")
#         axs[2].set_title("Difference (After − Before)")
#         axs[2].axis("off")
#         output_path = "/".join((title.split("/")[:-1]))
#         file_n = len(fnmatch.filter(os.listdir(output_path), '*.png'))
#         plt.savefig(f"./{title}_{file_n+1}.png")
#         plt.tight_layout()
#         plt.show()
#         plt.close()


def visualize_self_attention_region_edit(attn_b, attn_a, mask, title="self-region-edit", normalize=True):
    """
    Visualizes region-averaged self-attention before vs after edit.
    
    attn_* : torch.Tensor, shape (B, H*W, H*W)
        Self-attention maps (before and after).
    mask   : torch.Tensor, shape (H, W)
        Binary mask selecting query pixels of interest (the region whose attention we average).
    """
    B, HW, _ = attn_b.shape
    H = W = int(HW ** 0.5)

    # --- Flatten region mask to index query positions
    mask_flat = mask.flatten().bool().to(attn_b.device)

    # --- Average attention over the chosen region (queries)
    attn_before = attn_b[0, mask_flat].mean(dim=0)  # shape [HW]
    attn_after  = attn_a[0, mask_flat].mean(dim=0)  # shape [HW]

    # --- Reshape back to spatial map
    attn_before = attn_before.reshape(H, W)
    attn_after  = attn_after.reshape(H, W)

    if normalize:
        attn_before = attn_before / (attn_before.max() + 1e-8)
        attn_after  = attn_after / (attn_after.max() + 1e-8)

    # --- Plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Region-Averaged Self-Attention ({title})")

    axs[0].imshow(attn_before.detach().cpu(), cmap="magma")
    axs[0].set_title("Before edit")
    axs[0].axis("off")

    axs[1].imshow(attn_after.detach().cpu(), cmap="magma")
    axs[1].set_title("After edit")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"./{title}.png", bbox_inches="tight")
    plt.show()
    plt.close()


