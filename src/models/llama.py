from typing import List
from collections import OrderedDict

import os
import math
import shutil
import numpy as np
import coremltools.converters.mil as mil
from coremltools.converters.mil import Builder as mb, Var

from attention import (
    RoPEEmbedding,
    attention_monstrosity,
    make_rope_mask_indices,
    Mask,
    slice_update_stateful_attention,
)
from layers import RMSNorm, FFN2, Embedding, Head, Conv, Linear


def unpack_int32_to_intk(arr, k, as_uint4=True):
    # Ensure the input array is int32
    arr = arr.astype(np.int32)
    last_dim = arr.shape[-1]

    n = 32 // k

    # Create an output array with 8 times the length of the input array
    output = np.zeros(arr.shape[:-1] + (last_dim * n,), dtype=np.uint8)

    # Unpack each 4-bit integer
    for i in range(n):
        # Extract 4 bits, shift them to the least significant bits,
        # and mask to keep only the 4 least significant bits
        output[..., i::n] = (arr >> (k * i)) & (2**k - 1)

    if k == 4 and as_uint4:
        output = np.array(output).astype(mil.mil.types.np_uint4_dtype)

    return output


def qweight_to_lut(weights, scales, biases, nbits, as_uint4=True):
    # temp only for 2d weights
    lut = np.arange(0, 2**nbits, dtype=np.float16).reshape(
        1, 1, 2**nbits
    ) * np.expand_dims(np.array(scales), -1) + np.expand_dims(np.array(biases), -1)
    lut = np.expand_dims(lut, -1)
    weights = unpack_int32_to_intk(weights, nbits, as_uint4)
    return weights, lut

class OtherRMSNorm:
    def __init__(self, w, name):
        self.w = w
        self.name = name

    def __call__(self, x):
        sqrtn = np.sqrt(x.shape[-1])
        x = mb.expand_dims(x=x, axes=[-1, -2], name=f"{self.name}_expand")
        x = mb.l2_norm(x=x, name=f"{self.name}_l2_normalize")
        x = mb.squeeze(x=x, axes=[-1, -2], name=f"{self.name}_squeeze")
        x = mb.mul(x=x, y=(self.w * sqrtn).astype(np.float16), name=f"{self.name}_scale")
        return x

class Attention:
    def __init__(
        self,
        qkvproj: Linear,
        outproj: Linear,
        headdim,
        nqheads,
        nkvheads,
        block_index,
        channels_first,
        qnorm: RMSNorm = None,
        knorm: RMSNorm = None,
        rope: RoPEEmbedding = None,
        pre_normalization_and_pos_encoding=None,
        multi_query_head=None,
        repeat_interleave=None,
        index_in_attention_group: int = -1,
        state_implementation="",
        state_update_at="",
    ):
        self.qkvproj = qkvproj
        self.outproj = outproj
        self.headdim = headdim
        self.nqheads = nqheads
        self.nkvheads = nkvheads
        self.qnorm = qnorm
        self.knorm = knorm
        self.rope = rope
        self.channels_first = channels_first
        self.pre_normalization_and_pos_encoding = pre_normalization_and_pos_encoding
        self.multi_query_head = multi_query_head
        self.repeat_interleave = repeat_interleave
        self.block_index = block_index
        self.index_in_attention_group = index_in_attention_group
        self.state_implementation = state_implementation
        self.state_update_at = state_update_at

    def __call__(
        self,
        hidden_states: Var,
        mask=None,
        query_pos=None,
        pos_begin=None,
        pos_end=None,
        key_state=None,
        value_state=None,
        query_cos_emb=None,
        query_sin_emb=None,
    ):
        qkv = self.qkvproj(hidden_states, name=f"attention_{self.block_index}_qkvproj")
        attention, new_kheads, new_vheads, *_ = slice_update_stateful_attention(
            qkv=qkv,
            mask=mask,
            pos_begin=pos_begin,
            pos_end=pos_end,
            key_cache_state_tuple=key_state,
            value_cache_state_tuple=value_state,
            headdim=self.headdim,
            nqheads=self.nqheads,
            nkvheads=self.nkvheads,
            qnorm=self.qnorm,
            knorm=self.knorm,
            query_sin_emb=query_sin_emb,
            query_cos_emb=query_cos_emb,
            channels_first=self.channels_first,
            block_index=self.block_index,
        )
        # return attention, new_kheads, new_vheads
        out = self.outproj(attention, name=f"attention_{self.block_index}_outproj")
        if key_state is None:
            return out
        return out, new_kheads, new_vheads


class Block:
    def __init__(
        self,
        attn_norm: RMSNorm,
        attn: Attention,
        ffn_norm: RMSNorm,
        ffn: FFN2,
        # axis: int,
        channels_first: bool,
        block_index: int,
    ):
        self.attn_norm = attn_norm
        self.attn = attn
        self.ffn_norm = ffn_norm
        self.ffn = ffn
        # self.axis = axis
        self.channels_first = channels_first
        self.block_index = block_index

    def __call__(
        self,
        hidden_states: Var,
        mask=None,
        # axis=None,
        channels_first=None,
        query_pos=None,
        pos_begin=None,
        pos_end=None,
        key_state=None,
        value_state=None,
        query_sin_emb=None,
        query_cos_emb=None,
    ):
        # if axis is None:
        #     axis = self.axis
        if channels_first is None:
            channels_first = self.channels_first
        axis = 1 if channels_first else 2

        residual_1 = hidden_states
        hidden_states_post_attn_norm = self.attn_norm(
            hidden_states,
            prefix=f"block_{self.block_index}_attention",
            axes=[axis],
        )
        hidden_states_post_attn = self.attn(
            hidden_states_post_attn_norm,
            mask=mask,
            query_pos=query_pos,
            pos_begin=pos_begin,
            pos_end=pos_end,
            key_state=key_state,
            value_state=value_state,
            query_cos_emb=query_cos_emb,
            query_sin_emb=query_sin_emb,
        )
        if key_state is not None:
            # return hidden_states_post_attn
            hidden_states_post_attn, new_kheads, new_vheads = hidden_states_post_attn
        hidden_states = mb.add(
            x=residual_1,
            y=hidden_states_post_attn,
            name=f"block_{self.block_index}_residual_1",
        )
        residual_2 = hidden_states
        hidden_states_post_ffn_norm = self.ffn_norm(
            hidden_states, prefix=f"block_{self.block_index}_ffn", axes=[axis]
        )
        hidden_states_post_ffn = self.ffn(
            hidden_states_post_ffn_norm,
            prefix=f"block_{self.block_index}",
            axis=axis,
        )
        hidden_states = mb.add(
            x=hidden_states_post_ffn,
            y=residual_2,
            name=f"block_{self.block_index}_residual_2",
        )

        if key_state is not None:
            return hidden_states, new_kheads, new_vheads
        return hidden_states


class Model:
    def __init__(
        self,
        embedding: Embedding,
        head: Head,
        blocks: List[Block],
        finalnorm: RMSNorm,
        rope: RoPEEmbedding,
        mask: Mask,
        channels_first=True,
        apply_initial_embedding=True,
        apply_lm_head=True,
    ):
        self.embedding = embedding
        self.head = head
        self.blocks = blocks
        self.finalnorm = finalnorm
        self.rope = rope
        self.mask = mask
        self.channels_first = channels_first
        # self.attention_groups = attention_groups

        # self.state_implementation = state_implementation
        # self.state_update_at = state_update_at
        self.apply_initial_embedding = apply_initial_embedding
        self.apply_lm_head = apply_lm_head

    def __call__(
        self,
        input_ids=None,
        query_pos=None,
        hidden_state_state=None,
        query_pos_state=None,
        mask_state=None,
        query_sin_emb_state=None,
        query_cos_emb_state=None,
        mask=None,
        query_sin_emb=None,
        query_cos_emb=None,
        states=None,
        channels_first=None,
        num_blocks=-1,
        apply_initial_embedding=True,
        apply_lm_head=True,
        shift=0,
        propagate_state=False,
        return_mask_and_pos_emb=True,
    ):
        if apply_initial_embedding is None:
            apply_initial_embedding = self.apply_initial_embedding
        if apply_lm_head is None:
            apply_lm_head = apply_lm_head
        if num_blocks == -1:
            num_blocks = len(self.blocks) - shift
        if channels_first is None:
            channels_first = self.channels_first
        if channels_first:
            axis = 1
        else:
            axis = 2

        if mask is None:
            mask = self.mask.get_mask(query_pos, static=False)

        if query_sin_emb is None:
            _axis = 2
            cos_emb = self.rope.cos_emb.reshape(1, 1, -1, 1, 64)
            sin_emb = self.rope.sin_emb.reshape(1, 1, -1, 1, 64)
            query_sin_emb = mb.gather(
                x=sin_emb,
                indices=query_pos,
                axis=_axis,
                batch_dims=1,
                name="query_sin_emb",
            )
            query_cos_emb = mb.gather(
                x=cos_emb,
                indices=query_pos,
                axis=_axis,
                batch_dims=1,
                name="query_cos_emb",
            )

        if apply_initial_embedding:
            hidden_states = self.embedding(input_ids)
        elif input_ids is None:
            hidden_states = mb.read_state(input=hidden_state_state)
        else:
            hidden_states = input_ids

        if not apply_initial_embedding:
            mask = mask if mask else mb.read_state(input=mask_state)
            query_cos_emb = (
                query_cos_emb
                if query_cos_emb
                else mb.read_state(input=query_cos_emb_state)
            )
            query_sin_emb = (
                query_sin_emb
                if query_sin_emb
                else mb.read_state(input=query_sin_emb_state)
            )

        pos_begin = query_pos
        pos_end = mb.add(x=input_ids.shape[1], y=query_pos, name="end_pos")

        batch_size = query_pos.shape[0]
        if batch_size > 1:
            pos_begin = mb.split(x=pos_begin, axis=0, num_splits=batch_size)
            pos_end = mb.split(x=pos_end, axis=0, num_splits=batch_size)
        else:

        key_states, value_states = states
        key_states_read, value_states_read = [
            mb.read_state(input=key_states),
            mb.read_state(input=value_states),
        ]

        key_state = [key_states, key_states_read]
        value_state = [value_states, value_states_read]


        new_kheads, new_vheads = [], []
        block: Block
        all_hidden_states = []
        for i, block in enumerate(self.blocks[shift : num_blocks + shift]):
            hidden_states = block(
                hidden_states,
                mask=mask,
                channels_first=channels_first,
                query_pos=query_pos,
                pos_begin=pos_begin,
                pos_end=pos_end,
                key_state=key_state,
                value_state=value_state,
                query_cos_emb=query_cos_emb,
                query_sin_emb=query_sin_emb,
            )
            if key_state is not None:
                hidden_states, _new_kheads, _new_vheads = hidden_states
                new_kheads.append(_new_kheads)
                new_vheads.append(_new_vheads)

            all_hidden_states.append(hidden_states)

        # return (*all_hidden_states,)  # *new_kheads, *new_vheads

        _hidden_states = self.finalnorm(hidden_states, axes=[axis], prefix="final_norm")
        if not apply_lm_head:

            out = [_hidden_states]
            if return_mask_and_pos_emb:
                out += [mask, query_sin_emb, query_cos_emb]

            return out

        out = self.head(_hidden_states)
        return out
