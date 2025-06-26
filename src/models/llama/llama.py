from typing import List, Optional
from dataclasses import dataclass

import math
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from layers import LUTLinear, RMSNorm
from ops import (
    update_cache,
    gqa_attention,
    apply_rotary_pos_emb,
    build_causal_mask,
    gather_static,
)


class LlamaRMSNorm(RMSNorm):
    pass


@dataclass
class AttentionArgs:
    attention_mask: any
    sin_emb: any
    cos_emb: any
    kv_cache_write_idx_begin: any
    kv_cache_write_idx_end: any
    key_cache: any
    value_cache: any
    key_state: any
    value_state: any


class LlamaAttentionLayer:
    def __init__(
        self,
        q_proj: LUTLinear,
        k_proj: LUTLinear,
        v_proj: LUTLinear,
        o_proj: LUTLinear,
        num_query_heads: int,
        num_kv_heads: int,
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

    def __call__(
        self,
        hidden_states,
        attention_args: AttentionArgs,
        prefix: str,
        kv_cache_layer_write_idx: int,
    ):
        batch_size, hidden_dim, _, seqlen = hidden_states.shape
        # cache_seqlen = attention_args.key_cache.shape[-2]

        q_proj = self.q_proj(hidden_states, name=prefix + "q_proj_")
        headdim = q_proj.shape[1] // self.num_query_heads
        q = mb.reshape(
            x=q_proj,
            shape=[
                batch_size,
                self.num_query_heads,
                headdim,
                # seqlen,
                -1,
            ],
            name=f"{prefix}q_reshape",
        )
        q = mb.transpose(x=q, perm=[0, 1, 3, 2], name=f"{prefix}q_transpose")
        q_rot = apply_rotary_pos_emb(
            q,
            attention_args.sin_emb,
            attention_args.cos_emb,
            axis=3,
            prefix=prefix + "query_",
        )
        # return q_proj, q_rot

        k_proj = self.k_proj(hidden_states, name=prefix + "k_proj_")
        k = mb.reshape(
            x=k_proj,
            shape=[
                batch_size,
                self.num_kv_heads,
                headdim,
                # seqlen,
                -1,
            ],
            name=f"{prefix}k_reshape",
        )
        k = mb.transpose(x=k, perm=[0, 1, 3, 2], name=f"{prefix}k_transpose")
        k_rot = apply_rotary_pos_emb(
            k,
            attention_args.sin_emb,
            attention_args.cos_emb,
            axis=3,
            prefix=prefix + "key_",
        )
        k = update_cache(
            k_rot,
            attention_args.key_cache,
            attention_args.key_state,
            attention_args.kv_cache_write_idx_begin,
            attention_args.kv_cache_write_idx_end,
            kv_cache_layer_write_idx,
            prefix + "key_",
        )
        attention_args.key_cache = k
        k = mb.slice_by_index(
            x=k,
            begin=[kv_cache_layer_write_idx, 0, 0, 0],
            end=[
                kv_cache_layer_write_idx + batch_size,
                -1,
                -1,
                -1,
            ],
            begin_mask=[False, True, True, True],
            end_mask=[False, True, True, True],
            name=f"{prefix}key_slice_by_index",
        )

        v_proj = self.v_proj(hidden_states, name=prefix + "v_proj_")
        v = mb.reshape(
            x=v_proj,
            shape=[
                batch_size,
                self.num_kv_heads,
                headdim,
                # seqlen,
                -1,
            ],
            name=f"{prefix}v_reshape",
        )
        v = mb.transpose(x=v, perm=[0, 1, 3, 2], name=f"{prefix}v_transpose")
        v = update_cache(
            v,
            attention_args.value_cache,
            attention_args.value_state,
            attention_args.kv_cache_write_idx_begin,
            attention_args.kv_cache_write_idx_end,
            kv_cache_layer_write_idx,
            prefix + "value_",
        )
        attention_args.value_cache = v
        v = mb.slice_by_index(
            x=v,
            begin=[kv_cache_layer_write_idx, 0, 0, 0],
            end=[
                kv_cache_layer_write_idx + batch_size,
                -1,
                -1,
                -1,
            ],
            begin_mask=[False, True, True, True],
            end_mask=[False, True, True, True],
            name=f"{prefix}value_slice_by_index",
        )

        attention = gqa_attention(
            q_rot,
            k,
            v,
            prefix=prefix,
            scaling=np.array([1 / math.sqrt(headdim)], dtype=np.float16),
            attention_mask=attention_args.attention_mask,
        )
        attention = mb.concat(
            values=attention, axis=1, name=prefix + "attention_concat"
        )
        # attention = mb.transpose(
        #     x=attention, perm=[0, 1, 3, 2], name=prefix + "attention_transpose"
        # )
        attention = mb.reshape(
            x=attention,
            shape=[batch_size, self.num_query_heads * headdim, 1, -1],
            name=prefix + "attention_reshape",
        )
        output = self.o_proj(attention, name=prefix + "o_proj_")

        return output,


class LlamaMLP:
    def __init__(
        self,
        up_proj: LUTLinear,
        gate_proj: LUTLinear,
        down_proj: LUTLinear,
        # axis=1,
    ):
        self.up_proj = up_proj
        self.gate_proj = gate_proj
        self.down_proj = down_proj
        # self.axis = axis

    def __call__(self, hidden_states, prefix: str):
        g = mb.silu(
            x=self.gate_proj(hidden_states, name=f"{prefix}mlp_gate_proj"),
            name=f"{prefix}mlp_gate_activation",
        )
        up = self.up_proj(hidden_states, name=f"{prefix}mlp_up_proj")
        up = mb.mul(x=g, y=up, name=f"{prefix}mlp_gated_intermediate")
        down = self.down_proj(up, name=f"{prefix}mlp_down_proj")
        return down


class LlamaDecoderLayer:
    def __init__(
        self,
        input_layernorm: LlamaRMSNorm,
        attention: LlamaAttentionLayer,
        post_attention_layernorm: LlamaRMSNorm,
        ffn: LlamaMLP,
    ):
        self.input_layernorm = input_layernorm
        self.attention = attention
        self.post_attention_layernorm = post_attention_layernorm
        self.ffn = ffn

    def __call__(
        self,
        hidden_states,
        attention_args: AttentionArgs,
        prefix: str,
        kv_cache_layer_write_idx: int,
    ):
        residual = hidden_states
        # debug_iln = self.input_layernorm(
        #     hidden_states, prefix="debug_" + prefix + "input_norm_"
        # )
        # debug_attn, *other_attn_outputs = self.attention(
        #     hidden_states,
        #     attention_args=attention_args,
        #     prefix="debug_" + prefix + "attention_",
        #     kv_cache_layer_write_idx=kv_cache_layer_write_idx,
        # )
        # return (
        #     debug_attn,
        #     *other_attn_outputs,
        # )

        # debug_mlp = self.ffn(hidden_states, prefix="debug" + prefix)
        # debug_post_attn = self.post_attention_layernorm(
        #     hidden_states, prefix="debug_" + prefix + "post_attention_norm_"
        # )

        hidden_states = self.input_layernorm(
            hidden_states, prefix=prefix + "input_norm_"
        )
        hidden_states, *_ = self.attention(
            hidden_states,
            attention_args=attention_args,
            prefix=prefix,
            kv_cache_layer_write_idx=kv_cache_layer_write_idx,
        )
        hidden_states = mb.add(x=residual, y=hidden_states, name=f"{prefix}residual_1")
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states, prefix=prefix + "post_attention_norm_"
        )
        hidden_states = self.ffn(hidden_states, prefix=prefix)
        hidden_states = mb.add(x=residual, y=hidden_states, name=f"{prefix}residual_2")
        return hidden_states


class LlamaModel:
    def __init__(
        self,
        blocks: List[LlamaDecoderLayer],
        max_sequence_length=8192,
        sin_emb: Optional = None,
        cos_emb: Optional = None,
        layer_from: int = 0,
    ):
        self.blocks = blocks
        self.max_sequence_length = max_sequence_length
        self.causal_mask = build_causal_mask(max_sequence_length)
        self.sin_emb = sin_emb
        self.cos_emb = cos_emb
        self.layer_from = layer_from

    def __call__(
        self,
        hidden_states,
        positions,
        kv_write_idx=None,
        attention_mask: Optional = None,
        key_cache=None,
        value_cache=None,
        sin_emb: Optional = None,
        cos_emb: Optional = None,
        # attention_args: AttentionArgs,
    ):
        if attention_mask is None:
            attention_mask = gather_static(
                positions,
                self.causal_mask[: key_cache.shape[2], : key_cache.shape[2]],
                "attention_mask_",
                transpose=True,
            )
        if sin_emb is None:
            sin_emb = gather_static(positions, self.sin_emb, "sin_emb_")
        if cos_emb is None:
            cos_emb = gather_static(positions, self.cos_emb, "cos_emb_")

        # read_key_cache = None
        # read_value_cache = None
        read_key_cache = mb.read_state(input=key_cache)
        read_value_cache = mb.read_state(input=value_cache)
        seqlen = hidden_states.shape[3]
        if is_symbolic(seqlen):
            shape = mb.shape(x=hidden_states, name="input_shape")
            seqlen = mb.gather(x=shape, indices=3, name="sequence_length")
        # kv_write_idx_end = None
        kv_write_idx_end = mb.add(x=kv_write_idx, y=seqlen, name="kv_write_idx_end")
        attention_args = AttentionArgs(
            attention_mask,
            sin_emb,
            cos_emb,
            kv_write_idx,
            kv_write_idx_end,
            read_key_cache,
            read_value_cache,
            key_cache,
            value_cache,
        )

        block: LlamaDecoderLayer
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attention_args=attention_args,
                prefix=f"layer_{i + self.layer_from}_",
                kv_cache_layer_write_idx=(i + self.layer_from) * hidden_states.shape[0],
            )
        return hidden_states


def convert(model: LlamaModel):
    pass
