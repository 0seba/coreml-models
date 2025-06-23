from typing import List, Optional
from dataclasses import dataclass


import math
import numpy as np
from coremltools.converters.mil import Builder as mb


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
        cache_seqlen = attention_args.key_cache.shape[-2]

        q = self.q_proj(hidden_states, name=prefix + "q_proj_")
        print(q.shape)
        headdim = q.shape[1] // self.num_query_heads
        q = mb.reshape(
            x=q,
            shape=[
                batch_size,
                self.num_query_heads,
                headdim,
                seqlen,
            ],
            name=f"{prefix}q_reshape",
        )
        print(q.shape)
        q = mb.transpose(x=q, perm=[0, 1, 3, 2], name=f"{prefix}q_transpose")
        print(q.shape)
        q = apply_rotary_pos_emb(
            q, attention_args.sin_emb, attention_args.cos_emb, axis=3, prefix=prefix
        )
        print(q.shape)

        k = self.k_proj(hidden_states, name=prefix + "k_proj_")
        k = mb.reshape(
            x=k,
            shape=[
                batch_size,
                self.num_kv_heads,
                headdim,
                seqlen,
            ],
            name=f"{prefix}k_reshape",
        )
        k = mb.transpose(x=k, perm=[0, 1, 3, 2], name=f"{prefix}k_transpose")
        k = apply_rotary_pos_emb(
            k, attention_args.sin_emb, attention_args.cos_emb, axis=3, prefix=prefix
        )
        k = update_cache(
            k,
            attention_args.key_cache,
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
                self.num_kv_heads,
                cache_seqlen,
                headdim,
            ],
            name=f"{prefix}key_slice_by_index",
        )

        v = self.v_proj(hidden_states, name=prefix + "v_proj_")
        v = mb.reshape(
            x=v,
            shape=[
                batch_size,
                self.num_kv_heads,
                headdim,
                seqlen,
            ],
            name=f"{prefix}v_reshape",
        )
        v = mb.transpose(x=v, perm=[0, 1, 3, 2], name=f"{prefix}v_transpose")
        v = update_cache(
            v,
            attention_args.value_cache,
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
                self.num_kv_heads,
                cache_seqlen,
                headdim,
            ],
            name=f"{prefix}value_slice_by_index",
        )

        attention = gqa_attention(
            q,
            k,
            v,
            prefix=prefix,
            scaling=np.array([1 / math.sqrt(headdim)], dtype=np.float16),
            attention_mask=attention_args.attention_mask,
        )
        attention = mb.concat(
            values=attention, axis=1, name=prefix + "attention_concat"
        )
        print(attention.shape)
        output = self.o_proj(attention, name=prefix + "o_proj_")

        return output


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
        hidden_states = self.input_layernorm(
            hidden_states, prefix=prefix + "input_norm_"
        )
        hidden_states = self.attention(
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
    ):
        self.blocks = blocks
        self.max_sequence_length = max_sequence_length
        self.causal_mask = build_causal_mask(max_sequence_length)
        self.sin_emb = sin_emb
        self.cos_emb = cos_emb

    def __call__(
        self,
        hidden_states,
        kv_write_idx,
        positions,
        key_cache,
        value_cache,
        attention_mask: Optional = None,
        sin_emb: Optional = None,
        cos_emb: Optional = None,
        # attention_args: AttentionArgs,
    ):
        seqlen = hidden_states.shape[3]
        kv_write_idx_end = mb.add(x=kv_write_idx, y=seqlen, name="kv_write_idx_end")
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

        attention_args = AttentionArgs(
            attention_mask,
            sin_emb,
            cos_emb,
            kv_write_idx,
            kv_write_idx_end,
            mb.read_state(input=key_cache),
            mb.read_state(input=value_cache),
        )

        block: LlamaDecoderLayer
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attention_args=attention_args,
                prefix=f"layer_{i}_",
                kv_cache_layer_write_idx=i * hidden_states.shape[0],
            )
        return hidden_states


def convert(model: LlamaModel):
    pass
