from typing import List

import numpy as np
from coremltools.converters.mil import Builder as mb, Var

from attention import RoPEEmbedding, attention_monstrosity, make_rope_mask_indices, Mask
from layers import RMSNorm, FFN, Embedding, Head, Conv, Linear


class Attention:
    def __init__(
        self,
        qkvproj: Linear,
        outproj: Linear,
        headdim,
        nqheads,
        nkvheads,
        qnorm: RMSNorm,
        knorm: RMSNorm,
        rope: RoPEEmbedding,
        channels_first,
        pre_normalization_and_pos_encoding,
        multi_query_head,
        repeat_interleave,
        block_index,
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

    def __call__(self, hidden_states: Var, mask=None):
        qkv = self.qkvproj(hidden_states, name=f"attention_{self.block_index}_qkvproj")
        attention, *_ = attention_monstrosity(
            qkv=qkv,
            mask=mask,
            headdim=self.headdim,
            nqheads=self.nqheads,
            nkvheads=self.nkvheads,
            qnorm=self.qnorm,
            knorm=self.knorm,
            rope=self.rope,
            channels_first=self.channels_first,
            pre_normalization_and_pos_encoding=self.pre_normalization_and_pos_encoding,
            multi_query_head=self.multi_query_head,
            repeat_interleave=self.repeat_interleave,
            block_index=self.block_index,
        )
        out = self.outproj(attention, name=f"attention_{self.block_index}_outproj")
        return out


class Block:
    def __init__(
        self,
        attn_norm: RMSNorm,
        attn: Attention,
        ffn_norm: RMSNorm,
        ffn: FFN,
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
    ):
        # if axis is None:
        #     axis = self.axis
        axis = 1 if channels_first else 2

        residual = hidden_states
        hidden_states = self.attn_norm(
            hidden_states,
            prefix=f"block_{self.block_index}_attention",
            axes=[axis],
        )
        hidden_states = self.attn(hidden_states, mask)
        hidden_states = mb.add(
            x=residual, y=hidden_states, name=f"block_{self.block_index}_residual_1"
        )
        residual = hidden_states
        hidden_states = self.ffn_norm(
            hidden_states, prefix=f"block_{self.block_index}_ffn", axes=[axis]
        )
        hidden_states = self.ffn(
            hidden_states, prefix=f"block_{self.block_index}", axis=axis
        )
        hidden_states = mb.add(
            x=hidden_states, y=residual, name=f"block_{self.block_index}_residual_2"
        )
        return hidden_states


class OpenELM:
    def __init__(
        self,
        embedding: Embedding,
        head: Head,
        blocks: List[Block],
        finalnorm: RMSNorm,
        rope: RoPEEmbedding,
        mask: Mask,
        channels_first=True,
    ):
        self.embedding = embedding
        self.head = head
        self.blocks = blocks
        self.finalnorm = finalnorm
        self.rope = rope
        self.mask = mask
        self.channels_first = channels_first

    def __call__(self, input_ids, channels_first=None):
        if channels_first is None:
            channels_first = self.channels_first
        if channels_first:
            axis = 1
        else:
            axis = 2
        indices = make_rope_mask_indices(input_ids)
        self.rope.init_embedding_slices(indices, self.channels_first)
        mask = self.mask.get_mask(indices)
        hidden_states = self.embedding(input_ids)
        if channels_first:
            hidden_states = mb.transpose(
                x=hidden_states, perm=[0, 2, 1], name="input_embeddings_channels_first"
            )
        for block in self.blocks:
            hidden_states = block(
                hidden_states, mask=mask, channels_first=channels_first
            )
        hidden_states = self.finalnorm(hidden_states, axes=[axis], prefix="final_norm")
        return self.head(hidden_states)


def convert_linear(layer, name=None, channels_first=True, dtype=np.float32):
    weight = layer.weight.detach().numpy().astype(dtype)
    if channels_first:
        weight = np.expand_dims(weight, -1)
        return Conv(weight, name=name)
    else:
        return Linear(weight, name=name)


def convert_rmsnorm(layer, name, channels_first=True, dtype=np.float32):
    weight = layer.weight.detach().numpy().astype(dtype)
    if channels_first:
        axis = 1
        weight = np.expand_dims(weight, -1)
    else:
        axis = -1
    return RMSNorm(weight, layer.eps, axes=[axis], name=name)


def convert_ffn(layer, block_index, name, channels_first=True, dtype=np.float32):
    linear1 = convert_linear(
        layer.proj_1,
        channels_first=channels_first,
        name=f"block_{block_index}_ffn_in_proj",
        dtype=dtype,
    )
    linear2 = convert_linear(
        layer.proj_2,
        channels_first=channels_first,
        name=f"block_{block_index}_ffn_out_proj",
        dtype=dtype,
    )
    # TEMP hardcode activation
    activation = mb.silu
    if channels_first:
        axis = 1
    else:
        axis = -1

    return FFN(linear1, linear2, activation, layer.ffn_with_glu, block_index, axis=axis)


def from_torch(
    pytorch_model,
    channels_first,
    pre_normalization_and_pos_encoding,
    multi_query_head,
    repeat_interleave,
    dtype=np.float32,
):
    axis = 1 if channels_first else 2
    blocks: List[Block] = []
    # ropes = []
    rope = RoPEEmbedding(64, 2048, 2048, channels_first=channels_first, dtype=dtype)
    for i, layer in enumerate(pytorch_model.transformer.layers):
        attn = layer.attn
        hdim = layer.attn.head_dim

        attn_norm = convert_rmsnorm(
            layer.attn_norm, f"layer_{i}_attention_rmsnorm", channels_first, dtype
        )
        qnorm = convert_rmsnorm(
            layer.attn.q_norm, f"layer_{i}_attention_q_rmsnorm", channels_first, dtype
        )
        knorm = convert_rmsnorm(
            layer.attn.k_norm, f"layer_{i}_attention_k_rmsnorm", channels_first, dtype
        )

        # q, k, v = np.split(
        #     attn.qkv_proj.weight.detach().numpy(),
        #     indices_or_sections=[
        #         attn.head_dim * attn.num_q_heads,
        #         attn.head_dim * attn.num_q_heads + attn.head_dim * attn.num_k_heads,
        #         # attn.head_dim * attn.num_v_heads,
        #     ],
        #     axis=0,
        # )

        # qproj = Linear(q)
        # kproj = Linear(k)
        # vproj = Linear(v)
        qkvproj = convert_linear(
            attn.qkv_proj, f"block_{i}_attention_qkvproj", channels_first, dtype
        )
        outproj = convert_linear(
            layer.attn.out_proj, f"attention_{i}_outproj", channels_first, dtype
        )
        # if hdim not in ropes:
        #     ropes[hdim] = RoPEEmbedding(hdim, 2048, 2048, dtype=dtype)
        new_attn = Attention(
            qkvproj,
            outproj,
            hdim,
            layer.attn.num_q_heads,
            layer.attn.num_k_heads,
            qnorm,
            knorm,
            rope,
            channels_first,
            pre_normalization_and_pos_encoding,
            multi_query_head,
            repeat_interleave,
            block_index=i,
        )
        ffn = convert_ffn(layer.ffn, i, f"layer_{i}_ffn", channels_first, dtype)
        ffn_norm = convert_rmsnorm(
            layer.ffn_norm, f"layer_{i}_ffn_rmsnorm", channels_first, dtype
        )

        block = Block(
            attn_norm,
            new_attn,
            ffn_norm,
            ffn,
            channels_first,
            i,
        )

        blocks.append(block)

    embs = (
        pytorch_model.transformer.token_embeddings.weight.detach().numpy().astype(dtype)
    )

    tokemb = Embedding(
        embs,
        name="token_embedding",
    )
    head = Head(
        embs,
        # nsplits=2,
        channels_first=channels_first,
        split_size=16_000,
        prefix="lm_head",
    )
    finalnorm = convert_rmsnorm(
        pytorch_model.transformer.norm, "final_rmsnorm", channels_first, dtype
    )

    model = OpenELM(
        tokemb, head, blocks, finalnorm, rope, Mask(2048, dtype), channels_first
    )

    return model
