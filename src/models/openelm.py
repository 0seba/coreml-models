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
        # return qkv
        attention, *_ = attention_monstrosity(
            qkv=qkv,
            mask=mask,
            # mask=None,
            headdim=self.headdim,
            nqheads=self.nqheads,
            nkvheads=self.nkvheads,
            qnorm=self.qnorm,
            knorm=self.knorm,
            rope=self.rope,
            # rope=None,
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
        hidden_states = self.attn(hidden_states, mask=mask)
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

    def __call__(
        self,
        input_ids,
        # hidden_states,
        channels_first=None,
        num_blocks=-1,
    ):
        if channels_first is None:
            channels_first = self.channels_first
        if channels_first:
            axis = 1
        else:
            axis = 2
        # little hack since we have hidden_states that contain feature dimension which we do not want
        # NOTE: For some reason this does not build to CoreML
        # indices = mb.gather(x=hidden_states, indices=0, axis=axis)

        # indices = make_rope_mask_indices(indices)
        indices = make_rope_mask_indices(input_ids)
        mask = self.mask.get_mask(indices)
        self.rope.init_embedding_slices(indices, self.channels_first)
        
        hidden_states = self.embedding(input_ids)
        if channels_first:
            hidden_states = mb.transpose(
                x=hidden_states, perm=[0, 2, 1], name="input_embeddings_channels_first"
            )
        for block in self.blocks[:num_blocks]:
            hidden_states = block(
                hidden_states, mask=mask, channels_first=channels_first
            )
        hidden_states = self.finalnorm(hidden_states, axes=[axis], prefix="final_norm")
        return self.head(hidden_states)
        # return (
        #     # hidden_states,
        #     self.head(hidden_states),
        #     indices,
        #     mask,
        #     self.rope.query_sin_emb,
        #     self.rope.query_cos_emb,
        #     self.rope.key_cos_emb,
        #     self.rope.key_sin_emb,
        # )


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


if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn

    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.converters.mil as mil
    from transformers import AutoModelForCausalLM

    torch_model = AutoModelForCausalLM.from_pretrained(
        "apple/OpenELM-270M", trust_remote_code=True
    )

    precision = "fp16"

    if precision == "fp32":
        mtype = mil.input_types.types.fp32
        compute_precision = ct.precision.FLOAT32
        nptype = np.float32
    elif precision == "fp16":
        mtype = mil.input_types.types.fp16
        compute_precision = ct.precision.FLOAT16
        nptype = np.float16
    else:
        mtype = None
        compute_precision = None
        nptype = None

    channels_first = True
    pre_normalization_and_pos_encoding = False
    multi_query_head = False
    repeat_interleave = False

    coreml_model = from_torch(
        torch_model,
        channels_first=channels_first,
        pre_normalization_and_pos_encoding=pre_normalization_and_pos_encoding,
        multi_query_head=multi_query_head,
        repeat_interleave=repeat_interleave,
        dtype=nptype,
    )

    # shapes = [(1, 1280, seqlen) for seqlen in (32, 128, 256, 512, 1024, 2048)]
    shapes = [(1, seqlen) for seqlen in (32, 128, 256, 512, 1024, 2048)]
    enum_shape = mil.input_types.EnumeratedShapes(shapes=shapes)

    # fixed_shape = (1, 1280, 128)
    fixed_shape = (1, 128)
    # shape = enum_shape.symbolic_shape
    # shape = fixed_shape

    @mb.program(
        input_specs=[
            mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.int32),
            # mb.TensorSpec(fixed_shape, dtype=mil.input_types.types.int32),
            # mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.fp16),
        ],
        # opset_version=mil.builder.AvailableTarget.iOS17,
        opset_version=mil.builder.AvailableTarget.iOS18,
    )
    def coreml_model_progra(input_ids):
        return coreml_model(input_ids, num_blocks=1)
    # def coreml_model_progra(hidden_states):
    #     return coreml_model(hidden_states, num_blocks=-1)

    print(coreml_model_progra)

    # pipeline = ct.PassPipeline.DEFAULT
    # pipeline.remove_passes({"common::add_int16_cast"})
    cml_converted = ct.convert(
        coreml_model_progra,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_precision=ct.precision.FLOAT16,
        compute_precision=compute_precision,
        # minimum_deployment_target=ct.target.iOS17,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="input_ids", shape=ct.EnumeratedShapes(shapes)),
            # ct.TensorType(name="input_ids", shape=fixed_shape),
            # ct.TensorType(name="hidden_states", shape=ct.EnumeratedShapes(shapes)),
            # ct.TensorType(name="input_ids", shape=shape),
        ],
        # pass_pipeline=pipeline,
    )

    print(cml_converted._get_mil_internal())
    name = "openelm_270m_fp16_flex_ios18_mask_rope_dev"
    cml_converted.save(name)
