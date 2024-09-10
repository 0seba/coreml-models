import os
import shutil
from typing import List

import numpy as np
from safetensors import safe_open
import coremltools as ct
from coremltools.converters.mil import Builder as mb, Var
import coremltools.converters.mil as mil
from transformers import AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from ml_dtypes import bfloat16

from models.llama import (
    Block,
    Model,
    Attention,
    qweight_to_lut,
    unpack_int32_to_intk,
    OtherRMSNorm,
)
from attention import RoPEEmbedding, Mask
from layers import (
    RMSNorm,
    FFN2,
    LUTConv,
    QEmbedding,
    QHead,
    LUTLinear,
    GQLinear,
    NamedCall,
    Embedding,
    Head,
)


def safe_get_tensor(tensors, key):
    tensor = tensors.get_tensor(key)
    if tensor.dtype == bfloat16:
        return tensor.astype(np.float32).astype(np.float16)
    return tensor


class Linear(NamedCall):
    def __init__(
        self,
        w: np.ndarray,
        b: np.ndarray | None = None,
        name: None | str = None,
        channels_first=True,
    ):
        super().__init__(name)
        self.w = w
        self.b = b
        self.channels_first = channels_first
        if channels_first:
            self.op = mb.conv
        else:
            self.op = mb.linear

    def __call__(self, x, name=None):
        if self.b is not None:
            x = self.op(x=x, weight=self.w, bias=self.b, name=self.name(name))
        else:
            x = self.op(x=x, weight=self.w, name=self.name(name))
        return x


def expand_dims_for_conv(weight):
    # if len(weight.shape) == 2:
    return np.expand_dims(weight, 2)
    # elif weight.shape[2] != 1:
    #     raise f"Channels first weight with 3rd dim !=1, shape: {weight.shape}"
    # return weight


def convert_qlinear(
    tensors,
    name_prefix,
    bits,
    bias,
    is_quantized,
    op_name=None,
    channels_first=True,
    ngroups=None,
    dtype=np.float32,
):
    if bias:
        bias = safe_get_tensor(tensors, f"{name_prefix}.bias")
    else:
        bias = None
    if is_quantized:
        # w, s, b = (
        w, s, lut = (
            safe_get_tensor(tensors, f"{name_prefix}.weight.weight"),
            safe_get_tensor(tensors, f"{name_prefix}.weight.scales"),
            # safe_get_tensor(tensors, f"{name_prefix}.biases"),
            safe_get_tensor(tensors, f"{name_prefix}.weight.lut"),
        )
        if channels_first:
            w = expand_dims_for_conv(w)
            lut = expand_dims_for_conv(lut)
            s = expand_dims_for_conv(s)
        # w, lut = qweight_to_lut(w, s, b, bits)

        # return GQLinear(
        return LUTLinear(
            w,
            # unpack_int32_to_intk(w, bits),
            # bits,
            lut,
            bits,
            s,
            # b,
            b=bias,
            channels_first=channels_first,
            name=op_name,
        )
    else:
        w = safe_get_tensor(tensors, f"{name_prefix}.weight")
        if channels_first:
            w = expand_dims_for_conv(w)
        return Linear(w, bias, op_name, channels_first=channels_first)
    if channels_first:
        # w = np.expand_dims(w, -1)
        # lut = np.expand_dims(lut, -3)
        return LUTConv(
            w,
            lut,
            b=b,
            name=op_name,
        )
    else:
        return LUTLinear(
            w,
            lut,
            b=b,
            name=op_name,
        )


def convert_rmsnorm(weight, name, channels_first=True, dtype=np.float32):
    if channels_first:
        axis = 1
        weight = np.expand_dims(weight, -1)
    else:
        axis = -1
    eps = np.finfo(dtype).tiny
    return RMSNorm(weight, eps, axes=[axis], name=name)
    # return OtherRMSNorm(weight, name=name)


def convert_ffn(
    tensors, block_index, bits, is_quantized, channels_first=True, dtype=np.float32
):
    linear1 = convert_qlinear(
        tensors,
        f"model.layers.{block_index}.mlp.up_proj",
        bits,
        bias=False,
        channels_first=channels_first,
        is_quantized=is_quantized,
        # op_name=f"block_{block_index}_ffn_in_proj",
        # dtype=dtype,
    )
    linear2 = convert_qlinear(
        tensors,
        f"model.layers.{block_index}.mlp.down_proj",
        bits,
        bias=False,
        channels_first=channels_first,
        is_quantized=is_quantized,
        # op_name=f"block_{block_index}_ffn_in_proj",
        # dtype=dtype,
    )
    linearg = convert_qlinear(
        tensors,
        f"model.layers.{block_index}.mlp.gate_proj",
        bits,
        bias=False,
        channels_first=channels_first,
        is_quantized=is_quantized,
        # op_name=f"block_{block_index}_ffn_in_proj",
        # dtype=dtype,
    )
    # TEMP hardcode activation
    activation = mb.silu
    if channels_first:
        axis = 1
    else:
        axis = -1

    return FFN2(
        linear1, linearg, linear2, activation, True, f"block_{block_index}", axis=axis
    )


def convert_to_mil(
    tensors,
    channels_first,
    config: Qwen2Config,
    nbits,
    headdim,
    max_length=2048,
    dtype=np.float16,
    quantized_emb=False,
    split_head=False,
    batch_size=None,
    unquantized_layers=[],
):
    # bits = config.quantization["bits"]
    bits = nbits
    # headdim = 64

    blocks: List[Block] = []
    rope = RoPEEmbedding(
        headdim,
        max_length,
        max_length,
        freq_constant=config.rope_theta,
        channels_first=channels_first,
        dtype=np.float16,
    )

    for i in range(config.num_hidden_layers):
        quantized = i not in unquantized_layers
        # quantized = False
        attn_norm = convert_rmsnorm(
            safe_get_tensor(tensors, f"model.layers.{i}.input_layernorm.weight"),
            f"layer_{i}_attention_rmsnorm",
            channels_first,
            dtype,
        )
        qbias = safe_get_tensor(tensors, f"model.layers.{i}.self_attn.q_proj.bias")
        kbias = safe_get_tensor(tensors, f"model.layers.{i}.self_attn.k_proj.bias")
        vbias = safe_get_tensor(tensors, f"model.layers.{i}.self_attn.v_proj.bias")
        qkvb = np.concatenate(
            (qbias, kbias, vbias),
            axis=0,
        )
        if quantized:
            qw, qs, qb = (
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.q_proj.weight.weight"
                ),
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.q_proj.weight.scales"
                ),
                # safe_get_tensor(tensors, f"model.layers.{i}.self_attn.q_proj.biases"),
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.q_proj.weight.lut"
                ),
            )
            kw, ks, kb = (
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.k_proj.weight.weight"
                ),
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.k_proj.weight.scales"
                ),
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.k_proj.weight.lut"
                ),
                # safe_get_tensor(tensors, f"model.layers.{i}.self_attn.k_proj.biases"),
            )
            vw, vs, vb = (
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.v_proj.weight.weight"
                ),
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.v_proj.weight.scales"
                ),
                safe_get_tensor(
                    tensors, f"model.layers.{i}.self_attn.v_proj.weight.lut"
                ),
                # safe_get_tensor(tensors, f"model.layers.{i}.self_attn.v_proj.biases"),
            )

            # qw, qproj_lut = qweight_to_lut(qw, qs, qb, bits)
            # kw, kproj_lut = qweight_to_lut(kw, ks, kb, bits)
            # vw, vproj_lut = qweight_to_lut(vw, vs, vb, bits)
            # qkvproj_lut = np.concatenate((qproj_lut, kproj_lut, vproj_lut), axis=0)
            # if channels_first:
            #     # qkvproj_w = np.expand_dims(qkvproj_w, -1)
            #     # qkvproj_lut = np.expand_dims(qkvproj_lut, -3)
            #     qkvproj = LUTConv(
            #         qkvproj_w,
            #         qkvproj_lut,
            #         b=qkvb,
            #         name=f"block_{i}_attention_qkvproj",
            #     )
            # else:
            #     qkvproj = LUTLinear(
            #         qkvproj_w,
            #         qkvproj_lut,
            #         b=qkvb,
            #         name=f"block_{i}_attention_qkvproj",
            #     )

            qkvproj_w = np.concatenate((qw, kw, vw), axis=0)
            # qkvproj_w = unpack_int32_to_intk(qkvproj_w, bits)
            if channels_first:
                qkvproj_w = expand_dims_for_conv(qkvproj_w)

            ss, bb = (
                np.concatenate((qs, ks, vs), axis=0),
                np.concatenate((qb, kb, vb), axis=0),
            )
            if channels_first:
                ss = expand_dims_for_conv(ss)
                bb = expand_dims_for_conv(bb)
            # qkvproj = GQLinear(
            _bits = int(np.log2(bb.shape[-2]))
            assert 2**_bits == bb.shape[-2]
            qkvproj = LUTLinear(
                qkvproj_w,
                lut=bb,
                # qkvproj_lut,
                bits=_bits,
                s=ss,
                b=qkvb,
                channels_first=channels_first,
                name=f"block_{i}_attention_qkvproj",
            )
        else:
            (qw,) = (
                safe_get_tensor(tensors, f"model.layers.{i}.self_attn.q_proj.weight"),
            )
            (kw,) = (
                safe_get_tensor(tensors, f"model.layers.{i}.self_attn.k_proj.weight"),
            )
            (vw,) = (
                safe_get_tensor(tensors, f"model.layers.{i}.self_attn.v_proj.weight"),
            )
            qkvproj_w = np.concatenate((qw, kw, vw), axis=0)
            if channels_first:
                qkvproj_w = expand_dims_for_conv(qkvproj_w)
            qkvproj = Linear(
                qkvproj_w, qkvb, f"block_{i}_attention_qkvproj", channels_first
            )

        outproj = convert_qlinear(
            tensors,
            f"model.layers.{i}.self_attn.o_proj",
            bits,
            bias=False,
            op_name=f"attention_{i}_outproj",
            channels_first=channels_first,
            is_quantized=quantized,
        )
        ffn = convert_ffn(
            tensors, i, bits, channels_first=channels_first, is_quantized=quantized
        )

        new_attn = Attention(
            qkvproj,
            outproj,
            headdim,
            nqheads=config.num_attention_heads,
            nkvheads=config.num_key_value_heads,
            rope=rope,
            channels_first=channels_first,
            block_index=i,
        )

        ffn_norm = convert_rmsnorm(
            safe_get_tensor(
                tensors, f"model.layers.{i}.post_attention_layernorm.weight"
            ),
            f"layer_{i}_ffn_rmsnorm",
            channels_first,
            dtype,
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

    finalnorm = convert_rmsnorm(
        safe_get_tensor(tensors, f"model.norm.weight"),
        f"layer_{i}_ffn_rmsnorm",
        channels_first,
        dtype,
    )

    if quantized_emb:
        # weight, lut = qweight_to_lut(
        #     safe_get_tensor(tensors, "model.embed_tokens.weight"),
        #     safe_get_tensor(tensors, "model.embed_tokens.scales"),
        #     safe_get_tensor(tensors, "model.embed_tokens.biases"),
        #     bits,
        #     as_uint4=False,
        # )
        weight, lut, scales = (
            safe_get_tensor(tensors, "model.embed_tokens.weight.weight"),
            safe_get_tensor(tensors, "model.embed_tokens.weight.lut"),
            safe_get_tensor(tensors, "model.embed_tokens.weight.scales"),
        )
        if bits == 1:
            weight = np.array(weight).astype(mil.mil.types.np_uint1_dtype)
        elif bits == 2:
            weight = np.array(weight).astype(mil.mil.types.np_uint2_dtype)
        elif bits == 3:
            weight = np.array(weight).astype(mil.mil.types.np_uint3_dtype)
        elif bits == 4:
            weight = np.array(weight).astype(mil.mil.types.np_uint4_dtype)
        elif bits == 6:
            weight = np.array(weight).astype(mil.mil.types.np_uint6_dtype)
        emb = QEmbedding(
            weight,
            lut,
            scales=scales,
            nbits=bits,
            channels_first=channels_first,
            name="token_embedding",
        )
        head = QHead(
            weight,
            lut,
            channels_first=channels_first,
            name="lm_head",
            max_size=16536 if split_head else 1_000_000,
        )
    else:
        weight = safe_get_tensor(tensors, "model.embed_tokens.weight")
        # if channels_first:
        #     weight = expand_dims_for_conv(weight)
        # else:
        #     weight = np.transpose(weight)
        emb = Embedding(weight, "token_embedding", channels_first=channels_first)
        head = Head(
            weight,
            split_size=16384 if split_head else 1_000_000,
            channels_first=channels_first,
            prefix="lm_head",
        )

    return Model(
        emb,
        head,
        blocks,
        finalnorm,
        rope,
        Mask(max_length, dtype),
        headdim,
        channels_first,
    )


def convert(
    mil_model: Model,
    seqlen,
    cache_len,
    filename,
    channels_first,
    num_blocks,
    apply_initial_embedding,
    apply_lm_head,
    batch_size,
    headdim,
    model_config,
):
    dtype = mil.input_types.types.fp16
    if channels_first:
        if apply_initial_embedding:
            shape = (batch_size, seqlen)
        else:
            shape = (batch_size, 896, seqlen)
    else:
        if apply_initial_embedding:
            shape = (batch_size, seqlen)
        else:
            shape = (batch_size, seqlen, 896)
    state_spec = state_spec = [
        mb.StateTensorSpec(
            (
                model_config.num_hidden_layers * batch_size,
                model_config.num_key_value_heads,
                cache_len,
                headdim,
            ),
            dtype=mil.input_types.types.fp16,
        ),
        mb.StateTensorSpec(
            (
                model_config.num_hidden_layers * batch_size,
                model_config.num_key_value_heads,
                cache_len,
                headdim,
            ),
            dtype=mil.input_types.types.fp16,
        ),
    ]

    inp_spec = (
        mb.TensorSpec(shape, dtype=mil.input_types.types.int32)
        if apply_initial_embedding
        else mb.TensorSpec(shape, dtype=mil.input_types.types.fp16)
    )

    @mb.program(
        input_specs=[
            inp_spec,
            mb.TensorSpec(
                # (batch_size,), dtype=mil.input_types.types.int32
                (1,),
                dtype=mil.input_types.types.int32,
            ),  # query_pos
            # mb.TensorSpec(
            #     (1,), dtype=mil.input_types.types.int32
            #     # (batch_size,), dtype=mil.input_types.types.int32
            # ),  # query_pos
            *state_spec,
        ],
        # opset_version=mil.builder.AvailableTarget.iOS17,
        opset_version=mil.builder.AvailableTarget.iOS18,
    )
    def program(
        input_ids,
        query_pos1,
        # query_pos2,
        key_cache_state,
        value_cache_state,
    ):
        return mil_model(
            input_ids,
            # [query_pos1, query_pos2],
            [query_pos1],
            states=[key_cache_state, value_cache_state],
            apply_initial_embedding=apply_initial_embedding,
            apply_lm_head=apply_lm_head,
            return_mask_and_pos_emb=False,
            num_blocks=num_blocks,
        )

    print(program)

    pipeline = ct.PassPipeline.DEFAULT
    # TEMP should define a new custom pass
    # pipeline.remove_passes("common::const_elimination")
    pipeline.remove_passes({"common::add_int16_cast"})
    cml_converted = ct.convert(
        program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_units=ct.ComputeUnit.ALL,
        # compute_units=ct.ComputeUnit.CPU_AND_GPU,
        # compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT16,
        # minimum_deployment_target=ct.target.iOS17,
        minimum_deployment_target=ct.target.iOS18,
        # inputs=[
        #     ct.TensorType(name="input_ids", shape=(1, seqlen)),
        #     ct.TensorType(name="query_pos", shape=(1,)),
        # ],
        pass_pipeline=pipeline,
    )

    print("aaah")

    # name = f"qwen0.5b-instruct_shared_emb_head_stateful_inference_ctx_512_query_{qseqlength}.mlpackage"
    if os.path.exists(filename):
        shutil.rmtree(filename)
    try:
        cml_converted.save(filename)
    except Exception as e:
        print(e)
        cml_converted.save(f"F_{filename}")

    try:
        print(cml_converted._get_mil_internal())
    except Exception as e:
        print(e)


def make_embedding_model(mil_model: Model, save_path):
    shapes = [
        (1, seqlen) for seqlen in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    ]
    enum_shape = mil.input_types.EnumeratedShapes(shapes=shapes)

    @mb.program(
        input_specs=[
            mb.TensorSpec(enum_shape.symbolic_shape, dtype=mil.input_types.types.int32)
        ],
        opset_version=mil.builder.AvailableTarget.iOS18,
    )
    def program(input_ids):
        return mil_model.embedding(input_ids, channels_first=False)

    print(program)

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes({"common::add_int16_cast"})
    cml_converted = ct.convert(
        program,
        # compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        # compute_precision=compute_precision,
        # minimum_deployment_target=ct.target.iOS17,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="input_ids", shape=ct.EnumeratedShapes(shapes)),
        ],
        pass_pipeline=pipeline,
    )

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    try:
        cml_converted.save(save_path)
    except Exception as e:
        print(e)
        cml_converted.save(f"F_{save_path}")

    try:
        print(cml_converted._get_mil_internal())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-1.5B"
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Quantize tensors from a safetensors file.")
    parser.add_argument("--gs", required=True, type=int)
    parser.add_argument("--headdim", required=True, type=int)
    parser.add_argument("--qseqlen", required=True, type=int)
    parser.add_argument(
        "--bits", type=int, required=True, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--nblocks", type=int, required=True, help="Number of bits for quantization"
    )

    args = parser.parse_args()

    nbits = args.bits
    gs = args.gs
    # tensors_path = "/Users/sebastianamenabar/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/ff3a49fac17555b8dfc4db6709f480cc8f16a9fe/model.safetensors"
    # tensors_path = "/Users/sebastianamenabar/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
    # tensors_path = "/Users/sebastianamenabar/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a/model.safetensors"
    # nbits = 8
    tensors_path = "/Users/sebastianamenabar/Documents/mydeving/coreml-models/safetensors/Qwen2-1.5B-{}B-GS{}.safetensors"
    model_config: Qwen2Config = AutoConfig.from_pretrained(model_name)
    print(model_config)
    print("Reading tensors")
    tensors = safe_open(tensors_path.format(nbits, gs), framework="numpy")
    channels_first = True
    batch_size = 1
    qseqlen = args.qseqlen
    cache_len = 512
    quantized_emb = False
    split_head = True
    mil_model = convert_to_mil(
        tensors,
        channels_first,
        model_config,
        headdim=args.headdim,
        max_length=cache_len,
        quantized_emb=quantized_emb,
        split_head=split_head,
        nbits=nbits,
        batch_size=batch_size,
        unquantized_layers=(
            [0, model_config.num_hidden_layers - 1]
            if nbits != 16
            else [i for i in range(model_config.num_hidden_layers)]
        ),
        # unquantized_layers=[i for i in range(model_config.num_hidden_layers)]
    )
    num_blocks = args.nblocks
    if num_blocks == -1:
        num_blocks = model_config.num_hidden_layers
    apply_lm_head = True
    apply_initial_embedding = True
    # filename = f"QWEN-1.5B-{nbits}B-GS{gs}-LUT-SCALE-{qseqlen}-QL{'-NO-EMB' if apply_lm_head is False else ''}-{num_blocks}-BLOCKS-C{'F' if channels_first else 'L'}.mlpackage"
    filename = f"QWEN-1.5B-{nbits}B-{qseqlen}-QL{'-NO-EMB' if apply_lm_head is False else ''}-{num_blocks}-BLOCKS-C{'F' if channels_first else 'L'}.mlpackage"
    convert(
        mil_model,
        qseqlen,
        cache_len,
        filename,
        channels_first,
        num_blocks,
        apply_initial_embedding,
        apply_lm_head,
        batch_size,
        headdim=args.headdim,
        model_config=model_config,
        # unquantized_blocks=[0, num_blocks - 1],
    )

    # make_embedding_model(mil_model, "QWEN-05B-I-EMB.mlpackage")
