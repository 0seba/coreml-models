import os
import json
from typing import Dict, Any
from argparse import ArgumentParser

import numpy as np
import torch
from torch import Tensor
from safetensors.torch import load_file  # use torch safetensors to load bfloat16

import coremltools as ct
import coremltools.converters.mil as mil
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipelineManager

from layers import LUTLinear
from positional_encodings import compute_default_rope_parameters, compute_rope_embedding
from utils.quantization import unpack_weights
from utils.coreml_utils import print_compute_plan_sync
from .falcon_edge_bitnet import (
    FalconEdgeRMSNorm,
    FalconEdgeAttentionLayer,
    FalconEdgeBitnetDecoderLayer,
    FalconEdgeMLP,
    FalconEdgeBitnetModel,
)
from sampling.min_p import min_p

from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET
from custom_conv import conv

register_op(conv, opset_version=_IOS17_TARGET, allow_override=True)


Tensors = Dict[str, Tensor]
NORM_EPS = np.finfo(np.float16).tiny


def convert_rmsnorm(tensors: Tensors, prefix: str):
    weight = (
        tensors[f"{prefix}.weight"].float().half().unsqueeze(-1).unsqueeze(-1).numpy()
    )
    return FalconEdgeRMSNorm(weight, axes=(1,), eps=NORM_EPS)


def convert_linear(tensors: Tensors, prefix: str, use_quantized_lut=False):
    # TODO investigate if using vector quantization with 4 bit indices and
    # vectors of dim 2 is faster
    # ANS: vector quantization is not compatible with ANE
    w = unpack_weights(tensors[f"{prefix}.weight"].numpy(), dtype=np.int8)
    # w is a int array with values {-1, 0, 1}, transform them into indices
    w = w + 1
    w = np.expand_dims(w, axis=(-1, -2))
    # and now we build the lookup table
    scale = (
        (1 / tensors[f"{prefix}.weight_scale"].float().numpy())
        .astype(np.float16)
        .squeeze()
    )

    if use_quantized_lut:
        s = scale
        lut = np.array([-1, 0, 1, 2**7 - 1], dtype=np.int8)
    else:
        s = None
        lut = np.array([-scale, 0, scale, np.finfo(np.float16).max], dtype=np.float16)
    lut = lut.reshape((1, 1, 1, 1, 4, 1))

    return LUTLinear(w=w, lut=lut, bits=2, s=s)


def convert_attention(
    tensors: Tensors, prefix: str, num_query_heads: int, num_key_value_heads: int
):
    q_proj = convert_linear(tensors, prefix + ".q_proj")
    k_proj = convert_linear(tensors, prefix + ".k_proj")
    v_proj = convert_linear(tensors, prefix + ".v_proj")
    o_proj = convert_linear(tensors, prefix + ".o_proj")

    return FalconEdgeAttentionLayer(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_query_heads,
        num_key_value_heads,
    )


def convert_mlp(tensors: Tensors, prefix: str):
    up_proj = convert_linear(tensors, prefix + ".up_proj")
    gate_proj = convert_linear(tensors, prefix + ".gate_proj")
    down_proj = convert_linear(tensors, prefix + ".down_proj")

    return FalconEdgeMLP(up_proj, gate_proj, down_proj)


def build_model_from_safetensors(
    tensors: Tensors,
    config: Dict[str, Any],
    layer_from: int = 0,
    layer_to: int = -1,
    max_sequence_length: int = 2048,
):
    if layer_to == -1:
        layer_to = config["num_hidden_layers"]
    decoder_layers = []
    for i in range(layer_from, layer_to):
        input_layernorm = convert_rmsnorm(
            tensors, prefix=f"model.layers.{i}.input_layernorm"
        )
        attention = convert_attention(
            tensors,
            f"model.layers.{i}.self_attn",
            num_query_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
        )
        post_attention_layernorm = convert_rmsnorm(
            tensors, prefix=f"model.layers.{i}.post_attention_layernorm"
        )
        ffn = convert_mlp(tensors, prefix=f"model.layers.{i}.mlp")
        decoder_layer = FalconEdgeBitnetDecoderLayer(
            input_layernorm,
            attention,
            post_attention_layernorm,
            ffn,
        )
        decoder_layers.append(decoder_layer)

    inv_freq, attention_factor = compute_default_rope_parameters(
        config["head_dim"], config["rope_theta"]
    )
    sin_emb, cos_emb = compute_rope_embedding(
        inv_freq, attention_factor, max_sequence_length
    )
    model = FalconEdgeBitnetModel(
        decoder_layers,
        sin_emb=sin_emb.astype(np.float16),
        cos_emb=cos_emb.astype(np.float16),
        layer_from=layer_from,
    )
    return model, config


def convert(
    model: FalconEdgeBitnetModel,
    config: Dict[str, Any],
    batch_size=1,
    seq_len=32,
    cache_len=1024,
    package_dir=None,
    skip_model_load=False,
):
    headdim = config["head_dim"]
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    cache_length_sym = get_new_symbol()

    state_spec = [
        mb.StateTensorSpec(
            (
                num_hidden_layers * batch_size,
                config["num_key_value_heads"],
                cache_length_sym,
                headdim,
            ),
            dtype=mil.input_types.types.fp16,
        ),
        mb.StateTensorSpec(
            (
                num_hidden_layers * batch_size,
                config["num_key_value_heads"],
                cache_length_sym,
                headdim,
            ),
            dtype=mil.input_types.types.fp16,
        ),
    ]

    # lengths = [1, 8, 32, 64, 68, 72, 76, 80, 84,88, 92, 96, 128, 160]
    lengths = [1, 8, 16, 32, 48, 64, 128]
    hidden_states_input_shapes = [
        (batch_size, hidden_size, 1, seq_len) for seq_len in lengths
    ]
    positions_input_shapes = [(batch_size, seq_len) for seq_len in lengths]
    length_sym = get_new_symbol()

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                # mil.input_types.EnumeratedShapes(
                #     shapes=hidden_states_input_shapes
                # ).symbolic_shape,
                (batch_size, hidden_size, 1, length_sym),
                # (batch_size, hidden_size, 1, seq_len),
                dtype=mil.input_types.types.fp16,
            ),
            mb.TensorSpec(
                (batch_size, length_sym),
                # (batch_size, seq_len),
                dtype=mil.input_types.types.int32,
            ),
            mb.TensorSpec(
                (1,),
                dtype=mil.input_types.types.int32,
            ),
            mb.TensorSpec(
                (batch_size, 1, length_sym, cache_length_sym),
                dtype=mil.input_types.types.fp16,
            ),
            *state_spec,
        ],
        opset_version=mil.builder.AvailableTarget.iOS18,
    )
    # def program(hidden_states, kv_write_idx, positions, key_cache, value_cache):
    def program(
        hidden_states,
        positions,
        kv_write_idx,
        attention_mask,
        key_cache,
        value_cache,
    ):
        attention_mask = mb.transpose(x=attention_mask, perm=[0, 1, 3, 2], name="attention_mask_transposed")
        return model(
            hidden_states,
            positions,
            kv_write_idx,
            attention_mask,
            key_cache,
            value_cache,
        )

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
    pipeline.set_options(
        "common::materialize_symbolic_shape_program",
        {
            "function_name_to_materialization_map": {
                # As an example, let us assume the input is x (is0, is1, 1024)
                f"model_input_{l}_cache_{cache_length}": {
                    "hidden_states": (batch_size, hidden_size, 1, l),
                    "kv_write_idx": (batch_size,),
                    "positions": (batch_size, l),
                    "attention_mask": (batch_size, 1, l, cache_length),
                    "key_cache": (
                        num_hidden_layers * batch_size,
                        config["num_key_value_heads"],
                        cache_length,
                        headdim,
                    ),
                    "value_cache": (
                        num_hidden_layers * batch_size,
                        config["num_key_value_heads"],
                        cache_length,
                        headdim,
                    ),
                }
                for l in lengths
                for cache_length in [
                    512,
                    1024,
                    2048,
                    2048 + 1024,
                    4096,
                    4096 + 2048,
                    8192,
                ]
            }
        },
    )

    PassPipelineManager.apply_pipeline(program, pipeline)
    program.export_as_multifunction = True
    program.skip_all_passes = True
    # program.functions["flex"] = program.functions["main"]
    program.functions["main"] = program.functions["model_input_1_cache_1024"]
    # del program.functions["main"]
    # program.default_function_name = "model_length_1"

    mlmodel = ct.convert(
        program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=skip_model_load,
        # inputs=[
        #     ct.TensorType(
        #         shape=ct.EnumeratedShapes(hidden_states_input_shapes),
        #         name="hidden_states",
        #     ),
        #     ct.TensorType(shape=(1,), name="kv_write_idx"),
        #     ct.TensorType(
        #         shape=ct.EnumeratedShapes(positions_input_shapes), name="positions"
        #     ),
        # ],
        # pass_pipeline=pipeline,
        package_dir=package_dir,
    )

    return mlmodel


def convert_lm_head(
    tensors: Tensors,
    chunk_size=16_384,
    batch_size=1,
    hidden_size=2048,
    package_dir=None,
):
    final_rms_norm = convert_rmsnorm(tensors, "model.norm")
    ws = (
        tensors["lm_head.weight"]
        .unsqueeze(-1)
        .unsqueeze(-1)
        .float()
        .half()
        .split(chunk_size, dim=0)
    )
    ws = [w.numpy() for w in ws]
    lengths = [1, 8, 16, 32, 48, 64, 128]
    hidden_states_input_shapes = [
        (batch_size, hidden_size, 1, seq_len) for seq_len in lengths
    ]
    length_sym = get_new_symbol()

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                (batch_size, hidden_size, 1, length_sym),
                dtype=mil.input_types.types.fp16,
            ),
            mb.TensorSpec(
                (1,),
                dtype=mil.input_types.types.fp16,
            ),
            mb.TensorSpec(
                (1,),
                dtype=mil.input_types.types.fp16,
            ),
            mb.TensorSpec(
                (length_sym,),
                dtype=mil.input_types.types.fp32,
            ),
        ],
        opset_version=mil.builder.AvailableTarget.iOS18,
        # function_name="min_p",
    )
    def min_p_program(hidden_states, p, temp, random_number):
        hidden_states = final_rms_norm(hidden_states, "final_norm_")
        return min_p(hidden_states, ws, p, temp, random_number)

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                (batch_size, hidden_size, 1, length_sym),
                dtype=mil.input_types.types.fp16,
            ),
        ],
        opset_version=mil.builder.AvailableTarget.iOS18,
        # function_name="lm_head",
    )
    def lm_head_program(hidden_states):
        hidden_states = final_rms_norm(hidden_states, "final_norm_")
        logits_list = []
        for i, w in enumerate(ws):
            logits_chunk = mb.conv(
                x=hidden_states,
                weight=w,
                name=f"logits_chunk_{i}",
            )
            logits_list.append(logits_chunk)
        logits = mb.concat(values=logits_list, axis=1, name="logits")
        return logits

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
    pipeline.set_options(
        "common::materialize_symbolic_shape_program",
        {
            "function_name_to_materialization_map": {
                # As an example, let us assume the input is x (is0, is1, 1024)
                # f"main": {
                f"min_p_length_{l}": {
                    "hidden_states": (batch_size, hidden_size, 1, l),
                    "p": (1,),
                    "temp": (1,),
                    "random_number": (l,),
                }
                for l in lengths
            },
            # "source_function_name": "min_p",
        },
    )

    pipeline.remove_passes({"common::add_int16_cast"})
    pipeline.remove_passes({"common::add_fp16_cast"})
    PassPipelineManager.apply_pipeline(min_p_program, pipeline)
    min_p_program.export_as_multifunction = True
    min_p_program.skip_all_passes = True

    min_p_mlmodel = ct.convert(
        min_p_program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
        # pass_pipeline=pipeline,
        # package_dir=package_dir,
        inputs=[
            ct.TensorType(
                shape=ct.EnumeratedShapes(hidden_states_input_shapes),
                name="hidden_states",
            ),
            ct.TensorType(
                shape=(1,),
                name="p",
            ),
            ct.TensorType(
                shape=(1,),
                name="temp",
            ),
            ct.TensorType(
                shape=ct.EnumeratedShapes([(l,) for l in lengths]),
                name="random_number",
            ),
        ],
    )

    pipeline.insert_pass(0, "common::materialize_symbolic_shape_program")
    pipeline.set_options(
        "common::materialize_symbolic_shape_program",
        {
            "function_name_to_materialization_map": {
                f"lm_head_length_{l}": {
                    "hidden_states": (batch_size, hidden_size, 1, l),
                }
                for l in lengths
            },
        },
    )

    pipeline.remove_passes({"common::add_int16_cast"})
    pipeline.remove_passes({"common::add_fp16_cast"})
    PassPipelineManager.apply_pipeline(lm_head_program, pipeline)
    lm_head_program.export_as_multifunction = True
    lm_head_program.skip_all_passes = True

    lm_head_mlmodel = ct.convert(
        lm_head_program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
        # package_dir=package_dir,
        inputs=[
            ct.TensorType(
                shape=ct.EnumeratedShapes(hidden_states_input_shapes),
                name="hidden_states",
            ),
        ],
    )

    desc = MultiFunctionDescriptor()
    desc.add_model(
        model_path=min_p_mlmodel.package_path,
    )
    desc.remove_function("main")
    desc.add_function(
        model_path=min_p_mlmodel.package_path,
        src_function_name="main",
        target_function_name="min_p_flex",
    )
    desc.add_model(
        model_path=lm_head_mlmodel.package_path,
    )
    desc.remove_function("main")
    desc.add_function(
        model_path=lm_head_mlmodel.package_path,
        src_function_name="main",
        target_function_name="lm_head_flex",
    )
    desc.default_function_name = "lm_head_flex"
    save_multifunction(desc, package_dir)

    return desc


def export_embeddings(tensors: Tensors, package_dir: str):
    embs = tensors["model.embed_tokens.weight"].float().half().numpy()
    with open(package_dir, "wb") as f:
        np.save(f, embs, allow_pickle=False)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--layer_from", default=0, type=int)
    parser.add_argument("--layer_to", default=-1, type=int)
    parser.add_argument("--convert_model", default=False, action="store_true")
    parser.add_argument("--convert_lm_head", default=False, action="store_true")
    parser.add_argument("--export_embeddings", default=False, action="store_true")
    parser.add_argument("--cache_len", default=1024, type=int)

    parser.add_argument("--output_name", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    tensors_path = os.path.join(args.model_path, "model.safetensors")
    tensors = load_file(tensors_path)

    if args.convert_model:
        model, config = build_model_from_safetensors(
            tensors,
            config,
            args.layer_from,
            args.layer_to,
            max_sequence_length=2048
            * 4,  # this param is used to build the embeddings and attention mask
        )
        mlmodel = convert(
            model,
            config,
            package_dir=args.output_name + ".mlpackage" if args.output_name else None,
            cache_len=args.cache_len,
        )

    if args.convert_lm_head:
        lm_head_mlmodel = convert_lm_head(
            tensors, package_dir=args.output_name + "_lmhead.mlpackage"
        )

    if args.export_embeddings:
        export_embeddings(tensors, args.output_name + "_embeddings.npy")

    # print(mlmodel._get_mil_internal())
    # print_compute_plan_sync(
    #     mlmodel.get_compiled_model_path(), compute_unit=ct.ComputeUnit.CPU_AND_NE
    # )
    # mlmodel.save("falcon_edge_bitnet")
    import torch
    from transformers import AutoModel
    from transformers.models.llama.modeling_llama import (
        LlamaForCausalLM,
        LlamaDecoderLayer,
    )

    torch.random.manual_seed(42)
    model: LlamaForCausalLM = AutoModel.from_pretrained("tiiuae/Falcon-E-1B-Instruct")
    hidden_states = torch.randn(1, 8, 2048).half()
    hidden_states_np = hidden_states.transpose(-1, -2).unsqueeze(-2).numpy()

    torch_causal_mask = torch.arange(8)[:, None] <= torch.arange(8)[None, :]
    torch_causal_mask = torch_causal_mask[None, None, :, :]
    torch_layer: LlamaDecoderLayer = model.layers[0]
    position_embeddings = model.rotary_emb(hidden_states, torch.arange(8).unsqueeze(0))
    torch_pred = torch_layer(
        hidden_states,
        torch_causal_mask,
        torch.arange(8).unsqueeze(0),
        position_embeddings=position_embeddings,
    )
    print(torch_pred)

    state = mlmodel.make_state()
    coreml_pred = mlmodel.predict(
        {
            "hidden_states": hidden_states_np,
            "kv_write_idx": np.array([0], dtype=np.int32),
            "positions": np.arange(8, dtype=np.int32)[None, :],
        },
        state=state,
    )
    print(coreml_pred)
