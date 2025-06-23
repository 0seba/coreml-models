import os
import json
from typing import Dict, Any
from argparse import ArgumentParser

import numpy as np
from torch import Tensor
from safetensors.torch import load_file  # use torch safetensors to load bfloat16

import coremltools as ct
import coremltools.converters.mil as mil
from coremltools.converters.mil import Builder as mb

from layers import LUTLinear
from positional_encodings import compute_default_rope_parameters, compute_rope_embedding
from utils.quantization import unpack_weights
from .falcon_edge_bitnet import (
    FalconEdgeRMSNorm,
    FalconEdgeAttentionLayer,
    FalconEdgeBitnetDecoderLayer,
    FalconEdgeMLP,
    FalconEdgeBitnetModel,
)


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
    print(scale)

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
    model_path: str, layer_from: int, layer_to: int, max_sequence_length: int = 2048
):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(config)
    tensors_path = os.path.join(model_path, "model.safetensors")
    tensors = load_file(tensors_path)

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
    )
    return model, config


def convert(model: FalconEdgeBitnetModel, config: Dict[str, Any]):
    batch_size = 1
    seqlen = 8
    cache_len = 1024
    headdim = config["head_dim"]
    num_hidden_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]

    state_spec = [
        mb.StateTensorSpec(
            (
                num_hidden_layers * batch_size,
                config["num_key_value_heads"],
                cache_len,
                headdim,
            ),
            dtype=mil.input_types.types.fp16,
        ),
        mb.StateTensorSpec(
            (
                num_hidden_layers * batch_size,
                config["num_key_value_heads"],
                cache_len,
                headdim,
            ),
            dtype=mil.input_types.types.fp16,
        ),
    ]

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                (batch_size, hidden_size, 1, seqlen),
                dtype=mil.input_types.types.fp16,
            ),
            mb.TensorSpec(
                (1,),
                dtype=mil.input_types.types.int32,
            ),
            mb.TensorSpec(
                (
                    batch_size,
                    seqlen,
                ),
                dtype=mil.input_types.types.int32,
            ),
            *state_spec,
        ],
        opset_version=mil.builder.AvailableTarget.iOS18,
    )
    def program(hidden_states, kv_write_idx, positions, key_cache, value_cache):
        return model(hidden_states, kv_write_idx, positions, key_cache, value_cache)

    print(program)

    mlmodel = ct.convert(
        program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=False,
    )

    return mlmodel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--layer_from", type=int, required=True)
    # parser.add_argument("--layer_to", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, config = build_model_from_safetensors(
        args.model_path, 0, 1, max_sequence_length=2048
    )
    mlmodel = convert(model, config)
    # mlmodel.save("falcon_edge_bitnet.mlmodel")
    import torch
    from transformers import AutoModel
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

    model: LlamaForCausalLM = AutoModel.from_pretrained("tiiuae/Falcon-E-1B-Instruct")
    hidden_states = torch.randn(1, 8, 2048).half()
    hidden_states_np = hidden_states.transpose(-1, -2).unsqueeze(-2).numpy()

    torch_causal_mask = torch.arange(8)[:, None] <= torch.arange(8)[None, :]
    torch_causal_mask = torch_causal_mask[None, None, :, :]
    torch_layer: LlamaDecoderLayer = model.layers[0]
    position_embeddings = model.rotary_emb(hidden_states, torch.arange(8).unsqueeze(0))
    torch_pred = torch_layer(hidden_states, torch_causal_mask, torch.arange(8).unsqueeze(0), position_embeddings=position_embeddings)
    print(torch_pred)

    state = mlmodel.make_state()
    coreml_pred = mlmodel.predict({
        "hidden_states": hidden_states_np,
        "kv_write_idx": np.array([0], dtype=np.int32),
        "positions": np.arange(8, dtype=np.int32)[None, :],
    }, state=state)
    print(coreml_pred)
