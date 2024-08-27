from qwen2 import *

# def main():
if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn

    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.converters.mil as mil
    from transformers import AutoModelForCausalLM

    torch_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    precision = "fp16"
    # precision = "fp32"

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

    state_implementation = "none"
    state_implementation = "big_state"  # "interleaved", "per_block", "per_group_split", "per_group_slice", "per_group_gather", "big_state_gather", "big_state_slice", "max_num_heads"
    max_num_heads = 19
    state_update_at = ""
    state_update_at = "attention"
    seqlength = 512

    shift = 0
    num_blocks = -1
    coreml_model = from_torch(
        torch_model,
        channels_first=channels_first,
        pre_normalization_and_pos_encoding=pre_normalization_and_pos_encoding,
        multi_query_head=multi_query_head,
        repeat_interleave=repeat_interleave,
        dtype=nptype,
        state_implementation=state_implementation,
        state_update_at=state_update_at,
        num_blocks=num_blocks,
        shift=shift,
        max_length=seqlength,
    )

    # coreml_model.blocks = coreml_model.blocks[shift:]  # for testing/debugging stuff
    if num_blocks == -1:
        num_blocks = len(coreml_model.blocks)
    # coreml_model.blocks = coreml_model.blocks[:num_blocks]

    # shapes = [(1, 1280, seqlen) for seqlen in (32, 128, 256, 512, 1024, 2048)]
    shapes = [(1, seqlen) for seqlen in (32, 128, 256, 512, 1024, 2048)]
    enum_shape = mil.input_types.EnumeratedShapes(shapes=shapes)

    # fixed_shape = (1, 1280, 128)
    # fixed_shape = (1, 256)
    qseqlength = 1
    if channels_first:
        fixed_shape = (1, 896, qseqlength)
    else:
        fixed_shape = (1, qseqlength, 896)
    # shape = enum_shape.symbolic_shape
    # shape = fixed_shape

    block: Block

    num_states = 0
    if state_implementation == "big_state":
        num_states = num_blocks
        state_spec = sum(
            [
                [
                    # mb.StateTensorSpec(
                    #     # (1, block.attn.nkvheads, seqlength, 64),
                    #     (1, 1, seqlength - 1, 64),
                    #     # (1, block.attn.nkvheads, 64, seqlength),
                    #     dtype=mil.input_types.types.fp16,
                    # ) for _ in range(block.attn.nkvheads * 2)
                    # mb.StateTensorSpec(
                    #     (1, block.attn.nkvheads * 2, seqlength - 1, 64),
                    #     # (1, block.attn.nkvheads, 64, seqlength),
                    #     dtype=mil.input_types.types.fp16,
                    # ),
                    mb.StateTensorSpec(
                        (1, block.attn.nkvheads * 2, seqlength - 1, 64),
                        # (1, block.attn.nkvheads, 64, seqlength),
                        dtype=mil.input_types.types.fp16,
                    ),
                
                ]
                for block in coreml_model.blocks
            ],
            [],
        )
        state_spec = [
            mb.StateTensorSpec(
                (24, 2, seqlength, 64), dtype=mil.input_types.types.fp16,
            ),
            mb.StateTensorSpec(
                (24, 2, seqlength, 64), dtype=mil.input_types.types.fp16,
            ),
        ]
    # awfull fix to programatically create variable number of arguments and their name
    chunk_size = 24
    chunk_nheads = sum([block.attn.nkvheads for block in coreml_model.blocks[:chunk_size]])
    block: Block
    args_str = ",\n    ".join(
        # [f"key_state_{i},\n    value_state_{i}" for i in range(chunk_size)]
        # [f"key_state_{i}_{j},\n    value_state_{i}_{j}" for i, block in enumerate(coreml_model.blocks[:chunk_size]) for j in range(block.attn.nkvheads)]
        # [f"kv_cache_{i}" for i in range(chunk_size)]
        ["key_state", "value_state"]
    )
    # chunk_args_str = args_str[:chunk_size]
    func_def = f"""
def var_program(
    input_ids,
    query_pos,
    # mask,
    # query_sin_emb,
    # query_cos_emb,
    # hidden_state_state,
    # query_pos_state,
    # mask_state,
    # query_sin_emb_state,
    # query_cos_emb_state,
    {args_str}
):
    states = [\n    {args_str}\n    ]
    return coreml_model(
        input_ids=input_ids,
        query_pos=query_pos,
        # mask=mask,
        # query_sin_emb=query_sin_emb,
        # query_cos_emb=query_cos_emb,
        # hidden_state_state=hidden_state_state,
        # query_pos_state=query_pos_state,
        # mask_state=mask_state,
        # query_sin_emb_state=query_sin_emb_state,
        # query_cos_emb_state=query_cos_emb_state,
        states=states,
        num_blocks={chunk_size},
        apply_lm_head=True,
        propagate_state=False,
        apply_initial_embedding=True,
        return_mask_and_pos_emb=False,
    )
"""

    local_namespace = {}
    exec(func_def, globals(), local_namespace)
    program_func = local_namespace["var_program"]
    print(func_def)
    # print(program_func)

    pipeline = ct.PassPipeline.DEFAULT
    # pipeline.remove_passes({"common::add_int16_cast"})
    coreml_model_program = mb.program(
        input_specs=[
            mb.TensorSpec((1, qseqlength), dtype=mil.input_types.types.int32),
            # mb.TensorSpec(fixed_shape, dtype=mil.input_types.types.fp16),
            mb.TensorSpec((1,), dtype=mil.input_types.types.int32),  # query_pos
            # mb.TensorSpec((1, 1, seqlength), dtype=mil.input_types.types.fp16),  # mask
            # mb.TensorSpec(
            #     (1, 1, 1, 64), dtype=mil.input_types.types.fp16
            # ),  # query_sin_emb
            # mb.TensorSpec(
            #     (1, 1, 1, 64), dtype=mil.input_types.types.fp16
            # ),  # query_cos_emb
            # mb.StateTensorSpec((1, 896, 1), dtype=mil.input_types.types.fp16), # hidden_state_state
            # mb.StateTensorSpec((1,), dtype=mil.input_types.types.int16),  # query_pos_state
            # mb.StateTensorSpec((1, 1, seqlength), dtype=mil.input_types.types.fp16),  # mask
            # mb.StateTensorSpec(
            #     (1, 1, 1, 64), dtype=mil.input_types.types.fp16
            # ),  # query_sin_emb
            # mb.StateTensorSpec(
            #     (1, 1, 1, 64), dtype=mil.input_types.types.fp16
            # ),  # query_cos_emb
            # *state_spec[:chunk_size * 2],
            # *state_spec[:chunk_size],
            *state_spec
            # *state_spec[:chunk_nheads * 2],
        ],
        # opset_version=mil.builder.AvailableTarget.iOS17,
        opset_version=mil.builder.AvailableTarget.iOS18,
    )(program_func)

    print(coreml_model_program)
    cml_converted = ct.convert(
        coreml_model_program,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_units=ct.ComputeUnit.ALL,
        # compute_precision=ct.precision.FLOAT16,
        compute_precision=compute_precision,
        # minimum_deployment_target=ct.target.iOS17,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            # ct.TensorType(name="input_ids", shape=ct.EnumeratedShapes(shapes)),
            # ct.TensorType(name="input_ids", shape=fixed_shape),
            ct.TensorType(name="input_ids", shape=(1, qseqlength)),
            ct.TensorType(name="query_pos", shape=(1,)),
            # ct.TensorType(name="mask", shape=(1, 1, seqlength)),
            # ct.TensorType(name="query_sin_emb", shape=(1, 1, 64)),
            # ct.TensorType(name="query_cos_emb", shape=(1, 1, 64)),
        ],
        pass_pipeline=pipeline,
    )

    print("aaah")

    # name = f"qwen0.5b-instruct_shared_emb_head_stateful_inference_ctx_512_query_{qseqlength}.mlpackage"
    # name = f"qwen0.5b-instruct_chunked_shared_emb_head_stateful_inference_ctx_512_query_{qseqlength}.mlpackage"
    name = f"QWEN05B-{chunk_size}L-CTX-512-QLEN-{qseqlength}-CL.mlpackage"
    # name = f"QWEN05B-{chunk_size}L.mlpackage"
    if os.path.exists(name):
        shutil.rmtree(name)
    try:
        cml_converted.save(name)
    except Exception as e:
        print(e)
        cml_converted.save(f"F_{name}")

    try:
        print(cml_converted._get_mil_internal())
    except Exception as e:
        print(e)

    ################################ SECOND CHUNK ####################################

    # chunk_size = 21
    shift = chunk_size
    remaining_blocks = len(coreml_model.blocks) - chunk_size
    # remaining_blocks = 7
    args_str = ",\n    ".join(
        # [f"key_state_{i},\n    value_state_{i}" for i in range(shift, shift + remaining_blocks)]
        # [f"key_state_{i + shift}_{j},\n    value_state_{i + shift}_{j}" for i, block in enumerate(coreml_model.blocks[shift:shift + remaining_blocks]) for j in range(block.attn.nkvheads)]
        [f"kv_cache_{i}" for i in range(shift, shift + remaining_blocks)]
    )
    func_def = f"""
def var_program(
    hidden_state,
    # query_pos,
    # query_pos_state,
    mask,
    query_sin_emb,
    query_cos_emb,
    # mask_state,
    # query_sin_emb_state,
    # query_cos_emb_state,
    # hidden_state_state,
    {args_str}
):
    states = [\n    {args_str}\n    ]
    return coreml_model(
        input_ids=hidden_state,
        # hidden_state_state=hidden_state_state,
        # query_pos=query_pos,
        # mask_state=mask_state,
        # query_sin_emb_state=query_sin_emb_state,
        # query_cos_emb_state=query_cos_emb_state,
        states=states,
        num_blocks={remaining_blocks},
        shift={shift},
        apply_initial_embedding=False,
        mask=mask,
        query_cos_emb=query_cos_emb,
        query_sin_emb=query_sin_emb,
        # mask=False,
        # query_cos_emb=False,
        # query_sin_emb=False,
        apply_lm_head=True,
        return_mask_and_pos_emb=False,
    )
"""

    # local_namespace = {}
    # exec(func_def, globals(), local_namespace)
    # program_func = local_namespace["var_program"]
    # print(func_def)
    # # print(program_func)

    # coreml_model_program = mb.program(
    #     input_specs=[
    #         mb.TensorSpec((1, 896, 1), dtype=mil.input_types.types.fp16),
    #         # mb.TensorSpec((1,), dtype=mil.input_types.types.int32),  # query_pos
    #         # mb.StateTensorSpec((1,), dtype=mil.input_types.types.int16),  # query_pos
    #         mb.TensorSpec((1, 1, 1, seqlength), dtype=mil.input_types.types.fp16),  # mask
    #         # mb.TensorSpec((1, 1, seqlength, 1), dtype=mil.input_types.types.fp16),  # mask
    #         mb.TensorSpec(
    #             (1, 1, 1, 64), dtype=mil.input_types.types.fp16
    #             # (1, 1, 64, 1), dtype=mil.input_types.types.fp16
    #         ),  # query_sin_emb
    #         mb.TensorSpec(
    #             (1, 1, 1, 64), dtype=mil.input_types.types.fp16
    #             # (1, 1, 64, 1), dtype=mil.input_types.types.fp16
    #         ),  # query_cos_emb
    #         # mb.StateTensorSpec((1, 1, seqlength), dtype=mil.input_types.types.fp16),  # mask
    #         # mb.StateTensorSpec(
    #         #     (1, 1, 1, 64), dtype=mil.input_types.types.fp16
    #         # ),  # query_sin_emb
    #         # mb.StateTensorSpec(
    #         #     (1, 1, 1, 64), dtype=mil.input_types.types.fp16
    #         # ),  # query_cos_emb
    #         # mb.StateTensorSpec((1, 896, 1), dtype=mil.input_types.types.fp16),
    #         *state_spec[chunk_size:chunk_size + remaining_blocks],
    #         # *state_spec[chunk_size * 2:],
    #         # *state_spec[chunk_nheads * 2:],
    #     ],
    #     # opset_version=mil.builder.AvailableTarget.iOS17,
    #     opset_version=mil.builder.AvailableTarget.iOS18,
    # )(program_func)

    # print(coreml_model_program)
    # cml_converted = ct.convert(
    #     coreml_model_program,
    #     compute_units=ct.ComputeUnit.CPU_AND_NE,
    #     # compute_precision=ct.precision.FLOAT16,
    #     compute_precision=compute_precision,
    #     # minimum_deployment_target=ct.target.iOS17,
    #     minimum_deployment_target=ct.target.iOS18,
    #     # inputs=[
    #     #     ct.TensorType(name="input_ids", shape=(1, 896, 1)),
    #     #     ct.TensorType(name="query_pos", shape=(1,)),
    #     # ],
    # )

    # print("aaah")

    # name = f"qwen0.5b-instruct_inference_layer_state_chunk_{chunk_size}_{chunk_size + remaining_blocks}.mlpackage"
    # if os.path.exists(name):
    #     shutil.rmtree(name)
    # try:
    #     cml_converted.save(name)
    # except Exception as e:
    #     print(e)
    #     cml_converted.save(f"F_{name}")

    # try:
    #     print(cml_converted._get_mil_internal())
    # except Exception as e:
    #     print(e)

    # return coreml_model_program

# if __name__ == "__main__":
#     main()