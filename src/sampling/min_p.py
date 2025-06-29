import numpy as np
from coremltools.converters.mil import Builder as mb


def min_p(
    hidden_states,
    ws,
    min_p,
    temp,
    random_number,
    # gumbel_noise,
):
    # batch_size,
    # gumbel_noise = mb.random_uniform(shape=[])

    lses = []
    max_logits = []
    temp = mb.inverse(
        x=temp, epsilon=np.array(0.0, dtype=np.float16), name="temp_inverse"
    )
    logits_list = []
    max_indices = []
    indices_list = []
    # for i, logits_chunk in enumerate(logits_list):
    for i, w in enumerate(ws):
        logits_chunk = mb.conv(
            x=hidden_states,
            weight=w,
            name=f"logits_chunk_{i}",
        )
        logits_chunk = mb.mul(x=logits_chunk, y=temp, name=f"logits_chunk_{i}_mul")
        logits_list.append(logits_chunk)
        m = mb.reduce_max(
            x=logits_chunk, axes=(1,), keep_dims=True, name=f"logits_chunk_{i}_max"
        )
        index = mb.reduce_argmax(
            x=logits_chunk,
            axis=1,
            keep_dims=True,
            name=f"logits_chunk_{i}_argmax",
            output_dtype="uint16",
        )
        indices_list.append(index)
        # m, indices =  mb.topk(x=logits_chunk, k=1, axis=1, name=f"logits_chunk_{i}_topk")
        # max_indices.append(indices)
        max_logits.append(m)
        logits_chunk = mb.sub(x=logits_chunk, y=m, name=f"logits_chunk_{i}_sub")
        lse = mb.reduce_log_sum_exp(
            x=logits_chunk, axes=(1,), keep_dims=True, name=f"logits_chunk_{i}_lse_sub"
        )
        lse = mb.add(x=lse, y=m, name=f"logits_chunk_{i}_lse")
        lses.append(lse)

    lses = mb.concat(values=lses, axis=1, name=f"logits_lses")
    m = mb.reduce_max(x=lses, axes=(1,), name=f"logits_lses_max", keep_dims=True)
    lses = mb.sub(x=lses, y=m, name=f"logits_lses_sub")
    lses = mb.reduce_log_sum_exp(
        x=lses, axes=(1,), name=f"logits_lses_logsumexp", keep_dims=True
    )
    lse = mb.add(x=lses, y=m, name=f"logits_lse")
    max_logits = mb.concat(values=max_logits, axis=1, name="logits_max_logits_chunks")
    max_logit = mb.reduce_max(
        x=max_logits, axes=(1,), name="logits_max_logit", keep_dims=True
    )
    max_value = mb.sub(x=max_logit, y=lse, name="logits_max_logit_sub")
    max_prob = mb.exp(x=max_value, name="max_prob")

    min_p_tresh = mb.mul(x=max_prob, y=min_p, name="min_p_thresh")
    # min_p_sum = np.array(0.0, dtype=np.float16)

    probs_list = []
    b = mb.fill_like(
        ref_tensor=logits_list[0], value=np.array(0.0, dtype=np.float16), name="b"
    )  # TODO: cases where chunk size does not match
    for i, logits_chunk in enumerate(logits_list):
        logits_chunk = mb.sub(x=logits_chunk, y=lse, name=f"logits_chunk_{i}_sub")
        probs_chunk = mb.exp(x=logits_chunk, name=f"probs_chunk_{i}")
        mask = mb.greater_equal(
            x=probs_chunk, y=min_p_tresh, name=f"mask_probs_chunk_{i}"
        )
        mask_fp16 = mb.cast(x=mask, dtype="fp16", name=f"mask_chunk_{i}_fp16")
        masked_probs_chunk = mb.select(
            cond=mask,
            a=probs_chunk,
            b=mask_fp16,
            # b=b,
            # b=np.zeros((1, 1, 1, 1), dtype=np.float16),
            name=f"masked_probs_chunk_{i}",
        )
        probs_list.append(masked_probs_chunk)
        # probs_chunk_sum = mb.reduce_sum(
        #     x=masked_probs_chunk,
        #     axes=(1,),
        #     keep_dims=True,
        #     name=f"masked_probs_chunk_{i}_sum",
        # )
        # min_p_sum = mb.add(
        #     x=min_p_sum,
        #     y=probs_chunk_sum,
        #     name=f"masked_probs_chunk_{i}_sum_cumulative",
        # )

    probs = mb.concat(values=probs_list, axis=1, name="probs")
    probs = mb.cast(x=probs, dtype="fp32", name="probs_fp32")

    # argmax = mb.reduce_argmax(x=probs, axis=1, name="argmax", output_dtype="int32")
    probs_cumsum = mb.cumsum(x=probs, axis=1, exclusive=False, name="probs_cumsum")
    # probs_sum = mb.gather_along_axis(
    probs_sum = mb.gather(
        x=probs_cumsum,
        # indices=[[[[probs_cumsum.shape[1] - 1]]]],
        indices=[probs_cumsum.shape[1] - 1],
        axis=1,
        name="probs_sum",
    )
    # probs_sum = mb.split(
    #     x=probs_cumsum,
    #     split_sizes=[probs_cumsum.shape[1] - 1, 1],
    #     axis=1,
    #     name="probs_sum",
    # )[1]
    random_number = mb.mul(x=random_number, y=probs_sum, name="random_number_scaled")
    probs_mask = mb.greater(x=probs_cumsum, y=random_number, name="probs_greater")
    # probs = mb.select(
    #     cond=probs_greater,
    #     a=probs,
    #     b=np.array(100, dtype=np.float32),
    #     name="probs_select",
    # )
    probs_mask = mb.cast(x=probs_mask, dtype="int32", name="probs_greater_int32")
    sampled_index = mb.reduce_argmax(
        x=probs_mask, axis=1, name="sampled_index", output_dtype="int32", keep_dims=True
    )
    sampled_prob = mb.gather_along_axis(
        x=probs, indices=sampled_index, axis=1, name="sampled_index_probability"
    )
    # sampled_prob, sampled_value = mb.topk(x=probs, k=1, axis=1, name="topk")
    max_logit_index = mb.reduce_argmax(
        x=max_logits, axis=1, name="max_logit_index", keep_dims=True
    )

    indices_list = [
        mb.cast(x=indices, dtype="int32", name=f"indices_chunk_{i}_int32")
        for i, indices in enumerate(indices_list)
    ]
    indices = mb.concat(values=indices_list, axis=1, name="indices")
    argmax = mb.gather_along_axis(
        x=indices, axis=1, indices=max_logit_index, name="argmax_chunks"
    )
    offset = mb.mul(x=ws[0].shape[0], y=max_logit_index)
    argmax = mb.add(x=argmax, y=offset, name="argmax")

    return sampled_index, sampled_prob, argmax, max_prob
