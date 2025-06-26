from typing import Optional

import math
import numpy as np
from coremltools.converters.mil.mil import types
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

def build_causal_mask(indices):
    ones = mb.fill_like(ref_tensor=indices, value=np.array(1, dtype=np.int32), name="mask_ones")
    arange = mb.cumsum(x=ones, axis=-1, exclusive=True, name="mask_arange")  # exclusive is actually not required
    mask_left = mb.expand_dims(x=arange, axes=(-1,), name="mask_arange_left")
    mask_right = mb.expand_dims(x=arange, axes=(-2,), name="mask_arange_right")
    mask = mb.greater_or_equal(x=mask_left, y=mask_right, name="mask_bool")
    mask = mb.where(condition=mask, x=np.array(0, dtype=np.float16), y=np.array(-np.inf, dtype=np.float16), name="mask_fp16")
    if indices.rank == 1:
        mask = mb.expand_dims(x=mask, axes=(0, 1), name="mask_expand_dims")
    else:
        mask = mb.expand_dims(x=mask, axes=(0,), name="mask_expand_dims")
    return mask




def gather_static(indices, target, prefix, transpose=False):
    values = mb.gather(x=target, indices=indices, axis=0, name=f"{prefix}gather")
    if transpose:
        values = mb.transpose(
            x=values,
            perm=[0, 2, 1],
            name=f"{prefix}transpose",
        )
    values = mb.expand_dims(x=values, axes=(1,), name=f"{prefix}expand_dims")
    return values


def build_causal_mask(max_length, boolean=False):
    mask = np.arange(max_length, dtype=np.int32)
    mask = mask[:, None] >= mask[None, :]
    if boolean:
        return mask
    return np.where(
        mask, np.array(0, dtype=np.float16), np.array(-np.inf, dtype=np.float16)
    )


def rotate_half(x, prefix=None, axis=-1):
    if types.builtin_to_string(x.dtype) == "fp16":
        mone = np.float16(-1.0)
    else:
        mone = np.float32(-1.0)

    x1, x2 = mb.split(x=x, num_splits=2, axis=axis, name=f"{prefix}rotate_half_split")
    neg_x2 = mb.mul(x=x2, y=mone, name=f"{prefix}rotate_half_neg")
    return mb.concat(values=(neg_x2, x1), axis=axis, name=f"{prefix}rotate_half_concat")


def apply_rotary_pos_emb(hidden_states, sin_emb, cos_emb, axis: int, prefix: str):
    lhs = mb.mul(x=hidden_states, y=cos_emb, name=f"{prefix}rope_lhs_mult")
    xrot = rotate_half(x=hidden_states, prefix=prefix, axis=axis)
    rhs = mb.mul(x=xrot, y=sin_emb, name=f"{prefix}rope_rhs_mult")
    return mb.add(x=lhs, y=rhs, name=f"{prefix}rope")


def update_cache(
    update,
    cache,
    state,
    cache_write_start,
    cache_write_end,
    kv_layer_write_idx,
    prefix,
    channels_last: bool = True,
):
    # seqlen_idx = 2 if channels_last else 3
    # seqlen = update.shape[seqlen_idx]
    # if is_symbolic(seqlen):
    #     shape = mb.shape(x=update, name=f"{prefix}input_shape")
    #     seqlen = mb.gather(
    #         x=shape, indices=seqlen_idx, name=f"{prefix}sequence_length"
    #     )
    # cache_write_end = mb.add(
    #     x=cache_write_start, y=seqlen, name=f"{prefix}kv_write_idx_end"
    # )
    if channels_last:
        begin = mb.concat(
            values=(
                np.array([kv_layer_write_idx], dtype=np.int32),
                np.array([0], dtype=np.int32),
                cache_write_start,
                np.array([0], dtype=np.int32),
            ),
            axis=0,
            name=prefix + "slice_update_begin",
        )
        end = mb.concat(
            values=(
                np.array([kv_layer_write_idx + update.shape[0]], dtype=np.int32),
                np.array([update.shape[1]], dtype=np.int32),
                cache_write_end,
                np.array([update.shape[3]], dtype=np.int32),
            ),
            axis=0,
            name=prefix + "slice_update_end",
        )
    else:
        begin = mb.concat(
            values=(
                np.array([kv_layer_write_idx], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                cache_write_start,
            ),
            axis=0,
            name=prefix + "slice_update_begin",
        )
        end = mb.concat(
            values=(
                np.array([kv_layer_write_idx + update.shape[0]], dtype=np.int32),
                np.array([update.shape[1]], dtype=np.int32),
                np.array([update.shape[2]], dtype=np.int32),
                cache_write_end,
            ),
            axis=0,
            name=prefix + "slice_update_end",
        )

    cache = mb.slice_update(
        x=cache,
        update=update,
        begin=begin,
        end=end,
        name=prefix + "slice_update",
        begin_mask=[False] * 4,
        end_mask=[False] * 4,
        squeeze_mask=[False] * 4,
    )
    cache = mb.coreml_update_state(state=state, value=cache, name=prefix + "update_state")
    return cache


def update_kv_cache(
    key,
    value,
    kv_cache,
    kv_write_idx,  # current implementation supports only one contiguous write
    kv_layer_write_idx: int,
    prefix: str,
    value_hidden_last=False,
):
    b, numheads, _, headdim = key.shape
    seqlen = key.shape[2]

    k_cache, v_cache = kv_cache
    start = kv_write_idx
    end = mb.add(x=kv_write_idx, y=seqlen, name=prefix + "slice_update_end")

    if k_cache.shape[0] == 1:
        kv_layer_write_idx = 0

    begin = mb.concat(
        values=(
            np.array([kv_layer_write_idx], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([start], dtype=np.int32),
            np.array([0], dtype=np.int32),
        ),
        axis=0,
        name=prefix + "key_slice_update_begin",
    )
    end = mb.concat(
        values=(
            np.array([kv_layer_write_idx + b], dtype=np.int32),
            np.array([numheads], dtype=np.int32),
            np.array([end], dtype=np.int32),
            np.array([headdim], dtype=np.int32),
        ),
        axis=0,
        name=prefix + "key_slice_update_end",
    )
    k_cache = mb.slice_update(
        x=k_cache,
        update=key,
        begin=begin,
        end=end,
        name=prefix + "key_slice_update",
        begin_mask=[False] * 4,
        end_mask=[False] * 4,
        squeeze_mask=[False] * 4,
    )

    if value_hidden_last:
        v_cache = mb.slice_update(
            x=v_cache,
            update=value,
            begin=begin,
            end=end,
            name=prefix + "value_slice_update",
            begin_mask=[False] * 4,
            end_mask=[False] * 4,
            squeeze_mask=[False] * 4,
        )
    else:
        begin = mb.concat(
            values=(
                np.array([kv_layer_write_idx], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([start], dtype=np.int32),
            ),
            axis=0,
            name=prefix + "value_slice_update_begin",
        )
        end = mb.concat(
            values=(
                np.array([kv_layer_write_idx + b], dtype=np.int32),
                np.array([numheads], dtype=np.int32),
                np.array([headdim], dtype=np.int32),
                np.array([end], dtype=np.int32),
            ),
            axis=0,
            name=prefix + "value_slice_update_end",
        )
        v_cache = mb.slice_update(
            x=v_cache,
            update=value,
            begin=begin,
            end=end,
            name=prefix + "value_slice_update",
            begin_mask=[False] * 4,
            end_mask=[False] * 4,
            squeeze_mask=[False] * 4,
        )

    return k_cache, v_cache


def gqa_attention(
    query,
    key,
    value,
    prefix: str,
    scaling: Optional[np.array] = None,
    attention_mask: Optional = None,
    attn_logit_softcapping: Optional[float] = None,
    value_channels_last: bool = True,
):
    """
    query: (batch, heads, head_dim, sequence_length)
    key: (batch, heads, sequence, head_dim)
    values: (batch, heads, sequence, head_dim)
    kv_cache: ((batch, heads, cache_length, head_dim), (batch, heads, cache_length, head_dim))
    attention_mask: (batch, 1, cache_length, sequence_length)
    """
    batch_size, num_q_heads, seqlen, headdim = query.shape
    num_kv_heads = key.shape[1]
    assert (num_q_heads % num_kv_heads) == 0
    group_size = num_q_heads // num_kv_heads
    qs = mb.split(x=query, axis=1, num_splits=num_kv_heads, name=prefix + "query_split")
    ks = mb.split(x=key, axis=1, num_splits=num_kv_heads, name=prefix + "key_split")
    vs = mb.split(x=value, axis=1, num_splits=num_kv_heads, name=prefix + "value_split")

    per_head_attention = []
    for i in range(num_kv_heads):
        groupi = i
        qi = qs[i]
        if value_channels_last:
            scores = mb.matmul(
                x=ks[groupi],
                y=qi,
                name=prefix + f"group_{i}_scores",
                transpose_y=True,
            )  # (batch, heads, target_seqlen, source_seqlen)
        else:
            raise NotImplementedError()
        if attn_logit_softcapping is not None:
            attn_logit_softcapping = np.array(attn_logit_softcapping, dtype=np.float32)
            attn_logit_softcapping_inv = (1 / attn_logit_softcapping).astype(np.float16)
            scores = mb.mul(
                x=scores,
                y=attn_logit_softcapping_inv,
                name=prefix + f"group_{i}_scores_softcapping_in",
            )
            scores = mb.tanh(
                x=scores, name=prefix + f"group_{i}_scores_scoftcapping_tanh"
            )
            scores = mb.mul(
                x=scores,
                y=attn_logit_softcapping.astype(np.float16),
                name=prefix + f"group_{i}_scores_softcapping",
            )
        if scaling is not None:
            scores = mb.mul(
                x=scores, y=scaling, name=prefix + f"group_{i}_scores_scaled"
            )
        if attention_mask is not None:
            scores = mb.add(
                x=scores, y=attention_mask, name=prefix + f"group_{i}_scores_masked"
            )
        if value_channels_last:
            scores = mb.softmax(
                x=scores, axis=-2, name=prefix + f"group_{i}_scores_softmax"
            )
            output = mb.matmul(
                x=scores,
                y=vs[groupi],
                name=prefix + f"group_{i}_output",
                transpose_x=True,
            )  # (batch, heads, headdim, source_seqlen)
            output = mb.transpose(
                x=output,
                perm=[0, 1, 3, 2],
                name=prefix + f"group_{i}_output_transpose",
            )
            # output = mb.reshape(
            #     x=output,
            #     # shape=[batch_size, group_size * headdim, 1, seqlen],
            #     shape=[batch_size, group_size * headdim, 1, -1],
            #     name=prefix + f"group_{i}_output_reshaped",
            # )
        else:
            raise NotImplementedError()
        per_head_attention.append(output)

    # output = torch.cat(per_head_attention, dim=1)
    # return before concat, this way we could experiment later to perform
    # split output projection and the reduce sum, maybe faster due to
    # cache (?)
    return per_head_attention
