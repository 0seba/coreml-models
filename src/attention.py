import numpy as np

import coremltools as ct
import coremltools.converters.mil as mil
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Var, types

from utils import is_symbolic
from layers import Linear, RMSNorm


def make_rope_mask_indices(input_ids):
    ones = mb.fill_like(ref_tensor=input_ids, value=np.array(1, dtype=np.int32))
    indices = mb.cumsum(x=ones, axis=1, exclusive=True)
    return indices


def rope_rotate(
    x,
    pos_sin_1,
    pos_sin_2,
    pos_cos_1,
    pos_cos_2,
    prefix,
    axis=-1,
):
    neg_one = -1.0
    if types.builtin_to_string(x.dtype) == "fp16":
        neg_one = np.array(-1, dtype=np.float16)

    x1, x2 = mb.split(x=x, num_splits=2, axis=axis, name=f"{prefix}_rope_split")
    lhs_1 = mb.mul(x=x1, y=pos_cos_1, name=f"{prefix}_rope_lhs_1")
    lhs_2 = mb.mul(x=x2, y=pos_cos_2, name=f"{prefix}_rope_lhs_2")

    neg_x2 = mb.mul(x=x2, y=neg_one, name=f"{prefix}_rope_neg_x2")
    rhs_1 = mb.mul(x=neg_x2, y=pos_sin_1, name=f"{prefix}_rope_rhs_1")
    rhs_2 = mb.mul(x=x1, y=pos_sin_2, name=f"{prefix}_rope_rhs_2")

    return mb.add(x=lhs_1, y=rhs_1, name=f"{prefix}_rope_1"), mb.add(
        x=lhs_2,
        y=rhs_2,
        name=f"{prefix}_rope_2",
    )


class RoPEEmbedding:
    def __init__(
        self,
        hdim,
        qlen,
        klen,
        channels_first,
        freq_constant=10_000,
        dtype=np.float32,
        implementation="split_concat",
    ):
        self.hdim = hdim
        self.channels_first = channels_first
        self.freq_constant = freq_constant
        self.dtype = dtype
        self.length = max(qlen, klen)
        self.implementation = implementation
        self.cos_emb, self.sin_emb, self.M = self.compute_rope_embedding(
            hdim,
            freq_constant,
            max(qlen, klen),
            dtype,
        )

    def get_embedding_slices(self, query_length: Var, key_length: Var):
        # if types.builtin_to_string(query_length.dtype) == "fp16":
        #     dtype = np.float16
        # else:
        #     dtype = np.float32

        cos_emb, sin_emb, M = self.compute_rope_embedding(
            self.hdim,
            self.freq_constant,
            self.length,
            self.dtype,
        )

        qdiff = mb.sub(x=key_length, y=query_length)
        query_begin = mb.concat(
            values=(
                # np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                qdiff,
            ),
            axis=0,
        )
        query_size = mb.concat(
            values=(
                # np.array([1], dtype=np.int32),
                np.array([self.hdim], dtype=np.int32),
                np.array([1], dtype=np.int32),
                query_length,
            ),
            axis=0,
        )
        key_begin = [0] * 3
        key_size = mb.concat(
            values=(
                # np.array([1], dtype=np.int32),
                np.array([self.hdim], dtype=np.int32),
                np.array([1], dtype=np.int32),
                key_length,
            ),
            axis=0,
        )

        query_sin_emb = mb.slice_by_size(
            x=sin_emb,
            begin=query_begin,
            size=query_size,
            name=f"query_sin_emb_hdim_{self.hdim}",
        )
        query_cos_emb = mb.slice_by_size(
            x=cos_emb,
            begin=query_begin,
            size=query_size,
            name=f"query_cos_emb_hdim_{self.hdim}",
        )
        key_sin_emb = mb.slice_by_size(
            x=sin_emb,
            begin=key_begin,
            size=key_size,
            name=f"key_sin_emb_hdim_{self.hdim}",
        )
        key_cos_emb = mb.slice_by_size(
            x=cos_emb,
            begin=key_begin,
            size=key_size,
            name=f"key_cos_emb_hdim_{self.hdim}",
        )

        query_begin_M = mb.concat(
            values=(
                np.array([0], dtype=np.int32),
                qdiff,
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
            ),
            axis=0,
        )
        query_size_M = mb.concat(
            values=(
                np.array([1], dtype=np.int32),
                np.array([self.hdim], dtype=np.int32),
                query_length,
                np.array([self.hdim], dtype=np.int32),
            ),
            axis=0,
        )
        key_begin_M = [0] * 4
        key_size_M = mb.concat(
            values=(
                np.array([1], dtype=np.int32),
                np.array([self.hdim], dtype=np.int32),
                key_length,
                np.array([self.hdim], dtype=np.int32),
            ),
            axis=0,
        )
        query_M = mb.slice_by_size(
            x=M,
            begin=query_begin_M,
            size=query_size_M,
            name=f"query_M_hdim_{self.hdim}",
        )
        key_M = mb.slice_by_size(
            x=M,
            begin=key_begin_M,
            size=key_size_M,
            name=f"key_M_hdim_{self.hdim}",
        )

        return query_sin_emb, query_cos_emb, key_sin_emb, key_cos_emb, query_M, key_M

    def init_embedding_slices(
        self,
        # query_length: Var, key_length: Var,
        # input_ids,
        indices,
        channels_first=None,
        # query_pos=None,
        # cached=False,
        cache_length=0,
    ):
        if channels_first is None:
            channels_first = self.channels_first
        cos_emb, sin_emb, M = self.compute_rope_embedding(
            self.hdim,
            self.freq_constant,
            self.length,
            self.dtype,
        )
        if channels_first:
            _cos_emb = np.expand_dims(cos_emb.T, 0)
            _sin_emb = np.expand_dims(sin_emb.T, 0)
            axis = 2
        else:
            _cos_emb = np.expand_dims(cos_emb, 0)
            _sin_emb = np.expand_dims(sin_emb, 0)
            axis = 1

        qindices = indices

        if cache_length:
            self.query_cos_emb = mb.gather(
                x=_cos_emb,
                indices=qindices,
                axis=axis,
                # batch_dims=1,
                name="rope_query_cos_emb",
            )
            self.query_sin_emb = mb.gather(
                x=_sin_emb,
                indices=qindices,
                axis=axis,
                # batch_dims=1,
                name="rope_query_sin_emb",
            )
            if channels_first:
                self.key_cos_emb = _cos_emb[:, :, :cache_length]
                self.key_sin_emb = _sin_emb[:, :, :cache_length]
            else:
                self.key_cos_emb = _cos_emb[:, :cache_length, :]
                self.key_sin_emb = _sin_emb[:, :cache_length, :]

            return

        if not is_symbolic(indices.shape):
            if self.implementation == "split_concat":
                if channels_first:
                    # _cos_emb = np.expand_dims(cos_emb.T, 0)  # (1, 64, length)
                    # _sin_emb = np.expand_dims(sin_emb.T, 0)

                    self.query_cos_emb = _cos_emb[:, :, : indices.shape[1]]
                    self.query_sin_emb = _sin_emb[:, :, : indices.shape[1]]
                    self.key_cos_emb = _cos_emb[:, :, : indices.shape[1]]
                    self.key_sin_emb = _sin_emb[:, :, : indices.shape[1]]
                else:
                    # _cos_emb = np.expand_dims(cos_emb, 0)  # (1, 64, length)
                    # _sin_emb = np.expand_dims(sin_emb, 0)

                    self.query_cos_emb = _cos_emb[:, : indices.shape[1], :]
                    self.query_sin_emb = _sin_emb[:, : indices.shape[1], :]
                    self.key_cos_emb = _cos_emb[:, : indices.shape[1], :]
                    self.key_sin_emb = _sin_emb[:, : indices.shape[1], :]
            return

        # ones = mb.fill_like(ref_tensor=input_ids, value=np.array(1, dtype=np.int32))
        # indices = mb.cumsum(x=ones, axis=1, exclusive=True)
        # qindices = mb.sub(x=np.array(2048, dtype=np.int32), y=indices)
        if self.implementation == "split_concat":
            # if channels_first:
            #     _cos_emb = np.expand_dims(cos_emb.T, 0)
            #     _sin_emb = np.expand_dims(sin_emb.T, 0)
            #     axis = 2
            # else:
            #     _cos_emb = np.expand_dims(cos_emb, 0)
            #     _sin_emb = np.expand_dims(sin_emb, 0)
            #     axis = 1

            self.query_cos_emb = mb.gather(
                x=_cos_emb,
                indices=qindices,
                axis=axis,
                batch_dims=1,
                name="rope_query_cos_emb",
            )
            self.query_sin_emb = mb.gather(
                x=_sin_emb,
                indices=qindices,
                axis=axis,
                batch_dims=1,
                name="rope_query_sin_emb",
            )
            self.key_cos_emb = mb.gather(
                x=_cos_emb,
                indices=indices,
                axis=axis,
                batch_dims=1,
                name="rope_key_cos_emb",
            )
            self.key_sin_emb = mb.gather(
                x=_sin_emb,
                indices=indices,
                axis=axis,
                batch_dims=1,
                name="rope_key_sin_emb",
            )

        elif self.implementation == "split_mul":
            cos_emb_1, cos_emb_2 = np.split(
                cos_emb.T.reshape(1, 64, 2048), indices_or_sections=2, axis=1
            )
            sin_emb_1, sin_emb_2 = np.split(
                sin_emb.T.reshape(1, 64, 2048), indices_or_sections=2, axis=1
            )

            self.query_cos_emb_1 = mb.gather(
                x=cos_emb_1, indices=qindices, axis=2, batch_dims=1
            )
            self.query_cos_emb_2 = mb.gather(
                x=cos_emb_2, indices=qindices, axis=2, batch_dims=1
            )
            self.query_sin_emb_1 = mb.gather(
                x=sin_emb_1, indices=qindices, axis=2, batch_dims=1
            )
            self.query_sin_emb_2 = mb.gather(
                x=sin_emb_2, indices=qindices, axis=2, batch_dims=1
            )

            self.key_cos_emb_1 = mb.gather(
                x=cos_emb_1, indices=indices, axis=2, batch_dims=1
            )
            self.key_cos_emb_2 = mb.gather(
                x=cos_emb_2, indices=indices, axis=2, batch_dims=1
            )
            self.key_sin_emb_1 = mb.gather(
                x=sin_emb_1, indices=indices, axis=2, batch_dims=1
            )
            self.key_sin_emb_2 = mb.gather(
                x=sin_emb_2, indices=indices, axis=2, batch_dims=1
            )
        elif self.implementation == "rotation_matrix":
            # Not working yet, I think is related to symbolic shape propagation
            _M = np.expand_dims(M, 0)
            self.Mq = mb.gather(x=_M, indices=qindices, axis=1, batch_dims=1)
            self.Mk = mb.gather(x=_M, indices=indices, axis=1, batch_dims=1)

    def __call__(self, query, key, layer_index, axis=-1):
        # query_len = query.shape[2]
        # key_len = key.shape[2]
        # This is for a fixed lenght (No EnumeratedShapes)
        # query_sin_emb = self.sin_emb[key_len - query_len : key_len]
        # query_cos_emb = self.cos_emb[key_len - query_len : key_len]
        # key_sin_emb = self.sin_emb[:key_len]
        # key_cos_emb = self.cos_emb[:key_len]

        # First we get the query length and key length
        # qlen = mb.gather(x=mb.shape(x=query), indices=[2])
        # klen = mb.gather(x=mb.shape(x=key), indices=[2])
        # Then we multiply the array [1, 1] by our lengths to get
        # an array of size 2 [len, len]
        # this array needs to have size 2 because
        # mb.slice_by_index expects ranges for all dimensions,
        # and sin and cos embs have shape [2048, hdim]
        # qlen = mb.mul(x=np.ones((2,), dtype=np.int32), y=qlen)
        # klen = mb.mul(x=np.ones((2,), dtype=np.int32), y=klen)

        # using tile should be the same, and maybe the ANE
        # cannot execute such small multiplications (???)
        # qlen = mb.tile(x=qlen, reps=[2])
        # klen = mb.tile(x=klen, reps=[2])

        # q0 = mb.sub(x=klen, y=qlen)  # RoPE Operation key_len - query_len
        # query_sin_emb = mb.slice_by_index(
        #     x=self.sin_emb,
        #     begin=q0,
        #     end=klen,
        #     # second dimension is the dim dimension, which we want to
        #     # slice completely. instead of making begin and end tensors with values
        #     # [start, 0], [start, dim], I use the mask parameters to ignore
        #     # the second element of the tensors
        #     # I don't know if this is faster or slower, just seemed
        #     # easier to implement
        #     begin_mask=[False, True],
        #     end_mask=[False, True],
        # )
        # query_cos_emb = mb.slice_by_index(
        #     x=self.cos_emb,
        #     begin=q0,
        #     end=klen,
        #     begin_mask=[False, True],
        #     end_mask=[False, True],
        # )
        # key_sin_emb = mb.slice_by_index(
        #     x=self.sin_emb,
        #     begin=[0, 0],
        #     end=klen,
        #     begin_mask=[False, True],
        #     end_mask=[False, True],
        # )
        # key_cos_emb = mb.slice_by_index(
        #     x=self.cos_emb,
        #     begin=[0, 0],
        #     end=klen,
        #     begin_mask=[False, True],
        #     end_mask=[False, True],
        # )

        return self.apply_rotary_pos_emb(
            query,
            self.query_sin_emb,
            self.query_cos_emb,
            prefix=f"attention_{layer_index}_rope_query",
            axis=axis,
        ), self.apply_rotary_pos_emb(
            key,
            self.key_sin_emb,
            self.key_cos_emb,
            prefix=f"attention_{layer_index}_rope_key",
            axis=axis,
        )

    @staticmethod
    def compute_rope_embedding(hdim, rope_freq_constant, length, dtype):
        inv_freq = 1.0 / (
            rope_freq_constant ** ((np.arange(0, hdim, 2, dtype=np.float32)) / hdim)
        )
        pos_index = np.arange(length, dtype=np.float32)
        pos_index_theta = np.einsum("i,j->ij", pos_index, inv_freq)
        emb = np.concatenate((pos_index_theta, pos_index_theta), axis=-1)
        cos_emb = np.cos(emb)
        _sin_emb = np.sin(pos_index_theta)
        sin_emb = np.sin(emb)

        M = (
            np.vectorize(np.diag, signature="(n)->(n,n)")(cos_emb)
            - np.vectorize(np.diag, signature="(n),()->(2n,2n)")(_sin_emb, hdim // 2)
            + np.vectorize(np.diag, signature="(n),()->(2n,2n)")(_sin_emb, -hdim // 2)
        )

        return (
            cos_emb.astype(dtype),  # .T.reshape(hdim, 1, length),
            sin_emb.astype(dtype),  # .T.reshape(hdim, 1, length),
            # np.expand_dims(M.astype(dtype).T, 0),
            M.astype(dtype).reshape(1, hdim, length, hdim),
        )

    @staticmethod
    def rotate_half(x, prefix=None, axis=-1):
        if types.builtin_to_string(x.dtype) == "fp16":
            mone = np.float16(-1.0)
        else:
            mone = np.float32(-1.0)

        x1, x2 = mb.split(
            x=x, num_splits=2, axis=axis, name=f"{prefix}_rotate_half_split"
        )
        neg_x2 = mb.mul(x=x2, y=mone, name=f"{prefix}_rotate_half_neg")
        return mb.concat(
            values=(neg_x2, x1), axis=axis, name=f"{prefix}_rotate_half_concat"
        )

    @staticmethod
    def apply_rotary_pos_emb(x, pos_sin, pos_cos, prefix=None, axis=-1):
        lhs = mb.mul(x=x, y=pos_cos, name=f"{prefix}_rope_lhs_mult")
        xrot = RoPEEmbedding.rotate_half(x, prefix, axis)
        rhs = mb.mul(x=xrot, y=pos_sin, name=f"{prefix}_rope_rhs_mult")
        return mb.add(x=lhs, y=rhs, name=f"{prefix}_rope")


class Mask:
    def __init__(self, max_length, dtype=np.float32):
        self.causal_mask = np.expand_dims(
            np.triu(np.full((max_length, max_length), -np.inf, dtype=dtype), 1),
            axis=0,
        )

    def get_mask(self, indices, static=True):
        if not static or is_symbolic(indices.shape):
            mask = mb.gather(
                x=self.causal_mask,
                indices=indices,
                axis=1,
                batch_dims=1 if static else 0,
                name="mask_gather_0",
            )
            if is_symbolic(indices.shape):
                mask = mb.gather(
                    x=mask, indices=indices, axis=2, batch_dims=1, name="mask_gather_1"
                )
            return mask
        else:
            seqlen = indices.shape[1]
            return self.causal_mask[:, :seqlen, :seqlen]


def attention(
    x,
    mask,
    hdim,
    nqheads,
    nkvheads,
    qkvproj: Linear,
    outproj: Linear,
    block_index,
    pos_embedding: RoPEEmbedding,
    qnorm,
    knorm,
    channels_first=True,
    pre_norm_and_pos=False,
    multi_query_head=True,
):
    if types.builtin_to_string(x.dtype) == "fp16":
        hscale = np.float16(hdim**-0.5)
    else:
        hscale = np.float32(hdim**-0.5)
    if channels_first:
        axis = 1
    else:
        axis = -1

    qkv = qkvproj(x)

    if multi_query_head:
        pass
    else:
        if pre_norm_and_pos:
            head_splits = mb.split(
                x=qkv,
                split_sizes=[nqheads * hdim, nkvheads * hdim, nkvheads * hdim],
                axis=axis,
            )
            qheads = mb.reshape(
                x=head_splits[0],
                shape=(
                    0,
                    nqheads,
                    hdim,
                ),
            )
        else:
            head_splits = mb.split(x=qkv, num_splits=nqheads + nkvheads * 2, axis=axis)
            qheads = head_splits[:nqheads]
            kheads = head_splits[nqheads : nqheads + nkvheads]
            vheads = head_splits[nqheads + nkvheads :]

    attns = []
    num_groups = nqheads // nkvheads
    # mask = None
    for i in range(nkvheads):
        kh = kheads[i]
        kh = knorm(kh, f"attention_{block_index}_k_{i}", dim=axis)
        if pos_embedding.implementation == "split_concat":
            kh = RoPEEmbedding.apply_rotary_pos_emb(
                x=kh,
                pos_sin=pos_embedding.key_sin_emb,
                pos_cos=pos_embedding.key_cos_emb,
                prefix=f"attention_{block_index}_k_{i}",
                axis=axis,
            )
            pass
        elif pos_embedding.implementation == "split_mul":
            kh1, kh2 = rope_rotate(
                kh,
                pos_sin_1=pos_embedding.key_sin_emb_1,
                pos_sin_2=pos_embedding.key_sin_emb_2,
                pos_cos_1=pos_embedding.key_cos_emb_1,
                pos_cos_2=pos_embedding.key_cos_emb_2,
                prefix=f"attention_{block_index}_k_{i}",
                axis=axis,
            )
        elif pos_embedding.implementation == "rotation_matrix":
            kh = mb.transpose(x=kh, perm=[0, 2, 1])
            kh = mb.expand_dims(x=kh, axes=[-1])
            kh = mb.matmul(x=pos_embedding.Mk, y=kh)
            # kh = mb.einsum(values=(Mk, kh), equation='nchw,nwhu->nchu')
            kh = mb.squeeze(x=kh, axes=[-1])
            kh = mb.transpose(x=kh, perm=[0, 2, 1])
        vh = vheads[i]
        for j in range(num_groups):  # to perform gqa
            q_head_num = i * num_groups + j
            qh = qheads[q_head_num]
            qh = qnorm(qh, f"attention_{block_index}_q_{q_head_num}", dim=axis)

            if (
                pos_embedding.implementation == "split_concat"
                or pos_embedding.implementation == "rotation_matrix"
            ):
                if pos_embedding.implementation == "rotation_matrix":
                    qh = mb.transpose(x=qh, perm=[0, 2, 1])
                    qh = mb.expand_dims(x=qh, axes=[-1])
                    qh = mb.matmul(x=pos_embedding.Mq, y=qh)
                    # qh = mb.einsum(values=(Mk, qh), equation='nchw,nwhu->nchu')
                    qh = mb.squeeze(x=qh, axes=[-1])
                    qh = mb.transpose(x=qh, perm=[0, 2, 1])
                else:
                    qh = RoPEEmbedding.apply_rotary_pos_emb(
                        x=qh,
                        pos_sin=pos_embedding.query_sin_emb,
                        pos_cos=pos_embedding.query_cos_emb,
                        prefix=f"attention_{block_index}_q_{q_head_num}",
                        axis=axis,
                    )
                    pass

                scores = mb.matmul(
                    x=qh,
                    y=kh,
                    transpose_x=True,
                    name=f"attention_{block_index}_scores_head_{q_head_num}",
                )  # (B, L, H_DIM) @ (B, H_DIM, L) = (B, L, L)
            elif pos_embedding.implementation == "split_mul":
                qh1, qh2 = rope_rotate(
                    qh,
                    pos_sin_1=pos_embedding.query_sin_emb_1,
                    pos_sin_2=pos_embedding.query_sin_emb_2,
                    pos_cos_1=pos_embedding.query_cos_emb_1,
                    pos_cos_2=pos_embedding.query_cos_emb_2,
                    prefix=f"attention_{block_index}_q_{q_head_num}",
                    axis=1,
                )
                scores_1 = mb.matmul(
                    x=qh1,
                    y=kh1,
                    transpose_x=True,
                    name=f"attention_{block_index}_rope_scores_1_head_{q_head_num}",
                )  # (B, L, H_DIM) @ (B, H_DIM, L) = (B, L, L)
                scores_2 = mb.matmul(
                    x=qh2,
                    y=kh2,
                    transpose_x=True,
                    name=f"attention_{block_index}_rope_scores_2_head_{q_head_num}",
                )  # (B, L, H_DIM) @ (B, H_DIM, L) = (B, L, L)
                scores = mb.add(x=scores_1, y=scores_2)

            scores = mb.mul(
                x=scores,
                y=hscale,
                name=f"attention_{block_index}.scores_head_{q_head_num}",
            )
            if mask is None:
                pass
                # mask = mb.sub(x=x, y=mb.reshape(x=indices, ))
                # mask = mb.fill_like(ref_tensor=scores, value=np.array(1, dtype=np.float16))
                # mask = mb.cast(x=mask, dtype='int16')
                # mask = mb.cumsum(x=mask, axis=1, exclusive=True)
                # mask = mb.sub(x=mask, y=indices)
                # mask = mb.mul(x=mask, y=np.array(-np.inf, dtype=np.float16))
            # return scores
            # scores = mb.add(
            #     x=scores,
            #     y=mask,
            #     name=f"attention_{block_index}_masked_scores_head_{q_head_num}",
            # )
            weights = mb.softmax(
                x=scores,
                name=f"attention_{block_index}_softmax_head_{q_head_num}",
            )
            attention = mb.matmul(
                x=vh,
                y=weights,
                # x=weights,
                # y=vh,
                # transpose_y=True,
                name=f"attention_{block_index}_attention_head_{q_head_num}",
            )  # (B, L, L) @ (B, L, H_DIM) = (B, L, H_DIM)

            attns.append(attention)

    attention = mb.concat(
        values=attns,
        axis=axis,
        name=f"attention_{block_index}",
    )

    out = outproj(attention)

    return out


def attention_monstrosity(
    qkv,
    mask,
    headdim,
    nqheads,
    nkvheads,
    qnorm: RMSNorm | None,
    knorm: RMSNorm | None,
    rope: RoPEEmbedding | None,
    channels_first,
    pre_normalization_and_pos_encoding,
    multi_query_head,
    repeat_interleave,
    block_index=0,
    key_cache=None,
    value_cache=None,
):
    num_groups = nqheads // nkvheads
    if channels_first:
        axis = 1
    else:
        axis = 2
    if types.builtin_to_string(qkv.dtype) == "fp16":
        hscale = np.float16(headdim**-0.5)
    else:
        hscale = np.float32(headdim**-0.5)

    if repeat_interleave:
        # newshape = np.array([0, nqheads, hdim, -1])
        # TEMP batch size 1, I think it should be possible to have flexible batch size and sequence length
        # by customizing the operation definition
        # https://github.com/apple/coremltools/blob/7521b68fba363d4add0c772750d119e4d9815ce6/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_transformation.py#L237
        if channels_first:
            newshape = np.array([1, nqheads + nkvheads * 2, headdim, -1])
            qkv = mb.reshape(
                x=qkv,
                shape=newshape,
                name=f"attention_{block_index}_all_reshaped_heads",
            )
        else:
            # newshape = np.array([1, nqheads + nkvheads * 2, -1, hdim])
            qkv = mb.reshape(
                x=qkv,
                shape=[1, -1, nqheads + nkvheads * 2, headdim],
                name=f"attention_{block_index}_all_reshaped_heads",
            )
            qkv = mb.transpose(
                x=qkv,
                perm=[0, 2, 1, 3],
                name=f"attention_{block_index}_all_reshaped_heads_T",
            )

        qheads, kheads, vheads = mb.split(
            x=qkv,
            split_sizes=[nqheads, nkvheads, nkvheads],
            axis=1,
            name=f"attention_{block_index}_split_reshaped_heads",
        )
        if qnorm:
            qheads = qnorm(qheads, prefix=f"attention_{block_index}_q", axes=[axis + 1])
        if knorm:
            kheads = knorm(kheads, prefix=f"attention_{block_index}_k", axes=[axis + 1])

        if rope and rope.implementation == "split_concat":
            qheads = rope.apply_rotary_pos_emb(
                x=qheads,
                pos_sin=rope.query_sin_emb,
                pos_cos=rope.query_cos_emb,
                prefix=f"attention_{block_index}_q",
                axis=axis + 1,
            )
            kheads = rope.apply_rotary_pos_emb(
                x=kheads,
                pos_sin=rope.key_sin_emb,
                pos_cos=rope.key_cos_emb,
                prefix=f"attention_{block_index}_k",
                axis=axis + 1,
            )

        if repeat_interleave:
            kheads = mb.split(x=kheads, num_splits=nkvheads, axis=1)
            vheads = mb.split(x=vheads, num_splits=nkvheads, axis=1)
            kheads = [_k for _k in kheads for _ in range(num_groups)]
            vheads = [_v for _v in vheads for _ in range(num_groups)]
            kheads = mb.concat(
                values=kheads, axis=1, name=f"attention_{block_index}_k_interleave"
            )
            vheads = mb.concat(
                values=vheads, axis=1, name=f"attention_{block_index}_v_interleave"
            )

            # attention = mb.scaled_dot_product_attention(
            #     query=qheads,
            #     key=kheads,
            #     value=vheads,
            # )

            # QKV: (batch size, heads, hdim, length) when channels first
            if channels_first:
                attention = mb.scaled_dot_product_attention(
                    query=mb.transpose(x=qheads, perm=[0, 1, 3, 2]),
                    key=mb.transpose(x=kheads, perm=[0, 1, 3, 2]),
                    value=mb.transpose(x=vheads, perm=[0, 1, 3, 2]),
                    attn_mask=mask,
                )

                # scores = mb.matmul(
                #     x=qheads,
                #     y=kheads,
                #     transpose_x=True,
                #     name=f"attention_{block_index}_scores",
                # )
                # sm_axis = -1
            else:
                attention = mb.scaled_dot_product_attention(
                    query=qheads,
                    key=kheads,
                    value=vheads,
                    attn_mask=mask,
                )
            #     scores = mb.matmul(
            #         x=qheads,
            #         y=kheads,
            #         transpose_y=True,
            #         name=f"attention_{block_index}_scores",
            #     )
            #     sm_axis = -1
            # scores = mb.mul(
            #     x=scores,
            #     y=hscale,
            #     name=f"attention_{block_index}_scaled_scores",
            # )
            # if mask is not None:
            #     scores = mb.add(
            #         x=scores,
            #         y=mask,
            #         name=f"attention_{block_index}_masked_scaled_scores",
            #     )
            # weights = mb.softmax(
            #     x=scores,
            #     name=f"attention_{block_index}_softmax",
            #     axis=sm_axis,
            # )

            # if channels_first:
            #     attention = mb.matmul(
            #         x=vheads,
            #         y=weights,
            #         transpose_x=False,
            #         transpose_y=True,
            #         name=f"attention_{block_index}_attention",
            #     )
            #     # attention = mb.transpose(x=attention, perm=[0, 1, 3, 2])
            # else:
            #     attention = mb.matmul(
            #         x=weights,
            #         y=vheads,
            #         # transpose_x=True,
            #         name=f"attention_{block_index}_attention",
            #     )

            if channels_first:
                newshape = np.array([1, nqheads * headdim, -1])
                # attention = mb.transpose(x=attention, perm=[0, 2, 1, 3], name=f"attention_{block_index}_attention_retransposed")
            else:
                newshape = np.array([1, -1, nqheads * headdim])
                attention = mb.transpose(
                    x=attention,
                    perm=[0, 2, 1, 3],
                    name=f"attention_{block_index}_attention_retransposed",
                )

            attention = mb.reshape(
                x=attention,
                shape=newshape,
                name=f"attention_{block_index}_shape_restored",
            )
            return attention, qheads, kheads
    elif pre_normalization_and_pos_encoding:
        if multi_query_head:
            if channels_first:
                newshape = np.array([1, nqheads + nkvheads * 2, headdim, -1])
                qkv = mb.reshape(
                    x=qkv,
                    shape=newshape,
                    name=f"attention_{block_index}_all_reshaped_heads",
                )
            else:
                # newshape = np.array([1, nqheads + nkvheads * 2, -1, hdim])
                qkv = mb.reshape(
                    x=qkv,
                    shape=[1, -1, nqheads + nkvheads * 2, headdim],
                    name=f"attention_{block_index}_all_reshaped_heads",
                )
                qkv = mb.transpose(
                    x=qkv,
                    perm=[0, 2, 1, 3],
                    name=f"attention_{block_index}_all_reshaped_heads_T",
                )

            qheads, kheads, vheads = mb.split(
                x=qkv,
                split_sizes=[nqheads, nkvheads, nkvheads],
                axis=1,
                name=f"attention_{block_index}_split_reshaped_heads",
            )
            if qnorm:
                qheads = qnorm(
                    qheads, prefix=f"attention_{block_index}_q", axes=[axis + 1]
                )
            if knorm:
                kheads = knorm(
                    kheads, prefix=f"attention_{block_index}_k", axes=[axis + 1]
                )

            if rope and rope.implementation == "split_concat":
                qheads = rope.apply_rotary_pos_emb(
                    x=qheads,
                    pos_sin=rope.query_sin_emb,
                    pos_cos=rope.query_cos_emb,
                    prefix=f"attention_{block_index}_q",
                    axis=axis + 1,
                )
                kheads = rope.apply_rotary_pos_emb(
                    x=kheads,
                    pos_sin=rope.key_sin_emb,
                    pos_cos=rope.key_cos_emb,
                    prefix=f"attention_{block_index}_k",
                    axis=axis + 1,
                )
            qheads = mb.split(
                x=qheads,
                num_splits=nkvheads,
                axis=1,
                name=f"attention_{block_index}_q_group_split",
            )
            qiter = 1

            kheads = mb.split(
                x=kheads,
                num_splits=nkvheads,
                axis=1,
                name=f"attention_{block_index}_k_head_split",
            )
            vheads = mb.split(
                x=vheads,
                num_splits=nkvheads,
                axis=1,
                name=f"attention_{block_index}_v_head_split",
            )

        else:
            qkheads, vheads = mb.split(
                x=qkv,
                axis=axis,
                split_sizes=[
                    nqheads * headdim + nkvheads * headdim,
                    nkvheads * headdim,
                ],
                name=f"attention_{block_index}_qk_v_split",
            )

            vheads = mb.split(
                x=vheads,
                split_sizes=[headdim] * nkvheads,
                axis=axis,
                name=f"attention_{block_index}_v_split_heads",
            )

            if channels_first:
                newshape = np.array([1, nqheads + nkvheads, headdim, -1])
                qkheads = mb.reshape(
                    x=qkheads,
                    shape=newshape,
                    name=f"attention_{block_index}_qk_reshaped_heads",
                )
            else:
                # newshape = np.array([1, nqheads + nkvheads * 2, -1, hdim])
                qkheads = mb.reshape(
                    x=qkheads,
                    shape=[1, -1, nqheads + nkvheads, headdim],
                    name=f"attention_{block_index}_qk_reshaped_heads",
                )
                qkheads = mb.transpose(
                    x=qkheads,
                    perm=[0, 2, 1, 3],
                    name=f"attention_{block_index}_qk_reshaped_heads_T",
                )

            qheads, kheads = mb.split(
                x=qkheads,
                split_sizes=[nqheads, nkvheads],
                axis=1,
                name=f"attention_{block_index}_q_k_heads",
            )

            if qnorm:
                qheads = qnorm(
                    qheads, prefix=f"attention_{block_index}_q", axes=[axis + 1]
                )
            if knorm:
                kheads = knorm(
                    kheads, prefix=f"attention_{block_index}_k", axes=[axis + 1]
                )

            if rope and rope.implementation == "split_concat":
                qheads = rope.apply_rotary_pos_emb(
                    x=qheads,
                    pos_sin=rope.query_sin_emb,
                    pos_cos=rope.query_cos_emb,
                    prefix=f"attention_{block_index}_q",
                    axis=axis + 1,
                )
                kheads = rope.apply_rotary_pos_emb(
                    x=kheads,
                    pos_sin=rope.key_sin_emb,
                    pos_cos=rope.key_cos_emb,
                    prefix=f"attention_{block_index}_k",
                    axis=axis + 1,
                )

            if channels_first:
                qheads = mb.reshape(
                    x=qheads,
                    shape=[1, nqheads * headdim, -1],
                )
                kheads = mb.reshape(
                    x=kheads,
                    shape=[1, nkvheads * headdim, -1],
                )
            else:
                qheads = mb.reshape(
                    x=qheads,
                    shape=[1, -1, nqheads * headdim],
                )
                kheads = mb.reshape(
                    x=kheads,
                    shape=[1, -1, nkvheads * headdim],
                )

            qheads = mb.split(
                x=qheads,
                num_splits=nqheads,
                axis=1,
                name=f"attention_{block_index}_q_head_split",
            )
            kheads = mb.split(
                x=kheads,
                num_splits=nkvheads,
                axis=1,
                name=f"attention_{block_index}_k_head_split",
            )
            qiter = num_groups
    else:
        if multi_query_head:
            head_splits = mb.split(
                x=qkv,
                split_sizes=[nqheads * headdim] + [headdim] * nkvheads * 2,
                axis=axis,
            )
            qheads = head_splits[0]
            if channels_first:
                qheads = mb.reshape(x=qheads, shape=[1, nqheads, headdim, -1])
            else:
                qheads = mb.reshape(x=qheads, shape=[1, -1, nqheads, headdim])
                qheads = mb.transpose(x=qheads, perm=[0, 2, 1, 3])
            qheads = mb.split(x=qheads, num_splits=nkvheads, axis=1)
            kheads = head_splits[1 : 1 + nkvheads]
            vheads = head_splits[-nkvheads:]
            qiter = 1
        else:
            head_splits = mb.split(x=qkv, num_splits=nqheads + nkvheads * 2, axis=axis)
            qheads = head_splits[:nqheads]
            kheads = head_splits[nqheads : nqheads + nkvheads]
            vheads = head_splits[nqheads + nkvheads :]
            qiter = num_groups

    attns = []
    for i in range(nkvheads):
        kh = kheads[i]
        # if pre_normalization_and_pos_encoding and not multi_query_head:
        #     kh = mb.gather(x=kheads, axis=1, indices=[i], batch_dims=1)
        if not pre_normalization_and_pos_encoding:
            axis = 1 if channels_first else 2
            if knorm:
                kh = knorm(kh, f"attention_{block_index}_k_{i}", axes=[axis])
            if rope and rope.implementation == "split_concat":
                kh = RoPEEmbedding.apply_rotary_pos_emb(
                    x=kh,
                    pos_sin=rope.key_sin_emb,
                    pos_cos=rope.key_cos_emb,
                    prefix=f"attention_{block_index}_k_{i}",
                    axis=axis,
                )
        # else:
        #     kh = kheads[i]

        vh = vheads[i]
        # if pre_normalization_and_pos_encoding and not multi_query_head:
        #     vh = mb.squeeze(x=vh, axes=[1])
        for j in range(qiter):
            if multi_query_head:
                q_head_num = i
            else:
                q_head_num = i * num_groups + j

            qh = qheads[q_head_num]
            # if pre_normalization_and_pos_encoding and not multi_query_head:
            #     qh = mb.gather(x=qheads, axis=1, indices=[q_head_num], batch_dims=1)
            if not pre_normalization_and_pos_encoding:
                if channels_first and multi_query_head:  # (batch, head, hdim, len)
                    axis = 2
                elif channels_first:  # (batch, hdim, len)
                    axis = 1
                elif multi_query_head:  # (batch, head, len, hdim)
                    axis = 3
                else:  # (batch, len, hdim)
                    axis = 2

                if qnorm:
                    qh = qnorm(
                        qh, f"attention_{block_index}_q_{q_head_num}", axes=[axis]
                    )

                if rope and rope.implementation == "split_concat":
                    qh = RoPEEmbedding.apply_rotary_pos_emb(
                        x=qh,
                        pos_sin=rope.query_sin_emb,
                        pos_cos=rope.query_cos_emb,
                        prefix=f"attention_{block_index}_q_{q_head_num}",
                        axis=axis,
                    )
            # else:
            #     qh = qheads[q_head_num]

            scores = mb.matmul(
                x=qh,
                y=kh,
                transpose_x=channels_first,
                transpose_y=not channels_first,
                name=f"attention_{block_index}_scores_head_{q_head_num}",
            )  # (B, L, H_DIM) @ (B, H_DIM, L) = (B, L, L)
            scores = mb.mul(
                x=scores,
                y=hscale,
                name=f"attention_{block_index}_scaled_scores_head_{q_head_num}",
            )
            # if mask is None:
            #     pass
            # mask = mb.sub(x=x, y=mb.reshape(x=indices, ))
            # mask = mb.fill_like(ref_tensor=scores, value=np.array(1, dtype=np.float16))
            # mask = mb.cast(x=mask, dtype='int16')
            # mask = mb.cumsum(x=mask, axis=1, exclusive=True)
            # mask = mb.sub(x=mask, y=indices)
            # mask = mb.mul(x=mask, y=np.array(-np.inf, dtype=np.float16))
            # return scores
            if mask is not None:
                scores = mb.add(
                    x=scores,
                    y=mask,
                    name=f"attention_{block_index}_masked_scaled_scores_head_{q_head_num}",
                )
            weights = mb.softmax(
                x=scores,
                name=f"attention_{block_index}_softmax_head_{q_head_num}",
            )
            if channels_first:
                attention = mb.matmul(
                    x=vh,
                    y=weights,
                    transpose_y=True,  # I think weights may have to be transposed
                    name=f"attention_{block_index}_attention_{q_head_num}",
                )
            else:
                attention = mb.matmul(
                    x=weights,
                    y=vh,
                    transpose_y=False,
                    name=f"attention_{block_index}_attention_{q_head_num}",
                )

            attns.append(attention)

    if multi_query_head:
        attention = mb.concat(
            values=attns, axis=1, name=f"attention_{block_index}_concat_heads"
        )
        if channels_first:
            attention = mb.reshape(x=attention, shape=[1, nqheads * headdim, -1])
        else:
            attention = mb.transpose(x=attention, perm=[0, 2, 1, 3])
            attention = mb.reshape(x=attention, shape=[1, -1, nqheads * headdim])
    else:
        if channels_first:
            axis = 1
        else:
            axis = 2
        # axis += 1 if pre_normalization_and_pos_encoding else 0
        attention = mb.concat(
            values=attns,
            axis=axis,
            name=f"attention_{block_index}",
        )
        # if pre_normalization_and_pos_encoding:
        #     attention = mb.squeeze(
        #         x=attention, axes=[1], name=f"attention_{block_index}_squeezed"
        #     )

    return (attention,)


def stateful_attention(
    qkv,
    mask,
    query_pos,
    split_key_state,
    split_value_state,
    key_state,
    value_state,
    headdim,
    nqheads,
    nkvheads,
    update_index,
    qnorm: RMSNorm | None = None,
    knorm: RMSNorm | None = None,
    query_sin_emb=None,
    query_cos_emb=None,
    channels_first=False,
    block_index=0,
    state_implementation="",
    state_update_at="",
    # rope: RoPEEmbedding | None = None,
):
    if types.builtin_to_string(qkv.dtype) == "fp16":
        dtype = np.float16
    else:
        dtype = np.float32
    hscale = np.array(headdim**-0.5, dtype=dtype)
    num_groups = nqheads // nkvheads

    if channels_first:
        qkv = mb.reshape(
            x=qkv,
            shape=[1, nqheads + nkvheads * 2, 64, 1],
            name=f"attention_{block_index}_head_reshape",
        )
        qkv = mb.transpose(
            x=qkv, perm=[0, 1, 3, 2], name=f"attention_{block_index}_head_transpose"
        )
    else:
        qkv = mb.reshape(
            x=qkv,
            shape=[1, 1, nqheads + nkvheads * 2, 64],
            name=f"attention_{block_index}_head_reshape",
        )
        qkv = mb.transpose(
            x=qkv, perm=[0, 2, 1, 3], name=f"attention_{block_index}_head_transpose"
        )

    new_qheads, new_kheads, new_vheads = mb.split(
        x=qkv,
        split_sizes=[nqheads, nkvheads, nkvheads],
        axis=1,
        name=f"attention_{block_index}_head_split",
    )
    if qnorm is not None:
        new_qheads = qnorm(
            new_qheads, axes=[3], prefix=f"attention_{block_index}_q", squeeze=True
        )
    if knorm is not None:
        new_kheads = knorm(
            new_kheads, axes=[3], prefix=f"attention_{block_index}_k", squeeze=True
        )
    if query_sin_emb is not None:
        new_qheads = RoPEEmbedding.apply_rotary_pos_emb(
            new_qheads,
            query_sin_emb,
            query_cos_emb,
            prefix=f"attention_{block_index}_q",
        )
        new_kheads = RoPEEmbedding.apply_rotary_pos_emb(
            new_kheads,
            query_sin_emb,
            query_cos_emb,
            prefix=f"attention_{block_index}_k",
        )

    # _kheads = mb.read_state(
    #     input=key_state,
    # )
    # _vheads = mb.read_state(input=value_state)

    # scatter does not work on ANE
    # _kheads = mb.scatter(
    #     data=_kheads,
    #     updates=new_kheads,
    #     indices=query_pos,
    #     axis=2,
    #     name=f"attention_{block_index}_k_scatter",
    # )
    # _vheads = mb.scatter(
    #     data=_vheads,
    #     updates=new_vheads,
    #     indices=query_pos,
    #     axis=2,
    #     name=f"attention_{block_index}_v_scatter",
    # )

    _kheads = mb.scatter(
        data=split_key_state, updates=new_kheads, indices=query_pos, axis=2
    )
    _vheads = mb.scatter(
        data=split_value_state, updates=new_vheads, indices=query_pos, axis=2
    )

    # Nor does slice_update
    # begin = mb.repeat()
    # _kheads = mb.slice_update(x=_kheads, update=new_kheads, begin=[0, 0, 0, 0], end=[1, 3, 1, 64], begin_mask=[True, True, False, True], end_mask=[])
    # _vheads = mb.slice_update(x=_vheads, update=new_vheads, begin=[0, 0, 0, 0], end=[1, 3, 1, 64])

    ## But neither these operations stop ANE flow, computation continues on it, worst case all
    ## the state updates can be moved at the end of the program
    ## scatter seemed a bit simpler to implement, slice_update would have had to repeat query_pos 4 times
    ## and use as begin tensor, and create another adding 1 to use ase end tensor. Also would have needed to
    ## use begin_mask and end_mask. Timewise they seem very close, 2us in favor of slice_update, with
    ## hardcoded begin and end, without performing tile nor addition

    # For some reason updating here causes conversion to crash, thus the use of _kheads, kheads, _vheads and vheads
    # mb.coreml_update_state(state=key_states, value=kheads)
    # mb.coreml_update_state(state=value_states, value=vheads)

    # begin = np.array([0, update_index * 3, 0, 0])
    # end = np.array([-1, (update_index + 1) * 3, 1, -1])
    # add = np.array([0, 0, 1, 0])
    # update_mask = [True, False, False, True]
    # add = mb.mul(x=add, y=query_pos)
    # begin = mb.add(x=begin, y=add)
    # end = mb.add(x=end, y=add)

    # all_kheads = mb.slice_update(
    #     x=key_state[1],
    #     update=new_kheads,
    #     begin=begin,
    #     end=end,
    #     begin_mask=update_mask,
    #     end_mask=update_mask,
    # )
    # all_vheads = mb.slice_update(
    #     x=value_state[1],
    #     update=new_vheads,
    #     begin=begin,
    #     end=end,
    #     begin_mask=update_mask,
    #     end_mask=update_mask,
    # )

    if num_groups > 1:
        kheads = mb.split(
            x=_kheads,
            num_splits=nkvheads,
            axis=1,
            name=f"attention_{block_index}_k_interleave_split",
        )
        vheads = mb.split(
            x=_vheads,
            num_splits=nkvheads,
            axis=1,
            name=f"attention_{block_index}_v_interleave_split",
        )
        kheads = [_k for _k in kheads for _ in range(num_groups)]
        vheads = [_v for _v in vheads for _ in range(num_groups)]
        kheads = mb.concat(
            values=kheads, axis=1, name=f"attention_{block_index}_k_interleave_concat"
        )
        vheads = mb.concat(
            values=vheads, axis=1, name=f"attention_{block_index}_v_interleave_concat"
        )
    else:
        kheads = _kheads
        vheads = _vheads

    attention = mb.scaled_dot_product_attention(
        query=new_qheads,
        key=kheads,
        value=vheads,
        attn_mask=mask,
        name=f"attention_{block_index}_sdpa",
    )


    # mb.coreml_update_state(state=key_state[0], value=all_kheads)
    # mb.coreml_update_state(state=value_state[0], value=all_vheads)

    if channels_first:
        attention = mb.transpose(
            x=attention, perm=[0, 1, 3, 2], name=f"attention_{block_index}_out_tranpose"
        )
        attention = mb.reshape(
            x=attention,
            shape=[1, nqheads * headdim, 1],
            name=f"attention_{block_index}_out_reshape",
        )
    else:
        attention = mb.transpose(
            x=attention,
            perm=[0, 2, 1, 3],
            name=f"attention_{block_index}_out_transpose",
        )
        attention = mb.reshape(
            x=attention,
            shape=[1, 1, nqheads * headdim],
            name=f"attention_{block_index}_out_reshape",
        )


    if state_implementation == "per_block":
        return attention , _kheads, _vheads
    return (attention, new_kheads, new_vheads)  # , kheads, vheads


6
