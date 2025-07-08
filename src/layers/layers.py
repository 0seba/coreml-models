from typing import List

import math
import numpy as np
from coremltools.converters.mil import Builder as mb
import coremltools.converters.mil as mil
from coremltools.converters.mil.mil import Operation, Var, types

# from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
# from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET
# from .custom_conv import conv
# register_op(conv, opset_version=_IOS17_TARGET, allow_override=True)  #


class NamedCall:
    def __init__(self, name: str | None = None):
        self._name = name

    def name(self, name=None):
        return self._name if name is None else name


class WbGeneric(NamedCall):
    op: None = None

    def __init__(
        self, w: np.ndarray, b: np.ndarray | None = None, name: None | str = None
    ):
        super().__init__(name)
        self.w = w
        self.b = b

    def __call__(self, x, name=None):
        assert self.op, NotImplementedError("Undefined operation op")
        if self.b is not None:
            return self.op(x=x, weight=self.w, bias=self.b, name=self.name(name))

        return self.op(x=x, weight=self.w, name=self.name(name))


class Linear(WbGeneric):
    op = mb.linear


class Conv(WbGeneric):
    op = mb.conv


class LUTConv(NamedCall):
    def __init__(
        self,
        w: np.ndarray,
        lut: np.ndarray,
        b: np.ndarray | None = None,
        name: None | str = None,
    ):
        super().__init__(name)
        self.w = w
        self.lut = lut
        self.b = b

    def __call__(self, x, name=None):
        w = mb.constexpr_lut_to_dense(
            indices=self.w,
            lut=self.lut,
            name=self.name(name) + "_weight_dequantization",
        )
        w = mb.expand_dims(x=w, axes=[-2, -1])
        x = mb.expand_dims(x=x, axes=[-1])
        if self.b is not None:
            x = mb.conv(x=x, weight=w, bias=self.b, name=self.name(name))
        else:
            x = mb.conv(x=x, weight=w, name=self.name(name))
        x = mb.squeeze(x=x, axes=[-1])
        return x


class GQLinear(NamedCall):
    def __init__(
        self,
        w,
        bits,
        # mode,
        scales=None,
        biases=None,
        bias=None,
        name=None,
    ):
        super().__init__(name)
        # self.bits = bits
        self.bits = 8
        self.bias = bias
        # self.mode = mode

        if scales is not None:
            self.group_size = w.shape[1] // scales.shape[1]
            self.num_groups = scales.shape[1]
        elif biases is not None:
            self.group_size = w.shape[1] // biases.shape[1]
            self.num_groups = scales.shape[1]
        else:
            self.group_size = None
            self.num_groups = None

        self.w = w.reshape(w.shape[0], self.num_groups, self.group_size)
        self.scales = np.expand_dims(scales, -1)
        self.biases = np.expand_dims(biases, -1)

    def __call__(self, x, name=None):
        # lut = np.arange(0, 2**self.bits, dtype=np.float16)
        # lut = lut * np.expand_dims(np.array(self.scales), -1) + np.expand_dims(
        #     np.array(self.biases), -1
        # )
        # lut = np.expand_dims(lut, (2, -1))
        # _w = mb.constexpr_lut_to_dense(
        #     indices=np.reshape(
        #         self.w, (self.w.shape[0], self.num_groups, self.group_size)
        #     ),
        #     lut=lut,
        # )
        _w = mb.constexpr_blockwise_shift_scale(
            data=self.w,
            scale=self.scales,
            offset=self.biases,
        )
        _w = mb.reshape(x=_w, shape=(_w.shape[0], self.group_size * self.num_groups))
        # _w = mb.flatten2d(x=_w)
        # if self.scales is not None:
        #     scales = mb.tile(
        #         x=np.array(self.scales)[
        #             ...,
        #             None,
        #         ],
        #         reps=[1, 1, self.group_size],
        #     )
        #     scales = mb.reshape(x=scales, shape=(scales.shape[0], -1, 1, 1))
        #     _w = mb.mul(x=_w, y=scales)
        # if self.biases is not None:
        #     biases = mb.tile(
        #         x=np.array(self.biases)[
        #             ...,
        #             None,
        #         ],
        #         reps=[1, 1, 64],
        #     )
        #     biases = mb.reshape(x=biases, shape=(biases.shape[0], -1, 1, 1))
        #     _w = mb.add(x=_w, y=biases)
        # # x = mb.matmul(x=x, y=_w, transpose_y=True)
        # print(_w.shape)
        # x = mb.expand_dims(x=x, axes=[-1])
        if self.bias is not None:
            # x = mb.conv(x=x, weight=_w, bias=self.bias, name=self.name(name))
            x = mb.linear(x=x, weight=_w, bias=self.bias, name=self.name(name))
        else:
            # x = mb.conv(x=x, weight=_w, name=self.name(name))
            x = mb.linear(x=x, weight=_w, name=self.name(name))
        # x = mb.squeeze(x=x, axes=[-1])
        return x


class LUTLinear(NamedCall):
    def __init__(
        self,
        w: np.ndarray,
        lut: np.ndarray,
        bits,
        s: np.ndarray | None = None,
        b: np.ndarray | None = None,
        channels_first=True,
        name: None | str = None,
    ):
        super().__init__(name)
        # bits = int(np.log2(lut.shape[-2]))
        assert 2**bits == lut.shape[-2]
        if bits == 1:
            self.w = np.array(w).astype(mil.mil.types.np_uint1_dtype)
        elif bits == 2:
            self.w = np.array(w).astype(mil.mil.types.np_uint2_dtype)
        elif bits == 3:
            self.w = np.array(w).astype(mil.mil.types.np_uint3_dtype)
        elif bits == 4:
            self.w = np.array(w).astype(mil.mil.types.np_uint4_dtype)
        elif bits == 6:
            self.w = np.array(w).astype(mil.mil.types.np_uint6_dtype)
        else:
            self.w = w
        self.lut = lut
        self.b = b
        self.s = s
        self.channels_first = channels_first
        if channels_first:
            self.op = mb.conv
        else:
            self.op = mb.linear

    def __call__(self, x, name=None, vector_axis=None):
        w = mb.constexpr_lut_to_dense(
            indices=self.w,
            lut=self.lut,
            name=self.name(name) + "lut_dequantization",
            vector_axis=vector_axis,
        )
        if self.s is not None:
            w = mb.constexpr_blockwise_shift_scale(
                data=w,
                scale=self.s,
                name=self.name(name) + "weight_shift_scale",
            )
        if self.b is not None:
            x = self.op(x=x, weight=w, bias=self.b, name=self.name(name))
        else:
            x = self.op(x=x, weight=w, name=self.name(name))
        return x


class QEmbedding(NamedCall):
    def __init__(
        self,
        w: np.ndarray,
        lut: np.ndarray,
        nbits,
        name=None,
        channels_first=False,
        scales=None,
    ):
        super().__init__(name)
        self.w = w
        self.lut = lut
        self.nbits = nbits
        self.channels_first = channels_first
        self.scales = scales

    def __call__(self, indices, name=None, channels_first=None):
        if channels_first is None:
            channels_first = self.channels_first
        w = mb.constexpr_lut_to_dense(
            indices=self.w,
            lut=self.lut,  # name=self.name(name) + "_dequantize"
        )
        if self.scales is not None:
            w = mb.constexpr_blockwise_shift_scale(data=w, scale=self.scales)
        x = mb.gather(
            x=w,
            indices=indices,
            axis=0,
            name=self.name(name),
        )
        if channels_first:
            x = mb.transpose(
                x=x,
                perm=[0, 2, 1],
                name="input_embeddings_channels_first",
            )
        return x

        # TODO: perform gather without dequantizing all weights
        # if nbits = 4 we have to use a packed w, gather on int4 is not supported
        # x = mb.gather(
        #     x=self.w,
        #     indices=indices,
        #     axis=0,
        #     # name=self.name(name),
        # )
        # lut = mb.gather(
        #     x=self.lut,
        #     indices=x,
        #     axis=0,
        #     # name=self.name(name) + "_lut",
        # )
        # x = mb.constexpr_lut_to_dense(
        #     indices=x, lut=lut, # name=self.name(name) + "_dequantize"
        # )
        # return x


class Embedding(NamedCall):
    def __init__(
        self, w: np.ndarray, name=None, validate_indices=False, channels_first=False
    ):
        super().__init__(name)
        self.w = w
        self.validate_indices = validate_indices
        self.channels_first = channels_first

    def __call__(self, x, name=None, channels_first=None):
        if channels_first is None:
            channels_first = self.channels_first
        # gather_nd is incredibly slow
        # max_size = 2048
        # chunk_idx = mb.floor_div(x=x, y=max_size)
        # in_chunk_index = mb.mod(x=x, y=max_size)
        # index = mb.stack(
        #     values=(
        #         np.zeros(x.shape, dtype=np.int32),
        #         chunk_idx,
        #         in_chunk_index,
        #     ),
        #     axis=2,
        # )
        # print('EMB', self.w.shape, index.shape)
        # x = mb.gather_nd(x=self.w, indices=index, name=self.name(name), batch_dims=0)

        # if x.shape == (1, 1):
        #     max_size = 2048
        #     x = mb.squeeze(x=x)
        #     chunk_idx = mb.floor_div(x=x, y=max_size)
        #     in_chunk_index = mb.mod(x=x, y=max_size)
        #     begin = mb.stack(
        #         values=(
        #             chunk_idx,
        #             in_chunk_index,
        #             np.array(0, dtype=np.int32),
        #             # np.array(0, dtype=np.int32),
        #         ),
        #         axis=0,
        #     )
        #     begin = mb.cast(x=begin, dtype="int16")
        #     # end = np.array([0, 0, -1], dtype=np.int16)
        #     size = np.array([1, 1, self.w.shape[-1]], dtype=np.int16)
        #     # x = mb.slice_by_index(
        #     x = mb.slice_by_size(
        #         x=self.w,
        #         begin=begin,
        #         # end=end,
        #         size=size,
        #         # squeeze_mask=[True, True, False],
        #     )
        #     return x
        # return mb.squeeze(x=x, axes=[-1])

        x = mb.gather(
            # x=np.expand_dims(self.w.T, 0),
            x=self.w,
            indices=x,
            axis=0,
            name=self.name(name),
            # batch_dims=1,
            # validate_indices=self.validate_indices,
        )
        if self.channels_first:
            # x = mb.squeeze(x=x, axes=[-1])
            x = mb.transpose(
                x=x,
                perm=[0, 2, 1],
                name="input_embeddings_channels_first",
            )
        return x


class RMSNorm(NamedCall):
    def __init__(
        self, w: np.ndarray, eps: float | np.ndarray, axes: List[int], name=None
    ):
        super().__init__(name)
        self.w = w
        self.eps = np.array(eps)
        self.beta = np.inf
        self.axes = axes

    @staticmethod
    def stable_low_precision_normalize(x, eps, dimroot, prefix, axes):
        if types.builtin_to_string(x.dtype) == "fp16":
            dtype = np.float16
        else:
            dtype = np.float32
        eps = np.array(eps, dtype=dtype)
        beta = np.array(np.inf, dtype=dtype)
        # dimroot = np.array(dimroot, dtype=dtype)
        maxval = mb.abs(x=x, name=f"{prefix}rmsnorm_abs")
        maxval = mb.reduce_max(
            x=maxval, axes=axes, keep_dims=True, name=f"{prefix}rmsnorm_maxval"
        )
        maxval = mb.clip(
            x=maxval, alpha=eps, beta=beta, name=f"{prefix}rmsnorm_maxval_clipped"
        )
        xscaled = mb.real_div(x=x, y=maxval, name=f"{prefix}rmsnorm_scaled")

        # norm = mb.reduce_l2_norm(
        #     x=x, axes=axes, keep_dims=True, name=f"{prefix}_rmsnorm_norm"
        # ) # Not supported by ANE
        # # norm = mb.add(x=norm, y=eps, name=f"{prefix}_rmsnorm_clipped_norm")
        # norm = mb.clip(
        #     x=norm, alpha=eps, beta=beta, name=f"{prefix}_rmsnorm_norm_clipped"
        # )

        # Seems like reduce_l2_norm does not work on ANE, so we split in separate ops
        sq_sum = mb.reduce_sum_square(
            x=xscaled, axes=axes, keep_dims=True, name=f"{prefix}rmsnorm_squared_sum"
        )
        rsqrt = mb.rsqrt(x=sq_sum, epsilon=eps, name=f"{prefix}rmsnorm_rsqrt")
        xscaled = mb.mul(
            x=xscaled, y=dimroot.astype(dtype), name=f"{prefix}rmsnorm_dim_scaled"
        )
        # xscaled = mb.mul(x=x, y=dimroot.astype(dtype), name=f"{prefix}_rmsnorm_dim_scaled")
        # xnormed = mb.real_div(x=xscaled, y=norm, name=f"{prefix}_rmsnorm_normalized")
        xnormed = mb.mul(x=xscaled, y=rsqrt, name=f"{prefix}rmsnorm_normalized")
        return xnormed

    @staticmethod
    def normalize(x, eps, dimroot, prefix, axes):
        squared = mb.reduce_sum_square(
            x=x, axes=axes, keep_dims=True, name=f"{prefix}rmsnorm_squared_sum"
        )
        # squared_mean = mb.real_div(
        #     x=squared,
        #     y=np.array(x.shape[-1], dtype=np.float32),
        #     name=f"{prefix}_rmsnorm_squared_mean",
        # )
        norm_reciprocal = mb.rsqrt(
            x=squared, epsilon=eps, name=f"{prefix}rmsnorm_norm_reciprocal"
        )
        rmsnorm_reciprocal = mb.mul(
            x=norm_reciprocal, y=dimroot, name=f"{prefix}rmsnorm_rmsnorm_reciprocal"
        )
        return mb.mul(x=x, y=rmsnorm_reciprocal, name=f"{prefix}rmsnorm_normalized")

    def __call__(self, x, prefix=None, w=None, axes=None, squeeze=False):
        if axes is None:
            axes = self.axes
        shape = x.shape
        dims = [shape[i] for i in axes]  # x.shape[*axes] does not work
        dimroot = np.sqrt(np.prod(dims))
        if types.builtin_to_string(x.dtype) == "fp16":
            xnormed = RMSNorm.stable_low_precision_normalize(
                x, self.eps, dimroot, prefix, axes
            )
        else:
            xnormed = RMSNorm.normalize(x, self.eps, dimroot, prefix, axes)

        w = self.w if w is None else w
        if w is not None:
            if squeeze:  # Quick fix
                w = w.squeeze()
            return mb.mul(x=xnormed, y=w, name=f"{prefix}rmsnorm")
        return xnormed


class FFN:
    def __init__(
        self,
        win: Linear,
        wout: Linear,
        activation,
        is_glu: bool,
        prefix: str,
        axis: int = -1,
    ):
        self.win = win
        self.wout = wout
        self.activation = activation
        self.is_glu = is_glu
        self.prefix = prefix
        self.axis = axis

    def __call__(self, x, prefix=None, axis=None):
        if prefix is None:
            prefix = self.prefix
        if axis is None:
            axis = self.axis
        x = self.win(x, name=f"{prefix}_ffn_inproj")
        if self.is_glu:
            g, x = mb.split(
                x=x,
                num_splits=2,
                axis=axis,
                name=f"{prefix}_ffn_xg_split",
            )
            g = self.activation(x=g, name=f"{prefix}_ffn_g_activation")
            x = mb.mul(x=x, y=g, name=f"{prefix}_ffn_x_gated")
        else:
            x = self.activation(x=x, name=f"{prefix}_ffn_x_activation")
        x = self.wout(x, name=f"{prefix}_ffn_outproj")
        return x


class FFN2:
    def __init__(
        self,
        win: Linear,
        wg: Linear,
        wout: Linear,
        activation,
        is_glu: bool,
        prefix: str,
        axis: int = -1,
    ):
        self.win = win
        self.wg = wg
        self.wout = wout
        self.activation = activation
        self.is_glu = is_glu
        self.prefix = prefix
        self.axis = axis

    def __call__(self, x, prefix=None, axis=None):
        if prefix is None:
            prefix = self.prefix
        if axis is None:
            axis = self.axis
        intermediate = self.win(x, name=f"{prefix}_ffn_inproj")
        g = self.wg(x, f"{prefix}_ffn_g")
        g = self.activation(x=g, name=f"{prefix}_ffn_g_activation")
        x = mb.mul(x=intermediate, y=g, name=f"{prefix}_ffn_x_gated")
        x = self.wout(x, name=f"{prefix}_ffn_outproj")
        return x


class QHead(NamedCall):
    def __init__(
        self,
        w,
        lut,
        channels_first,
        s=None,
        max_size=16384,
        name=None,
        return_logsumexp=True,
    ):
        super().__init__(name)
        group_size = w.shape[0] // lut.shape[0]
        if w.shape[0] % max_size == 0:
            ios_1 = ios_2 = w.shape[0] // max_size
        else:
            ios_1 = np.cumsum([max_size] * (w.shape[0] // max_size))
            ios_2 = ios_1 // group_size

        self.w = np.split(w, indices_or_sections=ios_1)
        self.lut = np.split(lut, indices_or_sections=ios_2)
        self.channels_first = channels_first
        self.return_logsumexp = return_logsumexp
        if s is not None:
            self.s = np.split(s, indices_or_sections=ios_1)
        else:
            self.s = None

    def __call__(self, x, return_logsumexp=None):
        if return_logsumexp is None:
            return_logsumexp = self.return_logsumexp
        if self.channels_first:
            axis = 1
        else:
            axis = -1
        logits = []
        logsumexps = []
        for i, (w, lut) in enumerate(zip(self.w, self.lut)):
            w = mb.constexpr_lut_to_dense(
                indices=w, lut=lut, name=f"{self.name()}_w_dequantizations_{i}"
            )
            print(x.shape, w.shape, lut.shape)
            if self.s is not None:
                print(self.s[i].shape)
                w = mb.constexpr_blockwise_shift_scale(
                    data=w,
                    scale=self.s[i],
                )
            print(w.shape)
            if self.channels_first:
                _x = mb.conv(x=x, weight=w, name=f"{self.name()}_conv_{i}")
            else:
                _x = mb.linear(x=x, weight=w, name=f"{self.name()}_linear_{i}")
            if return_logsumexp:
                m = mb.reduce_max(x=_x, axes=[axis], keep_dims=True)
                x_m = mb.sub(x=_x, y=m)
                lse = mb.reduce_log_sum_exp(x=x_m, axes=[axis], keep_dims=True)
                lse = mb.add(x=m, y=lse)
                logsumexps.append(lse)
            logits.append(_x)

        output = (mb.concat(values=logits, axis=axis),)

        if return_logsumexp:
            lse = mb.concat(values=logsumexps, axis=axis)
            m = mb.reduce_max(x=lse, axes=[axis], keep_dims=True)
            lse_m = mb.sub(x=lse, y=m)
            lse = mb.reduce_log_sum_exp(x=lse_m, axes=[axis], keep_dims=True)
            lse = mb.add(x=m, y=lse, name="lse")
            output += (lse,)

        return output


class Head:
    ## TODO: using 64 padded vocab w is slightly faster 5%, have to find how to
    # fill result of output matmul with padded values with -inf and if that is faster
    # mb.gather or something of that kind
    def __init__(
        self,
        w: np.ndarray,
        split_size,
        channels_first: bool,
        topk=0,  # Seems that topk is not supported by ANE
        return_logits=True,
        cast=True,
        prefix=None,
        final_cat=True,
        return_logsumexp=True,
    ):
        self.w = w  # has same shape as input embeddings, (vocab_size, hidden dim)
        self.channels_first = channels_first
        self.nsplits = math.ceil(w.shape[0] / split_size)
        self.topk = topk
        # self.padsize = w.shape[0] - vocab_size
        # self.vocab_size = vocab_size
        split_sizes = [split_size for _ in range(w.shape[0] // split_size)]
        if w.shape[0] % split_size > 0:
            split_sizes.append(w.shape[0] % split_size)
        self.split_sizes = np.array(split_sizes, dtype=np.int32)
        self.return_logits = return_logits
        self.cast = cast
        self.prefix = prefix
        self.final_cat = final_cat
        self.return_logsumexp = return_logsumexp

    def __call__(self, x, prefix=None, channels_first=None, return_logsumexp=None):
        if return_logsumexp is None:
            return_logsumexp = self.return_logsumexp
        if prefix is None:
            prefix = self.prefix
        if channels_first is None:
            channels_first = self.channels_first

        axis = 1 if channels_first else 1

        # x = mb.expand_dims(x=x, axes=[1])
        # x = mb.matmul(y=x, x=self.w, transpose_y=not channels_first)
        # return x,

        # In this case we want to share the weights between input embedding and head weights
        # se we use matmul instead of conv or linear, because both of them apply some transformation to w
        # if len(self.split_sizes) == 1:
        #     if channels_first:
        #         return mb.conv(x=self.w, y=x, name="logits")
        #     else:
        #         return mb.linear(x=self.w, y=x, name="logits")

        ws = mb.split(
            x=self.w,
            split_sizes=self.split_sizes,
            axis=0,
            name="head_wsplits",
        )
        if len(self.split_sizes) == 1:
            ws = [ws]

        logits = []

        # Approximate topk on chunks is not faster than
        # an exact one on a big concat: this may be related
        # to ops of over 16_000 (likely 16_384) not supported
        # on ANE
        topk_vals = []
        topk_indices = []
        logsumexps = []
        for i in range(len(self.split_sizes)):
            # for i in range(self.nsplits):
            w = ws[i]
            if channels_first:
                if len(self.split_sizes) > 1:
                    w = mb.expand_dims(x=w, axes=[-1])
                # x = mb.transpose(x=x, perm=[0, 2, 1])
                logits_i = mb.conv(x=x, weight=w, name=f"logits_{i}")
                # logits_i = mb.linear(x=x, weight=w, name=f"logits_{i}")
                if return_logsumexp:
                    m = mb.reduce_max(x=logits_i, axes=[axis], keep_dims=True)
                    x_m = mb.sub(x=logits_i, y=m)
                    lse = mb.reduce_log_sum_exp(x=x_m, axes=[axis], keep_dims=True)
                    lse = mb.add(x=m, y=lse)
                    logsumexps.append(lse)
            else:
                # w = mb.transpose(x=w, perm=[1, 0], name=f"prediction_head_{i}")
                # transpose_y parameter does not work with big weights
                # I think this could also be a linear
                # logits_i = mb.matmul(x=x, y=w, transpose_x=True, name=f"logits_{i}")
                if len(self.split_sizes) > 1:
                    logits_i = mb.matmul(x=w, y=x, transpose_y=True, name=f"logits_{i}")
                else:
                    # we transposed w before
                    logits_i = mb.matmul(x=x, weight=w, name=f"logits_{i}")
                # I think linear is not supported on ANE because it internally performs transpose, which
                # # does not work with big weights
                if self.return_logsumpexp:
                    lse = mb.reduce_log_sum_exp(x=logits_i, axes=[-1], keep_dims=True)
                    logsumexps.append(lse)
            logits.append(logits_i)

            if self.topk > 0:
                topk_vals_i, topk_indices_i = mb.topk(
                    x=x, k=self.topk, name=f"logits_topk_{i}"
                )
                topk_vals.append(topk_vals_i)
                topk_indices.append(topk_indices_i)

        output = []
        if self.return_logits:
            if len(logits) > 1:
                if self.final_cat:
                    logits = [
                        mb.concat(
                            values=[logits[i] for i in range(len(self.split_sizes))],
                            axis=axis,
                            name="_logits",
                        )
                    ]
            else:
                logits = logits[0:1]
            if self.cast and types.builtin_to_string(x.dtype) == "fp16":
                logits = [mb.cast(x=logits[0], dtype="fp32", name=f"logits")]
            # else:
            #     # just for name consistency
            #     logits = [mb.identity(x=logits, name="logits")]

            output += logits

        if return_logsumexp:
            lse = mb.concat(values=logsumexps, axis=axis)
            m = mb.reduce_max(x=lse, axes=[axis], keep_dims=True)
            lse_m = mb.sub(x=lse, y=m)
            lse = mb.reduce_log_sum_exp(x=lse_m, axes=[axis], keep_dims=True)
            lse = mb.add(x=m, y=lse, name="lse")
            output += [lse]

        # if self.topk > 0:
        #     topk_vals = mb.concat(values=topk_vals, axis=2)
        #     topk_indices = mb.concat(values=topk_indices, axis=2)
        #     topk_vals, _topk_indices = mb.topk(x=x, k=self.topk, name="logits_topk")
        #     # topk_indices = mb.
        #     return x, topk_vals, topk_indices

        return output
