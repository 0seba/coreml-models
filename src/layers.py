from typing import List

import math
import numpy as np
from coremltools.converters.mil import Builder as mb
import coremltools.converters.mil as mil
from coremltools.converters.mil.mil import Operation, Var, types

from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET

from custom_conv import conv

register_op(conv, opset_version=_IOS17_TARGET, allow_override=True)  #


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


class Embedding(NamedCall):
    def __init__(self, w: np.ndarray, name=None, validate_indices=False):
        super().__init__(name)
        self.w = w
        self.validate_indices = validate_indices

    def __call__(self, x, name=None):
        return mb.gather(
            # x=np.expand_dims(self.w.T, 0),
            x=self.w,
            indices=x,
            axis=0,
            name=self.name(name),
            # batch_dims=1,
            # validate_indices=self.validate_indices,
        )


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
        dimroot = np.array(dimroot, dtype=dtype)
        maxval = mb.abs(x=x, name=f"{prefix}_rmsnorm_abs")
        maxval = mb.reduce_max(
            x=maxval, axes=axes, keep_dims=True, name=f"{prefix}_rmsnorm_maxval"
        )
        maxval = mb.clip(
            x=maxval, alpha=eps, beta=beta, name=f"{prefix}_rmsnorm_maxval_clipped"
        )
        xscaled = mb.real_div(x=x, y=maxval, name=f"{prefix}_rmsnorm_scaled")

        # norm = mb.reduce_l2_norm(
        #     x=xscaled, axes=[-1], keep_dims=True, name=f"{prefix}_rmsnorm_norm"
        # ) # Not supported by ANE
        # norm = mb.clip(
        #     x=norm, alpha=eps, beta=beta, name=f"{prefix}_rmsnorm_norm_clipped"
        # )

        # Seems like reduce_l2_norm does not work on ANE, so we split in separate ops
        sq_sum = mb.reduce_sum_square(
            x=xscaled, axes=axes, keep_dims=True, name=f"{prefix}_rmsnorm_squared_sum"
        )
        rsqrt = mb.rsqrt(x=sq_sum, epsilon=eps, name=f"{prefix}_rmsnorm_rsqrt")
        xscaled = mb.mul(x=xscaled, y=dimroot, name=f"{prefix}_rmsnorm_dim_scaled")
        # xnormed = mb.real_div(x=xscaled, y=norm, name=f"{prefix}_rmsnorm_x_normalized")
        xnormed = mb.mul(x=xscaled, y=rsqrt, name=f"{prefix}_rmsnorm_normalized")
        return xnormed

    @staticmethod
    def normalize(x, eps, dimroot, prefix, axes):
        squared = mb.reduce_sum_square(
            x=x, axes=axes, keep_dims=True, name=f"{prefix}_rmsnorm_squared_sum"
        )
        # squared_mean = mb.real_div(
        #     x=squared,
        #     y=np.array(x.shape[-1], dtype=np.float32),
        #     name=f"{prefix}_rmsnorm_squared_mean",
        # )
        norm_reciprocal = mb.rsqrt(
            x=squared, epsilon=eps, name=f"{prefix}_rmsnorm_norm_reciprocal"
        )
        rmsnorm_reciprocal = mb.mul(
            x=norm_reciprocal, y=dimroot, name=f"{prefix}_rmsnorm_rmsnorm_reciprocal"
        )
        return mb.mul(x=x, y=rmsnorm_reciprocal, name=f"{prefix}_rmsnorm_normalized")

    def __call__(self, x, prefix=None, w=None, axes=None):
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
            return mb.mul(x=xnormed, y=w, name=f"{prefix}_rmsnorm")
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

    def __call__(self, x, prefix=None, channels_first=None):
        if prefix is None:
            prefix = self.prefix
        if channels_first is None:
            channels_first = self.channels_first

        axis = 1 if channels_first else 2

        ws = mb.split(
            x=self.w,
            split_sizes=self.split_sizes,
            axis=0,
            name="head_wsplits",
        )

        logits = []

        # Approximate topk on chunks is not faster than
        # an exact one on a big concat: this may be related
        # to ops of over 16_000 (likely 16_384) not supported
        # on ANE
        topk_vals = []
        topk_indices = []
        for i in range(len(self.split_sizes)):
            # for i in range(self.nsplits):
            w = ws[i]
            if channels_first:
                w = mb.expand_dims(x=w, axes=[-1])
                logits_i = mb.conv(x=x, weight=w, name=f"logits_{i}")
            else:
                w = mb.transpose(x=w, perm=[1, 0], name=f"prediction_head_{i}")
                # transpose_y parameter does not work with big weights
                # I think this could also be a linear
                logits_i = mb.matmul(x=x, y=w, transpose_y=False, name=f"logits_{i}")
            logits.append(logits_i)

            if self.topk > 0:
                topk_vals_i, topk_indices_i = mb.topk(
                    x=x, k=self.topk, name=f"logits_topk_{i}"
                )
                topk_vals.append(topk_vals_i)
                topk_indices.append(topk_indices_i)

        output = tuple()
        if self.return_logits:
            logits = mb.concat(
                values=[logits[i] for i in range(len(self.split_sizes))],
                axis=axis,
                name="_logits",
            )
            if self.cast and types.builtin_to_string(x.dtype) == "fp16":
                logits = mb.cast(x=logits, dtype="fp32", name=f"logits")
            else:
                # just for name consistency
                logits = mb.identity(x=logits, name="logits")

            output += (logits,)

        # if self.topk > 0:
        #     topk_vals = mb.concat(values=topk_vals, axis=2)
        #     topk_indices = mb.concat(values=topk_indices, axis=2)
        #     topk_vals, _topk_indices = mb.topk(x=x, k=self.topk, name="logits_topk")
        #     # topk_indices = mb.
        #     return x, topk_vals, topk_indices

        return output
