import math
import numpy as np
from typing import List, Tuple

# from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil import Var, get_new_symbol, types
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

# from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17.conv import conv as _conv_iOS17
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET
from coremltools.converters.mil.mil.block import curr_opset_version
from coremltools.converters.mil.mil.ops.defs.iOS15 import _IOS15_TARGET


class ConvPoolingTypeInferenceCache(dict):
    """
    An utility class to cache the shape inference of ``conv`` and ``pool`` op.
    The cache mechanism makes sure ops with the same input shape (symbolic also),
    and params (``pad, stride, kernel``) would produce the same output shape.
    """

    @staticmethod
    def get_cache_key(
        input_shape: Tuple[int],
        pad_type: str,
        pad: Tuple[int],
        strides: Tuple[int],
        kernel: Tuple[int],
        ceil_mode: bool,
    ) -> Tuple[Tuple]:
        return (
            ("input_shape", input_shape),
            ("pad_type", pad_type),
            ("pad", pad),
            ("strides", strides),
            ("kernel", kernel),
            ("ceil_mode", ceil_mode),
        )

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"cache key {key} already exisit.")
        return dict.__setitem__(self, key, value)


CONV_POOLING_TYPE_INFERENCE_CACHE = ConvPoolingTypeInferenceCache()


def effective_kernel(kernel_shape, dilations):
    """

    Args:
        kernel_shape: tuple[int] representing the kernel shape in each
            given dimension.
        dilations: tuple[int] representing the dilation of the kernel
            in each given dimension.  Must be the same length as
            kernel_shape, and is assumed to give the dimensions in
            the same order as kernel_shape

    Returns: tuple[int] representing the effective shape of the kernel
        in each given dimension, with each dimension in the order given,
        taking into account dilation.
        See http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions
        Note that a dilation of 1 is equivalent to having no dilation.

    """
    if len(kernel_shape) != len(dilations):
        raise ValueError(
            f"kernel_shape ({len(kernel_shape)}) and dilations ({len(dilations)}) "
            f"must be the same length"
        )
    return tuple([(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)])


def aggregated_pad(
    pad_type,
    kernel_shape,
    input_shape=None,
    strides=None,
    dilations=None,
    custom_pad=None,
):
    """
    Args
        pad_type: string. Must be one of ('same', 'same_lower', 'valid', 'custom')

        kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

        input_shape: [iH, iW, ...]: spatial input dims (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

        strides: [sH, sW, ...]: spatial strides (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

        dilations: [dH, dW, ...]: dilations (excluding channels)
            If not provided, defaults to [1, 1, ...], effectively no dilation.

        custom_pad: Required iff pad_type == 'custom'.
            custom_pad[2*i], custom_pad[2*i+1] are before/after custom padding
            for spatial dim i.


    Returns:
        A tuple of total (before + after) padding for each spatial dimension in kernel_shape.
    """
    num_spatial_dims = len(kernel_shape)
    if dilations is None:
        dilations = [1] * num_spatial_dims
    elif len(dilations) != num_spatial_dims:
        raise ValueError(
            f"dilations must have same length as kernel_shape "
            f"({num_spatial_dims}, but got {len(dilations)})"
        )
    if pad_type in ["same", "same_lower"]:
        if input_shape is None or len(input_shape) != num_spatial_dims:
            raise ValueError(
                "For SAME padding input_shape must not be None and must have "
                "same length as kernel_shape ({}, but got {})".format(
                    num_spatial_dims,
                    len(input_shape) if input_shape is not None else "None",
                )
            )
        if strides is None or len(strides) != num_spatial_dims:
            raise ValueError(
                "For SAME padding strides must not be None and must have "
                "same length as kernel_shape ({}, but got {})".format(
                    num_spatial_dims, len(strides) if strides is not None else "None"
                )
            )
        effective_ks = effective_kernel(kernel_shape, dilations)
        return tuple(
            [
                (
                    int(max(0, s * math.ceil(float(i) / float(s)) - i + k - s))
                    if not is_symbolic(i)
                    else get_new_symbol()
                )
                for i, k, s in zip(input_shape, effective_ks, strides)
            ]
        )
    if pad_type == "valid":
        return tuple([0] * num_spatial_dims)
    if pad_type == "custom":
        if custom_pad is None or len(custom_pad) != 2 * num_spatial_dims:
            raise ValueError("Invalid custom_pad.")
        return tuple(
            [custom_pad[2 * d] + custom_pad[2 * d + 1] for d in range(num_spatial_dims)]
        )
    raise ValueError('Invalid padding pad_type "{}"'.format(pad_type))


def spatial_dimensions_out_shape(
    pad_type,
    input_shape,
    kernel_shape,
    strides,
    dilations=None,
    custom_pad=None,
    ceil_mode=False,
):
    """
    Args
        pad_type: string. Must be one of ('same', 'same_lower', 'valid', 'custom')

        input_shape: [iH, iW, ...]: spatial input dims (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

        kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

        strides: [sH, sW, ...]: spatial strides (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

        dilations: [dH, dW, ...]: dilations (excluding channels)
            If not provided, defaults to [1, 1, ...], effectively no dilation.

        custom_pad: Required iff pad_type == 'custom'.
            custom_pad[2*i], custom_pad[2*i+1] are before/after custom padding
            for spatial dim i.

        ceil_mode: determines the padding and output shape.
             When ceil mode is True:
                out_dim = floor((in_dim + pad_l + pad_r - kernel_size + (stride-1)) / stride) + 1
                if (out_dim-1) * stride >= in_dim + pad_l and (pad_l > 0 or pad_r > 0):
                    out_dim = out_dim - 1
            When ceil mode is False:
                out_dim = floor((in_dim + pad_l + pad_r - kernel_size) / stride) + 1

    Returns:
        A list of spatial output sizes for each spatial dimension of kernel_shape.

    """
    num_spatial_dims = len(kernel_shape)
    if dilations is None:
        dilations = [1] * num_spatial_dims
    if custom_pad is None:
        custom_pad = np.array([0] * num_spatial_dims * 2)
    if not (
        len(input_shape)
        == len(kernel_shape)
        == len(strides)
        == len(dilations)
        == len(custom_pad) / 2
    ):
        raise ValueError(
            f"input_shape (length {len(input_shape)}), "
            f"kernel_shape (length {len(kernel_shape)}), "
            f"strides (length {len(strides)}), "
            f"dilations (length {len(dilations)}), "
            f"and custom_pad (length {len(custom_pad)}) divided by two "
            "must all be the same length"
        )

    effective_ks = effective_kernel(kernel_shape, dilations)
    if isinstance(strides, np.ndarray):
        strides = tuple(strides.tolist())
    if isinstance(custom_pad, np.ndarray):
        custom_pad = tuple(custom_pad.tolist())
    cache_key = CONV_POOLING_TYPE_INFERENCE_CACHE.get_cache_key(
        input_shape,
        pad_type,
        custom_pad,
        strides,
        effective_ks,
        ceil_mode,
    )
    if cache_key in CONV_POOLING_TYPE_INFERENCE_CACHE:
        return CONV_POOLING_TYPE_INFERENCE_CACHE[cache_key]

    pad = aggregated_pad(
        pad_type=pad_type,
        kernel_shape=kernel_shape,
        input_shape=input_shape,
        strides=strides,
        dilations=dilations,
        custom_pad=custom_pad,
    )

    out_shape = []
    for r in range(num_spatial_dims):
        # only check if `input_shape` (spatial part of the input image) is symbolic, because:
        # * `input_shape` can be symbolic
        # * `pad` (aggregated from `input_shape` + ...) is symbolic only if `input_shape` is symbolic
        # * `effective_ks` (effective kernel size, determined from kernel size + dilations) cannot be symbolic
        # * strides cannot be symbolic
        if is_symbolic(input_shape[r]):
            if (
                not is_symbolic(pad[r])
                and pad[r] - effective_ks[r] == -1
                and strides[r] == 1
            ):
                out_shape.append(input_shape[r])
            else:
                out_shape.append(get_new_symbol())
        else:
            out_dim = 0
            if not ceil_mode:
                out_dim = math.floor(
                    (input_shape[r] + pad[r] - effective_ks[r]) / strides[r] + 1
                )
            else:
                out_dim = math.floor(
                    (input_shape[r] + pad[r] - effective_ks[r] + strides[r] - 1)
                    / strides[r]
                    + 1
                )
                if (out_dim - 1) * strides[r] >= input_shape[r] + pad[r] / 2 and pad[
                    r
                ] > 0:
                    out_dim = out_dim - 1
            if out_dim <= 0:
                raise ValueError(
                    f"spatial dimension {r} has invalid output size {out_dim}"
                )
            out_shape.append(out_dim)
    CONV_POOLING_TYPE_INFERENCE_CACHE[cache_key] = out_shape
    return out_shape


# @register_op(opset_version=_IOS17_TARGET, )
class conv(_conv_iOS17):
    """
    Perform convolution over input. Supports 1-D, 2-D, and 3-D convolution.

    The difference between this version and the iOS 15 :py:class:`~.iOS15.conv.conv` is that the
    ``weight`` and ``bias`` may have a different dtype than the input/output.
    """

    # input_spec = InputSpec(
    #     x=TensorInputType(type_domain="T"),
    #     weight=TensorInputType(type_domain="U"),
    #     bias=TensorInputType(optional=True, type_domain="U"),
    #     strides=TensorInputType(const=True, optional=True, type_domain=types.int32),
    #     pad_type=TensorInputType(const=True, optional=True, type_domain=types.str),
    #     pad=TensorInputType(const=True, optional=True, type_domain=types.int32),
    #     dilations=TensorInputType(const=True, optional=True, type_domain=types.int32),
    #     groups=TensorInputType(const=True, optional=True, type_domain=types.int32),
    # )

    # type_domains = {
    #     "T": (types.fp16, types.fp32),
    #     "U": (types.fp16, types.fp32),
    # }
    def type_inference(self):
        inshape = self.x.shape
        f_shape = self.weight.shape
        kernel_shape = f_shape[2:]
        C_out = f_shape[0]
        C_in = self.x.shape[1]
        groups = self.groups.val

        if self.bias is not None and (
            len(self.bias.shape) > 1 or self.bias.shape[0] != C_out
        ):
            msg = "# of bias values {} not equal to # output channels {}"
            raise ValueError(msg.format(self.bias.shape[0], C_out))
        if C_in % groups != 0:
            msg = "# of input channels {} not divisible by groups {}"
            raise ValueError(msg.format(C_in, groups))
        if C_in // groups != self.weight.shape[1]:
            msg = "C_in / groups = {}/{} != weight[1] ({})"
            raise ValueError(msg.format(C_in, groups, self.weight.shape[1]))

        strides = self.strides.val
        dilations = self.dilations.val

        # The same_lower padding is not supported in iOS15
        if curr_opset_version() == _IOS15_TARGET and self.pad_type.val == "same_lower":
            msg = "iOS15 version of conv does not support pad_type = `same_lower`"
            raise ValueError(msg)

        # Ignore self.pad if pad_type != custom
        custom_pad = None if self.pad_type.val != "custom" else self.pad.val

        is_weight_dynamic = not self.weight.is_descendant_of_const
        if is_weight_dynamic and any([True if d > 1 else False for d in dilations]):
            raise ValueError(
                "Convolution with dynamic weights does not support dilations!"
            )

        N = inshape[0]
        C_out = f_shape[0]
        # spatial dimensions
        d_out_shape = spatial_dimensions_out_shape(
            pad_type=self.pad_type.val,
            input_shape=inshape[2:],
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            custom_pad=custom_pad,
        )
        retshape = [N, C_out] + d_out_shape
        return types.tensor(self.x.dtype, tuple(retshape))
