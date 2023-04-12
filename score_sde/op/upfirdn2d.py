# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

""" Originated from https://github.com/rosinality/stylegan2-pytorch
The license for the original version of this file can be found in this directory (LICENSE_MIT).
"""

import os
from collections import abc
import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.utils.cpp_extension import load
# from . import conv2d_gradfix
# from WaveDiff.score_sde.op.conv2d_gradfix import conv2d

module_path = os.path.dirname(__file__)
# upfirdn2d_op = load(
#     "upfirdn2d",
#     sources=[
#         os.path.join(module_path, "upfirdn2d.cpp"),
#         os.path.join(module_path, "upfirdn2d_kernel.cu"),
#     ],
# )

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.conv2d` that supports
arbitrarily high order gradients with zero performance penalty."""

import warnings
import contextlib
import torch

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

# ----------------------------------------------------------------------------

enabled = False  # Enable the custom op by setting this to true.
weight_gradients_disabled = False  # Forcefully disable computation of gradients with respect to the weights.


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled
    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


# ----------------------------------------------------------------------------


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=True, weight_shape=weight.shape, stride=stride, padding=padding,
                               output_padding=output_padding, groups=groups, dilation=dilation).apply(input, weight,
                                                                                                      bias)
    return torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding,
                                                output_padding=output_padding, groups=groups, dilation=dilation)


# ----------------------------------------------------------------------------

def _should_use_custom_op(input):
    assert isinstance(input, torch.Tensor)
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False
    if input.device.type != 'cuda':
        return False
    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):
        return True
    warnings.warn(
        f'conv2d_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.conv2d().')
    return False


def _tuple_of_ints(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs


# ----------------------------------------------------------------------------

_conv2d_gradfix_cache = dict()


def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    # Parse arguments.
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    # Validate arguments.
    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else:  # transpose
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i]) for i in range(ndim))

    # Helpers.
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class Conv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            if not transpose:
                output = torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)
            else:  # transpose
                output = torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias,
                                                              output_padding=output_padding, **common_kwargs)
            ctx.save_for_backward(input, weight)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad_input = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p,
                                             **common_kwargs).apply(grad_output, weight, None)
                assert grad_input.shape == input.shape

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class Conv2dGradWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation(
                'aten::cudnn_convolution_backward_weight' if not transpose else 'aten::cudnn_convolution_transpose_backward_weight')
            flags = [torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                     torch.backends.cudnn.allow_tf32]
            grad_weight = op(weight_shape, grad_output, input, padding, stride, dilation, groups, *flags)
            assert grad_weight.shape == weight_shape
            ctx.save_for_backward(grad_output, input)
            return grad_weight

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight, None)
                assert grad2_grad_output.shape == grad_output.shape

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad2_input = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p,
                                              **common_kwargs).apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input.shape

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(transpose=False, weight_shape=weight.shape, stride=stride, padding=padding, output_padding=0, dilation=dilation, groups=groups).apply(input, weight, bias)
    return torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# ----------------------------------------------------------------------------

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    input = input.permute(0, 2, 3, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]

# class UpFirDn2dBackward(Function):
#     @staticmethod
#     def forward(
#         ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
#     ):
#
#         up_x, up_y = up
#         down_x, down_y = down
#         g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
#
#         grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
# #upfirdn2d_op.upfirdn2d
#         grad_input = upfirdn2d_native(
#             grad_output,
#             grad_kernel,
#             down_x,
#             down_y,
#             up_x,
#             up_y,
#             g_pad_x0,
#             g_pad_x1,
#             g_pad_y0,
#             g_pad_y1,
#         )
#         grad_input = grad_input.view(
#             in_size[0], in_size[1], in_size[2], in_size[3])
#
#         ctx.save_for_backward(kernel)
#
#         pad_x0, pad_x1, pad_y0, pad_y1 = pad
#
#         ctx.up_x = up_x
#         ctx.up_y = up_y
#         ctx.down_x = down_x
#         ctx.down_y = down_y
#         ctx.pad_x0 = pad_x0
#         ctx.pad_x1 = pad_x1
#         ctx.pad_y0 = pad_y0
#         ctx.pad_y1 = pad_y1
#         ctx.in_size = in_size
#         ctx.out_size = out_size
#
#         return grad_input
#
#     @staticmethod
#     def backward(ctx, gradgrad_input):
#         kernel, = ctx.saved_tensors
#
#         gradgrad_input = gradgrad_input.reshape(-1,
#                                                 ctx.in_size[2], ctx.in_size[3], 1)
#
#         gradgrad_out = upfirdn2d_op.upfirdn2d(
#             gradgrad_input,
#             kernel,
#             ctx.up_x,
#             ctx.up_y,
#             ctx.down_x,
#             ctx.down_y,
#             ctx.pad_x0,
#             ctx.pad_x1,
#             ctx.pad_y0,
#             ctx.pad_y1,
#         )
#         # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
#         gradgrad_out = gradgrad_out.view(
#             ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
#         )
#
#         return gradgrad_out, None, None, None, None, None, None, None, None

#
# class UpFirDn2d(Function):
#     @staticmethod
#     def forward(ctx, input, kernel, up, down, pad):
#         up_x, up_y = up
#         down_x, down_y = down
#         pad_x0, pad_x1, pad_y0, pad_y1 = pad
#
#         kernel_h, kernel_w = kernel.shape
#         batch, channel, in_h, in_w = input.shape
#         ctx.in_size = input.shape
#
#         input = input.reshape(-1, in_h, in_w, 1)
#
#         ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
#
#         out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
#         out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
#         ctx.out_size = (out_h, out_w)
#
#         ctx.up = (up_x, up_y)
#         ctx.down = (down_x, down_y)
#         ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)
#
#         g_pad_x0 = kernel_w - pad_x0 - 1
#         g_pad_y0 = kernel_h - pad_y0 - 1
#         g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
#         g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
#
#         ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
#
#         out =  upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
#
#         # out = upfirdn2d_op.upfirdn2d(
#         #     input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
#         # )
#         # out = out.view(major, out_h, out_w, minor)
#         out = out.view(-1, channel, out_h, out_w)
#
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         kernel, grad_kernel = ctx.saved_tensors
#
#         grad_input = UpFirDn2dBackward.apply(
#             grad_output,
#             kernel,
#             grad_kernel,
#             ctx.up,
#             ctx.down,
#             ctx.pad,
#             ctx.g_pad,
#             ctx.in_size,
#             ctx.out_size,
#         )
#
#         return grad_input, None, None, None, None

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x =  conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x

# def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
#     if input.device.type == "cpu":
#         out = upfirdn2d_native(
#             input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
#         )
#
#     else:
#         out = UpFirDn2d.apply(
#             input, kernel, (up, up), (down,
#                                       down), (pad[0], pad[1], pad[0], pad[1])
#         )
#
#     return out

#
# def upfirdn2d_ada(input, kernel, up=1, down=1, pad=(0, 0)):
#     if not isinstance(up, abc.Iterable):
#         up = (up, up)
#
#     if not isinstance(down, abc.Iterable):
#         down = (down, down)
#
#     if len(pad) == 2:
#         pad = (pad[0], pad[1], pad[0], pad[1])
#
#     if input.device.type == "cpu":
#         out = upfirdn2d_native(input, kernel, *up, *down, *pad)
#
#     else:
#         out = UpFirDn2d.apply(input, kernel, up, down, pad)
#
#     return out

def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    # Parse arguments.
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    # Validate arguments.
    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else: # transpose
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i]) for i in range(ndim))

    # Helpers.
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class Conv2d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            if not transpose:
                output = torch.nn.functional.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)
            else: # transpose
                output = torch.nn.functional.conv_transpose2d(input=input, weight=weight, bias=bias, output_padding=output_padding, **common_kwargs)
            ctx.save_for_backward(input, weight)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad_input = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p, **common_kwargs).apply(grad_output, weight, None)
                assert grad_input.shape == input.shape

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class Conv2dGradWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation('aten::cudnn_convolution_backward_weight' if not transpose else 'aten::cudnn_convolution_transpose_backward_weight')
            flags = [torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32]
            grad_weight = op(weight_shape, grad_output, input, padding, stride, dilation, groups, *flags)
            assert grad_weight.shape == weight_shape
            ctx.save_for_backward(grad_output, input)
            return grad_weight

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight, None)
                assert grad2_grad_output.shape == grad_output.shape

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                grad2_input = _conv2d_gradfix(transpose=(not transpose), weight_shape=weight_shape, output_padding=p, **common_kwargs).apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input.shape

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.
    Performs the following sequence of operations for each channel:
    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).
    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.
    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.
    4. Downsample the image by keeping every Nth pixel (`down`).
    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.
    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    # if impl == 'cuda' and x.device.type == 'cuda' and _init():
    #     return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0),
              max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
