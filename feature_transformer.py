import ctypes
import os
import sys
import glob
import torch
from torch import nn
from torch import autograd
import math

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)
forward_kernel = dll.feature_transformer_slice_forward
forward_kernel.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
backward_kernel = dll.feature_transformer_slice_backward
backward_kernel.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]

def _find_nearest_divisor(value, target):
    divisors = []
    for i in range(1, value+1):
        if value % i == 0:
            divisors.append((i, abs(target-i)))
    divisors.sort(key=lambda x:x[1])
    return divisors[0][0]

_num_threads_forward_cache = dict()
def _get_num_threads_for_forward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_forward_cache[output_size]

_num_threads_backward_cache = dict()
def _get_num_threads_for_backward(output_size):
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(output_size, optimal_num_threads)

    return _num_threads_backward_cache[output_size]

def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)
    return f

class FeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        forward_kernel(feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr()
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        backward_kernel(feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr()
            )

        return None, None, weight_grad, bias_grad

class DoubleFeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias):
        ctx.save_for_backward(feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias)

        assert len(feature_indices_0.shape) == 2
        assert len(feature_values_0.shape) == 2
        assert feature_indices_0.shape[0] == feature_values_0.shape[0]
        assert feature_indices_0.shape[1] == feature_values_0.shape[1]
        assert feature_indices_0.dtype == torch.int32
        assert feature_values_0.dtype == torch.float32

        assert len(feature_indices_1.shape) == 2
        assert len(feature_values_1.shape) == 2
        assert feature_indices_1.shape[0] == feature_values_1.shape[0]
        assert feature_indices_1.shape[1] == feature_values_1.shape[1]
        assert feature_indices_1.dtype == torch.int32
        assert feature_values_1.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_values_0.device == feature_indices_0.device
        assert feature_values_1.device == feature_indices_1.device
        assert feature_indices_0.device == feature_indices_1.device
        assert weight.device == feature_indices_0.device
        assert bias.device == feature_indices_0.device

        assert feature_indices_0.is_contiguous()
        assert feature_values_0.is_contiguous()
        assert feature_indices_1.is_contiguous()
        assert feature_values_1.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        print("enter forward")
        print(feature_indices_0)
        print(feature_values_0)
        print(weight)
        print(bias)

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        output0 = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)
        output1 = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        weight_ptr = weight.data_ptr()
        weight_ptr = ctypes.cast(weight_ptr, ctypes.POINTER(ctypes.c_float))

        bias_ptr = bias.data_ptr()
        bias_ptr = ctypes.cast(bias_ptr, ctypes.POINTER(ctypes.c_float))

        feature_indices_0_ptr = feature_indices_0.data_ptr()
        feature_indices_0_ptr = ctypes.cast(feature_indices_0_ptr, ctypes.POINTER(ctypes.c_int32))

        feature_values_0_ptr = feature_values_0.data_ptr()
        feature_values_0_ptr = ctypes.cast(feature_values_0_ptr, ctypes.POINTER(ctypes.c_float))

        output0_ptr = output0.data_ptr()
        output0_ptr = ctypes.cast(output0_ptr, ctypes.POINTER(ctypes.c_float))

        feature_indices_1_ptr = feature_indices_1.data_ptr()
        feature_indices_1_ptr = ctypes.cast(feature_indices_1_ptr, ctypes.POINTER(ctypes.c_int32))

        feature_values_1_ptr = feature_values_1.data_ptr()
        feature_values_1_ptr = ctypes.cast(feature_values_1_ptr, ctypes.POINTER(ctypes.c_float))

        output1_ptr = output1.data_ptr()
        output1_ptr = ctypes.cast(output1_ptr, ctypes.POINTER(ctypes.c_float))

        print("before forward_kernel0")
        forward_kernel(feature_indices_0_ptr,
                feature_values_0_ptr,
                weight_ptr,
                bias_ptr,
                output0_ptr
            )
        print("after forward_kernel0")
        
        forward_kernel(feature_indices_1_ptr,
                feature_values_1_ptr,
                weight_ptr,
                bias_ptr,
                output1_ptr
            )
        print("after forward_kernel1")

        return output0, output1

    @staticmethod
    def backward(ctx, grad_output_0, grad_output_1):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        print("enter backward")
        print(grad_output_0)
        print(grad_output_1)

        grad_output_0 = grad_output_0.contiguous()
        grad_output_1 = grad_output_1.contiguous()

        feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, weight, bias = ctx.saved_tensors

        device = feature_indices_0.device
        batch_size = feature_indices_0.shape[0]
        max_active_features = feature_indices_0.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        weight_grad_ptr = weight_grad.data_ptr()
        weight_grad_ptr = ctypes.cast(weight_grad_ptr, ctypes.POINTER(ctypes.c_float))

        bias_grad_ptr = bias_grad.data_ptr()
        bias_grad_ptr = ctypes.cast(bias_grad_ptr, ctypes.POINTER(ctypes.c_float))

        feature_indices_0_ptr = feature_indices_0.data_ptr()
        feature_indices_0_ptr = ctypes.cast(feature_indices_0_ptr, ctypes.POINTER(ctypes.c_int32))

        feature_values_0_ptr = feature_values_0.data_ptr()
        feature_values_0_ptr = ctypes.cast(feature_values_0_ptr, ctypes.POINTER(ctypes.c_float))

        grad_output_0_ptr = grad_output_0.data_ptr()
        grad_output_0_ptr = ctypes.cast(grad_output_0_ptr, ctypes.POINTER(ctypes.c_float))

        feature_indices_1_ptr = feature_indices_1.data_ptr()
        feature_indices_1_ptr = ctypes.cast(feature_indices_1_ptr, ctypes.POINTER(ctypes.c_int32))

        feature_values_1_ptr = feature_values_1.data_ptr()
        feature_values_1_ptr = ctypes.cast(feature_values_1_ptr, ctypes.POINTER(ctypes.c_float))

        grad_output_1_ptr = grad_output_1.data_ptr()
        grad_output_1_ptr = ctypes.cast(grad_output_1_ptr, ctypes.POINTER(ctypes.c_float))
 
    
        print("before backward_kernel0")
        backward_kernel(feature_indices_0_ptr,
                feature_values_0_ptr,
                weight_grad_ptr,
                bias_grad_ptr,
                grad_output_0_ptr
            )
        
        print("after backward_kernel0")
        backward_kernel(feature_indices_1_ptr,
                feature_values_1_ptr,
                weight_grad_ptr,
                bias_grad_ptr,
                grad_output_1_ptr
            )
        print("after backward_kernel1")

        return None, None, None, None, weight_grad, bias_grad

class FeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices, feature_values):
        return FeatureTransformerSliceFunction.apply(feature_indices, feature_values, self.weight, self.bias)

class DoubleFeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DoubleFeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1):
        return DoubleFeatureTransformerSliceFunction.apply(feature_indices_0, feature_values_0, feature_indices_1, feature_values_1, self.weight, self.bias)

if __name__ == '__main__':
    import time
    import sys
    import os

    def FeatureTransformerSliceFunctionEmulate(feature_indices, feature_values, weight, bias):
        batch_size = feature_indices.shape[0]
        num_inputs = weight.shape[0]
        max_active_features = feature_indices.shape[1]
        inputs = torch.zeros(batch_size, num_inputs, dtype=torch.float32, device=weight.device)
        for i in range(batch_size):
            for j in range(max_active_features):
                feature = feature_indices[i, j]
                value = feature_values[i, j]
                inputs[i, feature] += value

        return torch.mm(inputs, weight) + bias

    def test():
        BATCH_SIZE = 16
        INPUT_SIZE = 10
        MAX_ACTIVE_FEATURES = 32
        STRIDE = 128
        MAX_ERROR = 1e-4

        torch.manual_seed(0)
        weight0 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
        bias0 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        torch.manual_seed(0)
        weight1 = torch.rand(INPUT_SIZE, STRIDE, dtype=torch.float32, requires_grad=True)
        bias1 = torch.rand(STRIDE, dtype=torch.float32, requires_grad=True)
        indices0 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
        indices1 = (torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES) * INPUT_SIZE).to(dtype=torch.int32)
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

        output00 = FeatureTransformerSliceFunctionEmulate(indices0.clone(), values0.clone(), weight0, bias0)
        output01 = FeatureTransformerSliceFunctionEmulate(indices1.clone(), values1.clone(), weight0, bias0)
        output10, output11 = DoubleFeatureTransformerSliceFunction.apply(indices0.clone(), values0.clone(), indices1.clone(), values1.clone(), weight1, bias1)

        assert torch.max(output00.cpu() - output10.cpu()) < MAX_ERROR
        assert torch.max(output01.cpu() - output11.cpu()) < MAX_ERROR
        (output00 - output01).sum().backward()
        (output10 - output11).sum().backward()
        assert torch.max(weight0.grad.cpu() - weight1.grad.cpu()) < MAX_ERROR
        assert torch.max(bias0.grad.cpu() - bias1.grad.cpu()) < MAX_ERROR
        print('Tests passed.')

    def bench():
        INPUT_SIZE = 40960
        BATCH_SIZE = 8192
        ITERS = 64
        STRIDE = 264
        MAX_ACTIVE_FEATURES = 64

        layer = DoubleFeatureTransformerSlice(INPUT_SIZE, STRIDE)
        indices0 = torch.cat([torch.sort((torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4) * INPUT_SIZE), dim=1)[0].to(dtype=torch.int32), torch.full((BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32)], dim=1)
        values0 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)
        indices1 = torch.cat([torch.sort((torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES * 3 // 4)) * INPUT_SIZE, dim=1)[0].to(dtype=torch.int32), torch.full((BATCH_SIZE, MAX_ACTIVE_FEATURES // 4), -1, dtype=torch.int32)], dim=1)
        values1 = torch.rand(BATCH_SIZE, MAX_ACTIVE_FEATURES, dtype=torch.float32)

        output0, output1 = layer(indices0, values0, indices1, values1)

        device = indices0.device

        start = time.time()

        for i in range(ITERS):
            output0, output1 = layer(indices0, values0, indices1, values1)
            output0 = torch.clamp(output0, 0.0, 1.0)
            output1 = torch.clamp(output1, 0.0, 1.0)

            g = ((output0 - output1)**2).mean()
            g.backward()


        end = time.time()

        #for param in layer.parameters():
        #    print(param.grad)

        print('{} pos/s'.format((ITERS * BATCH_SIZE) / (end - start)))

    test()
    bench()