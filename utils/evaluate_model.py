import torch
import torch.nn as nn
import numpy as np
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

def elementwise_op_handler(inputs, outputs):
    """
    FLOPs for handling element - by - element operations (add, mul, etc.)
    """
    return np.prod(get_shape(outputs[0]))

def softmax_handler(inputs, outputs):
    return np.prod(get_shape(outputs[0])) * 2

def sum_mean_handler(inputs, outputs):
    return np.prod(get_shape(inputs[0]))

def expand_as_handler(inputs, outputs):
    return 0

def memory_intensive_op_handler(inputs, outputs):
    return 0

# Count the FLOPs and parameters of a model
def count_model_flops(model: nn.Module, dummy_input: torch.Tensor, custom_handlers=None):
    """
    Calculate the FLOPs and parameter count of the model, 
    supporting custom operation processors
    """
    if not isinstance(model, nn.Module) or not isinstance(dummy_input, torch.Tensor):
        raise TypeError("`model` must be an nn.Module and `dummy_input` a torch.Tensor.")

    # params
    total_params = sum(p.numel() for p in model.parameters())

    if custom_handlers is None:
        custom_handlers = {
            # Element-wise operation
            "aten::add": elementwise_op_handler,
            "aten::add_": elementwise_op_handler,
            "aten::mul": elementwise_op_handler,
            "aten::mul_": elementwise_op_handler,
            # Activation functions
            "aten::softmax": softmax_handler,
            # Aggregation operation
            "aten::sum": sum_mean_handler,
            "aten::mean": sum_mean_handler,
            # Memory/View Operations (FLOPs=0)
            "aten::expand_as": expand_as_handler,
            "aten::im2col": memory_intensive_op_handler,
            "aten::col2im": memory_intensive_op_handler,
        }

    # FlopCountAnalysis
    flop_analyzer = FlopCountAnalysis(model, dummy_input)
    flop_analyzer.set_op_handle(**custom_handlers)

    flop_analyzer.unsupported_ops_warnings(False)
    flop_analyzer.uncalled_modules_warnings(False)
    total_flops = flop_analyzer.total()

    return total_flops, total_params
    
    
# Measure the average inference time per sample
def measure_inference_time(model, dummy_input, num_iterations=100):
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_iterations, 1))
    for _ in range(10): _ = model(dummy_input) # Warm-up
    with torch.no_grad():
        for rep in range(num_iterations):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings[rep] = starter.elapsed_time(ender)
    return np.mean(timings)
      