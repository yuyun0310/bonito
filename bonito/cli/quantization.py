import torch
import torch.nn as nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

def convert_to_quantizable_layer(module):
    """
    Recursively convert supported layers to their quantizable versions.
    """
    print("#"*50)
    print(type(module), module)
    print("#"*50)
    mod = module
    if isinstance(module, nn.modules.conv.Conv1d):
        print("in 1")
        mod = quant_nn.QuantConv1d(module.in_channels, module.out_channels, module.kernel_size, 
                                   stride=module.stride, padding=module.padding, bias=(module.bias is not None))
    elif isinstance(module, nn.modules.linear.Linear):
        print("in 2")
        mod = quant_nn.QuantLinear(module.in_features, module.out_features, bias=(module.bias is not None))
    elif isinstance(module, nn.modules.rnn.LSTM):
        print("in 3")
        mod = quant_nn.QuantLSTM(module.input_size, module.hidden_size, module.num_layers, module.bias,
                                 module.batch_first, module.dropout, module.bidirectional, module.proj_size, module.device,
                                 module.dtype)
    
    for name, child in module.named_children():
        mod.add_module(name, convert_to_quantizable_layer(child))
    return mod

def convert_to_quantizable_model(pretrained_model):
    for name, module in pretrained_model.named_modules():
        print(f"{name}: {type(module).__name__}")


    quant_modules.initialize()
    quantized_model = convert_to_quantizable_layer(pretrained_model)

    for name, module in pretrained_model.named_modules():
        print(f"{name}: {type(module).__name__}")
    return quantized_model
