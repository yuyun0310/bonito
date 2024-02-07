"""
Bonito model viewer - display a model architecture for a given config.
"""

from re import M
import toml
import argparse
from bonito.util import load_symbol
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchviz import make_dot

def compare_params_in_layers(model, workdir):
    # Gather the names and sizes of all parameters
    param_names = [name for name, _ in model.named_parameters()]
    param_sizes = [param.nelement() for _, param in model.named_parameters()]

    # Plot
    plt.figure(figsize=(15, 10))
    idx = np.arange(len(param_names))
    plt.barh(idx, param_sizes, color='skyblue')
    plt.yticks(idx, param_names)
    plt.xlabel('Number of Parameters')
    plt.title('Parameters in Each Layer')

    plt.savefig(workdir + '/params.png')

    plt.show()

def visualize_model(model, workdir):
    N = 1  # Batch size: Using a single example for visualization
    C = 1  # Number of channels, as per the first Conv1d layer
    L = 100  # Sequence length: Arbitrary choice, adjust based on your needs

    # Generate a random input tensor
    input_tensor = torch.randn(N, C, L)

    model.eval()
    out = model(input_tensor)
    
    # Visualize the graph
    save_path = workdir + '/model_visualization.png'
    make_dot(out, params=dict(list(model.named_parameters()) + [('input_tensor', input_tensor)])).render(save_path, format="png")


def main(args):
    print(args.config)
    config = toml.load(args.config)

    # Directory to save graph
    workdir = args.dir + '/view'
    print(workdir)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training and overwrite files." % workdir)
        exit(1)
    os.makedirs(workdir, exist_ok=True)

    Model = load_symbol(config, "Model")
    model = Model(config)
    print(model)
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))
    
    # Define the path where you want to save the text file
    save_path = workdir + '/params_print.txt'

    # Open the file in write mode
    with open(save_path, 'w') as file:
        # Redirect the output of print(model) to the file
        print(model, file=file)
        print("Total parameters in model", sum(p.numel() for p in model.parameters()), file=file)

    compare_params_in_layers(model, workdir)
    visualize_model(model, workdir)


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config") # also the directory to save graphs generated
    parser.add_argument("--device", default="cpu") # or cuda
    parser.add_argument("--dir")
    
    return parser
