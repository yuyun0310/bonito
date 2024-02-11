#!/usr/bin/env python3

"""
Bonito model compression.
"""

import os
import sys

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from bonito.data import load_numpy, load_script
from bonito.util import __models__, default_config, default_data
from bonito.util import load_model, load_symbol, init, half_supported, accuracy, permute, decode_ref, get_parameters_count
from bonito.training import load_state, Trainer

import time
import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import LSTM
from torch.optim import AdamW
from torch.quantization import quantize_dynamic
import torch.nn as nn
import copy

def model_dequantization(quantized_model, original_model):
    print("Original Model Keys:")
    for key in original_model.state_dict().keys():
        print(key)

    print("\nQuantized Model Keys:")
    for key in quantized_model.state_dict().keys():
        print(key)
    # with torch.no_grad():
    #     for name, quantized_weight in quantized_model.state_dict().items():
    #         if "weight" in name or "bias" in name:
    #             # Directly copy the data for compatible parameters
    #             original_model.state_dict()[name].copy_(quantized_weight)
        
    with torch.no_grad():
        for name, param in original_model.named_parameters():
            if 'conv' in name or 'linear' in name:
                quantized_param = quantized_model.state_dict()[name]
                param.copy_(quantized_param)

    return original_model

def evaluate_model_quant(args, model, dequant_model, dataloader, device):
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

    model = model.to('cpu')

    seqs = []
    t0 = time.perf_counter()
    targets = []

    with torch.no_grad():
        for data, target, *_ in dataloader:
            targets.extend(torch.unbind(target, 0))
            data = data.to('cpu')

            log_probs = model(data)

            log_probs = log_probs.to('cuda')
            dequant_model = dequant_model.to('cuda')

            if hasattr(dequant_model, 'decode_batch'):
                seqs.extend(dequant_model.decode_batch(log_probs))
            else:
                seqs.extend([dequant_model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

    duration = time.perf_counter() - t0

    refs = [decode_ref(target, dequant_model.alphabet) for target in targets]
    accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
    print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

def evaluate_model(args, model, dataloader, device):
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

    seqs = []
    t0 = time.perf_counter()
    targets = []

    with torch.no_grad():
        for data, target, *_ in dataloader:
            targets.extend(torch.unbind(target, 0))
            
            data = data.to('cpu')
            model = model.to('cpu')

            log_probs = model(data)

            log_probs = log_probs.to('cuda')
            model = model.to('cuda')

            if hasattr(model, 'decode_batch'):
                seqs.extend(model.decode_batch(log_probs))
            else:
                seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

    duration = time.perf_counter() - t0

    refs = [decode_ref(target, model.alphabet) for target in targets]
    accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
    print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

# def evaluate_model_auto(args, model, dataloader, device):
#     accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

#     seqs = []
#     t0 = time.perf_counter()
#     targets = []

#     with torch.no_grad():
#         for data, target, *_ in dataloader:
#             targets.extend(torch.unbind(target, 0))
#             data = data.half()
#             data = data.to('cpu')
#             model = model.to('cpu')

#             log_probs = model(data)

#             log_probs = log_probs.to('cuda')
#             model = model.to('cuda')

#             if hasattr(model, 'decode_batch'):
#                 seqs.extend(model.decode_batch(log_probs))
#             else:
#                 seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

#     duration = time.perf_counter() - t0

#     refs = [decode_ref(target, model.alphabet) for target in targets]
#     accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

#     print("* mean      %.2f%%" % np.mean(accuracies))
#     print("* median    %.2f%%" % np.median(accuracies))
#     print("* time      %.2f" % duration)
#     print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

def time_evaluation(args, model, dataloader):
    """
        time evaluate: both use cpu
    """

    t0 = time.perf_counter()

    with torch.no_grad():
        for data, target, *_ in dataloader:
            
            data = data.to('cpu')
            model = model.to('cpu')

            log_probs = model(data)

            log_probs = log_probs.to('cuda')
            model = model.to('cuda')

    duration = time.perf_counter() - t0

    print("* time      %.2f" % duration)
    print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))


""""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print parameter number!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

def main(args):

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    if not args.pretrained:
        config = toml.load(args.config)
    else:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        pretrain_file = os.path.join(dirname, 'config.toml')
        config = toml.load(pretrain_file)
        if 'lr_scheduler' in config:
            print(f"[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']

    argsdict = dict(training=vars(args))

    print("[loading model]")
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False)
    else:
        model = load_symbol(config, 'Model')(config)

    for name, param in model.named_parameters():
        print(name, param.data.dtype)

    print("[loading data]")
    try:
        train_loader_kwargs, valid_loader_kwargs = load_numpy(
            args.chunks, args.directory, valid_chunks = args.valid_chunks
        )
    except FileNotFoundError:
        train_loader_kwargs, valid_loader_kwargs = load_script(
            args.directory,
            seed=args.seed,
            chunks=args.chunks,
            valid_chunks=args.valid_chunks,
            n_pre_context_bases=getattr(model, "n_pre_context_bases", 0),
            n_post_context_bases=getattr(model, "n_post_context_bases", 0),
        )

    loader_kwargs = {
        "batch_size": args.batch, "num_workers": 4, "pin_memory": True
    }
    train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
    valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)

    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))
    torch.save(model.state_dict(), os.path.join(workdir, "origin.tar"))

    # Evaluate the performance of the model before dynamic quantization
    print('*'*50)
    # evaluate_model(args, model, train_loader, args.device)
    # print('*'*50)
    evaluate_model(args, model, valid_loader, args.device)
    print('*'*50)

    print("Original Model Keys:")
    for key in model.state_dict().keys():
        print(key)

    model.to('cpu')  # Move the model to CPU for quantization
    print(type(model), model.seqdist)

    # Apply dynamic quantization to the LSTM and linear layers
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.LSTM, torch.nn.Linear},  # Specify the types of layers to quantize
        dtype=torch.qint8  # Use 8-bit integer quantization
    )
    
    torch.save(quantized_model.state_dict(), os.path.join(workdir, "quantized_model.tar"))
    torch.save(quantized_model.state_dict(), os.path.join(workdir, "quantized_model_config.toml"))
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    quantized_model.eval()

    print(type(quantized_model), quantized_model.seqdist)

    # # Prepare for evaluation
    # model_copy = copy.deepcopy(model)
    # dequantized_model = model_dequantization(quantized_model, model_copy)
    # dequantized_model.to(args.device)

    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False)
    else:
        model = load_symbol(config, 'Model')(config)

    model.to('cpu')
    model.eval()

    print('*'*50)
    # print("in evaluation")
    # evaluate_model_quant(args, quantized_model, model, train_loader, args.device)
    # print('*'*50)
    evaluate_model_quant(args, quantized_model, model, valid_loader, args.device)
    print('*'*50)

    size_model1 = os.path.getsize(os.path.join(workdir, "origin.tar"))
    size_model2 = os.path.getsize(os.path.join(workdir, "quantized_model.tar"))
    print("Size of Model 1:", size_model1, "bytes")
    print("Size of Model 2:", size_model2, "bytes")

    params_model1 = get_parameters_count(model)
    params_model2 = get_parameters_count(quantized_model)
    print("Params of Model 1:", params_model1)
    print("Params of Model 2:", params_model2)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default='dna_r9.4.1_e8_fast@v3.4')
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-3')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=1000, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    parser.add_argument("--quantile-grad-clip", action="store_true", default=False)
    return parser
