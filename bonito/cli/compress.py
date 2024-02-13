#!/usr/bin/env python3

"""
Bonito model compression.
"""

from ast import arg
import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

from bonito.data import load_numpy, load_script
from bonito.util import __models__, default_config
from bonito.util import load_model, load_symbol, init
from bonito.cli.quantization import model_structure_comparison, evaluate_accuracy, evaluate_time_cpu, evaluate_model_storage_compression_rate, save_quantized_model

import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic

def main(args):
    '''
    Prepare: workdir, device, load pretrained model, load dataset
    '''
    # workdir creation
    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    if args.device == 'cpu' and (not args.compare_time or args.quantized is None or not args.evaluate):
        print("[error] only evaluate time spent for models before and after qantization")
        exit(1)

    # device preparation
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

    # load pre-trained model
    print("[loading model]")
    if args.pretrained:
        print("[using pretrained model {}]".format(args.pretrained))
        model = load_model(args.pretrained, device, half=False)
    else:
        model = load_symbol(config, 'Model')(config)

    # load data
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

    # save config file and original pretrained model if not just evaluate
    if not args.evaluate:
        os.makedirs(workdir, exist_ok=True)
        argsdict = dict(training=vars(args))
        toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))
        torch.save(model.state_dict(), os.path.join(workdir, "weights_orig.tar"))

    '''
    Quantization
    '''
    if not args.evaluate:
        print("[quantize pre-trained model]")

        model.to('cpu')  # Move the model to CPU for quantization

        # Apply dynamic quantization to the LSTM and linear layers
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.LSTM, torch.nn.Linear},  # Specify the types of layers to quantize
            dtype=torch.qint8  # Use 8-bit integer quantization
        )
        model_state = quantized_model.module.state_dict() if hasattr(quantized_model, 'module') else quantized_model.state_dict()
        torch.save(model_state, os.path.join(workdir, "weights_quant.tar"))
    else:
        print('[load quantized model]')
        quantized_model = load_model(args.quantized, device, half=False, weights='quant')

    '''
    Evaluation
    '''
    model.eval()
    quantized_model.eval()

    if args.device == 'cuda':
        print("[compare model accuracy before and after quantization]")
        print("Before:")
        evaluate_accuracy(args, model, valid_loader)
        print()
        print("After:")
        evaluate_accuracy(args, quantized_model, valid_loader, model)
        print("Before:")

        print("[compare model size before and after quantization]")
        evaluate_model_storage_compression_rate("weights_orig.tar", "weights_quant.tar", workdir)

        print("[compare model structure before and after quantization]")
        model_structure_comparison(model, quantized_model, workdir)

    '''
    Evaluate time
    '''
    if args.device == 'cpu':
        print(['evaluate time on CPU'])
    else:
        print(['evaluate time on GPU'])

    print("Before:")
    evaluate_time_cpu(args, model, valid_loader)
    print()
    print("After:")
    evaluate_time_cpu(args, quantized_model, valid_loader)
    print()

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
    parser.add_argument("--device", default="cuda") # If 'cpu' then evaluate time spent.
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
    parser.add_argument('--compare_time', default=False) # compare time spent for prediction before and after quantization (must be on CPU)
    parser.add_argument('--quantized', default=None, type=Path) # If compare_time is True, then give the path to quantized model as well.
    parser.add_argument('--evaluate', default=False) # If only want to evaluate

    return parser
