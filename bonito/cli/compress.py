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
from bonito.cli.quantization import evaluate_model_size, measure_dynamic_memory_usage, model_structure_comparison, evaluate_accuracy, evaluate_time_cpu, evaluate_model_static_memory, print_model_info, save_quantized_model, static_quantization_wrapper, evaluate_runtime_memory
from bonito.cli.quantization import QuantizedFineTuner

import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.quantization import quantize_dynamic
import copy
from torch.optim import AdamW
from importlib import import_module

# from bonito_compression.bonito import bonito
# from memory_profiler import memory_usage
from bonito.nn import layers

def main(args):
    current_backend = torch.backends.quantized.engine
    print(f"Current quantization backend: {current_backend}")
    torch.backends.quantized.engine = 'fbgemm'
    print(f"Quantization backend set to: {torch.backends.quantized.engine}")
    '''
    Prepare: workdir, device, load pretrained model, load dataset
    '''
    # workdir creation
    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    if args.device == 'cpu' and (not args.compare_time or not args.evaluate):
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

    print(type(model))
    print(type(model.encoder), model.encoder)
    print(layers)

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
    print("^"* 50)
    total_samples = 0
    total_batches = 0
    for batch in train_loader:
        # Update sample count and batch count
        total_samples += batch[0].size(0)  # Assuming each batch contains input samples in the first dimension
        total_batches += 1
        # print(len(batch))
    print("train_loader size:", total_samples)
    

    total_samples = 0
    total_batches = 0
    for batch in valid_loader:
        # Update sample count and batch count
        total_samples += batch[0].size(0)  # Assuming each batch contains input samples in the first dimension
        total_batches += 1
    print("val_loader size:", total_samples)
    print("^"* 50)

    # save config file and original pretrained model if not just evaluate
    if not args.evaluate:
        os.makedirs(workdir, exist_ok=True)
        argsdict = dict(training=vars(args))
        toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))
        torch.save(model.state_dict(), os.path.join(workdir, "weights_orig.tar"))

    '''
    Quantization
    '''
    print("[quantize pre-trained model]")
    model_copy = copy.deepcopy(model)

    model_copy.to('cpu')  # Move the model to CPU for quantization

    if args.dynamic:
        # Apply dynamic quantization to the LSTM and linear layers
        quantized_model = quantize_dynamic(
            model_copy,
            {torch.nn.LSTM, torch.nn.Linear},  # Specify the types of layers to quantize
            dtype=torch.qint8  # Use 8-bit integer quantization
        )
        model_state = quantized_model.module.state_dict() if hasattr(quantized_model, 'module') else quantized_model.state_dict()
        torch.save(model_state, os.path.join(workdir, "weights_quant_dynamic.tar"))
    
    elif args.static:
        quantized_model = static_quantization_wrapper(model_copy)
        quantized_model.to('cpu')

        # Set the model to evaluation mode
        quantized_model.eval()

        # Specify the layers to be quantized
        quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(quantized_model, inplace=True)

        # Assuming calibration_dataset is a DataLoader object providing input tensors
        with torch.no_grad():
            for data, *_ in train_loader:
                data = data.to('cpu')
                quantized_model(data)

        quantized_model.to('cpu')
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)

        model_state = quantized_model.module.state_dict() if hasattr(quantized_model, 'module') else quantized_model.state_dict()
        torch.save(model_state, os.path.join(workdir, "weights_quant_static.tar"))

        print(quantized_model)
        for name, param in quantized_model.named_parameters():
            print(name, type(param), param.size())
        for param in quantized_model.parameters():
            print(param)

    '''
    Fine Tune
    '''
    # calib = ['no_calib', 'fine_tune', 'kl_distil']

    if args.calib == 'fine_tune':
        quantized_model.train()
        print('fine fune')
        if config.get("lr_scheduler"):
            sched_config = config["lr_scheduler"]
            lr_scheduler_fn = getattr(
                import_module(sched_config["package"]), sched_config["symbol"]
            )(**sched_config)
        else:
            lr_scheduler_fn = None
        trainer = QuantizedFineTuner(
            quantized_model, train_loader, valid_loader,
            device=device,
            criterion=None,
            use_amp=False,
            lr_scheduler_fn=lr_scheduler_fn
        )

        if (',' in args.lr):
            lr = [float(x) for x in args.lr.split(',')]
        else:
            lr = float(args.lr)
        quantized_model = trainer.fit(workdir, args.epochs, lr)

    elif args.calib == 'kl_distil':
        # quantized_model = knowledge_distillation()

        pass
    else:
        pass

    '''
    Evaluation
    '''
    model.eval()

    # Currently, only dynamic quantized model can be evaluated.
    quantized_model.eval()

    print(model)
    print(quantized_model)

    if args.device == 'cuda':
        print("[compare model accuracy before and after quantization]")
        print("Before:")
        evaluate_accuracy(args, model, valid_loader)
        print()
        print("After:")
        evaluate_accuracy(args, quantized_model, valid_loader)
        print()

        print("[compare model size before and after quantization]")
        evaluate_model_static_memory("weights_orig.tar", "weights_quant_dynamic.tar", workdir)

        print("[compare model structure before and after quantization]")
        model_structure_comparison(model, quantized_model, workdir)

        print_model_info(model)
        print_model_info(quantized_model)

        # evaluate_model_size(model, quantized_model, workdir)
        print("#"*50)
        print("#"*50)
        print("measure_dynamic_memory_usage")
        measure_dynamic_memory_usage(model, valid_loader)
        print("_"*50)
        measure_dynamic_memory_usage(quantized_model, valid_loader)
        print("#"*50)
        print("#"*50)

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

    '''
    Evaluate Runtime Memory Usage
    '''
    if args.device == 'cpu':
        print(['evaluate runtime memory usage on CPU'])
    else:
        print(['evaluate runtime memory usage on GPU'])

    print("Before:")
    evaluate_runtime_memory(model, valid_loader)
    print()
    print("After:")
    evaluate_runtime_memory(quantized_model, valid_loader)
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
    # parser.add_argument('--quantized', default=None, type=Path) # If compare_time is True, then give the path to quantized model as well.
    parser.add_argument('--evaluate', default=False) # If only want to evaluate
    group_method = parser.add_mutually_exclusive_group()
    group_method.add_argument("--dynamic", action="store_true", default=False)
    group_method.add_argument("--static", action="store_true", default=False)
    parser.add_argument("--calib", default='no_calib', type=str)

    return parser
