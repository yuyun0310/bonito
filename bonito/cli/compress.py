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
from bonito.util import load_model, load_symbol, init, half_supported, accuracy, permute, decode_ref
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

from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor

# Add the directory containing your module to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantization import convert_to_quantizable_model

def evaluate_model(args, model, dataloader, device):
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

    seqs = []
    t0 = time.perf_counter()
    targets = []

    with torch.no_grad():
        for data, target, *_ in dataloader:
            targets.extend(torch.unbind(target, 0))
            data = data.to(device)

            log_probs = model(data)

            if hasattr(model, 'decode_batch'):
                seqs.extend(model.decode_batch(log_probs))
            else:
                seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

    duration = time.perf_counter() - t0
    print("out")

    refs = [decode_ref(target, model.alphabet) for target in targets]
    accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
    print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

# def evaluate_model(args, model, dataloader, device):
#     accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

#     seqs = []
#     t0 = time.perf_counter()
#     targets = []

#     with torch.no_grad():
#         for data, target, *_ in dataloader:
#             targets.extend(torch.unbind(target, 0))
#             data = data.to(device)

#             log_probs = model(data).to(device)

#             # Directly use model.decode without checking for decode_batch
#             # Assuming permute function adjusts log_probs dimensions as needed
#             for p in log_probs.permute(1, 0, 2):  # Adjust permute() as necessary for your model's input format
#                 seq = model.decode(p)  # Decode each sequence individually
#                 seqs.append(seq)  # Append the decoded sequence to seqs list

#     duration = time.perf_counter() - t0

#     # Decoding reference sequences for comparison
#     refs = [decode_ref(target, model.alphabet) for target in targets]

#     # Calculating accuracies for each decoded sequence against its reference
#     accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

#     print("* mean      %.2f%%" % np.mean(accuracies))
#     print("* median    %.2f%%" % np.median(accuracies))
#     print("* time      %.2f" % duration)
#     print("* samples/s %.2E" % (len(data) * data.shape[2] / duration))

# # For pytorch-quantization, you might need to manually replace layers
# def replace_layers(model):
#     for name, module in model.named_children():
#         if isinstance(module, torch.nn.Linear):
#             quant_layer = QuantLinear(module.in_features, module.out_features, bias=module.bias is not None)
#             quant_layer.weight = module.weight
#             quant_layer.bias = module.bias
#             setattr(model, name, quant_layer)
#         elif isinstance(module, torch.nn.LSTM):
#             # Similar process for LSTM, but ensure you handle the complexities of LSTM layers
#             # This is a simplification, and you might need a more detailed conversion
#             setattr(model, name, QuantLSTM(module))  # This is a placeholder, adjust as needed
#         else:
#             replace_layers(module)

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
    evaluate_model(args, model, train_loader, args.device)
    print('*'*50)
    evaluate_model(args, model, valid_loader, args.device)
    print('*'*50)

    # model.to('cpu')  # Move the model to CPU for quantization

    # # Apply dynamic quantization to the LSTM and linear layers
    # quantized_model = quantize_dynamic(
    #     model,
    #     {torch.nn.LSTM, torch.nn.Linear},  # Specify the types of layers to quantize
    #     dtype=torch.qint8  # Use 8-bit integer quantization
    # )
    
    # # quantized_model.prep_for_save()
    # # quantized_model_path = 'path/to/save/quantized_model.tar'
    # torch.save(quantized_model.state_dict(), os.path.join(workdir, "quantized_model.tar"))
    # # torch.save(quantized_model.state_dict(), quantized_model_path)

    # # quantized_model = model.use_koi()
    # # print("[loading quantized_model]")
    # # if args.pretrained:
    # #     print("[using pretrained model {}]".format(args.pretrained))
    # #     quantized_model = load_model(args.pretrained, device, half=False, use_koi=True)
    # # else:
    # #     quantized_model = load_symbol(config, 'Model')(config)

    # # model = MyCustomModel(input_size, hidden_size, output_size)
    # # replace_layers(quantized_model)
    # # quantized_model = quantized_model.to(args.device)

    
    quant_modules.initialize()
    # Enable quantization for the entire model
    # quant_desc = QuantDescriptor(calib_method="histogram", num_bits=8)
    # quant_modules.initialize(model, quant_desc)

    # # Customize quantization configurations if necessary
    # quant_desc_input = QuantDescriptor(calib_method='histogram')
    # quant_modules.quantize_dynamic(model, qconfig_dict={'input': quant_desc_input, 'weight': quant_desc_input})
    # model.cuda()
    quantized_model = convert_to_quantizable_model(model)
    print(quantized_model)

    '''
    Train
    '''

    # optimizer = AdamW(quantized_model.parameters(), amsgrad=False, lr=args.lr)
    # criterion = quantized_model.seqdist.ctc_loss if hasattr(quantized_model, 'seqdist') else None

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        quantized_model, device, train_loader, valid_loader,
        use_amp=half_supported() and not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split,
        quantile_grad_clip=args.quantile_grad_clip
    )

    if (',' in args.lr):
        lr = [float(x) for x in args.lr.split(',')]
    else:
        lr = float(args.lr)
    trainer.fit(workdir, args.epochs, lr)

    quantized_model_retrained = trainer.model
    print(quantized_model_retrained)

    for name, param in quantized_model_retrained.named_parameters():
        print(name, param.data.dtype)

    print('*'*50)
    print("in evaluation")
    evaluate_model(args, quantized_model_retrained, valid_loader, args.device)
    print('*'*50)
    evaluate_model(args, quantized_model_retrained, train_loader, args.device)
    print('*'*50)

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
    parser.add_argument("--epochs", default=1, type=int)
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
