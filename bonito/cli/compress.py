#!/usr/bin/env python3

"""
Bonito model compression.
"""

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from bonito.data import load_numpy, load_script
from bonito.util import __models__, default_config, default_data
from bonito.util import load_model, load_symbol, init, half_supported
from bonito.training import load_state, Trainer

import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import LSTM
from torch.optim import AdamW

import warnings

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

    # # Run on cuda
    # val_loss, val_mean, val_median = test(model, device, valid_loader, criterion=criterion)
    # print("\n[start] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(workdir, val_loss, val_mean, val_median))
    # with open(os.path.join(workdir, 'accuracy.txt'), 'w') as accuracy_log:
    #     accuracy_log.write("[start] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(workdir, val_loss, val_mean, val_median))

    # for pruning_iter in range(1, args.pruning_iterations + 1):
    #     # Pruning
    #     print("Before pruning, model has %d params\n" % get_parameters_count(model))
    #     parameters_to_prune = model.get_parameters_to_prune()
    #     pruning_amount = 1 - (1 - args.prune_level) ** pruning_iter
    #     print("Pruning amount: %.3f" % pruning_amount)
    #     if args.structured:
    #         for module, param in parameters_to_prune:
    #             prune.ln_structured(module, param, amount=args.prune_level, n=1, dim=0)
    #     else:
    #         prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=args.prune_level)

    #     print("After pruning, model has %d params\n" % get_parameters_count(model))

    #     # Finetuning pruned model between iterations
    #     lr_scheduler = func_scheduler(
    #         optimizer, cosine_decay_schedule(1.0, 0.1), args.epochs * len(train_loader),
    #         warmup_steps=500, start_step=last_epoch*len(train_loader)
    #     )

    #     val_loss, val_mean, val_median = test(model, device, valid_loader, criterion=criterion)
    #     print("\n[prune {}] [untuned] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(pruning_iter, workdir, val_loss, val_mean, val_median))
    #     with open(os.path.join(workdir, 'accuracy.txt'), 'a') as accuracy_log:
    #         accuracy_log.write("\n[prune {}] [untuned] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(pruning_iter, workdir, val_loss, val_mean, val_median))

    #     for epoch in range(1 + last_epoch, args.epochs + 1 + last_epoch):
    #         try:
    #             with CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
    #                 train_loss, duration = train(
    #                     model, device, train_loader, optimizer, criterion=criterion,
    #                     use_amp=args.amp, lr_scheduler=lr_scheduler,
    #                     loss_log = loss_log
    #                 )

    #             torch.save(model.state_dict(), os.path.join(workdir, "weights_%s_%s.tar" % (pruning_iter, epoch)))

    #             val_loss, val_mean, val_median = test(
    #                 model, device, valid_loader, criterion=criterion
    #             )
    #         except KeyboardInterrupt:
    #             break

    #         print("\n[prune {}] [epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
    #             pruning_iter, epoch, workdir, val_loss, val_mean, val_median
    #         ))

    #         with open(os.path.join(workdir, 'accuracy.txt'), 'a') as accuracy_log:
    #             accuracy_log.write("\n[prune {}] [epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
    #                 pruning_iter, epoch, workdir, val_loss, val_mean, val_median
    #             ))

    #         with CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
    #             training_log.append(OrderedDict([
    #                 ('time', datetime.today()),
    #                 ('duration', int(duration)),
    #                 ('pruning_iter', pruning_iter),
    #                 ('epoch', epoch),
    #                 ('train_loss', train_loss),
    #                 ('validation_loss', val_loss),
    #                 ('validation_mean', val_mean),
    #                 ('validation_median', val_median)
    #             ]))

    #     torch.save(model.state_dict(), os.path.join(workdir, "weights_prune_%s.tar" % pruning_iter))

    # # Making pruned parameterisation permanent
    # for module, param in parameters_to_prune:
    #     prune.remove(module, param)

    # # prep_for_save() follows this: https://github.com/pytorch/pytorch/issues/33618
    # model.prep_for_save()

    # torch.save(model.state_dict(), os.path.join(workdir, "weights_final.tar"))
    # print("After pruning, model has %d params\n" % get_parameters_count(model))

    # # Sparsifying: Sparsification of model parameters is a technique used to reduce the 
    # # memory footprint and computational cost of deep learning models, 
    # # particularly when dealing with large models with many parameters.
    # model_state = model.state_dict()
    # for param_tensor in model_state:
    #     model_state[param_tensor] = model_state[param_tensor].to_sparse()
    # torch.save(model_state, os.path.join(workdir, "weights_final_sparse.tar"))


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory") # The new data and re-trained model will be saved here
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--directory", default=default_data) # Dataset dirctory
    parser.add_argument("--device", default="cuda") # Can be cpu
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--val_chunks", default=1000, type=int)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--pretrained", default="dna_r9.4.1_e8_fast@v3.4")
    parser.add_argument("--weights", default="0",type=str) # Suffix of weights file to use
    parser.add_argument("--prune_level", default=0.6, type=float)
    parser.add_argument("--structured", action="store_true", default=False)
    parser.add_argument("--pruning_iterations", default=1, type=int)
    return parser 
