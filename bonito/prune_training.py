"""
Bonito train
"""

import os
import re
import csv
from glob import glob
from functools import partial
from time import perf_counter
from collections import OrderedDict

from bonito.util import accuracy, decode_ref, permute, concat, match_names

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn.functional import ctc_loss
from torch.optim.lr_scheduler import LambdaLR

try: from apex import amp
except ImportError: pass


class CSVLogger:
    def __init__(self, filename):
        self.filename = str(filename)
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                self.columns = csv.DictReader(f).fieldnames
        else:
            self.columns = None
        self.fh = open(self.filename, 'a', newline='')
        self.csvwriter = csv.writer(self.fh, delimiter=',')
        self.count = 0

    def set_columns(self, columns):
        if self.columns:
            raise Exception('Columns already set')
        self.columns = list(columns)
        self.csvwriter.writerow(self.columns)

    def append(self, row):
        if self.columns is None:
            self.set_columns(row.keys())
        self.csvwriter.writerow([row.get(k, '-') for k in self.columns])
        self.count += 1
        if self.count > 100:
            self.count = 0
            self.fh.flush()

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ChunkDataSet:
    def __init__(self, chunks, targets, lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)


def const_schedule(y):
    """
    Constant Scheduler
    """
    return lambda t: y


def linear_schedule(y0, y1):
    """
    Linear Scheduler
    """
    return lambda t: y0 + (y1 - y0) * t


def cosine_decay_schedule(y0, y1):
    """
    Cosine Decay Scheduler
    """
    return lambda t: y1 + 0.5 * (y0 - y1) * (np.cos(t * np.pi) + 1.0)


def piecewise_schedule(knots, funcs):
    """
    Piecewise Scheduler
    """
    def f(t):
        i = np.searchsorted(knots, t)
        t0 = 0.0 if i == 0 else knots[i - 1]
        t1 = 1.0 if i == len(knots) else knots[i]
        return funcs[i]((t - t0) / (t1 - t0))
    return f


def func_scheduler(optimizer, func, total_steps, warmup_steps=None, warmup_ratio=0.1, start_step=0):
    """
    Learning Rate Scheduler
    """
    if warmup_steps:
        y0 = func(0.0)
        func = piecewise_schedule(
            [warmup_steps / total_steps],
            [linear_schedule(warmup_ratio * y0, y0), func]
        )
    return LambdaLR(optimizer, (lambda step: func((step + start_step) / total_steps)))


def load_state(dirname, device, model, optim, use_amp=False):
    """
    Load a model and optimizer state dict from disk
    """
    model.to(device)

    if use_amp:
        try:
            model, optimizer = amp.initialize(model, optim, opt_level="O1", verbosity=0)
        except NameError:
            print("[error]: Cannot use AMP: Apex package needs to be installed manually, See https://github.com/NVIDIA/apex")
            exit(1)

    weight_no = None

    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    if weight_files:
        weight_no = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])

    if weight_no:
        print("[picking up from epoch %s]" % weight_no)
        state_dict = torch.load(
            os.path.join(dirname, 'weights_%s.tar' % weight_no), map_location=device
        )
        state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        epoch = weight_no
    else:
        epoch = 0

    return epoch


def ctc_label_smoothing_loss(log_probs, targets, lengths, weights):
    T, N, C = log_probs.shape
    log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
    loss = ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean')
    label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
    return {'loss': loss + label_smoothing_loss, 'ctc_loss': loss, 'label_smooth_loss': label_smoothing_loss}


def train(model, device, train_loader, optimizer, use_amp=False, criterion=None, lr_scheduler=None, loss_log=None):

    if criterion is None:
        C = len(model.alphabet)
        weights = torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)]).to(device)
        criterion = partial(ctc_label_smoothing_loss, weights=weights)

    chunks = 0
    model.train()
    t0 = perf_counter()

    progress_bar = tqdm(
        total=len(train_loader), desc='[0/{}]'.format(len(train_loader.dataset)),
        ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
    )
    smoothed_loss = None
    max_norm = 2.0

    with progress_bar:

        for data, targets, lengths in train_loader:

            optimizer.zero_grad()

            chunks += data.shape[0]
            log_probs = model(data.to(device))
            losses = criterion(log_probs, targets.to(device), lengths.to(device))

            if not isinstance(losses, dict):
                losses = {'loss': losses}

            if use_amp:
                with amp.scale_loss(losses['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                losses['loss'].backward(retain_graph=True)

            params = amp.master_params(optimizer) if use_amp else model.parameters()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm).item()

            optimizer.step()

            if lr_scheduler is not None: lr_scheduler.step()

            losses = {k: v.item() for k,v in losses.items()}

            smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)

            progress_bar.set_postfix(loss='%.4f' % smoothed_loss)
            progress_bar.set_description("[{}/{}]".format(chunks, len(train_loader.dataset)))
            progress_bar.update()

            if loss_log is not None:
                loss_log.append({'chunks': chunks, 'time': perf_counter() - t0, 'grad_norm': grad_norm, **losses})

    return smoothed_loss, perf_counter() - t0


def test(model, device, test_loader, min_coverage=0.5, criterion=None):

    if criterion is None:
        C = len(model.alphabet)
        weights = torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)]).to(device)
        criterion = partial(ctc_label_smoothing_loss, weights=weights)

    seqs = []
    model.eval()
    test_loss = 0
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=min_coverage)

    with torch.no_grad():
        for batch_idx, (data, target, lengths) in enumerate(test_loader, start=1):
            log_probs = model(data.to(device))
            loss = criterion(log_probs, target.to(device), lengths.to(device))
            test_loss += loss['ctc_loss'] if isinstance(loss, dict) else loss
            if hasattr(model, 'decode_batch'):
                seqs.extend(model.decode_batch(log_probs))
            else:
                seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

    refs = [
        decode_ref(target, model.alphabet) for target in test_loader.dataset.targets
    ]
    accuracies = [
        accuracy_with_cov(ref, seq)[0] if len(seq) else 0. for ref, seq in zip(refs, seqs)
    ]

    mean = np.mean(accuracies)
    median = np.median(accuracies)
    return test_loss.item() / batch_idx, mean, median
