import os
import torch
import time
import toml
import numpy as np
from bonito.util import accuracy, decode_ref, permute, get_parameters_count
from bonito.training import ClipGrad, load_state
from bonito.io import CSVLogger
from memory_profiler import memory_usage
from torch.quantization import QuantStub, DeQuantStub
from datetime import datetime

from bonito.schedule import linear_warmup_cosine_decay
from bonito.util import accuracy, decode_ref, permute
from time import perf_counter
import torch.cuda.amp as amp
from tqdm import tqdm

class QuantizedModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModelWrapper, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
        
def static_quantization_wrapper(model):
    wrapped_model = QuantizedModelWrapper(model)
    return wrapped_model


def model_structure_comparison(model1, model2, workdir, report_file='model_comparison_report.txt'):
    '''
    The purpose of ths function is to check validity of evaluation method adopted.
    
    In evaluation, only model.decode_batch() needs a GPU environment.
    model.decode_batch(x) is only associated with input x and model.seqdist.
    model.seqdist is of type CTC_CRF (extend SequenceDist from koi.ctc [not open source]),
    which only related with models' alphabet, state_len and n_base.

    Besides, it is noticed that the config file will not change after quantization. Hence,
    model.decode_batch(x) could be shared with models. Only the predict output needs to be
    exactly from 2 models (before and after quantization), and this can be launched purely
    on CPU.
    '''
    with open(os.path.join(workdir, report_file), 'w') as f:
        def write_both(message):
            print(message)
            f.write(message + '\n')

        # Function to print and write layer details
        def report_model_structure(model, model_name):
            write_both(f"Type of {model_name}: {type(model)}")
            # Assuming the model has these attributes; adjust as necessary
            write_both(f"{model_name} Sequence Distribution: {model.seqdist}")
            write_both(f"{model_name} Base: {model.seqdist.n_base}, State Length: {model.seqdist.state_len}, Alphabet: {model.seqdist.alphabet}")
            write_both(f"{model_name} Configuration: {model.config}")
            write_both(f"{model_name} Structure:")
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # To avoid printing containers
                    params = sum(p.numel() for p in module.parameters())
                    write_both(f"  {name} - {type(module).__name__}: {params} parameters")

        # Report structure and parameters for model1
        report_model_structure(model1, "Model 1")
        
        # Report structure and parameters for model2
        report_model_structure(model2, "Model 2")

        # Using get_parameters_count to obtain total parameter counts
        params_model1 = get_parameters_count(model1)
        params_model2 = get_parameters_count(model2)
        write_both(f"\nTotal Parameters in Model 1: {params_model1}")
        write_both(f"Total Parameters in Model 2: {params_model2}")
        
def evaluate_accuracy(args, model, dataloader):
    print("evaluate ")
    model.eval()

    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

    seqs = []
    t0 = time.perf_counter()
    targets = []

    with torch.no_grad():
        for data, target, *_ in dataloader:
            targets.extend(torch.unbind(target, 0))

            model = model.to('cpu')
            data = data.to('cpu')

            log_probs = model(data)

            log_probs = log_probs.to('cuda')

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

def evaluate_time_cpu(args, model, dataloader):
    '''
    Evaluate run speed on CPU.
    '''
    model.eval()
    model = model.to('cpu')

    t0 = time.perf_counter()

    with torch.no_grad():
        for data, *_ in dataloader:
            data = data.to('cpu')
            model(data)

    duration = time.perf_counter() - t0

    print("* time      %.2f" % duration)
    print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

def runtime_simulator(model, dataloader):
    model.eval()
    model = model.to('cpu')

    with torch.no_grad():
        for data, *_ in dataloader:
            data = data.to('cpu')
            model(data)

def evaluate_runtime_memory(model, dataloader):
    mem_usage = memory_usage((runtime_simulator, (model, dataloader)), interval=0.5)
    print("Memory usage (in MB/0.5 sec):", mem_usage)
    print("Average memory usage (in MB/0.5 sec):", sum(mem_usage)/len(mem_usage))

def evaluate_model_storage_compression_rate(model_path1, model_path2, workdir):
    size_model1 = os.path.getsize(os.path.join(workdir, model_path1))
    size_model2 = os.path.getsize(os.path.join(workdir, model_path2))
    print("Size of Model 1:", size_model1, "bytes")
    print("Size of Model 2:", size_model2, "bytes")

def save_quantized_model(model, config, argsdict, workdir, file_path):
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))
    torch.save(model.state_dict(), os.path.join(workdir, file_path))

class QuantizedFineTuner:
    def __init__(
        self, model, train_loader, valid_loader, device, criterion=None,
        use_amp=False, lr_scheduler_fn=None, restore_optim=False,
        save_optim_every=10, grad_accum_split=1, quantile_grad_clip=False
    ):
        self.model = model.to('cpu')
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or model.loss
        self.use_amp = use_amp
        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.save_optim_every = save_optim_every
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.optimizer = None
        if quantile_grad_clip:
            self.clip_grad = ClipGrad()
        else:
            self.clip_grad = lambda parameters: torch.nn.utils.clip_grad_norm_(parameters, max_norm=2.0).item()

    def train_one_step(self, batch):
        self.optimizer.zero_grad()

        losses = None
        with amp.autocast(enabled=self.use_amp):
            for batch_ in zip(
                *map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)
            ):
                data_, targets_, lengths_, *args = (x.to(self.device) for x in batch_)

                data_ = data_.to('cpu')
                targets_ = targets_.to('cpu')
                lengths_ = lengths_.to('cpu')

                scores_ = self.model(data_, *args)

                data_ = data_.to(self.device)
                targets_ = targets_.to(self.device)
                lengths_ = lengths_.to(self.device)
                scores_ = scores_.to(self.device)

                # print("Calculate loss in train one step")
                losses_ = self.criterion(scores_, targets_, lengths_)

                if not isinstance(losses_, dict): losses_ = {'loss': losses_}

                total_loss = losses_.get('total_loss', losses_['loss']) / self.grad_accum_split
                print(total_loss)
                total_loss.requires_grad_()
                self.scaler.scale(total_loss).backward()

                losses = {
                    k: ((v.item() / self.grad_accum_split) if losses is None else (v.item() / self.grad_accum_split) + losses[k])
                    for k, v in losses_.items()
                }

        self.scaler.unscale_(self.optimizer)
        grad_norm = self.clip_grad(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return losses, grad_norm

    def train_one_epoch(self, loss_log, lr_scheduler):
        t0 = perf_counter()
        chunks = 0
        self.model.train()

        progress_bar = tqdm(
            total=len(self.train_loader), desc='[0/{}]'.format(len(self.train_loader.sampler)),
            ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
        )
        smoothed_loss = None

        with progress_bar:

            for batch in self.train_loader:

                chunks += batch[0].shape[0]

                losses, grad_norm = self.train_one_step(batch)

                smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)

                progress_bar.set_postfix(loss='%.4f' % smoothed_loss)
                progress_bar.set_description("[{}/{}]".format(chunks, len(self.train_loader.sampler)))
                progress_bar.update()

                if loss_log is not None:
                    lr = lr_scheduler.get_last_lr() if lr_scheduler is not None else [pg["lr"] for pg in optim.param_groups]
                    if len(lr) == 1: lr = lr[0]
                    loss_log.append({
                        'chunks': chunks,
                        'time': perf_counter() - t0,
                        'grad_norm': grad_norm,
                        'lr': lr,
                        **losses
                    })

                if lr_scheduler is not None: lr_scheduler.step()

        return smoothed_loss, perf_counter() - t0

    def validate_one_step(self, batch):
        data, targets, lengths, *args = batch
        with amp.autocast(enabled=self.use_amp):
            scores = self.model(data.to('cpu'), *(x.to('cpu') for x in args))
            scores = scores.to(self.device)
            losses = self.criterion(scores, targets.to(self.device), lengths.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()

        if hasattr(self.model, 'decode_batch'):
            seqs = self.model.decode_batch(scores)
        else:
            seqs = [self.model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
        refs = [decode_ref(target, self.model.alphabet) for target in targets]

        n_pre = getattr(self.model, "n_pre_context_bases", 0)
        n_post = getattr(self.model, "n_post_context_bases", 0)
        if n_pre > 0 or n_post > 0:
            refs = [ref[n_pre:len(ref)-n_post] for ref in refs]

        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]

        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            seqs, refs, accs, losses = zip(*(self.validate_one_step(batch) for batch in self.valid_loader))
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss = np.mean([(x['loss'] if isinstance(x, dict) else x) for x in losses])
        return loss, np.mean(accs), np.median(accs)

    def init_optimizer(self, lr, **kwargs):
        if isinstance(lr, (list, tuple)):
            if len(list(self.model.children())) != len(lr):
                raise ValueError('Number of lrs does not match number of model children')
            param_groups = [{'params': list(m.parameters()), 'lr': v} for (m, v) in zip(self.model.children(), lr)]
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr[0], **kwargs)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)

    def get_lr_scheduler(self, epochs, last_epoch=0):
        return self.lr_scheduler_fn(self.optimizer, self.train_loader, epochs, last_epoch)

    def fit(self, workdir, epochs=1, lr=2e-3, **optim_kwargs):
        if self.optimizer is None:
            self.init_optimizer(lr, **optim_kwargs)

        last_epoch = 0

        if self.restore_optim:
        # override learning rate to new value
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["initial_lr"] = pg["lr"] = lr[i] if isinstance(lr, (list, tuple)) else lr

        lr_scheduler = self.get_lr_scheduler(epochs, last_epoch=last_epoch)

        for epoch in range(1 + last_epoch, epochs + 1):
            try:
                with CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                    train_loss, duration = self.train_one_epoch(loss_log, lr_scheduler)

                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))
                if epoch % self.save_optim_every == 0:
                    torch.save(self.optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))

                val_loss, val_mean, val_median = self.validate_one_epoch()
            except KeyboardInterrupt:
                break

            print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                epoch, workdir, val_loss, val_mean, val_median
            ))

            with CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                training_log.append({
                    'time': datetime.today(),
                    'duration': int(duration),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                    'validation_mean': val_mean,
                    'validation_median': val_median
                })
            
            print({
                'time': datetime.today(),
                'duration': int(duration),
                'epoch': epoch,
                'train_loss': train_loss,
                'validation_loss': val_loss,
                'validation_mean': val_mean,
                'validation_median': val_median
            })
        return self.model
