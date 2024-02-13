import os
import torch
import time
import toml
import numpy as np
from bonito.util import accuracy, decode_ref, permute, get_parameters_count
from torch.quantization import QuantStub, DeQuantStub

class QuantizedModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModelWrapper, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        y = self.model(x)
        y = self.dequant(y)
        return y
        # try:
        #     x = self.quant(x)
        # except NotImplementedError:
        #     print("&" * 100)
        #     print("x = self.quant(x)")
        #     print("&" * 100)
        #     return None
            
        # try:
        #     y = self.model(x)
        # except NotImplementedError:
        #     print("&" * 100)
        #     print("y = self.model(x)")
        #     print("&" * 100)
        #     return None
        
        # try:
        #     y = self.dequant(y)
        # except NotImplementedError:
        #     print("&" * 100)
        #     print("y = self.dequant(y)")
        #     print("&" * 100)
        #     return None
        
        # return y
        
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

def evaluate_accuracy(args, model, dataloader, dequant_model=None):
    model.eval()

    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq)

    seqs = []
    t0 = time.perf_counter()
    targets = []

    support_model = dequant_model if dequant_model is not None else model

    with torch.no_grad():
        for data, target, *_ in dataloader:
            targets.extend(torch.unbind(target, 0))

            model = model.to('cpu')
            data = data.to('cpu')

            log_probs = model(data)

            log_probs = log_probs.to('cuda')
            support_model = support_model.to('cuda')

            if hasattr(support_model, 'decode_batch'):
                seqs.extend(support_model.decode_batch(log_probs))
            else:
                seqs.extend([support_model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])

    duration = time.perf_counter() - t0

    refs = [decode_ref(target, support_model.alphabet) for target in targets]
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

def evaluate_model_storage_compression_rate(model_path1, model_path2, workdir):
    size_model1 = os.path.getsize(os.path.join(workdir, model_path1))
    size_model2 = os.path.getsize(os.path.join(workdir, model_path2))
    print("Size of Model 1:", size_model1, "bytes")
    print("Size of Model 2:", size_model2, "bytes")

def save_quantized_model(model, config, argsdict, workdir, file_path):
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))
    torch.save(model.state_dict(), os.path.join(workdir, file_path))