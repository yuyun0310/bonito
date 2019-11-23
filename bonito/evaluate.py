"""
Bonito model evaluator
"""

import argparse
import time
import torch
import numpy as np
from itertools import starmap

from bonito.util import init, load_data, load_model
from bonito.util import decode_ctc, decode_ref, accuracy, poa


def main(args):

    init(args.seed, args.device)

    print("* loading data")
    poas = []
    chunks, targets, _ = load_data(limit=args.chunks, shuffle=args.shuffle)
    references = list(map(decode_ref, targets))

    for w in [int(i) for i in args.weights.split(',')]:

        print("* loading model", w)
        model = load_model(args.model_directory, args.device, weights=w)

        print("* calling")

        p = []

        t0 = time.perf_counter()

        with torch.no_grad():
            for i in range(0, int(args.chunks / args.batchsize)):
                tchunks = torch.tensor(np.expand_dims(chunks[i*args.batchsize:(i+1)*args.batchsize], axis=1))
                predictions = torch.exp(model(tchunks.to(args.device)))
                predictions = predictions.cpu()
                p.append(predictions.numpy())

        predictions = np.concatenate(p)

        duration = time.perf_counter() - t0

        sequences = list(map(decode_ctc, predictions))
        accuracies = list(starmap(accuracy, zip(references, sequences)))

        if args.poa: poas.append(sequences)

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)
        print("* samples/s %.2E" % (args.chunks * chunks.shape[1] / duration))

    if args.poa:
        print("* doing poa")
        # transpose so chunks are groups together
        poas = [list(batch) for batch in zip(*poas)]

        t0 = time.perf_counter()
        accuracies = list(starmap(accuracy, zip(references, poa(poas))))
        duration = time.perf_counter() - t0

        print("* mean poa      %.2f%%" % np.mean(accuracies))
        print("* median poa   %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("model_directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--chunks", default=500, type=int)
    parser.add_argument("--batchsize", default=100, type=int)
    parser.add_argument("--poa", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    return parser
