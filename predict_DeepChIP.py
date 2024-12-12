import argparse
import datetime
import json
import socket
from pathlib import Path
from typing import Callable, Iterable, Tuple

import models
import numpy as np
import torch
import train_DeepChIP
import utils

# import torcheval
from numpy.typing import ArrayLike
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parsing() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Parameters
    ----------
    python command-line

    Returns
    -------
    argparse.Namespace
        Namespace with all arguments as attributes
    """
    # Declaration of expected arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Config file from model training",
        type=str,
    )
    args, unparsed = parser.parse_known_args()
    if args.config:
        config = json.load(open(args.config))
        print("Using config file instead of default parameters")
    # populate default values with config if specified
    parser.add_argument(
        "-d",
        "--dataset_dir",
        help="Directory of the dataset to use for training",
        type=str,
        default=config["dataset_dir"] if config else None,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory name. Must not already exist",
        type=str,
        default=config["output_dir"] if config else None,
    )
    parser.add_argument(
        "-sp",
        "--split",
        help="Dataset split to predict on",
        nargs="+",
        default=["train", "valid", "test"],
        type=str,
    )
    parser.add_argument(
        "-arch",
        "--architecture",
        help="Name of the model architecture",
        type=str,
        default=config["architecture"] if config else None,
    )
    parser.add_argument(
        "-m",
        "--model_state",
        help="pytorch file with model state",
        type=str,
        default=str(Path(config["output_dir"], "model_state.pt")) if config else None,
    )
    parser.add_argument(
        "--paired",
        help="Indicates that dataset consists of paired reads",
        action="store_true",
        default=config["paired"] if config else False,
    )
    parser.add_argument(
        "-s",
        "--strand",
        help="Strand to perform training on, choose between 'for', 'rev' or "
        "'both' (default: %(default)s)",
        default=config["strand"] if config else "both",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of samples to use for training in parallel (default: %(default)s)",
        default=config["batch_size"] if config else 1024,
        type=int,
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        help="Number of parallel workers for preprocessing batches (default: %(default)s)",
        default=config["num_workers"] if config else 4,
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Indicates to show information messages",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def predict(model, dataloader, device, paired=True):
    res = []
    with torch.no_grad():
        for X, *_ in dataloader:
            if paired:
                X1, X2 = X
                X1, X2 = X1.to(device), X2.to(device)
                logits = model(X1, X2)
            else:
                X = X.to(device)
                logits = model(X)
            res.append(torch.sigmoid(logits).cpu())
    return torch.concat(res).squeeze()


def main(args):
    for split in args.split:
        dataset = train_DeepChIP.DatasetFromFiles(
            args.dataset_dir,
            split=split,
            strand=args.strand,
            paired=args.paired,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = args.architecture().to(args.device)
        model.load_state_dict(torch.load(args.model_state))
        preds = predict(model, dataloader, args.device)
        np.savez_compressed(
            Path(args.output_dir, f"preds_on_{Path(args.dataset_dir).name}_{split}"),
            preds=preds,
        )
        y = torch.tile(torch.tensor([1, 0]), (len(preds) // 2,)).reshape_as(preds)
        table = train_DeepChIP.tp_fp_tn_fn(preds, y, thres=0.5)
        acc = train_DeepChIP.accuracy(*table)
        prec = train_DeepChIP.precision(*table)
        rec = train_DeepChIP.recall(*table)
        tnr = train_DeepChIP.true_negative_rate(*table)
        with open(args.log_file, "a") as f:
            f.write(
                f"On {split}:\ttp={table[0]}\tfp={table[1]}\ttn={table[2]}\tfn={table[3]}\n"
            )
            f.write(
                f"accuracy = {acc}\nprecision = {prec}\nrecall = {rec}\ntrue negative rate = {tnr}\n"
            )


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    # Get arguments
    args = parsing()
    # Get cpu, gpu or mps device for training.
    args.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.verbose:
        print(f"Using {args.device} device")

    # Build output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Store arguments in config file
    config_filename = utils.safe_filename(
        Path(args.output_dir, f"preds_on_{Path(args.dataset_dir).name}_config.json")
    )
    args.log_file = str(
        utils.safe_filename(
            Path(args.output_dir, f"preds_on_{Path(args.dataset_dir).name}_log.txt")
        )
    )
    with open(config_filename, "w") as f:
        json.dump(
            {
                **vars(args),
                **{
                    "timestamp": str(tmstmp),
                    "machine": socket.gethostname(),
                },
            },
            f,
            indent=4,
        )
        f.write("\n")
    # Convert to non json serializable objects
    args.architecture = models.ARCHITECTURES[args.architecture]
    # Start computations, save total time even if there was a failure
    try:
        main(args)
    except:
        with open(args.log_file, "a") as f:
            f.write("Aborted\n")
            f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
        raise
    with open(args.log_file, "a") as f:
        f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
