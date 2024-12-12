import argparse
import datetime
import json
import socket
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Tuple

import models
import numpy as np
import torch
import utils
from numpy.typing import ArrayLike
from torch import nn
from torch.utils.data import DataLoader, Dataset

LOSSES = {"BCE": nn.BCELoss, "BCElogits": nn.BCEWithLogitsLoss}


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
        "-d",
        "--dataset_dir",
        help="Directory of the dataset to use for training",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory name. Must not already exist",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-arch",
        "--architecture",
        help="Name of the model architecture",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--paired",
        help="Indicates that dataset consists of paired reads",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--strand",
        help="Strand to perform training on, choose between 'for', 'rev' or "
        "'both' (default: %(default)s)",
        default="both",
        type=str,
    )
    parser.add_argument(
        "-lr",
        "--learn_rate",
        help="Value for learning rate (default: %(default)s)",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of training loops over the dataset (default: %(default)s)",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of samples to use for training in parallel (default: %(default)s)",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "-mt",
        "--max_train",
        help="Maximum number of batches per epoch for training (default: %(default)s)",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-mv",
        "--max_valid",
        help="Maximum number of batches per epoch for validation (default: %(default)s)",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        help="Number of parallel workers for preprocessing batches (default: %(default)s)",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-bal",
        "--balance",
        help="Indicates to balance weights inside batches",
        action="store_true",
    )
    parser.add_argument(
        "-loss",
        "--loss_fn",
        help="loss function for training (default: %(default)s)",
        default="BCElogits",
        type=str,
    )
    parser.add_argument(
        "-optim",
        "--optimizer_ctor",
        help="optimizer to use for training (default: %(default)s)",
        default="adam",
        type=str,
    )
    # parser.add_argument(
    #     "-rN",
    #     "--removeNs",
    #     help="Indicates to remove windows with Ns from training set",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "-r", "--remove_indices",
    #     help="Npz archive containing indices of labels to remove from "
    #          "training set, with one array per chromosome",
    #     default=None,
    #     type=str)
    # parser.add_argument(
    #     "--seed",
    #     help="Seed to use for random shuffling of training samples",
    #     default=None,
    #     type=int)
    parser.add_argument(
        "-p",
        "--patience",
        help="Number of epochs without improvement to wait before stopping "
        "training (default: %(default)s)",
        default=6,
        type=int,
    )
    # parser.add_argument(
    #     "-dist", "--distribute",
    #     action='store_true',
    #     help="Indicates to use both GPUs with MirrorStrategy.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Indicates to show information messages",
        action="store_true",
    )
    args = parser.parse_args()
    return args


class DatasetFromFiles(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str = "train",
        use_labels: bool = False,
        class_weights: dict = {0: 1, 1: 1},
        use_sample_weights: np.ndarray = False,
        file_suffix: str = None,
        transform: Callable = utils.idx_to_onehot,
        target_transform: Callable = None,
        strand: str = "for",
        paired: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        # Passed attributes
        self.split = split
        self.use_labels = use_labels
        self.class_weights = class_weights
        self.use_sample_weights = use_sample_weights
        self.transform = transform
        self.target_transform = target_transform
        self.strand = strand
        self.paired = paired
        self.rng = rng
        # Initializers for file parsing
        if use_labels:
            self.label_files = []
        if use_sample_weights:
            self.weight_files = []
        self.file_lengths = []
        self.file_mmaps = []
        self.shard_size = None
        self.read_shape = None
        self.dtype = None
        # Check file_suffix
        if file_suffix is None:
            # Check if first .npy file exists, if not assume data is in .npz
            if Path(dataset_dir, split + "_0.npy").exists():
                file_suffix = "npy"
            else:
                file_suffix = "npz"
        elif file_suffix not in {"npy", "npz"}:
            raise ValueError(
                f"file_suffix must be either npy or npz, not {file_suffix}"
            )
        # Check that files exists
        try:
            next(Path(dataset_dir).glob(split + "_*" + file_suffix))
        except StopIteration:
            raise FileNotFoundError(
                f"No {split} files found in {dataset_dir}. Files must be in "
                "numpy binary format, named as <split>_<number>.<npy or npz>"
            )
        # Prepare temporary dir to copy data from npz to npy for faster reading
        if file_suffix == "npz":
            self.tempdir = tempfile.TemporaryDirectory()
            self.dirname = self.tempdir.name
        else:
            self.dirname = dataset_dir
        # Loop over files to get info
        file_idx = 0
        file = Path(dataset_dir, f"{split}_{file_idx}.{file_suffix}")
        while file.exists():
            # Load file
            if file_suffix == "npy":
                reads = np.load(file)
                # Save file name and info
                self.file_lengths.append(len(reads))  # allowed to vary for last file
            elif file_suffix == "npz":
                with np.load(file) as f:
                    reads = f["reads"]
                # Copy as npy in temporary dir
                newfile = Path(self.dirname, file.stem + ".npy")
                np.save(newfile, reads)
                # Save file name and info
                self.file_lengths.append(len(reads))  # allowed to vary for last file
            if self.read_shape is None:
                self.read_shape = reads.shape[1:]
                self.dtype = reads.dtype
            else:
                if self.read_shape != reads.shape[1:]:
                    raise ValueError("All reads must have the same shape")
                if self.dtype != reads.dtype:
                    raise ValueError("All files must have the same dtype")
            # Store a memory map of the file to fast access without loading in RAM
            self.file_mmaps.append(
                np.memmap(
                    Path(self.dirname, f"{split}_{file_idx}.npy"),
                    dtype=self.dtype,
                    shape=(len(reads),) + self.read_shape,
                    mode="r",
                    offset=128,
                )
            )
            # Get corresponding label files
            if use_labels:
                label_file = Path(file.parent, "labels_" + file.stem + ".npy")
                if not label_file.exists():
                    raise FileNotFoundError(f"No label file found for {file}")
                self.label_files.append(label_file)
            # Get corresponding weight files
            if use_sample_weights:
                weight_file = Path(file.parent, "weights_" + file.stem + ".npy")
                if not weight_file.exists():
                    raise FileNotFoundError(f"No weight file found for {file}")
                self.weight_files.append(weight_file)
            # Get next file
            file_idx += 1
            file = Path(dataset_dir, f"{split}_{file_idx}.{file_suffix}")
        self.n_files = len(self.file_lengths)
        self.n_reads = sum(self.file_lengths)
        # Free space
        del reads
        # Determine shard_size and last_shard_size
        for length in self.file_lengths:
            if self.shard_size is None:
                self.shard_size = length
            elif self.last_shard_size != self.shard_size:
                raise ValueError("Intermediate files do not have same lengths")
            self.last_shard_size = length

    def __len__(self) -> int:
        return self.n_reads

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, float]:
        # Identify file from which to take and compute index from file start
        file_idx, idx_mod = divmod(idx, self.shard_size)
        # Get read using memmap
        read = self.file_mmaps[file_idx][idx_mod]
        if (self.strand == "both" and self.rng.random(1) > 0.5) or self.strand == "rev":
            # Take reverse sequence
            read = utils.RC_idx(read)
        if self.transform:  # Transform input (ex: one-hot encode)
            read = self.transform(read)
        if self.paired:  # Separate into 2 inputs
            read = (read[0], read[1])
        # Get label and transform
        if self.use_labels:
            # TODO
            raise NotImplementedError("use_labels is not implemented yet")
        else:
            label = (idx + 1) % 2
        if self.target_transform:
            label = self.target_transform(label)
        # Get weight
        if self.use_sample_weights:
            # TODO
            raise NotImplementedError("use_sample_weights is not implemented yet")
        else:
            weight = self.class_weights[label]
        return read, label, weight


def collate_classbalance(
    batch: Iterable[Tuple[ArrayLike, ArrayLike, ArrayLike]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq, label, weight = tuple(np.array(x) for x in zip(*batch))
    # sum weights by label
    y = np.squeeze(label)
    tot = np.size(weight, out_type=weight.dtype)
    tot_pos = np.reduce_sum(np.where(y, weight, 0))
    tot_neg = np.reduce_sum(np.where(y, weight, 0))
    new_weight = np.where(y, weight * tot / (2 * tot_pos), weight * tot / (2 * tot_neg))
    return tuple(torch.tensor(x, dtype=float) for x in (seq, label, new_weight))


def tp_fp_tn_fn(output: torch.Tensor, target: torch.Tensor, thres=0.5):
    positives = output > thres
    target_bool = target.type(positives.dtype)
    tp = torch.sum(torch.bitwise_and(positives, target_bool))
    fp = torch.sum(torch.bitwise_and(positives, ~target_bool))
    tn = torch.sum(torch.bitwise_and(~positives, ~target_bool))
    fn = torch.sum(torch.bitwise_and(~positives, target_bool))
    return torch.tensor([tp, fp, tn, fn], dtype=int).to(output.device)


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)


def precision(tp, fp, tn, fn):
    return tp / (tp + fp)


def recall(tp, fp, tn, fn):
    return tp / (tp + fn)


def true_negative_rate(tp, fp, tn, fn):
    return tn / (tn + fp)


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    device: str,
    max_batch_per_epoch: int,
    paired: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    test_loss = 0
    binarized_table = torch.zeros(4).to(device)
    # metrics = [
    #     torcheval.metrics.BinaryAccuracy(device=device),
    #     torcheval.metrics.BinaryAUPRC(device=device),
    #     torcheval.metrics.BinaryAUROC(device=device)
    # ]
    with torch.no_grad():
        for batch, (X, y, w) in enumerate(dataloader):
            if max_batch_per_epoch and batch >= max_batch_per_epoch:
                break
            if paired:
                X1, X2 = X
                X1, X2 = X1.to(device), X2.to(device)
                logits = model(X1, X2)
            else:
                X = X.to(device)
                logits = model(X)
            y = torch.reshape(y, logits.shape).type(logits.dtype).to(device)
            w = torch.reshape(w, logits.shape).type(logits.dtype).to(device)
            # Compute loss
            test_loss += loss_fn(weight=w)(logits, y)
            # save binarized predictions for computing accuracy
            binarized_table += tp_fp_tn_fn(nn.Sigmoid()(logits), y)
    test_loss /= batch
    test_acc = accuracy(*binarized_table)
    print(
        f"Avg test loss: {test_loss.mean().item():>8f}\tAvg test accuracy: {test_acc:>8f}\n"
    )
    return test_loss, test_acc


def main(args: argparse.Namespace) -> None:
    training_data = DatasetFromFiles(
        args.dataset_dir,
        split="train",
        strand=args.strand,
        paired=args.paired,
    )
    valid_data = DatasetFromFiles(
        args.dataset_dir,
        split="valid",
        strand=args.strand,
        paired=args.paired,
    )
    if args.balance:

        def collate_fn(batch):
            return collate_classbalance(batch)
    else:
        collate_fn = None
        # def collate_fn(batch):
        #     return tuple(torch.tensor(np.array(x), dtype=float) for x in zip(*batch))
    train_dataloader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Define model and optimizer
    model = args.architecture().to(args.device)
    optimizer = args.optimizer_ctor(model.parameters(), args.learn_rate)

    # Prepare files for logging and saving
    model_file = Path(args.output_dir, "model_state.pt")
    log_file = Path(args.output_dir, "epoch_logs.csv")
    with open(log_file, "w") as f:
        f.write("epoch\tstep\ttrain_loss\ttrain_accuracy\tvalid_loss\tvalid_accuracy\n")
    # Determine size of training dataset before running evaluation
    size = len(train_dataloader.dataset)
    if args.max_train:
        size = min(size, args.batch_size * args.max_train)
    # Initilialize counters for training steps
    step = 0
    wait = 0
    best_valid_loss = np.inf
    last_val_step = 0
    train_loss = 0
    binarized_table = torch.zeros(4).to(
        args.device
    )  # True and False positives and negatives
    model.train()
    # Start training
    for t in range(args.epochs):
        if args.verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        for batch, (X, y, w) in enumerate(train_dataloader):
            if args.paired:  # Extract X as a pair of tensors
                X1, X2 = X
                X1, X2 = X1.to(args.device), X2.to(args.device)
                logits = model(X1, X2)
            else:
                X = X.to(args.device)
                logits = model(X)
            # Reshape and type as logits
            y = torch.reshape(y, logits.shape).type(logits.dtype).to(args.device)
            w = torch.reshape(w, logits.shape).type(logits.dtype).to(args.device)
            # Compute loss
            loss = args.loss_fn(weight=w)(logits, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            if batch % 50 == 0:  # Show progress
                loss, current = loss.item(), (batch + 1) * len(y)
                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{last_val_step*len(y)+size:>5d}]"
                )
            # Update counters
            binarized_table += tp_fp_tn_fn(nn.Sigmoid()(logits), y)
            step += 1

            # Evaluation after fixed number of steps
            if args.max_train and step % args.max_train == 0:
                train_loss /= step - last_val_step
                train_acc = accuracy(*binarized_table).cpu().numpy()
                if args.verbose:
                    print(
                        f"Avg train loss: {train_loss:>8f}\tAvg train accuracy: {train_acc:>8f}\n"
                    )
                valid_loss, valid_acc = test(
                    valid_dataloader,
                    model,
                    args.loss_fn,
                    args.device,
                    args.max_valid,
                    args.paired,
                )
                valid_loss = valid_loss.mean().item()
                valid_acc = valid_acc.cpu().numpy()
                with open(log_file, "a") as f:
                    f.write(
                        f"{t}\t{step}\t{train_loss:>8f}\t{train_acc:>8f}\t{valid_loss:>8f}\t{valid_acc}\n"
                    )
                # Earlystopping
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), model_file)
                    wait = 0
                else:
                    wait += 1
                    if wait >= args.patience:
                        break
                # Reinitialize for training steps
                last_val_step = step
                train_loss = 0
                binarized_table = torch.zeros(4).to(args.device)
                model.train()
        # Propagate Earlystopping
        if wait >= args.patience:
            break
        # Evaluation after an epoch
        if not args.max_train:
            train_loss /= step - last_val_step
            train_acc = accuracy(*binarized_table).cpu().numpy()
            if args.verbose:
                print(
                    f"Avg train loss: {train_loss:>8f}\tAvg train accuracy: {train_acc:>8f}\n"
                )
            valid_loss, valid_acc = test(
                valid_dataloader,
                model,
                args.loss_fn,
                args.device,
                args.max_valid,
                args.paired,
            )
            valid_loss = valid_loss.mean().item()
            valid_acc = valid_acc.cpu().numpy()
            with open(log_file, "a") as f:
                f.write(
                    f"{t}\t{step}\t{train_loss:>8f}\t{train_acc:>8f}\t{valid_loss:>8f}\t{valid_acc}\n"
                )
            # Earlystopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_file)
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
            # Reinitialize for training steps
            last_val_step = step
            train_loss = 0
            binarized_table = torch.zeros(4).to(args.device)
            model.train()
    if args.verbose:
        print("Done!")
        print(f"Saved PyTorch Model State to {model_file}")


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
    print(f"Using {args.device} device")

    # Build output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    # Store arguments in config file
    with open(Path(args.output_dir, "config.json"), "w") as f:
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
    args.loss_fn = LOSSES[args.loss_fn]
    optimizer_ctors = {"adam": torch.optim.Adam}
    args.optimizer_ctor = optimizer_ctors[args.optimizer_ctor]

    # Start computations, save total time even if there was a failure
    try:
        main(args)
    except:
        with open(Path(args.output_dir, "train_time.txt"), "w") as f:
            f.write("Aborted\n")
            f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
        raise
    with open(Path(args.output_dir, "train_time.txt"), "w") as f:
        f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
