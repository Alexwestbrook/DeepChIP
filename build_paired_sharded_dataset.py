#!/usr/bin/env python

# This code takes an IP file of reads and a Control file of reads and creates
# a dataset separated into train, valid and test.
# The dataset in saved into an npz archive in the specified directory
#
# To execute this code run
# build_sharded_dataset.py -ip <IP file> -c <Control file> -out <directory>
# with other options available
#
# parameters :
# - IP file : npy file containing multiple read sequences from IP
# - Control file : npy file containing multiple read sequences from Control
# - directory : path of the directory to store the dataset files in

import argparse
import datetime
import gzip
import json
import platform
import socket
import subprocess
from itertools import repeat
from pathlib import Path

import numpy as np
import utils
from Bio import SeqIO


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip",
        "--ip_file_pair",
        help="IP paired fastq files",
        type=str,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--ctrl_file_pair",
        help="Control paired fastq files",
        type=str,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "-o", "--output_dir", help="output dataset directory", type=str, required=True
    )
    parser.add_argument(
        "-rl",
        "--read_length",
        help="number of base pairs to encode reads on, optional",
        type=int,
    )
    parser.add_argument(
        "-split",
        "--split_sizes",
        help="numbers of reads to put in the test and valid sets, "
        "set to 0 to ignore a split. If they sum to less than 1, "
        "they are considered as fractions of available reads (default: %(default)s)",
        default=[2**23, 2**23],
        type=float,
        nargs=2,
    )
    parser.add_argument(
        "-shard",
        "--shard_size",
        help="maximum number of reads in a shard (default: %(default)s)",
        default=2**24,
        type=int,
    )
    parser.add_argument(
        "-dN",
        "--discardN",
        help="indicates to only use fully sized reads",
        action="store_true",
    )
    args = parser.parse_args()
    # Check if the input data is valid
    for file in args.ip_file_pair + args.ctrl_file_pair:
        if not Path(file).exists():
            raise ValueError(
                f"file {file} does not exist.\n" "Please enter valid fastq file paths."
            )
    return args


def parse_fastq(filename, discardN=False, read_length=None):
    """Generator for reading fastq and skipping unwanted reads.

    Supports regular and gzipped fastq.

    Parameters
    ----------
    filename : str
        Fastq file containing reads
    discardN : bool, optional
        If True, reads with Ns are skipped, as well as reads shorter
        than `read_length`.
    read_length : int, optional
        Length of reads in bases. If `discardN` is set to True, reads
        shorter than this length will be discarded.

    Yields
    ------
    read : Bio.Seq.SeqRecord
        Successive read records in fastq
    """

    def skip(read):
        """Function defining whether a read should be skipped"""
        return discardN and (
            "N" in read.seq[: len(read) if read_length is None else read_length]
            or (read_length is not None and len(read.seq) < read_length)
        )

    # Use SeqIO to parse fastq, optionally reading from gzip file
    if filename.endswith("gz"):
        with gzip.open(filename, "rt") as f:
            for read in SeqIO.parse(f, format="fastq"):
                if skip(read):
                    continue
                yield read
    else:
        for read in SeqIO.parse(filename, "fastq"):
            if skip(read):
                continue
            yield read


def count_lines(filename):
    if platform.system() == "Linux":
        if filename.endswith(".gz"):
            zcat = subprocess.Popen(["zcat", filename], stdout=subprocess.PIPE)
            wc = subprocess.Popen(
                ["wc", "-l"], stdin=zcat.stdout, stdout=subprocess.PIPE
            )
            zcat.stdout.close()
            return int(wc.communicate()[0].partition(b"\n")[0])
        else:
            return int(
                subprocess.run(
                    ["wc", "-l", filename], capture_output=True
                ).stdout.partition(b" ")[0]
            )
    else:
        # from https://gist.github.com/zed/0ac760859e614cd03652
        if filename.endswith(".gz"):
            f = gzip.open(filename, "rt")
        else:
            f = open(filename)
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.read  # loop optimization
        buf = read_f(buf_size)
        while buf:
            lines += buf.count("\n")
            buf = read_f(buf_size)
        return lines


def parse_paired_fastq(filenames, discardN=False, read_length=None):
    """Generator for reading fastq and skipping unwanted reads.

    Supports regular and gzipped fastq.

    Parameters
    ----------
    filenames : List[str]
        Paired fastq files containing reads
    discardN : bool, optional
        If True, pairs with a read with Ns are skipped,
        as well as those with a read shorter than `read_length`.
    read_length : int, optional
        Length of reads in bases. If `discardN` is set to True,
        pairs with a read shorter than this length will be discarded.

    Yields
    ------
    read1, read2 : Bio.Seq.SeqRecord
        Successive read records in fastq
    """

    def skip(read):
        """Function defining whether a read should be skipped"""
        return discardN and (
            "N" in read.seq[: len(read) if read_length is None else read_length]
            or (read_length is not None and len(read.seq) < read_length)
        )

    # Parse both files simultaneously
    for read1, read2 in zip(*(parse_fastq(filename) for filename in filenames)):
        if read1.id != read2.id:
            raise ValueError("Paired reads do not have same ids")
        if skip(read1) or skip(read2):
            continue
        yield read1, read2


def infer_readlen_and_fullprop(file_pair, discardN=False, n_seq=1000, readlen=None):
    maxlen = 0
    n_max = 0
    if readlen is not None:
        n_rl = 0
    for i, reads in enumerate(zip(*(parse_fastq(file) for file in file_pair))):
        if i >= n_seq:
            break
        maxlen_here = max(len(read.seq) for read in reads)
        # count max length reads without Ns
        ismax_here = all(
            len(read.seq) == maxlen and "N" not in read.seq for read in reads
        )
        if maxlen_here > maxlen:
            maxlen = maxlen_here
            n_max = 1 if ismax_here else 0
        elif ismax_here:
            n_max += 1
        if readlen is not None:
            # Count more than read_length reads without Ns until read_length
            if all(
                len(read.seq) >= readlen and "N" not in read.seq[:readlen]
                for read in reads
            ):
                n_rl += 1
    maxprop = n_max / (4 * i)
    if readlen is not None:
        rlprop = n_rl / (4 * i)
        return maxlen, maxprop, rlprop
    else:
        return maxlen, maxprop


def process_fastq_and_save(
    ip_file_pair,
    ctrl_file_pair,
    output_dir,
    shard_size=2**24,
    split_sizes=[2**23, 2**23],
    read_length=None,
    discardN=False,
    log_file=None,
):
    """
    Read multiple fastq files and convert them into a sharded numpy dataset.

    The reads in the fastq files are tokenized and stored in shards.
    Each shard is a binary archive containing two arrays ids and reads:
    ids is an array of string ids in the fastq file and reads are the
    corresponding tokenized paired reads sequences, with shape
    (Number of reads, 2, read_length).

    Parameters
    ----------
    ip_file_pair : List[str]
        Paired-end fastq files containing ip reads
    ctrl_file_pair : List[str]
        Paired-end fastq files containing control reads
    output_dir : str
        Name of the output dataset directory, must be empty
    shard_size : int, default=2**24
        Number of reads in a shard
    split_sizes : tuple[int], default=[2**23, 2**23]
        Number of test and valid samples in this order, remaining samples are
        train samples. Set value to 0 to ignore a split.
        If split_sizes sum to less than 1, they are assumed to be fractions
        of the available reads.
    read_length : int, default=None
        Number of bases in reads, if None, the read length is inferred from
        the maximum length in the first 100 sequences from each file. All
        reads will be truncated or extended with N values to fit this length.
    discardN : bool, default=False
        If True, reads with N values are discarded, as well as reads shorter
        than `read_length`.
    log_file : str, optional
        File to print messages in
    """

    # helper functions
    def save_shard(shard):
        """Tokenize reads of a shard and save to npz archive"""
        print(f"saving shard {cur_split}_{cur_shard}...")
        if not discardN:  # extend reads before tokenizing
            shard = [read + "N" * (read_length - len(read)) for read in shard]
        reads = np.stack(utils.ordinal_encoder(shard)).reshape(-1, 2, read_length)
        np.savez_compressed(
            Path(output_dir, f"{cur_split}_{cur_shard}"), ids=ids, reads=reads
        )

    def get_split_iterators():
        """
        Return shard sizes for each split, in iterators.

        This is a generator function, generating an iterator per split, then
        an infinite iterator
        """
        for split_size in split_sizes:
            q, mod = divmod(split_size, shard_size)
            shard_sizes = [shard_size] * q
            if mod:
                shard_sizes.append(mod)
            yield iter(shard_sizes)
        yield repeat(shard_size)

    # Build output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Infer read length and proportion of reads with Ns from first 1000 sequences in each file
    if read_length is None:
        msg = "read length is unspecified, inferring read length from files"
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")
        print(msg)
    maxlens = []
    maxprops = []
    if read_length is not None:
        rlprops = []
    for file_pair in (ip_file_pair, ctrl_file_pair):
        if read_length is None:
            maxlen, maxprop = infer_readlen_and_fullprop(file_pair, readlen=read_length)
        else:
            maxlen, maxprop, rlprop = infer_readlen_and_fullprop(
                file_pair, readlen=read_length
            )
            rlprops.append(rlprop)
        maxlens.append(maxlen)
        maxprops.append(maxprop)
        if read_length is not None and maxlen != read_length:
            msg = f"Warning: max length seems to be {maxlen} in {file_pair} but user specified {read_length}"
            msg += f"\nAbout {maxprop*100}% of read pairs are of max length without Ns"
            msg += f"\nAbout {rlprop*100}% of read pairs are of user specified length or higher, without Ns"
        else:
            msg = f"Max length seems to be {maxlen} in {file_pair}"
            msg += f"\nAbout {maxprop*100}% of read pairs are of max length without Ns"
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")
        print(msg)
    if read_length is None:
        read_length = max(maxlens)
        rlprops = maxprops
    msg = f"Using read_length {read_length}"
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"{msg}\n")
    print(msg)

    # Convert split_sizes from fractions to count
    if 0 < sum(split_sizes) < 1:
        if not discardN:
            rlprops = (1, 1)
        # Count reads
        n_reads = (
            min(
                (count_lines(ip_file_pair[0]) // 4) * rlprops[0],
                (count_lines(ctrl_file_pair[0]) // 4) * rlprops[1],
            )
            * 2
        )
        split_sizes = tuple(int(n_reads * s) for s in split_sizes)
        msg += f"Using split sizes of {split_sizes}"
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")
        print(msg)

    # Handle train-valid-test splits
    splits = zip(["test", "valid", "train"], get_split_iterators())
    # Initialize first split and counters
    cur_split, cur_split_shards = next(splits)
    cur_shard_size = next(cur_split_shards)
    cur_shard = 0
    ids, shard = [], []

    # Read files
    for (ip1, ip2), (ctrl1, ctrl2) in zip(
        *(
            parse_paired_fastq(file_pair, discardN=discardN, read_length=read_length)
            for file_pair in (ip_file_pair, ctrl_file_pair)
        )
    ):
        ids.extend([ip1.id, ctrl1.id])
        shard.extend([str(read.seq) for read in (ip1, ip2, ctrl1, ctrl2)])
        # When shard is full, save it
        if len(ids) >= cur_shard_size:
            save_shard(shard)
            # Reinitialize
            ids, shard = [], []
            cur_shard += 1
            while True:
                try:
                    # Update next shard size
                    cur_shard_size = next(cur_split_shards)
                except StopIteration:
                    # Split is done, get to next split,
                    # loop again in case split is empty
                    cur_split, cur_split_shards = next(splits)
                    cur_shard = 0
                else:
                    # cur_shard_size was set successfully
                    break

    # Save last incomplete shard
    if len(ids) != 0:
        save_shard(shard)


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    # Get arguments
    args = parsing()

    # Build output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    # Store arguments in config file
    config_file = utils.safe_filename(Path(args.output_dir, "config.json"))
    args.log_file = str(utils.safe_filename(Path(args.output_dir, "dataset_log.txt")))
    with open(config_file, "w") as f:
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

    # Start computations, save total time even if there was a failure
    try:
        process_fastq_and_save(**vars(args))
    except:
        with open(args.log_file, "a") as f:
            f.write("Aborted\n")
            f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
        raise
    with open(args.log_file, "a") as f:
        f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
