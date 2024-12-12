#!/usr/bin/env python
import collections
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import Bio
import numpy as np
import scipy
from Bio import Seq
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from sklearn.preprocessing import OrdinalEncoder


def ordinal_encoder(
    inp: Union[Dict[str, str], List[str], str, Bio.Seq.Seq],
) -> Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]:
    # Format input
    if isinstance(inp, list):
        seqs = inp
    elif isinstance(inp, dict):
        seqs = list(inp.values())
    else:
        seqs = inp
    # Encode sequences
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=4, dtype=np.int8
    )
    encoder.fit(np.array(list("ACGT")).reshape((-1, 1)))
    seqs = [
        encoder.transform(np.array(list(seq)).reshape((-1, 1))).ravel() for seq in seqs
    ]
    # Format output
    if isinstance(inp, dict):
        seqs = dict(zip(inp.keys(), seqs))
    elif not isinstance(inp, list):
        seqs = seqs[0]
    return seqs


def make_iterator(obj: Any, length: int = 1) -> Iterator:
    if isinstance(obj, collections.abc.Iterable):
        return iter(obj)
    else:
        return (obj for _ in range(length))


def idx_to_onehot(idx: np.ndarray[int], dtype: type = np.float32) -> np.ndarray:
    return np.eye(5, dtype=dtype)[:, :-1][idx]


def RC_idx(idx: np.ndarray[int]) -> np.ndarray[int]:
    # Copy data array before modification
    idx = idx.copy()
    idx[idx == 0] = -1
    idx[idx == 3] = 0
    idx[idx == -1] = 3
    idx[idx == 1] = -1
    idx[idx == 2] = 1
    idx[idx == -1] = 2
    return np.flip(idx, axis=-1)


def slicer_on_axis(
    arr: np.ndarray,
    slc: Union[slice, Iterable[slice]],
    axis: Union[None, int, Iterable[int]] = None,
) -> Tuple[slice]:
    """Build slice of array along specified axis.

    This function can be used to build slices from unknown axis parameter.

    Parameters
    ----------
    arr: ndarray
        Input array
    slc: slice or iterable of slices
        Slices of the array to take.
    axis: None or int or iterable of ints, optional
        Axis along which to perform slicing. The default (None) is to take slices along first len(`slc`) dimensions.

    Returns
    -------
    tuple of slices
        Full tuple of slices to use to slice array

    Examples
    --------
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> arr
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> arr[slicer_on_axis(arr, slice(1, 3), axis=-1)]  # simple slice on last axis
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]],

           [[13, 14],
            [17, 18],
            [21, 22]]])
    >>> for axis, res in enumerate([arr[1:], arr[:, 1:], arr[:, :, 1:]]):  # unknown axis parameter
    >>>     print(np.all(arr[slicer_on_axis(arr, slice(1, None), axis=axis)] == res))
    True
    True
    True
    >>> arr[slicer_on_axis(arr, slice(None, -1))]  # no axis parameter
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)], axis=[0, 2])]  # multiple slices and axis
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)])]  # multiple slices without axis parameter
    array([[[ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, slice(1, None), axis=[1, 2])]  # single slice on multiple axis
    array([[[ 5,  6,  7],
            [ 9, 10, 11]],

           [[17, 18, 19],
            [21, 22, 23]]])
    """
    full_slice = [slice(None)] * arr.ndim
    if isinstance(slc, slice):
        if axis is None:
            axis = 0
        if isinstance(axis, int):
            full_slice[axis] = slc
        else:
            for ax in axis:
                full_slice[ax] = slc
    else:
        if axis is None:
            axis = list(range(len(slc)))
        elif not isinstance(axis, Iterable):
            raise ValueError("if slc is an iterable, axis must be an iterable or None")
        elif len(axis) != len(slc):
            raise ValueError("axis and slc must have same length")
        for s, ax in zip(slc, axis):
            if full_slice[ax] != slice(None):
                raise ValueError("Can't set slice on same axis twice")
            full_slice[ax] = s
    return tuple(full_slice)


def moving_sum(arr: np.ndarray, n: int, axis: Union[None, int] = None) -> np.ndarray:
    """Compute moving sum of array

    Parameters
    ----------
    arr: ndarray
        Input array
    n: int
        Length of window to compute sum on, must be greater than 0
    axis: None or int, optional
        Axis along which the moving sum is computed, the default (None) is to compute the moving sum over the flattened array.

    Returns
    -------
    ndarray
        Array of moving sum, with size along `axis` reduced by `n`-1.

    Examples
    --------
    >>> moving_sum(np.arange(10), n=2)
    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17])
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> moving_sum(arr, n=2, axis=-1)
    array([[[ 1,  3,  5],
            [ 9, 11, 13],
            [17, 19, 21]],

           [[25, 27, 29],
            [33, 35, 37],
            [41, 43, 45]]])
    """
    if n <= 0:
        raise ValueError(f"n must be greater than 0, but is equal to {n}")
    elif axis is None and n > arr.size:
        raise ValueError(
            f"Can't compute moving_sum of {n} values on flattened array with length {arr.size}"
        )
    elif axis is not None and n > arr.shape[axis]:
        raise ValueError(
            f"Can't compute moving_sum of {n} values on axis {axis} with length {arr.shape[axis]}"
        )
    res = np.cumsum(arr, axis=axis)
    res[slicer_on_axis(res, slice(n, None), axis=axis)] = (
        res[slicer_on_axis(res, slice(n, None), axis=axis)]
        - res[slicer_on_axis(res, slice(None, -n), axis=axis)]
    )
    return res[slicer_on_axis(res, slice(n - 1, None), axis=axis)]


def apply_on_index(
    func: Callable,
    *args: Tuple[np.ndarray, np.ndarray],
    length: int = None,
    neutral: int = 0,
) -> np.ndarray:
    """Applies a function on arrays specified as indices and values.

    Parameters
    ----------
    func : callable
        Function to compute on arrays, must take the number of values as first
        argument, then as many arrays as provided in args.
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, optional
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.
    neutral : int, optional
        Neutral element for the desired operation. This function requires that
        func has a neutral element.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    idxes, vals = zip(*args)
    if length is None:
        length = np.max([np.max(idx) for idx in idxes]) + 1
    n_per_pos = np.zeros((1, length))
    newvals = []
    for idx, val in args:
        val2D = val.reshape(-1, val.shape[-1])
        newval = np.full((len(val2D), length), neutral, dtype=val.dtype)
        newval[:, idx] = val2D
        n_per_pos[0, idx] += 1
        newvals.append(newval)
    res = func(n_per_pos, *newvals)
    res = np.where(n_per_pos == 0, np.nan, res)
    return res.reshape(vals[0].shape[:-1] + (-1,))


def mean_on_index(*args: Tuple[np.ndarray, np.ndarray], length: int = None):
    """Computes the mean of arrays specified as indices and values.

    Parameters
    ----------
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, optional
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    return apply_on_index(
        lambda n, *args: np.divide(sum(args), n, where=(n != 0)), *args, length=length
    )


def safe_filename(file: Union[Path, str]) -> Path:
    """Make sure file can be build without overriding an other.

    If file already exists, returns a new filename with a number in between
    parenthesis. If the parent of the file doesn't exist, it is created.

    Raises
    ------
    FileExistsError
        If one of the parents of the file to create is an existing file
    """
    file = Path(file)
    # Build parent directories if needed
    if not file.parent.is_dir():
        print("Building parent directories")
        file.parent.mkdir(parents=True)
    # Change filename if it already exists
    if file.exists():
        original_file = file
        file_dups = 0
        while file.exists():
            file_dups += 1
            file = Path(
                file.parent, original_file.stem + f"({file_dups})" + file.suffix
            )
            # python3.9: file.with_stem(original_file.stem + f'({file_dups})')
        print(f"{original_file} exists, changing filename to {file}")
    return file


def sliding_GC(
    seqs: np.ndarray, n: int, axis: int = -1, order: str = "ACGT", form: str = "token"
) -> np.ndarray:
    """Compute sliding GC content on encoded DNA sequences

    Sequences can be either tokenized or one-hot encoded.
    GC content will be computed by considering only valid tokens or one-hots.
    Valid tokens are between in the range [0, 4[, and valid one_hots have exactly one value equal to 1. However we check only if they sum to one.

    Parameters
    ----------
    seqs: ndarray
        Input sequences. Sequences are assumed to be read along last axis, otherwise change `axis` parameter.
    n: int
        Length of window to compute GC content on, must be greater than 0.
    axis: int, optional
        Axis along which to compute GC content. The default (-1) assumes sequences are read along last axis.
        Currently, form 'one_hot' doesn't support the axis parameter, it assumes one-hot values on last axis, and sequence on second to last axis.
    order: str, optional
        Order of encoding, must contain each letter of ACGT exactly once.
        If `form` is 'token', then value i corresponds to base at index i in order.
        If `form` is 'one_hot', then vector of zeros with a 1 at position i corresponds to base at index i in order.
    form: {'token', 'one_hot'}, optional
        Form of input array. 'token' for indexes of bases and 'one_hot' for one-hot encoded sequences, with an extra dimension.

    Returns
    -------
    ndarray
        Array of sliding GC content, with size along `axis` reduced by `n`-1, and optional one-hot dimension removed.
    """
    if form == "token":
        valid_mask = (seqs >= 0) & (seqs < 4)
        GC_mask = (seqs == order.find("C")) | (seqs == order.find("G"))
        if n > seqs.shape[axis]:
            n = seqs.shape[axis]
        return moving_sum(GC_mask, n, axis=axis) / moving_sum(valid_mask, n, axis=axis)
    elif form == "one_hot":
        valid_mask = seqs.sum(axis=-1) != 0
        GC_mask = seqs[:, [order.find("C"), order.find("G")]].sum(axis=-1)
        if n > seqs.shape[-1]:
            n = seqs.shape[-1]
        return moving_sum(GC_mask, n=n, axis=-1) / moving_sum(valid_mask, n=n, axis=-1)
    else:
        raise ValueError(f"form must be 'token' or 'one_hot', not {form}")


def smooth(values, window_size, mode="linear", sigma=1, padding="same"):
    if mode == "linear":
        box = np.ones(window_size) / window_size
    elif mode == "gaussian":
        box = gaussian(window_size, sigma)
        box /= np.sum(box)
    elif mode == "triangle":
        box = np.concatenate(
            (
                np.arange((window_size + 1) // 2),
                np.arange(window_size // 2 - 1, -1, -1),
            ),
            dtype=float,
        )
        box /= np.sum(box)
    else:
        raise NameError("Invalid mode")
    return convolve(values, box, mode=padding)


def masked_convolve(in1, in2, correct_missing=True, norm=True, valid_ratio=1./3., *args, **kwargs):
    """A workaround for np.ma.MaskedArray in scipy.signal.convolve. 
    It converts the masked values to complex values=1j. The complex space allows to set a limit
    for the imaginary convolution. The function use a ratio `valid_ratio` of np.sum(in2) to 
    set a lower limit on the imaginary part to mask the values.
    I.e. in1=[[1.,1.,--,--]] in2=[[1.,1.]] -> imaginary_part/sum(in2): [[1., 1., .5, 0.]]
    -> valid_ratio=.5 -> out:[[1., 1., .5, --]].

    Parameters
    ---------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    correct_missing : bool, optional
        correct the value of the convolution as a sum over valid data only, 
        as masked values account 0 in the real space of the convolution.
    norm : bool, optional
        if the output should be normalized to np.sum(in2).
    valid_ratio: float, optional
        the upper limit of the imaginary convolution to mask values. Defined by the ratio of np.sum(in2).
    *args, **kwargs: optional
        passed to scipy.signal.convolve(..., *args, **kwargs)
    
    Reference:
    https://stackoverflow.com/a/72855832
    """
    if not isinstance(in1, np.ma.MaskedArray):
        in1 = np.ma.array(in1)
    
    # np.complex128 -> stores real as np.float64
    con = scipy.signal.convolve(in1.astype(np.complex128).filled(fill_value=1j), 
                                in2.astype(np.complex128), 
                                *args, **kwargs)
    
    # split complex128 to two float64s
    con_imag = con.imag
    con = con.real
    mask = np.abs(con_imag/np.sum(in2)) > valid_ratio
    
    # con_east.real / (1. - con_east.imag): correction, to get the mean over all valid values
    # con_east.imag > percent: how many percent of the single convolution value have to be from valid values
    if correct_missing:
        correction = np.sum(in2) - con_imag
        con[correction!=0] *= np.sum(in2)/correction[correction!=0]
        
    if norm:
        con /= np.sum(in2)
        
    return np.ma.array(con, mask=mask)


def nan_smooth(values, window_size, mode="linear", sigma=1, padding="same", pad_masked=False):
    if mode == "linear":
        box = np.ones(window_size) / window_size
    elif mode == "gaussian":
        box = gaussian(window_size, sigma)
        box /= np.sum(box)
    elif mode == "triangle":
        box = np.concatenate(
            (
                np.arange((window_size + 1) // 2),
                np.arange(window_size // 2 - 1, -1, -1),
            ),
            dtype=float,
        )
        box /= np.sum(box)
    else:
        raise NameError("Invalid mode")
    values = np.ma.array(values, mask=~np.isfinite(values))
    if pad_masked:
        if padding == 'same':
            left_size = window_size // 2
            right_size = (window_size-1) // 2
        elif padding == 'full':
            left_size = window_size - 1
            right_size = window_size - 1
        values = np.ma.concatenate([
            np.ma.masked_all(left_size, dtype=values.dtype),
            values,
            np.ma.masked_all(right_size, dtype=values.dtype)])
        padding = 'valid'
    return masked_convolve(values, box, valid_ratio=(window_size-1)/window_size, mode=padding).filled(fill_value=np.nan)


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """Function from the numpy library"""
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def strided_window_view(
    x, window_shape, stride, axis=None, *, subok=False, writeable=False
):
    """Variant of `sliding_window_view` which supports stride parameter.

    The axis parameter doesn't work, the stride can be of same shape as
    window_shape, providing different stride in each dimension. If shorter
    than window_shape, stride will be filled with ones. it also doesn't
    support multiple windowing on same axis"""
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    # ADDED THIS ####
    stride = tuple(stride) if np.iterable(stride) else (stride,)
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError("`stride` cannot contain negative values")
    if len(stride) > len(window_shape):
        raise ValueError("`stride` cannot be longer than `window_shape`")
    elif len(stride) < len(window_shape):
        stride += (1,) * (len(window_shape) - len(stride))
    ########################

    # CHANGED THIS LINE ####
    # out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    # TO ###################
    out_strides = tuple(x.strides[ax] * stride[ax] for ax in range(x.ndim)) + tuple(
        x.strides[ax] for ax in axis
    )
    ########################

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        # CHANGED THIS LINE ####
        # x_shape_trimmed[ax] -= dim - 1
        # TO ###################
        x_shape_trimmed[ax] = int(np.ceil((x_shape_trimmed[ax] - dim + 1) / stride[ax]))
        ########################
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def strided_sliding_window_view(
    x, window_shape, stride, sliding_len, axis=None, *, subok=False, writeable=False
):
    """Variant of `strided_window_view` which slides in between strides.

    This will provide blocks of sliding window of `sliding_len` windows,
    with first windows spaced by `stride`
    The axis parameter determines where the stride and slide are performed, it
    can only be a single value."""
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    # ADDED THIS ####
    stride = tuple(stride) if np.iterable(stride) else (stride,)
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError("`stride` cannot contain negative values")
    if len(stride) == 1:
        stride += (1,)
    elif len(stride) > 2:
        raise ValueError("`stride` cannot be of length greater than 2")
    if sliding_len % stride[1] != 0:
        raise ValueError("second `stride` must divide `sliding_len` exactly")
    # CHANGED THIS ####
    # if axis is None:
    #     axis = tuple(range(x.ndim))
    #     if len(window_shape) != len(axis):
    #         raise ValueError(f'Since axis is `None`, must provide '
    #                          f'window_shape for all dimensions of `x`; '
    #                          f'got {len(window_shape)} window_shape '
    #                          f'elements and `x.ndim` is {x.ndim}.')
    # else:
    #     axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
    #     if len(window_shape) != len(axis):
    #         raise ValueError(f'Must provide matching length window_shape '
    #                          f'and axis; got {len(window_shape)} '
    #                          f'window_shape elements and {len(axis)} axes '
    #                          f'elements.')
    # TO ###################
    if axis is None:
        axis = 0
    ########################

    # CHANGED THIS LINE ####
    # out_strides = ((x.strides[0]*stride, )
    #                + tuple(x.strides[1:])
    #                + tuple(x.strides[ax] for ax in axis))
    # TO ###################
    out_strides = (
        x.strides[:axis]
        + (x.strides[axis] * stride[0], x.strides[axis] * stride[1])
        + x.strides[axis:]
    )
    ########################

    # CHANGED THIS ####
    # note: same axis can be windowed repeatedly
    # x_shape_trimmed = list(x.shape)
    # for ax, dim in zip(axis, window_shape):
    #     if x_shape_trimmed[ax] < dim:
    #         raise ValueError(
    #             'window shape cannot be larger than input array shape')
    #     x_shape_trimmed[ax] = int(np.ceil(
    #         (x_shape_trimmed[ax] - dim + 1) / stride))
    # out_shape = tuple(x_shape_trimmed) + window_shape
    # TO ###################
    x_shape_trimmed = [
        (x.shape[axis] - window_shape[axis] - sliding_len + stride[1]) // stride[0] + 1,
        sliding_len // stride[1],
    ]
    out_shape = window_shape[:axis] + tuple(x_shape_trimmed) + window_shape[axis:]
    ########################
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def get_legend(axes):
    handles, labels = [], []
    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        handles += handle
        labels += label
    return handles, labels


def z_score(arr):
    mean, std = np.nanmean(arr), np.nanstd(arr)
    return (arr - mean) / std