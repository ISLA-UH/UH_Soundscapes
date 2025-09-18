"""
This module contains utility functions for processing data.
"""
import numpy as np


def max_norm(data: np.ndarray) -> np.ndarray:
    """
    :param data: data array to normalize
    :return: maximum norm
    """
    return data / np.nanmax(np.abs(data))


def demean_norm(signal: np.ndarray) -> np.ndarray:
    """
    :param signal: input signal
    :return: demeaned and normalized signal
    """
    signal = signal - np.nanmean(signal)
    return signal / np.nanmax(np.abs(signal))


def rolling_mean(signal: np.ndarray, window_size: int = 13) -> np.ndarray:
    """
    Calculate the rolling mean of a signal using a specified window size.

    :param signal: The input signal as a NumPy array.
    :param window_size: The size of the rolling window.
    :return: A NumPy array containing the rolling mean of the input signal.
    """
    idx_i = 0
    roll_mean = []
    while idx_i < len(signal):
        if idx_i < window_size / 2:
            sig_slice = signal[:window_size]
            if len(sig_slice) != window_size:
                sig_slice = signal[:window_size + 1]
        elif idx_i > len(signal) - window_size / 2:
            sig_slice = signal[-window_size:]
            if len(sig_slice) != window_size:
                sig_slice = signal[-window_size - 1:]
        else:
            slice_start = int(idx_i - int(window_size / 2))
            sig_slice = signal[slice_start: slice_start + window_size]
            if len(sig_slice) != window_size:
                sig_slice = signal[slice_start: slice_start + window_size + 1]
        if len(sig_slice) != window_size:
            raise ValueError(f"Signal slice length {len(sig_slice)} does not match window size {window_size}.")
        roll_mean.append(np.nanmean(sig_slice))
        idx_i += 1
    if len(roll_mean) != len(signal):
        raise ValueError(f"Rolling mean length {len(roll_mean)} does not match signal length {len(signal)}.")
    return np.array(roll_mean)
