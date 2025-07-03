"""
Tutorial on how to read and plot downsampled ESC-50 audio waveforms and metadata from curated pickle files.

The files can be downloaded from https://drive.google.com/drive/folders/1QKHUzdT0et6xpKXm8LFOmHJ9_gx0J2ib?usp=sharing,
contact Milton Garcés (milton@isla.hawaii.edu) or Sarah Popenhagen (spopen@hawaii.edu) for access.

ESC-50 is an open-access dataset released by Karol Piczak in 2015. For information on ESC-50, see its:
    Original paper: https://doi.org/10.1145/2733373.2806390
    Webpage on Papers with Code: https://paperswithcode.com/dataset/esc-50
    GitHub repository: https://github.com/karolpiczak/ESC-50

The original 44.1 kHz .wav files and metadata can be downloaded from the GitHub repository. Conversion to numpy arrays
and downsampling was performed using the tensorflow function audio.decode_wav() and the tensorflow-io function
audio.resample().

For examples of how to use ESC-50 data in machine learning applications:
    Takazawa, S.K.; Popenhagen, S.K.; Ocampo Giraldo, L.A.; Hix, J.D.; Thompson, S.J.; Chichester, D.L.; Zeiler, C.P.;
Garcés, M.A. Explosion Detection Using Smartphones: Ensemble Learning with the Smartphone High-Explosive Audio
Recordings Dataset and the ESC-50 Dataset. Sensors 2024, 24, 6688. https://doi.org/10.3390/s24206688
    Popenhagen, S.K.; Takazawa, S.K.; Garcés, M.A. Rocket Launch Detection with Smartphone Audio and Transfer Learning.
Signals 2025, <ADD DOI (in review)>
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal

# TODO: Write Pickle Read Function
# replace paths below with the paths to the downloaded ESC-50 pickle files on your device
PICKLE_FILE_NAME_800Hz = "esc50_df_800Hz.pkl"
PICKLE_FILE_NAME_16kHz = "esc50_df_16kHz.pkl"

DIRECTORY_PATH = "/Users/mgarces/Documents/DATA_2025/INFRA_SETS"
# DIRECTORY_PATH = "/Users/spopen/redvox/data/rockets_data/datasets_pkl"
PATH_TO_PKL_800 = os.path.join(DIRECTORY_PATH, PICKLE_FILE_NAME_800Hz)
PATH_TO_PKL_16k = os.path.join(DIRECTORY_PATH, PICKLE_FILE_NAME_16kHz)
# PATH_TO_PKL = " "

class ESC50Labels:
    """
    A class containing the column names used in the ESC-50 pickle files.
    """
    def __init__(
            self,
            clip_id: str = "clip_id",
            audio_data: str = "waveform",
            audio_fs: str = "fs",
            esc50_target: str = "target",
            esc50_true_class: str = "true_class",
            yamnet_predicted_class: str = "inferred_class",
    ):
        """
        Defaults should be left in place for compatibility with the ESC-50 pickle files.
        :param clip_id: the ID string of the Freesound clip the audio was taken from, e.g. "freesound123456"
        :param audio_data: a numpy array containing the raw audio waveform amplitudes
        :param audio_fs: the sampling frequency of the audio waveform in Hz, e.g. 800 or 16000
        :param esc50_target: the target class number of the ESC-50 class, e.g. 37 for "clock_alarm"
        :param esc50_true_class: the name of the true ESC-50 class, e.g. "clock_alarm"
        :param yamnet_predicted_class: the name of the top class predicted by YAMNet, e.g. "Tools"
        """
        self.clip_id = clip_id
        self.audio_data = audio_data
        self.audio_fs = audio_fs
        self.esc50_target = esc50_target
        self.esc50_true_class = esc50_true_class
        self.yamnet_predicted_class = yamnet_predicted_class


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


def main():
    # load the datasets
    esc50_ds_16k = pd.read_pickle(PATH_TO_PKL_16k)
    esc50_ds_800 = pd.read_pickle(PATH_TO_PKL_800)
    # initiate an instance of ESC50Labels, a class containing the column names in the ESC-50 pickle files
    ds_labels = ESC50Labels()
    # We'll pick a random sample and plot both the 800Hz and 16kHz data to get a feel for the dataset.
    idx = random.randint(0, len(esc50_ds_16k) - 1)
    true_target = esc50_ds_800[ds_labels.esc50_target][esc50_ds_800.index[idx]]
    true_class = esc50_ds_800[ds_labels.esc50_true_class][esc50_ds_800.index[idx]]
    print(f"\nSelected sample (index {idx}) belongs to ESC-50 class {true_target}: {true_class}")
    for esc50_ds in [esc50_ds_16k, esc50_ds_800]:
        fs = int(esc50_ds[ds_labels.audio_fs][0])
        print(f"\nESC-50 dataset at {fs}Hz:\n")
        # get some details about the dataset and print them out
        n_signals = len(esc50_ds)
        n_clips = len(np.unique(esc50_ds[ds_labels.clip_id]))
        print(f"\tThis dataset contains {n_signals} 5 s long samples from {n_clips} different Freesound audio clips.\n")
        print(f"\tEach of the {n_signals} rows in the pandas DataFrame contains an audio waveform, the ID number of")
        print(f"\tthe Freesound clip it was taken from, the sampling frequency the audio was downsampled to, the")
        print(f"\tESC-50 class name and associated target class number, and the name of the top class predicted when")
        if fs == 16000:
            print(f"\tYAMNet is run on the sample.\n")
        else:
            print(f"\tYAMNet is run on the sample (after upsampling from {fs} to 16kHz).\n")
        # To plot ESC-50 waveforms, the time array must be reconstructed from the sample rate and waveform length
        sample_idx = esc50_ds.index[idx]
        print(f"\tPlotting sample {sample_idx} from the {fs} Hz ESC-50 dataset...\n")
        sample_fs = esc50_ds[ds_labels.audio_fs][sample_idx]
        sample_waveform = esc50_ds[ds_labels.audio_data][sample_idx]
        sample_time_array = np.arange(len(sample_waveform)) / sample_fs
        # We'll demean and normalize the waveform to the range [-1, 1] for cleaner visualization.
        sample_waveform = sample_waveform - rolling_mean(sample_waveform, window_size=13)
        sample_waveform = sample_waveform / np.nanmax(np.abs(sample_waveform))
        # We'll also extract the true class and the class predicted by YAMNet for this sample to add to the plot title.
        sample_esc50_class = esc50_ds[ds_labels.esc50_true_class][sample_idx]
        sample_yamnet_class = esc50_ds[ds_labels.yamnet_predicted_class][sample_idx]

        # To better understand the sample, we'll calculate and plot the Welch power spectral density (PSD) of the
        # waveform as well.
        nperseg = sample_fs * 0.48  # 0.48 seconds per segment
        f, Pxx_den = signal.welch(sample_waveform, sample_fs, nperseg=nperseg)

        # Figure set-up
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        xlabel = "Time (s)"
        title = f"ESC-50 audio downsampled to {int(sample_fs)}Hz"
        title += f"\nTrue class: {sample_esc50_class}\nClass predicted by YAMNet"
        if sample_fs < 16000.0:
            title += " after upsampling"
        title += f": {sample_yamnet_class}"
        # Plot the waveform
        ax[0].plot(sample_time_array, sample_waveform, lw=1, color="k")
        ax[1].plot(f, Pxx_den, lw=1, color="k")
        # Figure settings
        fontsize = 12
        ax[0].set(xlim=(sample_time_array[0], sample_time_array[-1]), ylim=(-1.1, 1.1))
        ax[0].set_title(title, fontsize=fontsize + 2)
        ax[1].set(xlim=(0, sample_fs / 2), ylim=(0, np.max(Pxx_den) * 1.05))
        ax[1].set_xlabel("Frequency (Hz)", fontsize=fontsize)
        ax[1].set_ylabel("Power spectral density (PSD)", fontsize=fontsize)
        ax[0].set_xlabel(xlabel, fontsize=fontsize)
        ax[0].set_ylabel("Normalized waveform", fontsize=fontsize)

        plt.subplots_adjust()
        print("\tDone.")
    # Show the figures
    plt.show()
    print("\nThis concludes the tutorial. See the comments at the top of the file for more information on ESC-50.")


if __name__ == "__main__":
    main()
