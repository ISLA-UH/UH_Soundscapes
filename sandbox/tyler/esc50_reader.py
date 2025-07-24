"""
ESC50 Reader using DatasetReader class
"""
from datetime import datetime, timezone, timedelta
import os
from typing import Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from quantum_inferno.cwt_atoms import cwt_chirp_from_sig
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_vert_example

from data_processing import rolling_mean
import dataset_reader as dsr
import plot_utils as plot_utils

TUTORIAL_PICKLE_FILE_NAME_800HZ = "esc50_tutorial_800Hz.pkl"
TUTORIAL_PICKLE_FILE_NAME_16KHZ = "esc50_tutorial_16kHz.pkl"
# CURRENT_DIRECTORY = os.getcwd()
CURRENT_DIRECTORY = "/Users/tyler/IdeaProjects/UH_Soundscapes/sandbox/sarah"
PATH_TO_TUTORIAL_PKL_800HZ = os.path.join(CURRENT_DIRECTORY, TUTORIAL_PICKLE_FILE_NAME_800HZ)
PATH_TO_TUTORIAL_PKL_16KHZ = os.path.join(CURRENT_DIRECTORY, TUTORIAL_PICKLE_FILE_NAME_16KHZ)
PATH_TO_PKL_800HZ = PATH_TO_TUTORIAL_PKL_800HZ
PATH_TO_PKL_16KHZ = PATH_TO_TUTORIAL_PKL_16KHZ
PKL_DIRECTORY = "/DIRECTORY/WITH/PKL/FILE"  # Replace with actual path, also used as output path
PKL_FILE_NAME = "ESC50_CHANGEME.pkl"  # Replace with actual file name
PATH_TO_PKL = os.path.join(PKL_DIRECTORY, PKL_FILE_NAME)


class ESC50Labels(dsr.DatasetLabels):
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
        super().__init__(event_id=clip_id)
        self.audio_data = audio_data
        self.audio_fs = audio_fs
        self.esc50_target = esc50_target
        self.esc50_true_class = esc50_true_class
        self.yamnet_predicted_class = yamnet_predicted_class


class ESC50Plot(plot_utils.PlotBase):
    """
    A class to plot ESC50 data using the BasePlot class.
    """
    def __init__(self, fig_size: Tuple[int, int] = (10, 7)) -> None:
        """
        Initialize the ESC50 plot class with default parameters.

        :param fig_size: Tuple of (width, height) for the figure size.  Default is (10, 7).
        """
        super().__init__(fig_size)


class ESC50Reader(dsr.DatasetReader):
    """
    A class to read and analyze the ESC50 dataset.

    Inherits from DatasetReader and uses ESC50Labels for column names.
    """
    def __init__(self, input_path: str, default_filename: str, show_info: bool = True, 
                 show_waveform_plots: bool = True, show_frequency_plots: bool = True,
                 save_data: bool = True, save_path: str = "."):
        """
        Initialize the SHAReDReader with the path to the dataset.

        :param input_path: path to the dataset file
        :param default_filename: default filename to use if the input file is not found
        :param show_info: if True, display dataset information. Default True.
        :param show_waveform_plots: if True, display waveform plots. Default True.
        :param show_frequency_plots: if True, display frequency plots. Default True.
        :param save_data: if True, save the processed data to a file. Default True.
        :param save_path: path to save the processed data. Default current directory.
        """
        super().__init__("ESC50", input_path, default_filename, ESC50Labels(),
                         show_info, show_waveform_plots, show_frequency_plots, save_data, save_path)
        self.esc50_plot = ESC50Plot()
        self.sample_rate = 0.

    def load_data(self):
        """
        Load the ASTRA dataset from the input_path.
        """
        super().load_data()
        self.sample_rate = int(self.data[self.dataset_labels.audio_fs])

    def print_metadata(self):
        """
        Print metadata about the dataset.
        """
        print(f"\nESC-50 dataset at {self.sample_rate}Hz:\n")
        # get some details about the dataset and print them out
        n_signals = len(self.data)
        n_clips = len(np.unique(self.data[self.dataset_labels.event_id]))
        print(f"\tThis dataset contains {n_signals} 5 s long samples from {n_clips} different Freesound audio clips.\n")
        print(f"\tEach of the {n_signals} rows in the pandas DataFrame contains an audio waveform, the ID number of")
        print(f"\tthe Freesound clip it was taken from, the sampling frequency the audio was downsampled to, the")
        print(f"\tESC-50 class name and associated target class number, and the name of the top class predicted when")
        print("\tYAMNet is run on the sample")
        print(f"{'' if self.sample_rate == 16000 else ' (after upsampling from {self.sample_rate} to 16kHz)'}.\n")
    
    def get_sample_waveform(self, idx: int) -> np.ndarray:
        """
        Get the sample waveform at a given index.

        :param idx: Index of the sample in the dataset.
        :return: The waveform as a NumPy array.
        """
        return self.data[self.dataset_labels.audio_data][self.data.index[idx]]

    def plot_waveforms(self, idx: int) -> Figure:
        """
        Plot the waveforms of the dataset at the given index.

        :param idx: Index of the sample in the dataset.
        :return: A matplotlib Figure object containing the waveform plot.
        """
        sample_idx = self.data.index[idx]
        sample_fs = self.data[self.dataset_labels.audio_fs][sample_idx]
        sample_waveform = self.get_sample_waveform(idx)
        time_array = np.arange(len(self.data[self.dataset_labels.audio_data][sample_idx])) / sample_fs
        # We'll demean and normalize the waveform to the range [-1, 1] for cleaner visualization.
        sample_waveform = sample_waveform - rolling_mean(sample_waveform, window_size=13)
        sample_waveform = sample_waveform / np.nanmax(np.abs(sample_waveform))
        # We'll also extract the true class and the class predicted by YAMNet for this sample to add to the plot title.
        sample_esc50_class = self.data[self.dataset_labels.esc50_true_class][sample_idx]
        sample_yamnet_class = self.data[self.dataset_labels.yamnet_predicted_class][sample_idx]

        # calculate and plot the Welch power spectral density (PSD)
        nperseg = self.sample_rate * 0.48  # 0.48 seconds per segment
        f, Pxx_den = signal.welch(sample_waveform, self.sample_rate, nperseg=nperseg)

        print(f"\tPlotting sample {sample_idx} from the {self.sample_rate} Hz ESC-50 dataset...\n")
        # Figure set-up
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        xlabel = "Time (s)"
        title = f"ESC-50 audio downsampled to {int(self.sample_rate)}Hz\n" \
                f"True class: {sample_esc50_class}\nClass predicted by YAMNet" \
                f"{' after upsampling' if self.sample_rate < 16000.0 else ''}: {sample_yamnet_class}"
        # Plot the waveform
        ax[0].plot(time_array, sample_waveform, lw=1, color="k")
        ax[1].plot(f, Pxx_den, lw=1, color="k")
        # Figure settings
        fontsize = 12
        ax[0].set(xlim=(time_array[0], time_array[-1]), ylim=(-1.1, 1.1))
        ax[0].set_title(title, fontsize=fontsize + 2)
        ax[1].set(xlim=(0, self.sample_rate / 2), ylim=(0, np.max(Pxx_den) * 1.05))
        ax[1].set_xlabel("Frequency (Hz)", fontsize=fontsize)
        ax[1].set_ylabel("Power spectral density (PSD)", fontsize=fontsize)
        ax[0].set_xlabel(xlabel, fontsize=fontsize)
        ax[0].set_ylabel("Normalized waveform", fontsize=fontsize)

        plt.subplots_adjust()
        return fig

if __name__=="__main__":
    import random
    esc800 = ESC50Reader(PATH_TO_PKL_800HZ, TUTORIAL_PICKLE_FILE_NAME_800HZ)
    esc16k = ESC50Reader(PATH_TO_PKL_16KHZ, TUTORIAL_PICKLE_FILE_NAME_16KHZ)
    esc800.load_data()
    esc16k.load_data()
    esc800.print_metadata()
    esc16k.print_metadata()
    idx = random.randint(0, len(esc16k.data) - 1)
    _ = esc800.plot_waveforms(idx)
    _ = esc16k.plot_waveforms(idx)
    plt.show()
