"""
ESC50 Reader using DatasetReader class

The files can be downloaded from https://www.higp.hawaii.edu/archive/isla/UH_Soundscapes/ESC50/

For an example on how to display metadata and plots, go to: 
https://github.com/ISLA-UH/UH_Soundscapes/blob/main/notebooks/reader_tutorial.ipynb

IDE note: inherited classes aren't properly recognized, so the IDE may not recognize
            some properties or methods.
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from uh_soundscapes.data_processing import rolling_mean
import uh_soundscapes.dataset_reader as dsr
from uh_soundscapes.standard_labels import ESC50Labels, StandardLabels


class ESC50DatasetLabels(dsr.DatasetLabels):
    """
    A class containing the column names used in ESC-50.

    Inherits from DatasetLabels and uses ESC50Labels for column names.
    """
    def __init__(self, el: ESC50Labels = ESC50Labels(), stl: StandardLabels = StandardLabels()):
        """
        Initialize the ESC50DatasetLabels with the column names used in ESC-50.

        :param el: ESC50Labels class with all the ESC-50 label names.
        """
        super().__init__(el, stl)


class ESC50Reader(dsr.DatasetReader, dsr.PlotBase):
    """
    A class to read and analyze the ESC50 dataset.

    Inherits from DatasetReader and uses ESC50Labels for column names.
    """
    def __init__(self, input_path: str, show_frequency_plots: bool = True, save_path: str = ".", 
                 subplots_rows: int = 2, subplots_cols: int = 1, fig_size: Tuple[int, int] = (10, 7), 
                 esc50_labels: ESC50Labels = ESC50Labels(), standard_labels: StandardLabels = StandardLabels()):
        """
        Initialize the ESC50Reader with the path to the dataset.

        :param input_path: path to the dataset file
        :param show_frequency_plots: if True, display frequency plots. Default True.
        :param save_path: path to save the processed data. Default current directory.
        :param subplots_rows: Number of rows in the subplot grid. Default is 1.
        :param subplots_cols: Number of columns in the subplot grid. Default is 1
        :param fig_size: Tuple of (width, height) for the figure size. Default is (10, 7).
        """
        dsr.DatasetReader.__init__(self, "ESC50", input_path, 
                                   ESC50DatasetLabels(esc50_labels, standard_labels), save_path)
        dsr.PlotBase.__init__(self, fig_size, subplots_rows, subplots_cols)
        self.show_frequency_plots = show_frequency_plots

    def new_figure(self, fig_size: Tuple[int, int] = None, subplot_rows: int = None, subplot_cols: int = None):
        """
        Create a new figure and Axes object for plotting.
        :param fig_size: Tuple of (width, height) for the figure size. If None, uses the last plot's fig_size.
        :param subplot_rows: Number of rows in the subplot grid. If None, uses the last plot's number of rows.
        :param subplot_cols: Number of columns in the subplot grid. If None, uses the last plot's number of columns.
        """
        if fig_size is None:
            fig_size = self.fig_size
        if subplot_rows is None:
            subplot_rows = self.ax.shape[0]
        if subplot_cols is None:
            subplot_cols = self.ax.shape[1]
        dsr.PlotBase.__init__(self, fig_size, subplot_rows, subplot_cols)

    def load_data(self):
        """
        Load the ESC-50 dataset from the input_path.
        """
        super().load_data()
        self.sample_rate = int(self.data[self.labels.event_labels.audio_fs].iloc[0])

    def print_metadata(self):
        """
        Print metadata about the dataset.
        """
        print(f"\nESC-50 dataset at {self.sample_rate}Hz:\n")
        # get some details about the dataset and print them out
        n_signals = len(self.data)
        clips, counts = np.unique(self.data[self.get_event_id_col()].values, return_counts=True)
        n_clips = len(clips)
        print(f"\tThis dataset contains {n_signals} 5 s long samples from {n_clips} different Freesound audio clips.\n")
        print(f"\tEach of the {n_signals} rows in the pandas DataFrame contains an audio waveform, the ID number of")
        print(f"\tthe Freesound clip it was taken from, the sample rate the audio was downsampled to, the ESC-50")
        print(f"\tclass name and associated target class number, and the name of the top class predicted when")
        last_line = ("\tYAMNet is run on the sample")
        if self.sample_rate != 16000:
            last_line += (f" (after upsampling from {int(self.sample_rate)}Hz to 16kHz)")
        last_line += " and the scores are meaned over the clip."
        print(last_line)
        print(f"\nAll Freesound clips in the dataset, their class labels, and the sample(s) taken from them:")
        for clip_id in clips:
            clip_ds = self.data[self.data[self.get_event_id_col()] == clip_id]
            classes, counts = np.unique(clip_ds[self.labels.event_labels.esc50_true_class], return_counts=True)
            class_summary = f"\t{clip_id}            "[:15] + ": "
            base_len = len(class_summary)
            for class_name, class_count in zip(classes, counts):
                if len(class_summary) > base_len:
                    class_summary += "\n\t                                       "[:base_len + 1]
                class_summary += f"'{class_name}'             "[:18] + f": {class_count} sample(s),"
            print(class_summary[:-1])

    def get_sample_waveform(self, idx: int) -> np.ndarray:
        """
        Get the sample waveform at a given index.

        :param idx: Index of the sample in the dataset.
        :return: The waveform as a NumPy array.
        """
        return self.data[self.labels.event_labels.audio_data][self.data.index[idx]]

    def touch_up_plot(self, xlabel: str, title: str, sample_rate: float, ax1_ymax: float, ax0_xmax: float):
        """
        Final adjustments to the plot, such as setting labels and limits.

        :param xlabel: Label for the x-axis.
        :param title: Title for the plot.
        :param sample_rate: Sampling rate of the audio data.
        :param ax1_ymax: Maximum y-value for the second subplot.
        :param ax0_xmax: Maximum x-value for the first subplot.
        """        
        self.ax[0].set(ylim=self.base_ylim, xlim=(0., ax0_xmax))
        # self.ax[0].set_title(title, fontsize=self.font_size + 2)
        self.ax[1].set(xlim=(0, sample_rate / 2), ylim=(0, ax1_ymax * 1.05))
        self.ax[1].set_xlabel("Frequency (Hz)", fontsize=self.font_size)
        self.ax[1].set_ylabel("Power spectral density (PSD)", fontsize=self.font_size)
        self.ax[0].set_xlabel(xlabel, fontsize=self.font_size)
        self.ax[0].set_ylabel("Normalized waveform", fontsize=self.font_size)
        for ax_i in self.ax:
            ax_i.tick_params(axis='both', which='both', labelsize='large')
        self.fig.align_ylabels(self.ax)
        plt.subplots_adjust(hspace=.3, top=0.875, right=0.95)

    def plot_clip(self, idx: Optional[int] = None):
        """
        Plot the ESC50 audio clip at the given index.

        :param idx: Index position of the sample in the dataset.
        """
        if idx is None:
            idx = 0
        if type(self.ax) == np.ndarray:
            ax_to_check = self.ax.flatten()[0]
        else:
            ax_to_check = self.ax
        if len(ax_to_check.get_children()) != self.n_children_base:
            self.new_figure(self.fig_size, 2, 1)
        sample_idx = self.data.index[idx]
        sample_fs = self.data[self.labels.event_labels.audio_fs][sample_idx]
        sample_waveform = self.get_sample_waveform(idx)
        time_array = np.arange(len(self.data[self.labels.event_labels.audio_data][sample_idx])) / sample_fs
        # We'll demean and normalize the waveform to the range [-1, 1] for cleaner visualization.
        sample_waveform = sample_waveform - rolling_mean(sample_waveform, window_size=13)
        sample_waveform = sample_waveform / np.nanmax(np.abs(sample_waveform))
        # We'll also extract the true class and the class predicted by YAMNet for this sample to add to the plot title.
        sample_esc50_class = self.data[self.labels.event_labels.esc50_true_class][sample_idx]
        sample_yamnet_class = self.data[self.labels.event_labels.yamnet_predicted_class][sample_idx]

        # calculate and plot the Welch power spectral density (PSD)
        nperseg = self.sample_rate * 0.48  # 0.48 seconds per segment
        psd_freq, psd = signal.welch(sample_waveform, self.sample_rate, nperseg=nperseg)

        print(f"Plotting sample {sample_idx} from the {self.sample_rate} Hz ESC-50 dataset.")
        # Figure set-up
        xlabel = "Time (s)"
        title = f"ESC-50 audio downsampled to {int(self.sample_rate)}Hz\n" \
                f"True class: '{sample_esc50_class}'\nClass predicted by YAMNet" \
                f"{' after upsampling' if self.sample_rate < 16000.0 else ''}: '{sample_yamnet_class}'"
        self.fig.suptitle(title, fontsize=self.font_size + 2)
        # Plot the waveforms
        self.ax[0].plot(time_array, sample_waveform, lw=1, color=self.waveform_color)
        self.ax[1].plot(psd_freq, psd, lw=1, color=self.waveform_color)
        self.touch_up_plot(xlabel, title, sample_fs, np.max(psd), np.max(time_array))

        if self.show_frequency_plots:
            print(f"Plotting the PSD of sample {sample_idx} from the {self.sample_rate} Hz ESC-50 dataset.")
            tfr_title = f"CWT and waveform from ESC-50 PKL index {sample_idx}"
            tfr_title += f" (clip ID: {self.data[self.get_event_id_col()][sample_idx]})"
            _ = self.plot_tfr(tfr_title, "", sample_fs, time_array, sample_waveform)
