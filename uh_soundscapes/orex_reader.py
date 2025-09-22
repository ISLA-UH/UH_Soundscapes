"""
OREX Reader Class

OREX example is simple enough that it doesn't need to inherit from the dataset_reader classes.

The files can be downloaded from https://www.higp.hawaii.edu/archive/isla/UH_Soundscapes/OREX/
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_mesh_wf_vert
from quantum_inferno.styx_stx import tfr_stx_fft
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

import uh_soundscapes.dataset_reader as dsr
from uh_soundscapes.data_processing import max_norm
from uh_soundscapes.standard_labels import OREXLabels, StandardLabels


class OREXDatasetLabels(dsr.DatasetLabels):
    """
    A class containing the column names used in OREX.

    Inherits from DatasetLabels and uses OREXLabels for column names.
    """
    def __init__(self, ol: OREXLabels = OREXLabels(), stl: StandardLabels = StandardLabels()):
        """
        Initialize the OREXDatasetLabels with the column names used in OREX.

        :param ol: OREXLabels class with all the OREX label names.
        """
        super().__init__(ol, stl)


class OREXReader(dsr.DatasetReader, dsr.PlotBase):
    """
    A class to read and analyze the OREX dataset.
    """
    def __init__(self, input_path: str, show_frequency_plots: bool = True, save_path: str = ".",
                 fig_size: Tuple[int, int] = (10, 7), orex_labels: OREXLabels = OREXLabels(), 
                 standard_labels: StandardLabels = StandardLabels()) -> None:
        """
        Initialize the OREX reader.

        :param input_path: str, path to the dataset file.
        :param show_frequency_plots: if True, display frequency plots. Default True.
        :param save_path: path to save the processed data. Default current directory.
        :param fig_size: Tuple of (width, height) for the figure size. Default is (10, 7).
        :param orex_labels: OREXLabels instance that lists all the OREX label names.  Default OREXLabels().
        :param standard_labels: StandardLabels instance that lists all the standardized label names.
                                Default StandardLabels().
        """
        # initialize the parent classes
        dsr.DatasetReader.__init__(self, "OREX", input_path, 
                                   OREXDatasetLabels(orex_labels, standard_labels), save_path)
        dsr.PlotBase.__init__(self, fig_size)
        self.show_frequency_plots = show_frequency_plots

    def plot_single_event(self, tick_label: str, timestamps: np.ndarray, data: np.ndarray):
        """
        plot a single event using the Axes object.

        :param tick_label: Label for the y-tick corresponding to this event.
        :param timestamps: Timestamps corresponding to the data.
        :param data: Data to be plotted.
        """
        self.t_max = max(self.t_max, timestamps.max())  # keep largest timestamp for x-axis limit
        self.ax.plot(timestamps, data + self.y_adj, lw=1, color=self.waveform_color)
        self.ticks.append(self.y_adj)
        self.tick_labels.append(tick_label)

    def touch_up_plot(self, xlabel: str, title: str):
        """
        Final adjustments to the plot, such as setting labels and limits.

        :param xlabel: Label for the x-axis.
        :param title: Title for the plot.
        """
        self.ax.set(xlabel=xlabel, xlim=(0, self.t_max),
                    ylim=(min(self.ticks) - 1.1 * self.y_adj_buff / 2,
                          max(self.ticks) + 1.1 * self.y_adj_buff / 2))
        self.ax.set_title(title, fontsize=self.font_size + 2)
        self.ax.set_xlabel(xlabel, fontsize=self.font_size)
        self.ax.yaxis.set_ticks(self.ticks)
        self.ax.yaxis.set_ticklabels(self.tick_labels)
        self.ax.tick_params(axis="y", labelsize="large")
        self.ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
        plt.subplots_adjust()
        self.fig.subplots_adjust(left=0.15, right=0.95)

    def plot_all(self):
        """
        Plot waveforms from arrays of labels, waveforms, and epochs.
        """
        self.y_adj_buff -= 0.2
        for station in self.data.index:
            sig_wf = self.data[self.labels.event_labels.audio_data][station]
            sig_wf = max_norm(sig_wf)
            sig_epoch_s = self.data[self.labels.event_labels.audio_epoch_s][station]
            sig_epoch_s = sig_epoch_s - sig_epoch_s[0]
            self.plot_single_event(self.data[self.labels.event_labels.station_label][station], sig_epoch_s, sig_wf)
            self.y_adj += int(self.y_adj_buff)
            if self.show_frequency_plots:
                self.plot_spectrogram(
                    timestamps=sig_epoch_s,
                    data=sig_wf,
                    label=self.data[self.labels.event_labels.station_label][station])
        self.touch_up_plot("Time (s) relative to signal", "OSIRIS-REx UH ISLA RedVox Signals")

    def plot_spectrogram(self, timestamps: np.ndarray, data: np.ndarray, label: str):
        # Load curated stations sampled at 800 Hz
        frequency_sample_rate_hz = 800

        # Averaging window sets lowest frequency of analysis (lower passband edge).
        fft_duration_ave_window_points_pow2 = 8192
        frequency_resolution_fft_hz = frequency_sample_rate_hz / fft_duration_ave_window_points_pow2

        # Order sets the atom resolution
        order_number_input: int = 3

        # Compute STX
        data /= np.std(data)        # Unit variance
        [stx_complex, _, frequency_stx_hz, _, _] = tfr_stx_fft(
            sig_wf=data,
            time_sample_interval=1 / frequency_sample_rate_hz,
            frequency_min=frequency_resolution_fft_hz,
            frequency_max=frequency_sample_rate_hz / 2,
            scale_order_input=order_number_input,
            n_fft_in=fft_duration_ave_window_points_pow2,
            is_geometric=True,
            is_inferno=False,
        )

        stx_power = 2 * np.abs(stx_complex) ** 2
        mic_stx_bits = to_log2_with_epsilon(np.sqrt(stx_power))

        # Select plot frequencies
        fmin_plot = 4 * frequency_resolution_fft_hz  # Octaves above the lowest frequency of analysis
        fmax_plot = frequency_sample_rate_hz / 2  # Nyquist

        # Plot the STX
        wf_base = ptb.WaveformPlotBase(label, f"STX for {self.dataset_name}")
        wf_panel = ptb.WaveformPanel(data, timestamps)
        mesh_base = ptb.MeshBase(timestamps, frequency_stx_hz, frequency_hz_ymin=fmin_plot, frequency_hz_ymax=fmax_plot)
        mesh_panel = ptb.MeshPanel(mic_stx_bits, colormap_scaling="range", cbar_units="log$_2$(Power)")
        stx = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)
