"""
OREX Reader Class

OREX example is simple enough that it doesn't need to inherit from the dataset_reader classes.

The files can be downloaded from https://www.higp.hawaii.edu/archive/isla/UH_Soundscapes/OREX/
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_mesh_wf_vert
from quantum_inferno.styx_stx import tfr_stx_fft
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

from soundscapes.data_processing import max_norm


class OREXReader:
    """
    A class to read and analyze the OREX dataset.
    """
    def __init__(self, input_path: str, input_filename: str) -> None:
        """
        Initialize the OREX reader.

        :param input_path: str, path to the dataset file.
        :param input_filename: str, name of the input file.
        """
        self.dataset_name = "OREX"
        self.input_file = os.path.join(input_path, input_filename)
        try:
            sig_np: np.ndarray = np.load(self.input_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading file at {self.input_file}: {e}.  Exiting program.")
            exit(1)
        self.sig_labels = sig_np['station_labels']
        self.sig_wfs = sig_np['station_wf']
        self.sig_epochs_s = sig_np['station_epoch_s']
        self.fig = None
        self.ax = None

    def plot_waveforms(self):
        """
        Plot waveforms from arrays of labels, waveforms, and epochs.

        :param labels: array of labels
        :param wfs: array of waveforms
        :param epochs_s: array of epochs in seconds
        """
        self.fig, self.ax = plt.subplots(figsize=[10, 8])
        for j in range(len(self.sig_labels)):
            sig_wf_j = self.sig_wfs[j, :]
            sig_epoch_s_j = self.sig_epochs_s[j, :]
            self.ax.plot(sig_epoch_s_j - sig_epoch_s_j[0], max_norm(sig_wf_j) / 1.4 + j, 'k')

        self.ax.set_yticks(np.arange(len(self.sig_labels)), labels=self.sig_labels)
        self.ax.set_title("OSIRIS-REx UH ISLA RedVox Signals")
        self.ax.set_xlabel("Time (s) relative to signal")
        plt.tight_layout()

    def plot_spectrogram(self):
        # Load curated stations sampled at 800 Hz
        frequency_sample_rate_hz = 800

        # Averaging window sets lowest frequency of analysis (lower passband edge).
        fft_duration_ave_window_points_pow2 = 8192
        print(f"Averaging period = {fft_duration_ave_window_points_pow2 / frequency_sample_rate_hz}s")
        frequency_resolution_fft_hz = frequency_sample_rate_hz / fft_duration_ave_window_points_pow2

        # Order sets the atom resolution
        order_number_input: int = 3

        # Compute STX
        for i in range(len(self.sig_labels)):
            sig_wf: np.ndarray = self.sig_wfs[i, :]
            sig_wf /= np.std(sig_wf)        # Unit variance
            sig_epoch_s: np.ndarray = self.sig_epochs_s[i, :]
            sig_time_s = sig_epoch_s - sig_epoch_s[0]
            sig_label: str = str(self.sig_labels[i])

            # Compute Stockwell transform
            [stx_complex, _, frequency_stx_hz, _, _] = tfr_stx_fft(
                sig_wf=sig_wf,
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
            wf_base = ptb.WaveformPlotBase(sig_label, f"STX for {self.dataset_name}")
            wf_panel = ptb.WaveformPanel(sig_wf, sig_time_s)
            mesh_base = ptb.MeshBase(sig_time_s, frequency_stx_hz, frequency_hz_ymin=fmin_plot, frequency_hz_ymax=fmax_plot)
            mesh_panel = ptb.MeshPanel(mic_stx_bits, colormap_scaling="range", cbar_units="log$_2$(Power)")
            stx = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

        plt.show()


if __name__=="__main__":
    orx = OREXReader(input_path=os.getcwd(), input_filename="orex_best_mics_800hz_1024pt.npz")
    orx.plot_waveforms()
    orx.plot_spectrogram()
    plt.show()
