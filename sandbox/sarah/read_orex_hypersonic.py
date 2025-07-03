"""
OSIRIS-REx reentry over Nevada and Utah
Stockwell transform with quantum inferno
"""
import os
import matplotlib.pyplot as plt
import numpy as np

import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_mesh_wf_vert
from quantum_inferno.styx_stx import tfr_stx_fft
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

print(__doc__)

# Path to the curated npz file
# input_file_homepath_jorge = os.path.join( os.environ.get('HOME'), "Apps", "poorman-template-matcher", "data", "raw")
input_file_homepath_magpc: str = "C:\\Users\\Milton.Garces\\Documents\\MAGPY\\poorman-template-matcher\\data\\raw"
input_file_homepath_magmac: str = "/Users/mgarces/Documents/DATA_2025/INFRA_SETS/"

input_file_npz: str = "orex_best_mics_800hz_1024pt.npz"
input_file_fullpath: str = os.path.join(input_file_homepath_magmac, input_file_npz)

def max_norm(data: np.ndarray):
    """
    :param data: data array to normalize
    :return: maximum norm
    """
    return data / np.nanmax(np.abs(data))


def plot_waveforms(labels: np.ndarray, wfs: np.ndarray, epochs_s: np.ndarray):
    """
    Plot waveforms from arrays of labels, waveforms, and epochs.

    :param labels: array of labels
    :param wfs: array of waveforms
    :param epochs_s: array of epochs in seconds
    """
    fig1, ax1 = plt.subplots(figsize=[10, 8])
    for j in range(len(labels)):
        sig_wf_j = wfs[j, :]
        sig_epoch_s_j = epochs_s[j, :]
        ax1.plot(sig_epoch_s_j - sig_epoch_s_j[0],
                 max_norm(sig_wf_j) / 1.4 + j, 'k')

    ax1.set_yticks(np.arange(len(labels)), labels=labels)
    ax1.set_title("OSIRIS-REx UH ISLA RedVox Signals")
    ax1.set_xlabel("Time (s) relative to signal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    EVENT_NAME = 'OREX'
    sig_np: np.ndarray = np.load(input_file_fullpath, allow_pickle=True)

    # Load curated stations sampled at 800 Hz
    frequency_sample_rate_hz = 800
    sig_labels = sig_np['station_labels']
    sig_wfs = sig_np['station_wf']
    sig_epochs_s = sig_np['station_epoch_s']

    # Averaging window sets lowest frequency of analysis (lower passband edge).
    fft_duration_ave_window_points_pow2 = 8*1024
    print('Averaging period = ', fft_duration_ave_window_points_pow2 / frequency_sample_rate_hz, 's')
    frequency_resolution_fft_hz = frequency_sample_rate_hz / fft_duration_ave_window_points_pow2

    # Order sets the atom resolution
    order_number_input: int = 3

    # Option to plot waveforms before computing TFR
    plot_wf: bool = True

    if plot_wf:
        plot_waveforms(sig_labels, sig_wfs, sig_epochs_s)

    # Compute STX
    for i in range(len(sig_labels)):
        sig_wf: np.ndarray = sig_wfs[i, :]
        sig_wf /= np.std(sig_wf)        # Unit variance
        sig_epoch_s: np.ndarray = sig_epochs_s[i, :]
        sig_time_s = sig_epoch_s - sig_epoch_s[0]
        sig_label: str = str(sig_labels[i])

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
        wf_base = ptb.WaveformPlotBase(sig_label, f"STX for {EVENT_NAME}")
        wf_panel = ptb.WaveformPanel(sig_wf, sig_time_s)
        mesh_base = ptb.MeshBase(sig_time_s, frequency_stx_hz, frequency_hz_ymin=fmin_plot, frequency_hz_ymax=fmax_plot)
        mesh_panel = ptb.MeshPanel(mic_stx_bits, colormap_scaling="range", cbar_units="log$_2$(Power)")
        stx = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

        plt.show()
