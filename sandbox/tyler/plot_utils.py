"""
This file contains utility functions for plotting data.
"""
from typing import Tuple

from matplotlib.figure import Figure
import numpy as np
from quantum_inferno.cwt_atoms import cwt_chirp_from_sig
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_vert_example


CBF_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

class PlotBase:
    """
    Base class for plotting utilities.
    """
    def __init__(self, fig_size: Tuple[int, int]) -> None:
        """
        Initialize the base plot class with default parameters.

        :param fig_size: Tuple of (width, height) for the figure size.
        """
        self.y_adj = 0
        self.y_adj_buff = 2.2
        self.t_max = 0
        self.ticks =[]
        self.tick_labels = []
        self.font_size = 12
        self.base_ylim = (self.y_adj - self.y_adj_buff / 2, self.y_adj + self.y_adj_buff / 2)
        self.marker_lines_zorder = 10
        self.waveform_color = "k"
    
    def plot_tfr(self, tfr_title: str, station_id: str, fs: float, timestamps: np.ndarray, data: np.ndarray) -> Figure:
        """
        Plot the time-frequency representation (TFR) of the given data.
        :param tfr_title: Title for the TFR plot.
        :param station_id: Identifier for the station.
        :param fs: Sampling frequency of the data.
        :param timestamps: Timestamps corresponding to the data.
        :param data: Audio data to be transformed.
        :return: Matplotlib Figure object containing the TFR plot.
        """
        _, cwt_bits, time_s, frequency_cwt_hz = cwt_chirp_from_sig(
            sig_wf=data,
            frequency_sample_rate_hz=fs,
            band_order_nth=3
        )
        return plot_wf_mesh_vert_example(
            station_id=station_id,
            wf_panel_a_sig=data,
            wf_panel_a_time=timestamps,
            mesh_time=time_s,
            mesh_frequency=frequency_cwt_hz,
            mesh_panel_b_tfr=cwt_bits,
            figure_title=tfr_title,
        )