"""
This file contains utility functions for plotting data.
"""
from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
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
        self.marker_lines_ylim = (self.y_adj - self.y_adj_buff / 2, self.y_adj + self.y_adj_buff / 2)
        self.marker_lines_zorder = 10
        self.waveform_color = "k"
        self.fig, self.ax = plt.subplots(figsize=fig_size)

    def plot_vlines(self, x_coords: List[float], colors: List[str], line_styles: List[str], labels: List[str]):
        """
        Plot vertical lines for the ticks and labels on the local Axes object.
        Pass empty strings for labels if no label is needed.

        :param x_coords: List of x-coordinates for the vertical lines.
        :param colors: List of colors for the vertical lines.
        :param line_styles: List of line styles for the vertical lines.
        :param labels: List of labels for the vertical lines.
        """
        if len(x_coords) != len(colors) or len(x_coords) != len(line_styles) or len(x_coords) != len(labels):
            raise ValueError("x_coords, colors, line_styles, and labels must have the same length.")
        for i in range(len(x_coords)):
            self.ax.vlines(
                ymin=self.marker_lines_ylim[0],
                ymax=self.marker_lines_ylim[1],
                x=x_coords[i],
                color=colors[i],
                zorder=self.marker_lines_zorder,
                label=labels[i],
                ls=line_styles[i],
                lw=2,
            )

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
    
    def touch_up_plot(self, xlabel: str, title: str):
        """
        Final adjustments to the plot, such as setting labels and limits.

        :param xlabel: Label for the x-axis.
        :param title: Title for the plot.
        """
        self.ax.set(xlabel=xlabel, xlim=(0, self.t_max), 
                    ylim=(min(self.ticks) - 1.1 * self.y_adj_buff / 2, max(self.ticks) + 1.1 * self.y_adj_buff / 2))
        self.ax.set_title(title, fontsize=self.font_size + 2)
        self.ax.set_xlabel(xlabel, fontsize=self.font_size)
        self.ax.yaxis.set_ticks(self.ticks)
        self.ax.yaxis.set_ticklabels(self.tick_labels)
        self.ax.tick_params(axis="y", labelsize="large")
        self.ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
        self.ax.legend(frameon=False, bbox_to_anchor=(.99, .99), loc='upper right', fontsize=self.font_size)
        plt.subplots_adjust()
    
    def plot_event(self):
        """
        Do all the plotting stuff here.  Implement for specific plot types.
        """
        pass
        # for x in data_df:
        #     self.plot_single_event()
        #     self.plot_vlines()
        #     if do_tfr_plot:
        #         self.plot_tfr()
        # self.touch_up_plot()
