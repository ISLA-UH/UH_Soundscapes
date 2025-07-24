"""
ASTRA Reader using DatasetReader class
"""
from datetime import datetime, timezone
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import dataset_reader as dsr
import plot_utils as plot_utils

TUTORIAL_PICKLE_FILE_NAME = "ASTRA_ART-1_1637620009.pkl"
CURRENT_DIRECTORY = os.getcwd()
PATH_TO_TUTORIAL_PKL = os.path.join(CURRENT_DIRECTORY, TUTORIAL_PICKLE_FILE_NAME)
# PATH_TO_TUTORIAL_PKL = "/Users/tyler/IdeaProjects/UH_Soundscapes/sandbox/sarah/ASTRA_ART-1_1637620009.pkl"
PATH_TO_PKL = PATH_TO_TUTORIAL_PKL


class ASTRALabels(dsr.DatasetLabels):
    """
    A class containing the column names used in ASTRA.
    """
    def __init__(
            self,
            station_id: str = "station_id",
            station_make: str = "station_make",
            station_model: str = "station_model_number",
            audio_data: str = "audio_wf_raw",
            first_sample_epoch_s: str = "first_sample_epoch_s",
            audio_fs: str = "audio_sample_rate_nominal_hz",
            station_lat: str = "station_latitude",
            station_lon: str = "station_longitude",
            launch_id: str = "launch_id",
            launch_pad_lat: str = "launch_pad_latitude",
            launch_pad_lon: str = "launch_pad_longitude",
            reported_launch_epoch_s: str = "reported_launch_epoch_s",
            s_aligned_toa_est: str = "start_aligned_arrival_time_estimate_epoch_s",
            p_aligned_toa_est: str = "peak_aligned_arrival_time_estimate_epoch_s",
            est_prop_dist_km: str = "estimated_propagation_distance_km",
            rocket_type: str = "rocket_type",
            rocket_model_number: str = "rocket_model_number",
            n_srbs: str = "n_solid_rocket_boosters",
    ):
        """
        Defaults should be left in place for most uses.
        :param station_id: column containing the recording smartphones' unique station ID numbers
        :param station_make: column containing the recording smartphones' makes
        :param station_model: column containing the recording smartphones' models
        :param audio_data: column containing the raw, uncalibrated audio data
        :param first_sample_epoch_s: column containing the epoch second of the first sample
        :param audio_fs: column containing the sample rate of the audio data in Hertz
        :param station_lat: column containing the recording smartphones' latitude in degrees
        :param station_lon: column containing the recording smartphones' longitude in degrees
        :param launch_id: column containing the launches' unique ID strings
        :param launch_pad_lat: column containing the launch pad latitudes in degrees
        :param launch_pad_lon: column containing the launch pad longitudes in degrees
        :param reported_launch_epoch_s: column containing the reported launch times in epoch seconds
        :param s_aligned_toa_est: column containing the start-aligned arrival time estimates in epoch seconds
        :param p_aligned_toa_est: column containing the peak-aligned arrival time estimates in epoch seconds
        :param est_prop_dist_km: column containing the estimated propagation distances in kilometers
        :param rocket_type: column containing the type of rockets launched (ex: "SpaceX Falcon 9")
        :param rocket_model_number: column containing the model number of the rockets launched (ex: "F9-B5")
        :param n_srbs: column containing the number of solid rocket boosters used
        """
        super().__init__(launch_id)
        self.station_id = station_id
        self.station_make = station_make
        self.station_model = station_model
        self.audio_data = audio_data
        self.audio_fs = audio_fs
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.launch_pad_lat = launch_pad_lat
        self.launch_pad_lon = launch_pad_lon
        self.reported_launch_epoch_s = reported_launch_epoch_s
        self.first_sample_epoch_s = first_sample_epoch_s
        self.s_aligned_toa_est = s_aligned_toa_est
        self.p_aligned_toa_est = p_aligned_toa_est
        self.est_prop_dist_km = est_prop_dist_km
        self.rocket_type = rocket_type
        self.rocket_model_number = rocket_model_number
        self.n_srbs = n_srbs


class ASTRAPlot(plot_utils.PlotBase):
    """
    A class to plot ASTRA data using the BasePlot class.
    """
    def __init__(self, fig_size: Tuple[int, int] = (10, 7)) -> None:
        """
        Initialize the ASTRA plot class with default parameters.

        :param fig_size: Tuple of (width, height) for the figure size.  Default is (10, 7).
        """
        super().__init__(fig_size)
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


class ASTRAReader(dsr.DatasetReader):
    """
    A class to read and analyze the ASTRA dataset.

    Inherits from DatasetReader and uses ASTRALabels for column names.
    """
    def __init__(self, input_path: str, default_filename: str, show_info: bool = True, 
                 show_waveform_plots: bool = True, show_frequency_plots: bool = True,
                 save_data: bool = True, save_path: str = ".", fig_size: Tuple[int, int] = (10, 7)) -> None:
        """
        Initialize the SHAReDReader with the path to the dataset.

        :param input_path: path to the dataset file
        :param default_filename: default filename to use if the input file is not found
        :param show_info: if True, display dataset information. Default True.
        :param show_waveform_plots: if True, display waveform plots. Default True.
        :param show_frequency_plots: if True, display frequency plots. Default True.
        :param save_data: if True, save the processed data to a file. Default True.
        :param save_path: path to save the processed data. Default current directory.
        :param fig_size: Tuple of (width, height) for the figure size. Default is (10, 7).
        """
        super().__init__("ASTRA", input_path, default_filename, ASTRALabels(),
                         show_info, show_waveform_plots, show_frequency_plots, save_data, save_path)
        self.astra_plot = ASTRAPlot(fig_size)

    def load_data(self):
        """
        Load the ASTRA dataset from the input_path.
        """
        super().load_data()
        
    def print_metadata(self):
        """
        Print the metadata of the dataset.
        """
        # We can get some details about the dataset and print them out
        num_ids = len(self.get_unique_event_ids()[0])
        len_data = len(self.data)
        print(f"This dataset contains {len_data} recording{'s' if len_data != 1 else ''} "
              f"from {num_ids} unique launch event{'s' if num_ids != 1 else ''}.")
        unique_id_counts = self.get_unique_event_ids()
        for launch_id, count in zip(unique_id_counts[0], unique_id_counts[1]):
            launch_df = self.data[self.data[self.dataset_labels.event_id] == launch_id]
            rocket_type = launch_df[self.dataset_labels.rocket_type][launch_df.index[0]]
            launch_date = launch_df[self.dataset_labels.reported_launch_epoch_s][launch_df.index[0]]
            date_string = (datetime.fromtimestamp(launch_date, tz=timezone.utc)).strftime("%d %b %Y")
            print(f"\t{rocket_type} launch {launch_id} on {date_string}: {count} recording(s)")

    def plot_event(self):
        launch_id = self.data[self.dataset_labels.event_id][self.data.index[0]]
        # We'll be plotting the waveforms from the launch relative to the mission's reported launch time.
        rep_launch_epoch_s = self.data[self.dataset_labels.reported_launch_epoch_s][self.data.index[0]]
        date_string = (datetime.fromtimestamp(rep_launch_epoch_s, tz=timezone.utc)).strftime("%d %B %Y")
        xlabel = f"Time (s) since launch"
        # For the title, we'll include some information on the launch included in the ASTRA dataset
        launch_n_srbs = self.data[self.dataset_labels.n_srbs][self.data.index[0]]
        launch_rocket_type = self.data[self.dataset_labels.rocket_type][self.data.index[0]]
        launch_rocket_model = self.data[self.dataset_labels.rocket_model_number][self.data.index[0]]
        title = f"Normalized ASTRA audio data from launch {launch_id} on {date_string}"
        title += f"\nRocket: {launch_rocket_type}, {launch_rocket_model} configuration ({launch_n_srbs} SRBs)"
        sa_toa_color, pa_toa_color = plot_utils.CBF_COLOR_CYCLE[0], plot_utils.CBF_COLOR_CYCLE[1]
        sorted_df = self.data.sort_values(by=self.dataset_labels.est_prop_dist_km)
        for station in sorted_df.index:
            # We'll start by normalizing the audio data from each station
            audio_data = self.data[self.dataset_labels.audio_data][station]
            audio_data = audio_data / np.nanmax(np.abs(audio_data))
            # The epoch time of the first sample of each recording is included in ASTRA
            start_time = self.data[self.dataset_labels.first_sample_epoch_s][station]
            # The sample rate of all the audio data in ASTRA is 800 Hz, but it is also included for convenience
            fs = self.data[self.dataset_labels.audio_fs][station]
            epoch_time = (np.array(range(len(audio_data))) / fs) + start_time
            relative_time = epoch_time - rep_launch_epoch_s
            # To speed up plot generation, trim the signal to start at the reported launch time
            first_idx = np.argwhere(relative_time >= 0).flatten()[0]
            relative_time = relative_time[first_idx:]
            audio_data = audio_data[first_idx:]
            est_prop_distance_km = self.data[self.dataset_labels.est_prop_dist_km][station]
            self.astra_plot.plot_single_event(f"{round(est_prop_distance_km, 1)} km", relative_time, audio_data)
            relative_start_toa_estimate = self.data[self.dataset_labels.s_aligned_toa_est][station] - rep_launch_epoch_s
            relative_peak_toa_estimate = self.data[self.dataset_labels.p_aligned_toa_est][station] - rep_launch_epoch_s
            v_labels = ["Start-aligned TOA estimate", "Peak-aligned TOA estimate"] if station == self.data.index[0] else []
            self.astra_plot.plot_vlines(
                x_coords=[relative_start_toa_estimate, relative_peak_toa_estimate],
                colors=[sa_toa_color, pa_toa_color],
                line_styles=["-", "--"],
                labels=v_labels 
            )
            station_id = f"{self.data[self.dataset_labels.station_id][station]} ({est_prop_distance_km:.1f} km)"
            if self.show_frequency_plots:
                self.astra_plot.plot_tfr(f"CWT and waveform from launch {launch_id}", 
                                         station_id, fs, relative_time, audio_data)
        self.astra_plot.touch_up_plot(xlabel, title)


if __name__=="__main__":
    ar = ASTRAReader(PATH_TO_PKL, TUTORIAL_PICKLE_FILE_NAME)
    ar.load_data()
    ar.print_metadata()
    ar.plot_event()
    plt.show()
    