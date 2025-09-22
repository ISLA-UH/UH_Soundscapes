"""
ASTRA Reader using DatasetReader class

The files can be downloaded from https://www.higp.hawaii.edu/archive/isla/UH_Soundscapes/ASTRA/

IDE note: inherited classes aren't properly recognized, so the IDE may not recognize
            some properties or methods.

"""
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import uh_soundscapes.dataset_reader as dsr
import uh_soundscapes.data_processing as dp
from uh_soundscapes.standard_labels import ASTRALabels, StandardLabels


class ASTRADatasetLabels(dsr.DatasetLabels):
    """
    A class containing the column names used in ASTRA.

    Inherits from DatasetLabels and uses ASTRALabels for column names.
    """
    def __init__(self, al: ASTRALabels = ASTRALabels(), stl: StandardLabels = StandardLabels()):
        """
        Initialize the ASTRADatasetLabels with the column names used in ASTRA.

        :param al: ASTRALabels class with all the ASTRA label names.
        """
        super().__init__(al, stl)


class ASTRAReader(dsr.DatasetReader, dsr.PlotBase):
    """
    A class to read and analyze the ASTRA dataset.

    Inherits from DatasetReader and PlotBase and uses ASTRALabels for column names.
    """

    def __init__(self, input_path: str, show_frequency_plots: bool = True, save_path: str = ".", 
                 fig_size: Tuple[int, int] = (10, 7), astra_labels: ASTRALabels = ASTRALabels(), 
                 standard_labels: StandardLabels = StandardLabels()):
        """
        Initialize the ASTRAReader with the path to the dataset.

        :param input_path: path to the dataset file
        :param show_frequency_plots: if True, display frequency plots. Default True.
        :param save_path: path to save the processed data. Default current directory.
        :param fig_size: Tuple of (width, height) for the figure size. Default is (10, 7).
        :param astra_labels: ASTRALabels instance that lists all the ASTRA label names.  Default ASTRALabels().
        :param standard_labels: StandardLabels instance that lists all the standardized label names. 
                                Default StandardLabels().
        """
        # initialize the parent classes
        dsr.DatasetReader.__init__(self, "ASTRA", input_path, 
                                   ASTRADatasetLabels(astra_labels, standard_labels), save_path)
        dsr.PlotBase.__init__(self, fig_size)
        self.show_frequency_plots = show_frequency_plots

    def plot_vlines(self, x_coords: List[float], colors: List[str], line_styles: List[str], labels: List[str],
                    ylim: Tuple[Optional[float], Optional[float]] = (None, None)):
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
        if ylim[0] is None or ylim[1] is None:
            ylim = self.base_ylim
        for i in range(len(x_coords)):
            self.ax.vlines(
                ymin=ylim[0],
                ymax=ylim[1],
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
                    ylim=(min(self.ticks) - 1.1 * self.y_adj_buff / 2,
                          max(self.ticks) + 1.1 * self.y_adj_buff / 2))
        self.ax.set_title(title, fontsize=self.font_size + 2)
        self.ax.set_xlabel(xlabel, fontsize=self.font_size)
        self.ax.yaxis.set_ticks(self.ticks)
        self.ax.yaxis.set_ticklabels(self.tick_labels)
        self.ax.tick_params(axis="y", labelsize="large")
        self.ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
        handles, labels = plt.gca().get_legend_handles_labels()
        labels_dict = dict(zip(labels, handles))
        self.ax.legend(frameon=False, bbox_to_anchor=(.99, .99), loc='upper right', fontsize=self.font_size,
                       handles=labels_dict.values(), labels=labels_dict.keys())
        plt.subplots_adjust()

    def print_metadata(self):
        """
        Print the metadata of the dataset.
        """
        # We can get some details about the dataset and print them out
        unique_id_counts = self.get_unique_event_ids()
        num_ids = len(unique_id_counts[0])
        len_data = len(self.data)
        print(f"This dataset contains {len_data} recording{'s' if len_data != 1 else ''} "
              f"from {num_ids} unique launch event{'s' if num_ids != 1 else ''}.")
        # metadata_df = self.compile_metadata(self.labels.event_labels.launch_id, self.labels.event_labels.event_metadata)
        for launch_id, count in zip(unique_id_counts[0], unique_id_counts[1]):
            launch_df = self.data[self.data[self.labels.event_labels.launch_id] == launch_id]
            rocket_type = launch_df[self.labels.event_labels.rocket_type][launch_df.index[0]]
            launch_date = launch_df[self.labels.event_labels.reported_launch_epoch_s][launch_df.index[0]]
            date_string = (datetime.fromtimestamp(launch_date, tz=timezone.utc)).strftime("%d %b %Y")
            print(f"\t{rocket_type} launch {launch_id} on {date_string}: {count} recording(s)")

    def plot_event(self, launch_id: str = ""):
        """
        Plot all data from a single event

        :param launch_id: launch ID string of the event to plot.  If empty or None, the first event in the 
                        dataset is plotted.  Default empty string.
        """
        event_id = self.labels.get_event_id()
        if not launch_id:
            sorted_df = self.data.sort_values(by=event_id)
            launch_id = self.data[event_id][sorted_df.index[0]]
        launch_df = self.data[self.data[event_id] == launch_id]
        sorted_df = launch_df.sort_values(by=self.labels.event_labels.est_prop_dist_km)
        # We'll be plotting the waveforms from the launch relative to the mission's reported launch time.
        rep_launch_epoch_s = self.data[self.labels.event_labels.reported_launch_epoch_s][sorted_df.index[0]]
        date_string = (datetime.fromtimestamp(rep_launch_epoch_s, tz=timezone.utc)).strftime("%d %B %Y")
        xlabel = f"Time (s) since launch"
        # For the title, we'll include some information on the launch included in the ASTRA dataset
        launch_n_srbs = self.data[self.labels.event_labels.n_srbs][sorted_df.index[0]]
        launch_rocket_type = self.data[self.labels.event_labels.rocket_type][sorted_df.index[0]]
        launch_rocket_model = self.data[self.labels.event_labels.rocket_model_number][sorted_df.index[0]]
        title = (f"Normalized ASTRA audio data from launch {launch_id} on {date_string}"
                 f"\nRocket: {launch_rocket_type}, {launch_rocket_model} configuration ({launch_n_srbs} SRBs)")
        sa_toa_color, pa_toa_color = self.cbf_colors[0], self.cbf_colors[1]
        print(f"Plotting data from {launch_rocket_type} launch {launch_id} on {date_string}")
        for station in sorted_df.index:
            # We'll start by demeaning, normalizing, and removing the NaNs in the audio data from each station
            audio_data = self.data[self.labels.event_labels.audio_data][station]
            audio_data = np.nan_to_num(dp.demean_norm(audio_data))
            # The epoch time of the first sample of each recording is included in ASTRA
            start_time = self.data[self.labels.event_labels.first_sample_epoch_s][station]
            # The sample rate of all the audio data in ASTRA is 800 Hz, but it is also included for convenience
            fs = self.data[self.labels.event_labels.audio_fs][station]
            epoch_time = (np.array(range(len(audio_data))) / fs) + start_time
            relative_time = epoch_time - rep_launch_epoch_s
            # To speed up plot generation, trim the signal to start at the reported launch time
            first_idx = np.argwhere(relative_time >= 0).flatten()[0]
            relative_time = relative_time[first_idx:]
            audio_data = audio_data[first_idx:]
            est_prop_distance_km = self.data[self.labels.event_labels.est_prop_dist_km][station]
            self.plot_single_event(f"{round(est_prop_distance_km, 1)} km", relative_time, audio_data)
            relative_start_toa_estimate = self.data[self.labels.event_labels.s_aligned_toa_est][station] - rep_launch_epoch_s
            relative_peak_toa_estimate = self.data[self.labels.event_labels.p_aligned_toa_est][station] - rep_launch_epoch_s
            v_labels = ["Start-aligned TOA estimate", "Peak-aligned TOA estimate"]
            self.plot_vlines(
                x_coords=[relative_start_toa_estimate, relative_peak_toa_estimate],
                colors=[sa_toa_color, pa_toa_color],
                line_styles=["-", "--"],
                labels=v_labels,
                ylim=(self.y_adj - 1, self.y_adj + 1)
            )
            self.y_adj -= int(self.y_adj_buff)
            if self.show_frequency_plots:
                station_id = f"{self.data[self.labels.event_labels.station_id][station]} ({est_prop_distance_km:.1f} km)"
                self.plot_tfr(f"CWT and waveform from launch {launch_id}", station_id, fs, relative_time, audio_data)
        self.touch_up_plot(xlabel, title)

    def plot_recording(self, launch_id: str = "", station_id: str = ""):
        """
        Plot data from a single station during a single event

        :param launch_id: launch ID string of the recording to plot.  If empty or None, the first event in the 
                            dataset is plotted.  Default empty string.
        :param station_id: station ID string of the recording to plot.  If empty or None, the first station in 
                            the event is plotted.  Default empty string.
        """
        if not launch_id:
            launch_id = self.data[self.labels.get_event_id()][self.data.index[0]]
        launch_df = self.data[self.data[self.labels.get_event_id()] == launch_id]
        sorted_df = launch_df.sort_values(by=self.labels.event_labels.est_prop_dist_km)
        if station_id:
            sorted_df = sorted_df[sorted_df[self.labels.event_labels.station_id] == station_id]
        station = sorted_df.index[0]
        # We'll be plotting the waveforms from the launch relative to the mission's reported launch time.
        rep_launch_epoch_s = self.data[self.labels.event_labels.reported_launch_epoch_s][station]
        date_string = (datetime.fromtimestamp(rep_launch_epoch_s, tz=timezone.utc)).strftime("%d %B %Y")
        xlabel = f"Time (s) since launch"
        # For the title, we'll include some information on the launch included in the ASTRA dataset
        launch_n_srbs = self.data[self.labels.event_labels.n_srbs][station]
        launch_rocket_type = self.data[self.labels.event_labels.rocket_type][station]
        launch_rocket_model = self.data[self.labels.event_labels.rocket_model_number][station]
        title = (f"Normalized ASTRA audio data from launch {launch_id} on {date_string}"
                 f"\nRocket: {launch_rocket_type}, {launch_rocket_model} configuration ({launch_n_srbs} SRBs)")
        sa_toa_color, pa_toa_color = self.cbf_colors[0], self.cbf_colors[1]
        print(f"Plotting data from {launch_rocket_type} launch {launch_id} on {date_string}")
        # We'll start by removing the NaNs, demeaning, and normalizing the audio data
        audio_data = self.data[self.labels.event_labels.audio_data][station]
        audio_data = dp.demean_norm(np.nan_to_num(audio_data))
        # The epoch time of the first sample of each recording is included in ASTRA
        start_time = self.data[self.labels.event_labels.first_sample_epoch_s][station]
        # The sample rate of all the audio data in ASTRA is 800 Hz, but it is also included for convenience
        fs = self.data[self.labels.event_labels.audio_fs][station]
        epoch_time = (np.array(range(len(audio_data))) / fs) + start_time
        relative_time = epoch_time - rep_launch_epoch_s
        # To speed up plot generation, trim the signal to start at the reported launch time
        first_idx = np.argwhere(relative_time >= 0).flatten()[0]
        relative_time = relative_time[first_idx:]
        audio_data = audio_data[first_idx:]
        est_prop_distance_km = self.data[self.labels.event_labels.est_prop_dist_km][station]
        self.plot_single_event(f"{round(est_prop_distance_km, 1)} km", relative_time, audio_data)
        relative_start_toa_estimate = self.data[self.labels.event_labels.s_aligned_toa_est][station] - rep_launch_epoch_s
        relative_peak_toa_estimate = self.data[self.labels.event_labels.p_aligned_toa_est][station] - rep_launch_epoch_s
        v_labels = ["Start-aligned TOA estimate", "Peak-aligned TOA estimate"]
        self.plot_vlines(
            x_coords=[relative_start_toa_estimate, relative_peak_toa_estimate],
            colors=[sa_toa_color, pa_toa_color],
            line_styles=["-", "--"],
            labels=v_labels
        )
        station_id = f"{self.data[self.labels.event_labels.station_id][station]} ({est_prop_distance_km:.1f} km)"
        if self.show_frequency_plots:
            self.plot_tfr(f"CWT and waveform from launch {launch_id}", station_id, fs, relative_time, audio_data)
        self.touch_up_plot(xlabel, title)
