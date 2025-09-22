"""
SHAReD Explosions Reader using DatasetReader class

The files can be downloaded from https://www.higp.hawaii.edu/archive/isla/UH_Soundscapes/SHAReD/

IDE note: inherited classes aren't properly recognized, so the IDE may not recognize
            some properties or methods.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from uh_soundscapes.data_processing import demean_norm
import uh_soundscapes.dataset_reader as dsr
from uh_soundscapes.standard_labels import SHAReDLabels, StandardLabels


class SHAReDDatasetLabels(dsr.DatasetLabels):
    """
    A class containing the column names used in SHAReD.

    Inherits from DatasetLabels and uses SHAReDLabels for column names.
    """
    def __init__(self, shl: SHAReDLabels = SHAReDLabels(), stl: StandardLabels = StandardLabels()):
        """
        Initialize the SHAReDDatasetLabels with the column names used in SHAReD.

        :param shl: SHAReDLabels class with all the SHAReD label names.
        """
        super().__init__(shl, stl)


class SHAReDReader(dsr.DatasetReader, dsr.PlotBase):
    """
    A class to read and analyze the SHAReD Explosions dataset.

    Inherits from DatasetReader and uses SHAReDLabels for column names.
    """
    def __init__(self, input_path: str, save_path: str = ".", subplots_rows: int = 3, subplots_cols: int = 2, 
                 fig_size: Tuple[int, int] = (10, 7), shared_labels: SHAReDLabels = SHAReDLabels(), 
                 standard_labels: StandardLabels = StandardLabels()):
        """
        Initialize the SHAReDReader with the path to the dataset.

        :param input_path: path to the dataset file
        :param save_path: path to save the processed data. Default current directory.
        :param subplots_rows: number of rows in the subplots. Default is 3.
        :param subplots_cols: number of columns in the subplots. Default is 2.
        :param fig_size: size of the figure for plotting. Default is (10, 7).
        :param shared_labels: SHAReDLabels instance that lists all the SHAReD label names.  Default SHAReDLabels().
        :param standard_labels: StandardLabels instance that lists all the standardized label names. 
                                Default StandardLabels().
        """
        dsr.DatasetReader.__init__(self, "SHAReD Explosions", input_path, 
                                   SHAReDDatasetLabels(shared_labels, standard_labels), save_path)
        dsr.PlotBase.__init__(self, fig_size, subplots_rows, subplots_cols)

    def load_data(self):
        """
        Load the SHAReD dataset from the input_path.
        """
        super().load_data()

    def print_metadata(self):
        """
        Show information about the dataset.
        """
        # We can get some details about the dataset and print them out
        num_ids = len(self.get_unique_event_ids()[0])
        len_data = len(self.data)
        print(f"This dataset contains {len_data} recording{'s' if len_data != 1 else ''} "
              f"from {num_ids} unique event{'s' if num_ids != 1 else ''}.")
        print(f"Each of the {num_ids} rows in the pandas DataFrame contains all the data collected by one smartphone")
        print(f"during one event, accompanied by a sample of ambient data from the smartphone's sensors, external location")
        print(f"information, and ground truth data about the smartphone and the explosion. Available fields are listed")
        print(f"in the SHAReDLabels class documentation.\n")

    def print_event_metadata(self):
        """
        Print the data for each event in the dataset.
        """
        for event_id, count in np.unique(self.data[self.labels.event_labels.event_id], return_counts=True):
            event_df = self.data[self.data[self.labels.event_labels.event_id] == event_id]
            eq_yield = event_df[self.labels.event_labels.source_yield_kg][event_df.index[0]]
            print(f"\tEvent {event_id}: {eq_yield} kg TNT eq. yield, {count} recording(s)")

    def print_event_info(self, event_id: int):
        """
        Print information about a specific event.

        :param event_id: The unique ID number of the event to print information for.
        """
        event_name = self.data[self.labels.event_labels.event_name][self.data.index[0]]
        source_yield = self.data[self.labels.event_labels.source_yield_kg][self.data.index[0]]
        smartphone_id = self.data[self.labels.event_labels.smartphone_id][self.data.index[0]]
        dist_m = self.data[self.labels.event_labels.distance_from_explosion_m][self.data.index[0]]
        print(f"\nExample event name: {event_name}, event ID number: {event_id}")
        print(f"{source_yield} kg TNT eq. detonation recorded by station {smartphone_id} at {round(dist_m)}m range.")

    def touch_up_plot(self, xmin: float, xmax: float, s_type: str = "base"):
        """
        Final adjustments to the plot, such as setting labels and limits.

        :param xmin: Minimum x-axis limit.
        :param xmax: Maximum x-axis limit.
        :param s_type: Sensor classification type. Default is "base". Unrecognized options will default to "base".
        """
        if s_type.lower() == "ambient":
            ax_idx = 0
            title_type = "Ambient"
        else:
            ax_idx = 1
            title_type = "Explosion"
        self.ax[2, 1].legend()
        for i in range(3):
            self.ax[i, ax_idx].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            self.ax[i, ax_idx].set_ylabel("Norm", fontsize=self.font_size)
        self.ax[0, ax_idx].set(xlim=(xmin, xmax), ylim=self.base_ylim)
        self.ax[0, ax_idx].set_title(f"{title_type} microphone", fontsize=self.font_size)
        self.ax[1, ax_idx].set_title(f"{title_type} barometer", fontsize=self.font_size)
        self.ax[2, ax_idx].set_title(f"{title_type} accelerometer", fontsize=self.font_size)
        self.ax[2, ax_idx].tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
        self.ax[2, ax_idx].set_xlabel(f"Time (s){' since event' if s_type == 'base' else ''}", fontsize=self.font_size)
        plt.subplots_adjust(hspace=0.3)

    def plot_sensor_data(self, station_idx: int, s_type: str = "base"):
        """
        Plot sensor data for a specific station.  Uses the s_type to determine which sensors to plot.

        :param station_index: Index of the station to plot data for.
        :param s_type: Sensor classification type.  Options are "base" and "ambient".  Default "base".
                        Unrecognized options will default to "base".
        """
        if s_type.lower() != "ambient":
            s_type = "base"
        if s_type == "ambient":
            index = 0
            labels = [self.labels.event_labels.ambient_microphone_time_s, 
                      self.labels.event_labels.ambient_microphone_data,
                      self.labels.event_labels.ambient_barometer_time_s, 
                      self.labels.event_labels.ambient_barometer_data,
                      self.labels.event_labels.ambient_accelerometer_time_s, 
                      self.labels.event_labels.ambient_accelerometer_data_x,
                      self.labels.event_labels.ambient_accelerometer_data_y,
                      self.labels.event_labels.ambient_accelerometer_data_z]
            start_time = self.data[self.labels.event_labels.ambient_microphone_time_s][station_idx][0]
        else:
            index = 1
            labels = [self.labels.event_labels.microphone_time_s, self.labels.event_labels.microphone_data,
                      self.labels.event_labels.barometer_time_s, self.labels.event_labels.barometer_data,
                      self.labels.event_labels.accelerometer_time_s, self.labels.event_labels.accelerometer_data_x,
                      self.labels.event_labels.accelerometer_data_y, self.labels.event_labels.accelerometer_data_z]
            start_time = self.data[self.labels.event_labels.explosion_detonation_time][station_idx]
        self.ax[0, index].plot(self.data[labels[0]][station_idx] - start_time,
                               demean_norm(self.data[labels[1]][station_idx]), lw=1, color="k")
        self.ax[1, index].plot(self.data[labels[2]][station_idx] - start_time,
                               demean_norm(self.data[labels[3]][station_idx]), lw=1, color="k")
        self.ax[2, index].plot(self.data[labels[4]][station_idx] - start_time,
                               demean_norm(self.data[labels[5]][station_idx]),
                               lw=1, label="x-axis", color=dsr.CBF_COLOR_CYCLE[0])
        self.ax[2, index].plot(self.data[labels[4]][station_idx] - start_time,
                               demean_norm(self.data[labels[6]][station_idx]),
                               lw=1, label="y-axis", color=dsr.CBF_COLOR_CYCLE[1])
        self.ax[2, index].plot(self.data[labels[4]][station_idx] - start_time,
                               demean_norm(self.data[labels[7]][station_idx]),
                               lw=1, label="z-axis", color=dsr.CBF_COLOR_CYCLE[2])

    def plot_data(self):
        station_idx = self.data.index[0]
        event_name = self.data[self.labels.event_labels.event_name][station_idx]
        event_id = self.data[self.labels.get_event_id()][station_idx]
        station_id = self.data[self.labels.event_labels.smartphone_id][station_idx]
        print(f"\nEvent name: {event_name}, event ID number: {event_id}, station ID: {station_id}")
        source_yield = self.data[self.labels.event_labels.source_yield_kg][station_idx]
        title_header = f"SHAReD event {event_name}"
        if source_yield is None:
            title_header += " source yield not included"
        else:
            title_header += f" {source_yield} kg TNT eq."
        # We'll plot the data from each sensor for both the "explosion" and "ambient" segments of data.
        detonation_ts = self.data[self.labels.event_labels.explosion_detonation_time][station_idx]
        start_audio_ts = self.data[self.labels.event_labels.microphone_time_s][station_idx][0] - detonation_ts
        end_audio_ts = self.data[self.labels.event_labels.microphone_time_s][station_idx][-1] - detonation_ts
        dist_m = self.data[self.labels.event_labels.distance_from_explosion_m][station_idx]

        title_line2 = (f"\nDistance from source: {int(dist_m)} m, scaled distance: "
                       f"{self.data[self.labels.event_labels.scaled_distance][station_idx]:.2f} m/kg^(1/3)")

        self.fig.suptitle(f"Normalized signals from {title_header}{title_line2}", fontsize=self.font_size + 2)
        self.plot_sensor_data(station_idx, s_type="base")
        self.touch_up_plot(start_audio_ts, end_audio_ts, "base")

        start_amb_ts = self.data[self.labels.event_labels.ambient_microphone_time_s][station_idx][0]
        end_amb_ts = self.data[self.labels.event_labels.ambient_microphone_time_s][station_idx][-1] - start_amb_ts
        self.plot_sensor_data(station_idx, s_type="ambient")
        self.touch_up_plot(0, end_amb_ts, "ambient")
