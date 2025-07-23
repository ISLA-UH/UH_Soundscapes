"""
SHAReD Explosions Reader using DatasetReader class
"""
from typing import Tuple

import dataset_reader as dsr
import plot_utils as plot_utils


class SHAReDLabels(dsr.DatasetLabels):
    """
    A class containing the column names used in the SHAReD dataset.
    """
    def __init__(self):
        super().__init__("training_validation_test")
        self.event_name: str = "event_name"
        self.source_yield_kg: str = "source_yield_kg"
        self.smartphone_id: str = "smartphone_id"
        self.microphone_time_s: str = "microphone_time_s"
        self.microphone_data: str = "microphone_data"
        self.microphone_sample_rate_hz: str = "microphone_sample_rate_hz"
        self.barometer_time_s: str = "barometer_time_s"
        self.barometer_data: str = "barometer_data"
        self.barometer_sample_rate_hz: str = "barometer_sample_rate_hz"
        self.accelerometer_time_s: str = "accelerometer_time_s"
        self.accelerometer_data_x: str = "accelerometer_data_x"
        self.accelerometer_data_y: str = "accelerometer_data_y"
        self.accelerometer_data_z: str = "accelerometer_data_z"
        self.accelerometer_sample_rate_hz: str = "accelerometer_sample_rate_hz"
        self.ambient_microphone_time_s: str = "ambient_microphone_time_s"
        self.ambient_microphone_data: str = "ambient_microphone_data"
        self.ambient_barometer_time_s: str = "ambient_barometer_time_s"
        self.ambient_barometer_data: str = "ambient_barometer_data"
        self.ambient_accelerometer_time_s: str = "ambient_accelerometer_time_s"
        self.ambient_accelerometer_data_x: str = "ambient_accelerometer_data_x"
        self.ambient_accelerometer_data_y: str = "ambient_accelerometer_data_y"
        self.ambient_accelerometer_data_z: str = "ambient_accelerometer_data_z"
        self.internal_location_latitude: str = "internal_location_latitude"
        self.internal_location_longitude: str = "internal_location_longitude"
        self.external_location_latitude: str = "external_location_latitude"
        self.external_location_longitude: str = "external_location_longitude"
        self.source_latitude: str = "source_latitude"
        self.source_longitude: str = "source_longitude"
        self.distance_from_explosion_m: str = "distance_from_explosion_m"
        self.scaled_distance: str = "scaled_distance"
        self.explosion_detonation_time: str = "explosion_detonation_time"
        self.internal_clock_offset_s: str = "internal_clock_offset_s"
        self.smartphone_model: str = "smartphone_model"
        self.effective_yield_category: str = "effective_yield_category"


class SHAReDPlot(plot_utils.PlotBase):
    def __init__(self, fig_size: Tuple[int, int] = (10, 7)) -> None:
        """
        Initialize the ASTRA plot class with default parameters.

        :param fig_size: Tuple of (width, height) for the figure size.  Default is (10, 7).
        """
        super().__init__(fig_size)


class SHAReDReader(dsr.DatasetReader):
    """
    A class to read and analyze the SHAReD Explosions dataset.

    Inherits from DatasetReader and uses SHAReDLabels for column names.
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
        super().__init__("SHAReD Explosions", input_path, default_filename, SHAReDLabels(),
                         show_info, show_waveform_plots, show_frequency_plots, save_data, save_path)

    def load_data(self):
        """
        Load the SHAReD dataset from the input_path.
        """
        super().load_data()
        # We can get some details about the dataset and print them out
        num_ids = len(self.get_unique_event_ids()[0])
        len_data = len(self.data)
        print(f"This dataset contains {len_data} recording{'s' if len_data != 1 else ''} "
              f"from {num_ids} unique event{'s' if num_ids != 1 else ''}.")
        print(f"Each of the {num_ids} rows in the pandas DataFrame contains all the data collected by one smartphone")
        print(f"during one event, accompanied by a sample of ambient data from the smartphone's sensors, external location")
        print(f"information, and ground truth data about the smartphone and the explosion. Available fields are listed")
        print(f"in the SHAReDLabels class documentation.\n")
