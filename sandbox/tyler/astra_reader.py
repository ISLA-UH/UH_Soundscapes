"""
ASTRA Reader using DatasetReader class
"""

import dataset_reader as dsr


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


class ASTRAReader(dsr.DatasetReader):
    """
    A class to read and analyze the ASTRA dataset.

    Inherits from DatasetReader and uses ASTRALabels for column names.
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
        super().__init__("ASTRA", input_path, default_filename, ASTRALabels(),
                         show_info, show_waveform_plots, show_frequency_plots, save_data, save_path)

    def load_data(self):
        """
        Load the ASTRA dataset from the input_path.
        """
        super().load_data()
        # We can get some details about the dataset and print them out
        num_ids = len(self.get_unique_event_ids()[0])
        len_data = len(self.data)
        print(f"This dataset contains {len_data} recording{'s' if len_data != 1 else ''} "
              f"from {num_ids} unique event{'s' if num_ids != 1 else ''}.")
