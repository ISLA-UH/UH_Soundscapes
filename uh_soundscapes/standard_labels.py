"""
This file contains classes defining the standard column names for pandas DataFrames and/or keys for NPZ files used in
various datasets. These classes help ensure consistency when accessing, manipulating, or merging data from different
sources. A standard set of labels is also included, along with methods to convert between different naming conventions
if needed.
"""
from typing import List
import pandas as pd


class StandardLabels():
    """
    A class containing the column names used in standardized datasets for machine learning applications.
    """
    def __init__(
            self,
            station_id: str = "station_id",
            station_network: str = "deployment_network",
            station_latitude: str = "station_latitude",
            station_longitude: str = "station_longitude",
            station_altitude: str = "station_altitude_m",
            audio_waveform: str = "audio_waveform",
            t0_epoch_s: str = "first_audio_sample_epoch_s",
            audio_fs: str = "audio_sample_rate_nominal_hz",
            event_id: str = "event_id",
            data_source: str = "data_source",
            ml_label: str = "machine_learning_label",
            source_latitude: str = "source_latitude",
            source_longitude: str = "source_longitude",
            source_altitude: str = "source_altitude_m",
            source_epoch_s: str = "source_epoch_s",
    ):
        """
        Defaults should be left in place for most uses.

        :param station_id: unique identifying string of the recording station, when applicable
        :param station_network: network smartphone was deployed on, when applicable
        :param station_latitude: latitude of the recording station in degrees, when applicable
        :param station_longitude: longitude of the recording station in degrees, when applicable
        :param station_altitude: altitude of the recording station in meters, when applicable
        :param audio_waveform: audio waveform data, typically a numpy array of raw audio samples
        :param t0_epoch_s: epoch second of the first audio sample, when applicable
        :param audio_fs: sample rate of the audio data in Hertz
        :param event_id: unique identifying string of the event associated with the audio data
        :param data_source: source of the data, e.g., "ASTRA", "SHAReD", "OREX", etc.
        :param ml_label: suggested label for machine learning applications, e.g., "explosion", "rocket", etc.
        :param source_latitude: latitude of the signal source in degrees, when applicable
        :param source_longitude: longitude of the signal source in degrees, when applicable
        :param source_altitude: altitude of the signal source in meters, when applicable
        :param source_epoch_s: epoch seconds of the source event, when applicable
        """
        self.station_id = station_id
        self.station_network = station_network
        self.station_lat = station_latitude
        self.station_lon = station_longitude
        self.station_alt = station_altitude
        self.audio_wf = audio_waveform
        self.t0_epoch_s = t0_epoch_s
        self.audio_fs = audio_fs
        self.event_id = event_id
        self.data_source = data_source
        self.ml_label = ml_label
        self.source_lat = source_latitude
        self.source_lon = source_longitude
        self.source_alt = source_altitude
        self.source_epoch_s = source_epoch_s
        self.standard_labels: List = [
            self.station_id,
            self.station_network,
            self.station_lat,
            self.station_lon,
            self.station_alt,
            self.audio_wf,
            self.t0_epoch_s,
            self.audio_fs,
            self.event_id,
            self.data_source,
            self.ml_label,
            self.source_lat,
            self.source_lon,
            self.source_alt,
            self.source_epoch_s
        ]

    def as_dict(self) -> dict:
        """
        :return: standard labels as a dictionary.
        """
        return {
            "station_id": self.station_id,
            "station_network": self.station_network,
            "station_latitude": self.station_lat,
            "station_longitude": self.station_lon,
            "station_altitude_m": self.station_alt,
            "audio_waveform": self.audio_wf,
            "first_audio_sample_epoch_s": self.t0_epoch_s,
            "audio_sample_rate_nominal_hz": self.audio_fs,
            "event_id": self.event_id,
            "data_source": self.data_source,
            "machine_learning_label": self.ml_label,
            "source_latitude": self.source_lat,
            "source_longitude": self.source_lon,
            "source_altitude_m": self.source_alt,
            "source_epoch_s": self.source_epoch_s
        }
    

class EventLabels():
    """
    A class for event specific labels.
    """
    def __init__(self, event_metadata: List[str] = [], station_metadata: List[str] = [], standardize_dict: dict = {}):
        """
        Initialize the event labels.

        :param event_metadata: list of column names associated with event-specific metadata
        :param station_metadata: list of column names associated with station-specific metadata
        :param standardize_dict: dictionary to map dataset-specific labels to standard labels
        """
        # event specific metadata
        self.event_metadata = event_metadata
        # station specific metadata
        self.station_metadata = station_metadata
        # dictionary to standardize labels
        self.standardize_dict = standardize_dict


class ASTRALabels(EventLabels):
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
            standard_labels: StandardLabels = StandardLabels(),
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
        :param standard_labels: instance of StandardLabels class for mapping to standard names
        """
        self.station_id = station_id
        self.station_make = station_make
        self.station_model = station_model
        self.audio_data = audio_data
        self.audio_fs = audio_fs
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.launch_id = launch_id
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

        # set the EventLabels properties
        # column names associated with event-specific metadata
        self.event_metadata = [
            self.launch_id,
            self.launch_pad_lat,
            self.launch_pad_lon,
            self.reported_launch_epoch_s,
            self.rocket_type,
            self.rocket_model_number,
            self.n_srbs
        ]

        # column names associated with station-specific metadata
        self.station_metadata = [self.station_id, self.station_make, self.station_model]

        # dictionary to map ASTRA labels to standard labels
        self.standardize_dict = {
            self.audio_data: standard_labels.audio_wf,
            self.first_sample_epoch_s: standard_labels.t0_epoch_s,
            self.audio_fs: standard_labels.audio_fs,
            self.station_id: standard_labels.station_id,
            self.station_lat: standard_labels.station_lat,
            self.station_lon: standard_labels.station_lon,
            self.launch_id: standard_labels.event_id,
            self.launch_pad_lat: standard_labels.source_lat,
            self.launch_pad_lon: standard_labels.source_lon,
            self.reported_launch_epoch_s: standard_labels.source_epoch_s}


class SHAReDLabels(EventLabels):
    """
    A class containing the column names used in the SHAReD dataset.
    """
    def __init__(self):
        """
        Defaults should be left in place for most uses.
        
        """
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
        self.event_id_number: str = "training_validation_test"

        # set the EventLabels properties
        # column names associated with event-specific metadata
        self.event_metadata = [
            self.event_name,
            self.event_id_number,
            self.source_yield_kg,
            self.effective_yield_category,
            self.source_longitude,
            self.source_latitude,
            self.explosion_detonation_time
        ]

        # column names associated with station-specific metadata
        self.station_metadata = [self.smartphone_id, self.smartphone_model]

        # dictionary to map SHAReD labels to standard labels
        standard_labels = StandardLabels()
        self.standardize_dict = {
            self.smartphone_id: standard_labels.station_id,
            self.event_name: standard_labels.event_id,
            self.microphone_data: standard_labels.audio_wf,
            self.microphone_sample_rate_hz: standard_labels.audio_fs,
            self.internal_location_latitude: standard_labels.station_lat,
            self.internal_location_longitude: standard_labels.station_lon,
            self.source_latitude: standard_labels.source_lat,
            self.source_longitude: standard_labels.source_lon,
            self.ambient_microphone_data: standard_labels.audio_wf,
            self.explosion_detonation_time: standard_labels.source_epoch_s}


class ESC50Labels(EventLabels):
    """
    A class containing the column names used in the ESC-50 pickle files.
    """
    def __init__(
            self,
            clip_id: str = "clip_id",
            audio_data: str = "waveform",
            audio_fs: str = "fs",
            esc50_target: str = "target",
            esc50_true_class: str = "true_class",
            yamnet_predicted_class: str = "inferred_class",
            standard_labels: StandardLabels = StandardLabels(),
    ):
        """
        Defaults should be left in place for compatibility with the ESC-50 pickle files.

        :param clip_id: the ID string of the Freesound clip the audio was taken from, e.g. "freesound123456"
        :param audio_data: a numpy array containing the raw audio waveform amplitudes
        :param audio_fs: the sampling frequency of the audio waveform in Hz, e.g. 800 or 16000
        :param esc50_target: the target class number of the ESC-50 class, e.g. 37 for "clock_alarm"
        :param esc50_true_class: the name of the true ESC-50 class, e.g. "clock_alarm"
        :param yamnet_predicted_class: the name of the top class predicted by YAMNet, e.g. "Tools"
        :param standard_labels: instance of StandardLabels class for mapping to standard names
        """
        self.clip_id = clip_id
        self.audio_data = audio_data
        self.audio_fs = audio_fs
        self.esc50_target = esc50_target
        self.esc50_true_class = esc50_true_class
        self.yamnet_predicted_class = yamnet_predicted_class

        # set the EventLabels properties
        # column names associated with event-specific metadata
        self.event_metadata = [self.clip_id, self.esc50_true_class, self.yamnet_predicted_class]

        # column names associated with station-specific metadata
        self.station_metadata = []

        # dictionary to map ESC-50 labels to standard labels
        self.standardize_dict = {
            self.clip_id: standard_labels.event_id,
            self.audio_data: standard_labels.audio_wf,
            self.audio_fs: standard_labels.audio_fs,
            self.esc50_true_class: standard_labels.ml_label}


class OREXLabels(EventLabels):
    """
    A class containing the keys used in the OSIRIS-REx NPZ file.
    """
    def __init__(
            self,
            station_id: str = "station_ids",
            station_label: str = "station_labels",
            station_make: str = "station_make",
            station_model: str = "station_model_number",
            station_network: str = "deployment_network",
            station_latitude: str = "best_location_latitude",
            station_longitude: str = "best_location_longitude",
            station_altitude: str = "best_location_altitude",
            audio_data: str = "station_wf",
            audio_epoch_s: str = "station_epoch_s",
            audio_fs: str = "audio_sample_rate_nominal_hz",
            event_id: str = "event_id",
            standard_labels: StandardLabels = StandardLabels(),
    ):
        """
        Defaults should be left in place for most uses.
        
        :param station_id: key associated with the unique ID string of the station used to record the signal
        :param station_label: key associated with the descriptive label string of the station
        :param station_make: key associated with the recording smartphone's make
        :param station_model: key associated with the recording smartphone's model
        :param station_network: key associated with the network on which the smartphone was deployed
        :param station_latitude: key associated with the latitude of the recording station in degrees
        :param station_longitude: key associated with the longitude of the recording station in degrees
        :param station_altitude: key associated with the altitude of the recording station in meters
        :param audio_data: key associated with the audio waveform of the signal
        :param audio_epoch_s: key associated with the time array of the audio waveform in epoch seconds
        :param audio_fs: key associated with the sample rate of the audio data in Hertz
        :param event_id: key associated with the unique ID string of the event associated with the signal
        :param standard_labels: instance of StandardLabels class for mapping to standard names
        """
        self.station_id = station_id
        self.station_label = station_label
        self.station_make = station_make
        self.station_model = station_model
        self.station_network = station_network
        self.station_lat = station_latitude
        self.station_lon = station_longitude
        self.station_alt = station_altitude
        self.audio_data = audio_data
        self.audio_epoch_s = audio_epoch_s
        self.audio_fs = audio_fs
        self.event_id = event_id

        # set the EventLabels properties
        # column names associated with event-specific metadata
        self.event_metadata = [self.event_id]

        # column names associated with station-specific metadata
        self.station_metadata = [
            self.station_id,
            self.station_label,
            self.station_make,
            self.station_model,
            self.station_network]

        # dictionary to map OREX labels to standard labels
        self.standardize_dict = {
            self.station_id: standard_labels.station_id,
            self.audio_data: standard_labels.audio_wf,
            self.audio_fs: standard_labels.audio_fs,
            self.event_id: standard_labels.event_id,
            self.station_lat: standard_labels.station_lat,
            self.station_lon: standard_labels.station_lon,
            self.station_alt: standard_labels.station_alt}


def standardize_df_columns(dataset: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """
    Standardize the columns of a pandas DataFrame using a label mapping dictionary.

    :param dataset: pandas DataFrame of dataset with original column names
    :param label_map: dictionary mapping original column names to standard column names
    :return: pandas DataFrame with standardized column names
    """
    for col in dataset.columns:
        if col in label_map.keys():
            dataset.rename(columns={col: label_map[col]}, inplace=True)
    return dataset
