"""
Ideas for standardizing datasets for machine learning applications. As of 08/13/2025, the included datasets are:
Aggregated Smartphone Timeseries of Rocket-generated Acoustics (ASTRA), Smartphone High-explosive Audio Recordings
Dataset (SHAReD), OSIRIS-REx UH ISLA hypersonic signals (OREX), and the environmental sound classification dataset
(ESC-50).
"""
import pandas as pd
import numpy as np
import os

DIRECTORY_PATH: str = "/Users/tyler/IdeaProjects/UH_Soundscapes/notebooks"
ASTRA_FILE_NAME: str = "ASTRA_tutorial.pkl"
SHARED_FILE_NAME: str = "SHAReD_tutorial.pkl"
ESC50_FILE_NAME: str = "esc50_tutorial_800Hz.pkl"
OREX_FILE_NAME: str = "orex_tutorial.npz"
ASTRA_STANDARDIZED_FILE_NAME: str = "ASTRA_standardized.pkl"
SHARED_STANDARDIZED_FILE_NAME: str = "SHAReD_standardized.pkl"
ASTRA_MD_FILE_NAME: str = "ASTRA_event_metadata.csv"
SHARED_MD_FILE_NAME: str = "SHAReD_event_metadata.csv"


class ASTRALabels:
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


class SHAReDLabels:
    """
    A class containing the column names used in the SHAReD dataset.
    """

    def __init__(self):
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


class ESC50Labels:
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
    ):
        """
        Defaults should be left in place for compatibility with the ESC-50 pickle files.
        :param clip_id: the ID string of the Freesound clip the audio was taken from, e.g. "freesound123456"
        :param audio_data: a numpy array containing the raw audio waveform amplitudes
        :param audio_fs: the sampling frequency of the audio waveform in Hz, e.g. 800 or 16000
        :param esc50_target: the target class number of the ESC-50 class, e.g. 37 for "clock_alarm"
        :param esc50_true_class: the name of the true ESC-50 class, e.g. "clock_alarm"
        :param yamnet_predicted_class: the name of the top class predicted by YAMNet, e.g. "Tools"
        """
        self.clip_id = clip_id
        self.audio_data = audio_data
        self.audio_fs = audio_fs
        self.esc50_target = esc50_target
        self.esc50_true_class = esc50_true_class
        self.yamnet_predicted_class = yamnet_predicted_class


class OREXLabels:
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
            audio_data: str = "station_wf",
            audio_epoch_s: str = "station_epoch_s",
            audio_fs: str = "audio_sample_rate_nominal_hz",
            event_id: str = "event_id",
    ):
        """
        Defaults should be left in place for most uses.
        :param station_id: key associated with the unique ID string of the station used to record the signal
        :param station_label: key associated with the descriptive label string of the station
        :param station_make: key associated with the recording smartphone's make
        :param station_model: key associated with the recording smartphone's model
        :param station_network: key associated with the network on which the smartphone was deployed
        :param audio_data: key associated with the audio waveform of the signal
        :param audio_epoch_s: key associated with the time array of the audio waveform in epoch seconds
        :param audio_fs: key associated with the sample rate of the audio data in Hertz
        :param event_id: key associated with the unique ID string of the event associated with the signal
        """
        self.station_id = station_id
        self.station_label = station_label
        self.station_make = station_make
        self.station_model = station_model
        self.station_network = station_network
        self.audio_data = audio_data
        self.audio_epoch_s = audio_epoch_s
        self.audio_fs = audio_fs
        self.event_id = event_id


class StandardLabels:
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


def select_astra_rocket_samples(astra_df: pandas.DataFrame):
    al = ASTRALabels()
    rocket_samples, t0s = [], []
    sample_dur = 5.
    for station in astra_df.index:
        t0 = astra_df[al.first_sample_epoch_s][station]
        peak_time = astra_df[al.p_aligned_toa_est][station]
        fs = astra_df[al.audio_fs][station]
        n_points_per_sample = int(sample_dur * fs)
        pa_toa_idx = int((peak_time - t0) * fs)
        first_sample_idx = int(pa_toa_idx - n_points_per_sample / 2)
        audio_data = astra_df[al.audio_data][station]
        audio_data = np.nan_to_num(audio_data)
        sample = audio_data[first_sample_idx:first_sample_idx + n_points_per_sample]
        rocket_samples.append(sample)
        t0s.append(t0 + first_sample_idx / fs)  # t0 is the epoch second of the first sample
    astra_df[al.audio_data] = rocket_samples
    astra_df[al.first_sample_epoch_s] = t0s
    return astra_df


def select_astra_noise_samples(astra_df, al=ASTRALabels()):
    noise_samples, t0s = [], []
    min_sample_dur = 0.96
    max_sample_dur = 5. * 10
    buffer_s = 60.
    for station in astra_df.index:
        t0 = astra_df[al.first_sample_epoch_s][station]
        fs = astra_df[al.audio_fs][station]
        n_points_per_sample_min = int(min_sample_dur * fs)
        n_points_per_sample_max = int(max_sample_dur * fs)
        sa_toa_idx = int((astra_df[al.s_aligned_toa_est][station] - t0) * fs)
        buffer_points = int(buffer_s * fs)  # 60 second buffer
        audio_data = astra_df[al.audio_data][station]
        audio_data = np.nan_to_num(audio_data)
        first_non_zero_idx = np.argwhere(audio_data != 0).flatten()[0]
        last_viable_noise_idx = sa_toa_idx - buffer_points
        n_viable_points = last_viable_noise_idx - first_non_zero_idx
        if n_viable_points >= n_points_per_sample_min:
            last_noise_idx = min(last_viable_noise_idx, first_non_zero_idx + n_points_per_sample_max)
            noise_sample = audio_data[first_non_zero_idx: last_noise_idx]
        else:
            noise_sample = np.array([])  # empty array if not enough viable data for at least 0.96 s sample
        noise_samples.append(noise_sample)
        t0s.append(t0 + first_non_zero_idx / fs)  # t0 is the epoch second of the first sample
    astra_df[al.audio_data] = noise_samples
    astra_df[al.first_sample_epoch_s] = t0s
    return astra_df


def get_astra_samples(raw_astra_df):
    stl = StandardLabels()
    noise_astra_df = raw_astra_df.copy()
    noise_astra_df = select_astra_noise_samples(noise_astra_df)
    rocket_astra_df = raw_astra_df.copy()
    rocket_astra_df = select_astra_rocket_samples(rocket_astra_df)
    al = ASTRALabels()
    rocket_astra_df = rocket_astra_df.rename(columns={
        al.audio_data: stl.audio_wf, al.first_sample_epoch_s: stl.t0_epoch_s, al.audio_fs: stl.audio_fs,
        al.station_id: stl.station_id, al.station_lat: stl.station_lat, al.station_lon: stl.station_lon,
        al.launch_id: stl.event_id, al.launch_pad_lat: stl.source_lat, al.launch_pad_lon: stl.source_lon,
        al.reported_launch_epoch_s: stl.source_epoch_s})
    rocket_astra_df[stl.station_network] = "FLORIDA"
    rocket_astra_df[stl.station_alt] = 0.0  # ASTRA stations are all surface stations
    rocket_astra_df[stl.data_source] = "ASTRA"
    rocket_astra_df[stl.source_alt] = 0.0  # ASTRA sources are all on the surface
    rocket_astra_df[stl.ml_label] = ["rocket"] * len(
        rocket_astra_df)  # suggested label for machine learning applications
    noise_astra_df = noise_astra_df.rename(columns={
        al.audio_data: stl.audio_wf, al.first_sample_epoch_s: stl.t0_epoch_s, al.audio_fs: stl.audio_fs,
        al.station_id: stl.station_id, al.station_lat: stl.station_lat, al.station_lon: stl.station_lon,
        al.launch_id: stl.event_id, al.launch_pad_lat: stl.source_lat, al.launch_pad_lon: stl.source_lon,
        al.reported_launch_epoch_s: stl.source_epoch_s})
    noise_astra_df[stl.ml_label] = ["noise"] * len(noise_astra_df)  # suggested label for machine learning applications
    noise_astra_df[stl.source_lat] = [np.nan] * len(noise_astra_df)
    noise_astra_df[stl.source_lon] = [np.nan] * len(noise_astra_df)
    noise_astra_df[stl.source_alt] = [np.nan] * len(noise_astra_df)
    noise_astra_df[stl.source_epoch_s] = [np.nan] * len(noise_astra_df)
    noise_astra_df[stl.station_network] = "FLORIDA"
    noise_astra_df[stl.station_alt] = 0.0  # ASTRA stations are all surface stations
    noise_astra_df[stl.data_source] = "ASTRA"
    return rocket_astra_df, noise_astra_df


def gen_metadata_df(df, event_column, metadata_columns):
    """
    Generate event metadata for a dataset.
    :param df: DataFrame containing the dataset.
    :param event_column: Column name containing event IDs.
    :param metadata_columns: List of column names to include in the metadata.
    :return: DataFrame with event metadata.
    """
    event_ids = df[event_column].unique()
    metadata_df = pd.DataFrame(index=event_ids, columns=metadata_columns)
    metadata_df[event_column] = event_ids
    for event in metadata_df.index:
        event_df = df[df[event_column] == event]
        for col in metadata_columns:
            if col in event_df.columns:
                metadata_df.at[event, col] = event_df[col].iloc[0]
            else:
                metadata_df.at[event, col] = np.nan
    return metadata_df


def standardize_astra():
    raw_astra_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ASTRA_FILE_NAME))
    astra_dsl = ASTRALabels()
    astra_event_metadata = gen_metadata_df(
        raw_astra_df,
        astra_dsl.launch_id,
        [astra_dsl.rocket_type, astra_dsl.rocket_model_number, astra_dsl.n_srbs, astra_dsl.reported_launch_epoch_s])
    rocket_astra_df, noise_astra_df = get_astra_samples(raw_astra_df)
    columns_to_keep = ['audio_waveform', 'first_audio_sample_epoch_s',
                       'audio_sample_rate_nominal_hz', 'station_id',
                       'station_latitude', 'station_longitude',
                       'event_id', 'source_latitude', 'source_longitude',
                       'source_epoch_s',
                       'machine_learning_label',
                       'deployment_network', 'station_altitude_m', 'data_source',
                       'source_altitude_m']
    rocket_astra_df = rocket_astra_df[columns_to_keep]
    noise_astra_df = noise_astra_df[columns_to_keep]
    astra_standardized_df = pd.concat([rocket_astra_df, noise_astra_df], ignore_index=True)
    return astra_standardized_df, astra_event_metadata


def standardize_shared():
    raw_shared_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, SHARED_FILE_NAME))
    shared_dsl = SHAReDLabels()
    standard_dsl = StandardLabels()
    for idx in raw_shared_df.index:
        if raw_shared_df[shared_dsl.event_name][idx] == "NNSS":
            raw_shared_df.at[idx, shared_dsl.event_name] = f"NNSS_{raw_shared_df[shared_dsl.event_id_number][idx]}"
    shared_event_metadata = gen_metadata_df(
        raw_shared_df,
        shared_dsl.event_name,
        [shared_dsl.explosion_detonation_time, shared_dsl.source_yield_kg, shared_dsl.effective_yield_category])
    explosion_columns_to_keep = [shared_dsl.event_name, shared_dsl.smartphone_id, shared_dsl.microphone_data,
                                 shared_dsl.microphone_time_s, shared_dsl.microphone_sample_rate_hz,
                                 shared_dsl.internal_location_latitude, shared_dsl.internal_location_longitude,
                                 shared_dsl.source_latitude, shared_dsl.source_longitude]
    ambient_columns_to_keep = [shared_dsl.event_name, shared_dsl.smartphone_id, shared_dsl.ambient_microphone_time_s,
                               shared_dsl.ambient_microphone_data, shared_dsl.microphone_sample_rate_hz,
                               shared_dsl.internal_location_latitude, shared_dsl.internal_location_longitude,
                               shared_dsl.source_latitude, shared_dsl.source_longitude]
    explosion_df = raw_shared_df[explosion_columns_to_keep]
    ambient_df = raw_shared_df[ambient_columns_to_keep]
    explosion_df[shared_dsl.microphone_time_s] = [t[0] for t in explosion_df[shared_dsl.microphone_time_s]]
    ambient_df[shared_dsl.ambient_microphone_time_s] = [np.nan] * len(ambient_df)
    ambient_df[shared_dsl.source_latitude] = [np.nan] * len(ambient_df)
    ambient_df[shared_dsl.source_longitude] = [np.nan] * len(ambient_df)
    shared_to_standard_cols = {
        shared_dsl.smartphone_id: standard_dsl.station_id,
        shared_dsl.event_name: standard_dsl.event_id,
        shared_dsl.microphone_data: standard_dsl.audio_wf,
        shared_dsl.microphone_time_s: standard_dsl.t0_epoch_s,
        shared_dsl.microphone_sample_rate_hz: standard_dsl.audio_fs,
        shared_dsl.internal_location_latitude: standard_dsl.station_lat,
        shared_dsl.internal_location_longitude: standard_dsl.station_lon,
        shared_dsl.source_latitude: standard_dsl.source_lat,
        shared_dsl.source_longitude: standard_dsl.source_lon,
        shared_dsl.ambient_microphone_data: standard_dsl.audio_wf,
        shared_dsl.ambient_microphone_time_s: standard_dsl.t0_epoch_s}
    for col in explosion_df.columns:
        if col in shared_to_standard_cols.keys():
            explosion_df = explosion_df.rename(columns={col: shared_to_standard_cols[col]})
    for col in ambient_df.columns:
        if col in shared_to_standard_cols.keys():
            ambient_df = ambient_df.rename(columns={col: shared_to_standard_cols[col]})
    ambient_df[standard_dsl.ml_label] = ["silence"] * len(ambient_df)
    explosion_df[standard_dsl.ml_label] = ["explosion"] * len(explosion_df)
    ambient_df[standard_dsl.data_source] = ["SHAReD"] * len(ambient_df)
    explosion_df[standard_dsl.data_source] = ["SHAReD"] * len(explosion_df)
    ambient_df[standard_dsl.station_alt] = [0.0] * len(ambient_df)  # SHAReD stations are all surface stations
    explosion_df[standard_dsl.station_alt] = [0.0] * len(explosion_df)  # SHAReD stations are all surface stations
    ambient_df[standard_dsl.source_alt] = [np.nan] * len(ambient_df)
    explosion_df[standard_dsl.source_alt] = [0.0] * len(explosion_df)  # SHAReD sources are all on the surface
    explosion_df[standard_dsl.station_network] = [x.split("_")[0] for x in explosion_df[standard_dsl.event_id]]
    ambient_df[standard_dsl.station_network] = [x.split("_")[0] for x in ambient_df[standard_dsl.event_id]]
    shared_standardized_df = pd.concat([explosion_df, ambient_df], ignore_index=True)
    return shared_standardized_df, shared_event_metadata


def main():
    astra_standard_df, astra_metadata_df = standardize_astra()
    shared_standard_df, shared_metadata_df = standardize_shared()
    astra_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILE_NAME))
    shared_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILE_NAME))
    astra_metadata_df.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_MD_FILE_NAME), index=True)
    shared_metadata_df.to_csv(os.path.join(DIRECTORY_PATH, SHARED_MD_FILE_NAME), index=True)


if __name__ == "__main__":
    main()
