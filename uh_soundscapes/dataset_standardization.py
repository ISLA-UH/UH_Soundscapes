"""
Standardize and/or merge acoustic datasets for machine learning applications.

As of 09/15/2025, the included datasets are: 
Aggregated Smartphone Timeseries of Rocket-generated Acoustics (ASTRA)
Smartphone High-explosive Audio Recordings Dataset (SHAReD)
OSIRIS-REx UH ISLA hypersonic signals (OREX)
Environmental sound classification dataset, fifty classes (ESC50).
"""
import numpy as np
import os
from typing import Dict, Tuple

from fastkml import kml
from fastkml.features import Placemark
from fastkml.utils import find_all
import pandas as pd
from pygeoif.geometry import Point as pyPoint

import uh_soundscapes.standard_labels as stl

STL = stl.StandardLabels()
SL = stl.SHAReDLabels()
AL = stl.ASTRALabels()
EL = stl.ESC50Labels()
OL = stl.OREXLabels()

CURRENT_DIRECTORY = os.getcwd()
# path to the directory where the original dataset files are stored
DIRECTORY_PATH: str = CURRENT_DIRECTORY

ASTRA_FILENAME: str = "ASTRA.pkl"  # name of the original ASTRA pickle file
ASTRA_STANDARDIZED_FILENAME: str = "ASTRA_standardized.pkl"  # name of the standardized ASTRA pickle file
ASTRA_EVENT_MD_FILENAME: str = "ASTRA_event_metadata.csv"  # name of the ASTRA event metadata CSV file
ASTRA_STATION_MD_FILENAME: str = "ASTRA_station_metadata.csv"  # name of the ASTRA station metadata CSV file

SHARED_FILENAME: str = "SHAReD.pkl"  # name of the original SHAReD pickle file
SHARED_STANDARDIZED_FILENAME: str = "SHAReD_standardized.pkl"  # name of the standardized SHAReD pickle file
SHARED_EVENT_MD_FILENAME: str = "SHAReD_event_metadata.csv"  # name of the SHAReD event metadata CSV file
SHARED_STATION_MD_FILENAME: str = "SHAReD_station_metadata.csv"  # name of the SHAReD station metadata CSV file

# TODO: INCLUDE 16kHZ VERSION OF ESC50?
ESC50_FILENAME: str = "ESC50_800Hz.pkl"  # name of the original ESC-50 pickle file (800 Hz version)
ESC50_STANDARDIZED_FILENAME: str = "ESC50_800Hz_standardized.pkl"  # name of the standardized ESC-50 pickle file
ESC50_EVENT_MD_FILENAME: str = "ESC50_event_metadata.csv"  # name of the ESC-50 event metadata CSV file

OREX_PKL_FILENAME: str = "OREX_UH_800Hz.pkl"  # name of the original OREX PKL file
OREX_STANDARDIZED_FILENAME: str = "OREX_standardized.pkl"  # name of the standardized OREX pickle file
OREX_STATION_MD_FILENAME: str = "OREX_station_metadata.csv"  # name of the OREX station metadata CSV file

MERGED_DS_FILENAME: str = "merged_standardized_dataset.pkl"  # name of the merged standardized dataset pickle file

STANDARDIZE_DATASETS: bool = True  # toggle dataset standardization
INCLUDE_ASTRA: bool = True  # include ASTRA data in the merged dataset
INCLUDE_SHAReD: bool = True  # include SHAReD data in the merged dataset
INCLUDE_OREX: bool = True  # include OREX data in the merged dataset
INCLUDE_ESC50: bool = True  # include ESC50 data in the merged dataset
SAVE_METADATA: bool = True  # toggle export of metadata files
MERGE_DATASETS: bool = True  # toggle dataset merging

UNKN_SURF_ALT: float = -9999.9  # float value to represent an unknown surface altitude


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Prints a summary of a standardized dataset

    :param df: pandas DataFrame containing a standardized dataset
    """
    sources, source_counts = np.unique(df[STL.data_source].values, return_counts=True)
    labels, label_counts = np.unique(df[STL.ml_label].values, return_counts=True)
    print("\nDataset Summary:")
    print(f"\tThis dataset contains {len(df)} signals from {len(sources)} different source datasets:")
    for source, count in zip(sources, source_counts):
        print(f"\t\t{count} signals from {source}")
    print(f"\tThe dataset contains {len(labels)} unique class labels:")
    for label, count in zip(labels, label_counts):
        print(f"\t\t{count} signals labeled as '{label}'")


def compile_metadata(df: pd.DataFrame, index_column: str, metadata_columns: list) -> pd.DataFrame:
    """
    Compile metadata for a dataset.

    :param df: DataFrame containing the dataset
    :param index_column: column name to use as the index for the metadata
    :param metadata_columns: list of column names to include in the metadata
    :return: DataFrame with event metadata
    """
    event_ids = df[index_column].unique()
    metadata_df = pd.DataFrame(index=event_ids, columns=metadata_columns)
    metadata_df[index_column] = event_ids
    for event in metadata_df.index:
        event_df = df[df[index_column] == event]
        for col in metadata_columns:
            if col in event_df.columns:
                metadata_df.at[event, col] = event_df[col].iloc[0]
            else:
                metadata_df.at[event, col] = np.nan
    return metadata_df


def select_astra_rocket_samples(astra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select ASTRA rocket samples from the full waveforms. Each sample is 5 seconds long and centered on the estimated
    peak arrival.

    :param astra_df: pandas DataFrame containing the full ASTRA waveforms
    :return: pandas DataFrame containing the ASTRA dataset with the waveforms trimmed to the selected samples
    """
    rocket_samples, t0s = [], []  # lists to hold the selected samples and their new t0 values
    sample_dur = 5.  # duration of the sample in seconds
    for station in astra_df.index:
        t0 = astra_df[AL.first_sample_epoch_s][station]  # original t0 of the recording
        peak_time = astra_df[AL.p_aligned_toa_est][station]  # peak arrival time of the recording
        fs = astra_df[AL.audio_fs][station]  # sample rate of the recording in Hz
        n_points_per_sample = int(sample_dur * fs)  # number of points in the sample
        pa_toa_idx = int((peak_time - t0) * fs)  # index of the peak arrival time in the full waveform
        first_sample_idx = int(pa_toa_idx - n_points_per_sample / 2)  # first index of the sample
        audio_data = astra_df[AL.audio_data][station]  # the full waveform
        audio_data = np.nan_to_num(audio_data)  # the full waveform, with NaN values filled with zeros
        sample = audio_data[first_sample_idx:first_sample_idx + n_points_per_sample]  # the selected sample
        rocket_samples.append(sample)  # append the sample
        t0s.append(t0 + first_sample_idx / fs)  # append the new t0 (epoch second of the first point in the sample)
    astra_df[AL.audio_data] = rocket_samples  # replace the full waveforms with the selected samples
    astra_df[AL.first_sample_epoch_s] = t0s  # replace the original t0 values with the new t0 values
    return astra_df


def select_astra_noise_samples(astra_df) -> pd.DataFrame:
    """
    Select ASTRA noise samples from the full waveforms. Each sample is 0.96-50.0 seconds long, ending at least 60
    seconds before the first possible arrival of the rocket launch signal at the station.

    :param astra_df: pandas DataFrame containing the full ASTRA waveforms
    :return: pandas DataFrame containing the ASTRA dataset with the waveforms trimmed to the selected samples
    """
    noise_samples, t0s = [], []  # lists to hold the selected samples and their new t0 values
    min_sample_dur = 0.96  # minimum sample duration in seconds
    max_sample_dur = 5. * 10  # maximum sample duration in seconds
    buffer_s = 60.  # minimum time in seconds between the end of the sample and the first possible signal arrival time
    for station in astra_df.index:
        t0 = astra_df[AL.first_sample_epoch_s][station]  # original t0 of the recording
        fs = astra_df[AL.audio_fs][station]  # sample rate of the recording in Hz
        n_points_per_sample_min = int(min_sample_dur * fs)  # minimum number of points in the sample
        n_points_per_sample_max = int(max_sample_dur * fs)  # maximum number of points in the sample
        sa_toa_idx = int((astra_df[AL.s_aligned_toa_est][station] - t0) * fs)  # idx of the first possible arrival time
        buffer_points = int(buffer_s * fs)  # 60 second buffer
        audio_data = astra_df[AL.audio_data][station]  # full waveform
        audio_data = np.nan_to_num(audio_data)  # full waveform, with NaNs filled with zeros
        first_non_zero_idx = np.argwhere(audio_data != 0).flatten()[0]  # idx of the first real point
        last_viable_noise_idx = sa_toa_idx - buffer_points  # idx of the last point that can be included
        n_viable_points = last_viable_noise_idx - first_non_zero_idx  # total number of viable 'noise' points
        if n_viable_points >= n_points_per_sample_min:
            last_noise_idx = min(last_viable_noise_idx, first_non_zero_idx + n_points_per_sample_max)
            noise_sample = audio_data[first_non_zero_idx: last_noise_idx]  # select the noise sample
        else:
            noise_sample = np.array([])  # empty array if not enough viable data for at least 0.96 s sample
        noise_samples.append(noise_sample)  # append the selected noise sample
        t0s.append(t0 + first_non_zero_idx / fs)  # append the new t0 (epoch second of the first point in the sample)
    astra_df[AL.audio_data] = noise_samples  # replace the full waveforms with the selected samples
    astra_df[AL.first_sample_epoch_s] = t0s  # replace the original t0 values with the new t0 values
    return astra_df


def get_astra_samples(raw_astra_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a raw ASTRA DataFrame into 2 DataFrames with the rocket and noise data samples, respectively.

    :param raw_astra_df: pandas DataFrame containing raw ASTRA data
    :return: two pandas DataFrames containing the rocket and noise samples from ASTRA, respectively
    """
    # add and fill station altitude, data source, and station network columns
    raw_astra_df[STL.station_alt] = [UNKN_SURF_ALT] * len(raw_astra_df)  # ASTRA stations are all surface stations
    raw_astra_df[STL.data_source] = ["ASTRA"] * len(raw_astra_df)  # all data is from the ASTRA dataset
    raw_astra_df[STL.station_network] = ["FLORIDA"] * len(raw_astra_df)  # all data was recorded on the Florida network
    # make a copy of the raw dataframe to select rocket samples from
    rocket_astra_df = raw_astra_df.copy()
    # select 5 second rocket samples centered on the peak aligned time of arrival
    rocket_astra_df = select_astra_rocket_samples(rocket_astra_df)
    # add source altitude and ML label columns
    rocket_astra_df[STL.source_alt] = UNKN_SURF_ALT  # ASTRA sources are all on the surface
    rocket_astra_df[STL.ml_label] = ["rocket"] * len(rocket_astra_df)  # suggested label for ML applications
    # rename columns to standard labels
    rocket_astra_df = stl.standardize_df_columns(dataset=rocket_astra_df, label_map=AL.standardize_dict)
    # fill in any missing standard columns with NaNs
    for col in STL.standard_labels:
        if col not in rocket_astra_df.columns:
            rocket_astra_df[col] = [np.nan] * len(rocket_astra_df)

    # make a copy of the raw dataframe to select noise samples from
    noise_astra_df = raw_astra_df.copy()
    # select < 50 second noise samples ending at least 60 seconds before the start-aligned time of arrival
    noise_astra_df = select_astra_noise_samples(noise_astra_df)
    # add and fill ML label column
    noise_astra_df[STL.ml_label] = ["noise"] * len(noise_astra_df)  # suggested label for ML applications
    # rename columns to standard labels
    noise_astra_df = stl.standardize_df_columns(dataset=noise_astra_df, label_map=AL.standardize_dict)
    # reset source location and time columns to NaN
    noise_astra_df[STL.source_lat] = [np.nan] * len(noise_astra_df)
    noise_astra_df[STL.source_lon] = [np.nan] * len(noise_astra_df)
    noise_astra_df[STL.source_alt] = [np.nan] * len(noise_astra_df)
    noise_astra_df[STL.source_epoch_s] = [np.nan] * len(noise_astra_df)
    # fill in any other missing standard columns with NaNs
    for col in STL.standard_labels:
        if col not in noise_astra_df.columns:
            noise_astra_df[col] = [np.nan] * len(noise_astra_df)
    return rocket_astra_df, noise_astra_df


def standardize_astra(astra_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardize the ASTRA dataset

    :param: astra_df: DataFrame containing the raw ASTRA data
    :return: pandas DataFrames containing the standardized data, event and station metadata
    """
    # compile ASTRA event metadata
    astra_event_metadata = compile_metadata(
        astra_df,
        AL.launch_id,
        AL.event_metadata)
    # compile ASTRA station metadata
    astra_station_metadata = compile_metadata(
        astra_df,
        AL.station_id,
        AL.station_metadata)
    # get ASTRA rocket and noise samples
    rocket_astra_df, noise_astra_df = get_astra_samples(astra_df)
    # keep only standard columns
    rocket_astra_df = rocket_astra_df[STL.standard_labels]
    noise_astra_df = noise_astra_df[STL.standard_labels]
    # concatenate rocket and noise dataframes
    astra_standardized_df = pd.concat([rocket_astra_df, noise_astra_df], ignore_index=True)
    return astra_standardized_df, astra_event_metadata, astra_station_metadata


def standardize_shared(shared_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardize the SHAReD dataset

    :param: shared_df: DataFrame containing the raw SHAReD data
    :return: pandas DataFrames containing the standardized data, event and station metadata
    """
    # change NNSS event names from "NNSS" to "NNSS_<event_id_number>" to make them unique
    for idx in shared_df.index:
        if shared_df[SL.event_name][idx] == "NNSS":
            shared_df.at[idx, SL.event_name] = f"NNSS_{shared_df[SL.event_id_number][idx]}"
    # compile SHAReD event metadata
    shared_event_metadata = compile_metadata(
        shared_df,
        SL.event_name,
        SL.event_metadata)
    # compile SHAReD station metadata
    shared_station_metadata = compile_metadata(
        shared_df,
        SL.smartphone_id,
        SL.station_metadata)
    # columns to keep for the explosion DataFrame
    explosion_columns = [SL.event_name, SL.smartphone_id, SL.microphone_data,
                         SL.microphone_time_s, SL.microphone_sample_rate_hz,
                         SL.internal_location_latitude, SL.internal_location_longitude,
                         SL.source_latitude, SL.source_longitude, SL.explosion_detonation_time]
    # columns to keep for the ambient DataFrame
    ambient_columns = [SL.event_name, SL.smartphone_id, SL.ambient_microphone_time_s,
                       SL.ambient_microphone_data, SL.microphone_sample_rate_hz,
                       SL.internal_location_latitude, SL.internal_location_longitude,
                       SL.source_latitude, SL.source_longitude]
    # create separate DataFrames for explosion and ambient data
    explosion_df = shared_df[explosion_columns]
    ambient_df = shared_df[ambient_columns]
    # add and fill first sample epoch second columns
    explosion_df[STL.t0_epoch_s] = [t[0] for t in explosion_df[SL.microphone_time_s]]
    ambient_df[STL.t0_epoch_s] = [t[0] for t in ambient_df[SL.ambient_microphone_time_s]]
    # add and fill data source columns
    explosion_df[STL.data_source] = ["SHAReD"] * len(explosion_df)
    ambient_df[STL.data_source] = ["SHAReD"] * len(ambient_df)
    # add and fill station altitude columns
    explosion_df[STL.station_alt] = [UNKN_SURF_ALT] * len(explosion_df)  # SHAReD stations are all surface stations
    ambient_df[STL.station_alt] = [UNKN_SURF_ALT] * len(ambient_df)  # SHAReD stations are all surface stations
    # add and fill source altitude columns
    explosion_df[STL.source_alt] = [UNKN_SURF_ALT] * len(explosion_df)  # SHAReD sources are all on the surface
    ambient_df[STL.source_alt] = [np.nan] * len(ambient_df)  # SHAReD ambient data has no identified source
    # add and fill station network columns
    explosion_df[STL.station_network] = [x.split("_")[0] for x in explosion_df[SL.event_name]]
    ambient_df[STL.station_network] = [x.split("_")[0] for x in ambient_df[SL.event_name]]
    # add and fill ML label columns with recommended class labels
    explosion_df[STL.ml_label] = ["explosion"] * len(explosion_df)
    ambient_df[STL.ml_label] = ["silence"] * len(ambient_df)
    # rename columns to standard labels and fill in any missing standard columns with NaNs
    explosion_df = stl.standardize_df_columns(dataset=explosion_df, label_map=SL.standardize_dict)
    for col in STL.standard_labels:
        if col not in explosion_df.columns:
            print(f"SHAReD explosion DataFrame missing column: {col}. Filling with NaN.")
            explosion_df[col] = [np.nan] * len(explosion_df)
    ambient_df = stl.standardize_df_columns(dataset=ambient_df, label_map=SL.standardize_dict)
    for col in STL.standard_labels:
        if col not in ambient_df.columns:
            print(f"SHAReD ambient DataFrame missing column: {col}. Filling with NaN.")
            ambient_df[col] = [np.nan] * len(ambient_df)
    # reset source location columns to NaN for ambient data
    ambient_df[SL.source_latitude] = [np.nan] * len(ambient_df)
    ambient_df[SL.source_longitude] = [np.nan] * len(ambient_df)

    # keep only the standard columns
    explosion_df = explosion_df[STL.standard_labels]
    ambient_df = ambient_df[STL.standard_labels]

    # concatenate explosion and ambient dataframes
    shared_standardized_df = pd.concat([explosion_df, ambient_df], ignore_index=True)
    return shared_standardized_df, shared_event_metadata, shared_station_metadata


def get_station_model(station_label_string: str) -> str:
    """
    Extract the station model from the OREX station label string

    :param station_label_string:
    :return: a string containing the station model
    """
    return station_label_string.split(" ")[-1].split("-")[0]


def get_station_network(station_label_string: str) -> str:
    """
    Extract the station network from the OREX station label string

    :param station_label_string:
    :return: a string containing the station network
    """
    return station_label_string.split(" ")[0]


def load_orex_ds(orex_path: str = os.path.join(DIRECTORY_PATH, OREX_PKL_FILENAME)) -> pd.DataFrame:
    """
    Loads OREX data from either a pkl or npz file.  Defaults to pkl.

    :param load_method: "pkl" or "npz" indicating which file format to load the OREX data from.  Default "pkl".
    :param orex_path: full path to file containing OREX data from the best stations.  Default pkl file path.
    :return: pandas DataFrame containing the OREX data
    """
    if os.path.isfile(orex_path):
        try:
            orex_df = pd.read_pickle(orex_path)
            return orex_df
        except Exception as e:
            raise RuntimeError(f"Error loading OREX PKL file: {e}")
    else:
        raise FileNotFoundError(f"PKL file at {orex_path} not found.")


def standardize_orex(
    orex_df: pd.DataFrame,
    orex_audio_fs_hz: float = 800.0,
    orex_event_id: str = "OREX",
    orex_ml_label: str = "hypersonic",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master function to standardize the OREX hypersonic dataset

    :param orex_df: pandas DataFrame containing the OREX data
    :param orex_audio_fs_hz: the sample rate of the OREX audio data in Hz
    :param orex_event_id: the event ID string to assign to all OREX signals
    :param orex_ml_label: the ML label string to assign to all OREX signals
    :return: pandas DataFrames containing the standardized dataset and the OREX metadata
    """
    # add and fill event ID, sample rate, and ML label columns if they are missing
    n_signals = len(orex_df)
    if OL.audio_fs not in orex_df.columns:
        orex_df[OL.audio_fs] = [orex_audio_fs_hz] * n_signals
    if OL.event_id not in orex_df.columns:
        orex_df[OL.event_id] = [orex_event_id] * n_signals
    if STL.ml_label not in orex_df.columns:
        orex_df[STL.ml_label] = [orex_ml_label] * n_signals
    if STL.data_source not in orex_df.columns:
        orex_df[STL.data_source] = ["UH_OREX"] * n_signals
    if STL.t0_epoch_s not in orex_df.columns:
        orex_df[STL.t0_epoch_s] = [time[0] for time in orex_df[OL.audio_epoch_s]]

    # extract station model and network data from station labels and add to the DataFrame
    if OL.station_model not in orex_df.columns:
        orex_df[OL.station_model] = [get_station_model(sls) for sls in orex_df[OL.station_label]]
    if OL.station_network not in orex_df.columns:
        orex_df[OL.station_network] = [get_station_network(sls) for sls in orex_df[OL.station_label]]

    # compile OREX station metadata
    orex_station_metadata = compile_metadata(
        orex_df,
        OL.station_id,
        OL.station_metadata)

    # rename columns to standard labels and fill in any missing standard columns with NaNs
    orex_df = stl.standardize_df_columns(dataset=orex_df, label_map=OL.standardize_dict)
    for col in STL.standard_labels:
        if col not in orex_df.columns:
            print(f"Standard column {col} missing from OREX DataFrame. Filling column with NaN.")
            orex_df[col] = [np.nan] * n_signals

    # keep only the standard labels
    orex_df = orex_df[STL.standard_labels]

    return orex_df, orex_station_metadata


def standardize_esc50(esc50_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master function to standardize the ESC-50 environmental sound dataset

    :param esc50_df: pandas DataFrame containing the ESC-50 data
    :return: pandas DataFrames containing the standardized dataset and the ESC-50 metadata
    """
    # add and fill event ID, sample rate, and ML label columns if they are missing
    n_signals = len(esc50_df)
    if STL.data_source not in esc50_df.columns:
        esc50_df[STL.data_source] = ["ESC-50"] * n_signals

    # compile ESC-50 event metadata
    esc50_event_metadata = compile_metadata(
        esc50_df,
        EL.clip_id,
        EL.event_metadata)

    # rename columns to standard labels and fill in any missing standard columns with NaNs
    esc50_df = stl.standardize_df_columns(dataset=esc50_df, label_map=EL.standardize_dict)
    for col in STL.standard_labels:
        if col not in esc50_df.columns:
            print(f"Standard column {col} missing from ESC-50 DataFrame. Filling column with NaN.")
            esc50_df[col] = [np.nan] * n_signals

    # keep only the standard labels
    esc50_df = esc50_df[STL.standard_labels]

    return esc50_df, esc50_event_metadata


def standardize_datasets():
    """
    Standardize all datasets specified by the config values at top of file.
    """
    datasets_to_merge = []

    if INCLUDE_ASTRA:
        if STANDARDIZE_DATASETS:
            # load raw ASTRA dataset
            astra_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ASTRA_FILENAME))
            # extract metadata and standardize the dataset
            astra_standard_df, astra_event_metadata, astra_station_metadata = standardize_astra(astra_df)
            # export the standardized dataset
            astra_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILENAME))
            print(f"Exported standardized ASTRA dataset to {os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILENAME)}")
            if SAVE_METADATA:
                # export metadata files
                astra_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_EVENT_MD_FILENAME), index=True)
                print(f"Exported ASTRA event metadata to {os.path.join(DIRECTORY_PATH, ASTRA_EVENT_MD_FILENAME)}")
                astra_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_STATION_MD_FILENAME), index=True)
                print(f"Exported ASTRA station metadata to {os.path.join(DIRECTORY_PATH, ASTRA_STATION_MD_FILENAME)}")
        else:
            # load standardized ASTRA dataset
            astra_standard_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILENAME))
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(astra_standard_df)

    if INCLUDE_SHAReD:
        if STANDARDIZE_DATASETS:
            # load raw SHAReD dataset
            shared_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, SHARED_FILENAME))
            # extract metadata and standardize the dataset
            shared_standard_df, shared_event_metadata, shared_station_metadata = standardize_shared(shared_df=shared_df)
            # export the standardized dataset
            shared_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILENAME))
            print(f"Exported standardized dataset to {os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILENAME)}")
            if SAVE_METADATA:
                # export metadata files
                shared_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, SHARED_EVENT_MD_FILENAME), index=True)
                print(f"Exported SHAReD event metadata to {os.path.join(DIRECTORY_PATH, SHARED_EVENT_MD_FILENAME)}")
                shared_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, SHARED_STATION_MD_FILENAME), index=True)
                print(f"Exported SHAReD station metadata to {os.path.join(DIRECTORY_PATH, SHARED_STATION_MD_FILENAME)}")
        else:
            # load standardized SHAReD dataset
            shared_standard_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILENAME))
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(shared_standard_df)

    if INCLUDE_OREX:
        if STANDARDIZE_DATASETS:
            # load raw OSIRIS-REx dataset
            orex_df = load_orex_ds(orex_path=os.path.join(DIRECTORY_PATH, OREX_PKL_FILENAME))
            # extract metadata and standardize the dataset
            orex_standard_df, orex_station_metadata = standardize_orex(
                orex_df=orex_df,
                orex_audio_fs_hz=800.0,
                orex_event_id="OREX",
                orex_ml_label="hypersonic")
            # export the standardized dataset
            orex_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, OREX_STANDARDIZED_FILENAME))
            print(f"Exported standardized OREX dataset to {os.path.join(DIRECTORY_PATH, OREX_STANDARDIZED_FILENAME)}")
            if SAVE_METADATA:
                # export the station metadata
                orex_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, OREX_STATION_MD_FILENAME), index=True)
                print(f"Exported OREX metadata to {os.path.join(DIRECTORY_PATH, OREX_STATION_MD_FILENAME)}")
        else:
            # load standardized OSIRIS-REx dataset
            orex_standard_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, OREX_STANDARDIZED_FILENAME))
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(orex_standard_df)

    if INCLUDE_ESC50:
        if STANDARDIZE_DATASETS:
            # load raw ESC-50 dataset
            esc50_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ESC50_FILENAME))
            esc50_standard_df, esc50_event_metadata = standardize_esc50(esc50_df=esc50_df)
            # export the standardized dataset
            esc50_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, ESC50_STANDARDIZED_FILENAME))
            print(f"Exported standardized dataset to {os.path.join(DIRECTORY_PATH, ESC50_STANDARDIZED_FILENAME)}")
            if SAVE_METADATA:
                # export metadata
                esc50_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, ESC50_EVENT_MD_FILENAME), index=True)
                print(f"Exported ESC-50 event metadata to {os.path.join(DIRECTORY_PATH, ESC50_EVENT_MD_FILENAME)}")
        else:
            # load standardized ESC-50 dataset
            esc50_standard_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ESC50_STANDARDIZED_FILENAME))
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(esc50_standard_df)

    # merge datasets into single DataFrame
    if MERGE_DATASETS:
        # concatenate all DataFrames in the list
        merged_df = pd.concat(datasets_to_merge, ignore_index=True)
        # export the merged dataset
        merged_path = os.path.join(DIRECTORY_PATH, MERGED_DS_FILENAME)
        merged_df.to_pickle(merged_path)
        print(f"Exported merged dataset to {merged_path}")
        summarize_dataset(merged_df)


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    standardize_datasets()
