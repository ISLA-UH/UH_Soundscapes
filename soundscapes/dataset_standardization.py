"""
Ideas for standardizing datasets for machine learning applications. As of 08/13/2025, the included datasets are:
Aggregated Smartphone Timeseries of Rocket-generated Acoustics (ASTRA), Smartphone High-explosive Audio Recordings
Dataset (SHAReD), OSIRIS-REx UH ISLA hypersonic signals (OREX), and the environmental sound classification dataset
(ESC-50).
"""
import numpy as np
import os
import pandas as pd
from typing import Dict, Tuple

from fastkml import kml
from fastkml.features import Placemark
from fastkml.utils import find_all
from pygeoif.geometry import Point as pyPoint

import soundscapes.standard_labels as stl

STL = stl.StandardLabels()
SL = stl.SHAReDLabels()
AL = stl.ASTRALabels()
EL = stl.ESC50Labels()
OL = stl.OREXLabels()

DIRECTORY_PATH: str = "/Users/tyler/Downloads/soundscapes_data/"

ASTRA_FILENAME: str = "ASTRA.pkl"
ASTRA_STANDARDIZED_FILENAME: str = "ASTRA_standardized.pkl"
ASTRA_EVENT_MD_FILENAME: str = "ASTRA_event_metadata.csv"
ASTRA_STATION_MD_FILENAME: str = "ASTRA_station_metadata.csv"

SHARED_FILENAME: str = "SHAReD.pkl"
SHARED_STANDARDIZED_FILENAME: str = "SHAReD_standardized.pkl"
SHARED_EVENT_MD_FILENAME: str = "SHAReD_event_metadata.csv"
SHARED_STATION_MD_FILENAME: str = "SHAReD_station_metadata.csv"

ESC50_FILENAME: str = "esc50_df_800Hz.pkl"
ESC50_STANDARDIZED_FILENAME: str = "ESC50_800Hz_standardized.pkl"
ESC50_EVENT_MD_FILENAME: str = "ESC50_event_metadata.csv"

OREX_NPZ_FILENAME: str = "orex_best_mics_800hz_1024pt.npz"
OREX_PKL_FILENAME: str = "orex_best_mics_800hz_1024pt.pkl"
OREX_STANDARDIZED_FILENAME: str = "OREX_standardized.pkl"
OREX_EVENT_MD_FILENAME: str = "OREX_event_metadata.csv"
OREX_STATION_MD_FILENAME: str = "OREX_station_metadata.csv"
OREX_FULL_FILENAME: str = "all_stations_3min_dataframe.pkl"
OREX_KML_FILENAME: str = "orex_redvox_best_stations_v03.kml"

MERGED_DS_FILENAME: str = "merged_standardized_dataset.pkl"

INCLUDE_ASTRA: bool = True
INCLUDE_SHAReD: bool = True
INCLUDE_OREX: bool = True
INCLUDE_ESC50: bool = True
SAVE_METADATA: bool = True
MERGE_DATASETS: bool = True


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


def select_astra_rocket_samples(astra_df: pd.DataFrame, al: stl.ASTRALabels = AL) -> pd.DataFrame:
    """
    Select ASTRA rocket samples using the labels given.

    :param astra_df: pandas DataFrame containing the raw ASTRA data
    :param al: ASTRALabels object containing the column labels for the ASTRA dataset
    :return: pandas DataFrame containing the selected ASTRA rocket samples
    """
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


def select_astra_noise_samples(astra_df: pd.DataFrame, al: stl.ASTRALabels = AL) -> pd.DataFrame:
    """
    selects ASTRA noise samples using the labels given.

    :param astra_df: pandas DataFrame containing the raw ASTRA data
    :param al: ASTRALabels object containing the column labels for the ASTRA dataset
    :return: pandas DataFrame containing the selected ASTRA noise samples
    """
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


def get_astra_samples(raw_astra_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get ASTRA samples for rocket and noise

    :param raw_astra_df: pandas DataFrame containing the raw ASTRA data
    :return: tuple of pandas DataFrames containing the rocket and noise samples
    """
    # add and fill station altitude, data source, and station network columns
    raw_astra_df[STL.station_alt] = [0.0] * len(raw_astra_df)  # ASTRA stations are all surface stations
    raw_astra_df[STL.data_source] = ["ASTRA"] * len(raw_astra_df)  # all data is from the ASTRA dataset
    raw_astra_df[STL.station_network] = ["FLORIDA"] * len(raw_astra_df)  # all data was recorded on the Florida network
    # make a copy of the raw dataframe to select rocket samples from
    rocket_astra_df = raw_astra_df.copy()
    # select 5 second rocket samples centered on the peak aligned time of arrival
    rocket_astra_df = select_astra_rocket_samples(rocket_astra_df)
    # add source altitude and ML label columns
    rocket_astra_df[STL.source_alt] = 0.0  # ASTRA sources are all on the surface
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
    explosion_df[STL.station_alt] = [-9999.9] * len(explosion_df)  # SHAReD stations are all surface stations
    ambient_df[STL.station_alt] = [-9999.9] * len(ambient_df)  # SHAReD stations are all surface stations
    # add and fill source altitude columns
    explosion_df[STL.source_alt] = [-9999.9] * len(explosion_df)  # SHAReD sources are all on the surface
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


def extract_orex_station_locs_pkl(
    station_label_map: Dict[str, str],
    full_orex_pkl_path: str = os.path.join(DIRECTORY_PATH, OREX_FULL_FILENAME)
) -> Dict[str, Dict[str, float]]:
    """
    Extract the best station locations from the full OREX PKL file

    :param station_label_map: dictionary mapping Redvox station IDs to station labels
    :param full_orex_pkl_path: full path of the OREX PKL file containing station location data
    :return: dictionary of station locations (lat, lon, alt) with station labels as keys
    """
    full_orex_df: pd.DataFrame = pd.read_pickle(full_orex_pkl_path)
    station_locs = {}
    for station in full_orex_df.index:
        station_id: str = full_orex_df["station_id"][station]
        if station_id not in station_label_map.keys():
            continue
        station_label = station_label_map[station_id]
        station_lats = full_orex_df[OL.station_lat][station]
        station_lons = full_orex_df[OL.station_lon][station]
        station_alts = full_orex_df[OL.station_alt][station]
        best_station_lat: float = np.nanmedian(station_lats).item()
        best_station_lon: float = np.nanmedian(station_lons).item()
        best_station_alt: float = np.nanmedian(station_alts).item()
        station_locs[station_label] = {'lat': best_station_lat,
                                       'lon': best_station_lon,
                                       'alt': best_station_alt}
        if station_label == "ALOFT S10-22":
            print(station_alts)
    return station_locs


def merge_coord(a: float, b: float) -> float:
    """
    Merge two coordinate values, returning their mean if they are within 10% of each other, otherwise NaN.

    :param a: first coordinate value
    :param b: second coordinate value
    :return: merged coordinate value
    """
    if a == 0.:
        a = np.nan
    if b == 0.:
        b = np.nan
    if np.isnan(a) and np.isnan(b):
        return np.nan
    elif np.isnan(a):
        return b
    elif np.isnan(b):
        return a
    else:
        mean: float = (a + b) / 2
        if abs(a - b) / mean > 0.1:  # if the two values differ by more than 10%, return NaN
            return np.nan
        else:
            return mean


def merge_orex_station_locs(
    station_label_map: Dict[str, str],
    orex_kml_path: str = os.path.join(DIRECTORY_PATH, OREX_KML_FILENAME),
    orex_full_pkl_path: str = os.path.join(DIRECTORY_PATH, OREX_FULL_FILENAME)
) -> Dict[str, Dict[str, float]]:
    """
    Compare the station location data extracted from OREX PKL and KML files and return the best available data

    :param orex_full_pkl_path: path to PKL file containing OREX station location data
    :param orex_kml_path: path to KML file containing OREX station location data
    :param station_label_map: dictionary mapping Redvox station IDs to OREX station labels
    :return: dictionary of OREX station locations (lat, lon, alt) with station labels as keys
    """
    # extract location data from both files
    kml_locs = extract_orex_station_locs_kml(orex_kml_path)
    pkl_locs = extract_orex_station_locs_pkl(station_label_map, orex_full_pkl_path)
    # get all unique station labels from both files
    station_labels = np.unique(np.array(list(kml_locs.keys()) + list(pkl_locs.keys())))
    # initialize an empty dictionary to hold merged locations
    merged_locs = {}
    # loop through all station labels
    for station_label in station_labels:
        # loop through both location dictionaries to ensure each has an entry for the current station label
        for loc_dict in [kml_locs, pkl_locs]:
            if station_label not in loc_dict.keys():
                # add an entry with NaNs if the station label is missing
                loc_dict[station_label] = {'lat': np.nan, 'lon': np.nan, 'alt': np.nan}
            if loc_dict[station_label]['alt'] == 0.0:
                # if altitude is 0.0, set it to NaN to avoid biasing the merged result
                loc_dict[station_label]['alt'] = np.nan
        # merge the coordinates from both dictionaries
        merged_locs[station_label] = {
            'lat': merge_coord(kml_locs[station_label]['lat'], pkl_locs[station_label]['lat']),
            'lon': merge_coord(kml_locs[station_label]['lon'], pkl_locs[station_label]['lon']),
            'alt': merge_coord(kml_locs[station_label]['alt'], pkl_locs[station_label]['alt']),
        }
    return merged_locs


def check_if_station(place_name: str) -> bool:
    """
    :param place_name: the name of the placemark
    :return: True if the placemark's name could correspond to a recording station, False otherwise
    """
    place_name_split = place_name.split(" ")
    if len(place_name_split) != 2 or place_name_split[0] not in ["CLIVE", "EUREK", "ALOFT", "WEND", "WARR"]:
        return False
    return True


def extract_orex_station_locs_kml(kml_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load locations of OSIRIS-REx recording stations from a KML file

    :param kml_file: full path of the file to load data from
    :return: dictionary of locations with identifiers
    """
    kml_data = kml.KML.parse(kml_file)
    locations: list[Placemark] = list(find_all(kml_data, of_type=Placemark))
    locations_dict = {}
    for place in locations:
        if place.geometry is None or type(place.geometry) is not pyPoint or check_if_station(place.name):
            continue
        locations_dict[place.name] = {"lon": place.geometry.x, "lat": place.geometry.y, "alt": place.geometry.z}
    return locations_dict


def load_orex_npz(orex_npz_path: str = os.path.join(DIRECTORY_PATH, OREX_NPZ_FILENAME)) -> pd.DataFrame:
    """
    :param orex_npz_path: string of full path to NPZ file containing OREX data from the best stations
    :return: pandas DataFrame containing the OREX data
    """
    orex_npz = np.load(orex_npz_path, allow_pickle=True)
    # convert npz to dataframe
    orex_df = pd.DataFrame()
    for field in orex_npz.files:
        field_element = orex_npz[field]
        if len(field_element.shape) > 1:
            field_element = [field_element[i, :] for i in range(field_element.shape[0])]
        orex_df[field] = field_element
    return orex_df


def load_orex_ds(
    load_method: str = "pkl",
    orex_path: str = os.path.join(DIRECTORY_PATH, OREX_PKL_FILENAME)
) -> pd.DataFrame:
    """
    Loads OREX data from either a pkl or npz file.  Defaults to pkl.

    :param load_method: "pkl" or "npz" indicating which file format to load the OREX data from.  Default "pkl".
    :param orex_path: full path to file containing OREX data from the best stations.  Default pkl file path.
    :return: pandas DataFrame containing the OREX data
    """
    pkl_path = os.path.join(DIRECTORY_PATH, OREX_PKL_FILENAME)
    npz_path = os.path.join(DIRECTORY_PATH, OREX_NPZ_FILENAME)
    if load_method == "pkl":
        if os.path.isfile(orex_path):
            orex_df = pd.read_pickle(orex_path)
        else:
            print("PKL file not found. Attempting to load from NPZ file instead.")
            orex_df = load_orex_npz(npz_path)
    elif load_method == "npz":
        if os.path.isfile(orex_path):
            orex_df = load_orex_npz(orex_path)
        else:
            print("NPZ file not found. Attempting to load from PKL file instead.")
            orex_df = pd.read_pickle(pkl_path)
    else:
        print("Invalid load_method parameter. Must be 'pkl' or 'npz'. Attempting to load from PKL file instead.")
        try:
            orex_df = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            print("PKL file not found. Attempting to load from NPZ file instead.")
            orex_df = load_orex_npz(npz_path)
    return orex_df


def add_orex_location_data(
    orex_df: pd.DataFrame,
    location_source: str = "merged",
    orex_kml_path: str = os.path.join(DIRECTORY_PATH, OREX_KML_FILENAME),
    orex_full_pkl_path: str = os.path.join(DIRECTORY_PATH, OREX_FULL_FILENAME)
) -> pd.DataFrame:
    """
    Add location data to the OREX DataFrame

    :param orex_df: pandas DataFrame containing the OREX data
    :param location_source: string ("pkl", "kml", or "merged") indicating which source of location data to use
    :param orex_kml_path: path to KML file containing OSIRIS-REx station location data
    :param orex_full_pkl_path: path to PKL file containing OSIRIS-REx station location data
    :return: pandas DataFrame containing the OREX data with added location data
    """
    orex_station_label_map = dict(zip(orex_df[OL.station_id], orex_df[OL.station_label]))
    if location_source.lower() == "pkl":
        if os.path.isfile(orex_full_pkl_path):
            station_locations = extract_orex_station_locs_pkl(orex_station_label_map, orex_full_pkl_path)
        else:
            print("Full OREX PKL file not found. Attempting to load from KML file instead.")
            station_locations = extract_orex_station_locs_kml(orex_kml_path)
    elif location_source.lower() == "kml":
        if os.path.isfile(orex_kml_path):
            station_locations = extract_orex_station_locs_kml(orex_kml_path)
        else:
            print("OREX KML file not found. Attempting to load from full PKL file instead.")
            station_locations = extract_orex_station_locs_pkl(orex_station_label_map, orex_full_pkl_path)
    else:  # default to 'merged' method
        if os.path.isfile(orex_kml_path) and os.path.isfile(orex_full_pkl_path):
            station_locations = merge_orex_station_locs(
                orex_station_label_map,
                orex_kml_path=orex_kml_path,
                orex_full_pkl_path=orex_full_pkl_path)
        else:
            print("One or both OREX location files not found. Attempting to load from whichever file is available.")
            if os.path.isfile(orex_kml_path):
                station_locations = extract_orex_station_locs_kml(orex_kml_path)
            else:
                print(f"No file found at {orex_kml_path}")
                station_locations = extract_orex_station_locs_pkl(orex_station_label_map, orex_full_pkl_path)

    # add station locations to the DataFrame
    lat, lon, alt = [], [], []
    for station in orex_df.index:
        station_label = orex_df[OL.station_label][station]
        lat.append(station_locations[station_label]["lat"])
        lon.append(station_locations[station_label]["lon"])
        alt.append(station_locations[station_label]["alt"])
    orex_df[OL.station_lat] = lat
    orex_df[OL.station_lon] = lon
    orex_df[OL.station_alt] = alt
    return orex_df


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

    # TODO: decide how to incorporate OREX event metadata--include estimates of source locations and times
    #  calculated for UPR poster?
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
    datasets_to_merge = []

    if INCLUDE_ASTRA:
        # load ASTRA dataset
        astra_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ASTRA_FILENAME))
        # extract metadata and standardize the dataset
        astra_standard_df, astra_event_metadata, astra_station_metadata = standardize_astra(astra_df)
        # export the standardized dataset
        astra_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILENAME))
        print(f"Exported standardized ASTRA dataset to {os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILENAME)}")
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(astra_standard_df)
        # export metadata files
        if SAVE_METADATA:
            astra_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_EVENT_MD_FILENAME), index=True)
            print(f"Exported ASTRA event metadata to {os.path.join(DIRECTORY_PATH, ASTRA_EVENT_MD_FILENAME)}")
            astra_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_STATION_MD_FILENAME), index=True)
            print(f"Exported ASTRA station metadata to {os.path.join(DIRECTORY_PATH, ASTRA_STATION_MD_FILENAME)}")

    if INCLUDE_SHAReD:
        # load SHAReD dataset
        shared_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, SHARED_FILENAME))
        # extract metadata and standardize the dataset
        shared_standard_df, shared_event_metadata, shared_station_metadata = standardize_shared(shared_df=shared_df)
        # export the standardized dataset
        shared_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILENAME))
        print(f"Exported standardized SHAReD dataset to {os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILENAME)}")
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(shared_standard_df)
        # export metadata files
        if SAVE_METADATA:
            shared_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, SHARED_EVENT_MD_FILENAME), index=True)
            print(f"Exported SHAReD event metadata to {os.path.join(DIRECTORY_PATH, SHARED_EVENT_MD_FILENAME)}")
            shared_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, SHARED_STATION_MD_FILENAME), index=True)
            print(f"Exported SHAReD station metadata to {os.path.join(DIRECTORY_PATH, SHARED_STATION_MD_FILENAME)}")

    if INCLUDE_OREX:
        # load OSIRIS-REx dataset
        orex_df = load_orex_ds(orex_path=os.path.join(DIRECTORY_PATH, OREX_NPZ_FILENAME), load_method="npz")
        # add station location data if missing
        if OL.station_lat not in orex_df.columns:
            orex_df = add_orex_location_data(orex_df,
                                             location_source="merged",
                                             orex_kml_path=os.path.join(DIRECTORY_PATH, OREX_KML_FILENAME),
                                             orex_full_pkl_path=os.path.join(DIRECTORY_PATH, OREX_FULL_FILENAME))
            orex_df.to_pickle(os.path.join(DIRECTORY_PATH, OREX_PKL_FILENAME))
        # extract metadata and standardize the dataset
        orex_standard_df, orex_station_metadata = standardize_orex(
            orex_df=orex_df,
            orex_audio_fs_hz=800.0,
            orex_event_id="OREX",
            orex_ml_label="hypersonic")
        # export the standardized dataset
        orex_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, OREX_STANDARDIZED_FILENAME))
        print(f"Exported standardized OREX dataset to {os.path.join(DIRECTORY_PATH, OREX_STANDARDIZED_FILENAME)}")
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(orex_standard_df)
        # export the station metadata
        if SAVE_METADATA:
            orex_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, OREX_STATION_MD_FILENAME), index=True)
            print(f"Exported OREX metadata to {os.path.join(DIRECTORY_PATH, OREX_STATION_MD_FILENAME)}")

    if INCLUDE_ESC50:
        esc50_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ESC50_FILENAME))
        esc50_standard_df, esc50_event_metadata = standardize_esc50(esc50_df=esc50_df)
        # export the standardized dataset
        esc50_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, ESC50_STANDARDIZED_FILENAME))
        print(f"Exported standardized ESC-50 dataset to {os.path.join(DIRECTORY_PATH, ESC50_STANDARDIZED_FILENAME)}")
        # add dataset to the list of datasets to merge
        datasets_to_merge.append(esc50_standard_df)
        # export metadata files
        if SAVE_METADATA:
            esc50_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, ESC50_EVENT_MD_FILENAME), index=True)
            print(f"Exported ESC-50 event metadata to {os.path.join(DIRECTORY_PATH, ESC50_EVENT_MD_FILENAME)}")

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
