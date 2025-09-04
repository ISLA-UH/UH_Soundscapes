"""
Ideas for standardizing datasets for machine learning applications. As of 08/13/2025, the included datasets are:
Aggregated Smartphone Timeseries of Rocket-generated Acoustics (ASTRA), Smartphone High-explosive Audio Recordings
Dataset (SHAReD), OSIRIS-REx UH ISLA hypersonic signals (OREX), and the environmental sound classification dataset
(ESC-50).
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List

from fastkml import kml, styles
from fastkml.features import Placemark
from fastkml.utils import find_all

from pygeoif.geometry import Point as pyPoint
import geopandas as gpd

import sandbox.sarah.standard_labels as stl

STL = stl.StandardLabels()
SL = stl.SHAReDLabels()
AL = stl.ASTRALabels()
EL = stl.ESC50Labels()
OL = stl.OREXLabels()

DIRECTORY_PATH: str = "/Users/spopen/redvox/data/rockets_data/datasets_pkl"

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

OREX_FILENAME: str = "orex_best_mics_800hz_1024pt.npz"
OREX_STANDARDIZED_FILENAME: str = "OREX_standardized.pkl"
OREX_EVENT_MD_FILENAME: str = "OREX_event_metadata.csv"
OREX_STATION_MD_FILENAME: str = "OREX_station_metadata.csv"
OREX_FULL_FILENAME: str = "all_stations_3min_dataframe.pkl"
OREX_KML_FILENAME: str = "OSIRIS-REx_deployment.kml"

MERGED_DS_FILENAME: str = "merged_standardized_dataset.pkl"

INCLUDE_ASTRA: bool = True
INCLUDE_SHAReD: bool = True
INCLUDE_OREX: bool = True
INCLUDE_ESC50: bool = True
SAVE_METADATA: bool = True


def select_astra_rocket_samples(astra_df, al=AL) -> pd.DataFrame:
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


def select_astra_noise_samples(astra_df, al=AL) -> pd.DataFrame:
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


def get_astra_samples(raw_astra_df) -> (pd.DataFrame, pd.DataFrame):
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


def compile_metadata(df, index_column, metadata_columns) -> pd.DataFrame:
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


def standardize_astra() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Master function to standardize the ASTRA dataset
    :return: pandas DataFrames containing the standardized data and metadata
    """
    # load ASTRA dataset
    raw_astra_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, ASTRA_FILENAME))
    # compile ASTRA event metadata
    astra_event_metadata = compile_metadata(
        raw_astra_df,
        AL.launch_id,
        AL.event_metadata)
    # compile ASTRA station metadata
    astra_station_metadata = compile_metadata(
        raw_astra_df,
        AL.station_id,
        AL.station_metadata)
    # get ASTRA rocket and noise samples
    rocket_astra_df, noise_astra_df = get_astra_samples(raw_astra_df)
    # keep only standard columns
    rocket_astra_df = rocket_astra_df[STL.standard_labels]
    noise_astra_df = noise_astra_df[STL.standard_labels]
    # concatenate rocket and noise dataframes
    astra_standardized_df = pd.concat([rocket_astra_df, noise_astra_df], ignore_index=True)
    return astra_standardized_df, astra_event_metadata, astra_station_metadata


def standardize_shared() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Master function to standardize the SHAReD dataset
    :return: pandas DataFrames containing the standardized data and metadata
    """
    # load SHAReD dataset
    raw_shared_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, SHARED_FILENAME))
    # change NNSS event names from "NNSS" to "NNSS_<event_id_number>" to make them unique
    for idx in raw_shared_df.index:
        if raw_shared_df[SL.event_name][idx] == "NNSS":
            raw_shared_df.at[idx, SL.event_name] = f"NNSS_{raw_shared_df[SL.event_id_number][idx]}"
    shared_event_metadata = compile_metadata(
        raw_shared_df,
        SL.event_name,
        SL.event_metadata)
    shared_station_metadata = compile_metadata(
        raw_shared_df,
        SL.smartphone_id,
        SL.station_metadata)
    explosion_columns = [SL.event_name, SL.smartphone_id, SL.microphone_data,
                         SL.microphone_time_s, SL.microphone_sample_rate_hz,
                         SL.internal_location_latitude, SL.internal_location_longitude,
                         SL.source_latitude, SL.source_longitude]
    ambient_columns = [SL.event_name, SL.smartphone_id, SL.ambient_microphone_time_s,
                       SL.ambient_microphone_data, SL.microphone_sample_rate_hz,
                       SL.internal_location_latitude, SL.internal_location_longitude,
                       SL.source_latitude, SL.source_longitude]
    explosion_df = raw_shared_df[explosion_columns]
    ambient_df = raw_shared_df[ambient_columns]
    # add and fill first sample epoch second columns
    explosion_df[STL.t0_epoch_s] = [t[0] for t in explosion_df[SL.microphone_time_s]]
    ambient_df[STL.t0_epoch_s] = [t[0] for t in ambient_df[SL.ambient_microphone_time_s]]
    # add and fill data set source columns
    explosion_df[STL.data_source] = ["SHAReD"] * len(explosion_df)
    ambient_df[STL.data_source] = ["SHAReD"] * len(ambient_df)
    # add and fill station altitude columns
    explosion_df[STL.station_alt] = [0.0] * len(explosion_df)  # SHAReD stations are all surface stations
    ambient_df[STL.station_alt] = [0.0] * len(ambient_df)  # SHAReD stations are all surface stations
    # add and fill source altitude columns
    explosion_df[STL.source_alt] = [0.0] * len(explosion_df)  # SHAReD sources are all on the surface
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
            explosion_df[col] = [np.nan] * len(explosion_df)
    ambient_df = stl.standardize_df_columns(dataset=ambient_df, label_map=SL.standardize_dict)
    for col in STL.standard_labels:
        if col not in ambient_df.columns:
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


def get_station_model(station_label_string) -> str:
    """
    Function to extract the station model from the OREX station label string
    :param station_label_string:
    :return: a string containing the station model
    """
    return station_label_string.split(" ")[-1].split("-")[0]


def get_station_network(station_label_string) -> str:
    """
    Function to extract the station network from the OREX station label string
    :param station_label_string:
    :return: a string containing the station network
    """
    return station_label_string.split(" ")[0]


def get_orex_station_locations_pkl() -> Dict[str, Dict[str, float]]:
    """
    Function to extract the best station locations from the full OREX PKL file
    :return: dictionary of station locations (lat, lon, alt) with station IDs as keys
    """
    full_orex_df = pd.read_pickle(os.path.join(DIRECTORY_PATH, OREX_FULL_FILENAME))
    station_locs = {}
    for station in full_orex_df.index:
        station_id = full_orex_df["station_id"][station]
        station_lats = full_orex_df[OL.station_lat][station]
        station_lons = full_orex_df[OL.station_lon][station]
        station_alts = full_orex_df[OL.station_alt][station]
        best_station_lat = np.nanmedian(station_lats).item()
        best_station_lon = np.nanmedian(station_lons).item()
        best_station_alt = np.nanmedian(station_alts).item()
        station_locs[station_id] = {'lat': best_station_lat,
                                    'lon': best_station_lon,
                                    'alt': best_station_alt}
    return station_locs


def load_kml(kml_file: str) -> Dict[str, Dict[str, float]]:
    """
    load location from a kml file
    :param kml_file: full path of the file to load data from
    :return: dictionary of locations with identifiers
    """
    kml_data = kml.KML.parse(kml_file)
    locations: list[Placemark] = list(find_all(kml_data, of_type=Placemark))
    set_locations = {}
    for place in locations:
        set_locations[place.name] = {"lon": place.geometry.x, "lat": place.geometry.y, "alt": place.geometry.z}
    return set_locations


def compare_kml_pkl_locs(max_dist_m: float = 20.0) -> Dict[str, str]:
    """
    Function to compare the coordinates in the KML and PKL files
    :return: dictionary mapping Redvox station IDs to KML station names if the KML and PKL coords are within max_dist_m
    """
    # get station locations from full OREX PKL file
    station_locations_pkl = get_orex_station_locations_pkl()
    # get station locations from OREX KML file
    station_locations_kml = load_kml(os.path.join(DIRECTORY_PATH, OREX_KML_FILENAME))
    station_ids = list(station_locations_pkl.keys())

    kml_station_id_map = {}

    for station_id in station_ids:
        pkl_lat = station_locations_pkl[station_id]['lat']
        pkl_lon = station_locations_pkl[station_id]['lon']
        pkl_point_2d = pyPoint(x=pkl_lon, y=pkl_lat)
        kml_stations, distances = [], []
        for kml_station in station_locations_kml.keys():
            kml_point_2d = pyPoint(x=station_locations_kml[kml_station]['lon'],
                                   y=station_locations_kml[kml_station]['lat'])
            points_df = gpd.GeoDataFrame({'geometry': [pkl_point_2d, kml_point_2d]}, crs='EPSG:4326')
            points_df = points_df.to_crs('EPSG:5234')
            points_df2 = points_df.shift()
            dist_m: float = points_df.distance(points_df2)[1]
            distances.append(dist_m)
            kml_stations.append(kml_station)
        dist_df = pd.DataFrame()
        dist_df["kml_station"] = kml_stations
        dist_df["dist_m"] = distances
        dist_df.sort_values(by="dist_m", inplace=True)
        print(f"Closest KML stations to Redvox station {station_id}:")
        for kml_station in dist_df.index:
            print(f"\t{dist_df['kml_station'][kml_station]}: {dist_df['dist_m'][kml_station]:.1f} meters")
        if dist_df["dist_m"][dist_df.index[0]] < max_dist_m:
            kml_station_id_map[station_id] = dist_df["kml_station"][dist_df.index[0]]
    stations_missing_from_kml = [sid for sid in station_ids if sid not in kml_station_id_map.keys()]
    print(f"Station in PKL file missing from KML:")
    for msid in stations_missing_from_kml:
        print(f"\t{msid}")
    return kml_station_id_map


def standardize_orex(location_source: str = 'pkl') -> (pd.DataFrame, pd.DataFrame):
    """
    Master function to standardize the OREX hypersonic dataset
    :param location_source: string ('pkl', 'kml', or 'best') indicating which source of station location data to use
    :return: pandas DataFrames containing the standardized dataset and the OREX metadata
    """
    # load OREX dataset
    orex_npz = np.load(os.path.join(DIRECTORY_PATH, OREX_FILENAME), allow_pickle=True)

    # convert npz to dataframe
    orex_df = pd.DataFrame()
    for field in orex_npz.files:
        field_element = orex_npz[field]
        if len(field_element.shape) > 1:
            field_element = [field_element[i, :] for i in range(field_element.shape[0])]
        orex_df[field] = field_element

    # add and fill event ID, sample rate, and ML label columns
    audio_fs_hz: float = 800.0
    orex_event_id: str = "OREX"
    ml_label: str = "hypersonic"
    n_signals = len(orex_df)
    orex_df[OL.audio_fs] = [audio_fs_hz] * n_signals
    orex_df[OL.event_id] = [orex_event_id] * n_signals
    orex_df[STL.ml_label] = [ml_label] * n_signals

    # extract station model and network data from station labels and add to the DataFrame
    orex_df[OL.station_model] = [get_station_model(sls) for sls in orex_df[OL.station_label]]
    orex_df[OL.station_network] = [get_station_network(sls) for sls in orex_df[OL.station_label]]

    if location_source == "kml":
        # TODO: find missing ground truth locations--another KML file somewhere with CLIVE stations?
        # # get station locations from OREX KML file
        # station_locations = load_kml(os.path.join(DIRECTORY_PATH, OREX_KML_FILENAME))
        raise NotImplementedError("KML location source not yet implemented.")
    elif location_source == "best":
        # TODO: decide how to incorporate KML coords (1-12 meter diff from PKL)
        #  Option 1: take mean of KML coords and 'best' samples median from PKL
        #  Option 2: add KML coords to 'best' samples before taking median
        #  Option 3: use KML or PKL alone
        raise NotImplementedError("Best location option not yet implemented.")
    else:
        # TODO: validate altitude values and units in PKL file
        # get station locations from full OREX PKL file
        station_locations = get_orex_station_locations_pkl()

    # add station locations to the DataFrame
    lat, lon, alt = [], [], []
    for station in orex_df.index:
        station_id = orex_df[OL.station_id][station]
        lat.append(station_locations[station_id]['lat'])
        lon.append(station_locations[station_id]['lon'])
        alt.append(station_locations[station_id]['alt'])
    orex_df[OL.station_lat] = lat
    orex_df[OL.station_lon] = lon
    orex_df[OL.station_alt] = alt

    # TODO: decide how to incorporate OREX event metadata--include estimates of source locations and times
    #  calculated for UPR poster?

    # compile OREX station metadata
    orex_station_metadata = compile_metadata(
        orex_df,
        OL.station_id,
        OL.station_metadata)

    # rename columns to standard labels and fill in any missing standard columns with NaNs
    # TODO: decide how to incorporate source location and time info--use estimates of source locations and times
    #  calculated for UPR poster?
    orex_df = stl.standardize_df_columns(dataset=orex_df, label_map=OL.standardize_dict)
    for col in STL.standard_labels:
        if col not in orex_df.columns:
            orex_df[col] = [np.nan] * n_signals

    # keep only the standard labels
    orex_df = orex_df[STL.standard_labels]
    return orex_df, orex_station_metadata


def main():
    datasets_to_merge = []

    if INCLUDE_ASTRA:
        astra_standard_df, astra_event_metadata, astra_station_metadata = standardize_astra()
        astra_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, ASTRA_STANDARDIZED_FILENAME))
        datasets_to_merge.append(astra_standard_df)
        if SAVE_METADATA:
            astra_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_EVENT_MD_FILENAME), index=True)
            astra_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, ASTRA_STATION_MD_FILENAME), index=True)

    if INCLUDE_SHAReD:
        shared_standard_df, shared_event_metadata, shared_station_metadata = standardize_shared()
        shared_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, SHARED_STANDARDIZED_FILENAME))
        datasets_to_merge.append(shared_standard_df)
        if SAVE_METADATA:
            shared_event_metadata.to_csv(os.path.join(DIRECTORY_PATH, SHARED_EVENT_MD_FILENAME), index=True)
            shared_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, SHARED_STATION_MD_FILENAME), index=True)

    if INCLUDE_OREX:
        compare_kml_pkl_locs()
        orex_standard_df, orex_station_metadata = standardize_orex(location_source='pkl')
        orex_standard_df.to_pickle(os.path.join(DIRECTORY_PATH, OREX_STANDARDIZED_FILENAME))
        datasets_to_merge.append(orex_standard_df)
        if SAVE_METADATA:
            orex_station_metadata.to_csv(os.path.join(DIRECTORY_PATH, OREX_STATION_MD_FILENAME), index=True)

    if INCLUDE_ESC50:
        print("ESC-50 standardization not yet implemented.")

    merged_df = pd.concat(datasets_to_merge, ignore_index=True)
    print(merged_df)
    print(merged_df.columns)
    merged_df.to_pickle(os.path.join(DIRECTORY_PATH, MERGED_DS_FILENAME))


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
