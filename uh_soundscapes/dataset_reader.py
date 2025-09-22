"""
Generic dataset reader
Assumes the dataset can be read via pickle and is a pandas dataframe.
Make inherited versions of the classes for specific datasets.
"""
from typing import Any, List, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from quantum_inferno.cwt_atoms import cwt_chirp_from_sig
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_vert_example

import uh_soundscapes.standard_labels as stdlbl

# Colors for multiple plots
CBF_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


class PlotBase:
    """
    Base class for plotting utilities.
    """
    def __init__(self, fig_size: Tuple[int, int], nrows: int = 1, ncols: int = 1):
        """
        Initialize the base plot class with default parameters.

        :param fig_size: Tuple of (width, height) for the figure size.
        """
        self.y_adj: int = 0
        self.y_adj_buff: float = 2.2
        self.t_max: int = 0
        self.ticks: List = []
        self.tick_labels: List[str] = []
        self.font_size = 12
        self.base_ylim: Tuple[float, float] = (self.y_adj - self.y_adj_buff / 2., self.y_adj + self.y_adj_buff / 2.)
        self.marker_lines_zorder: int = 10
        self.waveform_color: str = "k"
        self.fig_size: Tuple[int, int] = fig_size
        self.cbf_colors: List[str] = CBF_COLOR_CYCLE
        if nrows > 1 or ncols > 1:
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.fig_size)
        else:
            self.fig, self.ax = plt.subplots(figsize=self.fig_size)

    def plot_tfr(self, tfr_title: str, station_id: str, fs: float, timestamps: np.ndarray, data: np.ndarray) -> Figure:
        """
        Plot the time-frequency representation (TFR) of the given data.
        :param tfr_title: Title for the TFR plot.
        :param station_id: Identifier for the station.
        :param fs: Sampling frequency of the data.
        :param timestamps: Timestamps corresponding to the data.
        :param data: Audio data to be transformed.
        :return: Matplotlib Figure object containing the TFR plot.
        """
        _, cwt_bits, time_s, frequency_cwt_hz = cwt_chirp_from_sig(
            sig_wf=data,
            frequency_sample_rate_hz=fs,
            band_order_nth=3
        )
        return plot_wf_mesh_vert_example(
            station_id=station_id,
            wf_panel_a_sig=data,
            wf_panel_a_time=timestamps,
            mesh_time=time_s,
            mesh_frequency=frequency_cwt_hz,
            mesh_panel_b_tfr=cwt_bits,
            figure_title=tfr_title,
        )


class DatasetLabels:
    """
    Base class for dataset labels.  Inherited classes should implement specific labels.

    Properties:
        event_labels: EventLabels class that contains event-specific label names

        standard_labels: StandardLabels class that contains all the standardized label names
    """
    def __init__(self, event_labels: stdlbl.EventLabels, 
                 standard_labels: stdlbl.StandardLabels = stdlbl.StandardLabels()):
        """
        Initialize the dataset labels.  Inherited class will implement specific labels.

        :param event_labels: EventLabels class that contains event-specific label names.
        :param standard_labels: StandardLabels class that contains all the standardized label names.
        """
        self.event_labels = event_labels
        self.standard_labels = standard_labels

    def get_standard_labels(self) -> dict:
        """
        :return: standard labels for the dataset as a dictionary.
        """
        return self.standard_labels.as_dict()
    
    def get_event_id(self) -> str:
        """
        :return: event ID column name from the dataset labels.
        """
        for k, v in self.event_labels.standardize_dict.items():
            if v == self.standard_labels.event_id:
                return k
        return self.standard_labels.event_id


class DatasetReader:
    """
    A class to read and analyze datasets.

    Properties:
        dataset_name: str, name of the dataset.

        input_path: str, path to the dataset file.

        default_filename: str, default filename to use if the input file is not found.

        labels: DatasetLabels, labels for the dataset.

        save_data: bool, if True, save the processed data to a file. Default True.

        save_path: str, path to save the processed data. Default current directory.
    """
    def __init__(self, dataset_name: str, input_path: str, dataset_labels: DatasetLabels,
                 save_path: str = os.getcwd()):
        """
        Initialize the DatasetReader with the path to the dataset.

        :param dataset_name: name of the dataset
        :param input_path: path to the dataset file
        :param dataset_labels: DatasetLabels instance containing labels for the dataset
        :param save_path: path to save the processed data. Default current directory.
        """
        self.dataset_name: str = dataset_name
        self.input_path: str = input_path
        self.labels: DatasetLabels = dataset_labels
        self.save_path: str = save_path

        self.data: pd.DataFrame = pd.DataFrame()
        self.load_data()

    def get_event_id_col(self) -> str:
        """
        :return: event ID column name as a string.
        """
        return self.labels.get_event_id()

    def load_data(self):
        """
        Load the dataset from the input_path.
        """
        if not os.path.exists(self.input_path) and not os.path.isfile(self.input_path):
            raise FileNotFoundError(f"WARNING: {self.dataset_name} dataset pickle file not found at: {self.input_path}")
        try:
            self.data = pd.read_pickle(self.input_path)
        except Exception as e:
            print(f"Error loading dataset from {self.input_path}: {e}")

    def filter_data(self, filter_id: str, filter_value: Any) -> pd.DataFrame:
        """
        Filter the dataset based on a specific column and value.

        :param filter_id: name of the column to filter by.
        :param filter_value: value to filter the column by.
        :return: filtered DataFrame or empty DataFrame if filter_id not found.
        """
        if filter_id not in self.data.columns:
            print(f"Filter ID '{filter_id}' not found in dataset columns.")
            return pd.DataFrame()  # Return empty DataFrame if filter_id not found
        return self.data[self.data[filter_id] == filter_value]

    def get_unique_event_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Unique event IDs and their counts in the dataset.
        """
        return np.unique(self.data[self.get_event_id_col()], return_counts=True)
    
    def standardize_df_columns(self) -> pd.DataFrame:
        """
        Standardize the columns of the data using a label mapping dictionary.

        :return: pandas DataFrame with standardized column names
        """
        for col in self.data.columns:
            if col in self.labels.event_labels.standardize_dict.keys():
                self.data.rename(columns={col: self.labels.event_labels.standardize_dict[col]}, inplace=True)
        return self.data

    def compile_metadata(self, index_column: str, metadata_columns: list) -> pd.DataFrame:
        """
        Compile metadata for a dataset.

        :param index_column: column name to use as the index for the metadata
        :param metadata_columns: list of column names to include in the metadata
        :return: DataFrame with event metadata
        """
        event_ids = self.data[index_column].unique()
        metadata_df = pd.DataFrame(index=event_ids, columns=metadata_columns)
        metadata_df[index_column] = event_ids
        for event in metadata_df.index:
            event_df = self.data[self.data[index_column] == event]
            for col in metadata_columns:
                if col in event_df.columns:
                    metadata_df.at[event, col] = event_df[col].iloc[0]
                else:
                    metadata_df.at[event, col] = np.nan
        return metadata_df
