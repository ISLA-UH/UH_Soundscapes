"""
Generic dataset reader
Assumes the dataset can be read via pickle and is a pandas dataframe.
It's very likely this is gonna be an inherited class based on specific events and their needs
"""
from typing import Any, Tuple

import numpy as np
import os
import pandas as pd


class DatasetLabels:
    """
    Base class for dataset labels.  Inherited classes should implement specific labels.

    Properties:
        event_id: str, name of the event ID column in the dataset.
    """
    def __init__(self, event_id: str):
        """
        Initialize the dataset labels.  Inherited class will implement specific labels.

        :param event_id: str, name of the event ID column in the dataset.
        """
        self.event_id = event_id
        # sample rate

    def get_labels(self) -> dict:
        """
        Get the labels for the dataset.

        :return: labels for the dataset as a dictionary.
        """
        return {
            "event_id": self.event_id
            # sample rate
        }


class DatasetReader:
    """
    A class to read and analyze datasets.

    Properties:
        dataset_name: str, name of the dataset.

        input_path: str, path to the dataset file.

        default_filename: str, default filename to use if the input file is not found.

        dataset_labels: DatasetLabels, labels for the dataset.

        show_info: bool, if True, display dataset information.  Default True.

        show_waveform_plots: bool, if True, display waveform plots. Default True.

        show_frequency_plots: bool, if True, display frequency plots. Default True.

        save_data: bool, if True, save the processed data to a file. Default True.

        save_path: str, path to save the processed data. Default current directory.
    """
    def __init__(self, dataset_name: str, input_path: str, default_filename: str, dataset_labels: DatasetLabels,
                 show_info: bool = True, show_waveform_plots: bool = True, show_frequency_plots: bool = True,
                 save_data: bool = True, save_path: str = os.getcwd()):
        """
        Initialize the DatasetReader with the path to the dataset.

        :param dataset_name: name of the dataset
        :param input_path: path to the dataset file
        :param dataset_labels: DatasetLabels instance containing labels for the dataset
        :param default_filename: default filename to use if the input file is not found
        :param show_info: if True, display dataset information. Default True.
        :param show_waveform_plots: if True, display waveform plots. Default True.
        :param show_frequency_plots: if True, display frequency plots. Default True.
        :param save_data: if True, save the processed data to a file. Default True.
        :param save_path: path to save the processed data. Default current directory.
        """
        self.dataset_name: str = dataset_name
        self.input_path: str = input_path
        self.default_filename: str = default_filename
        self.dataset_labels: DatasetLabels = dataset_labels
        self.show_info: bool = show_info
        self.show_waveform_plots: bool = show_waveform_plots
        self.show_frequency_plots: bool = show_frequency_plots
        self.save_data: bool = save_data
        self.save_path: str = save_path

        self.data: pd.DataFrame = pd.DataFrame()
        self.load_data()

    def load_data(self):
        """
        Load the dataset from the input_path.
        """
        if not os.path.exists(self.input_path) and not os.path.isfile(self.input_path):
            print(f"WARNING: {self.dataset_name} dataset pickle file not found at: {self.input_path}")
            print(f"Using the subset of {self.dataset_name} included with this tutorial instead.")
            self.input_path = os.path.join(os.path.dirname(__file__), self.default_filename)
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
        return np.unique(self.data[self.dataset_labels.event_id], return_counts=True)

    def show_base_info(self):
        """
        Display information about the dataset if show_info is True.
        # TODO: looks like this will be inherited due to specific needs
        """
        pass

    def plot_frequency_waveforms(self):
        """
        Show frequency plots for the dataset if show_frequency_plots is True.
        TODO: you guessed it, inheritance
        """
        pass

    def save_event(self) -> str:
        """
        Save the processed data to the specified save path if save_data is True.
        # TODO: OH BOY, INHERITANCE AGAIN

        :return: path to saved data file or empty string if not saved
        """
        pass
        # if self.save_data:
        #     event_id_to_save = int(input("Enter the event ID number to save a subset of the SHAReD dataset: ").strip())
        #     all_ids = self.get_unique_event_ids()[0]
        #     # Check if the ID is in the dataset
        #     if event_id_to_save not in all_ids:
        #         print(f"Event ID '{event_id_to_save}' not found in dataset. Avaiable IDs are: {all_ids}")
        #         return ""
        #     # Create a subset of the DataFrame with only the recordings from the specified event
        #     subset_shared_ds = self.data[self.data[self.dataset_labels.event_id] == event_id_to_save]
        #     # Save the subset DataFrame to a new pickle file
        #     output_filename = f"SHAReD_event{event_id_to_save}.pkl"
        #     output_path = os.path.join(os.path.dirname(__file__), output_filename)
        #     subset_shared_ds.to_pickle(output_path)
        #     print(f"Subset of SHAReD containing event {event_id_to_save} data saved to: {output_path}")
        #     return output_path
        return ""
