"""
Generic dataset reader
Assumes the dataset can be read via pickle and is a pandas dataframe.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from quantum_inferno.cwt_atoms import cwt_chirp_from_sig
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_vert_example
from scipy.signal import welch


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
        self.dataset_name = dataset_name
        self.input_path = input_path
        self.default_filename = default_filename
        self.dataset_labels = dataset_labels
        self.show_info = show_info
        self.show_waveform_plots = show_waveform_plots
        self.show_frequency_plots = show_frequency_plots
        self.save_data = save_data
        self.save_path = save_path

        self.data = None
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

    def get_unique_event_ids(self) -> Tuple[np.array, np.array]:
        """
        :return: Unique event IDs and their counts in the dataset.
        """
        return np.unique(self.data[self.dataset_labels.event_id], return_counts=True)

    def show_info(self):
        """
        Display information about the dataset if show_info is True.
        # TODO: make generic?
        """
        event_ids, counts = np.unique(self.data[self.dataset_labels.event_id], return_counts=True)
        if self.show_info:
            for event_id, count in zip(event_ids, counts):
                event_df = self.data[self.data[self.dataset_labels.event_id] == event_id]
                eq_yield = event_df[self.dataset_labels.source_yield_kg][event_df.index[0]]
                print(f"\tEvent {event_id}: {eq_yield} kg TNT eq., {count} recordings")

    def plot_waveforms(self):
        """
        Plot waveforms for the dataset if show_waveform_plots is True.
        TODO: definitely functionalize
        """
        if self.show_waveform_plots:
            for event_id in self.data[self.dataset_labels.event_id].unique():
                event_df = self.data[self.data[self.dataset_labels.event_id] == event_id]
                plt.figure(figsize=(10, 6))
                plt.plot(event_df['time'], event_df['waveform'])
                plt.title(f"Waveform for Event ID: {event_id}")
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.grid()
                plt.show()

    def show_frequency_plots(self):
        """
        Show frequency plots for the dataset if show_frequency_plots is True.
        TODO: definitely functionalize
        """
        if self.show_frequency_plots:
            for event_id in self.data[self.dataset_labels.event_id].unique():
                event_df = self.data[self.data[self.dataset_labels.event_id] == event_id]
                f, pxx = welch(event_df['waveform'], fs=1/(event_df['time'][1] - event_df['time'][0]), nperseg=1024)
                plt.figure(figsize=(10, 6))
                plt.semilogy(f, pxx)
                plt.title(f"Frequency Spectrum for Event ID: {event_id}")
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power Spectral Density')
                plt.grid()
                plt.show()

    def save_data(self) -> str:
        """
        Save the processed data to the specified save path if save_data is True.

        :return: path to saved data file or empty string if not saved
        """
        if self.save_data:
            event_id_to_save = int(input("Enter the event ID number to save a subset of the SHAReD dataset: ").strip())
            # Check if the ID is in the dataset
            if event_id_to_save not in event_ids:
                print(f"Event ID '{event_id_to_save}' not found in dataset. Please check the available event IDs.")
                return ""
            # Create a subset of the DataFrame with only the recordings from the specified event
            subset_shared_ds = self.data[self.data[self.dataset_labels.event_id] == event_id_to_save]
            # Save the subset DataFrame to a new pickle file
            output_filename = f"SHAReD_event{event_id_to_save}.pkl"
            output_path = os.path.join(os.path.dirname(__file__), output_filename)
            subset_shared_ds.to_pickle(output_path)
            print(f"Subset of SHAReD containing event {event_id_to_save} data saved to: {output_path}")
            return output_path
        return ""
