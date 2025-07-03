"""
Tutorial on how to read and plot data from Aggregated Smartphone Timeseries of Rocket-generated Acoustics (ASTRA), an
open-access dataset.

ASTRA can be downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZKIS2K or
https://drive.google.com/drive/folders/1QKHUzdT0et6xpKXm8LFOmHJ9_gx0J2ib?usp=sharing (contact Milton Garcés
(milton@isla.hawaii.edu) or Sarah Popenhagen (spopen@hawaii.edu) for access to Google Drive folder).

The accompanying paper, Popenhagen and Garces (2025) can be found at: https://doi.org/10.3390/signals6010005

For details on the software used to collect and store the ASTRA audio data, refer to Garces, et al. (2020), which can
be found at: https://doi.org/10.3390/signals3020014
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_vert_example
from quantum_inferno.cwt_atoms import cwt_chirp_from_sig


# replace path below with the path to the downloaded ASTRA pickle file on your device
# if no file is found at the specified path, the script will use the subset of ASTRA included with this tutorial instead

TUTORIAL_PICKLE_FILE_NAME = "ASTRA_ART-1_1637620009.pkl"
PICKLE_FILE_NAME = "ASTRA.pkl"
DIRECTORY_PATH = "/Users/mgarces/Documents/DATA_2025/INFRA_SETS"
# DIRECTORY_PATH = "/Users/spopen/redvox/data/rockets_data/datasets_pkl"
PATH_TO_PKL = os.path.join(DIRECTORY_PATH, PICKLE_FILE_NAME)
PATH_TO_PKL = " "

if not os.path.isfile(PATH_TO_PKL):
    print(f"WARNING: ASTRA dataset pickle file not found at: '{PATH_TO_PKL}'")
    print("Using the subset of ASTRA included with this tutorial instead.\n")
    PATH_TO_PKL = os.path.join(os.path.dirname(__file__), TUTORIAL_PICKLE_FILE_NAME)

# a colorblind-friendly color cycle
CBF_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


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


def single_event_example_plots(astra_ds, launch_id, ds_labels, plot_tfrs=True):
    # If only the data from a specific launch is needed, a subset of the DataFrame can be created like this:
    launch_df = astra_ds[astra_ds[ds_labels.launch_id] == launch_id]
    # We'll be plotting the waveforms from the launch relative to the mission's reported launch time.
    rep_launch_epoch_s = launch_df[ds_labels.reported_launch_epoch_s][launch_df.index[0]]
    date_string = (datetime.fromtimestamp(rep_launch_epoch_s, tz=timezone.utc)).strftime("%d %B %Y")
    xlabel = f"Time (s) since launch"
    # For the title, we'll include some information on the launch included in the ASTRA dataset
    launch_n_srbs = launch_df[ds_labels.n_srbs][launch_df.index[0]]
    launch_rocket_type = launch_df[ds_labels.rocket_type][launch_df.index[0]]
    launch_rocket_model = launch_df[ds_labels.rocket_model_number][launch_df.index[0]]
    title = f"Normalized ASTRA audio data from launch {launch_id} on {date_string}"
    title += f"\nRocket: {launch_rocket_type}, {launch_rocket_model} configuration ({launch_n_srbs} SRBs)"
    # We'll also set some parameters for the figure
    fig, ax = plt.subplots(figsize=(10, 7))
    y_adj = 0
    y_adj_buff = 2.2
    t_max = 0
    ticks, tick_labels = [], []
    waveform_color, sa_toa_color, pa_toa_color = "k", CBF_COLOR_CYCLE[0], CBF_COLOR_CYCLE[1]
    # And sort the data by the estimated propagation distance
    launch_df = launch_df.sort_values(by=ds_labels.est_prop_dist_km)
    for station in launch_df.index:
        # We'll start by normalizing the audio data from each station
        audio_data = launch_df[ds_labels.audio_data][station]
        audio_data_abs_max = np.nanmax(np.abs(launch_df[ds_labels.audio_data][station]))
        audio_data = audio_data / audio_data_abs_max
        # The epoch time of the first sample of each recording is included in ASTRA
        start_time = launch_df[ds_labels.first_sample_epoch_s][station]
        # The sample rate of all the audio data in ASTRA is 800 Hz, but it is also included for convenience
        fs = launch_df[ds_labels.audio_fs][station]
        # With the sample rate, start time, and length of the audio data array, we can reconstruct the time array
        epoch_time = (np.array(range(len(audio_data))) / fs) + start_time
        # Epoch times are useful, but not very readable on a plot, so we'll convert the array to time since the launch
        relative_time = epoch_time - rep_launch_epoch_s
        # To speed up plot generation, we'll trim the signal to start at the reported launch time
        first_idx = np.argwhere(relative_time >= 0).flatten()[0]
        relative_time = relative_time[first_idx:]
        audio_data = audio_data[first_idx:]
        # We'll also keep track of the maximum time in the recording to set the x-axis limits later
        t_max = max(t_max, relative_time[-1])
        # The estimated propagation distance in kilometers is also included with each recording in ASTRA, along with the
        # ground truth latitudes and longitudes of the launch pad and the station
        est_prop_distance_km = launch_df[ds_labels.est_prop_dist_km][station]
        # We'll plot the normalized audio data from each station in order of their estimated propagation distances, with
        # the y-axis adjusted for each station
        ax.plot(relative_time, audio_data + y_adj, lw=1, color=waveform_color)
        ticks.append(y_adj)
        tick_labels.append(f"{round(est_prop_distance_km, 1)} km")
        # We'll also plot the estimated arrival times of the start and peak of the rocket launch signal as blue and
        # green lines, respectively. For detailed explanations of how these estimates were made, Popenhagen &
        # Garces, 2025 (link at the top of this file).
        relative_start_toa_estimate = launch_df[ds_labels.s_aligned_toa_est][station] - rep_launch_epoch_s
        relative_peak_toa_estimate = launch_df[ds_labels.p_aligned_toa_est][station] - rep_launch_epoch_s
        # We'll add labels to the first station's TOA estimate markers for clarity
        marker_lines_ylim = (y_adj - y_adj_buff / 2, y_adj + y_adj_buff / 2)
        marker_lines_zorder = 10
        if station == launch_df.index[0]:
            ax.vlines(
                ymin=marker_lines_ylim[0],
                ymax=marker_lines_ylim[1],
                x=relative_start_toa_estimate,
                color=sa_toa_color,
                zorder=marker_lines_zorder,
                label="Start-aligned TOA estimate",
                ls="-",
                lw=2,
            )
            ax.vlines(
                ymin=marker_lines_ylim[0],
                ymax=marker_lines_ylim[1],
                x=relative_peak_toa_estimate,
                color=pa_toa_color,
                zorder=marker_lines_zorder,
                label="Peak-aligned TOA estimate",
                ls="--",
                lw=2,
            )
        else:
            ax.vlines(
                ymin=marker_lines_ylim[0],
                ymax=marker_lines_ylim[1],
                x=[relative_start_toa_estimate, relative_peak_toa_estimate],
                color=[sa_toa_color, pa_toa_color],
                zorder=marker_lines_zorder,
                ls=["-", "--"],
                lw=2,
            )
        y_adj -= y_adj_buff
        # For each station, we can also plot the continuous wavelet transform (CWT) of the audio data using functions in
        # the quantum_inferno module
        if not plot_tfrs:
            continue
        tfr_title = f"CWT and waveform from launch {launch_id}"
        cwt, cwt_bits, time_s, frequency_cwt_hz = cwt_chirp_from_sig(
            sig_wf=audio_data,
            frequency_sample_rate_hz=fs,
            band_order_nth=3
        )
        _ = plot_wf_mesh_vert_example(
            station_id=f"{launch_df[ds_labels.station_id][station]} ({est_prop_distance_km:.1f} km)",
            wf_panel_a_sig=audio_data,
            wf_panel_a_time=relative_time,
            mesh_time=time_s,
            mesh_frequency=frequency_cwt_hz,
            mesh_panel_b_tfr=cwt_bits,
            figure_title=tfr_title,
        )
    # We'll add some finishing touches to the waveform plot settings
    fontsize = 12
    ax.set(xlabel=xlabel, xlim=(0, t_max), ylim=(min(ticks) - 1.1 * y_adj_buff / 2, max(ticks) + 1.1 * y_adj_buff / 2))
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_ticklabels(tick_labels)
    ax.tick_params(axis="y", labelsize="large")
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
    ax.legend(frameon=False, bbox_to_anchor=(.99, .99), loc='upper right', fontsize=fontsize)
    plt.subplots_adjust()


def main():
    # First, we'll read the ASTRA pickle file using pandas
    astra_ds = pd.read_pickle(PATH_TO_PKL)
    # We'll then initiate an instance of ASTRALabels, a class containing the column names in ASTRA
    ds_labels = ASTRALabels()
    # We can get some details about the dataset and print them out
    launch_ids, launch_counts = np.unique(astra_ds[ds_labels.launch_id], return_counts=True)
    n_signals = len(astra_ds)
    n_events = len(launch_ids)
    print(f"This dataset contains {n_signals} recording(s) from {n_events} unique launch event(s).")
    # We can print out the number of recordings for each launch event if we want
    answer = input("Would you like to see the number of recordings for each launch event? (y/n): ").strip().lower()
    if answer == 'y':
        for launch_id, count in zip(launch_ids, launch_counts):
            launch_df = astra_ds[astra_ds[ds_labels.launch_id] == launch_id]
            rocket_type = launch_df[ds_labels.rocket_type][launch_df.index[0]]
            launch_date = launch_df[ds_labels.reported_launch_epoch_s][launch_df.index[0]]
            date_string = (datetime.fromtimestamp(launch_date, tz=timezone.utc)).strftime("%d %b %Y")
            print(f"\t{rocket_type} launch {launch_id} on {date_string}: {count} recording(s)")

    # To plot ASTRA data, the time array must be reconstructed from the sample rate and epoch time of the first sample
    # As an example, we can plot the data from NASA's Artemis 1 launch.
    answer = input("\nWould you like to generate the ART-1 example plots? (y/n): ").strip().lower()
    if answer == "y":
        # The launch IDs in ASTRA are usually the flight number of the launch, but some launches, like Artemis 1, don't
        # have compact flight numbers, in which case the launch ID is an abbreviation of the launch's name. For
        # Artemis 1, the launch ID is "ART-1"
        artemis_launch_id = "ART-1"
        # We can pass this launch id to the plotting function defined above to generate time and time-frequency plots
        # for the Artemis 1 data in ASTRA.
        print(f"Plotting all available ASTRA audio data from: {artemis_launch_id}")
        single_event_example_plots(astra_ds, artemis_launch_id, ds_labels)
        # We'll show the figures
        print("Close all figures to proceed with the tutorial.")
        plt.show()
        plt.close()
    # To generate the waveform plots for all the launches in ASTRA, respond to the prompt below with 'y'
    answer1 = input("\nWould you like to plot data from all available ASTRA events? (y/n): ").strip().lower()
    if answer1 == "y":
        # To generate the time-frequency representations (TFRs) for all launches, respond to the prompt below with 'y'
        # (this will take longer to run and generate a lot of figures)
        answer2 = input("Would you like to plot the time-frequency representations? (y/n): ").strip().lower()
        tfr_plot = False
        if answer2 == "y":
            tfr_plot = True
        for launch_id in launch_ids:
            print(f"Plotting all available ASTRA audio data from: {launch_id}")
            single_event_example_plots(astra_ds, launch_id, ds_labels, tfr_plot)
            # We'll show the figures, and the tutorial will proceed after all figures are closed
            print("Close all figures to proceed with the tutorial.")
            plt.show()
            plt.close()
        print("Done.")
    # This concludes the plotting section of this tutorial. The following section will demonstrate how to save a subset
    # of the dataset containing only data from a single launch event to a new pickle file. This can be useful in some
    # cases as the full dataset is quite large and may not be needed for all applications.
    answer = input("\nWould you like to save a subset of the ASTRA dataset? (y/n): ").strip().lower()
    if answer == "y":
        # Enter a launch ID to save a subset of the ASTRA dataset
        launch_id_to_save = input("Enter the ID string of the launch you'd like to save: ").strip()
        # Check if the launch ID is in the dataset
        if launch_id_to_save not in launch_ids:
            print(f"Launch ID '{launch_id_to_save}' not found in ASTRA dataset. Please check the available launch IDs.")
        else:
            # Create a subset of the DataFrame with only the recordings from the specified launch
            subset_astra_ds = astra_ds[astra_ds[ds_labels.launch_id] == launch_id_to_save]
            # Print some details about the subset
            n_signals = len(subset_astra_ds)
            rocket_type = subset_astra_ds[ds_labels.rocket_type][subset_astra_ds.index[0]]
            launch_date = subset_astra_ds[ds_labels.reported_launch_epoch_s][subset_astra_ds.index[0]]
            date_string = (datetime.fromtimestamp(launch_date, tz=timezone.utc)).strftime("%d %b %Y")
            print(f"\tSelected: {rocket_type} launch {launch_id_to_save} on {date_string}: {n_signals} recording(s)")
            answer = input(f"Would you like to save all recordings from this launch? (y/n): ").strip().lower()
            if answer == "y":
                # Save the subset DataFrame to a new pickle file
                output_filename = f"ASTRA_{launch_id_to_save}_{n_signals}.pkl"
                output_path = os.path.join(os.path.dirname(__file__), output_filename)
                print(f"Saving subset of ASTRA containing all data from launch {launch_id_to_save} to: {output_path}")
            else:
                print("Available signals from the selected launch:")
                for station in subset_astra_ds.index:
                    station_id = subset_astra_ds[ds_labels.station_id][station]
                    dist_km = subset_astra_ds[ds_labels.est_prop_dist_km][station]
                    print(f"\tStation ID: {station_id} ({dist_km:.1f} km from launch pad)")
                answer = input("Enter the station ID of the recording you want to save: ").strip()
                # Check if the station ID is in the subset DataFrame
                if answer not in subset_astra_ds[ds_labels.station_id].values:
                    print(f"Station ID '{answer}' not found in the selected launch subset.")
                    output_path = None
                else:
                    output_filename = f"ASTRA_{launch_id_to_save}_{answer}.pkl"
                    subset_astra_ds = subset_astra_ds[subset_astra_ds[ds_labels.station_id] == answer]
                    output_path = os.path.join(os.path.dirname(__file__), output_filename)
                    print(f"Saving the station {answer} data from launch {launch_id_to_save} to: {output_path}")
            if output_path is not None:
                subset_astra_ds.to_pickle(output_path)
                print("Dataset saved.")
            else:
                print("Requested data not found. No file saved.")
    print("\nThis concludes the ASTRA dataset tutorial.")


if __name__ == "__main__":
    main()
