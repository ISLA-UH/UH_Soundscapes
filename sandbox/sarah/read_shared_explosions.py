"""
Tutorial on how to read and plot data from the Smartphone High-explosive Audio Recordings Dataset (SHAReD), an
open-access dataset.

SHAReD can be downloaded from Harvard Dataverse at https://doi.org/10.7910/DVN/ROWODP or Google Drive at
https://drive.google.com/drive/folders/1QKHUzdT0et6xpKXm8LFOmHJ9_gx0J2ib?usp=sharing (contact Milton Garcés
(milton@isla.hawaii.edu) or Sarah Popenhagen (spopen@hawaii.edu) for access to Google Drive folder).

The accompanying paper, Takazawa et al. (2024) can be found at: https://doi.org/10.3390/s24206688

For details on the software used to collect and store the SHAReD data, refer to Garces, et al. (2020), which can
be found at: https://doi.org/10.3390/signals3020014
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_vert_example
from quantum_inferno.cwt_atoms import cwt_chirp_from_sig

# replace path below with the path to the downloaded SHAReD pickle file on your device
TUTORIAL_PICKLE_FILE_NAME = "SHAReD_event26.pkl"
PICKLE_FILE_NAME = "SHAReD.pkl"
DIRECTORY_PATH = "/Users/mgarces/Documents/DATA_2025/INFRA_SETS"
# DIRECTORY_PATH = "/Users/spopen/redvox/data/rockets_data/datasets_pkl"
PATH_TO_PKL = os.path.join(DIRECTORY_PATH, PICKLE_FILE_NAME)
PATH_TO_PKL = " "

# if no file is found at the specified path, a subset of SHAReD included with this tutorial will be used instead
if not os.path.isfile(PATH_TO_PKL):
    print(f"WARNING: SHAReD dataset pickle file not found at: {PATH_TO_PKL}")
    print("Using the subset of SHAReD included with this tutorial instead.")
    PATH_TO_PKL = os.path.join(os.path.dirname(__file__), TUTORIAL_PICKLE_FILE_NAME)


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


def max_30hz_db(f: np.ndarray, power: np.ndarray):
    loc_hz = np.nanargmin(np.abs(f - 30.0))
    decibel = 10 * np.log10(power / power[loc_hz])
    return decibel


def demean_norm(signal: np.ndarray) -> np.ndarray:
    signal = signal - np.nanmean(signal)
    return signal / np.nanmax(np.abs(signal))


def main():
    # load the dataset
    shared_ds = pd.read_pickle(PATH_TO_PKL)
    # initiate an instance of SHAReDLabels, a class containing the column names in SHAReD
    ds_labels = SHAReDLabels()
    # get some details about the dataset and print them out
    n_signals = len(shared_ds)
    event_ids, counts = np.unique(shared_ds[ds_labels.event_id_number], return_counts=True)
    n_events = len(event_ids)
    print(f"This dataset contains {n_signals} recording(s) from {n_events} unique high explosive event(s).\n")
    print(f"Each of the {n_signals} rows in the pandas DataFrame contains all the data collected by one smartphone")
    print(f"during one event, accompanied by a sample of ambient data from the smartphone's sensors, external location")
    print(f"information, and ground truth data about the smartphone and the explosion. Available fields are listed")
    print(f"in the SHAReDLabels class documentation.\n")
    print(f"This dataset contains {n_signals} recordings from {n_events} unique events.\n")
    # We can print out the number of recordings for each event if we want
    answer = input("Would you like to see the number of recordings for each event? (y/n): ").strip().lower()
    if answer == 'y':
        for event_id, count in zip(event_ids, counts):
            event_df = shared_ds[shared_ds[ds_labels.event_id_number] == event_id]
            eq_yield = event_df[ds_labels.source_yield_kg][event_df.index[0]]
            print(f"\tEvent {event_id}: {eq_yield} kg TNT eq., {count} recording(s)")
    # We can pick an example event and plot all the "explosion" and "ambient" data to get a feel for the dataset. If you
    # would like to generate the plots, respond with 'y' to the prompt below.
    fontsize = 12  # font size for all potential plots
    example_event_id = 26
    answer = input("\nWould you like to generate plots from an example event? (y/n): ").strip().lower()
    if answer == 'y':
        # For our example, we'll look at a single event in SHAReD
        # To select a subset of SHAReD containing only data from this event, we can use the following line of code:
        sample_df = shared_ds[shared_ds[ds_labels.event_id_number] == example_event_id]
        # Once selected, we'll sort the subset by distance from the explosion to make it easier to visualize
        sample_df = sample_df.sort_values(by=ds_labels.distance_from_explosion_m)
        # Events in SHAReD were recorded in collaboration with either Idaho National Laboratory (INL) or Nevada National
        # Security Site (NNSS). The 'event_name' field in SHAReD contains ID strings starting with either "INL" (for
        # events recorded in collaboration with INL) or "NNSS" (for events recorded in collaboration with NNSS). INL
        # events contain information on the TNT equivalent source yield of the explosion, while NNSS events do not.
        # All NNSS events share the same event name ("NNSS"), but can be differentiated by their unique event ID numbers
        event_name = sample_df[ds_labels.event_name][sample_df.index[0]]
        print(f"\nExample event: {event_name}, ID number: {example_event_id}")
        source_yield = sample_df[ds_labels.source_yield_kg][sample_df.index[0]]
        if source_yield is None or np.isnan(source_yield):
            title_header = f"SHAReD event {event_name} (source yield not included)"
        else:
            title_header = f"SHAReD event {event_name} ({source_yield} kg TNT eq.)"
        # We'll plot all the audio waveforms from the event together to visualize the propagation of the explosion
        # signal, with their distance from the explosion site indicated on the y-axis. We'll also plot the multimodal
        # data from each station individually for both the "explosion" and "ambient" segments of data.
        fig, ax = plt.subplots(figsize=(10, 7))
        y_adj = 0
        y_adj_buff = 2.2
        ticks, tick_labels = [], []
        t000 = sample_df[ds_labels.explosion_detonation_time][sample_df.index[0]]
        for sample_idx in sample_df.index:
            t00 = shared_ds[ds_labels.microphone_time_s][sample_idx][0] - t000
            dt0 = shared_ds[ds_labels.microphone_time_s][sample_idx][-1] - t000
            dist_m = sample_df[ds_labels.distance_from_explosion_m][sample_idx]
            audio_data = demean_norm(shared_ds[ds_labels.microphone_data][sample_idx])
            ax.plot(shared_ds[ds_labels.microphone_time_s][sample_idx] - t000,
                    audio_data + y_adj,
                    lw=1, color="k")
            ticks.append(y_adj)
            tick_labels.append(f"{round(sample_df[ds_labels.distance_from_explosion_m][sample_idx])}m")
            y_adj -= y_adj_buff  # adjust y position for next plot
            title_line2 = f"\nDistance from source: {int(dist_m)} m, "
            title_line2 += f"scaled distance: {sample_df[ds_labels.scaled_distance][sample_idx]:.2f} m/kg^(1/3)"
            fig2, ax2 = plt.subplots(3, 2, figsize=(10, 7), sharex='col', sharey=True)
            fig2.suptitle(f"Normalized signals from {title_header}{title_line2}", fontsize=fontsize + 2)
            ax2[0, 1].plot(shared_ds[ds_labels.microphone_time_s][sample_idx] - t000,
                           audio_data,
                           lw=1, color="k")
            ax2[1, 1].plot(shared_ds[ds_labels.barometer_time_s][sample_idx] - t000,
                           demean_norm(shared_ds[ds_labels.barometer_data][sample_idx]),
                           lw=1, color="k")
            ax2[2, 1].plot(shared_ds[ds_labels.accelerometer_time_s][sample_idx] - t000,
                           demean_norm(shared_ds[ds_labels.accelerometer_data_x][sample_idx]),
                           lw=1, label="x-axis")
            ax2[2, 1].plot(shared_ds[ds_labels.accelerometer_time_s][sample_idx] - t000,
                           demean_norm(shared_ds[ds_labels.accelerometer_data_y][sample_idx]),
                           lw=1, label="y-axis")
            ax2[2, 1].plot(shared_ds[ds_labels.accelerometer_time_s][sample_idx] - t000,
                           demean_norm(shared_ds[ds_labels.accelerometer_data_z][sample_idx]),
                           lw=1, label="z-axis")
            ax2[2, 1].legend()
            ax2[0, 1].set(xlim=(t00, dt0), ylim=(-1.1, 1.1))
            ax2[0, 1].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            ax2[0, 1].set_title(f"Explosion microphone", fontsize=fontsize)
            ax2[0, 1].set_ylabel("Norm", fontsize=fontsize)
            ax2[1, 1].set_title("Explosion barometer", fontsize=fontsize)
            ax2[1, 1].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            ax2[1, 1].set_ylabel("Norm", fontsize=fontsize)
            ax2[2, 1].set_title("Explosion accelerometer", fontsize=fontsize)
            ax2[2, 1].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            ax2[2, 1].tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
            ax2[2, 1].set_ylabel("Norm", fontsize=fontsize)
            ax2[2, 1].set_xlabel("Time (s) since event", fontsize=fontsize)

            t10 = shared_ds[ds_labels.ambient_microphone_time_s][sample_idx][0]
            dt1 = shared_ds[ds_labels.ambient_microphone_time_s][sample_idx][-1] - t10
            ax2[0, 0].plot(shared_ds[ds_labels.ambient_microphone_time_s][sample_idx] - t10,
                           demean_norm(shared_ds[ds_labels.ambient_microphone_data][sample_idx]),
                           lw=1, color="k")
            ax2[1, 0].plot(shared_ds[ds_labels.ambient_barometer_time_s][sample_idx] - t10,
                           demean_norm(shared_ds[ds_labels.ambient_barometer_data][sample_idx]),
                           lw=1, color="k")
            ax2[2, 0].plot(shared_ds[ds_labels.ambient_accelerometer_time_s][sample_idx] - t10,
                           demean_norm(shared_ds[ds_labels.ambient_accelerometer_data_x][sample_idx]),
                           lw=1, label="x-axis")
            ax2[2, 0].plot(shared_ds[ds_labels.ambient_accelerometer_time_s][sample_idx] - t10,
                           demean_norm(shared_ds[ds_labels.ambient_accelerometer_data_y][sample_idx]),
                           lw=1, label="y-axis")
            ax2[2, 0].plot(shared_ds[ds_labels.ambient_accelerometer_time_s][sample_idx] - t10,
                           demean_norm(shared_ds[ds_labels.ambient_accelerometer_data_z][sample_idx]),
                           lw=1, label="z-axis")
            # ax2[2, 0].legend()
            ax2[0, 0].set(xlim=(0, dt1), ylim=(-1.1, 1.1))
            ax2[0, 0].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            ax2[0, 0].set_title(f"Ambient microphone", fontsize=fontsize)
            ax2[0, 0].set_ylabel("Norm", fontsize=fontsize)
            ax2[1, 0].set_title("Ambient barometer", fontsize=fontsize)
            ax2[1, 0].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            ax2[1, 0].set_ylabel("Norm", fontsize=fontsize)
            ax2[2, 0].set_title("Ambient accelerometer", fontsize=fontsize)
            ax2[2, 0].tick_params(axis="y", labelsize="large", left=True, labelleft=True)
            ax2[2, 0].tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
            ax2[2, 0].set_ylabel("Norm", fontsize=fontsize)
            ax2[2, 0].set_xlabel("Time (s)", fontsize=fontsize)

            plt.subplots_adjust(hspace=0.3)
        # We'll add some finishing touches to the audio waveform plot settings
        ax.set(ylim=(min(ticks) - 1.1 * y_adj_buff / 2, max(ticks) + 1.1 * y_adj_buff / 2))
        ax.set_title(f"Normalized explosion audio data from {title_header}", fontsize=fontsize + 2)
        ax.set_xlabel("Time (s) since event", fontsize=fontsize)
        ax.yaxis.set_ticks(ticks)
        ax.yaxis.set_ticklabels(tick_labels)
        ax.tick_params(axis="y", labelsize="large")
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
        plt.subplots_adjust()
        # Then generate the plots
        print(f"Plotting all data from selected event.")
        print("Close all figures to proceed with the tutorial.\n")
        plt.show()
        plt.close()
    # We can also take a look at the frequency and time-frequency characteristics of the data using functions in the
    # scipy and quantum_inferno modules
    answer = input("Would you like to generate example frequency and time-frequency plots? (y/n): ").strip().lower()
    if answer == 'y':
        # As an example, we'll look at the audio data from the closest recording to the explosion in our example event
        sample_df = shared_ds[shared_ds[ds_labels.event_id_number] == example_event_id]
        sample_df = sample_df.sort_values(by=ds_labels.distance_from_explosion_m)
        event_name = sample_df[ds_labels.event_name][sample_df.index[0]]
        source_yield = sample_df[ds_labels.source_yield_kg][sample_df.index[0]]
        dist_m = sample_df[ds_labels.distance_from_explosion_m][sample_df.index[0]]
        sample_fs = shared_ds[ds_labels.microphone_sample_rate_hz][sample_df.index[0]]
        audio_data = shared_ds[ds_labels.microphone_data][sample_df.index[0]]
        audio_data = audio_data / np.max(np.abs(audio_data))
        time = shared_ds[ds_labels.microphone_time_s][sample_df.index[0]]
        t0 = time[0]
        relative_time = time - t0
        nperseg = sample_fs * 0.48  # 0.48 seconds per segment
        # Calculate the PSD using scipy.signal.welch()
        f, psd = welch(audio_data, sample_fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
        # Convert the PSD to decibels relative to 30 Hz
        psd_dec = max_30hz_db(f, psd)
        # Plot the Welch PSD against log-scaled frequency
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(f, psd_dec, lw=1, color="k")
        ax.set(xscale="log", xlim=(1, sample_fs / 2))
        ax.tick_params(axis="y", labelsize="large")
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
        ax.set_title(f"Welch PSD of explosion audio from {event_name} at {int(dist_m)} m",
                     fontsize=fontsize)
        ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
        ax.set_ylabel("Decibels (dB) relative to 30 Hz", fontsize=fontsize)
        # Now, we'll calculate the continuous wavelet transform (CWT) of the waveform using the quantum_inferno module
        # and plot the results using the quantum_inferno.plot_templates.plot_wf_mesh_vert_example() function.
        tfr_title = f"CWT and waveform for {event_name} ({source_yield} kg TNT eq.)"
        cwt, cwt_bits, time_s, frequency_cwt_hz = cwt_chirp_from_sig(
            sig_wf=audio_data,
            frequency_sample_rate_hz=sample_fs,
            band_order_nth=3,
            cwt_type="conv"
        )
        _ = plot_wf_mesh_vert_example(
            station_id=f"{sample_df[ds_labels.smartphone_id][sample_df.index[0]]} ({int(dist_m)} m)",
            wf_panel_a_sig=audio_data,
            wf_panel_a_time=relative_time,
            mesh_time=time_s,
            mesh_frequency=frequency_cwt_hz,
            mesh_panel_b_tfr=cwt_bits,
            figure_title=tfr_title,
        )
        plt.subplots_adjust()
        print(f"\nPlotting frequency and time-frequency representations from {event_name} at {int(dist_m)} m")
        print("Close all figures to proceed with the tutorial.")
        plt.show()
    # This concludes the plotting section of this tutorial. The following section will demonstrate how to save a subset
    # of the dataset containing only data from a single event to a new pickle file. This can be useful in some
    # cases as the full dataset is quite large and may not be needed for all applications.
    answer = input("\nWould you like to save a subset of the SHAReD dataset? (y/n): ").strip().lower()
    if answer == "y":
        # Enter an event ID number to save a subset of the SHAReD dataset
        event_id_to_save = input("Enter the ID number of the event you'd like to save: ").strip()
        # Check if the input is a valid integer
        try:
            event_id_to_save = int(event_id_to_save)
            # Check if the ID is in the dataset
            if event_id_to_save not in event_ids:
                print(f"Event ID '{event_id_to_save}' not found in dataset. Please check the available event IDs.")
            else:
                # Create a subset of the DataFrame with only the recordings from the specified event
                subset_shared_ds = shared_ds[shared_ds[ds_labels.event_id_number] == event_id_to_save]
                # Save the subset DataFrame to a new pickle file
                output_filename = f"SHAReD_event{event_id_to_save}.pkl"
                output_path = os.path.join(os.path.dirname(__file__), output_filename)
                subset_shared_ds.to_pickle(output_path)
                print(f"Subset of SHAReD containing event {event_id_to_save} data saved to: {output_path}")
        except ValueError:
            print("Invalid input. Please enter a valid integer for the event ID number.")

    print("\nThis concludes the SHAReD dataset tutorial.")


if __name__ == "__main__":
    main()
