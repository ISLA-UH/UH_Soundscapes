"""
Build data window, print waveform and stx
Test gappy data scenario
"""
# Python libraries
import os
from typing import Tuple
from datetime import datetime, UTC

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# QI modules
from quantum_inferno import styx_cwt, scales_dyadic
import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_mesh_wf_vert

# RedVox modules
import redvox.common.date_time_utils as dt
from redvox.common.data_window import DataWindowConfig, DataWindow

# # Configuration file: MAWAR@ISLA
# from qi0_config_mawar import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
#     WINDOW_DURATION_S, HOP_S, STATIONS, OUTPUT_DIR, MICROS_TO_S

# Configuration file: GUAM@CSON
from qi0_config_guam import EVENT_NAME, INPUT_DIR, EPISODE_START_EPOCH_S, \
    WINDOW_DURATION_S, HOP_S, STATIONS, OUTPUT_DIR, MICROS_TO_S

FIG_OUT_DIR = os.path.join(OUTPUT_DIR, "img")
if not os.path.exists(FIG_OUT_DIR):
    os.makedirs(FIG_OUT_DIR, exist_ok=False)

def cw_pipeline(
        band_order_nth: float,
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        frequency_min_hz: float = 1.,
        frequency_max_hz = None,
        dictionary_type: str = "norm"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CWT

    :param band_order_nth: Nth order of constant Q bands
    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate in Hz
    :param cwt_type: one of "fft", or "morlet2". Default is "fft"
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
    :return: frequency_cwt_hz, time_cwt_s, cwt
    """
    if frequency_max_hz == None:
        frequency_max_hz = frequency_sample_rate_hz / 2.

    wavelet_points = len(sig_wf)
    time_cwt_s = np.arange(wavelet_points) / frequency_sample_rate_hz

    frequency_cwt_all_hz = scales_dyadic.log_frequency_hz_from_fft_points(
        frequency_sample_hz=frequency_sample_rate_hz,
        fft_points=wavelet_points,
        scale_order=band_order_nth,
        scale_base=scales_dyadic.Slice.G2)

    frequency_idx = np.where((frequency_cwt_all_hz >= frequency_min_hz) &
                             (frequency_cwt_all_hz <= frequency_max_hz))
    frequency_hz = frequency_cwt_all_hz[frequency_idx]

    cw_complex, _, _, _, amp = \
        styx_cwt.wavelet_centered_4cwt(band_order_nth=band_order_nth,
                              duration_points=wavelet_points,
                              scale_frequency_center_hz=frequency_hz,
                              frequency_sample_rate_hz=frequency_sample_rate_hz,
                              dictionary_type=dictionary_type)

    cwt = signal.fftconvolve(np.tile(sig_wf, (len(frequency_hz), 1)),
                                 np.conj(np.fliplr(cw_complex)), mode='same', axes=-1)
    cwt_power = np.abs(cwt)**2

    return frequency_hz, time_cwt_s, cwt_power


if __name__ == "__main__":
    """
    Guam draft pipeline with cwt for prop/tonal signals
    Temporary solution while station model framework is matured
    """
    # Additional options
    load_datawindow_from_file: bool = False
    save_datawindow: bool = False
    show_stx_fig: bool = False
    save_data: bool = False
    save_fig: bool = True

    # Constants and other important values
    order_number: int = 3
    frequency_sample_rate_hz: float = 800.
    frequency_highest_hz: float = 320.
    frequency_lowest_hz: float = 1.
    cyber_averaging_scale: float = frequency_sample_rate_hz / frequency_lowest_hz
    cyber_averaging_frame: float = .75 * np.pi * order_number * cyber_averaging_scale
    cyber_averaging_frame_pow2: int = int(2**np.ceil(np.log2(cyber_averaging_frame)))
    averaging_frame_s = cyber_averaging_frame_pow2 / frequency_sample_rate_hz

    DURATION_POW2_POINTS = int(2 ** np.ceil(np.log2(WINDOW_DURATION_S * frequency_sample_rate_hz)))
    DURATION_POW2_S = DURATION_POW2_POINTS / frequency_sample_rate_hz

    # TODO: Compute segments based on total span of data available
    # From window duration and batch duration
    start_segment = 701  # Where it stopped last; 0 to start from scratch - 701 is next break
    number_of_hours = 24
    end_segment = int(number_of_hours*60)  # 60 minute windows, 1440 total
    # start_segment = 170
    # end_segment = 171
    segment_counters = np.arange(start_segment, end_segment)
    number_of_segments = len(segment_counters)
    print("Staging Pipeline, v0.1")
    print(f"Min frame duration, s: {averaging_frame_s}")

    # As long as data window length is above min frame duration, all's well
    # We set the minimum frame duration and calculate the window start and end
    print(f"Requested duration, s: {WINDOW_DURATION_S}")
    print(f"Modified power of 2 duration, s: {DURATION_POW2_S}")
    print(f"Modified power of 2 duration, points: {DURATION_POW2_POINTS}")

    # Use a longer duration to batch process.  repeat below for each segment in the longer batch

    for time_index, segment_counter in enumerate(segment_counters):
        print(f"\nSegment counter: {segment_counter}")
        SEGMENT_START_EPOCH_S = EPISODE_START_EPOCH_S + segment_counter * HOP_S
        SEGMENT_END_EPOCH_S = SEGMENT_START_EPOCH_S + DURATION_POW2_S

        SEGMENT_START_DATETIME = dt.datetime_from_epoch_seconds_utc(int(SEGMENT_START_EPOCH_S))
        SEGMENT_END_DATETIME = dt.datetime_from_epoch_seconds_utc(int(SEGMENT_END_EPOCH_S))
        datetime_object_human = (
            datetime.utcfromtimestamp(int(SEGMENT_START_EPOCH_S)))
        SEGMENT_START_DATETIME_MSEED = datetime_object_human.strftime("%Y%m%dT%H%M%SZ")
        SEGMENT_NAME = EVENT_NAME + " " + str(SEGMENT_START_DATETIME_MSEED) + "_dur_" + str(WINDOW_DURATION_S) + 's'

        print(f"Event Name: {SEGMENT_NAME}")
        print(f"Start: {SEGMENT_START_DATETIME}")
        print(f"Start, MSEED: {SEGMENT_START_DATETIME_MSEED}")
        print(f"Stop: {SEGMENT_END_DATETIME}")
        print(datetime_object_human)

        if load_datawindow_from_file:
            # Load data window from report
            dw_path = os.path.join(OUTPUT_DIR, "dw_lz4", SEGMENT_NAME + ".pkl.lz4")
            print(f"DataWindow file path: {dw_path}")
            dw = DataWindow.deserialize(dw_path)

        else:
            dw_config = DataWindowConfig(
                input_dir=INPUT_DIR,
                start_datetime=SEGMENT_START_DATETIME,
                end_datetime=SEGMENT_END_DATETIME,
                station_ids=STATIONS)

            dw: DataWindow = DataWindow(event_name=SEGMENT_NAME, config=dw_config, debug=False,
                                        output_dir=os.path.join(OUTPUT_DIR, "dw_lz4"), out_type="LZ4")
            if save_datawindow:
                dw.save()

        # Loop over selected stations
        for station in dw.stations():
            # Station ID
            station_id_str: str = station.id()
            print(f"*** Station ID: {station_id_str} ***")
            # Get audio data
            sig_wf_sample_rate_hz = station.audio_sample_rate_nominal_hz()
            sig_wf_raw = station.audio_sensor().get_microphone_data()
            sig_epoch_start_micros = station.audio_sensor().first_data_timestamp()
            sig_epoch_micros = station.audio_sensor().data_timestamps()
            # Get Location Data
            # Continue if lat/lon not available, fill with nan or 0
            try:
                longitude_data = station.location_sensor().get_longitude_data()
                latitude_data = station.location_sensor().get_latitude_data()
                loc_time = station.location_sensor().get_gps_timestamps_data()
            except:
                print(f"Station {station_id_str} has no location data")
                longitude_data = np.nan
                latitude_data = np.nan
                loc_time = np.zeros(1)

            loc_time_clean = loc_time[~np.isnan(loc_time)]
            # Use SDK time conversion methods to convert to seconds
            sig_epoch_s = sig_epoch_micros * MICROS_TO_S
            sig_epoch_start_s = sig_epoch_start_micros * MICROS_TO_S
            sig_time_s = sig_epoch_s - sig_epoch_start_s

            # START SPECTRAL STX STAGING
            fft_duration_ave_window_points_pow2 = DURATION_POW2_POINTS

            # Remove non-nan mean
            sig_wf_raw2 = sig_wf_raw - np.nanmean(sig_wf_raw)

            # TODO: Computing for first 2^n points, extend to all points
            # Convert possible nans to zeros
            # sig_wf = np.nan_to_num(sig_wf_raw2[0:fft_duration_ave_window_points_pow2])
            sig_wf = np.nan_to_num(sig_wf_raw2)
            # sig_time_s = sig_time_s[0:fft_duration_ave_window_points_pow2]
            print(f"Signal number of points: {len(sig_wf)}")

            sig_power = np.abs(sig_wf)**2
            # sig_power_mean_array[time_index] = np.var(sig_wf)
            # sig_power_sum_array[time_index] = np.sum(sig_power)
            # sig_power_median_array[time_index] = np.median(sig_power)
            print(f"CWT of order: {order_number}")

            frequency_cwt_hz, time_cwt_s, cwt_power = cw_pipeline(
                band_order_nth = order_number,
                sig_wf = sig_wf,
                frequency_sample_rate_hz= frequency_sample_rate_hz,
                frequency_min_hz = frequency_lowest_hz,
                frequency_max_hz = frequency_highest_hz)

            cwt_power_per_window_per_band = np.sum(cwt_power, axis=1)

            cwt_pdf = cwt_power / np.sum(cwt_power)
            cwt_info = np.log2(cwt_pdf + scales_dyadic.EPSILON64)
            cwt_entropy = -np.sum(cwt_pdf*cwt_info)
            cwt_entropy_per_band = -np.sum(cwt_pdf*cwt_info, axis=1)

            cwt_power_mean_per_window_per_band = np.mean(cwt_power, axis=1)

            # station_ip: str = station_id_str[0:3] + '.6.1' + station_id_str[5:7] + '.' + station_id_str[7:]
            print(f"Station ID: {station_id_str}")
            # print(f"Equivalent IP: {station_ip}")
            print(f"Start time, Unix s: {EPISODE_START_EPOCH_S}")
            print(f"Band frequency, Hz: {frequency_cwt_hz}")
            print(f"Entropy per band, Bits: {cwt_entropy_per_band}")
            print(f"Total Entropy, Bits: {cwt_entropy}, {np.sum(cwt_entropy_per_band)}")
            fmin = frequency_cwt_hz[0]
            fmax = frequency_cwt_hz[-1]

            # Plot the STX
            # wf_base = ptb.WaveformPlotBase("", f"STX for {EVENT_NAME}, {order_number}")
            wf_base = ptb.WaveformPlotBase("", f"CWT for {station_id_str}_{SEGMENT_START_DATETIME_MSEED}")
            wf_panel = ptb.WaveformPanel(sig_wf, sig_time_s)
            mesh_base = ptb.MeshBase(time_cwt_s,
                                        frequency_cwt_hz,
                                        frequency_hz_ymin=fmin,
                                        frequency_hz_ymax=fmax)
            mesh_panel = ptb.MeshPanel(np.log2(cwt_power+scales_dyadic.EPSILON64),
                                        colormap_scaling="range",
                                        color_range=30,
                                        cbar_units="log$_2$(Power)")
            # TODO: rename to CWT
            stx = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

            if show_stx_fig:
                plt.show()
            if save_fig:
                FIG_NAME = EVENT_NAME + f"_{station_id_str}_{SEGMENT_START_DATETIME_MSEED}.png"
                stx.savefig(os.path.join(FIG_OUT_DIR, FIG_NAME))
