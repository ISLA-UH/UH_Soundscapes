'''
To push Cybersonic Data to UDL

Katie Stevens 2025
'''
import requests, base64
import sys, os
from datetime import datetime, timezone
import ssl
import xmlrpc
import xmlrpc.client
import urllib
import json
import shutil
import pandas as pd
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from dotenv import load_dotenv
import datetime

# QI modules
from quantum_inferno import styx_cwt, scales_dyadic
import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_mesh_wf_vert

# RedVox modules
import redvox.common.date_time_utils as dt
from redvox.common.data_window import DataWindowConfig, DataWindow

'''
TODO: Figure out what schema on UDL that would work best for our data
TODO: Update required JSON keys 
'''

#SERVICE_ENDPOINT = "https://imagery-test.unifieddatalibrary.com/filedrop/"
'''
Looking at Schemas: Track, Notification
Upload type and requirements will be different per schema. Check UDL to see what is needed.

Our fields:
- location: long & lat
- station id
- descriptor: what happened, hazard type, ml classes? 
- peak entropy
- peak frequency
- pecoc/ml output: anomoulous score. 
'''
MICROS_TO_S = 1E-6
REF_EPOCH_S = 1684926000
# Window duration and hop duration in seconds
EPISODE_START_EPOCH_S = REF_EPOCH_S
DURATION_S = 60
HOP_S = 60

# Additional options
load_datawindow_from_file: bool = False
save_datawindow: bool = False
show_figure: bool = False
show_stx_fig: bool = True
save_data: bool = False

# Constants and other important values
order_number: int = 3
frequency_sample_rate_hz: float = 800.
frequency_highest_hz: float = 320.
frequency_lowest_hz: float = 1.
cyber_averaging_scale: float = frequency_sample_rate_hz / frequency_lowest_hz
cyber_averaging_frame: float = .75 * np.pi * order_number * cyber_averaging_scale
cyber_averaging_frame_pow2: int = int(2**np.ceil(np.log2(cyber_averaging_frame)))
averaging_frame_s = cyber_averaging_frame_pow2 / frequency_sample_rate_hz

DURATION_POW2_POINTS = int(2**np.ceil(np.log2(DURATION_S * frequency_sample_rate_hz)))
DURATION_POW2_S = DURATION_POW2_POINTS / frequency_sample_rate_hz

start_segment = 0
end_segment = 2
segment_counters = np.arange(start_segment, end_segment)
number_of_segments = len(segment_counters)

def create_json_file(data):
    # Derived from ZuluWookiee Code

    classificationMarking = "U"
    #content = "3D Model" 
    #msgTime = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec='milliseconds') + 'Z' 
    description = "Lilipad Payload"
    source = "ARL" # Cybersonic? 
    dataMode = "TEST" # REAL 

    # Optional Fields: 
    keywords = [] 

    # This is an example of what would be posted to Notification Schema
    post_json_data = {
        "classificationMarking": classificationMarking,
        "description": description,
        "source": source,
        "dataMode": dataMode,
        "msgBody": data # The actual audio data
    }

    #post_json_data.update(data)

    # Write json to directory
    # with open("nameofupload.json", "w") as f:
    #     json.dump(post_json_data, f)

    return post_json_data

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

if __name__=="__main__":
    # load_dotenv()
    # USERNAME = os.getenv("USER")
    # PASSWORD = os.getenv("PASSWORD")
    # input_file = ""

    # for reading in json data as argument
    # for arg in sys.argv:
    #     if len(sys.argv) < 2:
    #         print("Please provide an input json file.")
    #         sys.exit(0)
    #     input_file = sys.argv[1]
    #     if not input_file.endswith(".json"):
    #         print("The file provided was not a .json file.")
    #         sys.exit(0)

    # with open(input_file, 'r') as f:
    #     data = json.load(f)
    # post_json_data = create_json_file(data)

    # key = USERNAME + ":" + PASSWORD
    # basicAuth = base64.b64encode(key.encode('utf-8')).decode("ascii")
    # udl_headers = {'accept': 'application/json',
    #                'content-type': 'application/json',
    #                'Authorization': 'Basic {auth}'.format(auth=basicAuth)}

    for time_index, segment_counter in enumerate(segment_counters):
        print('Segment counter: ', segment_counter)
        SEGMENT_START_EPOCH_S = EPISODE_START_EPOCH_S + segment_counter * HOP_S
        SEGMENT_END_EPOCH_S = SEGMENT_START_EPOCH_S + DURATION_POW2_S
        SEGMENT_NAME = "mawar_epoch_" + str(SEGMENT_START_EPOCH_S) + '_dur_' + str(DURATION_S) + 's'

        SEGMENT_START_DATETIME = dt.datetime_from_epoch_seconds_utc(int(SEGMENT_START_EPOCH_S))
        SEGMENT_END_DATETIME = dt.datetime_from_epoch_seconds_utc(int(SEGMENT_END_EPOCH_S))


        print('Event Name:', SEGMENT_NAME)
        print('Start:', SEGMENT_START_DATETIME)
        print('Stop:', SEGMENT_END_DATETIME)

        if load_datawindow_from_file:
            # Load data window from report
            # dw_path = os.path.join(OUTPUT_DIR, 'dw_lz4', SEGMENT_NAME + '.pkl.lz4')
            # print('DataWindow file path:', dw_path)
            # dw = DataWindow.deserialize(dw_path)
            print("load datawindow:")
        else:
            dw = DataWindow.from_config_file("dw.config.toml")
            # dw_config = DataWindowConfig(
            #     input_dir=INPUT_DIR,
            #     start_datetime=SEGMENT_START_DATETIME,
            #     end_datetime=SEGMENT_END_DATETIME)
            #     #station_ids=STATIONS)

            # dw: DataWindow = DataWindow(event_name=SEGMENT_NAME, config=dw_config, debug=False,
            #                             output_dir=os.path.join(OUTPUT_DIR, 'dw_lz4'), out_type="LZ4")
            # if save_datawindow:
            #     dw.save()

        station_list = dw.stations()
        # print(len(station_list))

        # Loop over selected stations
        for station in dw.stations():
            # Station ID
            station_id_str: str = station.id()
            print("Station ID:", station_id_str)
            # Get audio data
            sig_wf_sample_rate_hz = station.audio_sample_rate_nominal_hz()
            sig_wf_raw = station.audio_sensor().get_microphone_data()
            sig_epoch_start_micros = station.audio_sensor().first_data_timestamp()
            sig_epoch_micros = station.audio_sensor().data_timestamps()
            # Get Location Data
            longitude_data = station.location_sensor().get_longitude_data()
            latitude_data = station.location_sensor().get_latitude_data()
            loc_time = station.location_sensor().get_gps_timestamps_data()
            loc_time_clean = loc_time[~np.isnan(loc_time)]

            longitude = np.nanmean(longitude_data)
            latitude = np.nanmean(latitude_data)

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
            sig_wf = np.nan_to_num(sig_wf_raw2[0:fft_duration_ave_window_points_pow2])
            # sig_wf = np.nan_to_num(sig_wf_raw2)
            sig_time_s = sig_time_s[0:fft_duration_ave_window_points_pow2]
            print('Signal number of points:', len(sig_wf))

            sig_power = np.abs(sig_wf)**2
            # sig_power_mean_array[time_index] = np.var(sig_wf)
            # sig_power_sum_array[time_index] = np.sum(sig_power)
            # sig_power_median_array[time_index] = np.median(sig_power)
            print('CWT of order ', order_number)

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

            station_ip: str = station_id_str[0:3] + '.6.1' + station_id_str[5:7] + '.' + station_id_str[7:]
            print('Station ID: ', station_id_str)
            print('Equivalent IP:', station_ip)
            print('Start time, Unix s: ', EPISODE_START_EPOCH_S)
            print('Band frequency, Hz: ', frequency_cwt_hz)
            print('Entropy per band, Bits: ', cwt_entropy_per_band)
            print('Total Entropy, Bits: ', cwt_entropy, np.sum(cwt_entropy_per_band))

            start_time = dt.datetime_from_epoch_microseconds_utc(dw.start_date())
            end_time = dt.datetime_from_epoch_microseconds_utc(dw.end_date())

            data = {
                "stationId": station_id_str,
                "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "longitude": longitude,
                "latitude": latitude,
                "frequency": frequency_cwt_hz.tolist(),
                "entropy": cwt_entropy_per_band.tolist(),
                "totalEntropy": cwt_entropy
            }
            print("try to post to UDL")
            print("Saving generated Json...")
            json_data = create_json_file(data)
            with open("sample.json", 'w') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            exit(0)

            # print('Band frequency, mHz: ', 1000 * frequency_cwt_hz)
            # print('Entropy per band, mBits: ', 1000 * cwt_entropy_per_band)
            # print('Total Entropy, mBits: ', 1000 * cwt_entropy, 1000 * np.sum(cwt_entropy_per_band))
    
    try:
        print("try to post to UDL")
        print("Saving generated Json...")
        json_data = create_json_file(data)
        with open("sample.json", 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        # response = requests.post(SERVICE_ENDPOINT, data=post_json_data, headers=udl_headers)

        # response.raise_for_status()
        # if response.ok:
        #     print("Successful Post")
        # else:
        #     print(f"{response} Error occured.")

        # if result.status_code == 404:
        #     print(f"Request failed with status code: {result.status_code}")
        #     print("Resource not found.")
        #     sys.exit()
        # elif result.status_code == 401:
        #     print(f"Request failed with status code: {result.status_code}")
        #     print("Authentication Required.")
        #     sys.exit()
        # elif result.status_code == 503:
        #     print(f"Request failed with status code: {result.status_code}")
        #     print("Server is not ready to handle request. (UDL is probably down for maintenance)")
        #     sys.exit()
    except xmlrpc.client.ProtocolError as err:
        print("A protocol error occurred")
        print("URL: %s" % err.url)
        print("HTTP/HTTPS headers: %s" % err.headers)
        print("Error code: %d" % err.errcode)
        print("Error message: %s" % err.errmsg)
