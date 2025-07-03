import pandas as pd
import os
from pipeline_io import save_to_lz4, deserialize_dataframe, load_kml, write_kml
import folium
from folium import plugins
import random
import datetime


def read_lz4(folder_path, columns):
    """
    Reads lz4 file and loads as a pandas csv.
    :param folder_path: Folder path to lz4 files.
    :return: loaded df of lz4 files
    """
    full_df = pd.DataFrame(columns=columns)
    for file_path in os.listdir(folder_path):
        print(file_path)
        temp_df = deserialize_dataframe(os.path.join(folder_path, file_path))
        print(temp_df)
        full_df = pd.concat([full_df, temp_df])
    return full_df

if __name__ == "__main__":
    columns=["station_id", "lat", "lon", "start_time_s", "peak_freq", "peak_entropy",
             "total_entropy", "score.IForest.REDVOX.re5vmzjg", "proba.IForest.REDVOX.re5vmzjg",
             "pred.IForest.REDVOX.re5vmzjg"]
    data_path = "PATH/TO/IForest.lz4"
    x = read_lz4(data_path, columns)
    df = pd.DataFrame(x, columns=columns)

    guam_coordinates = [13.444304, 144.793732] #  lat, lon
    map = folium.Map(location=guam_coordinates, tiles="CartoDB Positron", zoom_start=11)

    times = df['start_time_s'].unique().tolist() # every 30 seconds
    time_index = []
    for time in times:
        t = datetime.datetime.fromtimestamp(time)
        t = t.strftime('%Y-%m-%dT%H:%M:%SZ')
        time_index.append(t)

    data = []
    i = 0
    by_ts = df.groupby('start_time_s')
    for ts, frame in by_ts:
        # Each list in data represents a time stamp. 
        per_ts = []
        for idx, row in frame.iterrows():
            # [lat, lon, weight (0-1)]
            point = [row['lat'], row['lon'], row['proba.IForest.REDVOX.re5vmzjg']]
            per_ts.append(point)
        data.append(per_ts)

    # print(data)
    folium.plugins.HeatMapWithTime(data, index=time_index).add_to(map)
    map.save('guam_heat_map.html')