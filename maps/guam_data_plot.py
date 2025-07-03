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
    
    data_path = "PATH/TO/LZ4"
    x = read_lz4(data_path, columns)
    df = pd.DataFrame(x, columns=columns)

    guam_coordinates = [13.444304, 144.793732] #  lat, lon
    map = folium.Map(location=guam_coordinates, tiles="CartoDB Positron", zoom_start=11)

    features = []
    df = df.sort_values('start_time_s')
    # color = '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    i = 0
    prev_coordinates = []
    prev_ts = ''
    for idx, row in df.iterrows(): 
        if (row['pred.IForest.REDVOX.re5vmzjg']):
            coordinates = [row['lon'], row['lat']]
            ts = datetime.datetime.fromtimestamp(row['start_time_s'])
            ts = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
            # print(f"Lon: {row['lon']} | Lat: {row['lat']} | Time: {ts}")
            # print("------------------------------------------------------------------------")
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": coordinates,
                    },
                    "properties": {
                        "popup": (f"Station ID: {row['station_id']}<br>"
                                  f"Lon: {row['lon']}<br>"
                                  f"Lat: {row['lat']}<br>"
                                  f"Time: {ts}<br>"
                                  f"Peak Frequency: {row['peak_freq']}<br>"
                                  f"Peak Entropy: {row['peak_entropy']}"),
                        "times": [ts],
                        "icon": "circle",
                        "iconstyle": {
                            "color": "red"
                        }
                    },
                }
            )
        else:
            coordinates = [row['lon'], row['lat']]
            ts = datetime.datetime.fromtimestamp(row['start_time_s'])
            ts = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
            # print(f"Lon: {row['lon']} | Lat: {row['lat']} | Time: {ts}")
            # print("------------------------------------------------------------------------")
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": coordinates,
                    },
                    "properties": {
                        "popup": (f"Station ID: {row['station_id']}<br>" \
                                  f"Lon: {row['lon']}<br>" \
                                  f"Lat: {row['lat']}<br>" \
                                  f"Time: {ts}<br>" \
                                  f"Peak Frequency: {row['peak_freq']}<br>" \
                                  f"Peak Entropy: {row['peak_entropy']}"),
                        "times": [ts],
                        "icon": "circle",
                        "iconstyle": {
                            "color": "blue"
                        }
                    },
                }
            )

    print(len(features))
    folium.plugins.TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT1M", # Makes the time iterate by minute
        duration="PT30S", # Points display for 30seconds
        add_last_point=True,
    ).add_to(map)

    map.save('guam_map.html')
