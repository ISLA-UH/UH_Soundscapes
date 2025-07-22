"""
Visualizations of outputs from outlier_detection.py. Visualizations
See cson_pipeline
"""
from typing import List

import folium
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import os
import pandas as pd

from outlier_detection import impute_latlon


def make_visualizations(preds: pd.DataFrame, model_name: str, model_id: str, event_coordinates: List[float],
                        output_dir: str = os.getcwd()):
    """
    Makes plots based on outputs and export results to the output_dir.

    :param preds: Predictions made by model
    :param model_name: Name of model
    :param model_id: ID of study
    :param event_coordinates: Latitude and longitude in degrees, used for map visualization
    :param output_dir: Directory to save visualizations, default current directory
    """
    out_path = os.path.join(output_dir, "vis", model_name, model_id)
    if not os.path.exists(out_path):  # create directory and parent directories if they don't exist
        os.makedirs(out_path, exist_ok=True)
    stations = preds["station_id"].unique()
    prob_cmap = plt.get_cmap("RdBu_r")
    cont_features = ["peak_freq", "peak_entropy", "total_entropy"]
    preds["datetime"] = pd.to_datetime(preds["start_time_s"], unit="s")
    preds = impute_latlon(preds)
    for station in stations:
        station_preds = preds[preds["station_id"] == station]
        for feature in cont_features:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            sm = matplotlib.cm.ScalarMappable(cmap=prob_cmap, norm=norm)
            sm.set_array([])
            for i in range(len(station_preds)):
                plt.plot(station_preds["datetime"].iloc[i: i + 2], station_preds[feature].iloc[i: i + 2],
                         c=prob_cmap(station_preds[f"proba.{model_name}.REDVOX.{model_id}"].iloc[i]))
                plt.title(f"{feature} vs. datetime, Station: {station}")
                plt.xlabel("datetime")
                plt.ylabel(f"{feature}")

            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.ax.set_ylabel("Outlier Probability", rotation=-90, va="bottom")
            plt.xticks(rotation=45)
            plt.tight_layout()
            station_out_path = os.path.join(out_path, station)
            if not os.path.exists(station_out_path):
                os.mkdir(station_out_path)  # make station specific directory
            plt.savefig(os.path.join(station_out_path, f"{feature}vsdatetime.{station}.png"))
            plt.clf()

    # guam_coordinates = [13.444304, 144.793732] # TODO: Change to get the median points eventually
    _map = folium.Map(location=event_coordinates, tiles="CartoDB Positron", zoom_start=11)

    features = []

    for _, row in preds.iterrows():
        coordinates = [row["lon"], row["lat"]]
        ts = row["datetime"]
        ts = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        color = matplotlib.colors.rgb2hex(prob_cmap(row[f"proba.{model_name}.REDVOX.{model_id}"]))
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
                                f"Peak Entropy: {row['peak_entropy']}<br>"),
                    "times": [ts],
                    "icon": "circle",
                    "iconstyle": {"color": f'{color}'}
                },
            }
        )
    folium.plugins.TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features
        },
        period="PT15S",
        duration="PT30S",
        add_last_point=True,
    ).add_to(_map)

    _map.save(os.path.join(out_path, "map.html"))
