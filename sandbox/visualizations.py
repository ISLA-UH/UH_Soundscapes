"""
Visualizations of outputs from outlier_detection.py. Visualizations
See cson_pipeline
"""
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import folium
from folium import plugins


def make_visualizations(preds, model_name, model_id):
    """
    Makes plots based on outputs and exports to correct folder.
    :param preds: Predictions made by model
    :param model_name: Name of model
    :param model_id: ID of study
    :return:
    """
    stations = preds["station_id"].unique()
    prob_cmap = plt.get_cmap("RdBu_r")
    cont_features = ["peak_freq", "peak_entropy", "total_entropy"]
    preds["datetime"] = pd.to_datetime(preds["start_time_s"], unit="s")
    from outlier_detection import impute_latlon
    preds = impute_latlon(preds)
    for station in stations:

        station_preds = preds[preds["station_id"] == station]
        for feature in cont_features:

            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = mpl.cm.ScalarMappable(cmap=prob_cmap, norm=norm)
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
            if not os.path.exists(f"vis/{model_name}/{model_id}/{station}"):

                os.mkdir(f"vis/{model_name}/{model_id}/{station}")

            plt.savefig(f"vis/{model_name}/{model_id}/{station}/{feature}vsdatetime.{station}.png")
            plt.clf()

    guam_coordinates = [13.444304, 144.793732] # TODO: Change to get the median points eventually
    _map = folium.Map(location=guam_coordinates, tiles="CartoDB Positron", zoom_start=11)

    features = []

    for idx, row in preds.iterrows():

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

    _map.save(f"vis/{model_name}/{model_id}/map.html")
