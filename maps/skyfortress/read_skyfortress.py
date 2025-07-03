# import matplotlib.pyplot as plt
# import geopandas
# from cartopy import crs as ccrs
# from geodatasets import get_path

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import folium
from folium import plugins
import random



df = pd.read_json('skyfortress.json')
by_trkid = df.groupby("trkId")
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


# [df['lat'].mean(), df['lon'].mean()]
map = folium.Map(location=[21.4389, -158.0001], tiles="CartoDB Positron", zoom_start=11)

features = []
for trk, frame in by_trkid:
    frame = frame.sort_values('ts')
    color = '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # coordinates = frame[['lon', 'lat']].values.tolist()
    # print(coordinates)
    # timestamps = frame['ts'].values.tolist()
    # print(timestamps)
    # print("---------------------------------------------")
    i = 0
    prev_coordinates = []
    prev_ts = ''
    for idx, row in frame.iterrows(): 
        # Add each Point
        coordinates = [row['lon'], row['lat']]
        ts = row['ts']
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates,
                },
                "properties": {
                    "popup": (f"Track ID: {trk}<br>Lon: {row['lon']}<br>Lat: {row['lat']}<br>Time: {ts}"),
                    "times": [ts],
                    # This makes colored circles. Couldn't figure out how to make the markers the same color
                    # as the lines but it got messy anyways.
                    # "icon": "circle",
                    # "iconstyle": {
                    #     "fillColor": color,
                    #     "fillOpacity": 0.6,
                    #     "stroke": "false",
                    #     "radius": 13,
                    # }
                },
            }
        )
        if i > 0:
            # Add each line between tracks.
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [prev_coordinates, coordinates],
                    },
                    "properties": {
                        "times": [prev_ts, ts],
                        "style": {
                            "color": color
                        }

                    }
                }
            )
        prev_coordinates = coordinates
        prev_ts = ts
        i += 1

folium.plugins.TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1H",
    add_last_point=True,
).add_to(map)

map.save('skyfotress_tracks.html')