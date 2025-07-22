"""
This module applies ML to data
"""
import os
import random
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from pipeline_io import deserialize_dataframe, save_to_lz4
from visualizations import make_visualizations
"""
Decision boundary displaying under development.
"""
GU_MIN_LAT = 13.227058
GU_MAX_LAT = 14.204108
GU_MIN_LON = 144.598961
GU_MAX_LON = 145.301743

ALPHABET = string.ascii_lowercase + string.digits
MODELS = {"IForest": IForest(n_jobs=-1, behaviour="new"), "LOF": LOF(n_jobs=-1), "OCSVM": OCSVM()}
PARAM_GRID = {
    "IForest": {
        "n_estimators": np.arange(200, 1001, 200),
        "contamination": np.arange(0.1, 0.51, 0.1)
    },
    "LOF": {
        "n_neighbors": np.linspace(0, 1, num=1, dtype=int),
        "contamination": np.arange(0.1, 0.51, 0.1)
    },
    "OCSVM": {
        "gamma": np.logspace(-11, 0, 12) * 2,
        "nu": np.arange(0.1, 1, 0.1),
        "contamination": np.arange(0.1, 0.51, 0.1)
    }
}


# noinspection PyUnusedLocal
def silhouette_score_helper(estimator, x, y=None):
    """
    Helper for the silhouette score function to use in gridscore
    :param estimator: trained estimator used to predict
    :param x: Input matrix / data
    :param y: Labels. Ignored since this is unsupervised case.
    :return: score - the silhouette score of the input and output
    """
    labels = estimator.predict(x)

    # Silhouette score should be -1 if all labels are inliers
    if all(i == labels[0] for i in labels):
        return -1
    score = silhouette_score(x, labels, random_state=42)
    return score


def read_lz4(folder_path: str):
    """
    Reads lz4 file and loads as a pandas csv.
    :param folder_path: Folder path to lz4 files.
    :return: loaded df of lz4 files
    """
    full_df = pd.DataFrame(columns=["station_id", "lat", "lon", "start_time_s", "peak_freq", "peak_entropy",
                                    "total_entropy"])
    # Deserializes any lz4 in the data folder
    for file_path in os.listdir(folder_path):
        temp_df = deserialize_dataframe(os.path.join(folder_path, file_path))
        full_df = pd.concat([full_df, temp_df])
    return full_df.drop_duplicates().reset_index(drop=True).dropna()


def impute_latlon(x):
    jitter_mask = (x["lat"] < GU_MIN_LAT) | (x["lat"] > GU_MAX_LAT) | (x["lon"] < GU_MIN_LON) | (x["lon"] > GU_MAX_LON)
    jitter = x[jitter_mask]
    x_drop_jitter = x.drop(jitter.index)[["station_id", "lat", "lon"]]
    for station_id in jitter["station_id"].unique():
        station_drop = x_drop_jitter[x_drop_jitter["station_id"] == station_id]
        lat_median = station_drop["lat"].median()
        lon_median = station_drop["lon"].median()
        x.loc[jitter_mask, ["lat"]] = lat_median
        x.loc[jitter_mask, ["lon"]] = lon_median

    return x

def preprocessing(x: pd.DataFrame, trained_kmeans: any=None):
    """
    Preprocesses REDVOX pandas dataframe and outputs as processed numpy arrays ready for training.
    :param x: REDVOX data in pandas dataframe
            fields of x: station_id, lat, lon, start_time_s, peak_freq, peak_entropy, total_entropy
    :param trained_kmeans: Prior trained kmeans - used for after executing preprocessing on X_train.  Default None
    :return: Preprocessed version of x in numpy array
            fields of Output: 0 - lat/lon cluster, 1 - start minute, 2 - peak frequency, 3 - peak entropy,
            4 - total entropy
    """
    # Four columns: lat/long cluster, start minute, peak energy, peak frequency
    # Number of records is unchanged
    processed = np.zeros((len(x), 5))
    x = impute_latlon(x)

    geo = x[["lat", "lon"]]

    if trained_kmeans is not None or trained_kmeans:
        best_kmeans = trained_kmeans
    else:
        # Lat/long clustering
        # Silhouette score is used to evaluate KMeans performance
        n_stations = len(x["station_id"].unique())
        silhouette_avg = np.zeros(n_stations - 2)

        for num_clusters in range(n_stations - 2):
            kmeans = KMeans(n_clusters=num_clusters + 2, random_state=42)
            kmeans.fit(geo)
            silhouette_avg[num_clusters] = silhouette_score(geo, kmeans.labels_, random_state=42)

        opt_clusters = np.argmax(silhouette_avg) + 2
        best_kmeans = KMeans(n_clusters=opt_clusters, random_state=42)
        best_kmeans.fit(geo)

    processed[:, 0] = best_kmeans.predict(geo)
    processed[:, 0] /= len(best_kmeans.cluster_centers_)
    # Start minute
    datetime_start = pd.to_datetime(x["start_time_s"], unit='s')
    processed[:, 1] = datetime_start.dt.hour * 60 + datetime_start.dt.minute + datetime_start.dt.second / 60
    processed[:, 1] /= 1440

    # Peak frequency
    processed[:, 2] = (x["peak_freq"] - x["peak_freq"].min()) / (x["peak_freq"].max() - x["peak_freq"].min())

    # Peak entropy
    processed[:, 3] = ((x["peak_entropy"] - x["peak_entropy"].min()) /
                       (x["peak_entropy"].max() - x["peak_entropy"].min()))

    # Total entropy
    processed[:, 4] = ((x["total_entropy"] - x["total_entropy"].min()) /
                       (x["total_entropy"].max() - x["total_entropy"].min()))
    return processed, best_kmeans


def fit_eval(estimator, x_train, x_test):
    """
    Trains and evaluates IForest model exporting score values as well as feature importances of trained model.
    :param estimator: string for model name
    :param x_train: Training set
    :param x_test: Testing set
    :return: Exports scores and feature importances
    """
    # Need a better way to evaluate model. Silhouette score is fine, but Synthetic AUROC could provide more insight.
    tscv = TimeSeriesSplit(n_splits=5)
    PARAM_GRID["LOF"]["n_neighbors"] = np.linspace(1, len(next(tscv.split(x_train))[0]), num=10, dtype=int)
    grid_search = GridSearchCV(estimator=MODELS[estimator], param_grid=PARAM_GRID[estimator], cv=tscv,
                               scoring=silhouette_score_helper)
    grid_search.fit(x_train)
    pred = grid_search.predict(x_test)

    print(f"params: {grid_search.best_params_}")
    print(f"cv score: {grid_search.best_score_}")
    print(f"test score: {silhouette_score(x_test, pred, random_state=42)}")

    return grid_search


if __name__ == "__main__":
    model_name = sys.argv[1]
    _id = ''.join(random.choices(ALPHABET, k=8))
    column_names = ["lat_long_cluster", "start_time_min", "peak_freq", "peak_entropy", "total_entropy"]

    # Reads lz4 file.
    # Replace with folder name with all lz4 files
    print("Reading data...")
    X = read_lz4("./data")
    df = X.copy()

    # 0.8/0.2 train test split
    print("Splitting data...")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    print("Preprocessing data...")
    X_train_ind = X_train.index
    X_train, opt_kmeans = preprocessing(X_train)
    X_test_ind = X_test.index
    X_test, _ = preprocessing(X_test, trained_kmeans=opt_kmeans)

    print("Fitting and evaluating...")
    op_clf = fit_eval(model_name, X_train, X_test)

    feature_importances = []
    if model_name == "IForest":
        feature_importances = op_clf.best_estimator_.feature_importances_
    else:
        results = permutation_importance(op_clf, X_test, None, n_repeats=10, random_state=42)
        feature_importances = np.array([importance for _, importance in enumerate(results.importances_mean)])

    feature_importances = feature_importances / sum(feature_importances)
    feature_importances = pd.Series(feature_importances, index=column_names)
    feature_importances.plot(kind="bar")
    plt.ylabel("importance")
    plt.xlabel("feature")
    plt.title(f"Feature importance of study {model_name} {_id}")
    plt.tight_layout()

    if not os.path.exists("vis"):
        os.mkdir("vis")

    if not os.path.exists(f"vis/{model_name}"):
        os.mkdir(f"vis/{model_name}")

    os.mkdir(f"vis/{model_name}/{_id}")
    plt.savefig(f"vis/{model_name}/{_id}/features.REDVOX.png")
    plt.clf()

    X_preprocess, _ = preprocessing(X, trained_kmeans=opt_kmeans)
    df[f"score.{model_name}.REDVOX.{_id}"] = op_clf.decision_function(X_preprocess)
    df[f"proba.{model_name}.REDVOX.{_id}"] = op_clf.predict_proba(X_preprocess)[:, 1]
    df[f"pred.{model_name}.REDVOX.{_id}"] = op_clf.predict(X_preprocess)
    save_to_lz4(df, file_name=f"score.{model_name}.REDVOX.{_id}.lz4")
    make_visualizations(df, model_name, _id)
