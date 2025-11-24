import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Haversine 计算经纬度距离
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # 地球半径（km）

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def preprocess_copy(df):
    df_new = df.copy()
    df_new["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df_new["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors='coerce')
    df_new["trip_duration"] = df["trip_duration"]
    df_new["pickup_hour"] = df_new["pickup_datetime"].dt.hour
    df_new["distance_km"] = haversine(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )

    # 速度 km/h
    df_new["speed_kmh"] = df_new["distance_km"] / (df_new["trip_duration"] / 3600)
    # 去除异常速度
    df_new.loc[df_new["speed_kmh"] > 200, "speed_kmh"] = np.nan

    return df_new


def plot_duration(df, max_min=3600):
    duration = df["trip_duration"]
    duration = duration[duration < max_min]

    plt.figure(figsize=(8,4))
    plt.hist(duration, bins=50)
    plt.title("Trip Duration Distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.show()

def plot_distance(df, max_km=30):
    dist = df["distance_km"]
    dist = dist[dist < max_km]

    plt.figure(figsize=(8,4))
    plt.hist(dist, bins=50)
    plt.title("Trip Distance Distribution")
    plt.xlabel("Distance (km)")
    plt.ylabel("Count")
    plt.show()

def plot_speed(df, max_speed=120):
    speed = df["speed_kmh"]
    speed = speed[speed < max_speed]

    plt.figure(figsize=(8,4))
    plt.hist(speed, bins=50)
    plt.title("Speed Distribution")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Count")
    plt.show()

def plot_pickup_hour(df):
    hours = df["pickup_hour"]

    plt.figure(figsize=(8,4))
    plt.hist(hours, bins=24, range=(0,24))
    plt.title("Pickup Hour Distribution")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.xticks(range(24))
    plt.show()

def plot_pickup_location(df, n=50000):
    sample = df.sample(min(n, len(df)))

    plt.figure(figsize=(6,6))
    plt.scatter(sample["pickup_longitude"], sample["pickup_latitude"], s=1, alpha=0.3)
    plt.title("Pickup Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def plot_dropoff_location(df, n=50000):
    sample = df.sample(min(n, len(df)))

    plt.figure(figsize=(6,6))
    plt.scatter(sample["dropoff_longitude"], sample["dropoff_latitude"], s=1, alpha=0.3)
    plt.title("Dropoff Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def visualize_all(df):
    plot_duration(df)
    plot_distance(df)
    plot_speed(df)
    plot_pickup_hour(df)
    # plot_pickup_location(df)
    # plot_dropoff_location(df)

def feature_importance(df):
    # 处理后数据
    df_new = preprocess_copy(df)

    # 选择特征
    feature_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude", "pickup_hour", "distance_km"]
    X = df_new[feature_cols]
    y = df_new["trip_duration"]

    # 去除缺失
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # 随机森林
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # 特征重要性
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return importance, model


def plot_importance(importance_df):
    plt.figure(figsize=(6,4))
    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.gca().invert_yaxis()
    plt.show()