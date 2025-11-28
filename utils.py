import seaborn as sns
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

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算两点之间的方位角（单位：度）
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # 计算经度差
    dLon = lon2 - lon1
    
    # 计算方位角公式 (Bearing Formula)
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    
    # 计算弧度并转回角度
    bearing = np.degrees(np.arctan2(y, x))
    
    # 将范围调整到 [0, 360)
    return (bearing + 360) % 360

def preprocess_copy(df):
    df_new = df.copy()
    
    # 时间处理
    df_new["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df_new["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors='coerce')
    df_new["trip_duration"] = df["trip_duration"]
    df_new["pickup_hour"] = df_new["pickup_datetime"].dt.hour
    # 月份与星期几
    df_new["month"] = df_new["pickup_datetime"].dt.month
    df_new["day_of_week"] = df_new["pickup_datetime"].dt.dayofweek


    # 计算行程距离 km
    df_new["distance_km"] = haversine(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )

    # 速度 km/h
    df_new["speed_kmh"] = df_new["distance_km"] / (df_new["trip_duration"] / 3600)
    df_new.loc[df_new["speed_kmh"] > 160, "speed_kmh"] = np.nan  # 异常速度设为 NaN
    # 经纬度范围过滤
    lon_min, lon_max = -74.3, -73.7
    lat_min, lat_max = 40.5, 41.0

    df_new.loc[
        (df_new["pickup_longitude"] < lon_min) | (df_new["pickup_longitude"] > lon_max) |
        (df_new["pickup_latitude"] < lat_min) | (df_new["pickup_latitude"] > lat_max),
        ["pickup_longitude", "pickup_latitude"]
    ] = np.nan

    df_new.loc[
        (df_new["dropoff_longitude"] < lon_min) | (df_new["dropoff_longitude"] > lon_max) |
        (df_new["dropoff_latitude"] < lat_min) | (df_new["dropoff_latitude"] > lat_max),
        ["dropoff_longitude", "dropoff_latitude"]
    ] = np.nan

    df_new["direction"] = calculate_bearing(
        df_new["pickup_latitude"], df_new["pickup_longitude"],
        df_new["dropoff_latitude"], df_new["dropoff_longitude"]
    )

    # 去除 distance_km 为 0 或负值的异常
    df_new.loc[df_new["distance_km"] <= 0, "distance_km"] = np.nan

    return df_new


sns.set_theme(style="whitegrid")  # 设置 Seaborn 主题

def plot_duration(df, max_min=3600):
    duration = df["trip_duration"]
    duration = duration[duration < max_min]

    plt.figure(figsize=(8,4))
    sns.histplot(duration, bins=50, kde=False, color="skyblue")
    plt.title("Trip Duration Distribution", fontsize=14)
    plt.xlabel("Seconds", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.savefig("./imgs/duration_distribution.png", dpi=300)
    plt.show()

def plot_distance(df, max_km=30):
    dist = df["distance_km"]
    dist = dist[dist < max_km]

    plt.figure(figsize=(8,4))
    sns.histplot(dist, bins=50, kde=False, color="salmon")
    plt.title("Trip Distance Distribution", fontsize=14)
    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.savefig("./imgs/distance_distribution.png", dpi=300)
    plt.show()

def plot_speed(df, max_speed=120):
    speed = df["speed_kmh"]
    speed = speed[speed < max_speed]

    plt.figure(figsize=(8,4))
    sns.histplot(speed, bins=50, kde=False, color="lightgreen")
    plt.title("Speed Distribution", fontsize=14)
    plt.xlabel("Speed (km/h)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.savefig("./imgs/speed_distribution.png", dpi=300)
    plt.show()

def plot_pickup_hour(df):
    hours = df["pickup_hour"]

    plt.figure(figsize=(8,4))
    sns.histplot(hours, bins=24, binrange=(0,24), kde=False, color="mediumpurple")
    plt.title("Pickup Hour Distribution", fontsize=14)
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(range(0,24))
    plt.savefig("./imgs/pickup_hour_distribution.png", dpi=300)
    plt.show()

def plot_pickup_location(df, n=50000):
    sample = df.sample(min(n, len(df)))

    plt.figure(figsize=(6,6))
    sns.scatterplot(
        x="pickup_longitude", y="pickup_latitude", 
        data=sample, s=10, alpha=0.3, color="dodgerblue"
    )
    plt.title("Pickup Locations", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.savefig("./imgs/pickup_locations.png", dpi=300)
    plt.show()

def plot_dropoff_location(df, n=50000):
    sample = df.sample(min(n, len(df)))

    plt.figure(figsize=(6,6))
    sns.scatterplot(
        x="dropoff_longitude", y="dropoff_latitude", 
        data=sample, s=10, alpha=0.3, color="tomato"
    )
    plt.title("Dropoff Locations", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.savefig("./imgs/dropoff_locations.png", dpi=300)
    plt.show()

def plot_month(df):
    month = df["month"]
    plt.figure(figsize=(8,4))
    sns.histplot(month, bins=12, binrange=(0.5,12.5), discrete=True, kde=False, color="gold")
    plt.title("Month Distribution", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(range(1,13))
    plt.savefig("./imgs/month_distribution.png", dpi=300)
    plt.show()

def plot_day_of_week(df):
    day_of_week = df["day_of_week"]

    plt.figure(figsize=(8,4))
    sns.histplot(day_of_week, bins=7, binrange=(0.5,7.5), discrete=True, kde=False, color="gold")
    plt.title("Day of Week Distribution", fontsize=14)
    plt.xlabel("Day of Week", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.savefig("./imgs/day_of_week_distribution.png", dpi=300)
    plt.show()

def plot_direction(df):
    direction = df["direction"]

    plt.figure(figsize=(8,4))
    sns.histplot(direction, bins=36, binrange=(0.5,360.5), discrete=True, kde=False, color="gold")
    plt.title("Direction Distribution", fontsize=14)
    plt.xlabel("Direction (°)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(range(0,360,15))
    plt.savefig("./imgs/direction_distribution.png", dpi=300)
    plt.show()

def visualize_all(df):
    plot_duration(df)
    plot_distance(df)
    plot_speed(df)
    plot_pickup_hour(df)
    plot_pickup_location(df)
    plot_dropoff_location(df)
    plot_month(df)
    plot_day_of_week(df)
    plot_direction(df)

def feature_importance(df):
    # 处理后数据
    df_new = preprocess_copy(df)

    # 选择特征
    feature_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
                    "dropoff_longitude", "dropoff_latitude", "pickup_hour", "distance_km","month","day_of_week","direction"]
    
    # 去除缺失
    valid_idx = df_new[feature_cols].dropna().index
    X = df_new.loc[valid_idx, feature_cols]
    y = df_new.loc[valid_idx, "trip_duration"]

    # # 只用十万的数据
    # if len(X) > 100000:
    #     sample_idx = X.sample(n=100000, random_state=42).index
    #     X = X.loc[sample_idx]
    #     y = y.loc[sample_idx]
    # # -----------------------------------------------------

    # 随机森林
    model = RandomForestRegressor(
        n_estimators=100,  # 减少树的数量，省内存
        max_depth=30,     # 限制树的深度，省内存
        random_state=42,
        n_jobs=4 # 改用 4 个线程
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