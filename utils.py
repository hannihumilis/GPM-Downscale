import numpy as np


def format_lat_lon(latitudes, longitudes):
    lat_dirs = np.where(latitudes >= 0, 'N', 'S')
    lon_dirs = np.where(longitudes >= 0, 'E', 'W')

    lat_strs = [f"{abs(lat):.2f}°{dir}" for lat, dir in zip(latitudes, lat_dirs)]
    lon_strs = [f"{abs(lon):.2f}°{dir}" for lon, dir in zip(longitudes, lon_dirs)]

    return lat_strs, lon_strs
