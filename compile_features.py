import rasterio
import pyproj
import cv2
import numpy as np
import pickle
import os
import argparse


patch_size = 384


def pad_features_to_patch_size(features, patch_size):
    height, width, _ = features["optical"].shape
    target_height = patch_size * int(height // patch_size + 1)
    target_width = patch_size * int(width // patch_size + 1)
    pad_height = (target_height - height) // 2
    pad_width = (target_width - width) // 2
    for feature in features:
        feature_arr = features[feature]
        _, _, n_bands = feature_arr.shape
        padded = np.zeros((target_height, target_width, n_bands))
        padded[pad_height:pad_height + height, pad_width:pad_width + width, :] = feature_arr
        features[feature] = padded
    return features, (height, width), (pad_height, pad_width)


def main():
    with open("dataset_stats.pickle", "rb") as file:
        mins, maxs = pickle.load(file)
    
    with rasterio.open(args.optical, "r") as file:
        meta = file.meta
        bands = []
        for band_idx in range(file.count):
            band = file.read(band_idx + 1)
            bands.append(band)
        optical = np.stack(bands, axis=-1)
        optical = np.where(~np.isnan(optical), optical, 0)
        optical = (optical - mins["optical"]) / (maxs["optical"] - mins["optical"])
    height, width, _ = optical.shape
    
    # make meta template
    meta["dtype"] = None
    meta["count"] = -1

    # make lat/lon grid
    transform = meta["transform"]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    if meta["crs"].to_epsg() != 4326:
        target_crs = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(meta["crs"], target_crs, always_xy=True)
        xs = np.reshape(xs, (-1, ))
        ys = np.reshape(ys, (-1, ))
        xs, ys = transformer.transform(xs, ys)
    lat = np.array(ys)
    lon = np.array(xs)
    lat = np.reshape(lat, (height, width, 1))
    lon = np.reshape(lon, (height, width, 1))
    
    with rasterio.open(args.elevation, "r") as file:
        elev = file.read(1)
        elev = (elev - mins["dem"]) / (maxs["dem"] - mins["dem"])
        elev = cv2.resize(elev, (width, height))
    with rasterio.open(args.slope, "r") as file:
        slope = file.read(1)
        slope = (slope - mins["slope"]) / (maxs["slope"] - mins["slope"])
        slope = cv2.resize(slope, (width, height))
    dem = np.stack([elev, slope], axis=-1)

    features = {
        "optical": optical,
        "dem": dem,
        "lat": lat,
        "lon": lon,
    }

    for feature, path in [
        ("co_pol_sar", args.co_pol_sar), ("cross_pol_sar", args.cross_pol_sar), 
        ("in_sar", args.in_sar), ("thermal", args.thermal),
    ]:
        if path:
            with rasterio.open(path, "r") as file:
                arr = file.read()
                arr = (arr - mins[feature]) / (maxs[feature] - mins[feature])
                arr = cv2.resize(arr, (width, height))
            features[feature] = arr
    
    features, _, (pad_height, pad_width) = pad_features_to_patch_size(features, patch_size)
    attrs = {
        "height": height,
        "width": width,
        "pad_height": pad_height,
        "pad_width": pad_width,
        "meta": meta,
    }

    with open(args.output, "wb") as file:
        pickle.dump((features, attrs), file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="features.pickle", help="Output path")
    parser.add_argument("--optical", help="Path to optical .tif raster (REQUIRED)")
    parser.add_argument("--elevation", help="Path to elevation .tif raster (REQUIRED)")
    parser.add_argument("--slope", help="Path to slope .tif raster (REQUIRED)")

    parser.add_argument("--co_pol_sar", help="Path to co_pol_sar .tif raster")
    parser.add_argument("--cross_pol_sar", help="Path to cross_pol_sar .tif raster")
    parser.add_argument("--in_sar", help="Path to in_sar .tif raster")
    parser.add_argument("--thermal", help="Path to thermal .tif raster")

    args = parser.parse_args()

    if not args.optical or not args.elevation or not args.slope:
        raise ValueError("At least optical, elevation and slope must be provided.")

    main()
