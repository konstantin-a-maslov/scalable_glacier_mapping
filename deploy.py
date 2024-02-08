import config
import rasterio
import scipy.ndimage
import numpy as np
import models.mapping
import utils
import pickle
import os
import gc
from tqdm import tqdm


def apply_model(model, features, batch_size=config.data.batch_size):
    patch_size = config.data.patch_size
    height, width, _ = features["optical"].shape
    weighted_prob = np.zeros((height, width, config.data.n_outputs))
    weights = gaussian_kernel(patch_size)[..., np.newaxis]
    counts = np.zeros((height, width, 1))

    patches = {feature: [] for feature in features.keys() if feature not in {"lat", "lon"}}
    if args.region_encoding or args.coordinate_encoding:
        patches["location"] = []
    n_patches = 0
    rows_cols = []

    row = 0
    while row + patch_size <= height:
        col = 0 
        while col + patch_size <= width:
            # patch = {}
            for feature, arr in features.items():
                if feature == "lat" or feature == "lon":
                    continue
                patches[feature].append(arr[row:row + patch_size, col:col + patch_size, :])

            if args.region_encoding:
                patches["location"].append(region_vector)
            if args.coordinate_encoding:
                lat_arr = features["lat"]
                lon_arr = features["lon"]
                lat = lat_arr[row + patch_size // 2, col + patch_size // 2, 0]
                lon = lon_arr[row + patch_size // 2, col + patch_size // 2, 0]
                coordinates_vector = np.array([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)])
                patches["location"].append(coordinates_vector)

            rows_cols.append((row, col))
            n_patches += 1

            if n_patches >= batch_size:
                for feature in patches.keys():
                    patches[feature] = np.array(patches[feature])
                patch_probs = model.predict(patches, verbose=2)

                for (row_pred, col_pred), patch_prob in zip(rows_cols, patch_probs):
                    weighted_prob[
                        row_pred:row_pred + patch_size, 
                        col_pred:col_pred + patch_size, :
                    ] += (weights * patch_prob)
                    counts[
                        row_pred:row_pred + patch_size, 
                        col_pred:col_pred + patch_size, :
                    ] += weights

                patches = {feature: [] for feature in patches.keys()}
                n_patches = 0
                rows_cols = []

            col += (patch_size // 2)
        row += (patch_size // 2)
    
    if n_patches > 0:
        for feature in patches.keys():
            patches[feature] = np.array(patches[feature])
        patch_probs = model.predict(patches, verbose=0)

        for (row_pred, col_pred), patch_prob in zip(rows_cols, patch_probs):
            weighted_prob[
                row_pred:row_pred + patch_size, 
                col_pred:col_pred + patch_size, :
            ] += (weights * patch_prob)
            counts[
                row_pred:row_pred + patch_size, 
                col_pred:col_pred + patch_size, :
            ] += weights

    prob = weighted_prob / counts
    return prob


def gaussian_kernel(size, mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(distance - mu)**2 / 2/ sigma**2) / np.sqrt(2 / np.pi) / sigma
    return kernel


def smooth(arr):
    return scipy.ndimage.median_filter(arr, size=args.smoothing)


def shannon_entropy(x, eps=1e-6, C=2):
    x = np.clip(x, eps, 1 - eps)
    return -np.sum(x * np.log(x) / np.log(C), axis=-1)


def shannon_confidence(x):
    return 1 - shannon_entropy(x)


def main():
    train_dataloader = utils.build_dataloader(
        config.data.train_sampler_builder,
        config.data.train_sampler_args, 
        config.data.train_plugins, 
        config.data.batch_size,
        labels="outlines"
    )

    model_builder = config.model.model_builder
    model_args = config.model.model_args
    weights_path = os.path.join("weights", f"{config.model.model_name}_weights.h5")
    model = utils.build_model(
        model_builder, model_args, config.data, 
        weights_path=weights_path, mode="testing"
    )

    with open(args.input, "rb") as file:
        features, attrs = pickle.load(file)
        height, width = attrs["height"], attrs["width"]
        pad_height, pad_width = attrs["pad_height"], attrs["pad_width"]
        meta = attrs["meta"]

    probs = None
    for _ in tqdm(range(args.n)):
        prob = apply_model(model, features)
        if probs is None:
            probs = prob
        else:
            probs += prob

    probs /= args.n

    del model
    gc.collect()

    probs = probs[pad_height:pad_height + height, pad_width:pad_width + width, :]
    pred = np.argmax(probs, axis=-1)
    confidence = shannon_confidence(probs)

    if calibration_model:
        confidence = np.reshape(confidence, (height * width, 1))
        batch_size = 1024 * 32
        for i in tqdm(range(0, len(confidence), batch_size)):
            batch_slice = slice(i, i + batch_size)
            confidence[batch_slice] = calibration_model.predict(confidence[batch_slice])[:, np.newaxis]
        confidence[confidence < 0] = 0
        confidence[confidence > 1] = 1
        confidence = np.reshape(confidence, (height, width))

    meta.update(count=1, dtype=np.uint8)
    with rasterio.open(os.path.join(args.output, "outlines.tif"), "w", **meta) as file:
        if args.smoothing:
            pred = smooth(pred)
        file.write(pred.astype(np.uint8), 1)

    meta.update(dtype=np.float32)
    with rasterio.open(os.path.join(args.output, "confidence.tif"), "w", **meta) as file:
        file.write(confidence.astype(np.float32), 1)
        

if __name__ == "__main__":
    parser = utils.create_cli_parser()
    parser.add_argument("input", help="Input .pickle path")
    parser.add_argument("-o", "--output", default=".", help="Output folder path")
    parser.add_argument("--n", default=1, type=int, help="Number of inference runs")
    parser.add_argument("-cm", "--calibration_model", help="Confidence calibration model path")
    parser.add_argument("-rv", "--region_vector", help="Region vector path")
    parser.add_argument("--smoothing", type=int, help="Median filter smoothing factor for outlines")
    args = parser.parse_args()
    if args.n > 1:
        args.mcdropout = True
    if args.region_vector:
        args.region_encoding = True
    utils.update_config_from_args(config, args)

    calibration_model = None
    region_vector = None
    if args.calibration_model:
        with open(args.calibration_model, "rb") as file:
            calibration_model = pickle.load(file)
    if args.region_vector:
        with open(args.region_vector, "rb") as file:
            region_vector = pickle.load(file)

    main()
