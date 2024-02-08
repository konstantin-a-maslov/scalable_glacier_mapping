import h5py
import numpy as np
import config
import dataloaders
import utils
import os
from tqdm import tqdm


def read_tile(tile):
    pad_height = tile.attrs["padding_height"]
    pad_width = tile.attrs["padding_width"]
    features = {_: np.array(tile[_])[np.newaxis, ...] for _ in tile.keys() if _ != config.data.labels}
    groundtruth = np.array(tile[config.data.labels])
    return features, groundtruth, (pad_height, pad_width)


def apply_model(model, features):
    patch_size = config.data.patch_size
    _, height, width, _ = features["optical"].shape
    weighted_prob = np.zeros((height, width, config.data.n_outputs))
    weights = gaussian_kernel(patch_size)[..., np.newaxis]
    counts = np.zeros((height, width, 1))

    row = 0
    while row + patch_size <= height:
        col = 0 
        while col + patch_size <= width:
            patch = {}
            for feature, arr in features.items():
                if len(arr.shape) == 4:
                    patch[feature] = arr[:, row:row + patch_size, col:col + patch_size, :]
                else:
                    patch[feature] = arr
            if args.n > 1:
                patch_prob = None
                for _ in range(args.n):
                    if not patch_prob:
                        patch_prob = model.predict(patch)[0]
                    else:
                        patch_prob += model.predict(patch)[0]
                patch_prob /= args.n
            else:
                patch_prob = model.predict(patch)[0]
            weighted_prob[row:row + patch_size, col:col + patch_size, :] += (weights * patch_prob)
            counts[row:row + patch_size, col:col + patch_size, :] += weights
            col += (patch_size // 2)
        
        row += (patch_size // 2)
    
    prob = weighted_prob / counts
    return prob


def gaussian_kernel(size, mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(distance - mu)**2 / 2 / sigma**2) / np.sqrt(2 / np.pi) / sigma
    return kernel


def main():
    if config.model.inference_dropout:
        raise NotImplementedError()

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

    for tile_name in tqdm(dataset.keys()):
        tile = dataset[tile_name]
        features, true, (pad_height, pad_width) = read_tile(tile)

        if len(train_dataloader.plugins) > 0:
            skip_tile = False
            for _plugin in train_dataloader.plugins:
                if isinstance(_plugin, dataloaders.TileFilter):
                    if not _plugin.apply_filters(tile):
                        skip_tile = True
                        break

            if skip_tile:
                continue

            plugin = None
            for _plugin in train_dataloader.plugins:
                if isinstance(_plugin, dataloaders.AddRegionVector) \
                    or isinstance(_plugin, dataloaders.AddCoordinatesVector):
                    plugin = _plugin

            if isinstance(plugin, dataloaders.AddRegionVector):
                region = tile.attrs["region"]
                features["location"] = plugin.get_region_vector(region)[np.newaxis, ...]
                # unknown = np.zeros(plugin.n_regions + 1)
                # unknown[-1] = 1
                # features["location"] = unknown[np.newaxis, ...]

            if isinstance(plugin, dataloaders.AddCoordinatesVector):
                lat = (tile.attrs["ymin"] + tile.attrs["ymax"]) / 2
                long = (tile.attrs["xmin"] + tile.attrs["xmax"]) / 2
                features["location"] = plugin.get_coordinates_vector(lat, long)[np.newaxis, ...]

        prob = apply_model(model, features)
        pred = np.argmax(prob, axis=-1)

        prob = prob[pad_height:-pad_height, pad_width:-pad_width, :]
        pred = pred[pad_height:-pad_height, pad_width:-pad_width]
        true = true[pad_height:-pad_height, pad_width:-pad_width, -1]

        group = predictions_dataset.create_group(tile_name)
        group.create_dataset("prob", data=prob)
        group.create_dataset("pred", data=pred)
        group.create_dataset("true", data=true)
        group.attrs["tile_name"] = tile_name
        group.attrs["region"] = tile.attrs["region"]
        group.attrs["subregion"] = tile.attrs["subregion"]
        

if __name__ == "__main__":
    parser = utils.create_cli_parser()
    parser.add_argument("--val", action="store_true", help="Use validation set for inference")
    parser.add_argument("--n", default=1, type=int, help="Number of inference runs")
    args = parser.parse_args()
    if args.n > 1:
        args.mcdropout = True
    utils.update_config_from_args(args)

    if args.val:
        dataset = h5py.File(config.data.val_dataset_path, "r")
    else:
        dataset = h5py.File(config.data.test_dataset_path, "r")
    predictions_dataset_dir = os.path.join(config.data.predictions_dir, config.model.model_name)
    if not os.path.exists(predictions_dataset_dir):
        os.makedirs(predictions_dataset_dir, exist_ok=True)

    predictions_dataset_path = os.path.join(
        predictions_dataset_dir, f"predictions{'_val' if args.val else ''}.hdf5"
    )
    if os.path.isfile(predictions_dataset_path):
        os.remove(predictions_dataset_path)
    predictions_dataset = h5py.File(predictions_dataset_path, "w")

    main()

    predictions_dataset.close()
    dataset.close()
