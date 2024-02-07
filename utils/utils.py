import dataloaders
import dataloaders.filters
import warnings
import argparse


def update_config(
    config, model=None, name=None, region_encoding=None, coordinate_encoding=None, features=None, 
    regions=None, subregions=None, labels=None, weights_path=None, contrast=None, occlusion=None, 
    noise=None, label_smoothing=None
):
    if region_encoding and coordinate_encoding:
        raise NotImplementedError("Simultaneous region and coordinate encodings are not implemented!")
    if model != "glavitu" and (region_encoding or coordinate_encoding):
        raise NotImplementedError("Region/coordinate encoding is implemented only for GlaViTU!")

    if model:
        set_model(config, model)
    if name:
        set_model_name(config, name)
    if region_encoding:
        add_region_encoding(config)
    if coordinate_encoding:
        add_coordinate_encoding(config)
    if features:
        set_features(config, features)
    if regions:
        set_regions(config, regions)
    if subregions:
        set_subregions(config, subregions)
    if labels:
        warnings.warn("Setting labels is deprecated, change it in code manually if really intended.", DeprecationWarning)
        # set_target_labels(config, labels)
    if weights_path:
        set_weights_path(config, weights_path)
    if contrast:
        add_contrast_augmentation(config)
    if occlusion:
        add_feature_occlusion(config)
    if noise:
        add_noise_augmentation(config)
    if label_smoothing:
        add_label_smoothing(config, label_smoothing)


def create_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="glavitu", choices=["glavitu", "deeplab"], help="Model architecture")
    parser.add_argument("-n", "--name", help="Model name")

    parser.add_argument("--region_encoding", action="store_true", help="Use of region encoding")
    parser.add_argument("--coordinate_encoding", action="store_true", help="Use of coordinate encoding")

    parser.add_argument("-f", "--features", default=["optical", "dem"], nargs="*", help="Training features")
    parser.add_argument("-r", "--regions", nargs="*", help="Regions of interest")
    parser.add_argument("-sr", "--subregions", nargs="*", help="Subregions of interest")
    parser.add_argument("-l", "--labels", default="outlines", help="Name of target labels (DEPR)")
    parser.add_argument("-w", "--weights_path", help="Path to starting point weights")

    parser.add_argument("--contrast", action="store_true", help="Use of random gamma correction augmentation")
    parser.add_argument("--occlusion", action="store_true", help="Use of feature occlusion")
    parser.add_argument("--noise", action="store_true", help="Use of gaussian noise/shift augmentation")

    parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing parameter")

    return parser


def update_config_from_args(args):
    config.cli_args = args
    update_config(
        config, 
        model=args.model,
        name=args.name,
        region_encoding=args.region_encoding,
        coordinate_encoding=args.coordinate_encoding,
        features=args.features,
        regions=args.regions,
        subregions=args.subregions,
        labels=args.labels,
        weights_path=args.weights_path,
        contrast=args.contrast,
        occlusion=args.occlusion,
        noise=args.noise,
        label_smoothing=args.label_smoothing,
    )


def update_config_from_cli(config):
    parser = create_cli_parser()
    args = parser.parse_args()
    update_config_from_args(args)


def build_dataloader(sampler_builder, sampler_args, plugins, batch_size, labels, len_factor=1):
    sampler = sampler_builder(**sampler_args)
    return dataloaders.DataLoader(sampler, plugins, batch_size, labels, len_factor)


def build_model(model_builder, model_args, data_config=None, weights_path=None, mode="training"):
    if mode not in {"training", "testing"}:
        raise ValueError()
    # if location is used but not specified, try to infer it from data_config
    if "use_location" in model_args and "location_size" not in model_args and data_config:
        plugins = data_config.train_plugins
        if len(plugins) == 0:
            raise ValueError()
        for plugin in plugins:
            if isinstance(plugin, dataloaders.AddRegionVector):
                model_args["location_size"] = plugin.n_regions + 1
        if "location_size" not in model_args:
            raise NotImplementedError()
    elif "use_location" in model_args and "location_size" not in model_args:
        raise ValueError()
    # initialize input_shapes and n_outputs for model
    if data_config:
        model_args["input_shapes"] = data_config.input_shapes
        model_args["n_outputs"] = data_config.n_outputs
    # build the model
    model = model_builder(**model_args)
    # load weights if requested
    if weights_path:
        if isinstance(model, tuple):
            model[0].load_weights(weights_path)
        else:
            model.load_weights(weights_path)
    # choose proper subnet for mode
    if isinstance(model, tuple):
        if mode == "training":
            model = model[0]
        else:
            model = model[1]
    return model


def set_model_name(config, model_name):
    config.model.model_name = model_name
    config.model.model_args.update({
        "name": model_name 
    })


def set_model(config, model):
    if model == "glavitu":
        pass # default choice

    elif model == "deeplab":
        import configs.models.deeplab
        config.model = configs.models.deeplab
    
    else:
        raise NotImplementedError()


def add_region_encoding(config):
    config.data.train_plugins += [
        dataloaders.AddRegionVector(),
        dataloaders.Augmentation([
            dataloaders.transformations.set_region_to_unknown(p=0.1)
        ])
    ]
    config.data.val_plugins += [
        dataloaders.AddRegionVector(),
        dataloaders.RepeatWithMandatoryTransformations([
            dataloaders.transformations.set_region_to_unknown(p=1.0)
        ])
    ]
    config.model.model_args.update({
        "use_location": True
    })


def add_coordinate_encoding(config):
    config.data.train_plugins += [dataloaders.AddSinCosCoordinatesVector()]
    config.data.val_plugins += [dataloaders.AddSinCosCoordinatesVector()]
    config.model.model_args.update({
        "use_location": True,
        "location_size": 4
    })


def set_features(config, features):
    patch_size = config.data.patch_size
    SHAPES = {
        "optical": (patch_size, patch_size, 6),
        "dem": (patch_size, patch_size, 2),
        "co_pol_sar": (patch_size, patch_size, 2),
        "cross_pol_sar": (patch_size, patch_size, 2),
        "in_sar": (patch_size, patch_size, 2),
        "thermal": (patch_size, patch_size, 1)
    }

    config.data.features = features
    config.data.train_sampler_args["features"] = features
    config.data.val_sampler_args["features"] = features

    config.data.input_shapes = {
        feature: SHAPES[feature] for feature in features
    }
    config.data.train_plugins += [
        dataloaders.TileFilter([dataloaders.filters.feature_filter(features)])
    ]
    config.data.val_plugins += [
        dataloaders.TileFilter([dataloaders.filters.feature_filter(features)])
    ]


def set_regions(config, regions):
    config.data.train_plugins += [
        dataloaders.TileFilter([dataloaders.filters.region_filter(regions)])
    ]
    config.data.val_plugins += [
        dataloaders.TileFilter([dataloaders.filters.region_filter(regions)])
    ]


def set_subregions(config, subregions):
    config.data.train_plugins += [
        dataloaders.TileFilter([dataloaders.filters.subregion_filter(subregions)])
    ]
    config.data.val_plugins += [
        dataloaders.TileFilter([dataloaders.filters.subregion_filter(subregions)])
    ]


def set_target_labels(config, labels):
    LABEL_SIZES = {
        "outlines": 2,
        "bright_dark_outlines": 3
    }

    config.data.labels = labels
    config.data.train_sampler_args["labels"] = labels
    config.data.val_sampler_args["labels"] = labels
    config.data.n_outputs = LABEL_SIZES[labels]


def set_weights_path(config, weights_path):
    config.model.weights_path = weights_path


def add_contrast_augmentation(config, gamma_l=0.8, gamma_r=1.2):
    features = config.data.features
    contrast_augmentation = dataloaders.Augmentation(
        [
            dataloaders.transformations.add_gamma_correction(features, gamma_l, gamma_r),
        ]
    )
    config.data.train_plugins += [contrast_augmentation]

    
def add_feature_occlusion(config, max_obstruction_size=192):
    features = config.data.features
    feature_occlusion = dataloaders.Augmentation(
        [
            dataloaders.transformations.groupwise_feature_occlusion(features, max_obstruction_size)
        ]
    )
    config.data.train_plugins += [feature_occlusion]


def add_noise_augmentation(config, shift=0.025, noise=0.005):
    features = config.data.features
    noise_augmentation = dataloaders.Augmentation(
        [
            dataloaders.transformations.add_gaussian_shift(features, [shift for _ in features]),
            dataloaders.transformations.add_gaussian_noise(features, [noise for _ in features]),
        ]
    )
    config.data.train_plugins += [noise_augmentation]


def add_label_smoothing(config, smoothing=0.1):
    label_smoothing = dataloaders.AddLabelSmoothing(smoothing=smoothing)
    config.data.train_plugins += [label_smoothing]
