import configs.data as data
import configs.models.glavitu as model
import configs.training as training
import configs.finetuning as finetuning
import configs.swa as swa


cli_args = None


use_region_encoding = False
use_coordinate_encoding = False
use_sincos_coordinate_encoding = False

#####################################################################
# import dataloaders
# import dataloaders.filters
# data.train_plugins += [
#     # dataloaders.TileFilter([lambda tile_group: tile_group.attrs["region"] == "HMA"])
#     dataloaders.TileFilter([dataloaders.filters.region_filter({"ALP"})])
# ]
# data.val_plugins += [
#     # dataloaders.TileFilter([lambda tile_group: tile_group.attrs["region"] == "HMA"])
#     dataloaders.TileFilter([dataloaders.filters.region_filter({"ALP"})])
# ]
#####################################################################

#####################################################################
if [use_region_encoding, use_coordinate_encoding, use_sincos_coordinate_encoding].count(True) > 1:
    raise NotImplementedError()


import dataloaders
import dataloaders.transformations


if use_region_encoding:
    data.train_plugins += [
        dataloaders.AddRegionVector(),
        dataloaders.Augmentation([
            dataloaders.transformations.set_region_to_unknown(p=0.1)
        ])
    ]
    data.val_plugins += [
        dataloaders.AddRegionVector(),
        dataloaders.RepeatWithMandatoryTransformations([
            dataloaders.transformations.set_region_to_unknown(p=1.0)
        ])
    ]
    model.model_args.update({
        "use_location": True
    })

if use_coordinate_encoding:
    data.train_plugins += [dataloaders.AddCoordinatesVector()]
    data.val_plugins += [dataloaders.AddCoordinatesVector()]
    model.model_args.update({
        "use_location": True,
        "location_size": 2
    })

if use_sincos_coordinate_encoding:
    data.train_plugins += [dataloaders.AddSinCosCoordinatesVector()]
    data.val_plugins += [dataloaders.AddSinCosCoordinatesVector()]
    model.model_args.update({
        "use_location": True,
        "location_size": 4
    })
