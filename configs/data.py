import dataloaders
import dataloaders.transformations
import h5py
import os


train_dataset_path = ...
val_dataset_path = ...
test_dataset_path = ...

predictions_dir = ...

n_outputs = 2
patch_size = 384
batch_size = 8

features = ["optical", "dem"]
labels = "outlines"
input_shapes = {
    "optical": (patch_size, patch_size, 6),
    "dem": (patch_size, patch_size, 2)
}

train_sampler_builder = dataloaders.RandomSampler
train_sampler_args = {
    "dataset": h5py.File(train_dataset_path, "r"), 
    "patch_size": patch_size, 
    "features": features,
    "labels": labels
}
train_plugins = [
    dataloaders.Augmentation([
        dataloaders.transformations.random_vertical_flip(),
        dataloaders.transformations.random_horizontal_flip(),
        dataloaders.transformations.random_rotation(),
        dataloaders.transformations.crop_and_scale(patch_size=patch_size)
    ])
]

val_sampler_builder = dataloaders.ConsecutiveSampler
val_sampler_args = {
    "dataset": h5py.File(val_dataset_path, "r"), 
    "patch_size": patch_size, 
    "features": features,
    "labels": labels
}
val_plugins = []
