import numpy as np
import tensorflow as tf
import dataloaders.transformations


def get_tile_features(tile_data, features):
    tile_features = {}
    for feature in features + ["outlines"]:
        feature_arr = np.array(tile_data[feature])
        if len(feature_arr.shape) == 2:
            feature_arr = feature_arr[..., np.newaxis]
        tile_features[feature] = feature_arr
    return tile_features


# def pad_features_to_patch_size(tile_features, patch_size, mins):
#     height, width, _ = tile_features["groundtruth"].shape
#     target_height = patch_size * int(height // patch_size + 1)
#     target_width = patch_size * int(width // patch_size + 1)
#     pad_height = (target_height - height) // 2
#     pad_width = (target_width - width) // 2
#     for feature in tile_features:
#         feature_arr = tile_features[feature]
#         if feature != "groundtruth":
#             padded = mins[feature][np.newaxis, np.newaxis, :]
#             padded = np.repeat(padded, target_height, axis=0)
#             padded = np.repeat(padded, target_width, axis=1)
#         else:
#             padded = np.zeros((target_height, target_width, 1))
#         padded[pad_height:pad_height + height, pad_width:pad_width + width, :] = feature_arr
#         tile_features[feature] = padded
#     return tile_features, (pad_height, pad_width)


# def group_features(tile_features_list, groups):
#     grouped = {}
#     for group in groups:
#         group_batch = []
#         for tile_features in tile_features_list:
#             group_arrs = []
#             for feature in groups[group]:
#                 group_arrs.append(tile_features[feature])
#             group_stack = np.concatenate(group_arrs, axis=-1)
#             group_batch.append(group_stack)
#         grouped[group] = np.array(group_batch)
#     groundtruth_batch = []
#     for tile_features in tile_features_list:
#         groundtruth_batch.append(tile_features["groundtruth"])
#     grouped["groundtruth"] = np.array(groundtruth_batch)
#     return grouped


# class BaseDataLoader(tf.keras.utils.Sequence):
#     def index_dataset(self):
#         raise NotImplementedError()

#     def sample(self):
#         raise NotImplementedError()

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         batch = self.sample()
#         batch = group_features(batch, self.feature_groups)
#         batch_x = {_: batch[_] for _ in batch if _ != "groundtruth"}
#         batch_y = batch["groundtruth"]
#         return batch_x, batch_y


# class RandomSamplingDataLoader(BaseDataLoader):
#     def __init__(
#         self, dataset, tile_filters, feature_groups, batch_size, patch_size, len, pad_mins, 
#         transformations=None
#     ):
#         self.dataset = dataset
#         self.tile_filters = tile_filters
#         self.feature_groups = feature_groups
#         self.features = []
#         for group in feature_groups:
#             self.features += feature_groups[group]
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.len = len
#         self.pad_mins = pad_mins
#         self.transformations = transformations
#         self.index_dataset()

#     def index_dataset(self):
#         self.tiles = []
#         for tile in self.dataset.keys():
#             tile_data = self.dataset[tile]
#             filters = [filter(tile, tile_data) for filter in self.tile_filters]
#             if all(filters):
#                 self.tiles.append(tile)

#     def sample(self):
#         batch = []
#         for _ in range(self.batch_size):
#             image = self.sample_image()
#             patch = self.sample_patch(image)
#             dataloaders.transformations.apply_transformations(
#                 patch, self.transformations
#             )
#             batch.append(patch)
#         return batch

#     def sample_image(self):
#         tile = np.random.choice(self.tiles)
#         tile_data = self.dataset[tile]
#         image = get_tile_features(tile_data, self.features)
#         image, _ = pad_features_to_patch_size(image, self.patch_size, self.pad_mins)
#         return image

#     def sample_patch(self, image):
#         height, width, _ = image["groundtruth"].shape
#         y = np.random.choice(height - self.patch_size)
#         x = np.random.choice(width - self.patch_size)
#         patch = {}
#         for feature in image:
#             feature_patch = image[feature][y:y + self.patch_size, x:x + self.patch_size, :]
#             patch[feature] = feature_patch
#         return patch


# class RandomSamplingDataLoaderPrecomputed(BaseDataLoader):
#     def __init__(self, dataset, batch_size, patch_size, len, augmentation=None):
#         self.dataset = dataset
#         self.tiles = list(dataset.keys())
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.len = len
#         self.augmentation = augmentation

#     def __getitem__(self, idx):
#         batch_list = self.sample()
#         features = batch_list[0].keys()
#         batch = {}
#         for feature in features:
#             feature_list = []
#             for item in batch_list:
#                 feature_list.append(item[feature])
#             batch[feature] = np.array(feature_list)
#         batch_x = {_: batch[_] for _ in batch if _ != "groundtruth"}
#         batch_y = batch["groundtruth"]
#         # return batch_x, batch_y
#         return batch_x, [batch_y, batch_y]

#     def sample(self):
#         batch = []
#         for _ in range(self.batch_size):
#             image = self.sample_image()
#             patch = self.sample_patch(image)
#             dataloaders.transformations.apply_transformations(
#                 patch, self.augmentation
#             )
#             batch.append(patch)
#         return batch

#     def sample_image(self):
#         tile = np.random.choice(self.tiles)
#         image = self.dataset[tile]
#         return image

#     def sample_patch(self, image):
#         height, width, _ = image["groundtruth"].shape
#         y = np.random.choice(height - self.patch_size)
#         x = np.random.choice(width - self.patch_size)
#         patch = {}
#         for feature in image.keys():
#             feature_patch = image[feature][y:y + self.patch_size, x:x + self.patch_size, :]
#             patch[feature] = feature_patch
#         return patch


# class ConsecutiveSamplingDataLoader(BaseDataLoader):
#     def __init__(
#         self, dataset, tile_filters, feature_groups, batch_size, patch_size, pad_mins, 
#         transformations=None
#     ):
#         self.dataset = dataset
#         self.tile_filters = tile_filters
#         self.feature_groups = feature_groups
#         self.features = []
#         for group in feature_groups:
#             self.features += feature_groups[group]
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.len = 0
#         self.pad_mins = pad_mins
#         self.transformations = transformations
#         self.index_dataset()
#         self.reset()

#     def index_dataset(self):
#         self.tiles = []
#         for tile in self.dataset.keys():
#             tile_data = self.dataset[tile]
#             filters = [filter(tile, tile_data) for filter in self.tile_filters]
#             if all(filters):
#                 self.tiles.append(tile)
#                 height, width, _ = tile_data["groundtruth"].shape
#                 n_patches = int(np.ceil(height / self.patch_size)) * \
#                     int(np.ceil(width / self.patch_size))
#                 self.len += n_patches

#     def sample(self):
#         batch = []
#         for _ in range(self.batch_size):
#             patch = self.sample_patch()
#             dataloaders.transformations.apply_transformations(
#                 patch, self.transformations
#             )
#             batch.append(patch)
#         return batch

#     def sample_image(self):
#         if self.tile_index >= len(self.tiles):
#             self.reset()
#         tile = self.tiles[self.tile_index]
#         tile_data = self.dataset[tile]
#         image = get_tile_features(tile_data, self.features)
#         image, _ = pad_features_to_patch_size(image, self.patch_size, self.pad_mins)
#         return image

#     def sample_patch(self):
#         image = self.sample_image()
#         height, width, _ = image["groundtruth"].shape
#         if self.x + self.patch_size > width:
#             self.x = 0
#             self.y += self.patch_size
#         if self.y + self.patch_size > height:
#             self.y = 0
#             self.x = 0
#             self.tile_index += 1
#             image = self.sample_image()
#         patch = {}
#         for feature in image:
#             feature_patch = image[feature][self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :]
#             patch[feature] = feature_patch
#         self.x += self.patch_size
#         return patch

#     def reset(self):
#         self.tile_index = 0
#         self.x = 0
#         self.y = 0


# class ConsecutiveSamplingDataLoaderPrecomputed(BaseDataLoader):
#     def __init__(self, dataset, batch_size, patch_size, augmentation=None):
#         self.dataset = dataset
#         self.tiles = list(dataset.keys())
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.len = 0
#         self.augmentation = augmentation
#         self.index_dataset()
#         self.reset()

#     def __getitem__(self, idx):
#         batch_list = self.sample()
#         features = batch_list[0].keys()
#         batch = {}
#         for feature in features:
#             feature_list = []
#             for item in batch_list:
#                 feature_list.append(item[feature])
#             batch[feature] = np.array(feature_list)
#         batch_x = {_: batch[_] for _ in batch if _ != "groundtruth"}
#         batch_y = batch["groundtruth"]
#         # return batch_x, batch_y
#         return batch_x, [batch_y, batch_y]

#     def index_dataset(self):
#         for tile in self.tiles:
#             tile_data = self.dataset[tile]
#             height, width, _ = tile_data["groundtruth"].shape
#             n_patches = (height // self.patch_size) * (width // self.patch_size)
#             self.len += n_patches

#     def sample(self):
#         batch = []
#         for _ in range(self.batch_size):
#             patch = self.sample_patch()
#             dataloaders.transformations.apply_transformations(
#                 patch, self.augmentation
#             )
#             batch.append(patch)
#         return batch

#     def sample_image(self):
#         if self.tile_index >= len(self.tiles):
#             self.reset()
#         tile = self.tiles[self.tile_index]
#         image = self.dataset[tile]
#         return image

#     def sample_patch(self):
#         image = self.sample_image()
#         height, width, _ = image["groundtruth"].shape
#         if self.x + self.patch_size > width:
#             self.x = 0
#             self.y += self.patch_size
#         if self.y + self.patch_size > height:
#             self.y = 0
#             self.x = 0
#             self.tile_index += 1
#             image = self.sample_image()
#         patch = {}
#         for feature in image.keys():
#             feature_patch = image[feature][self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :]
#             patch[feature] = feature_patch
#         self.x += self.patch_size
#         return patch

#     def reset(self):
#         self.tile_index = 0
#         self.x = 0
#         self.y = 0


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, sampler, plugins, batch_size, labels="outlines", len_factor=1):
        self.sampler = sampler
        self.plugins = plugins
        self.batch_size = batch_size
        self.labels = labels
        self.len_factor = len_factor
        self.__index()

    def __index(self):
        self.sampler.set_dataloader(self)
        self.before_indexing_plugins = []
        self.after_indexing_plugins = []
        self.on_sampling_plugins = []
        self.on_finalising_plugins = []
        for plugin in self.plugins:
            self.add_plugin(plugin)
        for plugin in self.before_indexing_plugins:
            plugin.before_indexing(self.sampler)
        self.sampler.index()
        for plugin in self.after_indexing_plugins:
            plugin.after_indexing(self.sampler)
    
    def __index_plugin(self, plugin):
        if plugin.has_before_indexing_behaviour:
            self.before_indexing_plugins.append(plugin)
        if plugin.has_after_indexing_behaviour:
            self.after_indexing_plugins.append(plugin)
        if plugin.has_on_sampling_behaviour:
            self.on_sampling_plugins.append(plugin)
        if plugin.has_on_finalising_behaviour:
            self.on_finalising_plugins.append(plugin)

    def add_plugin(self, plugin):
        plugin.set_dataloader(self)
        self.__index_plugin(plugin)

    def __len__(self):
        return self.sampler.n_patches // self.batch_size * self.len_factor

    def __getitem__(self, idx):
        if idx == 0:
            self.sampler.reset()
        self.batch_list = []
        self.sample_idx = 0
        while self.sample_idx < self.batch_size:
            sample = self.sampler.sample()
            for plugin in self.on_sampling_plugins:
                sample = plugin.on_sampling(sample)
            self.batch_list.append(sample)
            self.sample_idx += 1
        batch_x, batch_y = self.__reformat(self.batch_list)
        for plugin in self.on_finalising_plugins:
            batch_x, batch_y = plugin.on_finalising(batch_x, batch_y)
        return batch_x, batch_y
        
    def __reformat(self, batch_list):
        features = batch_list[0].keys()
        batch = {}
        for feature in features:
            feature_list = []
            for item in batch_list:
                feature_list.append(item[feature])
            batch[feature] = np.array(feature_list)
        batch_x = {_: batch[_] for _ in batch if _ != self.labels}
        batch_y = batch[self.labels]
        return batch_x, batch_y
