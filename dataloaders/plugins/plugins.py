import abc
import dataloaders.transformations
import numpy as np


class Plugin(abc.ABC):
    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def get_sampler(self):
        if not self.dataloader:
            raise ValueError()
        return self.dataloader.sampler

    def __init_subclass__(cls, **kwargs):
        cls.has_before_indexing_behaviour = not (cls.before_indexing == Plugin.before_indexing)
        cls.has_after_indexing_behaviour = not (cls.after_indexing == Plugin.after_indexing)
        cls.has_on_sampling_behaviour = not (cls.on_sampling == Plugin.on_sampling)
        cls.has_on_finalising_behaviour = not (cls.on_finalising == Plugin.on_finalising)

    def before_indexing(self, sampler):
        pass

    def after_indexing(self, sampler):
        pass

    def on_sampling(self, sample):
        return sample

    def on_finalising(self, batch_x, batch_y):
        return batch_x, batch_y


class TileFilter(Plugin):
    def __init__(self, filters):
        self.filters = filters

    def before_indexing(self, sampler):
        dataset = sampler.dataset
        tiles = sampler.tiles
        filtered_tiles = []
        for tile in tiles:
            tile_group = dataset[tile]
            if self.apply_filters(tile_group):
                filtered_tiles.append(tile)
        sampler.tiles = filtered_tiles

    def apply_filters(self, tile_group):
        filter_outputs = [filter(tile_group) for filter in self.filters]
        return all(filter_outputs)


class RemapNoData(Plugin):
    def __init__(self, features, old_nodata, new_nodata):
        self.features = features
        self.old_nodata = old_nodata
        self.new_nodata = new_nodata

    def on_sampling(self, sample):
        for feature in self.features:
            mask = (sample[feature] == self.old_nodata)
            sample[feature][mask] = self.new_nodata
        return sample


class Augmentation(Plugin):
    def __init__(self, transformations):
        self.transformations = transformations

    def on_sampling(self, sample):
        dataloaders.transformations.apply_transformations(
            sample, self.transformations
        )
        return sample


class AddDeepSupervision(Plugin):
    def __init__(self, n_branches=2):
        self.n_branches = n_branches

    def on_finalising(self, batch_x, batch_y):
        batch_y = [batch_y for _ in range(self.n_branches)]
        return batch_x, batch_y


class AddLabelSmoothing(Plugin):
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def on_finalising(self, batch_x, batch_y):
        n_classes = batch_y.shape[-1]
        batch_y = batch_y * (1 - self.smoothing) + self.smoothing / n_classes
        return batch_x, batch_y


class RepeatWithMandatoryTransformations(Plugin):
    def __init__(self, transformations):
        self.transformations = transformations

    def after_indexing(self, sampler):
        sampler.n_patches *= 2

    def on_sampling(self, sample):
        sample_copy = {key: value.copy() for key, value in sample.items()}
        dataloaders.transformations.apply_transformations(
            sample_copy, self.transformations
        )
        self.dataloader.batch_list.append(sample_copy)
        self.dataloader.sample_idx += 1
        return sample


class AddRegionVector(Plugin):
    def after_indexing(self, sampler):
        regions = list(sampler.regions)
        regions.sort()
        self.n_regions = len(regions)
        self.mapping = {region: idx for idx, region in enumerate(regions)}

    def on_sampling(self, sample):
        sampler = self.get_sampler()
        region = sampler.tile_group.attrs["region"]
        sample["location"] = self.get_region_vector(region)
        return sample

    def get_region_vector(self, region):
        region_vector = np.zeros(self.n_regions + 1)
        region_idx = self.mapping[region]
        region_vector[region_idx] = 1
        return region_vector


class AddCoordinatesVector(Plugin):
    def on_sampling(self, sample):
        sampler = self.get_sampler()
        attrs = sampler.tile_group.attrs
        x, y, patch_size = sampler.x, sampler.y, sampler.patch_size
        x = x - attrs["padding_width"] + patch_size / 2
        y = attrs["height"] - y - attrs["padding_height"] - patch_size / 2
        xmin, xmax, ymin, ymax = attrs["xmin"], attrs["xmax"], attrs["ymin"], attrs["ymax"]
        height, width = attrs["original_height"], attrs["original_width"]
        long = x * (xmax - xmin) / width + xmin
        lat = y * (ymax - ymin) / height + ymin
        sample["location"] = self.get_coordinates_vector(lat, long)
        return sample

    def get_coordinates_vector(self, lat, long):
        coordinates_vector = np.array([lat / 90, long / 180])
        return coordinates_vector


class AddSinCosCoordinatesVector(AddCoordinatesVector):
    def get_coordinates_vector(self, lat, long):
        lat, long = lat * np.pi / 180, long * np.pi / 180
        coordinates_vector = np.array([np.sin(lat), np.cos(lat), np.sin(long), np.cos(long)])
        return coordinates_vector
