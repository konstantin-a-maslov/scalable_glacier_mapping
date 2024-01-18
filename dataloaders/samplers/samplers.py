import numpy as np
import abc


class Sampler(abc.ABC):
    def __init__(self, dataset, patch_size):
        self.dataset = dataset
        self.tiles = list(dataset.keys())
        self.patch_size = patch_size

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def access_tile_group(self, tile):
        return self.dataset[tile]

    def index(self):
        self.n_patches = 0 
        self.regions = set()
        for tile in self.tiles:
            attrs = self.dataset[tile].attrs
            height, width = attrs["height"], attrs["width"]
            region = attrs["region"]
            self.n_patches += (height // self.patch_size) * (width // self.patch_size)
            self.regions.add(region)

    def reset(self):
        pass

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError


class RandomSampler(Sampler):
    def __init__(self, dataset, patch_size, features, labels="outlines"):
        super(RandomSampler, self).__init__(dataset, patch_size)
        self.features = features
        self.labels = labels

    def sample(self):
        self.sample_image()
        self.sample_patch()
        return self.patch

    def sample_image(self):
        tile = np.random.choice(self.tiles)
        self.tile_group = self.access_tile_group(tile)

    def sample_patch(self):
        height, width = self.tile_group.attrs["height"], self.tile_group.attrs["width"]
        self.y = np.random.choice(height - self.patch_size)
        self.x = np.random.choice(width - self.patch_size)
        self.patch = {}
        for feature in self.features:
            feature_patch = self.tile_group[feature][
                self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
            ]
            self.patch[feature] = feature_patch
        self.patch[self.labels] = self.tile_group[self.labels][
            self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
        ].astype(np.double)


class ConsecutiveSampler(Sampler):
    def __init__(self, dataset, patch_size, features, labels="outlines"):
        super(ConsecutiveSampler, self).__init__(dataset, patch_size)
        self.features = features
        self.labels = labels
        self.reset()

    def reset(self):
        self.tile_idx = 0
        self.x = 0
        self.y = 0

    def sample(self):
        self.sample_image()
        height, width = self.tile_group.attrs["height"], self.tile_group.attrs["width"]
        if self.x + self.patch_size > width:
            self.x = 0
            self.y += self.patch_size
        if self.y + self.patch_size > height:
            self.y = 0
            self.x = 0
            self.tile_idx += 1
            self.sample_image()
        self.patch = {}
        for feature in self.features:
            feature_patch = self.tile_group[feature][
                self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
            ]
            self.patch[feature] = feature_patch
        self.patch[self.labels] = self.tile_group[self.labels][
            self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
        ].astype(np.double)
        self.x += self.patch_size
        return self.patch

    def sample_image(self):
        if self.tile_idx >= len(self.tiles):
            self.reset()
        tile = self.tiles[self.tile_idx]
        self.tile_group = self.access_tile_group(tile)


class RAMTileGroup:
    def __init__(self, tile_group, keys=None):
        self.datasets = {
            _: tile_group[_][:] 
            for _ in (tile_group.keys() 
            if keys is None else keys)
        }
        self.attrs = {
            _: tile_group.attrs[_] 
            for _ in tile_group.attrs.keys()
        }

    def __getitem__(self, key):
        return self.datasets[key].copy()


def move_sampler_to_ram(sampler):
    import types

    sampler.cached_tile_groups = {}
    sampler.nonram_access_tile_group = sampler.access_tile_group

    def ram_access_tile_group(self, tile):
        if tile not in self.cached_tile_groups:
            tile_group = self.nonram_access_tile_group(tile)
            ram_tile_group = RAMTileGroup(tile_group)
            self.cached_tile_groups[tile] = ram_tile_group
        return self.cached_tile_groups[tile]

    sampler.access_tile_group = types.MethodType(ram_access_tile_group, sampler)
    return sampler
