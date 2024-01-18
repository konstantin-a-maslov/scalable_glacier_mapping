def region_filter(regions):
    def filter(tile_group):
        return (tile_group.attrs["region"] in regions)
    return filter


def subregion_filter(subregions):
    def filter(tile_group):
        return (tile_group.attrs["subregion"] in subregions)
    return filter


def feature_filter(features):
    def filter(tile_group):
        for feature in features:
            if feature not in tile_group.keys():
                return False
        return True
    return filter
