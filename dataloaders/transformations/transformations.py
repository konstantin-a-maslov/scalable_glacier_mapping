import numpy as np
import cv2


# def groundtruth_to_onehot(n_classes):
#     def transform(patch):
#         groundtruth = patch["outlines"]
#         height, width, _ = groundtruth.shape
#         transformed = np.zeros((height, width, n_classes))
#         for class_index in range(n_classes):
#             transformed[:, :, class_index][groundtruth[:, :, -1] == class_index] = 1
#         patch["outlines"] = transformed
#     return transform


# def stretch_features_to_range(mins, maxs):
#     def transform(patch):
#         for feature in patch:
#             if feature == "outlines":
#                 continue
#             if len(patch[feature].shape) != 3:
#                 continue
#             original = patch[feature]
#             transformed = (original - mins[feature]) / (maxs[feature] - mins[feature])
#             patch[feature] = transformed
#     return transform


def random_vertical_flip(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            transformed = np.flip(original, axis=0)
            patch[feature] = transformed
    return transform


def random_horizontal_flip(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            transformed = np.flip(original, axis=1)
            patch[feature] = transformed
    return transform


def random_rotation(p=0.75):
    def transform(patch):
        if np.random.random() > p:
            return
        k = np.random.choice([1, 2, 3])
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            transformed = np.rot90(original, k, axes=(1, 0))
            patch[feature] = transformed
    return transform


def crop_and_scale(patch_size, scale=0.8, p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        scale_coef = np.random.random() * (1 - scale) + scale
        new_size = int(scale_coef * patch_size)
        y = np.random.choice(patch_size - new_size)
        x = np.random.choice(patch_size - new_size)
        for feature in patch:
            if len(patch[feature].shape) != 3:
                continue
            original = patch[feature]
            crop = original[y:y + new_size, x:x + new_size, :]
            _, _, depth = crop.shape
            transformed = np.empty((patch_size, patch_size, depth))
            for channel_idx in range(depth):
                transformed[:, :, channel_idx] = cv2.resize(
                    crop[:, :, channel_idx], (patch_size, patch_size), 
                    interpolation=cv2.INTER_NEAREST
                )
            patch[feature] = transformed
    return transform


def set_region_to_unknown(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        location = np.zeros_like(patch["location"])
        location[-1] = 1
        patch["location"] = location
    return transform


def groupwise_feature_occlusion(features, max_obstruction_size, p=0.5, nodata=0):
    def transform(patch):
        for feature in features:
            if np.random.random() > p:
                continue
            obstruction_height = int(np.random.random() * max_obstruction_size)
            obstruction_width = int(np.random.random() * max_obstruction_size)
            height, width, _ = patch[feature].shape
            y = np.random.choice(height - obstruction_height)
            x = np.random.choice(width - obstruction_width)
            patch[feature][y:y + obstruction_height, x:x + obstruction_width, :] = nodata
    return transform


def feature_occlusion(features, max_obstruction_size, p=0.5, nodata=0):
    def transform(patch):
        for feature in features:
            height, width, n_features = patch[feature].shape
            for feature_idx in range(n_features):
                if np.random.random() > p:
                    continue
                obstruction_height = int(np.random.random() * max_obstruction_size)
                obstruction_width = int(np.random.random() * max_obstruction_size)
                y = np.random.choice(height - obstruction_height)
                x = np.random.choice(width - obstruction_width)
                patch[feature][y:y + obstruction_height, x:x + obstruction_width, feature_idx] = nodata
    return transform


def add_gaussian_noise(features, sigmas, p=0.5, nodata=0):
    def transform(patch):
        # if np.random.random() > p:
        #     return
        for feature, sigma in zip(features, sigmas):
            if np.random.random() > p:
                continue
            shape = patch[feature].shape
            noise = np.random.normal(scale=sigma, size=shape)
            mask = (patch[feature] == nodata)
            patch[feature] += noise
            patch[feature][mask] = nodata
    return transform


def add_gaussian_shift(features, sigmas, p=0.5, nodata=0):
    def transform(patch):
        # if np.random.random() > p:
        #     return
        for feature, sigma in zip(features, sigmas):
            if np.random.random() > p:
                continue
            shape = (patch[feature].shape[-1], )
            shift = np.random.normal(scale=sigma, size=shape)
            mask = (patch[feature] == nodata)
            patch[feature] += shift
            patch[feature][mask] = nodata
    return transform


def add_gamma_correction(features, gamma_l, gamma_r, p=0.5, nodata=0, eps=1e-6):
    def transform(patch):
        # if np.random.random() > p:
        #     return
        for feature in features:
            if np.random.random() > p:
                continue
            n_channels = patch[feature].shape[-1]
            gammas = np.random.rand(n_channels) * (gamma_r - gamma_l) + gamma_l
            mask = (patch[feature] == nodata)
            # not tested!!! adjustment with shifting by min
            min_val = np.min(patch[feature][~mask])
            patch[feature] += (min_val + eps)
            patch[feature][...] = patch[feature] ** gammas
            # patch[feature][np.isnan(patch[feature])] = 0
            patch[feature] -= (min_val + eps)
            patch[feature][mask] = nodata
    return transform


def add_dem_shift(magnitude, p=0.5, nodata=0):
    def transform(patch):
        if np.random.random() > p:
            return
        shift = (np.random.rand() - 0.5) * 2 * magnitude
        mask = (patch["dem"][..., 0] == nodata)
        patch["dem"][..., 0] += shift
        patch["dem"][mask, 0] = nodata
    return transform


def subtract_dem_mean(nodata=0):
    def transform(patch):
        dem_mean = np.mean(patch["dem"][..., 0])
        mask = (patch["dem"][..., 0] == nodata)
        patch["dem"][..., 0] -= dem_mean
        patch["dem"][mask, 0] = nodata
    return transform


def apply_transformations(patch, transformations):
    if not transformations:
        return 
    for transformation in transformations:
        transformation(patch)
