import numpy as np
import cv2


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


def groupwise_feature_occlusion(features, max_obstruction_size, p=0.5):
    def transform(patch):
        for feature in features:
            if np.random.random() > p:
                continue
            obstruction_height = int(np.random.random() * max_obstruction_size)
            obstruction_width = int(np.random.random() * max_obstruction_size)
            height, width, _ = patch[feature].shape
            y = np.random.choice(height - obstruction_height)
            x = np.random.choice(width - obstruction_width)
            patch[feature][y:y + obstruction_height, x:x + obstruction_width, :] = 0
    return transform


def feature_occlusion(features, max_obstruction_size, p=0.5):
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
                patch[feature][y:y + obstruction_height, x:x + obstruction_width, feature_idx] = 0
    return transform


def add_gaussian_noise(features, sigmas, p=0.5):
    def transform(patch):
        for feature, sigma in zip(features, sigmas):
            if np.random.random() > p:
                continue
            shape = patch[feature].shape
            noise = np.random.normal(scale=sigma, size=shape)
            patch[feature] += noise
    return transform


def add_gaussian_shift(features, sigmas, p=0.5):
    def transform(patch):
        for feature, sigma in zip(features, sigmas):
            if np.random.random() > p:
                continue
            shape = (patch[feature].shape[-1], )
            shift = np.random.normal(scale=sigma, size=shape)
            patch[feature] += shift
    return transform


def add_gamma_correction(features, gamma_l, gamma_r, p=0.5):
    def transform(patch):
        for feature in features:
            if np.random.random() > p:
                continue
            n_channels = patch[feature].shape[-1]
            gammas = np.random.rand(n_channels) * (gamma_r - gamma_l) + gamma_l
            patch[feature][...] = patch[feature] ** gammas
            patch[feature][np.isnan(patch[feature])] = 0
    return transform


def apply_transformations(patch, transformations):
    if not transformations:
        return 
    for transformation in transformations:
        transformation(patch)
