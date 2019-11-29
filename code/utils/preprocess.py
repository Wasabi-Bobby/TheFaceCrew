# Module for augementing image dataset as a preprocessing step
import numpy as np

"""
Augmentations to implement:
- normalization
- rotation
- translation
- pixelation
- mirroring
? change backgrounds
? add random noise


Look into:
- AutoAugment
"""



def make_rotations(dataset, angles):
"""
Augment dataset with rotations of source images

Args
    dataset: source dataset
    angles: list of positive angles for mirroring. Function will use negatives of each angle as well.

Returns
    A new dataset composed of the source dataset and augmentations
"""
    return dataset



def make_translations(dataset):
"""
Augment dataset with translations of source images

Args
    dataset: source dataset

Returns
    A new dataset composed of the source dataset and augmentations
"""    
    return dataset



def make_pixelation(dataset):
"""
Augment dataset by pixelating image (make it blurry)

Args
    dataset: source dataset

Returns
    A new dataset composed of the source dataset and augmentations
"""    
    return dataset



def make_mirrored(dataset):
"""
Augment dataset by mirroring source images

Args
    dataset: source dataset

Returns
    A new dataset composed of the source dataset and augmentations

Notes:
    ?Mirror both x and y axis?
"""
    return dataset
