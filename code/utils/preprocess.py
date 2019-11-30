# Module for augementing image dataset as a preprocessing step
import math
import numpy as np
from scipy import ndimage

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
        angles: list of positive angles (in degrees) for mirroring. Function will use negatives of each angle as well.
    """
    was_flattened = (len(dataset[0].shape) == 1)

    augmented_dataset = []
    
    for image in dataset:
        for angle in angles:
            if was_flattened:
                dim = image.shape[0]
                new_dim = int(math.sqrt(dim))
                image = np.reshape(image, (new_dim, new_dim))
                
            rotated_pos = ndimage.rotate(image, angle)
            rotated_neg = ndimage.rotate(image, -angle)

            if was_flattened:
                rotated_pos = rotated_pos.flatten()
                rotated_neg = rotated_neg.flatten()

            augmented_dataset.append(rotated_pos)
            augmented_dataset.append(rotated_neg)
            
    dataset.extend(augmented_dataset)


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
