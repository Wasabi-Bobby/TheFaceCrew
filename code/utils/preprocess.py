# Module for augementing image dataset as a preprocessing step
import math
import numpy as np
import cv2
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

def unflatten(image):
    dim = image.shape[0]
    new_dim = int(math.sqrt(dim))
    image = np.reshape(image, (new_dim, new_dim))

    return image

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
        if was_flattened:
            image = unflatten(image)

        for angle in angles:
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



def make_mirrored(dataset, fliplist):
    """
    Augment dataset by mirroring source images

    Args
        dataset: source dataset
        fliplist: list of desired flips. 
            0: flips around x-axis
            1: flips around y-axis
           -1: flips both
    """
    was_flattened = (len(dataset[0].shape) == 1)    

    augmented_dataset = []
    for image in dataset:
        if was_flattened:
            image = unflatten(image)

        for flip in fliplist:
            altered_image = cv2.flip(image, flip)

            if was_flattened:
                altered_image = altered_image.flatten()

            augmented_dataset.append(altered_image)

    dataset.extend(augmented_dataset)
