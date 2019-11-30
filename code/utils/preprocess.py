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



def make_blurry(dataset, filter_size):
    """
    Augment dataset by pixelating image (make it blurry)

    Args
        dataset: source dataset
        filter_size: size of kernel to convolve

    Returns
        A new dataset composed of the source dataset and augmentations
    """
    kernel = np.ones((filter_size, filter_size))
    k_width = filter_size
    k_height = filter_size
    was_flattened = (len(dataset[0].shape) == 1)
    augmented_dataset = []

    for image in dataset:
        if was_flattened:
            image = unflatten(image)

        i_height = image.shape[0]
        i_width = image.shape[1]
        blurry_image = np.zeros_like(image.astype(np.int32))

        for y in range(0, i_height - k_height):
            for x in range(0, i_width - k_width):
                # Extract the sub_matrix at current position
                sub_matrix = image[y:y+k_height,x:x+k_width]

                # element-wise multiplication with kernel
                sum_matrix = sub_matrix * kernel
                # sum the matrix and set values of img_out
                asum = np.sum(sum_matrix)
                blurry_image[y,x] = asum

        if was_flattened:
            blurry_image.flatten()

        augmented_dataset.append(blurry_image)
        break

    dataset.extend(augmented_dataset)


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
