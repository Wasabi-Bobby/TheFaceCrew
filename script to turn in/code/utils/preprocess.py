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

def make_rotations(dataset, labels, angles):
    """
    Augment dataset with rotations of source images

    Args
        dataset: source dataset
        angles: list of positive angles (in degrees) for mirroring. Function will use negatives of each angle as well.

    Returns
        A tuple of augmented images and their corresponding labels
    """
    was_flattened = (len(dataset[0].shape) == 1)
    augmented_dataset = []
    augmented_labels = []
    
    for image, label in zip(dataset, labels):
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
            augmented_labels.append(label)
            augmented_labels.append(label)
            
    return (augmented_dataset, augmented_labels)


def make_translations(dataset, labels):
    """
    Augment dataset with translations of source images. Shift image around by 10 pixels

    Args
        dataset: source dataset

    Returns
        A tuple of augmented images and their corresponding labels
    """
    offset = 10
    translations = [
        (0, offset),
        (0, -offset),
        (offset, 0),
        (-offset, 0),
        (-offset, -offset),
        (-offset, offset),
        (offset, -offset),
        (offset, offset)
    ]

    was_flattened = (len(dataset[0].shape) == 1)
    augmented_dataset = []
    augmented_labels = []
    
    for image, label in zip(dataset, labels):
        if was_flattened:
            image = unflatten(image)
            
        height = image.shape[0]
        width = image.shape[1]
        
        for t_x, t_y in translations:
            new_image = np.zeros(image.shape)
            t_mat = np.array([[1,0,t_x],[0,1,t_y],[0,0,1]])

            for x in range(0, width):
                for y in range(0, height):
                    old_coords = np.array([[x],[y],[1]])
                    new_coords = t_mat.dot(old_coords) # translation here

                    if new_coords[0] > 0 and new_coords[0] < width and new_coords[1] > 0 and new_coords[1] < height:
                        new_image[new_coords[1], new_coords[0]] = image[y, x]
                        
            if was_flattened:
                new_image.flatten()
            augmented_dataset.append(new_image)
            augmented_labels.append(label)

    return (augmented_dataset, augmented_labels)


def make_blurry(dataset, labels, filter_size):
    """
    Augment dataset by pixelating image (make it blurry)

    Args
        dataset: source dataset
        filter_size: size of kernel to convolve

    Returns
        A tuple of augmented images and their corresponding labels
    """
    kernel = np.ones((filter_size, filter_size))
    k_width = filter_size
    k_height = filter_size
    border_size = int(filter_size / 2)

    was_flattened = (len(dataset[0].shape) == 1)
    augmented_dataset = []
    augmented_labels = []

    for image, label in zip(dataset, labels):
        if was_flattened:
            image = unflatten(image)

        blurry_image = np.zeros_like(image)
        # pad image
        image = cv2.copyMakeBorder(image, border_size, border_size + 1, border_size, border_size + 1, cv2.BORDER_REPLICATE)
        i_height = image.shape[0]
        i_width = image.shape[1]

        for y in range(0, i_height - k_height):
            for x in range(0, i_width - k_width):
                # Extract the sub_matrix at current position
                sub_matrix = image[y:y+k_height,x:x+k_width]

                # element-wise multiplication with kernel
                sum_matrix = sub_matrix * kernel
                # sum the matrix and set values of img_out
                asum = np.sum(sum_matrix) / (k_width * k_height)
                blurry_image[y,x] = asum

        if was_flattened:
            blurry_image.flatten()

        augmented_dataset.append(blurry_image)
        augmented_labels.append(label)

    return (augmented_dataset, augmented_labels)

    
def make_mirrored(dataset, labels, fliplist):
    """
    Augment dataset by mirroring source images

    Args
        dataset: source dataset
        fliplist: list of desired flips. 
            0: flips around x-axis
            1: flips around y-axis
           -1: flips both

    Returns
        A tuple of augmented images and their corresponding labels
    """
    was_flattened = (len(dataset[0].shape) == 1)    
    augmented_dataset = []
    augmented_labels = []
    
    for image, label in zip(dataset, labels):
        if was_flattened:
            image = unflatten(image)

        for flip in fliplist:
            altered_image = cv2.flip(image, flip)

            if was_flattened:
                altered_image = altered_image.flatten()

            augmented_dataset.append(altered_image)
            augmented_labels.append(label)

    return (augmented_dataset, augmented_labels)
