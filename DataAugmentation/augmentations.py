# Module for augementing image dataset as a preprossecing step
import numpy as np

'''
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
'''

# Normalize dataseet
# For each pixel, subtract the dataset mean and
# divide by standard deviation
# Note: We may want to normalize features as well
def normalize_dataset(dataset):
    return dataset
