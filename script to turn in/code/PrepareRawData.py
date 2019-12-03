"""
Script for preparing the image set to be used. Does things like:
    - size all images correctly
    - infer image labels from filenames
    - package data as dict and pickle
"""
import os
import cv2
import importlib
from PIL import Image
import numpy as np
import utils.imageset
import utils.preprocess as pp

fileType = ".pgm"
directory = "/home/cu3us/Tmp/MIT-CBCL-facerec-database/training-synthetic"

height = 64
width = 64
data = []
labels = []
for fileName in os.listdir(directory):
    if fileType in fileName:
        label = fileName[:4]
        # img = cv2.imread(os.path.join(directory, fileName), 0) # Read in as gray-scale
        # img = cv2.resize(img, (115,115))
        # img = img.flatten()
        img = Image.open(os.path.join(directory, fileName)).convert('L')
        img = img.resize((height, width), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float64).reshape((64, 64, 1))
        # img = img.flatten()
        
        data.append(img)
        labels.append(label)

# print(len(data))
# print(len(labels))
# rotation_data, rotation_labels = pp.make_rotations(data, labels, {90})
# print(len(rotation_data))
# print(len(rotation_labels))

# rotation_dataset = {
#     "data": rotation_data,
#     "labels": rotation_labels    
#     }

# utils.imageset.save_dataset(rotation_dataset, "/home/cu3us/Git/TheFaceCrew/data/pickles/only-augmented_train-synthetic_batch_grayscale_2D_rotated")



# print(len(data))
# print(len(labels))
# mirrored_data, mirrored_labels = pp.make_mirrored(data, labels, {-1,0,1})
# print(len(mirrored_data))
# print(len(mirrored_labels))

# mirrored_dataset = {
#     "data": mirrored_data,
#     "labels": mirrored_labels    
#     }

# utils.imageset.save_dataset(mirrored_dataset, "/home/cu3us/Git/TheFaceCrew/data/pickles/only-augmented_train-synthetic_batch_grayscale_2D_mirrored")



# print(len(data))
# print(len(labels))
# blurry_data, blurry_labels = pp.make_blurry(data, labels, 5)
# print(len(blurry_data))
# print(len(blurry_labels))

# blurry_dataset = {
#     "data": blurry_data,
#     "labels": blurry_labels    
#     }

# utils.imageset.save_dataset(blurry_dataset, "/home/cu3us/Git/TheFaceCrew/data/pickles/only-augmented_train-synthetic_batch_grayscale_2D_blurry")

print(len(data))
print(len(labels))
translated_data, translated_labels = pp.make_translations(data, labels)
print(len(translated_data))
print(len(translated_labels))

translated_dataset = {
    "data": translated_data,
    "labels": translated_labels    
    }

utils.imageset.save_dataset(translated_dataset, "/home/cu3us/Git/TheFaceCrew/data/pickles/only-augmented_train-synthetic_batch_grayscale_2D_translated")

