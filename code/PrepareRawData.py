"""
Script for preparing the image set to be used. Does things like:
    - size all images correctly
    - infer image labels from filenames
    - package data as dict and pickle
"""
import os
import cv2
import utils.imageset

fileType = ".pgm"
# directory = "/media/cu3us/7E05-F06D/ComputerVision/TermProject/MIT-CBCL-facerec-database/test/"
directory = "/media/cu3us/7E05-F06D/ComputerVision/TermProject/MIT-CBCL-facerec-database/training-synthetic/"
data = []
labels = []
for fileName in os.listdir(directory):
    # print(fileName)
    if fileType in fileName:
        label = fileName[:4]
        img = cv2.imread(os.path.join(directory, fileName), 0) # Read in as gray-scale
        img = cv2.resize(img, (115,115))
        img = img.flatten()
        data.append(img)
        labels.append(label)

print(len(data))
print(data[0].shape)

# cv2.imwrite("tst.png", data[0])
        
dataset = {
    "data": data,
    "labels": labels    
    }

# utils.imageset.save_dataset(dataset, "/media/cu3us/7E05-F06D/ComputerVision/TermProject/TheFaceCrew/data/pickles/test_batch_grayscale_2D")
utils.imageset.save_dataset(dataset, "/media/cu3us/7E05-F06D/ComputerVision/TermProject/TheFaceCrew/data/pickles/train-synthetic_batch_grayscale_flattened")
