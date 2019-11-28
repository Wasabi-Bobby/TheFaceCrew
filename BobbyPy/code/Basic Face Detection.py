import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import scipy
from scipy.sparse.linalg import eigs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import collections
import glob
import cv2
import os

# Used only the knowledge of how to convert this dict file into a manageable numpy file from here
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py
# Took since this downloading and managing is not really the main concept I am being challenged on

# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _images_per_file


def convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = convert_images(raw_images)

    return images, cls

#################################
# Back to my code now
#################################


def create_vector_using_image(image_directory, image_name):
    # Setting to the path of the cat image
    image_path = image_directory + "\\" + image_name
    image = Image.open(image_path, 'r').convert('L')

    # Creating an array to pixel color information at each x and y coordinate
    image_vector = np.array(image)

    return image_vector


def get_image_directory():

    # Grabbing the directory path of this code
    dir_path = os.getcwd()

    # Grabbing the folder name in case that was changed
    code_folder_name = os.path.basename(dir_path)

    # This should work regardless of current code folder name ( Assuming the data is located in the data folder! )
    image_directory = dir_path[:-len(code_folder_name)] + "data"

    return image_directory


# Give this an int to decide which of the number data sets to return
# Second parameter should be an int from 1-1000 saying how many of those images to return
def get_number_image(which_number, number_of_digits):
    # Getting path to number which number directory
    image_dir = get_image_directory() + "\\DigitDataset\\" + str(which_number)

    # Going to be converted to numpy matrix later
    image_list = []
    # Offset is laid out for 0 starting at 9001, so need a bit of if else logic to combat this
    if which_number != 0:
        image_name_offset = (which_number - 1) * 1000 + 1
    else:
        image_name_offset = 9001

    # Actually grabbing images
    for i in range(number_of_digits):
        image_name = "image" + str(image_name_offset) + ".png"
        image_list.append(create_vector_using_image(image_dir, image_name))
        # Next image
        image_name_offset += 1

    image_matrix = np.array(image_list)

    return image_matrix


# Creates image when given the vector of the image, the directory to place the image and the image name
def create_image(image_vector, image_directory, image_name):
    image = Image.fromarray(image_vector, 'L')
    # Need to save to a certain image path
    image_path = image_directory + "\\" + image_name
    image.save(image_path)
    image.show()


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Output of layer3 is 64 channels of 7x7 images
        self.fc1 = nn.Linear(64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

# Used https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/ to get an idea on how to save my torch model
def train_and_test(data_matrix, data_labels_array, test_matrix, test_labels_array):
    model = BasicNet()
    # How I originally get the data is a double and not a float so I need to .float() a few times
    model = model.float()

    # num_epochs = 5
    # batch_size = 4
    # learning_rate = 0.001

    num_epochs = 1
    batch_size = 50
    learning_rate = 0.01

    # Train the model
    total_step = len(data_matrix)
    loss_list = []
    acc_list = []

    print(np.shape(data_matrix))
    data_matrix = np.transpose(data_matrix, (0, 3, 1, 2))
    train_loader = convert_np_to_tensor(data_matrix, data_labels_array, True, batch_size)

    # This automatically includes softmax activation, so no need to define that
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        print("Current Epoch: " + str(epoch + 1))
        for i, (images, labels) in enumerate(train_loader):
            #Forward pass + loss calculation
            outputs = model(images.float())
            loss = loss_function(outputs, labels.long())
            loss_list.append(loss.item())

            # Zero the gradient
            optimizer.zero_grad()

            # Back prop + optimize
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

    # Wanted to save this for future reference for myself
    # torch.save(model.state_dict(), get_image_directory())
    # print('Saved model parameters to disk.')

    # test_matrix = test_matrix[:1000]
    # test_matrix = np.transpose(test_matrix, (0, 3, 1, 2))
    # test_labels_array = test_labels_array[:1000]
    # test_acc_list = []
    # test_batch = 1000
    #
    # test_loader = convert_np_to_tensor(test_matrix, test_labels_array, False, test_batch)

    # model.eval()
    # # For how to grab the confusion matrix https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/
    # confusion_matrix = np.zeros([10, 10], int)
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         outputs = model(images.float())
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         for i, l in enumerate(labels):
    #             confusion_matrix[l.item(), predicted[i].item()] += 1
    #
    #     print('Test Accuracy of the model on the 1000 test images: {} %'.format((correct / total) * 100))

    #return confusion_matrix
    return

# https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
# For appending numpy into tensor flow data set
def convert_np_to_tensor(data, labels, shuffle, batch_size):
    train_data = []
    for i in range(len(data)):
        train_data.append([data[i], labels[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)

    return trainloader


def load_images_from_list(directory, list):
    image_list = np.zeros(shape=(1, 576, 768, 3), dtype=float)
    raw_list = []

    for i in range(len(list)):
        #print(directory + "/" + list[i])
        im = Image.open(directory + "/" + list[i])
        image_list = np.concatenate(image_list, (np.array(im, dtype=float)))

    print(image_list)
    print(np.shape(image_list))

    return image_list


def grab_all_files_in_directory(directory):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    return onlyfiles


def grab_labels_from_list(list):
    label_list = []

    for i in range(len(list)):
        data = list[i].split("_")
        label_list.append(data[0][-1])


    return np.array(label_list)


def main():
    data_directory = get_image_directory()
    training_image_directory = data_directory + "\\training"
    test_image_directory = data_directory + "\\test"

    training_list = grab_all_files_in_directory(training_image_directory)

    train_labels = grab_labels_from_list(training_list)
    train_data = load_images_from_list(training_image_directory, training_list)

    train_and_test(train_data, train_labels, train_data, train_labels)

    return


main()
