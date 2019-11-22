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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
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
        # output = self.fc1(output)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

# Used https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/ to get an idea on how to save my torch model
def train_and_test(data_matrix, data_labels_array, test_matrix, test_labels_array):
    model = Net()
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

    test_matrix = test_matrix[:1000]
    test_matrix = np.transpose(test_matrix, (0, 3, 1, 2))
    test_labels_array = test_labels_array[:1000]
    test_acc_list = []
    test_batch = 1000

    test_loader = convert_np_to_tensor(test_matrix, test_labels_array, False, test_batch)

    model.eval()
    # For how to grab the confusion matrix https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/
    confusion_matrix = np.zeros([10, 10], int)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

        print('Test Accuracy of the model on the 1000 test images: {} %'.format((correct / total) * 100))

    return confusion_matrix

# https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
# For appending numpy into tensor flow data set
def convert_np_to_tensor(data, labels, shuffle, batch_size):
    train_data = []
    for i in range(len(data)):
        train_data.append([data[i], labels[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)

    return trainloader


def Fisher_LDF(data_matrix, data_labels_array, test_matrix, test_labels_array):

    H, mean_array, covariance_array = Fisher_Training(data_matrix, data_labels_array)

    conf_matrix = Fisher_Test(test_matrix, test_labels_array, H, mean_array, covariance_array)

    return conf_matrix

def Fisher_Training(data_matrix, data_labels_array):
    data_matrix = data_matrix.reshape((10000, (32*32*3), 1))[:10000]
    data_labels_array = data_labels_array[:10000]

    print(np.shape(data_matrix))
    print(np.shape(data_labels_array))

    # 0-10 array for class mean
    class_mean_array    = np.zeros(shape=(10, (32 * 32 * 3), 1))
    overall_class_mean = np.zeros(shape=((32 * 32 * 3), 1))

    # 0-10 array for each class' covariance
    class_covariance_array = np.zeros(shape=(10, (32 * 32 * 3), (32 * 32 * 3)))

    # Getting the N for every item in the batch (we don't use unique)
    unique, counts = np.unique(data_labels_array, return_counts=True)
    print(counts)

    # Setting class mean of the label (min label is 0, max is 9 so it is fine)
    for i in range(len(data_matrix)):
        label = data_labels_array[i]
        class_mean_array[label] = class_mean_array[label] + data_matrix[i]

    # Finalizing each class mean and overall mean
    for i in range(len(class_mean_array)):
        class_mean_array[i] = class_mean_array[i] / counts[i]
        overall_class_mean = overall_class_mean + class_mean_array[i]

    overall_class_mean = np.divide(overall_class_mean, len(class_mean_array))

    data_class_list = []
    print("Class covariance calculating")
    for i in range(len(counts)):
        data_class_list.append(data_matrix[np.argwhere(data_labels_array==i)].reshape(counts[i], (32 * 32 * 3)))
        i_cov = np.cov((data_class_list[i]).T)
        class_covariance_array[i] = i_cov

    # Used to be the old way of setting up the class covariance array (took 25 minutes previously...)
    # for i in range(len(data_matrix)):
    #     if i % 100 == 0:
    #         print("Class covariance currently at " + str(i))
    #     label = data_labels_array[i]
    #     subtracted_feature = np.subtract(data_matrix[i], class_mean_array[label])
    #     class_covariance_array[label] = class_covariance_array[label] + np.dot(subtracted_feature, subtracted_feature.T)
    #
    # for i in range(len(class_covariance_array)):
    #     class_covariance_array[i] = class_covariance_array[i] / counts[i]

    print("B calculation begins")

    B = np.zeros(shape=((32 * 32 * 3), (32 * 32 * 3)))

    for i in range(len(class_mean_array)):
        mean_difference = class_mean_array[i] - overall_class_mean
        B = B + np.dot(mean_difference, mean_difference.T)

    print("A calculation begins")

    A = np.zeros(shape=((32 * 32 * 3), (32 * 32 * 3)))
    for i in range(len(class_covariance_array)):
        A = A + class_covariance_array[i]

    print("Computing H")
    A_inv = np.linalg.inv(A)
    A_inv = np.nan_to_num(A_inv)
    eig_val, eig_vect = np.linalg.eig(np.dot(A_inv, B))
    eig_val = np.abs(eig_val)

    # Checking result
    rank = len(class_mean_array) - 1
    max_vals_array = np.flip(eig_val.argsort()[-rank:])
    H = []
    for i in range(len(max_vals_array)):
        H.append(eig_vect[:, max_vals_array[i]])

    H = np.array(H).T

    fisher_mean_array       = np.zeros(shape=(10, rank, 1))
    fisher_covariance_array = np.zeros(shape=(10, rank, rank))



    # Calculating both fisher variables
    for i in range(len(fisher_mean_array)):
        fisher_mean_array[i] = np.dot(H.T, class_mean_array[i])
        fisher_covariance_array[i] = np.dot(np.dot(H.T, class_covariance_array[i]), H)

    print("Done Training!")

    return H, fisher_mean_array, fisher_covariance_array

def Fisher_Test(test_matrix, test_labels_array, H, mean_array, covariance_array):
    test_matrix = test_matrix.reshape((10000, (32 * 32 * 3), 1))[:1000]
    test_labels_array = test_labels_array[:1000]

    print(np.shape(test_matrix))
    print(np.shape(test_labels_array))
    print(np.shape(covariance_array[0]))

    class_confusion_matrix = np.zeros(shape=(10, 10))

    num_classes = 10

    print("Calculating test confusion matrix using Fisher LDF and melanohbis distance as classifier")

    for i in range(1000):
        d_matrix = np.zeros(shape=(10, 1))
        f_space = np.dot(H.T, test_matrix[i])
        for j in range(num_classes):
            f_sub_m = f_space - mean_array[j]
            d_result = np.dot(np.dot(f_sub_m.T, np.linalg.pinv(covariance_array[j])), f_sub_m)
            d_matrix[j] = d_result

        label = test_labels_array[i]
        lowest_distance = np.argmin(d_matrix)
        # print("Lowest distance = " + str(lowest_distance))
        # print("Label = " + str(label))
        class_confusion_matrix[label][lowest_distance] = class_confusion_matrix[label][lowest_distance] + 1

    current_correct_count = 0

    for i in range(10):
        current_correct_count = current_correct_count + class_confusion_matrix[i][i]

    total_accuracy = (current_correct_count / len(test_labels_array)) * 100
    print("Accuracy = " + str(total_accuracy))
    print("Error = " + str(100 - total_accuracy))

    return class_confusion_matrix


def main():
    data_directory = get_image_directory()

    data_matrix, data_labels_array = load_data(data_directory + "\\data_batch_1")
    test_matrix, test_labels_array = load_data(data_directory + "\\test_batch")
    # a = np.zeros(shape=(9, 9))
    # np.linalg.pinv(a)
    # a = np.zeros(shape=(10, 9, 9))
    # np.linalg.inv(a[0])

    conf_matrix = Fisher_LDF(data_matrix, data_labels_array, test_matrix, test_labels_array)
    print(conf_matrix)

    nn_conf_matrix = train_and_test(data_matrix, data_labels_array, test_matrix, test_labels_array)
    print(nn_conf_matrix)

    return


main()
