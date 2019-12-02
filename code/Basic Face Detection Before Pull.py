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
import cv2
import os


def get_image_directory():

    # Grabbing the directory path of this code
    dir_path = os.getcwd()

    # Grabbing the folder name in case that was changed
    code_folder_name = os.path.basename(dir_path)

    # This should work regardless of current code folder name ( Assuming the data is located in the data folder! )
    image_directory = dir_path[:-len(code_folder_name)] + "data"

    return image_directory


# Creates image when given the vector of the image, the directory to place the image and the image name
def create_image(image_vector, image_directory, image_name):
    image = Image.fromarray(image_vector, 'L')
    # Need to save to a certain image path
    image_path = image_directory + "\\" + image_name
    image.save(image_path)
    image.show()


class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=2),
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
        self.fc1 = nn.Linear(64, 10)
        # This layer isn't actually used since it gives 10 outputs in the first layer
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.size(0), -1)
        output = F.relu(self.fc1(output))
        # See above comment why this is commented out
        output = self.fc2(output)
        return output

# Used https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/ to get an idea on how to save my torch model
def train_and_test(data_matrix, data_labels_array, test_matrix, test_labels_array, num_epochs=10, batch_size=20, learning_rate=0.01, mod_name="model"):
    model = BasicNet()
    # How I originally get the data is a double and not a float so I need to .float() a few times
    model = model.float()

    num_epochs = num_epochs
    batch_size = batch_size
    learning_rate = learning_rate

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
        # print("Accuracy = " + str(correct / total * 100))

    # Wanted to save this for future reference for myself
    # TODO: Read this and make sure you run this in admin or else this won't work
    torch.save(model.state_dict(), get_image_directory() + "/models/" + mod_name)
    print('Saved model parameters to disk.')

    #test_matrix = test_matrix[:1000]
    test_matrix = np.transpose(test_matrix, (0, 3, 1, 2))
    #test_labels_array = test_labels_array[:1000]
    test_acc_list = []
    test_batch = batch_size

    test_loader = convert_np_to_tensor(test_matrix, test_labels_array, True, test_batch)

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

    print(confusion_matrix)

    return confusion_matrix

# https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
# For appending numpy into tensor flow data set
def convert_np_to_tensor(data, labels, shuffle, batch_size):
    train_data = []
    for i in range(len(data)):
        train_data.append([data[i], labels[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)

    return trainloader

# This does some preprocessing by making the size 64 by 64
def load_images_from_list_RGB(directory, list):
    height = 64
    width = 64
    #raw_list = np.array(Image.open(directory + "/" + list[0]), dtype=float)
    raw_list = np.zeros(shape=(len(list), width, height, 3), dtype=np.float64)
    for i in range(len(list)):
        #if i > 0:
        #print(directory + "/" + list[i])
        item_location = directory + "/" + list[i]
        im = Image.open(directory + "/" + list[i]).convert('RGB')
        #print(item_location)

        #im.show()

        im = im.resize((height, width), Image.ANTIALIAS)

        #im.show()

        #print(im.size)

        next_im = np.array(im, dtype=np.float64)
        raw_list[i] = next_im

    # print(raw_list)
    # print(np.shape(raw_list))

    return raw_list

# This does some preprocessing by making the size 64 by 64
# This also grabs them as grayscale
def load_images_from_list_Grayscale(directory, list):
    height = 64
    width = 64
    #raw_list = np.array(Image.open(directory + "/" + list[0]), dtype=float)
    raw_list = np.zeros(shape=(len(list), width, height, 1), dtype=np.float64)
    for i in range(len(list)):
        #if i > 0:
        #print(directory + "/" + list[i])
        item_location = directory + "/" + list[i]
        im = Image.open(directory + "/" + list[i]).convert('L')
        #print(item_location)

        #im.show()

        im = im.resize((height, width), Image.ANTIALIAS)

        #im.show()

        #print(im.size)

        next_im = np.array(im, dtype=np.float64).reshape((64, 64, 1))
        raw_list[i] = next_im

    # print(raw_list)
    # print(np.shape(raw_list))

    return raw_list

def grab_all_files_in_directory(directory):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    return onlyfiles


def grab_labels_from_list(list):
    label_list = []

    for i in range(len(list)):
        data = list[i].split("_")
        label = (int(data[0][-1]))
        label_list.append(label)


    return np.array(label_list)

def normalize(x):
    """
    https://stackoverflow.com/questions/49429734/trying-to-normalize-python-image-getting-error-rgb-values-must-be-in-the-0-1
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

# Sets standard deviation of data to one and shifts to have a mean of zero
def standardize_data(data):
    '''
    Currently does not work ( Going to have to make my own, which just requires a loop of some sort
    :param data: np array
    :return: standardized np array
    '''

    standardized_data = (data - np.mean(data)) / np.std(data)
    return standardized_data


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# http://dlib.net/face_landmark_detection.py.html where I got this from for grabbing faces and important features to feed into network
def grab_boundingbox_face():
    import dlib
    import glob

    data_directory = get_image_directory()
    predictor_path = data_directory + "/dlib-models/shape_predictor_68_face_landmarks.dat"
    faces_folder_path = data_directory + "/training-dlib"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

        win.add_overlay(dets)

        #dlib.hit_enter_to_continue()


def main():
    data_directory = get_image_directory()
    training_image_directory = data_directory + "\\training"
    test_image_directory = data_directory + "\\test"

    training_list = grab_all_files_in_directory(training_image_directory)
    test_list = grab_all_files_in_directory(test_image_directory)

    train_labels = grab_labels_from_list(training_list)
    train_data = load_images_from_list_RGB(training_image_directory, training_list)

    test_labels = grab_labels_from_list(test_list)
    test_data   = load_images_from_list_RGB(test_image_directory, test_list)

    grab_boundingbox_face()


    # print(np.shape(test_labels))
    # print(np.shape(test_data))

    #train_and_test(train_data, train_labels, test_data, test_labels)

    train_data = normalize(train_data)
    test_data = normalize(test_data)

    train_data = standardize_data(train_data)
    test_data = standardize_data(test_data)



    # Batch size = 5
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,  5, 0.001, "lowLR_1_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  5, 0.001, "lowLR_1_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  5, 0.001, "lowLR_2_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  5, 0.001, "lowLR_3_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 5, 0.001, "lowLR_4_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   5, 0.01, "medLR_0_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  5, 0.01, "medLR_1_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  5, 0.01, "medLR_2_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  5, 0.01, "medLR_3_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 5, 0.01, "medLR_4_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   5, 0.1, "highLR_0_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  5, 0.1, "highLR_1_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  5, 0.1, "highLR_2_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  5, 0.1, "highLR_3_0_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 5, 0.1, "highLR_4_0_std")
    #
    # # Batch size = 25
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   25, 0.001, "lowLR_0_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  25, 0.001, "lowLR_1_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  25, 0.001, "lowLR_2_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  25, 0.001, "lowLR_3_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 25, 0.001, "lowLR_4_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   25, 0.01, "medLR_0_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  25, 0.01, "medLR_1_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  25, 0.01, "medLR_2_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  25, 0.01, "medLR_3_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 25, 0.01, "medLR_4_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   25, 0.1, "highLR_0_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  25, 0.1, "highLR_1_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  25, 0.1, "highLR_2_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  25, 0.1, "highLR_3_1_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 25, 0.1, "highLR_4_1_std")
    #
    # # Batch size = 50
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   50, 0.001, "lowLR_0_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  50, 0.001, "lowLR_1_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  50, 0.001, "lowLR_2_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  50, 0.001, "lowLR_3_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 50, 0.001, "lowLR_4_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   50, 0.01, "medLR_0_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  50, 0.01, "medLR_1_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  50, 0.01, "medLR_2_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  50, 0.01, "medLR_3_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 50, 0.01, "medLR_4_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 5,   50, 0.1, "highLR_0_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 10,  50, 0.1, "highLR_1_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 25,  50, 0.1, "highLR_2_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 50,  50, 0.1, "highLR_3_2_std")
    # train_and_test(train_data, train_labels, test_data, test_labels, 100, 50, 0.1, "highLR_4_2_std")

    return


main()
