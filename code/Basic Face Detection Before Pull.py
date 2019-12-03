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


def get_data_directory():

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

class AdvNet(nn.Module):

    def __init__(self):
        super(AdvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=2),
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
        self.fc1 = nn.Linear(4160, 100)
        # This layer isn't actually used since it gives 10 outputs in the first layer
        self.fc2 = nn.Linear(100, 10)

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
    torch.save(model.state_dict(), get_data_directory() + "/models/" + mod_name)
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

        print('Test Accuracy of the model on the ' + str(len(test_matrix)) + ' test images: {} %'.format((correct / total) * 100))

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


def load_images_from_list(directory, image_name_list, exclusion_list, rect_list):
    '''
    # This does some preprocessing by making the size 64 by 64
    # This also grabs them as Grayscale
    :param directory: directory where images are located
    :param image_name_list: Names of all images
    :param exclusion_list: List of images to not grab
    :param rect_list: Four ints saying where the bounding box is (
    :return: A list of images that is cropped around the bounding box
    '''
    height = 128
    width = 128
    # raw_list = np.array(Image.open(directory + "/" + list[0]), dtype=float)
    raw_list = np.zeros(shape=(len(rect_list), width, height, 1), dtype=np.float64)
    current_rect_index = 0
    for i in range(len(image_name_list)):
        if exclusion_list[i] != 1:
            rect = rect_list[current_rect_index]
            item_location = directory + "/" + image_name_list[i]
            im = Image.open(item_location).convert('L')
            im = im.crop((rect.left(), rect.top(), rect.right(), rect.bottom()))

            #im.show()
            im = im.resize((height, width), Image.ANTIALIAS)
            #im.show()

            save_new_image(im, get_data_directory() + "/processed-training-images/", image_name_list[i])

            # print(im.size)

            next_im = np.array(im, dtype=np.float64).reshape((width, height, 1))
            raw_list[current_rect_index] = next_im
            current_rect_index += 1

    # print(raw_list)
    # print(np.shape(raw_list))

    return raw_list


def load_preprocessed_images_from_list(folder_name, image_name_list):
    '''
    # This does some preprocessing by making the size 64 by 64
    # This also grabs them as Grayscale
    :param directory: directory where images are located
    :param image_name_list: Names of all images
    :param exclusion_list: List of images to not grab
    :param rect_list: Four ints saying where the bounding box is (
    :return: A list of images that is cropped around the bounding box
    '''
    height = 128
    width = 128

    raw_list = np.zeros(shape=(len(image_name_list), width, height, 1), dtype=np.float64)

    for i in range(len(image_name_list)):
        item_location = get_data_directory() + folder_name + "/" + image_name_list[i]
        im = Image.open(item_location).convert('L')
        im = im.resize((height, width), Image.ANTIALIAS)

        next_im = np.array(im, dtype=np.float64).reshape((width, height, 1))
        raw_list[i] = next_im

    return raw_list


def save_new_image(image, image_directory, image_name):
    print("Saving the following file at the path")
    print(str(image_directory + image_name))
    image.save(image_directory + image_name, 'JPEG')

    return


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


def grab_labels_from_list_exclusion(list, exclusion_list):
    label_list = []

    for i in range(len(list)):
        # If it able to grab a face from it
        if exclusion_list[i] != 1:
            data = list[i].split("_")
            label = (int(data[0][-1]))
            label_list.append(label)

    return np.array(label_list)


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


def grab_boundingbox_face(image_directory):
    '''
    http://dlib.net/face_landmark_detection.py.html where I got this from
    Grabbs faces and important features to feed into network
    :param training_image_directory: Directory of the images
    :return: bounding boxes of every face and 68 features of every face as well as what faces were not grabbed
    '''
    import dlib
    import glob

    data_directory = get_data_directory()
    predictor_path = data_directory + "/dlib-models/shape_predictor_68_face_landmarks.dat"
    faces_folder_path = image_directory

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # win = dlib.image_window()

    shape_list = []
    bounding_face_list = []
    # If it is a 1 in the list then it was excluded and a 0 means included
    # Going to be used when grabbing labels and when figuring out what to bound
    exclusion_list = []

    current_index = 0

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        # Do this now before we try finding a face
        exclusion_list.append(1)

        #win.clear_overlay()
        #win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            # Found a face so make this zero
            exclusion_list[current_index] = 0
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

            bounding_face_list.append(d)



            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))

            shape_list.append(shape)
            # Draw the face landmarks on the screen.
            #win.add_overlay(shape)

        current_index += 1


        #win.add_overlay(dets)

    bounding_face_list = np.array(bounding_face_list)
    shape_list = np.array(shape_list)
    # Need to reshape to be (len, 1) not (len,)
    # bounding_face_list = bounding_face_list.reshape(len(bounding_face_list))
    # shape_list = shape_list.reshape(len(shape_list))

    # TODO: Comment these out before turning in
    print(np.shape(bounding_face_list))
    print(np.shape(shape_list))
    print(exclusion_list)
    print(np.shape(bounding_face_list))

    return bounding_face_list, shape_list, exclusion_list


def grab_shape_list(face_folder_name):
    '''
    Grabs the key features of each image in image list to draw each feature to feed into the NN
    :param image_list: Images to feed into this
    :param bounding_face_list: Bounding box of faces
    :return: a list with a bunch of shapes
    '''
    import numpy as np
    import dlib
    import glob

    width = 128
    height = 128
    shape_list = []
    faces_folder_path = get_data_directory() + face_folder_name
    counter = 0
    # win = dlib.image_window()

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        predictor_path = get_data_directory() + "/dlib-models/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)

        img = np.zeros((width, height), dtype=np.uint8)

        dlib_img = dlib.load_rgb_image(f)
        # win.clear_overlay()
        # win.set_image(dlib_img)

        # The predictor goes slightly out of bounds, gotta reel this back to not go outta bounds
        shape = predictor(dlib_img, dlib.rectangle(0,0, width, height))
        # win.add_overlay(shape)
        # TODO: UNCOMMENT WHEN YOU WANT TO KNOW HOW IT SHOULD LOOK
        #dlib.hit_enter_to_continue()
        shape = shape_to_np(shape)


        '''
        The predictor predicts values outside of the picture, so we must bound this with a threshold
        '''
        super_threshold_indices = shape >= height
        shape[super_threshold_indices] = height - 1

        jaw = shape[0:17]
        right_eyebrow = shape[17:22]
        left_eyebrow = shape[22:27]
        nose = shape[27:36]
        right_eye = shape[36:42]
        left_eye = shape[42:48]
        outer_mouth = shape[48:61]
        inner_mouth = shape[61:68]


        img = draw_shape(img, jaw)
        img = draw_shape(img, right_eyebrow)
        img = draw_shape(img, left_eyebrow)
        img = draw_shape(img, nose)
        img = draw_shape(img, right_eye)
        img = draw_shape(img, left_eye)
        img = draw_shape(img, outer_mouth)
        img = draw_shape(img, inner_mouth)

        img = img.reshape(width, height)
        img = np.rot90(img)
        img = np.rot90(img)
        img = np.rot90(img)
        # img = np.rot90(img, axes=(-2,-1))

        # for i in range(128):
        #     print(img[i])

        # pil_img = Image.fromarray(img, 'L')
        # pil_img.show()
        # win.set_image(img)

        img = img.reshape(width, height, 1)
        f_img = img.astype(np.float64)

        # if np.array_equal(f_img, img):
        #     print("Same same")

        shape_list.append(f_img)
        counter += 1

    shape_list = np.array(shape_list).reshape(len(shape_list), width, height, 1)

    return shape_list


def draw_shape(img, shape):
    '''
    Takes in shapes as input and
    :param img: image to modify
    :param shape: shape to draw ( a series of x and y paired coordinates)
    :return: a new image that has the shape drawn
    '''
    from skimage.draw import line_aa

    # print("Drawing the next shape")
    for i in range(len(shape) - 1):
        # print("First shape x and y : " + str(shape[i]))
        # print("Second shape x and y : " + str(shape[i+1]))
        rr, cc, val = line_aa(shape[i][0], shape[i][1], shape[i + 1][0], shape[i + 1][1])
        # print(val)
        # print(rr)
        # print(cc)
        img[rr, cc] = val * 255

    return img


def create_and_train_model(data_matrix, data_labels_array, num_epochs=10, batch_size=20, learning_rate=0.01, mod_name="model"):
    model = AdvNet()
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
    torch.save(model.state_dict(), get_data_directory() + "/models/" + mod_name)
    print('Saved model parameters to disk.')

    return


def create_data_list(bounding_face_list, shape_list, image_directory, exclusion_list):
    '''
    :param bounding_face_list: Bounding box of faces
    :param image_directory: Directory to grab images
    :param exclusion_list: What images we should ignore
    :param shape_list: shape key points of faces
    :return: returns a list with the two combined and with shape list being more descript
    '''
    list_images = grab_all_files_in_directory(image_directory)
    image_list = load_images_from_list(image_directory, list_images, exclusion_list, bounding_face_list)
    shape_list = grab_shape_list("/processed-training-images/")

    print(np.shape(image_list))
    print(np.shape(shape_list))
    # Combining cropped face and shape images into one array
    data = []
    for i in range(len(image_list)):
        data.append(np.vstack((image_list[i], shape_list[i])))

    data = np.array(data).reshape(len(data), 256, 128, 1)
    print(np.shape(data))

    labels = grab_labels_from_list_exclusion(list_images, exclusion_list)

    return data, labels


def create_test_list(bounding_face_list, image_directory, exclusion_list):
    '''
    :param bounding_face_list: Bounding box of faces
    :param image_directory: Directory to grab images
    :param exclusion_list: What images we should ignore
    :return: returns a list with the two combined and with shape list being more descript
    '''
    list_images = grab_all_files_in_directory(image_directory)
    image_list = load_test_images_from_list(image_directory, list_images, exclusion_list, bounding_face_list)
    shape_list = grab_shape_test_list("/processed-test-images/")

    print(np.shape(image_list))
    print(np.shape(shape_list))
    # Combining cropped face and shape images into one array
    data = []
    for i in range(len(image_list)):
        data.append(np.vstack((image_list[i], shape_list[i])))

    data = np.array(data).reshape(len(data), 256, 128, 1)
    print(np.shape(data))

    labels = grab_labels_from_list_exclusion(list_images, exclusion_list)

    return data, labels


def grab_boundingbox_face_test(image_directory):
    '''
    http://dlib.net/face_landmark_detection.py.html where I got this from
    Grabs faces and important features to feed into network from the test images
    :param training_image_directory: Directory of the images
    :return: bounding boxes of every face and 68 features of every face as well as what faces were not grabbed
    '''
    import dlib
    import glob

    data_directory = get_data_directory()
    predictor_path = data_directory + "/dlib-models/shape_predictor_68_face_landmarks.dat"
    faces_folder_path = image_directory

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    shape_list = []
    bounding_face_list = []
    # If it is a 1 in the list then it was excluded and a 0 means included
    # Going to be used when grabbing labels and when figuring out what to bound
    exclusion_list = []

    current_index = 0

    for f in glob.glob(os.path.join(faces_folder_path, "*.pgm")):
        print("Processing file: {}".format(f))
        # TODO: FIX THIS
        img = Image.open(f).convert('L')
        img = np.array(img, dtype=np.uint8)
        print(np.shape(img))
        #img = dlib.load_grayscale_image(f)
        # img = dlib.load_rgb_image(f)
        # Do this now before we try finding a face
        exclusion_list.append(1)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            # Found a face so make this zero
            exclusion_list[current_index] = 0
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

            bounding_face_list.append(d)



            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))

            shape_list.append(shape)
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

        current_index += 1


        win.add_overlay(dets)

    bounding_face_list = np.array(bounding_face_list)
    shape_list = np.array(shape_list)
    # Need to reshape to be (len, 1) not (len,)
    # bounding_face_list = bounding_face_list.reshape(len(bounding_face_list))
    # shape_list = shape_list.reshape(len(shape_list))

    # TODO: Comment these out before turning in
    print(np.shape(bounding_face_list))
    print(np.shape(shape_list))
    print(exclusion_list)
    print(np.shape(bounding_face_list))

    return bounding_face_list, shape_list, exclusion_list


def load_test_images_from_list(directory, image_name_list, exclusion_list, rect_list):
    '''
    # This does some preprocessing by making the size 64 by 64
    # This also grabs them as Grayscale
    :param directory: directory where images are located
    :param image_name_list: Names of all images
    :param exclusion_list: List of images to not grab
    :param rect_list: Four ints saying where the bounding box is (
    :return: A list of images that is cropped around the bounding box
    '''
    height = 128
    width = 128
    # raw_list = np.array(Image.open(directory + "/" + list[0]), dtype=float)
    raw_list = np.zeros(shape=(len(rect_list), width, height, 1), dtype=np.float64)
    current_rect_index = 0
    for i in range(len(image_name_list)):
        if exclusion_list[i] != 1:
            rect = rect_list[current_rect_index]
            item_location = directory + "/" + image_name_list[i]
            im = Image.open(item_location).convert('L')
            im = im.crop((rect.left(), rect.top(), rect.right(), rect.bottom()))

            #im.show()
            im = im.resize((height, width), Image.ANTIALIAS)
            #im.show()

            save_new_image(im, get_data_directory() + "/processed-test-images/", image_name_list[i])

            # print(im.size)

            next_im = np.array(im, dtype=np.float64).reshape((width, height, 1))
            raw_list[current_rect_index] = next_im
            current_rect_index += 1

    # print(raw_list)
    # print(np.shape(raw_list))

    return raw_list


def grab_shape_test_list(face_folder_name):
    '''
    Grabs the key features of each image in image list to draw each feature to feed into the NN
    :param image_list: Images to feed into this
    :param bounding_face_list: Bounding box of faces
    :return: a list with a bunch of shapes
    '''
    import numpy as np
    import dlib
    import glob

    width = 128
    height = 128
    shape_list = []
    faces_folder_path = get_data_directory() + face_folder_name
    counter = 0
    # TODO: Disable win
    #win = dlib.image_window()

    for f in glob.glob(os.path.join(faces_folder_path, "*.pgm")):
        predictor_path = get_data_directory() + "/dlib-models/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)

        img = np.zeros((width, height), dtype=np.uint8)

        dlib_img = dlib.load_rgb_image(f)
        #win.clear_overlay()
        #win.set_image(dlib_img)

        # The predictor goes slightly out of bounds, gotta reel this back to not go outta bounds
        shape = predictor(dlib_img, dlib.rectangle(0,0, width, height))
        #win.add_overlay(shape)
        # TODO: UNCOMMENT WHEN YOU WANT TO KNOW HOW IT SHOULD LOOK
        #dlib.hit_enter_to_continue()
        shape = shape_to_np(shape)


        '''
        The predictor predicts values outside of the picture, so we must bound this with a threshold
        '''
        super_threshold_indices = shape >= height
        shape[super_threshold_indices] = height - 1

        jaw = shape[0:17]
        right_eyebrow = shape[17:22]
        left_eyebrow = shape[22:27]
        nose = shape[27:36]
        right_eye = shape[36:42]
        left_eye = shape[42:48]
        outer_mouth = shape[48:61]
        inner_mouth = shape[61:68]


        img = draw_shape(img, jaw)
        img = draw_shape(img, right_eyebrow)
        img = draw_shape(img, left_eyebrow)
        img = draw_shape(img, nose)
        img = draw_shape(img, right_eye)
        img = draw_shape(img, left_eye)
        img = draw_shape(img, outer_mouth)
        img = draw_shape(img, inner_mouth)

        img = img.reshape(width, height)
        img = np.rot90(img)
        img = np.rot90(img)
        img = np.rot90(img)
        # img = np.rot90(img, axes=(-2,-1))

        # for i in range(128):
        #     print(img[i])

        pil_img = Image.fromarray(img, 'L')
        # save_new_image(save_new_image, "shape-test-images", name)
        # save_new_image(pil_img, )
        # pil_img.show()
        #win.set_image(img)

        img = img.reshape(width, height, 1)
        f_img = img.astype(np.float64)

        # if np.array_equal(f_img, img):
        #     print("Same same")

        shape_list.append(f_img)
        counter += 1

    shape_list = np.array(shape_list).reshape(len(shape_list), width, height, 1)

    return shape_list


def create_test_preprocessed_list(folder_name, image_list):
    loaded_image_list = load_preprocessed_images_from_list(folder_name, image_list)
    shape_list = grab_shape_test_list("/processed-test-images/")

    loaded_image_list = normalize(loaded_image_list)
    shape_list = normalize(shape_list)

    print(np.shape(loaded_image_list))
    print(np.shape(shape_list))
    # Combining cropped face and shape images into one array
    data = []
    for i in range(len(loaded_image_list)):
        data.append(np.vstack((loaded_image_list[i], shape_list[i])))

    data = np.array(data).reshape(len(data), 256, 128, 1)
    print(np.shape(data))

    labels = grab_labels_from_list(image_list)

    return data, labels



# def create_data_preprocessed_list(folder_name, image_list):
#     '''
#
#     :param image_directory:
#     :param image_list:
#     :return:
#     '''
#     loaded_image_list = load_preprocessed_images_from_list(folder_name, image_list)
#     shape_list = grab_shape_list("/processed-training-images/")
#
#     print(np.shape(loaded_image_list))
#     print(np.shape(shape_list))
#     # Combining cropped face and shape images into one array
#     data = []
#     for i in range(len(loaded_image_list)):
#         data.append(np.vstack((loaded_image_list[i], shape_list[i])))
#
#     data = np.array(data).reshape(len(data), 256, 128, 1)
#     print(np.shape(data))
#
#     labels = grab_labels_from_list(image_list)
#
#     return data, labels

def create_data_preprocessed_list(folder_name, image_list):
    '''

    :param image_directory:
    :param image_list:
    :return:
    '''
    loaded_image_list = load_preprocessed_images_from_list(folder_name, image_list)
    shape_list = grab_shape_list("/processed-training-images/")

    loaded_image_list = normalize(loaded_image_list)
    shape_list = normalize(shape_list)

    print(np.shape(loaded_image_list))
    print(np.shape(shape_list))
    # Combining cropped face and shape images into one array
    data = []
    for i in range(len(loaded_image_list)):
        data.append(np.vstack((loaded_image_list[i], shape_list[i])))

    data = np.array(data).reshape(len(data), 256, 128, 1)
    print(np.shape(data))

    labels = grab_labels_from_list(image_list)

    return data, labels


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


def test_and_report(test_matrix, test_labels_array, model_name, batch_size):
    device = torch.device('cpu')
    model = AdvNet()
    model.load_state_dict(torch.load(get_data_directory() + "/models/" + model_name, map_location=device))
    # test_matrix = test_matrix[:1000]
    test_matrix = np.transpose(test_matrix, (0, 3, 1, 2))
    # test_labels_array = test_labels_array[:1000]
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

        print('Test Accuracy of the model on the ' + str(len(test_matrix)) + ' test images: {} %'.format((correct / total) * 100))

    print(confusion_matrix)


    return



def main():
    data_directory = get_data_directory()
    training_image_directory = data_directory + "\\training"
    training_preprocessed_image_directory = data_directory + "\\processed-training-images"
    # TODO: Replace function testing with data testing
    # function_testing_directory = data_directory + "/function-testing"
    test_image_directory = data_directory + "\\test"
    test_preprocessed_image_directory = data_directory + "\\processed-test-images"

    # Grabing training information
    # Bounding box will preprocess
    # train_bounding_face_list, train_shape_list, train_exclusion_list = grab_boundingbox_face(training_image_directory)
    # train_data , train_labels = create_data_list(train_bounding_face_list, train_shape_list, training_image_directory, train_exclusion_list)

    '''
        Change these values for different tests
    '''
    num_epoch      = 10
    batch_size     = 10
    learning_rate  =  0.001


    #TODO: CHANGE MODEL NAME WITH EACH RUN

    model_name = "BEST-MODEL_"


    print("Retrieving training data list")
    train_pre_list = grab_all_files_in_directory(training_preprocessed_image_directory)
    print("Creating data and label")
    train_pre_data, train_pre_label = create_data_preprocessed_list("\\processed-training-images", train_pre_list)
    print("Starting to train")
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "0"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "1"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "2"))

    num_epoch = 100
    batch_size = 3
    learning_rate = 0.0001

    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "3"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "4"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "5"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "6"))

    num_epoch = 100
    batch_size = 9
    learning_rate = 0.00001

    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "7"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "8"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "9"))

    num_epoch = 100
    batch_size = 10
    learning_rate = 0.1

    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "10"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "11"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "12"))

    num_epoch = 100
    batch_size = 15
    learning_rate = 0.01

    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "10"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "11"))
    create_and_train_model(train_pre_data, train_pre_label, num_epoch, batch_size, learning_rate, (model_name + "12"))

    # Preprocesses the data
    #test_bounding_face_list, test_shape_list, test_exclusion_list = grab_boundingbox_face_test(test_image_directory)
    #test_data, test_labels = create_test_list(test_bounding_face_list, test_image_directory, test_exclusion_list)
    print("Creating test data and label")
    test_data, test_labels = create_test_preprocessed_list("\\processed-test-images", grab_all_files_in_directory(test_preprocessed_image_directory))
    print("Testing")
    for i in range(13):
        test_and_report(test_data, test_labels, (model_name + str(i)), batch_size)


    # bounding_face_list, shape_list = grab_boundingbox_face(test_image_directory)
    # test_data   = create_data_list(bounding_face_list, shape_list)
    # test_labels = grab_labels_from_list(test_list)


    '''
        Below is the way we create and test the models that are very naive (did this on purpose to show some better results)
        Best possible result I saw was about 20%, but that might have been pure luck or random
        The most it does to process images is just resize (so no cropping at all) 
    '''
    # data_directory = get_data_directory()
    # training_image_directory = data_directory + "\\training"
    # test_image_directory = data_directory + "\\test"
    #
    # training_list = grab_all_files_in_directory(training_image_directory)
    # test_list = grab_all_files_in_directory(test_image_directory)
    #
    # train_labels = grab_labels_from_list(training_list)
    # train_data = load_images_from_list_RGB(training_image_directory, training_list)
    #
    # test_labels = grab_labels_from_list(test_list)
    # test_data   = load_images_from_list_RGB(test_image_directory, test_list)
    #
    # # print(np.shape(test_labels))
    # # print(np.shape(test_data))
    #
    # train_data = normalize(train_data)
    # test_data = normalize(test_data)
    #
    # train_data = standardize_data(train_data)
    # test_data = standardize_data(test_data)



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
