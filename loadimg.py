import os.path
import os
import cv2
import numpy as np


# Main script containing the image loading part. In keras-nn-example.py an example of a CNN using images loaded in that way is given.

# Root directory containing the images to classify, folder are used as labels
# train --> cat --> img1, img2...imgN
#       --> dog --> img1, img2...imgN
#
# test  --> cat --> img1, img2...imgN
#       --> dog --> img1, img2...imgN


def getimagedataandlabels(root_dir, image_w, image_h, verbose, mode):

    X_data=[]
    Y_data=[]
    classes_from_directories = []  # to determine the classes from the root folder structure automatically

    for directory, subdirectories, files in os.walk(root_dir):
        if verbose:
            print(directory)
        subdirectories.sort()
        for subdirectory in subdirectories:
            if verbose:
                print(subdirectory)
            classes_from_directories.append(subdirectory)
        for file in files:
            # print(file)
            # print(directory)
            if file != '.DS_Store':  # fix for MAC...
                imagepath = os.path.join(directory, file)
                current_image_class_splitt = imagepath.split('/')
                current_image_class = current_image_class_splitt[len(current_image_class_splitt) - 2]
                img = cv2.imread(imagepath)
                img = cv2.resize(img, (image_w, image_h))
                img_rescale = np.true_divide(img, 255.)  # imshow expects values in the range [0, 1]
                if mode == "custom_seq":
                    img_rescale = np.expand_dims(img, axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
                X_data.append(np.asarray(img_rescale, dtype="float32"))
                Y_data.append(current_image_class)
                #print imagepath

    return np.array(X_data), np.array(Y_data), classes_from_directories

if __name__ == '__main__':
    #sub folders in the root directory respresen classes, for example '/Users/michael/polyps' can contain a /neg/images... and pos/images...
    root_dir_train = 'INSERT PATH HERE'
    root_dir_test = 'INSERT PATH HERE'

    #Reshaping the images durring the loading process
    image_w, image_h = 256,256

    x_train, y_train = getimagedataandlabels(root_dir_train,image_w,image_h)
    print("Training data and labels loaded")
    print(x_train.shape)
    print(y_train.shape)

    print("Test data and labels loaded")
    x_test, y_test = getimagedataandlabels(root_dir_test,image_w,image_h)
    print(x_test.shape)
    print(y_test.shape)
