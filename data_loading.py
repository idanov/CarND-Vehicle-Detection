from sklearn.model_selection import train_test_split
from random import shuffle
from image_helpers import img_load, img_convert, single_img_features
import numpy as np
import glob


# Define a function to extract features from a list of images
# Have this function call single_img_features()
def extract_features(img_list, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in img_list:
        # Read in each one by one
        image = img_load(file)
        # apply color conversion if other than 'RGB'
        image = img_convert(image, color_space)
        file_features = single_img_features(image, spatial_size,
                                            hist_bins, orient,
                                            pix_per_cell, cell_per_block, hog_channel,
                                            spatial_feat, hist_feat, hog_feat
                                            )
        features.append(file_features)
    # Return list of feature vectors
    return features


def split_data(data, test_size, random=True):
    if random:
        rand_state = np.random.randint(0, 100)
        data_train, data_test = train_test_split(data, test_size=test_size, random_state=rand_state)
    else:
        needle = int(len(data) * (1 - test_size))
        data_train, data_test = data[:needle], data[needle:]
    return data_train, data_test


def load_vehicle_data(data_dir, test_size=0.2, verbose=True):
    # Get all non-car images (they could safely be shuffled)
    non_car = glob.glob(data_dir + '/non-vehicles/**/*.png')
    non_car_train_list, non_car_test_list = split_data(non_car, test_size=test_size)

    # Get KITTI car images (they could safely be shuffled)
    kitti_car = glob.glob(data_dir + '/vehicles/KITTI_extracted/*.png')
    kitti_car_train, kitti_car_test = split_data(kitti_car, test_size=test_size)

    # Get GTI car images by type (these cannot be shuffled due to being sequences)
    # Start with GTI-far
    gti_far = sorted(glob.glob(data_dir + '/vehicles/GTI_Far/*.png'))
    gti_far_train, gti_far_test = split_data(gti_far, test_size=test_size, random=False)
    # Add with GTI-left
    gti_left = sorted(glob.glob(data_dir + '/vehicles/GTI_Left/*.png'))
    gti_left_train, gti_left_test = split_data(gti_left, test_size=test_size, random=False)
    # Add with GTI-right
    gti_right = sorted(glob.glob(data_dir + '/vehicles/GTI_Right/*.png'))
    gti_right_train, gti_right_test = split_data(gti_right, test_size=test_size, random=False)
    # Add with GTI-middle-close
    gti_middle = sorted(glob.glob(data_dir + '/vehicles/GTI_MiddleClose/*.png'))
    gti_middle_train, gti_middle_test = split_data(gti_middle, test_size=test_size, random=False)

    # Combine all train and test car examples
    car_train_list = kitti_car_train + gti_far_train + gti_left_train + gti_right_train + gti_middle_train
    car_test_list = kitti_car_test + gti_far_test + gti_left_test + gti_right_test + gti_middle_test
    # Print number of examples for car and non-car images
    if verbose:
        print("Non-car images (train/test):", len(non_car_train_list), "/", len(non_car_test_list))
        print("Car images (train/test):", len(car_train_list), "/", len(car_test_list))
    # Merge training car and non-car images and generate target values
    X_train_list = car_train_list + non_car_train_list
    y_train = [1] * len(car_train_list) + [0] * len(car_train_list)
    # Merge test car and non-car images and generate target values
    X_test_list = car_test_list + non_car_test_list
    y_test = [1] * len(car_test_list) + [0] * len(car_test_list)

    # Shuffle training examples
    combined_train = list(zip(X_train_list, y_train))
    shuffle(combined_train)
    X_train_list, y_train = zip(*combined_train)
    # Shuffle test examples
    combined_test = list(zip(X_test_list, y_test))
    shuffle(combined_test)
    X_test_list, y_test = zip(*combined_test)
    # Print information about the train/test split
    if verbose:
        print("Total train/test split:", len(X_train_list), "/", len(X_test_list))

    return X_train_list, y_train, X_test_list, y_test
