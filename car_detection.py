import cv2
import time
import itertools
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from image_helpers import single_img_features, img_convert, img_hog_features, img_load


class CarDetector:
    def __init__(self, color_space='HLS', spatial_size=(16, 16), hist_bins=32,
                 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                 spatial_feat=True, hist_feat=True, hog_feat=True, fast=False):
        self.__color_space = color_space
        self.__spatial_size = spatial_size
        self.__hist_bins = hist_bins
        self.__orient = orient
        self.__pix_per_cell = pix_per_cell
        self.__cell_per_block = cell_per_block
        self.__hog_channel = hog_channel
        self.__spatial_feat = spatial_feat
        self.__hist_feat = hist_feat
        self.__hog_feat = hog_feat
        self.__feature_scaler = None
        self.__model = None
        self.__fast = fast

    def __extract_features(self, img_list):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in img_list:
            # Read in each one by one
            image = img_load(file)
            # apply color conversion if other than 'RGB'
            image = img_convert(image, self.__color_space)
            file_features = single_img_features(image, self.__spatial_size,
                                                self.__hist_bins, self.__orient,
                                                self.__pix_per_cell, self.__cell_per_block, self.__hog_channel,
                                                self.__spatial_feat, self.__hist_feat, self.__hog_feat, self.__fast
                                                )
            features.append(file_features)
        # Return list of feature vectors
        return features

    def train_model(self, X_train_list, y_train, X_test_list, y_test):
        t1 = time.time()
        X_train = self.__extract_features(X_train_list)
        X_test = self.__extract_features(X_test_list)
        t2 = time.time()
        print(round(t2 - t1, 2), 'seconds to extract the features of all images...')

        # Use standard scaler for features
        self.__feature_scaler = StandardScaler()
        # Fit a per-column scaler on train data
        self.__feature_scaler.fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = self.__feature_scaler.transform(X_train)
        X_test = self.__feature_scaler.transform(X_test)

        print('Feature vector length:', len(X_train[0]))
        # Check the training time for the SVC
        t1 = time.time()
        # Use a linear SVC
        self.__model = LinearSVC(dual=False)
        self.__model.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t1, 2), 'seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.__model.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t1 = time.time()
        n_predict = 10
        print('My SVC predicts: ', self.__model.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t1, 5), 'seconds to predict', n_predict, 'labels with SVC')

    # Define a function that takes
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and step size (for both x and y)
    @staticmethod
    def slide_window(x_start_stop, y_start_stop, xy_window, xy_step):
        x_start = x_start_stop[0]
        x_stop = x_start_stop[1] - xy_window[0]
        y_start = y_start_stop[0]
        y_stop = y_start_stop[1] - xy_window[1]
        # Compute the number of pixels per step in x/y
        x_step = xy_step[0]
        y_step = xy_step[1]
        # Enumerate all x/y coordinates of the windows
        x_range = range(x_start, x_stop + 1, x_step)
        y_range = range(y_start, y_stop + 1, y_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through the cartesian product of all x and y window coordinates
        for x_pos, y_pos in itertools.product(x_range, y_range):
            # Append window position to list
            window_list.append(((x_pos, y_pos), (x_pos + xy_window[0] - 1, y_pos + xy_window[1] - 1)))
        # Return the list of windows
        return window_list

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def __search_windows(self, img, windows, ytop, ybottom, window_size):
        # 1) Calculate window size to feature image size scale
        scale = window_size / 64
        cell_per_window = 64 // self.__pix_per_cell - 1
        # 2) Extract only the region of interest from the image
        img = img[ytop:ybottom, :, :]
        # 3) Apply color conversion if other than 'RGB'
        img = img_convert(img, self.__color_space)
        # 4) Resize image for calculating hog
        img_to_hog = img.copy()
        if scale != 1:
            w = int(((img.shape[1] / scale) // self.__pix_per_cell) * self.__pix_per_cell)
            h = int(((img.shape[0] / scale) // self.__pix_per_cell) * self.__pix_per_cell)
            img_to_hog = cv2.resize(img, (w, h))
        # 5) Calculate hog over the region of interest
        if not self.__fast:
            hog_image = img_hog_features(img_to_hog, orient=self.__orient, pix_per_cell=self.__pix_per_cell,
                                         cell_per_block=self.__cell_per_block, hog_channel=self.__hog_channel,
                                         feature_vec=False, fast=False)
        # 6) Create an empty list to receive positive detection windows
        on_windows = []
        # 7) Iterate over all windows in the list
        for tl, br in windows:
            # 8) Extract the test window from original image
            test_img = cv2.resize(img[tl[1] - ytop:br[1] - ytop, tl[0]:br[0]], (64, 64))
            # 9) Extract features for that window using single_img_features()
            features = single_img_features(test_img,
                                           spatial_size=self.__spatial_size, hist_bins=self.__hist_bins,
                                           orient=self.__orient, pix_per_cell=self.__pix_per_cell,
                                           cell_per_block=self.__cell_per_block, hog_channel=self.__hog_channel,
                                           spatial_feat=self.__spatial_feat, hist_feat=self.__hist_feat, hog_feat=self.__fast)
            # 10) Extract hog features for that window
            if self.__hog_feat and not self.__fast:
                # carefully convert real image coordinates to hog indices
                to_hog = lambda x: int(((x + 1) / scale) // self.__pix_per_cell)
                hog_roi = hog_image[to_hog(tl[1] - ytop):to_hog(tl[1] - ytop) + cell_per_window,
                          to_hog(tl[0]):to_hog(tl[0]) + cell_per_window]
                features = np.concatenate((features, hog_roi.ravel()))
            # 11) Scale extracted features to be fed to classifier
            test_features = self.__feature_scaler.transform(np.array(features).reshape(1, -1))
            # 12) Predict using your classifier
            prediction = self.__model.predict(test_features)
            # 13) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append((tl, br))
        # 14) Return windows for positive detections
        return on_windows

    def detect_cars(self, image, win_sizes, ytop=0):
        assert self.__model is not None, "A model should be trained first using self.train_model()"
        all_bounding_boxes = []
        for win_size, blocks_per_step, ybottom in win_sizes:
            # Calculate the hog cell size in a non-scaled image
            # There are (64 / pix_per_cell) number of cells in a 64px image,
            # So before resizing the window, a hog cell will have hog_cell_size pixels
            hog_cell_size = (self.__pix_per_cell * win_size) // 64
            step = blocks_per_step * hog_cell_size
            # Calculate sliding windows with a step, multiple of the hog cell size
            windows = self.slide_window(x_start_stop=(0, 1280), y_start_stop=(ytop, ybottom),
                                        xy_window=(win_size, win_size), xy_step=(step, step))
            # Find bounding boxes
            bounding_boxes = self.__search_windows(image, windows, ytop=ytop, ybottom=ybottom, window_size=win_size)
            all_bounding_boxes.extend(bounding_boxes)
        return all_bounding_boxes


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def labeled_bboxes(labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    # Return the image
    return bboxes
