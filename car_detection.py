import itertools
import cv2
import numpy as np
from image_helpers import single_img_features, img_convert, img_hog_features


# Define a function that takes
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and step size (for both x and y)
def slide_window(x_start_stop=(0, 1280), y_start_stop=(0, 720), xy_window=(64, 64), xy_step=(32, 32)):
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
def search_windows(img, windows, clf, scaler, ytop, ybottom, window_size,
                   color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):

    # 1) Calculate window size to feature image size scale
    scale = window_size / 64
    cell_per_window = 64 // pix_per_cell - 1
    # 2) Extract only the region of interest from the image
    img = img[ytop:ybottom, :, :]
    # 3) Apply color conversion if other than 'RGB'
    img = img_convert(img, color_space)
    # 4) Resize image for calculating hog
    img_to_hog = img.copy()
    if scale != 1:
        w = int(img.shape[1] / scale)
        h = int(img.shape[0] / scale)
        img_to_hog = cv2.resize(img, (w, h))
    # 5) Calculate hog over the region of interest
    hog_image = img_hog_features(img_to_hog, orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel=hog_channel)
    # 6) Create an empty list to receive positive detection windows
    on_windows = []
    # 7) Iterate over all windows in the list
    for tl, br in windows:
        # 8) Extract the test window from original image
        test_img = cv2.resize(img[tl[1] - ytop:br[1] - ytop, tl[0]:br[0]], (64, 64))
        # 9) Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       spatial_size=spatial_size, hist_bins=hist_bins, hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=False)
        # 10) Extract hog features for that window
        if hog_feat:
            # carefully convert real image coordinates to hog indices
            to_hog = lambda x: int(((x + 1) / scale) // pix_per_cell)
            hog_roi = hog_image[to_hog(tl[1] - ytop):to_hog(tl[1] - ytop) + cell_per_window,
                                to_hog(tl[0]):to_hog(tl[0]) + cell_per_window]
            features = np.concatenate((features, hog_roi.ravel()))
        # 11) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 12) Predict using your classifier
        prediction = clf.predict(test_features)
        # 13) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append((tl, br))
    # 14) Return windows for positive detections
    return on_windows


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


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img
