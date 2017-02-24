import numpy as np
import cv2
from skimage.feature import hog


# Define a function for loading images as RGB
def img_load(fname):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Define a function for converting images to a given color space
def img_convert(image, color_space='RGB'):
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = np.copy(image)
    return image


# Define a function to return HOG features and visualization (if requested)
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a helper function to take hog features for specified image channels
def img_hog_features(feature_image, orient, pix_per_cell, cell_per_block, hog_channel=0):
    if hog_channel == 'ALL':
        features = []
        for channel in range(feature_image.shape[2]):
            hog_features = get_hog_features(feature_image[:, :, channel],
                                            orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=False)
            features.append(np.expand_dims(hog_features, axis=-1))
    else:
        hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=False)
        features = [np.expand_dims(hog_features, axis=-1)]
    return np.concatenate(features, axis=-1)


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a single image window
def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(img, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(img, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        hog_features = img_hog_features(img, orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block, hog_channel=hog_channel)
        # 8) Append features to list
        img_features.append(hog_features.ravel())

    # 9) Return concatenated array of features
    return np.concatenate(img_features)
