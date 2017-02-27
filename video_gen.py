from functools import reduce

from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from car_detection import *
from data_loading import load_vehicle_lists

X_train_list, y_train, X_test_list, y_test = load_vehicle_lists("data/", test_size=0.15)

# Parameters
detector = CarDetector(color_space='YCrCb', spatial_size=(16, 16), hist_bins=32,
                       orient=16, pix_per_cell=16, cell_per_block=2, hog_channel='ALL',
                       spatial_feat=True, hist_feat=True, hog_feat=True, fast=True)

detector.train_model(X_train_list, y_train, X_test_list, y_test)
# Window size and sliding step in number of hog cells (>=8 means no overlapping)
win_sizes = [(48, 2, 560),
             (64, 2, 576),
             (96, 1, 608),
             (128, 1, 624)]
heatmap_history = []
max_history = 5
history_thresh = 3
frame_thresh = 2


def windows_to_cars(image, bboxes, thresh):
    # Empty heat image
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add bounding boxes from the current frame
    heat = add_heat(heat, bboxes)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, thresh)
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    bboxes = labeled_bboxes(labels)
    return bboxes


def process_image(image):
    global heatmap_history
    global max_history
    global frame_thresh
    global history_thresh
    bboxes = detector.detect_cars(image, win_sizes, ytop=400)
    # Get boxes for the current frame
    current_bboxes = windows_to_cars(image, bboxes, frame_thresh)
    # Get and update history
    hist = reduce((lambda x, y: x + y), heatmap_history, [])
    heatmap_history = heatmap_history[-(max_history-1):] + [current_bboxes]
    # Generate final bboxes
    final_bboxes = windows_to_cars(image, current_bboxes + hist, history_thresh)
    # Draw final bboxes
    draw_img = draw_boxes(image, final_bboxes, color=(0, 0, 255), thick=6)
    return draw_img

input_video = 'project_video.mp4'
output_video = 'project_video_detected.mp4'

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

