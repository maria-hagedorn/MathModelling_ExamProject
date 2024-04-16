import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
    
def apply_nocloud_filter(rgb_image):
    """Applies the NoCloud filter to an RGB image."""
    R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    NoCloud = R + G - 2 * B
    return cv2.normalize(NoCloud, None, 0, 255, cv2.NORM_MINMAX)

def convert_to_grayscale(image):
    """Converts an image to grayscale, checking if it is RGB."""
    if image.ndim == 3 and image.shape[2] == 3:  # Check if the image is RGB
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:  # The image is already grayscale
        return image
    else:
        raise ValueError("Unexpected image format. Expected RGB or grayscale.")

def calculate_optical_flow(prev_image, current_image, prev_features=None):
    """ Calculates and returns the optical flow between two images.
        
        prev_features define which method is used to calculate the optical flow.
    """

    if prev_features is not None:
        next_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_image, current_image, prev_features, None)
        return next_features
    else:
        return cv2.calcOpticalFlowFarneback(prev_image, current_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def detect_features_to_track(image):
    """Uses goodFeaturesToTrack for feature detection."""
    return cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

base_dir = 'data'
output_dir = 'data/flows'   # Directory for output flow data
mask_path = 'data/mask.npy' # Denmark mask file path

# Compile a regular expression pattern to match directories like "202403xx"
pattern = re.compile(r"202403\d\d$")

# List and sort directories of the image folders
days = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d)) and pattern.match(d)]

for day in days:
    # Path to the day directory
    day_dir = os.path.join(base_dir, day)
    image_arrays = sorted([f for f in os.listdir(day_dir) if f.endswith('.npy')])

    # Reset the previous image for each day, night period is not considered
    prev_image = None 
    prev_features = None

    for i, file_name in enumerate(image_arrays):
        # Path to the image array file
        image_path = os.path.join(day_dir, file_name)
        current_image = np.load(image_path)

        # Check if the image array is RGB
        if current_image.ndim == 3 and current_image.shape[2] == 3:
            # Apply the NoCloud filter
            nocloud_image = apply_nocloud_filter(current_image)

            # Convert to grayscale
            grayscale_image = convert_to_grayscale(nocloud_image)

            if prev_image is not None:
                if prev_features is None:
                    prev_features = detect_features_to_track(prev_image)
                    next_features = calculate_optical_flow(prev_image, grayscale_image, prev_features)
                flow = calculate_optical_flow(prev_image, grayscale_image)
                flow_file_path = os.path.join(output_dir, f'flow_{day}_{i-1}_{i}')
                np.save(flow_file_path, flow)

            prev_image = grayscale_image

        else:
            print("The loaded array is not in the expected RGB format.")
