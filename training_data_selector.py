import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from scipy.signal import argrelextrema


def image_list_sum(folder_path):
    """
    Loads images, filters out non-image files, and calculates their sum.
    :param folder_path: path to the folder containing the images.
    :return: 3D float image_volume_array of the images stacked, and the summed_array of every image added together
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    file_paths = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(valid_extensions)
    ]

    # Load images as floats and create an image volume array
    image_volume_array = np.array([np.array(Image.open(file_path), dtype=float) for file_path in file_paths])

    # Sum all the images in the stack to get a total for every pixel
    summed_array = np.sum(image_volume_array, axis=0)

    return image_volume_array, summed_array


def average_pixel_difference_calc(average_image, image_volume_array):
    """
    Calculate the average pixel difference between each image and the average image.
    :param average_image: the average image array.
    :param image_volume_array: 3D float image stack array.
    :return: list of (index, difference score), sorted by difference score in descending order.
    """
    total_pixels = average_image.size

    difference_array = np.abs(image_volume_array - average_image[np.newaxis, :, :])
    scores = np.sum(difference_array, axis=(1, 2)) / total_pixels
    difference_scores = list(enumerate(scores))

    average_difference_scores_by_key = sorted(difference_scores, key=lambda ele: ele[0])
    average_difference_scores_by_item = sorted(difference_scores, key=lambda ele: ele[1], reverse=True)

    return average_difference_scores_by_key, average_difference_scores_by_item


def local_extrema_by_mode(array, mode, order=1, index_offset=1):
    """
    Given a 1D array dataset and the type of extrema, this program
    :param array: 1D dataset being analyzed
    :param mode: "max", "min", or "both" determines which extrema values to look for
    :param order: the number of points considered on either side of a potential local extrema
    :param index_offset: the starting index for your data (default to 1 to match image stack numbering)
    :return: total_extrema: total number of local extrema found, extrema: Dictionary with max and or min keys holding
    tuples of the extrema indices
    """
    extrema = {}

    if mode in ("max", "both"):
        extrema["max"] = argrelextrema(array, np.greater, order=order)[0] + index_offset

    if mode in ("min", "both"):
        extrema["min"] = argrelextrema(array, np.less, order=order)[0] + index_offset

    total_extrema = sum(len(indices) for indices in extrema.values())

    return total_extrema, extrema


def training_slice_selector(dataset_path, desired_number_of_slices, mode="both", idx_offset=1):
    """
    Selects the desired number of image slices from the input dataset for training data using the local extrema of the
    average pixel difference scores.
    :param dataset_path: the path to the image stack dataset folder
    :param desired_number_of_slices: how many image slices you want to identify for use as training data
    :return: list of the local maxima and minima slice numbers totaling the desired number of training slices
    """

    root = tk.Tk()
    root.withdraw()  # Hide the root window

    if dataset_path is None:
        dataset_path = filedialog.askdirectory(title="Select dataset folder")
    print("Dataset folder path:", dataset_path)

    # Load images and calculate initial average
    img_vol_array, img_sum = image_list_sum(dataset_path)

    if desired_number_of_slices >= img_vol_array.shape[0]:
        raise ValueError("The dataset is not large enough to select this many slices")

    avg_img = img_sum / img_vol_array.shape[0]
    avg_diff, avg_diff_sorted = average_pixel_difference_calc(avg_img, img_vol_array)
    average_difference_array = np.array([score for _, score in avg_diff])

    # Order determines how many points on either side of the local extrema are considered to classify it as such
    order = 1
    total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order, idx_offset)

    # If the number of local extrema slices is greater than the number of desired slices, increase the order
    if total_extrema > desired_number_of_slices:
        while total_extrema > desired_number_of_slices:
            order += 1
            temp_total_extrema, temp_local_extrema = local_extrema_by_mode(average_difference_array, mode, order, idx_offset)

            if temp_total_extrema < desired_number_of_slices:
                # If the increase in order reduces the number of slices below the desired number, break the loop
                break

            # If the increase in order does not return fewer slices than desired, update the local extrema
            total_extrema = temp_total_extrema
            local_extrema = temp_local_extrema

        # Inform the user how many images were requested and how many were identified by the closest order
        print(f"Order required to select a minimum of {training_data_quantity} training image slices: {order}")

    # If the number of extrema slices returned at order 1 is less than desired, inform the user
    else:
        print(f"To select {training_data_quantity} slices for training data, please provide more data or change mode.")

    print("Total training slices returned: ", total_extrema)
    print()

    return local_extrema, average_difference_array


# Main script
folder_path = None

# Number of training image slices you want identified from the dataset
training_data_quantity = 3

# Input the dataset path and the number of images
local_extrema, avg_diff_array = training_slice_selector(folder_path, training_data_quantity, mode="both", idx_offset=1)

print(local_extrema)

