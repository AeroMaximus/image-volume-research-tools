import math
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from scipy.signal import argrelextrema


def get_total_size(file_paths):
    """
    Calculates the total size of a list of files
    :param file_paths: list of files
    :return: total size of the list
    """
    total_size = 0
    for file_path in file_paths:
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
        else:
            print(f"Warning: {file_path} is not a valid file.")
    return total_size


def image_list_avg(folder_path):
    """
    Loads images, filters out non-image files, and calculates their average image.
    :param folder_path: path to the folder containing the images.
    :return: list of file paths to each image, the average image as an array of values
    """
    valid_extensions = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp")
    file_paths = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(valid_extensions)
    ]

    # Calculate size of dataset
    total_size = get_total_size(file_paths)
    print(f"Dataset size: {total_size: .2f} GB")

    summed_array = 0

    for file_path in file_paths:
        summed_array += np.array(Image.open(file_path), dtype=float)

    avg_img = summed_array / len(file_paths)

    return file_paths, avg_img


def average_pixel_difference_calc(average_image, dataset_file_paths):
    """
    Calculate the average pixel difference between each image and the average image.
    :param dataset_file_paths: list of file paths to images in dataset
    :param average_image: the average image array.
    :return: list of difference scores
    """
    total_pixels = average_image.size
    difference_scores = []

    for file_path in dataset_file_paths:
        image_array = np.array(Image.open(file_path))
        difference_array = np.abs(image_array - average_image)
        difference_score = np.sum(difference_array) / total_pixels
        difference_scores.append(difference_score)

    return difference_scores


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

    if dataset_path:
        print("Dataset folder path:", dataset_path)
    else:
        raise ValueError("No dataset path provided")

    # Load images and calculate initial average image
    file_paths, avg_img = image_list_avg(dataset_path)
    number_of_images = len(file_paths)

    if desired_number_of_slices >= number_of_images:
        raise ValueError("The dataset is not large enough to select this many slices")

    average_difference_array = np.array(average_pixel_difference_calc(avg_img, file_paths))

    # Order determines how many points on either side of the local extrema are considered to classify it as such
    order = 1
    total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order, idx_offset)

    # If the number of local extrema slices is greater than the number of desired slices, increase the order
    if total_extrema > desired_number_of_slices:

        # Start at the highest order possible and work down
        order = math.floor(number_of_images/2)
        total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order, idx_offset)

        # Loop ends when the order is the largest possible to give desired results
        while total_extrema < desired_number_of_slices and order >= 1:
            order -= 1
            total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order, idx_offset)

        # Inform the user how many images were requested and how many were identified by the closest order
        print(f"Order to select a minimum of {training_data_quantity} training image slices: {order}")

    # If the number of extrema slices returned at order 1 is less than desired, inform the user
    else:
        print(f"To select {training_data_quantity} slices for training data, please provide more data or change mode.")

    print("Total training slices returned: ", total_extrema)
    print()

    return local_extrema, average_difference_array


# Main script
folder_path = None

# Number of training image slices you want identified from the dataset
training_data_quantity = 15

# Input the dataset path and the number of images
local_extrema, avg_diff_array = training_slice_selector(folder_path, training_data_quantity, mode="both", idx_offset=0)

print(local_extrema)

