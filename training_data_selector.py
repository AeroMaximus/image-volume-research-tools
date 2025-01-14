import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from scipy.signal import argrelextrema


def image_list_sum(folder_path):
    """
    Saves a stack of grayscale images from a folder as a list of float numpy arrays and sums them into a single array.
    :param folder_path: path to the folder containing the images.
    :return: the list of (index, image array) tuples, and the sum of those arrays.
    """
    image_list = []
    summed_array = None

    # Iterate through all files in the folder
    for idx, file_name in enumerate(sorted(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, file_name)

        # Open each image, convert to array, and append to the list
        with Image.open(file_path) as img:
            # img = img.convert("L")  # Convert to grayscale
            image_array = np.array(img, dtype=float)
            image_list.append((idx, image_array))

            # Sum the arrays
            if summed_array is None:
                summed_array = image_array
            else:
                summed_array = summed_array + image_array

    return image_list, summed_array


def average_image_calculator(summed_img_array, image_list_len):
    """
    Calculate the average image from a summed array.
    :param summed_img_array: summed image array.
    :param image_list_len: number of images averaged.
    :return: average image array as a float.
    """
    return summed_img_array / image_list_len


def average_pixel_difference_calc(average_image, tuple_list):
    """
    Calculate the average pixel difference between each image and the average image.
    :param average_image: the average image array.
    :param tuple_list: list of (index, image array) tuples.
    :return: list of (index, difference score), sorted by difference score in descending order.
    """
    total_pixels = average_image.size
    difference_scores = []

    for index, img_array in tuple_list:
        diff = np.abs(img_array - average_image)
        score = np.sum(diff) / total_pixels
        difference_scores.append((index, score))

    average_difference_scores_by_key = sorted(difference_scores, key=lambda ele: ele[0])
    average_difference_scores_by_item = sorted(difference_scores, key=lambda ele: ele[1], reverse=True)

    return average_difference_scores_by_key, average_difference_scores_by_item


def average_subset_conversion(average_array, image_list, remove_index):
    """
    Remove the most different image and recalculate the average image.
    :param average_array: current average image array.
    :param image_list: current list of (index, image array) tuples.
    :param remove_index: index of the image to remove.
    :return: new average image, updated image list.
    """
    list_len = len(image_list)
    remove_image = None

    # Remove the selected image
    new_image_list = []
    for idx, (img_idx, img_array) in enumerate(image_list):
        if img_idx == remove_index:
            remove_image = img_array
        else:
            new_image_list.append((img_idx, img_array))

    # Recalculate the average
    if remove_image is not None:
        subset_avg = (average_array - (remove_image / list_len)) * (list_len / (list_len - 1))
    else:
        raise ValueError(f"Image with index {remove_index} not found.")

    return subset_avg, new_image_list


def training_slice_selector(dataset_path, desired_number_of_slices):
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
    img_tuple_list, img_sum = image_list_sum(dataset_path)

    if desired_number_of_slices >= len(img_tuple_list):
        raise ValueError("The dataset is not large enough to select this many slices")

    avg_img = average_image_calculator(img_sum, len(img_tuple_list))
    avg_diff, avg_diff_sorted = average_pixel_difference_calc(avg_img, img_tuple_list)

    avg_diff_array = np.empty(len(img_tuple_list), dtype=object)

    print("Initial scores for all images:")
    for img_index, score in avg_diff:
        img_num = img_index + 1
        print(f"Image {img_num}: Average pixel difference {score:.4f}")
        avg_diff_array[img_index] = score
    print()

    # Order determines how many points on either side of the local extrema are considered to classify it as such
    order = 1
    local_maxima = argrelextrema(avg_diff_array, np.greater, order=order)
    local_minima = argrelextrema(avg_diff_array, np.less, order=order)

    # If the number of local extrema slices is greater than the number of desired slices, increase the order
    if (local_maxima[0].size + local_minima[0].size) > desired_number_of_slices:
        while (local_maxima[0].size + local_minima[0].size) > desired_number_of_slices:
            order += 1
            temp_local_maxima = argrelextrema(avg_diff_array, np.greater, order=order)
            temp_local_minima = argrelextrema(avg_diff_array, np.less, order=order)

            if (temp_local_maxima[0].size + temp_local_minima[0].size) < desired_number_of_slices:
                # If the increase in order reduces the number of slices below the desired number, break the loop
                break
            else:
                # If the increase in order does not return fewer slices than desired, update the local extrema
                local_maxima = temp_local_maxima
                local_minima = temp_local_minima

    # If the number of extrema slices returned at order 1 is less than desired, inform the user
    else:
        print("The desired number of slices could not be identified, please provide additional data")

    # Converts the indices (starting at 0) to slice numbers (starting from 1)
    local_maxima_slice_numbers = local_maxima[0] + 1
    local_minima_slice_numbers = local_minima[0] + 1

    # Inform the user how many images were requested and how many were identified by the closest order
    print(f"Order required to select a minimum of {training_data_quantity} training image slices: {order}")
    print("Total training slices returned: ", local_maxima[0].size + local_minima[0].size)
    print()

    return local_maxima_slice_numbers, local_minima_slice_numbers


# Main script
folder_path = None

# Number of training image slices you want identified from the dataset
training_data_quantity = 10

# Input the dataset path and the number of images
local_max, local_min = training_slice_selector(folder_path, training_data_quantity)

print(local_max)
print(local_min)

