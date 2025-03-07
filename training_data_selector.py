import math
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import progressbar
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
    file_paths = sorted(
        [entry.path for entry in os.scandir(folder_path) if
         entry.is_file() and entry.name.lower().endswith(tuple(valid_extensions))]
    )

    # Calculate size of dataset
    total_size = get_total_size(file_paths) / 1073741824
    print(f"Dataset size: {total_size: .2f} GB")

    summed_array = 0

    print("Calculating Average Image")
    summation_progress_bar = progressbar.ProgressBar(max_value=len(file_paths), redirect_stdout=True)
    summation_progress_bar.update(0)

    for f, file_path in enumerate(file_paths):
        summed_array += np.array(Image.open(file_path), dtype=float)
        summation_progress_bar.update(f+1)

    summation_progress_bar.finish()

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

    print("Calculating Difference Scores")
    scoring_progress_bar = progressbar.ProgressBar(max_value=len(dataset_file_paths), redirect_stdout=True)
    scoring_progress_bar.update(0)

    for f, file_path in enumerate(dataset_file_paths):
        image_array = np.array(Image.open(file_path))
        # difference_array = np.abs(image_array - average_image)

        # Mean Square Error formula
        difference_array = (image_array - average_image) ** 2
        difference_score = np.sum(difference_array) / total_pixels
        difference_scores.append(difference_score)

        scoring_progress_bar.update(f+1)

    scoring_progress_bar.finish()

    return difference_scores

def img_diff_plot(average_difference_array, idx_offset, number_of_images):
    mean_score = np.mean(average_difference_array)
    median_score = np.median(average_difference_array)

    x = np.arange(idx_offset, idx_offset + number_of_images)
    plt.plot(x, average_difference_array, label='Average MSE per slice', marker='o')

    # Adding mean and median lines
    plt.axhline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
    plt.axhline(median_score, color='g', linestyle='-.', label=f'Median: {median_score:.2f}')

    # Adding labels and title
    plt.xlabel('Slice')
    plt.ylabel('Average MSE')
    plt.legend()

    # Add major grid lines
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')

    # Add minor grid lines
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.show()

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


def training_slice_selector(dataset_path=None, desired_number_of_slices=None, mode="both", idx_offset=0, plot_prev=False):
    """
    Selects the desired number of image slices from the input dataset for training data using the local extrema of the
    average pixel difference scores.
    :param idx_offset: the index your dataset begins numbering at (typically either 0 or 1 unless using a subset)
    :param mode: 'max' to identify the most unique images, 'min' for the least unique images, or 'both' for a
                combination of the two
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

    # Score each image
    average_difference_array = np.array(average_pixel_difference_calc(avg_img, file_paths))

    while True:
        if idx_offset is None:
            try:
                print()
                idx_offset = int(input("Starting index not specified, please enter it now: "))
            except:
                print()
                print("Invalid index, please enter an integer")
                continue
        else:
            break

    if plot_prev is True:
        img_diff_plot(average_difference_array, idx_offset, number_of_images)

    if desired_number_of_slices is None:
        while True:
            try:

                print()
                desired_number_of_slices = int(input("Enter the number of training images (or 0 to change mode, -1 to exit): "))

                if desired_number_of_slices == 0:
                    print()
                    while True:
                        mode = input("Enter 'max' to identify the most unique images, 'min' for the least unique images, "
                                        "or 'both' for a combination of the two: ")
                        if mode in ('max', 'min', 'both'):
                            break
                        else:
                            print("\nInvalid mode\n")
                    continue

                if desired_number_of_slices == -1:
                    if 'local_extrema' not in locals():
                        total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, 1,
                                                                             idx_offset)
                    break

                # Order determines how many points on either side of the local extrema are considered to classify it as such
                order = 1
                total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order,
                                                                     idx_offset)

                if desired_number_of_slices >= number_of_images:
                    print("The dataset is not large enough to select this many slices. Try again.")
                    continue

                # If the number of local extrema slices is greater than the number of desired slices, increase the order
                if total_extrema > desired_number_of_slices:

                    # Start at the highest order possible and work down
                    order = math.floor(number_of_images / 2)
                    total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order,
                                                                         idx_offset)

                    # Loop ends when the order is the largest possible to give desired results
                    while total_extrema < desired_number_of_slices and order >= 1:
                        order -= 1
                        total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order,
                                                                             idx_offset)

                    # Inform the user how many images were requested and how many were identified by the closest order
                    print(f"Order to select a minimum of {desired_number_of_slices} training image slices: {order}")

                # If the number of extrema slices returned at order 1 is less than desired, inform the user
                else:
                    print(
                        f"To select {desired_number_of_slices} slices for training data, please provide more data or change mode.")

                print("Total training slices returned: ", total_extrema)
                print()
                print(f"Selected slices: {local_extrema}")

            except ValueError:
                print("Invalid input. Please enter an integer.")

    else:
        # Order determines how many points on either side of the local extrema are considered to classify it as such
        order = 1
        total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order, idx_offset)

        # If the number of local extrema slices is greater than the number of desired slices, increase the order
        if total_extrema > desired_number_of_slices:

            # Start at the highest order possible and work down
            order = math.floor(number_of_images / 2)
            total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order,
                                                                 idx_offset)

            # Loop ends when the order is the largest possible to give desired results
            while total_extrema < desired_number_of_slices and order >= 1:
                order -= 1
                total_extrema, local_extrema = local_extrema_by_mode(average_difference_array, mode, order,
                                                                     idx_offset)

        # If the number of extrema slices returned at order 1 is less than desired, inform the user
        else:
            print(f"To select {desired_number_of_slices} slices for training data, please provide more data or change mode.")

    slices = np.arange(idx_offset, idx_offset+number_of_images)
    average_difference_array = np.column_stack((slices, average_difference_array))

    return local_extrema, average_difference_array

def main():
    """Intended future features are changing the difference scoring method and having the option to export the identified
     training data to a new directory"""

    # Input the dataset path, enter None if you wish to browse for the directory (None by default)
    folder_path = None

    # Number of training image slices you want identified from the dataset, enter None to be prompted (None by default)
    training_data_quantity = None

    # Enter the index of the first image in the dataset, enter None to be prompted (0 by default)
    starting_index = None

    # Change whether you want a preview of the difference scores vs image slices to appear (False by default)
    plot_preview = False

    local_extrema, avg_diff_array = training_slice_selector(folder_path, training_data_quantity, mode="both",
                                                            idx_offset=starting_index, plot_prev=plot_preview)

    print(f"Final Slice Selection: {local_extrema}")

if __name__ == '__main__':
    main()
