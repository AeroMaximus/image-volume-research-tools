import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix


def image_stacker(folder_path):
    """
    Converts folders containing image stacks to a 3D array of the intensity values (image volume).
    :param folder_path: Path to the folder containing images of identical size that will be stacked into a volume
    :return: Resulting image volume of the stacked images converted to a 3D array
    """
    image_list = []

    # Iterates through all the files in the folder
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)

        # Opens each image, converts it to an array, then appends the array to the list of images in the folder to save
        with Image.open(file_path) as img:
            image_array = np.array(img)
            image_list.append(image_array)

    # Converts the saved list of images into a 3D array by stacking them along the axis normal to the image plane
    image_volume = np.stack(image_list, axis=2)

    return image_volume


def confusion_matrix_statistics(pos_label, roi_mask_input, print_labels, ground_truth_folder_path=None,
                                predicted_folder_path=None, roi_mask_folder_path=None):
    """
    Allows you to select datasets to compare and calculate the confusion matrix of. Prints relevant statistics such as
    F1-score, precision, and recall.
    :param pos_label: The greyscale integer value considered to be positive, it should be equal to (2^(# of bits) - 1)
    :param roi_mask_input: Boolean value for if you have a specific region of interest to isolate
    :param print_labels: Boolean value that determines if the printed outputs have labels or not
    :param ground_truth_folder_path: Optional variable to enter the path to the ground truth folder to skip browsing
    :param predicted_folder_path: Optional variable to enter the path to the predicted folder to skip browsing
    :param roi_mask_folder_path: Optional variable to enter the path to the ROI mask folder to skip browsing
    :return: The confusion matrix
    """

    if not isinstance(roi_mask_input, bool):
        raise ValueError("Mask parameter must be a boolean")
    if not isinstance(print_labels, bool):
        raise ValueError("Print_labels parameter must be a boolean")

    root = tk.Tk()
    root.withdraw()  # Hide the root window

    if ground_truth_folder_path is None:
        ground_truth_folder_path = filedialog.askdirectory(title="Select the Ground Truth dataset folder")

    if ground_truth_folder_path:
        print("Ground truth folder:", ground_truth_folder_path)
    else:
        raise ValueError("No ground truth dataset selected")

    if predicted_folder_path is None:
        predicted_folder_path = filedialog.askdirectory(title="Select the Predicted dataset folder")

    if predicted_folder_path:
        print("Predicted folder:", predicted_folder_path)
    else:
        raise ValueError("No predicted dataset selected")

    if roi_mask_input is True:
        if roi_mask_folder_path is None:
            roi_mask_folder_path = filedialog.askdirectory(title="Select the ROI mask dataset folder")

        if roi_mask_folder_path:
            print("ROI Mask folder:", roi_mask_folder_path)
        else:
            print("No ROI mask selected, calculations will include the entire dataset.")
            roi_mask_input = False

    # Converts image stacks to image volumes
    ground_truth = image_stacker(ground_truth_folder_path)
    predicted = image_stacker(predicted_folder_path)

    if np.shape(ground_truth) != np.shape(predicted):
        raise ValueError("Datasets are not the same size")

    # Flatten the images to 1D arrays for use with sklearn's f1_score function
    if roi_mask_input is True:
        # If there's a mask dataset, turn it into indexes of 0s and 1s
        roi_mask = image_stacker(roi_mask_folder_path)
        normalized_roi_mask = roi_mask/pos_label
        roi_size = np.sum(normalized_roi_mask)
        print("ROI voxels: ", roi_size)

        if np.size(ground_truth) != np.size(roi_mask):
            raise ValueError("Mask dimensions don't match the datasets")

        roi_mask_flat = normalized_roi_mask.flatten() == 1

        # Only convert the 1 indexes to the flat form for comparison
        ground_truth_flat = ground_truth.flatten()[roi_mask_flat]
        predicted_flat = predicted.flatten()[roi_mask_flat]
    else:
        # If there is no mask, flatten the entire dataset
        ground_truth_flat = ground_truth.flatten()
        predicted_flat = predicted.flatten()

    # Calculate the values of the confusion matrix
    confusion_img_matrix = confusion_matrix(ground_truth_flat, predicted_flat, labels=[0, pos_label])
    tn, fp, fn, tp = confusion_img_matrix.ravel()

    # Calculates the precision from the true and false positive values for use in the f1 score calculation
    precision = tp / (tp + fp)

    # Calculates the recall values for use in the f1 score calculation
    recall = tp / (tp + fn)

    # Calculate the F1-score from the precision and recall values to provide a hollistic comparison
    f1 = 2 * (precision * recall) / (precision + recall)

    # Calculate pixel error
    pixel_error = (fn + fp) / (tp + fn + fp + tn)

    if print_labels:
        print("\nTrue positive (TP):", tp)
        print("False negative (FN):", fn)
        print("False positive (FP):", fp)
        print("True negative (TN):", tn)
        print("\nPrecision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Pixel Error:", pixel_error)
    else:
        print(tp)
        print(fn)
        print(fp)
        print(tn)
        print(precision)
        print(recall)
        print(f1)
        print(pixel_error)
        print("\n")

    stat_bloc = [tp, fn, fp, tn, precision, recall, f1, pixel_error]
    stat_bloc = np.transpose(stat_bloc)

    return stat_bloc

