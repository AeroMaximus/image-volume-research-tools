import os
import numpy as np
from PIL import Image
from discrete_local_extrema_finder import local_extrema


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

    return average_difference_scores_by_key, average_difference_scores_by_item # sorted(difference_scores, key=lambda x: x[1], reverse=True)


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


# Main script
folder_path = r"path to your dataset image stack"
# Number of training images you want identified
training_data_quantity = 10

# Load images and calculate initial average
img_tuple_list, img_sum = image_list_sum(folder_path)
avg_img = average_image_calculator(img_sum, len(img_tuple_list))
avg_diff, avg_diff_sorted = average_pixel_difference_calc(avg_img, img_tuple_list)

avg_diff_array = np.empty(len(img_tuple_list), dtype=object)

print("Initial scores for all images:")
for img_index, score in avg_diff:
    img_num = img_index + 1
    print(f"Image {img_num}: Average pixel difference {score:.4f}")
    avg_diff_array[img_index] = score
print()

order = 1
local_maxima, local_minima = local_extrema(avg_diff_array, order)

while (local_maxima[0].size + local_minima[0].size) > training_data_quantity:
    order += 1
    local_maxima, local_minima = local_extrema(avg_diff_array, order)

# Converts the indices (starting at 0) to slice numbers (starting from 1)
local_maxima_slice_numbers = local_maxima[0] + 1
local_minima_slice_numbers = local_minima[0] + 1

print(f"Order required to select {training_data_quantity} training images: {order}")

print(local_maxima_slice_numbers)

print(local_minima_slice_numbers)

# # Iteratively remove the most different image
# for iteration in range(training_data_quantity):
#     avg_diff, avg_diff_sorted = average_pixel_difference_calc(avg_img, img_tuple_list)
#     most_different_index = avg_diff_sorted[0][0]
#
#     if iteration == 0:
#         print("Initial scores for all images:")
#         for index, score in avg_diff:
#             print(f"Image index {index}: Difference score {score:.4f}")
#         print()
#
#     print(f"Iteration {iteration + 1}: Removing image index {most_different_index + 1}, score {avg_diff_sorted[0][1]:.4f}")
#     avg_img, img_tuple_list = average_subset_conversion(avg_img, img_tuple_list, most_different_index)

    # Next step should be to revise the code so that after the initial average pixel differences are calculated, graph
    # them against the slice number and check the local extrema

