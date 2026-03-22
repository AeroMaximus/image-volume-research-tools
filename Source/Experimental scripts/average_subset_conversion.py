import os


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

