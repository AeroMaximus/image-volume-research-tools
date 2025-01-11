# image-volume-research-tools

Author: Maximus Chen

A collection of Python scripts and functions that I’ve developed for my research to do basic analysis on image volumes, particularly, segmentations of CT scans by deep learning algorithms.

The confusion_matrix_stats_calculator file is contains functions I wrote to calculate the confusion matrix and related stats such as F1 score of 3D image volume dataset comparisons. There is a demo script using this function that you can use but please see the function itself for documentation.

The key_phrase_folder_search file contains a function for searching a directory for a file containing a keyword or phrase in the file name.

The Training Data selector script is still a work in progress and will have it's logic changed soon, but it currently calculates the average image of a greyscale image stack dataset, subtracts the average image from every image to create pixel intensity difference maps, and then averages those maps to assign every image an average pixel intensity difference. The most different image is removed from the stack and identified to the user to use as training data and the remaining images are put through the process again.