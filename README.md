# image-volume-research-tools

Author: Maximus Chen

A collection of Python scripts and functions that I’ve developed for my research to do basic analysis on image volumes, particularly, segmentations of CT scans by deep learning algorithms.

The confusion_matrix_stats_calculator file contains functions I wrote to calculate the confusion matrix and related stats such as F1 score of 3D image volume dataset comparisons.

The Training Data selector script is still a work in progress, but it currently calculates the average image of a greyscale image stack dataset and then finds the average mean square error (MSE) between every image and the average image. Since every image in the stack has an average MSE that roughly determines how different each image is to the entire dataset, we can select a mix of the local maximum and minimum scoring images to use as training data. Local minimums should be the images most similar to the entire dataset and thus hopefully representative of most of the dataset. Local maximums will be the most different images in the dataset and will add diversity to the training data.

Any unmentioned scripts and functions are likely test scripts I included but are not being used or updated further.