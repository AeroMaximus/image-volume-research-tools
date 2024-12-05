from confusion_matrix_stats_calculator import confusion_matrix_statistics

"""
Leave the following folder paths equal to none the first time and then once you've selected the desired folders, 
they will be printed for you to copy into the ground truth and ROI mask variables so you only need to browse for the 
next predicted folder each time. See the functions themselves for further documentation.
"""
ground_truth_folder = None
predicted_folder = None  # Leave as None if you wish to be prompted to browse for the folder each time
ROI_mask_folder = None

confusion_matrix_statistics(255, True, False, ground_truth_folder, predicted_folder,
                            ROI_mask_folder)
