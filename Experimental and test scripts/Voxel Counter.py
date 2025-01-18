import numpy as np

"""
Note that this script was only used once to verify voxel counts and is not being updated or used.
"""

# Path to a uint16_scv raw CT scan file
scv_file_path = r"path to your file"

# Read the SCV file in binary mode
with open(scv_file_path, "rb") as scv_file:
    # Read raw data from file, skipping the first 1024 bytes
    raw = np.frombuffer(scv_file.read(), dtype=np.uint16, offset=1024)

print(raw[1])

# Gray value intensity threshold separating the background air from the subject of the CT scan
global_threshold = 12515

# Set every non-air voxel to 1 and the rest to 0, sum the voxels to find the number of non-air voxels
no_air_voxels = np.sum(np.where(raw > global_threshold, 1, 0))
# Set every air voxel to 1 and the others to 0, sum the voxels to find the number of air voxels
air_voxels = np.sum(np.where(raw <= global_threshold, 1, 0))

# Sum the two to ensure we get the correct total number of voxels
total = no_air_voxels + air_voxels

print(raw.shape)
print("Sample voxels:", no_air_voxels)
print("Air voxels:", air_voxels)
print("Total Voxels:", total)
