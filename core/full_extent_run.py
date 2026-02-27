# [WIP] - Test in Google Colab to ensure compatibility with both Colab/Local

import os
import sys

is_colab = "google.colab" in sys.modules

# Google Colab handling
if is_colab:
    from google.colab import drive
    drive.mount("/content/drive")

# Import local files
from core.full_extent.A_warping import sgd_mesh_warp_adaptive

# Parameters
image_path_one = "./drive/MyDrive/Colab Notebooks/landarea.png"
image_path_two = "./drive/MyDrive/Colab Notebooks/image_03.png"

warped_result, learned_disp = sgd_mesh_warp_adaptive(image_path_one, image_path_two)