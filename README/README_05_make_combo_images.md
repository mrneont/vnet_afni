What is this README for?
------------------------------

README_make_combo_images.md is for illustrating the steps required to create the combo images from the predicted masks after training or testing. The combo images are basically the predicted mask and groundtruth overlayed on the original dataset. These images help in visually identifying the performance of the model compared to the ground truth. 

The scripts are present in the 'make_combo_images_scripts' folder 
1) do_make_combo.tcsh
2) make_combo_image_folders.tcsh
3) make_combo_images_wrapper.py

Steps for creating combo images from the given predicted masks
------------------------------
1. Download+git checkout the 'make_combo_images_scripts' folder from git repo.

2. Start interactive session on biowulf: 'sinteractive --mem=20g'

3. Activate the conda env: 'conda activate vnet_tech_2024_01_02'

4. Load AFNI programs: 'module afni'

5. export PYTHONPATH=/usr/local/apps/afni/current-py3/linux_rocky_8

6. Use the 'make_combo_images_wrapper.py' program to create an combo images from a given set of predicted masks.

   Command to run: python make_combo_images_wrapper.py
-pred_mask_dir /path_to_predicted_masks_folder
-seed_num 42
-exec_mode swarm


do_make_combo.tcsh shell script creates combo images of each predicted mask in two steps. 
first the predicated mask is added to the groundtruth, next the result is overlayed on the original. 

make_combo_image_folders.tcsh shell script prepares the folders
-> pred_mask
-> orig
-> target
-> pred_plus_target
-> combo_images



