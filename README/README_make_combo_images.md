What is this README for?
------------------------------

README_make_combo_images.md is for illustrating the steps required to create the combo images from the predicted masks after training or testing. The combo images are basically the predicted mask and groundtruth overlayed on the original dataset. These images help in visually identifying the performance of the model compared to the ground truth. 

The scripts are present in the 'make_combo_images_scripts' folder 
1) do_make_combo.tcsh
2) run_make_combo_images.tcsh

`tcsh run_make_combo_images.tcsh  $1 $2 `   
`tcsh run_make_combo_images.tcsh  /datapath_result_folder  phase `    

do_make_combo.tcsh shell script creates combo images of each predicted mask in two steps. 
first the predicated mask is added to the groundtruth, next the result is overlayed on the original. 

run_make_combo_images.tcsh shell script prepares the folders, and swarms the do_make_combo.tcsh shell script corresponding to each predicated mask. 



