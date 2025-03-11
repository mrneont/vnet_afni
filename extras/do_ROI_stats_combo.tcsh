#!/bin/tcsh

# Script to combine the pred_mask and the target(groundtruth)

# notes
# 1) 'do_pred_plus_target.tcsh' is a post-script used before the
#    'do_combo_orig_olay_pred_plus_target.tcsh' script.
# 2)  An intermediate result is created where the predicted mask and the target are added.
# 3)  The intermediate result is named 'out_suffix.nii.gz'. The suffix is the the name of the dataset. 
# 4) 'do_combo_orig_olay_pred_plus_target.tcsh' script overlays the 'out_suffix.nii.gz'
#     over the priginal dataset

#folder structure
#  parent_dir     : parent directory (The full path is given )
#  dir_pred_mask  : folder which has the predicted masks 
#  dir_target     : folder which has the target masks (groundtruth)
#  dir_out        : folder where the result 'out_suffix.nii.gz' is written/stored

# usage example 
# tcsh do_pred_plus_target_cmd.tcsh /Users/narayanaswamyy2/AFNI_VNET/FQC_DA_c14ed246 

echo "Script to evaluate the ROI stats for the combo images"
set parent_dir    = $1  # cmd line arguement 

cd ${parent_dir}

echo ${PWD}

set all_combo_mask = (*.nii.gz)

#echo ${all_pred_mask}

foreach dset_combo_mask ( ${all_combo_mask} )

    echo "inside for loop"
    
    echo "++ DSET: ${dset_combo_mask}"

    set base  = `3dROIstats -quiet -nzvoxels -mask  "${dset_combo_mask}"  "${dset_combo_mask}"`

    #nzvoxel

    echo ${dset_combo_mask} $base >> log.csv
    #3dROIstats -quiet -nzvolume -mask  ${dset_combo_mask}  ${dset_combo_mask} >> log.csv


end