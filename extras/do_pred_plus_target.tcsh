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

echo "Script to combine the pred_mask and the target(groundtruth)"
set parent_dir    = $1  # cmd line arguement 
set dir_pred_mask = "training_pred_mask"
set dir_target    = "training_target_mask"
set dir_out       = "training_pred_plus_target"

cd ${parent_dir}/${dir_pred_mask}
echo ${PWD}

set all_pred_mask = (*.nii.gz)

#echo ${all_pred_mask}

foreach dset_pred_mask ( ${all_pred_mask} )

	echo "inside for loop"
	
	echo "++ DSET: ${dset_pred_mask}"
	set dset_target = `python -c "print('target_000_train_subj_'+ \
	                             ''.join('sub-' + '${dset_pred_mask}'.split('sub-')[1]))"`
	
	set dset_out = `python -c "print('out_' + \
	               ''.join('sub-' + '${dset_pred_mask}'.split('sub-')[1]))"`

	echo ${dset_out}

	echo "${parent_dir}/${dir_target}/${dset_target}"

	# combo is overlap mapped to 1(green)
	# a is vnet_pred_mask mapped to 3(red)
	# b is FS/target mask mapped to 2(blue)
	3dcalc -prefix ${parent_dir}/${dir_out}/${dset_out}\
       -expr 'bool(ispositive(a-0.5)+b)*(4-(ispositive(a-0.5)+2*b))' \
       -a ${parent_dir}/${dir_pred_mask}/${dset_pred_mask} \
       -b ${parent_dir}/${dir_target}/${dset_target} \
       -overwrite 

end	