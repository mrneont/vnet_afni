#!/bin/tcsh

# notes about the script : 
# + Script to prepare a folder for  make_combo_images.
# make_combo_images_folder
# --> orig
# --> pred_mask
# --> pred_plus_target
#--> combo_images


# set pred_mask_dir : pred_mask_output_folder
set pred_mask_output_dir      =  $1

#echo ${pred_mask_output_dir}
# The combo images pertaining to ${pred_mask_output_dir}
# is present in the ${make_combo_images_dir}

set basename = `basename ${pred_mask_output_dir}`
#echo ${basename}
set make_combo_images_dir =  `python -c "print('make_combo_images_'+'${basename}')"`
#echo ${make_combo_images_dir}


set combo_orig_dir       = "orig"
set combo_target_dir     = "target"
set combo_pred_mask_dir  = "pred_mask"
set pred_plus_target_dir = "pred_plus_target"
set combo_images_dir     = "combo_images"
set scripts_dir          = "scripts"


mkdir ${make_combo_images_dir}
mkdir ${make_combo_images_dir}/${combo_orig_dir} 
mkdir ${make_combo_images_dir}/${combo_target_dir} 
mkdir ${make_combo_images_dir}/${combo_pred_mask_dir} 
mkdir ${make_combo_images_dir}/${pred_plus_target_dir}
mkdir ${make_combo_images_dir}/${combo_images_dir}
mkdir ${make_combo_images_dir}/${scripts_dir}


set pred_mask_key = "ch01_ep-100_train"
set orig_key      = "orig_train_subj_sub"
set target_key    = "target_000_train_subj_sub"
set scripts_dir = ${make_combo_images_dir}/${scripts_dir}
echo ${scripts_dir}
#copy the pred_masks

cp  ${pred_mask_output_dir}/*"${pred_mask_key}"*   \
                    ${make_combo_images_dir}/${combo_pred_mask_dir} 

#copy the original data 
cp  ${pred_mask_output_dir}/*"${orig_key}"*   \
                    ${make_combo_images_dir}/${combo_orig_dir} 

#copy the target data 
cp  ${pred_mask_output_dir}/*"${target_key}"*   \
                    ${make_combo_images_dir}/${combo_target_dir} 


cd ${make_combo_images_dir}/${combo_pred_mask_dir} 
echo ${PWD}

set all_pred_mask = (*.nii.gz)

#do Pred_mask + target
foreach dset_pred_mask ( ${all_pred_mask} )

    echo ${dset_pred_mask}
    set new_extension = ".tcsh"
    #print('file_name=',file_name)


    

    set temp = `python -c "print('combo_sub_' + '${dset_pred_mask}'.split('sub-')[1])"`
    set script_fl = `python -c "print('${temp}'.split('.')[0]+ '${new_extension}')"`
    echo ${script_fl}
   
    
   


end 




