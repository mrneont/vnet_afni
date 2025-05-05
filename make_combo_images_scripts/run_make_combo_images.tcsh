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

# 
set dir_parent = $2
#echo ${pred_mask_output_dir}
# The combo images pertaining to ${pred_mask_output_dir}
# is present in the ${make_combo_images_dir}

set basename = `basename ${pred_mask_output_dir}`

#echo ${basename}
set make_combo_images_dir =  `python -c "print('make_combo_images_'+'${basename}')"`
#echo ${make_combo_images_dir}
set scr_swarm     = "swarm_make_combo.txt"
if ( -e ${scr_swarm} ) then
    \rm ${scr_swarm}
endif

if ( -d ${make_combo_images_dir} ) then
    echo "folder exists"
    exit
else
    echo "folder does not exist"
    echo "folder will be created"

set dir_orig       = "orig"
set dir_target     = "target"
set dir_pred_mask  = "pred_mask"
set dir_pred_plus_target = "pred_plus_target"
set dir_combo      = "combo_images"
#set dir_swarm      = "swarms"
set dir_logs       = "logs"


mkdir -p ${dir_parent}/${make_combo_images_dir}
mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_orig} 
mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_target} 
mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_pred_mask} 
mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_pred_plus_target}
mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_combo}
#mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_swarm}
mkdir -p ${dir_parent}/${make_combo_images_dir}/${dir_logs}


set pred_mask_key = "ch01_ep-100_train"
set orig_key      = "orig_train_subj_sub"
set target_key    = "target_000_train_subj_sub"

#copy the pred_masks

cp  ${pred_mask_output_dir}/*"${pred_mask_key}"*   \
                    ${dir_parent}/${make_combo_images_dir}/${dir_pred_mask} 

#copy the original data 
cp  ${pred_mask_output_dir}/*"${orig_key}"*   \
                    ${dir_parent}/${make_combo_images_dir}/${dir_orig} 

#copy the target data 
cp  ${pred_mask_output_dir}/*"${target_key}"*   \
                    ${dir_parent}/${make_combo_images_dir}/${dir_target} 


cd ${dir_parent}/${make_combo_images_dir}/${dir_pred_mask} 
#echo ${PWD}




#cat swarm_make_combo.txt
set all_pred_mask = (*.nii.gz)

#do Pred_mask + target
foreach dset_pred_mask ( ${all_pred_mask} )

    #echo ${dset_pred_mask}
    #echo "tcsh -xf do_make_combo.tcsh ${make_combo_images_dir} ${dset_pred_mask} \\"    >> ${scr_swarm}
    
    set prefix = `python -c "print(''.join('sub-' + '${dset_pred_mask}'.split('sub-')[1]))"`
    set prefix = `python -c "print('${prefix}'.split('.')[0])"`
    set logtxt = `python -c "print('log_'+ '${prefix}'+'.txt')"`
    #echo ${logtxt}

    echo "tcsh -xf ${dir_parent}/do_make_combo.tcsh ${dir_parent}/${make_combo_images_dir} ${dset_pred_mask}  |& tee ../${dir_logs}/${logtxt}" >> ${dir_parent}/${scr_swarm}

    
   
    
   


end 


# -------------------------------------------------------------------------
# run swarm command
#cp ${dirparent}/do_make_combo.tcsh  ${dirparent}/${make_combo_images_dir}/${dir_pred_mask} 
#echo "dirparent"
#echo ${dir_parent}
echo ${PWD}

set cmd = "combo"
echo "++ And start swarming: ${scr_swarm}"

swarm                                                              \
    -f ${dir_parent}/${scr_swarm}                                                \
    --partition=norm,quick                                         \
    --threads-per-process=2                                       \
    --gb-per-process=3                                            \
    --time=0:15:00                                                \
    --logdir=../${dir_logs}                                           \
    --job-name=job_${cmd}                                          \
    --merge-output                                                 \
    --usecsh




