#!/bin/tcsh


# script to make QC images of mask over anatomical
# Script to overlay the pred_mask+target(groundtruth) over original.
# 'do_pred_plus_target.tcsh' is a script to be used before using 
#  do_combo_orig_olay_pred_plus_target.tcsh
# 'out_suffix.nii.gz' is created by 'do_pred_plus_target.tcsh'



#folder structure
#  parent_dir            : parent directory (The full path is given )
#  dir_pred_plus_target  : folder where the result 'out_suffix.nii.gz' is written/stored
#                          pred_mask+target 
#  dir_orig              : folder which has the original data
#  dir_combo             : folder where the QC volumes of pred_mask+target is overlayed on original data 

# even works appropriately if the mask is not binary, but floats in
# range [0, 1]

set parent_dir = $1
set dir_pred_plus_target = "training_pred_plus_target"
set dir_orig = "training_orig" # "../FQC_training_orig"
set dir_combo = "training_combo_pred_traget_orig" # "../training_combo_pred_traget_orig"

echo "${dir_pred_plus_target}"
echo "${dir_orig}"

cd ${parent_dir}/${dir_pred_plus_target}
set all_pred_plus_target = ( out*.nii.gz )
echo "**************************************************************************************"
echo ${all_pred_plus_target}

foreach dset_mask ( ${all_pred_plus_target} )
    echo "++ DSET: ${dset_mask}"

    set dset_orig = `python -c "print('orig_train_subj_'+ \
                                 ''.join('sub-' + '${dset_mask}'.split('sub-')[1]))"`

    echo "++ orig :${dset_orig}"

    set base  = `3dinfo -prefix_noext "${dset_mask}"`
    
    
    set prefix = `python -c "print(''.join('sub-' + '${base}'.split('sub-')[1]))"`
    echo ${prefix}
    # output prefix for montage image
    set opref = IMG_${prefix}

    # make three PNGs
    @chauffeur_afni                                  \
        -ulay  ${parent_dir}/${dir_orig}/${dset_orig}                          \
        -olay  ${parent_dir}/${dir_pred_plus_target}/${dset_mask}                          \
        -box_focus_slices AMASK_FOCUS_OLAY           \
        -ulay_range 0% 98%                           \
        -func_range 3t                                  \
        -cbar "RedBlueGreen"                   \
        -pbar_posonly                                \
        -opacity     4                               \
        -blowup      2                               \
        -prefix      ${parent_dir}/${dir_combo}/${opref}                        \
        -montx 3 -monty 3                            \
        -set_xhairs OFF                              \
        -label_mode 1 -label_size 4                  \
        -cmd2script        ${parent_dir}/${dir_combo}/${opref}_run.tcsh   \
        -do_clean 

    # glue together separate PNGs
    2dcat                                            \
        -gap     5                                   \
        -gap_col 150 150 150                         \
        -nx 1                                        \
        -ny 3                                        \
        -prefix  ${parent_dir}/${dir_combo}/${opref}_FINAL.jpg                  \
        ${parent_dir}/${dir_combo}/${opref}*{sag,axi,cor}*png

    if ( $status ) then
        echo "** ERROR: exit, badness for: ${dset_mask}"
        exit 1
    endif


    # clean up, remove separate PNGs
    \rm ${parent_dir}/${dir_combo}/${opref}*{sag,axi,cor}*png

    echo "++ Done: ${opref}_FINAL.jpg"


end


exit
