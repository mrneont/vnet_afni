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

#echo "Script to combine the pred_mask and the target(groundtruth)"
set parent_dir    = $1  # cmd line arguement 
set dset_pred_mask = $2


set dir_target = "target"
set dir_pred_mask = "pred_mask"
set dir_pred_plus_target       = "pred_plus_target"
set dir_orig      = "orig"
set dir_combo      = "combo_images"


set dset_target = `python -c "print('target_000_train_subj_'+ \
                   ''.join('sub-' + '${dset_pred_mask}'.split('sub-')[1]))"`

set dset_out = `python -c "print('out_' + \
               ''.join('sub-' + '${dset_pred_mask}'.split('sub-')[1]))"`

set dset_orig = `python -c "print('orig_train_subj_'+ \
                                 ''.join('sub-' + '${dset_pred_mask}'.split('sub-')[1]))"`

echo ${dset_orig}

echo ${dset_out}

set base  = `3dinfo -prefix_noext "${dset_target}"`


echo ${base}


set prefix = `python -c "print(''.join('sub-' + '${dset_target}'.split('sub-')[1]))"`
set prefix = `python -c "print('${prefix}'.split('.')[0])"`
echo "prefix"
echo ${prefix}
    # output prefix for montage image
set opref = IMG_${prefix}

echo "${parent_dir}/${dir_target}/${dset_target}"

	# combo is overlap mapped to 1(green)
	# a is vnet_pred_mask mapped to 3(red)
	# b is FS/target mask mapped to 2(blue)
3dcalc -prefix ${parent_dir}/${dir_pred_plus_target}/${dset_out}\
    -expr 'bool(ispositive(a-0.5)+b)*(4-(ispositive(a-0.5)+2*b))' \
    -a ${parent_dir}/${dir_pred_mask}/${dset_pred_mask} \
    -b ${parent_dir}/${dir_target}/${dset_target} \
    -overwrite 

     # make three PNGs
@chauffeur_afni                                  \
        -ulay  ${parent_dir}/${dir_orig}/${dset_orig}                          \
        -olay  ${parent_dir}/${dir_pred_plus_target}/${dset_out}                          \
        -box_focus_slices AMASK_FOCUS_OLAY           \
        -ulay_range 0% 98%                           \
        -func_range 3t                                  \
        -cbar "RedBlueGreen"                   \
        -pbar_posonly                                \
        -opacity     4                               \
        -blowup      2                               \
        -prefix      ${parent_dir}/${dir_combo}/${opref}                        \
        -montx 6 -monty 1                            \
        -set_xhairs OFF                              \
        -label_mode 1 -label_size 4                  \
        -do_clean 
        #-cmd2script        ${parent_dir}/${dir_combo}/${opref}_run.tcsh   \
        

    # glue together separate PNGs
    2dcat                                            \
        -gap     5                                   \
        -gap_col 150 150 150                         \
        -nx 1                                        \
        -ny 3                                        \
        -prefix  ${parent_dir}/${dir_combo}/${opref}.jpg                  \
        ${parent_dir}/${dir_combo}/${opref}*{sag,axi,cor}*png

if ( $status ) then
    echo "** ERROR: exit, badness for: ${dset_out}"
    exit 1
endif


    # clean up, remove separate PNGs
\rm ${parent_dir}/${dir_combo}/${opref}*{sag,axi,cor}*png


end	