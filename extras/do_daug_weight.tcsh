#!/bin/tcsh

# This script runs 3dDepthMap on each mask in a dset, and creates a
# parallel directory of EDT  dsets.  
# --------------------------------------------------------------------------

# set Data augmentation directory and filename: daug
set daug  =  $1

# seperate out the  folder name 

# daug = orig folder/orig_data
set daug_dir = `dirname ${daug}`
echo "++ The orig folder :   ${daug_dir}"

# seperate  out the filename of the dataset 
# orig_data
set dset = `basename ${daug}`
echo "++ The orig basename :   ${dset}"

# -------------------------------------------------------------------
# mask_daug = mask folder/mask_data
set mask_daug = ${daug:gas/orig/mask/}
# mask folder
set mask_dir = `dirname ${mask_daug}`
#echo "++ The mask folder :    ${mask_dir}"
# mask_data
set dset_mask = ${dset:gas/orig/mask/}
#echo "++ The mask basename :   ${dset_mask}"
# -------------------------------------------------------------------

# edt_daug = edt folder/edt_data
set edt_daug = ${daug:gas/orig/edt/}
# edt folder
set edt_dir  = `dirname ${edt_daug}`
#echo "++ The edt folder :    ${edt_dir}"

# edt data
set dset_wtexp = `basename ${edt_daug}`
#echo "++ The edt basename :    ${dset_wtexp}"

# -------------------------------------------------------------------

set dset_dpth  = "${dset_mask:gas/mask.nii.gz/dpth.nii.gz/}"
set dset_abs   = "${dset_mask:gas/mask.nii.gz/abs.nii.gz/}"
set dset_wts   = "${dset_mask:gas/mask.nii.gz/wts.nii.gz/}"
set dset_wtexp = "${dset_mask:gas/mask.nii.gz/edt.nii.gz/}"

#echo ${dset_dpth}
#echo ${dset_abs}
#echo ${dset_wts}
#echo ${dset_wtexp}

cd ${mask_dir} # access the copies of dataset masks 

#echo ${PWD}

set flr        = 0.2
set dist_scale = 10   # 6.25  #12.5

#calculate the depth_map
3dDepthMap                                         \
    -overwrite \
    -nz_are_neg                                       \
    -input   ${dset_mask}                \
    -prefix  ${edt_dir}/${dset_dpth}


# finding the abs(depth_map)
3dcalc \
    -overwrite \
    -a  ${edt_dir}/${dset_dpth} \
    -expr    "abs(a)" \
    -prefix  ${edt_dir}/${dset_abs}

# dpth_abs_min = min(abs(depth_map))
set dpth_abs_min = `3dBrickStat -min -slow ${edt_dir}/${dset_abs}`

# wts = abs(depth_map) - min(abs(depth_map))
3dcalc \
    -overwrite \
    -a  ${edt_dir}/${dset_abs} \
    -expr    "a-${dpth_abs_min}" \
    -prefix  ${edt_dir}/${dset_wts}


#wts_exp = (1-flr)*exp(-0.693*wts/dist_scale)+flr
3dcalc \
    -overwrite \
    -a   ${edt_dir}/${dset_wts} \
    -expr    "(1-${flr})*exp(-0.693*a/${dist_scale})+${flr}" \
    -prefix  ${edt_dir}/${dset_wtexp}


rm ${edt_dir}/${dset_dpth}
rm ${edt_dir}/${dset_abs}
rm ${edt_dir}/${dset_wts}


