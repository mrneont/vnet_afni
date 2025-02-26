#!/bin/tcsh

# This script is to create transformed datasets from the original datasets
# based on the transformation specified in the 1D parameter files. 

#  daug = $1 -> is the orig dset fname with abs path of the data augmentation dir
#  ${argv[2-13]} is the list of params required for affine transformation. 

# --------------------------------------------------------------------------------

# check for correct number of cmd line args
if ( $#argv != 13 ) then
    echo "** ERROR: Wrong number (${#argv}) of cmd line args."
    echo "          Read the script for details."
    exit 1
endif

# --------------------------------------------------------------------------------

# set Data augmentation directory and filename: daug
set daug  =  $1
set params = ( ${argv[2-13]} )
echo ${params}

# to be more threadsafe, add random string to intermediate temp files
set rndm = "`3dnewid -fun11`"

# separate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

# move into the working dir
cd ${daug_dir} # access the copies of dataset 

# intermed files
set tmp_1D   = temp_${rndm}_aff.1D
set tmp_mask = temp_${rndm}_mask.nii.gz

# save affine params to 1D file
echo "${params}" >> ${tmp_1D}

echo "Params file"
echo "${tmp_1D}"
# separate  out the basename of the dataset 
set dset = `basename ${daug}`
echo ${dset}

set mask_daug = ${daug:gas/orig/mask/}
set mask_dir = `dirname ${mask_daug}`
echo ${mask_dir}

set mask = ${dset:gas/orig/mask/}
echo ${mask}

# get dtype of mask, to match with final output
set dtype = `3dinfo -datum ${mask_dir}/${mask}`

#echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
echo "doing affine"
3dAllineate -overwrite -input ${dset}       \
                -master  ${dset}   \
                -prefix ${dset}  \
                -final wsinc5  \
                -1Dparam_apply ${tmp_1D}

# now use linear interp, and binarize below
3dAllineate -echo_edu -overwrite -input ${mask_dir}/${mask}       \
                -master ${mask_dir}/${mask}       \
                -prefix ${tmp_mask}  \
                -final linear -float          \
                -1Dparam_apply ${tmp_1D}

# threshold mask and binarize (and make sure the output dtype matches
# the input dtype); the thr >0.3 is meant to be a bit more generous
# than just 3dAllineate with NN mode
3dcalc                             \
    -overwrite                     \
    -a ${tmp_mask}                 \
    -expr "step(a-0.3)"            \
    -prefix ${mask_dir}/${mask}    \
    -datum ${dtype}                \
    -nscale 

# clean up intermed files
\rm ${tmp_1D} ${tmp_mask}

