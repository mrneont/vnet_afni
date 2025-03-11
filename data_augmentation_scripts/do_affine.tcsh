#!/bin/tcsh

# This script is to create transformed datasets from the original datasets
# based on the transformation specified in the 1D parameter files. 

#  daug = $1 -> is the absolute path of the data augmentation dir
#  ${argv[2-13]} is the list of params required for affine transformation. 

# using the script 
# tcsh do_affine.tcsh 

# set Data augmentation directory and filename: daug
set daug  =  $1
set params = ( ${argv[2-13]} )
echo ${params}



# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo "dirname = ${daug_dir}"


cd ${daug_dir} # access the copies of dataset 

set folder1D = ${daug_dir}/file1D
mkdir -p ${folder1D}

echo " FOLDER1D = ${folder1D}"
# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
echo "basename = ${dset}"

#sub-001001_orig
set file1D =  `python -c "print('temp_'+'${dset}'.split('_')[0] +'.1D')"`
echo "file1D = ${file1D}"
echo "${params}" >> ${folder1D}/${file1D}

set tmp_mask = `python -c "print('temp_'+'${dset}'.split('_')[0] +'_mask.nii.gz')"`


set mask_daug = ${daug:gas/orig/mask/}
set mask_dir = `dirname ${mask_daug}`
echo "mask_dir = ${mask_dir}"

set mask = ${dset:gas/orig/mask/}
#echo ${mask}

# get dtype of mask, to match with final output
set dtype = `3dinfo -datum ${mask_dir}/${mask}`

#echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
echo "doing affine"
3dAllineate -overwrite -input ${dset}       \
                -master  ${dset}   \
                -prefix ${dset}  \
                -final wsinc5  \
                -1Dparam_apply ${folder1D}/${file1D}

3dAllineate -overwrite -input ${mask_dir}/${mask}       \
                -master ${mask_dir}/${mask}       \
                -prefix ${tmp_mask}  \
                -final linear -float          \
                -1Dparam_apply ${folder1D}/${file1D}


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
\rm  ${tmp_mask}

