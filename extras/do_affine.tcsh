#!/bin/tcsh

# This script is to create transformed datasets from the original datasets
# based on the transformation specified in the 1D parameter files. 

#  daug = $1 -> is the absolute path of the data augmentation dir
#  ${argv[2-13]} is the list of params required for affine transformation. 


# set Data augmentation directory and filename: daug
set daug  =  $1
set params = ( ${argv[2-13]} )
echo ${params}


# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}
echo "${params}" >> ${daug_dir}/temp.1D

cd ${daug_dir} # access the copies of dataset 

echo "Params file"
echo ${daug_dir}/temp.1D
# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
echo ${dset}

set mask_daug = ${daug:gas/orig/mask/}
set mask_dir = `dirname ${mask_daug}`
echo ${mask_dir}

set mask = ${dset:gas/orig/mask/}
echo ${mask}

#echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
echo "doing affine"
3dAllineate -overwrite -input ${dset}       \
                -master  ${dset}   \
                -prefix ${dset}  \
                -final wsinc5  \
                -1Dparam_apply ${daug_dir}/temp.1D

3dAllineate -overwrite -input ${mask_dir}/${mask}       \
                -master ${mask_dir}/${mask}       \
                -prefix ${mask_dir}/${mask}  \
                -final NN          \
                -1Dparam_apply ${daug_dir}/temp.1D

rm ${daug_dir}/temp.1D

