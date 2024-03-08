#!/bin/tcsh

# script to introduce the noise into the datasets. 

# set Data augmentation directory and filename: daug
set daug      =  $1
set axis_ran  =  $2
set noise_lvl =  $3


# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

cd ${daug_dir} # access the copies of dataset 

# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
echo ${dset}


# find the median  of the voxel value in the dset
set vals   = `3dBrickStat -slow -automask -median ${dset}`

set med    = ${vals[2]}
    
3dcalc                   \
        -overwrite               \
        -a  ${dset}               \
        -expr "a + gran(0,${med})* ${noise_lvl}"          \
        -prefix ${dset} \
        -datum float \
        -nscale 

# mask will not be altered 