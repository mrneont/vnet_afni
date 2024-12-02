#!/bin/tcsh

# script to introduce the gain inhomogentity into the datasets. 

# daug = $1 -> is the absolute path of the data augmentation dir

# set Data augmentation directory and filename: daug
set daug   =  $1
set axis   =  $2
set window =  $3 

# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

cd ${daug_dir} # access the copies of dataset 

# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
echo ${dset}


#window over which the scaling factor varies 
# the scaling factor varies between values 'win_min' and 'window' 

set win_min = `ccalc 1-${window}/2` 

echo ${win_min}

echo "window"
echo ${window}

# same voxels on the corresponding axis 
set nmat      = `3dinfo -nj ${dset}`



3dcalc                                         \
    -overwrite                                 \
    -a  ${dset}   \
    -expr "a * (${window} * (1-${axis}/${nmat})+ ${win_min})" \
    -prefix  ${dset}



