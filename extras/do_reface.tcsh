#!/bin/tcsh


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

# access the copies of dataset 
cd ${daug_dir}

@afni_refacer_run   \
    -input ${dset}  \
    -mode_reface    \
    -prefix ${dset} \
    -no_images 


