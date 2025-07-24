#!/bin/tcsh

# notes about the script : 
# + Script to prepare a folder for  data_augmentation.
# + The data_augmentation folder is prepared by  creating copies 
#     of the original datasets and the masks
# + The number of copies are named serially with leading zeros. 
#    example: 
#    If original dataset were 'sub-114_orig.nii.gz'
#    Copies will be named as below 
#    sub-114000_orig.nii.gz  - original copy
#    sub-114001_orig.nii.gz  - augmented 
#    sub-114002_orig.nii.gz  - augmented
#    sub-114003_orig.nii.gz  - augmented

#    If original dataset were 'sub-114_mask.nii.gz'
#    Copies will be named as below 
#    sub-114000_mask.nii.gz  - original copy
#    sub-114001_mask.nii.gz  - augmented 
#    sub-114002_mask.nii.gz  - augmented
#    sub-114003_mask.nii.gz  - augmented


# dataset_dir  : is the dataset directory
# data_aug_dir : is the data Data augmentation directory 

# set Data augmentation directory : DA_dir
set dataset_dir      =  $1
set data_aug_dir     =  $2
# number of copies of the dataset to be made 
set num_cp           =  $3

set dataset_orig_dir = "orig"
set dataset_mask_dir = "mask"

set dataaug_orig_dir = "orig"
set dataaug_mask_dir = "mask"
set dataaug_scripts  = "scripts"
set dataaug_logs     = "logs"

# make sure output directory doesn't exist already
if ( -d "${data_aug_dir}" ) then
    echo "** ERROR: top level dir '${data_aug_dir}' already exists" 
    echo "   Please move/remove it and try again"
    exit 1
endif

\mkdir -p ${data_aug_dir} 
\mkdir -p ${data_aug_dir}/${dataaug_orig_dir}
\mkdir -p ${data_aug_dir}/${dataaug_mask_dir}
\mkdir -p ${data_aug_dir}/${dataaug_scripts}
\mkdir -p ${data_aug_dir}/${dataaug_logs}

echo "++ ${data_aug_dir}/${dataaug_mask_dir}"

# get list of all  dsets
cd ${dataset_dir}/${dataset_orig_dir}
set all_dsets = ( *orig.nii* )
cd -

echo "++ Found ${#all_dsets} dsets: ${all_dsets}"

# loop over all dsets, making copies
foreach dset_orig ( ${all_dsets} )
    echo "++ Process dset: ${dset_orig}"

    # loop over num_cp times and append zero-based count labels for each copy
    @ nmax = ${num_cp} - 1
    foreach i ( `seq 0 1 ${nmax}` )
        # set prefix for the copies of the dataset 
        set prefix       = `python -c "print(str(${i}).rjust(3,'0'))"`
        echo ${prefix}

        set dataset_copy = `python -c "print('${dset_orig}'.split('_')[0] +'${prefix}'+\
                                    '_'+'orig'+\
                                    '.'+'${dset_orig}'.split('.')[-2] +\
                                    '.'+'${dset_orig}'.split('.')[-1])"`
        echo ${dataset_copy}

        set dset_mask      = ${dset_orig:gas/orig/mask/}
        set dset_mask_copy = ${dataset_copy:gas/orig/mask/}

        echo ${dset_mask}
        echo ${dset_mask_copy}

        # copy orig dset
        \cp  ${dataset_dir}/${dataset_orig_dir}/${dset_orig}   \
             ${data_aug_dir}/${dataaug_orig_dir}/${dataset_copy}

        if ( $status) then
            echo "** Failed to copy: ${dset_orig} -> ${dataset_copy}"
            exit 2
        endif

        # copy mask dset
        \cp  ${dataset_dir}/${dataset_mask_dir}/${dset_mask}   \
             ${data_aug_dir}/${dataaug_mask_dir}/${dset_mask_copy}

        if ( $status) then
            echo "** Failed to copy: ${dset_mask} -> ${dset_mask_copy}"
            exit 2
        endif
    end             
end 

# successful exit
exit 0
