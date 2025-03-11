#!/bin/tcsh

# This script provides a rule for another augmentation method we have
# seen that might be useful: lowering the brightness and contrast
# within the brain region of the input anatomical dataset. The script
# does it by raising values within the mask to some power a bit less
# than 1. Blurring is used to smoothly connect the power modulation
# spatially.

# ---------------------------------------------------------------------------

# set Data augmentation directory and filename: daug
set daug   =  $1

# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

cd ${daug_dir} # access the copies of dataset 

# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
echo ${dset}



# the main factor determining the dimming. Making this value smaller
# makes the dimming more extreme.  Values of 3-5 seem reasonable,
# where 3 gives pretty extreme dimming and 5 is mild but noticeable.
set dim_fac = 5




set mask_daug = ${daug:gas/orig/mask/}
set mask_dir = `dirname ${mask_daug}`
echo "mask_dir = ${mask_dir}"

set mask = ${dset:gas/orig/mask/}
echo ${mask}


# temp dsets

set mask_fl   = `python -c "print('temp_'+'${dset}'.split('_')[0] +'_maskfloat.nii.gz')"`
set mask_blur = `python -c "print('temp_'+'${dset}'.split('_')[0] +'_maskblur.nii.gz')"`
# ---------------------------------------------------------------------------

# make sure mask is float type, before blurring



# make sure mask is float type, before blurring
3dcalc                      \
    -overwrite              \
    -a ${mask}         \
    -expr 'a'               \
    -prefix ${mask_fl} \
    -datum float -nscale

# same voxels on the corresponding axis 
set nmat      = `3dinfo -nj ${dset}`

# blur the mask (these values to be used in the power calc)
3dmerge                 \
    -overwrite          \
    -1blur_fwhm 3       \
    -prefix ${mask_blur} \
    ${mask_fl}


# apply mask values as a power modulator on voxel values.
3dcalc                            \
    -overwrite                    \
    -a ${mask_blur}          \
    -b ${dset}               \
    -expr "b**(1-a/${dim_fac})"   \
    -prefix ${dset}      \
    -datum float -nscale



\rm  ${mask_fl} ${mask_blur}

