#!/bin/tcsh

# This program takes data from a parent directory, where associated
# mask+orig (=anatomical) dataset pairs are organized within
# directories that have the following directory structure within some
# parent directory:
#
#     dir_parent
#     |-- testing
#     |   |-- mask
#     |   `-- orig
#     |-- training
#     |   |-- mask
#     |   `-- orig
#     `-- validation
#         |-- mask
#         `-- orig
#
# ... and creates a new data tree with the same structure, but with
# new/resampled matrix dimensions

# PA Taylor (SSCC, NIMH, NIH, USA)

# ===========================================================================
# user chosen info

# string for output tree identification
set label    = 208ish
# dimensions for output matrices (input FS dset matrices are: 256 x 256 x 256)
set nslice_x = 208
set nslice_y = 208
set nslice_z = 192

# get input mask+orig datasets from specified parent directory
set dir_parent = data_00_basic_iso_256
set all_dset   = `find ${dir_parent}/ -name "*mask*nii*"`

# make output directory tree
mkdir -p data_00_basic_iso_${label}/testing/mask
mkdir -p data_00_basic_iso_${label}/testing/orig
mkdir -p data_00_basic_iso_${label}/training/mask
mkdir -p data_00_basic_iso_${label}/training/orig
mkdir -p data_00_basic_iso_${label}/validation/mask
mkdir -p data_00_basic_iso_${label}/validation/orig

# loop over datasets, and make resampled versions
foreach dset ( ${all_dset} )

    # create names for output, with 
    set oname_mask = `echo "${dset}" | sed "s/256/${label}/g"`
    set dset_orig  = `echo "${dset}" | sed "s/mask/orig/g"`
    set oname_orig = `echo "${dset_orig}" | sed "s/256/${label}/g"`

    echo "++ proc: $dset"

    # ----- autobox to center trimming around brain (hopefully)

    # get ijk extent pairs in x,y,z ordering
    set ijk_ran =  `3dAutobox -overwrite \
                    -input "${dset}" -extent_ijkord -prefix __tmp.nii`

    # do mathematics of centering desired matrix sizes around the
    # autoboxed region as best as possible
    @ diff_x = ${ijk_ran[3]} - ${ijk_ran[2]}
    @ diff_y = ${ijk_ran[6]} - ${ijk_ran[5]}
    @ diff_z = ${ijk_ran[9]} - ${ijk_ran[8]}

    set pad_R = `echo "scale=0; ($nslice_x - $diff_x) / 2" | bc`
    set pad_L = `echo "scale=0; $nslice_x - $diff_x - $pad_R - 1" | bc`

    set pad_A = `echo "scale=0; ($nslice_y - $diff_y) / 2" | bc`
    set pad_P = `echo "scale=0; $nslice_y - $diff_y - $pad_A - 1" | bc`

    set pad_I = `echo "scale=0; ($nslice_z - $diff_z) / 2" | bc`
    set pad_S = `echo "scale=0; $nslice_z - $diff_z - $pad_I -1" | bc`

    # ----- apply calculated slicing

    # apply to the mask dataset
    3dZeropad                        \
        -overwrite                   \
        -L ${pad_L} -R ${pad_R}      \
        -A ${pad_A} -P ${pad_P}      \
        -I ${pad_I} -S ${pad_S}      \
        -prefix ${oname_mask}        \
        __tmp.nii

    # ... and to the orig dset, making sure they are on the same grid
    3dZeropad                        \
        -overwrite                   \
        -master ${oname_mask}        \
        -prefix ${oname_orig}        \
        ${dset_orig}

end

echo ""
echo "++ DONE. Checkout the new tree of datasets:"
echo "       data_00_basic_iso_${label}/"
echo ""

exit 0
