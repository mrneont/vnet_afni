#!/bin/tcsh

# This script runs 3dEulerDist on each mask in a dset, and creates a
# parallel directory of EDT and weight dsets.  The weights come in 2
# varieties: linear by distance (*_wtlin.nii.gz) and exponential by
# distance (*_wtexp.nii.gz).

# --------------------------------------------------------------------------

set here      = ${PWD}

### --- User parameter to set --- absolute path to toplevel dir
set top_level = ${here}/data_00_basic_iso_64

# subdirs (some exist, some to be made)
set all_tdir  = ( testing  training  validation )
set dir_mask  = mask
set dir_wt    = weight

# -------------------------------------------------------------------------

cd ${top_level}

# loop over training, testing, etc. dirs
foreach tdir ( ${all_tdir} )
    cd ${tdir}

    echo "------------ work in: ${tdir} ----------------------------"

    # make the weight dir, if necessary
    \mkdir -p ${dir_wt}

    # get list of all mask dsets
    cd ${dir_mask}
    set all_mask = ( *mask.nii* )
    cd -

    # loop over each mask
    foreach dset_mask ( ${all_mask} )
        echo "++ Proc subj mask:   ${dset_mask}"
        set dset_edt   = "${dset_mask:gas/mask.nii.gz/edt.nii.gz/}"
        set dset_wtlin = "${dset_mask:gas/mask.nii.gz/wtlin.nii.gz/}"
        set dset_wtexp = "${dset_mask:gas/mask.nii.gz/wtexp.nii.gz/}"

        3dEulerDist                                          \
            -overwrite                                       \
            -zeros_are_zero                                  \
            -input   ${dir_mask}/${dset_mask}                \
            -prefix  ${dir_wt}/${dset_edt}    

        set max_edt  = `3dinfo -dmaxus ${dir_wt}/${dset_edt}`
        echo "++ max_edt = ${max_edt}"

        # dset with linear weight-by-distance
        3dcalc                                               \
            -overwrite                                       \
            -a       ${dir_wt}/${dset_edt}                   \
            -expr    "step(a)*(1.1 - a/${max_edt})"          \
            -prefix  ${dir_wt}/${dset_wtlin}

        # dset with exponential weight-by-distance
        3dcalc                                               \
            -overwrite                                       \
            -a       ${dir_wt}/${dset_edt}                   \
            -expr    "step(a)*exp(- a/${max_edt})"           \
            -prefix  ${dir_wt}/${dset_wtexp}

    end

    cd ${top_level}
end
