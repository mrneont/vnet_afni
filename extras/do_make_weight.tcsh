#!/bin/tcsh

# This script runs 3dDepthMap on each mask in a dset, and creates a
# parallel directory of EDT  dsets.  
# --------------------------------------------------------------------------

set here      = ${PWD}

### --- User parameter to set --- absolute path to toplevel dir
set top_level = ${here}/data_00_basic_iso_64

# subdirs (some exist, some to be made)
set all_tdir  = ( testing training  validation )
set dir_mask  = mask
set dir_edt    = edt

# -------------------------------------------------------------------------

cd ${top_level}

# loop over training, testing, etc. dirs
foreach tdir ( ${all_tdir} )
    cd ${tdir}

    echo "------------ work in: ${tdir} ----------------------------"

    # make the weight dir, if necessary
    \mkdir -p ${dir_edt}

    # get list of all mask dsets
    cd ${dir_mask}
    set all_mask = ( *mask.nii* )
    cd -

    # loop over each mask
    foreach dset_mask ( ${all_mask} )
        echo "++ Proc subj mask:   ${dset_mask}"
        set dset_edt   = "${dset_mask:gas/mask.nii.gz/edt.nii.gz/}"
        #set dset_wtlin = "${dset_mask:gas/mask.nii.gz/wtlin.nii.gz/}"
        #set dset_wtexp = "${dset_mask:gas/mask.nii.gz/weight.nii.gz/}"

        3dDepthMap                                         \
            -nz_are_neg                                       \
            -input   ${dir_mask}/${dset_mask}                \
            -prefix  ${dir_edt}/${dset_edt}    

 

    end

    cd ${top_level}
end
