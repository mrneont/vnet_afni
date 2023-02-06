#!/bin/tcsh

# This script runs 3dDepthMap on each mask in a dset, and creates a
# parallel directory of EDT  dsets.  
# --------------------------------------------------------------------------

set here      = ${PWD}

### --- User parameter to set --- absolute path to toplevel dir
set top_level = ${here}/data_00_basic_iso_128

# subdirs (some exist, some to be made)
set all_tdir   = ( training validation testing )
set dir_mask   = mask
set dir_edt    = edt
set dir_del    = del  #This is a temporary folder created to store intermediate files. 
set flr        = 0.2
set dist_scale = 10   # 6.25  #12.5

# -------------------------------------------------------------------------

cd ${top_level}

# loop over training, testing, etc. dirs
foreach tdir ( ${all_tdir} )
    cd ${tdir}

    echo "------------ work in: ${tdir} ----------------------------"

    # make the weight dir, if necessary
    #\mkdir -p ${dir_edt}

    # get list of all mask dsets
    cd ${dir_mask}
    
    set all_mask = ( *mask.nii* )
    cd -



    # loop over each mask
    foreach dset_mask ( ${all_mask} )
        echo "++ Proc subj mask:   ${dset_mask}"
        
        set dset_dpth  = "${dset_mask:gas/mask.nii.gz/dpth.nii.gz/}"
        set dset_abs   = "${dset_mask:gas/mask.nii.gz/abs.nii.gz/}"
        set dset_wts   = "${dset_mask:gas/mask.nii.gz/wts.nii.gz/}"
        set dset_wtexp = "${dset_mask:gas/mask.nii.gz/edt.nii.gz/}"

        #calculate the depth_map
        3dDepthMap                                         \
            -overwrite \
            -nz_are_neg                                       \
            -input   ${dir_mask}/${dset_mask}                \
            -prefix  ${dir_del}/${dset_dpth}    

        # finding the abs(depth_map)
        3dcalc \
            -overwrite \
            -a       ${dir_del}/${dset_dpth} \
            -expr    "abs(a)" \
            -prefix  ${dir_del}/${dset_abs}
        
        # dpth_abs_min = min(abs(depth_map))
        set dpth_abs_min = `3dBrickStat -min -slow ${dir_del}/${dset_abs}`

        # wts = abs(depth_map) - min(abs(depth_map))
        3dcalc \
            -overwrite \
            -a        ${dir_del}/${dset_abs} \
            -expr    "a-${dpth_abs_min}" \
            -prefix  ${dir_del}/${dset_wts}
        

        #wts_exp = (1-flr)*exp(-0.693*wts/dist_scale)+flr
        3dcalc \
        -overwrite \
        -a       ${dir_del}/${dset_wts} \
        -expr    "(1-${flr})*exp(-0.693*a/${dist_scale})+${flr}" \
        -prefix  ${dir_edt}/${dset_wtexp}
        


    end

    cd ${top_level}
end
