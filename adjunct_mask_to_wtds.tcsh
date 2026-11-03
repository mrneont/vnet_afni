#!/bin/tcsh

# A short script to estimate the weight dataset (wtds) for a given mask.
# 
# The user provides exactly two arguments:
# + the name of the input dataset (binarized mask dataset)
# + the name of the output dataset (the wtds, to be created)
# 
# Example:
#
#   adjunct_mask_to_wtds.tcsh                   \
#       training/mask/sub-456_mask.nii.gz       \
#       training/wtds/sub-456_wtds.nii.gz
#
# auth : Y Narayana Swamy (SSCC, NIMH, NIH, USA)
#        RC Reynolds (SSCC, NIMH, NIH, USA)
#        PA Taylor (SSCC, NIMH, NIH, USA)
# 
# ============================================================================
# read in cmd line args

# I/O data names, including any path info
set dset_in  = "$1"
set dset_out = "$2"

if ( "${dset_out}" == "" ) then
    echo "** ERROR: need to provide 2 command line args:"
    echo "          DSET_IN DSET_OUT"
    exit -1
endif

# ============================================================================
# parameters for depth mask

set flr        = 0.2    # minimum/floor value within mask
set dist_scale = 10     # scale for expon fade; smaller -> steeper slope

# ----------------------------------------------------------------------------
# get/set/make relevant directories

# get directory info
set here = ${PWD}
set odir = `dirname "${dset_out}"`

# set workdir
set tmp  = `3dnewid -fun11`
set wdir = "${odir}/__m2w_${tmp}"

\mkdir -p "${wdir}"

# ============================================================================
# do actual work

# ----- calculate the depth map

# inside/outside of mask are just positive distance values to mask edge
set dset_depth = "${wdir}/data_00_depth.nii.gz"
3dDepthMap                                                                  \
    -overwrite                                                              \
    -input       "${dset_in}"                                               \
    -prefix      "${dset_depth}"

if ( $status ) then
    echo "** ERROR: 3dDepthMap failed"
    exit -1
endif

# ----- get min value of dset_depth

set min_depth = `3dBrickStat -min -slow "${dset_depth}"`

if ( $status ) then
    echo "** ERROR: 3dBrickStat failed"
    exit -1
endif

# ----- calc final weight dataset (wtds) 

# scale depth map to get wtds (final values should be between 0.2-1.0)
3dcalc                                                                      \
    -overwrite                                                              \
    -a       "${dset_depth}"                                                \
    -expr    "(1-${flr})*exp(-0.693*(a-${min_depth})/${dist_scale})+${flr}" \
    -prefix  "${dset_out}"

if ( $status ) then
    echo "** ERROR: 3dcalc failed"
    exit -1
endif

# ----- done, finish up

# clean up wdir
\rm -rf "${wdir}"

echo "++ Done making wtds : ${dset_out}"

exit 0
