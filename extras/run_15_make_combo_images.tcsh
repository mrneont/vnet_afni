#!/bin/tcsh

# MAKE_COMBO: run combo image generation for VNET
# -> using make_combo_images_wrapper.py

# This script processes a corresponding do_*.tcsh script.
# Can be run on either a slurm/swarm system (like Biowulf) or on a desktop.

# To execute:
#     tcsh run_15_make_combo_images.tcsh

# ---------------------------------------------------------------------------

# use slurm? 1 = yes, 0 = no, def: use if available
set use_slurm = $?SLURM_CLUSTER_NAME

# set unique command name
set cmd           = 15_make_combo_images

# directory structure
set dir_scr       = $PWD
set dir_inroot    = ..
set dir_log       = ${dir_inroot}/logs
set dir_swarm     = ${dir_inroot}/swarms

# names for logs and swarm
set cdir_log      = ${dir_log}/logs_${cmd}
set scr_swarm     = ${dir_swarm}/swarm_${cmd}.txt
set scr_cmd       = ${dir_scr}/do_${cmd}.tcsh

# --------------------------------------------------------------------------
# create log + swarm directories
\mkdir -p ${cdir_log}
\mkdir -p ${dir_swarm}

# remove old swarm file if exists
if ( -e ${scr_swarm} ) then
    \rm ${scr_swarm}
endif

# --------------------------------------------------------------------------
# build swarm execution script

set log = ${cdir_log}/log_${cmd}.txt

echo "tcsh -xf ${scr_cmd}                \\"    >> ${scr_swarm}
echo "     |& tee ${log}"                       >> ${scr_swarm}

# --------------------------------------------------------------------------
# run the swarm file

cd ${dir_scr}

echo "++ And start running: ${scr_swarm}"


# load conda setup for Biowulf if needed
source /data/NIMH_SSCC/ptaylor/miniconda3/etc/profile.d/conda.csh >& /dev/null

tcsh ${scr_swarm}
