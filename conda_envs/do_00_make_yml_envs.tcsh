#!/bin/tcsh

# a starter script to create a series of YML files and associated
# envs, with different pytorch versions

# =====================================================================

# list of all Pytorch versions
set all_ver = ( 1.3 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 )

# loop over and create new yml files
foreach ver ( $all_ver )

    # root of output name and YML file
    set oname = test_env_${ver}
    set ofile = ${oname}.yml

    # create YML file; be sure not to indent closing 'EOF'
    # (Q: should we include cuda-toolkit for some linux?)
cat <<EOF >> ${ofile}
name: ${oname}         # conda env create -f ${ofile}
dependencies:
  - python
  - pytorch=="${ver}"   
  - matplotlib
  - numpy 
  - jupyter                   # useful python interface
  - ipython
  - nibabel

EOF

    # create conda env, and log the process
    conda env create -f ${ofile} |& tee log_${ver}.txt
end
