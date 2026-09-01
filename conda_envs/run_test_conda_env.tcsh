#!/bin/tcsh




set dir_parent = $1


set dir_output        = "test_pred_mask"
set dir_logs          = "logs"
set scr_swarm         = "swarm_test_conda_env.txt"

mkdir -p ${dir_parent}/${dir_output}
mkdir -p ${dir_parent}/${dir_logs}


if ( -e ${scr_swarm} ) then
    \rm ${scr_swarm}
endif

echo ${dir_parent}/${dir_output}
rm -rf ${dir_parent}/${dir_output}/*

echo ${dir_parent}/${dir_logs}
rm -rf ${dir_parent}/${dir_logs}/*

#for loop to write the shell scripts
#given the list of conda envs 

#conda_env  = "test_torch_v1_13_0  test_torch_v2_0_0 "

set conda_env = (test_torch_v1_13_0 test_torch_v2_0_0)



#other parameters : 
# test i/p_data dir  = $2
# test o/p_data dir  = $3
# checkpoint_path    = $4
#map_location        = $5





echo ${PWD}
foreach i  (${conda_env})
    echo $i
    #tcsh do_conda_env.tcsh $i
    set logtxt = `python -c "print('log_'+ '${i}'+'.txt')"`
    echo "tcsh do_conda_env.tcsh ${i}  |& tee  ${dir_logs}/${logtxt}" >> ${dir_parent}/${scr_swarm}
end






# -------------------------------------------------------------------------
# run swarm command

echo "dirparent"
echo ${dir_parent}
echo ${PWD}

set cmd = "conda_env"
echo "++ And start swarming: ${scr_swarm}"

swarm                                                              \
    -f ${dir_parent}/${scr_swarm}                                                \
    --partition=norm,quick                                         \
    --threads-per-process=1                                       \
    --gb-per-process=10                                            \
    --time=0:30:00                                                \
    --logdir=${dir_logs}                                           \
    --job-name=${cmd}                                          \
    --merge-output                                                 \
    --usecsh




