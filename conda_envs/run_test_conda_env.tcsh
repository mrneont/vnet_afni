#!/bin/tcsh




set dir_parent = $1

set dir_scripts       = "scripts"
set dir_output        = "test_pred_mask"
set dir_logs          = "logs"
set scr_swarm         = "swarm_test_conda_env.txt"

mkdir -p ${dir_parent}/${dir_output}
mkdir -p ${dir_parent}/${dir_logs}


if ( -e ${scr_swarm} ) then
    \rm ${scr_swarm}
endif

#for loop to write the shell scripts
#given the list of conda envs 

#conda_env  = [test_torch_v1_13_0  test_torch_v2_0_0 test_torch_v2_2_2]

#python lib_ml_test.py \
#    -d /data/NIMH_SSCC/narayanaswamyy2/data_setup/test_one_dataset/validation \
#    -o 'test_one_torch_v2_0_0' \
#    -ch '/data/NIMH_SSCC/narayanaswamyy2/CNNouts/checkpoint' \
#    -m 'cpu'

#other parameters : 
# test i/p_data dir  = $2
# test o/p_data dir  = $3
# checkpoint_path    = $4
#map_location        = $5

cd ${dir_parent}/${dir_scripts}

set all_test_scr = (*.tcsh)

#this can be modified for each conda env in the list 'conda_env'
foreach  env_scr ( ${all_test_scr} )

    echo ${env_scr}

    set prefix = `python -c "print('${env_scr}'.split('.')[0])"`
    set logtxt = `python -c "print('log_'+ '${prefix}'+'.txt')"`

    echo ${logtxt}
    echo "tcsh ${env_scr}  |& tee ../${dir_logs}/${logtxt}" >> ${dir_parent}/${scr_swarm}


end 


# -------------------------------------------------------------------------
# run swarm command
#cp ${dirparent}/do_make_combo.tcsh  ${dirparent}/${make_combo_images_dir}/${dir_pred_mask} 
#echo "dirparent"
#echo ${dir_parent}
echo ${PWD}

set cmd = "conda_env"
echo "++ And start swarming: ${scr_swarm}"

swarm                                                              \
    -f ${dir_parent}/${scr_swarm}                                                \
    --partition=norm,quick                                         \
    --threads-per-process=1                                       \
    --gb-per-process=10                                            \
    --time=0:30:00                                                \
    --logdir=../${dir_logs}                                           \
    --job-name=${cmd}                                          \
    --merge-output                                                 \
    --usecsh




