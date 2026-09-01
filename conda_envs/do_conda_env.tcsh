#!/bin/tcsh

set env = $1


set output_folder = `python -c "print('${env}'+'_out')"`
set dir_output = "test_pred_mask"
echo ${dir_output}
conda activate ${env}
echo python -ver
python -V

mkdir -p ${dir_output}/${output_folder}

python /data/NIMH_SSCC/narayanaswamyy2/vnet_afni_9May25/lib_ml_test.py \
    -d /data/NIMH_SSCC/narayanaswamyy2/data_setup/test_one_dataset/validation \
    -o  ${dir_output}/${output_folder} \
    -ch '/data/NIMH_SSCC/narayanaswamyy2/CNNouts/checkpoint' \
    -m 'cpu'