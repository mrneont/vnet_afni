#!/bin/tcsh

# This script is to create transformed datasets from the original datasets
# based on the transformation specified in the 1D parameter files. 


set here        = ${PWD}

echo "${here}"
### absolute path to 1D params file dir
set param_level = ${here}/param_1D


# subdirs (some exist, some to be made)

set orig_folder = '/Users/narayanaswamyy2/RR_AFNI_VNET/mridataset/data_augment_master/FQC_data_FS_208/training/orig'
set mask_folder = '/Users/narayanaswamyy2/RR_AFNI_VNET/mridataset/data_augment_master/FQC_data_FS_208/training/mask'
set trans_folder = '/Users/narayanaswamyy2/RR_AFNI_VNET/mridataset/data_augment_master/transformation'
set param_folder = '/Users/narayanaswamyy2/RR_AFNI_VNET/mridataset/data_augment_master/transformation/param_1D_files'
cd  ${orig_folder}

set dset      = sub-705_orig.nii.gz 

set dset_mask = sub-705_mask.nii.gz

set param_fl  = shift_minus30y.1D

	
# print the 1D param file which contains the 12 parameters
# required for transformation
echo " ${param_fl}"

	

set prefix      = `python -c  "print('${param_fl}'.split('.')[0])"`

echo "prefix= ${prefix}"

set dset_trans  =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

set dset_mask_trans  =  `python -c "print('${dset_mask}'.split('.')[0]+'_'+ \
					    '${prefix}' + '.' + '${dset_mask}'.split('.')[-2]+ \
					     '.'+ '${dset_mask}'.split('.')[-1] )"`
	

3dAllineate -overwrite -input ${orig_folder}/${dset}       \
                -master ${orig_folder}/${dset}   \
                -prefix ${trans_folder}/${dset_trans}  \
                -final wsinc5  \
                -1Dparam_apply ${param_folder}/${param_fl}


3dAllineate -overwrite -input ${mask_folder}/${dset_mask}       \
                -master ${mask_folder}/${dset_mask}       \
                -prefix ${trans_folder}/${dset_mask_trans}  \
                -final NN          \
                -1Dparam_apply ${param_folder}/${param_fl}

