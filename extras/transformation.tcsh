#!/bin/tcsh

# This script is to create transformed datasets from the original datasets
# based on the transformation specified in the 1D parameter files. 


set here        = ${PWD}

echo "${here}"
### absolute path to 1D params file dir
set param_level = ${here}/param_1D


# subdirs (some exist, some to be made)

cd ${param_level}

set dir_trans  = ${PWD}/transformed
mkdir ${dir_trans}

set dset_trans_dir      = ${dir_trans}/orig
set dset_mask_trans_dir = ${dir_trans}/mask
mkdir ${dset_trans_dir}
mkdir ${dset_mask_trans_dir}

set dset      = pac_125_orig.nii.gz 

set dset_mask = pac_125_mask.nii.gz


set fl1D = ( *.1D)
# loop over all the 1D param files in the folder 
foreach param_fl  (${fl1D})
	

	# print the 1D param file which contains the 12 parameters
	# required for transformation
	echo " ${param_fl}"

	

	set prefix      = `python -c  "print('${param_fl}'.split('.')[0])"`

	

	set dset_trans  =  `python -c "print('${dset}'.split('_')[0]+'_'+  \
					    '${prefix}' + '_' + '${dset}'.split('_')[1]+ \
					     '_'+ '${dset}'.split('_')[2] )"`

	set dset_mask_trans  =  `python -c "print('${dset_mask}'.split('_')[0]+'_'+ \
					    '${prefix}' + '_' + '${dset_mask}'.split('_')[1]+ \
					     '_'+ '${dset_mask}'.split('_')[2] )"`
	

	3dAllineate -overwrite -input ${dset}       \
                -master ${dset}   \
                -prefix ${dset_trans_dir}/${dset_trans}  \
                -final wsinc5  \
                -1Dparam_apply ${param_fl}


   	3dAllineate -overwrite -input ${dset_mask}       \
                -master ${dset_mask}       \
                -prefix ${dset_mask_trans_dir}/${dset_mask_trans}  \
                -final NN          \
                -1Dparam_apply ${param_fl}





end 