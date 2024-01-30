#!/bin/tcsh

# script to introduce the gain inhomogentity into the datasets. 

# DA_dir      : is the data Data augmentation directory 
# dataset_dir : is the dataset directory

#window over which the scaling factor varies 
# the scaling factor varies between values 'win_min' and 'window' 
set window  = 1.2   # within [0,2]
set win_min = `ccalc 1-${window}/2` 


# set Data augmentation directory : DA_dir
set DA_dir       =  $1
set dataset_dir  =  $2
set fl_name = 'FILE.txt'

# list of orig dataset into which the gain inhomogenity will be applied
set v=`cat  ${DA_dir}/${fl_name}`
#echo ${v}
set i=26

set orig_folder = "training/orig"
set mask_folder = "training/mask"
set trans_folder = "gain"
echo " orig_folder = ${orig_folder}"

cd  ${dataset_dir}/${orig_folder}
echo " mask_folder = ${mask_folder} "

set axis =(i j k) # the three different axis 

while ( $i < = 40 )
    set dset = $v[$i]
    

	echo ${dset}
	
	# dimension of the dset
	set nmat      = `3dinfo -ni ${dataset_dir}/${orig_folder}/${dset}`
	set axis_ran = (`ccalc -i '1+iran(2)'`)
	echo "axis_ran = ${axis_ran}"
	foreach x (${axis_ran}) # counter for the array 'axis'
    	
    	echo ${axis[${x}]}
		# prefix denotes the type of artifact inrtoduced into the dset
    	set prefix  = `python -c "print('g'+ '${axis[${x}]}')"`
    	echo " prefix = ${prefix} "
    	# dset_trans is the transformed dset
    	set dset_trans  =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

		echo " dset = ${dset} "
        set mask = ${dset:gas/orig/mask/}
        echo " mask = ${mask} "

        set mask_trans = "${dset_trans:gas/orig/mask/}"
        echo " mask_trans = ${mask_trans} "

		3dcalc                                         \
    		-overwrite                                 \
    		-a ${dataset_dir}/${orig_folder}/${dset}   \
    		-expr "a * (${win_min} + ${window}* ${axis[${x}]}/${nmat})" \
    		-prefix   ${DA_dir}/${trans_folder}/${dset_trans}
	
		cp  ${dataset_dir}/${mask_folder}/${mask} ${DA_dir}/${trans_folder}/${mask_trans}

    end
    @ i = $i + 1
end
