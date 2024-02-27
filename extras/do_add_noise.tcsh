#!/bin/tcsh

# script to introduce the noise into the datasets. 

# DA_dir      : is the data Data augmentation directory 
# dataset_dir : is the dataset directory


# set Data augmentation directory : DA_dir
set DA_dir       =  $1
set dataset_dir  =  $2
set fl_name = 'FILE.txt'
# list of orig dataset into which the gain inhomogenity will be applied
set v=`cat  ${DA_dir}/${fl_name}`

set i=11

set orig_folder = "training/orig"
set mask_folder = "training/mask"
set trans_folder = "shading"
echo " orig_folder = ${orig_folder}"

cd  ${dataset_dir}/${orig_folder}
echo " mask_folder = ${mask_folder} "

# noise level of 10% 20% 30%
set noise_lvl = (0.1 0.2 0.3)
set percent = (10 20 30) #SNR percentage 

while ( $i < = 25 )
    set dset = $v[$i]
    echo ${dset}

    # find the median  of the voxel value in the dset
    set vals   = `3dBrickStat -slow -automask -median ${dataset_dir}/${orig_folder}/${dset}`
    
    set med    = ${vals[2]}

    echo ${med}

    set axis_ran = (`ccalc -i '1+iran(2)'`)
    echo ${axis_ran}
    foreach x (${axis_ran}) # counter for the array 'axis'
        
        # prefix denotes the type of artifact inrtoduced into the dset

        set prefix  = `python -c "print('noise'+ '${percent[${x}]}')"`

        # dset_trans is the transformed dset
        

        set dset_trans  =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

        echo " dset_trans = ${dset_trans} "     

        echo " dset = ${dset} "
        set mask = ${dset:gas/orig/mask/}
        echo " mask = ${mask} "

        set mask_trans = "${dset_trans:gas/orig/mask/}"
        echo " mask_trans = ${mask_trans} "
        # add zero-centered Gaussian random noise (whose stdev = dset's
        # median), and then scale that to reach our stated SNR
        #echo  ${noise_lvl[${x}]}

        3dcalc                   \
        -overwrite               \
        -a ${dataset_dir}/${orig_folder}/${dset}               \
        -expr "a + gran(0,${med})* ${noise_lvl[${x}]}"          \
        -prefix ${DA_dir}/${trans_folder}/${dset_trans} \
        -datum float \
        -nscale 
        #copy the corresponding mask file
        cp  ${dataset_dir}/${mask_folder}/${mask} ${DA_dir}/${trans_folder}/${mask_trans}

    end

    @ i = $i + 1
end
