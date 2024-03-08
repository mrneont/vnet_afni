#!/bin/tcsh

# script to introduce the gibbs/ringing artifact into the datasets. 

# DA_dir      : is the data Data augmentation directory 
# dataset_dir : is the dataset directory

# set Data augmentation directory : DA_dir
set DA_dir       =  $1
set dataset_dir  =  $2
set fl_name      = 'FILE.txt'
# list of orig dataset into which the gain inhomogenity will be applied
set v=`cat  ${DA_dir}/${fl_name}`
#echo ${v}
set i=1

set orig_folder  = "training/orig"
set mask_folder  = "training/mask"
set trans_folder = "gibbs" 
echo " orig_folder = ${orig_folder}"

#cd  ${orig_folder}
echo " mask_folder = ${mask_folder} "

#echo "HELLO"
#echo ${dset_list}
set rad = 100


while ( $i < = 10 )
    set dset = $v[$i]
    

    echo ${dset}

	set dset_FFT  =  `python -c "print('${dset}'.split('.')[0] +'_'+'FFT'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

    set dset_ZP =  `python -c "print('${dset}'.split('.')[0] +'_'+'FFT'+\
    								'_'+'ZP'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

    set dset_roi =  `python -c "print('${dset}'.split('.')[0] +'_'+'FFT'+\
    								'_'+'ZP'+\
    								'_'+'roi'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`
    
    set dset_trans = `python -c "print('${dset}'.split('.')[0] +'_'+'FFT'+\
    								'_'+'gibbs'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`
    echo ${dset_FFT} 
    echo ${dset_ZP}                                 
    echo ${dset_roi}   
    echo ${dset_trans}   


	# FFT to complex
	3dFFT -overwrite   -altIN  -complex  -prefix ${DA_dir}/${trans_folder}/${dset_FFT} \
           ${dataset_dir}/${orig_folder}/${dset}

    3dZeropad -overwrite -prefix ${DA_dir}/${trans_folder}/${dset_ZP} \
              -RL $rad -AP $rad  ${DA_dir}/${trans_folder}/${dset_FFT}

    3dZeropad  -overwrite -prefix ${DA_dir}/${trans_folder}/${dset_roi} \
               -master ${dataset_dir}/${orig_folder}/${dset}  ${DA_dir}/${trans_folder}/${dset_ZP}

    3dFFT -overwrite  -inverse -prefix  ${DA_dir}/${trans_folder}/${dset_trans} \
          -abs ${DA_dir}/${trans_folder}/${dset_roi}

    rm ${DA_dir}/${trans_folder}/${dset_ZP}
    rm ${DA_dir}/${trans_folder}/${dset_FFT}
    rm ${DA_dir}/${trans_folder}/${dset_roi}

    echo " dset = ${dset} "
    set mask = ${dset:gas/orig/mask/}
    echo " mask = ${mask} "

    set mask_trans = "${dset_trans:gas/orig/mask/}"
    echo " mask_trans = ${mask_trans} "
    
    cp  ${dataset_dir}/${mask_folder}/${mask} ${DA_dir}/${trans_folder}/${mask_trans}


    @ i = $i + 1



end