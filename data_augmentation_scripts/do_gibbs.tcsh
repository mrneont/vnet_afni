#!/bin/tcsh

# script to introduce the gibbs/ringing artifact into the datasets. 

# set Data augmentation directory and filename: daug
set daug  =  $1
set rad       =  $2 

# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

cd ${daug_dir} # access the copies of dataset 

# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
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
    

echo ${dset_FFT} 
echo ${dset_ZP}                                 
echo ${dset_roi}   
 

# FFT to complex
3dFFT -overwrite   -altIN  -complex  -prefix  ${dset_FFT}  \
            ${dset}

3dZeropad -overwrite -prefix  ${dset_ZP} \
              -RL $rad -AP $rad  ${dset_FFT}

3dZeropad  -overwrite -prefix ${dset_roi} \
               -master ${dset}  ${dset_ZP}

3dFFT -overwrite  -inverse -prefix  ${dset} \
          -abs  ${dset_roi}

# remove the intermediate files
rm  ${dset_ZP}
rm  ${dset_FFT}
rm  ${dset_roi}

# mask will not be altered 
