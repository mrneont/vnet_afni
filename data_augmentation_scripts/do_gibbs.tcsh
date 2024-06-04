#!/bin/tcsh

# script to introduce the gibbs/ringing artifact into the datasets. 

# set Data augmentation directory and filename: daug
set daug  =  $1
set range =  $2

# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

cd ${daug_dir} # access the copies of dataset 

# seperate  out the basename of the dataset 
set dset_in = `basename ${daug}`
echo ${dset_in}


set dset_res  = `python -c "print('${dset_in}'.split('.')[0] +'_'+'00_rai'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`


set dset_zp  = `python -c "print('${dset_in}'.split('.')[0] +'_'+'01_zp'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_fft = `python -c "print('${dset_in}'.split('.')[0] +'_'+'02_complex'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_bpa = `python -c "print('${dset_in}'.split('.')[0] +'_'+'03_bp_abs'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_bpp = `python -c "print('${dset_in}'.split('.')[0] +'_'+'03_bp_pha'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_bpc = `python -c "print('${dset_in}'.split('.')[0] +'_'+'04_bp_comp'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_ifft = `python -c "print('${dset_in}'.split('.')[0] +'_'+'05_ifft'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_izp = `python -c "print('${dset_in}'.split('.')[0] +'_'+'06_izp'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`

set dset_ires = `python -c "print('${dset_in}'.split('.')[0] +'_'+'07_ires'+\
                                    '.'+ 'nii' +\
                                    '.'+ 'gz' )"`



# -----------------------------------------------------------------------------

# get input orientation and datum type
set ori_in = `3dinfo -orient ${dset_in}`
set dat_in = `3dinfo -datum ${dset_in}`

# ----------------------------------------------------------------------------
#IKJ #
# make RAI orient, so IJK maps onto XYZ easily
3dresample                                                \
    -overwrite                                            \
    -orient RAI                                           \
    -prefix ${dset_res}                                   \
    -input  ${dset_in}

# convenient way to know size, and to have powers of 2 in all dir
3dZeropad                                                 \
    -overwrite                                            \
    -RL 256 -AP 256 -IS 256                               \
    -prefix ${dset_zp}                                    \
    ${dset_res}

# do the FFT, with "shift" on so baseline is in middle of image
3dFFT                                                     \
    -overwrite                                            \
    -complex                                              \
    -altIN                                                \
    -prefix   ${dset_fft}                                 \
    -input    ${dset_zp}

# bounds on each side
# range 70 to 100 
set dx = ${range}
set dy = ${range}
set dz = 0
@   xd = 256 - ${dx} - 1
@   yd = 256 - ${dy} - 1
@   zd = 256 - ${dz} - 1

# 3dcalc cannot output complex dataset; need to do Mag/Ph or Re/Im separately
3dcalc \
    -overwrite \
    -cx2r ABS  \
    -a ${dset_fft} \
    -expr "a" \
    -prefix ${dset_bpa}                                             \
    -float                                                          \
    -nscale
3dcalc \
    -overwrite \
    -cx2r PHASE  \
    -a ${dset_fft} \
    -expr "a*(1+0.75*not(within(i,$dx,$xd)*within(j,$dy,$yd)*within(k,$dz,$zd)))" \
    #-expr "a"                   \
    -prefix ${dset_bpp}                                             \
    -float                                                          \
    -nscale

# recombined bandpassed Mag/Ph
3dTwotoComplex                                            \
    -overwrite                                            \
    -MP                                                   \
    -prefix  ${dset_bpc}                                   \
    ${dset_bpa}                                            \
    ${dset_bpp}

# invert the FFT on the bandpassed dataset
3dFFT                                                     \
    -overwrite                                            \
    -inverse                                              \
    -prefix   ${dset_ifft}                                 \
    -input    ${dset_bpc}

# change back to original grid dims                       \
3dZeropad                                                 \
    -overwrite                                            \
    -master  ${dset_res}                                  \
    -prefix  ${dset_izp}                                  \
    ${dset_ifft} 

# change back to original orientation
3dresample                                                \
    -overwrite                                             \
    -orient ${ori_in}                                      \
    -input  ${dset_izp}                                   \
    -prefix  ${dset_ires}

# no negative values, ensure short output type
3dcalc \
    -overwrite              \
    -a ${dset_ires} \
    -expr 'min(abs(a),255)'       \
    -prefix ${dset_in}     \
    -datum ${dat_in}        \
    -nscale


# remove the intermediate files
rm  ${dset_res}
rm  ${dset_zp}
rm  ${dset_fft}
rm  ${dset_bpa}

rm  ${dset_bpp}
rm  ${dset_bpc}
rm  ${dset_ifft}
rm  ${dset_izp}
rm  ${dset_ires}


    
