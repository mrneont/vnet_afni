#!/bin/tcsh


# FFT to complex
3dFFT -overwrite   -altIN  -complex  -prefix sub_001_orig_FFT.nii.gz sub_001_orig.nii.gz


# back to 
#3dFFT -overwrite   -inverse -prefix  pac_125_FFT_DONE.nii.gz   pac_125_orig_128sz_FFT.nii.gz


3dcalc -a sub_001_orig.nii.gz                                              \
	    -expr 'step(300-(i-128)*(i-128)-(j-128)*(j-128)-(k-128)*(k-128))' \
	    -prefix ball.nii.gz \
	    -overwrite


3dcalc -overwrite \
  -a sub_001_orig_FFT.nii.gz \
  -b ball.nii.gz \
        -expr 'a*b'  \
        -prefix sub_001_orig_FFT_roi.nii.gz




# back to 
3dFFT -overwrite  -altIN -inverse -prefix  sub_001_orig_FFT_gibbs.nii.gz   sub_001_orig_FFT_roi.nii.gz




3dcalc -overwrite \
  -a sub_001_orig.nii.gz \
  -b sub_001_orig_FFT_gibbs.nii.gz \
        -expr 'a-b'  \
        -prefix sub_256_res.nii.gz

