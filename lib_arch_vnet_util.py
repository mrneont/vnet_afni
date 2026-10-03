#!/usr/bin/env python

# A library of various utilities for: archimedes_vnet.py
# ============================================================================

import torch
import numpy as np

from afnipy import afni_base as ab

from communifti import lib_nibabel_read_nifti  as lnrn

# ============================================================================

def load_orig_dset(inset, do_perc_thr=True, do_zscore=True, 
                   set_dtype=np.float32, verb=1):
    """Read in inset (NIFTI anatomical dset) using nibabel, and do things
like percentile-based thresholding (if do_perc_thr=True) and Z-scoring
(if do_zscore=True) of it.  The header is also stored separately, to
help with writing out data.

The 3D dset is stored as a torch tensor array.  In order to be used as
an input to the model, it is also unsqueezed to insert an extra dim in
the [0]th index, so 3D data of dim [A, B, C] -> [1, A, B, C].

Parameters
----------
inset : str
    the name of the orig (3D, anatomical) dset being input
do_perc_thr : bool
    do simple processing to the data, in the form of percentile-based 
    thresholding
do_zscore : bool
    do simple processing to the data, in the form of Z-scoring the matrix 
    values
set_dtype : dtype
    a parameter passed along to read_nifti_to_nibabel(), probably just leave 
    at this for now
verb : int
    verbosity level

Returns
-------
is_fail : int
    0 for success, nonzero for failure
data_orig : torch tensor
    the (4D) torch tensor being output
hdr_orig : nibabel header
    the header of the inset, to pass along for later processing

    """

    BAD_RETURN = (-1, None, None)

    # read NIFTI to tmp data array (needs proc) and header obj
    is_fail, orig_data, hdr_orig = \
        lnrn.read_nifti_to_nibabel(inset, set_dtype=set_dtype, verb=verb)
    if is_fail :
        ab.EP1("Could not read in NIFTI: {}".format(inset))
        return BAD_RETURN

    # simple proc 1: percentile-based thresholding of data
    if do_perc_thr :
        top99_thresh = np.percentile(orig_data, 99)
        orig_data[orig_data > top99_thresh] = top99_thresh
        down2_thresh = np.percentile(orig_data, 2)
        orig_data[orig_data < down2_thresh] = down2_thresh

    # simple proc 2: z-score conversion
    if do_zscore :
        is_fail, orig_data = z_scoring(orig_data)
        if is_fail :
            ab.EP1("Could not read z-score NIFTI: {}".format(inset))
            return BAD_RETURN

    # numpy array -> torch tensor, with extra dim added at the start
    data_orig = torch.from_numpy(orig_data).unsqueeze(0)

    return 0, data_orig, hdr_orig

# ----------------------------------------------------------------------------

def z_scoring(img):

    BAD_RETURN = (-1, np.ndarray(0))

    data_mean = img.mean()
    data_std  = img.std()

    if data_std :
        Z_normalized = (img - data_mean) / data_std
    else:
        ab.EP1("std dev of img dset was zero?")
        return BAD_RETURN

    return 0, Z_normalized

# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")
