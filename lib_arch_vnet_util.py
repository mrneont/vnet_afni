#!/usr/bin/env python

# A library of various utilities for: archimedes_vnet.py
# ============================================================================

import torch
import numpy as np

from     afnipy import afni_base as ab

from  vnet_afni import lib_arch_vnet_defs as DEF

from communifti import lib_nibabel_read_nifti  as lnrn

# ============================================================================

def load_orig_dset(inset, do_perc_thr=True, scale_mode='z_scoring', 
                   set_dtype=np.float32, verb=1):
    """Read in inset (NIFTI anatomical dset) using nibabel, and do
things like percentile-based thresholding (if do_perc_thr=True) and
scaling (e.g., Z-scoring or min/max scaling) of it.  The header is
also stored separately, to help with writing out data.

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
scale_mode : str
    do simple processing to the data, in the form of normalizing the data
    in some way; allowed values are 'z_scoring' and 'min_max_scale'
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

    if scale_mode is not None and not(scale_mode in DEF.LIST_scale_mode) :
        msg = "Unknown scale_mode: {}\n".format(scale_mode)
        msg+= "Please select one from among:\n"
        msg+= "{}".format(DEF.STR_scale_mode)
        ab.EP1(msg)
        return BAD_RETURN


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
    if scale_mode is None :
        pass

    elif scale_mode == 'z_scoring' :
        is_fail, orig_data = z_scoring(orig_data)
        if is_fail :
            ab.EP1("Could not z-score NIFTI: {}".format(inset))
            return BAD_RETURN

    elif scale_mode == 'min_max_scale' :
        is_fail, orig_data = scale_min_max(orig_data)
        if is_fail :
            ab.EP1("Could not min/max scale NIFTI: {}".format(inset))
            return BAD_RETURN

    else:
        ab.EP1("Unknown scale_mode: {}".format(scale_mode))
        return BAD_RETURN


    # numpy array -> torch tensor, with extra dim added at the start
    data_orig = torch.from_numpy(orig_data).unsqueeze(0)

    return 0, data_orig, hdr_orig

# ----------------------------------------------------------------------------
# scale modes, simple functions

def z_scoring(img):

    BAD_RETURN = (-1, np.ndarray(0))

    data_mean = img.mean()
    data_std  = img.std()

    if np.isfinite(data_std) and data_std :
        Z_normalized = (img - data_mean) / data_std
    else:
        ab.EP1("std dev of img dset was zero?")
        return BAD_RETURN

    return 0, Z_normalized

def scale_min_max(img):

    BAD_RETURN = (-1, np.ndarray(0))

    dmin = img.min()
    dmax = img.max()

    if dmax == dmin :
        ab.EP1("Cannot min-max scale constant dset")
        return BAD_RETURN

    MM_normalized = (img - dmin) / (dmax - dmin)

    return 0, MM_normalized

# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")
