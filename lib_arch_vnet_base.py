#!/usr/bin/env python

# A library of base functionality for: archimedes_vnet.py
# ============================================================================

import os, glob

import torch
import numpy as np

from afnipy import afni_base as ab

from communifti import lib_nibabel_read_nifti  as lnrn

# ============================================================================

class ArchInputTree:
    """Object for the data tree for training and validation.  This defines
and checks for expected directories and datasets.  

If has_wtds is True, then we search for a directory containing weight
datasets for each input dset.

The basic DATA_DIR directory structure looks like this:

    dir_root/
    `-- orig/

If has_mask is True (which is the case for training and validation),
then it would look like this:

    dir_root/
    `-- mask/
    `-- orig/

... and if also has_wtds is True, then it looks like this:

    dir_root/
    `-- mask/
    `-- orig/
    `-- wtds/

In each of dir_root's subdirectories, there must be correspondingly
named files, where the only part of the filename that differs is the
subdir name.  That is: 

    sub-001_mask.nii.gz, sub-001_orig.nii.gz, sub-001_wtds.nii.gz

There must be a corresponding dataset across each of the
subdirectories that is being used.

Parameters
----------
dir_root : str
    the root directory of the data tree
has_mask : bool
    should the tree be checked for a directory of mask datasets ('mask')?
has_wtds : bool
    should the tree be checked for a directory of weight datasets ('wtds')?
verb : int
    verbosity level

    """

    def __init__(self, dir_root, has_mask=True, has_wtds=False, verb=1):

        # ----- set up attributes

        # main input variables
        self.status        = 0                           # not used
        self.verb          = verb

        self.has_mask      = has_mask
        self.has_wtds      = has_wtds
        self.all_label     = []

        # need to know present working dir, for hopping around
        self.pwd           = None

        # top dir (all subdirs of interest are decorators)
        self.dir_root      = dir_root

        # dset names: store in a dictionary, where each subdir is a key
        self.all_dset         = {}
        self.all_dset['mask'] = []
        self.all_dset['orig'] = []
        self.all_dset['wtds'] = []


        # ----- take action(s)

        tmp = self.basic_setup()
        if tmp : return

        tmp = self.check_tree_dirs()
        if tmp : return

        tmp = self.check_tree_files()
        if tmp : return

    # ----- methods

    def basic_setup(self):
        """Simple string formatting, like: make sure not '/' at end"""

        # need to know present working dir
        self.pwd = os.getcwd()

        # don't want backslashes, for aesthetics
        self.dir_root.rstrip('/')

        # make a list of subdirs to check
        self.all_label = ['orig']
        if self.has_mask :
            self.all_label.append('mask')
        if self.has_wtds :
            self.all_label.append('wtds')

        return 0

    def check_tree_dirs(self):
        """Go through a series of directory checks about what should exist in
        the tree"""

        if self.verb :
            ab.IP("Check tree dirs exist")

        BAD_RETURN = -1

        if not(os.path.isdir(self.dir_root)) :
            msg = "Tree check, no dir_root: "
            msg+= "{}".format(self.dir_root)
            ab.EP1(msg)
            return BAD_RETURN

        if self.has_mask and not(os.path.isdir(self.get_subdir_path('mask'))) :
            msg = "Tree check, no dir_mask: "
            msg+= "{}".format(self.dir_mask)
            ab.EP1(msg)
            return BAD_RETURN

        if not(os.path.isdir(self.get_subdir_path('orig'))) :
            msg = "Tree check, no dir_orig: "
            msg+= "{}".format(self.dir_orig)
            ab.EP1(msg)
            return BAD_RETURN

        if self.has_wtds and not(os.path.isdir(self.get_subdir_path('wtds'))) :
            msg = "Tree check, no dir_wtds: "
            msg+= "{}".format(self.dir_wtds)
            ab.EP1(msg)
            return BAD_RETURN

        return 0

    def check_tree_files(self):
        """Go through a series of directory checks about what should exist in
        the tree"""

        if self.verb :
            ab.IP("Check tree files for names and consistency")

        BAD_RETURN = -1

        # ... and then use the label for each to glob equivalently
        for label in self.all_label:
            os.chdir(self.get_subdir_path(label))

            self.all_dset[label] = glob.glob("*{}.nii*".format(label))
            # NB: list must be sorted, for verifying cross-dir partners below
            self.all_dset[label].sort()

            os.chdir(self.pwd)

        # verify that the label str appears in each filename only once
        # (from glob above, we already know it exists _at least_ once)
        nbad = 0
        for label in self.all_label:
            for dset in self.all_dset[label]:
                if dset.count(label) > 1 :
                    msg = "dset {} has too many instances of ".format(dset)
                    msg+= "{} to be a valid filename here".format(label)
                    ab.EP1(msg)
                    nbad+= 1
        if nbad :
            return BAD_RETURN

        # verify consistency of filenames across subdirs in a pairwise
        # fashion (when there are at least 2 subdirs)
        for ii in range(1, len(self.all_label)):
            label0 = self.all_label[ii-1]
            label1 = self.all_label[ii]
            
            # simplest check: two dirs have same num of files
            n0 = self.count_subdir_files(label0)
            n1 = self.count_subdir_files(label1) 
            if n0 != n1 :
                msg = "The {} and {} subdirs ".format(label0, label1)
                msg+= "have differing file counts: {} and {}.".format(n0, n1)
                msg+= "\nSo, we know they can't have matched files"
                ab.EP1(msg)
                return BAD_RETURN

            # can we find cross-subdir matches, i.e., filenames
            # differing only in label (like sub-123_mask.nii.gz and
            # sub-123_orig.nii.gz), for each dset? this check requires
            # sorting has been done previously
            for jj in range(n0):
                f0 = self.all_dset[label0][jj]
                f1 = self.all_dset[label1][jj]
                g0 = f0.replace(label0, label1) # change label only
                if g0 != f1 :
                    msg = "At least one file mismatch between these "
                    msg+= "subdirs: {} and {}\n".format(label0, label1)
                    msg+= "One of these doesn't appear to have a partner: "
                    msg+= "{} and {}.".format(g0, f1)
                    ab.EP1(msg)
                    return BAD_RETURN

        return 0

    def get_subdir_path(self, label):
        """Simply append dir-of-interest, as defined by the str 'label', to
        dir_root (no existence checked)"""

        return self.dir_root + '/' + label

    def count_subdir_files(self, label):
        """How many files exist in a subdir (as defined by the str
        'label')?"""

        return len(self.all_dset[label])


    # ----- decorators

    #@property

# ============================================================================

def load_orig_dset(inset, do_perc_thr=True, do_zscore=True, set_dtype=np.float32,
                   verb=1):
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
    do simple processing to the data, in the form of percentile-based thresholding
do_zscore : bool
    do simple processing to the data, in the form of Z-scoring the matrix values
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
