#!/usr/bin/env python

# A library of base functionality for: archimedes_vnet.py
# ============================================================================

import os, glob

from afnipy import afni_base as ab

# ============================================================================

class ArchRootTree:
    """Object for managing the input tree for running both training and
validation of the VNET. This would likely be the indir for
archimedes_vnet.py, and would also likely contain two directories of
split data (for training and validation).  This object defines and checks for
expected directories and datasets.

Parameters
----------
dir_root : str
    the root directory of the data tree (contains data split directories)
has_mask : bool
    should the split data be checked for a directory of mask datasets ('mask')?
has_wtds : bool
    should the split data be checked for a directory of weight datasets ('wtds')?
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
        self.all_split               = {}
        self.all_split['training']   = None
        self.all_split['validation'] = None

        # ----- take action(s)

        tmp = self.basic_setup()
        if tmp : return

        tmp = self.check_tree_dirs()
        if tmp : return

        tmp = self.load_split_trees()
        if tmp : return

    # ----- methods

    def basic_setup(self):
        """Simple string formatting, like: make sure not '/' at end"""

        # need to know present working dir
        self.pwd = os.getcwd()

        # don't want backslashes, for aesthetics
        self.dir_root.rstrip('/')

        return 0

    def check_tree_dirs(self):
        """Go through a series of directory checks about what should exist in
        the tree"""

        if self.verb :
            ab.IP("Check root-level tree dirs exist")

        BAD_RETURN = -1

        if not(os.path.isdir(self.dir_root)) :
            msg = "Tree check, no dir_root: "
            msg+= "{}".format(self.dir_split)
            ab.EP1(msg)
            return BAD_RETURN

        for split in self.all_split :
            ddd = self.get_splitdir_path(split)
            if not(os.path.isdir(ddd)) :
                msg = "Tree check, no dir_{}: {}".format(split, ddd)
                ab.EP1(msg)
                return BAD_RETURN

        return 0

    def load_split_trees(self):
        """Load in the split trees, and make sure they valid"""

        if self.verb :
            ab.IP("Load in split trees with data")

        BAD_RETURN = -1

        for split in self.all_split :
            self.all_split[split] = \
                ArchSplitTree(
                    self.get_splitdir_path(split),
                    has_mask = self.has_mask, 
                    has_wtds = self.has_wtds,
                    verb     = self.verb
                )
            if isinstance(self.all_split[split], int) :
                msg = "Failed to load split tree for: {}".format(split)
                ab.EP1(msg)
                return BAD_RETURN

        return 0

    def get_splitdir_path(self, split):
        """Simply append dir-of-interest, as defined by the str split, to
        dir_root (no existence checked)"""

        return self.dir_root + '/' + split

    def count_splitdir_files(self, split):
        """How many files exist in a subdir (as defined by the str
        'label')?"""

        return len(self.all_split[split])


# ============================================================================

class ArchSplitTree:
    """Object for managing the tree of data splits, like training and
validation.  This defines and checks for expected directories and
datasets.

If has_wtds is True, then we search for a directory containing weight
datasets for each input dset.

The basic data split directory structure looks like this (but likely
with more sub*.nii.gz datasets):

    dir_split/
    `-- orig/
        `-- sub-000_orig.nii.gz
        `-- sub-001_orig.nii.gz

If has_mask is True (which is the case for training and validation),
then it would look like this:

    dir_split/
    `-- mask/
        `-- sub-000_mask.nii.gz
        `-- sub-001_mask.nii.gz
    `-- orig/
        `-- sub-000_orig.nii.gz
        `-- sub-001_orig.nii.gz

... and if also has_wtds is True, then it looks like this:

    dir_split/
    `-- mask/
        `-- sub-000_mask.nii.gz
        `-- sub-001_mask.nii.gz
    `-- orig/
        `-- sub-000_orig.nii.gz
        `-- sub-001_orig.nii.gz
    `-- wtds/
        `-- sub-000_wtds.nii.gz
        `-- sub-001_wtds.nii.gz

In each of dir_split's subdirectories, there must be correspondingly
named files, where the only part of the filename that differs is the
subdir name.  That is: 

    sub-001_mask.nii.gz, sub-001_orig.nii.gz, sub-001_wtds.nii.gz

There must be a corresponding dataset across each of the
subdirectories that is being used.

Parameters
----------
dir_split : str
    the split directory of the data tree
has_mask : bool
    should the tree be checked for a directory of mask datasets ('mask')?
has_wtds : bool
    should the tree be checked for a directory of weight datasets ('wtds')?
verb : int
    verbosity level

    """

    def __init__(self, dir_split, has_mask=True, has_wtds=False, verb=1):

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
        self.dir_split     = dir_split

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
        self.dir_split.rstrip('/')

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
            ab.IP("Check split-level tree dirs exist")

        BAD_RETURN = -1

        ddd = self.dir_split
        if not(os.path.isdir(ddd)) :
            msg = "Tree check, no dir_split: {}".format(ddd)
            ab.EP1(msg)
            return BAD_RETURN

        ddd = self.get_subdir_path('mask')
        if self.has_mask and not(os.path.isdir(ddd)) :
            msg = "Tree check, no dir_mask: {}".format(ddd)
            ab.EP1(msg)
            return BAD_RETURN

        ddd = self.get_subdir_path('orig')
        if not(os.path.isdir(ddd)) :
            msg = "Tree check, no dir_orig: {}".format(ddd)
            ab.EP1(msg)
            return BAD_RETURN

        ddd = self.get_subdir_path('wtds')
        if self.has_wtds and not(os.path.isdir(ddd)) :
            msg = "Tree check, no dir_wtds: {}".format(ddd)
            ab.EP1(msg)
            return BAD_RETURN

        return 0

    def check_tree_files(self):
        """Go through a series of directory checks about what should exist in
        the tree"""

        if self.verb :
            ab.IP("Check split-level tree files for names and consistency")

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
        dir_split (no existence checked)"""

        return self.dir_split + '/' + label

    def count_subdir_files(self, label):
        """How many files exist in a subdir (as defined by the str
        'label')?"""

        return len(self.all_dset[label])


    # ----- decorators

    #@property

# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")
