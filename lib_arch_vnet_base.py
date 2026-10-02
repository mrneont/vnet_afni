#!/usr/bin/env python

# A library of base functionality for: archimedes_vnet.py
# ============================================================================

import os, sys, copy

# ============================================================================

class ArchInputTree:
    """Object for the data tree for training and validation.  This defines
and checks for expected directories and datasets.  

If has_wtds is True, then we search for a directory containing weight
datasets for each input dset.

The basic DATA_DIR directory structure looks like this:

    dir_root/
    `-- mask/
    `-- orig/

and if has_wtds is True, then it looks like this:

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
has_wtds : bool
    should the tree be checked for a directory of weights?
verb : int
    verbosity level

    """

    def __init__(self, dir_root, has_wtds=False, verb=1):

        # ----- set up attributes

        # main input variables
        self.status        = 0                           # not used
        self.verb          = verb
        self.has_wtds      = has_wtds

        # dirs
        self.dir_root      = dir_root 

        # ----- take action(s)

        tmp = self.check_tree()
        if tmp : return

    # ----- methods

    def check_tree(self):
        """****"""

        BAD_RETURN = -1

        return 0


    # ----- decorators

    @property
    def some_decorator(self):
        """****"""

        return 'something'


# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")
