#!/usr/bin/env python

# A library of functions for populating VNET weight-dataset directories.
# ============================================================================

import os, glob

from afnipy import afni_base as ab

# ============================================================================

LIST_split = [
    'training',
    'validation',
]

PROG_mask_to_wtds = 'adjunct_mask_to_wtds.tcsh'

# ============================================================================

class MainObj:
    """Object for creating VNET wtds datasets from mask datasets."""

    def __init__(self, user_inobj=None):

        # ----- set up attributes

        self.status        = 0
        self.user_inobj    = user_inobj

        self.verb          = 1
        self.overwrite     = False
        self.indir         = None

        # ----- take action(s)

        if user_inobj:
            self.status = self.load_from_inopts()
            if self.status: return

            self.status = self.basic_setup()
            if self.status: return

            self.status = self.run_all_splits()
            if self.status: return

    # ----- methods

    def load_from_inopts(self):
        """Populate values from the command line interface object."""

        if not(self.user_inobj):
            ab.WP("No user_inobj? Nothing to do.")
            return 0

        io = self.user_inobj

        self.verb      = io.verb
        self.overwrite = io.overwrite
        self.indir     = io.indir

        return 0

    def basic_setup(self):
        """Check the input VNET tree and expected mask directories."""

        BAD_RETURN = -1

        if not(self.indir):
            ab.EP1("Need to provide an indir")
            return BAD_RETURN

        if not(os.path.isdir(self.indir)):
            ab.EP1("Cannot find indir: {}".format(self.indir))
            return BAD_RETURN

        for split in LIST_split:
            dir_split = os.path.join(self.indir, split)
            dir_mask  = os.path.join(dir_split, 'mask')
            dir_wtds  = os.path.join(dir_split, 'wtds')

            if not(os.path.isdir(dir_split)):
                ab.EP1("Cannot find split dir: {}".format(dir_split))
                return BAD_RETURN

            if not(os.path.isdir(dir_mask)):
                ab.EP1("Cannot find mask dir: {}".format(dir_mask))
                return BAD_RETURN

            if os.path.isdir(dir_wtds) and not(self.overwrite):
                msg = "The wtds dir exists already: {}".format(dir_wtds)
                msg+= "\nUse -overwrite to replace its associated outputs"
                ab.EP1(msg)
                return BAD_RETURN

        return 0

    def run_all_splits(self):
        """Create wtds datasets for training and validation splits."""

        BAD_RETURN = -1

        for split in LIST_split:
            is_fail = self.run_one_split(split)
            if is_fail:
                return BAD_RETURN

        return 0

    def run_one_split(self, split):
        """Create all wtds datasets for one VNET split."""

        BAD_RETURN = -1

        dir_split = os.path.join(self.indir, split)
        dir_mask  = os.path.join(dir_split, 'mask')
        dir_wtds  = os.path.join(dir_split, 'wtds')

        if self.verb:
            ab.IP("Process masks: {}".format(split))

        try:
            os.makedirs(dir_wtds, exist_ok=self.overwrite)
        except OSError:
            ab.EP1("Could not create wtds dir: {}".format(dir_wtds))
            return BAD_RETURN

        mask_list = glob.glob(os.path.join(dir_mask, '*_mask.nii*'))
        mask_list.sort()

        if not len(mask_list):
            ab.EP1("No mask datasets found in: {}".format(dir_mask))
            return BAD_RETURN

        for ii, fname_mask in enumerate(mask_list):

            fname_base = os.path.basename(fname_mask)
            if fname_base.endswith('_mask.nii.gz'):
                fname_wtds = fname_base[:-12] + '_wtds.nii.gz'
            elif fname_base.endswith('_mask.nii'):
                fname_wtds = fname_base[:-9] + '_wtds.nii.gz'
            else:
                ab.EP1("Unexpected mask filename: {}".format(fname_mask))
                return BAD_RETURN

            path_wtds = os.path.join(dir_wtds, fname_wtds)

            if self.verb:
                ab.IP("Dset {:04d} / {:04d}: {}".format(
                    ii+1, len(mask_list), fname_base))

            cmd = '{} "{}" "{}"'.format(
                PROG_mask_to_wtds, fname_mask, path_wtds
            )

            com = ab.shell_com(cmd, capture=1)
            stat = com.run()
            if stat:
                ab.EP1("Failed to make wtds from mask: {}".format(fname_mask))
                return BAD_RETURN

        return 0


# ============================================================================

if __name__ == "__main__":

    print("++ No example")
