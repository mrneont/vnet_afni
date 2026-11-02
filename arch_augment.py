#!/usr/bin/env python

"""Prepare and optionally execute AFNI V-net data augmentation."""

import sys

from afnipy    import afni_base         as ab
from vnet_afni import lib_arch_aug_opts as LAO

# ==========================================================================

def main(argv=None):
    """Read and check inputs, then create the augmentation processing object.

    Return an exit status and the processing object.  For help or an input
    error, return None in place of the processing object.
    """

    # Read command-line inputs before creating the processing object.
    inobj = LAO.InOpts(argv=argv)

    rv = inobj.process_options()
    if rv > 0:
        return 0, None

    if rv < 0:
        ab.EP1('failed to process options')
        return 1, None

    co = inobj.check_options()
    if co :
        ab.EP1('option processing failed during check')
        return 1, None

    # Load the processing library after help and option parsing.
    from vnet_afni import lib_arch_aug_base as LAB

    mainobj = LAB.MainObj(user_inobj=inobj, args_orig=inobj.args_orig)
    if mainobj.status:
        return 1, mainobj

    return 0, mainobj


# ==========================================================================

if __name__ == '__main__':

    stat, mainobj = main()
    sys.exit(stat)
