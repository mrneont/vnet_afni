#!/usr/bin/env python

# python3 status: compatible

# system libraries
import sys

# AFNI libraries
from    afnipy import option_list   as OL
from    afnipy import afni_base     as ab

from vnet_afni import lib_arch_vnet_wtds as LAVW

# ----------------------------------------------------------------------
# globals

g_help_string = """Overview ~1~

This program helps prepare a data tree that is going to be used to
make a VNET, by making weight dataset ('wtds') volumes for each
existing mask.

The input directory is expected to contain training/ and validation/
data split directories, each with a mask/ subdirectory.  

This program will create a new wtds/ subdirectory parallel to it. For
each mask/*_mask.nii* dataset, an associated wtds/*_wtds.nii.gz
dataset will be created by running the supplementary
adjunct_mask_to_wtds.tcsh script.

One can also use the -swarm_script option to create a swarmable script
(i.e., one full call to the adjunct*.tcsh script for each mask dataset)
for carrying out this procedure, which can be executed according to the 
user's desired settings.

auth: Y Narayana Swamy (SSCC, NIMH, NIH, USA)
      PA Taylor (SSCC, NIMH, NIH, USA)

--------------------------------------------------------------------------

Usage ~1~

-indir INDIR     :(req) VNET input directory containing training/ and
                  validation/ directories

-overwrite       :allow pre-existing wtds/ directories and overwrite their
                  associated output datasets

-swarm_script SS :write a swarmable command file SS instead of running the
                  mask-to-wtds commands directly; each line will contain one
                  adjunct_mask_to_wtds.tcsh call with full input/output paths

-swarm_cmd_file SCF
                 :write a command file SCF for running the swarm script, which
                  the user can either edit/refine, or execute directly; 
                  NB: this option requires also using -swarm_script

-verb VVV        :control verbosity 
                  (def: 1)

-help, -h        :display program help file

-show_valid_opts :show valid options for this program

--------------------------------------------------------------------------

Examples ~1~

  1. Basic usage):

     arch_vnet_make_wtds.py                      \\
         -indir data_vnet

  2. Basic usage for making a swarm script and executable swarm command, 
     to be run separately to do all the work:

     arch_vnet_make_wtds.py                      \\
         -indir           data_vnet              \\
         -swarm_script    run_vnet_wtds.swarm    \\
         -swarm_cmd_file  run_vnet_wtds.tcsh

"""


class InOpts:
    """Object for storing and parsing command line inputs."""

    def __init__(self):

        self.status     = 0
        self.valid_opts = None
        self.user_opts  = None

        self.verb       = 1
        self.indir      = None
        self.overwrite  = False
        self.swarm_script   = None
        self.swarm_cmd_file = None

        self.init_options()

    def init_options(self):
        """Prepare the set of all options."""

        self.valid_opts = OL.OptionList('valid opts')

        self.valid_opts.add_opt('-help', 0, [],
                                helpstr='display program help')

        self.valid_opts.add_opt('-show_valid_opts', 0, [],
                                helpstr='display all valid options')

        self.valid_opts.add_opt('-indir', 1, [],
                                helpstr='VNET input directory')

        self.valid_opts.add_opt('-overwrite', 0, [],
                                helpstr='overwrite pre-existing wtds outputs')

        self.valid_opts.add_opt('-swarm_script', 1, [],
                                helpstr='write swarm command file')

        self.valid_opts.add_opt('-swarm_cmd_file', 1, [],
                                helpstr='write command file to run swarm')

        self.valid_opts.add_opt('-verb', 1, [],
                                helpstr='set the verbosity level')

        return 0

    def process_options(self):
        """Read command line options."""

        self.valid_opts.check_special_opts(sys.argv)

        if len(sys.argv) <= 1 or '-help' in sys.argv or '-h' in sys.argv:
            print(g_help_string)
            return 1

        if '-show_valid_opts' in sys.argv:
            self.valid_opts.show('', 1)
            return 1

        self.user_opts = OL.read_options(sys.argv, self.valid_opts)
        uopts = self.user_opts
        if not uopts:
            return -1

        val, err = uopts.get_type_opt(int, '-verb')
        if val is not None and not err:
            self.verb = val

        err_base = "Problem interpreting use of opt: "

        for opt in uopts.olist:

            if opt.name == '-indir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP(err_base + opt.name)
                self.indir = val

            elif opt.name == '-overwrite':
                self.overwrite = True

            elif opt.name == '-swarm_script':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP(err_base + opt.name)
                self.swarm_script = val

            elif opt.name == '-swarm_cmd_file':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    ab.EP(err_base + opt.name)
                self.swarm_cmd_file = val

        return 0

    def check_options(self):
        """Perform final command line checks before execution."""

        BAD_RETURN = -1

        if self.indir is None:
            ab.EP1("missing -indir option")
            return BAD_RETURN

        if self.swarm_cmd_file and not(self.swarm_script):
            ab.EP1("-swarm_cmd_file requires -swarm_script")
            return BAD_RETURN

        return 0


# ----------------------------------------------------------------------------

def main():

    inobj = InOpts()

    rv = inobj.process_options()
    if rv > 0:
        return 0, None
    if rv < 0:
        ab.EP1('failed to process options')
        return 1, None

    rv = inobj.check_options()
    if rv:
        ab.EP1('failed whilst checking options')
        return rv, None

    mainobj = LAVW.MainObj(user_inobj=inobj)
    if not(mainobj) or mainobj.status:
        return 1, mainobj

    return 0, mainobj


# ============================================================================

if __name__ == '__main__':

    stat, mainobj = main()
    sys.exit(stat)
