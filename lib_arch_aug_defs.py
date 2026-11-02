#!/usr/bin/env python

"""Defaults, shared definitions and help for arch_augment.py."""

import os

# ============================================================================
# program information and paths

g_version = 'arch_augment.py 1.0 (Nov 2, 2026)'
g_history = '''
Nov 2, 2026  : migrate data_augmentation_wrapper.py to AFNI-style objects
               and option parsing; use the explicit input/output flags of
               the updated scripts.

Nov 2, 2026  : expand command construction, option parsing and dataset
               planning into explicit steps; add method descriptions.

Nov 2, 2026  : reuse ArchSplitTree for input directory and orig-mask checks.
               Execute generated runners with ab.shell_com.
'''

DIR_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIST_exec_mode = ['None', 'swarm', 'shell']
STR_exec_mode = ', '.join(LIST_exec_mode)


DOPTS = {
    'indir'       : None,
    'outdir'      : None,
    'num_cp'      : 3,
    'seed_num'    : None,
    'exec_mode'   : LIST_exec_mode[0],
    'script_dir'  : DIR_scripts,
    'verb'        : 1,
}

# Command option, attribute, type, short help.  Defaults live only in DOPTS.
LIST_opts = [
    ('-indir', 'indir', str,
     'required input split directory: orig/ and mask/'),
    ('-outdir', 'outdir', str, 'required new output split directory'),
    ('-num_cp', 'num_cp', int, 'total copies per input, including copy 000'),
    ('-seed_num', 'seed_num', int, 'seed for augmentation planning'),
    ('-exec_mode', 'exec_mode', str, 'execution mode: ' + STR_exec_mode),
    ('-script_dir', 'script_dir', str,
     'directory containing the do_*.tcsh scripts'),
    ('-verb', 'verb', int, 'verbosity level'),
]

LIST_terminal_opts = [
    ('-help', 'display help'),
    ('-h', 'display help'),
    ('-hist', 'display program history'),
    ('-ver', 'display version'),
    ('-show_valid_opts', 'display valid options'),
]


# ============================================================================
# output tree, files and commands

LIST_subdirs      = ['orig', 'mask', 'scripts', 'logs']
LIST_nii_ext      = ['.nii.gz', '.nii']
FNAME_master      = 'master_script.tcsh'
FNAME_swarm       = 'run_swarm.tcsh'
FNAME_serial      = 'run_shell.tcsh'
FNAME_plan        = 'daug.json'
FNAME_cmd         = 'log_arch_augment.txt'

LIST_common_flags = [
    '-inset_orig',
    '-inset_mask',
    '-prefix_orig',
    '-prefix_mask',
]

DICT_aug_script = {
    'reface'           : 'do_reface.tcsh',
    'gibbs'            : 'do_gibbs.tcsh',
    'affine'           : 'do_affine.tcsh',
    'gain_inhom'       : 'do_gain_inhomogenity.tcsh',
    'zipper'           : 'do_zipper_noise.tcsh',
    'add_noise'        : 'do_add_noise.tcsh',
    'contrast_core'    : 'do_contrast_variation_core.tcsh',
    'contrast_shading' : 'do_contrast_variation_shading.tcsh',
}

LIST_swarm_opts = [
    '--module afni',
    '--partition=norm,quick',
    '--threads-per-process=4',
    '--gb-per-process=3',
    '--time=00:30:00',
    '--job-name=job_vnet_aug',
    '--merge-output',
    '--usecsh',
]

# ============================================================================
# setup for data augmentation phases: stages and relative percentages

# for each augmentation phase, there is a breakdown into items with a relative
# percentage of occurring, which should add up to 100%
DICT_phase_types = {
    'daug_ph0': {
        'reface' : 20,
        'empty'  : 80,
    },

    'daug_ph1': {
        'gibbs'  : 15,
        'affine' : 45,
        'empty'  : 40,
    },

    'daug_ph2': {
        'gain_inhom'   : 30,
        'zipper'       : 10,
        'add_noise'    : 30,
        'contrast_var' : 30,
    },
}

LIST_phases = list(DICT_phase_types.keys())
LIST_phases.sort()

# ============================================================================
# augmentation probabilities and parameter ranges (legacy distributions)

LIST_affine_types = ['rotation', 'scale', 'shear', 'shift']
LIST_axis_xyz     = ['x', 'y', 'z']
LIST_axis_ijk     = ['i', 'j', 'k']
LIST_gain_axis_weights = [10, 80, 10]

# ranges for random selection
RANGE_gibbs       = (70, 100)
RANGE_rotation    = (-30, 30)
RANGE_scale       = (0.8, 1.2)
RANGE_shear       = (-0.1, 0.1)
RANGE_gain_window = (2.0, 2.8)

# specific mode values that are possible
LIST_shift        = [-5, -10, -15, -20, -25, -30, 5, 10, 15, 20, 25, 30]
LIST_noise_level  = [0.1, 0.2, 0.3]
MODE_reface       = 'mode_reface'
CONTRAST_dim_fac  = 5
CONTRAST_axis     = 'j'
CONTRAST_window   = 2

# 3dAllineate -1Dparam_apply: shifts, Rz/Rx/Ry, scales, shears.
LIST_affine_identity = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
DICT_rotation_index  = {'x': 4, 'y': 5, 'z': 3}
DICT_shear_index     = {'x': 9, 'y': 11, 'z': 10}

# ============================================================================

g_help_string = '''
Overview ~1~

Prepare paired anatomical/mask copies and scripts for V-net augmentation.
The input is one split directory (for example, training), containing
orig/*_orig.nii[.gz] and matching mask/*_mask.nii[.gz] datasets.
Input pairing uses lib_arch_vnet_base.ArchSplitTree, as in training.
Each pair must have the same extension (.nii or .nii.gz), and the two
folders must contain corresponding filenames and equal dataset counts.
The shared tree also requires each folder label (orig or mask) to occur
only once in its corresponding filename.

Copy 000 is unaugmented. Each later copy has three randomly selected phases:
  0: refacing (20%) or no operation (80%).
  1: Gibbs (15%), affine (45%), or no operation (40%).
  2: gain inhomogeneity (30%), zipper (10%), noise (30%), contrast (30%).

# --------------------------------------------------------------------------

Options ~1~

-indir DIR       : (req) input split directory.

-outdir DIR      : (req) output directory; must not already exist.

-num_cp N        : total copies per input, including copy 000
                   (def: {num_cp}).

-seed_num N      : integer planning seed
                   (def: {seed_num}).

-exec_mode MODE  : None, shell, or swarm (default: {exec_mode}).
                   None copies datasets and writes scripts without
                   running AFNI.
                   shell runs each dataset script serially with its own log.
                   swarm submits the generated master file via run_swarm.tcsh.

-script_dir DIR  : location of the updated do_*.tcsh scripts.
                   All supported augmentation scripts must be present,
                   even if a run does not select every augmentation type.
                   (def: directory containing arch_augment.py)

-verb N          : verbosity, a nonnegative integer
                   (def: {verb})

-help, -h        : display help.

-hist            : display history.

-ver             : display version.

-show_valid_opts : display all valid options.

# --------------------------------------------------------------------------

Outputs ~1~

orig/, mask/, scripts/, and logs/.
scripts/ contains per-dataset scripts, master_script.tcsh, run_shell.tcsh,
run_swarm.tcsh,
daug.json (all copies, phase choices, parameters, and source/output paths),
and the copied augmentation scripts. log_arch_augment.txt records the command.

Dataset names retain the full input stem, followed by a three-digit copy
index (expanding if necessary). Both .nii and .nii.gz inputs are supported.
Generated scripts use absolute paths and stop when a processing step fails.
They update only the output copies; the source datasets are never modified.

The seed controls Python phase/parameter selection, not random numbers inside
AFNI programs. Plans are reproducible but differ from the legacy random
stream. Swarm resource options retain the legacy defaults; edit
run_swarm.tcsh as needed.

# --------------------------------------------------------------------------

Examples ~1~

  arch_augment.py -indir DATA/training -outdir AUG/training -num_cp 3

  arch_augment.py -indir DATA/training -outdir AUG/training \\
      -exec_mode shell

To run prepared jobs later, use tcsh scripts/run_shell.tcsh for serial
execution, or tcsh scripts/run_swarm.tcsh for swarm. master_script.tcsh is
the swarm command list; run_shell.tcsh also stops after a failed dataset.
For a fresh augmentation, choose a new output directory; rerunning prepared
scripts applies augmentation again to the existing copies.

'''.format(**DOPTS)
