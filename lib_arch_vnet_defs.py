#!/usr/bin/env python

# A library of default settings for: archimedes_vnet.py
# ============================================================================

from afnipy import lib_format_cmd_str as lfcs

# ============================================================================
# training parameter options (and default settings)

# List of all possible net architectures to choose from. 
LIST_architecture  = [
    'vnet_orig',     # default 
 #   'Cerebrum',     # NB: we are just using vnet_orig at present
]
STR_architecture = ', '.join(LIST_architecture)

# -----

# List of all possible optimizers
LIST_optimizer = [
    'Adam',              # default
#    'Adam16',           # NB: this doesn't seem to be enabled/usable
]
STR_optimizer = ', '.join(LIST_optimizer)

## This is not needed at present
### default epsilon value for Adam optimizer, if precision if half or mixed
##EPS_adam_nonfull_prec = 1.0e-4

# -----

# List of all possible data normalizations
LIST_scale_mode = [
    'z_scoring',          # default
    'min_max_scale',
]
STR_scale_mode = ', '.join(LIST_scale_mode)

# -----

LIST_precision = [
    'full',              # default
    #'half',             # not implemented at present
    'mixed',
]
STR_precision = ', '.join(LIST_precision)

# -----

# List of all possible loss functions
LIST_loss_func = [
    "WtSorensen_Dice",   # default
    "Sorensen_Dice_mean",
    "Sorensen_Dice_single_channel",
]
STR_loss_func = ', '.join(LIST_loss_func)

# ... and a related list: each loss_func that uses weights
LIST_loss_func_has_wtds = [
    "WtSorensen_Dice",
]
STR_loss_func_has_wtds = ', '.join(LIST_loss_func_has_wtds)

# -----

# List of available devices to use
LIST_device = [
    'auto',     # default
    'cpu',
    'cuda',
    'mps',   # mac 
]
STR_device  = ', '.join(LIST_device)

# ============================================================================
# default values, parameters and settings for the main obj

DOPTS = {
    'user_opts'            : [],         # command the user ran
    'verb'                 : 1,
    'overwrite'            : '',
    'do_clean'             : 'Yes',
    'do_log'               : 'No',
    'do_log_loss'          : 'Yes',
    'do_plot_loss'         : 'Yes',
    'indir'                : '',
    'outdir'               : None,
    'workdir'              : '',
    'architecture'         : LIST_architecture[0],
    'scale_mode'           : LIST_scale_mode[0],
    'optimizer'            : LIST_optimizer[0],
    'precision'            : LIST_precision[0],
    'loss_func'            : LIST_loss_func[0],
    'device'               : LIST_device[0],
    'num_cpu'              : -1,
    'max_epoch'            : 100,
    'learn_rate'           : 0.0001,
    'seed'                 : 42,
    'batch_size'           : 1,
    'do_weight_norm'       : 'No',
    'do_shuffle'           : 'Yes',  
    'do_strict_load'       : 'Yes',  
    'restart_checkpoint'   : None,
    'save_checkpoint_rate' : None,
    'save_checkpoint_list' : [],
    'save_mask_rate'       : None,
    'save_mask_list'       : [],
}

DOPTS_all_keys = list(DOPTS.keys())

# ============================================================================

def get_arg_str(argv, do_niceify=True):
    """Parse the command line call argv, which is a list of all the pieces
therein.  The pieces can just be joined together with whitespace, or
more nicely spaced out across multiple lines and vertically aligned
with do_niceify.

Parameters
----------
argv: list
    list of strings, being the pieces of the command line command
do_niceify: bool
    should we space things out nicely across multiple lines?
    (def: of course we should!)

Returns
-------
arg_str : str
    The single string that is the full command line command

    """

    arg_str = ' '.join(argv)

    if do_niceify :
        # might be simpler way to get from parser?
        all_opts = ["-" + key for key in DOPTS_all_keys]

        # create str
        is_diff, arg_str = \
            lfcs.afni_niceify_cmd_str(arg_str, list_cmd_args = all_opts)

    return arg_str

# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")
