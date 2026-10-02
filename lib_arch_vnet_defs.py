#!/usr/bin/env python

# A library of default settings for: archimedes_vnet.py
# ============================================================================

#import os, sys, copy

# ============================================================================
# training parameter options (and default settings)

# List of all possible net architectures to choose from. 
LIST_architecture  = [
    'vnet_orig',     # default 
    'Cerebrum',
]
STR_architecture = ', '.join(LIST_architecture)

# -----

# List of all possible optimizers
LIST_optimizer = [
    'Adam',              # default
    'Adam16',
]
STR_optimizer = ', '.join(LIST_optimizer)

# default epsilon value for Adam optimizer, if precision if half or mixed
EPS_adam_nonfull_prec = 1.0e-4

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
    'half',
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
LIST_loss_func_w_wt = [
    "WtSorensen_Dice",
]
STR_loss_func_w_wt = ', '.join(LIST_loss_func_w_wt)

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
    'num_epoch'            : 100,
    'learn_rate'           : 0.0001,
    'seed'                 : 42,
    'batch_size'           : 1,
    'do_weight_norm'       : 'No',
    'do_shuffle'           : 'Yes',  
    'restart_checkpoint'   : None,
    'save_checkpoint_rate' : None,
    'save_checkpoint_list' : [],
    'save_mask_rate'       : None,
    'save_mask_list'       : [],
}

# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")
