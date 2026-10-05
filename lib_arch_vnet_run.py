#!/usr/bin/env python

# A library of functions for dealing with archimedes_vnet.py
#
# auth : PA Taylor (SSCC, NIMH, NIH, USA)
# ----------------------------------------------------------------------------
# ver 1.0 : start mapping original vnet funcs to AFNI format/objs
# ============================================================================

import sys, os, copy, glob
import platform
from  datetime import datetime

from    afnipy import afni_base          as ab
from    afnipy import afni_util          as au
from    afnipy import lib_torch_util     as ltu

from vnet_afni import lib_arch_vnet_defs as DEF
from vnet_afni import lib_arch_vnet_base as LAVB
from vnet_afni import lib_arch_vnet_util as LAVU
from vnet_afni import lib_ml_losses      as lml
from vnet_afni import lib_ml_models      as lmm
from vnet_afni import lib_ml_cerebrum    as lmc

import numpy as np

import torch
from   torch.utils.data import Dataset, DataLoader

from communifti import lib_nibabel_read_nifti  as lnrn
from communifti import lib_nibabel_write_nifti as lnwn

# ============================================================================
# the allowed split/phase keywords when looping through epochs

LIST_phase = [
    'training',
    'validation',
]
STR_phase = ', '.join(LIST_phase)

# ============================================================================

# NB: we define a 'sample' as a matched packet of volumes to be used
# for training or validation: orig + mask (+ wtds, optionally). A
# 'sample set' is the indexed collection of such samples used for
# training or validation.  
# 
# This hopefully avoids overloading the term 'dataset', which we often
# use in AFNI for a single NIFTI volume. A sample_set can (and
# generally will) be more than one dataset, in that sense.

class ArchVnetSampleSet(Dataset):
    """PyTorch dataset wrapper around one ArchSplitTree.

    ArchSplitTree has already checked that orig, mask and optional wtds
    datasets correspond.  This class just loads and prepares one indexed
    sample of volumes.
    """

    def __init__(self, split_tree, scale_mode='z_scoring',
                 has_wtds=False, verb=1):

        self.split_tree = split_tree
        self.scale_mode = scale_mode
        self.has_wtds   = has_wtds
        self.verb       = verb


    def __len__(self):

        return self.split_tree.count_subdir_files('orig')


    def __getitem__(self, index):

        # ----- orig

        fname_orig = self.split_tree.all_dset['orig'][index]
        path_orig  = os.path.join(
            self.split_tree.get_subdir_path('orig'),
            fname_orig
        )

        is_fail, data_orig, hdr_orig = LAVU.load_orig_dset(
            path_orig,
            do_perc_thr = True,
            scale_mode  = self.scale_mode,
            verb        = 0
        )
        if is_fail :
            msg = "Failed to load orig dset: {}".format(path_orig)
            raise RuntimeError(msg)

        # ----- mask

        fname_mask = self.split_tree.all_dset['mask'][index]
        path_mask  = os.path.join(
            self.split_tree.get_subdir_path('mask'),
            fname_mask
        )

        is_fail, arr_mask, _ = lnrn.read_nifti_to_nibabel(
            path_mask,
            set_dtype = np.float32,
            verb      = 0
        )
        if is_fail :
            msg = "Failed to load mask dset: {}".format(path_mask)
            raise RuntimeError(msg)

        # [D,H,W] -> [1,D,H,W]
        data_mask = torch.from_numpy(arr_mask).unsqueeze(0)

        # ----- output sample

        sample = {
            'orig'     : data_orig,
            'mask'     : data_mask,
            'fname'    : fname_orig,
            'hdr_orig' : hdr_orig,
        }

        # ----- optional weight dset

        if self.has_wtds :

            fname_wtds = self.split_tree.all_dset['wtds'][index]
            path_wtds  = os.path.join(
                self.split_tree.get_subdir_path('wtds'),
                fname_wtds
            )

            is_fail, arr_wtds, _ = lnrn.read_nifti_to_nibabel(
                path_wtds,
                set_dtype = np.float32,
                verb      = 0
            )
            if is_fail :
                raise RuntimeError(
                    "Failed to load weight dset: {}".format(path_wtds)
                )

            sample['wtds'] = torch.from_numpy(arr_wtds).unsqueeze(0)

        return sample


def collate_arch_vnet_samples(sample_list):
    """Collate samples while preserving nibabel header objects as a list."""

    batch = {
        'orig'     : torch.stack([x['orig'] for x in sample_list]),
        'mask'     : torch.stack([x['mask'] for x in sample_list]),
        'fname'    : [x['fname'] for x in sample_list],
        'hdr_orig' : [x['hdr_orig'] for x in sample_list],
    }

    if 'wtds' in sample_list[0] :
        batch['wtds'] = torch.stack([x['wtds'] for x in sample_list])

    return batch


# ============================================================================

class MainObj:
    """Object for setting up archimedes_vnet.py, to carry out training.

Parameters
----------
inobj : InOpts object 
    object constructed from running TEMPLATE_tool.py on the command
    line. At present, the only way to really provide inputs here.

    """

    def __init__(self, user_inobj=None, args_orig=[]):

        # ----- set up attributes

        # main input variables
        self.status          = 0
        self.user_opts       = DEF.DOPTS['user_opts']  # command the user ran
        self.user_inobj      = user_inobj
        self.args_orig       = args_orig

        # general variables
        self.verb            = DEF.DOPTS['verb']
        self.overwrite       = DEF.DOPTS['overwrite']
        self.do_clean        = DEF.DOPTS['do_clean']
        self.do_log          = DEF.DOPTS['do_log']
        self.do_log_loss     = DEF.DOPTS['do_log_loss']

        # main data variables
        self.indir           = DEF.DOPTS['indir']
        self.outdir          = DEF.DOPTS['outdir']     # None or str
        self.workdir         = DEF.DOPTS['workdir']
        self.time_start      = None
        self.time_finish     = None
        self.time_duration   = None

        # control variables
        self.max_epoch       = DEF.DOPTS['max_epoch']
        self.learn_rate      = DEF.DOPTS['learn_rate']
        self.architecture    = DEF.DOPTS['architecture']
        self.optimizer       = DEF.DOPTS['optimizer']
        self.scale_mode      = DEF.DOPTS['scale_mode']
        self.precision       = DEF.DOPTS['precision']
        self.loss_func       = DEF.DOPTS['loss_func']
        self.device          = DEF.DOPTS['device']
        self.seed            = DEF.DOPTS['seed']
        self.num_cpu         = DEF.DOPTS['num_cpu']
        self.batch_size      = DEF.DOPTS['batch_size']
        self.do_weight_norm  = DEF.DOPTS['do_weight_norm']
        self.do_shuffle      = DEF.DOPTS['do_shuffle']
        self.do_strict_load  = DEF.DOPTS['do_strict_load']
        self.restart_checkpoint   = DEF.DOPTS['restart_checkpoint']
        self.save_checkpoint_rate = DEF.DOPTS['save_checkpoint_rate']
        self.save_checkpoint_list = DEF.DOPTS['save_checkpoint_list'].copy()
        self.save_mask_rate  = DEF.DOPTS['save_mask_rate']
        self.save_mask_list  = DEF.DOPTS['save_mask_list'].copy()

        # things created in the processing/training setup
        self.sysname         = None     # platform system
        self.loss_has_wtds   = False    # some loss_func need a weight dset

        # model parameters: here, just for binary classification from 1 vol
        self.num_channel_in  = 1        # single vol input
        self.num_class_out   = 2        # num channels out (binary classifier)

        # model attributes, set/made/loaded below
        self.net             = None     # the (v)net model itself
        self.net_optim       = None     # optimizer-in-action for network
        self.grad_scaler     = None

        self.data_tree       = None
        self.loader_train    = None
        self.loader_valid    = None
        self.net_loss        = None
        self.outdir_mask     = None
        self.fname_log_cmd   = None
        self.fname_log_loss_training   = None
        self.fname_log_loss_validation = None

        # ----- take action(s)

        # prelim stuff
        if user_inobj :
            self.status = self.load_from_inopts()
            if self.status : return

            self.status = self.basic_setup()
            if self.status : return

            self.status = self.set_device_and_cpus()
            if self.status : return

            self.status = self.make_net()
            if self.status : return

            self.status = self.load_optimizer()
            if self.status : return

            self.status = self.make_outdir()
            if self.status : return

            ### NOT NEEDED SO FAR ***
            ###tmp = self.make_workdir()
            ###if tmp : return

            self.status = self.make_dataloaders()
            if self.status : return

            self.status = self.make_loss()
            if self.status : return

            self.status = self.run_training()
            if self.status : return

            self.status = self.finish_cmd_log()
            if self.status : return

            ### NOT NEEDED SO FAR ***
            ###if self.do_clean :
            ###    self.status = self.remove_workdir()
            ###    if self.status : return

    # ----- methods

    def load_from_inopts(self):
        """Populate the input values using the command line interface
        input. The user information is provided as the self.user_inobj
        object, which gets parsed and redistributed here.
        """
        
        if not(self.user_inobj) :
            ab.WP("No user_inobj? Nothing to do.")
            return 0

        # shorter name to use, and less confusing with 'self' usage
        io = self.user_inobj

        if io.user_opts is not None :
            self.user_opts = io.user_opts

        # general variables
        if io.verb is not None :
            self.verb = io.verb
        if io.overwrite is not None :
            self.overwrite = io.overwrite
        if io.do_clean is not None :
            self.do_clean = io.do_clean
        if io.do_log is not None :
            self.do_log = io.do_log

        if io.do_log_loss is not None :
            self.do_log_loss = io.do_log_loss

        # main data variables
        if io.indir is not None :
            self.indir = io.indir
        if io.outdir is not None :
            self.outdir = io.outdir
        if io.workdir is not None :
            self.workdir = io.workdir

        # control variables

        if io.architecture is not None :
            self.architecture = io.architecture
        if io.scale_mode is not None :
            self.scale_mode = io.scale_mode
        if io.optimizer is not None :
            self.optimizer = io.optimizer
        if io.precision is not None :
            self.precision = io.precision
        if io.max_epoch is not None :
            self.max_epoch = io.max_epoch
        if io.learn_rate is not None :
            self.learn_rate = io.learn_rate
        if io.loss_func is not None :
            self.loss_func = io.loss_func
        if io.device is not None :
            self.device = io.device
        if io.seed is not None :
            self.seed = io.seed
        if io.num_cpu is not None :
            self.num_cpu = io.num_cpu
        if io.batch_size is not None :
            self.batch_size = io.batch_size
        if io.do_weight_norm is not None :
            self.do_weight_norm = io.do_weight_norm
        if io.do_strict_load is not None :
            self.do_strict_load = io.do_strict_load
        if io.do_shuffle is not None :
            self.do_shuffle = io.do_shuffle
        if io.restart_checkpoint is not None :
            self.restart_checkpoint = io.restart_checkpoint
        if io.save_checkpoint_rate is not None :
            self.save_checkpoint_rate = io.save_checkpoint_rate
        if io.save_checkpoint_list is not None :
            self.save_checkpoint_list = io.save_checkpoint_list
        if io.save_mask_rate is not None :
            self.save_mask_rate = io.save_mask_rate
        if io.save_mask_list is not None :
            self.save_mask_list = io.save_mask_list

        return 0


    def basic_setup(self):
        """Run through basic checks of what has been input, and fill in any
        further information that is needed (like wdir, etc.)"""

        self.time_start = datetime.now()

        # check basic requirements

        if not(self.indir) :
            ab.EP("Need to provide an indir")
        else:
            dir_exists = os.path.isdir(self.indir)
            if not(dir_exists) :
                ab.EP("Failed to find indir")

        if not(self.outdir) :
            ab.EP("Need to provide an outdir")
        else:
            dir_exists = os.path.isdir(self.outdir)
            if dir_exists :
                msg = "The outdir {} exists already".format(self.outdir)
                if not(self.overwrite) :
                    ab.EP(msg)
                else:
                    msg+= "\n... but -overwrite is used, so we continue"
                    ab.WP(msg)

        if self.restart_checkpoint is not None :
            file_exists = os.path.isfile(self.restart_checkpoint)
            if not(file_exists) :
                msg = "Cannot find restart_checkpoint: "
                msg+= "{}".format(self.restart_checkpoint)
                ab.EP(msg)

        if self.architecture : 
            if self.architecture not in DEF.LIST_architecture :
                msg = "Invalid value after -architecture: "
                msg+= "{}\n".format(self.architecture)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_architecture)
                ab.EP(msg)

        # determine the loss func
        if self.loss_func : 
            if self.loss_func not in DEF.LIST_loss_func :
                msg = "Invalid value after -loss_func: "
                msg+= "{}\n".format(self.loss_func)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_loss_func)
                ab.EP(msg)
        # ... and check whether it will need a weight
        if self.loss_func in DEF.LIST_loss_func_has_wtds :
            self.loss_has_wtds = True

        if self.scale_mode : 
            if self.scale_mode not in DEF.LIST_scale_mode :
                msg = "Invalid value after -scale_mode: "
                msg+= "{}\n".format(self.scale_mode)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_scale_mode)
                ab.EP(msg)

        if self.precision : 
            if self.precision not in DEF.LIST_precision :
                msg = "Invalid value after -precision: "
                msg+= "{}\n".format(self.precision)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_precision)
                ab.EP(msg)

        if self.optimizer : 
            if self.optimizer not in DEF.LIST_optimizer :
                msg = "Invalid value after -optimizer: "
                msg+= "{}\n".format(self.optimizer)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_optimizer)
                ab.EP(msg)

        if self.learn_rate <= 0.0 :
            msg = "Invalid value after -learn_rate: "
            msg+= "{}\n".format(self.learn_rate)
            msg+= "Cannot be <= 0.0"
            ab.EP(msg)
            
        if self.batch_size <= 0 :
            msg = "Invalid value after -batch_size: "
            msg+= "{}\n".format(self.batch_size)
            msg+= "Cannot be <= 0"
            ab.EP(msg)

        if self.max_epoch < 0 :
            msg = "Invalid value after -max_epoch: "
            msg+= "{}\n".format(self.max_epoch)
            msg+= "Cannot be < 0"
            ab.EP(msg)

        if self.save_checkpoint_rate is not None :
            if self.save_checkpoint_rate <= 0 :
                msg = "Invalid value after -save_checkpoint_rate: "
                msg+= "{}\n".format(self.save_checkpoint_rate)
                msg+= "Cannot be <= 0"
                ab.EP(msg)

        # if using save_checkpoint_list, values must be ints
        if len(self.save_checkpoint_list) :
            is_fail, self.save_checkpoint_list = \
                convert_list_of_str_to_int(self.save_checkpoint_list, 
                                           verb=self.verb)
            if is_fail :
                msg = "Invalid values after -save_checkpoint_list: "
                msg+= "{}\n".format(self.save_checkpoint_list)
                msg+= "Could not convert all to int"
                ab.EP(msg)

        # check about combining save_checkpoint_rate with the list, or just
        # verify the list (default or user-entered)
        is_fail, self.save_checkpoint_list = \
            combine_list_rate_max(
                self.save_checkpoint_list,
                self.save_checkpoint_rate,
                self.max_epoch,
                label="checkpoint",
                verb=self.verb
            )
        if is_fail :
            msg = "Could not merge in values from -save_checkpoint_rate: "
            msg+= "{}\n".format(self.save_checkpoint_rate)
            ab.EP(msg)

        # finalize checkpoint list: always include self.max_epochs
        if self.max_epoch not in self.save_checkpoint_list :
            self.save_checkpoint_list.append(self.max_epoch)

        if self.save_mask_rate is not None :
            if self.save_mask_rate <= 0 :
                msg = "Invalid value after -save_mask_rate: "
                msg+= "{}\n".format(self.save_mask_rate)
                msg+= "Cannot be <= 0"
                ab.EP(msg)

        # if using save_mask_list, values must be ints
        if len(self.save_mask_list) :
            is_fail, self.save_mask_list = \
                convert_list_of_str_to_int(self.save_mask_list, 
                                           verb=self.verb)
            if is_fail :
                msg = "Invalid values after -save_mask_list: "
                msg+= "{}\n".format(self.save_mask_list)
                msg+= "Could not convert all to int"
                ab.EP(msg)

        # check about combining save_mask_rate with the list, or just
        # verify the list (default or user-entered)
        is_fail, self.save_mask_list = \
            combine_list_rate_max(
                self.save_mask_list,
                self.save_mask_rate,
                self.max_epoch,
                label="mask",
                verb=self.verb
            )
        if is_fail :
            msg = "Could not merge in values from -save_mask_rate: "
            msg+= "{}\n".format(self.save_mask_rate)
            ab.EP(msg)

        # store platform system name 
        self.sysname = platform.system()

        # setup device
        if self.device : 
            if self.device not in DEF.LIST_device :
                msg = "Invalid value after -device: "
                msg+= "{}\n".format(self.device)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_device)
                ab.EP(msg)

        # generate basic items

        # generate wdir with random component, if none provided
        if not(self.workdir) :
            cmd  = '3dnewid -fun11'
            com  = ab.shell_com(cmd, capture=1)
            stat = com.run()
            rstr = com.so[0].strip()
            self.workdir = '__wdir_TEMPLATE_' + rstr

        # convert bool-ish opts to bools
        self.do_weight_norm = au.convert_to_bool_yn10(self.do_weight_norm)
        self.do_strict_load = au.convert_to_bool_yn10(self.do_strict_load)
        self.do_shuffle     = au.convert_to_bool_yn10(self.do_shuffle)
        self.do_clean       = au.convert_to_bool_yn10(self.do_clean)
        self.do_log         = au.convert_to_bool_yn10(self.do_log)
        self.do_log_loss    = au.convert_to_bool_yn10(self.do_log_loss)

        return 0

    def set_device_and_cpus(self):
        """Choose the device to be used; this might override the user's
        choice, based on the realities of the system. 

        Also set how many CPUs to use. There are different ways to
        specify; can also do nothing and just let system decide. This
        is managed in the torch-related library in AFNI."""

        BAD_RETURN = -1

        # get/set the device
        is_fail, self.device = ltu.select_device_general(dev_in=self.device,
                                                         verb=self.verb)
        if is_fail :
            ab.EP1("Failed select device")
            return BAD_RETURN

        # another precision-based consideration
        if self.precision == 'mixed':
            if self.device != 'cuda':
                msg = "Mixed precision training requires device to be 'cuda'"
                ab.EP1(msg)
                return BAD_RETURN

            self.grad_scaler = torch.amp.GradScaler('cuda')

        # get/set number of CPUs to use
        is_fail = ltu.set_torch_cpus(num_cpu=self.num_cpu, verb=self.verb)
        if is_fail :
            ab.EP1("Failed select num_cpu")
            return BAD_RETURN

        # make use of random seed, if provided by user
        if self.seed is not None:
            torch.manual_seed(self.seed)

        return 0

    def make_net(self):
        """Create the network with the desired model. This function will also
        load a restart_checkpoint, if that has been provided"""

        if self.verb :
            ab.IP("Make network architecture: {}".format(self.architecture))

        BAD_RETURN = -1

        # NB: verb level is 'downgraded' here, to keep useful default
        # outputs for both this program and 3dBrainTeaser. So, we
        # cheat a bit here.
        if self.architecture == 'vnet_orig' :
            try:
                self.net = lmm.VNet_orig(
                    in_channels = self.num_channel_in,
                    num_class   = self.num_class_out,
                    wt_norm     = self.do_weight_norm, 
                    verb        = max(self.verb-1, 0)
                )
            except:
                ab.EP1("Failed to make network: {}".format(self.architecture))
                return BAD_RETURN

        # this if-branch is now impossible, bc Cerebrum is not enabled now
        elif 0 and self.architecture == 'Cerebrum' :
            try:
                self.net = lmc.Cerebrum(
                    in_channels = self.num_channel_in,
                    num_class   = self.num_class_out,
                    wt_norm     = self.do_weight_norm, 
                    verb        = max(self.verb-1, 0)
                )
            except:
                ab.EP1("Failed to make network: {}".format(self.architecture))
                return BAD_RETURN

        else:
            msg = "Unknown architecture: {}\n".format(self.architecture)
            msg+= "Try again, using one from the allowed list:\n"
            msg+= "{}".format(DEF.STR_architecture)
            ab.EP1(msg)
            return BAD_RETURN

        # load "restart" checkpoint?
        if self.restart_checkpoint :
            if not(os.path.isfile(self.restart_checkpoint)) :
                msg = "Cannot find restart checkpoint to load: "
                msg+= "{}".format(self.restart_checkpoint)
                ab.EP1(msg)
                return BAD_RETURN

            try:
                tload = torch.load(
                    self.restart_checkpoint,
                    weights_only=True,
                    map_location=torch.device(self.device),
                )
                # in case strict=False, add in potential debugging
                # about differences in the state load that could
                # happen (NB: we don't want these to happen!)
                missing, unexpected = \
                    self.net.load_state_dict(tload, 
                                             strict=self.do_strict_load)
                if missing :
                    ab.WP("Missing checkpoint keys: {}".format(missing))
                if unexpected :
                    ab.WP("Unexpected checkpoint keys: {}".format(unexpected))
            except:
                msg = "Failed to load checkpoint: "
                msg+= "{}".format(self.restart_checkpoint)
                ab.EP1(msg)
                return BAD_RETURN

        # move the model to the device
        self.net.to(self.device)

        return 0

    def load_optimizer(self):
        """Setup the optimizer, using network parameters."""

        if self.verb :
            ab.IP("Load optimizer: {}".format(self.optimizer))

        BAD_RETURN = -1

        # only one optimizer at present
        if self.optimizer == 'Adam':
            self.net_optim = torch.optim.Adam(
                self.net.parameters(),
                lr = self.learn_rate,
            )

        else:
            msg = "Unknown optimizer: {}\n".format(self.optimizer)
            msg+= "Try again, using one from the allowed list:\n"
            msg+= "{}".format(DEF.STR_optimizer)
            ab.EP1(msg)
            return BAD_RETURN

        return 0

    def make_outdir(self):
        """make the outdir, if it doesn't exist already."""

        BAD_RETURN = -1

        # create output directory before writing checkpoints or other outputs
        try:
            os.makedirs(self.outdir, exist_ok=self.overwrite)
        except OSError:
            ab.EP1("Could not create outdir: {}".format(self.outdir))
            return BAD_RETURN

        # log the command that was run (in nice format) and its start time
        self.fname_log_cmd = os.path.join(self.outdir, 'log_cmd_run.txt')
        try:
            cmd_run = DEF.get_arg_str(self.args_orig, do_niceify=True)

            with open(self.fname_log_cmd, 'w') as fff:
                tts = self.time_start.isoformat(timespec='seconds')
                fff.write("# started: {}\n\n".format(tts))
                fff.write("# cmd run:\n")
                fff.write("{}\n".format(cmd_run))
        except OSError:
            ab.EP1("Could not write command log: {}".format(
                self.fname_log_cmd))
            return BAD_RETURN

        # initialize per-epoch loss logs
        if self.do_log_loss :
            self.fname_log_loss_training = os.path.join(
                self.outdir, 'log_loss_training.dat')
            self.fname_log_loss_validation = os.path.join(
                self.outdir, 'log_loss_validation.dat')
            for fname in [self.fname_log_loss_training,
                          self.fname_log_loss_validation]:
                try:
                    with open(fname, 'w') as fff:
                        # match the format spacing of the numbers, below
                        txt = "{:7s} ".format('# epoch')
                        txt+= "{:>10s} {:>10s} ".format('min', 'max')
                        txt+= "{:>10s} {:>10s}\n".format('mean', 'stdev')
                        fff.write(txt)
                except OSError:
                    ab.EP1("Could not create loss log: {}".format(fname))
                    return BAD_RETURN

        # keep predicted masks together in their own output subdirectory
        if self.nsave_mask :
            self.outdir_mask = os.path.join(self.outdir, 'pmask')
            try:
                os.makedirs(self.outdir_mask, exist_ok=self.overwrite)
            except OSError:
                ab.EP1("Could not create mask outdir: {}".format(
                    self.outdir_mask))
                return BAD_RETURN

        return 0

    def make_dataloaders(self):
        """Create training and validation datasets/loaders."""

        if self.verb :
            ab.IP("Make training and validation dataloaders")

        BAD_RETURN = -1

        # This object verifies the training/validation directory trees and
        # the correspondence among orig/mask/optional-wtds datasets.
        self.data_tree = LAVB.ArchRootTree(
            self.indir,
            has_mask = True,
            has_wtds = self.loss_has_wtds,
            verb     = self.verb
        )
        if self.data_tree.status :
            ab.EP1("Could not load main input dir correctly")
            return BAD_RETURN

        split_train = self.data_tree.all_split['training']
        split_valid = self.data_tree.all_split['validation']

        if split_train is None or split_valid is None :
            ab.EP1("Failed to make training/validation data trees")
            return BAD_RETURN

        sample_set_train = ArchVnetSampleSet(
            split_train,
            scale_mode = self.scale_mode,
            has_wtds   = self.loss_has_wtds,
            verb       = self.verb
        )
        if not len(sample_set_train):
            ab.EP1("No training datasets found")
            return BAD_RETURN

        sample_set_valid = ArchVnetSampleSet(
            split_valid,
            scale_mode = self.scale_mode,
            has_wtds   = self.loss_has_wtds,
            verb       = self.verb
        )
        if not len(sample_set_valid):
            ab.EP1("No validation datasets found")
            return BAD_RETURN

        self.loader_train = DataLoader(
            sample_set_train,
            batch_size = self.batch_size,
            shuffle    = self.do_shuffle,
            collate_fn = collate_arch_vnet_samples
        )

        # Retain the old behavior of validating one dataset at a time.
        self.loader_valid = DataLoader(
            sample_set_valid,
            batch_size = 1,
            shuffle    = False,
            collate_fn = collate_arch_vnet_samples
        )

        if self.verb :
            ab.IP("Num training dsets   : {}".format(len(sample_set_train)))
            ab.IP("Num validation dsets : {}".format(len(sample_set_valid)))

        return 0

    def make_loss(self):
        """Create the selected loss-function object."""

        if self.verb :
            ab.IP("Make loss function: {}".format(self.loss_func))

        BAD_RETURN = -1

        loss_name = 'CalcLoss_' + self.loss_func

        try:
            loss_class = getattr(lml, loss_name)
            self.net_loss = loss_class()
        except (AttributeError, TypeError):
            ab.EP1("Failed to make loss function: {}".format(loss_name))
            return BAD_RETURN

        return 0

    def run_epoch(self, loader, phase='training', epoch=None):
        """Run one training or validation epoch.

        Parameters
        ----------
        loader : DataLoader
            loader for the desired data split
        phase : str
            one of allowed phase/splits
        epoch : int
            current epoch index; used for optional predicted-mask output

        Returns
        -------
        is_fail : int
            0 for success
        losses : list
            scalar loss for each batch
        """

        BAD_RETURN = (-1, [])
        
        if not(phase in LIST_phase) :
            msg = "Unknown phase in run_epoch : {}\n".format(phase)
            msg+= "Should be from this list:\n{}".format(STR_phase)
            ab.EP1(msg)
            return BAD_RETURN

        # convenient to have this as a boolean, below
        do_train = False

        if phase == 'training' :
            self.net.train()
            do_train = True
        elif phase == 'validation' :
            self.net.eval()
        else:
            # should never reach
            msg = "Reached an impossible point, via phase: {}".format(phase)
            ab.EP1(msg)
            return BAD_RETURN

        if self.verb :
            ab.IP("Start {} epoch".format(phase))

        # add this for mixed precision and cuda behavior; amp =
        # Automatic Mixed Precision, PyTorch automatically chooses
        # when to use float16 or float32 for operations
        use_amp = (
            self.precision == 'mixed'
            and self.device == 'cuda'
        )

        losses = []

        # no gradients are needed during validation
        with torch.set_grad_enabled(do_train):

            num_batch = len(loader)
            for ibatch, batch in enumerate(loader):

                if self.verb :
                    ab.IP("Batch {:04d} / {:04d}".format(ibatch+1, num_batch))

                data_orig = batch['orig'].to(self.device)
                data_mask = batch['mask'].to(self.device)

                if self.loss_has_wtds :
                    data_wtds = batch['wtds'].to(self.device)

                if do_train :
                    self.net_optim.zero_grad(set_to_none=True)

                with torch.autocast(
                        device_type = 'cuda',
                        dtype       = torch.float16,
                        enabled     = use_amp):

                    # Cerebrum has a different forward signature than Vnet
                    if self.architecture == 'Cerebrum' :
                        pred = self.net(data_orig, self.verb)
                    else:
                        pred = self.net(data_orig)

                    if self.loss_has_wtds :
                        loss = self.net_loss(pred, data_mask, data_wtds)
                    else:
                        loss = self.net_loss(pred, data_mask)

                if do_train :
                    if use_amp :
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.step(self.net_optim)
                        self.grad_scaler.update()
                    else:
                        loss.backward()
                        self.net_optim.step()

                losses.append(float(loss.detach().cpu()))

                if epoch in self.save_mask_list :
                    is_fail = self.write_pred_masks(pred, batch, phase, epoch)
                    if is_fail :
                        return BAD_RETURN

        return 0, losses

    def get_pred_mask_fname(self, fname_orig, phase, epoch):
        """Return output name for one predicted foreground mask."""

        fname_base = fname_orig.replace('_orig', '')
        if fname_base.endswith('.nii.gz') :
            fname_base = fname_base[:-7]
        elif fname_base.endswith('.nii') :
            fname_base = fname_base[:-4]

        if len(phase) > 5 :
            ppp = phase[:5]
        else:
            ppp = phase

        fname_out = "{}_pmask_{}_{:04d}.nii.gz".format(
            fname_base, ppp, epoch)

        return os.path.join(self.outdir_mask, fname_out)

    def write_pred_masks(self, pred, batch, phase, epoch):
        """Write predicted foreground masks for one batch."""

        BAD_RETURN = -1

        for bb in range(pred.shape[0]) :
            fname_out = self.get_pred_mask_fname(
                batch['fname'][bb], phase, epoch)

            if self.verb :
                ab.IP("Writing out pred_mask to file:\n{}".format(fname_out))

            # make an appropriate type and convert to numpy array on cpu
            arr = (pred[bb][1]
                   .detach()
                   .to(device='cpu', dtype=torch.float32)
                   .numpy())

            # actually write out the file to disk
            try:
                lnwn.BabelNiftiWrite(
                    arr, batch['hdr_orig'][bb], fname_out,
                    map_rules    = "afni_rules",
                    do_overwrite = bool(self.overwrite),
                    do_rm_exts   = True,
                    verb         = self.verb )
            except Exception:
                ab.EP1("Failed to write pred_mask: {}".format(fname_out))
                return BAD_RETURN

        return 0

    def run_training(self):
        """Run all training and validation epochs."""

        BAD_RETURN = -1

        for epoch in range(self.num_epoch):

            if self.verb :
                ab.IP("Epoch {:04d} / {:04d}".format(epoch, self.max_epoch))

            # run training phase
            is_fail, train_losses = self.run_epoch(
                self.loader_train,
                phase = 'training',
                epoch = epoch,
            )
            if is_fail :
                ab.EP1("Training failed at epoch {}".format(epoch))
                return BAD_RETURN

            # save checkpoint after training phase, when requested
            if epoch in self.save_checkpoint_list :
                checkpoint_fname = os.path.join(
                    self.outdir,
                    "checkpoint_train_{:04d}.pt".format(epoch),
                )
                if self.verb :
                    ab.IP("Save checkpoint: {}".format(checkpoint_fname))
                try:
                    torch.save(self.net.state_dict(), checkpoint_fname)
                except Exception:
                    ab.EP1("Failed to save checkpoint: {}".format(
                        checkpoint_fname))
                    return BAD_RETURN

            # run validation phase
            is_fail, valid_losses = self.run_epoch(
                self.loader_valid,
                phase = 'validation',
                epoch = epoch,
            )
            if is_fail :
                ab.EP1("Validation failed at epoch {}".format(epoch))
                return BAD_RETURN

            train_mean = np.mean(train_losses)
            valid_mean = np.mean(valid_losses)

            if self.do_log_loss :
                is_fail = self.write_loss_log(epoch, 
                                              'training', 
                                              train_losses)
                if is_fail :
                    return BAD_RETURN
                is_fail = self.write_loss_log(epoch, 
                                              'validation', 
                                              valid_losses)
                if is_fail :
                    return BAD_RETURN

            if self.verb :
                ab.IP("Mean training loss   : {:.6f}".format(train_mean))
                ab.IP("Mean validation loss : {:.6f}".format(valid_mean))

        return 0

    def write_loss_log(self, epoch, phase, losses):
        """Append per-epoch loss statistics for one phase."""

        BAD_RETURN = -1

        arr = np.asarray(losses, dtype=float)
        vals1 = (arr.min(), arr.max())
        vals2 = (arr.mean(), arr.std())

        if phase == 'training' :
            fname = self.fname_log_loss_training
        elif phase == 'validation' :
            fname = self.fname_log_loss_validation
        else:
            ab.EP1("Unknown phase for loss log: {}".format(phase))
            return BAD_RETURN

        try:
            with open(fname, 'a') as fff:
                txt = "   {:04d} ".format(epoch)
                txt+= "{:10.6f} {:10.6f} ".format(*vals1)
                txt+= "{:10.6f} {:10.6f}\n".format(*vals2)
                fff.write(txt)
        except OSError:
            ab.EP1("Could not append loss log: {}".format(fname))
            return BAD_RETURN

        return 0

    def finish_cmd_log(self):
        """Append the command completion time, if its log exists."""

        if not(self.fname_log_cmd) :
            return 0

        self.time_finish   = datetime.now()
        self.time_duration = self.time_finish - self.time_start

        try:
            with open(self.fname_log_cmd, 'a') as fff:
                ttf = self.time_finish.isoformat(timespec='seconds')
                ttd = str(self.time_duration).split('.')[0] # no frac sec
                fff.write("\n# finished: {}\n".format(ttf))
                fff.write("# duration: {}\n".format(ttd))
        except OSError:
            ab.EP1("Could not finish command log: {}".format(
                self.fname_log_cmd))
            return -1

        return 0

    def make_workdir(self):
        """Make the workdir"""

        BAD_RETURN = -10

        cmd  = '\\mkdir -p "{}" '.format(self.workdir)
        com  = ab.shell_com(cmd, capture=1)
        stat = com.run()

        if stat :
            ab.EP1("Could not make workdir")
            return BAD_RETURN

        return 0

    def remove_workdir(self):
        """Remove the workdir"""

        BAD_RETURN = -10

        # see if we have a workdir to remove (if not, just return)
        if not(os.path.isdir(self.workdir)) :
            return 0

        cmd  = '\\rm -rf "{}" '.format(self.workdir)
        com  = ab.shell_com(cmd, capture=1)
        stat = com.run()

        if stat :
            ab.EP1("Could not remove workdir: {}".format(self.workdir))
            return BAD_RETURN

        return 0

    # ----- decorators

    @property
    def num_epoch(self):
        """the total number of epochs; also, the number to use at the
        top of loops over epochs, because of half-open intervals; that
        is, [0, self.max_epoch] is the same as [0, self.num_epoch)"""

        return self.max_epoch + 1

    @property
    def nsave_checkpoint(self):
        """the number of epochs for saving checkpoints"""

        return len(self.save_checkpoint_list)

    @property
    def nsave_mask(self):
        """the number of epochs for saving masks"""

        return len(self.save_mask_list)


# ----------------------------------------------------------------------------

def combine_list_rate_max(Alist, Brate, Cmax, label=None, verb=1):
    """Combine a list of integers Alist with another list generated from
the integers: rate Brate and maximum allowed number Cmax.  Return the
combined set of numbers in a sorted list that does not contain
repeats.  It is possible that Alist is empty and/or Brate is None.

Values outside of the interval [0, Cmax] are rejected.

For example, if these are inputs:
    Alist : [2, 12, 17]
    Brate : 5
    Cmax  : 20
then the final output will be:
    Dlist : [0, 2, 5, 10, 12, 15, 17, 20].   

Parameters
----------
Alist : list
    list of integers
Brate : int
    rate for generating a sequence of ints; must have Brate>0; if Brate is
    None, then only the list is checked
Cmax : int
    max number for the rate to end at (so last one can be at Cmax itself)
label : str
    str info for verb-y output, to label what the list is for
verb : int
    verbosity level

Returns
-------
is_fail : int
    0 for success, nonzero for failure
Dlist : list
    list of integers combining any non-null inputs

    """

    Dlist = []

    BAD_RETURN = (-1, [])

    if Brate is not None and Brate <= 0 :
        ab.EP1("Brate value must be >0, not: {}".format(Brate))
        return BAD_RETURN
    if Cmax < 0 :
        ab.EP1("Cmax value must be >=0, not: {}".format(Cmax))
        return BAD_RETURN

    if Brate is not None :
        for ii in range(0, Cmax+1, Brate):
            Dlist.append(ii)

    # combine lists
    Dlist.extend(list(Alist))
    # ... and remove any duplicates
    Dlist = list(set(Dlist))
    # ... and reject values outside of valid range
    tmp = []
    for x in Dlist :
        if x>=0 and x<=Cmax :
            tmp.append(x)
        else:
            msg = "Removing invalid 'save' index "
            if label :
                msg+= " for {}".format(label)
            msg+= ": {}".format(x)
            ab.WP(msg)
    Dlist = copy.deepcopy(tmp)  # deepcopy may be unnecessary here, but fine
    # ... and sort
    Dlist.sort()

    if verb :
        msg = "Int list"
        if label :
            msg+= " for {}".format(label)
        msg+= ": {}".format(', '.join([str(x) for x in Dlist]))
        ab.IP(msg)


    return 0, Dlist

def convert_list_of_str_to_int(L, verb=1):
    """L is a list of strings. Try converting each element to an int, and
return the resulting list. 

Parameters
----------
L : list
    list of strings (which should be converted to int)
verb : int
    verbosity level

Returns
-------
is_fail : int
    0 for success, nonzero for failure
M : list
    list of integers

    """

    M = []

    BAD_RETURN = (-1, [])

    if not isinstance(L, list):
        ab.EP1("input L must be a list, not: {}".format(au.simple_time(L)))
        return BAD_RETURN
    
    try: 
        M = [int(x) for x in L]
    except:
        ab.EP1("could not convert all of L to int: {}".format(L))
        return BAD_RETURN

    return 0, M

# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")

