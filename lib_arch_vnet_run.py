#!/usr/bin/env python

# A library of functions for dealing with archimedes_vnet.py
#
# auth : PA Taylor (SSCC, NIMH, NIH, USA)
# ----------------------------------------------------------------------------
# ver 1.0 : start mapping original vnet funcs to AFNI format/objs
# ============================================================================

import sys, os, copy, glob
import platform

from   afnipy import afni_base          as ab
from   afnipy import afni_util          as au
from   afnipy import lib_torch_util     as ltu

from   afnipy import lib_arch_vnet_defs as DEF

# ============================================================================

# ----------------------------------------------------------------------------


# ============================================================================

class MainObj:
    """Object for setting up archimedes_vnet.py, to carry out training.

Parameters
----------
inobj : InOpts object 
    object constructed from running TEMPLATE_tool.py on the command
    line. At present, the only way to really provide inputs here.

    """

    def __init__(self, user_inobj=None):

        # ----- set up attributes

        # main input variables
        self.status          = 0                        # not used
        self.user_opts       = DEF.DOPTS['user_opts']  # command the user ran
        self.user_inobj      = user_inobj

        # general variables
        self.verb            = DEF.DOPTS['verb']
        self.overwrite       = DEF.DOPTS['overwrite']
        self.do_clean        = DEF.DOPTS['do_clean']
        self.do_log          = DEF.DOPTS['do_log']

        # main data variables
        self.indir           = DEF.DOPTS['indir']
        self.outdir          = DEF.DOPTS['outdir']     # None or str
        self.workdir         = DEF.DOPTS['workdir']

        # control variables
        self.num_epoch       = DEF.DOPTS['num_epoch']
        self.learn_rate      = DEF.DOPTS['learn_rate']
        self.architecture    = DEF.DOPTS['architecture']
        self.optimizer       = DEF.DOPTS['optimizer']
        self.scale_mode      = DEF.DOPTS['scale_mode']
        self.architecture    = DEF.DOPTS['architecture']
        self.precision       = DEF.DOPTS['precision']
        self.loss_func       = DEF.DOPTS['loss_func']
        self.device          = DEF.DOPTS['device']
        self.seed            = DEF.DOPTS['seed']
        self.num_cpu         = DEF.DOPTS['num_cpu']
        self.batch_size      = DEF.DOPTS['batch_size']
        self.do_weight_norm  = DEF.DOPTS['do_weight_norm']
        self.do_shuffle      = DEF.DOPTS['do_shuffle']
        self.restart_checkpoint   = DEF.DOPTS['restart_checkpoint']
        self.save_checkpoint_rate = DEF.DOPTS['save_checkpoint_rate']
        self.save_checkpoint_list = DEF.DOPTS['save_checkpoint_list']
        self.save_mask_rate  = DEF.DOPTS['save_mask_rate']
        self.save_mask_list  = DEF.DOPTS['save_mask_list']

        self.sysname         = None           # platform system

        # ----- take action(s)

        # prelim stuff
        if user_inobj :
            tmp = self.load_from_inopts()
            if tmp : return

            tmp = self.basic_setup()
            if tmp : return

            ##### **** ADD THIS WITHIN LTU, and update
            ##### **** lib_vnet_test.py similarly
            ###tmp = self.set_cpus()
            ###if tmp : return

            tmp = self.make_workdir()
            if tmp : return

            # ****

            if self.do_clean :
                tmp10 = self.remove_workdir()
                if tmp10 : return

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
        if io.num_epoch is not None :
            self.num_epoch = io.num_epoch
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
        if io.do_train_shuffle is not None :
            self.do_train_shuffle = io.do_train_shuffle
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
                ab.EP("The outdir {} exists already".format(self.outdir))

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

        if self.loss_func : 
            if self.loss_func not in DEF.LIST_loss_func :
                msg = "Invalid value after -loss_func: "
                msg+= "{}\n".format(self.loss_func)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_loss_func)
                ab.EP(msg)

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

        if self.num_epoch <= 0 :
            msg = "Invalid value after -num_epoch: "
            msg+= "{}\n".format(self.num_epoch)
            msg+= "Cannot be <= 0"
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
                msg+= "{}\n".format(self.save_checkpoint_rate)
                msg+= "Could not convert all to int"
                ab.EP(msg)

        # if using save_checkpoint_rate, have to combine with list
        # (default or user-entered)
        if self.save_checkpoint_rate is not None :
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

        # finalize checkpoint list: always include self.num_epochs
        if self.num_epoch not in self.save_checkpoint_list :
            self.save_checkpoint_list.append(self.num_epoch)

        if self.save_mask_rate is not None :
            if self.save_mask_rate <= 0 :
                msg = "Invalid value after -save_mask_rate: "
                msg+= "{}\n".format(self.save_mask_rate)
                msg+= "Cannot be <= 0"
                ab.EP(msg)

        # if using save_mask_rate, have to combine with list
        # (default or user-entered)
        if self.save_mask_rate is not None :
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

        # setup device (may replace user-entered parameter
        if self.device : 
            if self.device not in DEF.LIST_device :
                msg = "Invalid value after -device: "
                msg+= "{}\n".format(self.device)
                msg+= "Please selection from this list:\n"
                msg+= "{}".format(DEF.STR_device)
                ab.EP(msg)
        is_fail, self.device = ltu.select_device_general(dev_in=self.device,
                                                         verb=self.verb)
        if is_fail :
            ab.EP1("Failed select device")
            return BAD_RETURN

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
        self.do_shuffle     = au.convert_to_bool_yn10(self.do_shuffle)
        self.do_clean       = au.convert_to_bool_yn10(self.do_clean)
        self.do_log         = au.convert_to_bool_yn10(self.do_log)

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
    def max_epoch(self):
        """the number to use at the top of loops over epochs, because of half
        open intervals; that is, [0, self.num_epoch] is the same as
        [0, self.max_epoch)"""

        return self.num_epoch + 1

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
    rate for generating a sequence of ints; must have Brate>0
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

    if Brate <= 0 :
        ab.EP1("Brate value must be >0, not: {}".format(Brate))
        return BAD_RETURN
    if Cmax <= 0 :
        ab.EP1("Cmax value must be >0, not: {}".format(Cmax))
        return BAD_RETURN

    for ii in range(0, Cmax+1, Brate):
        Dlist.append(ii)

    # combine lists
    Dlist.extend(list(Alist))
    # ... and remove any duplicates
    Dlist = list(set(Dlist))
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

