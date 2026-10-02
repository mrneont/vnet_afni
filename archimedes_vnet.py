#!/usr/bin/env python

# python3 status: compatible

# system libraries
import sys, os

# AFNI libraries
from   afnipy import option_list   as OL
from   afnipy import afni_util     as UTIL
from   afnipy import afni_base     as BASE

from   afnipy import lib_arch_vnet_defs as LAVD
from   afnipy import lib_arch_vnet_run  as LAVR

# ----------------------------------------------------------------------
# globals

# combine all entries for help here
g_help_dict   = {
    **LAVD.DOPTS, 
    'STR_architecture' : LAVD.STR_architecture,
    'STR_optimizer'    : LAVD.STR_optimizer,
    'STR_scale_mode'   : LAVD.STR_scale_mode,
    'STR_precision'    : LAVD.STR_precision,
    'STR_loss_func'    : LAVD.STR_loss_func,
    'STR_device'       : LAVD.STR_device,
}

g_help_string = """Overview ~1~

This program is for running a VNET  the start of a python program

auth = Y Narayana Swamy (SSCC, NIMH, NIH, USA)
       RC Reynolds (SSCC, NIMH, NIH, USA)
       PA Taylor (SSCC, NIMH, NIH, USA)

------------------------------------------------------------------------

Usage ~1~

-indir INDIR   :(req) name of the input directory of training data

-outdir OUTDIR :(req) name of the output directory of training data

-num_epoch NE  :number of epochs to use for training; these start counting 
                at 1 
                (def: {num_epoch})

-learn_rate LR :learning rate during training, which is the step size while
                optimizing; smaller values can improve stability, but 
                also take more time/computational resources                
                (def: {learn_rate})

-batch_size BS :number of dsets to batch together during training; increasing
                this should boost robustness, but uses/requires more memory
                (def: {batch_size})

-do_shuffle DS :if using a batch_size >1, can choose to shuffle during which
                dsets are batched together, which is probably a good idea
                (def: {do_shuffle})

-do_weight_norm DWN 
               :use weight normalization during training?
                (def: {do_weight_norm})

-loss_func LF  :choose a particular loss function for training, from among
                this list:
                  {STR_loss_func}
                (def: {loss_func})

-device D      :choose a particular device to run the training on, from among
                this list:
                  {STR_device}
                (def: {device})

-architecture AR :choose a particular architecture for training, from among
                this list:
                  {STR_architecture}
                (def: {architecture})

-optimizer OP  :choose a particular optimizer for training, from among
                this list:
                  {STR_optimizer}
                (def: {optimizer})

-precision PR  :choose a particular precision for training, from among
                this list:
                  {STR_precision}
                (def: {precision})

-seed S        :set a particular seed for any randomization steps
                (def: {seed})

-num_cpu NCPU  :number of CPU threads to use during skullstripping; 
                a negative value means that the system decides.
                NB: to know how many CPUs are available, you can run:
                     afni_system_check.py -disp_num_cpu
                (def: {num_cpu})

-restart_checkpoint RC 
               :to start training from an already-created checkpoint,
                provide its name here
                (def: {do_weight_norm})

-save_checkpoint_rate SCR :write out a checkpoint at a regular interval;
                if using this option, a checkpoint will be written for the
                [0]th epoch and then every SCR-th one up to num_epochs. For
                example, if num_epochs is 20 and SCR is 5, then checkpoints
                will be written out at these epochs: 0, 5, 10, 15, 20.
                (def: only save the final checkpoint)

-save_checkpoint_list SCL 
               :write out a checkpoint at specified epochs; users
                can provide a list of one or more integers in the range:
                [0, num_epoch].
                (def: only save the final checkpoint)

       NB: the above -save_checkpoint_* options can both be used; the result
           is to create a list of their union and num_epoch (without repeats).

-save_mask_rate SMR 
               :write out estimated masks at a regular interval;
                if using this option, masks will be written for the
                [0]th epoch and then every SCR-th one up to
                num_epochs. For example, if num_epochs is 20 and SCR
                is 5, then maskss will be written out at these epochs:
                0, 5, 10, 15, 20.  
                (def: no masks written out)

-save_mask_list SML 
               :write out estimated masks at specified epochs; users
                can provide a list of one or more integers in the range:
                [0, num_epoch].
                (def: no masks written out)

       NB: the above -save_masks_* options can both be used; the result
           is to create a list of their union (without repeats).


****

-workdir WD    :working directory name, without path; the working dir
                will be subdirectory of the output location
                (def: name with random chars) ***STILL KEEP??***

-do_clean DC   :state whether to clean up any intermediate files;
                allowed values are:  Yes, 1, No, 0
                (def: '{do_clean}')

-do_log        :add this opt to turn on making a text log of all the
                shell commands that are run when this program is
                executed.  Mainly for debugging purposes.

-help, -h      :display program help file

-hist          :display program history

-ver           :display program version number

-verb  VVV     :control verbosity (def: {verb})

-show_valid_opts :show valid options for this program

------------------------------------------------------------------------

Notes ~1~

***

------------------------------------------------------------------------

Examples ~1~

 ****

           

""".format(**g_help_dict)

g_history = """
  archimedes_vent.py history:

  0.1   Sep 28, 2026 :: started this command line interface 
"""

g_prog    = g_history.split()[0]
g_ver     = g_history.split("\n")[-2].split("::")[0].strip()
g_version = g_prog + " version " + g_ver

class InOpts:
    """Object for storing any/all command line inputs, and just checking
that any input files do, in fact, exist.  Option parsing and other
checks happen in a subsequent object.

    """

    def __init__(self):
        # main variables
        self.status          = 0                       # exit value
        self.valid_opts      = None
        self.user_opts       = None

        # general variables
        self.verb            = LAVD.DOPTS['verb']
        self.do_clean        = None
        self.overwrite       = None
        self.do_log          = None

        # main data variables
        self.indir           = None
        self.outdir          = None
        self.workdir         = None

        # control variables
        self.architecture    = None
        self.scale_mode      = None
        self.optimizer       = None
        self.precision       = None
        self.num_epoch       = None
        self.learn_rate      = None
        self.loss_func       = None
        self.device          = None
        self.seed            = None
        self.num_cpu         = None
        self.batch_size      = None
        self.do_weight_norm  = None
        self.do_train_shuffle     = None
        self.restart_checkpoint   = None
        self.save_checkpoint_rate = None
        self.save_checkpoint_list = None
        self.save_mask_rate  = None
        self.save_mask_list  = None


        # ----- take action(s)

        # prelim stuff
        tmp1 = self.init_options()

    # ----- methods

    def init_options(self):
        """
        Prepare the set of all options, with very short help descriptions
        for each.
        """

        self.valid_opts = OL.OptionList('valid opts')

        # short, terminal arguments

        self.valid_opts.add_opt('-help', 0, [],           \
                        helpstr='display program help')
        self.valid_opts.add_opt('-hist', 0, [],           \
                        helpstr='display the modification history')
        self.valid_opts.add_opt('-show_valid_opts', 0, [],\
                        helpstr='display all valid options')
        self.valid_opts.add_opt('-ver', 0, [],            \
                        helpstr='display the current version number')

        # required parameters

        self.valid_opts.add_opt('-indir', 1, [], 
                        helpstr='name of input directory')

        self.valid_opts.add_opt('-outdir', 1, [], 
                        helpstr='name of output directory')

        # optional parameters

        self.valid_opts.add_opt('-workdir', 1, [], 
                        helpstr='name of workdir (no path)')

        self.valid_opts.add_opt('-num_epoch', 1, [], 
                        helpstr='number of epochs (iterations) for training')

        self.valid_opts.add_opt('-learn_rate', 1, [], 
                        helpstr='learning rate for training')

        self.valid_opts.add_opt('-seed', 1, [], 
                        helpstr='seed value (int) for randomization steps')

        self.valid_opts.add_opt('-num_cpu', 1, [], 
                        helpstr='specify number of CPUs to use')

        self.valid_opts.add_opt('-batch_size', 1, [], 
                        helpstr='batch size (int) for training')

        self.valid_opts.add_opt('-do_weight_norm', 1, [], 
                        helpstr='do weight normalization (or not)')

        self.valid_opts.add_opt('-do_shuffle', 1, [], 
                        helpstr='if batch size >1, shuffle dsets in training')

        self.valid_opts.add_opt('-restart_checkpoint', 1, [], 
                        helpstr='restart training from specified checkpoint')

        self.valid_opts.add_opt('-save_checkpoint_rate', 1, [], 
                        helpstr='write a checkpoint every n-th epoch')

        self.valid_opts.add_opt('-save_checkpoint_list', 1, [], 
                        helpstr='write a checkpoint at every listed epoch')

        self.valid_opts.add_opt('-save_mask_rate', 1, [], 
                        helpstr='write masks every n-th epoch')

        self.valid_opts.add_opt('-save_mask_list', 1, [], 
                        helpstr='write masks at every listed epoch')

        self.valid_opts.add_opt('-architecture', 1, [], 
                        helpstr='network architecture type for training')

        self.valid_opts.add_opt('-scale_mode', 1, [], 
                        helpstr='data normalization type for training')

        self.valid_opts.add_opt('-optimizer', 1, [], 
                        helpstr='name of optimizer to use during training')

        self.valid_opts.add_opt('-precision', 1, [], 
                        helpstr='type of dset precision to use for training')

        self.valid_opts.add_opt('-loss_func', 1, [], 
                        helpstr='type of loss function to use for training')

        self.valid_opts.add_opt('-device', 1, [], 
                        helpstr='device to use (cpu, cuda, etc.)')

        # general options

        self.valid_opts.add_opt('-do_clean', 1, [], 
                        helpstr="turn on/off removal of intermediate files")

        self.valid_opts.add_opt('-do_log', 0, [], 
                        helpstr="turn on/off logging shell cmd execution")

        self.valid_opts.add_opt('-overwrite', 0, [], 
                        helpstr='overwrite preexisting outputs')

        self.valid_opts.add_opt('-verb', 1, [], 
                        helpstr='set the verbose level (default is 0)')

        return 0

    def process_options(self):
        """return  1 on valid and exit        (e.g. -help)
           return  0 on valid and continue    (e.g. do main processing)
           return -1 on invalid               (bad things, panic, abort)
        """

        # process any optlist_ options
        self.valid_opts.check_special_opts(sys.argv)

        # process terminal options without the option_list interface
        # (so that errors are not reported)
        # return 1 (valid, but terminal)

        # if no arguments are given, apply -help
        if len(sys.argv) <= 1 or '-help' in sys.argv:
           print(g_help_string)
           return 1

        if '-hist' in sys.argv:
           print(g_history)
           return 1

        if '-show_valid_opts' in sys.argv:
           self.valid_opts.show('', 1)
           return 1

        if '-ver' in sys.argv:
           print(g_version)
           return 1

        # ============================================================
        # read options specified by the user
        self.user_opts = OL.read_options(sys.argv, self.valid_opts)
        uopts = self.user_opts            # convenience variable
        if not uopts: return -1           # error condition

        # ------------------------------------------------------------
        # process non-chronological options, verb comes first

        val, err = uopts.get_type_opt(int, '-verb')
        if val != None and not err: self.verb = val

        # ------------------------------------------------------------
        # process options sequentially, to make them like a script

        err_base = "Problem interpreting use of opt: "

        for opt in uopts.olist:

            # main options

            if opt.name == '-indir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.indir = val

            if opt.name == '-outdir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.outdir = val

            elif opt.name == '-workdir':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.workdir = val

            # control options

            elif opt.name == '-num_epoch':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.num_epoch = val

            elif opt.name == '-learn_rate':
                val, err = uopts.get_type_opt(float, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.learn_rate = val

            elif opt.name == '-seed':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.seed = val

            elif opt.name == '-num_cpu':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.num_cpu = val

            elif opt.name == '-batch_size':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.batch_size = val

            elif opt.name == '-do_weight_norm':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.do_weight_norm = val

            elif opt.name == '-do_shuffle':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.do_shuffle = val

            elif opt.name == '-restart_checkpoint':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.restart_checkpoint = val

            elif opt.name == '-save_checkpoint_rate':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.save_checkpoint_rate = val

            # list (of int)
            elif opt.name == '-save_checkpoint_list':
                val, err = uopts.get_string_list('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.save_checkpoint_list = val

            elif opt.name == '-save_mask_rate':
                val, err = uopts.get_type_opt(int, '', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.save_mask_rate = val

            # list (of int)
            elif opt.name == '-save_mask_list':
                val, err = uopts.get_string_list('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.save_mask_list = val

            elif opt.name == '-architecture':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.architecture = val

            elif opt.name == '-scale_mode':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.scale_mode = val

            elif opt.name == '-optimizer':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.optimizer = val

            elif opt.name == '-precision':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.precision = val

            elif opt.name == '-loss_func':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.loss_func = val

            elif opt.name == '-device':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.device = val

            # general options

            elif opt.name == '-do_clean':
                val, err = uopts.get_string_opt('', opt=opt)
                if val is None or err:
                    BASE.EP(err_base + opt.name)
                self.do_clean = val

            elif opt.name == '-do_log':
                self.do_log = True

            elif opt.name == '-overwrite':
                self.overwrite = '-overwrite'

            # ... verb has already been checked above

        return 0

    def check_options(self):
        """perform any final tests before execution; most checks are done
        in the main object"""

        if self.verb > 1:
            BASE.IP("Begin processing options")

        # required opt
        if self.indir is None :
            BASE.EP1("missing -indir option")
            return -1

        if self.outdir is None:
            BASE.EP1("missing -outdir option")
            return -1

        return 0

    def test(self, verb=3):
        """one might want to be able to run internal tests,
           alternatively, test from the shell
        """
        print('------------------------ initial tests -----------------------')
        self.verb = verb
        
        print('------------------------ reset files -----------------------')

        print('------------------------ should fail -----------------------')

        print('------------------------ more tests ------------------------')

        return None

    # ----- decorators

    @property
    def ninset(self):
        """number of insets"""
        return len(self.inset)


# ----------------------------------------------------------------------------

def main():

    # init option-reading obj
    inobj = InOpts()
    if not(inobj) :  
        return 1, None

    # process (= read) options
    rv = inobj.process_options()
    if rv > 0: 
        # exit with success (e.g. -help)
        return 0, None
    if rv < 0:
        # exit with error status
        BASE.EP1('failed to process options')
        return 1, None

    # check the options
    rv2 = inobj.check_options()
    if rv2 :
        # exit with error status
        BASE.EP1('failed whilst checking options')
        return rv2, None

    # use options to create main object
    mainobj = LAVR.MainObj( user_inobj=inobj )
    if not mainobj :  
        return 1

    # write out log/history of what has been done (not done by default, to
    # save some time, bc this takes a mini-while)
    if inobj.do_log :
        olog = 'log_archimedes_vnet.txt'
        BASE.IP('creating log: {}'.format(olog))
        UTIL.write_afni_com_log(olog)

    return 0, mainobj

# ============================================================================

if __name__ == '__main__':

    stat, mainobj = main()
    sys.exit(stat)


