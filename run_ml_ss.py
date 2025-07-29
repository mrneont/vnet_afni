#!/usr/bin/env python

import os
import sys
import datetime
import argparse      as argp
import lib_ml_train  as lmt
import lib_ml_losses as lml

# -----------------------------------------------------------------------
# 
__version__ = '1.0.00'; verdate = 'Jun 2, 2021'
# [PT] version number of code running
__version__ = '1.0.01'; verdate = 'Jul 25, 2025'
# update help parsing
__version__ = '1.0.02'; verdate = 'Jul 28, 2025'
# more option usage changes, merge some opts
__version__ = '1.0.03'; verdate = 'Jul 29, 2025'
# change half_prec and mixed_prec opts -> single '-precision' opt 

# -----------------------------------------------------------------------
# default opts

# Q: should do_write_nifti just be controlled by save_mask_*? why use
# a separate switch
DEF = {
        'input_dir'  : '',
        'output_dir' : '',
        'num_epochs' : 5,
        'learning_rate' : 0.0001,
        'seed' : 42,
        'verb' : 1,
        'train_batch_size' : 1,
        'architecture' : lmt.list_net_arch[0],
        'scale_mode' : lmt.list_scale_mode[0],
        'do_weight_norm' : 0,
        'loss_func' : lml.DEF_CalcLoss,
        'optimizer' : lmt.list_optimizer[0],
        'precision' : lmt.list_precision[0],
        'do_write_nifti' : 0,    
        'do_train_shuffle' : 1,  
        'restart_from_checkpoint' : '',
        'save_checkpoint_rate' : 0,
        'save_checkpoint_list' : [],
        'save_mask_rate' : 0,
        'save_mask_list' : [],
}



# -----------------------------------------------------------------------


# for expanding help information
epilog_data_struc = ''' 
------------------------------------------------------------------------

Input data structure
--------------------

The basic DATA_DIR directory structure looks like
this:

    DATA_DIR/
    ├── testing
    │   ├── mask
    │   └── orig
    ├── training
    │   ├── mask
    │   └── orig
    └── validation
        ├── mask
        └── orig

... where each pair of mask/ and orig/ subsubdirectories contains
files like SUB-001_mask.nii.gz and SUB-001_orig.nii.gz, respectively.
That is, within a given set like "testing", each file in "mask" must
have a partner in "orig".

Typical fractions of the total N subjects for each directory could be:

    training   : 70%
    validation : 20%
    testing    : 10%

'''

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(",")))

def get_args():
    """Use Python's arg-parser to get input values from the command line.  

NB: as soon as the 'nargs=..' kwarg is used for an option, then the
value saved within the argparser object is a list.  This is true even
if using 'nargs=1'.

    """

    # [PT] Using this formatter_class: ArgumentDefaultsHelpFormatter
    #      ... crushes newlines in the text.
    intro  = 'Train the VNet on MRI data and target masks'
    parser = argp.ArgumentParser(prog=str(sys.argv[0]).split('/')[-1],
                                 add_help=False,
                                 description = intro,
                                 epilog = epilog_data_struc,
                                 formatter_class=argp.RawTextHelpFormatter) 

    parser.add_argument('-ver', action='version', 
                        version='%(prog)s {}'.format(__version__))

    parser.add_argument("-input_dir", 
                        type=str,
                        dest='data_dir',
                        default=DEF['input_dir'],
                        help='(req) name of input dir (see below)' + '\n' +
                        '(def: {})'.format(DEF['input_dir']))

    parser.add_argument('-output_dir',
                        metavar='OD', 
                        dest='outdir',
                        type=str, default=DEF['output_dir'],
                        help='(req) name of output directory' + '\n' +
                        '(def: {})'.format(DEF['output_dir']))

    parser.add_argument('-num_epochs',
                        metavar='NUM_EPOCHS', 
                        dest='epochs',
                        type=int, default=DEF['num_epochs'],
                        help='Number of epochs, or iterations' + '\n' +
                        "(def: {})".format(DEF['num_epochs']))

    parser.add_argument("-learning_rate",
                        metavar='LRATE', 
                        dest="learning_rate", 
                        type=float, default=DEF['learning_rate'],
                        help='Learning rate' + '\n' +
                        '(def: {})'.format(DEF['learning_rate']))

    parser.add_argument("-seed",
                        dest='seed',
                        type=int, default=DEF['seed'],
                        help='seed value (int) for randomized steps' + '\n' +
                        '(def: {})'.format(DEF['seed']))

    parser.add_argument("-verb",
                        metavar='VERB', 
                        dest='verb', 
                        type=int, default=DEF['verb'],
                        help='verbosity for code running' + '\n' +
                        '(def: {})'.format(DEF['verb']))

    parser.add_argument("-train_batch_size", 
                        metavar='TBS', 
                        dest='train_batch_size', 
                        type=int, default=DEF['train_batch_size'], 
                        help='batch size for training' + '\n' +
                        '(def: {})'.format(DEF['train_batch_size']))

    parser.add_argument("-architecture", 
                        dest="net_arch", 
                        type=str, default=DEF['architecture'],
                        help="network architecture type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_net_arch)) + '\n' +
                        '(def: {})'.format(DEF['architecture']))

    parser.add_argument("-precision", 
                        dest="precision", 
                        type=str, default=DEF['precision'],
                        help="data type precision type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_precision)) + '\n' +
                        '(def: {})'.format(DEF['precision']))

    parser.add_argument("-scale_mode", 
                        dest="scale_mode", 
                        type=str, default=DEF['scale_mode'],
                        help="data normalization type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_scale_mode)) + '\n' +
                        '(def: {})'.format(DEF['scale_mode']))

    # does not seem so useful
    parser.add_argument("-do_weight_norm", 
                        dest="weight_norm", 
                        type=int, default=DEF['do_weight_norm'],
                        help='do weight normalization' + '\n' +
                        '(def: {})'.format(DEF['do_weight_norm']))

    # very useful to have on always, for stability
    parser.add_argument("-do_train_shuffle", 
                        dest="tr_shuf", 
                        type=int, default=DEF['do_train_shuffle'],
                        help='shuffle datasets during training' + '\n' +
                        '(def: {})'.format(DEF['do_train_shuffle']))

    parser.add_argument("-restart_from_checkpoint", 
                        dest="restart", 
                        type=str, default=DEF['restart_from_checkpoint'],
                        help='restart using specified checkpoint' + '\n' +
                        '(def: {})'.format(DEF['restart_from_checkpoint']))

    parser.add_argument("-loss_func", 
                        dest="loss_func", 
                        type=str, default=DEF['loss_func'],
                        help="loss function type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lml.list_CalcLoss)) + '\n' +
                        '(def: {})'.format(DEF['loss_func']))

    parser.add_argument("-optimizer", 
                        dest="optimizer", 
                        type=str, default=DEF['optimizer'],
                        help="optimizer type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_optimizer)) + '\n' +
                        '(def: {})'.format(DEF['optimizer']))

    parser.add_argument("-do_write_nifti", 
                        dest='do_nifti',
                        type=int, default=DEF['do_write_nifti'],
                        help='flag to turn on the writing of NIFTI' + '\n' +
                        "datasets while processing" + '\n' +
                        '(def: {})'.format(DEF['do_write_nifti']))

    parser.add_argument("-save_checkpoint_rate", 
                        metavar='CRATE', 
                        dest="save_chpt_rate", 
                        type=int, default=DEF['save_checkpoint_rate'],
                        help='write a checkpoint*.pt every CRATE epochs' + 
                        '\n' +
                        '(def: {})'.format(DEF['save_checkpoint_rate']))

    parser.add_argument("-save_checkpoint_list", nargs='+',
                        metavar='CLIST', 
                        dest="save_chpt_list",
                        type=int, default=DEF['save_checkpoint_list'],
                        help='write a checkpoint*.pt at each listed epoch' +
                        '\n' +
                        '(def: {})'.format(DEF['save_checkpoint_list']))

    parser.add_argument("-save_mask_rate", 
                        metavar='MRATE', 
                        dest="save_mask_rate", 
                        type=int, default=DEF['save_mask_rate'],
                        help='write mask results every MRATE epochs' + 
                        '\n' +
                        '(def: {})'.format(DEF['save_mask_rate']))

    parser.add_argument("-save_mask_list", nargs='+',
                        metavar='MLIST', 
                        dest="save_mask_list",
                        type=int, default=DEF['save_mask_list'],
                        help='write mask results at each listed epoch' +
                        '\n' +
                        '(def: {})'.format(DEF['save_mask_list']))

    parser.add_argument('-help', action="store_true", 
                        default=False,
                        help='display help in terminal') 

    parser.add_argument('-hview', action="store_true", 
                        default=False,
                        help='display help in a text editor') 

    args = parser.parse_args()

    # display program version (later, when an AFNI program, do_view differs)
    do_help  = args.help
    do_hview = args.hview
    if len(sys.argv) == 1 or do_help or do_hview :
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()

def prep_outdir(din, verb=1):
    """Check+prepare the outdir entered by the user.  At the moment, we
    allow outdir to exist already, and contents from run_ml_ss.py will
    be simply and brutally overwritten.

    Any final '/' will be stripped, because it would be attached later
    by path-joining commands

    Parameters
    ----------

    din       : Name of output directory entered by the user.  Can be
                absolute or relative path.

    Returns
    -------

    dout      : String obj for actual path to be used.

    """

    # check for any input
    if not(din):
        return ""

    # remove any '/' at end, or just copy
    dout = din.rstrip('/')

    # report if dir exists already; just reporting at the moment
    does_exist = os.path.isdir(dout)
    path_isabs = os.path.isabs(dout)

    # check/make outdir
    if does_exist :
        if verb :
            print("++ Path of outdir ({}) exists already".format(dout)) 
    else:
        if verb :
            print("++ Path of outdir ({}) does NOT already exist".format(dout)) 
            print("   Will make it now.") 
        os.mkdir(dout)

    # report abs path (might prefer using this later, if program hops
    # around to different places, but not for now)
    if verb :
        abspath = os.path.abspath(dout)
        print("++ Absolute path of outdir will be: {}".format(abspath))

    return dout

def writeout_args(argv, outdir, ofile='log_cmd.txt', ver='0.0.0',
                  state=None, verb=1):
    """Store the command used to make this run.  Simple/no formatting at
    the moment

    Parameters
    ----------
    
    argv       : the list of terminal commands used.
    outdir     : output directory where all outputs (including what is 
                 written here) will go.
    ver        : version of code (str).
    ofile      : name of file to be written in the outdir.

    Returns nothing, just writes a text file.

    """

    otxt  = '/'.join([outdir, ofile])
    ocmd  = ' '.join(argv)
    opwd  = os.getcwd()
    
    dt    = datetime.datetime.now()
    odate = dt.strftime("%Y/%m/%d")
    otime = dt.strftime("%H:%M:%S")

    fff  = open(otxt, mode='w')

    fff.write("{:15s} : {:15s}\n".format("Run cmd", ocmd))
    fff.write("\n")
    fff.write("{:15s} : {:15s}\n".format("Prog ver", str(ver)))
    fff.write("{:15s} : {:15s}\n".format("Output dir", opwd))
    fff.write("{:15s} : {:15s}\n".format("Run date", odate))
    fff.write("{:15s} : {:15s}\n".format("Run start time", otime))
    fff.write("\n")

    if state :
        fff.write("{}".format(state))

    fff.close()

    if verb :
        print("++ Store executed command in text file: {}".format(otxt))


def check_opt_allowed(X, the_list, desc_bad=None): 
    '''See if element X is contained in the_list; the desc_bad is the
    description output if it ain't.

    '''

    is_ok = the_list.__contains__(X)

    if not(is_ok) :
        print("** ERROR: {} {}\n".format(desc_bad, X))

    return is_ok

def create_valid_epoch_list(N, list_idx, rate_idx, add_final_idx=False):
    """When there are N epochs (so, epoch indices is in [0, N-1]), use
either a user-provided list of indices list_idx or a rate of indices
rate_idx, to generate a list of epoch indices at which something will
happen.

Note: the user can only provide either a list_idx or a rate_idx, not both.

If the list_idx contains invalid indices, which are here defined as
i<=-N or i>=N, then we exit with error.

We just quietly ignore any repeated index in list_idx.

If rate_idx<0 or rate_idx>N, then we exit with error.

If add_final_idx is True, then in all cases where N>=1, the final
epoch index N-1 will be included. If add_final_idx is False, then that
will not necessarily be added. Probably add_final_idx should be True
when creating a checkpoint list, and False for a mask list (because
there might be a lot of masks...).

Parameters
----------
N : int
    number of epochs, so indices are in range [0, N-1]
list_idx : list of int
    list of integer indices for writing
rate_idx : int
    rate at which specify epochs to do something at (must be >0)
add_final_idx : bool
    switch to ensure that index N-1 is always present in the output list.

Returns
-------
all_idx : list of idx
    list of all indices

    """

    all_idx = []

    # don't think this should happen, but here we go
    if not(N):
        print("+* WARN: Number of epochs is 0")
        return []
    
    if rate_idx < 0 :
        print("** ERROR: cannot have negative rate_idx :", rate_idx)
        sys.exit(7)
    elif rate_idx >= N :
        print("** ERROR: rate_idx={} is greater than num_epoch={}"
              "".format(rate_idx, N))
        sys.exit(7)

    # can't have 2 methods of creating list used
    if len(list_idx) and rate_idx :
        print("** ERROR: user provided both a list_idx and rate_idx.")
        print("   That is not allowed---only provide one of those.")
        sys.exit(7)

    elif len(list_idx) :
        # in theory, the user-provided list could be out of order
        for lll in list_idx :
            if lll >= 0 and lll < N :
                if not(lll in all_idx ) :
                    all_idx.append(lll)
            elif lll > -N and lll < 0 :
                mmm = N + lll
                if not(mmm in all_idx ) :
                    all_idx.append(mmm)
            else:
                print("** ERROR: index '{}' out of range".format(lll))
                print("   -N < i < N, where N = {}".format(N))
                sys.exit(8)

    elif rate_idx :
        all_idx = [x for x in range(rate_idx-1, N, rate_idx)]

    # in all cases, we include the final epoch in the list, which is N-1
    if not(N-1 in all_idx) and add_final_idx :
        all_idx.append(N-1)

    return all_idx



def get_args_state(args):
    '''
    Get a nice string for outputting the state of the variables
    '''

    ostr = ''

    for key in vars(args):
        ostr+= "{:15s} : {:15s}\n".format(str(key), str(vars(args)[key]))

    return ostr



if __name__ == '__main__':

    # get args (path and parameter settings) , and prepare to pass
    # along to the main training net prog
    args          = get_args()
    data_path     = args.data_dir
    num_epochs    = int(args.epochs)
    lr            = float(args.learning_rate)
    tr_bsize      = args.train_batch_size
    seed          = int(args.seed)
    net_arch      = args.net_arch
    loss_func     = args.loss_func
    optimizer     = args.optimizer
    verb          = int(args.verb)
    precision     = args.precision
    wt_norm       = int(args.weight_norm)
    tr_shuf       = args.tr_shuf
    do_nifti      = args.do_nifti
    scale_mode    = args.scale_mode
    restart       = args.restart  

    # these will be parsed/checked/used below
    save_chpt_rate = int(args.save_chpt_rate)
    save_chpt_list = args.save_chpt_list
    save_mask_rate = int(args.save_mask_rate)
    save_mask_list = args.save_mask_list

    # check input dir
    if not(os.path.isdir(data_path)) :
        print("** ERROR: '-input_dir ..' is not valid: {}".format(data_path))
        sys.exit(5)

    # check+format output dir
    outdir = prep_outdir(args.outdir, verb=verb)
    if not(outdir) :
        print("** ERROR: this path is not valid: {}".format(args.outdir))
        sys.exit(5)

    if  not(check_opt_allowed( net_arch, lmt.list_net_arch, 
                               desc_bad='This network architecture ' + 
                               'is not in the List:' )) or \
        not(check_opt_allowed( optimizer, lmt.list_optimizer, 
                               desc_bad='This optimizer ' + 
                               'is not in the List:' )) or \
        not(check_opt_allowed( loss_func, lml.list_CalcLoss,
                               desc_bad='This loss function ' + 
                               'is not in the List:' )) or \
        not(check_opt_allowed( scale_mode, lmt.list_scale_mode,
                               desc_bad='This data normalization ' + 
                               'is not in the List:' ))  or \
        not(check_opt_allowed( precision, lmt.list_precision,
                               desc_bad='This precision ' + 
                               'is not in the List:' ))  :
        sys.exit(5)

    # ----- check more things

    if len(restart) :
        if not(os.path.isfile(restart)) :
            print("** ERROR: specified restart file does not exist:", restart)
            sys.exit(3)

    # create list of epoch indices at which to write a checkpoint 
    epoch_chpt_list = create_valid_epoch_list(num_epochs, 
                                              save_chpt_list,
                                              save_chpt_rate,
                                              add_final_idx = True)
    # create list of epoch indices at which to write predicted masks 
    epoch_mask_list = create_valid_epoch_list(num_epochs, 
                                              save_mask_list,
                                              save_mask_rate,
                                              add_final_idx = False)

    if verb :
        print("++ Calculating {} epochs, indices: 0 to {}".format(num_epochs,
                                                                  num_epochs-1))
        print("++ The (0-based) list of epoch indices to write a checkpoint:")
        print("   " + ', '.join([str(x) for x in epoch_chpt_list]))
        print("++ The (0-based) list of epoch indices to write prediced masks:")
        print("   " + ', '.join([str(x) for x in epoch_mask_list]))

    if len(epoch_mask_list) and not(do_nifti) :
        print("+* WARN: you asked for masks at epoch intervals, ")
        print("   but -do_write_nifti is 0, so won't do it, oddly.")

    # save command used
    str_args = get_args_state(args)
    writeout_args( sys.argv, outdir, ver=__version__, state=str_args,
                   verb=verb )

    # inserting new ussage of mask_oplist and chpt_oplist replacements
    net = lmt.train_net( data_path, num_epochs, lr, tr_bsize, seed, net_arch, 
                         loss_func, optimizer, precision, wt_norm, 
                         tr_shuf, restart, scale_mode, do_nifti, 
                         outdir, epoch_mask_list, epoch_chpt_list, 
                         verb )

    # done successfully
    sys.exit(0)
