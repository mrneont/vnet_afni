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
#
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

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_args():

    # [PT] Using this formatter_class: ArgumentDefaultsHelpFormatter
    #      ... crushes newlines in the text.
    intro  = 'Train the VNet on MRI data and target masks'
    parser = argp.ArgumentParser(prog = 'run_ml_ss.py',
                                 description = intro,
                                 epilog = epilog_data_struc,
                                 formatter_class=argp.RawTextHelpFormatter) 

    parser.add_argument('-V', '--version', action='version', 
                        version='%(prog)s {}'.format(__version__))

    parser.add_argument('-d', "--data_dir", 
                        type=dir_path, 
                        help="Path to the data directory (structure below)")

    def_E = 5 
    parser.add_argument('-e', '--epochs', 
                        metavar='E', 
                        dest='epochs',
                        type=int, default=def_E,
                        help='Number of epochs' + '\n' +
                        "(def: {})".format(str(def_E)))

    def_LRATE = 0.0001
    parser.add_argument("-l", "--learning_rate", 
                        metavar='LRATE', 
                        dest="learning_rate", 
                        type=float, default=def_LRATE, 
                        help='Learning rate' + '\n' +
                        '(def: {})'.format(str(def_LRATE)))

    def_S = None
    parser.add_argument('-s', '--seed', 
                        metavar='S', 
                        dest='seed',
                        type=int, default=def_S,
                        help='Set seed for random value gen' + '\n' +
                        '(def: {})'.format(str(def_S)))

    def_OD = '.'
    parser.add_argument('-o', '--outdir', 
                        metavar='OD', 
                        dest='outdir',
                        type=str, default=def_OD,
                        help='Set name of output directory' + '\n' +
                        '(def: {})'.format(str(def_OD)))

    def_verb = 1
    parser.add_argument("-v", "--verb", 
                        metavar='VERB', 
                        dest='verb', 
                        type=int, default=def_verb, 
                        help='verbosity for code running' + '\n' +
                        '(def: {})'.format(str(def_verb)))

    def_train_batch_size = 1
    parser.add_argument("-trb", "--train_batch_size", 
                        metavar='TRAIN_BATCH_SIZE', 
                        dest='train_batch_size', 
                        type=int, default=def_train_batch_size, 
                        help='batch size for training' + '\n' +
                        '(def: {})'.format(str(def_train_batch_size)))


    def_net_arch = lmt.list_net_arch[0]
    parser.add_argument("-a", "--architecture", 
                        dest="net_arch", 
                        type=str, default=def_net_arch,
                        help="network architecture type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_net_arch)) + '\n' +
                        '(def: {})'.format(str(def_net_arch)))


    def_half_prec = 0
    parser.add_argument("-hp", "--half_prec", 
                        dest="half_prec", 
                        type=int, default=def_half_prec,
                        help='half precision' + '\n' +
                        '(def: {})'.format(str(def_half_prec)))

    def_mixed_prec = 0
    parser.add_argument("-mp", "--mixed_prec", 
                        dest="mixed_prec", 
                        type=int, default=def_mixed_prec,
                        help='mixed precision' + '\n' +
                        '(def: {})'.format(str(def_mixed_prec)))

    def_data_norm = lmt.list_data_norm[0]
    parser.add_argument("-dn", "--data_norm", 
                        dest="data_norm", 
                        type=str, default=def_data_norm,
                        help="data normalization type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_data_norm)) + '\n' +
                        '(def: {})'.format(str(def_data_norm)))


    def_wt_norm = 0
    parser.add_argument("-w", "--weight_norm", 
                        dest="weight_norm", 
                        type=int, default=def_wt_norm,
                        help='weight normalization' + '\n' +
                        '(def: {})'.format(str(def_wt_norm)))

    def_restart = 0
    parser.add_argument("-r", "--restart", 
                        dest="restart", 
                        type=int, default=def_restart,
                        help='restart using checkpoint weights' + '\n' +
                        '(def: {})'.format(str(def_restart)))  

    def_loss_func = lml.DEF_CalcLoss
    parser.add_argument("-L", "--Loss", 
                        dest="loss_func", 
                        type=str, default=def_loss_func,
                        help="loss function type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lml.list_CalcLoss)) + '\n' +
                        '(def: {})'.format(str(def_loss_func)))

    def_optimizer = lmt.list_optimizer[0]
    parser.add_argument("-O", "--optimizer", 
                        dest="optimizer", 
                        type=str, default=def_optimizer,
                        help="optimizer type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(lmt.list_optimizer)) + '\n' +
                        '(def: {})'.format(str(def_optimizer)))

    def_nifti = "don't write out dsets"
    parser.add_argument("-W", "--write_nifti", 
                        dest='do_nifti',
                        action='store_true',
                        #const=def_nifti,
                        help='flag to turn on the writing of NIFTI' + '\n' +
                        "datsets while processing" + '\n' +
                        '(def: {})'.format(str(def_nifti)))

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
    if din[-1] == '/' :
        dout = din[:-1]
    else:
        dout = din

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

def get_args_state(args):
    '''
    Get a nice string for outputting the state of the variables
    '''

    ostr = ''

    for key in vars(args):
        ostr+= "{:15s} : {:15s}\n".format(str(key), str(vars(args)[key]))

    return ostr



if __name__ == '__main__':

    # [PT] a trick so that putting in *no* args prompts the help to be
    # shown
    if len(sys.argv) == 1 :
        sys.argv.append('-h')

    # get args (path and parameter settings) , and prepare to pass
    # along to the main training net prog
    args       = get_args()
    data_path  = args.data_dir
    epochs     = args.epochs
    lr         = args.learning_rate
    tr_bsize   = args.train_batch_size
    seed       = args.seed
    net_arch   = args.net_arch
    loss_func  = args.loss_func
    optimizer  = args.optimizer
    verb       = args.verb
    half_prec  = args.half_prec
    mixed_prec = args.mixed_prec
    wt_norm    = args.weight_norm
    do_nifti   = args.do_nifti
    data_norm  = args.data_norm
    restart    = args.restart  
    outdir     = prep_outdir(args.outdir, verb=verb)

    if not(outdir) :
        print("ERROR: this path is not valid: {}".format(args.outdir))
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
        not(check_opt_allowed( data_norm, lmt.list_data_norm,
                               desc_bad='This data normalization ' + 
                               'is not in the List:' ))  :
        sys.exit(5)

    # save command used
    str_args = get_args_state(args)
    writeout_args( sys.argv, outdir, ver=__version__, state=str_args,
                   verb=verb )

    net = lmt.train_net( data_path, epochs, lr, tr_bsize, seed, net_arch, loss_func,
                         optimizer, half_prec, mixed_prec, wt_norm, restart, data_norm, do_nifti, 
                         outdir, verb )
