import os
import sys
import datetime
import argparse     as argp
import lib_ml_train as lmt
 
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
    parser = argp.ArgumentParser(description = intro,
                                 epilog = epilog_data_struc,
                                 formatter_class=argp.RawTextHelpFormatter) 

    parser.add_argument('-d',"--data_dir", 
                        type=dir_path, 
                        help="Path to the data directory (structure below)")

    parser.add_argument('-e', '--epochs', 
                        metavar='E', 
                        dest='epochs',
                        type=int, default=5,
                        help='Number of epochs (def: 5)')

    parser.add_argument("-l", "--learning_rate", 
                        metavar='LRATE', 
                        dest="learning_rate", 
                        type=float, default=0.001, 
                        help="Learning rate (def: 0.001)")

    parser.add_argument('-s', '--seed', 
                        metavar='S', 
                        dest='seed',
                        type=int, default=None,
                        help='Set seed for random value gen (def: None)')

    parser.add_argument('-o', '--outdir', 
                        metavar='OD', 
                        dest='outdir',
                        type=str, default='.',
                        help='Set name of output directory (def: .)')

    parser.add_argument("-v", "--verb", 
                        dest="verb", 
                        type=int, default=1, 
                        help="verbosity for code running (def: 1)")

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
            print("++ Path of outdir ({}) does NOT already".format(dout)) 
            print("   Will make it now.") 
        os.mkdir(dout)

    # report abs path (might prefer using this later, if program hops
    # around to different places, but not for now)
    if verb :
        abspath = os.path.abspath(dout)
        print("++ Absolute path of outdir will be: {}".format(abspath))

    return dout

def writeout_args(argv, outdir, ofile='cmd_args.txt', verb=1):
    """Store the command used to make this run.  Simple/no formatting at
    the moment

    Parameters
    ----------
    
    argv       : the list of terminal commands used.
    outdir     : output directory where all outputs (including what is 
                 written here) will go.
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

    fff.write("# Run date  : {}\n".format(odate))
    fff.write("# Run time  : {}\n".format(otime))
    fff.write("# Run loc   : {}\n".format(opwd))
    fff.write("# Run cmd   :\n{}".format(ocmd))

    fff.close()

    if verb :
        print("++ Store executed command in text file: {}".format(otxt))


if __name__ == '__main__':

    # [PT] a trick so that putting in *no* args prompts the help to be
    # shown
    if len(sys.argv) == 1 :
        sys.argv.append('-h')

    # get args (path and parameter settings) , and prepare to pass
    # along to the main training net prog
    args      = get_args()
    data_path = args.data_dir
    epochs    = args.epochs
    lr        = args.learning_rate
    seed      = args.seed
    verb      = args.verb
    outdir    = prep_outdir(args.outdir, verb=verb)

    if not(outdir) :
        print("ERROR: this path is not valid: {}".format(args.outdir))
        sys.exit(5)

    # save command used
    writeout_args(sys.argv, outdir, verb=verb)

    net = lmt.train_net(data_path, epochs, lr, seed, outdir, verb)
