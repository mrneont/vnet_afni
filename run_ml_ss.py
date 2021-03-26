import os
import sys
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

    parser.add_argument("-v", "--verb", 
                        dest="verb", 
                        type=int, default=1, 
                        help="verbosity for code running (def: 1)")

    return parser.parse_args()

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
    
    net = lmt.train_net(data_path, epochs, lr, seed, verb)
    #lmt.test(data_path)
