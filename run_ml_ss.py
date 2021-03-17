import os
import sys
import argparse     as argp
import lib_ml_train as mlt

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_args():

    intro  = 'Train the VNet on MRI data and target masks'
    parser = argp.ArgumentParser(description = intro,
                        formatter_class=argp.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d',"--data_dir", 
                        type=dir_path, 
                        help="Path to the data directory")

    parser.add_argument('-e', '--epochs', 
                        metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')

    parser.add_argument("-l", "--learning_rate", 
                        dest="learning_rate", default=0.001, 
                        type=float, help="Learning rate. Default: 0.001")
    return parser.parse_args()

if __name__ == '__main__':

    # [PT] a trick so that putting in *no* args prompts the help to be
    # shown
    if len(sys.argv) == 1 :
        sys.argv.append('-h')

    args = get_args()
    #data_path = 'data/pretrain/pretrain_vnet_res8'
    data_path = args.data_dir
    epochs=args.epochs
    lr = args.learning_rate
    
    net = mlt.train_net(data_path,epochs,lr)
    #mlt.test(data_path)
