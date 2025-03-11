import sys
sys.path.append('/Users/narayanaswamyy2/abin')
from afnipy import afni_base as ab
import os 
import argparse as argp
import pandas   as pd



def get_data_augment_args():

    parser = argp.ArgumentParser(prog = 'data_augmentation_wrapper.py',
                                    formatter_class=argp.RawTextHelpFormatter)

    # data_dir is the data path for the dataset folder

    parser.add_argument("-d", "--data_dir")

    return parser.parse_args()




def main():

    args        = get_data_augment_args()
   
    
    # data_dir is the data path for the dataset folder
    data_path   = args.data_dir 

    
    cmd  = '''tcsh do_ROI_stats_combo.tcsh {param1} '''.\
                            format(param1= data_path)

    com  = ab.shell_com(cmd, capture=1)
    stat = com.run()
    # print the status of preparing the data_augmentation folder
    
    if (stat == 0):
        print("Status msg : roi_stats done succesfully ")
    else :
        print("Status msg :roi_stats not created: exiting")
        sys.exit(1)


    

main()


