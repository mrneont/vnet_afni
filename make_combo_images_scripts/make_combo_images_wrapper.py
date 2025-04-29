import sys
import os 
import argparse      as argp
import json
import random
import numpy as np
import os
import glob
import re #regular expression


def get_make_combo_args():

    parser = argp.ArgumentParser(prog = 'make_combo_images_wrapper.py',
                                    formatter_class=argp.RawTextHelpFormatter)
   
    # data_dir is the  path for the pred_mask_output_dir

    parser.add_argument("-d", "--pred_mask_output_dir")

    parser.add_argument("-c", "--combo_images_dir")

   

    return parser.parse_args()



def write_script(da_dict,fl_name, daug):
    # set the filename for the shell script based on the dataset
    dir_name  = os.path.dirname(fl_name)
    file_name = os.path.basename(fl_name)
    
    new_extension = ".tcsh"
    #print('file_name=',file_name)
    script_fl = file_name.split('.')[0] +new_extension
    script_fl_path_str = os.path.join(os.path.dirname(dir_name), 'scripts', script_fl)
    print('script_fl =', script_fl)
    f = open(script_fl_path_str, "w")
    f.write("#!/bin/tcsh")
    f.write('\n')

    # notes 
    # + the individual shell script need the 
    #   absolute path to the dataset 
    #print("write_script fl_name =  ",fl_name)


    
        f.write('\n')

    
    # create the edt data for all the masks 

    
    f.write("tcsh do_daug_weight.tcsh {}""".format(fl_name))

    f.write('\n')
    f.close()


def main():

    print('sys.version_info = ',sys.version_info)
    if sys.version_info<(3,10,0):
        sys.stderr.write("You need python 3.10 or later to run this script. \
                        \nThe 'match' statement was introduced in Python 3.10.\
                        \nSo if you're using an older version, \
                        you'll need to upgrade to use it.")
        exit(1)

    random.seed(42)
    print('Random seed =', random.random())

    args        = get_make_combo_args()
    

    pred_mask_path     = args.pred_mask_output_dir
    #print(" data augmentation folder is: {}".format(da_path))
    
    # data_dir is the data path for the dataset folder
    data_path   = args.data_dir 

    # number of copies of the dataset to be made 
    num_cp      = args.num_cp   
    
    if(os.path.isdir(da_path) == True):
        print('\nStatus msg : Data augmentation folder already exists')
        print('Please change the data_augmentation directory name and retry\n')
        sys.exit(1) 

    cmd  = '''tcsh make_folders_combo.tcsh {param1} {param2} {param3}'''.\
                            format(param1= data_path, param2= da_path,\
                            param3= num_cp)

    com  = ab.shell_com(cmd, capture=1)
    stat = com.run()
    # print the status of preparing the  folder
    if (stat == 0):
        print("Status msg : folder succesfully created")
    else :
        print("Status msg : folder not created: exiting")
        sys.exit(1)
    
  

main()