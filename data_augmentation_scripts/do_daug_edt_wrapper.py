
import sys
import os 
import argparse      as argp
import glob

'''
************************************************************************************
* This is a wrapper code which calls the do_daug_weight.tcsh 
* This code is used for creating depth data for the validation datasets. 
* 
* Inputs required are as below
*  -d  | --data_dir (data path for the dataset folder)
*  -s  | --script_path for the run_edt.tcsh script
*
* sample usage of the code : python data_augmentation_wrapper.py 
*                           -d '/Users/name/dataset' 
*                           -s '/Users/name/dataset_aug/scripts'
*
*
*************************************************************************************
'''
def get_data_augment_args():

    parser = argp.ArgumentParser(prog = 'data_augmentation_wrapper.py',
                                    formatter_class=argp.RawTextHelpFormatter)
    
    # data_dir is the data path for the dataset folder

    parser.add_argument("-d", "--data_dir")

    parser.add_argument("-s", "--script_path")
    
    return parser.parse_args()


def main():

    args        = get_data_augment_args()
    #data_augment_path is the data_path for data augmentation folder
    data_path     = args.data_dir
    script_path   = args.script_path
    orig_path_str = os.path.join(data_path, 'orig', '*.nii.gz')
    # list of copies of dataset
    orig_data_list = glob.glob(orig_path_str)
    orig_data_list.sort()

    run_edt = "run_edt.tcsh"

    fl_path_str = os.path.join(script_path,run_edt)
    fl        = open(fl_path_str, "w")
    
    

    fl.write("#!/bin/tcsh")
    fl.write('\n')
    for fl_name in orig_data_list:
        # condition to check whether to daug or not
        fl_basename = os.path.basename(fl_name)
        
        name = fl_basename.split('.')[0]
        print(name)

        fl.write("tcsh do_daug_weight.tcsh {}""".format(fl_name))
        fl.write('\n')

    fl.write('\n')
    fl.close()    






main()