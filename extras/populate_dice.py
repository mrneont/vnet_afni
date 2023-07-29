import nibabel as nib
import os
import sys
import glob
import numpy 	as np
import pandas 	as pd
import argparse as argp
from IPython.display import display


#****************************************************************************************************
#  Purpose : This python file computes the dice metric of the pred_mask from different epochs and 
#          target mask and populates the csv file.    
#
#  Usage : python populate_dice.py -d 'dir/folder/outdir' -f 'table_file_name' -m 2
#
#
#****************************************************************************************************

BIG = 1000

def get_dice(pred, gt):
    # num is the total number of classes, include the background
    dice = 2.0*np.sum(pred*gt)/(np.sum(pred)+np.sum(gt))
            
    return dice

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_args():

    parser = argp.ArgumentParser()



    
    parser.add_argument('-d', "--data_dir", 
                        type=dir_path, 
                        help="Path to the data directory")

    parser.add_argument('-f', "--filename" ,
                            help= "output file", 
                            type=argp.FileType('w'))

    def_m = 1
    parser.add_argument('-m', "--multiple" ,
                            help= "multiple", default=def_m, 
                            type=int)

    return parser.parse_args()




if __name__ == '__main__':

    args       = get_args()
    foldername = args.data_dir
    filename   = args.filename
    multiple   = args.multiple

    
    df= pd.DataFrame()
    for idx in range(0, BIG, multiple): # counter for number of epochs 
    

        
        list_pred_mask = [os.path.basename(x) for x in  glob.glob(os.path.join(foldername,"ch01_ep-*_train_subj-pac_*.nii.gz"))
                    if float(os.path.basename(x)[8:11]) == idx]

        targ_path_str = os.path.join(foldername, 'target_000_train*.nii.gz')
        list_target = glob.glob(targ_path_str)

        sorted_list_pred_mask = sorted(list_pred_mask) 
        sorted_list_target    = sorted(list_target)
    
        if(len(sorted_list_pred_mask)==0):
            break
        print('idx =',idx)

        #print('sorted_list_pred_mask =',len(sorted_list_pred_mask))
        #print('sorted_list_target =',len(sorted_list_target))
    
        #add condition that sorted_list_pred_mask == sorted_list_target
          
        dice_list=[]    
        
        for idy in range(len(sorted_list_target)): # counter for dataset
        
            #compute dice for pred_mask and target 
        
            #print('idy =',idy)
            pred_path_str   = os.path.join(foldername, sorted_list_pred_mask[idy])
            pred_mask_data1 = nib.load(pred_path_str)
            target_data2    = nib.load( sorted_list_target[idy])
            pred_mask_data1 = np.asanyarray(pred_mask_data1.dataobj).astype('float32')
            target_data2    = np.asanyarray(target_data2.dataobj).astype('float32')
            dice            = get_dice(pred_mask_data1,target_data2)
        
        
            #print("{:0.4f}".format(dice_metric))
            dice_list.append(dice)
            #print('dice_list size',len(dice_list))
            pd_dice_list = pd.Series(dice_list)
            
        df[f'EPOCH_{idx}'] = pd_dice_list.values
        
        #display(df)
    
    df.to_csv(filename.name + '.csv')

    print('Loop ended.')