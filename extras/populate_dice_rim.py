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

    parser.add_argument('-e', "--edt_dir", 
                        type=dir_path, 
                        help="Path to the edt directory")

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
    edt_foldername = args.edt_dir
    filename   = args.filename
    multiple   = args.multiple

    
    df_inner_rim = pd.DataFrame()
    df_outer_rim = pd.DataFrame()
    for idx in range(0, BIG, multiple): # counter for number of epochs 
    

        
        list_pred_mask = [os.path.basename(x) for x in  glob.glob(os.path.join(foldername,"ch01_ep-*_train_subj-pac_*.nii.gz"))
                    if float(os.path.basename(x)[8:11]) == idx]

        targ_path_str = os.path.join(foldername, 'target_000_train*.nii.gz')
        list_target = glob.glob(targ_path_str)

        edt_path_str = os.path.join(edt_foldername, '*.nii.gz')
        list_edt = glob.glob(edt_path_str)

        sorted_list_pred_mask = sorted(list_pred_mask) 
        sorted_list_target    = sorted(list_target)
        sorted_list_edt   = sorted(list_edt)
        #print(sorted_list_edt)
    
        if(len(sorted_list_pred_mask)==0):
            break
        print('idx =',idx)

        #print('sorted_list_pred_mask =',len(sorted_list_pred_mask))
        #print('sorted_list_target =',len(sorted_list_target))
    
        #add condition that sorted_list_pred_mask == sorted_list_target
          
        dice_inner_rim_list = [] 
        dice_outer_rim_list = []    
        
        for idy in range(len(sorted_list_target)): # counter for dataset
        
            #compute dice for pred_mask and target 
        
            #print('idy =',idy)
            pred_path_str   = os.path.join(foldername, sorted_list_pred_mask[idy])
            pred_mask_data1 = nib.load(pred_path_str)
            target_data2    = nib.load( sorted_list_target[idy])
            edt_data3        = nib.load( sorted_list_edt[idy])
            pred_mask_data1 = np.asanyarray(pred_mask_data1.dataobj).astype('float32')
            target_data2    = np.asanyarray(target_data2.dataobj).astype('float32')
            edt_data3       = np.asanyarray(edt_data3.dataobj).astype('float32')

            
            # define rim region 
            rim = (edt_data3 > 0.6)

            #define inner rim region 
            inner_rim = np.where(rim == target_data2, rim, 0)
            pred_inner_rim   = pred_mask_data1 * inner_rim
            targ_inner_rim   = target_data2 * inner_rim
            dice_inner_rim   = get_dice(pred_inner_rim, pred_inner_rim)

            
            #define outer rim region 
            b= np.logical_not(target_data2)
            outer_rim = np.where(rim == b, rim,0)
            
            pred_outer_rim  = pred_mask_data1 * outer_rim
            targ_outer_rim  = target_data2 * outer_rim
            dice_outer_rim  = get_dice(pred_outer_rim, pred_outer_rim)
        
            #print("{:0.4f}".format(dice_metric))
            dice_inner_rim_list.append(dice_inner_rim)
            dice_outer_rim_list.append(dice_outer_rim)
            #print('dice_list size',len(dice_list))
            pd_dice_inner_rim_list = pd.Series(dice_inner_rim_list)
            pd_dice_outer_rim_list = pd.Series(dice_outer_rim_list)
            
        df_inner_rim[f'EPOCH_{idx}'] = pd_dice_inner_rim_list.values
        df_outer_rim[f'EPOCH_{idx}'] = pd_dice_outer_rim_list.values
        
        #display(df)
    
    df_inner_rim.to_csv(filename.name +'_inner_rim'+ '.csv')
    df_outer_rim.to_csv(filename.name +'_outer_rim'+'.csv')

    print('Loop ended.')