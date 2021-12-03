import nibabel as nib
import os
import sys
import glob
import numpy 	as np
import pandas 	as pd
import argparse as argp
from IPython.display import display

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

    return parser.parse_args()




if __name__ == '__main__':

    args       = get_args()
    foldername = args.data_dir
    filename   = args.filename

    dn = []

    for idx in range(BIG): # counter for number of epochs 
    
        list_pred_mask = [x for x in glob.glob(os.path.join(foldername, "predmask_OPP_*_train_*.nii.gz"))
                        if float(x[-20:-18]) == idx]
        list_target    = [y for y in glob.glob(os.path.join(foldername, "target_*_train*.nii.gz"))
                        if float(y[-20:-18]) == idx]
    
        
        sorted_list_pred_mask = sorted(list_pred_mask) 
        sorted_list_target    = sorted(list_target)
    
        if(len(sorted_list_pred_mask)==0):
            break
    
        #add condition that sorted_list_pred_mask == sorted_list_target
          
        dice=[]    
        df1= pd.DataFrame()
        for idy in range(len(sorted_list_target)): # counter for dataset
        
            #compute dice for pred_mask and target 
        
        
            pred_mask_data1 = nib.load(sorted_list_pred_mask[idy])
            target_data2    = nib.load( sorted_list_target[idy])
            pred_mask_data1 = np.asanyarray(pred_mask_data1.dataobj).astype('float32')
            target_data2    = np.asanyarray(target_data2.dataobj).astype('float32')
            dice_metric     = get_dice(pred_mask_data1,target_data2)
        
        
            #print("{:0.4f}".format(dice))
        
            df1 = df1.append({idx :dice_metric}, ignore_index=True)
            
    
        dn.append(df1)
        
        
    dn = pd.concat(dn, axis=1)
    dn.columns = [f'EPOCH_{c}' for c in dn]  
    display(dn)
    
    dn.to_csv(filename.name + '.csv')

    print('Loop ended.')