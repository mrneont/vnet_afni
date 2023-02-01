import nibabel as nib
import os
import sys
import glob
import numpy 	as np
import pandas 	as pd
import argparse as argp
from IPython.display import display


#******************************************************************************
#  Purpose : This python file computes the dice metric of the pred_mask_rim 
#            and the target_mask_rim over different epochs and populates the 
#            csv file.
#         
# expected folder structure: 
# base_folder
#           -> datafolder
#           -> targ_foldername
#           -> rim_foldername
#
#  Usage : python populate_dice_rim.py -d 'dir/base_folder/datafolder'
#                                      -t 'dir/base_folder/targ_foldername'
#                                      -r 'dir/base_folder/rim_foldername'
#                                      -o 'dir/base_folder'
#                                      -f 'df_file_name'
#
#  df : here is the data frame
#******************************************************************************

BIG = 1000

def get_dice(pred, gt):
    '''
    +  This function calculates the dice between pred_mask and the target mask
    +  Input:
            pred -> predicted mask 
            gt   -> ground truth/ target
    +  Return: the dice value

    '''

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

    parser.add_argument('-r', "--rim_dir", 
                        type=dir_path, 
                        help="Path to the rim directory")

    parser.add_argument('-t', "--targ_dir", 
                        type=dir_path, 
                        help="Path to the targ directory")

    parser.add_argument('-o', "--out_dir", 
                        type=dir_path, 
                        help="Path to the output directory")

    parser.add_argument('-f', "--filename" ,
                            help= "output file", 
                            type=argp.FileType('w'))

    return parser.parse_args()



if __name__ == '__main__':

    args       = get_args()
    # folder name where the pred_mask are stored
    foldername = args.data_dir
    # csv file where the dice values are to be populated
    filename   = args.filename
    # folder name where the mask_rim are stored
    rim_foldername = args.rim_dir
    # folder name where the target_masks are stored
    targ_foldername = args.targ_dir
    # folder name where the filename.csv is to be written
    out_foldername = args.out_dir  # 
    
    dn = []

    for idx in range(0,BIG,5): # counter for number of epochs 
    
        print('epoch_number =',idx)
        #list_pred_mask has only the basename of the pred_mask files
        list_pred_mask = [os.path.basename(x) for x in  glob.glob(os.path.join
                            (foldername,"ch01_ep-*_train_subj-pac_*.nii.gz"))
                                if float(os.path.basename(x)[8:11]) == idx]

        #print(list_pred_mask)
        targ_path_str = os.path.join(targ_foldername, '*.nii.gz')
        list_targ = glob.glob(targ_path_str)
        
        
        #print(len(list_pred_mask))
        rim_path_str       = (os.path.join(rim_foldername, "rim_*.nii.gz"))
        list_rim = glob.glob(rim_path_str)               
        
    
        sorted_list_pred_mask = sorted(list_pred_mask) 
        sorted_list_target    = sorted(list_targ)
       
        sorted_list_rim       = sorted(list_rim)
        #print(len(sorted_list_pred_mask), len(sorted_list_target),
        #                                            len(sorted_list_rim))
        if(len(sorted_list_pred_mask)==0):
            break
    
        #[YNS]add condition that sorted_list_pred_mask == sorted_list_target
          
        dice=[]    
        df1= pd.DataFrame()

        for idy in range(len(sorted_list_pred_mask)): # counter for dataset
        
            #compute dice for pred_mask and rim 
            
            
            #print('pred_mask=',sorted_list_pred_mask[idy])
            #print('rim_filename=',rim_filename[0])
            #print('idy =',idy)
            # sorted_list_pred_mask has only the basenames of the pred_mask files. 
            pred_mask_data1 = nib.load(os.path.join(foldername, sorted_list_pred_mask[idy]))
            target_data2    = nib.load(sorted_list_target[idy])
            rim_data        = nib.load(sorted_list_rim[idy])

            pred_mask_data1 = np.asanyarray(pred_mask_data1.dataobj).astype('float64')
            target_data2    = np.asanyarray(target_data2.dataobj).astype('float64')
            rim_data        = np.asanyarray(rim_data.dataobj).astype('float64')
            rim_data        = (rim_data > 0.5).astype(np.int_)
            pred_rim        = pred_mask_data1 * rim_data
            targ_rim        = target_data2 * rim_data
            dice_metric     = get_dice(pred_rim,targ_rim)
        
        
            #print("{:0.4f}".format(dice_metric))
        
            df1 = df1.append({idx :dice_metric}, ignore_index=True)
            
        
        
        dn.append(df1)
        
        
       
    dn = pd.concat(dn, axis=1)
    dn.columns = [f'EPOCH_{c}' for c in dn]  
    display(dn)

    #print(out_foldername)
    #print(filename.name)
    out_fname = os.path.join(out_foldername,filename.name)
    #print('out_fname',out_fname)
    dn.to_csv(out_fname)


       

    print('Loop ended.')

