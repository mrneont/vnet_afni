import os

import numpy as np
import nibabel as nib

#data_dims = (256, 256, 256)


def vol_generator(foldername):
    
    orig_data_path = os.path.join(foldername,'orig')
    orig_data_list  = sorted(os.listdir(orig_data_path))
    orig_set_size = len(orig_data_list)
    
    mask_data_path = os.path.join(foldername,'mask')
    mask_data_list  = sorted(os.listdir(mask_data_path))
    mask_set_size = len(mask_data_list)

    x = np.zeros((orig_set_size, 32, 32,32), dtype=np.float)
    y = np.zeros((mask_set_size,32, 32,32), dtype=np.float)
    
    for index in range(orig_set_size):
        
        orig_filename = orig_data_list[index]
        orig_data_file = os.path.join(orig_data_path, orig_filename)
        orig_image = nib.load(orig_data_file)
        # from:
        # https://www.programcreek.com/python/example/98176/nibabel.load
        data = np.asanyarray(orig_image.dataobj).astype('int16')  
        x[index] = data
        
        mask_filename = mask_data_list[index]
        mask_data_file = os.path.join(mask_data_path, mask_filename)
        mask_image = nib.load(mask_data_file)
        data = np.asanyarray(mask_image.dataobj).astype('int16') 
        y[index] = data
        
    return(x,y)
    

def SSData_path(data_path):
    #print(data_path)
    
    training_path = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')
    
    return(str(training_path), str(validation_path))




