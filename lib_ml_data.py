import os

import numpy   as np
import nibabel as nib

#data_dims = (256, 256, 256)


def vol_generator(foldername):
    """Take a subdirectory (training, validation, etc.) and populate matrices
    for the dsets.

    Parameters
    ==========

    foldername   : Subdirectory of data (see program help for  
                   directory sub-structure), such as '.../training'
                   or '.../validation'

                   Any 'foldername' should contain 2 directories of
                   dsets: 'mask' and 'orig'.  These should each
                   contain N files (matched across the directories).


    Returns
    =======

    x            : A list of N arrays, where N is the number of dsets 
                   in the 'orig' subdir

    y            : A list of N arrays, where N is the number of dsets 
                   in the 'mask' subdir

    x and y should have the same length.  The size of each element
    should be the same: an MxMxM array of data.

    """

    orig_data_path = os.path.join(foldername,'orig')
    orig_data_list = sorted(os.listdir(orig_data_path))
    orig_set_size  = len(orig_data_list)
    
    mask_data_path = os.path.join(foldername,'mask')
    mask_data_list = sorted(os.listdir(mask_data_path))
    mask_set_size  = len(mask_data_list)

    # [PT] Q: Shouldn't the dimensions here come from the dsets
    # themselves?  I don't think they should be hardwired as 32x32x32.
    # For example, could get dimensions from the first dset, and make
    # sure all the others match. (And I think at the moment we will
    # constrain ourselves to *always* have uniform input
    # dimensions??--- this means we should also check+exit with error
    # if that is not the case.)
    x = np.zeros((orig_set_size, 32, 32, 32), dtype=np.float)
    y = np.zeros((mask_set_size, 32, 32, 32), dtype=np.float)
    
    for index in range(orig_set_size):
        
        orig_filename  = orig_data_list[index]
        orig_data_file = os.path.join(orig_data_path, orig_filename)
        orig_image     = nib.load(orig_data_file)
        # from:
        # https://www.programcreek.com/python/example/98176/nibabel.load
        ### [PT] Q: Should this always be "astype('int16')"?  Could we
        ### save space with making the mask data binarized?  And is
        ### there an issue that x was initialized above as type
        ### 'float', while now the data array has type int16?
        data           = np.asanyarray(orig_image.dataobj).astype('int16')  
        x[index]       = data
        
        mask_filename  = mask_data_list[index]
        mask_data_file = os.path.join(mask_data_path, mask_filename)
        mask_image     = nib.load(mask_data_file)
        ### [PT] Q: Should this always be "astype('int16')"?  Could we
        ### save space with making the mask data binarized (bool
        ### type)?  As above, is there a problem that the y array was
        ### initialized with float type, and this is int16?
        data           = np.asanyarray(mask_image.dataobj).astype('int16') 
        y[index]       = data
        
    return x, y
    

def SSData_path(data_path):
    #print(data_path)
    
    training_path   = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')
    
    return(str(training_path), str(validation_path))




