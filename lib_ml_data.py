import os
import sys

import lib_nibabel_utils     as lnu  

import numpy   as np
import nibabel as nib
import torch.utils.data as data
import glob

class mridataset(data.Dataset):

    def __init__(self, root_path):
        self.orig_data_list = [x for x in glob.glob(os.path.join(root_path, 'orig','*.nii.gz'))]
        self.mask_data_list = [x for x in glob.glob(os.path.join(root_path, 'mask','*.nii.gz'))]
        


    def __getitem__(self, index):
        
        self.orig_image  = nib.load(self.orig_data_list[index])
        self.mask_image  = nib.load(self.mask_data_list[index])
        self.orig_data   = np.asanyarray(self.orig_image.dataobj).astype('float32')  
        self.mask_data   = np.asanyarray(self.mask_image.dataobj).astype('float32')  
        
        return (self.orig_data, self.mask_data)

    def __len__(self):
        return len(self.orig_data_list)




def mat_generator(foldername, verb=1):
    """Take a subdirectory (training, validation, etc.) and populate matrices
    for the dsets.

    Parameters
    ==========

    foldername   : Subdirectory of data (see program help for  
                   directory sub-structure), such as '.../training'
                   or '.../validation'.

                   Any 'foldername' should contain 2 directories of
                   dsets: 'mask' and 'orig'.  These should each
                   contain N files (matched across the directories).


    Returns
    =======

    orig_mat     : A list of N arrays (each H x W x D of individual
                   datasets), where N is the number of dsets in the
                   'orig' subdir.

    mask_mat     : A list of N arrays (each H x W x D of individual
                   datasets), where N is the number of dsets in the
                   'mask' subdir.

    The output orig_mat and mask_mat should have the same length.  The
    size of each element should be the same: an MxMxM array of data.

    """

    orig_data_path = os.path.join(foldername, 'orig')
    orig_data_list = sorted(os.listdir(orig_data_path))
    orig_set_size  = len(orig_data_list)
    
    mask_data_path = os.path.join(foldername, 'mask')
    mask_data_list = sorted(os.listdir(mask_data_path))
    mask_set_size  = len(mask_data_list)

    if orig_set_size != mask_set_size :
        
        print("** ERROR: mismatched number of orig ({}) and mask ({}) dsets."
              "".format(orig_set_size, mask_set_size))
        print("   Check dataset directories:\n"
              "     {}\n"
              "     {}\n"
              "".format( orig_data_path, mask_data_path ))

        sys.exit(3)

    # get matrix size from first dset in list, and can check later
    # that it matches for everyone. This is necessary to get the
    # matrix size from the input dsets
    if True :
        orig_filename  = orig_data_list[0]
        orig_data_file = os.path.join(orig_data_path, orig_filename)
        orig_image     = nib.load(orig_data_file)
        orig_shape     = orig_image.shape

        mask_filename  = mask_data_list[0]
        mask_data_file = os.path.join(mask_data_path, mask_filename)
        mask_image     = nib.load(mask_data_file)
        mask_shape     = mask_image.shape

        if len(orig_shape) != 3 or len(mask_shape) != 3 :
            print("** ERROR: orig and mask dsets should be 3D volumes, not\n"
                  "'{}' and '{}' dimensional".format(len(orig_shape), 
                                                     len(mask_shape)))
            sys.exit(2)
        else:
            str1 = ' '.join([str(x) for x in orig_shape])
            str2 = ' '.join([str(x) for x in mask_shape])
            if str1 != str2 :
                print("** ERROR: orig and mask dsets have same shape,\n"
                      " but don't:\n"
                      "   {} : {}\n"
                      "   {} : {}\n".format(orig_data_file, str1,
                                            mask_data_file, str2))
                sys.exit(2)
            
        # if we reach here, we should be happy that the number of orig
        # dsets matches that of mask dsets; and the dimensions of the
        # orig dsets should match that of the mask ones (at least for
        # the [0]th examples of each list
        ndset_and_dims = (orig_set_size, orig_shape[0], orig_shape[1],
                          orig_shape[2])

    # matrix dims come from dsets themselves
    orig_mat = np.zeros(ndset_and_dims, dtype=np.float32)
    mask_mat = np.zeros(ndset_and_dims, dtype=np.float32)
    
    if verb > 1 :
        print("\nLoading orig+mask dsets:\n")
        print("  {:^30s} : {:^30s}".format('orig files', 'mask files'))
        print("  {:30s}  : {:30s}".format('-'*30, '-'*30))

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
        orig_data       = np.asanyarray(orig_image.dataobj).astype('float32')  
        orig_mat[index] = orig_data
        
        mask_filename  = mask_data_list[index]
        mask_data_file = os.path.join(mask_data_path, mask_filename)
        mask_image     = nib.load(mask_data_file)
        ### [PT] Q: Should this always be "astype('int16')"?  Could we
        ### save space with making the mask data binarized (bool
        ### type)?  As above, is there a problem that the y array was
        ### initialized with float type, and this is int16?
        ##### [PT: Apr 5, 2020] There is still a type mismatch: 'mask'
        ##### defined above has np.float32; this is being read in
        ##### 'astype' bool... but it probably gets immediately
        ##### converted to float32?  Python arrays can only have 1
        ##### type, so implicit type conversion will have to take
        ##### place, but this should be dealt with consistently.
        mask_data       = np.asanyarray(mask_image.dataobj).astype('bool') 
        mask_mat[index] = mask_data
        
        if verb > 1 :
            print("  {:30s} : {:30s}".format(orig_filename, mask_filename))

        check_pair = is_mask_orig_pair(orig_filename, mask_filename)
        if not(check_pair) :
            print("** ERROR: mismatched orig+mask pair:\n"
                  "   {:30s} : {:30s}".format(orig_filename, mask_filename))
            sys.exit(3)

    return orig_mat, mask_mat
    

def SSData_path(data_path):
    #print(data_path)
    
    training_path   = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')
    
    return(str(training_path), str(validation_path))

def is_mask_orig_pair(A, B):
    """Check if 2 dataset named A and B have filenames that match *except*
    for one containing 'orig' and the other 'mask'.  Basically, check the
    prefix_noext for each, removing 'mask' and 'orig'

    Parameters
    ==========
    A, B    : (str) file names

    Return
    ======
    
    int     : 1 if they match, 0 if they don't

    """
    
    # check that names match for each pair: later generalize

    x = A.replace('.gz', '')
    x = x.replace('.nii', '')
    x = x.replace('_orig', '')

    y = B.replace('.gz', '')
    y = y.replace('.nii', '')
    y = y.replace('_mask', '')

    if x == y :
        return 1
    else:
        return 0
