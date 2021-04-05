import os
import sys

import numpy   as np
import nibabel as nib

#data_dims = (256, 256, 256)


def vol_generator(foldername, verb=1):
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

    orig         : A list of N arrays, where N is the number of dsets 
                   in the 'orig' subdir

    mask         : A list of N arrays, where N is the number of dsets 
                   in the 'mask' subdir

    x and y should have the same length.  The size of each element
    should be the same: an MxMxM array of data.

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

    # [PT] Q: Shouldn't the dimensions here come from the dsets
    # themselves?  I don't think they should be hardwired as 32x32x32.
    # For example, could get dimensions from the first dset, and make
    # sure all the others match. (And I think at the moment we will
    # constrain ourselves to *always* have uniform input
    # dimensions??--- this means we should also check+exit with error
    # if that is not the case.)
    #### [PT: Apr 5, 2020] the dimensions here should be read read in
    #### from the dset, using an initial 'read' with nibable,
    #### extracting the information from the file header.  '32x32x32'
    #### should *not* be hardwired here, because our dsets will not
    #### not always have that property.  Reading 1 dset and getting
    #### that info should be OK, because all dsets should have that
    #### property---though we should certainly check this for
    #### consistency across dsets once one is read in (at the moment).
    orig = np.zeros((orig_set_size, 32, 32, 32), dtype=np.float32)
    mask = np.zeros((mask_set_size, 32, 32, 32), dtype=np.float32)
    

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
        orig_data      = np.asanyarray(orig_image.dataobj).astype('float32')  
        orig[index]    = orig_data
        
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
        mask_data      = np.asanyarray(mask_image.dataobj).astype('bool') 
        mask[index]    = mask_data
        
        if verb > 1 :
            print("  {:30s} : {:30s}".format(orig_filename, mask_filename))

        check_pair = is_mask_orig_pair(orig_filename, mask_filename)
        if not(check_pair) :
            print("** ERROR: mismatched orig+mask pair:\n"
                  "   {:30s} : {:30s}".format(orig_filename, mask_filename))
            sys.exit(3)

    return orig , mask
    

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
