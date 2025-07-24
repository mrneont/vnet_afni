
import os
import sys

import lib_nibabel_utils     as lnu  

import numpy   as np
import nibabel as nib
import torch.utils.data as data
import glob


class mridataset(data.Dataset):
    """+ A 'mridataset' Dataset class is substantiated for the Dataloader.

+ The Dataset class is used by the Dataloader class.

+ A custom Dataset class must implement three functions: __init__,
  __len__, and __getitem__.

+ The __init__ function initializes the data directory, annotation
  file and any data tranforms.

+ The __len__ function returns the number of samples in the dataset.

+ The __getitem__ function loads and returns a sample from the dataset
  at the given index.

  Based on the index, it converts the mri data into torch tensor using
  nibabel 'load'.

  It retrieves the corresponding groundtruth/mask
    """

    def __init__(self, root_path, use_dpth_wts=None, no_mask=None, verb=0):

        # initialize 
        self.orig_data_list = []
        self.mask_data_list = []
        self.dpth_data_list = []        # only populated if use_dpth_wts

        self.orig_head_list = []        # store all the headers

        self.root_path      = root_path
        self.use_dpth_wts   = use_dpth_wts
        self.no_mask        = no_mask
        self.verb           = verb

        # ===============================================================

        orig_path_str = os.path.join(self.root_path, 'orig', '*.nii.gz')
        self.orig_data_list = glob.glob(orig_path_str)
        self.orig_data_list.sort()

        if self.verb > 1 :
            print("++ All orig dsets:")
            for x in self.orig_data_list:
                print(x)

        # verify that we have data
        Norig_dset = len(self.orig_data_list)
        if Norig_dset == 0 :
            print("** ERROR: no orig datasets found in '{}'"
                  "".format(orig_path_str))
            sys.exit(3)
        else:
            print("++ Found {} orig dsets".format(Norig_dset))

        # length of root string, used below
        Nroot = len(self.root_path)

        # lists of dsets made in parallel, must match item for item
        for orig_dset in self.orig_data_list:
            A = nib.load(orig_dset)
            self.orig_head_list.append(A.header.copy())
            orig_file = orig_dset[Nroot:]

            if (self.no_mask==0) :
                mask_file = orig_file.replace('orig', 'mask')
                mask_dset = ''.join([self.root_path, mask_file])
                self.mask_data_list.append(mask_dset)

            if self.use_dpth_wts :
                dpth_file = orig_file.replace('orig', 'edt')
                dpth_dset = ''.join([self.root_path, dpth_file])
                self.dpth_data_list.append(dpth_dset)

        if (self.no_mask==0) :
            # verify that all the mask and dpth files exist
            MISSING_DSET = 0
            for dset in self.mask_data_list :
                if not(os.path.isfile(dset)):
                    MISSING_DSET = 1
                    print("** ERROR: required mask dataset not found: "
                          "{}".format(dset))
            if self.use_dpth_wts :
                for dset in self.dpth_data_list :
                    if not(os.path.isfile(dset)):
                        MISSING_DSET = 1
                        print("** ERROR: required dpth dataset not found: "
                              "{}".format(dset))
            if MISSING_DSET :
                sys.exit(5)

        if self.verb > 1:
            print("++ Check matching of input dsets:")
            for i in range(len(self.orig_data_list)):
                if self.use_dpth_wts :
                    print("{:20s} --- {:20s}--- {:20s}"
                          "".format(self.orig_data_list[i], 
                                    self.mask_data_list[i],
                                    self.dpth_data_list[i]))
                else:
                    print("{:20s} --- {:20s}"
                          "".format(self.orig_data_list[i], 
                                    self.mask_data_list[i]))

    def __getitem__(self, index):
        """Output set of 3 index-th datasets as arrays.

        The third item in the tuple might be an empty array, if the
        dpth weights are not being used.  The DataLoader needed each
        item to be an array (otherwise the third item might have been None).

        Return
        ------
        orig_data     : array of original data
        mask_data     : array of mask data
        dpth_data     : if use_dpth_wts, an array of depth_map data; else, empty

        """
        
        # [PT] Q: do we always want these thesholding bits done, or
        # should those be controlled by a user option?
        orig_fname = os.path.basename(self.orig_data_list[index])
        orig_image  = nib.load(self.orig_data_list[index])
        orig_data   = np.asanyarray(orig_image.dataobj).astype('float32')
        top99_thresh = np.percentile(orig_data, 99) 
        orig_data[orig_data >top99_thresh] = top99_thresh
        down2_thresh = np.percentile(orig_data, 2) 
        orig_data[orig_data < down2_thresh] = down2_thresh
        if (self.no_mask==0) :
            mask_image  = nib.load(self.mask_data_list[index])
            mask_data   = np.asanyarray(mask_image.dataobj).astype('float32')
        else:
            mask_data   = np.ndarray(0) 

        if self.use_dpth_wts :
            dpth_image  = nib.load(self.dpth_data_list[index])
            dpth_data   = np.asanyarray(dpth_image.dataobj).astype('float32')
        else:
            dpth_data   = np.ndarray(0) 

        if type(index) != tuple :
            index = tuple([index])

        
        return orig_data, mask_data, dpth_data, orig_fname, index

    def __len__(self):
        return len(self.orig_data_list)

# ------------------------------------------------------------------------



# one idea of scaling the input dsets, to have a range of values [0,
# 1], to start  
def min_max_scale(img):
   
    data_min   = img.min()
    data_max   = img.max()
    #print("orig_data max = {}".format(data_max))
    #print("orig_data min = {}".format(data_min))
    normalized = (img - data_min) / (data_max - data_min)

    return normalized

def z_scoring(img):
   
    data_mean  = img.mean()
    data_std   = img.std()

    Z_normalized = (img - data_mean) / data_std
    #print("orig_data max = {}".format(Z_normalized.max()))
    #print("orig_data min = {}".format(Z_normalized.min()))

    return Z_normalized

# ------------------------------------------------------------------------
    
def mat_generator(foldername, verb=1):
    """Take a subdirectory (training, validation, etc.) and populate
    matrices for the dsets.

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
        orig_data       = np.asanyarray(orig_image.dataobj).astype('float32')  
        orig_mat[index] = orig_data
        
        mask_filename  = mask_data_list[index]
        mask_data_file = os.path.join(mask_data_path, mask_filename)
        mask_image     = nib.load(mask_data_file)
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

    training_path   = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')
    
    return (str(training_path), str(validation_path))

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
