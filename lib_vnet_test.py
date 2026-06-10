#!/usr/bin/env python

import os
import platform

# this needs to be done before torch import
# **** see if this is actually helpful???  ****
os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'   # 128 KB

import torch
import numpy                     as np
import nibabel                   as nib

from afnipy import afni_base     as ab
from afnipy import afni_util     as au

from communifti import lib_nibabel_read_nifti  as lnrn
from communifti import lib_nibabel_write_nifti as lnwn

from . import lib_ml_models      as lmm
from . import lib_nibabel_utils  as lnu


# ============================================================================

class VnetTestObj:
    """Object for running a Vnet model on a test dataset.

Parameters
----------
inset : str
    name of input (anatomical) dset
prefix : str
    file name for the estimated/calculated pred_mask 
checkpoint : str
    name of a checkpoint file to use, i.e., the trained model to apply
device : str
    a keyword for the device to use, primarily either CPU or GPU (=cuda);
    there are only a few specific values this can take, see
    lib_vnet_defs.LIST_all_device
do_overwrite : bool
    should the outputs here be able to overwrite pre-existing dsets?
verb : int
    verbosity level

Returns
-------
VnetTestObj : obj
    this object with all its bells and whistles; but importantly,
    during the runtime this object will output a mask file, of name
    'prefix'

    """

    def __init__(self, inset, prefix='mask_new.nii.gz', 
                 checkpoint=None, device='cpu', 
                 do_overwrite=False, verb=1):

        # ----- set up attributes

        # main input variables
        self.status           = 0                  # not used

        # main input variables
        self.inset            = inset
        self.prefix           = prefix

        self.checkpoint       = checkpoint
        self.device           = device
        
        # data loaded in
        self.data_orig        = None              # from inset arr
        self.hdr_orig         = None              # nifti header of inset
        self.data_mask        = None              # from mask arr

        # data calculated
        self.data_pred_mask   = None              # the mask Vnet predicts

        # general variables
        self.verb             = verb
        self.do_overwrite     = do_overwrite

        # ----- take action(s)

        tmp = self.basic_setup()
        if tmp : return

        tmp = self.load_data()
        if tmp : return

        tmp = self.load_model()
        if tmp : return

        tmp = self.run_model()
        if tmp : return

        tmp = self.write_output()
        if tmp : return

    # ----- methods

    def run_model(self):
        """Run the vnet: calc data_pred_mask"""

        # some things that won't work here:
        # + not useful bc no nn.Linear (in Conv3d)
        #   torch.quantization.quantize_dynamic(self.model, 
        #                                      {torch.nn.Linear},
        #                                      dtype=torch.qint8)
        # + 
        #   memory_format=torch.channels_last

        print("++ Run vnet", flush=True)
        
        # get correct shape for tensor
        orig_data = self.data_orig.unsqueeze(1).to(self.device)

        self.model.to(self.device)
        device_model = next(self.model.parameters()).device

        print("HEY: data_orig shape:", orig_data.shape)
        print("HEY DEVICE DATA :", orig_data.device, flush=True)
        print("HEY DEVICE MODEL:", device_model, flush=True)

        # useful for inference
        self.model.eval()

        # this mode is like torch.no_grad(), saves more memory, and is
        # fine since we will not use grads later
        with torch.inference_mode():
            # main calculation: estimate the mask with vnet
            self.data_pred_mask = self.model.forward(orig_data)

        return 0

    def write_output(self):
        """Save pred_mask to disk"""

        ab.IP("Writing out pred_mask to file: {}".format(self.prefix))

        # convert torch.Tensor to np.array
        arr   = (self.data_pred_mask[0][1]).cpu().detach().numpy()

        lnwn.BabelNiftiWrite( arr, self.hdr_orig, self.prefix,
                              map_rules    = "afni_rules",
                              do_overwrite = self.do_overwrite,
                              do_rm_exts   = True, 
                              verb         = self.verb )

        return 0

    def load_data(self):
        """Load datasets/volumes into arrays.  The specific loading function
        for each dset contains more description, but in general dsets
        get stored as torch tensors, with dimensionality setup for
        running in the model.
        """

        BAD_RETURN = -1

        # load main input dset
        is_fail = self.load_inset()
        if is_fail :
            return BAD_RETURN

        return 0

    '''  ## NOT using right now
    def load_comp_mask(self):
        """Read in comparison mask data (if present), using nibabel.

        The 3D dset is stored as a torch tensor array.  In order to be
        used as an input to the model, it is also unsqueezed to insert
        an extra dim in the [0]th index, so 3D data of dim [A, B, C] ->
        [1, A, B, C].
        """

        BAD_RETURN = -1

        if self.have_comp_mask :
            mask_image = nib.load(self.comp_mask)
            mask_data  = np.asanyarray(mask_image.dataobj).astype('float32')

            self.data_comp_mask = torch.from_numpy(mask_data).unsqueeze(0)

        return 0
'''    

    def load_inset(self):
        """Read in inset anatomical dset, using nibabel, and do things like
        percentile-based thresholding and Z-scoring of it.  The header
        is also stored separately, to help with writing out data.

        The 3D dset is stored as a torch tensor array.  In order to be
        used as an input to the model, it is also unsqueezed to insert
        an extra dim in the [0]th index, so 3D data of dim [A, B, C] ->
        [1, A, B, C].

        """

        BAD_RETURN = -1

        # read NIFTI to tmp data array (needs proc) and header obj
        is_fail, orig_data, self.hdr_orig = \
            lnrn.read_nifti_to_nibabel(self.inset, set_dtype=np.float32, 
                                       verb=self.verb)
        if is_fail :
            ab.EP1("Could not read in NIFTI: {}".format(self.inset))
            return BAD_RETURN 

        # simple proc 1: percentile-based thresholding of data
        top99_thresh = np.percentile(orig_data, 99) 
        orig_data[orig_data >top99_thresh] = top99_thresh
        down2_thresh = np.percentile(orig_data, 2) 
        orig_data[orig_data < down2_thresh] = down2_thresh

        # simple proc 2: z-score conversion
        is_fail, orig_data = z_scoring(orig_data)
        if is_fail :
            return BAD_RETURN 

        self.data_orig = torch.from_numpy(orig_data).unsqueeze(0)   

        return 0

    def load_model(self):
        """Make announcements, verify that datasets exist"""

        ab.IP("Using device: {}".format(self.device))

        # some of this containment of threads is useful, esp. on macOS
        # *** add opt to control this from command line
        if platform.system() == 'Darwin':
            torch.set_num_threads(4)

        self.model = lmm.VNet_orig(in_channels = 1, 
                                   num_class   = 2,
                                   wt_norm     = 0, 
                                   verb        = self.verb)

        tload = torch.load(self.checkpoint,
                           map_location = torch.device(self.device))

        self.model.load_state_dict(tload, strict=False)

        return 0                           

    def basic_setup(self):
        """Verify that datasets and other input choices exist"""

        # check basic requirements

        ab.IP("Prepare vnet")

        # (req)
        if not(self.inset) :
            ab.EP("Need to provide an inset")
        else:
            nfail = au.check_all_dsets_exist([self.inset], label='inset', 
                                             verb=self.verb)
            if nfail :
                ab.EP("Failed to load inset")

        # (req) checkpoint --- ** at some point have a default choice
        if self.checkpoint :
            is_ok = os.path.isfile(self.checkpoint)
            if not(is_ok) :
                ab.EP("Failed to load checkpoint")

        # (req)
        if not(self.prefix) : 
            ab.EP("Need to provide a prefix")

        if os.path.isfile(self.prefix) and not(self.do_overwrite) :
            msg = "Output dset exists: '{}'\n".format(self.prefix)
            msg+= "Either active overwriting, or move/remove dset"
            ab.EP(msg)

        return 0

    # ----- decorators

    ##@property
    ##def have_comp_mask(self):
    ##    """was a comparison mask input? return 0 for no and 1 for yes"""
    ##    if self.comp_mask : return 1
    ##    else:               return 0

# ----------------------------------------------------------------------------

def z_scoring(img):
   
    BAD_RETURN = (-1, np.ndarray(0))

    data_mean  = img.mean()
    data_std   = img.std()

    if data_std :
        Z_normalized = (img - data_mean) / data_std
    else:
        ab.EP1("std dev of img dset was zero?")
        return BAD_RETURN

    return 0, Z_normalized


# ============================================================================

if __name__ == "__main__" :

    # an example use case
    print("++ No example")

