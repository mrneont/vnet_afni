#!/usr/bin/env python

import os
import platform

# this needs to be done before torch import
os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'   # 128 KB

import torch
import numpy                     as np
import nibabel                   as nib

from afnipy import afni_base      as ab
from afnipy import afni_util      as au
from afnipy import lib_torch_util as ltu

from communifti import lib_nibabel_read_nifti  as lnrn
from communifti import lib_nibabel_write_nifti as lnwn

from . import lib_ml_models      as lmm
from . import lib_nibabel_utils  as lnu

# ============================================================================
# NOTES

### memory format
# channels_last_3d is the 3D equivalent of channels_last.  It tells
# PyTorch to store feature maps as (N, D, H, W, C) internally, which
# maps conveniently on macOS ARM (esp. with MPS kernels).  See docs:
# https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html

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
num_cpu : int
    integer to specify number of CPUs to use, via OMP_NUM_THREADS;
    negative value means to stay with system default
device : str
    keyword for the device: 'cpu', 'mps', 'cuda', or 'auto'.
    'auto' selects MPS on Apple Silicon, otherwise CPU.
do_overwrite : bool
    should the outputs here be able to overwrite pre-existing dsets?
do_compile : bool
    wrap the model with torch.compile() (PyTorch ≥ 2.0).
    Adds a one-time compilation cost but speeds up repeated forward passes.
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
                 checkpoint=None, device='auto', num_cpu=-1,
                 do_overwrite=False, do_compile=False, verb=1):

        # ----- set up attributes

        # main input variables
        self.inset            = inset
        self.prefix           = prefix

        self.checkpoint       = checkpoint
        self.device           = device         # checked (maybe changed) below
        self.num_cpu          = num_cpu        # number of CPUs, if >0
        self.sysname          = None           # platform system

        self.data_orig        = None
        self.hdr_orig         = None
        self.data_mask        = None

        # to be calculated
        self.data_pred_mask   = None

        # general variables
        self.verb             = verb
        self.do_overwrite     = do_overwrite
        self.do_compile       = do_compile
        self.status           = 0                  # not used

        # ----- take action(s)

        tmp = self.basic_setup()
        if tmp : return

        tmp = self.set_cpus()
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
        """Run the vnet: calc data_pred_mask.
        
        NB: at the moment we simply switch between ~optimizing options
        for macoS-ARM and Linux-Intel.  Will work further to unify
        opts, or split this function more cleanly.
        """

        ab.IP("Run vnet on device: {}".format(self.device))

        if self.sysname == "Darwin" :
            # particularly efficient mem format on macOS-mps (see NOTES)
            mem_fmt = torch.channels_last_3d

            orig_data = (self.data_orig
                         .unsqueeze(1)
                         .to(self.device, non_blocking=True)   # async copy
                         .to(memory_format=mem_fmt))

            self.model.to(self.device)
            # keep wts contiguous in same layout
            self.model.to(memory_format=mem_fmt)

            self.model.eval()

            # autocast: on MPS this runs the forward pass in float16, halving
            # memory bandwidth and giving a free speed boost at negligible
            # accuracy cost for inference.  On CPU it is a no-op (float32).
            ac_device = 'mps' if self.device == 'mps' else 'cpu'
            ac_dtype  = torch.float16 if self.device == 'mps' else torch.float32

            with torch.inference_mode():
                # because autocast might be used, convert back to float32 below;
                # *** decide if this only applies of self.device=='mps' 
                #     not cuda ***;
                # *** and decide about casting back to float32 here ***
                with torch.autocast(device_type=ac_device, dtype=ac_dtype,
                                     enabled=(self.device == 'mps')):
                    self.data_pred_mask = self.model.forward(orig_data)

        else:
            # get correct shape for tensor
            orig_data = self.data_orig.unsqueeze(1).to(self.device)

            self.model.to(self.device)
            device_model = next(self.model.parameters()).device

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

        ab.IP("Writing out pred_mask to file:\n{}".format(self.prefix))

        # convert torch.Tensor to np.array
        arr = (self.data_pred_mask[0][1]
               .to(torch.float32)      # cast back from fp16 if autocasting
               .cpu()
               .detach()
               .numpy())

        lnwn.BabelNiftiWrite( arr, self.hdr_orig, self.prefix,
                              map_rules    = "afni_rules",
                              do_overwrite = self.do_overwrite,
                              do_rm_exts   = True,
                              verb         = self.verb )

        return 0

    def load_data(self):
        """Load datasets/volumes into arrays.  

        The specific loading function
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
        orig_data[orig_data > top99_thresh] = top99_thresh
        down2_thresh = np.percentile(orig_data, 2)
        orig_data[orig_data < down2_thresh] = down2_thresh

        # simple proc 2: z-score conversion
        is_fail, orig_data = z_scoring(orig_data)
        if is_fail :
            return BAD_RETURN

        self.data_orig = torch.from_numpy(orig_data).unsqueeze(0)
        return 0

    def set_cpus(self):
        """Set how many CPUs to use. There are different ways to specify; can
        also do nothing and just let system decide.
        """

        if self.num_cpu > 0 :
            # user-specified route

            torch.set_num_threads(self.num_cpu)
            torch.set_num_interop_threads(self.num_cpu)

            if self.verb:
                ab.IP("User opt: using {} CPU thread(s)".format(self.num_cpu))

            return 0

        if self.sysname == 'Darwin':
            # if on macOS: estimate based on number of performance cores

            # M-series chips have 4–12 performance cores; use them all.
            # torch.get_num_threads() respects PYTORCH_CPU_ALLOC_CONF if set,
            # so only override when the user has not already done so.
            n_perf_cores = _count_arm_perf_cores()
            torch.set_num_threads(n_perf_cores)

            if self.verb:
                ab.IP("macOS: using {} CPU thread(s)".format(n_perf_cores))

            return 0

        # default
        num_threads = torch.get_num_threads()
        if self.verb:
            ab.IP("Default: using {} CPU thread(s)".format(num_threads))

        return 0

    def load_model(self):
        """Make announcements, verify that datasets exist; 
        optionally compile the model."""

        ab.IP("Using device: {}".format(self.device))

        self.model = lmm.VNet_orig(in_channels = 1, 
                                   num_class   = 2,
                                   wt_norm     = 0, 
                                   verb        = self.verb)

        tload = torch.load(self.checkpoint,
                           map_location = torch.device(self.device))

        self.model.load_state_dict(tload, strict=False)

        if self.do_compile:
            is_fail, can_compile = ltu.torch_can_compile(verb=self.verb)
            if is_fail :
                return BAD_RETURN
            
            if can_compile :
                self.model = torch.compile(self.model)

        return 0

    def basic_setup(self):
        """Verify that datasets and other input choices exist."""

        BAD_RETURN = 2

        ab.IP("Prepare vnet")

        # (req) input dset
        if not self.inset:
            ab.EP1("Need to provide an inset")
            return BAD_RETURN
        else:
            nfail = au.check_all_dsets_exist([self.inset], label='inset',
                                             verb=self.verb)
            if nfail:
                ab.EP1("Failed to load inset")
                return BAD_RETURN

        # (req) checkpoint *** at some point, have a default choice
        if self.checkpoint:
            if not(os.path.isfile(self.checkpoint)) :
                ab.EP1("Failed to load checkpoint")
                return BAD_RETURN

        # (req)
        if not(self.prefix) :
            ab.EP1("Need to provide a prefix")
            return BAD_RETURN

        if os.path.isfile(self.prefix) and not(self.do_overwrite) :
            msg = "Output dset exists: '{}'\n".format(self.prefix)
            msg+= "Either activate overwriting, or move/remove dset"
            ab.EP1(msg)
            return BAD_RETURN

        # store platform system name 
        self.sysname = platform.system()

        # setup device (may replace user-entered parameter
        is_fail, self.device = ltu.select_device_general(dev_in=self.device,
                                                         verb=self.verb)
        if is_fail :
            ab.EP1("Failed select device")
            return BAD_RETURN

        return 0

    # ----- decorators

    ## none at present

# ----------------------------------------------------------------------------

def _count_arm_perf_cores() -> int:
    """Return the number of performance cores on Apple Silicon.

    Uses sysctl if available (macOS); falls back to logical CPU count.
    On M1 that's 4 P-cores; on M1 Pro/Max/Ultra it's 8–16.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ['sysctl', '-n', 'hw.perflevel0.logicalcpu'],
            stderr=subprocess.DEVNULL
        )
        return max(1, int(out.strip()))
    except Exception:
        return max(1, os.cpu_count() or 4)


def z_scoring(img):

    BAD_RETURN = (-1, np.ndarray(0))

    data_mean = img.mean()
    data_std  = img.std()

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

