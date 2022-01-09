import nibabel as nib
import numpy   as np
import copy

### some comments here about I/O
# https://stackoverflow.com/questions/44397617/change-data-type-in-numpy-and-nibabel/45589431

### **To do: remove an NIFTI header extensions, too

fname_dsetA = 'pac_755_mask.nii.gz'

dsetA = nib.load(fname_dsetA)
headA = dsetA.header

headA_cp = copy.deepcopy(headA)
headA_cp['datatype'] = 16        # output will be float? 


# oddly, datatypeA is type() np.ndarry, but has not shape() or len()
# properties;  however, it appears to be treatable as a scalar (int)
datatypeA = headA['datatype']

# currently, each dset is read in as type float32---the data from
# Francisco+Patrick is all byte, actually, prob for memory
# conservation?
arrA_fl   = np.asanyarray(dsetA.dataobj).astype('float32')
arrA_byte = np.asanyarray(dsetA.dataobj).astype('byte')
# NB: putting 'byte' instead of 'byte' above produces a dset of dtype=int8

# ------------------ make a new dset from this one ------------------

if headA['sizeof_hdr'] == 348:
    print("++ I am NIFTI-1 format")
    # I put 'None' in the [1]th because that affine value can/should
    # be in the header in the [2]th arg?
    dsetB_fl   = nib.Nifti1Image(arrA_fl, None, header=headA)
    dsetB_flcp = nib.Nifti1Image(arrA_fl, None, header=headA_cp)
    dsetB_fl10 = nib.Nifti1Image(arrA_fl*10, None, header=headA)
    dsetB_byte = nib.Nifti1Image(arrA_byte, None, header=headA)
elif headA['sizeof_hdr'] == 540:
    print("++ I am NIFTI-2 format")
    dsetB_fl   = nib.Nifti2Image(arrA_fl, None, header=headA)
    dsetB_flcp = nib.Nifti2Image(arrA_fl, None, header=headA_cp)
    dsetB_fl10 = nib.Nifti2Image(arrA_fl*10, None, header=headA)
    dsetB_byte = nib.Nifti2Image(arrA_byte, None, header=headA)
else:
    raise IOError('Input image header problem')

headB_fl       = dsetB_fl.header
datatypeB_fl   = headB_fl['datatype']
headB_flcp     = dsetB_flcp.header
datatypeB_flcp = headB_flcp['datatype']
headB_fl10     = dsetB_fl10.header
datatypeB_fl10 = headB_fl10['datatype']
headB_byte     = dsetB_byte.header
datatypeB_byte = headB_byte['datatype']

print("++ datatype of input dsetA      : {}".format( datatypeA ))
print("++ datatype of input dsetB_fl   : {}".format( datatypeB_fl ))
print("++ datatype of input dsetB_flcp : {}".format( datatypeB_flcp ))
print("++ datatype of input dsetB_fl10 : {}".format( datatypeB_fl10 ))
print("++ datatype of input dsetB_byte : {}".format( datatypeB_byte ))


nib.save(dsetA,      'outnew_A.nii.gz')
nib.save(dsetB_fl,   'outnew_B_fl.nii.gz')
nib.save(dsetB_flcp, 'outnew_B_flcp.nii.gz')
nib.save(dsetB_fl10, 'outnew_B_fl10.nii.gz')
nib.save(dsetB_byte, 'outnew_B_byte.nii.gz')




