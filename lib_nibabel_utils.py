# supplementary function around nibabel
import os
import numpy   as np
import nibabel as nib





### Nibabel dset orientation notes, mostly from here:
### https://nipy.org/nibabel/image_orientation.html
#
# + An LPI dset in AFNI (one whose xcoords go from L->R, ycoords from
#   P->A and zcoords from I->S) would be called RAS (or 'RAS+') in
#   nibabel
#
# + "By convention, nibabel world axes are always in RAS+"
#
### Nibabel [q,s]form_code comments
# 
# + For some reason, nibabel defaults are:
#        qform_code = 0
#        sform_code = 2
#
# --------------------------------------------------------------------------

# Dictionary to convert between AFNI and nibabel orientation labeling.
# The same dictionary would map AFNI->nibabel and nibabel->AFNI
dict_flip_orient = { 'R' : 'L',
                     'L' : 'R',
                     'A' : 'P',
                     'P' : 'A',
                     'I' : 'S',
                     'S' : 'I'}

# --------------------------------------------------------------------------





def write_out_nifti_vol( arr3d, fname='dset.nii.gz', outdir=None,
                         head=None,
                         aorient=None, affmat=np.eye(4),
                         sform_code=1, qform_code=1):
    """Write out a 3D array arr3d as a NIFTI dataset.  This function uses
    nibabel under the hood, but inputs are in standard AFNI convention.


    Params
    ------
    arr3d       : (3D array or pytorch tensor) dset to be written out 
                  as a NIFTI vol
    fname       : (str) output filename, which can include path 
    outdir      : (str) optional way to provide output dir path (could just
                  be as part of fname, as well)
    head        : (nibabel NIFTI header) header from nibabel; will get
                  precedence over remaining kwargs, which are header elements
    aorient     : (str, len=3) orientation of dset to be output, in standard
                  AFNI interpretation (nibabel's is opposite of this)
                  **NOT USED RIGHT NOW: affmat CONTROLS THIS**
                  **STILL NEED TO FIGURE OUT IF/HOW TO USE THIS**
    affmat      : (arr, size=(4,4)) the affine matrix of the dset,
                  determining voxelsize and any obliquity; what
                  nibabel.Nifti1Image() calls 'affine'; but if the head 
                  kwarg is used, that gets precedence
    sform_code  : (arr, size=unsized; putting in int is fine) default is 1,
                  for orig space
    qform_code  : (arr, size=unsized; putting in int is fine) default is 1,
                  for orig space


    Returns
    -------
    1 on success (NIFTI dataset written to disk), 0 on failure (not writing 
    dset).  If head is used, some parts of it will still be overwritten, 
    to make sure the output header is consistent with the data itself

    """

    if fname.__contains__('/') and outdir != None :
        print("** ERROR: can't both provide path in fname AND use outdir")
        return 0
    elif outdir :
        if outdir[-1] == '/' :
            fname = outdir + fname
        else:
            fname = outdir + '/' + fname

    if head :
        # temp header, to get a few pieces of info
        tmp = nib.Nifti1Image(arr3d, np.eye(4))

        head['datatype']  = tmp.header['datatype']    # to get from arr dtype
        head['scl_slope'] = tmp.header['scl_slope']   # should be non-info
        head['scl_inter'] = tmp.header['scl_inter']   # should be non-info
        head['dim']       = tmp.header['dim']         # to make based on arr
        head['extents']   = tmp.header['extents']     # should be empty
        head['cal_min']   = 0                         # nullify
        head['cal_max']   = 0                         # nullify
        head.extensions.clear()                       # remove extensions

        ovol = nib.Nifti1Image(arr3d, affine=None, header=head)
    else:
        ovol = nib.Nifti1Image(arr3d, affine=affmat)
        ovol.header['qform_code'] = qform_code
        ovol.header['sform_code'] = sform_code

    nib.save(ovol, fname)

    return 1






def make_names_of_dsets(outdir, orig_fname, strepoch, phase):
    """Create a series of informative (not brief!) names for outputting
    datasets.

    Parameters
    ----------
    outdir   : (str)
               rel or abs path to output directory
    strepoch : (str)
               epoch number
    phase    : (str)
               which phase of processing are we in, such as 
               'val', 'train', etc.
    idx      : (str) 
               related to index of dataset

    Return
    ------
    fname_targ : (str)
               output name of the target dataset
    fname_pred_ch00_back : (str)
               output name of the background channel dataset
    fname_pred_ch00_fore : (str)
               output name of the foreground channel dataset

    """

    # removing the file_extension "nii.gz" to access the subject id
    subj_id = orig_fname.rsplit( ".", 2 )[0]

    pref_orig = "{}_{}_{}_{:s}".format( 'orig', phase, 'subj', subj_id[:-5])

    # subj_id[:-5] removes the suffix "_orig"
    fname_orig      = "{}/{}.nii.gz".format( outdir,pref_orig)

  
    pref_targ  = "{}_{}_{}_{}_{:s}".format( 'target', strepoch, phase,'subj', subj_id[:-5])

    fname_targ = "{}/{}.nii.gz".format( outdir, pref_targ )

    channel_suffix = "{}_{}_{}-{:s}".format(strepoch, phase, 'subj', subj_id[:-5])

    #background channel->00
    pref_pred_ch00_back  = "{}-{}".format('ch00_ep', channel_suffix)

    fname_pred_ch00_back = "{}/{}.nii.gz".format( outdir, pref_pred_ch00_back )

    #foreground channel->01
    pref_pred_ch01_fore  = "{}-{}".format('ch01_ep', channel_suffix)

    fname_pred_ch01_fore = "{}/{}.nii.gz".format( outdir, pref_pred_ch01_fore )

    return fname_orig, fname_targ, fname_pred_ch00_back, fname_pred_ch01_fore















# =========================================================================
# deal with dset orientation issues

def ISVALID_orient(X):
    """Check if orient is valid combination of letters.

    Return 1 if it is, or 0 if it isn't."""

    if type(X) != str :
        print("** ERROR: Invalid type input.  Must be str, not", type(X))
        return 0
    if len(X) != 3 :
        print("** ERROR: Invalid len of str.  Must be 3, not", len(X))
        return 0

    osum = np.zeros(3, dtype=int)

    osum[0]+= X.__contains__('R') + X.__contains__('L') 
    osum[1]+= X.__contains__('A') + X.__contains__('P') 
    osum[2]+= X.__contains__('I') + X.__contains__('S') 

    for i in range(3):
        if osum[i] != 1:
            return 0

    return 1


def orient_perm_rai_to_other(X):
    """Calculate a 3x3 permutation matrix to go from RAI to another
    orientation Y (which is a len=3 str).  

    This forms the basis of more general permutations. See:
    orient_perm_A_to_B().

    Return 3x3 perm array.

    """

    P = np.zeros((3,3), dtype=int)

    if not(ISVALID_orient(X)) :
        print("** ERROR: invalid orientation str:", X)
        return P

    W = 'RAI'
        
    for i in range(3):
        for j in range(3):
            if W[i] == X[j] :
                P[i, j] = 1
            elif dict_flip_orient[W[i]] == X[j] :
                P[i, j] = -1

    return P


def orient_perm_A_to_B(A, B):
    """Calculate a 3x3 permutation matrix to go from orientation A to
    another orientation B (each is a len=3 str).

    Return 3x3 perm array.

    """

    PA = orient_perm_rai_to_other(A)
    PB = orient_perm_rai_to_other(B)

    PAINV = np.linalg.inv(PA)

    PAB = np.zeros((3, 3), dtype=int) 

    for i in range(3):
        for j in range(3):
            for k in range(3):
                PAB[i, j]+= PAINV[i, k] * PB[k, j]

    return PAB


def flip_orient(ostr):
    """Flip the orientation string ostr, for example from 'RAI' to 'LPS'.

    This is useful for converting between, say, AFNI and nibabel
    namings of dset orientation (converts both AFNI to nibabel, and
    vice versa).

    Params
    ------

    ostr :   3 char string of dset orient

    Return
    ------

    ostr_flip : flipped, 3 char string of dset orient.

    """

    if type(ostr) != str :  return ""
    if len(ostr)  != 3   :  return ""
    
    achar = ''.join(dict_flip_orient)

    for c in ostr:
        if not(achar.__contains__(c)) :
            print("** ERROR: unrecognized char '{}' in orient str".format(c))
            return ""

    ostr_inv = "".join([dict_flip_orient[c] for c in ostr])

    return ostr_inv


def afni_orient_nib_dset(x):
    """Get the orientation from a nibabel dset object, and provide it in
    AFNI format ('LPI' means that xcoord goes from L->R, etc.), which
    appears to be opposite of nibabel notation.

    Params
    ------

    x : nibabel dset obj

    Return
    ------
    
    orient : string orient (AFNI-style), like LPI

    """

    nib_orient  = ''.join(nib.aff2axcodes(x.affine))
    afni_orient = flip_orient(nib_orient)

    return afni_orient


def reorient_mat44(M44A, ocharA, ocharB):
    """Take a 4x4 array (e.g., a full affine matrix) and its orientation
string ocharA (e.g., 'LAI'), and reorient it to a new orientation
(e.g., 'SRA').

    Parameters
    ----------
    
    M44A    : (array, dim=(4,4)) affine matrix, comprised of a 3x3 
              submatrix, a 3x1 origin vector, and the bottom row is a 1x4
              vector (0,0,0,1).
    ocharA  : (str, len=3) orientation of input matrix
    ocharB  : (str, len=3) orientation of output matrix


    Returns
    -------

    M44B  : (array, dim=(4,4)) output array that has been permuted

    """

    P33  = orient_perm_A_to_B( ocharA, ocharB )
    
    M44B = np.zeros((4, 4), dtype=float)

    M44B[:3, :3] = np.matmul(P33, M44A[:3, :3])
    M44B[:3, 3]  = np.matmul(P33, M44A[:3, 3])
    M44B[3, 3]   = 1

    return M44B



# --------------------------------------------------------------------------
# This is TEMPORARY way to give some correct header info to the output
# dsets we have for 32x32x32 testing at the moment.  Later, we will
# use some of the below functions to pass along that info from the
# input.

# get what orient appears to be to nibabel (it is 'RSP' in AFNI)
TMP_32iso_orient_nib     = 'RSP' #flip_orient('RSP') 

# What the 8mm iso datasets appear to have as RAI header info:
TEMP_M44_32iso_RAI       = np.eye(4)*8
#TEMP_M44_32iso_RAI[:3,3] = np.array([-126.607697, -132.524399, -105.403000])
TEMP_M44_32iso_RAI[:3,3] = np.array([-124.947701, -130.799896, -123.744102])
TEMP_M44_32iso_RAI[3,3]  = 1

# ... get it in RSP, which is what all inputs appear to be.
TEMP_M44_32iso_nib_ori = reorient_mat44(TEMP_M44_32iso_RAI, 
                                        'LPI', 
                                        TMP_32iso_orient_nib)

# --------------------------------------------------------------------------


# write the target masks into output directory 
# [PT] starting to translate this to having more correct header info.
#    + pieces are still hardwired now, but these should overlay now
def write_tensor_to_disk_nifti(tt, fname=None, head=None):
    """This function writes a torch tensor volume to disk as a NIFTI file.

    Inputs
    ------
    tt               : (torch.Tensor) 3D volume
    fname            : (str) full path+name of output dset (build name
                       before using this func).    
    head             : (nibabel NIFTI header) header from nibabel

    """ 

    if not(fname) :
        print("** ERROR: need fname for writing out tensor to nifti.")
        sys.exit(7)

    # convert torch.Tensor to np.array
    arr   = tt.cpu().detach().numpy()

    if head :
        write_out_nifti_vol( arr, fname,
                             head=head )
    else:
        write_out_nifti_vol( arr, fname,
                             affmat = TEMP_M44_32iso_nib_ori )

