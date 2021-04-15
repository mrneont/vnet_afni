# supplementary function around nibabel

import nibabel as nib


### Dset orientation notes, mostly from here:
### https://nipy.org/nibabel/image_orientation.html
#
# + An LPI dset in AFNI (one whose xcoords go from L->R, ycoords from
#   P->A and zcoords from I->S) would be called RAS (or 'RAS+') in
#   nibabel
#
# + "By convention, nibabel world axes are always in RAS+"
 
# Dictionary to convert between AFNI and nibabel orientation labeling.
# The same dictionary would map AFNI->nibabel and nibabel->AFNI
dict_flip_orient = { 'R' : 'L',
                     'L' : 'R',
                     'A' : 'P',
                     'P' : 'A',
                     'I' : 'S',
                     'S' : 'I'}

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
