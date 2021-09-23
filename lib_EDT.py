import sys, copy
import numpy             as np
import matplotlib.pyplot as plt
import numpy as np

BIG = 10**10    # -> assumes max matrix dimension is < 10**5

# NB: should make the testing criterion for is "big" big enough use
# the voxel/pixel edges, too, because that could theoretically matter

# ==========================================================================
#
# This code calculates the Euclidean Distance Transform (EDT) for 3D
# volumes following this nice, efficient algorithm, by Felzenszwalb
# and Huttenlocher (2012;  FH2012):
#
#   Felzenszwalb PF, Huttenlocher DP (2012). Distance Transforms of
#   Sampled Functions. Theory of Computing 8:415-428.
#   https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf
#
# Another useful/illustrative resource abotu this is by Philip Rideout:
#
#   https://prideout.net/blog/distance_fields/
#
# The current code here extends/tweaks the FH2012 algorithm to a more
# general case of having several different ROIs present, for running
# in 3D (trivial extension), and for having voxels of non-unity and
# non-isotropic lengths.  It does this by utilizing the fact that at
# its very heart, the FH2012 algorithm works line by line and can even
# be thought of as working boundary-by-boundary.
#
# Here, the zero-valued "background" is also just treated like an ROI,
# with one difference.  At a FOV boundary, the zero-valued
# ROI/backgroud is treated as open, so that the EDT value at each
# "zero" voxel is always to one of the shapes within the FOV.  For
# nonzero ROIs, one can treat the FOV boundary *either* as an ROI edge
# (EDT value there will be 1 edge length) *or* as being open.
#
# written by:  PA Taylor (NIH)
#
# ==========================================================================

#version = '1.1' ; date = 'June 30, 2021'
# [PT] cleaned up code, removed older/simpler functions
#    + added in plotting function and more checks
#    + deal with some special situations of constant inputs
#    + even put in doc/signatures per function (whoa)
#
#version = '1.2' ; date = 'July 1, 2021'
# [PT] rename, simplify/clarify parts
#
version = '2.0' ; date = 'July 2, 2021'
# [PT] fairly major change: redo how Euclidean_DT_delta() works, to
#      accommodate non-unity length edges (i.e., non isotropic
#      voxels/pixels of edgelen=1).  
#    + Some instability had occurred in some cases, but that seems
#      fixed with current mechanism---staying closer to the original
#      alg with a temporary rescaling
#    + Now, we also go through by order of decreasing edge length---this
#      might be unnecessary, though.
#    + Also, removing "dist-to-surf-of-ROI" calc temporarily---needs more
#      work/checking.
#    + Also, fixed 2D calc (edgelengths were assigned to wrong dims)
#    + ** but haven't switched 2D calc to working in order of voxel dims!
#
version = '2.1' ; date = 'July 2, 2021'
# [PT] tweak plotting
#
# ==========================================================================

# --------------------------------------------------------------------------
# supplementary functions to check various dset features and sort, etc.

def check_matrix_dims(im):
    '''BIG must be >= max(dim)**2.  Check that here.  

    Return 0 if dimensions are OK, 1 otherwise.
    '''

    max_SH = max(np.shape(im))
    if max_SH >= BIG**0.5 :
        print("** ERROR: this dset has too large a matrix dimension\n"
              "   ({}) for the current BIG value ({}).  For this dset,\n"
              "   BIG must be >= {}**2 = {}.\n"
              "".format(max_SH, BIG, max_SH, max_SH**2))
        return 1

    return 0

def check_matrix_types(im) :
    '''Input dset must be an array (not a list) because of indexing.  Make
    sure each element is an int.

    Return 0 if types are OK, 1 otherwise.
    '''

    BADNESS = 0

    if type(im) != np.ndarray :
        BADNESS+= 1
        print("** ERROR: input must be of type np.array (not {})"
              "".format(type(im)))
    #if not isinstance(im.dtype, (int, np.integer)) :
    if not [int, np.int32].__contains__(im.dtype) :
    #if im.dtype != int :
        BADNESS+= 1
        print("** ERROR: input must have dtype=int (not {}), "
              "since this is a map of ROIs"
              "".format(im.dtype))

    if BADNESS :
        return 1

    return 0

def sort_perm_1D(x, reverse=False):
    '''Calculate the permutation of indices needed to sort the 1D ordered
    collection 'x'. Default is ascending order (reverse=False); can
    have descending order (reverse=True).  

    Returns a tuple of the permutation indices (same length as input).

    '''

    L = len(x)
    X = [ (x[i], i) for i in range(L)]
    X.sort(reverse=reverse)
    a, b = zip(*X)
    return b

# --------------------------------------------------------------------------
# EDT functions: split into hierarchy of tasks

def Euclidean_DT_delta(f0, delta = 1.0):
    '''This is a supplementary function for the EDT calcultion (of any
    dimension): perform the classical Euclidean Distance Transform
    (EDT) of Felzenszwalb and Huttenlocher (2012), for any uniform spacing.

    The input 'f0' is a 1D list/array of values representing either:
    + distance**2 values (the case after 1 dimension of the input
      volume has been processed), or
    + values that are either 0 or BIG to express membership in an ROI,
      which is mainly a translation of Eq. 1(q) in F&H2012's
      expression in the middle of page 416 (the case when processing the
      first dimension of the input volume).
    
    Assumes that: len(f) < sqrt(10**10).

    In this version, all elements have equal length along a given
    dimension (but it can vary *across* dimensions), set by 'delta'.
    Distance units are (the square of) whatever units delta has.

    Parameters
    ----------

    f0       : 1D array or list. Either distance**2 values (or, to start,
               values binarized to 0 or BIG).

    delta    : element edge length along this dimension.

    Returns
    -------

    Df       : 1D list of distance**2 values (either updating initial list,
               or providing first distances based on ROI
               membership/boundary).

    '''

    # To deal with anisotropic and non-unity-edge-length elements,
    # first scale non-BIG distances to be "as if" there were edge=1
    # voxels, and then at end scale back.

    # [PT] Comment: unlike in earlier thinking (even before current
    # scale down/up approach), do NOT want to mult 'BIG' by 'delta',
    # because pixels/voxels can be anisotropic here.

    n    = len(f0)
    k    = 0
    v    = [0]*n
    z    = [0.]*(n+1)
    z[0] = -BIG         
    z[1] =  BIG 

    # Scale down, if not using unity element
    f = copy.deepcopy(f0)
    if delta != 1 :
        for i in range(n):
            if f[i] != BIG:
                f[i]/= delta**2

    for q in range(1, n):
        s = (f[q] + q**2) - (f[v[k]] + v[k]**2)
        s/= 2. * (q - v[k])
        while s <= z[k]: 
            k-= 1
            s = (f[q] + q**2) - (f[v[k]] + v[k]**2)
            s/= 2. * (q - v[k])
        k+= 1
        v[k]   = q
        z[k]   = s
        z[k+1] = BIG 

    k   = 0
    Df  = [0]*n
    for q in range(n):
        while z[k+1] < q :
            k+= 1
        Df[q] = (q - v[k])**2 + f[v[k]]

    # Scale back up, if not using unity element
    Df0 = copy.deepcopy(Df)
    if delta != 1 :
        for i in range(n):
            if Df0[i] != BIG:
                Df0[i]*= delta**2

    return Df0


def run_EDTD_per_line( roi_line, dist2_line, 
                       delta = 1.0,
                       bounds_are_zero=True ):
    '''This is a supplementary function for the EDT calcultion (of any
    dimension): get set to run through a full line of the image.

    This function walks through one line of the volume, and passes
    ROI-piece-by-ROI-piece to the EDT algorithm.  This behavior
    (dividing up each ROI interval into a 1D EDT problem) is what
    allows us to have multiple ROIs within one volume efficiently
    here.

    Parameters
    ----------

    roi_line     : (1D list) ROI values, i.e., a map of ROIs along this axis.

    dist2_line   : (1D list or array) distance**2 values.

    delta        : (int or float) element edge length along this dimension.

    bounds_are_zero : (bool) the value determines how to treat ROIs at
                   FOV boundaries:
                   True means to treat the dataset as if it is bounded
                   by all zeros (so the FOV "closes" each nonzero
                   ROI);
                   False means to treat the dataset as if it is
                   bounded by an infinite extension of each ROI (so
                   the FOV is "open" for each ROI.
                   Zero-valued ROIs (= background) are not affected by
                   this value.

    Returns
    -------

    line_out     : 1D list of updated distance** values.

    '''

    Na = len(roi_line)
    idx = 0 

    line_out = np.zeros(Na)

    while idx < Na :
        # Get interval of line with current ROI value
        roi = roi_line[idx]

        n = idx
        while n < Na :
            if roi_line[n] != roi :
                break
            n+=1
        n -= 1
        # n now has the index of last matching element

        # Decide if we need to pad with 0 on either side, represents
        # boundary of another ROI (or zeropadded edge)
        ll    = []
        start = 0
        stop  = None    # for pythonic indexing
        # pad at start?
        if idx != 0 or (bounds_are_zero and roi != 0) :
            ll.append(0)
            start = 1
        # Put actual values from dist**2 field...
        for m in range(idx, n+1): 
            ll.append(dist2_line[m]) 
        # Pad at end?
        if n != Na-1 or (bounds_are_zero and roi != 0) :
            ll.append(0)
            stop = -1

        # Now do dist calc for this interval of line, and save result
        out_1d            = Euclidean_DT_delta(ll, delta=delta)
        line_out[idx:n+1] = out_1d[start:stop]

        idx = n+1

    return line_out


def calc_EDT_2D( im, 
                 do_sqrt = True, bounds_are_zero = True,
                 zeros_are_zeroed = False,
                 edims = (1.0, 1.0) ):
    '''The main function that is called for 2D (= image) EDT calculation.

    See calc_EDT_3D() for the 3D (= volumetric) version.

    Parameters
    ----------

    im           : 2D array (not list, because of index selection within)
                   of dtype=int, containing a map of ROIs.

    do_sqrt      : if True, the output image of EDT values is distance
                   values; otherwise, the values are distance**2 (because
                   that is what the program works with).

    bounds_are_zero : (bool) the value determines how to treat ROIs at
                   FOV boundaries:
                   True means to treat the dataset as if it is bounded
                   by all zeros (so the FOV "closes" each nonzero
                   ROI);
                   False means to treat the dataset as if it is
                   bounded by an infinite extension of each ROI (so
                   the FOV is "open" for each ROI.
                   Zero-valued ROIs (= background) are not affected by
                   this value.

    zeros_are_zeroed : if False, EDT values are output for the
                   zero-valued region; otherwise, zero them out, so EDT
                   values are only reported in the non-zero ROIs. NB: the
                   EDT values are still calculated everywhere; it is just
                   a question of zeroing out later (no time saved for True).

    edims        : (2D tuple) element dimensions (here, pixel edge lengths)


    Returns
    -------

    odt          : 2D array of final distance (or distance**2) values 
                   throughout the FOV (or for the non-zero ROIs only)

    '''

    # Check inputs for usability, and prep size stuff

    SH  = np.shape(im)      # dimensions
    ND  = len(SH)           # number of dimensions

    if ND != 2 :
        print("** ERROR: need to be a 2D array for this function")
        sys.exit(4)

    #if check_matrix_dims(im) or check_matrix_types(im):
    #    sys.exit(5)

    # Special cases: input is all constant (esp. if all zeros)
    if np.min(im) == np.max(im) :
        if np.min(im) == 0 :
            return np.zeros(SH)
        elif not(bounds_are_zero) :
            return np.zeros(SH)

    # Initialize the "output" or answer array
    odt = np.ones(SH)*BIG
  
    # First pass: start with all BIGs
    for ii in range(SH[0]) :
        # get a line...
        aa = im[ii,:]
        # ... and then calc with it, and save results
        odt[ii,:] = run_EDTD_per_line( aa, odt[ii,:],
                                       delta = edims[1],
                                       bounds_are_zero = \
                                           bounds_are_zero )

    # 2nd pass: carry on from first pass's results
    for jj in range(SH[1]) :
        # Get a line...
        aa = im[:,jj] 
        # ... and then calc with it, and save results
        odt[:,jj] = run_EDTD_per_line( aa, odt[:,jj],
                                       delta = edims[0],
                                       bounds_are_zero = \
                                           bounds_are_zero )

    # Zero out EDT values in "zero" ROI?
    if zeros_are_zeroed :
        odt = odt * im.astype(bool) 

    if do_sqrt :
        return np.sqrt(odt)
    else :
        return odt

def calc_EDT_3D( im, 
                 do_sqrt = True, bounds_are_zero = True,
                 zeros_are_zeroed = False,
                 edims = (1,1,1) ):
    '''The main function that is called for 3D (= volume) EDT calculation.

    See calc_EDT_2D() for the 2D (= imagey) version.

    Parameters
    ----------

    im           : 3D array (not list, because of index selection within)
                   of dtype=int, containing a map of ROIs.

    do_sqrt      : if True, the output image of EDT values is distance
                   values; otherwise, the values are distance**2 (because
                   that is what the program works with).

    bounds_are_zero : switch for how to treat FOV boundaries for
                   nonzero-ROIs; True means they are ROI boundaries
                   (so FOV is a "closed" boundary for the ROI), and
                   False means that the ROI continues "infinitely" at
                   FOV boundary (so it is "open").  Zero-valued ROIs
                   (= background) are not affected by this value.

    zeros_are_zeroed : if False, EDT values are output for the
                   zero-valued region; otherwise, zero them out, so EDT
                   values are only reported in the non-zero ROIs. NB: the
                   EDT values are still calculated everywhere; it is just
                   a question of zeroing out later (no time saved for True).

    edims        : (3D tuple) element dimensions (here, pixel edge lengths)


    Returns
    -------

    odt          : 3D array of final distance (or distance**2) values 
                   throughout the FOV (or for the non-zero ROIs only)
    '''

    # Check inputs for usability, and prep size stuff

    SH  = np.shape(im)      # dimensions
    ND  = len(SH)           # number of dimensions

    if ND != 3 :
        print("** ERROR: need to be a 3D array for this function")
        sys.exit(4)

    if check_matrix_dims(im) or check_matrix_types(im):
        sys.exit(5)

    # Special cases: input is all constant (esp. if all zeros)
    if np.min(im) == np.max(im) :
        if np.min(im) == 0 :
            return np.zeros(SH)
        elif not(bounds_are_zero) :
            return np.zeros(SH)

    # Initialize the "output" or answer array
    odt = np.ones(SH)*BIG
  
    # We now go through the axes in order of decreasing element edge
    # length, to avoid pathology in the EDT alg
    vox_ord_rev = sort_perm_1D(edims, reverse=True)
    for nn in vox_ord_rev :
        if nn == 0:
            odt = calc_EDT_3D_dim0( im, odt, SH, edims, 
                                    bounds_are_zero=bounds_are_zero )
        elif nn == 1 :
            odt = calc_EDT_3D_dim1( im, odt, SH, edims, 
                                    bounds_are_zero=bounds_are_zero )
        elif nn == 2 :
            odt = calc_EDT_3D_dim2( im, odt, SH, edims, 
                                    bounds_are_zero=bounds_are_zero )

    # Zero out EDT values in "zero" ROI?
    if zeros_are_zeroed :
        odt = odt * im.astype(bool) 

    if do_sqrt :
        return np.sqrt(odt)
    else :
        return odt

# --------------------------------------------------------------------------
# tiny supplements: so we can work through matrix in order of voxel
# edge length.  Haven't made similar ones for 2D **yet**

def calc_EDT_3D_dim2( im, odt, SH, 
                      edims=(1,1,1), bounds_are_zero=True ):
    '''Small, supplementary piece of the main function calc_EDT_3D().
    Separated out so that we can go through each axis in order of
    descending edge lengths. Part of a set of calc_EDT_3D_dim?()
    funcs, each doing the same thing along different axes.

    '''
    for ii in range(SH[0]) :
        for jj in range(SH[1]) :
            # get a line...
            aa = im[ii,jj,:]
            # ... and then calc with it, and save results
            odt[ii,jj,:] = run_EDTD_per_line( aa, odt[ii,jj,:],
                                              delta = edims[2],
                                              bounds_are_zero = \
                                                  bounds_are_zero )
    return odt


def calc_EDT_3D_dim1( im, odt, SH, 
                      edims=(1,1,1), bounds_are_zero=True ):
    '''Small, supplementary piece of the main function calc_EDT_3D().
    Separated out so that we can go through each axis in order of
    descending edge lengths.  Part of a set of calc_EDT_3D_dim?()
    funcs, each doing the same thing along different axes.

    '''

    for ii in range(SH[0]) :
        for kk in range(SH[2]) :
            # get a line...
            aa = im[ii,:,kk]
            # ... and then calc with it, and save results
            odt[ii,:,kk] = run_EDTD_per_line( aa, odt[ii,:,kk],
                                              delta = edims[1],
                                              bounds_are_zero = \
                                                  bounds_are_zero )
    return odt


def calc_EDT_3D_dim0( im, odt, SH, 
                      edims=(1,1,1), bounds_are_zero=True ):
    '''Small, supplementary piece of the main function calc_EDT_3D().
    Separated out so that we can go through each axis in order of
    descending edge lengths.  Part of a set of calc_EDT_3D_dim?()
    funcs, each doing the same thing along different axes.

    '''

    for jj in range(SH[1]) :
        for kk in range(SH[2]) :
                # get a line...
                aa = im[:,jj,kk] 
                # ... and then calc with it, and save results
                odt[:,jj,kk] = run_EDTD_per_line( aa, odt[:,jj,kk],
                                                  delta = edims[0],
                                                  bounds_are_zero = \
                                                      bounds_are_zero )
    return odt

# --------------------------------------------------------------------------
# some examples for plotting results

def plot_EDT_2D( arr_EDT, arr_ROI,
                 fname     = "IMAGE_2D.svg",
                 panelsize = (3, 4),
                 edims     = (1, 1) ):
    '''Make a plot of the results.  The figure will have 2 columns and 1
    rows: left column is the input ROI map, and right column is the
    output EDT estimates.

    Parameters
    ----------

    arr_EDT      : (2D array) EDT estimates
    
    arr_ROI      : (2D array) ROI map

    fname        : (str) output filename of figure; include extension for
                   format
    
    panelsize    : (2D tuple) dimensions of individual figure panel, each 
                   element in units of inches

    edims        : (2D tuple) element dimensions (here, pixel edge lengths)

    Returns
    -------

    Nothing, but should output the 'fname' image.

    '''

    figsize = (panelsize[0]*2, panelsize[1])

    plt.figure(fname, figsize=figsize)

    SH  = np.shape(arr_EDT)      # dimensions
    ND  = len(SH)                # number of dimensions

    # for colorbar/ticks
    mind    = np.min(edims)
    maxcbar = mind*int(np.max(arr_EDT) / float(mind))

    plt.subplot(1, 2, 1) 
    plt.title("ROIs")
    plt.imshow(arr_ROI.T, 
       interpolation=None, 
       cmap='tab20', 
       origin='lower',
       aspect='equal',
       vmin=0,
       vmax=20,
       extent=(0, SH[0]*edims[0], 0, SH[1]*edims[1]))
    plt.colorbar(orientation='horizontal', shrink=0.9,
                 ticks=[0,10,20])

    ax = plt.subplot(1, 2, 2) 
    plt.title("EDT") 
    plt.imshow(arr_EDT.T, 
       interpolation=None, 
       cmap='gist_earth_r', #'gist_heat_r',
       origin='lower',
       aspect='equal',
       vmin=mind,
       vmax=np.max(arr_EDT),
       extent=(0, SH[0]*edims[0], 0, SH[1]*edims[1]))
    plt.colorbar(orientation='horizontal', shrink=0.9,
                 ticks=np.linspace(mind, maxcbar, 3))

    ax.set_yticklabels([])
            
    plt.tight_layout()
    plt.savefig(fname)


def plot_EDT_3D( arr_EDT, arr_ROI,
                 Nsli      = 8,
                 fname     = "IMAGE_3D.svg",
                 panelsize = (9, 6),
                 edims       = (1, 1, 1) ):
    '''Make a plot of the results.  The figure will have Nsli columns and
    2 rows: top row is the input ROI map, and bottom row is the output
    EDT estimates.

    Parameters
    ----------

    arr_EDT      : (3D array) EDT estimates
    
    arr_ROI      : (3D array) ROI map

    Nsli         : (int) number of slices to show, selected along [2]th axis.
                   A special keyword value "ALL" means that all slices
                   will be shown---could be very large!.
                     
    fname        : (str) output filename of figure; include extension for
                   format
    
    panelsize    : (2D tuple) dimensions of individual figure panel, each 
                   element in units of inches

    edims        : (3D tuple) element dimensions (here, voxel edge lengths)

    Returns
    -------

    Nothing, but should output the 'fname' image.

    '''

    SH  = np.shape(arr_EDT)      # dimensions
    ND  = len(SH)                # number of dimensions

    # set of slices to view
    if Nsli == 'ALL' or Nsli == SH[2] :
        Nsli = SH[2]
        all_sli = np.arange(Nsli)
    else: 
        all_sli = ((np.arange(1, Nsli+1)/float(Nsli+1))*SH[2] - 1).astype(int)

    # for colorbar/ticks
    mind    = np.min(edims)
    maxcbar = mind*int(np.max(arr_EDT) / float(mind))

    figsize = (panelsize[0]*Nsli, panelsize[1]*2)

    plt.figure(fname, figsize=figsize)

    for sli in range(Nsli):
        ax = plt.subplot(2, Nsli, sli+1) 
        plt.title("ROIs (slice {})".format(all_sli[sli]))
        plt.imshow(arr_ROI[:,:,all_sli[sli]].T, 
           interpolation="nearest", # needed for SVG, not: None
           cmap='tab20', 
           origin='lower',
           aspect='equal',
           vmin=0, 
           vmax=20, 
           extent=(0, SH[0]*edims[0], 0, SH[1]*edims[1]))
        if sli == Nsli-1 :
            plt.colorbar(orientation='vertical', shrink=0.9,
                         ticks=[0,10,20])

        if sli: 
            ax.set_yticklabels([])

    for sli in range(Nsli):
        ax = plt.subplot(2, Nsli, Nsli+sli+1) 
        plt.title("EDT") #.format(all_sli[sli]))
        plt.imshow(arr_EDT[:,:,all_sli[sli]].T, 
           interpolation="nearest", # needed for SVG, not: None
           cmap='gist_earth_r', #'gist_heat_r',
           origin='lower',
           aspect='equal',
           vmin=0, #mind,
           vmax=np.max(arr_EDT),
           extent=(0, SH[0]*edims[0], 0, SH[1]*edims[1]))
        if sli == Nsli-1 :
            plt.colorbar(orientation='vertical', shrink=0.9,
                         ticks=np.linspace(mind, maxcbar, 3))

        if sli: 
            ax.set_yticklabels([])

    plt.tight_layout()
    #plt.show()
    plt.savefig(fname)

# --------------------------------------------------------------------------
# some example ROI arrays to use

def make_3D_ex_00():
    '''Make a 3D array with ROIs (= map of ROIs).  The max ROI value is
    <20, for ease of Python colorbar stuff.

    Returns
    -------

    vol      : (3D array, dtype=int) map of ROIs

    edims    : (1D array, len=3) voxel dims

    '''

    # make a test image: a map of various ROIs
    vol   = np.zeros((80, 40, 20), dtype=int)
    edims = (0.5, 1, 2)  

    LX, LY, LZ = np.shape(vol)

    vol[4:18, 2:8, 4:8]     = 1
    vol[2:8, 4:18, 4:8]     = 4
    vol[21:30, 0:10, 11:14] = 7
    for i in range(LX):
        for j in range(LY):
            for k in range(LZ):
                if (15-i)**2 + (25-j)**2 + (10-k)**2 < 31:
                    vol[i,j,k] = 2
    vol[17:19, :, 3:6]           = 1
    vol[:, 19:21, 3:6]           = 10
    vol[60:70,28:36,14:19]       = 17
    for i in range(LX):
        for j in range(LY):
            for k in range(LZ):
                if (65-i)**2 + (10-j)**2 + (10-k)**2 < 75:
                    if not((60-i)**2 + (7-j)**2 + (10-k)**2 < 31):
                        vol[i,j,k] = 2

    vol[40:56, 25:33, 4:8]     = 5    # a square depending on vox size

    return vol, edims

def make_3D_ex_01():

    # make a test image: a map of various ROIs
    vol   = np.zeros((4, 4, 4), dtype=int)
    edims = (1, 1, 2)  

    vol[2:3, 2:4, 2:4] = 1
    vol[0:2, 0:3, 1:3] = 4

    return vol, edims

def make_3D_ex_02():

    # make a test image: a map of various ROIs
    vol   = np.zeros((6, 6, 6), dtype=int)
    edims = (1, 1, 1)  

    vol[1:5, 1:5, 1:5] = 1

    return vol, edims






# ==========================================================================


if __name__ == "__main__" :

    # ----------------------------------------------------------------------
    # a 3D example: Set it up (ROIs+voxel dims), run the transform,
    # and plot results

    
    print("++ Calc 3D EDT for example_00")
    vol_00_rois, vol_00_edims = make_3D_ex_00()

    vol_00_EDT  = calc_EDT_3D( vol_00_rois, 
                               bounds_are_zero=True,
                               edims=vol_00_edims )
    print("++ Plotting 3D EDT results")
    plot_EDT_3D( vol_00_EDT, vol_00_rois,
                 Nsli  = 4,
                 fname = "TEST_00.svg",
                 edims = vol_00_edims )

    print("++ Calc 3D EDT for example_01")
    vol_01_rois, vol_01_edims = make_3D_ex_01()
    vol_01_EDT  = calc_EDT_3D( vol_01_rois, 
                               bounds_are_zero=True,
                               edims=vol_01_edims )
    print("++ Plotting 3D EDT results")
    plot_EDT_3D( vol_01_EDT, vol_01_rois,
                 Nsli  = 4,
                 fname = "TEST_01.svg",
                 edims = vol_01_edims )

    print("++ Calc 3D EDT for example_02")
    vol_02_rois, vol_02_edims = make_3D_ex_02()
    vol_02_EDT  = calc_EDT_3D( vol_02_rois, 
                               bounds_are_zero=True,
                               edims=vol_02_edims )
    print("++ Plotting 3D EDT results")
    plot_EDT_3D( vol_02_EDT, vol_02_rois,
                 Nsli  = 6,
                 fname = "TEST_02.svg",
                 edims = vol_02_edims )


    print(" vol_00_EDT ", vol_00_EDT)
    print(" vol_00_rois ", vol_00_rois)


    sys.exit(0)


    # ------------------------------------------------------------------------
    # And now write out some files

    # get the dimensions (same for all arrays here)
    S = np.shape(vol)

    # ----------------- write out : the input ROI map (int)
    fname = "data_vol_sh.dat"
    fff   = open(fname, 'wb') # open for business, 'w'riting 'b'inary
    # this appears to be the requisite ordering of walking through array
    for i in range(S[2]):
        for j in range(S[1]):
            for k in range(S[0]):
                fff.write(np.short(vol[k, j, i])) # alias for np.int16
    fff.close()

    # ----------------- write out : the EDT, everywhere (float)
    fname = "data_vol_EDT_fl.dat"
    fff   = open(fname, 'wb') # open for business, 'w'riting 'b'inary
    # this appears to be the requisite ordering of walking through array
    for i in range(S[2]):
        for j in range(S[1]):
            for k in range(S[0]):
                fff.write(np.float32(vol_EDT[k, j, i]))
    fff.close()

    # ----------------- write out : the EDT, ignore where input is 0 (float)
    fname = "data_vol_EDT_nz_fl.dat"
    fff   = open(fname, 'wb') # open for business, 'w'riting 'b'inary
    # this appears to be the requisite ordering of walking through array
    for i in range(S[2]):
        for j in range(S[1]):
            for k in range(S[0]):
                fff.write(np.float32(vol_EDT_nz[k, j, i]))
    fff.close()
