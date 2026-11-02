#!/bin/tcsh

# Augmentation script for vnet_afni: add ringing artifact.
#
# written by: YA Narayanaswamy (SSCC, NIMH, NIH, USA)
#             RC Reynolds (SSCC, NIMH, NIH, USA)
#             PA Taylor (SSCC, NIMH, NIH, USA)
# ======================================================================

# ----------------------------------------------------------------------
# top level setup

# Propagate failures inside backquote substitutions to $status.
set anyerror

# AFNI's global options are available when @global_parse is installed.
which @global_parse >& /dev/null
if ( $status == 0 ) then
    @global_parse `basename "$0"` "$*"
    if ( $status ) exit 0
endif

set version   = "1.0"
set rev_dat   = "Nov 1, 2026"
set this_prog = "do_gibbs.tcsh"
set abbrev    = "gibbs"
set here      = "$cwd"

# ----------------------- set defaults --------------------------

set inset_orig  = ""
set inset_mask  = ""
set prefix_orig = ""
set prefix_mask = ""
set odir        = ""
set opref       = ""
set mdir        = ""
set mpref       = ""
set wdir        = ""
set work_path   = ""
set DO_CLEAN    = 1
set range       = ""

# ------------------- process options, a la rr ----------------------

if ( $#argv == 0 ) goto SHOW_HELP

set ac = 1
while ( $ac <= $#argv )
    # terminal options
    if ( ( "$argv[$ac]" == "-h" ) || ( "$argv[$ac]" == "-help" ) ) then
        goto SHOW_HELP
    endif
    if ( "$argv[$ac]" == "-ver" ) goto SHOW_VERSION

    if ( "$argv[$ac]" == "-echo" ) then
        set echo
    else if ( "$argv[$ac]" == "-inset_orig" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set inset_orig = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-inset_mask" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set inset_mask = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-prefix_orig" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set prefix_orig = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-prefix_mask" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set prefix_mask = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-param_range" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set range = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-workdir" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set wdir = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-no_clean" ) then
        set DO_CLEAN = 0

    else
        echo "** ERROR: unexpected option #$ac = '$argv[$ac]'"
        goto BAD_EXIT
    endif
    @ ac += 1
end

# =======================================================================
# ======================== ** Verify + setup ** =========================
# =======================================================================

# -workdir must be a simple directory name, as described in the help.
if ( "$wdir" != "" ) then
    if ( "$wdir" =~ */* || "$wdir" =~ -* || \
         "$wdir" == "." || "$wdir" == ".." ) then
        echo "** ERROR: -workdir requires a simple directory name."
        goto BAD_EXIT
    endif
endif

if ( "${inset_orig}" == "" ) then
    echo "** ERROR: need to provide -inset_orig."
    goto BAD_EXIT
endif

if ( "${inset_mask}" == "" ) then
    echo "** ERROR: need to provide -inset_mask."
    goto BAD_EXIT
endif

if ( "${prefix_orig}" == "" ) then
    echo "** ERROR: need to provide -prefix_orig."
    goto BAD_EXIT
endif

if ( "${prefix_mask}" == "" ) then
    echo "** ERROR: need to provide -prefix_mask."
    goto BAD_EXIT
endif

if ( "${range}" == "" ) then
    echo "** ERROR: need to provide -param_range (1 value(s))."
    goto BAD_EXIT
endif

foreach value ( "${inset_orig}" "${inset_mask}" "${prefix_orig}" \
                "${prefix_mask}" "${range}" )
    if ( "$value" =~ -[A-Za-z]* ) then
        echo "** ERROR: an option value appears to be another option: $value"
        goto BAD_EXIT
    endif
end

# Resolve relative paths before entering the isolated working directory.
if ( "$inset_orig" !~ /* )  set inset_orig  = "${here}/${inset_orig}"
if ( "$inset_mask" !~ /* )  set inset_mask  = "${here}/${inset_mask}"
if ( "$prefix_orig" !~ /* ) set prefix_orig = "${here}/${prefix_orig}"
if ( "$prefix_mask" !~ /* ) set prefix_mask = "${here}/${prefix_mask}"

set odir  = "`dirname "$prefix_orig"`"
if ( $status ) goto BAD_EXIT
set opref = "`basename "$prefix_orig"`"
if ( $status ) goto BAD_EXIT
set mdir  = "`dirname "$prefix_mask"`"
if ( $status ) goto BAD_EXIT
set mpref = "`basename "$prefix_mask"`"
if ( $status ) goto BAD_EXIT

if ( -d "$prefix_orig" || -d "$prefix_mask" ) then
    echo "** ERROR: output names must name datasets, not directories."
    goto BAD_EXIT
endif

if ( ! -d "$odir" ) then
    echo "++ Making new output directory: $odir"
    \mkdir -p "$odir"
    if ( $status ) goto BAD_EXIT
endif
if ( ! -d "$mdir" ) then
    \mkdir -p "$mdir"
    if ( $status ) goto BAD_EXIT
endif

# Canonicalize the output parent dirs so paths survive subsequent cd commands.
cd "$odir"
if ( $status ) goto BAD_EXIT
set odir = "$cwd"
set prefix_orig = "${odir}/${opref}"
cd "$mdir"
if ( $status ) goto BAD_EXIT
set mdir = "$cwd"
set prefix_mask = "${mdir}/${mpref}"
cd "$here"
if ( $status ) goto BAD_EXIT

if ( "$prefix_orig" == "$prefix_mask" ) then
    echo "** ERROR: anatomical and mask outputs need different names."
    goto BAD_EXIT
endif

# make workdir name, if nec
if ( "$wdir" == "" ) then
    set tmp_code = "`3dnewid -fun11`"  # should be essentially unique hash
    if ( $status ) goto BAD_EXIT
    if ( "$tmp_code" == "" ) goto BAD_EXIT
    set wdir = __workdir_${abbrev}_${tmp_code}
endif
set work_path = "${odir}/${wdir}"

# A fresh directory avoids collisions and makes cleanup ownership explicit.
# Do not reuse an existing directory, including an existing symbolic link.
if ( -e "$work_path" || -l "$work_path" ) then
    echo "** ERROR: working directory already exists: $work_path"
    goto BAD_EXIT
endif
echo "++ Making working directory: $work_path"
\mkdir -p "$work_path"
if ( $status ) goto BAD_EXIT
cd "$work_path"
if ( $status ) goto BAD_EXIT
onintr BAD_EXIT

# =======================================================================
# =========================== ** Main work ** ===========================
# =======================================================================

# Copies replace the originals' assumption that inputs are disposable copies.
set dset_in = orig_00_copy.nii.gz
set mask_in = mask_00_copy.nii.gz

3dcopy "$inset_orig" "$dset_in"
if ( $status ) goto BAD_EXIT

3dcopy "$inset_mask" "$mask_in"
if ( $status ) goto BAD_EXIT

# -------------------------------------------------------------------------
# do processing

# some temp dsets
set dset_res  = orig_01_rai.nii.gz
set dset_zp   = orig_02_zp.nii.gz
set dset_fft  = orig_03_complex.nii.gz
set dset_bpa  = orig_04_bp_abs.nii.gz
set dset_bpp  = orig_05_bp_pha.nii.gz
set dset_bpc  = orig_06_bp_comp.nii.gz
set dset_ifft = orig_07_ifft.nii.gz
set dset_izp  = orig_08_izp.nii.gz
set dset_ires = orig_09_ires.nii.gz

# get input orientation and datum type
set ori_in = `3dinfo -orient ${dset_in}`
if ( $status ) goto BAD_EXIT
set dat_in = `3dinfo -datum ${dset_in}`
if ( $status ) goto BAD_EXIT

# ----------------------------------------------------------------------------
# IKJ #
# make RAI orient, so IJK maps onto XYZ easily
3dresample                                                                   \
    -overwrite                                                               \
    -orient     RAI                                                          \
    -prefix     ${dset_res}                                                  \
    -input      ${dset_in}
if ( $status ) goto BAD_EXIT

# convenient way to know size, and to have powers of 2 in all dir
3dZeropad                                                                    \
    -overwrite                                                               \
    -RL         256                                                          \
    -AP         256                                                          \
    -IS         256                                                          \
    -prefix     ${dset_zp}                                                   \
                ${dset_res}
if ( $status ) goto BAD_EXIT

# do the FFT, with "shift" on so baseline is in middle of image
3dFFT                                                                        \
    -overwrite                                                               \
    -complex                                                                 \
    -altIN                                                                   \
    -prefix     ${dset_fft}                                                  \
    -input      ${dset_zp}
if ( $status ) goto BAD_EXIT

# bounds on each side
# range 70 to 100
set dx = ${range}
set dy = ${range}
set dz = 0
@   xd = 256 - ${dx} - 1
if ( $status ) goto BAD_EXIT
@   yd = 256 - ${dy} - 1
if ( $status ) goto BAD_EXIT
@   zd = 256 - ${dz} - 1
if ( $status ) goto BAD_EXIT

# 3dcalc cannot output complex dataset; need to do Mag/Ph or Re/Im separately
3dcalc                                                                       \
    -overwrite                                                               \
    -cx2r       ABS                                                          \
    -a          ${dset_fft}                                                  \
    -expr       "a"                                                          \
    -prefix     ${dset_bpa}                                                  \
    -float                                                                   \
    -nscale
if ( $status ) goto BAD_EXIT

    # -expr "a"                   \
3dcalc                                                                       \
    -overwrite                                                               \
    -cx2r       PHASE                                                        \
    -a          ${dset_fft}                                                  \
    -expr       "a*(1+0.75*not(within(i,$dx,$xd)*within(j,$dy,$yd)*within(k,$dz,$zd)))" \
    -prefix     ${dset_bpp}                                                  \
    -float                                                                   \
    -nscale
if ( $status ) goto BAD_EXIT

# recombined bandpassed Mag/Ph
3dTwotoComplex                                                               \
    -overwrite                                                               \
    -MP                                                                      \
    -prefix     ${dset_bpc}                                                  \
                ${dset_bpa}                                                  \
                ${dset_bpp}
if ( $status ) goto BAD_EXIT

# invert the FFT on the bandpassed dataset
3dFFT                                                                        \
    -overwrite                                                               \
    -inverse                                                                 \
    -prefix     ${dset_ifft}                                                 \
    -input      ${dset_bpc}
if ( $status ) goto BAD_EXIT

# change back to original grid dims                       \
3dZeropad                                                                    \
    -overwrite                                                               \
    -master     ${dset_res}                                                  \
    -prefix     ${dset_izp}                                                  \
                ${dset_ifft}
if ( $status ) goto BAD_EXIT

# change back to original orientation
3dresample                                                                   \
    -overwrite                                                               \
    -orient     ${ori_in}                                                    \
    -input      ${dset_izp}                                                  \
    -prefix     ${dset_ires}
if ( $status ) goto BAD_EXIT

# no negative values, ensure short output type
3dcalc                                                                       \
    -overwrite                                                               \
    -a          ${dset_ires}                                                 \
    -expr       'min(abs(a),255)'                                            \
    -prefix     orig_10_clip.nii.gz                                          \
    -datum      ${dat_in}                                                    \
    -nscale
if ( $status ) goto BAD_EXIT

# ---------------------------------------------------------------------
# Write out both requested datasets after all processing succeeds.

# outputs
set orig_result = orig_10_clip.nii.gz
set mask_result = "$mask_in"                # output mask matches input

3dcopy -overwrite "$orig_result" "$prefix_orig"
if ( $status ) goto BAD_EXIT

3dcopy -overwrite "$mask_result" "$prefix_mask"
if ( $status ) goto BAD_EXIT

# move out of wdir to the odir
cd "$odir"
if ( $status ) goto BAD_EXIT
set whereout = "$cwd"

if ( $DO_CLEAN == 1 ) then
    echo "++ Clean working dir"
    \rm -rf "$work_path"
    if ( $status ) goto BAD_EXIT
else
    echo "++ NOT removing temporary working dir: $work_path"
endif

cat << EOF

++ DONE. See the outputs:
   $prefix_orig
   $prefix_mask

EOF
goto GOOD_EXIT

# ========================================================================
# ========================================================================

SHOW_HELP:
cat << EOF
-------------------------------------------------------------------------
Overview ~1~

Apply the original Fourier-domain ringing/artifact operation.

ver     : ${version}
revdate : ${rev_dat}

-------------------------------------------------------------------------
Options ~1~

-inset_orig    IN_O    :(req) input anatomical dataset

-inset_mask    IN_M    :(req) input mask dataset, given explicitly

-prefix_orig   PREF_O  :(req) output anatomical dataset name/prefix

-prefix_mask   PREF_M  :(req) output mask dataset name/prefix

-param_range   RR      :(req) Integer dx and dy bound;
                        typical values might be in [70, 100].

-workdir       WDIR    :temporary directory name beneath the anatomy output
                        directory; no slash; must not already exist

-no_clean              :keep the working directory 
                        (def: remove wdir on success)

-echo                  :very verbose output for troubleshooting

-help, -h              :show help

-ver                   :show version

-------------------------------------------------------------------------
Notes ~1~

Both '-inset_* ..' inputs and both '-prefix_* ..' outputs are required.

Relative paths start at the calling directory. Temporary datasets in
the workdir have simple numbered names within the working
directory. Failed runs leave that directory for diagnosis. Final
copies use AFNI overwrite behavior. 

The output mask is an unchanged dataset copy.

-------------------------------------------------------------------------
Examples ~1~

tcsh do_gibbs.tcsh                                \
    -inset_orig        p1/anat.nii.gz             \
    -inset_mask        p1/mask.nii.gz             \
    -prefix_orig       p2/anat.nii.gz             \
    -prefix_mask       p2/mask.nii.gz             \
    -param_range       80

EOF

goto GOOD_EXIT

SHOW_VERSION:
    echo "version $version ($rev_dat)"
    goto GOOD_EXIT

FAIL_MISSING_ARG:
    echo "** ERROR: Missing argument(s) after option flag: '$argv[$ac]'"
    goto BAD_EXIT

BAD_EXIT:
    echo "** ERROR: $this_prog failed."
    if ( "$work_path" != "" ) then
        if ( -d "$work_path" ) then
            echo "++ Working directory retained: $work_path"
        endif
    endif
    exit 1

GOOD_EXIT:
    exit 0
