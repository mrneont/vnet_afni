#!/bin/tcsh

# Augmentation script for vnet_afni.
#
# This script introduces zipper noise into the orig dataset.
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
set this_prog = "do_zipper_noise.tcsh"
set abbrev    = "zipper_noise"
set here      = "$cwd"

# ----------------------- set defaults --------------------------

set inset_orig  = ""
set inset_mask  = ""
set prefix_orig = ""
set prefix_mask = ""
set prefix_qc   = ""
set odir        = ""
set opref       = ""
set mdir        = ""
set mpref       = ""
set wdir        = ""
set work_path   = ""
set DO_CLEAN    = 1
set axis        = "j"

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

    else if ( "$argv[$ac]" == "-prefix_qc" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set prefix_qc = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-param_axis" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set axis = "$argv[$ac]"

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

if ( "${axis}" == "" ) then
    echo "** ERROR: need to provide -param_axis (1 value(s))."
    goto BAD_EXIT
endif

foreach value ( "${inset_orig}" "${inset_mask}" "${prefix_orig}" \
                "${prefix_mask}" "${prefix_qc}" "${axis}" )
    if ( "$value" =~ -[A-Za-z]* ) then
        echo "** ERROR: an option value appears to be another option: $value"
        goto BAD_EXIT
    endif
end

if ( "${axis}" != "i" && "${axis}" != "j" && "${axis}" != "k" ) then
    echo "** ERROR: axis must be one of these values: i, j, k."
    echo "   Cannot be: ${axis}"
    goto BAD_EXIT
endif

# Resolve relative paths before entering the isolated working directory.
if ( "$inset_orig" !~ /* )  set inset_orig  = "${here}/${inset_orig}"
if ( "$inset_mask" !~ /* )  set inset_mask  = "${here}/${inset_mask}"
if ( "$prefix_orig" !~ /* ) set prefix_orig = "${here}/${prefix_orig}"
if ( "$prefix_mask" !~ /* ) set prefix_mask = "${here}/${prefix_mask}"

if ( "$prefix_qc" != "" && "$prefix_qc" != "0" ) then
    if ( "$prefix_qc" !~ /* ) set prefix_qc = "${here}/${prefix_qc}"
    set qc_prog = `which adjunct_mask_qc_image`
    if ( $status ) goto BAD_EXIT
endif

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

# example of the expression: 
#   step(5-(abs(i-65)))*step(1-(abs(k-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2
#
# 5: half-width in i-direction is 5-1
# 65: centered around i=65
# step(mod(j,4)-1):  walk through j indices, and divide by 4; if
# remainder is 0,1 -> step(...) # = 0; if remainder is 2,3 -> step(...) =
# 1
# (-1)**SOMETHING:  create value that is either -1 or 1, depending on odd
# or even SOMETHING
# step(iran(9)-7) : generate a random int in range [0,9], and if that
# value is >7, return 1;
# else, 0

if ( "$axis" == "k" ) then
    3dcalc                                                                   \
        -a          ${dset_in}                                               \
        -expr       "step(5-(abs(i-65)))*step(1-(abs(j-65)))*((-1)**step(mod(k, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
        -prefix     noise_00_zipper.nii.gz                                   \
        -overwrite
    if ( $status ) goto BAD_EXIT
else if ( "$axis" == "i" ) then
    3dcalc                                                                   \
        -a          ${dset_in}                                               \
        -expr       "step(5-(abs(j-65)))*step(1-(abs(k-65)))*((-1)**step(mod(i, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
        -prefix     noise_00_zipper.nii.gz                                   \
        -overwrite
    if ( $status ) goto BAD_EXIT
else if ( "$axis" == "j" ) then
    3dcalc                                                                   \
        -a          ${dset_in}                                               \
        -expr       "step(5-(abs(k-65)))*step(1-(abs(i-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
        -prefix     noise_00_zipper.nii.gz                                   \
        -overwrite
    if ( $status ) goto BAD_EXIT
else 
    echo "** Unrecognized axis value: ${axis}"
    echo "   Should be one of: i, j, k"
    goto BAD_EXIT
endif

3dcalc                                                                       \
    -overwrite                                                               \
    -a          ${dset_in}                                                   \
    -b          noise_00_zipper.nii.gz                                       \
    -c          b'[0,2,0,0]'                                                 \
    -d          b'[0,1,1,0]'                                                 \
    -expr       "a + (150*b)-a*step(c)+ (150*d)"                             \
    -prefix     orig_01_zipper.nii.gz
if ( $status ) goto BAD_EXIT

# ---------------------------------------------------------------------
# Write out both requested datasets after all processing succeeds.

# outputs
set orig_result = orig_01_zipper.nii.gz
set mask_result = "$mask_in"                # output mask matches input

3dcopy -overwrite "$orig_result" "$prefix_orig"
if ( $status ) goto BAD_EXIT

3dcopy -overwrite "$mask_result" "$prefix_mask"
if ( $status ) goto BAD_EXIT

# Make the optional QC image from the published output datasets.
if ( "$prefix_qc" != "" && "$prefix_qc" != "0" ) then
    set qc_dir = "`dirname "$prefix_qc"`"
    if ( $status ) goto BAD_EXIT
    if ( ! -d "$qc_dir" ) then
        \mkdir -p "$qc_dir"
        if ( $status ) goto BAD_EXIT
    endif

    adjunct_mask_qc_image                                                    \
        -input      "$prefix_mask"                                           \
        -ulay       "$prefix_orig"                                           \
        -outdir     "$work_path"                                             \
        -prefix     qc_00_mask
    if ( $status ) goto BAD_EXIT

    # The helper appends _QC.jpg; use the exact filename requested here.
    \mv "$work_path/qc_00_mask_QC.jpg" "$prefix_qc"
    if ( $status ) goto BAD_EXIT
endif

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

Apply the original zipper-noise operation.

ver     : ${version}
revdate : ${rev_dat}

-------------------------------------------------------------------------
Options ~1~

-inset_orig    IN_O    :(req) input anatomical dataset

-inset_mask    IN_M    :(req) input mask dataset, given explicitly

-prefix_orig   PREF_O  :(req) output anatomical dataset name/prefix

-prefix_mask   PREF_M  :(req) output mask dataset name/prefix

-prefix_qc     JPG     :optional QC image filename, including its path.
                        Overlay the output mask on the output orig dataset.
                        (def: empty; 0 also disables QC)

-param_axis     AA     :axis for zipper appearance; must be one of: i, j, k
                        (def: ${axis})

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

The output mask is an unchanged dataset copy.

-------------------------------------------------------------------------
Examples ~1~

tcsh do_zipper_noise.tcsh                         \
    -inset_orig        p1/anat.nii.gz             \
    -inset_mask        p1/mask.nii.gz             \
    -prefix_orig       p2/anat.nii.gz             \
    -prefix_mask       p2/mask.nii.gz             \
    -param_axis        j

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
