#!/bin/tcsh

# Augmentation script for vnet_afni: shade across the whole volume.
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
set this_prog = "do_contrast_variation_shading.tcsh"
set abbrev    = "contrast_variation_shading"
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
set dim_fac     = "5"
set axis        = "j"
set window      = "2"

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

    else if ( "$argv[$ac]" == "-param_dim_fac" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set dim_fac = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-param_axis" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set axis = "$argv[$ac]"

    else if ( "$argv[$ac]" == "-param_window" ) then
        if ( $ac >= $#argv ) goto FAIL_MISSING_ARG
        @ ac += 1
        set window = "$argv[$ac]"

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

foreach value ( "${inset_orig}" "${inset_mask}" "${prefix_orig}" \
                "${prefix_mask}" "${dim_fac}" "${axis}" "${window}" )
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

# temp dsets
set mask_fl   = mask_01_float.nii.gz
set mask_blur = mask_02_blur.nii.gz

# make sure mask is float type, before blurring
3dcalc                                                                       \
    -overwrite                                                               \
    -a          "$mask_in"                                                   \
    -expr       'a'                                                          \
    -prefix     "$mask_fl"                                                   \
    -datum      float                                                        \
    -nscale
if ( $status ) goto BAD_EXIT

# same voxels on the corresponding axis
set nmat = ( `3dinfo -n${axis} "$dset_in"` )
if ( $status ) goto BAD_EXIT

# blur the mask (these values to be used in the power calc)
3dmerge                                                                      \
    -overwrite                                                               \
    -1blur_fwhm  3                                                           \
    -prefix      "$mask_blur"                                                \
                 "$mask_fl"
if ( $status ) goto BAD_EXIT

# apply mask values as a power modulator on voxel values.
3dcalc                                                                       \
    -overwrite                                                               \
    -a          "$mask_blur"                                                 \
    -b          "$dset_in"                                                   \
    -expr       "b**(1-a/${dim_fac})"                                        \
    -prefix     orig_01_core.nii.gz                                          \
    -datum      float                                                        \
    -nscale
if ( $status ) goto BAD_EXIT

# contrast variation in the core + shading from top to bottom
# window over which the scaling factor varies
# the scaling factor varies between values 'win_min' and 'window'

set win_min = "`ccalc 1-${window}/2`"
if ( $status ) goto BAD_EXIT

echo "++ win min : ${win_min}"
echo "++ window  : ${window}"

# same voxels on the corresponding axis
set nmat = ( `3dinfo -n${axis} orig_01_core.nii.gz` )
if ( $status ) goto BAD_EXIT

3dcalc                                                                       \
    -overwrite                                                               \
    -a          orig_01_core.nii.gz                                          \
    -expr       "a * (${window} * (1-${axis}/${nmat})+ ${win_min})"          \
    -prefix     orig_02_shading.nii.gz
if ( $status ) goto BAD_EXIT

# ---------------------------------------------------------------------
# Write out both requested datasets after all processing succeeds.

# outputs
set orig_result = orig_02_shading.nii.gz
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

Apply power modulation followed by the original linear shading.


ver     : ${version}
revdate : ${rev_dat}

-------------------------------------------------------------------------
Options ~1~

-inset_orig    IN_O    :(req) input anatomical dataset

-inset_mask    IN_M    :(req) input mask dataset, given explicitly

-prefix_orig   PREF_O  :(req) output anatomical dataset name/prefix

-prefix_mask   PREF_M  :(req) output mask dataset name/prefix

-param_dim_fac  DF     :dimming factor (power-modulation divisor).
                        (def: ${dim_fac})

-param_axis     AA     :axis for shading variation; must be one of: i, j, k
                        (def: ${axis})

-param_window   WW     :shading window width
                        (def: ${window})

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

tcsh do_contrast_variation_shading.tcsh           \
    -inset_orig        p1/anat.nii.gz             \
    -inset_mask        p1/mask.nii.gz             \
    -prefix_orig       p2/anat.nii.gz             \
    -prefix_mask       p2/mask.nii.gz             \
    -param_dim_fac     5                          \
    -param_axis        j                          \
    -param_window      2

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
