#!/bin/tcsh

# script to make QC images of mask over anatomical
#
# even works appropriately if the mask is not binary, but floats in
# range [0, 1]


set dir_mask = "data_00_basic_iso_192/training/mask"
set dir_orig = "${dir_mask:gas/mask/orig/}"

set all_mask = ( ${dir_mask}/*mask*nii* )

foreach dset_mask ( ${all_mask} )
    echo "++ DSET: ${dset_mask}"

    # orig dset name is same as mask, with replacement
    set dset_orig = "${dset_mask:gas/mask/orig/}"

    set base  = `3dinfo -prefix_noext "${dset_mask}"`
    set rname = "${base:gas/\_mask//}"

    # output prefix for montage image
    set opref = IMG_${rname}

    # make three PNGs
    @chauffeur_afni                                  \
        -ulay  ${dset_orig}                          \
        -olay  ${dset_mask}                          \
        -box_focus_slices AMASK_FOCUS_OLAY           \
        -ulay_range 0% 98%                           \
        -func_range 1                                \
        -cbar "Reds_and_Blues_Inv"                   \
        -pbar_posonly                                \
        -opacity     4                               \
        -blowup      2                               \
        -prefix      ${opref}                        \
        -montx 6 -monty 1                            \
        -set_xhairs OFF                              \
        -label_mode 1 -label_size 4                  \
        -do_clean 

    # glue together separate PNGs
    2dcat                                            \
        -gap     5                                   \
        -gap_col 150 150 150                         \
        -nx 1                                        \
        -ny 3                                        \
        -prefix  ${opref}_FINAL.jpg                  \
        ${opref}*{sag,axi,cor}*png

    if ( $status ) then
        echo "** ERROR: exit, badness for: ${dset_mask}"
        exit 1
    endif


    # clean up, remove separate PNGs
    \rm ${opref}*{sag,axi,cor}*png

    echo "++ Done: ${opref}_FINAL.jpg"
end


exit
