#!/bin/tcsh


# make output dir for all images
\mkdir -p QC01

set opref = QC01/Vrel

@chauffeur_afni                                                       \
    -ulay              predmask_opp_E4_val_0009.nii.gz                               \
    #-box_focus_slices  AMASK_FOCUS_ULAY                               \
    -olay              target4_val_0009.nii.gz                         \
    -cbar              Reds_and_Blues_Inv                             \
    -ulay_range        2% 98%                                        \
    -func_range        3                                              \
    -opacity           5                                              \
    -prefix            ${opref}                                       \
    -set_xhairs        OFF                                            \
    -montx 3 -monty 3                                                 \
    -label_mode 1 -label_size 4
    