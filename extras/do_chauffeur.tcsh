#!/bin/tcsh


set opref = TEST1

@chauffeur_afni                                                       \
    -ulay              target3_val_0009.nii.gz                         \
    -olay              predmask*E3_val_0009.nii.gz                \
    -box_focus_slices  target3_val_0009.nii.gz \
    -cbar              Reds_and_Blues_Inv                             \
    -pbar_posonly                                                    \
    -blowup        4                                					\
    -ulay_range        2% 98%                                        \
    -func_range        1                                              \
    -opacity           5                                              \
    -prefix            ${opref}                                       \
    -set_xhairs        OFF                                            \
    -montx 3 -monty 3                                                 \
    -label_mode 1 -label_size 4
