#!/bin/tcsh -e

set bup  = 4          # can be less for 128iso, e.g., 2


#foreach epoch ( 000 001 005 010 015 )
foreach ii ( `seq 9 1 12` )
    foreach epoch ( 000 001 002 003 004 )
        set phase = 'train'
        set subj  = `printf "%04d" ${ii}`

        set dset_targ = target_${epoch}_${phase}_${subj}.nii.gz
        set dset_pred = predmask_OPP_${epoch}_${phase}_${subj}.nii.gz
        set opref     = img_${epoch}_${phase}_${subj}

        @chauffeur_afni                                                       \
            -ulay              ${dset_targ}                                \
            -olay              ${dset_pred}                                 \
            -box_focus_slices  ${dset_targ}                                 \
            -cbar              Reds_and_Blues_Inv                             \
            -pbar_posonly                                                    \
            -blowup            ${bup}                             				\
            -ulay_range        2% 98%                                        \
            -func_range        1                                              \
            -opacity           5                                              \
            -prefix            ${opref}                                       \
            -set_xhairs        OFF                                            \
            -montx 1 -monty 1                                                 \
            -label_mode 1 -label_size 4
    end
end
