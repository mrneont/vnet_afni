#!/bin/tcsh

# script to copy FROM one flat dir that contains mask/aseg/orig dsets 
# TO a new dir that has a subdir+subsubdir structure for the VNET

# user defines the "from" and "to" top directories, as well as the
# relative fractions for validation, training and testing dsets.

# at the moment, no check occurs that numbers match and that the
# fractions are reasonable... but afterward, I run "tree OUTDIR" to
# see the structure there, and it looks good.

# --------------- user defines these dir names and fractions

set pret_dir = pretrain_vnet_iso_32    # input dir: copy FROM
set main_dir = data_00_basic           # output dir: copy TO

# define fraction for each group (test_frac is "remaining")
set train_frac = 0.7
set valid_frac = 0.2
set test_frac  = `echo "scale = 2; ( 1 - ${train_frac} - ${valid_frac} ) / 1" \
                    | bc`

# ----------------- names of subdirs --------------------------

set subdir    = ( training validation testing )
set subsubdir = ( mask orig )

# --------------- get all subj, by mask dset --------------------

cd ${pret_dir}
set all_subj = ( *_mask.nii.gz )
set nsubj    = ${#all_subj}
cd -

# -----------------------------------------

# get integer number of each of 3 categories
set ntrain = `echo "scale = 0; ${train_frac} * ${nsubj} / 1" | bc`
set nvalid = `echo "scale = 0; ${valid_frac} * ${nsubj} / 1" | bc`
set ntest  = `echo "scale = 0; ${nsubj} - ${ntrain} - ${nvalid}" | bc`

@ valid_bot = ${ntrain} + 1
@ valid_top = ${ntrain} + ${nvalid}
@ test_bot  = ${valid_top} + 1

echo "++ Nsubj       : ${nsubj}"
echo ""
echo "++ train frac  : ${train_frac}"
echo "++ valid frac  : ${valid_frac}"
echo "++ test frac   : ${test_frac}"
echo ""
echo "++ Ntrain      : ${ntrain}"
echo "++ Nvalid      : ${nvalid}"
echo "++ Ntest       : ${ntest}"

set all_train = ( ${all_subj[1-${ntrain}]} )
set all_valid = ( ${all_subj[${valid_bot}-${valid_top}]} )
set all_test  = ( ${all_subj[${test_bot}-${nsubj}]} )


#echo ${all_subj}
#echo ""
#echo ${all_train}
#echo ""
#echo ${all_valid}
#echo ""
#echo ${all_test}
#echo ""

# --------------------------------------------------------------------
# create the new dir structure

\mkdir -p ${main_dir}

foreach ss ( ${subdir} ) 
    foreach tt ( ${subsubdir} ) 
        \mkdir -p ${main_dir}/${ss}/${tt}
    end
end

# --------------------------------------------------------------------
# populate

set name_subdir = training
foreach aa ( ${all_train} )
    set subj = ${aa:gas/_mask.nii.gz//}   # subj id part
    
    # copy each of mask and orig dsets to their new loc
    foreach dtype ( ${subsubdir} )
        \cp ${pret_dir}/${subj}_${dtype}.nii.gz       \
            ${main_dir}/${name_subdir}/${dtype}/${subj}_${dtype}.nii.gz 
    end
end


set name_subdir = validation
foreach aa ( ${all_valid} )
    set subj = ${aa:gas/_mask.nii.gz//}   # subj id part
    
    # copy each of mask and orig dsets to their new loc
    foreach dtype ( ${subsubdir} )
        \cp ${pret_dir}/${subj}_${dtype}.nii.gz       \
            ${main_dir}/${name_subdir}/${dtype}/${subj}_${dtype}.nii.gz 
    end
end


set name_subdir = testing
foreach aa ( ${all_test} )
    set subj = ${aa:gas/_mask.nii.gz//}   # subj id part
    
    # copy each of mask and orig dsets to their new loc
    foreach dtype ( ${subsubdir} )
        \cp ${pret_dir}/${subj}_${dtype}.nii.gz       \
            ${main_dir}/${name_subdir}/${dtype}/${subj}_${dtype}.nii.gz 
    end
end
