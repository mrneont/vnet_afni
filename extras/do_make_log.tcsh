#!/bin/tcsh

set all_dset = ( FT_NEW4.results/*HEAD FT_NEW4.results/*.nii* )

set ocsv     = FILENAME.csv


# -------------------------------

# initialize+clear CSV file:
printf "" > ${ocsv}

# make column labels: 
printf "%3s, %3s, %3s, %4s, %4s, %4s, %s\n"  \
        ni nj nk vdi vdj vdk prefix        \
        >> ${ocsv}
    

foreach dset ( ${all_dset} )
    echo "++ dset: ${dset}"

    set dset_info = `3dinfo            \
                        -n4            \
                        -ad3           \
                        -prefix_noext  \
                        -av_space      \
                        "${dset}"`

    set dim_grid  = ( ${dset_info[1-3]} ) # ignore [4], which is time
    set dim_ad3   = ( ${dset_info[5-7]} )
    set prefix_ne = ${dset_info[8]}
    set av_space  = ${dset_info[9]}

    # get just *name* of dset, even if path is included
    set dset_base = `basename ${dset}`
    
    printf "%3d, %3d, %3d, %4.2f, %4.2f, %4.2f, %s\n"  \
            ${dim_grid} ${dim_ad3} ${dset_base}        \
            >> ${ocsv}
    
end

