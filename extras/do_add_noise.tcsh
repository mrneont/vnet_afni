#!/bin/tcsh

# list of orig dataset into which the gain inhomogenity will be applied
set dset_list = ( *.gz) 

# noise level of 10% 20% 30%
set noise_lvl = (0.1 0.2 0.3)
set percent = (10 20 30) #SNR percentage 

foreach dset  (${dset_list})

	echo ${dset}

	# find the median  of the voxel value in the dset
	set vals   = `3dBrickStat -slow -automask -median ${dset}`
	
	set med    = ${vals[2]}

	echo ${med}

	foreach x (1 2 3) # counter for the array 'axis'
    	
		# prefix denotes the type of artifact inrtoduced into the dset

    	set prefix  = `python -c "print('noise'+ '${percent[${x}]}')"`

    	# dset_trans is the transformed dset
    	set dset_trans  =  `python -c "print('${dset}'.split('_')[0]+'_'+ \
					       '${prefix}' + '_' + '${dset}'.split('_')[1]+ \
					       '_'+ '${dset}'.split('_')[2] )"`

    	# add zero-centered Gaussian random noise (whose stdev = dset's
		# median), and then scale that to reach our stated SNR
		#echo  ${noise_lvl[${x}]}

		3dcalc                   \
   		-overwrite  			 \
    	-a ${dset}     		  \
    	-expr "a + gran(0,${med})* ${noise_lvl[${x}]}"          \
    	-prefix ${dset_trans}


    end

end
