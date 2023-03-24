#!/bin/tcsh


#window over which the scaling factor varies 
# the scaling factor varies between values 'win_min' and 'window' 
set window  = 1.2   # within [0,2]
set win_min = `ccalc 1-${window}/2` 

# list of orig dataset into which the gain inhomogenity will be applied
set dset_list = ( *.gz) 

set axis =(i j k) # the three different axis 


foreach dset  (${dset_list})

	echo ${dset}
	
	# dimension of the dset
	set nmat      = `3dinfo -ni ${dset}`

	foreach x (1 2 3) # counter for the array 'axis'
    	
		# prefix denotes the type of artifact inrtoduced into the dset
    	set prefix  = `python -c "print('g'+ '${axis[${x}]}')"`

    	# dset_trans is the transformed dset
    	set dset_trans  =  `python -c "print('${dset}'.split('_')[0]+'_'+ \
					       '${prefix}' + '_' + '${dset}'.split('_')[1]+ \
					       '_'+ '${dset}'.split('_')[2] )"`

		echo ${dset_trans}

		3dcalc                                         \
    		-overwrite                                 \
    		-a ${dset}                                 \
    		-expr "a * (${win_min} + ${window}* ${axis[${x}]}/${nmat})" \
    		-prefix ${dset_trans}
	
		

    end

end
