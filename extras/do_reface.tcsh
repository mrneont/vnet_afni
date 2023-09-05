#!/bin/tcsh

set here      = ${PWD}

echo ${here}
set orig   =    ${here}/orig
echo ${orig}

cd ${orig}
set dset_list =  (*.gz) 

echo ${dset_list}

foreach dset  (${dset_list})

	echo ${dset}

	set prefix      = `python -c "print('reface' )"`

	
	set folder      = `python -c "print('${dset}'.split('.')[0]+'_'+'reface' +'_'+'QC')"`

	set auxfl       =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
									'.'+ 'face' +\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`
    echo ${auxfl}
	echo ${folder}

	set dset_trans  =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`


	@afni_refacer_run                                                     \
		-input ${dset}                                                 \
		-mode_reface                                                      \
		-prefix ${here}/${dset_trans}

	rm  ${here}/${auxfl}
	rm -r ${here}/${folder}

end 
#end_time=$(date +%s)
#elapsed=$(( end_time - start_time ))
#echo ${elapsed}


