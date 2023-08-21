#!/bin/tcsh

# list of orig dataset into which the gain inhomogenity will be applied
set dset_list = ( *.gz) 
set axis = (i j k) # the three different axis 

foreach dset  (${dset_list})



	echo ${dset}

    # example of the expression 
    # step(5-(abs(i-65)))*step(1-(abs(k-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2

    # 5: half-width in i-direction is 5-1
    # 65: centered around i=65
    # step(mod(j,4)-1):  walk through j indices, and divide by 4; if remainder is 0,1 -> step(...) # = 0; if remainder is 2,3 -> step(...) = 1
    # (-1)**SOMETHING:  create value that is either -1 or 1, depending on odd or even SOMETHING
    # step(iran(9)-7) : generate a random int in range [0,9], and if that value is >7, return 1; 
    # else, 0
    foreach x (1 2 3) # counter for the array 'axis'

        echo ${x}
        
        if ( ${x} == 1 ) then 
            set xp  = `expr $x + 1`
            set xpp = `expr $x + 2`
        else if ( ${x} == 2 ) then
            set xp = `expr $x + 1`
            set xpp = `expr $x - 1`
        else #(${x} == 3 )
            set xp = `expr $x - 2`
            set xpp = `expr $x - 1`

        endif
        
        echo ${x} ${xp} ${xpp} 
        set prefix      = `python -c "print('zipper'+ '${axis[${x}]}' +'${axis[${xp}]}' )"`
        echo ${prefix}
        
        set dset_trans  =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

        
        
        
        if ( ${x} == 1 ) then 
            3dcalc              \
                -a ${dset}\
                -expr "step(5-(abs(i-65)))*step(1-(abs(j-65)))*((-1)**step(mod(k, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
                -prefix zip1.nii.gz \
                -overwrite
        else if ( ${x} == 2 ) then
            3dcalc              \
                -a ${dset}\
                -expr "step(5-(abs(j-65)))*step(1-(abs(k-65)))*((-1)**step(mod(i, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
                -prefix zip1.nii.gz \
                -overwrite
        else #(${x} == 3 )
            3dcalc              \
                -a ${dset}\
                -expr "step(5-(abs(k-65)))*step(1-(abs(i-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
                -prefix zip1.nii.gz \
                -overwrite

        endif
        
        
        

        3dcalc                                 \
            -overwrite                         \
            -a pac_125_orig_128sz.nii.gz       \
            -b zip1.nii.gz                     \
            -c b'[0,2,0,0]'   \
            -d b'[0,1,1,0]'     \
            -expr "a + (150*b)-a*step(c)+ (150*d)"   \
            -prefix ${dset_trans}

        rm zip1.nii.gz
    end

end
