#!/bin/tcsh


# script to introduce the zipper noise into the datasets. 

# DA_dir      : is the data Data augmentation directory 
# dataset_dir : is the dataset directory


# set Data augmentation directory : DA_dir
set DA_dir       =  $1
set dataset_dir  =  $2
set fl_name = 'FILE.txt'
# list of orig dataset into which the gain inhomogenity will be applied
set v=`cat  ${DA_dir}/${fl_name}`
#echo ${v}
set i=41

set orig_folder  = "training/orig"
set mask_folder  = "training/mask"
set trans_folder = "zipper"
echo " orig_folder = ${orig_folder}"

cd  ${dataset_dir}/${orig_folder}
echo " mask_folder = ${mask_folder} "

while ( $i < = 50 )
    set dset = $v[$i]
    echo ${dset}
    # example of the expression 
    # step(5-(abs(i-65)))*step(1-(abs(k-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2

    # 5: half-width in i-direction is 5-1
    # 65: centered around i=65
    # step(mod(j,4)-1):  walk through j indices, and divide by 4; if remainder is 0,1 -> step(...) # = 0; if remainder is 2,3 -> step(...) = 1
    # (-1)**SOMETHING:  create value that is either -1 or 1, depending on odd or even SOMETHING
    # step(iran(9)-7) : generate a random int in range [0,9], and if that value is >7, return 1; 
    # else, 0
    # the axis is randomly picked # ccalc another option #iran+1
    set axis_ran = (`ccalc -i '1+iran(2)'`)
    echo ${axis_ran}
    foreach x (${axis_ran}) # counter for the array 'axis'

        echo ${x}
  
        if ( ${x} == 1 ) then 
            set prefix      = `python -c "print('zipperk')"`
            3dcalc              \
                -a ${dataset_dir}/${orig_folder}/${dset}  \
                -expr "step(5-(abs(i-65)))*step(1-(abs(j-65)))*((-1)**step(mod(k, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
                -prefix zip1.nii.gz \
                -overwrite
        else if ( ${x} == 2 ) then
            set prefix      = `python -c "print('zipperi')"`
            3dcalc              \
                -a ${dataset_dir}/${orig_folder}/${dset} \
                -expr "step(5-(abs(j-65)))*step(1-(abs(k-65)))*((-1)**step(mod(i, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
                -prefix zip1.nii.gz \
                -overwrite
        else #(${x} == 3 )
            set prefix      = `python -c "print('zipperj')"`
            3dcalc              \
                -a ${dataset_dir}/${orig_folder}/${dset} \
                -expr "step(5-(abs(k-65)))*step(1-(abs(i-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
                -prefix zip1.nii.gz \
                -overwrite

        endif
        
        
        set dset_trans  =  `python -c "print('${dset}'.split('.')[0] +'_'+'${prefix}'+\
                                    '.'+'${dset}'.split('.')[-2] +\
                                    '.'+'${dset}'.split('.')[-1])"`

        echo " dset = ${dset} "
        set mask = ${dset:gas/orig/mask/}
        echo " mask = ${mask} "

        set mask_trans = "${dset_trans:gas/orig/mask/}"
        echo " mask_trans = ${mask_trans} "

        3dcalc                                 \
            -overwrite                         \
            -a ${dataset_dir}/${orig_folder}/${dset}       \
            -b zip1.nii.gz                     \
            -c b'[0,2,0,0]'   \
            -d b'[0,1,1,0]'     \
            -expr "a + (150*b)-a*step(c)+ (150*d)"   \
            -prefix ${DA_dir}/${trans_folder}/${dset_trans}


        cp  ${dataset_dir}/${mask_folder}/${mask}  ${DA_dir}/${trans_folder}/${mask_trans}

        rm zip1.nii.gz
    end
    @ i = $i + 1
end
