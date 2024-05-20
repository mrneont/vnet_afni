#!/bin/tcsh


# script to introduce the zipper noise into the datasets. 
# daug = $1 -> is the absolute path of the data augmentation dir

# set Data augmentation directory and filename: daug
set daug   =  $1
set axis   =  $2
set window =  $3 

# seperate out the  folder name 
set daug_dir = `dirname ${daug}`
echo ${daug_dir}

cd ${daug_dir} # access the copies of dataset 

# seperate  out the basename of the dataset 
set dset = `basename ${daug}`
echo ${dset}
# example of the expression 
    # step(5-(abs(i-65)))*step(1-(abs(k-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2

    # 5: half-width in i-direction is 5-1
    # 65: centered around i=65
    # step(mod(j,4)-1):  walk through j indices, and divide by 4; if remainder is 0,1 -> step(...) # = 0; if remainder is 2,3 -> step(...) = 1
    # (-1)**SOMETHING:  create value that is either -1 or 1, depending on odd or even SOMETHING
    # step(iran(9)-7) : generate a random int in range [0,9], and if that value is >7, return 1; 
    # else, 0




if ( axis == k ) then 
    3dcalc              \
        -a ${dset}  \
        -expr "step(5-(abs(i-65)))*step(1-(abs(j-65)))*((-1)**step(mod(k, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
        -prefix zip1.nii.gz \
        -overwrite
else if ( axis == i ) then
    3dcalc              \
        -a ${dset} \
        -expr "step(5-(abs(j-65)))*step(1-(abs(k-65)))*((-1)**step(mod(i, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
        -prefix zip1.nii.gz \
        -overwrite
else #(axis == j )
    3dcalc              \
        -a ${dset} \
        -expr "step(5-(abs(k-65)))*step(1-(abs(i-65)))*((-1)**step(mod(j, 4)-1)*(-1)**step(iran(9)-7)+1)/2" \
        -prefix zip1.nii.gz \
        -overwrite

endif

3dcalc                                 \
    -overwrite                         \
    -a ${dset}       \
    -b zip1.nii.gz                     \
    -c b'[0,2,0,0]'   \
    -d b'[0,1,1,0]'     \
    -expr "a + (150*b)-a*step(c)+ (150*d)"   \
    -prefix ${dset}

rm zip1.nii.gz
