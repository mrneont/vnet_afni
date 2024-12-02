import sys
sys.path.append('/Users/narayanaswamyy2/abin')
from afnipy import afni_base as ab
import os 
import argparse      as argp
import json
import random
import numpy as np
import os
import glob
import re #regular expression


'''
************************************************************************************
* This is a wrapper code which nests the modular data augmentation scripts
* Inputs required are as below
* flag |  input
*  -a  | --data_augment_path (data_path for data augmentation folder)
*  -d  | --data_dir (data path for the dataset folder)
*  -c  | -- number of copies of the dataset
* 
* + Please note : Create a log folder to capture the results of
*   master_script.tcsh
* 
* sample usage of the code : python data_augmentation_wrapper.py 
*                           -a '/Users/name/data_augmentation' 
*                           -d '/Users/name/dataset' 
*                           -c num_copies
*
* 
* List of data augmentation types supported are 
* 1) gibbs artifact
* 2) affine_transformations 
* 3) gain_inhomogenity
* 4) zipper noise
* 5) various % of noise added to dataset
* 6) refacing 
*************************************************************************************

 -smallrange   = Set all the parameter ranges to be smaller (about half) than
                 the default ranges, which are rather large for many purposes.
                * Default angle range    is plus/minus 30 degrees
                * Default shift range    is plus/minus 32% of grid size
                * Default scaling range  is plus/minus 20% of grid size
                * Default shearing range is plus/minus 0.1111
'''
# List of data augmentation in different phases 
list_daug_ph0    = ['reface','empty']
list_daug_ph1    = ['gibbs','affine']
list_daug_ph2    = ['gain_inhom','zipper','add_noise']
list_daug = []

def get_data_augment_args():

    parser = argp.ArgumentParser(prog = 'data_augmentation_wrapper.py',
                                    formatter_class=argp.RawTextHelpFormatter)
    # data_augment_path is the data_path for data augmentation folder
    parser.add_argument("-a", "--data_augment_path")
    # data_dir is the data path for the dataset folder

    parser.add_argument("-d", "--data_dir")

    # number of copies of the dataset to be made 
    parser.add_argument("-c", "--num_cp")

    return parser.parse_args()



def get_daug_dict(fl_basename): 
    
    global daug1_keys,daug1_values,daug_ph1_val
    global daug2_keys,daug2_values,daug_ph2_val

    daug1_keys  =[]
    daug1_values=[]
    daug2_keys  =[]
    daug2_values=[]
    list_param1D=[]

    #do phase0 data augmentation 
    # phase0 -> refacing is done randomly to 20% of da 
    phase0_rand_item = random.choices([0,1], weights=(20, 80))
    phase0_rand      = phase0_rand_item[0]
    phase0_daug      = list_daug_ph0[phase0_rand]
    print('phase_0)',phase0_daug)
    daug0_keys   = ['type']
    daug0_values = [phase0_daug]

    
    #Nested dict for parameters
    
    #do phase1 data augmentation 
    
    # + phase1 augmentation is randomly picked from 'list_daug_ph1'
    # + the two types of augmentations in phase 1 are gibbs and affine
    # + 15% weightage is given to gibbs and 
    #   75% weightage os given to affine 
    # + phase1_rand_item is either 0 or 1. 
    # + 0 is given 15% weightage and 1 is given 75% weightage 
    phase1_rand_item = random.choices([0,1], weights=(15, 85))
    phase1_rand      = phase1_rand_item[0]
    phase1_daug      = list_daug_ph1[phase1_rand]
    print('phase_1)',phase1_daug)
    
    
    #do phase2 data augmentation 
    # randomly select a number between 0 and len(list)-1
    phase2_rand = random.randint(0, len(list_daug_ph2)-1)
    
    phase2_daug = list_daug_ph2[phase2_rand]
    print('phase_2)',phase2_daug)

    match phase1_daug:
    
        case 'gibbs' :
            #do_something(gibbs)
            print("augmentation type : doing gibbs")
            radius   = random.randint(70, 100)
            daug1_keys = ['type', 'radius']
            daug1_values = [phase1_daug, radius]

        case 'affine':
            #do_something(affine)
            # Default shift range  is plus/minus 32% of grid size
            # Default scaling range  is plus/minus 20% of grid size
            # Default shearing range is plus/minus 0.1111
            # Default angle range    is plus/minus 30 degrees
            # yns: option to pick without replacement 
            affine_rand = random.choices(['rotation','scale','shear','shift'], k=2)
            # condition to avoid random selection of same two augmentation types
            while (affine_rand[0] == affine_rand[1]):
                affine_rand = random.choices(['rotation','scale','shear','shift'], k=2)

            affine_type = affine_rand[0]+'_'+ affine_rand[1]
            # DEFINITION OF AFFINE TRANSFORMATION PARAMETERS
            #pt : # x-shift  y-shift  z-shift   \
            #       z-angle  x-angle  y-angle \
            #       x-scale  y-scale  z-scale  \
            #      y/x-shear  z/x-shear  z/y-shear 
            # param_map = [#1  #2  #3  #4 #5 #6  #7  #8  #9  #10  #11  #12]
            
            # param_map = [Shx Shy Shz Rz Rx Ry Scx Scy Scz  Sheyx Shezx Shezy]
           

            # empty list for param1D of 2 types of affine_transforms
            param1D=[[],[]]
            # for loop over the     
            for x in range(len(affine_rand)):
                #print x= affine data augmentation type 
                #print('x =',affine_rand[x])
                list_axis    =['x','y','z']
                match affine_rand[x]:

                    case 'rotation':
                        #do_something(rotation)
                        # Default angle range is plus/minus 30 degrees
                        
                        rand_axis = random.choice(list_axis)
                        rand_rot   = random.randint(-30, 30)
                        # condition to avoid the rotation =0
                        while (rand_rot == 0):
                            rand_rot   = random.randint(-30, 30)
                        if(rand_axis == 'x'):
                            Rx = rand_rot
                            Rz = 0
                            Ry = 0
                        elif (rand_axis == 'y'):
                            Ry = rand_rot
                            Rx = 0
                            Rz = 0
                        else : # (rand_axis == 'z'):
                            Rz = rand_rot
                            Rx = 0
                            Ry = 0
                        print("augmentation type : doing rotation")
                        param1D[x] = [0, 0, 0, Rz, Rx, Ry, 0, 0, 0, 0, 0, 0]
                        
                    case 'shift':

                        print("augmentation type : doing shift")
                        list_shift   = [-5,-10,-15,-20,-25,-30,5,10,15,20,25,30]
                        rand_shift   = random.choice(list_shift)
                        rand_axis    = random.choice(list_axis)
                        if(rand_axis == 'x'):
                            Shx = rand_shift
                            Shy = 0
                            Shz = 0
                        elif (rand_axis == 'y'):
                            Shx = 0
                            Shy = rand_shift
                            Shz = 0
                        else : # (rand_axis == 'z'):
                            Shx = 0
                            Shy = 0
                            Shz = rand_shift
                        
                        param1D[x] = [Shx, Shy, Shz, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    
                    case 'scale':
                        #do_something(affine_trans)
                        print("augmentation type : doing scale")
                        # scaling applies to all axis
                        #yns range is (0.8,1.2) 
                        rand_scale   = round(random.uniform(0.8, 1.2), 1)
                        
                        Scz          = rand_scale
                        Scx          = rand_scale
                        Scy          = rand_scale
                        
                        
                        param1D[x] = [ 0, 0, 0, 0, 0, 0, Scx, Scy, Scz, 0, 0, 0]
                        
                    case 'shear':
                        #do_something(affine_trans)
                        print("augmentation type : doing shear")
                        rand_axis  = random.choice(list_axis)
                        rand_shear = round(random.uniform(-0.1, 0.1), 2)
                        # condition to avoid the scaling =0
                        while (rand_shear == 0):
                            rand_shear   = round(random.uniform(-0.1, 0.1), 2)
                        if(rand_axis == 'x'):
                            Shex = rand_shear
                            Shey = 0
                            Shez = 0
                        elif (rand_axis == 'y'):
                            Shex = 0
                            Shey = rand_shear
                            Shez = 0
                        else : # (rand_axis == 'z'):
                            Shex = 0
                            Shey = 0
                            Shez = rand_shear

                        param1D[x] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, Shex, Shez, Shey]
                        
                        
            #adding element wise the param list of the two  affine transformations
            list_param1D  = [x + y for x, y in zip(param1D[0], param1D[1])]
            param1D      =  " ".join(map(str, list_param1D))
            print(param1D)
            daug1_keys   = ['type', 'param1D']
            daug1_values = [affine_type, param1D]

    match phase2_daug:

        case 'gain_inhom':
            #do_something(gain_inhom)
            print("augmentation type : doing gain_inhom")
            #window over which the scaling factor varies 
            #the scaling factor varies between values 'win_min' and 'win_max'
            # within [2,2.8]
            list_axis    =['i','j','k']
            rand_axis    =random.choices(list_axis,weights=(10, 80, 10))
            win_min = 2
            win_max = 2.8
            #Returns a random float number up to 1 decimal places
            window  = round(random.uniform(win_min, win_max), 1)
            daug2_keys   = ['type', 'axis', 'window']
            daug2_values = [phase2_daug, rand_axis, window]
        case 'zipper':
            #do_something(zipper)
            print("augmentation type : doing zipper")
            list_axis    =['i','j','k']
            rand_axis    =random.choice(list_axis)
            zip_width    = 5
            daug2_keys   = ['type', 'axis', 'zip_width']
            daug2_values = [phase2_daug, rand_axis, zip_width]
        case 'add_noise':
            #do_something(add_noise)
            # noise level of 10% 20% 30%
            print("augmentation type : doing add_noise")
            noise_lvl       =[0.1, 0.2, 0.3]
            rand_noise_lvl  =random.choice(noise_lvl)
            list_axis    =['i','j','k']
            rand_axis    =random.choice(list_axis)
            daug2_keys   = ['type', 'axis', 'noise_lvl']
            daug2_values = [phase2_daug, rand_axis, rand_noise_lvl]
            



    keys = ['fl_basename', 'daug_ph0', 'daug_ph1', 'daug_ph2']
    daug_ph0_val = dict(zip(daug0_keys, daug0_values))
    daug_ph1_val = dict(zip(daug1_keys, daug1_values))
    daug_ph2_val = dict(zip(daug2_keys, daug2_values))
    values = [fl_basename, daug_ph0_val, daug_ph1_val, daug_ph2_val]
    
    daug_dict = dict(zip(keys, values))
    
    
    print(daug_dict)
    return daug_dict


def write_script(da_dict,fl_name, daug):
    # set the filename for the shell script based on the dataset
    dir_name  = os.path.dirname(fl_name)
    file_name = os.path.basename(fl_name)
    
    new_extension = ".tcsh"
    #print('file_name=',file_name)
    script_fl = file_name.split('.')[0] +new_extension
    script_fl_path_str = os.path.join(os.path.dirname(dir_name), 'scripts', script_fl)
    print('script_fl =', script_fl)
    f = open(script_fl_path_str, "w")
    f.write("#!/bin/tcsh")
    f.write('\n')

    # notes 
    # + the individual shell script need the 
    #   absolute path to the dataset 
    #print("write_script fl_name =  ",fl_name)


    

    # phase-1  data augmentation shell script
    if (daug == 1):

        # phase-0  
        if (da_dict['daug_ph0']['type'] == 'reface'):
            f.write("tcsh do_reface.tcsh {}""".format(fl_name))
            f.write('\n')

        if (da_dict['daug_ph1']['type'] == 'gibbs'):#(weightage =15%)
            gibbs_radius = da_dict['daug_ph1']['radius']
            print('gibbs_radius =',gibbs_radius)
            f.write("tcsh do_gibbs.tcsh {} {}""".format(fl_name,\
                                                        gibbs_radius))
            f.write('\n')
        else: # all cominations of affine_transform(weightage =85%)
            param1D = da_dict['daug_ph1']['param1D']
            f.write("tcsh do_affine.tcsh {} {}""".format(fl_name,\
                                                        param1D))
            f.write('\n')
        # phase-2  data augmentation shell script

        phase2_daug = da_dict['daug_ph2']['type']

        match phase2_daug:

            case 'gain_inhom':
                axis   = da_dict['daug_ph2']['axis']
                window = da_dict['daug_ph2']['window']
                f.write("tcsh do_gain_inhomogenity.tcsh {} {} {}""".format(fl_name,\
                                                        axis, window))

            case 'zipper':
                axis  = da_dict['daug_ph2']['axis']
                width = da_dict['daug_ph2']['zip_width']
                f.write("tcsh do_zipper_noise.tcsh {} {} {}""".format(fl_name,\
                                                        axis, width))

            case 'add_noise':
                axis         = da_dict['daug_ph2']['axis']
                noise_level  = da_dict['daug_ph2']['noise_lvl']
                f.write("tcsh do_add_noise.tcsh {} {} {}""".format(fl_name,\
                                                        axis, noise_level))

        f.write('\n')

    
    # create the edt data for all the masks 

    
    f.write("tcsh do_daug_weight.tcsh {}""".format(fl_name))

    f.write('\n')
    f.close()


def main():

    print('sys.version_info = ',sys.version_info)
    if sys.version_info<(3,10,0):
        sys.stderr.write("You need python 3.10 or later to run this script. \
                        \nThe 'match' statement was introduced in Python 3.10.\
                        \nSo if you're using an older version, \
                        you'll need to upgrade to use it.")
        exit(1)

    random.seed(42)
    print('Random seed =', random.random())

    args        = get_data_augment_args()
    #data_augment_path is the data_path for data augmentation folder
    da_path     = args.data_augment_path
    #print(" data augmentation folder is: {}".format(da_path))
    
    # data_dir is the data path for the dataset folder
    data_path   = args.data_dir 

    # number of copies of the dataset to be made 
    num_cp      = args.num_cp   
    
    if(os.path.isdir(da_path) == True):
        print('\nStatus msg : Data augmentation folder already exists')
        print('Please change the data_augmentation directory name and retry\n')
        sys.exit(1) 

    cmd  = '''tcsh make_copies_data_aug.tcsh {param1} {param2} {param3}'''.\
                            format(param1= data_path, param2= da_path,\
                            param3= num_cp)

    com  = ab.shell_com(cmd, capture=1)
    stat = com.run()
    # print the status of preparing the data_augmentation folder
    if (stat == 0):
        print("Status msg :Data augmentation folder succesfully created")
    else :
        print("Status msg :Data augmentation folder not created: exiting")
        sys.exit(1)
    
    # read copies of dataset
    orig_path_str = os.path.join(da_path, 'orig', '*.nii.gz')
    # list of copies of dataset
    orig_data_list = glob.glob(orig_path_str)
    orig_data_list.sort()
    #print(orig_data_list)
    master_fl = "master_script.tcsh"
    master_fl_path_str = os.path.join(da_path,'scripts',master_fl)
    fl        = open(master_fl_path_str, "w")
    #fl.write("#!/bin/tcsh")
    #fl.write('\n \n \n')
    for fl_name in orig_data_list:
        # condition to check whether to daug or not
        fl_basename = os.path.basename(fl_name)
        print("\n Processing :",fl_basename)
        name = fl_basename.split('.')[0]
        print(name)
        log_fl = "log_"+name +".txt"
        print(log_fl)
        fl_num = int(re.search(r'\d+', fl_basename).group(0))
        if ((fl_num%1000) ==0):
            print('Status msg : Original copy of dset: no data augmentation')
            daug = 0 # flag to depict data_augmentation
            daug_dict = {} # empty dictionary
            print("fl_name =",fl_name)
            write_script(daug_dict,fl_name,daug)

        else: 
            print('Status msg : do augmentation')
            daug_dict = get_daug_dict(fl_basename)

            # write scripts based on the dictionary for each dataset
            daug = 1 # flag to depict data_augmentation
            write_script(daug_dict,fl_name,daug)
            print("daug_dict = ",daug_dict)
            list_daug.append(daug_dict)
            #|& tee logs/log_sub-001001_orig.txt
        fl.write("tcsh -x {}.tcsh |& tee ../logs/{}""".format(fl_basename.split('.')[0],log_fl))
        fl.write('\n')
            # list of dict
            

    # Serializing json
    json_object = json.dumps(list_daug, indent=4)
    # Writing to sample.json
    with open("daug.json", "w") as outfile:
        outfile.write(json_object)

    fl.close()

main()