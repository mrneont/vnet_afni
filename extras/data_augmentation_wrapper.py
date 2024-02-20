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
*  -l  | --file_list (file having the list of datasets )
*  -da | --data_aug (data augmentation type)
*  -ph | --phase (training phase or validation phase)
* 
* sample usage of the code : python data_augmentation_wrapper.py 
*							-a '/Users/name/data_augmentation' 
*							-d '/Users/name/dataset' 
*							
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
'''
# ver2 has folder and copies created 
# ver3 read fl_name and deciede whether daug is needed 
# ver4 will have the dict added to it 

# List of data augmentation in different phases 
list_daug_ph1 = ['gibbs','shift','rotation','scale','shear']
list_daug_ph2 = ['gain_inhom','zipper','add_noise']
list_daug_ph3 = ['refacing']
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
    
    #Nested dict for parameters
    
    #do phase1 data augmentation 
    phase1_rand = random.randint(0, 3)
    
    phase1_daug = list_daug_ph1[phase1_rand]
    print('phase_1)',phase1_daug)
    
    
    #do phase2 data augmentation 
    phase2_rand = random.randint(0, 2)
    
    phase2_daug = list_daug_ph2[phase2_rand]
    print('phase_2)',phase2_daug)
    
    #do phase3 data augmentation 
    phase3_daug = list_daug_ph3[0]
    print('phase_3)',phase3_daug)
    
    match phase1_daug:
        case 'gibbs' :
            #do_something(gibbs)
            print("augmentation type : doing gibbs")
            daug1_keys = ['type', 'radius']
            daug1_values = [phase1_daug, 100]
        
        case 'shift':
            #do_something(shift)
            # Default shift range  is plus/minus 32% of grid size
            print("augmentation type : doing shift")
            list_shift   = [5,10,15,20,25,30]
            rand_shift   = random.choice(list_shift)
            list_param1D = [0, 0, -20, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            daug1_keys   = ['type', 'param1D']
            daug1_values = [phase1_daug, list_param1D]

        case 'rotation':
            #do_something(affine_trans)
            print("augmentation type : doing rotation")
            list_param1D = [0, 0, 0, 0, 0, -30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            daug1_keys   = ['type', 'param1D']
            daug1_values = [phase1_daug, list_param1D]
            

        case 'scale':
            #do_something(affine_trans)
            print("augmentation type : doing scale")
            list_param1D = [0, 0, 0, 0, 0, 0, -0.7, -0.7, -0.7, 0.0, 0.0, 0.0]
            daug1_keys   = ['type', 'param1D']
            daug1_values = [phase1_daug, list_param1D]

        case 'shear':
            #do_something(affine_trans)
            print("augmentation type : doing shear")
            list_param1D = [0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09]
            daug1_keys   = ['type', 'param1D']
            daug1_values = [phase1_daug, list_param1D]

       
    
    

    match phase2_daug:

    	case 'gain_inhom':
    		#do_something(gain_inhom)
    		print("augmentation type : doing gain_inhom")
    		#window over which the scaling factor varies 
    		#the scaling factor varies between values 'win_min' and 'win_max'
    		# within [0,2]
    		list_axis    =['i','j','k']
    		rand_axis    =random.choice(list_axis)
    		win_min = 0
    		win_max = 2
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
    		



    keys = ['fl_basename', 'daug_ph1', 'daug_ph2', 'daug_ph3']
    daug_ph1_val = dict(zip(daug1_keys, daug1_values))
    daug_ph2_val = dict(zip(daug2_keys, daug2_values))
    values = [fl_basename, daug_ph1_val, daug_ph2_val, phase3_daug]
    
    daug_dict = dict(zip(keys, values))
    
    
    print(daug_dict)
    return daug_dict


def main():

	
	args        = get_data_augment_args()
	#data_augment_path is the data_path for data augmentation folder
	da_path     = args.data_augment_path
	#print(" data augmentation folder is: {}".format(da_path))
	
	# data_dir is the data path for the dataset folder
	data_path   = args.data_dir	

	# number of copies of the dataset to be made 
	num_cp      = args.num_cp	
	
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


	for fl_name in orig_data_list:
		# condition to check whether to daug or not
		fl_basename = os.path.basename(fl_name)
		print("\n Processing :",fl_basename)
		fl_num = int(re.search(r'\d+', fl_basename).group(0))
		if ((fl_num%1000) ==0):
			print('Status msg : Original copy of dset: no data augmentation')
		else: 
			print('Status msg : do augmentation')
			daug_dict = get_daug_dict(fl_basename)
			# list of dict
			list_daug.append(daug_dict)

	# Serializing json
	json_object = json.dumps(list_daug, indent=4)
	# Writing to sample.json
	with open("daug.json", "w") as outfile:
		outfile.write(json_object)



main()