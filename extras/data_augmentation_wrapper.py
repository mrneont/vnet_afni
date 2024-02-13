import sys
sys.path.append('/Users/narayanaswamyy2/abin')
from afnipy import afni_base as ab
import os 
import argparse      as argp

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
*							-l 'file_list.txt' 
* 						    -ph 'training' 
*							-da 'affine_trans'
*
* 
* List of data augmentation types supported are 
* 1) gibbs artifact
* 2) affine_transformations 
* 3) gain_inhomogenity
* 4) zipper noise
* 5) various % of noise added to dataset
*************************************************************************************
'''

def get_data_augment_args():

	parser = argp.ArgumentParser(prog = 'data_augmentation_wrapper.py',
									formatter_class=argp.RawTextHelpFormatter)
	# data_augment_path is the data_path for data augmentation folder
	parser.add_argument("-a", "--data_augment_path")
	# data_dir is the data path for the dataset folder

	parser.add_argument("-d", "--data_dir")
	# file having the list of datasets 
	# in a random order for data augmentation
	parser.add_argument("-l", "--file_list")
	# string to specify the data augmentation type

	parser.add_argument("-da", "--data_aug")
	# string to specify the training phase or validation phase
	parser.add_argument("-ph", "--phase")


	return parser.parse_args()






def main():

	
	args        = get_data_augment_args()
	#data_augment_path is the data_path for data augmentation folder
	da_path     = args.data_augment_path
	print(" data augmentation folder is: {}".format(da_path))
	
	# data_dir is the data path for the dataset folder
	data_path   = args.data_dir	
	# file having the list of datasets 
	# in a random order for data augmentation
	file_list   = args.file_list
	# string to specify the data augmentation type
	daug        = args.data_aug
	phase       = args.phase

	orig_folder = os.path.join(data_path, phase,'orig')
	
	#daug  = 'add_noise'
	match daug:
		case 'gibbs' :
			#do_something(gibbs)
			print("doing gibbs")
			cmd  = '''tcsh do_gibbs_artifact.tcsh {param1} {param2} {param3}'''.format(param1= ${da_path}, param2= ${data_path},param3= ${file_list})
			com  = ab.shell_com(cmd, capture=1)
			stat = com.run()
			#abin_path = com.so[0]
			#print(stat)
			#print(com.so)
			#print(com.se)
		case 'affine_trans':
			#do_something(affine_trans)
			print("doing affine_trans")
		case 'gain_inhom':
			#do_something(gain_inhom)
			print("doing gain_inhom")
		case 'zipper':
			#do_something(zipper)
			print("doing zipper")
		case 'add_noise':
			#do_something(add_noise)
			print("doing add_noise")



main()