What is this README for?
------------------------------

Data augmentation is a suite of programs to create pseudo artifacts in copies of the original MRI data. The copies of the data created during data augmentation are used to train the Vnet model. 

List of data augmentation types supported are
------------------------------
* phase-0
  * refacing is done randomly to 20% of data
* phase-1    
  * gibbs artifact - 15% weightage
  * affine_transformations - 45% weightage
* phase-3  
  * gain_inhomogenity
  * zipper noise
  * various % of noise added to dataset
  * contrast_variation inside core + top to bottom

Steps for creating data augmentation folder from the given dataset 
-----------------------------------------------------------
1) Download/ git checkout  the ‘data_augmentation_scripts’ folder  from git repo.
2) Fetch memory on biowulf using command  ‘sinteractive --mem=20g’
3) Activate the conda env ‘conda activate vnet_tech_2024_01_02_gpu’
4) Run ‘module afni’ command to load afni. 
5) export PYTHONPATH=/usr/local/apps/afni/current-py3/linux_rocky_8 [  *clarify*  ]
6) Run ‘data_augmentation_wrapper.py’ library file to create the augmented dataset.
   
   Path_to_original_dataset_augment is prepared which contains orig, maks, logs and  scripts folder
   
   `python data_augmentation_wrapper.py  -d /path_to_original_dataset  -a /path_to_original_dataset_augment  -c  #_no_of_copies`
   
7) Copy all the data_augmentation scripts from ‘data_augmentation_scripts’ folder to Path_to_original_dataset_augment/scripts folder.
8) Path_to_original_dataset_augment/scripts folder also contains the shell scripts to run the data augmentation for each dataset in orig folder. The ‘run_swarm.tcsh’ shell
   script runs all the data augmentation scripts in parallel. This can be checked using the command ‘sjobs’
10) 





 

