What is this README for?
------------------------------------------------------------------------------

Data augmentation is a suite of programs to create pseudo artifacts in
copies of the original MRI data. The copies of the data created during
data augmentation are used to train the Vnet model.


List of data augmentation types supported are
------------------------------------------------------------------------------

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
------------------------------------------------------------------------------
1) Download+git checkout the 'data_augmentation_scripts' folder from git repo.
2) Start interactive session on biowulf:  'sinteractive --mem=20g'
3) Activate the conda env: 'conda activate vnet_tech_2024_01_02'
4) Load AFNI programs: 'module afni'
5) export PYTHONPATH=/usr/local/apps/afni/current-py3/linux_rocky_8 [ *clarify* ]
6) Use the 'data_augmentation_wrapper.py' program to create an augmented
   copy of the input orig+mask datasets
   
   input directory  : path_to_original_training_data
   + contains these dirs: orig, mask

   output directory : path_to_augmented_training_data
   + contains these dirs: orig, mask, logs, scripts
   + if swarm/shell execution run, then also these dirs: edt

   Command to run:
   python data_augmentation_wrapper.py                       \
       -input_dir           path_to_original_training_data   \
       -output_dir          path_to_augmented_training_data  \
       -num_cp              3                                \
       -exec_mode           swarm
   
  ... which starts a swarm job to run all the data augmentation
  scripts in parallel. This can be checked using the command 'sjobs'






 

