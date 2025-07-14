What is this repository for?
------------------------------

Test_conda_env is a suite of programs/scripts mostly used by the developer to test the different conda environements to run the Vnet model. 

The model is tested for stability of different versions of pytorch.

The yml files corresponding to different versions of pytorch are created and the list of the conda enviroment is maintained. 

The yml files to  create the different conda environments are in the folder 
/data/NIMH_SSCC/narayanaswamyy2/test_env_yml

sample list of  Pytorch versions = 
( 1.3, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7 )

The shell scripts to activate and test the different conda environments are in the folder :
/data/NIMH_SSCC/narayanaswamyy2/test_env_yml_scripts

The two scripts used to activate and test the different conda environments are 
1. run_test_conda_env.tcsh
2. do_conda_env.tcsh

Steps to test the different conda environments 
------------------------------
1. The run_test_conda_env.tcsh maintains a list of the different conda environments created. The developer can make changes to this list at any given point in time by adding yml files in the 'test_env_yml' folder. 

e.g. conda_env = (test_torch_v1_13_0  test_torch_v2_0_0 test_torch_v2_2_0 test_torch_v2_3_0 test_torch_v2_4_0 \
	         test_torch_v2_5_1 test_torch_v2_6_0 test_torch_v2_7_0 )

2. Run the 'run_test_conda_env.tcsh' using the sample code below. This creates the 'test_pred_mask' and 'logs' folders and 'swarm_test_conda_env.txt' file. 
`tcsh run_test_conda_env.tcsh  $dir_parent`

3. The 'for' loop in the 'run_test_conda_env.tcsh' loops over the list of conda environments. The 'do_conda_env.tcsh' command is written to the 'swarm_test_conda_env.txt' to test each of the conda environment,  which in turn is swarmed on the biowulf.

4. The predicted masks created during testing of conda environments are written into the 'test_pred_mask' folder with suitable suffix to differentiate the different conda environment outputs. 

5. The log files are stored in the 'logs' folder.  





