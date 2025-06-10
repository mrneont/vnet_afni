README_testing_datasets.md is for illustrating the steps involved in testing of the Vnet model. 

1. The model is tested after training and the model weights are stored as checkpoint.pt files with the specified epoch number as suffix. 

2. The model is tested when neural network is set to 'eval' mode. During this phase the weights of the model are frozen/ not altered. 

3. Multiple checkpoint.pt files can be added to the 'checkpoint.pt_folder_path'. The library file 'lib_ml_test.py' has a 'for loop' which goes over multiple checkpoint.pt in the 'checkpoint.pt_folder_path'. The test data is tested against each of the models and the loss value are tabulated in separate log files. 

4. A sample version of setting the testing of model for a particular dataset is as below:

`python lib_ml_test.py -d 'testing_dataset_path' -o 'output_dir_path' -ch 'checkpoint.pt_folder_path'  -m 'map_location' `

5. The 'map_location' is the device on which the checkpoint.pt/model weights are loaded/mapped. The default location in 'cpu' and the user has a choice to set 'map_location' to 'gpu' if GPU are avaailable. 

6. When the groundtruth is not available the masks are predicted and are stored in the 'output_dir_path'. The dice score is not calculated nor tabulated in a log file. 

7. Testing on Biowulf using sbatch command: [TBD]
    a)  
   

8. Developer's POV  

a) The 100th epoch seems to be working fine for both the 'Sorensen_Dice_mean' and 'WtSorensen_Dice' loss functions.  

b) the user can define a desired stopping criterion to select the checkpoint.pt pertaining to a particular epoch. 
