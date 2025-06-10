What is this README for?
------------------------------
README_finetuning.md is for illustrating the steps required to finetune the model using an already existing checkpoint.pt(model).

Fine-tuning in deep learning is a process where a pre-trained model is further trained on a new dataset to improve its performance on a specific task.

Steps for finetuning the model  
----------------------------------------------
1. Assuming that a working directory of the vnet_afni already exists and the model is already trained for a particular dataset. 
Having run the model, the checkpoint.pt files for various epochs exists in the output directory. One of the checkpoint.pt is selected as a model to start finetuning, which is then copied into the working directory of the 'vnet_afni'.

2. A sample version of setting the finetuning of model is as below:
The flag '-r' denotes the 'restart'. When the restart flag is set to 1, the model loads the checkpoint.pt file from the specified location [TBD] check : currently the checkpoint.pt is hardwired, the path and file can be given as an arguement???

   `python run_ml_ss.py 
      -d 'training_dataset_path' 
      -s 'seed'
      -o 'output_dir_path' 
      -e 'epoch_num'
      -l 'learning_rate' 
      -dn 'data_normalization' 
      -L  'Loss_function'
      -W  'write_to_output_dir_path' 
      -trb 'batch_size_training' 
      -tr_shuf 'shuffle_flag'
      -mask_everyn 1 
      -chpt_everyn 1 
      -r 1 `
   
3. The above command allows the training to continue after the model is loaded. The user can expect the predicted masks to be pretty good fit even in the first epoch, since the training was restarted from a pretrained model. The predicted masks are stored in 'output_dir_path' along with the subsequent checkpoint.pt files.
   
4. The checkpoint.pt files created from finetuning can be used for testing. Please follow the steps provided in README_testing_datasets.md

5. During finetuning the model if the 'Loss_function' used is 'WtSorensen_Dice', the depth information data needs to created.
   The scripts 'do_daug_edt_wrapper.py' and 'do_daug_weight.tcsh' are used for creating the depth data.

6. A sample version of usage of 'do_daug_edt_wrapper.py' is as below:
   
     `python do_daug_edt_wrapper.py
            -d 'dataset_path/training_OR_validation' 
            -s 'data_augmentation_scripts_folder' `
   
   Executing the above command creates a script 'run_edt.tcsh' for every dataset in the 'dataset_path/training_OR_validation'.
   
   
8. The depth data is created for both the training as well as for validation datasets by executing the 'run_edt.tcsh'. It is stored in the 'edt' folder alongside the 'orig' and 'mask' folders. After the depth data is created the finetuning of the model can be executed by following step 2.

9. Testing on Biowulf using sbatch command: [TBD]
    

10. Developer's POV :
    a) The learning rate can be maintained at 1e-4.
    b) Generally the training dataset used for finetuning is small, so 'batch_size_training' can be set to 2 instead of 6.
   
   

