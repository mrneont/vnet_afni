What is this README for?
------------------------------
README_finetuning.md is for illustrating the steps required to finetune the model using an already existing checkpoint.pt(model).

Fine-tuning in deep learning is a process where a pre-trained model is further trained on a new dataset to improve its performance on a specific task.

Assuming that a working directory of the vnet_afni already exists and the model is already trained for a particular dataset. 
Having run the model, the checkpoint.pt files for various epochs exists in the output directory. 

Steps for finetuning the model  
----------------------------------------------
1. One of the checkpoint.pt is selected as a model to start finetuning, the follwoing command is used
The flag '-r' denotes the 'restart'. When the restart flag is set to 1, the model loads the checkpoint.pt file from the specified location [TBD] 

   `python run_ml_ss.py 
      -d 'training_dataset_path' 
      -s seed 
      -o 'output_dir_path' 
      -e epoch_num 
      -l learning_rate 
      -dn data_normalization 
      -L  Loss_function 
      -W  write_to_output_dir_path 
      -trb batch_size_training 
      -tr_shuf shuffle_flag 
      -mask_everyn 1 
      -chpt_everyn 1 
      -r 1 `


