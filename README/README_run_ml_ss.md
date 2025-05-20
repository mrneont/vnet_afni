What is this README for?
------------------------------

README_run_ml_ss.md is for illustrating the different aspects of setting the training of the Vnet model. 

The model is trained when neural network is set to 'train' mode. During this phase the weights of the model are altered by the optimization algorithm. There are various hyper parameters which determine the training of the neural network. All of the hyperparameters and other parameters reqiored for the training of the model are set as arguements of 'run_ml_ss.py'

A sample version of setting the training of model is as below:
   
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
   





