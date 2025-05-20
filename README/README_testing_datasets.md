README_testing_datasets.md is for illustrating the steps involved in testing of the Vnet model. 

The model is tested when neural network is set to 'eval' mode. During this phase the weights of the model are not altered. 

A sample version of setting the testing of model for a particular dataset is as below:

`python lib_ml_test.py -d 'testing_dataset_path' -o 'output_dir_path' -ch 'checkpoint.pt folfer path'  -m map_location `
   

