import torch
import os, io
from torchsummary import summary
import sys
sys.path.append('../')
import lib_ml_models as lmm

'''
This code checks the compatibility of any given dataset in terms of its dimensions 
with the neural network under consideration. 

Note:
+The dimensions of the dataset needs to be a part of the list 'data_sizes'

+The neural network/model under consideration needs to be specified as 'net'

+The compatibity of the input data set is checked using the pytorch API 'summary'

+'summary' is an API, which provides the fine visualization of the model given 
 the dimensions of the input dataset. 

+'summary' is very useful in providing both the details of each layer in the model
  and the output shape at each layer. This helps the user to do a dry run(without the 
  actual input dataset) and gauge the model mismatch early on.

+ 'print(model)' feature would print all the layers of the model, but 'summary' gives additional 
  details about the output shape at each layer in the model, the number of trainable parameters
  (weights and biases) in each layer, the approximate size of the input dataset, and the memory requirements 
  for forward and backward pass of the neural network. 

+ Syntax of 'summary' API
  
  summary(model,input_size)

  where: 
  model      = neural network in consideration
  input_size = (C_in,D,H,W)
  C_in       = in_channels of input_data
  D          = Depth of the input_data
  H          = Height of the input_data
  W 		 = Width of the input_data
  batch_size is not specified in the input_data dimension and is assumed to be 1. 

+ Note that in the output shape at each layer would be something like 
  '[-1, 16, 32, 64, 64]' '[-1, 32, 16, 32, 32]'
  The -1  would represent an arbitrary number for the batch dimension.
  While the other shapes are fixed to the shown values, the user can use any batch size for this model.


'''



# list of a sample input data sizes. The sample input_data_size are checked for thier 
# compatibility with the neural network under consideration. 
# more choices of sample input data dimensions could be appended to the list. 

data_sizes = [(1, 8, 32,32),(1, 16, 32,32),(1, 24, 32,32), (1, 64, 32,32),
              (1, 32, 24,24),(1, 16, 8,8),(1, 16, 16,16),(1, 32, 64,64)]



# the compatibility status is logged into the file below
data_size_check = '/'.join(['input_data_size_check.txt'])

# list of compatibility status messages
msg = ['supported','not supported']

# title of columns in the log file  'datasize' and 'compatibility'
with io.open(data_size_check, 'a') as data_size_check_log:
        data_size_check_log.write(" {:>10s}  {:>16s} \n".format('datasize', 
                                								'compatibility'))

# Specify the model/neural network under consideration
net = lmm.VNet_orig(in_channels=1, num_class=2, wt_norm = 0,verb=0)


# loop over the sample inputdata dimensions in the list 'data_sizes'
for i in range(len(data_sizes)):

	
  	with io.open(data_size_check, 'a') as data_size_check_log:

  		data_size_check_log.write(','.join(str(x) for x in data_sizes[i]))

  	try:
      #compatibilty check of the model and the input_data size using 'summary' API
  		summary(net, data_sizes[i]) 

  		with io.open(data_size_check, 'a') as data_size_check_log:
  				data_size_check_log.write(" {:->16s}  \n".format(msg[0]))
  	except:
  		with io.open(data_size_check, 'a') as data_size_check_log:
  				data_size_check_log.write(" {:->16s}  \n".format(msg[1]))



