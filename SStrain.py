
import sys
import torch

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from  SSdata import SSData_path,vol_generator
from SSmodels import VNet_org
from SSlosses import SoftDiceLoss




# data_path contains the dataset currently being used to train the model
# each dataset contains the training set and the validation set
#data_path = '/Users/yamunasn/Vnet_afni/dataset/pretrain_vnet'

#data dimensions of the volume  
#data_dims = (256, 256, 256)


# number of slices in cor, sag and axl
#num_slices_sag = data_dims[0]
#num_slices_cor = data_dims[1]
#num_slices_axl = data_dims[2]

def train_net(data_path,epochs,lr):
	

	(training_path,validation_path) = SSData_path(data_path)
	(xtrain,ytrain) = vol_generator(training_path)
	(xval,yval)=vol_generator(validation_path)
	n_train= xtrain.shape[0]
	#print(n_train)
	net = VNet_org(in_channels=1, num_class=2)
	#load loss func
	loss = SoftDiceLoss(xtrain,ytrain)

	# load optimizor
	optimizer = optim.Adam(net.parameters(), lr=lr)

	train_loader = DataLoader(xtrain,shuffle=False,batch_size=1)
	print(len(train_loader))
	step = 0

	for epoch in range(epochs):


		net.train()

		epoch_loss = 0

		for train_data in train_loader:
			

			
			mri_data = train_data.unsqueeze(0)
			#print('mri_data shape is =',mri_data.shape)
			mri_data = torch.tensor(mri_data, dtype=torch.float32)
			#print('DATA shape is =',train_data_i.shape)
			masks_pred = net(mri_data)
			optimizer.zero_grad()




def main():
    data_path = '/Users/yamunasn/Vnet_afni/dataset/pretrain_vnet'
    epochs=5
    lr=0.001
    train_net(data_path,epochs,lr)


main()	