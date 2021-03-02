
import sys
import torch

from torch import optim
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from lib_ml_data   import SSData_path, vol_generator
from lib_ml_models import VNet_org
from lib_ml_losses import DiceLoss,dice
import numpy as np


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
    #traingen = vol_generator(training_path)
    #print(len(traingen))
    (xtrain,ytrain) = vol_generator(training_path)
    #(xval,yval)=vol_generator(validation_path)
    #n_train= xtrain.shape[0]
    #print(n_train)
    net = VNet_org(in_channels=1, num_class=1)
    # load optimizor
    optimizer = optim.Adam(net.parameters(), lr=lr)

    mri_train_loader = DataLoader(xtrain,shuffle=False,batch_size=1)
    mask_train_loader = DataLoader(ytrain,shuffle=False,batch_size=1)
    #print(len(train_loader))
    step = 0
    dash =  '-' * 20
    epochend = '='*60

    for epoch in range(epochs):

        print('EPOCH:',epoch)
        net.train()

        epoch_loss = 0
        i=1

        for mri_data,mask_data in zip(mri_train_loader,mask_train_loader):

            print('ith datavol = ',i)
            mri_data = mri_data.unsqueeze(0)
            #print('mri_data shape is =',mri_data.shape)
            mask_data = mask_data.unsqueeze(0)
            #print('mask_data shape is =',mask_data.shape)

            mri_data = torch.tensor(mri_data, dtype=torch.float32)
            mask_data = torch.tensor(mask_data, dtype=torch.float32)

            masks_pred = net(mri_data)
            print('masks_pred shape is =',masks_pred.shape)
            loss = DiceLoss()
            LOSS = loss.forward(mask_data,masks_pred)

            print('LOSS = ',LOSS)
            i=i+1
            optimizer.zero_grad()
            LOSS.backward()
            optimizer.step()
            epoch_loss += LOSS.item()
            #print(dash)
        print(epochend)
        print("epoch %d epochloss:%0.3f" % (epoch, epoch_loss))
    torch.save(net.state_dict(), 'model_weights.pt')
    return net


def test(data_path):

    (training_path,validation_path) = SSData_path(data_path)
        
    (xval,yval)=vol_generator(validation_path)
    mri_val_loader = DataLoader(xval,shuffle=False,batch_size=1)
    mask_val_loader = DataLoader(yval,shuffle=False,batch_size=1)
    model = VNet_org(in_channels=1, num_class=1)
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    dicescore = []
    dash =  '-' * 60
    with torch.no_grad():
        count = 0
        for mri_valdata,mask_valdata in zip(mri_val_loader,mask_val_loader):
            
            mri_valdata = mri_valdata.unsqueeze(0)
            #print('mri_data shape is =',mri_data.shape)
            mask_valdata = mask_valdata.unsqueeze(0)
            #print('mask_data shape is =',mask_data.shape)

            mri_valdata = torch.tensor(mri_valdata, dtype=torch.float32)
            mask_valdata = torch.tensor(mask_valdata, dtype=torch.float32)
            y = model(mri_valdata)
            dicescore.append(dice(y, mask_valdata, smooth=1.0))

            count += 1
        print(dash)    
        print('DICE SCORE for validation data = ',dicescore)

  

       
      
