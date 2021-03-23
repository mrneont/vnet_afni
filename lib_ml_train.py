
import sys
import time
import numpy                as np
import matplotlib.pyplot    as plt
import torch

from   torch            import optim
from   torch.utils.data import DataLoader
import torch.nn             as nn

import lib_ml_data          as lmd
import lib_ml_models        as lmm
import lib_ml_losses        as lml
import pytorchtools         as ptt

### unused:
#from   tqdm             import tqdm

### import whole file with abbrev, to see where functions are more
### easily:
#from   lib_ml_data      import SSData_path, vol_generator
#from   lib_ml_models    import VNet_org
#from   lib_ml_losses    import DiceLoss, dice
#from   pytorchtools     import EarlyStopping

# -----------------------------------------------------------------------

# data_path contains the dataset currently being used to train the model
# each dataset contains the training set and the validation set
#data_path = '/Users/yamunasn/Vnet_afni/dataset/pretrain_vnet'

#data dimensions of the volume  
#data_dims = (256, 256, 256)


# number of slices in cor, sag and axl
#num_slices_sag = data_dims[0]
#num_slices_cor = data_dims[1]
#num_slices_axl = data_dims[2]
def visualize_loss(avg_train_losses,avg_valid_losses):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_losses)+1), avg_train_losses, 
             label='Training Loss')
    plt.plot(range(1,len(avg_valid_losses)+1), avg_valid_losses,
             label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')



def train_net(data_path, epochs, lr, verb):
    """
    Main training function. Sends training to either GPU or CPU.

    Parameters
    ==========

    data_path    : top level directory of data (see program help for  
                   directory sub-structure)
    epochs       : number of epochs for network (int)
    lr           : learning rate parameter 
    verb         : verbosity for stdout

    Returns
    =======

    [***the full network?  maybe describe more what this is...]

    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if verb :
        print('The device being used is =', device)
        print('The number of epochs is  =', epochs)

    # Here, get the paths, and then make lists of the training and
    # validation dsets.  In both cases, the 'x*' member is the 'orig'
    # dset, and the 'y*' member is the mask dset.
    (training_path, validation_path) = lmd.SSData_path(data_path)
    #traingen = lmd.vol_generator(training_path)
    #print(len(traingen))
    (xtrain, ytrain) = lmd.vol_generator(training_path, verb=verb)
    (xval, yval)     = lmd.vol_generator(validation_path, verb=verb)

    Ntrain = xtrain.shape[0]
    Nval   = xval.shape[0]

    #n_train= xtrain.shape[0]
    #print(n_train)

    # Set up network
    ### [PT] Q: the number of channels here is determined by.... ?
    ### and the number of classes is determined by having a binary
    ### mask, right?
    net = lmm.VNet_org(in_channels=1, num_class=1)
    net.to(device)

    # load optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    mri_train_loader  = DataLoader(xtrain, shuffle=False, batch_size=1)
    mask_train_loader = DataLoader(ytrain, shuffle=False, batch_size=1)
    mri_val_loader    = DataLoader(xval,   shuffle=False, batch_size=1)
    mask_val_loader   = DataLoader(yval,   shuffle=False, batch_size=1)
    #print(len(train_loader))
    step     = 0
    dash     = '-' * 20
    epochend = '='*60

    # initialize the early_stopping object
    patience         = 5
    early_stopping   = ptt.EarlyStopping(patience=patience, verbose=True)
    avg_train_losses = []
    avg_valid_losses = []
    
    start = time.time()
    for epoch in range(epochs):

        print('EPOCH:', epoch)
        
        i=1
        
        ###################
        # train the model #
        ###################
        train_losses=[]
        net.train() # prep model for training
        for mri_data, mask_data in zip(mri_train_loader, mask_train_loader):

            print('training dset : {:5d} / {}'.format(i, Ntrain))

            mri_data, mask_data = mri_data.to(device), mask_data.to(device)
            mri_data  = mri_data.unsqueeze(0)   # 1 x (dimensions of dset)
            #print('mri_data shape is =',mri_data.shape)
            mask_data = mask_data.unsqueeze(0)
            #print('mask_data shape is =',mask_data.shape)

            mri_data   = torch.tensor(mri_data, dtype=torch.float32)
            mask_data  = torch.tensor(mask_data, dtype=torch.float32)

            masks_pred = net(mri_data)
            if verb :
                print('masks_pred shape is =', masks_pred.shape)
            loss = lml.DiceLoss()
            LOSS = loss.forward(mask_data,masks_pred)

            if verb :
                print('LOSS = ', LOSS)
            i=i+1
            optimizer.zero_grad()
            LOSS.backward()
            optimizer.step()
            train_losses.append(LOSS.item())
            #print(dash)

        print(epochend)

        ######################    
        # validate the model #
        ######################
        net.eval() # prep model for evaluation
        valid_losses = []
        with torch.no_grad():
            count = 0
            for mri_valdata,mask_valdata in zip(mri_val_loader,mask_val_loader):
                
                ## [PT] Q: why is this one line?  this doesn't look
                ## like it has to be a tuple on LHS, so why not be 2
                ## lines?
                mri_valdata, mask_valdata = mri_valdata.to(device), mask_valdata.to(device)
                mri_valdata = mri_valdata.unsqueeze(0)
                #print('mri_data shape is =',mri_data.shape)
                mask_valdata = mask_valdata.unsqueeze(0)
                #print('mask_data shape is =',mask_data.shape)

                mri_valdata  = torch.tensor(mri_valdata,  dtype=torch.float32)
                mask_valdata = torch.tensor(mask_valdata, dtype=torch.float32)
                y = net(mri_valdata)
                valid_losses.append(lml.dice(y, mask_valdata, smooth=1.0))

                count += 1
                #print('validatecount=', count)
                print('validating dset : {:5d} / {}'.format(count, Nval))

        
        # calculate average loss over an epoch
        
        end = time.time()
        print(f"Runtime of the program is {end - start}")
        print("Runtime of the program is {}".format(end - start))

        #train_loss = np.average(train_losses)
        #train_loss = np.mean(train_losses)
        train_losses = torch.as_tensor(train_losses)
        train_loss   = torch.mean(train_losses)
        #train_loss = torch.mean(train_losses).detach().cpu().numpy()
        #valid_loss = np.average(valid_losses)
        valid_losses = torch.as_tensor(valid_losses)
        valid_loss   = torch.mean(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print('avg training loss is   = ', avg_train_losses)
        print('avg validation loss is = ', avg_valid_losses)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # early_stopping needs the validation loss to check if it has
        # decreased, and if it has, it will make a checkpoint of the
        # current model
        early_stopping(valid_loss, net)
        visualize_loss(avg_train_losses,avg_valid_losses)
    #torch.save(net.state_dict(), 'model_weights.pt')
    return net


def test(data_path):

    (training_path, validation_path) = lmd.SSData_path(data_path)
        
    (xval, yval)    = lmd.vol_generator(validation_path)
    mri_val_loader  = DataLoader(xval,shuffle=False,batch_size=1)
    mask_val_loader = DataLoader(yval,shuffle=False,batch_size=1)
    model = lmm.VNet_org(in_channels=1, num_class=1)
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    dicescore = []
    dash =  '-' * 60
    with torch.no_grad():
        count = 0
        for mri_valdata, mask_valdata in zip(mri_val_loader, mask_val_loader):
            
            mri_valdata = mri_valdata.unsqueeze(0)
            #print('mri_data shape is =',mri_data.shape)
            mask_valdata = mask_valdata.unsqueeze(0)
            #print('mask_data shape is =',mask_data.shape)

            mri_valdata = torch.tensor(mri_valdata, dtype=torch.float32)
            mask_valdata = torch.tensor(mask_valdata, dtype=torch.float32)
            y = model(mri_valdata)
            dicescore.append(lml.dice(y, mask_valdata, smooth=1.0))

            count += 1
        print(dash)    
        print('DICE SCORE for validation data = ', dicescore)

  

       
      
