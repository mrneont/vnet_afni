
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

# -----------------------------------------------------------------------

#data dimensions of the volume  
#data_dims = (256, 256, 256)


# number of slices in cor, sag and axl
#num_slices_sag = data_dims[0]
#num_slices_cor = data_dims[1]
#num_slices_axl = data_dims[2]
def data_normalize(img):
   
    
    #mean = img.mean()
    #std = img.std()
    data_min = img.min()
    data_max = img.max()
    normalized = (img - data_min) / (data_max -data_min )
    #normalized = (img - mean) / std
    return normalized

def visualize_loss(avg_train_losses, avg_val_losses, outdir = '.'):

    oimage = '/'.join([outdir, 'loss_plot.png'])

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_losses)+1), avg_train_losses, 
             label='Training Loss')
    plt.plot(range(1,len(avg_val_losses)+1), avg_val_losses,
             label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_val_losses.index(min(avg_val_losses))+1 
    
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig(oimage, bbox_inches='tight')



def train_net(data_path, epochs, lr, seed, outdir, verb):
    """
    Main training function. Sends training to either GPU or CPU.

    Parameters
    ==========

    data_path    : top level directory of data (see program help for  
                   directory sub-structure)
    epochs       : number of epochs for network (int)
    lr           : learning rate parameter 
    seed         : for random number generation in torch (int, or None);
                   if None, no seed is set
    outdir       : directory for various outputs
    verb         : verbosity for stdout

    Returns
    =======

    [***the full network?  maybe describe more what this is...]

    """
    # check the device available 
    loss_file  = '/'.join([outdir,'log_loss.txt'])
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

        if seed != None :
            torch.manual_seed(seed)

    if verb :
        print('DEVICE BEING USED :', device)
        print('NUMBER OF EPOCHS  :', epochs)

    loss_log  = open(loss_file, mode='w')

    loss_log.write("DEVICE BEING USED  : {}\n".format(device))
    loss_log.write("NUMBER OF EPOCHS  : {}\n".format(epochs))
    
    

    # Here, get the paths, and then make lists of the training and
    # validation dsets.  In both cases, the 'x*' member is the 'orig'
    # dset, and the 'y*' member is the mask dset.
    #(training_path, validation_path) = lmd.SSData_path(data_path)
    path_train, path_val     = lmd.SSData_path(data_path)

    # populate the matrices from the dataset
    ### [PT: Apr 5, 2021] maybe this should be called
    ### "mat_generator()", since that is what it generates?
    ##### [PT] renaming output to be more descriptive: these appear to
    ##### be matrices, of training data, for orig and mask dsets 
    #orig_train, mask_train   = lmd.vol_generator(path_train, verb=verb)
    #orig_val, mask_val       = lmd.vol_generator(path_val,   verb=verb)
    orig_train_mat, mask_train_mat = lmd.mat_generator(path_train, verb=verb)
    orig_val_mat,   mask_val_mat   = lmd.mat_generator(path_val,   verb=verb)

    # dataset size
    #Ntrain = orig_train.shape[0]
    #Nval   = orig_val.shape[0]
    Ntrain = orig_train_mat.shape[0]
    Nval   = orig_val_mat.shape[0]

    if verb:
        print('TRAINING DATASET PATH   :', path_train)
        print('VALIDATION DATASET PATH :', path_val)
        print('ORIGINAL TRAINING DATASET SIZE   :', Ntrain)
        print('ORIGINAL VALIDATION DATASET SIZE :', Nval)


    loss_log.write("TRAINING DATASET PATH   :{}\n".format(path_train))
    loss_log.write("VALIDATION DATASET PATH   :{}\n".format(path_val))
    loss_log.write("ORIGINAL TRAINING DATASET SIZE    :{}\n".format(Ntrain))
    loss_log.write("ORIGINAL VALIDATION DATASET SIZE   :{}\n".format(Nval))

    
    
        
    # Set up network

    # Task : binary segmentation 
    # in_channels = 1 , size = (H X W X Depth): in this case the entire MRI vol
    # num_class = Output channel  = 1 ,  size = (H X W X Depth)
    # num_class = 1 since the task is binary segmentation. 

    net = lmm.VNet_org(in_channels=1, num_class=1,verb=verb)

    # move model to device
    net.to(device)

    # print the model summary
    if verb :
        print('MODEL SUMMARY :\n' )
        print(net)

    # load optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    #pytorch dataloader 
    orig_train_loader = DataLoader(orig_train_mat, shuffle=False, batch_size=1)
    mask_train_loader = DataLoader(mask_train_mat, shuffle=False, batch_size=1)
    orig_val_loader   = DataLoader(orig_val_mat,   shuffle=False, batch_size=1)
    mask_val_loader   = DataLoader(mask_val_mat,   shuffle=False, batch_size=1)
    #print('orig_train_loader  DATA TYPE =',orig_train_loader[0].dtype)
    #print('mask_train_loader DATA TYPE     =',mask_train_loader[0].dtype)
    
    step     = 0
    dash     = '-' * 20
    epochend = '=' * 80

    # initialize the early_stopping object
    patience         = 5
    early_stopping   = ptt.EarlyStopping(patience=patience, verbose=True)
    avg_train_losses = []
    avg_val_losses = []
    start            = time.time()


    for epoch in range(epochs):

        if verb:
            print('EPOCH:', epoch)
        
        i=1
        
        loss_log.write("EPOCH  :{}\n".format(epoch))
       
        ###################
        # train the model #
        ###################
        train_losses=[]
        net.train() # prep model for training
        for orig_data, mask_data in zip(orig_train_loader, mask_train_loader):


            if i==1 :
                loss_log.write("ORIGINAL TRAINING DATA DIM : {}\n".format(orig_data.shape))
                loss_log.write("MASK TRAINING DATA DIM : {}\n".format(mask_data.shape))
                loss_log.write("ORIGINAL TRAINING DATA TYPE : {}\n".format(orig_data.dtype))
                loss_log.write("MASK TRAINING DATA TYPE : {}\n".format(mask_data.dtype))
                
                

            
            orig_data = data_normalize(orig_data)
            if i==1 :
                loss_log.write(" DATA NORMALISED (0,1) \n")
                loss_log.write(" ORIG DATA MAX : {}".format(orig_data.max()))
                loss_log.write(" ""MIN : {}\n".format(orig_data.min()))
            orig_data = orig_data.to(device)
            mask_data = mask_data.to(device)
            ### CONV3D requires i/p in the format of:
            ### (batchsz =1, Channels=1, Depth =256, Height=256 width=256)
            # So, try to bring each data into the format: (1 X 1 X D X H X W)
            orig_data = orig_data.unsqueeze(0) 
            mask_data = mask_data.unsqueeze(0) 
            
            # F.conv3d expects the data to be Double
            mask_train_pred = net(orig_data,verb)

            if verb > 1 :
                print('PREDICTED MASK SIZE :', mask_train_pred.shape)

            if i==1 :
                loss_log.write("PREDICTED MASK SIZE : {}\n".format(mask_train_pred.shape))

            loss_log.write("  \n")
            loss_log.write("PREDICTED MASK MAX : {}".format(mask_train_pred.max()))
            loss_log.write(" ""MIN : {}\n".format(mask_train_pred.min()))
            # creating an instance of loss function
            loss = lml.DiceLoss()
            # compare the predicted mask and the target data 
            LOSS = loss.forward(mask_train_pred, mask_data)

            if verb :
                print('training dset : {:5d} / {}'.format(i, Ntrain))
                print('LOSS = ', LOSS)

            loss_log.write("training dataset : {:5d}/{} ".format(i, Ntrain))
            loss_log.write(" ""LOSS: {}\n".format(LOSS))
            i=i+1
            optimizer.zero_grad()
            LOSS.backward()
            optimizer.step()
            train_losses.append(LOSS.item())
            #print(dash)

        

        ######################    
        # validate the model #
        ######################
        net.eval() # prep model for evaluation
        val_losses = []
        with torch.no_grad():
            count = 0
            for orig_valdata, mask_valdata \
                in zip(orig_val_loader, mask_val_loader):
                
                orig_valdata  = orig_valdata.to(device)
                mask_valdata  = mask_valdata.to(device)
                ### CONV3D requires i/p in the format of: (batchsz = 1,
                ###   Channels=1, Depth =256, Height=256, width=256)
                # So, trying to bring the data into the format of (1 X
                # 1 X D X H X W)
                orig_valdata  = orig_valdata.unsqueeze(0)
                mask_valdata  = mask_valdata.unsqueeze(0)

                #RuntimeError: expected scalar type Double but found Float
                #F.conv3d expects the data to be Double, hence typecasting 
                #orig_valdata  = torch.tensor(orig_valdata, dtype=torch.float32)
                #mask_valdata  = torch.tensor(mask_valdata, dtype=torch.float32)

                if verb > 1 :
                    print('ORIGINAL VALIDATION DATA DIM  =', orig_valdata.shape)
                    print('MASK VALIDATION DATA DIM      =', mask_valdata.shape)
                    print('ORIGINAL VALIDATION DATA TYPE =', orig_valdata.dtype)
                    print('MASK VALIDATION DATA TYPE     =', mask_valdata.dtype)

                mask_val_pred = net(orig_valdata,verb)
                val_losses.append(lml.dice(mask_val_pred, mask_valdata, 
                                             smooth=1.0))

                print('validating dset : {:5d} / {}'.format(count, Nval))
                
                loss_log.write("validating dataset : {:5d}/{} ".format(count, Nval))
                loss_log.write(" ""LOSS: {}\n".format(val_losses[count]))
                count += 1
                


                
                
        # calculate average loss over an epoch
        
        end = time.time()
        #print(f"Runtime of the program is {end - start}")
        print("Runtime of the program is {}".format(end - start))

        # computing the loss pertaining to the training data
        train_losses = torch.as_tensor(train_losses)
        train_loss   = torch.mean(train_losses)
        # computing the loss pertaining to the validation data
        val_losses = torch.as_tensor(val_losses)
        val_loss   = torch.mean(val_losses)

        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)
        if verb :
            print('AVG TRAINING LOSS   = ', avg_train_losses)
            print('AVG VALIDATION LOSS = ', avg_val_losses)
        if early_stopping.early_stop:
            print("EARLY STOPPING")
            break


        loss_log.write("AVG TRAINING LOSS : {}\n".format(avg_train_losses))
        loss_log.write("AVG VALIDATION LOSS  : {}\n".format(avg_val_losses))
        loss_log.write("RUN TIME OF PROGRAM : {}\n".format(end - start))
        print(epochend)
        # early_stopping needs the validation loss to check if it has
        # decreased, and if it has, it will make a checkpoint of the
        # current model
        early_stopping(val_loss, net)
        visualize_loss(avg_train_losses, avg_val_losses, outdir=outdir)
        
    #torch.save(net.state_dict(), 'model_weights.pt')
    loss_log.write("EPOCH END : {}\n".format(epochend))
    loss_log.close()
    return net


def test(data_path):

    # DATA PATH
    training_path, validation_path = lmd.SSData_path(data_path)
    
     # populate the matrices from the dataset
    orig_val, mask_val = lmd.mat_generator(validation_path)

    # pytorch data loaders
    orig_val_loader = DataLoader(orig_val, shuffle=False, batch_size=1)
    mask_val_loader = DataLoader(mask_val, shuffle=False, batch_size=1)

    # set up the network
    # Task : binary segmentation 
    ### in_channels = 1 , size = (H X W X Depth) : in this case the
    ### entire MRI volume
    # num_class = Output channel  = 1 , size = (H X W X Depth)
    # num_class = 1 since the task is binary segmentation. 
    model  = lmm.VNet_org(in_channels=1, num_class=1)

    # load the weights of the model
    model.load_state_dict(torch.load('model_weights.pt'))

    # prep the model for evaluation mode
    model.eval()

    dicescore = []
    dash =  '-' * 60

    # the gradients are not altered during the validation phase
    with torch.no_grad(): 
        count = 0
        for orig_valdata, mask_valdata in zip(orig_val_loader, mask_val_loader):
            
            orig_valdata = orig_valdata.unsqueeze(0)
            
            mask_valdata = mask_valdata.unsqueeze(0)
            
            orig_valdata = torch.tensor(orig_valdata, dtype=torch.float32)
            mask_valdata = torch.tensor(mask_valdata, dtype=torch.float32)

            if verb:
                print('ORIGINAL VALIDATION DATA DIM  =',orig_valdata.shape)
                print('MASK VALIDATION DATA DIM      =',mask_valdata.shape)
                print('ORIGINAL VALIDATION DATA TYPE =',orig_valdata.dtype)
                print('MASK VALIDATION DATA TYPE     =',mask_valdata.dtype)

            mask_val_pred = model(orig_valdata)

            dicescore.append(lml.dice(mask_val_pred, mask_valdata, smooth=1.0))

            count += 1
        print(dash)    
        print('DICE SCORE for validation data = ', dicescore)

  

       
      
