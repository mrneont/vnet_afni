
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
import nibabel as nib
import lib_ml_cerebrum as lmc
import os


# one idea of scaling the input dsets, to have a range of values [0,
# 1], to start
def data_normalize(img):
   
    data_min   = img.min()
    data_max   = img.max()
    normalized = (img - data_min) / (data_max - data_min)

    return normalized

# write the predicated masks into output directory 
def mask_pred_save(mask_pred, phase, count, outdir = '.'):

    mask_pred_sq    = torch.squeeze(mask_pred) # squeeze the channel dimension
    mask_pred_sq_np = mask_pred_sq.cpu().detach().numpy()
    pred_fname      = ("predmask_{}_{:04d}.nii.gz".format(phase, count))
    pred_fname_path = '/'.join([outdir, pred_fname])
    output_image    = nib.Nifti1Image(mask_pred_sq_np, affine=np.eye(4))

    nib.save(output_image, pred_fname_path)

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
    """

    # file to track the training and validation loss 
    loss_file = '/'.join([outdir,'log_loss.txt'])
    dash      = '-' * 20
    epochend  = '=' * 80
    step      = 0         

    # check the device available 
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

        if seed != None :
            torch.manual_seed(seed)

    if verb :
        print('DEVICE BEING USED :', device)
        print('NUMBER OF EPOCHS  :', epochs)

    
    # Task : binary segmentation 
    # in_channels = 1, size = (H X W X Depth): in this case the entire MRI vol
    # num_class = Output channel  = 1 ,  size = (H X W X Depth)
    # num_class = 1 since the task is binary segmentation. 

     # Set up network - Initialize the net with the desired model
    net = lmm.VNet_org(in_channels=1, num_class=1,verb=verb)
    #net = lmc.Cerebrum(in_channels=1, num_class=1,verb=verb)
    
    # move model to device
    net.to(device)

    # print the model summary
    if verb > 1:
        print('MODEL SUMMARY :\n' )
        print(net)

    # load optimizer
    optimizer      = optim.Adam(net.parameters(), lr=lr)

    train_datapath = os.path.join(data_path, 'training')
    train_set      = lmd.mridataset(train_datapath)
    Ntrain         = len(train_set)
    
    val_datapath   = os.path.join(data_path, 'validation')
    val_set        = lmd.mridataset(val_datapath)
    Nval           = len(val_set)
   
    dataloaders = {
        'train': DataLoader(train_set, shuffle=False, batch_size=1),
        'val'  : DataLoader(val_set,   shuffle=False, batch_size=1)
    }

    # initialize the early_stopping object
    patience         = 5
    early_stopping   = ptt.EarlyStopping(patience=patience, verbose=True)
    avg_train_losses = []   # calculated over the entire dataset in an epoch
    avg_val_losses   = []   # calculated over the entire dataset in an epoch
    start            = time.time()

    for epoch in range(epochs): # start of FOR loop for EPOCHS

        if verb:
            print('EPOCH:', epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # start of FOR loop for PHASE
            if phase == 'train':
                # Set model to training mode
                net.train()  
                train_losses = [] 
                dataset_size = Ntrain
            elif phase == 'val':
                # Set model to evaluate mode
                net.eval()   
                val_losses   = []
                dataset_size = Nval 
            else:
                print("This should never happen! 'phase' is: {}"
                      "".format(phase))

            loss_log  = open(loss_file, mode='a')
            loss_log.write("EPOCH  :{}\n".format(epoch))

            print("Starting phase: ", phase)

            # creating an instance of loss function
            loss_log.write("PHASE  :{}\n".format(phase))
            loss_log.write("{0!s:13} {1!s:10} {2!s:10} {3!s:10}\n"
                           "".format('dataset_num', 'LOSS.item',
                                     'min_val', 'max_val'))
            loss = lml.get_dice()

            i = 1 # index for the datafile/volume in the DATASET

            for orig_data, mask_data in dataloaders[phase]:

                orig_data = data_normalize(orig_data)
           
                orig_data = orig_data.to(device)
                mask_data = mask_data.to(device)

                ### CONV3D requires i/p in the format of:
                ### (batchsz=1, Channels=1, Depth=256, Height=256, width=256)
                # Try to bring each data into the format: (1 X 1 X D X H X W)
                orig_data = orig_data.unsqueeze(0) 
                mask_data = mask_data.unsqueeze(0) 
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward propagation required in both training and
                # validation phase

                # set gradient calculation only for training phase
                with torch.set_grad_enabled(phase == 'train'): 

                    # predict the mask using MRI orig_data
                    mask_pred = net(orig_data, verb) 

                    ## reminder: F.conv3d expects the data to be Double

                    # compare the predicted mask and the target data 
                    LOSS = loss.forward(mask_pred, mask_data)

                # backward propagation and optimization only if in
                # training phase
                if phase == 'train':
                    LOSS.backward()
                    optimizer.step()
                    train_losses.append(LOSS.item())

                elif phase == 'val':
                    # save the model weights
                    val_losses.append(LOSS.item())
                    
                loss_log.write(" {}/{} {:15.4f} {:8.2f} {:8.2f}\n "
                               "".format(i, dataset_size, LOSS, 
                                         mask_pred.min(), mask_pred.max()))
                if verb :
                    print("dset : {:5d} / {}  LOSS = {:1.4f}"
                          "".format(i, dataset_size, LOSS))
                
                if epoch == (epochs-1):
                    # save the pred masks in output dir
                    mask_pred_save(mask_pred, phase, i, outdir = outdir)
                
                i+= 1 # end of FOR loop for ORIG_DATA, MASK_DATA
            end = time.time() # end of FOR loop for PHASE
            loss_log.close()
        # computing the loss pertaining to the training data
        train_losses = torch.as_tensor(train_losses)
        train_loss   = torch.mean(train_losses)

        # computing the loss pertaining to the validation data
        val_losses = torch.as_tensor(val_losses)
        val_loss   = torch.mean(val_losses)

        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)
        early_stopping(val_loss, net)
        visualize_loss(avg_train_losses, avg_val_losses, outdir=outdir)

        print(epochend) # end of FOR loop for EPOCHS

    #loss_log.close()
    return net
