
import os
import sys
import time
import numpy                as np
import matplotlib.pyplot    as plt
from   matplotlib.lines import Line2D
import nibabel              as nib

import torch
from   torch            import optim
from   torch.utils.data import DataLoader
import torch.nn             as nn

import lib_ml_data          as lmd
import lib_ml_models        as lmm
import lib_ml_losses        as lml
import lib_ml_cerebrum      as lmc
import lib_nibabel_utils    as lnu
import pytorchtools         as ptt

# --------------------------------------------------------------------------

# List of all possible net architectures to choose from.  Add any
# others here (the if-condition to use one is below). The [0th]
# one is the default.
list_net_arch = [ 'vnet_orig',
                  'Cerebrum',
                  ]

# List of all possible optimizers to choose from.  Add any others here
# (the if-condition to use one is below). The [0th] one is the
# default.
list_optimizer = [ 'Adam',
                   ]

# --------------------------------------------------------------------------


def plot_grad_flow(named_parameters, epoch, count, outdir = '.'):
    ave_grads = []
    layers = []
    grad_fname      = ("grad_{}_{}.png".format(epoch, count))
    grad_fname_path = '/'.join([outdir, grad_fname])
    plt.figure(figsize=(10,9))
    for n, p in named_parameters:
        #print("n = {}".format(n))
        #print("p = {}".format(p)) # parameter containing tensor 
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            #print("ave_grads = {}".format(ave_grads))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(grad_fname_path, bbox_inches = "tight")
    #fig.savefig(oimage, bbox_inches='tight')

def plot_grad_flow_new(named_parameters,epoch, count, outdir = '.'):
    '''Plots the gradients flowing through different layers in the net
    during training.  Can be used for checking for possible gradient
    vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards()
    as "plot_grad_flow(self.model.named_parameters())" to visualize
    the gradient flow

    '''
    ave_grads = []
    max_grads= []
    layers = []
    grad_fname      = ("grad_{}_{}.png".format(epoch, count))
    grad_fname_path = '/'.join([outdir, grad_fname])
    plt.figure(figsize=(10,9))
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], 
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(grad_fname_path, bbox_inches = "tight")

# one idea of scaling the input dsets, to have a range of values [0,
# 1], to start
def data_normalize(img):
   
    data_min   = img.min()
    data_max   = img.max()
    #print("orig_data max = {}".format(data_max))
    #print("orig_data min = {}".format(data_min))
    normalized = (img - data_min) / (data_max - data_min)

    return normalized

def z_scoring(img):
   
    data_mean  = img.mean()
    data_std   = img.std()

    
    Z_normalized = (img - data_mean) / data_std
    #print("orig_data max = {}".format(Z_normalized.max()))
    #print("orig_data min = {}".format(Z_normalized.min()))

    return Z_normalized

# write the predicated masks into output directory 
# [PT] starting to translate this to having more correct header info.
#    + pieces are still hardwired now, but these should overlay now
def mask_pred_save(mask_pred, epoch, phase, count, outdir = '.'):

    
    mask_pred_np    = mask_pred.cpu().detach().numpy()
    pred_fname      = ("predmask_E{}_{}_{:04d}.nii.gz".format(epoch, 
                                                              phase, 
                                                              count))
    pred_fname_path = '/'.join([outdir, pred_fname])
    #output_image    = nib.Nifti1Image(mask_pred_np, affine=np.eye(4))
    #nib.save(output_image, pred_fname_path)

    lnu.write_out_nifti_vol(mask_pred_np, pred_fname_path,
                            affmat=lnu.TEMP_M44_32iso_nib_ori)

# [PT] starting to translate this to having more correct header info.
#    + pieces are still hardwired now, but these should overlay now
def mask_pred_save_opp(mask_pred_opp, epoch, phase, count, outdir = '.'):

    
    mask_pred_opp_np    = mask_pred_opp.cpu().detach().numpy()
    pred_opp_fname      = ("predmask_opp_E{}_{}_{:04d}.nii.gz".format(epoch, 
                                                                      phase, 
                                                                      count))
    pred_opp_fname_path = '/'.join([outdir, pred_opp_fname])
    #output_image        = nib.Nifti1Image(mask_pred_opp_np, affine=np.eye(4))
    #nib.save(output_image, pred_opp_fname_path)

    lnu.write_out_nifti_vol(mask_pred_opp_np, pred_opp_fname_path,
                            affmat=lnu.TEMP_M44_32iso_nib_ori)


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


def train_net(data_path, epochs, lr, seed, net_arch, loss_func,
              optimizer, wt_norm, outdir, verb): 
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
    net_arch     : network architecture name, from available list (str)
    loss_func    : loss function name, from available list (str)
    optimizer    : optimizer name, from available list (str)
    outdir       : directory for various outputs
    verb         : verbosity for stdout

    Returns
    =======
    """

    # file to track the training and validation loss 
    loss_file = '/'.join([outdir, 'log_loss.txt'])
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
    if net_arch == 'vnet_orig' :
        net = lmm.VNet_orig(in_channels=1, num_class=2, wt_norm = wt_norm, verb=verb)
    elif net_arch == 'Cerebrum' :
        net = lmc.Cerebrum(in_channels=1, num_class=2, wt_norm = wt_norm, verb=verb)
    else:
        print("There is no network architecture here of name: {}"
              "".format(net_arch))
        sys.exit(1)

    print("++ Network architecture type: {}".format(net_arch))

    # move model to device
    if device == 'cuda':
        net.to(device).half()
    else: # device == 'cpu'
        net.to(device)


    # print the model summary
    if verb > 1:
        print('MODEL SUMMARY :\n' )
        print(net)

    # load optimizer
    if optimizer == 'Adam':
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
        print('EPOCH:', epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # start of FOR loop for PHASE
            print("Starting phase: ", phase)

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
            loss_log.write("PHASE  :{}\n".format(phase))
            loss_log.write("{0!s:13} {1!s:10} {2!s:10} {3!s:10}\n"
                           "".format('dataset_num', 'LOSS.item',
                                     'min_val', 'max_val'))

            # creating an instance of loss function
            if 1 :
                loss = lml.CalcLoss_SoftDice_00()

            i = 1 # index for the datafile/volume in the DATASET

            for orig_data, mask_data in dataloaders[phase]:

                
                if 1:
                    orig_data = z_scoring(orig_data)
                else: 
                    orig_data = data_normalize(orig_data)

                if device =='cuda':
                    orig_data = orig_data.to(device).half()
                    mask_data = mask_data.to(device).half()
                else:# device == 'cpu'
                    orig_data = orig_data.to(device)
                    mask_data = mask_data.to(device)

                ### CONV3D requires i/p in the format of:
                ### (batchsz=1, Channels=1, Depth=256, Height=256, width=256)
                # Try to bring each data into the format: (1 X 1 X D X H X W)
                orig_data = orig_data.unsqueeze(0) 
                mask_data = mask_data.unsqueeze(0) 
            
                # zero the parameter gradients
                optimizer.zero_grad()
                #before = list(net.parameters())[0].clone()

                # forward propagation required in both training and
                # validation phase

                # set gradient calculation only for training phase
                with torch.set_grad_enabled(phase == 'train'): 
                    ## reminder: F.conv3d expects the data to be Double

                    # predict the mask using MRI orig_data
                    mask_pred = net.forward(orig_data, verb) 


                    # compare the predicted mask and the target data 
                    LOSS = loss.forward(mask_pred, mask_data)

                # backward propagation and optimization only if in
                # training phase
                if phase == 'train':
                    LOSS.backward()
                    optimizer.step()
                    train_losses.append(LOSS.item())
                    #print('net.named_parameters()')
                    #print(net.named_parameters())
                    #plot_grad_flow_new(net.named_parameters(), 
                    #                   epoch, i, outdir = outdir)
                    #print([z.grad for z in list(net.parameters())])
                    

                elif phase == 'val':
                    # save the model weights
                    val_losses.append(LOSS.item())
                
                #after = list(net.parameters())[0].clone()
                #for j in range(len(before)):
                    #print('CHANGE in Parameters \n')
                    #print(torch.equal(before[j].data, after[j].data))
                loss_log.write(" {}/{} {:15.4f} {:8.2f} {:8.2f}\n "
                               "".format(i, dataset_size, LOSS, 
                                         mask_pred.min(), mask_pred.max()))
                if verb :
                    print("dset : {:5d} / {}  LOSS = {:1.4f}"
                          "".format(i, dataset_size, LOSS))
                
                if epoch == (epochs-1):
                    # save the pred masks in output dir
                    mask_pred_save(mask_pred[0][0], epoch, phase, i, 
                                   outdir = outdir)
                    mask_pred_save_opp(mask_pred[0][1], epoch, phase, i, 
                                   outdir = outdir)
                
                i+= 1 # end of FOR loop for ORIG_DATA, MASK_DATA
            end = time.time() # end of FOR loop for PHASE
            

        # computing the loss pertaining to the training data
        train_losses = torch.as_tensor(train_losses)
        train_loss   = torch.mean(train_losses)

        # computing the loss pertaining to the validation data
        val_losses = torch.as_tensor(val_losses)
        val_loss   = torch.mean(val_losses)

        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)
        loss_log.write(" avg_train_losses {} \n ".format(avg_train_losses))
        loss_log.write(" avg_val_losses {} \n ".format(avg_val_losses))
        early_stopping(val_loss, net)
        visualize_loss(avg_train_losses, avg_val_losses, outdir=outdir)
        print("Time taken for all epochs= {}\n".format(end-start))
        loss_log.write("Time taken for all epochs= {}\n".format(end-start))
        print(epochend) # end of FOR loop for EPOCHS
    loss_log.close()
    return net
