
import os, io
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

from   lib_fp16util     import convert_network
import torch.backends.cudnn as cudnn
from   lib_adam_fp16    import Adam16

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
                   'Adam16',
                   ]

list_data_norm = [  'min_max_scale',
                    'z_scoring',]
# --------------------------------------------------------------------------

def plot_grad_flow_new(named_parameters, strepoch, count, outdir = '.'):
    '''Plot the gradients flowing through different layers in the net
    during training.  Can be used for checking for possible gradient
    vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards()
    as "plot_grad_flow(self.model.named_parameters())" to visualize
    the gradient flow.

    Inputs
    ------
    named_parameters :  iterator over module/net, yielding both the name of the layer 
                        as well as the parameter/weight.

    strepoch         : (str) zeropadded epoch number
    count            : (int) index for the datafile/volume in the DATASET
    outdir           : (str) directory for outputting image

    '''

    ave_grads       = []
    max_grads       = []
    layers          = []
    grad_fname      = ("grad_{}_{}.png".format(strepoch, count))
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
    plt.ylim(bottom=-0.001, top=0.02) # zoom in on the lower gradient regions
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
def min_max_scale(img):
   
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

# --------------------------------------------------------------------------

def train_net(data_path, num_epochs, lr, seed, net_arch, loss_func,
              optimizer, half_prec, wt_norm, data_norm, do_nifti, outdir, verb): 
    """
    Main training function. Sends training to either GPU or CPU.

    Parameters
    ==========

    data_path    : top level directory of data (see program help for  
                   directory sub-structure)
    num_epochs   : number of epochs for network (int)
    lr           : learning rate parameter 
    seed         : for random number generation in torch (int, or None);
                   if None, no seed is set
    net_arch     : network architecture name, from available list (str)
    loss_func    : loss function name, from available list (str)
    optimizer    : optimizer name, from available list (str)
    do_nifti     : binary switch about whether to write out nifti dsets 
                   during the network run
    outdir       : directory for various outputs
    verb         : verbosity for stdout

    Returns
    =======
    """

    # file to track the training and validation loss 
    loss_file_pre = '/'.join([outdir, 'log_loss'])
    summ_file_pre = '/'.join([outdir, 'log_summ'])
    perf_file     = '/'.join([outdir, 'log_performance.txt'])
    dash          = '-' * 20
    epochend      = '=' * 80
    step          = 0         

    # check the device available 
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

        if seed != None :
            torch.manual_seed(seed)

    if verb :
        print("++ {:30s} : {}".format('Device being used', device))
        print("++ {:30s} : {}".format('Number of epochs', num_epochs))
        print("++ {:30s} : {}".format('Weight norm', wt_norm))
        print("++ {:30s} : {}".format('data normalization', data_norm))
        print("++ {:30s} : {}".format('loss function', loss_func))
        print("++ {:30s} : {}".format('Network architecture', net_arch))
        
    if half_prec == 1:
        print("++ {:30s} : {}".format('Precision', 'half'))
    else :
        print("++ {:30s} : {}".format('Precision', 'full'))


    # Task : binary segmentation 
    # in_channels = 1, size = (H X W X Depth): in this case the entire MRI vol
    # num_class = Output channel  = 1 ,  size = (H X W X Depth)
    # num_class = 1 since the task is binary segmentation. 

    # Set up network - Initialize the net with the desired model
    if net_arch == 'vnet_orig' :
        net = lmm.VNet_orig(in_channels=1, num_class=2, wt_norm = wt_norm, 
                            verb=verb)
    elif net_arch == 'Cerebrum' :
        net = lmc.Cerebrum(in_channels=1, num_class=2, wt_norm = wt_norm, 
                           verb=verb)
    else:
        print("** This should never happen! 'network architecture' is {}" 
              "".format(net_arch))
        sys.exit(1)


    if (device == torch.device('cpu')) and (half_prec == 1):
        print("** The Half Precision operations are not supported in CPU ")
        sys.exit(2)
    
    # move model to device
    net.to(device)
    
    if device == torch.device('cuda'):
        if half_prec == 1:
            
            net = convert_network(net, dtype=torch.float16)
            ### the following gives error
            #net = convert_network(net, dtype = torch.cuda.HalfTensor) 
            #torch.backends.cudnn.enabled 
    

    # print the model summary
    if verb > 1:
        print('MODEL SUMMARY :\n' )
        print(net)

    # load optimizer
    if optimizer == 'Adam':
        if half_prec == 1:

            optimizer = Adam16(net.parameters(), lr=lr, 
                           betas=(0.9, 0.999), 
                           eps=1e-8, weight_decay=0)
        else : #(half_prec == 0)
           
            optimizer = optim.Adam(net.parameters(), lr=lr)
    
    else:
        print("** This should never happen! 'optimizer' is {}" 
              "".format(optimizer))
        sys.exit(3)


    train_datapath = os.path.join(data_path, 'training')
    train_set      = lmd.mridataset(train_datapath, verb=verb)
    Ntrain         = len(train_set)
    
    val_datapath   = os.path.join(data_path, 'validation')
    val_set        = lmd.mridataset(val_datapath, verb=verb)
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
   

    with io.open(perf_file, 'a') as perf_log:
        perf_log.write("# {:5s}  "
                       "{:>12s}  {:>12s}  {:>12s} "
                       "{:>12s}  {:>12s}  {:>12s}\n"
                       "".format('epoch', 
                                 'train_loss', 'tr_loss_med', 'tr_loss_std',
                                 'val_loss', 'val_loss_med', 'val_loss_std'))
    
    start = time.time()

    for epoch in range(num_epochs): # start of FOR loop for EPOCHS
        print("++ {:30s} : {}".format('Start of epoch', epoch))
        strepoch  = "{:03d}".format(epoch)
        summ_file = summ_file_pre + '_' + strepoch +'.txt'

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # start of FOR loop for PHASE
            print("++ {:30s} : {}".format('Start of phase', phase))
            loss_file = loss_file_pre + '_' + strepoch + '_' + phase +'.txt'

            if phase == 'train':
                # Set model to training mode
                net.train()  
                train_losses = [] 
                Ndset = Ntrain
            elif phase == 'val':
                # Set model to evaluate mode
                net.eval()   
                val_losses   = []
                Ndset = Nval 
            else:
                print("** This should never happen! 'phase' is: {}"
                      "".format(phase))
                sys.exit(4)

            with io.open(loss_file, 'a') as loss_log:
                loss_log.write("# {:>5s}  {:>10s}  {:>10s}  {:>10s}\n"
                               "".format('dset', 'loss', 
                                         'min_val', 'max_val'))

            # creating an instance of loss function
            if loss_func == 'SoftDice_00' :
                loss = lml.CalcLoss_SoftDice_00()
            elif loss_func == 'WtSoftDice_01' :
                loss = lml.CalcLoss_WtSoftDice_01()
            elif loss_func == 'Sorensen_Dice_mean' :
                loss = lml.CalcLoss_Sorensen_Dice_mean()
            elif loss_func == 'Sorensen_Dice_single_channel' :
                loss = lml.CalcLoss_Sorensen_Dice_single_channel()
            elif loss_func == 'WtSorensen_Dice_03' :
                loss = lml.CalcLoss_WtSorensen_Dice_03()
            elif loss_func == 'WtSorensen_Dice_single_channel' :
                loss = lml.CalcLoss_WtSorensen_Dice_single_channel()

                
            else:
                print("This should never happen! 'loss_func' is: {}"
                      "".format(loss_func))
                sys.exit(5)

            idx = 1 # index for the datafile/volume in the DATASET
            for orig_data, mask_data in dataloaders[phase]:

                if data_norm == 'z_scoring':
                    orig_data = z_scoring(orig_data)

                elif data_norm == 'min_max_scale': 
                    orig_data = min_max_scale(orig_data)

                    #[YNS] add the other data normalizations 
                    
                else:
                    print("This should never happen! 'data normalization' is: {}"
                      "".format(data_norm))
                    sys.exit(6)

                if (device == torch.device('cuda') and (half_prec == 1)): # device == "cuda"
                    
                    orig_data = orig_data.to(device).half()
                    mask_data = mask_data.to(device).half()
                
                else: 

                    orig_data = orig_data.to(device)
                    mask_data = mask_data.to(device)

                if idx < 2 :
                    print("++ {:30s} : {}".format('orig_data.type', 
                                                  orig_data.type()))

                ### CONV3D requires input in the format of:
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
                    # The data to the neural net is of type 'torch.FloatTensor'
                    mask_pred = net.forward(orig_data, verb) 
                    # The output mask_pred is of type 'torch.FloatTensor'

                    # compare the predicted mask and the target data 
                    # + mask_data and mask_data is of type torch.FloatTensor  
                    LOSS = loss.forward(mask_pred, mask_data)
                    # the output of loss.forward is a single value of
                    # type 'torch.DoubleTensor'

                # backward propagation and optimization only if in
                # training phase
                if phase == 'train':
                    LOSS.backward()
                    optimizer.step()
                    train_losses.append(LOSS.item())
                    #print('net.named_parameters()')
                    #print(net.named_parameters())
                    #plot_grad_flow_new(net.named_parameters(), 
                                    #epoch, idx, outdir = outdir)
                    #print([z.grad for z in list(net.parameters())])
                elif phase == 'val':
                    # save the model weights
                    val_losses.append(LOSS.item())
                
                #after = list(net.parameters())[0].clone()
                #for j in range(len(before)):
                    #print('CHANGE in Parameters \n')
                    #print(torch.equal(before[j].data, after[j].data))
                with io.open(loss_file, 'a') as loss_log:
                    loss_log.write("  {:>5d}  {:10.4f}  {:10.4f}  {:10.4f}\n"
                                   "".format(idx, LOSS, 
                                             mask_pred.min(), mask_pred.max()))

                if verb :
                    this_str = "... dset {:5d} / {:5d}".format(idx, Ndset)
                    this_str+= ", loss"
                    print("   {:30s} : {:.4f}"
                          "".format(this_str, LOSS))

                if (do_nifti and not(half_prec)) :
                    pref_targ = "{}_{}_{}_{:04d}".format( 'target', 
                                                          strepoch, 
                                                          phase, 
                                                          idx )
                    fname_targ = "{}/{}.nii.gz".format( outdir,
                                                        pref_targ )
                    lnu.write_tensor_to_disk_nifti( mask_data[0][0], 
                                                    fname=fname_targ )

                    pref_pred = "{}_{}_{}_{:04d}".format( 'predmask', 
                                                          strepoch, 
                                                          phase, 
                                                          idx )
                    fname_pred = "{}/{}.nii.gz".format( outdir,
                                                        pref_pred )
                    lnu.write_tensor_to_disk_nifti( mask_pred[0][0], 
                                                    fname=fname_pred )

                    pref_pred_OPP = "{}_{}_{}_{:04d}".format( 'predmask_OPP',
                                                              strepoch, 
                                                              phase, 
                                                              idx )
                    fname_pred_OPP = "{}/{}.nii.gz".format( outdir,
                                                            pref_pred_OPP )
                    lnu.write_tensor_to_disk_nifti( mask_pred[0][1], 
                                                    fname=fname_pred_OPP )

                
                idx+= 1 # end of FOR loop for ORIG_DATA, MASK_DATA

            end = time.time() # end of FOR loop for PHASE

        # computing the loss pertaining to the training data
        train_losses = torch.as_tensor(train_losses)
        train_loss   = torch.mean(train_losses)

        # computing the loss pertaining to the validation data
        val_losses = torch.as_tensor(val_losses)
        val_loss   = torch.mean(val_losses)

        with io.open(perf_file, 'a') as perf_log:
            perf_log.write("  {:5d}  "
                           "{:12.4f}  {:12.4f}  {:12.4f} "
                           "{:12.4f}  {:12.4f}  {:12.4f}\n"
                           "".format(epoch,
                                     train_loss, torch.median(train_losses),
                                     torch.std(train_losses),
                                     val_loss, torch.median(val_losses),
                                     torch.std(val_losses)))

        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        tot_time = "{:0.6f}".format(end-start)
        with io.open(summ_file, 'a') as summ_log:
            summ_log.write("train_loss_mean   : {:0.6f}\n".format(train_loss))
            summ_log.write("val_loss_mean     : {:0.6f}\n".format(val_loss))
            summ_log.write("total time to now : {}\n".format(tot_time))

        early_stopping(val_loss, net)
        visualize_loss(avg_train_losses, avg_val_losses, outdir=outdir)

        print("++ {:30s} : {} s".format(' --- total time to now', tot_time))
        print(epochend) # end of FOR loop for EPOCHS

    ### [PT] I *think* these closes are unnecessary, now using io.open()
    loss_log.close()
    perf_log.close()

    return net
