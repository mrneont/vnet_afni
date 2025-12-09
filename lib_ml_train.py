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

from   lib_fp16util     import network_to_half
import torch.backends.cudnn as cudnn
from   lib_adam_fp16    import Adam16

from torch.cuda.amp import autocast, GradScaler
### NB: couple warnings with autocast when running:
# lib_ml_train.py:221: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
# python3.13/site-packages/torch/amp/grad_scaler.py:136: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
### ... so likely the autocast should be inside a 'device == cuda'-type check?

# --------------------------------------------------------------------------

# List of 'list of various choices'. The choices could be appended
# to the list and there exists an if condition in main training function
# 'train_net()' to choose one of them from the list.  The [0th] choice
# is always the default.  

# List of all possible net architectures to choose from. 
list_net_arch  = [ 'vnet_orig',
                   'Cerebrum',
]

# List of all possible optimizers to choose from. 
list_optimizer = [ 'Adam',
                   'Adam16',
]

# List of all possible data normalizations to choose from. 
list_scale_mode = [ 'z_scoring',      # default
                    'min_max_scale',
]

list_precision = [ 'full',       # default
                   'half',
                   'mixed',
]

# --------------------------------------------------------------------------


def train_net(data_path, num_epochs, lr, tr_bsize, seed, net_arch, 
              loss_func, optimizer, precision, wt_norm, 
              tr_shuf, restart, scale_mode, do_nifti,
              outdir, epoch_mask_list, epoch_chpt_list, 
              verb): 
    """
    Main training function. Sends training to either GPU or CPU.

    Parameters
    ==========

    data_path    : top level directory of data (see program help for  
                   directory sub-structure)
    num_epochs   : number of epochs for network (int)
    lr           : learning rate parameter 
    tr_bsize     : batch size for training 
    seed         : for random number generation in torch (int, or None);
                   if None, no seed is set
    net_arch     : network architecture name, from available list (str)
    loss_func    : loss function name, from available list (str)
    optimizer    : optimizer name, from available list (str)
    precision    : ***
    wt_norm      : normalization of ***
    tr_shuf      : shuffle datasets during training
    restart      : Restart using checkpoint (this var is path to checkpoint)
    scale_mode   : initial brightness scaling of orig dset
    do_nifti     : binary switch about whether to write out nifti dsets 
                   during the network run
    outdir       : directory for various outputs
    epoch_mask_list : list of epoch indices when pred_mask written out
    epoch_chpt_list : list of epoch indices when checkpoint written out
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
        print("++ Running on device: cuda")
    else:
        device = torch.device('cpu')
        print("++ Running on device: cpu")

        if seed != None :
            torch.manual_seed(seed)

    # Some loss functions use weights derived from an additional input
    # 'depth map' (dpth) dataset, and others do not.
    USE_DPTH_WTS = lml.dict_CalcLoss[loss_func]

    if verb :
        print("++ {:40s} : {}".format('Device being used', device))
        print("++ {:40s} : {}".format('Number of epochs', num_epochs))
        print("++ {:40s} : {}".format('Batch size for training', tr_bsize))
        print("++ {:40s} : {}".format('Weight norm', wt_norm))
        print("++ {:40s} : {}".format('Initial scaling mode', scale_mode))
        print("++ {:40s} : {}".format('Loss function', loss_func))
        print("++ {:40s} : {}".format('Using weight datasets', USE_DPTH_WTS))
        print("++ {:40s} : {}".format('Restart using checkpoint weights', restart))
        print("++ {:40s} : {}".format('Network architecture', net_arch))
        print("++ {:40s} : {}".format('Shuffle training datasets', tr_shuf))
        print("++ {:40s} : {}".format('Precision', precision))


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

    # load model from given checkpoint file? (restart is file path)
    if len(restart) :
        if not(os.path.isfile(restart)) :
            print("** ERROR: specified restart file does not exist:", restart)
            sys.exit(3)
        net.load_state_dict(torch.load(restart),strict=False)

    if (device == torch.device('cpu')) and (precision == 'half'):
        print("** The Half Precision operations are not supported in CPU ")
        sys.exit(2)
    
    # move model to device
    net.to(device)
    
    if device == torch.device('cuda') and precision == 'half' :
        net = network_to_half(net)

    # print the model summary
    if verb > 1:
        print('MODEL SUMMARY :\n' )
        print(net)

    # load optimizer
    if optimizer == 'Adam':
        if precision in ['half', 'mixed'] : 
            optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-4)
        else : # precision == 'full'
            optimizer = optim.Adam(net.parameters(), lr=lr)
    
    else:
        print("** This should never happen! 'optimizer' is {}" 
              "".format(optimizer))
        sys.exit(3)


    train_datapath = os.path.join(data_path, 'training')
    train_set      = lmd.mridataset(train_datapath, 
                                    use_dpth_wts=USE_DPTH_WTS, 
                                    mask_available= 0,
                                    verb=verb)
    Ntrain         = len(train_set)
    
    val_datapath   = os.path.join(data_path, 'validation')
    val_set        = lmd.mridataset(val_datapath, 
                                    use_dpth_wts=USE_DPTH_WTS, 
                                    mask_available= 0,
                                    verb=verb)
    Nval           = len(val_set)
   
    dataloaders = {
        'train': DataLoader(train_set, shuffle = tr_shuf, batch_size = tr_bsize),
        'val'  : DataLoader(val_set,   shuffle = False, batch_size = 1)
    }
    print('training shuffle tr_shuf = ',tr_shuf)
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

    if precision == 'mixed' :
        scaler = GradScaler()

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
            if loss_func == 'Sorensen_Dice_mean' :
                loss = lml.CalcLoss_Sorensen_Dice_mean()

            elif loss_func == 'Sorensen_Dice_single_channel':
                loss = lml.CalcLoss_Sorensen_Dice_single_channel()

            elif loss_func == 'WtSorensen_Dice' :
                loss = lml.CalcLoss_WtSorensen_Dice()
            
            else:
                print("This should never happen! 'loss_func' is: {}"
                      "".format(loss_func))
                sys.exit(5)

            idx = 1 # index for the number of steps in an epoch
            for (orig_data, mask_data, dpth_data, orig_fname_tup, index_tup) in dataloaders[phase]:

                
                # orig_data and mask_data start as (full) arrays here
                # dpth_data will be either full array or an empty one
                
                index_dset = np.array2string (index_tup[0].numpy())
                
                #  index_tup is equal to the first element in the tuple 'index_tup'
                index_tup = index_tup[0]
                
                # batchsize should be equal to the length of the index_tup
                # 'index_tup' has indices of the data volumes in the train/validation dataset 
                bsize = len(index_tup)


                
                # ---- scale/normalize the input data in some fashion
                if scale_mode == 'z_scoring':
                    orig_data = lmd.z_scoring(orig_data)

                elif scale_mode == 'min_max_scale': 
                    orig_data = lmd.min_max_scale(orig_data)

                    #[YNS] add the other data normalizations 
                    
                else:
                    print("** ERROR: This should never happen! "
                          "'data normalization' is: {}".format(scale_mode))
                    sys.exit(6)

                # ---- possible GPU niceties
                if (device == torch.device('cuda')) and precision == 'half' :
                    # device == "cuda"
                    orig_data = orig_data.to(device).half()
                    mask_data = mask_data.to(device).half()
                    if USE_DPTH_WTS :
                        dpth_data = dpth_data.to(device).half()
                
                else: 
                    orig_data = orig_data.to(device)
                    mask_data = mask_data.to(device)
                    if USE_DPTH_WTS :
                        dpth_data = dpth_data.to(device)

                if verb > 2 :
                    print("++ {:30s} : {}".format('orig_data.type', 
                                                  orig_data.type()))

                ### CONV3D requires input in the format of:
                ### (batchsz=1, Channels=1, Depth=256, Height=256, width=256)
                # Try to bring each data into the format: (1 X 1 X D X H X W)
                orig_data = orig_data.unsqueeze(1) 
                mask_data = mask_data.unsqueeze(1) 
                dpth_data = dpth_data.unsqueeze(1) 
                
                print('train orig_data size ',orig_data.size())
                print('train mask_data size ',mask_data.size())
                print('train dpth_data size ',dpth_data.size())
                # [PT] Q: should dpth_data also be unsqueezed here, if it
                # is being used?

            
                # zero the parameter gradients
                optimizer.zero_grad()


                # + net.parameters() function returns the network's
                #   learnable/trainable parameters.
                # + The parameters of a layer(in the network) are its
                #   weights and biases.
                # + print(torch.equal(before[j].data, after[j].data))
                #   is a quick and dirty way of finding if the weights
                #   of the network layers are still being
                #   learned(changing) or have become stagnant
                # + torch.equal(before[j].data, after[j].data) is
                #   False means the network's weights are changing
                # + Note: the weights do not change during the validation 
                #   phase  
                #   before = list(net.parameters())[0].clone()
                
                # forward propagation required in both training and
                # validation phase set gradient calculation only for
                # training phase
                with torch.set_grad_enabled(phase == 'train'): 

                    # Invoke Network's forward() method to run
                    # it. This is a Object Oriented way of doing
                    # things.
                    ## reminder: F.conv3d expects the data to be Double

                    # predict the mask using MRI orig_data
                    # The data to the neural net is of type 'torch.FloatTensor'
                    if precision == 'mixed' :                        
                        with autocast():
                            mask_pred = net.forward(orig_data) 
                    #print('pred_mask size',mask_pred.size())
                    # The output mask_pred is of type 'torch.FloatTensor'

                    # Invoke loss function forward() method to run it.
                    # compare the predicted mask and the target data 
                    # + mask_data and mask_data is of type torch.FloatTensor  

                            ### Q: is this indentation correct?
                            ### (inside "with ..."?); now guessing it
                            ### is, but will verify, from above
                            ### indentation of comments
                            if USE_DPTH_WTS :
                                LOSS = loss.forward(mask_pred, mask_data, dpth_data)
                            else:
                                LOSS = loss.forward(mask_pred, mask_data)

                    else: 
                        
                        mask_pred = net.forward(orig_data) 
                        if USE_DPTH_WTS :
                                LOSS = loss.forward(mask_pred, mask_data, dpth_data)
                        else:
                                LOSS = loss.forward(mask_pred, mask_data)
                    '''
                    # Notes about how the dice and loss is calculated when batch_size >1 

                            When the batch size is greater than 2, 
                            For eg., when training batch size = 2, 
                            then orig_data.size() =  torch.Size([2, 1, D, H, W])
                                  mask_data.size() =  torch.Size([2, 1, D, H, W])
                            i.e., [batchsz, Channels, Depth, Height , width ]

                            the dice for the above data will be something like 
                            dice = tensor([[0.8069, 0.2942],
                                           [0.7746, 0.2963]], grad_fn=<DivBackward0>)

                            the return from loss function will be the mean of the dice. 

                            Loss = 1-dice.mean()
                    '''
                    #print('LOSS.item()=',LOSS.item())
                    # the output of loss.forward is a single value of
                    # type 'torch.DoubleTensor'

                # backward propagation (where gradients are computed)
                # and optimization only if in training phase
                if phase == 'train':
                    if precision == 'mixed' :
                        scaler.scale(LOSS).backward()
                        scaler.step(optimizer)
                    else: 
                        LOSS.backward()
                        optimizer.step()
                    #print('LOSS.item()=',LOSS.item())
                    train_losses.append(LOSS.item())

                    # Updates the scale for next iteration.
                    if precision == 'mixed' :
                        scaler.update()
                    # + plot the gradient flow to check poosible
                    #   gradient vanishing / exploding problems.
                    # + named parameters() provide an iterator that
                    #   includes both the parameter label/name and the
                    #   parameter.
                    #ptt.plot_grad_flow(net.named_parameters(), epoch, 
                    #                   idx, outdir = outdir)
                elif phase == 'val':
                    # save the model weights
                    val_losses.append(LOSS.item())
                
                #after = list(net.parameters())[0].clone()
                #for j in range(len(before)):
                    #print("{:30s} : {}".format('Network has stopped learning = ', 
                        #torch.equal(before[j].data, after[j].data)))
                    
                    
                with io.open(loss_file, 'a') as loss_log:
                    loss_log.write("  {:>5d}  {:10.4f}  {:10.4f}  {:10.4f}\n"
                                   "".format(idx, LOSS, 
                                             mask_pred.min(), mask_pred.max()))

                if verb :
                    sss = str(int(index_tup[0]))
                    if len(index_tup) > 1:
                        sss+= "-" + str(int(index_tup[-1]))
                    this_str = "... dset {:11s} / {:5d}".format(sss, Ndset)
                    this_str+= ", loss"
                    print("   {:30s} : {:.4f}".format(this_str, LOSS))

                # [PT] Q: why can't we output dsets if half_prec is True?
                if do_nifti and not(precision == 'half') :

                    ### loop over each dataset in the batchsize(bsize)
                    for bb in range (bsize):
                        index      = index_tup[bb]
                        orig_fname = orig_fname_tup[bb] 
                        
                        if phase == 'train' :
                            orig_head  = train_set.orig_head_list[index] #global lists
                            
                        elif phase == 'val' :
                            orig_head = val_set.orig_head_list[index]
                            
                        else: 
                            orig_head = None

                        # orig_fname is in a form tuple
                        fname_orig, fname_targ, fname_pred_ch00_back, fname_pred_ch01_fore = \
                            lnu.make_names_of_dsets(outdir, orig_fname, strepoch, phase)

                        # this only needs to be written out in first iteration
                        ### Q : do we always want this written out if
                        ### do_nifti is True? should we just let the
                        ### epoch_mask_list tell us when to write out?
                        is_hp = (precision == 'half')
                        if epoch == 0 :
                            # write orig_data in the form of nifti file
                            
                            lnu.write_tensor_to_disk_nifti(orig_data[bb][0], 
                                                        fname=fname_orig,
                                                        head=orig_head, 
                                                        half_prec=is_hp)

                            lnu.write_tensor_to_disk_nifti( mask_data[bb][0], 
                                                        fname=fname_targ,
                                                        head=orig_head, 
                                                        half_prec=is_hp)

                        # [PT] I don't think this needs to be written out
                        # generally, at present.  Just at higher verbosity seems fine?
                        if verb > 3 :
                            lnu.write_tensor_to_disk_nifti( mask_pred[bb][0], 
                                                        fname=fname_pred_ch00_back,
                                                        head=orig_head, 
                                                        half_prec=is_hp)
                        
                        # pred_mask is written into the 'out' folder.
                        if epoch in epoch_mask_list :
                            lnu.write_tensor_to_disk_nifti( mask_pred[bb][1], 
                                                    fname=fname_pred_ch01_fore,
                                                    head=orig_head, 
                                                    half_prec=is_hp)
                        # end of bb loop : for bb in range (bsize)
                    
                    

                idx+= 1 # number of steps in an epoch
                # end of the loop for all batches in the epoch

            end = time.time()
            # end of FOR loop for PHASE

            # save the checkpoint 
            if epoch in epoch_chpt_list and phase == 'train':
                checkpoint_flname = "{}/{}_{}_{}{}".format(outdir,'checkpoint',
                                                 phase, strepoch,'.pt')
                if verb :
                    print("++ Save checkpoint:", checkpoint_flname)
                torch.save(net.state_dict(), checkpoint_flname)

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
        ptt.visualize_loss(avg_train_losses, avg_val_losses, outdir=outdir)

        print("++ {:30s} : {} s".format(' --- total time to now', tot_time))
        print(epochend) # end of FOR loop for EPOCHS

    ### [PT] I *think* these closes are unnecessary, now using io.open()
    loss_log.close()
    perf_log.close()

    return net

