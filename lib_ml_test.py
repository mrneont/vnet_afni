import os, io
import sys
import time
import numpy                as np
import nibabel              as nib
import argparse             as argp

import torch
from   torch            import optim
from   torch.utils.data import DataLoader
import torch.nn 

import lib_ml_data          as lmd
import lib_ml_models        as lmm
import lib_ml_losses        as lml
import lib_nibabel_utils    as lnu
import run_ml_ss            as rms



dicescore = []

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def get_test_args():

    parser = argp.ArgumentParser(prog = 'lib_ml_test.py',
                                formatter_class=argp.RawTextHelpFormatter)

    parser.add_argument('-d', "--data_path",type=dir_path)

    def_OD = '.'
    parser.add_argument('-o', '--outdir')

    return parser.parse_args()

def test_net(data_path,outdir):

    # Set up network 
    model = lmm.VNet_orig(in_channels=1, num_class=2, wt_norm = 0, 
                         verb=0)
   
    model.load_state_dict(torch.load('checkpoint.pt'),strict=False)

    model.eval()

    # datapath
    # DataLoader setup
    test_datapath = os.path.join(data_path, 'validation')
    test_set      = lmd.mridataset(test_datapath, 
                                    use_dpth_wts= 0, 
                                    verb=0)
    
    Ntest         = len(test_set)

    test_dataloader = DataLoader(test_set,shuffle=False,batch_size=1)
    
    # creating an instance of loss function
    loss = lml.CalcLoss_Sorensen_Dice_mean()

    
    dash =  '-' * 60

    with torch.no_grad():
        
        count = 1
        
        for (orig_data, mask_data, dpth_data, orig_fname) in test_dataloader:
            
            # dummy variables in testing
            phase = 'test'
            strepoch = 1
            # ---- scale/normalize the input data in some fashion
            # [YNS] include options for other normalization    
            orig_data = lmd.z_scoring(orig_data)

            # CONV3D requires input in the format of:
            # (batchsz=1, Channels=1, Depth=256, Height=256, width=256)
            # Try to bring each data into the format: (1 X 1 X D X H X W)
            orig_data = orig_data.unsqueeze(1) 
            mask_data = mask_data.unsqueeze(1) 
            print('data size',orig_data.size())


  
            idxm1 = count - 1
            orig_head = test_set.orig_head_list[idxm1]
                


            pred_mask = model.forward(orig_data)
            print('pred_mask size',pred_mask.size())

            fname_orig, fname_targ, fname_pred_ch00_back, fname_pred_ch01_fore = \
                        lnu.make_names_of_dsets(outdir, orig_fname[0], 1, phase)

            lnu.write_tensor_to_disk_nifti(orig_data[0][0], 
                                                        fname=fname_orig,
                                                        head=orig_head)

            lnu.write_tensor_to_disk_nifti( mask_data[0][0], 
                                                        fname=fname_targ,
                                                        head=orig_head)


            lnu.write_tensor_to_disk_nifti( pred_mask[0][1], 
                                                    fname=fname_pred_ch01_fore,
                                                    head=orig_head )

            
            LOSS = loss.forward(pred_mask, mask_data)

            dicescore.append(LOSS.item())
           

            count += 1
            print('count =',count)
        print(dash)  
        print('DICE SCORE for validation data = ',dicescore)
        print(dash) 
        # pending[YNS] : write the dice scores into a log file in output directory   


def main():
    args      = get_test_args()
    data_path = args.data_path
    outdir    = rms.prep_outdir(args.outdir)
    test_net(data_path,outdir)


if __name__ == "__main__":
    main()