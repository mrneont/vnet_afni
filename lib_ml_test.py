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

#********************************************************************************
#* The code is to test the model weights. 
#* The model weights are stored as 'checkpoint.pt' file after the training phase
#* In this code model weights are loaded from the checkpoint.pt file using torch.load()
#* 
#*
#* Usage: python lib_ml_test.py -d data_path_testdata -o data_path_output  -m 'cpu'
#* the flag '-m' is either cuda or the cpu depending on the device chosen to test the data
#*
#* 'data_path_testdata' should contain folder 'orig' and 'mask'
#********************************************************************************

# before using the lib_ml_test.py
# Please create a folder with all the checkpoint.pt files which are required to be tested
# the checkpoint.pt files are read succesively and the test_data are evaluated for each model

# List of map_locations(devices) to choose from. 
list_map_loc = [ 'cpu','cuda']
checkpt_file_list = []
origfl_list = []



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

    parser.add_argument('-ch', "--checkpoint_path",type=dir_path)

    def_map_loc = list_map_loc[0]
    parser.add_argument("-m", "--map_loc", 
                        dest="map_loc", 
                        type=str, default=def_map_loc,
                        help="map_location type; valid arguments\n" +
                        "include:" + '\n  ' +
                        "{}".format('\n  '.join(list_map_loc)) + '\n' +
                        '(def: {})'.format(str(def_map_loc)))

    return parser.parse_args()

def test_net(data_path, outdir, checkpoint_path, map_loc):

    print("++ Device on which the model weights are mapped =", map_loc)
    # Set up network 
    model = lmm.VNet_orig(in_channels=1, num_class=2, wt_norm = 0, 
                         verb=0)

    
    for name in os.listdir(checkpoint_path):
        #print('checkpoint flname = ',name)
        checkpt_file_list.append(name)
    print(checkpt_file_list)

    origfl_path =  os.path.join(data_path,'orig')
    for flname in os.listdir(origfl_path):
        #print('origfl flname = ',flname)
        origfl_list.append(flname)

    

    #perf_file     = '/'.join([outdir, 'log_performance.txt'])

    # datapath
    # DataLoader setup
    test_datapath = os.path.join(data_path)
    test_set      = lmd.mridataset(test_datapath, 
                                    use_dpth_wts= 0, 
                                    verb=0)
    
    Ntest         = len(test_set)

    test_dataloader = DataLoader(test_set,shuffle=False,batch_size=1)
    
    # creating an instance of loss function
    loss = lml.CalcLoss_Sorensen_Dice_mean()

    
    dash =  '-' * 60
    print(dash)
    strepoch = 1
    
    loss_file_pre = '/'.join([outdir, 'log_loss'])

    for name in os.listdir(checkpoint_path):

        perf_file = loss_file_pre + '_' + name +'.txt'
        with io.open(perf_file, 'a') as perf_log:
                perf_log.write("# {:>20s}  {:>10s}  \n"
                               "".format('dset','loss'))
        #dicescore = []
        chi = 2
        print('checkpoint flname:', name)
        chfl_name = os.path.join(checkpoint_path, name)
        print('checkpoint flname path:', chfl_name)
        model.load_state_dict(torch.load(chfl_name,
                                map_location=torch.device(map_loc)),
                                strict=False)

        model.eval()
        with torch.no_grad():
        
            count = 1
        
            
            for (orig_data, mask_data, dpth_data, orig_fname,index) in test_dataloader:
            
                # dummy variables in testing
                phase = 'test'
                print('strepoch =',strepoch)
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
                        lnu.make_names_of_dsets(outdir, orig_fname[0], strepoch, phase)
                
                
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
                print('LOSS.item()= ',LOSS.item())
                score = 1-LOSS.item()
                #dicescore.append(1-LOSS.item())
                #print("{}".format(orig_fname))
                #print(perf_file)
                #print(type(score))
                with io.open(perf_file, 'a') as perf_log:
                    perf_log.write("  {:12s} {:12.4f} \n".format(orig_fname[0],float(score)))

                count += 1
                print('count =',count)
            
            
            #print(df)
            chi += 1
            strepoch += 1
            #print(dash)  
            #print('DICE SCORE for validation data = ',dicescore)
            #print(dash)

    #df.to_csv('test_dice'+ '.csv') 
    perf_log.close()
        # pending[YNS] : write the dice scores into a log file in output directory   


def main():
    args      = get_test_args()
    data_path = args.data_path
    outdir    = rms.prep_outdir(args.outdir)
    checkpoint_path = args.checkpoint_path
    # map_loc : - Device on which the model weights are mapped
    map_loc   = args.map_loc
    test_net(data_path,outdir, checkpoint_path, map_loc)


if __name__ == "__main__":
    main()