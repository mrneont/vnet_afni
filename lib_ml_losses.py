import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import lib_EDT
import numpy as np
import lib_nibabel_utils    as lnu


# List of all lost function suffixes, so we can check if user has
# input a valid one.  Every time a new loss function is added, its
# name should be entered here.  The [0]th one in the list is the
# default.
list_CalcLoss = [ "Sorensen_Dice_mean",
                  "Sorensen_Dice_single_channel",
                  "WtSorensen_Dice"]

# =========================================================================



def make_one_hot_stack(gt, num_classes):
    '''
    + The basic idea of one-hot encoding is to create new variables that 
      take on values 0 and 1 to represent the original categorical values
      (labels).
    + This function creates a binary mask for each of the class 

    Parameters
    ----------
    data        : is the data for which the masks need to be created. 
    num_classes : is the number of  classes in the data.
                  It is also equal to the number of output channels in the 
                  predicted data from the neural network. 
    
    A label is a category into which a record falls, usually in the
    context of predictive modeling.

    #num_classes = (data.max()+1).int() # (highest cardinal number of the label+1)
    #num_classes = number of o/p channels in the predicated data

    Returns
    -------
    masks       : returns binary masks for each class/label of datatype 
                  torch.FloatTensor

    '''
    
    # the number of masks created is equal to the number of classes, 
    # the size of each mask is equal to the (D, H, W) of ground truth
    values = [gt == i for i in range(num_classes)]
    
    masks = torch.stack(values, dim=2).float()
    masks_squeezed = torch.squeeze(masks ,dim=0)
    #print('masks size \n',masks_squeeze.size())
    #print('gt size \n'   ,gt.size())
    #print(' mask type \n' , masks_squeezed.type())

    '''
    #in order to check the masks, write the masks as a NIFTI dataset 
   
    #lnu.write_tensor_to_disk_nifti(masks_squeezed[0][0], 'mask_0_0')
    #lnu.write_tensor_to_disk_nifti(masks_squeezed[0][1], 'mask_0_1')

    '''

    return masks_squeezed


def make_one_hot_scatter(gt, num_classes):
    '''
    + The basic idea of one-hot encoding is to create new variables that 
      take on values 0 and 1 to represent the original categorical values
      (labels).
    + This function creates a binary mask for each of the class 

    Parameters
    ----------
    gt          : is the data for which the masks need to be created. 
    num_classes : is the number of  classes in the data.
                  It is also equal to the number of output channels in the 
                  predicted data from the neural network. 
    
    A label is a category into which a record falls, usually in the context of 
    predictive modeling.

    #num_classes = (data.max()+1).int() # (highest cardinal number of the label+1)
    #num_classes = number of o/p channels in the predicated data

    Returns
    -------
    masks       : returns binary masks for each class/label of datatype 
                  torch.FloatTensor
    
    '''
    
    
    # gt_size = (batch_sz,ch,D, H, W)
    gt_size   = gt.size()
    batch_sz  = gt_size[0]
    num_ch    = num_classes
    depth     = gt_size[2]
    height    = gt_size[3]
    width     = gt_size[4]

    # define the masks tensor
    # the number of masks created is equal to the number of classes, 
    # the size of each mask is equal to the (D, H, W) of ground truth
    masks       = torch.zeros([batch_sz, num_ch, depth, height, width]) 
    #print('masks size \n',masks.size())
    #print('gt size \n'   ,gt.size())

    #dimension here is (batch_sz, channels, D, H, W)
    #tensor.scatter_() in this case operates along dim = 1
    masks.scatter_(1, gt.data.long(), 1)  ##(dim, index, src) 
    #print(' mask type \n' , masks.type())

    '''
    #in order to check the masks, write the masks as a NIFTI dataset 

    lnu.write_tensor_to_disk_nifti(masks[0][0], 'mask_0_0')
    lnu.write_tensor_to_disk_nifti(masks[0][1], 'mask_0_1')

    '''
    return masks    



    


class CalcLoss_Sorensen_Dice_mean(nn.Module):

    '''
    +  Sorensen–Dice index = 2*|X intersection Y| / |X| + |Y|
    +  Ref:
       https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    '''

    def __init__(self,):
        super(CalcLoss_Sorensen_Dice_mean, self).__init__()
        

    def forward(self, pred, gt , depth_map):

        # predicted_mask dim is in the format of (batch_sz,num_out_ch,
        # D, H, W)
        # num_out_ch is the number of output channels and is equal to
        # the number of classes

        num_out_ch   = pred.size()[1]
        #target       = make_one_hot_scatter(gt, num_classes = num_out_ch)
        target       = make_one_hot_stack(gt, num_classes = num_out_ch)

        numerator    = 2.0 * torch.sum(pred * target, dim=(2, 3, 4))
        denominator  = torch.sum(pred + target, dim=(2, 3, 4))
        
        dice         = numerator/denominator
       
          
        return 1 - torch.mean(dice)


class CalcLoss_Sorensen_Dice_single_channel(nn.Module):
    

    '''
    +  Sorensen–Dice index = 2*|X intersection Y| / |X| + |Y|
    +  Ref:
       https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
       gt = dims(1, 1, 32, 32, 32).
       pred, target = dims(1, 2, 32, 32, 32).

    '''

    def __init__(self,):
        super(CalcLoss_Sorensen_Dice_single_channel, self).__init__()
        

    def forward(self, pred, gt, depth_map):

        num_out_ch   = pred.size()[1]
        target       = make_one_hot_scatter(gt, num_classes = num_out_ch)
        
        numerator    = 2.0 * torch.sum(pred * target, dim=(2, 3, 4))
        denominator  = torch.sum(pred + target, dim=(2, 3, 4))
        
        dice         = (numerator)/denominator

        ###print("dice = ",dice)
        # this part of the code logic requires re-visit when handling
        # multi-class data if the denominator of the predicted brain
        # mask is zero them return loss as 1 (high)
        if denominator[0][1] == 0: #channel containing predicted brain mask.
            return 1  # check the return type 

        # returning the dice loss pertaining to the channel containing
        # predicted brain mask.
        return 1 - dice[0][1]
       


class CalcLoss_WtSorensen_Dice(nn.Module):

    '''
    +  Sorensen–Dice index = 2*|X intersection Y| / |X| + |Y|
    +  Ref:
       https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    '''

    def __init__(self,):
        super(CalcLoss_WtSorensen_Dice, self).__init__()
        

    def forward(self, pred, gt , depth_map):

        # predicted_mask dim is in the format of (batch_sz,num_out_ch,
        # D, H, W)
        # num_out_ch is the number of output channels and is equal to
        # the number of classes

        num_out_ch   = pred.size()[1]
        #target       = make_one_hot_scatter(gt, num_classes = num_out_ch)
        target       = make_one_hot_stack(gt, num_classes = num_out_ch)

        depth_map = depth_map.unsqueeze(0) 
        #print('depth_map size= ',depth_map.size())

        depth_map_abs =  torch.abs(depth_map)

        #lnu.write_tensor_to_disk_nifti(depth_map[0][0], 'depth_map')
        #print('depth_map_abs size= ',depth_map_abs.size())
        wts           = depth_map_abs - depth_map_abs.min()
        #print('wts size= ',wts.size())

        flr = 0.2
        dist_scale = 50


        wts_exp = (1-flr)*torch.exp(-0.693*wts/dist_scale)+ flr
        #print('wts_exp size= ',wts_exp.size())

        # in order to check the wts, write the wts_exp as a NIFTI dataset 

        #print('wts_exp[0][0] size= ',wts_exp[0][0].size())
        #lnu.write_tensor_to_disk_nifti(wts_exp[0][0], 'wts_exp')
        
        wtPT = wts_exp * pred * target
        PT   = pred * target
        #print('size of wts_exp = ',wts_exp.size())
        #print('size of wtPT = ',wtPT.size())
        #print('size of PT = ',PT.size())

        numerator    = 2.0 * torch.sum(wts_exp * pred * target, dim=(2, 3, 4))
        denominator  = torch.sum(wts_exp * (pred+target), dim=(2, 3, 4))
        
        dice         = numerator/denominator
       
          
        return 1 - torch.mean(dice)
