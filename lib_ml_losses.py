import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import lib_EDT
import numpy as np


# List of all lost function suffixes, so we can check if user has
# input a valid one.  Every time a new loss function is added, its
# name should be entered here.  The [0]th one in the list is the
# default.
list_CalcLoss = [ "SoftDice_00",
                  "WtSoftDice_01",
                  ]

# =========================================================================

def make_one_hot(labels, classes):
    '''
    + This function puts together a representation of categorical 
       variables as binary vectors
    + Here the categories are different classes/channels

    + Input labels is the ground truth data itself

    + Return a torch tensor 'target' having the number of channels
      equal to the  number of classes

    '''
    shape = list(labels.size())
    shape[1] = classes
    
    
    one_hot =torch.zeros(shape,device=labels.device)
    target = one_hot.scatter_(1, labels.data.long(), 1)
    
    return target





class CalcLoss_SoftDice_00(nn.Module):

    def __init__(self,):
        super(CalcLoss_SoftDice_00, self).__init__()

    def forward(self, pred, gt):
        '''Input two dsets, and calculate the loss function between them.
    This is for the 2-channel case:  brain, and nonbrain.

    For this loss: the soft-Dice is calculated for each channel; the
    results are averaged; and the output is one minus that value.

    Each dset has 5 dimensions, with each index telling:
        (batch_size, n_channels, Depth, Height, Width)

    For example, the gt.size() is: (1, 1, 32, 32, 32), 
    and the pred.size() is:        (1, 2, 32, 32, 32).

    Args:
        pred     : 'prediction' dataset
        gt       : 'answer' dataset

    Return:
        value    : single-valued Tensor-type (so, a scalar tensor?)

        '''
        
        # separate out pred into separate dsets, one volume per
        # channel; same size as pred
        target      = make_one_hot(gt, classes=pred.size()[1])


        numerator   = 2.0 * (pred * target).sum(dim=(2, 3, 4))
        denominator = 1.0 + pred.pow(2).sum(dim=(2, 3, 4)) + \
            target.sum(dim=(2, 3, 4))

        
        dice = numerator/denominator
       
        dice_mean = dice.mean()
       
       

        return 1 - dice_mean




class CalcLoss_WtSoftDice_01(nn.Module):

    def __init__(self,):
        super(CalcLoss_WtSoftDice_01, self).__init__()

    def forward(self, pred, gt):
        '''Input two dsets, and calculate the loss function between them.
    This is for the 2-channel case:  brain, and nonbrain.
    
    Each dset has 5 dimensions, with each index telling:
        (batch_size, n_channels, Depth, Height, Width)

    For example, the gt.size() is: (1, 1, 32, 32, 32), 
    and the pred.size() is:        (1, 2, 32, 32, 32).

    Args:
        pred     : 'prediction' dataset
        gt       : 'answer' dataset

    Return:
        value    : single-valued Tensor-type (so, a scalar tensor?)

    + WtSoftDice_01 function has an extended feature to SoftDice_00. 
    + The predicted mask is mupltiplied by the depth_info as weights. 
    + The depth_info emphasises the voxels at the edge 
    + The depth_info is calculated using the function "calc_EDT_3D"
      imported from lib_EDT

        '''
        
        # separate out pred into separate dsets, one volume per
        # channel; same size as pred
        target      = make_one_hot(gt, classes=pred.size()[1])

      
        gt = gt.int()
        np_gt = gt.numpy()
        
        
        target_EDT  = lib_EDT.calc_EDT_3D( np_gt[0][0], do_sqrt = True, 
                               bounds_are_zero=True,
                               edims=(8, 8, 8))

       
        wts  = target_EDT/target_EDT.max()
        wtsexp = np.exp(-wts)
        
        
        # Please note : The neural network goes into saturation when  
        # exponential(depth_info) is normalized.
        # the best way to go about is to normalize the depth_info
        # and then calculate the exp(norm_depth_info)
        
        wts = torch.from_numpy(wtsexp)
        

 
        # channel corresponding to the non brainm region 
       
        num1   = 2.0 * ( pred[0][0] * target[0][0]).sum()
        denom1 = 1.0 + pred[0][0].pow(2).sum() + \
            target[0][0].sum()

        # channel corresponding to the brain region 
        # the wts are multiplied only to the channel corresponding to the brain region 
        num2  = 2.0 * (wts* pred[0][1] * target[0][1]).sum()
        denom2 = 1.0 +  (wts*pred[0][1]).pow(2).sum() + \
            target[0][1].sum()


        dice1 = num1 / denom1

        dice2 = num2 / denom2

        
        dice = [dice1, dice2]

        wtdicemean  = (dice1+dice2)/2
     

        return 1 - wtdicemean



class Calc_Sorensen_Dice_02(nn.Module):

    '''
    +  Sorensen–Dice index = 2|X∩Y|/ |X|+|Y|
    +  Ref :https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    '''

    def __init__(self,):
        super(Calc_Sorensen_Dice_02, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, gt):

        target       = make_one_hot(gt, classes=pred.size()[1])

        
        
        numerator    = 2.0 *  torch.sum(pred * target, dim=(2, 3, 4))
        denominator  = torch.sum(pred + target, dim=(2, 3, 4))

        
        dice = numerator/(denominator)
       
        return 1 - torch.mean(dice)
       





class Calc_WtSorensen_Dice_03(nn.Module):

    '''
    +  Sorensen–Dice index = 2|X∩Y|/ |X|+|Y|
    +  Ref :https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    '''

    def __init__(self,):
        super(Calc_WtSorensen_Dice_03, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, gt):

        target       = make_one_hot(gt, classes=pred.size()[1])

        gt = gt.int()
        np_gt = gt.numpy()
        
        
        target_EDT  = lib_EDT.calc_EDT_3D( np_gt[0][0], do_sqrt = True, 
                               bounds_are_zero=True,
                               edims=(8, 8, 8))

       
        wts  = target_EDT/target_EDT.max()
        wtsexp = np.exp(-wts)
        
        
        # Please note : The neural network goes into saturation when  
        # exponential(depth_info) is normalized.
        # the best way to go about is to normalize the depth_info
        # and then calculate the exp(norm_depth_info)
        
        wts = torch.from_numpy(wtsexp)
        

        
        
        numerator    = 2.0 *  torch.sum(wts*pred * target, dim=(2, 3, 4))
        denominator  = torch.sum(wts *pred + wts*target, dim=(2, 3, 4))

        
        dice = numerator/(denominator)
       
        return 1 - torch.mean(dice)











# -------------------------------------------------------------------------
# The boneyard of older cost functions.  For inspiration.

'''
class dice_thrsh(nn.Module):  
    def __init__(self):
        super(dice_thrsh, self).__init__()

    def forward(self, pred, gt):

        
        pred_max = torch.max(pred)
        cutoff   = 0.25*pred_max

        # Binarize, with thresholding to preserve the gradient attribute. 
        C      = (pred - cutoff) 
        C[C<0] = 0
        C[C>0] = 1

        denom = torch.sum(C) + torch.sum(gt)

        if denom :
            dice = 2.0 * torch.sum(C*gt) / denom
        else:
            dice = 0

        loss_value = (1-dice) 
        
        return loss_value




class get_dice(nn.Module):
    def __init__(self):
        super(get_dice, self).__init__()
    
    def forward(self,pred, gt):

        #print('pred MAX', pred.max())
        #print('pred MIN', pred.min())
        
        dice = 2.0*torch.sum(pred*gt)/(1.0+torch.sum(pred**2)+torch.sum(gt**2))
      
        return 1-dice 



def dice(inputs, targets, smooth=1.0):

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    

    intersection = (inputs * targets).sum()   
    dice = 2.*intersection + smooth
    dice/= inputs.sum() + targets.sum() + smooth
    #print(dice)

    return 1-dice

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent
        # activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = 2.*intersection + smooth
        dice/= inputs.sum() + targets.sum() + smooth
        
        return 1-dice
'''
