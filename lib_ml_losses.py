import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable   ### [PT] this doesn't seem to be used?


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

