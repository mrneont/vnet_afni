import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
                         
# ==================================================================
# pieces to be used within the model

class RepeatConv(nn.Module): 
    """
    Repeat 'Conv + ReLU' n_conv times.

    NB:  out_channels = in_channels.
    """

    def __init__(self, n_channels, n_conv, wt_norm):
        super(RepeatConv, self).__init__()
        
        conv_list = []
        for i in range(n_conv):
            if wt_norm ==1:
                conv_list.append(weight_norm(nn.Conv3d(n_channels, n_channels,   
                                        kernel_size=5, padding=2)))
            else:
                conv_list.append(nn.Conv3d(n_channels, n_channels,   
                                        kernel_size=5, padding=2))

            conv_list.append(nn.ReLU())
        
        self.conv = nn.Sequential(*conv_list)
        
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """
    NB:  out_channels = 2 * in_channels.
    """

    def __init__(self, in_channels, out_channels, n_conv, wt_norm): 
        super(Down, self).__init__()
        
        if wt_norm ==1:

            self.downconv = nn.Sequential(
                weight_norm(nn.Conv3d( in_channels, out_channels, 
                       kernel_size=2, stride=2)),
                nn.ReLU())

        else:
            self.downconv = nn.Sequential(
                (nn.Conv3d( in_channels, out_channels, 
                       kernel_size=2, stride=2)),
                nn.ReLU())
        

        # repeat the 'n_conv' convolution layers 
        self.conv = RepeatConv(out_channels, n_conv, wt_norm) 
        
    def forward(self, x):
        out = self.downconv(x)
        return out + self.conv(out)

class Up(nn.Module):
    """
    NB:  out_channels = in_channels / 2.
    """
    def __init__(self, in_channels, out_channels, n_conv, wt_norm): 
        super(Up, self).__init__()
        
        if wt_norm ==1:
            self.upconv = nn.Sequential(
                weight_norm(nn.ConvTranspose3d( in_channels, int(out_channels/2), 
                                kernel_size=2, stride=2)),
                nn.ReLU())

        else:
            self.upconv = nn.Sequential(
                nn.ConvTranspose3d( in_channels, int(out_channels/2), 
                                kernel_size=2, stride=2),
                nn.ReLU())
            
        

        # repeat the 'n_conv' convolution layers 
        self.conv = RepeatConv(out_channels, n_conv, wt_norm) 
        
    def forward(self, x, down):
        x   = self.upconv(x)
        # residual connection only in the decoder part 
        cat = torch.cat((x, down), dim=1) 
        return cat + self.conv(cat)

# ==================================================================
# the model

class VNet_orig(nn.Module):
    """Main model: VNet_orig is implementation of Fig 2 from the paper:
       https://arxiv.org/pdf/1606.04797.pdf

    Here we set up the main model.

    Parameters (init)
    =================

    in_channels    : number of input channels
    num_class      : number of classes in the dataset

    Objects (init)
    =================

    down*          : The enoder has the down-convolutional layers
                     each followed by a ReLU activation function
                        

    up*            : The decoder has the up-convolutional layers
                     each followed by a ReLU activation function 
                    

    Methods
    =======

    forward        : Forward() method is the actual network
                     transformation.  The forward pass of the network
                     defines the computational graph of the model.
                     The forward method is the mapping that maps an
                     input tensor to a prediction output tensor.
                     Edges will be activation functions that produce
                     output Tensors from input Tensors.
                     Backpropagating through this graph then allows
                     you to easily compute gradients

    """

    def __init__(self, in_channels, num_class, wt_norm, verb):
        super(VNet_orig, self).__init__()

        #                -- ENCODER --
        #  input layer : down1 = (Conv3d, ReLu)
        # hidden layer : down2 = downconv(Conv3d,Relu) => 2x(Conv3d,Relu)
        # hidden layer : down3 = downconv(Conv3d,Relu) => 3x(Conv3d,Relu)
        # hidden layer : down4 = downconv(Conv3d,Relu) => 3x(Conv3d,Relu)
        # hidden layer : down5 = downconv(Conv3d,Relu) => 3x(Conv3d,Relu)
        if (wt_norm ==1):
            self.down1 = nn.Sequential(
                weight_norm(nn.Conv3d(1, 16, kernel_size=5, padding=2)), 
                nn.ReLU())
        else :
            self.down1 = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=5, padding=2),
                nn.ReLU())

        self.down2 = Down(16,  32,  2, wt_norm)    
        self.down3 = Down(32,  64,  3, wt_norm)  
        self.down4 = Down(64,  128, 3, wt_norm)
        self.down5 = Down(128, 256, 3, wt_norm) 
        
        #                -- DECODER --
        # hidden layer : up1 = upconv(ConvTr3d,Relu) => 3x(Conv3d,Relu)
        # hidden layer : up2 = upconv(ConvTr3d,Relu) => 3x(Conv3d,Relu)
        # hidden layer : up3 = upconv(ConvTr3d,Relu) => 2x(Conv3d,Relu)
        # hidden layer : up4 = upconv(ConvTr3d,Relu) => 1x(Conv3d,Relu)
        #    out layer : up5 = (Conv3d,Relu)
        self.up1 = Up(256, 256, 3, wt_norm)
        self.up2 = Up(256, 128, 3, wt_norm)
        self.up3 = Up(128,  64, 2, wt_norm)
        self.up4 = Up(64,   32, 1, wt_norm)

        if (wt_norm ==1):
            self.up5 = nn.Sequential(
                weight_norm(nn.Conv3d(32, num_class, kernel_size=1)),
                nn.Softmax(dim=1))
        else:
            self.up5 = nn.Sequential(
                nn.Conv3d(32, num_class, kernel_size=1),
                nn.Softmax(dim=1))
            
        
        
        #count = 1
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
                #print('weight init {:d}'.format(count))
                #count= count+1
        '''
         
    def forward(self, x, verb):
        down1 = self.down1(x) + torch.cat(16*[x], dim=1)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4) 
       
        up1 = self.up1(down5, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)
        up5 = self.up5(up4)

        if verb >=2 :
            print('\n')
            print('*****************  NETWORK LAYERS *************************')
            print('ENCODER PART')
            print('input layer  : down1 shape', down1.shape)
            print('hidden layer : down2 shape', down2.shape)
            print('hidden layer : down3 shape', down3.shape)
            print('hidden layer : down4 shape', down4.shape)
            print('hidden layer : down5 shape', down5.shape)
            print('DECODER PART')
            print('hidden layer : up1 shape', up1.shape)
            print('hidden layer : up2 shape', up2.shape)
            print('hidden layer : up3 shape', up3.shape)
            print('hidden layer : up4 shape', up4.shape)
            print('output layer : up5 shape', up5.shape)
            print('*************************************************************')

        return up5



