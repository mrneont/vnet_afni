import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm


# ==========================================================================
'''
+   This code is the pytorch implementation of the Vnet neural network architecture proposed in the 
    paper https://arxiv.org/pdf/1606.04797.pdf
 
+   The Vnet(volumetric neural net) processes 3D data by performing volumetric convolutions.

+   VNet_orig() function is implementation of Fig 2 from the above mentioned paper. The function VNet_orig
    incorporates the various stages that operate at different resolutions both on the left side(enocder) 
    and the right side(decoder) of the network. 
 
+   Vnet Architecture explained : It has the Encoder on the left hand side(LHS) and decoder on the 
    right hand side 
+   please note that we keep increasing the channels on the encoder side and decreasing the channels on the decoder side 
 
    -- ENCODER --
        #  input layer : down1 = (Conv3d, ReLu)
        # hidden layer : down2 = downconv(Conv3d,Relu) => 2x(Conv3d,Relu) (means 'downconv' followed by RepeatConv repeated twice)
        # hidden layer : down3 = downconv(Conv3d,Relu) => 3x(Conv3d,Relu) (means 'downconv' followed by RepeatConv repeated thrice)
        # hidden layer : down4 = downconv(Conv3d,Relu) => 3x(Conv3d,Relu)
        # hidden layer : down5 = downconv(Conv3d,Relu) => 3x(Conv3d,Relu)
                                                          
                                                         
+   We keep increasing the channels on the encoder side at each stage from 1 to 16 to 32 to 64 to 128 to 256 
+   The first layer in the encoder is the input layer(denoted by the variable down1 in this code)made from 
    volumetric kernels of size 5 x 5 x 5 voxels
+   The volumetric kernels are implemented using the  Conv3d function from Pytorch
    torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride)
  
    where in_channels - Number of channels in the input image
         out_channels - Number of channels produced by the convolution
         kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution

    The input data is in the format: (1 X 1 X D X H X W) of datatype torch.FloatTensor
   (batchsz=1, Channels=1, Depth=256, Height=256, width=256)

+   down1 layer has conv3d and relu activation built in sequentially using nn.Sequential() function from pytorch 
+   The other 3 functions defined in this code are RepeatConv(), Down(),  Up() 
+   The function Down() incorporates the down convolution denoted by variable 'downconv' and RepeatConv()
+   'downconv' is Conv3d and relu activation built in sequentially which aides in the downsampling 
+   RepeatConv() is an ensemble of  kernels(5 x 5 x 5 voxels) in tandem of 2 or 3 

+   The function Up() incorporates the up convolution denoted by variable 'upconv' and RepeatConv()
+   'upconv'is ConvTranspose3d and relu activation built in sequentially which aides in the upsampling
+   The volumetric kernels for up sampling are implemented using the  ConvTranspose3d function from Pytorch
+   torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride) 
+   We keep decreasing the channels on the decode side at each stage from 256 to 128 to 64 to 32 to 16  
                -- DECODER --
        # hidden layer : up1 = upconv(ConvTr3d,Relu) => 3x(Conv3d,Relu) (means 'upconv' followed by RepeatConv() repeated thrice)
        # hidden layer : up2 = upconv(ConvTr3d,Relu) => 3x(Conv3d,Relu)
        # hidden layer : up3 = upconv(ConvTr3d,Relu) => 2x(Conv3d,Relu)
        # hidden layer : up4 = upconv(ConvTr3d,Relu) => 1x(Conv3d,Relu)
        #    out layer : up5 = (Conv3d,Relu)

+ After the architecture of the neural network has been defined/set,  the neural network is 
   initiated/called  using the 'forward' method in the training and the validation phases
''' 
# ==================================================================


class RepeatConv(nn.Module): 
    """

    This function builds an ensemble of  kernels(5 x 5 x 5 voxels) in tandem of 2 or 3 

    Repeat 'Conv + ReLU' n_conv times.

    NB:  out_channels = in_channels.

    Params
    ------
    n_channels: is the number of channels after the downconvolution or upconvolution at each stage 
    n_conv    : is the number of tandem repeats of the volumentric kernel
    wt_norm   : is the flag to denote the normalization of weights 

    Returns
    -------
    This function returns the convolved data/ feature map of type 'torch.FloatTensor'

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
        
        #print("++ {:30s} : {}".format('RepeatConv() return data_type', self.conv(x).type()))
        return self.conv(x)

class Down(nn.Module):
    """
    This function is used to implement the encoder part of the neural network. 
    It is called whenever there is a need to increase the number of out_channels comapred to the in_channels.
    It has 2 parts 1) downsampling  2) feature extraction
    The downsampling is done by 'downconv'. 'downconv' is variable used to denotes the Conv3d and ReLU activation function 
    RepeatConv() is an ensemble of  kernels(5 x 5 x 5 voxels) in tandem of 2 or 3 
    out_channels = 2 * in_channels.
    
    Params
    ------
    in_channels   : Number of channels in the input data at each stage 
    out_channels  : Number of channels produced by the downconvolution
    n_conv        : is the number of tandem repeats of the volumentric kernel
    wt_norm       : is the flag to denote the normalization of weights 

    Returns
    -------
    This function returns the convolved data/ feature map of type 'torch.FloatTensor'

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
        
        #print("++ {:30s} : {}".format('Down() return data_type', out.type()))
       
        return out + self.conv(out)

class Up(nn.Module):
    """
    This function is used to implement the decoder part of the neural network. 
    It is called whenever there is a need to decrease the number of out_channels comapred to the in_channels.
    It has 2 parts 1) upsampling  2) feature extraction
    The upsampling is done by 'upconv'. 'upconv' is variable used to denotes the ConvTranspose3d and ReLU activation function 
    RepeatConv() is an ensemble of  kernels(5 x 5 x 5 voxels) in tandem of 2 or 3 
    
    out_channels = in_channels/2

    Params
    ------
    in_channels   : Number of channels in the input data at each stage 
    out_channels  : Number of channels produced by the upconvolution
    n_conv        : is the number of tandem repeats of the volumentric kernel
    wt_norm       : is the flag to denote the normalization of weights 

    Returns
    -------
    This function returns the convolved data/ feature map of type 'torch.FloatTensor'

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
        #print("++ {:30s} : {}".format('Up() return data_type', (cat + self.conv(cat)).type()))
        return cat + self.conv(cat)

# ==================================================================
# the model

class VNet_orig(nn.Module):
    """Main model: VNet_orig is implementation of Fig 2 from the paper:
       https://arxiv.org/pdf/1606.04797.pdf

    Here we set up the main model. 
    The VNet_orig uses the functions  Down(), Up() and RepeatConv() defined in this file 
    to put together neural network of predefined architecture  

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

        # the final layer in the neural network uses the Softmax() function
        # Reason of using softmax function[YNS-TBD] 
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
         
    def forward(self, x, verb=0):
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


'''
ENCODER PART
input layer  : down1 shape torch.Size([1, 16, 32, 32, 32])
hidden layer : down2 shape torch.Size([1, 32, 16, 16, 16])
hidden layer : down3 shape torch.Size([1, 64, 8, 8, 8])
hidden layer : down4 shape torch.Size([1, 128, 4, 4, 4])
hidden layer : down5 shape torch.Size([1, 256, 2, 2, 2])
DECODER PART
hidden layer : up1 shape torch.Size([1, 256, 4, 4, 4])
hidden layer : up2 shape torch.Size([1, 128, 8, 8, 8])
hidden layer : up3 shape torch.Size([1, 64, 16, 16, 16])
hidden layer : up4 shape torch.Size([1, 32, 32, 32, 32])
output layer : up5 shape torch.Size([1, 2, 32, 32, 32])
'''
