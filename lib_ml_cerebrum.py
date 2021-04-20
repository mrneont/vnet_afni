import torch
import torch.nn as nn
import numpy as np

class conv3dblk(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(conv3dblk, self).__init__()

		self.conv3d = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
            ) 

	def forward(self, x):
		out = self.conv3d(x)
		return out 


class downconv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(downconv, self).__init__()

		self.downconv3d = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size,stride,padding),
            nn.ReLU()
            )

	def forward(self, x):

		out = self.downconv3d(x)
		return out

class upconv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):

		super(upconv, self).__init__()

		self.upconv3d = nn.Sequential(
			nn.ConvTranspose3d(in_channels, out_channels, kernel_size,stride,padding),
			nn.ReLU()
			)


	def forward(self, x):

		out = self.upconv3d(x)
		return out 


class Cerebrum(nn.Module):

	def __init__(self, in_channels, num_class,verb):

		super(Cerebrum, self).__init__()

		self.enc_lvl1_block1_op = conv3dblk(1,48,3,1,1)
		self.down_conv_1to2_op  = downconv(48,96,3,2,1)



		# -----------------------------------
		#         ENCODER - LEVEL 2
		# -----------------------------------
		self.enc_lvl2_block1_op =  conv3dblk(96,96,3,1,1)
		self.enc_lvl2_block2_op =  conv3dblk(96,96,3,1,1)
		self.down_conv_2to3_op  = downconv(96,192,3,2,1)

		# -----------------------------------
		#         BOTTLENECK LAYER
		# -----------------------------------

		self.bottleneck_block1_op =  conv3dblk(192,192,3,1,1)
		self.bottleneck_block2_op =  conv3dblk(192,192,3,1,1)
		self.bottleneck_block3_op =  conv3dblk(192,192,3,1,1)


		# -----------------------------------
		#         DECODER - LEVEL 2
		# -----------------------------------

		self.up_conv_3to2_op = upconv(192,96,4,2,1)   
		self.dec_lvl2_block1_op = conv3dblk(96,96,3,1,1)
		self.dec_lvl2_block2_op = conv3dblk(96,96,3,1,1)

		# -----------------------------------
		#         DECODER - LEVEL 1
		# -----------------------------------

		self.up_conv_2to1_op = upconv(96,48,4,2,1)
		self.dec_lvl1_block1_op = conv3dblk(48,48,3,1,1)
		self.output_layer_op = nn.Sequential(nn.Conv3d(48, 1, kernel_size=1),
            nn.Softmax(dim=3)
            )
#nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')

		for m in self.modules():
			if isinstance(m, nn.Conv3d):

				nn.init.xavier_uniform(m.weight.data)

			elif isinstance(m, nn.ConvTranspose3d):

				nn.init.xavier_normal_(m.weight.data)


	def forward(self, x,verb):

		# -----------------------------------
		#         ENCODER - LEVEL 1
		# -----------------------------------

		enc_lvl1_block1 = self.enc_lvl1_block1_op(x)+ torch.cat(48*[x], dim=1)


		down_conv_1to2 = self.down_conv_1to2_op(enc_lvl1_block1)

		# -----------------------------------
		#         ENCODER - LEVEL 2
		# -----------------------------------

		enc_lvl2_block1 = self.enc_lvl2_block1_op(down_conv_1to2)


		enc_lvl2_block2 = self.enc_lvl2_block2_op(enc_lvl2_block1)


		down_conv_2to3 = self.down_conv_2to3_op(enc_lvl2_block2)

		# -----------------------------------
		#         BOTTLENECK LAYER
		# -----------------------------------


		bottleneck_block1 = self.bottleneck_block1_op(down_conv_2to3)


		bottleneck_block2 = self.bottleneck_block2_op(bottleneck_block1)


		bottleneck_block3 = self.bottleneck_block3_op(bottleneck_block2)


		# -----------------------------------
		#         DECODER - LEVEL 2
		# -----------------------------------

		up_conv_3to2 = self.up_conv_3to2_op(bottleneck_block3)


		skip_conn_lvl2 = enc_lvl2_block1+up_conv_3to2


		dec_lvl2_block1 = self.dec_lvl2_block1_op(skip_conn_lvl2)

		dec_lvl2_block2 = self.dec_lvl2_block2_op(dec_lvl2_block1)

		# -----------------------------------
		#         DECODER - LEVEL 1
		# -----------------------------------

		up_conv_2to1 = self.up_conv_2to1_op(dec_lvl2_block2)


		skip_conn_lvl1 = up_conv_2to1+enc_lvl1_block1


		dec_lvl1_block1 = self.dec_lvl1_block1_op(skip_conn_lvl1)


		output_layer = self.output_layer_op(dec_lvl1_block1)


		return output_layer