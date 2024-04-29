import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import attention_block

class attn_config:
	'''
	Hyperparameters to tune for attention layer
	'''
	#try 16, 32, 64. if fast enough training time & enough gpu memory, 64 is the best
	embed_dim = 64 

	max_len = None # = spatial_dimention squared

	# option 1: 'cross_attn'
	# option 2: 'self_attn'
	# option 3: 'mamba'
	layer_type = 'cross_attn'

	in_channels = None #output channels of conv layer

	# option 1: 'learnedPE' 
	# option 2: 'NoPE' - no positional encoding
	# option 3: 'RoPE' - rotary positional encoding, sota of nlp tasks (e.g. chatgpt)
	# only matters to self and cross attn
	pe_type = 'learnedPE'

def normal_init(m, mean, std):
	"""
	Helper function. Initialize model parameter with given mean and std.
	"""
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		# delete start
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()
		# delete end



class ConvDown(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super().__init__(**kwargs)
		self.layers = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(inplace=True)
		)
	def forward(self, x):
		return self.layers(x)

class ConvUp(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super().__init__(**kwargs)
		self.layers = nn.Sequential(
			nn.ConvTranspose2d(in_channels*2, out_channels, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self, x, skip):
		x = torch.cat([x, skip], dim=1)
		return self.layers(x)

class residualBlock(nn.Module):
	def __init__(self, n_channels):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(n_channels),
			nn.ReLU(True),
			nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(n_channels)
		)
		self.final = nn.ReLU(True)
	
	def forward(self, x):
		return self.final(x + self.layers(x))

class residualAttn(nn.Module):
	def __init__(self, n_channels, spatial_dim):
		super().__init__()
		self.residual = residualBlock(n_channels)
		# self.attention = attention_block.BaseNet(attn_config.embed_dim, spatial_dim ** 2, attn_config.layer_type, 
										        #  n_channels, attn_config.pe_type)

	def forward(self, x, orig_img):
		x = self.residual(x)
		# x = self.attention(x, orig_img)
		return x

class gen_with_attn(nn.Module):
	def __init__(self, scale : int=1, img_dim : int=256, down_sample=2, n_blocks = 6, *args, **kwargs):
		super().__init__(*args, **kwargs)
		scale = 2 ** (5 + scale)
		initial = nn.Sequential(
				 nn.Conv2d(3, scale, kernel_size=4, stride=2, padding=1),
				 nn.BatchNorm2d(scale),
				 nn.LeakyReLU(True)
		)
		self.downs = nn.ModuleList([initial])
		down_sample -= 1
		for i in range(down_sample):  # add downsampling layers
			mult = 2 ** i
			self.downs.append(ConvDown(scale * mult, scale * mult * 2))

		self.resAttns = nn.ModuleList(residualAttn(scale * mult * 2, img_dim / (2**down_sample)) for _ in range(n_blocks))
		

		self.ups = nn.ModuleList()
		for i in range(down_sample):  # add upsampling layers
			mult = 2 ** (down_sample - i)
			self.ups.append(ConvUp(scale * mult, int(scale * mult / 2)))
		self.final = nn.ConvTranspose2d(scale * 2, 3, kernel_size=4, stride=2, padding=1)

	def forward(self, x):
		orig_img = x.clone()
		skips = []

		for down in self.downs:
			x = down(x)
			skips.append(x.clone())
		
		for resAttn in self.resAttns:
			x = resAttn(x, orig_img)

		for up, skip in zip(self.ups, skips[::-1]):
			x = up(x, skip)
		x = torch.cat([x, skips[0]], dim=1)
		return self.final(x)
		

		

class generator(nn.Module):
  # initializers
	def __init__(self, scale : int=1):
		super().__init__()
		
		# Unet generator encoder
		self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(512)
		self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.bn5 = nn.BatchNorm2d(512)
		self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.bn6 = nn.BatchNorm2d(512)
		self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.bn7 = nn.BatchNorm2d(512)
		self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

		# Unet generator decoder

		self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.dbn1 = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
		self.dbn2 = nn.BatchNorm2d(512)
		self.deconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
		self.dbn3 = nn.BatchNorm2d(512)
		self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
		self.dbn4 = nn.BatchNorm2d(512)
		self.deconv5 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
		self.dbn5 = nn.BatchNorm2d(256)
		self.deconv6 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
		self.dbn6 = nn.BatchNorm2d(128)
		self.deconv7 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
		self.dbn7 = nn.BatchNorm2d(64)
		self.deconv8 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

  # weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, input):

		e1 = F.leaky_relu(self.conv1(input), negative_slope=0.2)
		e2 = F.leaky_relu(self.bn2(self.conv2(e1)), negative_slope=0.2)
		e3 = F.leaky_relu(self.bn3(self.conv3(e2)), negative_slope=0.2)
		e4 = F.leaky_relu(self.bn4(self.conv4(e3)), negative_slope=0.2)
		e5 = F.leaky_relu(self.bn5(self.conv5(e4)), negative_slope=0.2)
		e6 = F.leaky_relu(self.bn6(self.conv6(e5)), negative_slope=0.2)
		e7 = F.leaky_relu(self.bn7(self.conv7(e6)), negative_slope=0.2)
		e8 = F.leaky_relu(self.conv8(e7), negative_slope=0.2)

		# decoding

		d1 = F.relu(self.dbn1(self.deconv1(e8)))
		d1 = torch.cat([d1, e7], 1)
		d2 = F.relu(self.dbn2(self.deconv2(d1)))
		d2 = torch.cat([d2, e6], 1)
		d3 = F.relu(self.dbn3(self.deconv3(d2)))
		d3 = torch.cat([d3, e5], 1)
		d4 = F.relu(self.dbn4(self.deconv4(d3)))
		d4 = torch.cat([d4, e4], 1)
		d5 = F.relu(self.dbn5(self.deconv5(d4)))
		d5 = torch.cat([d5, e3], 1)
		d6 = F.relu(self.dbn6(self.deconv6(d5)))
		d6 = torch.cat([d6, e2], 1)
		d7 = F.relu(self.dbn7(self.deconv7(d6)))
		d7 = torch.cat([d7, e1], 1)
		d8 = F.tanh(self.deconv8(d7))

		output = d8

		return output