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

	in_channels = None #output channels of conv layer

	attn_type = 'cross_attn' #choose from: 'cross_attn', 'self_attn', 'mamba', None

	# option 1: 'learnedPE' 
	# option 2: 'NoPE' - no positional encoding
	# option 3: 'RoPE' - rotary positional encoding, sota of nlp tasks (e.g. chatgpt)
	# only matters to self and cross attn
	pe_type = 'learnedPE'

	def __str__(self):
		attributes = [f"{key} = {getattr(attn_config, key)}" for key in dir(attn_config) 
						if not key.startswith("__") and not callable(getattr(attn_config, key))]
		return "\n".join(attributes)

def normal_init(m, mean, std):
	"""
	Helper function. Initialize model parameter with given mean and std.
	"""
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		# delete start
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()
		# delete end


class ConvolutionalBlock(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		is_downsampling: bool = True,
		add_activation: bool = True,
		**kwargs
	):
		super().__init__()
		if is_downsampling:
			self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
				nn.InstanceNorm2d(out_channels),
				nn.ReLU(inplace=True) if add_activation else nn.Identity(),
			)
		else:
			self.conv = nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
				nn.InstanceNorm2d(out_channels),
				nn.ReLU(inplace=True) if add_activation else nn.Identity(),
			)

	def forward(self, x):
		return self.conv(x)

class identity_attn(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, orig_img): 
		return x

class ResidualAttentionBlock(nn.Module):
	def __init__(self, channels: int, spatial_dim):
		super().__init__()
		self.block = nn.Sequential(
			ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
			ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
		)
		self.attention = attention_block.BaseNet(attn_config.embed_dim, spatial_dim ** 2, attn_config.attn_type, 
												 channels, attn_config.pe_type) if attn_config.attn_type is not None else identity_attn()
	def forward(self, x, orig_img):
		x = F.relu(x + self.block(x), True)
		x = self.attention(x, orig_img)
		return x




class gen_with_attn(nn.Module):
	def __init__(
		self, img_channels: int =3, num_features: int = 64, n_blocks: int = 6, n_downsamples: int = 2
	):
		"""
		Generator consists of 2 layers of downsampling/encoding layer, 
		followed by 6 residual blocks for 128 × 128 training images 
		and then 3 upsampling/decoding layer. 
		
		The network with 6 residual blocks can be written as: 
		c7s1–64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, and c7s1–3.

		"""
		super().__init__()
		self.initial_layer = nn.Sequential(
			nn.Conv2d(
				img_channels,
				num_features,
				kernel_size=7,
				stride=1,
				padding=3,
				padding_mode="reflect",
			),
			nn.InstanceNorm2d(num_features),
			nn.ReLU(inplace=True),
		)

		self.downsampling_layers = nn.ModuleList([
			ConvolutionalBlock(
				num_features * 2**i, 
				num_features * 2**(i+1),
				is_downsampling=True, 
				kernel_size=3, 
				stride=2, 
				padding=1,
			) for i in range(n_downsamples)])

		self.residual_attention = nn.ModuleList(
			ResidualAttentionBlock(num_features * 4, 256 / (2 ** n_downsamples)) for _ in range(n_blocks))

		self.upsampling_layers = nn.ModuleList([
			ConvolutionalBlock(
				num_features * 2**(i+1), 
				num_features * 2**i,
				is_downsampling=False, 
				kernel_size=3, 
				stride=2, 
				padding=1,
				output_padding=1,
			)
			for i in reversed(range(n_downsamples))])

		self.last_layer = nn.Conv2d(
			num_features * 1,
			img_channels,
			kernel_size=7,
			stride=1,
			padding=3,
			padding_mode="reflect",
		)
		self.apply(lambda m: normal_init(m, 0.0, 0.02))
	
	def forward(self, x):
		orig_img = x.clone()
		x = self.initial_layer(x)
		for layer in self.downsampling_layers:
			x = layer(x)
		for layer in self.residual_attention:
			x = layer(x, orig_img)
		for layer in self.upsampling_layers:
			x = layer(x)
		return torch.tanh(self.last_layer(x))

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