import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels,
				4,
				stride,
				1,
				bias=True,
				padding_mode="reflect",
			),
			nn.InstanceNorm2d(out_channels),
			nn.LeakyReLU(0.2, inplace=True),
		)

	def forward(self, x):
		return self.conv(x)

class discriminator(nn.Module):
	def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
		super().__init__()
		self.initial = nn.Sequential(
			nn.Conv2d(
				in_channels,
				features[0],
				kernel_size=4,
				stride=2,
				padding=1,
				padding_mode="reflect",
			),
			nn.LeakyReLU(0.2, inplace=True),
		)

		layers = []
		in_channels = features[0]
		for feature in features[1:]:
			layers.append(
				Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
			)
			in_channels = feature
		layers.append(
			nn.Conv2d(
				in_channels,
				1,
				kernel_size=4,
				stride=1,
				padding=1,
				padding_mode="reflect",
			)
		)
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		x = self.initial(x)
		return torch.sigmoid(self.model(x))
	"""
	# initializers
	def getDim(self, input, layer):
		return input*(2**(layer-1))

	def __init__(self, scale=1):
	super(discriminator, self).__init__()
	out_dim=64*scale
	self.conv1 = nn.Conv2d(6, out_dim, kernel_size=4, stride=2, padding=1)
	self.conv2 = nn.Conv2d(self.getDim(out_dim, 1), self.getDim(out_dim, 2), kernel_size=4, stride=2, padding=1)
	self.bn2 = nn.BatchNorm2d(self.getDim(out_dim, 2))
	self.conv3 = nn.Conv2d(self.getDim(out_dim, 2), self.getDim(out_dim, 3), kernel_size=4, stride=2, padding=1)
	self.bn3 = nn.BatchNorm2d(self.getDim(out_dim, 3))
	self.conv4 = nn.Conv2d(self.getDim(out_dim, 3), self.getDim(out_dim, 4), kernel_size=4, stride=2, padding=1)
	self.bn4 = nn.BatchNorm2d(self.getDim(out_dim, 4))
	self.conv5 = nn.Conv2d(self.getDim(out_dim, 4), 1, kernel_size=4, stride=1, padding=1)

	# weight_init
	def weight_init(self, mean, std):
	for m in self._modules:
		if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
			# delete start
			m.weight.data.normal_(mean, std)
			m.bias.data.zero_()

	# forward method
	def forward(self, input):

	x1 = F.leaky_relu(self.conv1(input))
	x2 = F.leaky_relu(self.bn2(self.conv2(x1)), negative_slope=0.2)
	x3 = F.leaky_relu(self.bn3(self.conv3(x2)), negative_slope=0.2)
	x4 = F.leaky_relu(self.bn4(self.conv4(x3)), negative_slope=0.2)
	x5 = F.sigmoid(self.conv5(x4))

	x = x5

	return x
	"""
