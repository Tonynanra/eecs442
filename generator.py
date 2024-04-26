import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
  """
  Helper function. Initialize model parameter with given mean and std.
  """
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    # delete start
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()
    # delete end
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class generator(nn.Module):
  # initializers
	def __init__(self, scale : int=1):
		super(generator, self).__init__()
		
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