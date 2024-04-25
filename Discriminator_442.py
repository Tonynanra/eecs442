import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):
  # initializers
  def getDim(input, layer):
     return input*(2**(layer-1))
  
  def __init__(self, scale):
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