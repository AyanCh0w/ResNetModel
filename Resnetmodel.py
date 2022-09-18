import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  def __init__(self, inpChannels, outChannels, stride=1, downsample=None):
    super().__init__()
    self.downsample = downsample
    self.conv1 = nn.Conv2d(inpChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.batch1 = nn.BatchNorm2d(outChannels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1, bias=False)
    self.batch2 = nn.BatchNorm2d(outChannels)
  
  def forward(self, x):
    identity = x
    output = self.conv1(x)
    output = self.batch1(output)
    output = self.relu(output)
    output = self.conv2(output)
    output = self.batch2(output)
    output = self.relu(output)
    if self.downsample != None:
      identity = self.downsample(identity)
    return output + identity


class ResNet(nn.Module):
  def __init__(self, layerSizes, numClasses):
    super().__init__()
    self.layerSizes = layerSizes
    self.numClasses = numClasses

    self.conv = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
    self.batchnorm = nn.BatchNorm2d(32)
    self.relu = nn.ReLU()

    self.layer1 = self.makeLayer(32, layerSizes[0])
    self.layer2 = self.makeLayer(64, layerSizes[1])
    self.layer3 = self.makeLayer(128, layerSizes[2])
    self.layer4 = self.makeLayer(256, layerSizes[3])
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.flatten = nn.Flatten()
    self.classify = nn.Linear(512, numClasses)

  def forward(self, x):
    output = self.conv(x)
    output = self.batchnorm(output)
    output = self.relu(output)
    output = self.layer1(output)
    output = self.layer2(output)
    output = self.layer3(output)
    output = self.layer4(output)
    output = self.avgpool(output)
    output = self.flatten(output)
    output = self.classify(output)
    return output

  def makeLayer(self, inpChannels, numBlocks):
    layers = []
    layers.append(BasicBlock(inpChannels, inpChannels*2, stride=2, downsample=nn.Sequential(nn.Conv2d(inpChannels, inpChannels*2, 1, 2, bias=False), nn.BatchNorm2d(inpChannels*2))))
    for i in range(1, numBlocks):
      layers.append(BasicBlock(inpChannels*2, inpChannels*2))
    return nn.Sequential(*layers)

layerSizes = [3, 4, 6, 3] #Same Sizes resnet uses
outputClasses = 7 #Ex 7 different emotions to predict of an image
model = ResNet(layerSizes, outputClasses)