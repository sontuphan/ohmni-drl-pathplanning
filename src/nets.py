import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SHAPE = (64, 64)


class MobileNet(torch.nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # General info
        self.input_shape = IMG_SHAPE
        self.output_shape = (2,)
        # Layers
        self.model = models.mobilenet_v2(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.last_channel, 2)
        # Loss and Optimizer
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters())
        # Preprocess
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = F.relu(self.model(x))
        return x

    def normalize_data(self, imgs):
        imgs = np.array(imgs, dtype=np.float32) / 255
        imgs = torch.from_numpy(imgs)
        return self.preprocess(imgs)


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # General info
        self.input_shape = IMG_SHAPE
        self.output_shape = (2,)
        # Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # Loss and Optimizer
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters())

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(self.input_shape[0])))
        convh = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(self.input_shape[1])))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, self.output_shape[0])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def normalize_data(self, imgs):
        imgs = np.array(imgs)
        (batch_size, _, _, channel_size) = imgs.shape
        imgs = np.resize(imgs, (batch_size,) + IMG_SHAPE+(channel_size,))
        imgs = imgs/255
        imgs = imgs - 1
        imgs = imgs.transpose((0, 3, 1, 2))
        return torch.from_numpy(imgs).float()
