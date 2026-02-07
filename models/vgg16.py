import torch
from torch import nn
from typing import Tuple
# from data.augmentation import DataAugmentation
from config import Config

class VGG16:
    """VGG16 model"""
    def __init__(self, config):
        self.config = config

    def build_vgg16_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """Build VGG16 model with BatchNormalization"""
        class VGGForCIFAR10(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                in_channels = input_shape[0]
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout1', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 2
                self.features.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
                self.features.add_module('bn3', nn.BatchNorm2d(128, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu3', nn.ReLU(inplace=True))
                self.features.add_module('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1))
                self.features.add_module('bn4', nn.BatchNorm2d(128, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu4', nn.ReLU(inplace=True))
                self.features.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
                
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout2', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 3
                self.features.add_module('conv5', nn.Conv2d(128, 256, kernel_size=3, padding=1))
                self.features.add_module('bn5', nn.BatchNorm2d(256, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu5', nn.ReLU(inplace=True))
                self.features.add_module('conv6', nn.Conv2d(256, 256, kernel_size=3, padding=1))
                self.features.add_module('bn6', nn.BatchNorm2d(256, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu6', nn.ReLU(inplace=True))
                self.features.add_module('conv7', nn.Conv2d(256, 256, kernel_size=3, padding=1))
                self.features.add_module('bn7', nn.BatchNorm2d(256, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu7', nn.ReLU(inplace=True))
                self.features.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
                
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout3', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 4
                self.features.add_module('conv8', nn.Conv2d(256, 512, kernel_size=3, padding=1))
                self.features.add_module('bn8', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu8', nn.ReLU(inplace=True))
                self.features.add_module('conv9', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn9', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu9', nn.ReLU(inplace=True))
                self.features.add_module('conv10', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn10', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu10', nn.ReLU(inplace=True))
                self.features.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
                
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout4', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 5
                self.features.add_module('conv11', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn11', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu11', nn.ReLU(inplace=True))
                self.features.add_module('conv12', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn12', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu12', nn.ReLU(inplace=True))
                self.features.add_module('conv13', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn13', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu13', nn.ReLU(inplace=True))
                self.features.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
                dense_drop1 = getattr(self.config, "fc_dropout_rate1", self.config.dropout_rate)
                dense_drop2 = getattr(self.config, "fc_dropout_rate2", self.config.dropout_rate)
                
                # Classifier
                dense_size1 = 256 if num_classes == 10 else 512
                self.classifier = nn.Sequential(
                    nn.Linear(512, dense_size1),
                    nn.BatchNorm1d(dense_size1, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dense_drop1),
                    nn.Linear(dense_size1, 512),
                    nn.BatchNorm1d(512, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dense_drop2),
                    nn.Linear(512, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return VGGForCIFAR10(self.config)

    def build_cifar100_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """Build VGG16 model with BatchNormalization"""
        class VGGForCIFAR100(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                in_channels = input_shape[0]
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout1', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 2
                self.features.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
                self.features.add_module('bn3', nn.BatchNorm2d(128, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu3', nn.ReLU(inplace=True))
                self.features.add_module('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1))
                self.features.add_module('bn4', nn.BatchNorm2d(128, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu4', nn.ReLU(inplace=True))
                self.features.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
                
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout2', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 3
                self.features.add_module('conv5', nn.Conv2d(128, 256, kernel_size=3, padding=1))
                self.features.add_module('bn5', nn.BatchNorm2d(256, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu5', nn.ReLU(inplace=True))
                self.features.add_module('conv6', nn.Conv2d(256, 256, kernel_size=3, padding=1))
                self.features.add_module('bn6', nn.BatchNorm2d(256, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu6', nn.ReLU(inplace=True))
                self.features.add_module('conv7', nn.Conv2d(256, 256, kernel_size=3, padding=1))
                self.features.add_module('bn7', nn.BatchNorm2d(256, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu7', nn.ReLU(inplace=True))
                self.features.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
                
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout3', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 4
                self.features.add_module('conv8', nn.Conv2d(256, 512, kernel_size=3, padding=1))
                self.features.add_module('bn8', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu8', nn.ReLU(inplace=True))
                self.features.add_module('conv9', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn9', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu9', nn.ReLU(inplace=True))
                self.features.add_module('conv10', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn10', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu10', nn.ReLU(inplace=True))
                self.features.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
                
                if self.config.use_spatial_dropout:
                    self.features.add_module('spatial_dropout4', nn.Dropout2d(p=self.config.spatial_dropout_rate))
                
                # Block 5
                self.features.add_module('conv11', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn11', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu11', nn.ReLU(inplace=True))
                self.features.add_module('conv12', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn12', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu12', nn.ReLU(inplace=True))
                self.features.add_module('conv13', nn.Conv2d(512, 512, kernel_size=3, padding=1))
                self.features.add_module('bn13', nn.BatchNorm2d(512, momentum=self.config.batch_norm_momentum))
                self.features.add_module('relu13', nn.ReLU(inplace=True))
                self.features.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
                dense_drop1 = getattr(self.config, "fc_dropout_rate1", self.config.dropout_rate)
                dense_drop2 = getattr(self.config, "fc_dropout_rate2", self.config.dropout_rate)
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dense_drop1),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256, momentum=self.config.batch_norm_momentum),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dense_drop2),
                    nn.Linear(256, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return VGGForCIFAR100(self.config)