import torch
from torch import nn
from pytorch_tcn import TCN
import transformers
from transformers import MobileViTV2Model

class EEGMobileVit_TCN(nn.Module):
    def __init__(self):
        super().__init__()

        # TCN layer
        self.tcn = TCN(
            num_inputs=129,
            num_channels=[64, 128, 256],  # for three layers
            kernel_size=3,
            dropout=0.75,
            causal=True,
            use_norm='weight_norm',
            activation='relu',
            kernel_initializer='xavier_uniform'
        )

        # Convolutional layers with batch normalization
        self.conv1 = torch.nn.Conv2d(1, 256, kernel_size=(1, 36), stride=(1, 36), padding=(0,2))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = torch.nn.Conv2d(256, 768, kernel_size=(256, 1), stride=(256, 1), padding=(0,0))
        self.bn2 = nn.BatchNorm2d(768)

        self.relu = nn.ReLU()

        # ViT configuration
        model_name = "apple/mobilevitv2-1.0-imagenet1k-256"
        config = transformers.MobileViTV2Config.from_pretrained(model_name)
        config.update({'num_channels': 768, 'image_size': (1, 14), 'patch_size': 1, 'conv_kernel_size ': 3})

        model = transformers.MobileViTV2ForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 1000, bias=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(1000, 2, bias=True)
        )
        self.ViT = model
            
    def forward(self, x):
        x = self.tcn(x.squeeze(1))  # Adjust for time dimension
        
        # Reshape and apply convolutions and batch normalization
        x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
    
        # Pass through ViT
        x = self.ViT(x).logits
        return x
