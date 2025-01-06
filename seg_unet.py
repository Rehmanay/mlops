import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, num_classes, n_channels=3, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=num_classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = UNet(num_classes=58)
    input = torch.Tensor(1, 3, 832, 544)
    output = model(input)
    print(f'output of the model is : {output.shape}')