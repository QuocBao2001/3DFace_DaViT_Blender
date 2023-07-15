import torch
import torch.nn as nn
from DaViT_source.timm.models import create_model

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Up_non_cat(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class Fine_net(nn.Module):
    def __init__(self, input_channels=6, skip_layers_dims = [384, 192, 96], out_channels=3):
        super().__init__()

        # encoder of fine net is DaViT tiny model
        self.DaViT =  create_model(
            model_name='DaViT_tiny',
            checkpoint_path='/root/baonguyen/3d_face_reconstruction/data/DaViT/DaViT_Pretrained.pth.tar')

        # change input dims to match input channels
        self.DaViT.patch_embeds[0].proj = nn.Conv2d(input_channels, 
                                                    96, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        # Initialize variable to store the output of DaViT.patch_embeds
        self.patch_embeds_output_0 = None
        self.patch_embeds_output_1 = None
        self.patch_embeds_output_2 = None

        # register forward hook
        self.DaViT.patch_embeds[0].proj.register_forward_hook(self.save_patch_embeds_output_0)
        self.DaViT.patch_embeds[1].proj.register_forward_hook(self.save_patch_embeds_output_1)
        self.DaViT.patch_embeds[2].proj.register_forward_hook(self.save_patch_embeds_output_2)

        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(1000, 256))
        self.l2 = nn.Sequential(nn.Linear(256, 256 * self.init_size ** 2))

        # Decoder for texture
        self.tex_up1 = Up(256 + 384, 256)
        self.tex_up2 = Up(256 + 192, 128)
        self.tex_up3 = Up(128 + 96, 64)

        # upsample to match output size
        self.tex_up4 = Up_non_cat(64, 32)
        self.tex_up5 = Up_non_cat(32, out_channels)

    
    def save_patch_embeds_output_0(self, module, input, output):
        self.patch_embeds_output_0 = output

    def save_patch_embeds_output_1(self, module, input, output):
        self.patch_embeds_output_1 = output

    def save_patch_embeds_output_2(self, module, input, output):
        self.patch_embeds_output_2 = output

    def forward(self, img_stack):
        # encoder
        x = self.DaViT(img_stack)
        
        # decode for texture
        x = self.l1(x)
        x = self.l2(x)
        x = x.view(x.shape[0], 256, self.init_size, self.init_size)  #(256, 8, 8)
        
        t = self.tex_up1(x, self.patch_embeds_output_2)
        t = self.tex_up2(t, self.patch_embeds_output_1)
        t = self.tex_up3(t, self.patch_embeds_output_0)
        t = self.tex_up4(t)
        t = self.tex_up5(t)

        return t