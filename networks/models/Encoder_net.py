import torch
import torch.nn as nn
from DaViT_source.timm.models import create_model

class DaViT_Encoder(nn.Module):
    def __init__(self, output_dims):
        super(DaViT_Encoder, self).__init__()
        # load DaViT tiny  model
        self.DaViT =  create_model(
            model_name='DaViT_tiny')
        # get number of features in the last layer of DaViT Tiny
        self.DaViT_out_fts = self.DaViT.head.out_features
        # add Linear layer to match output dims
        # self.outLayer = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(self.DaViT_out_fts, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dims)
        # ) 
        self.outLayer = nn.Sequential(
            # nn.BatchNorm1d(self.DaViT_out_fts),
            nn.ReLU(),
            nn.Linear(self.DaViT_out_fts, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dims)
        ) 
        # nn.init.ones_(self.outLayer[5].weight)

    def forward(self, x):
        x = self.DaViT(x)
        x = self.outLayer(x)
        return x

