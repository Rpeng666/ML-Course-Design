from . Encoder import Encoder 
from . Decoder import Decoder
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)

        return X