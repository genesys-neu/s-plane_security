import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerNN(nn.Module):
    def __init__(self, classes: int = 1, num_feats: int = 6, slice_len: int = 40, nhead: int = 3, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False):
        super(TransformerNN, self).__init__()
        self.norm = nn.LayerNorm(num_feats)

        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(num_feats, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats * slice_len, 256)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(256, classes)


    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        # src = self.norm(src) should not be necessary since output can be already normalized
        # pass through encoder layers
        t_out = self.transformer_encoder(src)
        # flatten already contextualized KPIs
        t_out = torch.flatten(t_out, start_dim=1)
        # Pass through MLP classifier
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = torch.sigmoid(output)
        return output

model = TransformerNN()
print(sum(p.numel() for p in model.parameters()))