from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class GeneralTransformerEncoderLayer(nn.Module):
    r"""GeneralTransformerEncoderLayer is made up of Multiheadattention and feedforward network.
    This is exactly taken from pytorch code and modified so that you can pass q,k,v.
    Thus the same layer can be used for normal transformers and also co attentional transformers
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> query = torch.rand(10, 32, 512)
        >>> key = torch.rand(15, 32, 512)
        >>> value = torch.rand(15, 32, 512)
        >>> out = encoder_layer(query, key, value)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5):
        super(GeneralTransformerEncoderLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(GeneralTransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            query: query vector (required).
            key: key vector (optional). If not given key = query
            value: value vector (optional). If not given value = query
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if not key:
            key = query
        if not value:
            value = query
        query2 = self.multihead_attn(query, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        return query


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
