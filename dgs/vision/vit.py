import torch

from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat


class Vit(nn.Module):
    def __init__(
        self,
        image_size: int,
        channels: int = 3,
        patch_size: int = 16,
        vit_dim: int = 1024,
        num_transformer_layers: int = 6,
        nheads: int = 8,
        transformer_mlp_dim: int = 2048,
        classifier_type: str = "token",
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        transformer_activation: str = "gelu",
    ):
        """
        Vision Transformer: AN IMAGE IS WORTH 16X16 WORDS https://arxiv.org/pdf/2010.11929.pdf

        Arguments:
            image_size (int): The input image size. If rectangular images, make sure to give max(H,W). No Default.
            channels (int): Number of input channels. Default 3.
            patch_size (int): The patch size for the Vit. Default 16 (16 X 16 pixels).
            vit_dim (int): The dimension of the vision transformer. This is the output shape of the feedforward layer before the transformers
                This is also the output shape of the VIT. Default 1024.
            num_transformer_layers (int): The number of transformer encoder blocks to include. Default 6.
            nheads (int): Number of heads in Multi-HeadSelf-Attention. Default 8.
            transformer_mlp_dim (int): The dimension of the MLP layer used in each transformer block. Default 2048.
            classifier_type (str): The classifier type to use. One of 'token', 'gap'. Default 'token'.
                See here for more details: https://github.com/google-research/vision_transformer/issues/60.
            transformer_dropout (float): Dropout for each transformer layer. Default 0.1
            embedding_dropout (float): Dropout for the pos_embedding + patch_embedding for VIT. Default 0.1
            transformer_activation (str): Activation function for each of the transformer. Default 'gelu'


        Input: image (Batch, Channel, Height, Width)
        Output: latent representation (Batch, vit_dim)

        Usage:
            >> image_size = 256
            >> vit_dim = 1024
            >> image_encoder = Vit(image_size=image_size, vit_dim=vit_sim)
            >> image = torch.randn(5, 3, image_size, image_size)
            >> output = image_encoder(image)
            >> print(output.shape) # torch.Size([5, 1024])

        References:
            - https://github.com/lucidrains/vit-pytorch (Pytorch VIT implementation)
            - https://github.com/google-research/vision_transformer (official implementation)
        """
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        assert vit_dim % nheads == 0, "The vit_dim must be divisible by the nheads for MultiheadSelfAttention to work"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert classifier_type in {"token", "gap"}, "classifier_type type must be either token, gap"
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, vit_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, vit_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim, nhead=nheads, dim_feedforward=transformer_mlp_dim, dropout=transformer_dropout, activation=transformer_activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_transformer_layers)

        self.classifier_type = classifier_type
        self.to_latent = nn.Identity()
        self.final_norm = nn.LayerNorm(vit_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image is of shape B, C, H, W
        x = self.to_patch_embedding(image)
        batch_size, n, _ = x.shape  # B, N, P^2 * C where N = H*W/P^2, check the vit paper for  more info
        class_tokens = repeat(self.class_token, "() n d -> b n d", b=batch_size)
        x = torch.cat((class_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.embedding_dropout(x)
        x = self.transformer(x)
        if self.classifier_type == "token":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.final_norm(x)
