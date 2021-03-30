import torch

from torch import nn
from transformers import BertModel
from einops import rearrange

from ..blocks.vision.vit import Vit
from ..blocks.my_transformers import CoAttentionModule


class TilBert(nn.Module):
    """TILBERT - fully Transformer based architecture for combined Image-and-Language tasks.
    This model is inspired from ViT and ViLBERT.
    """

    def __init__(
        self,
        vit_model: Vit,
        bert_model: BertModel,
        coattention_transformer_config: dict,
        output_transformer_config: dict,
        num_of_combined_coattention_and_output_transformers: int = 1,
        classifier_type: str = "token",
        merge_mode: str = "mul",
    ) -> None:
        super(TilBert, self).__init__()
        # vit model, the default configs output: Batch_size, num_patches, 768
        self.vit_model = vit_model
        # bert model pretrained: bert-base-uncased outputs, Batch_size, Num tokens, 768
        self.bert_model = bert_model
        assert vit_model.vit_dim == bert_model.config.hidden_size, "ViT's output shape {} not equal to Bert's output shape {}".format(
            vit_model.vit_dim, bert_model.config.hidden_size
        )
        self.coattention_module = CoAttentionModule(
            coattention_transformer_config=coattention_transformer_config,
            output_transformer_config=output_transformer_config,
            num_of_combined_coattention_and_output_transformers=num_of_combined_coattention_and_output_transformers,
        )

        self.classifier_type = classifier_type
        self.merge_mode = merge_mode

    def forward(self, image: torch.Tensor, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor, text_token_type_ids: torch.Tensor):
        image_embedding = self.vit_model(image)
        text_embedding = self.bert_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask, token_type_ids=text_token_type_ids
        ).last_hidden_state

        # hugging face gives outputs in batchsize, seqlen, embedding
        text_embedding = rearrange(text_embedding, "b l e -> l b e")
        # check if batch axis align
        assert image_embedding.shape[1] == text_embedding.shape[1]

        image_embedding, text_embedding = self.coattention_module(image_embedding=image_embedding, text_embedding=text_embedding)

        # now pick the [CLS] token in both the cases
        if self.classifier_type == "token":
            image_embedding = image_embedding[0]
            text_embedding = text_embedding[0]

        # average the tokens
        elif self.classifier_type == "gap":
            image_embedding = image_embedding.mean(dim=0)
            text_embedding = text_embedding.mean(dim=0)

        if self.merge_mode == "mul":
            return image_embedding * text_embedding

        elif self.merge_mode == "add":
            return image_embedding + text_embedding

        elif self.merge_mode == "concat":
            return torch.cat([image_embedding, text_embedding], axis=0)
