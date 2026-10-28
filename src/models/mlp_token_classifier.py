import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertMLPForTokenClassification(BertPreTrainedModel):
    """
    BERT token classification model with an MLP prediction head.

    Architecture:
        BERT encoder
            -> dropout
            -> linear layer
            -> activation
            -> dropout
            -> final linear layer
            -> token label logits
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)

        mlp_hidden_size = getattr(config, "mlp_hidden_size", config.hidden_size)
        mlp_dropout = getattr(config, "mlp_dropout", config.hidden_dropout_prob)
        mlp_activation = getattr(config, "mlp_activation", "gelu")

        if mlp_activation == "relu":
            activation_layer = nn.ReLU()
        elif mlp_activation == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported MLP activation: {mlp_activation}")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            activation_layer,
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_size, config.num_labels),
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
