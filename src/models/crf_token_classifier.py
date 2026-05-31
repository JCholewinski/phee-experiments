import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertCRFForTokenClassification(BertPreTrainedModel):
    """
    BERT token classification model with a CRF decoding layer.

    Architecture:
        BERT encoder
            -> dropout
            -> linear emission layer
            -> CRF layer
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf = CRF(config.num_labels, batch_first=True)

        self.ce_loss_weight = getattr(config, "crf_ce_loss_weight", 1.0)
        self.crf_loss_weight = getattr(config, "crf_loss_weight", 0.1)
        
        self.post_init()

    def _get_emissions(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
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
        emissions = self.classifier(sequence_output)

        return emissions, outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        crf_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        emissions, outputs = self._get_emissions(
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

        # loss = None
        # if labels is not None:
        #     mask = attention_mask.bool()

        #     loss = -self.crf(
        #         emissions,
        #         labels,
        #         mask=mask,
        #         reduction="mean",
        #     )
        loss = None
        if labels is not None:
            if crf_mask is not None:
                mask = crf_mask.bool()
            else:
                mask = attention_mask.bool()

            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0  # 0 = O

            crf_loss = -self.crf(
                emissions,
                crf_labels,
                mask=mask,
                reduction="token_mean",
            )

            ce_loss_fct = CrossEntropyLoss(ignore_index=-100)
            ce_loss = ce_loss_fct(
                emissions.view(-1, self.num_labels),
                labels.view(-1),
            )

            loss = self.ce_loss_weight * ce_loss + self.crf_loss_weight * crf_loss

        if not return_dict:
            output = (emissions,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=emissions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, input_ids=None, attention_mask=None, token_type_ids=None, crf_mask=None):
        emissions, _ = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        if crf_mask is not None:
            mask = crf_mask.bool()
        else:
            mask = attention_mask.bool()

        return self.crf.decode(emissions, mask=mask)
