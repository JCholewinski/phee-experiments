import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertFrozenLinearCRFForTokenClassification(BertPreTrainedModel):
    """
    Two-stage Linear + CRF.

    Stage 1:
        Train normal BERT + linear token classifier.

    Stage 2:
        Load trained checkpoint, freeze BERT + linear classifier,
        train only CRF transitions on top of fixed emission scores.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.post_init()
        self.freeze_encoder_and_classifier()

    def freeze_encoder_and_classifier(self):
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = False

    def _get_emissions(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs,
        )

        sequence_output = outputs[0]

        # Do not apply dropout during CRF-only training,
        # because BERT + classifier are frozen.
        emissions = self.classifier(sequence_output)

        return emissions, outputs

    def _compact_for_crf(self, emissions, labels=None, crf_mask=None):
        """
        Converts tokenizer-level sequence into compact CRF sequence.

        Example:
        tokens:   [CLS] word1 ##sub word2 [SEP] [PAD]
        crf_mask:   1     1    0    1    0     0

        CRF receives:
        [CLS] word1 word2
        """
        if crf_mask is None:
            raise ValueError("crf_mask is required.")

        batch_size, _, num_labels = emissions.shape
        device = emissions.device

        crf_mask = crf_mask.bool()
        active_lengths = crf_mask.sum(dim=1)
        max_len = int(active_lengths.max().item())

        compact_emissions = emissions.new_zeros(batch_size, max_len, num_labels)
        compact_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        compact_labels = None
        if labels is not None:
            compact_labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for i in range(batch_size):
            active_positions = torch.where(crf_mask[i])[0]
            length = active_positions.numel()

            compact_emissions[i, :length] = emissions[i, active_positions]
            compact_mask[i, :length] = True

            if labels is not None:
                selected_labels = labels[i, active_positions].clone()
                selected_labels[selected_labels == -100] = 0
                compact_labels[i, :length] = selected_labels

        return compact_emissions, compact_labels, compact_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        crf_mask=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        emissions, outputs = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs,
        )

        loss = None

        if labels is not None:
            compact_emissions, compact_labels, compact_mask = self._compact_for_crf(
                emissions=emissions,
                labels=labels,
                crf_mask=crf_mask,
            )

            loss = -self.crf(
                compact_emissions,
                compact_labels,
                mask=compact_mask,
                reduction="mean",
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=emissions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        crf_mask=None,
    ):
        emissions, _ = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        compact_emissions, _, compact_mask = self._compact_for_crf(
            emissions=emissions,
            labels=None,
            crf_mask=crf_mask,
        )

        return self.crf.decode(compact_emissions, mask=compact_mask)