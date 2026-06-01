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
    
    def _apply_bio_constraints(self):
        """
        Enforce basic BIO constraints in the CRF transition matrix.

        Invalid:
            START -> I-X
            O -> I-X
            B-Y -> I-X where X != Y
            I-Y -> I-X where X != Y

        Valid:
            B-X -> I-X
            I-X -> I-X
        """
        neg_value = -10000.0

        id2label = {
            int(k): v for k, v in self.config.id2label.items()
        }

        with torch.no_grad():
            for to_id, to_label in id2label.items():
                if not to_label.startswith("I-"):
                    continue

                to_type = to_label[2:]

                # Sequence should not start with I-X
                self.crf.start_transitions[to_id] = neg_value

                for from_id, from_label in id2label.items():
                    valid_previous = (
                        from_label == f"B-{to_type}"
                        or from_label == f"I-{to_type}"
                    )

                    if not valid_previous:
                        self.crf.transitions[from_id, to_id] = neg_value

    def _apply_zero_bio_constraints(self):
        """
        Use CRF decoding only for BIO-valid transitions.

        All valid transitions get score 0.
        Invalid BIO transitions get a large negative score.
        This prevents learned/random CRF transitions from dominating emissions.
        """
        neg_value = -10000.0

        id2label = {int(k): v for k, v in self.config.id2label.items()}

        with torch.no_grad():
            # Reset all transitions to neutral
            self.crf.transitions.fill_(0.0)
            self.crf.start_transitions.fill_(0.0)
            self.crf.end_transitions.fill_(0.0)

            for to_id, to_label in id2label.items():
                if not to_label.startswith("I-"):
                    continue

                to_type = to_label[2:]

                # Cannot start a sequence with I-X
                self.crf.start_transitions[to_id] = neg_value

                for from_id, from_label in id2label.items():
                    valid_previous = (
                        from_label == f"B-{to_type}"
                        or from_label == f"I-{to_type}"
                    )

                    if not valid_previous:
                        self.crf.transitions[from_id, to_id] = neg_value

    def _apply_soft_bio_constraints(self):
        """
        Softly discourage invalid BIO transitions instead of fully banning them.
        This makes CRF decoding more conservative and closer to emission argmax.
        """
        penalty = getattr(self.config, "invalid_transition_penalty", -2.0)

        id2label = {int(k): v for k, v in self.config.id2label.items()}

        with torch.no_grad():
            self.crf.transitions.fill_(0.0)
            self.crf.start_transitions.fill_(0.0)
            self.crf.end_transitions.fill_(0.0)

            for to_id, to_label in id2label.items():
                if not to_label.startswith("I-"):
                    continue

                to_type = to_label[2:]

                # Starting with I-X is suspicious, but not impossible
                self.crf.start_transitions[to_id] = penalty

                for from_id, from_label in id2label.items():
                    valid_previous = (
                        from_label == f"B-{to_type}"
                        or from_label == f"I-{to_type}"
                    )

                    if not valid_previous:
                        self.crf.transitions[from_id, to_id] = penalty

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
            self._apply_zero_bio_constraints()
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
    
    def _build_bio_transition_scores(self, device, dtype):
        """
        Build a controlled BIO transition matrix for conservative decoding.

        transition_scores[from_label, to_label]

        Valid transitions get score 0.
        Invalid BIO transitions get a configurable negative penalty.
        """
        penalty = getattr(self.config, "invalid_transition_penalty", -4.0)

        id2label = {
            int(k): v for k, v in self.config.id2label.items()
        }

        transition_scores = torch.zeros(
            self.num_labels,
            self.num_labels,
            device=device,
            dtype=dtype,
        )

        start_scores = torch.zeros(
            self.num_labels,
            device=device,
            dtype=dtype,
        )

        for to_id, to_label in id2label.items():
            if not to_label.startswith("I-"):
                continue

            to_type = to_label[2:]

            # Starting directly with I-X is discouraged.
            start_scores[to_id] = penalty

            for from_id, from_label in id2label.items():
                valid_previous = (
                    from_label == f"B-{to_type}"
                    or from_label == f"I-{to_type}"
                )

                if not valid_previous:
                    transition_scores[from_id, to_id] = penalty

        return transition_scores, start_scores

    def _constrained_viterbi_decode(self, emissions, mask):
        """
        Conservative Viterbi decoding using emission scores and BIO transition penalties.

        emissions: batch_size x seq_len x num_labels
        mask:      batch_size x seq_len

        Important:
        The mask may contain gaps because subword continuation tokens are ignored.
        Therefore, we must gather emissions from active positions instead of taking
        emissions[:, :seq_len].
        """
        batch_paths = []

        transition_scores, start_scores = self._build_bio_transition_scores(
            device=emissions.device,
            dtype=emissions.dtype,
        )

        batch_size = emissions.size(0)

        for batch_idx in range(batch_size):
            active_positions = torch.nonzero(
                mask[batch_idx],
                as_tuple=False,
            ).squeeze(-1)

            if active_positions.numel() == 0:
                batch_paths.append([])
                continue

            seq_emissions = emissions[batch_idx, active_positions]

            # First active timestep
            scores = seq_emissions[0] + start_scores
            backpointers = []

            # Remaining active timesteps
            for t in range(1, seq_emissions.size(0)):
                candidate_scores = (
                    scores.unsqueeze(1)
                    + transition_scores
                    + seq_emissions[t].unsqueeze(0)
                )

                best_scores, best_previous_labels = candidate_scores.max(dim=0)

                scores = best_scores
                backpointers.append(best_previous_labels)

            # Backtrack
            best_last_label = int(scores.argmax().item())
            best_path = [best_last_label]

            for previous_labels in reversed(backpointers):
                best_last_label = int(previous_labels[best_last_label].item())
                best_path.append(best_last_label)

            best_path.reverse()
            batch_paths.append(best_path)

        return batch_paths

    def decode_constrained(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        crf_mask=None,
    ):
        """
        Decode with controlled BIO-constrained Viterbi instead of torchcrf.decode().
        This is more conservative and keeps emissions dominant.
        """
        emissions, _ = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        emission_scale = getattr(self.config, "emission_scale", 5.0)
        emissions = emissions * emission_scale

        if crf_mask is not None:
            mask = crf_mask.bool()
        else:
            mask = attention_mask.bool()

        return self._constrained_viterbi_decode(emissions, mask)

    def decode(self, input_ids=None, attention_mask=None, token_type_ids=None, crf_mask=None):
        emissions, _ = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        emission_scale = getattr(self.config, "emission_scale", 3.0)
        emissions = emissions * emission_scale

        self._apply_soft_bio_constraints()

        if crf_mask is not None:
            mask = crf_mask.bool()
        else:
            mask = attention_mask.bool()

        return self.crf.decode(emissions, mask=mask)
