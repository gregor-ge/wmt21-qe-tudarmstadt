import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.adapters.heads import PredictionHead
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput
)


class BatchNormClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name='qe_da',
        num_labels=1,
        layers=2,
        activation_function="tanh",
        id2label=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def z_norm(self, inputs):
        mean = inputs.mean(0, keepdim=True)
        var = inputs.var(0, unbiased=False, keepdim=True)
        return (inputs - mean) / torch.sqrt(var + 1e-9)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        cls_output = cls_output if cls_output is not None else outputs[0][:, 0]
        cls_ouput = self.z_norm(cls_output)
        print('norming')
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs
