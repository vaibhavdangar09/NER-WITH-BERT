import torch
from torch import nn
from transformers import BertModel

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_ner_labels=9, num_pos_labels=17):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.ner_head = nn.Linear(self.bert.config.hidden_size, num_ner_labels)
        self.pos_head = nn.Linear(self.bert.config.hidden_size, num_pos_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        ner_logits = self.ner_head(sequence_output)
        pos_logits = self.pos_head(sequence_output)
        return ner_logits, pos_logits
