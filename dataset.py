import json
import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_len = tokens["input_ids"].shape[1]

        # Pad or truncate NER and POS labels to match input length
        ner_labels = item["ner_labels"][:input_len] + [0] * (input_len - len(item["ner_labels"]))
        pos_labels = item["pos_labels"][:input_len] + [0] * (input_len - len(item["pos_labels"]))

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "ner_labels": torch.tensor(ner_labels, dtype=torch.long),
            "pos_labels": torch.tensor(pos_labels, dtype=torch.long),
        }
