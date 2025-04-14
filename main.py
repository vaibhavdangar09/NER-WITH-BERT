import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW

from model import MultiTaskBERT
from dataset import MultiTaskDataset

def train_model(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, ner_labels, pos_labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["ner_labels"].to(device),
                batch["pos_labels"].to(device),
            )

            optimizer.zero_grad()
            ner_logits, pos_logits = model(input_ids, attention_mask)

            loss_fn = torch.nn.CrossEntropyLoss()
            ner_loss = loss_fn(ner_logits.view(-1, ner_logits.shape[-1]), ner_labels.view(-1))
            pos_loss = loss_fn(pos_logits.view(-1, pos_logits.shape[-1]), pos_labels.view(-1))
            loss = ner_loss + pos_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = MultiTaskDataset("data/train.json", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = MultiTaskBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_model(model, train_loader, optimizer, epochs=3)

    # ✅ Save the trained model
    torch.save(model.state_dict(), "multi_task_bert.pth")
    print("Model saved to multi_task_bert.pth")
