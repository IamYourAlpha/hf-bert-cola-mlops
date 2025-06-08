import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2"):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.num_classes = 2  # Binary classification for CoLA

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        h_cls = outputs.last_hidden_state[:, 0]  # CLS token representation
        logits = self.classifier(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # print (f"Training step {batch_idx}, loss: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(batch["label"].cpu(), preds.cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True)
        # print(f"Validation step {batch_idx}, loss: {loss.item()}, accuracy: {val_acc.item()}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer
