import torch
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, data):
        return self.tokenizer(
            data["sentence"], truncation=True, padding="max_length", max_length=128
        )

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(
                self.tokenize_data,
                batched=True,
            )
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )
            self.val_data = self.val_data.map(
                self.tokenize_data,
                batched=True,
            )
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    data_module = DataModule()
    data_module.prepare_data()
    data_module.setup("fit")

    for batch in data_module.train_dataloader():
        print(batch)
        break  # Just to show one batch
