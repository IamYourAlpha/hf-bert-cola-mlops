
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    # callbacks
    # ModelCheckpoint saves the model with the lowest validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    # EarlyStopping stops training when the validation loss does not improve for a given number of epochs
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    logger_instance = WandbLogger(
        project="cola-classification",
        name="cola_model",
        log_model=True,
        save_dir="./wandb_logs"
    )    

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="gpu",
        devices=1,
        max_epochs=2,
        fast_dev_run=False,
        logger=logger_instance,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
