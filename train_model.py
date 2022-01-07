import os
from argparse import ArgumentParser

from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.DataModule import DataModule
from src.LightningModule import LightningModule
import pytorch_lightning as pl


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    datamodule = DataModule(
        train_path=relative_to(__file__, "../data/FashionMNIST/train"),
        val_path=relative_to(__file__, "../data/FashionMNIST/test"),
        batch_size=args.batch_size,
        num_workers=0,
    )
    datamodule.setup()

    model = LightningModule(
        batch_size=args.batch_size,
        dataset_size=datamodule.dataset_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        n_classes=datamodule.n_classes,
    )
    callbacks = [
        LearningRateMonitor(log_momentum=True),
    ]

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.n_epochs,
        callbacks=callbacks,
        auto_lr_find=True,
        checkpoint_callback=False,
    )
    # lr_finder = trainer.tuner.lr_find(model, datamodule)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()


    trainer.tune(model, datamodule)
    trainer.fit(model, datamodule)




def relative_to(file, relative: str) -> Path:
    dir_name = os.path.dirname(file)
    return Path(os.path.join(dir_name, relative))


if __name__ == "__main__":
    main()
