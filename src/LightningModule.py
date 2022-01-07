import math
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

from src.Model import seq_model


class LightningModule(pl.LightningModule):
    def __init__(
            self,
            *,
            n_classes: int,
            batch_size: int,
            dataset_size: int,
            n_epochs: int,
            lr: float = 5e-4,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.network = seq_model(n_classes=n_classes)


    def training_step(
            self,
            batch: Any,
            batch_idx: int,
            optimizer_idx: Optional[int] = None,
            **kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self.network(x)


        loss = F.cross_entropy(logits, y)
        accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())

        self.log("training/accuracy", accuracy, on_epoch=True, on_step=True)
        self.log("training/loss", loss, on_epoch=True, on_step=True)

        return {"loss": loss}

    def validation_step(
            self,
            batch: Any,
            batch_idx: int,
            **kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self.network(x)

        loss = F.cross_entropy(logits, y)
        accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())

        self.log("validation/accuracy", accuracy, on_epoch=True)
        self.log("validation/loss", accuracy, on_epoch=True)

        return {"loss": loss}


    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Any]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        total_steps = math.ceil(self.hparams.dataset_size / self.hparams.batch_size) * self.hparams.n_epochs
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.hparams.lr,
                total_steps=total_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
