from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate


class LitModule(pl.LightningModule):
    """Class for training best shot models.

    Args:
        cfg: Experiment config.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.model = instantiate(cfg.model)

    def configure_optimizers(self) -> Dict[str, nn.Module]:
        """Setup optimizers and schedulers."""

        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

        if "scheduler" in self.cfg:
            scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)

            scheduler = {
                "scheduler": scheduler,
                "interval": self.cfg.vars.run.scheduler_interval,
                "frequency": 1,
                "monitor": self.cfg.vars.run.scheduler_monitor_val,
            }

            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Implements forward step and logs loss and metrics.

        Args:
            batch: Input batch.
            batch_idx: batch index (not used).

        Returns:
            Train batch loss.
        """
        model_output = self.model(**batch)

        self.log(
            "train_loss",
            model_output["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return model_output["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Implements validation step and logs loss and metrics.

        Args:
            batch: Input batch.
            batch_idx: batch index (not used).
        """
        model_output = self.model(**batch)

        self.log(
            "val_loss",
            model_output["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
