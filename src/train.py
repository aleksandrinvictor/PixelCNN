"""Run experiment."""

from pytorch_lightning.loggers import WandbLogger
from src.datasets import get_dataloaders_from_cfg
from src.pl import LitModule
from pytorch_lightning import Trainer, seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
from hydra.utils import instantiate


def from_config(
    cfg: DictConfig,
) -> Tuple[LitModule, Dict[str, DataLoader], List[Callback], Logger]:
    """Creates model, datamodule, callbacks and logger from config.

    Args:
        cfg: Experiment config.

    Returns:
        Tuple (model, datamodule, callbacks, logger).
    """
    train_dataloader, val_dataloader = get_dataloaders_from_cfg(cfg)

    if cfg.pretrained_path is not None:
        lit_module = LitModule.load_from_checkpoint(checkpoint_path=cfg.pretrained_path, cfg=cfg, strict=False)
    else:
        lit_module = LitModule(cfg)

    callbacks = instantiate(cfg.callbacks)

    if "logger" in cfg:
        exp_logger = instantiate(cfg.logger)

        if isinstance(exp_logger, WandbLogger):
            exp_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    else:
        exp_logger = None

    return lit_module, {"train": train_dataloader, "val": val_dataloader}, callbacks, exp_logger


@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    seed_everything(config.seed)

    lit_module, dataloaders, callbacks, logger = from_config(config)

    trainer = Trainer(
        max_epochs=config.num_epochs,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
        accelerator=config.accelerator,
        devices=config.devices,
    )

    trainer.fit(lit_module, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"])


if __name__ == "__main__":
    main()
