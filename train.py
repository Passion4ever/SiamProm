import logging
import warnings

import hydra
import torch
from omegaconf import OmegaConf
from srcs.trainer import Trainer
from srcs.utils import instantiate, set_global_random_seed

OmegaConf.register_new_resolver("power", lambda x: 4**x)
OmegaConf.register_new_resolver("divide", lambda x: x // 2)
logger = logging.getLogger("train")


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg):
    warnings.filterwarnings("ignore")
    set_global_random_seed(cfg.seed)
    device = torch.device(f"cuda:{str(cfg.device)}")

    # 2. dataloader
    dataloaders = instantiate(cfg.data, is_func=True)()

    # 3. model
    model = instantiate(cfg.model.arch).to(device)
    # logger.info(model)

    # 4. loss
    ce_loss = instantiate(cfg.model.loss.ce_loss)
    ct_loss = instantiate(cfg.model.loss.ct_loss)

    # 5. metrics
    metrics = [instantiate(met, is_func=True) for met in cfg["metrics"]]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # 6. optim
    optimizer = instantiate(cfg.model.optim, trainable_params)

    # 7. lr_scheduler
    lr_scheduler = instantiate(cfg.model.lr_scheduler, optimizer)

    # 8. trainer
    trainer = Trainer(
        model,
        [ce_loss, ct_loss],
        optimizer,
        metrics,
        config=cfg,
        device=device,
        lr_schduler=lr_scheduler,
    )
    best_mcc = trainer.train(
        train_loader=dataloaders[2], val_loader=dataloaders[:2], log_train_metrics=False
    )

    return best_mcc


if __name__ == "__main__":
    main()
