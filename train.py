import logging
import warnings

import hydra
import mindspore as ms
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
    ms.set_context(device_target=cfg.device)
    # device = torch.device(f"cuda:{str(cfg.device)}")

    # 2. dataloader
    dataloaders = instantiate(cfg.data, is_func=True)()

    # 3. model
    model = instantiate(cfg.model.arch)
    # logger.info(model)

    # 4. loss
    ce_loss = instantiate(cfg.model.loss.ce_loss)
    ct_loss = instantiate(cfg.model.loss.ct_loss)

    # 5. metrics
    metrics = [instantiate(met, is_func=True) for met in cfg["metrics"]]

    # 6. optim
    optimizer = instantiate(cfg.model.optim, model.trainable_params())

    # 7. lr_scheduler
    lr_scheduler = instantiate(cfg.model.lr_scheduler, optimizer)

    # 8. trainer
    trainer = Trainer(
        model,
        [ce_loss, ct_loss],
        optimizer,
        metrics,
        config=cfg,
        lr_schduler=lr_scheduler,
    )


    trainer.train(train_loader=dataloaders[2], val_loader=dataloaders[:2])



if __name__ == "__main__":
    main()