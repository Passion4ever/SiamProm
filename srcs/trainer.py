# Passion4ever
import logging
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast,GradScaler

from .utils import EarlyStopping, MetricTracker

logger = logging.getLogger('trainer')

class Trainer:

    def __init__(self, model, criterion, optimizer, metric_ftns, config, device, lr_schduler=None):
        # Prepare config and device
        self.config = config
        self.device = device
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare model, criterion, metrics, optimizer, lr_scheduler
        self.model = model
        self.ce_lossfunc, self.contr_lossfunc = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_schduler

        # Prepare trainer config
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.mnt_metric = cfg_trainer['monitor_metric']
        self.early_stopping = EarlyStopping(**config['trainer']['early_stop'])
        self.start_epoch = 1

        # Prepare MetricTracker
        self.metric_ftns = metric_ftns
        self.loss_metrics = MetricTracker()
        self.train_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns])

        self.scaler = GradScaler()

        if config.resume is not None:
            self._resume_ckpt(config.resume)

    def _train_epoch(self, train_loader, epoch, metric_tracker):
        """Training logic for an epoch"""

        self.model.train()

        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            seq_1, seq_2, label, label_1, label_2 = [data.to(self.device) for data in batch]
            self.optimizer.zero_grad()

            # Forward pass
            with autocast():
                out_1 = self.model(seq_1)
                out_2 = self.model(seq_2)
                c_loss = self.contr_lossfunc(out_1, out_2, label)
                out_3 = self.model.predict(seq_1)
                out_4 = self.model.predict(seq_2)
                p_loss = self.ce_lossfunc(out_3, label_1) + self.ce_lossfunc(out_4, label_2)
                loss = c_loss + p_loss

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metric track
            step = (epoch - 1) * len(train_loader) + batch_idx + 1
            loss_dic = {
                't_loss': loss.item(),
                'p_loss': p_loss.item(),
                'c_loss': c_loss.item()
            }
            metric_tracker.add(loss_dic, epoch, step)

        return metric_tracker.epoch_result(), time.time() - t0

    def _val_epoch(self, val_loader, epoch, metric_tracker):
        self.model.eval()

        score_lis = []
        label_lis = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                seq, label = [data.to(self.device) for data in batch]
                # Forward pass
                output = self.model.predict(seq)
                loss = self.ce_lossfunc(output, label)

                # Predict
                y_scores, y_preds = torch.max(output.data, 1)
                y_scores = y_scores.tolist()
                y_preds = y_preds.tolist()
                y_true = label.tolist()

                score_lis.extend(y_scores)
                label_lis.extend(y_true)

                # Step data
                step = (epoch - 1) * len(val_loader) + batch_idx + 1
                # Metric Track
                metric_result = {met.__name__: met(y_true, y_preds) for met in self.metric_ftns}
                metric_result.update({'loss': loss.item()})
                metric_tracker.add(metric_result, epoch, step)

        return metric_tracker.epoch_result()
    
    def train(self, train_loader, val_loader=None, log_train_metrics=True):
        train_dataloader, val_dataloader = val_loader

        for epoch in range(self.start_epoch, self.epochs + 1):

            # train and val every epoch
            loss_log, train_time = self._train_epoch(train_loader, epoch, self.loss_metrics)
            train_log = self._val_epoch(train_dataloader, epoch, self.train_metrics)
            val_log = self._val_epoch(val_dataloader, epoch, self.valid_metrics)

            # lr_scheduler call
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_log[self.mnt_metric])

            # earlystop call
            update, self.best_score, counts = self.early_stopping(val_log[self.mnt_metric])

            # epoch info
            self._log_epoch(epoch, train_time, counts, self.early_stopping.patience)
            # train_loss info
            self._log_metrics(loss_log, 'Loss')
            # val info
            self._log_metrics(train_log, 'Val_train_data')
            self._log_metrics(val_log, 'Val_val_data')

            # save and earlystop
            if update:
                self._save_ckpt(epoch, save_best=update)

            # if epoch % self.save_period == 0:
            #     self._save_ckpt(epoch)
                
            if self.early_stopping.early_stop:
                logger.info(f"{'':12s}EarlyStop!!!")
                break
        # save res
        with open(f'./mcc={self.best_score}', 'w') as f:
            f.write(f'{self.best_score}')

        return self.best_score
    
    def _save_ckpt(self, epoch, save_best=False):
        """Saving checkpoints

        Args:
            epoch: current epoch number
            save_best: if True, save the checkpoint to 'model_best.pth'
        """
        model = type(self.model).__name__
        optimizer = type(self.optimizer).__name__
        state = {
            # 'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'monitor_metric': {self.mnt_metric: self.best_score},
            'config': self.config
        }

        if save_best:
            best_path = str(self.config.save_dir + 'ckpt_best.pth')
            torch.save(state, best_path)
            logger.info("Saving current best: ckpt_best.pth ...")
        else:
            ckpt_name = str(self.config.save_dir + f'ckpt_epoch_{epoch}.pth')
            torch.save(state, ckpt_name)
            logger.info(f"Saving checkpoint: {ckpt_name} ...")

    def _resume_ckpt(self, resume_path):
        """Resume from saved checkpoints

        Args:
            resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        logger.info(f"Loading checkpoint: {resume_path} ...")
        ckpt = torch.load(resume_path)
        self.best_score = ckpt['monitor_metric'][self.mnt_metric]
        self.early_stopping.best_score = self.best_score
        print(self.best_score)
        print(f"ckpt_model:{ckpt['config']['model']['_target_']}")
        print(f"cur_model:{self.config['model']['_target_']}")
        if ckpt['config']['model']['_target_'] != self.config['model']['_target_']:
            logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(ckpt['model_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if ckpt['config']['optimizer']['_target_'] != self.config['optimizer']['_target_']:
            logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(ckpt['optim_state_dict'])

        logger.info(
            f"Checkpoint loaded. Resume checkpoint.")

    
    def _log_epoch(self, epoch, epoch_time, counts, patience):

        epoch_msg = f"{'✅'if counts == 0 else ''}EPOCH: {epoch} "\
                    f"EARLYSTOP_COUNTS: ({counts:02d}/{patience}) "\
                    f"USE: {epoch_time:9.6f}s"
        logger.info(f"{epoch_msg:-^120s}")
    
    def _log_metrics(self, metric_dict, description=''):
        logger.info(f"{description:^16s}: "
                         f"{' '.join([f'[{key}: {value:.6f}]' for key, value in metric_dict.items()])}")