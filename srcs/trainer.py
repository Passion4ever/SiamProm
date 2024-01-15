# Passion4ever
import logging
import time
from pathlib import Path


import mindspore as ms
from mindspore import ops
from .utils import EarlyStopping, MetricTracker

logger = logging.getLogger('trainer')

class Trainer:

    def __init__(self, model, criterion, optimizer, metric_ftns, config, lr_schduler=None):
        # Prepare config and device
        self.config = config
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


        self.grad_fn = ms.value_and_grad(self._forward_fn, None, self.optimizer.parameters, has_aux=True)

    def _forward_fn(self, data, label):
        seq_1, seq_2 = data
        c_label, label_1, label_2 = label
        out_1 = self.model(seq_1)
        out_2 = self.model(seq_2)
        out_3 = self.model.predict(seq_1)
        out_4 = self.model.predict(seq_2)
        label_1 = label_1.astype(ms.int32)
        label_2 = label_2.astype(ms.int32)
        c_loss = self.contr_lossfunc(out_1, out_2, c_label)
        p_loss = self.ce_lossfunc(out_3, label_1) + self.ce_lossfunc(out_4, label_2)
        loss = c_loss + p_loss

        return loss, (out_1, out_2, out_3, out_4)

    def _train_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss
    

    def _train_epoch(self, train_loader, epoch, metric_tracker):
        """Training logic for an epoch"""

        self.model.set_train()

        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            data, label = batch[:2], batch[2:]

            loss = self._train_step(data, label)
            # Metric track
            step = (epoch - 1) * len(train_loader) + batch_idx + 1
            loss_dic = {
                'loss': loss.item(),
            }
            metric_tracker.add(loss_dic, epoch, step)

        return metric_tracker.epoch_result(), time.time() - t0

    def _val_epoch(self, val_loader, epoch, metric_tracker):

        self.model.set_train(False)

        score_lis = []
        label_lis = []
        for batch_idx, batch in enumerate(val_loader):
            seq, label = batch
            # Forward pass
            output = self.model.predict(seq)
            loss = self.ce_lossfunc(output, label)

            # Predict
            y_scores, y_preds = ops.max(output, 1)
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
    
    def train(self, train_loader, val_loader=None):
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
                best_path = str(self.config.save_dir + 'ckpt_best.ckpt')
                ms.save_checkpoint(self.model, best_path)

            if epoch % self.save_period == 0:
                saved_path = str(self.config.save_dir + f'epoch_{epoch}.ckpt')
                self._save_ckpt(self.model, saved_path)
                
            if self.early_stopping.early_stop:
                logger.info(f"{'':12s}EarlyStop!!!")
                break

        return self.best_score
    
    def _log_epoch(self, epoch, epoch_time, counts, patience):
        epoch_msg = f"{'✅'if counts == 0 else ''}EPOCH: {epoch} "\
                    f"EARLYSTOP_COUNTS: ({counts:02d}/{patience}) "\
                    f"USE: {epoch_time:9.6f}s"
        logger.info(f"{epoch_msg:-^120s}")
    
    def _log_metrics(self, metric_dict, description=''):
        logger.info(f"{description:^16s}: "
                         f"{' '.join([f'[{key}: {value:.6f}]' for key, value in metric_dict.items()])}")
        