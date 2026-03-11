import os
import time
import torch
import numpy as np

from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics
from src.utils.swanlab_tracker import SwanLabTracker

class BaseEngine():
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, \
                 swanlab_cfg=None, log_interval=0):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._log_interval = max(0, int(log_interval))

        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 
        if swanlab_cfg is None:
            swanlab_cfg = {}
        default_exp_name = "{}-s{}".format(self.model.__class__.__name__, self._seed)
        self._swanlab = SwanLabTracker(
            enabled=swanlab_cfg.get("enabled", False),
            logger=self._logger,
            project=swanlab_cfg.get("project", "LargeST"),
            experiment_name=swanlab_cfg.get("experiment_name", default_exp_name),
            config=swanlab_cfg.get("config", {}),
            mode=swanlab_cfg.get("mode", "cloud"),
            logdir=swanlab_cfg.get("logdir", self._save_path),
            description=swanlab_cfg.get("description", ""),
            job_type=swanlab_cfg.get("job_type", "train"),
            lark_webhook_url=swanlab_cfg.get("lark_webhook_url", ""),
            lark_secret=swanlab_cfg.get("lark_secret", ""),
        )

    def _assert_finite(self, name, tensor):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("{} contains NaN/Inf".format(name))

    def close(self):
        self._swanlab.finish()

    def _log_test_metrics(self, test_mae, test_rmse, test_mape):
        for i, (mae, rmse, mape) in enumerate(zip(test_mae, test_rmse, test_mape), start=1):
            self._swanlab.log(
                {
                    "test/horizon_{}/mae".format(i): mae,
                    "test/horizon_{}/rmse".format(i): rmse,
                    "test/horizon_{}/mape".format(i): mape,
                },
                step=i,
            )

        avg_mae = np.mean(test_mae)
        avg_rmse = np.mean(test_rmse)
        avg_mape = np.mean(test_mape)
        self._swanlab.log(
            {
                "test/avg_mae": avg_mae,
                "test/avg_rmse": avg_rmse,
                "test/avg_mape": avg_mape,
            }
        )
        return avg_mae, avg_rmse, avg_mape


    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}.pt'.format(self._seed)
        state_dict = self.model.state_dict()
        if bool(getattr(self.model, "checkpoint_trainable_only", False)):
            trainable_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
            state_dict = {k: v for k, v in state_dict.items() if k in trainable_names}
            self._logger.info(
                "Save trainable-only checkpoint with {} tensors.".format(len(state_dict))
            )
        torch.save(state_dict, os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}.pt'.format(self._seed)
        ckpt_path = os.path.join(save_path, filename)
        try:
            state_dict = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=self._device)
        strict = not bool(getattr(self.model, "checkpoint_trainable_only", False))
        self.model.load_state_dict(state_dict, strict=strict)


    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        total_batch = self._dataloader['train_loader'].num_batch
        start_time = time.time()
        for batch_idx, (X, label) in enumerate(self._dataloader['train_loader'].get_iterator(), start=1):
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            self._assert_finite("train/pred", pred)
            self._assert_finite("train/label", label)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            self._assert_finite("train/loss", loss)
            self._assert_finite("train/rmse", torch.tensor(rmse, device=pred.device))
            self._assert_finite("train/mape", torch.tensor(mape, device=pred.device))

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
            if self._log_interval > 0 and (
                batch_idx == 1
                or batch_idx % self._log_interval == 0
                or batch_idx == total_batch
            ):
                elapsed = time.time() - start_time
                avg_batch_time = elapsed / batch_idx
                eta = avg_batch_time * max(total_batch - batch_idx, 0)
                self._logger.info(
                    "Train Batch {}/{}, Loss {:.4f}, RMSE {:.4f}, MAPE {:.4f}, "
                    "Avg {:.2f}s/batch, ETA {:.1f} min".format(
                        batch_idx,
                        total_batch,
                        np.mean(train_loss),
                        np.mean(train_rmse),
                        np.mean(train_mape),
                        avg_batch_time,
                        eta / 60.0,
                    )
                )
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, \
                                             (t2 - t1), (v2 - v1), cur_lr))
            self._swanlab.log(
                {
                    "train/loss": mtrain_loss,
                    "train/rmse": mtrain_rmse,
                    "train/mape": mtrain_mape,
                    "valid/loss": mvalid_loss,
                    "valid/rmse": mvalid_rmse,
                    "valid/mape": mvalid_mape,
                    "train/epoch_time_sec": (t2 - t1),
                    "valid/epoch_time_sec": (v2 - v1),
                    "optim/lr": cur_lr,
                },
                step=epoch + 1,
            )

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
                self._swanlab.log(
                    {
                        "valid/best_loss": min_loss,
                        "valid/best_epoch": epoch + 1,
                    },
                    step=epoch + 1,
                )
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    self._swanlab.log(
                        {
                            "train/early_stop_epoch": epoch + 1,
                            "valid/best_loss": min_loss,
                        },
                        step=epoch + 1,
                    )
                    break

        self.evaluate('test')


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                self._assert_finite("{}/pred".format(mode), pred)
                self._assert_finite("{}/label".format(mode), label)

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            if not np.isfinite(mae) or not np.isfinite(mape) or not np.isfinite(rmse):
                raise FloatingPointError(
                    "val metrics contain NaN/Inf: mae={}, rmse={}, mape={}".format(
                        mae, rmse, mape
                    )
                )
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
            avg_mae, avg_rmse, avg_mape = self._log_test_metrics(
                test_mae, test_rmse, test_mape
            )
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(avg_mae, avg_rmse, avg_mape))
