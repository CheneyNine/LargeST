import numpy as np
import torch

from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse


class STEVE_Engine(BaseEngine):
    def __init__(self, variant_loss_weight, invariant_loss_weight, mi_loss_weight, **args):
        super(STEVE_Engine, self).__init__(**args)
        self._variant_loss_weight = variant_loss_weight
        self._invariant_loss_weight = invariant_loss_weight
        self._mi_loss_weight = mi_loss_weight

        train_loader = self._dataloader["train_loader"]
        self._total_train_iters = max(1, self._max_epochs * train_loader.num_batch)

    def _compute_mask_value(self, label):
        mask_value = torch.tensor(0, device=label.device)
        if label.min() < 1:
            mask_value = label.min()
        return mask_value

    def _progress(self):
        return min(1.0, float(self._iter_cnt) / float(self._total_train_iters))

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader["train_loader"].shuffle()

        for X, label in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
            outputs = self.model(X, label, progress=self._progress(), training=True)
            pred = outputs["prediction"]

            pred, label = self._inverse_transform([pred, label])
            mask_value = self._compute_mask_value(label)
            if self._iter_cnt == 0:
                print("Check mask value", mask_value)

            pred_loss = self._loss_fn(pred, label, mask_value)
            loss = (
                pred_loss
                + self._variant_loss_weight * outputs["variant_loss"]
                + self._invariant_loss_weight * outputs["invariant_loss"]
                + self._mi_loss_weight * outputs["mi_loss"]
            )

            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            self._iter_cnt += 1

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def evaluate(self, mode):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label, progress=1.0, training=False)["prediction"]
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        if mode == "test":
            test_mae = []
            test_mape = []
            test_rmse = []
            print("Check mask value", mask_value)
            for horizon in range(self.model.horizon):
                res = compute_all_metrics(preds[:, horizon, :], labels[:, horizon, :], mask_value)
                log = "Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                self._logger.info(log.format(horizon + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            avg_mae, avg_rmse, avg_mape = self._log_test_metrics(
                test_mae, test_rmse, test_mape
            )
            log = "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            self._logger.info(log.format(avg_mae, avg_rmse, avg_mape))
