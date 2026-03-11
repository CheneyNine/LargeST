import numpy as np
import torch
import torch.nn.functional as F

from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse


class CALFEngine(BaseEngine):
    def __init__(
        self,
        feature_loss="smooth_l1",
        output_loss="smooth_l1",
        task_loss="smooth_l1",
        feature_w=0.01,
        output_w=1.0,
        task_w=1.0,
        **args
    ):
        super(CALFEngine, self).__init__(**args)
        self._feature_loss_name = str(feature_loss)
        self._output_loss_name = str(output_loss)
        self._task_loss_name = str(task_loss)
        self._feature_w = float(feature_w)
        self._output_w = float(output_w)
        self._task_w = float(task_w)

    def _compute_mask_value(self, label):
        mask_value = torch.tensor(0, device=label.device)
        if label.min() < 1:
            mask_value = label.min()
        return mask_value

    def _apply_mask(self, a, b, mask_value):
        if torch.isnan(mask_value):
            mask = ~torch.isnan(b)
        else:
            mask = (b != mask_value)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        return a * mask, b * mask

    def _elementwise_loss(self, pred, target, loss_name, mask_value):
        pred, target = self._apply_mask(pred, target, mask_value)
        if loss_name == "l1":
            loss = torch.abs(pred - target)
        elif loss_name == "smooth_l1":
            loss = F.smooth_l1_loss(pred, target, reduction="none")
        elif loss_name == "mse":
            loss = (pred - target) ** 2
        else:
            raise ValueError("Unsupported CALF loss: {}".format(loss_name))
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def _feature_loss(self, inter_time, inter_text):
        losses = []
        for idx, (feat_time, feat_text) in enumerate(zip(inter_time[::-1], inter_text[::-1])):
            if self._feature_loss_name == "l1":
                cur = F.l1_loss(feat_time, feat_text)
            elif self._feature_loss_name == "smooth_l1":
                cur = F.smooth_l1_loss(feat_time, feat_text)
            elif self._feature_loss_name == "mse":
                cur = F.mse_loss(feat_time, feat_text)
            else:
                raise ValueError("Unsupported CALF feature loss: {}".format(self._feature_loss_name))
            losses.append((0.8 ** idx) * cur)
        if not losses:
            return torch.tensor(0.0, device=self._device)
        return sum(losses)

    def _compute_total_loss(self, outputs, label):
        pred_time = outputs["outputs_time"]
        pred_text = outputs["outputs_text"]
        inter_time = outputs["intermidiate_time"]
        inter_text = outputs["intermidiate_text"]

        pred_time_denorm, pred_text_denorm, label_denorm = self._inverse_transform(
            [pred_time, pred_text, label]
        )
        mask_value = self._compute_mask_value(label_denorm)

        task_loss = self._elementwise_loss(
            pred_time_denorm, label_denorm, self._task_loss_name, mask_value
        )
        output_loss = self._elementwise_loss(
            pred_time_denorm, pred_text_denorm, self._output_loss_name, mask_value
        )
        feature_loss = self._feature_loss(inter_time, inter_text)

        total_loss = (
            self._task_w * task_loss
            + self._output_w * output_loss
            + self._feature_w * feature_loss
        )
        return total_loss, pred_time_denorm, label_denorm, mask_value

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader["train_loader"].shuffle()

        for X, label in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))

            outputs = self.model(X, label)
            loss, pred_time, label, mask_value = self._compute_total_loss(outputs, label)

            mape = masked_mape(pred_time, label, mask_value).item()
            rmse = masked_rmse(pred_time, label, mask_value).item()

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
                outputs = self.model(X, label)
                pred_time = outputs["outputs_time"]
                pred_time, label = self._inverse_transform([pred_time, label])

                preds.append(pred_time.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":
            loss = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return loss, mape, rmse

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
