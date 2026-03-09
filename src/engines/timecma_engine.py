import numpy as np
import torch
from datetime import datetime
from datetime import timedelta

from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse


class TimeCMA_Engine(BaseEngine):
    def __init__(
        self,
        generate_embeddings_on_the_fly=False,
        embedding_method="gpt2",
        prompt_start_datetime="",
        prompt_freq_minutes=5,
        **args
    ):
        super(TimeCMA_Engine, self).__init__(**args)
        self._generate_embeddings_on_the_fly = bool(generate_embeddings_on_the_fly)
        self._embedding_method = embedding_method
        self._prompt_freq_minutes = int(prompt_freq_minutes)
        self._prompt_start_datetime = self._parse_prompt_start_datetime(prompt_start_datetime)
        self._x_offsets = np.arange(-(self.model.seq_len - 1), 1, 1, dtype=np.int64)

    def _parse_prompt_start_datetime(self, start_datetime):
        value = str(start_datetime).strip()
        if not value:
            return datetime(2019, 1, 1, 0, 0, 0)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(value, fmt)
                if fmt == "%Y-%m-%d":
                    return parsed.replace(hour=0, minute=0, second=0)
                return parsed
            except ValueError:
                continue
        raise ValueError(
            "Invalid prompt_start_datetime: {}. Use one of YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, YYYY-MM-DDTHH:MM:SS".format(
                value
            )
        )

    def _compute_mask_value(self, label):
        mask_value = torch.tensor(0, device=label.device)
        if label.min() < 1:
            mask_value = label.min()
        return mask_value

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            return batch[0], batch[1], batch[2], batch[3]
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            return batch[0], batch[1], batch[2], None
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1], None, None
        raise ValueError("Unexpected batch format in TimeCMA_Engine")

    def _build_prompt_marks(self, idx_ind):
        if idx_ind is None:
            return None
        idx_ind = np.asarray(idx_ind).reshape(-1)
        history_index = idx_ind[:, None] + self._x_offsets[None, :]

        marks = np.zeros((history_index.shape[0], history_index.shape[1], 6), dtype=np.int64)
        for i in range(history_index.shape[0]):
            for t in range(history_index.shape[1]):
                dt = self._prompt_start_datetime + timedelta(
                    minutes=self._prompt_freq_minutes * int(history_index[i, t])
                )
                marks[i, t, 0] = dt.year
                marks[i, t, 1] = dt.month
                marks[i, t, 2] = dt.day
                marks[i, t, 3] = dt.weekday()
                marks[i, t, 4] = dt.hour
                marks[i, t, 5] = dt.minute
        return self._to_device(self._to_tensor(marks))

    def _prepare_embeddings(self, embeddings, x_tensor, idx_ind=None):
        if embeddings is not None:
            return self._to_device(self._to_tensor(embeddings))

        if not self._generate_embeddings_on_the_fly:
            return None

        input_mark = self._build_prompt_marks(idx_ind) if self._embedding_method == "gpt2" else None

        if self._embedding_method == "gpt2":
            with torch.no_grad():
                return self.model.generate_prompt_embeddings(
                    x_tensor[..., : self.model.ts_dim],
                    input_mark=input_mark,
                    method=self._embedding_method,
                )
        return self.model.generate_prompt_embeddings(
            x_tensor[..., : self.model.ts_dim],
            input_mark=input_mark,
            method=self._embedding_method,
        )

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader["train_loader"].shuffle()

        for batch in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()

            X, label, embeddings, idx_ind = self._unpack_batch(batch)
            X, label = self._to_device(self._to_tensor([X, label]))
            embedding_tensor = self._prepare_embeddings(embeddings, X, idx_ind=idx_ind)

            pred = self.model(X, label, embeddings=embedding_tensor)
            pred, label = self._inverse_transform([pred, label])

            mask_value = self._compute_mask_value(label)
            if self._iter_cnt == 0:
                print("Check mask value", mask_value)

            loss = self._loss_fn(pred, label, mask_value)
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
            for batch in self._dataloader[mode + "_loader"].get_iterator():
                X, label, embeddings, idx_ind = self._unpack_batch(batch)
                X, label = self._to_device(self._to_tensor([X, label]))

                if embeddings is not None:
                    embedding_tensor = self._to_device(self._to_tensor(embeddings))
                elif self._generate_embeddings_on_the_fly:
                    input_mark = (
                        self._build_prompt_marks(idx_ind)
                        if self._embedding_method == "gpt2"
                        else None
                    )
                    embedding_tensor = self.model.generate_prompt_embeddings(
                        X[..., : self.model.ts_dim],
                        input_mark=input_mark,
                        method=self._embedding_method,
                    )
                else:
                    embedding_tensor = None

                pred = self.model(X, label, embeddings=embedding_tensor)
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

            log = "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
