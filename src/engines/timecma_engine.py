import numpy as np
import torch

from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse


class TimeCMA_Engine(BaseEngine):
    def __init__(self, generate_embeddings_on_the_fly=False, embedding_method="gpt2", **args):
        super(TimeCMA_Engine, self).__init__(**args)
        self._generate_embeddings_on_the_fly = bool(generate_embeddings_on_the_fly)
        self._embedding_method = embedding_method

    def _compute_mask_value(self, label):
        mask_value = torch.tensor(0, device=label.device)
        if label.min() < 1:
            mask_value = label.min()
        return mask_value

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1], None
        raise ValueError("Unexpected batch format in TimeCMA_Engine")

    def _prepare_embeddings(self, embeddings, x_tensor):
        if embeddings is not None:
            return self._to_device(self._to_tensor(embeddings))

        if not self._generate_embeddings_on_the_fly:
            return None

        if self._embedding_method == "gpt2":
            with torch.no_grad():
                return self.model.generate_prompt_embeddings(
                    x_tensor[..., : self.model.ts_dim],
                    method=self._embedding_method,
                )
        return self.model.generate_prompt_embeddings(
            x_tensor[..., : self.model.ts_dim],
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

            X, label, embeddings = self._unpack_batch(batch)
            X, label = self._to_device(self._to_tensor([X, label]))
            embedding_tensor = self._prepare_embeddings(embeddings, X)

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
                X, label, embeddings = self._unpack_batch(batch)
                X, label = self._to_device(self._to_tensor([X, label]))

                if embeddings is not None:
                    embedding_tensor = self._to_device(self._to_tensor(embeddings))
                elif self._generate_embeddings_on_the_fly:
                    embedding_tensor = self.model.generate_prompt_embeddings(
                        X[..., : self.model.ts_dim],
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
