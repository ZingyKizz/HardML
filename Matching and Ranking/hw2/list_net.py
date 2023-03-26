import numpy as np
import torch
from torch import Tensor
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


def _compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    if gain_scheme == "const":
        return y_value
    elif gain_scheme == "exp2":
        return 2 ** y_value - 1


def _dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str, k: int) -> float:
    # допишите ваш код здесь
    indices = torch.argsort(ys_pred, descending=True)[:k]
    sorted_by_pred = ys_true[indices]
    gain = _compute_gain(sorted_by_pred, gain_scheme)
    discount = np.log2(np.arange(2, len(gain) + 2))
    return (gain / discount).sum().item()


def _ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = "const", k: int = None) -> float:
    # допишите ваш код здесь
    sorted_ys_true, _ = torch.sort(ys_true, descending=True)[:k]
    res = _dcg(ys_true, ys_pred, gain_scheme, k)
    ideal_res = _dcg(sorted_ys_true, sorted_ys_true, gain_scheme, k)
    if ideal_res == 0:
        return 0
    return res / ideal_res


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        # допишите ваш код здесь
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)
        self.X_train = torch.FloatTensor(
            self._scale_features_in_query_groups(X_train, self.query_ids_train)
        )
        self.X_test = torch.FloatTensor(
            self._scale_features_in_query_groups(X_test, self.query_ids_test)
        )

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        # допишите ваш код здесь
        scaler = StandardScaler()
        res = np.full_like(inp_feat_array, fill_value=np.NaN)
        for query_id in np.unique(inp_query_ids):
            query_mask = (inp_query_ids == query_id)
            res[query_mask] = scaler.fit_transform(inp_feat_array[query_mask])
        return res

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        # допишите ваш код здесь
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        # допишите ваш код здесь
        ndcg_collection = []
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            ndcg = self._eval_test_set()
            ndcg_collection.append(ndcg)
            print(f"Epoch: {epoch}, NDCG: {ndcg}")
        return ndcg_collection

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        q = torch.nn.functional.softmax(batch_pred)
        p = torch.nn.functional.softmax(batch_ys)
        return -torch.sum(p * torch.log(q))

    def _train_one_epoch(self) -> None:
        self.model.train()
        # допишите ваш код здесь
        batches = self._iter_batches(self.X_train, self.ys_train, self.query_ids_train, shuffle=True)
        for X_batch, y_batch in batches:
            self.optimizer.zero_grad()
            y_pred_batch = self.model(X_batch)
            loss = self._calc_loss(y_batch, y_pred_batch.view(-1))
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            # допишите ваш код здесь
            batches = self._iter_batches(self.X_test, self.ys_test, self.query_ids_test)
            for X_batch, y_batch in batches:
                y_pred_batch = self.model(X_batch)
                ndcgs.append(
                    self._ndcg_k(y_batch, y_pred_batch.view(-1), ndcg_top_k=10)
                )
            return np.mean(ndcgs)

    def _iter_batches(self, X, y, queries, shuffle=False):
        qs = np.unique(queries)
        if shuffle:
            np.random.shuffle(qs)
        for q in qs:
            mask = (queries == q)
            yield X[mask], y[mask]

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # допишите ваш код здесь
        return _ndcg(ys_true, ys_pred, gain_scheme="exp2", k=ndcg_top_k)


if __name__ == "__main__":
    solution = Solution()
    solution.fit()
