from torch import Tensor
import torch
import numpy as np


def sort_by_pred(ys_true: Tensor, ys_pred: Tensor) -> Tensor:
    indices = torch.argsort(ys_pred, descending=True)
    return ys_true[indices]


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # допишите ваш код здесь
    sorted_by_pred = sort_by_pred(ys_true, ys_pred)
    res = 0
    for i in range(len(sorted_by_pred)):
        for j in range(i + 1, len(sorted_by_pred)):
            if sorted_by_pred[i] < sorted_by_pred[j]:
                res += 1
    return res


def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    if gain_scheme == "const":
        return y_value
    elif gain_scheme == "exp2":
        return 2 ** y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # допишите ваш код здесь
    sorted_by_pred = sort_by_pred(ys_true, ys_pred)
    gain = compute_gain(sorted_by_pred, gain_scheme)
    discount = np.log2(np.arange(2, len(gain) + 2))
    return (gain / discount).sum()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    return dcg(ys_true, ys_pred, gain_scheme) / dcg(ys_true, ys_true, gain_scheme)


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    sorted_by_pred = sort_by_pred(ys_true, ys_pred)
    return sorted_by_pred[:k].sum() / min(k, sorted_by_pred.sum())


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    sorted_by_pred = sort_by_pred(ys_true, ys_pred).numpy().tolist()
    return 1 / (sorted_by_pred.index(1) + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    # допишите ваш код здесь
    p_rel = sort_by_pred(ys_true, ys_pred)
    p_look = torch.empty_like(ys_true)
    p_look[0] = 1
    for i in range(1, len(ys_true)):
        p_look[i] = p_look[i - 1] * (1 - p_rel[i - 1]) * (1 - p_break)
    return (p_look * p_rel).sum().item()


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    sorted_by_pred = sort_by_pred(ys_true, ys_pred)
    indices = (sorted_by_pred == 1).nonzero()
    precisions = []
    for i in indices:
        k = i + 1
        precision_at_k = sorted_by_pred[:k].sum() / k
        precisions.append(precision_at_k)
    return np.mean(precisions) if precisions else -1
