import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


def compute_metrics(targets, predictions, top_k: int = 100) -> dict:
    """计算 MSE、相关系数及 Top-k 指标

    targets 与 predictions 为顺序一致的一维数组
    top_k 为筛选数量，返回命中率及所选变体的真实 fitness 均值
    """
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    if (
        targets.ndim != 1
        or predictions.shape != targets.shape
        or targets.size == 0
    ):
        raise ValueError("标签和预测必须是等长、非空的一维数组")

    if (
        not np.isfinite(targets).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("标签或预测包含 NaN 或无穷值")

    if top_k < 1:
        raise ValueError("top_k 必须大于 0")

    # 常数数组的相关系数记为 NaN
    valid_correlation = (
        len(targets) > 1
        and np.ptp(targets) > 0
        and np.ptp(predictions) > 0
    )

    if valid_correlation:
        spearman = float(spearmanr(targets, predictions).statistic)
        pearson = float(pearsonr(targets, predictions).statistic)
    else:
        spearman = float("nan")
        pearson = float("nan")

    k = min(top_k, len(targets))

    # 同分时保留输入顺序
    predicted_top = np.argsort(
        -predictions, kind="stable"
    )[:k]
    true_top = np.argsort(
        -targets, kind="stable"
    )[:k]

    # 命中数为预测与真实 Top-k 的交集大小
    hits = np.intersect1d(predicted_top, true_top).size

    return {
        "mse": float(mean_squared_error(targets, predictions)),
        "spearman": spearman,
        "pearson": pearson,
        "top_k": k,
        "top_k_hit": float(hits / k),
        # 预测 Top-k 对应的真实 fitness 均值
        "top_k_mean_fitness": float(targets[predicted_top].mean()),
    }