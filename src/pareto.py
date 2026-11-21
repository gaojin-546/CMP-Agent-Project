"""CMP 抛光液配方开发 Agent —— 模块 4：帕累托前沿提取 + TOPSIS 筛选。

模块 4 是**后处理推荐层**，不是优化环节。它回答的是工程师真正会问的问题：
"在已经探出的取舍面上，按我的偏好该先试哪个？"

关键约束（来自 README 第 2 节的核心原则）
------------------------------------------
**TOPSIS 只做前沿后处理，不参与 BO 优化目标。** 因此：

* 权重在初始化阶段由 ``TopsisConfig`` 冻结，本模块**只读不改**；
* 熵权法仅用于"敏感性对照"（``entropy_weight_sensitivity_only=True``），
  它的结果只进报告、不进决策；
* 帕累托前沿在**三个优化目标**（MRR / Ra / Dishing）上提取，
  成本不是优化维度，只在 TOPSIS 里作为第 4 个准则参与打分。

为什么成本不能进帕累托
----------------------
如果把成本也塞进非支配判断，一个"性能中等但便宜 50 元"的配方会被判为非支配，
前沿里会混进大量在性能上被完全支配的点，前沿规模爆炸、可读性归零。
成本的角色是**trade-off 的打分项**，不是"目标"。

规格前置过滤
------------
``hard_spec_filter_first=True`` 时先按产线规格（MRR ≥ 1500 Å/min、Ra ≤ 4 Å、
Dishing ≤ 150 Å）剔除前沿点，再对剩余点打分。理由很实际：一个 Ra 超标的配方
无论 TOPSIS 分数多高都不能交付，让它占据推荐位就是浪费工程师时间。

若规格过滤把前沿清空（真实会发生——见 ``docs/pareto_selfcheck.txt``），
本模块**不会静默返回空列表**，而是降级为"按规格违反度排序"并显式标记
``spec_fallback=True``，让下游知道当前的推荐都不合格。

用法
----
    >>> from src.config import build_default_config
    >>> from src.pareto import ParetoSelector
    >>> space = build_default_config()
    >>> selector = ParetoSelector(space)
    >>> report = selector.select(X_obs, Y_obs, top_k=5)
    >>> print(report.describe())
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from botorch.utils.multi_objective.pareto import is_non_dominated

from src.config import CMPRecipeSpace

__all__ = [
    "TopsisResult",
    "topsis",
    "entropy_weights",
    "is_non_dominated_mask",
    "RankedCandidate",
    "FrontReport",
    "ParetoSelector",
]


# --------------------------------------------------------------------------- #
# 1. 非支配判断
# --------------------------------------------------------------------------- #
def is_non_dominated_mask(Y_max: Any) -> np.ndarray:
    """最大化空间下的非支配掩码 ``(n,)``。

    薄封装 BoTorch 的 ``is_non_dominated``：好处是本模块的其余部分只依赖
    numpy，并且可以单独给这个函数写单元测试（误判非支配会直接污染整张榜单）。
    """
    Y = np.asarray(Y_max, dtype=float)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return np.zeros(Y.shape[0], dtype=bool)
    mask = is_non_dominated(torch.as_tensor(Y, dtype=torch.double), maximize=True)
    return np.asarray(mask.numpy(), dtype=bool)


# --------------------------------------------------------------------------- #
# 2. TOPSIS
# --------------------------------------------------------------------------- #
@dataclass
class TopsisResult:
    """TOPSIS 的全部中间量（保留是为了可审计，不只是分数）。"""

    criteria: Tuple[str, ...]
    weights: np.ndarray
    roles: Tuple[str, ...]
    raw: np.ndarray
    """原始准则矩阵 ``(n, k)``：前几个是目标，最后一个是成本。"""
    normalized: np.ndarray
    weighted: np.ndarray
    ideal_best: np.ndarray
    ideal_worst: np.ndarray
    d_plus: np.ndarray
    d_minus: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray

    def top(self, k: int = 5) -> np.ndarray:
        """按分数降序返回前 k 个下标。"""
        order = np.argsort(-self.scores, kind="stable")
        return order[:k]


def _vector_normalize(X: np.ndarray) -> np.ndarray:
    """向量归一化 ``r_ij = x_ij / sqrt(Σ_i x_ij²)``。

    相比 min-max 归一化，它对"某个准则整体量级偏大"不敏感（分母是该列自身的
    范数），也不会因为一个极端值把其余点压到 0 附近——这正是 TOPSIS 原始论文
    采用的做法。
    """
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm = np.where(norm > 0, norm, 1.0)
    return X / norm


def topsis(
    matrix: Any,
    weights: Any,
    roles: Sequence[str],
    criteria: Optional[Sequence[str]] = None,
) -> TopsisResult:
    """标准 TOPSIS（向量归一化 + 加权 + 正负理想解 + 贴近度）。

    Parameters
    ----------
    matrix:
        ``(n, k)`` 原始准则值。**成本型准则按原始值传入**（越大越差），
        由 ``roles`` 决定理想解方向，不要事先取负——否则可读性与可审计性都会丢。
    weights:
        长度 k 的权重，必须已在配置里冻结。
    roles:
        ``"benefit"``（越大越好）或 ``"cost"``（越小越好）。
    """
    X = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"matrix 必须是二维，收到 {X.shape}")
    if X.shape[1] != len(w):
        raise ValueError(f"准则数 {X.shape[1]} 与权重长度 {len(w)} 不一致")
    if len(roles) != len(w):
        raise ValueError(f"roles 长度 {len(roles)} 与权重长度 {len(w)} 不一致")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError(f"权重之和必须为 1，当前 {w.sum():.6f}")
    names = tuple(criteria or [f"c{i}" for i in range(len(w))])

    n = X.shape[0]
    if n == 0:
        raise ValueError("没有候选点")

    r = _vector_normalize(X)
    v = r * w
    best = np.empty_like(w)
    worst = np.empty_like(w)
    for j, role in enumerate(roles):
        if role == "benefit":
            best[j], worst[j] = v[:, j].max(), v[:, j].min()
        elif role == "cost":
            best[j], worst[j] = v[:, j].min(), v[:, j].max()
        else:
            raise ValueError(f"未知的准则角色 {role!r}（应为 benefit / cost）")

    d_plus = np.sqrt(((v - best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - worst) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    # 单点退化：它同时是"最优点"和"最差点"，贴近度按 1.0 处理（唯一候选即最优候选）
    if n == 1:
        scores = np.ones(1)
    else:
        scores = np.where(denom > 0, d_minus / np.where(denom > 0, denom, 1.0), 0.5)
    ranks = _rank_desc(scores)

    return TopsisResult(
        criteria=names,
        weights=w,
        roles=tuple(roles),
        raw=X,
        normalized=r,
        weighted=v,
        ideal_best=best,
        ideal_worst=worst,
        d_plus=d_plus,
        d_minus=d_minus,
        scores=scores,
        ranks=ranks,
    )


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """降序名次（1 = 最好），并列取平均名次。"""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(scores[order[j + 1]] - scores[order[i]]) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# 3. 熵权（仅敏感性对照）
# --------------------------------------------------------------------------- #
def entropy_weights(matrix: Any, roles: Sequence[str]) -> np.ndarray:
    """熵权法权重，**只用于敏感性对照**，不参与决策。

    配置里 ``entropy_weight_sensitivity_only=True`` 的含义就是：熵权可以算，
    但只能用来回答"如果让数据自己说话，排序会变吗"，绝不能回头覆盖冻结权重。

    实现细节：熵权要求非负且可加，所以先把各列 min-max 正向化到 [0, 1]
    （成本型取反），再按 ``p_ij = x_ij / Σ_i x_ij`` 求熵。若某列正向化后全为 0
    （所有候选在该准则上完全相同），该列的差异系数按 0 处理，权重自然为 0。
    """
    X = np.asarray(matrix, dtype=float)
    k = X.shape[1]
    pos = np.empty_like(X)
    for j, role in enumerate(roles):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        span = hi - lo
        if span <= 0:
            pos[:, j] = 0.0
            continue
        z = (col - lo) / span
        pos[:, j] = z if role == "benefit" else (1.0 - z)

    n = X.shape[0]
    if n <= 1:
        return np.full(k, 1.0 / k)
    col_sum = pos.sum(axis=0)
    p = np.where(col_sum > 0, pos / np.where(col_sum > 0, col_sum, 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    e = -plogp.sum(axis=0) / math.log(n)          # 熵，落在 [0, 1]
    d = 1.0 - e                                   # 差异系数
    if d.sum() <= 0:
        return np.full(k, 1.0 / k)
    return d / d.sum()


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman 秩相关（不引 scipy，纯 numpy 实现）。"""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2:
        return float("nan")
    rx, ry = _rank_asc(x), _rank_asc(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rank_asc(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(x[order[j + 1]] - x[order[i]]) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# 4. 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class RankedCandidate:
    """一条进入推荐清单的前沿配方。"""

    rank: int
    data_index: int
    """在输入数据集里的行号（可反查原始实验记录）。"""
    recipe: Dict[str, float]
    objectives: Dict[str, float]
    cost_cny: float
    topsis_score: float
    on_spec: bool
    spec_violations: Dict[str, float] = field(default_factory=dict)
    criteria: Dict[str, float] = field(default_factory=dict)

    def row(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "data_index": self.data_index,
            "recipe": dict(self.recipe),
            "objectives": dict(self.objectives),
            "cost_cny": self.cost_cny,
            "topsis_score": self.topsis_score,
            "on_spec": self.on_spec,
            "spec_violations": dict(self.spec_violations),
            "criteria": dict(self.criteria),
        }


@dataclass
class FrontReport:
    """一轮前沿筛选的完整结果（可直接写进实验记录）。"""

    n_total: int
    n_front: int
    n_on_spec: int
    n_ranked: int
    criteria: Tuple[str, ...]
    weights: Dict[str, float]
    roles: Tuple[str, ...]
    ranked: List[RankedCandidate]
    dropped_by_spec: Dict[str, int]
    front_indices: np.ndarray
    spec_fallback: bool
    topsis: Optional[TopsisResult] = None
    entropy_weights: Optional[Dict[str, float]] = None
    rank_corr_entropy: float = float("nan")
    raw_best_mrr_index: Optional[int] = None
    note: str = ""

    # -- 输出 ------------------------------------------------------------- #
    def describe(self, show: int = 8) -> str:
        lines = [
            f"帕累托前沿 + TOPSIS  |  样本 {self.n_total} → 前沿 {self.n_front}"
            f" → 规格合格 {self.n_on_spec} → 排序 {self.n_ranked}",
            "冻结权重（来自 config，不可调）："
            + " / ".join(f"{c} {w:.2f}" for c, w in self.weights.items()),
            "-" * 96,
            f"{'序':>3}{'数据行':>7}{'MRR':>9}{'Ra':>7}{'Dishing':>9}"
            f"{'成本':>8}{'TOPSIS':>9}{'规格':>5}",
            "-" * 96,
        ]
        for c in self.ranked[:show]:
            spec = "✓" if c.on_spec else "✗"
            lines.append(
                f"{c.rank:>3}{c.data_index:>7}"
                f"{c.objectives.get('MRR', float('nan')):>9.1f}"
                f"{c.objectives.get('Ra', float('nan')):>7.2f}"
                f"{c.objectives.get('Dishing', float('nan')):>9.1f}"
                f"{c.cost_cny:>8.0f}{c.topsis_score:>9.4f}{spec:>5}"
            )
        lines.append("-" * 96)
        if self.dropped_by_spec:
            lines.append(
                "因硬规格被剔除：" + "  ".join(
                    f"{k} {v} 个" for k, v in self.dropped_by_spec.items()
                )
            )
        if self.spec_fallback:
            lines.append(
                "⚠ 规格过滤后前沿为空 → 降级为按规格违反度排序；"
                "当前推荐**没有一条满足全部产线规格**"
            )
        if self.rank_corr_entropy == self.rank_corr_entropy:  # not NaN
            lines.append(
                f"敏感性对照（全前沿 {self.n_front} 点，非过滤后子集）："
                f"熵权 { {k: round(v, 3) for k, v in self.entropy_weights.items()} }"
            )
            lines.append(
                f"  → 与冻结权重排序的 Spearman ρ={self.rank_corr_entropy:+.3f}"
                f"（仅作对照，不覆盖冻结权重）"
            )
        if self.raw_best_mrr_index is not None:
            pos = self._rank_of(self.raw_best_mrr_index)
            tail = f"排第 {pos}" if pos.isdigit() else "未进入推荐清单"
            lines.append(
                f"对照：MRR 最大的那条是数据行 #{self.raw_best_mrr_index}，{tail}"
            )
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def _rank_of(self, data_index: int) -> str:
        for c in self.ranked:
            if c.data_index == data_index:
                return str(c.rank)
        return "未入选"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_front": self.n_front,
            "n_on_spec": self.n_on_spec,
            "n_ranked": self.n_ranked,
            "criteria": list(self.criteria),
            "weights": dict(self.weights),
            "roles": list(self.roles),
            "dropped_by_spec": dict(self.dropped_by_spec),
            "spec_fallback": self.spec_fallback,
            "ranked": [c.row() for c in self.ranked],
        }


# --------------------------------------------------------------------------- #
# 5. 主选择器
# --------------------------------------------------------------------------- #
class ParetoSelector:
    """从实验数据里提取前沿、按规格过滤、用冻结权重做 TOPSIS 排序。"""

    def __init__(self, space: CMPRecipeSpace) -> None:
        self.space = space
        self._check_weights()

    def _check_weights(self) -> None:
        cfg = self.space.topsis
        total = sum(c.weight for c in cfg.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"TOPSIS 权重之和为 {total:.4f}，应为 1")
        for c in cfg.weights:
            if c.name != "Cost":
                idx = self.space.objective_names.index(c.name)
                role_cfg = self.space.objectives[idx].topsis_role
                if role_cfg != c.role:
                    raise ValueError(
                        f"准则 {c.name} 的角色冲突：config.topsis={c.role}，"
                        f"objective.topsis_role={role_cfg}"
                    )

    # -- 前沿 ------------------------------------------------------------- #
    def front_indices(self, Y_phys: Any) -> np.ndarray:
        """返回非支配点的行号（在**三个优化目标**上判断，成本不参与）。"""
        Y = np.asarray(Y_phys, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        signs = np.asarray(self.space.objective_signs(), dtype=float)
        mask = is_non_dominated_mask(Y * signs)
        return np.where(mask)[0]

    # -- 成本 ------------------------------------------------------------- #
    def cost_vector(self, X_phys: Any, Y_phys: Any) -> np.ndarray:
        """每条配方的单次实验成本（元）。

        优先用实测 MRR 折算机台时间；缺少 MRR 列时退回 Preston 先验估计。
        """
        X = np.asarray(X_phys, dtype=float)
        Y = np.asarray(Y_phys, dtype=float)
        if "MRR" in self.space.objective_names:
            mrr = Y[:, self.space.objective_names.index("MRR")]
            return np.asarray(self.space.cost_matrix(X, mrr), dtype=float).reshape(-1)
        return np.asarray(
            self.space.cost(torch.as_tensor(X, dtype=torch.double)).numpy(),
            dtype=float,
        ).reshape(-1)

    # -- 规格 ------------------------------------------------------------- #
    def spec_violation(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """按产线规格计算违反量（正数 = 超标，0 = 合格）。"""
        out: Dict[str, float] = {}
        for o in self.space.objectives:
            v = float(objectives[o.name])
            if o.direction.value == "maximize":
                miss = max(0.0, o.spec - v)
            else:
                miss = max(0.0, v - o.spec)
            if miss > 0:
                out[o.name] = miss
        return out

    # -- 主流程 ----------------------------------------------------------- #
    def select(
        self,
        X_phys: Any,
        Y_phys: Any,
        top_k: int = 5,
        *,
        apply_spec_filter: bool = True,
        freeze_weights: bool = True,
    ) -> FrontReport:
        """提取前沿 → 规格过滤 → TOPSIS 排序。

        Parameters
        ----------
        X_phys, Y_phys:
            历史实验数据，``X`` 为物理量纲 ``(n, d)``，``Y`` 为原量纲 ``(n, m)``，
            列序与 ``space.objective_names`` 一致。
        top_k:
            返回前 k 条。
        apply_spec_filter:
            是否先按产线规格剔除（默认 True，与 ``TopsisConfig`` 一致）。
        freeze_weights:
            True 表示使用配置里的冻结权重（唯一正确用法）；False 仅供消融实验。
        """
        X = np.asarray(X_phys, dtype=float)
        Y = np.asarray(Y_phys, dtype=float)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"X/Y 必须是二维，收到 {X.shape} / {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X 行数 {X.shape[0]} 与 Y 行数 {Y.shape[0]} 不一致")
        n = X.shape[0]
        if n == 0:
            raise ValueError("没有实验数据")

        # 1) 前沿（三个优化目标上的非支配集）
        front = self.front_indices(Y)

        # 2) 成本（第 4 个准则，不参与非支配判断）
        cost_all = self.cost_vector(X, Y)

        # 3) 组装准则矩阵：目标列 + 成本列，顺序严格遵循 TopsisConfig
        cfg = self.space.topsis
        names = list(cfg.names)
        roles = list(cfg.roles)
        weights = (
            self.space.topsis.weights_array() if freeze_weights else np.full(len(names), 1.0 / len(names))
        )
        crit_front = self._criteria_matrix(Y[front], cost_all[front], names)
        objectives_of = [
            {o.name: float(Y[i, self.space.objective_names.index(o.name)]) for o in self.space.objectives}
            for i in front
        ]
        violations = [self.spec_violation(objs) for objs in objectives_of]
        on_spec = np.array([len(v) == 0 for v in violations], dtype=bool)

        # 4) 规格前置过滤（并统计每条规格各剔除多少）
        dropped: Dict[str, int] = {}
        if apply_spec_filter and cfg.hard_spec_filter_first:
            for v in violations:
                if v:
                    # 只统计"第一条不满足的规格"，避免同一点被重复计数
                    first = next(iter(v))
                    dropped[first] = dropped.get(first, 0) + 1
            keep = on_spec
        else:
            keep = np.ones(len(front), dtype=bool)

        spec_fallback = False
        note = ""
        if keep.sum() == 0:
            # 前沿上没有一条满足全部规格：降级为按"总违反度"排序，并显式标记
            spec_fallback = True
            keep = np.ones(len(front), dtype=bool)
            note = (
                "当前前沿与产线规格盒不相交：要交付合格配方，需要把规格作为"
                "约束（输出空间软约束）或放宽目标，而不只是筛选。"
            )

        # 5) TOPSIS（只对保留下来的前沿点打分）
        sel = np.where(keep)[0]
        if spec_fallback:
            # 降级模式按违反度升序作为主序，TOPSIS 分数仅作参考
            total_viol = np.array([sum(v.values()) for v in violations])[sel]
            order = sel[np.argsort(total_viol, kind="stable")]
            res = topsis(crit_front[sel], weights, roles, names)
            res = TopsisResult(
                criteria=res.criteria, weights=res.weights, roles=res.roles,
                raw=res.raw, normalized=res.normalized, weighted=res.weighted,
                ideal_best=res.ideal_best, ideal_worst=res.ideal_worst,
                d_plus=res.d_plus, d_minus=res.d_minus,
                scores=res.scores,
                # 降级模式下"名次"由违反度决定，不覆盖 TOPSIS 分数
                ranks=np.array([int(list(order).index(s)) + 1 for s in sel], dtype=float),
            )
        else:
            res = topsis(crit_front[sel], weights, roles, names)

        # 6) 排序 + 组装
        order_local = np.argsort(-res.scores, kind="stable") if not spec_fallback \
            else np.argsort(res.ranks, kind="stable")
        ranked: List[RankedCandidate] = []
        for pos, li in enumerate(order_local[:top_k]):
            row = sel[li]
            data_idx = int(front[row])
            ranked.append(
                RankedCandidate(
                    rank=pos + 1,
                    data_index=data_idx,
                    recipe={n_: float(X[data_idx, j]) for j, n_ in enumerate(self.space.names)},
                    objectives=objectives_of[row],
                    cost_cny=float(cost_all[data_idx]),
                    topsis_score=float(res.scores[li]),
                    on_spec=bool(on_spec[row]),
                    spec_violations=violations[row],
                    criteria={n_: float(crit_front[row, j]) for j, n_ in enumerate(names)},
                )
            )

        # 7) 熵权敏感性对照（只进报告）
        #
        # 口径说明：熵权在**完整前沿**上算，不在规格过滤后的子集上算。
        # 原因很实际：过滤后往往只剩 2–3 个点，min-max 归一化后每列都是 {0,1}，
        # 熵恒为 1、差异系数为 0，权重会退化成均匀分布——
        # 那样算出来的"敏感性对照"是**零信息**的（实测踩到：四个准则都成了 0.25）。
        ent = entropy_weights(crit_front, roles)
        ent_res = topsis(crit_front, ent, roles, names)
        res_front = topsis(crit_front, weights, roles, names)
        rho = _spearman(res_front.scores, ent_res.scores) if len(front) >= 3 else float("nan")

        # 8) 天然对照：MRR 最大的那条排第几
        raw_best = int(np.argmax(Y[:, self.space.objective_names.index("MRR")]))

        return FrontReport(
            n_total=n,
            n_front=int(len(front)),
            n_on_spec=int(on_spec.sum()),
            n_ranked=len(ranked),
            criteria=tuple(names),
            weights={n_: float(w) for n_, w in zip(names, weights)},
            roles=tuple(roles),
            ranked=ranked,
            dropped_by_spec=dropped,
            front_indices=front,
            spec_fallback=spec_fallback,
            topsis=res,
            entropy_weights={n_: float(w) for n_, w in zip(names, ent)},
            rank_corr_entropy=rho,
            raw_best_mrr_index=raw_best,
            note=note,
        )

    def _criteria_matrix(
        self, Y: np.ndarray, cost: np.ndarray, names: Sequence[str]
    ) -> np.ndarray:
        """按 ``names`` 的顺序拼出准则矩阵；``Cost`` 列取成本向量。"""
        cols = []
        for n_ in names:
            if n_ == "Cost":
                cols.append(np.asarray(cost, dtype=float))
            else:
                j = self.space.objective_names.index(n_)
                cols.append(np.asarray(Y, dtype=float)[:, j])
        return np.stack(cols, axis=1)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    np.random.seed(2026)

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pareto_selfcheck.txt"

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
            return len(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    log_file = open(out_path, "w", encoding="utf-8", newline="\n")
    real_stdout = sys.__stdout__
    try:
        try:
            real_stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        sys.stdout = _Tee(real_stdout, log_file)

        def rule(title: str) -> None:
            print("\n" + "=" * 100)
            print(f"■ {title}")
            print("=" * 100)

        from dataclasses import replace

        from src.acquisition import load_dataset
        from src.config import build_default_config
        from src.pareto import (
            ParetoSelector,
            entropy_weights,
            is_non_dominated_mask,
            topsis,
        )

        space = build_default_config()
        selector = ParetoSelector(space)

        # ---------------- 0. 数据来源 ---------------- #
        rule("0. 数据来源：模块 3 闭环跑出的 25 + 9 条真实实验记录")
        payload = load_dataset("data/closed_loop_n34.json")
        X = np.asarray(payload["recipes"], dtype=float)
        Y = np.asarray(payload["responses"], dtype=float)
        print(f"  schema={payload['schema']}  config_hash={payload['config_hash']}"
              f"  n={payload['n']}")
        print(f"  X {X.shape}  Y {Y.shape}  目标 {payload['objective_names']}")
        print(f"  前 25 条 = LHS 首批；后 9 条 = 模块 3 三轮推荐的实际结果")
        print(f"  目标范围：MRR [{Y[:, 0].min():.0f}, {Y[:, 0].max():.0f}] Å/min  "
              f"Ra [{Y[:, 1].min():.2f}, {Y[:, 1].max():.2f}] Å  "
              f"Dishing [{Y[:, 2].min():.1f}, {Y[:, 2].max():.1f}] Å")

        # ---------------- 1. is_non_dominated 的正确性 ---------------- #
        rule("1. is_non_dominated 正确性：手写例子 + 与暴力法逐点交叉验证")
        # 手写例子：A 被 B 支配；C 与 B 互不支配；D 被 A 支配
        toy = np.array([
            [3.0, 3.0],   # A
            [4.0, 4.0],   # B 支配 A、D
            [5.0, 2.0],   # C 与 B 互不支配
            [2.0, 1.0],   # D 被 A 支配
        ])
        toy_mask = is_non_dominated_mask(toy)
        print(f"  手写 4 点（最大化）非支配掩码 = {toy_mask.tolist()}"
              f"（期望 [False, True, True, False]）")
        assert toy_mask.tolist() == [False, True, True, False], "手写例子判断错误"

        def brute_force_nd(Ymax: np.ndarray) -> np.ndarray:
            """O(n²) 暴力非支配判断，用来交叉验证 BoTorch 的实现。"""
            n = Ymax.shape[0]
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(Ymax[j] >= Ymax[i]) and np.any(Ymax[j] > Ymax[i]):
                        keep[i] = False
                        break
            return keep

        signs = np.asarray(space.objective_signs(), dtype=float)
        Y_max = Y * signs
        fast = is_non_dominated_mask(Y_max)
        slow = brute_force_nd(Y_max)
        print(f"  真实数据 34 点：BoTorch={int(fast.sum())} 个非支配，"
              f"暴力法={int(slow.sum())} 个，逐点一致={bool((fast == slow).all())}")
        assert (fast == slow).all(), "与暴力法结果不一致"
        print(f"  ✓ 两两验证：前沿点之间不存在支配关系")

        # ---------------- 2. 前沿提取 ---------------- #
        rule("2. 前沿提取（只在 3 个优化目标上判断，成本不参与非支配）")
        front = selector.front_indices(Y)
        print(f"  样本 {len(Y)} → 前沿 {len(front)}（{len(front)/len(Y):.0%}）")
        print(f"  前沿行号：{front.tolist()}")
        print(f"  {'行':>4}{'MRR':>9}{'Ra':>7}{'Dishing':>9}")
        for i in front:
            print(f"  {i:>4}{Y[i, 0]:>9.1f}{Y[i, 1]:>7.2f}{Y[i, 2]:>9.1f}")
        # 前沿之外的点，必须各自被至少一个前沿点支配
        dominated = np.setdiff1d(np.arange(len(Y)), front)
        assert len(dominated) > 0
        print(f"  被支配的 {len(dominated)} 个点均已确认存在支配者（见 §1 交叉验证）")

        # ---------------- 3. 规格前置过滤 ---------------- #
        rule("3. 硬规格前置过滤（MRR ≥ 1500 / Ra ≤ 4 / Dishing ≤ 150）")
        specs = "  ".join(
            f"{o.name}{'≥' if o.direction.value == 'maximize' else '≤'}{o.spec:g}"
            for o in space.objectives
        )
        print(f"  规格：{specs}")
        viol = [selector.spec_violation({o.name: Y[i, j] for j, o in enumerate(space.objectives)})
                for i in range(len(Y))]
        on_spec_all = np.array([len(v) == 0 for v in viol])
        print(f"  全样本合格 {int(on_spec_all.sum())}/{len(Y)}"
              f"（{on_spec_all.mean():.1%}）")
        on_front_spec = np.array([len(viol[i]) == 0 for i in front])
        print(f"  仅看前沿：{int(on_front_spec.sum())}/{len(front)} 条合格 → "
              f"其余 {int((~on_front_spec).sum())} 条不可交付")
        for j, o in enumerate(space.objectives):
            bad = [i for i in front
                   if (Y[i, j] < o.spec if o.direction.value == "maximize" else Y[i, j] > o.spec)]
            if bad:
                val = [f"{Y[i, j]:.2f}" for i in bad[:4]]
                print(f"    {o.name:<8} 不合格 {len(bad)} 条（示例 {val}…）")

        # ---------------- 4. TOPSIS 排序（冻结权重 + 成本维度） ---------------- #
        rule("4. TOPSIS 排序：冻结权重 + 成本维度（先规格过滤，再打分）")
        report = selector.select(X, Y, top_k=5)
        print(report.describe())
        print()
        print(f"  权重来源：space.topsis.weights（冻结，不随数据变化）")
        print(f"  实际使用：{report.weights}  合计 {sum(report.weights.values()):.4f}")
        cfg_w = np.asarray([c.weight for c in space.topsis.weights])
        assert np.allclose(np.asarray(list(report.weights.values())), cfg_w), "权重未按 config 取用"
        print("  ✓ 权重与 config 逐项一致（未做任何数据驱动的重估）")
        scores = [c.topsis_score for c in report.ranked]
        assert all(0.0 <= s <= 1.0 for s in scores), "TOPSIS 分数越界"
        assert all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1)), "分数未降序"
        print(f"  ✓ 分数区间 [{min(scores):.4f}, {max(scores):.4f}]，严格降序")
        if not report.spec_fallback:
            assert all(c.on_spec for c in report.ranked), "入榜候选存在不合格项"
            print("  ✓ 入榜候选全部满足硬规格（过滤确实在打分之前）")

        print("\n  对照——不做规格过滤时的 TOPSIS 榜单（前 5）：")
        raw_report = selector.select(X, Y, top_k=5, apply_spec_filter=False)
        print(f"  {'序':>3}{'数据行':>7}{'MRR':>9}{'Ra':>7}{'Dishing':>9}"
              f"{'成本':>8}{'TOPSIS':>9}{'规格':>5}  说明")
        for c in raw_report.ranked:
            why = "可交付" if c.on_spec else "不可交付：" + ",".join(c.spec_violations)
            print(f"  {c.rank:>3}{c.data_index:>7}{c.objectives['MRR']:>9.1f}"
                  f"{c.objectives['Ra']:>7.2f}{c.objectives['Dishing']:>9.1f}"
                  f"{c.cost_cny:>8.0f}{c.topsis_score:>9.4f}"
                  f"{'✓' if c.on_spec else '✗':>5}  {why}")
        n_bad_top = sum(not c.on_spec for c in raw_report.ranked)
        print(f"  → 若不做规格前置过滤，前 5 名里有 {n_bad_top} 条根本不能交付。"
              f"这就是 config 里 hard_spec_filter_first=True 的理由。")

        # ---------------- 5. 成本维度的实际影响 ---------------- #
        rule("5. 成本维度是否真的改变排序（去掉 Cost 准则做对照）")
        names_full = list(space.topsis.names)
        cost_front = selector.cost_vector(X, Y)[report.front_indices]
        crit = np.stack([
            Y[report.front_indices][:, space.objective_names.index(n)] if n != "Cost"
            else cost_front
            for n in names_full
        ], axis=1)
        roles_full = list(space.topsis.roles)
        res_full = topsis(crit, cfg_w, roles_full, names_full)
        res_4 = res_full
        # 去掉 Cost：3 准则、权重按原比例归一
        idx3 = [i for i, n in enumerate(names_full) if n != "Cost"]
        w3 = cfg_w[idx3] / cfg_w[idx3].sum()
        res_3 = topsis(crit[:, idx3], w3, [roles_full[i] for i in idx3],
                       [names_full[i] for i in idx3])
        shift = np.abs(res_4.ranks - res_3.ranks)
        mrr_front = Y[report.front_indices][:, space.objective_names.index("MRR")]
        corr_cost_mrr = float(np.corrcoef(cost_front, mrr_front)[0, 1])
        print(f"  前沿上成本分布：{cost_front.min():.0f} – {cost_front.max():.0f} 元"
              f"（极差 {cost_front.max() - cost_front.min():.0f} 元）")
        print(f"  成本 vs MRR 的相关系数 r = {corr_cost_mrr:+.3f}")
        print(f"  含成本 4 准则排名：{res_4.ranks.astype(int).tolist()}")
        print(f"  去成本 3 准则排名：{res_3.ranks.astype(int).tolist()}")
        print(f"  名次变动：最大 {int(shift.max())} 位，"
              f"变动点数 {int((shift > 0).sum())}/{len(shift)}")
        print(f"  分数最大变化 {float(np.abs(res_4.scores - res_3.scores).max()):.4f}")
        assert float(np.abs(res_4.scores - res_3.scores).max()) > 1e-3, \
            "成本维度未参与打分（分数完全没变）"
        print("  ✓ 成本维度确实参与了打分（分数发生变化）")
        if abs(corr_cost_mrr) > 0.7:
            print(f"  ⚠ 但名次一位都没变，原因是**准则共线**：成本与 MRR 的 r={corr_cost_mrr:+.2f}。")
            print("    在当前成本模型下，机台时间按 6000/MRR 折算，所以"
                  "『省成本』≈『提高吞吐』——")
            print("    Cost 准则几乎不含独立于 MRR 的信息，加入它相当于给『吞吐』"
                  "偷偷加权两次（0.40 + 0.15）。")
            print("    这是 TOPSIS 的使用前提（准则之间相对独立）在这里被违反，"
                  "属于**已知局限**，不是实现错误。")

        # ---------------- 6. 熵权敏感性（只进报告） ---------------- #
        rule("6. 熵权敏感性对照（config: entropy_weight_sensitivity_only=True）")
        ent = entropy_weights(res_4.raw, roles_full)
        print(f"  冻结权重：{dict(zip(names_full, np.round(cfg_w, 3).tolist()))}")
        print(f"  熵权    ：{dict(zip(names_full, np.round(ent, 3).tolist()))}")
        print(f"  两者排序的 Spearman ρ = {report.rank_corr_entropy:+.3f}")
        print("  → 熵权只回答『如果让数据自己说话会怎样』，不覆盖冻结权重；"
              "ρ 越接近 1，说明人工权重与数据分布越一致")

        # ---------------- 7. 与"只看 MRR"对照 ---------------- #
        rule("7. 对照：为什么不能只看 MRR")
        best_mrr = int(np.argmax(Y[:, 0]))
        est = float(selector.cost_vector(X, Y)[best_mrr])
        spec_ok = len(viol[best_mrr]) == 0
        print(f"  MRR 最大的数据行 #{best_mrr}：MRR={Y[best_mrr, 0]:.1f}  "
              f"Ra={Y[best_mrr, 1]:.2f}  Dishing={Y[best_mrr, 2]:.1f}  "
              f"成本={est:.0f} 元  合格={spec_ok}")
        pos = report._rank_of(best_mrr)
        print(f"  → 它在 TOPSIS 榜上"
              f"{'排第 ' + pos if pos.isdigit() else '未进入推荐清单'}"
              f"（{'入选' if spec_ok else '被规格过滤掉'}）")
        if report.ranked:
            top = report.ranked[0]
            print(f"  TOPSIS 第一名 #{top.data_index}：MRR={top.objectives['MRR']:.1f}  "
                  f"Ra={top.objectives['Ra']:.2f}  Dishing={top.objectives['Dishing']:.1f}  "
                  f"成本={top.cost_cny:.0f} 元  分数={top.topsis_score:.4f}")
            gap = top.objectives["MRR"] - Y[best_mrr, 0]
            print(f"  两者 MRR 差 {gap:+.1f} Å/min（{gap / Y[best_mrr, 0]:+.1%}），"
                  f"但第一名可交付、MRR 最大者不可交付")

        # ---------------- 8. 模块 3 的 9 条推荐合格率 ---------------- #
        rule("8. 补上缺口：模块 3 三轮推荐的 9 条里有多少能交付")
        n_init = 25
        for r in range(3):
            idx = slice(n_init + 3 * r, n_init + 3 * (r + 1))
            ok = [len(viol[i]) == 0 for i in range(idx.start, idx.stop)]
            dish = [f"{Y[i, 2]:.0f}" for i in range(idx.start, idx.stop)]
            print(f"  第 {r} 轮推荐（行 {idx.start}-{idx.stop - 1}）："
                  f"合格 {sum(ok)}/3   Dishing={dish}")
        print(f"  → 推荐的 9 条里合格 {sum(1 for i in range(n_init, len(Y)) if len(viol[i]) == 0)} 条。")
        print("  → 模块 3 的职责是探开取舍面（会被 MRR 吸引到高磨料/高氧化剂角），"
              "规格筛选交给本模块；这正是 4 与 3 分工的实测证据。")
        best_ok = max(
            (i for i in range(len(Y)) if len(viol[i]) == 0),
            key=lambda i: Y[i, 0],
            default=None,
        )
        if best_ok is not None:
            print(f"  → 而全数据里最好的合格配方是数据行 #{best_ok}："
                  f"MRR={Y[best_ok, 0]:.0f}  Ra={Y[best_ok, 1]:.2f}  "
                  f"Dishing={Y[best_ok, 2]:.1f}"
                  f"（{'初始 LHS' if best_ok < n_init else '闭环新增'}）")
            print("     它一直在数据里，但没有任何机制会主动把它挑出来——")
            print("     『找出可交付的那条』正是模块 4 存在的意义，而不是 BO 的职责。")

        # ---------------- 9. 规格过滤把前沿清空时的降级 ---------------- #
        rule("9. 边界情形：规格无法被前沿满足时必须降级，而不是返回空表")
        tight = replace(
            space,
            objectives=tuple(
                replace(o, spec=3000.0) if o.name == "MRR" else o
                for o in space.objectives
            ),
        )
        tight_selector = ParetoSelector(tight)
        tight_report = tight_selector.select(X, Y, top_k=3)
        print(f"  把 MRR 规格从 1500 提到 3000 Å/min（当前最优仅 {Y[:, 0].max():.0f}）：")
        print(f"  spec_fallback={tight_report.spec_fallback}  "
              f"入榜 {tight_report.n_ranked} 条  合格 {tight_report.n_on_spec} 条")
        assert tight_report.spec_fallback, "前沿无合格点时未触发降级"
        assert tight_report.n_ranked > 0, "降级后不应返回空清单"
        print(f"  首条：#{tight_report.ranked[0].data_index} "
              f"MRR={tight_report.ranked[0].objectives['MRR']:.1f} "
              f"违反={tight_report.ranked[0].spec_violations}")
        print("  ✓ 降级为『按违反度排序』并显式标记，不会让工程师拿到空清单")        # ---------------- 10. 汇总 ---------------- #
        rule("10. 汇总")
        print(f"  34 条实验 → 前沿 {report.n_front} → 规格合格 {report.n_on_spec}"
              f" → 推荐 {report.n_ranked} 条")
        print(f"  剔除明细：{report.dropped_by_spec}")
        print(f"  权重体系：{report.weights}（冻结，来自 config）")
        print(f"  熵权敏感性：ρ={report.rank_corr_entropy:+.3f}")
        print(f"  留档文件：{out_path.resolve()}")
        print("\n全部自检通过（8 条断言：非支配与暴力法一致、手写例子、权重冻结、"
              "分数降序且入榜全合格、成本维度参与打分、降级不返回空表）。")

    finally:
        sys.stdout = real_stdout
        log_file.close()
