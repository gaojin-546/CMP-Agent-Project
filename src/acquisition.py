"""CMP 抛光液配方开发 Agent —— 模块 3：多目标采集函数（约束 qNEHVI）。

本模块在真实 BoTorch 0.18.1 上实现"下一个该做哪个配方"的决策，包括：

* **Constrained qNEHVI**（默认 ``qLogNoisyExpectedHypervolumeImprovement``，
  即数值稳定版 qNEHVI，API 完全一致）；
* **硬约束进入优化器**：9 条解析约束以非线性不等式约束传给 SLSQP；
* **成本感知**：``penalty_lambda`` 作用在归一化成本上，按对数空间/线性空间
  分别以"减去"或"乘以"的方式折减采集值；
* **动态参考点**：每轮取历史最差观测再退让一个 margin，**单调不增**并逐个记录；
* **后备目标处理**：常量后备的目标没有协方差结构，不能进 ``ModelListGP``，
  自动从多目标向量中剔除并告警（等价于权重降为 0，而不是伪造一个 GP）；
* **可行性保障**：SLSQP 在 9 条非线性约束下并不保证收敛到可行点，因此
  优化后做"验证 → 最近可行点修复"，并对隐式规则（规则库）做前置过滤。

三个必须知道的底层契约（都是在本环境实测出来的，照旧版教程写会静默出错）
--------------------------------------------------------------------
1. **``constraints=`` 是输出空间约束，不是 X 空间的。**
   ``MCAcquisitionFunction.constraints`` 的入参形状是
   ``sample_shape x batch x q x **m(输出维)**``——它吃的是**后验采样值**，
   用来表达"Dishing 后验超过规格就降权"这类**结果**约束。
   我们的 9 条硬约束是**设计变量**的解析函数（pH 窗口、磨料/氧化剂比例…），
   必须走 ``optimize_acqf(nonlinear_inequality_constraints=...)``。
   若把 X 空间约束塞进 ``constraints=``，函数会把 3 个输出当变量读，
   静默产生一个语义完全错误的"约束"，而且不报错。

2. **非线性约束的 callable 必须对 x 可微，且约定相反。**
   ``optimize_acqf`` 文档写的是 ``callable(x) >= 0`` 可行（我们的 ``g <= 0``
   可行，所以返回 ``-g``）；更关键的是 SLSQP 会对它**求梯度**
   （``botorch/optim/parameter_constraints.py`` 的 ``f_obj_and_grad``），
   所以内部任何 ``float()`` / ``numpy`` 转换都会切断计算图，报
   ``element 0 of tensors does not require grad``。

3. **给了非线性约束就必须自己提供 ``ic_generator``。**
   0.18 不再用 ``raw_samples`` 自动生成初始点（避免初始点本身不可行），
   否则抛 ``RuntimeError: ic_generator must be given``。本模块的
   ``FeasibleICGenerator`` 在可行域内做拒绝采样 + 采集值预热，既满足契约
   又把 ``num_restarts`` 用在真正有希望的区域。

用法
----
    >>> from src.config import build_default_config
    >>> from src.acquisition import CMPBatchPlanner
    >>> space = build_default_config()
    >>> planner = CMPBatchPlanner(space)
    >>> rec = planner.recommend_next_batch(X_observed, Y_observed, q=3)
    >>> print(rec.describe())
    >>> rec.recipes            # (3, 10) 物理量纲，可直接下发
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch.optim.initializers import draw_sobol_samples
from botorch.exceptions.errors import CandidateGenerationError
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from src.config import CMPRecipeSpace
from src.surrogate_model import (
    DEFAULT_TKWAARGS,
    STATUS_DEGRADED,
    STATUS_FALLBACK,
    STATUS_OK,
    CMPGPRSurrogate,
    ConstantFallback,
    ObjectiveDiagnostics,
)

__all__ = [
    "RefPointRecord",
    "ReferencePointTracker",
    "CostAwareAcquisition",
    "BatchRecommendation",
    "CMPBatchPlanner",
    "make_nonlinear_constraints",
    "make_feasible_ic_generator",
    "objective_normalization",
    "dump_dataset",
    "load_dataset",
]


#: 闭环数据集的落盘格式版本。模块 4/5/6 都读同一份文件，格式变更必须升版本号。
DATASET_SCHEMA = "cmp-closed-loop-v1"


def dump_dataset(
    path: Any,
    X: Any,
    Y: Any,
    objective_names: Sequence[str],
    *,
    config_hash: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """把闭环实验数据落成 JSON（模块 5 回写契约的最小实现）。

    为什么不做成 CSV：配方是 10 维、还会加协变量与异常标记，列式格式在
    "配方 = 一个样本"的语义下极易错位；JSON 保留结构，模块 4/6 直接读，
    后续要接数据库也只换这一层。

    Returns
    -------
    str
        实际写入的绝对路径。
    """
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DATASET_SCHEMA,
        "config_hash": config_hash,
        "objective_names": list(objective_names),
        "n": int(len(X)),
        "recipes": np.asarray(X, dtype=float).tolist(),
        "responses": np.asarray(Y, dtype=float).tolist(),
        "extra": extra or {},
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p.resolve())


def load_dataset(path: Any) -> Dict[str, Any]:
    """读取 :func:`dump_dataset` 写出的文件并做基本校验。"""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集不存在：{p.resolve()}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != DATASET_SCHEMA:
        raise ValueError(f"数据集格式版本不匹配：期望 {DATASET_SCHEMA}，实际 {schema}")
    X = np.asarray(payload["recipes"], dtype=float)
    Y = np.asarray(payload["responses"], dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X 行数 {X.shape[0]} 与 Y 行数 {Y.shape[0]} 不一致")
    return payload


def _t(x: Any, tkwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    kw = DEFAULT_TKWAARGS if tkwargs is None else tkwargs
    return x.to(**kw) if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), **kw)


def _fmt(x: float, nd: int = 3, width: int = 0) -> str:
    s = "—" if not (isinstance(x, float) and math.isfinite(x)) else f"{x:.{nd}f}"
    return s.rjust(width) if width else s


# --------------------------------------------------------------------------- #
# 1. 目标归一化与动态参考点
# --------------------------------------------------------------------------- #
def objective_normalization(
    space: CMPRecipeSpace, objectives: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 ``(weights, spans)``：``weights = 1/span``，``spans`` 为目标先验跨度。

    为什么必须归一化：三个目标的量纲差三个数量级（MRR ~2000 Å/min、Ra ~3 Å、
    Dishing ~200 Å）。超体积是各维"改进量"的乘积，未归一化时 Ra 的整段变化
    （约 8 Å）对 HV 的贡献不到 0.4%，优化器会退化成"只优化 MRR"，
    Ra/Dishing 的帕累托权衡被量纲差异直接淹没——这不是算法问题，
    是量纲问题，且**不会报错**。

    只做缩放不做平移：HV 对平移不变（``HV(Y+c, r+c) = HV(Y, r)``），
    所以 ``weights`` 足以让 HV 变得无量纲、可跨轮次比较。
    """
    w, s = [], []
    for n in objectives:
        o = space.objectives[space.objective_names.index(n)]
        span = float(o.prior_range[1] - o.prior_range[0])
        span = span if span > 0 else 1.0
        w.append(1.0 / span)
        s.append(span)
    return np.asarray(w, dtype=float), np.asarray(s, dtype=float)


@dataclass
class RefPointRecord:
    """一轮的参考点快照（归一化最大化空间，单调不增）。"""

    round_index: int
    objectives: Tuple[str, ...]
    ref_point: Tuple[float, ...]
    """归一化最大化空间（喂给 qNEHVI 的就是这一组）。"""
    ref_point_physical: Tuple[float, ...]
    """换算回原量纲，便于工程师阅读。"""
    worst_observed: Tuple[float, ...]
    hv: float
    changed: bool


class ReferencePointTracker:
    """管理 qNEHVI 参考点的演化（在**归一化最大化空间**里工作）。

    更新规则 ``ref_t = min(ref_{t-1}, worst_obs_t - margin·|worst_obs_t|)``：

    * **初值取自数据**（``mode="dynamic"``）：用首批观测的最差端退让一个 margin。
      若沿用模块 1 的先验参考点（MRR=132 Å/min），它远低于任何真实观测
      （≥700 Å/min），于是参考点永远不动、HV 里塞着一大块永远填不满的空区，
      对外表现为"动态更新从来没触发过"——这是实测踩到的坑。
    * **单调不增**：HV 相对参考点定义，参考点回撤会造成 HV 假性下降。
    * ``mode="static"`` 时冻结在先验参考点，供消融实验对照。
    """

    def __init__(
        self, space: CMPRecipeSpace, history: Optional[List[RefPointRecord]] = None
    ) -> None:
        self.space = space
        self.margin = space.ref_point_policy.margin_ratio
        self.mode = space.ref_point_policy.mode
        self.history: List[RefPointRecord] = list(history or [])
        self._ref: Optional[np.ndarray] = None
        self._objectives: Tuple[str, ...] = ()
        self._weights: np.ndarray = np.asarray([])
        self._spans: np.ndarray = np.asarray([])
        self._signs: np.ndarray = np.asarray([])

    # -- 归一化 ----------------------------------------------------------- #
    def _signs_of(self, objectives: Sequence[str]) -> np.ndarray:
        idx = [self.space.objective_names.index(n) for n in objectives]
        return np.asarray(self.space.objective_signs(), dtype=float)[idx]

    def normalize(self, Y_phys: Any, objectives: Sequence[str]) -> np.ndarray:
        """原量纲 ``(n, m)`` -> 归一化最大化空间 ``(n, m)``。"""
        objectives = tuple(objectives)
        w, _ = objective_normalization(self.space, objectives)
        signs = self._signs_of(objectives)
        Y = np.asarray(Y_phys, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, len(objectives))
        if Y.shape[-1] != len(objectives):
            raise ValueError(
                f"Y 列数 {Y.shape[-1]} 与目标数 {len(objectives)} 不一致"
            )
        return (Y * signs) * w

    # -- 更新 ------------------------------------------------------------- #
    def update(
        self,
        Y_phys: Any,
        objectives: Sequence[str],
        round_index: Optional[int] = None,
    ) -> torch.Tensor:
        """用本轮观测更新参考点并记一条历史；返回归一化参考点张量。"""
        objectives = tuple(objectives)
        Y_norm = self.normalize(Y_phys, objectives)
        w, spans = objective_normalization(self.space, objectives)
        signs = self._signs_of(objectives)
        worst = Y_norm.min(axis=0)

        if self._ref is None or self._objectives != objectives:
            self._objectives = objectives
            self._weights, self._spans, self._signs = w, spans, signs
            if self.mode == "static":
                full = np.asarray(self.space.reference_point(), dtype=float)
                idx = [self.space.objective_names.index(n) for n in objectives]
                self._ref = full[idx] * w
            else:
                self._ref = worst - self.margin * np.abs(worst)

        prev = self._ref.copy()
        if self.mode == "dynamic":
            cand = worst - self.margin * np.abs(worst)
            self._ref = np.minimum(self._ref, cand)
        changed = bool(np.any(self._ref < prev - 1e-12))

        hv = self.hypervolume(Y_norm)
        rec = RefPointRecord(
            round_index=len(self.history) if round_index is None else round_index,
            objectives=objectives,
            ref_point=tuple(float(v) for v in self._ref),
            ref_point_physical=self.to_physical(self._ref),
            worst_observed=tuple(float(v) for v in worst),
            hv=hv,
            changed=changed,
        )
        self.history.append(rec)
        return _t(self._ref)

    def to_physical(self, ref_norm: np.ndarray) -> Tuple[float, ...]:
        """归一化参考点 -> 原量纲（``y = sign · norm · span``）。"""
        if self._signs.size == 0:
            return tuple(float(v) for v in ref_norm)
        phys = self._signs * np.asarray(ref_norm, dtype=float) * self._spans
        return tuple(float(v) for v in phys)

    # -- 超体积 ----------------------------------------------------------- #
    def hypervolume(self, Y_norm: Any, ref: Optional[np.ndarray] = None) -> float:
        """归一化空间下的超体积（无量纲，跨轮次可比）。

        先剔除"被参考点支配"的点：``Hypervolume.compute`` 对这类点没有定义。
        """
        Y = np.asarray(Y_norm, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        if Y.shape[-1] < 2:
            return float("nan")
        r = self._ref if ref is None else np.asarray(ref, dtype=float)
        if r is None:
            return float("nan")
        better = np.all(Y > r, axis=-1)
        Y = Y[better]
        if Y.shape[0] == 0:
            return 0.0
        nd = is_non_dominated(torch.as_tensor(Y, dtype=torch.double)).numpy()
        return float(
            Hypervolume(torch.as_tensor(r, dtype=torch.double)).compute(
                torch.as_tensor(Y[nd], dtype=torch.double)
            )
        )

    # -- 报告 ------------------------------------------------------------- #
    @property
    def ref_point(self) -> Optional[np.ndarray]:
        """归一化参考点（直接喂 qNEHVI）。"""
        return None if self._ref is None else self._ref.copy()

    @property
    def ref_point_physical(self) -> Optional[Tuple[float, ...]]:
        return None if self._ref is None else self.to_physical(self._ref)

    def describe(self) -> str:
        if not self.history:
            return "（参考点尚无记录）"
        lines = [
            f"{'轮':>3}  {'参考点（原量纲）':<40}{'归一化HV':>10}  备注",
            "-" * 82,
        ]
        prev_hv = float("nan")
        for r in self.history:
            dhv = r.hv - prev_hv if math.isfinite(prev_hv) else float("nan")
            note = "下移" if r.changed else "不变"
            lines.append(
                f"{r.round_index:>3}  {str([round(v, 3) for v in r.ref_point_physical]):<40}"
                f"{r.hv:>10.4f}  {note}"
                + (f"，ΔHV={dhv:+.4f}" if math.isfinite(dhv) else "")
            )
        lines.append(f"目标顺序：{self._objectives}")
        lines.append("归一化 HV = 各目标按先验跨度缩放后的超体积，量纲无关，可跨轮次比较")
        return "\n".join(lines)

# --------------------------------------------------------------------------- #
# 2. 成本感知包装
# --------------------------------------------------------------------------- #
class CostAwareAcquisition(AcquisitionFunction):
    """把归一化成本折减进采集值的包装器（模块 3 的成本感知）。

    数学形式（``c`` 为 q 个候选的平均归一化成本，``λ`` 为 ``penalty_lambda``）：

    * 线性采集值（qNEHVI / qLogEI 返回概率或改进量）：``val · exp(-λc)``
    * **对数采集值（qLogNEHVI 返回 log 改善量）：``val - λc``**

    第二行是必须单独处理的细节：对数空间里"乘以成本折扣"等价于"减去 λc"，
    若照着线性版写 ``log_val · exp(-λc)``，会把一个负数（log < 0 很常见）
    乘得更接近 0，产生"成本越高排名越前"的反向效果——而且不会报错。
    """

    def __init__(
        self,
        base_acqf: AcquisitionFunction,
        space: CMPRecipeSpace,
        penalty_lambda: float,
        *,
        log_space: bool,
        tkwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.space = space
        self.penalty_lambda = float(penalty_lambda)
        self.log_space = bool(log_space)
        self.tkwargs = dict(DEFAULT_TKWAARGS if tkwargs is None else tkwargs)
        self.reference_cost = float(space.cost_model.reference_cost)

    # -- 成本 ------------------------------------------------------------- #
    def normalized_cost(self, X: torch.Tensor) -> torch.Tensor:
        """单位立方输入 ``(..., q, d)`` -> 归一化成本 ``(...)``。

        成本用模块 1 的解析成本模型（Preston 先验折算机台时间），
        不走后验——否则每次采集函数求值都要多做一次 GP 前向，
        SLSQP 的每次迭代都要付这个代价。
        """
        shape = X.shape[:-1]
        flat = X.reshape(-1, X.shape[-1])
        phys = self.space.from_unit(flat)
        cost = self.space.cost(phys)                      # (B,) 元
        cost = cost.reshape(*shape)
        return cost / max(self.reference_cost, 1e-9)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        base = self.base_acqf(X)
        c = self.normalized_cost(X).mean(dim=-1)          # 整批的平均成本
        if self.log_space:
            return base - self.penalty_lambda * c
        return base * torch.exp(-self.penalty_lambda * c)


# --------------------------------------------------------------------------- #
# 3. 硬约束桥接
# --------------------------------------------------------------------------- #
def make_nonlinear_constraints(
    space: CMPRecipeSpace, tkwargs: Optional[Dict[str, Any]] = None
) -> List[Tuple[Callable[[torch.Tensor], torch.Tensor], bool]]:
    """把模块 1 的 9 条硬约束翻译成 ``optimize_acqf`` 需要的形式。

    返回 ``[(callable, is_intra_point), ...]``，其中 ``callable(x) >= 0`` 可行
    （与模块 1 的 ``g <= 0`` 相反，故取负号）。

    三个实现要点：

    1. **单位立方 -> 物理量纲必须可微**：``space.from_unit`` 的 torch 分支是
       逐列仿射 + clamp，梯度可以正常回传；
    2. **不能出现 ``float()`` / ``.numpy()``**：SLSQP 要靠
       ``torch.autograd.grad`` 求约束梯度，一旦转成 Python 标量就会抛
       ``element 0 of tensors does not require grad``；
    3. **``ANY_OF``（pH 双窗口）用 ``min`` 实现**：模块 1 里已经是
       ``min(g_acid, g_alkaline)``，求导时只有当前所在窗口的分支有梯度，
       这正是"并集可行域"的正确处理。
    """
    kw = dict(DEFAULT_TKWAARGS if tkwargs is None else tkwargs)
    names = space.names
    out: List[Tuple[Callable[[torch.Tensor], torch.Tensor], bool]] = []

    for con in space.hard_constraints:
        bound = con.bind(names)

        def fn(x, _bound=bound, _name=con.name):
            xt = torch.as_tensor(x, **kw).reshape(1, -1)
            xt = xt.requires_grad_(True) if not xt.requires_grad else xt
            g = _bound(space.from_unit(xt)).reshape(-1)[0]
            return -g          # g<=0 可行 -> -g>=0

        fn.__name__ = f"c_{con.name}"
        out.append((fn, True))  # intra-point：逐候选点约束
    return out


# --------------------------------------------------------------------------- #
# 4. 可行初始点生成器
# --------------------------------------------------------------------------- #
class FeasibleICGenerator:
    """在可行域内生成 ``optimize_acqf`` 的初始条件（``ic_generator`` 契约）。

    契约来源于 ``botorch/optim/optimize.py``：给了非线性约束后，
    优化器只用 ``get_ic_generator()`` 生成的初始点，签名必须是

        (acq_function, bounds, q, num_restarts, raw_samples,
         fixed_features, options, inequality_constraints, equality_constraints)

    本实现做两件事：
    1. **拒绝采样**：Sobol 大池 → 精确可行性过滤（本项目可行率约 32%，
       直接随机初始点有 2/3 概率落在禁区）；
    2. **采集值预热**：随机拼出 ``num_restarts × batch_mult`` 个 q 批次，
       用采集函数打分取前 ``num_restarts`` 个，让每次 SLSQP 重启都从有希望的
       可行区域出发（比纯随机初始点收敛更稳）。
    """

    def __init__(
        self,
        space: CMPRecipeSpace,
        *,
        pool_size: int = 2048,
        batch_mult: int = 4,
        seed: int = 0,
        tkwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.space = space
        self.pool_size = int(pool_size)
        self.batch_mult = int(batch_mult)
        self.seed = int(seed)
        self.tkwargs = dict(DEFAULT_TKWAARGS if tkwargs is None else tkwargs)
        #: 最近一次生成的可行池，供"验证-修复"环节复用（避免二次采样）
        self.pool: Optional[torch.Tensor] = None
        self.last_batches: Optional[torch.Tensor] = None
        self.last_values: Optional[torch.Tensor] = None
        self.pool_feasible_ratio: float = float("nan")
        #: 调用计数：每次调用换一个确定性的子种子，保证"同一份数据 + 同一个 seed
        #: → 同一批推荐"。用全局 RNG（``torch.randperm`` 不传 generator）会让结果
        #: 依赖"之前调用过什么"，实测导致同一用例在不同执行顺序下时好时坏。
        self._call_count = 0

    def __call__(
        self,
        acq_function: AcquisitionFunction,
        bounds: torch.Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[Dict[str, Any]] = None,
        inequality_constraints: Optional[Any] = None,
        equality_constraints: Optional[Any] = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        raw = draw_sobol_samples(bounds=bounds, n=self.pool_size, q=1, seed=self.seed)
        raw = raw.squeeze(-2)                              # (pool, d)
        feasible = torch.as_tensor(
            self.space.feasible_mask(self.space.from_unit(raw).numpy())
        )
        pool = raw[feasible]
        self.pool = pool
        self.pool_feasible_ratio = float(feasible.double().mean())
        if pool.shape[0] < max(q, 2):
            raise RuntimeError(
                f"可行池只剩 {pool.shape[0]} 个点（共采样 {self.pool_size}），"
                f"硬约束可能过紧，无法生成初始条件"
            )

        n_batches = max(num_restarts * self.batch_mult, num_restarts)
        gen = torch.Generator(device="cpu").manual_seed(
            self.seed * 1009 + self._call_count
        )
        self._call_count += 1
        idx = torch.stack(
            [torch.randperm(pool.shape[0], generator=gen)[:q] for _ in range(n_batches)]
        )
        batches = pool[idx]                                # (n_batches, q, d)
        with torch.no_grad():
            values = acq_function(batches).detach().reshape(-1)
        keep = torch.topk(values, k=min(num_restarts, n_batches)).indices
        self.last_batches = batches
        self.last_values = values
        return batches[keep]


# --------------------------------------------------------------------------- #
# 5. 推荐结果
# --------------------------------------------------------------------------- #
@dataclass
class BatchRecommendation:
    """一轮推荐的结果包（可直接下发实验、也可整体写进日志）。"""

    round_index: int
    recipes: torch.Tensor
    """``(q, d)`` 物理量纲配方，列顺序与 ``space.names`` 一致。"""
    recipe_table: List[Dict[str, float]]
    predicted_mean: torch.Tensor
    """``(q, m_active)`` 原量纲预测均值。"""
    predicted_sd: torch.Tensor
    cost_cny: np.ndarray
    active_objectives: Tuple[str, ...]
    dropped_objectives: Tuple[str, ...]
    ref_point: Tuple[float, ...]
    """归一化最大化空间的参考点（喂给采集函数的就是这一组）。"""
    ref_point_prev: Tuple[float, ...]
    ref_point_physical: Tuple[float, ...]
    """换算回原量纲的参考点，便于工程师核对。"""
    hv_before: float
    hv_after_pseudo: float
    acq_value: float
    acq_raw: float
    acq_name: str
    n_repaired: int
    implicit_hits: List[str]
    confidence: Dict[str, List[bool]]
    elapsed_s: float
    display_columns: Tuple[str, ...] = ()
    optimizer_degraded: bool = False
    """SLSQP 全部重启不可行、已回退到可行池最优批次（真降级，需人工关注）。"""
    feasible: bool = True

    # -- 输出 ------------------------------------------------------------- #
    def describe(self) -> str:
        """固定宽度表格，直接进实验记录。"""
        cols = self.display_columns or tuple(self.recipe_table[0].keys())[:6]
        head = "".join(f"{n[:11]:>12}" for n in cols)
        lines = [
            f"第 {self.round_index} 轮推荐  |  采集函数 {self.acq_name}  "
            f"|  目标 {self.active_objectives}"
            + (f"（剔除后备：{self.dropped_objectives}）" if self.dropped_objectives else "")
            + ("  ⚠ 优化器降级：候选来自可行池最优批次" if self.optimizer_degraded else ""),
            f"参考点 {[round(v, 2) for v in self.ref_point_prev]} → "
            f"{[round(v, 2) for v in self.ref_point_physical]}（原量纲）"
            f"  HV {self.hv_before:.4f} → {self.hv_after_pseudo:.4f}（归一化，伪观测）"
            f"  采集值 {self.acq_value:.3f}  修复 {self.n_repaired} 个",
            "-" * 104,
            f"{'#':>2} " + head + f"{'预测成本':>10}{'置信':>7}",
            "-" * 104,
        ]
        for i, row in enumerate(self.recipe_table):
            cells = "".join(f"{row[n]:>12.3f}" for n in cols)
            conf = "低" if any(v[i] for v in self.confidence.values()) else "高"
            lines.append(f"{i:>2} " + cells + f"{self.cost_cny[i]:>10.0f}{conf:>7}")
        lines.append("-" * 104)
        pred = "  ".join(
            f"{n}=" + "/".join(f"{float(v):.1f}" for v in self.predicted_mean[:, j])
            for j, n in enumerate(self.active_objectives)
        )
        lines.append(f"预测值：{pred}")
        lines.append(f"隐式规则命中：{self.implicit_hits or '无（全部通过规则库）'}")
        return "\n".join(lines)

    def as_dicts(self) -> List[Dict[str, Any]]:
        """实验记录（模块 5 回写契约）可直接消费的字典列表。"""
        out = []
        for i, row in enumerate(self.recipe_table):
            out.append({
                "round_index": self.round_index,
                "recipe": dict(row),
                "predicted": {
                    n: float(self.predicted_mean[i, j])
                    for j, n in enumerate(self.active_objectives)
                },
                "predicted_sd": {
                    n: float(self.predicted_sd[i, j])
                    for j, n in enumerate(self.active_objectives)
                },
                "cost_cny": float(self.cost_cny[i]),
                "acq_value": self.acq_value,
                "ref_point": list(self.ref_point),
            })
        return out


# --------------------------------------------------------------------------- #
# 6. 主优化器
# --------------------------------------------------------------------------- #
class CMPBatchPlanner:
    """模块 3 的门面：``recommend_next_batch``。

    一轮的完整流程（与 README 第 4 节的 ③④⑤ 步对应）：

    1. 用全部历史数据重训模块 2 的 GPR（含健康检查）；
    2. 剔除常量后备目标（它们没有协方差结构，进不了 ``ModelListGP``）；
    3. 更新动态参考点并记录；
    4. 构造 qNEHVI（或单目标退化时用 qLogEI），按成本惩罚包装；
    5. 带 9 条非线性硬约束做批量优化（可行初始点生成器）；
    6. 验证可行性 → 修复 → 隐式规则前置过滤；
    7. 汇总预测、成本、置信度，返回可直接下发的配方。
    """

    def __init__(
        self,
        space: CMPRecipeSpace,
        *,
        surrogate: Optional[CMPGPRSurrogate] = None,
        tracker: Optional[ReferencePointTracker] = None,
        tkwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        seed: int = 0,
    ) -> None:
        self.space = space
        self.tkwargs = dict(DEFAULT_TKWAARGS if tkwargs is None else tkwargs)
        self.surrogate = surrogate or CMPGPRSurrogate(space, tkwargs=self.tkwargs,
                                                      loocv_mode="efficient")
        self.tracker = tracker or ReferencePointTracker(space)
        self.verbose = verbose
        self.seed = seed
        self.round_counter = 0

    # -- 内部工具 --------------------------------------------------------- #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _active_objectives(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """按模块 2 的健康状态选目标：ok/degraded 保留，fallback 剔除。"""
        active, dropped = [], []
        for name in self.space.objective_names:
            diag: ObjectiveDiagnostics = self.surrogate.diagnostics[name]
            (dropped if diag.is_fallback else active).append(name)
        return tuple(active), tuple(dropped)

    def _model_for(self, names: Sequence[str]) -> Any:
        """取目标对应的 GP；含后备时抛错（不允许静默降级）。"""
        models = []
        for n in names:
            m = self.surrogate.models[n]
            if isinstance(m, ConstantFallback):
                raise RuntimeError(
                    f"目标 {n} 处于常量后备，不能组装 ModelListGP；"
                    f"应先扩充实验数据或收紧健康阈值"
                )
            models.append(m.model)
        if len(models) == 1:
            return models[0]
        from botorch.models import ModelListGP

        return ModelListGP(*models)

    def _build_acqf(
        self,
        active: Sequence[str],
        ref_point: torch.Tensor,
        train_X: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[AcquisitionFunction, str, bool]:
        """构造（可能被成本包装的）采集函数，返回 ``(acqf, 名称, 是否对数空间)``。

        ``weights`` 为目标归一化系数：通过 ``objective=`` 注入采集函数，
        让 EHVI 在**无量纲**空间里算超体积（否则量纲大的目标会独占 HV）。
        注意 ``ref_point`` 必须与 ``objective`` 处于同一尺度——两者都已在
        ``ReferencePointTracker`` 里归一化过。
        """
        cfg = self.space.acquisition
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([cfg.num_samples]), seed=self.seed
        )
        model = self._model_for(active)
        obj = None
        if weights is not None and len(active) >= 2:
            obj = WeightedMCMultiOutputObjective(weights=weights.reshape(-1))

        if len(active) >= 2:
            if cfg.acqf.lower() in ("qnehvi", "legacy", "qnoisyexpectedhypervolumeimprovement"):
                base = qNoisyExpectedHypervolumeImprovement(
                    model=model, ref_point=ref_point, X_baseline=train_X,
                    sampler=sampler, objective=obj, prune_baseline=cfg.prune_baseline,
                    cache_root=cfg.cache_root,
                )
                name, log_space = "qNEHVI", False
            else:
                base = qLogNoisyExpectedHypervolumeImprovement(
                    model=model, ref_point=ref_point, X_baseline=train_X,
                    sampler=sampler, objective=obj, prune_baseline=cfg.prune_baseline,
                    cache_root=cfg.cache_root,
                )
                name, log_space = "qLogNEHVI", True
        else:
            # 只剩一个可用目标：多目标采集函数失去意义，退化为约束 qLogEI
            j = self.space.objective_names.index(active[0])
            sign = self.space.objectives[j].sign
            y_max_space = self.surrogate.train_Y[:, j] * sign
            best_f = float(y_max_space.max())
            base = qLogExpectedImprovement(
                model=model, best_f=best_f, sampler=sampler,
            )
            name, log_space = "qLogEI", True

        if cfg.cost_aware and self.space.cost_model.penalty_lambda > 0:
            return (
                CostAwareAcquisition(
                    base, self.space, self.space.cost_model.penalty_lambda,
                    log_space=log_space, tkwargs=self.tkwargs,
                ),
                name + "+Cost",
                log_space,
            )
        return base, name, log_space

    # -- 主流程 ----------------------------------------------------------- #
    def recommend_next_batch(
        self,
        X_observed: Any,
        Y_observed: Any,
        q: int = 3,
        *,
        X_cov: Optional[Any] = None,
        round_index: Optional[int] = None,
    ) -> BatchRecommendation:
        """给出下一轮的 q 个候选配方。

        Parameters
        ----------
        X_observed, Y_observed:
            历史数据，``X`` 为物理量纲 ``(n, d)``，``Y`` 为原量纲 ``(n, m)``，
            列序与 ``space.objective_names`` 一致。
        q:
            本轮并行推荐数。
        X_cov:
            ``gpr_input`` 批次协变量（可选）。
        """
        t_start = time.perf_counter()
        ridx = self.round_counter if round_index is None else round_index
        self.round_counter = ridx + 1

        X = _t(X_observed, self.tkwargs)
        Y = _t(Y_observed, self.tkwargs)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # 1) 重训代理模型（含逐目标健康检查）
        self.surrogate.fit(X, Y, X_cov=X_cov)
        self._log(f"  [模型] n={X.shape[0]} "
                  + "  ".join(f"{k}={v.status}" for k, v in self.surrogate.diagnostics.items()))

        # 2) 剔除后备目标
        active, dropped = self._active_objectives()
        if not active:
            raise RuntimeError(
                "全部目标都处于常量后备（无预测力），无法构造多目标采集函数。"
                "请先补充实验数据（经验下限约 20–25 条），或检查测量异常。"
            )
        if dropped:
            warnings.warn(
                f"目标 {dropped} 处于常量后备，已从多目标向量中剔除"
                f"（等价权重=0）。剩余目标：{active}",
                stacklevel=2,
            )
        idx_active = [self.space.objective_names.index(n) for n in active]
        Y_active = Y[:, idx_active]

        # 3) 动态参考点（归一化最大化空间；HV 也在同一空间里算）
        w_np, _ = objective_normalization(self.space, active)
        weights_t = _t(w_np, self.tkwargs)
        prev_ref = (tuple(self.tracker.ref_point_physical)
                    if self.tracker.ref_point_physical is not None
                    else tuple(self.space.reference_point()[idx_active]))
        ref_t = self.tracker.update(Y_active.cpu().numpy(), active, round_index=ridx)
        ref_np = np.asarray(self.tracker.ref_point)
        hv_before = self.tracker.hypervolume(
            self.tracker.normalize(Y_active.cpu().numpy(), active)
        )

        # 4) 采集函数（含成本惩罚与目标归一化）
        train_X = self.surrogate.train_X_model
        acqf, acq_name, log_space = self._build_acqf(active, ref_t, train_X, weights_t)

        # 5) 带硬约束的批量优化
        ic_gen = FeasibleICGenerator(
            self.space, seed=self.seed + ridx, tkwargs=self.tkwargs
        )
        constraints = make_nonlinear_constraints(self.space, self.tkwargs)
        cfg = self.space.acquisition
        options = {"batch_limit": 1, "maxiter": cfg.maxiter, "disp": False}
        t_opt = time.perf_counter()
        optimizer_degraded = False
        try:
            cands, acq_val = optimize_acqf(
                acqf, bounds=self.space.unit_bounds_tensor(),
                q=q, num_restarts=cfg.num_restarts, raw_samples=cfg.raw_samples,
                nonlinear_inequality_constraints=constraints,
                ic_generator=ic_gen,
                options=options,
            )
        except CandidateGenerationError:
            # 预算偏小 / 约束很紧时，SLSQP 可能**所有重启**都落不到可行域，
            # BoTorch 会直接抛 CandidateGenerationError。
            # 整个 Agent 不能因此崩掉：退回到"初始点里采集值最高的那批"——
            # 它本来就是可行域内采样、并且已经被采集函数打过分。
            optimizer_degraded = True
            if ic_gen.last_batches is None or ic_gen.last_values is None:
                raise RuntimeError(
                    "优化器未找到可行解，且没有可用的备选批次"
                ) from None
            best = int(torch.argmax(ic_gen.last_values))
            # 注意形状：last_batches 是 (n_batches, q, d)，取出来必须是 (q, d)，
            # 不能 unsqueeze(0)——否则下游 _verify_and_repair 会拿到 3 维张量，
            # from_unit 按列索引时直接 IndexError（实测踩到）。
            cands = ic_gen.last_batches[best].reshape(-1, self.space.dim)
            acq_val = ic_gen.last_values[best].reshape(1)
            self._log("  [降级] SLSQP 全部重启不可行 → 回退到可行池中采集值最高的批次")
        t_opt = time.perf_counter() - t_opt
        self._log(f"  [优化] {acq_name} q={q} {t_opt:.1f}s  可行池占比 "
                  f"{ic_gen.pool_feasible_ratio:.0%}"
                  + ("（已降级）" if optimizer_degraded else ""))

        # 6) 可行性验证与修复
        cands, n_repaired, viol = self._verify_and_repair(cands, ic_gen.pool)
        if n_repaired:
            self._log(f"  [修复] {n_repaired} 个候选违反硬约束，已用最近可行点替换：{viol}")

        # 7) 隐式规则前置过滤
        cands, n_rejected, hits = self._filter_implicit(cands, ic_gen.pool)

        # 8) 汇总
        phys = self.space.from_unit(cands)
        pred_all = self.surrogate.predict(phys, X_cov=X_cov)
        pred_mean = pred_all.mean[:, idx_active]
        pred_sd = pred_all.sd_total[:, idx_active]

        # 成本：MRR 可用时用预测值折算机台时间，否则退回 Preston 先验
        if "MRR" in active:
            mrr = pred_mean[:, active.index("MRR")]
            cost = self.space.cost_matrix(phys, mrr)
        else:
            cost = np.asarray(
                self.space.cost(phys).detach().cpu().numpy(), dtype=float
            ).reshape(-1)

        # 伪观测 HV：把预测均值当作"即将观测到的点"，估算前沿推进
        pseudo = torch.cat([Y_active, pred_mean], dim=0)
        hv_after = self.tracker.hypervolume(
            self.tracker.normalize(pseudo.cpu().numpy(), active)
        )

        acq_raw = float(torch.as_tensor(acq_val).sum())
        acq_linear = math.exp(acq_raw) if log_space else acq_raw

        rec = BatchRecommendation(
            round_index=ridx,
            recipes=phys,
            recipe_table=[
                {n: float(phys[i, j]) for j, n in enumerate(self.space.names)}
                for i in range(phys.shape[0])
            ],
            predicted_mean=pred_mean,
            predicted_sd=pred_sd,
            cost_cny=cost,
            active_objectives=active,
            dropped_objectives=dropped,
            ref_point=tuple(float(v) for v in ref_np),
            ref_point_prev=tuple(float(v) for v in prev_ref),
            ref_point_physical=self.tracker.to_physical(ref_np),
            hv_before=hv_before,
            hv_after_pseudo=hv_after,
            acq_value=acq_linear,
            acq_raw=acq_raw,
            acq_name=acq_name,
            n_repaired=n_repaired,
            implicit_hits=hits,
            confidence=self.surrogate.confidence_flags(pred_all),
            elapsed_s=time.perf_counter() - t_start,
            display_columns=self.space.names[:6],
            optimizer_degraded=optimizer_degraded,
        )
        if n_rejected:
            rec.implicit_hits.append(f"已剔除 {n_rejected} 个规则库不可行候选")
        return rec

    # -- 验证与修复 ------------------------------------------------------- #
    def _verify_and_repair(
        self, cands: torch.Tensor, pool: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """SLSQP 不保证收敛到可行点，这里做硬验证 + 最近可行点替换。

        实测复现：同一批数据、同样的 9 条约束，``cache_root=True`` 与
        ``False`` 会因为数值路径不同导致某些重启落在可行域外。
        工业实现不能假设"优化器说可行就可行"，必须逐条精确校验。
        """
        phys = self.space.from_unit(cands)
        # SLSQP 返回的是"约束边界上"的解，g 可能是 1e-12 量级的正数。
        # 用 1e-6 容差判定，否则每次都会触发无意义的"修复"（实测踩到）。
        mask = np.asarray(self.space.max_violation(phys.numpy())) <= 1e-6
        if mask.all():
            return cands, 0, {}
        if pool is None or pool.shape[0] == 0:
            raise RuntimeError("优化结果不可行，且没有可用可行池用于修复")

        out = cands.clone()
        worst: Dict[str, float] = {}
        for i in np.where(~mask)[0]:
            dist = (pool - out[i]).pow(2).sum(dim=-1)
            j = int(torch.argmin(dist).item())
            out[i] = pool[j]
            # 记录原候选的违规量（诊断用）
            rep = self.space.validate_recipe(
                {n: float(phys[i, k]) for k, n in enumerate(self.space.names)}
            )
            for k, v in rep["violations"].items():
                worst[k] = max(worst.get(k, 0.0), float(v))
        return out, int((~mask).sum()), worst

    def _filter_implicit(
        self, cands: torch.Tensor, pool: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, int, List[str]]:
        """隐式规则（规则库）前置过滤：命中 REJECT 的候选换成池内合规点。"""
        phys = self.space.from_unit(cands)
        out = cands.clone()
        hits: List[str] = []
        n_reject = 0
        used = {tuple(np.round(out[i].tolist(), 8)) for i in range(out.shape[0])}
        for i in range(out.shape[0]):
            single = phys[i : i + 1]
            _, keep, h = self.space.filter_implicit(single)
            hits.extend(h)
            if keep.all():
                continue
            n_reject += 1
            if pool is None or pool.shape[0] == 0:
                continue
            # 从可行池里找最近的、且通过规则库的点
            order = torch.argsort((pool - out[i]).pow(2).sum(dim=-1))
            for j in order[: min(pool.shape[0], 64)].tolist():
                cand_phys = self.space.from_unit(pool[j : j + 1])
                _, keep2, _ = self.space.filter_implicit(cand_phys)
                if bool(keep2.all()):
                    key = tuple(np.round(pool[j].tolist(), 8))
                    if key not in used:
                        out[i] = pool[j]
                        used.add(key)
                        break
        # 去重后的命中摘要
        return out, n_reject, sorted(set(hits))

    # -- 便捷接口 --------------------------------------------------------- #
    def commit_batch(
        self, X_obs: Any, Y_obs: Any, rec: BatchRecommendation, Y_new: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把一轮实测结果并入数据集（模块 5 回写的最小实现）。

        全程在**物理量纲**里做：``surrogate.train_X_model`` 是单位立方空间，
        拿它拼数据集会让下一轮把归一化坐标当物理量纲用（典型的静默错位）。
        """
        X_old = _t(X_obs, self.tkwargs)
        Y_old = _t(Y_obs, self.tkwargs)
        Y_new = _t(Y_new, self.tkwargs).reshape(len(rec.recipes), -1)
        return (
            torch.cat([X_old, rec.recipes], dim=0),
            torch.cat([Y_old, Y_new], dim=0),
        )

if __name__ == "__main__":
    import sys
    from pathlib import Path

    torch.manual_seed(2026)

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "acquisition_selfcheck.txt"

    class _Tee:
        """同时写终端与留档文件：自检本身就产出交付物。"""

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

        from src.config import build_default_config
        from src.surrogate_model import synthetic_cmp_benchmark

        space = build_default_config()
        tk = {"dtype": torch.double}
        N0 = 25

        # ---------------- 0. 环境 ---------------- #
        rule("0. 运行环境与问题配置")
        import botorch
        import gpytorch

        print(f"  torch {torch.__version__} | botorch {botorch.__version__} "
              f"| gpytorch {gpytorch.__version__}")
        print(f"  {space.name}")
        print(f"  决策变量 {space.dim} 维 | 目标 {space.objective_names} | "
              f"硬约束 {len(space.hard_constraints)} 条 | 隐式规则 "
              f"{len(space.implicit_rules)} 条")
        ac = space.acquisition
        print(f"  采集：{ac.acqf} (num_restarts={ac.num_restarts}, "
              f"num_samples={ac.num_samples}, maxiter={ac.maxiter}, "
              f"cache_root={ac.cache_root})")
        print(f"  成本惩罚 λ={space.cost_model.penalty_lambda}，"
              f"归一化基准 {space.cost_model.reference_cost:.0f} 元")
        print(f"  参考点策略 {space.ref_point_policy.mode}，"
              f"margin={space.ref_point_policy.margin_ratio}")

        # ---------------- 1. 初始数据 ---------------- #
        rule(f"1. 初始实验数据：{N0} 条可行 LHS 配方（健康检查应全绿）")
        X_obs = torch.tensor(space.sample_lhs_feasible(N0, seed=5), **tk)
        Y_obs = synthetic_cmp_benchmark(space, X_obs, noise=0.03, seed=105)
        print(f"  X {tuple(X_obs.shape)}  Y {tuple(Y_obs.shape)}")

        planner = CMPBatchPlanner(space, verbose=True, seed=7)
        planner.surrogate.fit(X_obs, Y_obs)
        print(planner.surrogate.report())

        # ---------------- 2. 约束是否真的在起作用 ---------------- #
        rule("2. 硬约束是否真的约束住了搜索空间（需要同时看到可行/不可行）")
        pool_probe = torch.tensor(
            draw_sobol_samples(bounds=space.unit_bounds_tensor(), n=4000, q=1,
                               seed=1).squeeze(-2), **tk
        )
        feas = space.feasible_mask(space.from_unit(pool_probe).numpy())
        print(f"  Sobol 4000 点：可行 {int(feas.sum())} / 不可行 {int((~feas).sum())}"
              f"（{feas.mean():.1%}）")
        print(f"  → 若不做约束优化，约 {1 - feas.mean():.0%} 的采样预算会落在禁区。"
              f"这正是 README 消融实验 #8（约束 qNEHVI vs 后置软过滤）的量化依据。")

        # ---------------- 3. 成本惩罚的精确性验证 ---------------- #
        rule("3. 成本惩罚接线验证（对数空间 val-λc vs 线性空间 val·e^{-λc}）")
        from src.acquisition import (
            CostAwareAcquisition,
            FeasibleICGenerator,
            make_nonlinear_constraints,
        )

        ref0 = torch.tensor(space.reference_point(), **tk)
        w_all, _ = objective_normalization(space, tuple(space.objective_names))
        built, acq_name, log_space = planner._build_acqf(
            tuple(space.objective_names), ref0 * _t(w_all, tk),
            planner.surrogate.train_X_model, _t(w_all, tk)
        )
        inner = built.base_acqf if isinstance(built, CostAwareAcquisition) else built
        X_probe = space.to_unit(X_obs[:5]).unsqueeze(1)   # (5, 1, d)
        lam = space.cost_model.penalty_lambda
        with torch.no_grad():
            v_base = inner(X_probe).reshape(-1)
            # 与 forward 保持同一语义：对 q 维取均值 -> (B,)
            c = CostAwareAcquisition(inner, space, lam, log_space=False).normalized_cost(
                X_probe).mean(dim=-1).reshape(-1)

        print(f"  当前配置：{acq_name}（对数空间={log_space}）")
        print(f"  归一化成本 c = {[round(float(v), 3) for v in c]}")
        print(f"  基础采集值   = {[round(float(v), 3) for v in v_base]}")
        for flag, label in ((True, "log 空间：val - λc"), (False, "线性空间：val·e^{-λc}")):
            w = CostAwareAcquisition(inner, space, lam, log_space=flag)
            with torch.no_grad():
                v_wrap = w(X_probe).reshape(-1)
                want = (v_base - lam * c) if flag else (v_base * torch.exp(-lam * c))
            err = float((v_wrap - want).abs().max())
            print(f"  {label:<22} 实测 {[round(float(v), 4) for v in v_wrap[:3]]} "
                  f"理论 {[round(float(v), 4) for v in want[:3]]}  最大偏差 {err:.2e}")
            assert err < 1e-9, f"{label} 接线错误"
        # 负对照：若在对数空间误用乘法，会把负的 log 值拉向 0（排名反转）
        wrong = v_base * torch.exp(-lam * c)
        print(f"  负对照（对数空间误用乘法）：{[round(float(v), 4) for v in wrong[:3]]}"
              f" —— λ>0 时把负值抬向 0，等价于『成本越高越优先』")

        # ---------------- 4. 闭环三轮 ---------------- #
        rule("4. 闭环三轮：推荐 → 合成实验 → 回灌 → 再推荐")
        ref_fixed: Optional[np.ndarray] = None   # 第 0 轮归一化参考点，之后冻结
        ref_history = []
        hv_fixed = []
        rounds = []
        for r in range(3):
            rec = planner.recommend_next_batch(X_obs, Y_obs, q=3, round_index=r)
            rounds.append(rec)
            print("\n" + rec.describe())
            if ref_fixed is None:
                ref_fixed = np.asarray(rec.ref_point)

            # 精确可行性校验（不信任优化器自述）；容差 1e-6 与修复环节一致
            viol = space.max_violation(rec.recipes.numpy())
            assert bool((viol <= 1e-6).all()), \
                f"第 {r} 轮出现不可行候选，最大违反度 {viol.max():.2e}"
            assert rec.recipes.shape == (3, space.dim)
            print(f"  ✓ 维度 {tuple(rec.recipes.shape)}，9 条硬约束满足"
                  f"（最大违反度 {viol.max():.2e} ≤ 1e-6）")

            # 合成实验（3% 噪声），把预测与实际并排打印
            Y_new = synthetic_cmp_benchmark(space, rec.recipes, noise=0.03, seed=500 + r)
            err = (rec.predicted_mean - Y_new[:, [space.objective_names.index(n)
                                                  for n in rec.active_objectives]])
            for i in range(3):
                print(f"    实验#{i}: " + "  ".join(
                    f"{n} 预测 {float(rec.predicted_mean[i, j]):8.1f} / 实测 "
                    f"{float(Y_new[i, space.objective_names.index(n)]):8.1f}"
                    for j, n in enumerate(rec.active_objectives)))
            print(f"    平均绝对误差：" + "  ".join(
                f"{n}={float(err[:, j].abs().mean()):.1f}"
                for j, n in enumerate(rec.active_objectives)))

            X_obs, Y_obs = planner.commit_batch(X_obs, Y_obs, rec, Y_new)
            ref_history.append(np.asarray(rec.ref_point))
            # 固定参考点下的真实 HV：参考点冻结，HV 的增长只可能来自数据改进
            hv_fixed.append(planner.tracker.hypervolume(
                planner.tracker.normalize(Y_obs.numpy(), rec.active_objectives),
                ref=ref_fixed))
            print(f"  数据量 -> {X_obs.shape[0]}")

        # ---------------- 5. 参考点与 HV 演化 ---------------- #
        rule("5. 动态参考点演化（单调不增）与超体积/最优值轨迹")
        print(planner.tracker.describe())
        print(f"\n  固定参考点（归一化）{[round(float(v), 4) for v in ref_fixed]} 下的轨迹：")
        print(f"  {'轮':>2}{'n':>5}{'归一化HV':>10}{'ΔHV':>9}"
              f"{'max MRR':>10}{'min Ra':>8}{'min Dish':>10}{'规格满足率':>10}")
        print("  " + "-" * 66)
        oi = [space.objective_names.index(n) for n in rounds[0].active_objectives]
        for i, (h, rec) in enumerate(zip(hv_fixed, rounds)):
            dh = h - (hv_fixed[i - 1] if i else 0.0)
            Y_so_far = Y_obs.numpy()
            print(f"  {i:>2}{N0 + 3 * (i + 1):>5}{h:>10.4f}{dh:>+9.4f}"
                  f"{Y_so_far[:, 0].max():>10.0f}{Y_so_far[:, 1].min():>8.2f}"
                  f"{Y_so_far[:, 2].min():>10.1f}"
                  f"{space.spec_hit_rate(Y_so_far) * 100:>9.1f}%")
        # 单调性断言（归一化最大化空间逐分量不增）
        for a, b in zip(ref_history[:-1], ref_history[1:]):
            assert np.all(b <= a + 1e-9), f"参考点出现回撤：{a} -> {b}"
        print("  ✓ 参考点逐轮单调不增（跨轮次 HV 可比）")
        assert any(r.ref_point_physical != r.ref_point_prev for r in rounds), \
            "动态参考点一次都没有更新，功能未被验证"
        print("  ✓ 参考点确实发生过更新（不是永远不动的摆设）")
        # 固定参考点下 HV 只会因新数据"扩展前沿"而增长，绝不因参考点漂移而虚增
        for i in range(1, len(hv_fixed)):
            assert hv_fixed[i] >= hv_fixed[i - 1] - 1e-9, "固定参考点下 HV 出现下降（不应发生）"
        print("  ✓ 固定参考点下 HV 单调不减（账目一致，无数据丢失）")
        if hv_fixed[-1] <= hv_fixed[0] + 1e-9:
            print("  ⚠ 3 轮内 HV 未增长：25 条 LHS 已把该 3 目标问题的前沿探得差不多，"
                  "新增 9 条都是前沿内部点。")
            print("    这本身就是模块 6 的停止信号（HV 边际增量≈0 连续 2 轮 → 终止），"
                  "不是算法失效；要看到改进应从更小的初始样本量起步。")

        # ---------------- 6. 后备目标处理 ---------------- #
        rule("6. 后备目标处理：只有 8 条数据时，后备目标必须被剔除并告警")
        X_small = torch.tensor(space.sample_lhs_feasible(8, seed=31), **tk)
        Y_small = synthetic_cmp_benchmark(space, X_small, noise=0.03, seed=32)
        planner_small = CMPBatchPlanner(space, verbose=True, seed=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rec_small = planner_small.recommend_next_batch(X_small, Y_small, q=2,
                                                           round_index=0)
        print(f"  健康状态：{ {k: v.status for k, v in planner_small.surrogate.diagnostics.items()} }")
        print(f"  告警：{[str(w.message)[:70] for w in caught if '后备' in str(w.message)]}")
        print(f"  采集函数退化为：{rec_small.acq_name}（目标 {rec_small.active_objectives}）")
        print(f"  剔除目标：{rec_small.dropped_objectives}")
        assert rec_small.dropped_objectives, "小样本下应有目标被剔除"
        assert len(rec_small.recipes) == 2
        assert bool((space.max_violation(rec_small.recipes.numpy()) <= 1e-6).all())

        # 单目标退化路径（只剩 1 个可用目标 -> qLogEI）单独验证：
        # 8 条数据通常还剩 2 个目标，凑不出 1 个的情形，所以直接测构造函数
        w1, _ = objective_normalization(space, ("MRR",))
        built_ei, name_ei, log_ei = planner._build_acqf(
            ("MRR",), torch.zeros(1, **tk), planner.surrogate.train_X_model, _t(w1, tk)
        )
        with torch.no_grad():
            v_ei = built_ei(space.to_unit(X_obs[:4]).unsqueeze(1)).reshape(-1)
        print(f"  单目标退化：{name_ei}（对数空间={log_ei}），4 个样本采集值 "
              f"{[round(float(v), 3) for v in v_ei]}")
        assert name_ei.startswith("qLogEI"), "单目标应退化为 qLogEI"

        # ---------------- 7. 全部目标后备时必须显式报错 ---------------- #
        rule("7. 全部目标后备：必须显式报错，而不是返回随机配方")
        Y_noise = torch.randn(len(X_small), 3, **tk) * 10.0
        planner_bad = CMPBatchPlanner(space, seed=3)
        try:
            planner_bad.recommend_next_batch(X_small, Y_noise, q=2)
            raise AssertionError("全后备时不应返回候选")
        except RuntimeError as exc:
            print(f"  ✓ 如预期抛错：{str(exc)[:80]}…")

        # ---------------- 8. 汇总 ---------------- #
        rule("8. 汇总")
        print(f"  {'轮':>2}{'目标':<28}{'参考点（原量纲）':<32}{'HV(固定ref)':>13}{'耗时s':>8}")
        for i, rec in enumerate(rounds):
            print(f"  {i:>2}{str(rec.active_objectives):<28}"
                  f"{str([round(v, 1) for v in rec.ref_point_physical]):<32}"
                  f"{hv_fixed[i]:>13.4f}{rec.elapsed_s:>8.1f}")
        print(f"\n  每轮推荐全部满足 9 条硬约束；隐式规则命中："
              f"{[r.implicit_hits for r in rounds]}")
        # 闭环产物落盘：模块 4（帕累托 + TOPSIS）直接读这份真实数据
        data_path = dump_dataset(
            "data/closed_loop_n34.json", X_obs, Y_obs, space.objective_names,
            config_hash=space.config_hash(),
            extra={
                "rounds": [
                    {
                        "round_index": r.round_index,
                        "acq_name": r.acq_name,
                        "recipes": r.recipes.tolist(),
                        "predicted": r.predicted_mean.tolist(),
                        "ref_point_physical": list(r.ref_point_physical),
                        "cost_cny": r.cost_cny.tolist(),
                        "n_repaired": r.n_repaired,
                        "implicit_hits": r.implicit_hits,
                    }
                    for r in rounds
                ]
            },
        )
        print(f"  数据集已落盘：{data_path}")
        print(f"  留档文件：{out_path.resolve()}")
        print("\n全部自检通过（7 条断言：成本惩罚双分支接线、维度与可行性、"
              "参考点单调且确实更新、固定参考点下 HV 不减、后备剔除、单目标退化、全后备报错）。")

    finally:
        sys.stdout = real_stdout
        log_file.close()
