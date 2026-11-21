"""CMP 抛光液配方开发 Agent —— 模块 1：问题定义与参数空间配置。

本模块是整个闭环 Agent 的"初始化层"，把 CMP 配方研发需求翻译成 BoTorch
可以直接消费的数据结构：

* 决策变量（``DecisionVariable``）：上下限、类型、单位、机理注释；
* 优化目标（``Objective``）：方向、产线规格、TOPSIS 冻结权重、BoTorch 符号；
* 确定性硬约束（``HardConstraint``）：解析表达式，直接进 Constrained qNEHVI；
* 隐式约束规则库（``ImplicitRule``）：无解析式，走采样前置过滤；
* 批次协变量（``BatchCovariate``）：显式建模为 GPR 额外输入维度或随机效应；
* 成本模型（``CostModel``）：组分/晶圆/工时成本，供成本感知采样与 TOPSIS 成本维度；
* 实验与采集配置（``ExperimentPlan`` / ``AcquisitionConfig`` / ``ReferencePointPolicy``）；
* 代理模型先验（``GPRPriorConfig``）：交给模块 2 训练 GPR 时使用。

设计约定（务必遵守，否则模块 2/3/4 会静默不一致）
--------------------------------------------------
1. **变量顺序即列顺序**：``space.variables`` 的顺序就是设计矩阵 ``X`` 的列顺序；
   所有约束、成本、编码函数都按 ``space.index`` 取列，不要自己写死下标。
2. **约束符号约定**：所有硬约束函数返回形状 ``(n,)`` 的"违反度"，``<= 0`` 表示可行。
   这与 BoTorch 的 ``ConstrainedMCObjective`` 一致：``feasible = (g(X) <= 0).all()``。
3. **目标符号约定**：BoTorch 内部统一按"最大化"处理，故 ``Objective.sign``
   把最小化指标取负（Ra、Dishing）。参考点（ref_point）也必须在同一符号空间里算。
4. **约束只放解析式**：能写成闭式不等式的才进采集函数；浆料长期稳定性、团聚、
   发泡这类没有可靠解析式的，一律进规则库（``implicit_rules``），不进优化器。
5. **TOPSIS 权重冻结**：``TopsisConfig.weights`` 初始化后不再随轮次调整，
   熵权仅用于敏感性对照（``entropy_weight_sensitivity_only``），避免权重漂移污染决策。

BoTorch 最小接入示例
--------------------
    >>> from src.config import build_default_config
    >>> import torch
    >>> cfg = build_default_config()
    >>> bounds = cfg.bounds_tensor(dtype=torch.double)           # (2, d) 物理量纲
    >>> bounds_unit = cfg.unit_bounds_tensor(dtype=torch.double) # (2, d) 单位立方
    >>> cons = cfg.constraint_callables()                        # [callable(X)->(n,)]
    >>> ref = cfg.reference_point(dtype=torch.double)            # (m,) 最大化符号空间
    >>> X_unit = cfg.to_unit(recipes_physical)                   # 物理 -> 单位立方

    # 送入 BoTorch 的典型写法
    # from botorch.acquisition.multi_objective.monte_carlo import (
    #     qNoisyExpectedHypervolumeImprovement,
    # )
    # acqf = qNoisyExpectedHypervolumeImprovement(
    #     model=model, ref_point=ref, X_baseline=train_X_unit,
    #     constraints=cons, prune_baseline=cfg.acquisition.prune_baseline,
    # )
    # candidates, _ = optimize_acqf(acqf, bounds=cfg.unit_bounds_tensor(),
    #                               q=cfg.experiment.q, num_restarts=24, raw_samples=512)

版本：见 ``CMPRecipeSpace.version`` 与 ``config_hash()``，供模块 8 审计。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # torch 是可选依赖：做纯配置检查 / 装机校验时不需要它
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


__all__ = [
    "VariableKind",
    "Scale",
    "Direction",
    "ConstraintLogic",
    "Severity",
    "DecisionVariable",
    "Objective",
    "HardConstraint",
    "ImplicitRule",
    "BatchCovariate",
    "CostModel",
    "ExperimentPlan",
    "AcquisitionConfig",
    "ReferencePointPolicy",
    "StoppingCriteria",
    "TopsisCriterion",
    "TopsisConfig",
    "ModelHealthConfig",
    "PrestonPrior",
    "GPRPriorConfig",
    "CMPRecipeSpace",
    "FAMILY_PRESETS",
    "ABRASIVE_PRICES",
    "RecordSchema",
    "build_config",
    "build_default_config",
]


# --------------------------------------------------------------------------- #
# 0. 后端无关的小工具（同一份代码同时吃 numpy 与 torch）
# --------------------------------------------------------------------------- #
def _is_torch(x: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def _ensure_2d(x: Any) -> Any:
    """单条配方 ``(d,)`` 自动升成 ``(1, d)``，避免约束函数取列时形状错乱。"""
    if getattr(x, "ndim", 1) == 1:
        return x.reshape(1, -1)
    return x


def _ewmin(*terms: Any) -> Any:
    """逐元素取最小；用于 ``ANY_OF``（OR）约束组，可行域为各分支并集。

    注意：torch 的 ``minimum/maximum`` **不接受 Python 标量**
    （``TypeError: argument 'other' must be Tensor, not float``），
    这里统一把标量提升成同 dtype/device 的张量。模块 1 只跑 numpy 时
    不会暴露这个问题，一旦张量流经成本模型（模块 3 的梯度计算）就会炸。
    """
    out = terms[0]
    for t in terms[1:]:
        if _is_torch(out):
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, dtype=out.dtype, device=out.device)
            out = torch.minimum(out, t)
        else:
            out = np.minimum(out, t)
    return out


def _ewmax(*terms: Any) -> Any:
    out = terms[0]
    for t in terms[1:]:
        if _is_torch(out):
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, dtype=out.dtype, device=out.device)
            out = torch.maximum(out, t)
        else:
            out = np.maximum(out, t)
    return out


class _NamedView:
    """按变量名取列的只读视图，避免在约束/成本函数里写死列号。

    ``v["ph"]`` 返回形状 ``(n,)`` 的一列；同时支持 numpy 数组与 torch 张量。
    """

    __slots__ = ("_idx", "_x")

    def __init__(self, names: Sequence[str], x: Any) -> None:
        self._idx = {n: i for i, n in enumerate(names)}
        self._x = _ensure_2d(x)

    def __getitem__(self, name: str) -> Any:
        try:
            return self._x[:, self._idx[name]]
        except KeyError as exc:  # pragma: no cover - 拼错变量名时快速失败
            raise KeyError(f"未知变量 {name!r}；可用变量：{sorted(self._idx)}") from exc

    def __len__(self) -> int:
        return self._x.shape[0]


# --------------------------------------------------------------------------- #
# 1. 枚举与决策变量
# --------------------------------------------------------------------------- #
class VariableKind(str, Enum):
    """变量类型。类别变量不进连续采集函数，见 ``CMPRecipeSpace.contextual_variables``。"""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


class Scale(str, Enum):
    """归一化尺度。跨数量级的浓度建议用 ``LOG``（如表面活性剂 0.01–0.5 wt%）。"""

    LINEAR = "linear"
    LOG = "log"


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintLogic(str, Enum):
    """约束组逻辑。``ANY_OF`` 用于"物理上互斥的多个稳定窗口"（并集可行域）。"""

    ALL_OF = "all_of"
    ANY_OF = "any_of"


class Severity(str, Enum):
    REJECT = "reject"  # 直接丢弃候选
    WARN = "warn"  # 保留但打标，交由工程师确认
    FLAG = "flag"  # 仅记录，不影响推荐


@dataclass(frozen=True)
class DecisionVariable:
    """一个决策变量（配方组分或工艺参数）。

    Attributes
    ----------
    name:
        英文列名，同时是设计矩阵列名与约束里的键，必须保持稳定。
    unit:
        物理单位，仅用于展示与日志（不做换算）。
    low, high:
        物理量纲上下限（闭区间）。
    kind:
        ``CONTINUOUS`` / ``INTEGER`` / ``CATEGORICAL``。
    scale:
        ``LINEAR`` 或 ``LOG``；``LOG`` 要求 ``low > 0``。
    categories:
        仅 ``CATEGORICAL`` 使用。
    mechanism:
        该变量对 CMP 性能的机理影响，用于特征重要性对照与工程师复核。
    """

    name: str
    unit: str
    low: float
    high: float
    kind: VariableKind = VariableKind.CONTINUOUS
    scale: Scale = Scale.LINEAR
    categories: Tuple[str, ...] = ()
    mechanism: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind is VariableKind.CATEGORICAL:
            if len(self.categories) < 2:
                raise ValueError(f"{self.name}: 类别变量至少需要 2 个水平")
            object.__setattr__(self, "low", 0.0)
            object.__setattr__(self, "high", float(len(self.categories) - 1))
        if not (self.low < self.high):
            raise ValueError(f"{self.name}: 需要 low < high，得到 ({self.low}, {self.high})")
        if self.scale is Scale.LOG and self.low <= 0:
            raise ValueError(f"{self.name}: LOG 尺度要求 low > 0，得到 low={self.low}")

    @property
    def span(self) -> float:
        return self.high - self.low

    @property
    def n_columns(self) -> int:
        """在模型输入矩阵中占用的列数（类别变量按 one-hot 展开）。"""
        if self.kind is VariableKind.CATEGORICAL:
            return len(self.categories)
        return 1

    # -- 归一化 ----------------------------------------------------------- #
    def to_unit(self, values: Any) -> Any:
        """物理量纲 -> [0, 1]（``LOG`` 尺度先取对数再线性归一）。"""
        v = values
        if self.scale is Scale.LOG:
            eps = 1e-12
            if _is_torch(v):
                v = v.clamp_min(eps)
                return (torch.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
            v = np.maximum(v, eps)
            return (np.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / self.span

    def from_unit(self, u: Any) -> Any:
        """[0, 1] -> 物理量纲；``INTEGER`` 取整，越界裁剪到上下限。"""
        if self.scale is Scale.LOG:
            lo, hi = math.log(self.low), math.log(self.high)
            if _is_torch(u):
                v = torch.exp(lo + u.clamp(0.0, 1.0) * (hi - lo))
            else:
                v = np.exp(lo + np.clip(u, 0.0, 1.0) * (hi - lo))
        else:
            if _is_torch(u):
                v = self.low + u.clamp(0.0, 1.0) * self.span
            else:
                v = self.low + np.clip(u, 0.0, 1.0) * self.span
        if self.kind is VariableKind.INTEGER:
            v = torch.round(v) if _is_torch(v) else np.round(v)
        return v


# --------------------------------------------------------------------------- #
# 2. 优化目标
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Objective:
    """一个待优化指标。

    注意 ``sign``：BoTorch 的多目标采集函数只认"越大越好"，因此最小化指标
    需要取负后再送进去；``prior_range`` 一律按原量纲给，换算在参考点里统一处理。
    """

    name: str
    unit: str
    direction: Direction
    spec: float
    """产线规格阈值；``MAXIMIZE`` 时要求 ``y >= spec``，``MINIMIZE`` 时 ``y <= spec``。"""
    prior_range: Tuple[float, float]
    """先验可达范围（原量纲），用于初始化 ref_point 与域外判定的尺度参考。"""
    topsis_role: str = "benefit"
    """TOPSIS 属性类型：``benefit`` 收益型 / ``cost`` 成本型。"""
    description: str = ""
    mechanism: str = ""

    @property
    def sign(self) -> float:
        """BoTorch 最大化空间符号：最大化 +1，最小化 -1。"""
        return 1.0 if self.direction is Direction.MAXIMIZE else -1.0

    def to_botorch(self, y: Any) -> Any:
        """原量纲实测值 -> BoTorch 最大化符号空间。"""
        return y * self.sign

    def from_botorch(self, y: Any) -> Any:
        """BoTorch 最大化符号空间 -> 原量纲。"""
        return y * self.sign

    def is_on_spec(self, y: Any) -> Any:
        """逐元素判断是否满足产线规格，返回 bool 数组/张量。"""
        if self.direction is Direction.MAXIMIZE:
            return y >= self.spec
        return y <= self.spec


# --------------------------------------------------------------------------- #
# 3. 约束
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HardConstraint:
    """确定性硬约束（有解析式，进 Constrained qNEHVI）。

    ``fn`` 接收 :class:`_NamedView`，返回形状 ``(n,)`` 的违反度，``<= 0`` 可行。
    ``expression`` 是给人看的数学式，会写进版本快照供审计。
    """

    name: str
    expression: str
    fn: Callable[[_NamedView], Any]
    logic: ConstraintLogic = ConstraintLogic.ALL_OF
    members: Tuple[str, ...] = ()
    description: str = ""
    rationale: str = ""

    def bind(self, names: Sequence[str]) -> Callable[[Any], Any]:
        """绑定列序，返回可直接喂给 BoTorch 的 ``callable(X) -> (n,)``。"""

        def _bound(x: Any) -> Any:
            return self.fn(_NamedView(names, x))

        _bound.__name__ = f"constraint__{self.name}"
        return _bound


@dataclass(frozen=True)
class ImplicitRule:
    """隐式约束规则（无解析式，采样后前置过滤，不进优化器）。

    ``violates`` 返回 ``True`` 的位置即为命中规则的候选。
    """

    name: str
    violates: Callable[[_NamedView], Any]
    severity: Severity = Severity.REJECT
    action: str = ""
    rationale: str = ""

    def bind(self, names: Sequence[str]) -> Callable[[Any], Any]:
        def _bound(x: Any) -> Any:
            return self.violates(_NamedView(names, x))

        _bound.__name__ = f"rule__{self.name}"
        return _bound


# --------------------------------------------------------------------------- #
# 4. 批次协变量
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatchCovariate:
    """批次协变量：不与配方耦合、但会污染响应的系统性因素。

    处理方式三选一：

    * ``gpr_input``：作为 GPR 的额外输入维度（推荐，能学出漂移方向）；
    * ``random_effect``：批次随机效应（批次内归一化 / 偏移项）；
    * ``metadata``：只记录不建模。
    """

    name: str
    unit: str
    handling: str = "gpr_input"
    description: str = ""
    typical_range: Tuple[float, float] = (0.0, 1.0)
    batch_constant: bool = True
    """该协变量在同一批次内是否应当恒定。

    * ``True``（环境温湿度、机台号、抛光垫累计片数）→ 批次级变量，
      同一 ``batch_id`` 内出现两个不同值就说明台账有问题；
    * ``False``（``slurry_age_hours``：配液到上机的时间）→ 天然逐条变化，
      同一批次内不同记录本来就该不一样，不能拿来做一致性校验。
    """

    def __post_init__(self) -> None:
        if self.handling not in {"gpr_input", "random_effect", "metadata"}:
            raise ValueError(f"{self.name}: 未知 handling={self.handling!r}")


# --------------------------------------------------------------------------- #
# 5. 成本模型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostModel:
    """单次实验成本模型（币种统一为 CNY）。

    成本 = 浆料配制 + 耗材 + 晶圆摊销 + 计量 + 机台工时 + 人工。

    **成本的主导项是产能而不是药剂**：在铜 CMP 体系里，一次实验的浆料成本
    只有几十元，而 300 mm 图形片摊销 + CMP 机台工时 + 计量合计在两千元量级。
    因此机台工时按目标去除量 / MRR 折算（``tool_hours``）——去除速率越快，
    单位实验摊销的机台成本越低，这才是工程师真正在权衡的"成本"。
    换成氧化铈体系时磨料单价上升一个量级，组分项才开始分叉（见 ``ABRASIVE_PRICES``）。

    浆料成本对配方是线性的，机台成本对 MRR 是 ``1/x`` 的——两者都可用于
    成本感知采样的解析惩罚项，也可作为 TOPSIS 的"成本型"维度。
    """

    currency: str = "CNY"
    # -- 浆料配制 --------------------------------------------------------- #
    slurry_volume_ml: float = 1000.0
    slurry_density_g_ml: float = 1.05
    abrasive_price_per_kg: float = 45.0
    """按当前配方族的磨料商品价设定（胶体二氧化硅 30% 固含折算）。"""
    oxidizer_price_per_kg: float = 8.0  # H2O2（31% 水溶液）
    complexing_price_per_kg: float = 15.0  # 甘氨酸
    surfactant_price_per_kg: float = 60.0  # PEG / PVP
    inhibitor_price_per_kg: float = 120.0  # BTA
    water_price_per_kg: float = 0.02
    # -- 固定成本 --------------------------------------------------------- #
    patterned_wafer_cost: float = 4200.0
    """300 mm 图形片采购价（Dishing / 图形片验证必须用图形片）。"""
    wafer_reuse_factor: float = 6.0
    pad_cost_per_experiment: float = 95.0  # 抛光垫磨损摊销
    metrology_cost: float = 210.0  # MRR / Ra / Dishing 计量合计
    labor_cost_per_hour: float = 160.0
    # -- 机台工时 --------------------------------------------------------- #
    target_removal_a: float = 6000.0
    """单次抛光的标称去除量（Å），用于把 MRR 折算成抛光时间。"""
    tool_hour_cost: float = 2400.0
    """CMP 机台工时成本（含折旧与产能机会成本），元/小时。"""
    handling_hours: float = 0.40
    """上下片、清洗、传片等与配方无关的固定工时（h）。"""
    # -- 成本感知采样 ----------------------------------------------------- #
    penalty_lambda: float = 0.18
    """采集函数成本惩罚强度（作用在归一化成本上）；0 表示关闭成本感知（消融 7）。"""
    reference_cost: float = 2200.0
    """归一化基准成本（元），取典型实验成本，使惩罚项量级与 HV 改进相当。"""

    # -- 分项 ------------------------------------------------------------- #
    def slurry_cost(self, view: _NamedView) -> Any:
        """单次实验的浆料成本（元），随配方线性变化。"""
        mass_kg = self.slurry_volume_ml * self.slurry_density_g_ml / 1000.0
        cost = (
            view["abrasive_wt_pct"] / 100.0 * self.abrasive_price_per_kg
            + view["oxidizer_wt_pct"] / 100.0 * self.oxidizer_price_per_kg
            + view["complexing_wt_pct"] / 100.0 * self.complexing_price_per_kg
            + view["surfactant_wt_pct"] / 100.0 * self.surfactant_price_per_kg
            + view["inhibitor_wt_pct"] / 100.0 * self.inhibitor_price_per_kg
            + self.water_price_per_kg
        )
        return cost * mass_kg

    def tool_hours(self, mrr: Any) -> Any:
        """抛光工时（h）= 固定上下片时间 + 目标去除量 / MRR。"""
        mrr_safe = _ewmax(mrr, 1.0)
        return self.handling_hours + self.target_removal_a / mrr_safe / 60.0

    def tool_cost(self, mrr: Any) -> Any:
        return self.tool_hours(mrr) * self.tool_hour_cost

    @property
    def fixed_cost(self) -> float:
        """与配方、MRR 都无关的固定成本（元/次）。"""
        return (
            self.patterned_wafer_cost / max(self.wafer_reuse_factor, 1.0)
            + self.pad_cost_per_experiment
            + self.metrology_cost
            + self.labor_cost_per_hour * self.handling_hours
        )

    def total(self, view: _NamedView, mrr: Any) -> Any:
        """单次实验总成本（元/次）。``mrr`` 为对应配方的去除速率（Å/min）。"""
        return self.fixed_cost + self.slurry_cost(view) + self.tool_cost(mrr)

    def normalized(self, view: _NamedView, mrr: Any) -> Any:
        """归一化成本，供成本惩罚项使用（1.0 为典型值）。"""
        return self.total(view, mrr) / max(self.reference_cost, 1e-9)


# --------------------------------------------------------------------------- #
# 6. 实验 / 采集 / 停止 / TOPSIS / 模型先验
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentPlan:
    """实验预算与并行度。"""

    n_init: int = 16
    """首批实验数（LHS 补充采样）。"""
    n_init_from_history: int = 6
    """首批中由历史知识库相似性检索贡献的样本上限（只做先验，不替代新基底校准）。"""
    q: int = 3
    """每轮并行推荐数（qNEHVI 的 q）。"""
    n_rounds: int = 10
    """最大闭环轮数。"""
    lhs_seed: int = 20260101
    min_samples_per_objective: int = 8
    """低于该样本数时只允许探索性采样，不做利用性推荐。"""


@dataclass(frozen=True)
class AcquisitionConfig:
    """多目标采集函数配置（模块 3）。"""

    acqf: str = "qLogNEHVI"
    """采集函数：``qLogNEHVI``（默认）/ ``qNEHVI``（legacy，仅做对照实验）。

    默认用**对数版**不是口味问题：BoTorch 0.18 在对 ``qNoisyExpectedHypervolumeImprovement``
    实例化时会主动发 ``NumericsWarning``——“有已知数值问题、会导致次优优化性能，
    强烈建议直接换成 qLogNoisyExpectedHypervolumeImprovement，API 完全一致”
    （见 arXiv:2310.20708）。两者算法相同，差别只在 EHVI 用线性还是 log 空间计算；
    log 版在改进量很小时也不会丢精度。
    代价：返回值的量纲是 log，成本惩罚必须写成 ``val - λc`` 而不是 ``val·e^{-λc}``，
    模块 3 的 ``CostAwareAcquisition`` 已经按这个区别分支处理。
    """
    num_restarts: int = 6
    raw_samples: int = 512
    num_samples: int = 128
    """蒙特卡洛采样数（每个候选抽多少条后验样本）。128 是精度/耗时折中；
    降到 64 约快 1.3 倍，但小 q 时采集值方差明显变大。"""
    maxiter: int = 70
    """SLSQP 单次重启的最大迭代数。非线性约束下收敛较慢；70 在"精度/耗时"上
    比较平衡（100 时每轮多花约 40%，采集值仅小幅提升）。"""
    prune_baseline: bool = True
    cache_root: bool = True
    """缓存基线协方差的 Cholesky 根：实测把一轮优化从 22.6 s 压到 13.7 s
    （纯 Python EHVI 内核下）。代价是显存/内存占用，n<200 时可放心开。"""
    cost_aware: bool = True
    failure_prob_threshold: float = 0.35
    """失效概率超过该阈值的候选直接压低采样优先级（安全约束，不并入主目标）。"""
    exploration_trigger_rounds: int = 3
    """连续多少轮推荐落在同一狭窄区域即判定局部最优陷阱。"""
    exploration_trigger_radius: float = 0.08
    """狭窄区域判定半径（单位立方空间）。"""
    exploration_bonus: float = 0.5
    """触发探索增强时对后验不确定项的加权系数。"""


@dataclass(frozen=True)
class ReferencePointPolicy:
    """qNEHVI 参考点策略（模块 3）。

    更新规则：``ref_t = min(ref_{t-1}, worst_observed_t - margin)``。
    参考点单调不增，保证跨轮次超体积可比；同时不因单轮异常值把参考点
    永久拉到不可达区域。
    """

    mode: str = "dynamic"
    """``static`` / ``dynamic``；复现实验与消融可用 static。"""
    margin_ratio: float = 0.12
    """沿"更差"方向退让的比例：``margin = margin_ratio * |worst|``。"""
    update_rule: str = "running_worst"
    log_every_round: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"static", "dynamic"}:
            raise ValueError(f"未知 ref_point mode={self.mode!r}")


@dataclass(frozen=True)
class StoppingCriteria:
    """收敛与终止判据（模块 6）。"""

    max_rounds: int = 10
    hv_rel_improvement_min: float = 0.01
    """超体积相对增量低于该值视为停滞。"""
    hv_patience_rounds: int = 2
    spacing_target: float = 0.05
    """Spacing 目标值（越小说明前沿点分布越均匀）。"""
    spec_hit_rate_target: float = 0.02
    """规格满足率目标（本轮实验批次中达标样本占比）。"""
    require_spec_hit: bool = True
    """拿到满足产线规格的合格配方即可提前终止。"""


@dataclass(frozen=True)
class TopsisCriterion:
    name: str
    weight: float
    role: str  # benefit / cost
    description: str = ""


@dataclass(frozen=True)
class TopsisConfig:
    """TOPSIS 后处理配置（模块 4）。

    权重在初始化阶段一次性冻结，后续轮次沿用；熵权只做敏感性对照，
    绝不参与 BO 的目标构造——这是"TOPSIS 只做前沿后处理"原则的落地方式。
    """

    weights: Tuple[TopsisCriterion, ...]
    entropy_weight_sensitivity_only: bool = True
    normalize: str = "vector"
    """向量归一化：``r_ij = x_ij / sqrt(sum_i x_ij^2)``。"""
    hard_spec_filter_first: bool = True
    """先剔除不合规（如 Ra 超标）的前沿点，再打分。"""

    def __post_init__(self) -> None:
        total = sum(c.weight for c in self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"TOPSIS 权重之和必须为 1，当前为 {total:.6f}")

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.weights)

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(c.role for c in self.weights)

    def weights_array(self) -> np.ndarray:
        """权重向量（与 ``names`` 同序），供模块 4 直接使用。"""
        return np.asarray([c.weight for c in self.weights], dtype=float)


@dataclass(frozen=True)
class ModelHealthConfig:
    """模型健康检查阈值（模块 2）。

    三层判定，避免"一刀切"把弱信号模型直接砍成常数：

    * **后备（fallback）**：无预测力或比常数还差 —— R² ≤ ``min_loo_r2``，
      或相对误差 > ``loo_relative_rmse_fatal``，或长度尺度全塌缩；
    * **降级（degraded）**：能排序但精度不够交付 —— R² < ``degraded_loo_r2``
      或相对误差 > ``loo_relative_rmse_max``；模型保留，但预测标低置信度、
      过度噪声后验方差；
    * **正常（ok）**：其余情况。

    相对误差的分母是**目标标称量级 mean(|y|)**，不是样本标准差。用标准差做
    分母会让阈值等价于一个 R² 门槛（``rel_rmse = sqrt(1-R²)``），把"测量噪声大"
    和"模型没学到东西"混为一谈——而这两件事的处理方式完全不同。
    """

    loo_relative_rmse_max: float = 0.35
    """交付级容差：相对误差超过此值即降级（仍可用于排序，但不能作为交付预测）。"""
    loo_relative_rmse_fatal: float = 1.0
    """灾难级：相对误差超过标称量级（等于或差于直接预测均值）→ 后备。"""
    min_loo_r2: float = 0.0
    """LOOCV R² 低于此值视为无预测力 → 后备。"""
    degraded_loo_r2: float = 0.5
    """LOOCV R² 低于此值标记降级。"""
    lengthscale_min_ratio: float = 0.02
    """长度尺度低于输入范围该比例视为"维度塌缩"。"""
    lengthscale_max_ratio: float = 20.0
    fallback_model: str = "constant"
    """``constant``：仅输出均值、方差取观测先验，把预算让给可学习目标。"""
    fallback_flag_in_prediction: bool = True
    """后备状态下预测一律打"低置信度"标签（模块 7 联动）。"""


@dataclass(frozen=True)
class PrestonPrior:
    """Preston 方程 ``MRR = k_p * P * v`` 的先验参数。

    ``v`` 由抛光盘/载片台转速换算线速度（取盘心半径处线速度近似），单位 m/s。
    """

    k_mean: float = 220.0
    """k_p 先验均值，单位 Å·min⁻¹ / (psi·m·s⁻¹)；铜 CMP 实测约 120–320。"""
    k_sigma_log: float = 0.45
    pad_diameter_m: float = 0.6
    """抛光盘直径（m），300 mm 晶圆对应 800 mm 抛光盘时取 0.6–0.7。"""

    def velocity(self, platen_rpm: Any, carrier_rpm: Any) -> Any:
        """相对线速度近似：``v = pi * D * (rpm_platen + rpm_carrier) / 120``。"""
        return math.pi * self.pad_diameter_m * (platen_rpm + carrier_rpm) / 120.0

    def mrr(self, pressure_psi: Any, platen_rpm: Any, carrier_rpm: Any) -> Any:
        """Preston 机理均值预测（Å/min），用作 MRR 的 GPR 均值函数。"""
        return self.k_mean * pressure_psi * self.velocity(platen_rpm, carrier_rpm)


@dataclass(frozen=True)
class GPRPriorConfig:
    """GPR 核超参数先验（模块 2 训练时使用）。"""

    kernel: str = "Matern52_ARD"
    lengthscale_prior: Tuple[str, float, float] = ("LogNormal", 0.0, 1.2)
    """(分布名, 对数均值, 对数标准差)。"""
    outputscale_prior: Tuple[str, float, float] = ("LogNormal", -1.0, 0.6)
    noise_prior: Tuple[str, float, float] = ("LogNormal", -4.0, 0.8)
    use_heteroscedastic_noise: bool = False
    mean_module: str = "preston"
    """``preston``（MRR 用机理均值）/ ``constant`` / ``zero``。"""
    ood_mahalanobis_threshold: float = 3.0
    """马氏距离阈值，超过即判为域外（模块 7）。"""


# --------------------------------------------------------------------------- #
# 7. 参数空间主体
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CMPRecipeSpace:
    """一个 CMP 配方优化问题的完整定义（模块 1 的唯一真源）。

    变量顺序即设计矩阵列顺序；``bounds()`` / ``constraint_callables()`` /
    ``reference_point()`` 都从这里派生，模块 2/3/4 只读不改。
    """

    name: str
    description: str
    variables: Tuple[DecisionVariable, ...]
    objectives: Tuple[Objective, ...]
    hard_constraints: Tuple[HardConstraint, ...]
    implicit_rules: Tuple[ImplicitRule, ...]
    covariates: Tuple[BatchCovariate, ...]
    cost_model: CostModel
    experiment: ExperimentPlan
    acquisition: AcquisitionConfig
    ref_point_policy: ReferencePointPolicy
    stopping: StoppingCriteria
    topsis: TopsisConfig
    gpr_prior: GPRPriorConfig
    model_health: ModelHealthConfig
    preston: PrestonPrior = PrestonPrior()
    family: str = "cu_peroxide_glycine"
    contextual_variables: Tuple[DecisionVariable, ...] = ()
    """类别型"配方族"上下文（磨料种类 / 氧化剂种类）。不参与连续 BO：
    按族拆分成独立子问题（分层 BO），避免不同化学体系共用一个核。"""
    version: str = "0.1.0"

    # -- 基本访问 --------------------------------------------------------- #
    def __post_init__(self) -> None:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("决策变量名重复")
        if not self.variables:
            raise ValueError("决策变量不能为空")
        if not self.objectives:
            raise ValueError("优化目标不能为空")
        if self.experiment.q < 1:
            raise ValueError("q 必须 >= 1")
        overlap = set(names) & {c.name for c in self.covariates}
        if overlap:
            raise ValueError(f"协变量与决策变量重名：{sorted(overlap)}")
        if "Cost" not in self.topsis.names:
            raise ValueError("TOPSIS 必须包含 Cost 维度")
        for o in self.objectives:
            if o.name not in self.topsis.names:
                raise ValueError(f"目标 {o.name} 缺少 TOPSIS 权重")

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def objective_names(self) -> Tuple[str, ...]:
        return tuple(o.name for o in self.objectives)

    @property
    def index(self) -> Dict[str, int]:
        return {v.name: i for i, v in enumerate(self.variables)}

    def var(self, name: str) -> DecisionVariable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"未知变量 {name!r}")

    # -- 边界 / 归一化 ---------------------------------------------------- #
    def bounds(self) -> np.ndarray:
        """物理量纲边界，形状 ``(2, d)``：第 0 行下限、第 1 行上限。"""
        return np.array(
            [[v.low for v in self.variables], [v.high for v in self.variables]], dtype=float
        )

    def unit_bounds(self) -> np.ndarray:
        """单位立方边界 ``(2, d)``，恒为 0/1（BoTorch 推荐在单位立方上优化）。"""
        return np.vstack([np.zeros(self.dim), np.ones(self.dim)])

    def bounds_tensor(self, dtype: Any = None, device: Any = None) -> Any:
        """torch 张量边界（物理量纲），可直接喂 ``Normalize`` / ``optimize_acqf``。"""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("未安装 torch：先 `pip install torch` 或改用 bounds()")
        return torch.tensor(self.bounds(), dtype=dtype or torch.double, device=device)

    def unit_bounds_tensor(self, dtype: Any = None, device: Any = None) -> Any:
        """torch 张量边界（单位立方）。"""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("未安装 torch：先 `pip install torch`")
        return torch.tensor(self.unit_bounds(), dtype=dtype or torch.double, device=device)

    def to_unit(self, x: Any) -> Any:
        """设计矩阵物理量纲 -> 单位立方（逐列按各自尺度归一）。"""
        x = _ensure_2d(x)
        cols = [v.to_unit(x[:, i]).reshape(-1, 1) for i, v in enumerate(self.variables)]
        return torch.cat(cols, dim=-1) if _is_torch(x) else np.concatenate(cols, axis=-1)

    def from_unit(self, u: Any) -> Any:
        """单位立方 -> 设计矩阵物理量纲。"""
        u = _ensure_2d(u)
        cols = [v.from_unit(u[:, i]).reshape(-1, 1) for i, v in enumerate(self.variables)]
        return torch.cat(cols, dim=-1) if _is_torch(u) else np.concatenate(cols, axis=-1)

    def sample_lhs(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """拉丁超立方采样（单位立方），返回物理量纲矩阵 ``(n, d)``。

        纯 numpy 实现，不引入 scipy：每列按分层随机置换，保证一维投影均匀。
        """
        rng = np.random.default_rng(self.experiment.lhs_seed if seed is None else seed)
        u = np.empty((n, self.dim), dtype=float)
        for j in range(self.dim):
            strata = (np.arange(n) + rng.random(n)) / n
            u[:, j] = rng.permutation(strata)
        return self.from_unit(u)

    def sample_lhs_feasible(
        self, n: int, seed: Optional[int] = None, max_draws: int = 200
    ) -> np.ndarray:
        """可行域内的 LHS 采样：拒绝采样补齐。

        全盒 LHS 有相当比例的样本踩硬约束（尤其 pH 双窗口与磨料/氧化剂比例），
        首批实验若用纯 LHS 会浪费配额，故用拒绝采样保证首批 16 条全部可上机。
        若 ``max_draws`` 次仍不足，返回已收集的样本并给出告警。

        .. warning:: **只保证硬约束可行，不过滤隐式规则库。**
           规则库里的 REJECT 级规则（如"高固含 + 强酸会沉降"）不会在这里被过滤，
           因此首批里可能出现规则库判定为不可行的配方（实测发生过：模块 5 的
           校验器抓到 `E022`，磨料 11.7 wt% + pH 2.6）。
           这是有意的取舍：首批要覆盖空间，规则由下游（模块 5 校验器 / 模块 3
           的隐式过滤）兜底；但生产使用前应把规则库也接进采样器。
        """
        rng = np.random.default_rng(self.experiment.lhs_seed if seed is None else seed)
        collected: List[np.ndarray] = []
        have = 0
        for _ in range(max_draws):
            need = n - have
            if need <= 0:
                break
            cand = self.sample_lhs(max(need * 3, 8), seed=int(rng.integers(0, 2**31 - 1)))
            cand = cand[self.feasible_mask(cand)]
            if cand.size:
                collected.append(cand)
                have += cand.shape[0]
        if have < n:
            print(
                f"[warn] 仅采到 {have}/{n} 条可行样本；检查约束是否过紧",
                file=sys.stderr,
            )
        return np.vstack(collected)[:n] if collected else np.empty((0, self.dim))

    # -- 约束 ------------------------------------------------------------- #
    def constraint_callables(self) -> List[Callable[[Any], Any]]:
        """绑定后的硬约束列表，``g_i(X) <= 0`` 表示可行（BoTorch 约定）。"""
        return [c.bind(self.names) for c in self.hard_constraints]

    def named_constraint_callables(self) -> List[Tuple[str, Callable[[Any], Any]]]:
        """带名字版本，便于日志里区分"哪条约束卡住了采样"。"""
        return [(c.name, c.bind(self.names)) for c in self.hard_constraints]

    def implicit_rule_callables(self) -> List[Tuple[str, Callable[[Any], Any]]]:
        return [(r.name, r.bind(self.names)) for r in self.implicit_rules]

    def feasible_mask(self, x: Any) -> np.ndarray:
        """硬约束可行性掩码（numpy 输入）。"""
        x = np.asarray(x, dtype=float)
        mask = np.ones(x.shape[0], dtype=bool)
        for c in self.hard_constraints:
            mask &= np.asarray(c.bind(self.names)(x), dtype=float).reshape(-1) <= 0.0
        return mask

    def max_violation(self, x: Any) -> np.ndarray:
        """每条样本的最大违反度（``<= 0`` 严格可行）。

        比布尔掩码更有用：它给出"离可行域有多远"，模块 3 用它做带容差的
        可行性判定（SLSQP 返回的解常常恰好压在边界上，``g`` 是 1e-12 级的正数，
        用严格 ``<= 0`` 会把数值零判成违约，触发无意义的修复）。
        """
        x = np.asarray(x, dtype=float)
        v = np.full(x.shape[0], -np.inf, dtype=float)
        for c in self.hard_constraints:
            g = np.asarray(c.bind(self.names)(x), dtype=float).reshape(-1)
            v = np.maximum(v, g)
        return v

    def constraint_diagnostics(self, x: Any) -> Dict[str, float]:
        """各条硬约束的通过率（%，0–100），用于调试参数空间是否设得过紧。"""
        x = np.asarray(x, dtype=float)
        out: Dict[str, float] = {}
        for c in self.hard_constraints:
            g = np.asarray(c.bind(self.names)(x), dtype=float).reshape(-1)
            out[c.name] = float((g <= 0.0).mean() * 100.0)
        out["__all__"] = float(self.feasible_mask(x).mean() * 100.0)
        return out

    def filter_implicit(self, x: Any) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """隐式规则前置过滤：返回 ``(保留的 X, 保留掩码, 命中规则摘要)``。

        ``REJECT`` 级规则直接丢弃候选；``WARN`` / ``FLAG`` 只打标不丢弃。
        """
        x = np.asarray(x, dtype=float)
        keep = np.ones(x.shape[0], dtype=bool)
        hits: List[str] = []
        for rule in self.implicit_rules:
            flagged = np.asarray(rule.bind(self.names)(x), dtype=bool).reshape(-1)
            if flagged.any():
                hits.append(f"{rule.name}({int(flagged.sum())})")
            if rule.severity is Severity.REJECT:
                keep &= ~flagged
        return x[keep], keep, hits

    def estimate_mrr(self, x: Any) -> Any:
        """用 Preston 先验粗估 MRR（Å/min），仅用于成本折算，不进 GPR 训练。"""
        view = _NamedView(self.names, x)
        return self.preston.mrr(view["down_force_psi"], view["platen_rpm"], view["carrier_rpm"])

    def cost(self, x: Any, mrr: Any = None) -> Any:
        """单次实验总成本（元）。``mrr`` 缺省时用 Preston 先验估计。"""
        view = _NamedView(self.names, x)
        if mrr is None:
            mrr = self.estimate_mrr(x)
        return self.cost_model.total(view, mrr)

    def cost_matrix(self, x: Any, y_mrr: Sequence[float]) -> np.ndarray:
        """批量成本向量（元），供 TOPSIS 的 Cost 维度使用（用实测 MRR）。"""
        return np.asarray(
            self.cost_model.total(_NamedView(self.names, x), np.asarray(y_mrr, dtype=float)),
            dtype=float,
        ).reshape(-1)

    # -- 目标 / 参考点 / 规格 --------------------------------------------- #
    def objective_matrix(self) -> Tuple[Tuple[float, ...], ...]:
        """目标符号矩阵，形状 ``(m, 1)``，供 ``GenericMCObjective`` 使用。

        对默认空间即 ``((+1,), (-1,), (-1,))``：MRR 越大越好，Ra/Dishing 越小越好。
        """
        return tuple((o.sign,) for o in self.objectives)

    def objective_signs(self) -> np.ndarray:
        """目标符号向量 ``(m,)``：+1 最大化，-1 最小化。"""
        return np.asarray([o.sign for o in self.objectives], dtype=float)

    def reference_point(self, dtype: Any = None, device: Any = None) -> Any:
        """初始化参考点（BoTorch 最大化符号空间），形状 ``(m,)``。

        规则：取各目标先验范围的"最差端"，沿更差方向退让
        ``margin = margin_ratio * |最差端|``。这样参考点永远严格劣于任何预期观测，
        第一轮超体积就有定义；用相对退让而非绝对量，保证换指标量纲后无需重调。
        """
        vals = []
        for o in self.objectives:
            lo, hi = o.prior_range
            ends = sorted([o.sign * lo, o.sign * hi])
            worst = ends[0]
            margin = self.ref_point_policy.margin_ratio * abs(worst)
            vals.append(worst - margin)
        if _TORCH_AVAILABLE and (dtype is not None or device is not None):
            return torch.tensor(vals, dtype=dtype or torch.double, device=device)
        return np.asarray(vals, dtype=float)

    def update_reference_point(self, observed_y: Any) -> np.ndarray:
        """动态更新参考点（单调不增），返回最大化符号空间的 ``(m,)``。

        ``ref_t = min(ref_{t-1}, worst_obs_t - margin)``：
        * 单调不增 → 跨轮次超体积可比，不会因为参考点回撤造成 HV 虚降；
        * 只在观测突破先验下限时才下移 → 避免个别异常值把参考点永久拉飞。
        """
        y = np.asarray(observed_y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        cur = np.asarray(self.reference_point(), dtype=float)
        if self.ref_point_policy.mode != "dynamic":
            return cur
        for j, o in enumerate(self.objectives):
            col = y[:, j] * o.sign
            worst = float(np.min(col))
            margin = self.ref_point_policy.margin_ratio * abs(worst)
            cur[j] = min(cur[j], worst - margin)
        return cur

    def spec_hit_rate(self, y: Any) -> float:
        """规格满足率：所有目标同时达标的样本比例（模块 6 指标）。"""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        ok = np.ones(y.shape[0], dtype=bool)
        for j, o in enumerate(self.objectives):
            ok &= np.asarray(o.is_on_spec(y[:, j]), dtype=bool)
        return float(ok.mean())

    # -- 校验 / 序列化 ----------------------------------------------------- #
    def validate_recipe(self, recipe: Mapping[str, float]) -> Dict[str, Any]:
        """校验单条配方：越界、硬约束、隐式规则一次报全。"""
        missing = [n for n in self.names if n not in recipe]
        if missing:
            return {"ok": False, "error": f"缺少变量：{missing}"}
        x = np.array([[float(recipe[n]) for n in self.names]])
        report: Dict[str, Any] = {"ok": True, "violations": {}, "implicit": []}
        for v, val in zip(self.variables, x[0]):
            if not (v.low <= val <= v.high):
                report["ok"] = False
                report["violations"][f"bounds::{v.name}"] = float(val)
        for c in self.hard_constraints:
            g = float(np.asarray(c.bind(self.names)(x)).reshape(-1)[0])
            if g > 0:
                report["ok"] = False
                report["violations"][c.name] = round(g, 4)
        for r in self.implicit_rules:
            if bool(np.asarray(r.bind(self.names)(x)).reshape(-1)[0]):
                report["implicit"].append(r.name)
                if r.severity is Severity.REJECT:
                    report["ok"] = False
        report["cost_cny"] = round(float(np.asarray(self.cost(x)).reshape(-1)[0]), 1)
        return report

    def to_dict(self) -> Dict[str, Any]:
        """JSON 可序列化快照（不含可调用对象），用于版本管理与审计。"""
        return {
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "description": self.description,
            "variables": [
                {
                    "name": v.name,
                    "unit": v.unit,
                    "low": v.low,
                    "high": v.high,
                    "kind": v.kind.value,
                    "scale": v.scale.value,
                    "categories": list(v.categories),
                }
                for v in self.variables
            ],
            "objectives": [
                {
                    "name": o.name,
                    "unit": o.unit,
                    "direction": o.direction.value,
                    "spec": o.spec,
                    "prior_range": list(o.prior_range),
                    "topsis_role": o.topsis_role,
                }
                for o in self.objectives
            ],
            "hard_constraints": [
                {"name": c.name, "expression": c.expression, "logic": c.logic.value}
                for c in self.hard_constraints
            ],
            "implicit_rules": [
                {"name": r.name, "severity": r.severity.value} for r in self.implicit_rules
            ],
            "covariates": [{"name": c.name, "handling": c.handling} for c in self.covariates],
            "cost_model": {
                "currency": self.cost_model.currency,
                "fixed_cost_cny": round(self.cost_model.fixed_cost, 2),
                "patterned_wafer_cost": self.cost_model.patterned_wafer_cost,
                "tool_hour_cost": self.cost_model.tool_hour_cost,
                "abrasive_price_per_kg": self.cost_model.abrasive_price_per_kg,
                "penalty_lambda": self.cost_model.penalty_lambda,
            },
            "experiment": {
                "n_init": self.experiment.n_init,
                "q": self.experiment.q,
                "n_rounds": self.experiment.n_rounds,
            },
            "acquisition": {
                "acqf": self.acquisition.acqf,
                "q": self.experiment.q,
                "cost_aware": self.acquisition.cost_aware,
            },
            "ref_point_policy": {
                "mode": self.ref_point_policy.mode,
                "margin_ratio": self.ref_point_policy.margin_ratio,
            },
            "topsis": {
                "weights": [
                    {"name": c.name, "weight": c.weight, "role": c.role}
                    for c in self.topsis.weights
                ],
                "frozen": True,
            },
            # 健康阈值与 GPR 先验也是实验协议的一部分：改一个阈值就可能让
            # 某个目标从"可用"变成"后备"，因此必须进指纹，否则同一条实验
            # 数据会对应两个不同的判定口径（模块 8 审计要求）。
            "model_health": {
                "loo_relative_rmse_max": self.model_health.loo_relative_rmse_max,
                "loo_relative_rmse_fatal": self.model_health.loo_relative_rmse_fatal,
                "min_loo_r2": self.model_health.min_loo_r2,
                "degraded_loo_r2": self.model_health.degraded_loo_r2,
                "lengthscale_min_ratio": self.model_health.lengthscale_min_ratio,
                "lengthscale_max_ratio": self.model_health.lengthscale_max_ratio,
                "fallback_model": self.model_health.fallback_model,
            },
            "gpr_prior": {
                "kernel": self.gpr_prior.kernel,
                "lengthscale_prior": list(self.gpr_prior.lengthscale_prior),
                "outputscale_prior": list(self.gpr_prior.outputscale_prior),
                "noise_prior": list(self.gpr_prior.noise_prior),
                "mean_module": self.gpr_prior.mean_module,
                "heteroscedastic_noise": self.gpr_prior.use_heteroscedastic_noise,
            },
        }

    def config_hash(self) -> str:
        """配置内容哈希（前 12 位），作为实验批次的配置指纹。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        """人类可读的空间摘要，直接打印即可贴进实验记录（避免 CJK 表格对齐问题）。"""
        lines = [
            f"# {self.name}",
            f"  family={self.family}  version={self.version}  hash={self.config_hash()}",
            f"  {self.description}",
            "",
            f"决策变量（{self.dim} 维，顺序即设计矩阵列顺序）",
        ]
        for i, v in enumerate(self.variables, 1):
            lines.append(
                f"  {i:>2}. {v.name:<22} [{v.low:g}, {v.high:g}] {v.unit:<8} {v.scale.value:<7}"
                f" {v.mechanism}"
            )
        lines += ["", "优化目标（BoTorch 统一最大化，最小化指标取负）"]
        wmap = {c.name: c.weight for c in self.topsis.weights}
        for o in self.objectives:
            arrow = "↑" if o.direction is Direction.MAXIMIZE else "↓"
            lines.append(
                f"  {arrow} {o.name:<9} {o.direction.value:<9} spec={o.spec:<7g}"
                f" 先验范围={o.prior_range}  TOPSIS 权重={wmap[o.name]:.2f} ({o.topsis_role})"
            )
        lines += ["", f"确定性硬约束（{len(self.hard_constraints)} 条，进 Constrained qNEHVI，g<=0 可行）"]
        for c in self.hard_constraints:
            lines.append(f"  [{c.logic.value:<6}] {c.name}: {c.expression}")
        lines += ["", f"隐式约束规则库（{len(self.implicit_rules)} 条，采样后前置过滤）"]
        for r in self.implicit_rules:
            lines.append(f"  [{r.severity.value:<6}] {r.name}: {r.action}")
        lines += ["", "成本 / 实验 / 采集"]
        cm = self.cost_model
        mid = np.mean(self.bounds(), axis=0, keepdims=True)  # 空间中心的代表配方
        slurry_mid = float(np.asarray(cm.slurry_cost(_NamedView(self.names, mid))).reshape(-1)[0])
        lines.append(
            f"  成本：固定 {cm.fixed_cost:.0f} 元 + 浆料 {slurry_mid:.1f} 元（中心配方）"
            f" + 机台 {cm.tool_hour_cost:.0f} 元/h × ({cm.handling_hours:.2f}h + {cm.target_removal_a:g}/MRR/60)"
        )
        lines.append(
            f"  成本感知：λ={cm.penalty_lambda}（归一化基准 {cm.reference_cost:.0f} 元）"
        )
        lines.append(
            f"  实验：n_init={self.experiment.n_init}（其中历史先验≤{self.experiment.n_init_from_history}）"
            f" q={self.experiment.q} rounds={self.experiment.n_rounds}"
        )
        lines.append(
            f"  采集：{self.acquisition.acqf}  ref_point={self.ref_point_policy.mode}"
            f"（margin={self.ref_point_policy.margin_ratio}）  "
            f"失效概率阈值={self.acquisition.failure_prob_threshold}"
        )
        lines.append(
            f"  初始 ref_point（最大化空间）= {np.round(np.asarray(self.reference_point()), 2).tolist()}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 8. 默认问题的构造
# --------------------------------------------------------------------------- #
#: 磨料商品单价（元/kg，按固含折算），决定不同配方族的成本量级差异。
ABRASIVE_PRICES: Dict[str, float] = {
    "colloidal_silica": 45.0,
    "calcined_ceria": 620.0,
    "alpha_alumina": 90.0,
}


# --------------------------------------------------------------------------- #
# 化学物质目录（界面下拉的预设，也是"体系元数据"的词表）
# --------------------------------------------------------------------------- #
#: 每个角色下的 ``(目录代号, 展示名)``。代号必须能安全用于目录名
#: （只允许字母/数字/下划线），因为网页端会把它拼进
#: ``data/projects/{基底}/{磨料}_{氧化剂}_{络合剂}/``。
#:
#: 清单来源：CMP 公开文献与工业配方中的常见组分（磨料/氧化剂/络合剂/
#: 表面活性剂/缓蚀剂/pH 调节剂），并用网络检索做了类别核对
#: （如 "pH adjusting agent"、"azole corrosion inhibitor" 等术语）。
#: 它只是**预设**，界面允许用户手输任意新物质——目录代号的规范化见
#: ``src/app.py: slugify_token``。
CHEMISTRY_CATALOG: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "substrate": (
        ("Cu", "铜 Cu"),
        ("W", "钨 W"),
        ("Oxide", "氧化硅介质 SiO2"),
        ("SiN", "氮化硅 SiN"),
        ("Al", "铝 Al"),
        ("Co", "钴 Co"),
        ("Ru", "钌 Ru"),
        ("Si", "单晶硅 Si"),
    ),
    "abrasive": (
        ("SiO2", "胶体二氧化硅 Colloidal Silica"),
        ("CeO2", "氧化铈 Ceria"),
        ("Al2O3", "氧化铝 Alumina"),
        ("ZrO2", "氧化锆 Zirconia"),
        ("TiO2", "二氧化钛 Titania"),
        ("Diamond", "金刚石 Diamond"),
    ),
    "oxidizer": (
        ("H2O2", "过氧化氢 H2O2"),
        ("KIO3", "碘酸钾 KIO3"),
        ("APS", "过硫酸铵 APS"),
        ("KMnO4", "高锰酸钾 KMnO4"),
        ("NaClO", "次氯酸钠 NaClO"),
        ("HNO3", "硝酸 HNO3"),
        ("FeNitrate", "硝酸铁 Fe(NO3)3"),
        ("O3", "臭氧水 O3"),
    ),
    "complexing": (
        ("Glycine", "甘氨酸 Glycine"),
        ("CitricAcid", "柠檬酸 Citric Acid"),
        ("EDTA", "EDTA"),
        ("OxalicAcid", "草酸 Oxalic Acid"),
        ("TartaricAcid", "酒石酸 Tartaric Acid"),
        ("AceticAcid", "乙酸 Acetic Acid"),
        ("AmmoniumCitrate", "柠檬酸铵 Ammonium Citrate"),
        ("Ammonia", "氨水 Ammonia"),
    ),
    "surfactant": (
        ("PEG", "聚乙二醇 PEG"),
        ("PVP", "聚乙烯吡咯烷酮 PVP"),
        ("TX100", "Triton X-100"),
        ("SDS", "十二烷基硫酸钠 SDS"),
        ("CTAB", "十六烷基三甲基溴化铵 CTAB"),
        ("PAA", "聚丙烯酸 PAA"),
        ("Pluronic", "Pluronic 嵌段共聚物"),
    ),
    "inhibitor": (
        ("BTA", "苯并三氮唑 BTA"),
        ("TTA", "甲基苯并三氮唑 TTA"),
        ("BTA_OH", "羟基苯并三氮唑 OH-BTA"),
        ("Benzimidazole", "苯并咪唑 Benzimidazole"),
        ("Thiourea", "硫脲 Thiourea"),
        ("Cystine", "胱氨酸 Cystine"),
    ),
    "ph_adjuster": (
        ("KOH", "氢氧化钾 KOH"),
        ("NH4OH", "氨水 NH4OH"),
        ("TMAH", "四甲基氢氧化铵 TMAH"),
        ("HNO3", "硝酸 HNO3"),
        ("H2SO4", "硫酸 H2SO4"),
        ("CitricAcid", "柠檬酸 Citric Acid"),
        ("AceticAcid", "乙酸 Acetic Acid"),
    ),
}

#: 允许出现在目录名里的字符（与 app 端的规范化规则保持一致）。
_SAFE_TOKEN_RE = __import__("re").compile(r"^[A-Za-z0-9_]{1,64}$")


def is_path_safe_token(token: str) -> bool:
    """目录代号是否只含字母/数字/下划线（跨平台安全）。

    网页端把用户手输的化学物质名规范化后才建目录；这个函数供测试与
    数据校验使用，防止 ``Fe(NO3)3`` 这类带括号的名字混进路径。
    """
    return bool(_SAFE_TOKEN_RE.match(token or ""))

#: 配方族预设：不同化学体系不能共用一个核，按族分层做 BO。
#: ``ph_windows`` 允许给多个区间——物理上 H2O2 只在酸性/碱性两端稳定。
FAMILY_PRESETS: Dict[str, Dict[str, Any]] = {
    "cu_peroxide_glycine": {
        "description": "铜 CMP：胶体二氧化硅 + H2O2 + 甘氨酸（酸性 / 碱性双稳定窗口）",
        "abrasive": "colloidal_silica",
        "oxidizer": "H2O2",
        "ph_windows": ((2.0, 5.5), (8.5, 11.0)),
        "preston_k": 220.0,
    },
    "cu_alkaline_complex": {
        "description": "铜 CMP：碱性过氧化氢 + 络合体系（仅碱性窗口，抑制点蚀）",
        "abrasive": "colloidal_silica",
        "oxidizer": "H2O2",
        "ph_windows": ((8.5, 11.0),),
        "preston_k": 260.0,
    },
    "oxide_ceria_anionic": {
        "description": "氧化物 CMP：氧化铈 + 阴离子表面活性剂（无氧化剂，靠 Ce³⁺/Ce⁴⁺ 化学活性）",
        "abrasive": "calcined_ceria",
        "oxidizer": "H2O2",
        "ph_windows": ((5.5, 8.5),),
        "preston_k": 520.0,
    },
}


@dataclass(frozen=True)
class _RecordField:
    """实验记录字段（模块 5 数据回写契约）。"""

    name: str
    type: str
    required: bool = True
    description: str = ""


#: 模块 5 结构化回写契约：一轮实验的完整记录列。
RecordSchema: Tuple[_RecordField, ...] = (
    _RecordField("experiment_id", "str", True, "实验唯一编号，如 R03-E02"),
    _RecordField("round_id", "int", True, "闭环轮次"),
    _RecordField("batch_id", "str", True, "配液批次号（随机效应 / 批次归一化的分组键）"),
    _RecordField("config_hash", "str", True, "下发时的问题配置指纹（模块 8 审计）"),
    _RecordField("recipe", "dict", True, "变量名 -> 物理量纲值"),
    _RecordField("y_mrr", "float", True, "MRR 实测值，Å/min"),
    _RecordField("y_ra", "float", True, "Ra 实测值，Å"),
    _RecordField("y_dishing", "float", True, "Dishing 实测值，Å"),
    _RecordField("y_stability_h", "float", False, "浆料稳定性观察时长（h），未测填 NaN"),
    _RecordField("covariates", "dict", True, "pad_use_count / slurry_age_hours / ..."),
    _RecordField("cost_cny", "float", True, "本次实验实付成本"),
    _RecordField("anomaly", "enum", True, "none / measurement（仪器操作失误，可剔除）/ response（配方导致，保留并标记）"),
    _RecordField("failure", "enum", True, "none / mix_fail / scratch / agglomeration / other（供失效概率模型学习）"),
    _RecordField("uncertainty", "dict", False, "预测时的 epistemic / aleatoric / OOD 分层记录"),
    _RecordField("operator", "str", False, "操作人"),
    _RecordField("timestamp", "iso8601", True, "上机完成时间"),
)


def _build_variables(
    abrasive: str, oxidizer: str, ph_windows: Tuple[Tuple[float, float], ...]
) -> Tuple[DecisionVariable, ...]:
    """构造决策变量列表（顺序即列顺序，勿随意调整）。"""
    ph_lo = min(w[0] for w in ph_windows)
    ph_hi = max(w[1] for w in ph_windows)
    abrasive_cap = 12.0 if abrasive == "colloidal_silica" else 6.0
    return (
        DecisionVariable(
            name="abrasive_wt_pct",
            unit="wt%",
            low=0.5,
            high=abrasive_cap,
            mechanism="磨料机械去除主体；过高→划伤/团聚，过低→MRR 不足",
            note="商品浆料固含折算后的最终浓度",
        ),
        DecisionVariable(
            name="oxidizer_wt_pct",
            unit="wt%",
            low=0.2,
            high=8.0,
            mechanism=f"{oxidizer} 氧化被抛表面形成软质氧化层，决定化学去除速率",
        ),
        DecisionVariable(
            name="complexing_wt_pct",
            unit="wt%",
            low=0.1,
            high=4.0,
            mechanism="络合剂（甘氨酸）溶解金属离子、加速氧化层脱除；过量→过度腐蚀",
        ),
        DecisionVariable(
            name="surfactant_wt_pct",
            unit="wt%",
            low=0.0,
            high=0.50,
            mechanism="活性剂吸附抑制凹处去除，降低 Dishing；过量→MRR 塌陷/起泡",
        ),
        DecisionVariable(
            name="inhibitor_wt_pct",
            unit="wt%",
            low=0.0,
            high=0.30,
            mechanism="缓蚀剂（BTA）在铜面成膜，压制静态腐蚀与 Dishing",
        ),
        DecisionVariable(
            name="ph",
            unit="-",
            low=ph_lo,
            high=ph_hi,
            mechanism="决定氧化剂稳定性、Zeta 电位与络合形态，是最强的化学杠杆之一",
        ),
        DecisionVariable(
            name="down_force_psi",
            unit="psi",
            low=0.5,
            high=5.0,
            mechanism="Preston 方程中的 P：直接放大机械去除，也放大划伤与 Dishing",
        ),
        DecisionVariable(
            name="platen_rpm",
            unit="rpm",
            low=30.0,
            high=150.0,
            mechanism="决定相对线速度 v，与压力共同构成机械作用项 P·v",
        ),
        DecisionVariable(
            name="carrier_rpm",
            unit="rpm",
            low=27.0,
            high=135.0,
            mechanism="载片台转速影响面内均匀性与剪切；与盘转速之差决定去除均一性",
        ),
        DecisionVariable(
            name="slurry_flow_ml_min",
            unit="mL/min",
            low=50.0,
            high=300.0,
            mechanism="供液不足→干抛/划伤与去除不均；过量→浪费并抬高成本",
        ),
    )


def _build_objectives() -> Tuple[Objective, ...]:
    return (
        Objective(
            name="MRR",
            unit="Å/min",
            direction=Direction.MAXIMIZE,
            spec=1500.0,
            prior_range=(150.0, 3200.0),
            topsis_role="benefit",
            description="材料去除速率，产线要求不低于 1500 Å/min 以满足产能",
            mechanism="Preston: MRR = k_p·P·v；化学项通过氧化层厚度调制有效 k_p",
        ),
        Objective(
            name="Ra",
            unit="Å",
            direction=Direction.MINIMIZE,
            spec=4.0,
            prior_range=(1.2, 10.0),
            topsis_role="cost",
            description="表面粗糙度，产线要求不高于 4 Å",
        ),
        Objective(
            name="Dishing",
            unit="Å",
            direction=Direction.MINIMIZE,
            spec=150.0,
            prior_range=(30.0, 450.0),
            topsis_role="cost",
            description="图形片碟形凹陷，产线要求不高于 150 Å",
        ),
    )


def _build_constraints(
    ph_windows: Tuple[Tuple[float, float], ...], pad_diameter_m: float
) -> Tuple[HardConstraint, ...]:
    """构造确定性硬约束。

    全部写成 ``g(X) <= 0``；``ph_stability_window`` 用 ``ANY_OF`` 表达
    "酸性窗口或碱性窗口"这种并集可行域（逐元素取 min 等价于逻辑或）。
    """

    # 1) 氧化剂稳定 pH 窗口
    if len(ph_windows) == 1:
        lo, hi = ph_windows[0]
        ph_c = HardConstraint(
            name="ph_stability_window",
            expression=f"{lo:g} <= pH <= {hi:g}",
            fn=lambda v: _ewmax(lo - v["ph"], v["ph"] - hi),
            logic=ConstraintLogic.ALL_OF,
            description="氧化剂稳定 pH 窗口",
            rationale="过氧化氢在近中性区分解最快（金属离子催化），放气导致去除率漂移与缺陷",
        )
    else:
        terms = ", ".join(f"[{lo:g}, {hi:g}]" for lo, hi in ph_windows)
        ph_c = HardConstraint(
            name="ph_stability_window",
            expression=f"pH in {terms}",
            fn=lambda v: _ewmin(
                *[_ewmax(lo - v["ph"], v["ph"] - hi) for lo, hi in ph_windows]
            ),
            logic=ConstraintLogic.ANY_OF,
            members=tuple(f"window_{i}" for i in range(len(ph_windows))),
            description="H2O2 酸性 / 碱性双稳定窗口（并集可行域）",
            rationale="pH 5.5–8.5 区间 H2O2 分解速率最高，浆料寿命与去除率不可控",
        )

    return (
        ph_c,
        HardConstraint(
            name="abrasive_oxidizer_ratio_low",
            expression="abrasive / oxidizer >= 0.3",
            fn=lambda v: 0.3 * v["oxidizer_wt_pct"] - v["abrasive_wt_pct"],
            description="磨料/氧化剂质量比下限",
            rationale="氧化剂相对磨料过量时化学去除占主导，表面出现蚀坑与氧化层残留",
        ),
        HardConstraint(
            name="abrasive_oxidizer_ratio_high",
            expression="abrasive / oxidizer <= 15",
            fn=lambda v: v["abrasive_wt_pct"] - 15.0 * v["oxidizer_wt_pct"],
            description="磨料/氧化剂质量比上限",
            rationale="磨料过量而氧化剂不足时进入纯机械抛光区，划伤与 Dishing 显著恶化",
        ),
        HardConstraint(
            name="total_solids",
            expression="abrasive + complexing + surfactant + inhibitor <= 20",
            fn=lambda v: (
                v["abrasive_wt_pct"]
                + v["complexing_wt_pct"]
                + v["surfactant_wt_pct"]
                + v["inhibitor_wt_pct"]
                - 20.0
            ),
            description="总固含上限（wt%）",
            rationale="总固含超过 20 wt% 后黏度骤升，供液管路沉积、颗粒沉降不可控",
        ),
        HardConstraint(
            name="surfactant_to_abrasive",
            expression="surfactant <= 0.25 * abrasive",
            fn=lambda v: v["surfactant_wt_pct"] - 0.25 * v["abrasive_wt_pct"],
            description="表面活性剂/磨料比例上限",
            rationale="活性剂过量包覆会屏蔽磨料机械作用，导致 MRR 塌陷并大量起泡",
        ),
        HardConstraint(
            name="mechanical_dose",
            expression=f"P * pi * {pad_diameter_m:g} * (rpm_p + rpm_c) / 120 <= 12",
            fn=lambda v: v["down_force_psi"]
            * math.pi
            * pad_diameter_m
            * (v["platen_rpm"] + v["carrier_rpm"])
            / 120.0
            - 12.0,
            description="机械作用强度 P·v 上限（psi·m/s）",
            rationale="高压力 × 高转速超过抛光盘/抛光垫的允许比压与剪切，导致垫面快速磨损与划伤",
        ),
        HardConstraint(
            name="chelator_to_oxidizer",
            expression="complexing <= 4 * oxidizer",
            fn=lambda v: v["complexing_wt_pct"] - 4.0 * v["oxidizer_wt_pct"],
            description="络合剂/氧化剂比例上限",
            rationale="络合剂远过量会直接溶解金属（化学腐蚀）而非去除氧化层，造成蚀坑与粗化",
        ),
        HardConstraint(
            name="ionic_strength_proxy",
            expression="0.28*complexing + 0.15*abs(pH - 7) <= 1.2",
            fn=lambda v: 0.28 * v["complexing_wt_pct"]
            + 0.15 * np.abs(v["ph"] - 7.0)
            - 1.2
            if not _is_torch(v["ph"])
            else 0.28 * v["complexing_wt_pct"]
            + 0.15 * torch.abs(v["ph"] - 7.0)
            - 1.2,
            description="离子强度代理上限（胶体稳定性）",
            rationale="盐类压缩双电层致胶体失稳；本体系盐负荷来自 pH 调节剂与络合剂阴离子，"
            "H2O2 为中性分子不贡献离子强度。系数需按实测 Zeta 电位标定",
        ),
        HardConstraint(
            name="flow_per_rpm",
            expression="flow >= 0.4 * platen_rpm",
            fn=lambda v: 0.4 * v["platen_rpm"] - v["slurry_flow_ml_min"],
            description="单位转速最低供液量（mL/min per rpm）",
            rationale="高转速下供液不足造成局部干抛，划伤率与面内去除不均同步上升",
        ),
    )


def _build_implicit_rules() -> Tuple[ImplicitRule, ...]:
    """隐式约束规则库：无可靠解析式，只做前置过滤与打标。"""
    return (
        ImplicitRule(
            name="high_loading_acid_settling",
            violates=lambda v: (v["abrasive_wt_pct"] > 8.0) & (v["ph"] < 4.0),
            severity=Severity.REJECT,
            action="丢弃：高固含浆料在强酸下快速沉降（胶体失稳）",
            rationale="pH 靠近等电点时 Zeta 电位趋零，颗粒团聚 + 重力沉降，"
            "放置 2 h 内即分层；该规则对氧化铈体系尤为敏感（经验规则）",
        ),
        ImplicitRule(
            name="surfactant_foaming",
            violates=lambda v: v["surfactant_wt_pct"] > 0.35,
            severity=Severity.WARN,
            action="打标：需确认消泡工艺，否则供液泵气蚀",
            rationale="活性剂接近/超过 CMC 后起泡严重，气泡夹带导致去除率波动",
        ),
        ImplicitRule(
            name="oxidizer_thermal_decomposition",
            violates=lambda v: (v["oxidizer_wt_pct"] > 6.0) & (v["ph"] > 9.5),
            severity=Severity.WARN,
            action="打标：碱性高浓度 H2O2 需控温 ≤25 ℃ 并缩短浆料放置时间",
            rationale="碱性条件 H2O2 分解活化能降低，放热可自加速",
        ),
        ImplicitRule(
            name="high_shear_agglomeration",
            violates=lambda v: (v["abrasive_wt_pct"] > 10.0) & (v["slurry_flow_ml_min"] < 90.0),
            severity=Severity.WARN,
            action="打标：高固含 + 低流量 → 局部高剪切团聚风险",
            rationale="窄间隙处剪切速率高，团聚体破碎后二次絮凝产生大颗粒划伤",
        ),
        ImplicitRule(
            name="inhibitor_overdose",
            violates=lambda v: (v["inhibitor_wt_pct"] > 0.2) & (v["ph"] < 3.0),
            severity=Severity.FLAG,
            action="记录：强酸下 BTA 成膜能力下降，需补做腐蚀电位测试",
            rationale="BTA 在低 pH 下质子化，吸附膜不完整",
        ),
    )


def _build_covariates() -> Tuple[BatchCovariate, ...]:
    return (
        BatchCovariate(
            "pad_use_count",
            "片",
            handling="gpr_input",
            description="抛光垫累计使用片数，垫面釉化使 MRR 单调下降",
            typical_range=(0.0, 400.0),
        ),
        BatchCovariate(
            "slurry_age_hours",
            "h",
            handling="gpr_input",
            description="配液到上机的时间，影响颗粒沉降与氧化剂有效浓度",
            typical_range=(0.0, 48.0),
            batch_constant=False,  # 同批次内每片晶圆的等待时间本来就不同
        ),
        BatchCovariate(
            "ambient_temp_c",
            "℃",
            handling="gpr_input",
            description="洁净间温度，影响浆料黏度与化学反应速率",
            typical_range=(18.0, 26.0),
        ),
        BatchCovariate(
            "ambient_humidity_pct",
            "%RH",
            handling="gpr_input",
            description="湿度影响抛光垫含水率与摩擦状态",
            typical_range=(35.0, 65.0),
        ),
        BatchCovariate(
            "tool_id",
            "-",
            handling="random_effect",
            description="机台编号，跨机台存在系统性偏移（随机效应）",
            typical_range=(1.0, 3.0),
        ),
    )


def _build_topsis() -> TopsisConfig:
    """TOPSIS 冻结权重：性能 85% + 成本 15%，初始化后不再调整。"""
    return TopsisConfig(
        weights=(
            TopsisCriterion("MRR", 0.40, "benefit", "去除速率，越高越好"),
            TopsisCriterion("Ra", 0.25, "cost", "粗糙度，越低越好"),
            TopsisCriterion("Dishing", 0.20, "cost", "碟形凹陷，越低越好"),
            TopsisCriterion("Cost", 0.15, "cost", "单次实验成本（元），越低越好"),
        ),
        entropy_weight_sensitivity_only=True,
        normalize="vector",
        hard_spec_filter_first=True,
    )


def build_config(family: str = "cu_peroxide_glycine", version: str = "0.1.0") -> CMPRecipeSpace:
    """按配方族构造完整问题定义。默认：铜 CMP + 胶体二氧化硅 + H2O2 + 甘氨酸。"""
    if family not in FAMILY_PRESETS:
        raise KeyError(f"未知配方族 {family!r}；可选：{sorted(FAMILY_PRESETS)}")
    preset = FAMILY_PRESETS[family]
    ph_windows: Tuple[Tuple[float, float], ...] = tuple(
        tuple(float(x) for x in w) for w in preset["ph_windows"]  # type: ignore[misc]
    )
    preston = PrestonPrior(k_mean=float(preset["preston_k"]))
    cost_model = replace(
        CostModel(), abrasive_price_per_kg=ABRASIVE_PRICES[preset["abrasive"]]
    )

    return CMPRecipeSpace(
        name=f"CMP 抛光液配方优化 / {family}",
        description=preset["description"],
        variables=_build_variables(preset["abrasive"], preset["oxidizer"], ph_windows),
        objectives=_build_objectives(),
        hard_constraints=_build_constraints(ph_windows, preston.pad_diameter_m),
        implicit_rules=_build_implicit_rules(),
        covariates=_build_covariates(),
        cost_model=cost_model,
        experiment=ExperimentPlan(),
        acquisition=AcquisitionConfig(),
        ref_point_policy=ReferencePointPolicy(),
        stopping=StoppingCriteria(),
        topsis=_build_topsis(),
        gpr_prior=GPRPriorConfig(),
        model_health=ModelHealthConfig(),
        preston=preston,
        family=family,
        contextual_variables=(
            DecisionVariable(
                name="abrasive_type",
                unit="-",
                low=0,
                high=1,
                kind=VariableKind.CATEGORICAL,
                categories=tuple(t for t, _ in CHEMISTRY_CATALOG["abrasive"]),
                mechanism="磨料硬度/粒径决定机械去除与划伤倾向，属配方族上下文",
            ),
            DecisionVariable(
                name="oxidizer_type",
                unit="-",
                low=0,
                high=1,
                kind=VariableKind.CATEGORICAL,
                categories=tuple(t for t, _ in CHEMISTRY_CATALOG["oxidizer"]),
                mechanism="氧化剂决定 pH 稳定窗口与氧化电位，属配方族上下文",
            ),
            # 以下三类是"添加剂种类"：它们改变响应曲面，但其**浓度**已经在
            # 连续变量里（surfactant_wt_pct / inhibitor_wt_pct），
            # 所以种类只作为分类上下文进台账，不进 GPR 的连续输入。
            DecisionVariable(
                name="surfactant_type",
                unit="-",
                low=0,
                high=1,
                kind=VariableKind.CATEGORICAL,
                categories=tuple(t for t, _ in CHEMISTRY_CATALOG["surfactant"]),
                mechanism="表面活性剂种类决定吸附构型：抑制凹处去除（降 Dishing）或屏蔽磨料（降 MRR）",
            ),
            DecisionVariable(
                name="inhibitor_type",
                unit="-",
                low=0,
                high=1,
                kind=VariableKind.CATEGORICAL,
                categories=tuple(t for t, _ in CHEMISTRY_CATALOG["inhibitor"]),
                mechanism="缓蚀剂种类决定铜面钝化膜强度：BTA 类成膜快、对 Dishing 与静态腐蚀影响大",
            ),
            DecisionVariable(
                name="ph_adjuster_type",
                unit="-",
                low=0,
                high=1,
                kind=VariableKind.CATEGORICAL,
                categories=tuple(t for t, _ in CHEMISTRY_CATALOG["ph_adjuster"]),
                mechanism="pH 调节剂决定离子强度与缓冲能力，间接影响胶体稳定性与氧化剂分解速率",
            ),
        ),
        version=version,
    )


def build_default_config() -> CMPRecipeSpace:
    """默认问题（铜 CMP，酸性/碱性双稳定窗口）。"""
    return build_config("cu_peroxide_glycine")


# --------------------------------------------------------------------------- #
# 9. RAG 知识库配置（切片策略、路径、服务地址）
# --------------------------------------------------------------------------- #
def _repo_root() -> "Path":
    return Path(__file__).resolve().parents[1]


def load_dotenv(path: Optional[Any] = None, *, override: bool = False) -> Dict[str, str]:
    """极简 ``.env`` 解析器（不引入 python-dotenv，3.11/3.15 都能用）。

    只支持 ``KEY=VALUE``、``#`` 注释、可选引号；解析结果写入 ``os.environ``
    （默认不覆盖已有环境变量，便于命令行临时覆盖）。
    """
    p = Path(path) if path is not None else _repo_root() / ".env"
    loaded: Dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None or v == "" else v


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RAGConfig:
    """RAG 子系统的全部可调项（切片、检索、路径、LLM）。

    为什么切片是"大 chunk"：CMP 文献里一段配方往往横跨多行
    （磨料 5 wt%、pH 4.2、压力 2 psi、转速 93/87 rpm…），默认 750 字符的
    细切会把它们劈开，检索到的片段缺关键参数就没法回答"这个体系用什么配方"。
    因此默认 ``chunk_size=1800``、``chunk_overlap=300``，并且**按 Markdown
    标题先分段、再在段内合并**，让一个 chunk 尽量覆盖完整的配方块。
    """

    # -- 服务 --
    service_url: str = "http://127.0.0.1:8770"
    service_port: int = 8770

    # -- 原始资产（绝对路径；BGE-M3 有 2.3 GB，绝不复制）--
    kb_dir: str = r"E:\private_rag_cmp\private_kb"
    record_file: str = r"E:\private_rag_cmp\kb_record.json"
    model_dir: str = r"E:\private_rag_cmp\model\bge-m3"
    archive_dir: str = r"E:\private_rag_cmp\summary_archive"
    collection: str = "langchain"

    # -- 切片（化工文献：大块，保配方完整）--
    chunk_size: int = 1800
    chunk_overlap: int = 300

    # -- 检索 --
    top_k: int = 4
    retrieve_multiplier: int = 2

    # -- LLM --
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    llm_temperature: float = 0.15
    llm_timeout_s: int = 120

    # -- 行为 --
    max_upload_mb: int = 20
    health_timeout_s: int = 3

    @classmethod
    def from_env(cls, dotenv_path: Optional[Any] = None) -> "RAGConfig":
        load_dotenv(dotenv_path)
        return cls(
            service_url=_env_str("RAG_SERVICE_URL", "http://127.0.0.1:8770").rstrip("/"),
            service_port=_env_int("RAG_PORT", 8770),
            kb_dir=_env_str("RAG_KB_DIR", r"E:\private_rag_cmp\private_kb"),
            record_file=_env_str("RAG_RECORD_FILE", r"E:\private_rag_cmp\kb_record.json"),
            model_dir=_env_str("RAG_MODEL_DIR", r"E:\private_rag_cmp\model\bge-m3"),
            archive_dir=_env_str("RAG_ARCHIVE_DIR", r"E:\private_rag_cmp\summary_archive"),
            collection=_env_str("RAG_COLLECTION", "langchain"),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 1800),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 300),
            top_k=_env_int("RAG_TOP_K", 4),
            retrieve_multiplier=_env_int("RAG_RETRIEVE_MULTIPLIER", 2),
            llm_model=_env_str("DEEPSEEK_MODEL", "deepseek-chat"),
            llm_base_url=_env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=_env_str("DEEPSEEK_API_KEY", ""),
            llm_timeout_s=_env_int("DEEPSEEK_TIMEOUT", 120),
            max_upload_mb=_env_int("RAG_MAX_UPLOAD_MB", 20),
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())

    def describe_paths(self) -> str:
        return (f"KB={self.kb_dir}  |  model={self.model_dir}  |  "
                f"chunk={self.chunk_size}/{self.chunk_overlap}")


#: 进程级单例：模块导入时就加载 .env（网页与 RAG 服务共用同一份配置）
RAG_CONFIG = RAGConfig.from_env()


# --------------------------------------------------------------------------- #
# 9. 自检：``python -m src.config``
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:  # Windows 控制台 GBK 下也能正确输出中文
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    cfg = build_default_config()
    print(cfg.describe())

    print("\n" + "=" * 100)
    print("自检 1：边界与单位立方往返")
    x = cfg.sample_lhs(5, seed=7)
    rt = cfg.from_unit(cfg.to_unit(x))
    assert np.allclose(rt, x, atol=1e-9), "物理量纲 <-> 单位立方 往返失败"
    print(f"  LHS 5 条配方，往返最大误差 {np.abs(rt - x).max():.2e}")

    print("\n自检 2：硬约束可行性判定")
    baseline = {
        "abrasive_wt_pct": 3.0,
        "oxidizer_wt_pct": 2.0,
        "complexing_wt_pct": 1.0,
        "surfactant_wt_pct": 0.10,
        "inhibitor_wt_pct": 0.02,
        "ph": 4.0,
        "down_force_psi": 2.0,
        "platen_rpm": 93.0,
        "carrier_rpm": 87.0,
        "slurry_flow_ml_min": 200.0,
    }
    rep = cfg.validate_recipe(baseline)
    print(f"  基准配方 ok={rep['ok']} 命中={rep['violations']} 成本={rep['cost_cny']} 元/次")
    assert rep["ok"], "基准配方应当可行"

    for label, patch in (
        ("pH=7 落在分解带", {"ph": 7.0}),
        ("表面活性剂过量", {"surfactant_wt_pct": 0.5, "abrasive_wt_pct": 1.0}),
        ("高压高转速", {"down_force_psi": 5.0, "platen_rpm": 150.0, "carrier_rpm": 135.0}),
    ):
        r = cfg.validate_recipe({**baseline, **patch})
        print(f"  {label}: ok={r['ok']} 命中={list(r['violations'])}")
        assert not r["ok"], f"{label} 应判为不可行"

    print("\n自检 3：随机采样可行性（含中间 pH 带，整体不应为 100%）")
    xs = cfg.sample_lhs(2000, seed=1)
    diag = cfg.constraint_diagnostics(xs)
    for k, v in diag.items():
        tag = "全部硬约束" if k == "__all__" else k
        print(f"  通过率 {tag:<32} {v:6.1f}%")
    kept, keep_mask, hits = cfg.filter_implicit(cfg.sample_lhs_feasible(16, seed=3))
    print(f"  可行域 LHS 首批 16 条；隐式过滤后 {len(kept)} 条；命中 {hits}")

    print("\n自检 4：参考点 / 目标符号 / 规格满足率")
    print(f"  初始化 ref_point = {np.round(np.asarray(cfg.reference_point()), 2).tolist()}")
    print(f"  目标符号矩阵   = {np.asarray(cfg.objective_matrix()).ravel().tolist()}")
    good = np.array([[1800.0, 3.2, 120.0], [2600.0, 2.1, 90.0]])
    bad = np.array([[900.0, 5.5, 260.0], [100.0, 9.9, 440.0]])
    print(f"  合格批次规格满足率 = {cfg.spec_hit_rate(good):.0%}")
    print(f"  不合格批次规格满足率 = {cfg.spec_hit_rate(bad):.0%}")
    print(f"  观测极差批次后 ref_point = {np.round(cfg.update_reference_point(bad), 2).tolist()}")
    print("  （仅当观测劣于先验下限时才下移，保证跨轮次 HV 可比）")

    print("\n自检 5：成本模型对 MRR 与磨料浓度的敏感性")
    mid = np.mean(cfg.bounds(), axis=0, keepdims=True)
    for mrr in (600.0, 1500.0, 3000.0):
        c = float(np.asarray(cfg.cost(mid, mrr=mrr)).reshape(-1)[0])
        print(f"  MRR={mrr:6.0f} Å/min -> {c:7.0f} 元/次")
    for a, name in ((1.0, "silica@1wt%"), (8.0, "silica@8wt%")):
        r = cfg.validate_recipe(
            {**baseline, "abrasive_wt_pct": a, "oxidizer_wt_pct": max(2.0, a / 6.0)}
        )["cost_cny"]
        print(f"  {name:<14} -> {r:7.1f} 元/次")

    print(f"\n配置指纹 hash={cfg.config_hash()}")
    print("全部自检通过。")
