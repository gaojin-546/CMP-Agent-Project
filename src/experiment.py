"""CMP 抛光液配方开发 Agent —— 模块 5：实验台账 + 校验器 + 数据三分。

模块 5 是"环境交互层"的落地面。它解决的不是"怎么存数据"，而是三个更硬的
工业问题：

1. **数据回写契约**：一条实验记录到底有哪些字段、哪些必填、哪些可以为空。
   模块 1 的 ``RecordSchema`` 定义契约，本模块负责**落地、校验、迁移**。
2. **异常三分**（本模块的核心）：
   * ``measurement``：仪器故障 / 操作失误 → **可剔除**，但必须留在台账里可追溯；
   * ``response``：配方本身导致的反常结果 → **必须保留**，只打标记。删掉它
     就是把真实物理当噪声扔掉，模型会学到"这片区域不存在"的错误先验；
   * ``failure``：配不出、划伤、团聚等 → **不进主目标训练**，单独建模（失效概率
     分类 GPR 的标注样本），因为它们根本没有可信的性能值。
3. **批次协变量校验**：``pad_use_count`` / ``ambient_temp_c`` 这类协变量是
   GPR 的输入维度，脏的协变量会直接污染核函数。校验要区分"缺失"（可容忍，
   但会降低模型可用维度）与"非法"（负的使用片数、>100% 的湿度 → 必须报错）。

设计原则：**校验器不修改数据**。它只报告 ``level``（error/warning）、
``code``、``field`` 与样例记录号；修数据是人的决定，台账要保留原始值。

用法
----
    >>> from src.config import build_default_config
    >>> from src.experiment import ExperimentLedger
    >>> space = build_default_config()
    >>> ledger = ExperimentLedger.from_dataset("data/closed_loop_n34.json", space)
    >>> report = ledger.validate()
    >>> print(report.describe())
    >>> split = ledger.split()
    >>> split.train_X.shape, split.train_Y.shape      # 可直接喂模块 2/3
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.config import CMPRecipeSpace, RecordSchema

__all__ = [
    "Anomaly",
    "Failure",
    "ExperimentRecord",
    "ValidationIssue",
    "ValidationReport",
    "CovariateReport",
    "DataSplit",
    "ExperimentLedger",
    "LEDGER_SCHEMA",
]

#: 台账文件格式版本。与模块 3 的 ``cmp-closed-loop-v1`` 不同：
#: 那一版只有 recipe + responses，这一版补齐了协变量、异常标记、批次与成本。
LEDGER_SCHEMA = "cmp-ledger-v2"

#: 目标名 -> 台账字段名（严格遵循模块 1 的 RecordSchema）。
Y_FIELD: Dict[str, str] = {"MRR": "y_mrr", "Ra": "y_ra", "Dishing": "y_dishing"}


# --------------------------------------------------------------------------- #
# 1. 枚举与记录
# --------------------------------------------------------------------------- #
class Anomaly(str, Enum):
    """响应异常标签（决定该记录是否进入主目标训练）。"""

    NONE = "none"
    MEASUREMENT = "measurement"
    """仪器故障 / 操作失误：可剔除出训练集，但必须留在台账里。"""
    RESPONSE = "response"
    """配方本身导致的反常结果：**必须保留**并打标记（它可能是真实物理）。"""


class Failure(str, Enum):
    """失效标签（不进入主目标训练，供失效概率分类器学习）。"""

    NONE = "none"
    MIX_FAIL = "mix_fail"
    SCRATCH = "scratch"
    AGGLOMERATION = "agglomeration"
    OTHER = "other"


@dataclass
class ExperimentRecord:
    """一条实验记录（字段与模块 1 的 ``RecordSchema`` 一一对应）。"""

    experiment_id: str
    round_id: int
    batch_id: str
    config_hash: str
    recipe: Dict[str, float]
    y: Dict[str, Optional[float]]
    covariates: Dict[str, float] = field(default_factory=dict)
    cost_cny: Optional[float] = None
    anomaly: Anomaly = Anomaly.NONE
    failure: Failure = Failure.NONE
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    operator: str = ""
    timestamp: str = ""
    notes: str = ""

    # -- 序列化 ----------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "round_id": self.round_id,
            "batch_id": self.batch_id,
            "config_hash": self.config_hash,
            "recipe": {k: float(v) for k, v in self.recipe.items()},
        }
        for name, field_name in Y_FIELD.items():
            out[field_name] = None if self.y.get(name) is None else float(self.y[name])  # type: ignore[arg-type]
        out.update({
            "covariates": {k: float(v) for k, v in self.covariates.items()},
            "cost_cny": None if self.cost_cny is None else float(self.cost_cny),
            "anomaly": self.anomaly.value,
            "failure": self.failure.value,
            "uncertainty": dict(self.uncertainty),
            "operator": self.operator,
            "timestamp": self.timestamp,
            "notes": self.notes,
        })
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(d["experiment_id"]),
            round_id=int(d["round_id"]),
            batch_id=str(d["batch_id"]),
            config_hash=str(d.get("config_hash", "")),
            recipe={k: float(v) for k, v in dict(d["recipe"]).items()},
            y={n: (None if d.get(f) is None else float(d[f])) for n, f in Y_FIELD.items()},
            covariates={k: float(v) for k, v in dict(d.get("covariates") or {}).items()},
            cost_cny=None if d.get("cost_cny") is None else float(d["cost_cny"]),
            anomaly=Anomaly(d.get("anomaly", "none")),
            failure=Failure(d.get("failure", "none")),
            uncertainty=dict(d.get("uncertainty") or {}),
            operator=str(d.get("operator", "")),
            timestamp=str(d.get("timestamp", "")),
            notes=str(d.get("notes", "")),
        )


# --------------------------------------------------------------------------- #
# 2. 校验结果
# --------------------------------------------------------------------------- #
@dataclass
class ValidationIssue:
    """一条（可聚合的）校验问题。"""

    level: str          # "error" / "warning"
    code: str           # 机器可读的错误码，便于下游按类别处理
    field: str
    message: str
    count: int = 1
    examples: List[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.level == "error"


class ReportBuilder:
    """把同一 ``(code, field, message)`` 的问题聚合成一条，避免刷屏。"""

    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str, str, str], ValidationIssue] = {}

    def add(
        self, level: str, code: str, field_: str, message: str, record_id: str = ""
    ) -> None:
        key = (level, code, field_, message)
        issue = self._items.get(key)
        if issue is None:
            issue = ValidationIssue(level=level, code=code, field=field_, message=message)
            self._items[key] = issue
        else:
            issue.count += 1
        if record_id and len(issue.examples) < 3:
            issue.examples.append(record_id)

    def build(self) -> List[ValidationIssue]:
        order = {"error": 0, "warning": 1}
        return sorted(
            self._items.values(),
            key=lambda i: (order.get(i.level, 9), -i.count, i.code),
        )


@dataclass
class ValidationReport:
    """整本台账的校验结果。"""

    n_records: int
    issues: List[ValidationIssue]

    @property
    def n_errors(self) -> int:
        return sum(i.count for i in self.issues if i.is_error)

    @property
    def n_warnings(self) -> int:
        return sum(i.count for i in self.issues if not i.is_error)

    @property
    def ok(self) -> bool:
        """有 error 即视为不可直接用于训练；warning 不影响可用性。"""
        return self.n_errors == 0

    def codes(self, level: Optional[str] = None) -> Dict[str, int]:
        """按错误码汇总计数。

        注意必须**累加**而不是简单赋值：同一个错误码可能作用在不同字段上
        （例如 ``covariate_out_of_typical`` 同时命中 pad_use_count 与
        ambient_temp_c），用字典推导式会互相覆盖、把计数悄悄改小。
        """
        out: Dict[str, int] = {}
        for i in self.issues:
            if level is None or i.level == level:
                out[i.code] = out.get(i.code, 0) + i.count
        return out

    def describe(self, width: int = 30) -> str:
        if not self.issues:
            return f"校验通过：{self.n_records} 条记录，无问题。"
        lines = [
            f"台账校验：{self.n_records} 条记录  →  "
            f"{self.n_errors} 个 error，{self.n_warnings} 个 warning",
            "-" * 96,
            f"{'级别':<8}{'错误码':<28}{'字段':<18}{'次数':>5}  说明",
            "-" * 96,
        ]
        for i in self.issues:
            flag = "ERROR" if i.is_error else "warn"
            lines.append(
                f"{flag:<8}{i.code[:26]:<28}{i.field[:16]:<18}{i.count:>5}  {i.message}"
            )
            if i.examples:
                lines.append(f"{'':<8}  例：{', '.join(i.examples)}")
        lines.append("-" * 96)
        lines.append("结论：" + ("erroneous，需人工修正后才能训练" if not self.ok
                                 else "可训练（warning 不阻塞，但会降低模型可用维度）"))
        return "\n".join(lines)


@dataclass
class CovariateReport:
    """批次协变量的结构化覆盖度报告（给台账看板用）。"""

    names: Tuple[str, ...]
    n_records: int
    coverage: Dict[str, float]
    out_of_typical: Dict[str, int]
    missing: Dict[str, int]
    handling: Dict[str, str]

    def describe(self) -> str:
        lines = [
            f"批次协变量校验（{self.n_records} 条记录，{len(self.names)} 个协变量）",
            "-" * 84,
            f"{'协变量':<24}{'用途':<14}{'覆盖率':>8}{'缺失':>6}{'超出典型范围':>12}",
            "-" * 84,
        ]
        for n in self.names:
            lines.append(
                f"{n:<24}{self.handling[n]:<14}{self.coverage[n]:>7.0%}"
                f"{self.missing[n]:>6}{self.out_of_typical[n]:>12}"
            )
        lines.append("-" * 84)
        lines.append(
            "缺失不阻塞训练（模块 2 会把该维度当未知），但超出典型范围会让 GPR 外推，"
            "应结合模块 7 的域外标志一起看。"
        )
        return "\n".join(lines)


@dataclass
class DataSplit:
    """三分结果：可训练 / 剔除 / 标记保留 / 失效（单独建模）。"""

    space: CMPRecipeSpace
    train_X: np.ndarray
    train_Y: np.ndarray
    train_ids: List[str]
    response_flagged_ids: List[str]
    excluded_measurement_ids: List[str]
    failure_X: np.ndarray
    failure_labels: List[str]
    failure_ids: List[str]
    incomplete_ids: List[str]

    # -- 摘要 ------------------------------------------------------------- #
    @property
    def n_train(self) -> int:
        return int(self.train_X.shape[0])

    def summary(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_train + len(self.excluded_measurement_ids)
            + len(self.failure_ids) + len(self.incomplete_ids),
            "n_train": self.n_train,
            "n_response_flagged": len(self.response_flagged_ids),
            "n_excluded_measurement": len(self.excluded_measurement_ids),
            "n_failure": len(self.failure_ids),
            "n_incomplete": len(self.incomplete_ids),
        }

    def describe(self) -> str:
        s = self.summary()
        lines = [
            f"数据三分：{s['n_total']} 条记录",
            f"  ✅ 进主目标训练      {s['n_train']:>3} 条"
            f"（含 {s['n_response_flagged']} 条 response 异常，已标记但保留）",
            f"  ⛔ 测量异常剔除      {s['n_excluded_measurement']:>3} 条"
            f"（保留在台账，不进训练）",
            f"  🔬 失效样本单独建模  {s['n_failure']:>3} 条"
            f"（供失效概率分类，不参与主目标）",
            f"  ❓ 数据不完整        {s['n_incomplete']:>3} 条"
            f"（既无性能值也无失效标签）",
        ]
        if s["n_train"]:
            lines.append(
                f"  训练矩阵：X {self.train_X.shape}  Y {self.train_Y.shape}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. 台账
# --------------------------------------------------------------------------- #
class ExperimentLedger:
    """实验台账：记录的容器 + 校验器 + 三分器。"""

    def __init__(self, space: CMPRecipeSpace, records: Optional[List[ExperimentRecord]] = None,
                 provenance: Optional[Dict[str, Any]] = None) -> None:
        self.space = space
        self.records: List[ExperimentRecord] = list(records or [])
        #: 记录每个字段是"实测"还是"派生/缺失"，迁移时必须如实标注
        self.provenance: Dict[str, Any] = dict(provenance or {})

    # -- 基本 ------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterable[ExperimentRecord]:
        return iter(self.records)

    def append(self, record: ExperimentRecord) -> None:
        """追加一条记录（不做校验，校验由 :meth:`validate` 统一做）。"""
        self.records.append(record)

    @property
    def objective_names(self) -> Tuple[str, ...]:
        return self.space.objective_names

    # -- 迁移：从模块 3 的闭环数据集 -------------------------------------- #
    @classmethod
    def from_dataset(
        cls, path: Any, space: CMPRecipeSpace
    ) -> "ExperimentLedger":
        """把模块 3 的 ``cmp-closed-loop-v1`` 数据集迁移成完整台账。

        迁移必须**如实标注派生字段**：那一版只有配方与响应，
        协变量、批次、成本、时间戳要么从 ``extra.rounds`` 里取，要么根本不存在。
        这里把不存在的东西留空（而不是编一个看起来合理的值），
        并在 ``provenance`` 里写清楚——否则台账会骗人，比没有台账更危险。
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema != "cmp-closed-loop-v1":
            raise ValueError(f"数据集格式不匹配：期望 cmp-closed-loop-v1，实际 {schema}")

        objective_names = list(payload.get("objective_names", space.objective_names))
        X = np.asarray(payload["recipes"], dtype=float)
        Y = np.asarray(payload["responses"], dtype=float)
        n = X.shape[0]
        config_hash = str(payload.get("config_hash", ""))
        extra = payload.get("extra") or {}
        rounds_meta = extra.get("rounds") or []

        # 前 n_init 条为首批 LHS（模块 3 的闭环里固定 25 条），之后每轮 q 条
        q = int(rounds_meta[0]["recipes"].__len__()) if rounds_meta else 3
        n_init = n - q * len(rounds_meta) if rounds_meta else n

        # 成本：闭环的 extra 里逐轮存过，按行号对齐
        cost_by_row: Dict[int, float] = {}
        for r_i, meta in enumerate(rounds_meta):
            costs = meta.get("cost_cny") or []
            for k, c in enumerate(costs):
                row = n_init + r_i * q + k
                if row < n:
                    cost_by_row[row] = float(c)

        records: List[ExperimentRecord] = []
        for i in range(n):
            round_id = 0 if i < n_init else 1 + (i - n_init) // q
            records.append(
                ExperimentRecord(
                    experiment_id=f"E{i:03d}",
                    round_id=round_id,
                    # 旧格式没有批次号：用"每轮一个配液批次"的约定派生，并记在 provenance 里
                    batch_id=f"B{round_id:02d}",
                    config_hash=config_hash,
                    recipe={name: float(X[i, j]) for j, name in enumerate(space.names)},
                    y={
                        name: float(Y[i, j])
                        for j, name in enumerate(objective_names)
                        if j < Y.shape[1]
                    },
                    covariates={},          # 旧格式没有协变量：留空，绝不编造
                    cost_cny=cost_by_row.get(i),
                    anomaly=Anomaly.NONE,
                    failure=Failure.NONE,
                    timestamp="",            # 旧格式没有时间戳
                    notes="migrated from cmp-closed-loop-v1",
                )
            )

        provenance = {
            "source": str(Path(path).name),
            "source_schema": schema,
            "target_schema": LEDGER_SCHEMA,
            "measured_fields": ["recipe", "y_mrr", "y_ra", "y_dishing", "cost_cny(逐轮)"],
            "derived_fields": {
                "experiment_id": "按行号生成 E000…",
                "round_id": "前 25 条=0（首批），之后每 3 条一轮",
                "batch_id": "按轮派生 B00/B01…（旧格式无真实批次号）",
            },
            "missing_fields": ["covariates", "operator", "timestamp", "uncertainty"],
            "note": "缺失字段保持为空，由 validate() 报 warning；禁止用推测值填充",
        }
        return cls(space, records, provenance)

    # -- 校验 ------------------------------------------------------------- #
    def validate(self, *, strict_units: bool = True) -> ValidationReport:
        """结构化校验：结构 → 配方 → 响应 → 协变量 → 成本 → 时间戳 → 批次。

        顺序是有意的：先保证"字段在、配方合法"，再看"数值可信"，
        最后才做跨记录的批次一致性——否则会在脏数据上浪费时间。
        """
        b = ReportBuilder()
        required = {f.name for f in RecordSchema if f.required}
        seen_ids: Dict[str, str] = {}
        batch_cov: Dict[str, Dict[str, float]] = {}
        prev_ts: Optional[datetime] = None
        cov_names = tuple(c.name for c in self.space.covariates)
        cov_handling = {c.name: c.handling for c in self.space.covariates}
        cov_range = {c.name: c.typical_range for c in self.space.covariates}
        cov_batch_constant = {c.name: c.batch_constant for c in self.space.covariates}

        for rec in self.records:
            rid = rec.experiment_id

            # --- R1 结构 --- #
            for f_ in ("experiment_id", "round_id", "batch_id", "recipe"):
                if f_ not in rec.to_dict() or rec.to_dict()[f_] in (None, {}):
                    b.add("error", "missing_field", f_, f"必填字段 {f_} 缺失", rid)
            if rid in seen_ids:
                b.add("error", "duplicate_id", "experiment_id",
                      f"实验编号重复（另一条为 {seen_ids[rid]}）", rid)
            else:
                seen_ids[rid] = rid

            # --- R2 配方 --- #
            if set(rec.recipe) != set(self.space.names):
                missing = set(self.space.names) - set(rec.recipe)
                extra = set(rec.recipe) - set(self.space.names)
                if missing:
                    b.add("error", "recipe_missing_var", "recipe",
                          f"缺少设计变量 {sorted(missing)}", rid)
                if extra:
                    b.add("error", "recipe_unknown_var", "recipe",
                          f"未知设计变量 {sorted(extra)}", rid)
            else:
                x = np.array([[rec.recipe[n] for n in self.space.names]])
                for v, val in zip(self.space.variables, x[0]):
                    if not (v.low <= val <= v.high):
                        b.add("error", "recipe_out_of_bounds", v.name,
                              f"{v.name}={val:.3g} 超出 [{v.low:g}, {v.high:g}]", rid)
                for con in self.space.hard_constraints:
                    g = float(np.asarray(con.bind(self.space.names)(x)).reshape(-1)[0])
                    # 容差 1e-6：SLSQP 返回的边界解常压在 g≈1e-13 上，
                    # 用严格 g>0 会把数值零判成违约（模块 3 已踩过同一个坑）。
                    if g > 1e-6:
                        b.add("error", "recipe_hard_constraint", con.name,
                              f"违反硬约束 {con.name}（{con.expression}），"
                              f"超界 {g:.3g}", rid)
                for rule in self.space.implicit_rules:
                    if bool(np.asarray(rule.bind(self.space.names)(x)).reshape(-1)[0]):
                        # 隐式规则只应约束**候选生成**，不能追溯性否定已完成的实验：
                        # 这条数据是真实测出来的，删掉它等于抹掉一段真实物理。
                        # 所以哪怕严重级是 REJECT，在台账里也只记 warning。
                        if rule.severity.value == "reject":
                            b.add("warning", "recipe_implicit_reject", rule.name,
                                  f"命中 REJECT 级规则（{rule.action}）——数据保留："
                                  f"规则用于约束候选生成，不用于否定既成实验", rid)
                        else:
                            b.add("warning", "recipe_implicit_rule", rule.name,
                                  rule.action, rid)

            # --- R3 响应 --- #
            has_failure = rec.failure is not Failure.NONE
            for name in self.objective_names:
                val = rec.y.get(name)
                if val is None:
                    if not has_failure:
                        b.add("error", "y_missing", f"y_{name.lower()}",
                              f"{name} 缺失且无失效标签（记录不可用）", rid)
                    continue
                if not math.isfinite(float(val)):
                    b.add("error", "y_not_finite", f"y_{name.lower()}",
                          f"{name}={val} 非有限值", rid)
                    continue
                if float(val) < 0:
                    b.add("error", "y_negative", f"y_{name.lower()}",
                          f"{name}={val:.3g} 为负，物理上不可能", rid)
                # 先验范围不是硬边界，只用来提示"这条数据在经验之外"。
                #
                # 关键细节：**"优于先验"不能当成异常**。Dishing 只有 8 Å
                # （先验下限 30 Å）是我们梦寐以求的结果，把它标成"疑单位错误"
                # 等于让校验器对好消息报警——实测踩到过这个坑。
                # 因此单位错误只在跨越**一个数量级**时才报（10× 是最典型的
                # 单位混用，如 nm 当 Å、Å/min 当 nm/min）。
                obj = self.space.objectives[self.space.objective_names.index(name)]
                lo, hi = obj.prior_range
                if strict_units and (val < lo / 10.0 or val > hi * 10.0):
                    b.add("warning", "y_unit_suspect", f"y_{name.lower()}",
                          f"{name}={val:.3g} 与先验范围 [{lo:g}, {hi:g}] 相差一个数量级"
                          f"以上，疑单位混用", rid)
                elif val < lo or val > hi:
                    direction = (
                        "优于先验下限" if val < lo else
                        ("优于先验上限" if obj.direction.value == "maximize"
                         else "劣于先验上限")
                    )
                    b.add("warning", "y_out_of_prior_range", f"y_{name.lower()}",
                          f"{name}={val:.3g} 超出先验范围 [{lo:g}, {hi:g}]"
                          f"（{direction}）", rid)

            # --- R4 协变量 --- #
            if not rec.covariates:
                b.add("warning", "covariates_missing", "covariates",
                      "整条记录没有批次协变量（模型将缺少漂移维度）", rid)
            else:
                unknown = set(rec.covariates) - set(cov_names)
                if unknown:
                    b.add("warning", "covariate_unknown", "covariates",
                          f"未知协变量 {sorted(unknown)}（会被忽略）", rid)
                for name in cov_names:
                    if name not in rec.covariates:
                        b.add("warning", "covariate_missing_field", name,
                              f"缺少协变量 {name}", rid)
                        continue
                    v = float(rec.covariates[name])
                    if not math.isfinite(v):
                        b.add("error", "covariate_not_finite", name,
                              f"{name} 非有限值", rid)
                        continue
                    lo, hi = cov_range[name]
                    if name in ("pad_use_count", "slurry_age_hours") and v < 0:
                        b.add("error", "covariate_negative", name,
                              f"{name}={v:g} 不能为负", rid)
                    if name == "ambient_humidity_pct" and not (0 <= v <= 100):
                        b.add("error", "covariate_out_of_physical", name,
                              f"湿度 {v:g}% 超出 0–100", rid)
                    if name == "tool_id" and v not in (1.0, 2.0, 3.0):
                        b.add("error", "covariate_bad_category", name,
                              f"tool_id={v:g} 不在 {{1,2,3}} 内", rid)
                    if v < lo or v > hi:
                        b.add("warning", "covariate_out_of_typical", name,
                              f"{name}={v:g} 超出典型范围 [{lo:g}, {hi:g}]，"
                              f"模型将外推", rid)

            # --- R5 成本 --- #
            if rec.cost_cny is not None:
                c = float(rec.cost_cny)
                if not math.isfinite(c) or c < 0:
                    b.add("error", "cost_invalid", "cost_cny", f"成本 {c} 非法", rid)
                else:
                    # 用模块 1 的成本模型反算，偏差过大说明口径不一致
                    try:
                        x = np.array([[rec.recipe[n] for n in self.space.names]])
                        mrr = rec.y.get("MRR")
                        expect = float(np.asarray(
                            self.space.cost(x, mrr=None if mrr is None else float(mrr))
                        ).reshape(-1)[0])
                        if abs(c - expect) / max(expect, 1e-9) > 0.30:
                            b.add("warning", "cost_inconsistent", "cost_cny",
                                  f"记录成本 {c:.0f} 元与成本模型 {expect:.0f} 元"
                                  f"偏差 >30%", rid)
                    except Exception:
                        pass

            # --- R6 配置指纹 --- #
            if rec.config_hash and rec.config_hash != self.space.config_hash():
                b.add("warning", "config_hash_mismatch", "config_hash",
                      f"记录指纹 {rec.config_hash} ≠ 当前 {self.space.config_hash()}"
                      f"（参数空间可能已变更）", rid)

            # --- R7 时间戳 --- #
            if not rec.timestamp:
                b.add("warning", "timestamp_missing", "timestamp", "缺少时间戳", rid)
            else:
                try:
                    ts = datetime.fromisoformat(rec.timestamp.replace("Z", "+00:00"))
                    if prev_ts is not None and ts < prev_ts:
                        b.add("warning", "timestamp_out_of_order", "timestamp",
                              "时间戳早于前一条记录（台账顺序可疑）", rid)
                    prev_ts = ts
                except ValueError:
                    b.add("error", "timestamp_unparsable", "timestamp",
                          f"时间戳无法解析：{rec.timestamp}", rid)

            # --- R8 异常/失效组合 --- #
            if has_failure and any(rec.y.get(n) is not None for n in self.objective_names):
                b.add("warning", "failure_with_values", "failure",
                      "失效样本带有性能值（部分失效？这些值不参与主目标）", rid)
            if rec.anomaly is Anomaly.RESPONSE and all(
                rec.y.get(n) is None for n in self.objective_names
            ):
                b.add("error", "response_without_values", "anomaly",
                      "标记为 response 异常却没有性能值（应改为 failure）", rid)

            # --- R9 批次一致性 --- #
            # 只比较"批次级"协变量：环境温湿度、机台号、垫片数在同一批内应当恒定。
            # slurry_age_hours 天然逐条变化（同批次先后抛光的等待时间不同），
            # 拿它做一致性校验会产生大量假警报（实测一次报了 24 条）。
            if rec.covariates:
                key = rec.batch_id
                batch_level = {
                    k: float(v) for k, v in rec.covariates.items()
                    if k in cov_names and cov_batch_constant.get(k, True)
                }
                if key in batch_cov:
                    if any(
                        abs(batch_level.get(k, float("nan")) - v) > 1e-9
                        for k, v in batch_cov[key].items()
                        if k in batch_level
                    ):
                        b.add("warning", "batch_covariate_inconsistent", "batch_id",
                              f"同一批次 {key} 内的批次级协变量不一致", rid)
                else:
                    batch_cov[key] = batch_level

        return ValidationReport(n_records=len(self.records), issues=b.build())

    # -- 协变量报告 ------------------------------------------------------- #
    def covariate_report(self) -> CovariateReport:
        covs = list(self.space.covariates)
        n = len(self.records)
        coverage, missing, oot = {}, {}, {}
        for c in covs:
            present = [r.covariates[c.name] for r in self.records if c.name in r.covariates]
            coverage[c.name] = len(present) / n if n else 0.0
            missing[c.name] = n - len(present)
            lo, hi = c.typical_range
            oot[c.name] = sum(1 for v in present if v < lo or v > hi)
        return CovariateReport(
            names=tuple(c.name for c in covs),
            n_records=n,
            coverage=coverage,
            out_of_typical=oot,
            missing=missing,
            handling={c.name: c.handling for c in covs},
        )

    # -- 三分 ------------------------------------------------------------- #
    def split(self) -> DataSplit:
        """按异常三分法切出四个集合。

        判定顺序（重要）：
        1. ``failure != none`` → 失效集（**即使有性能值也不进训练**，因为它的
           性能值来自一个崩掉的实验，只会污染模型）；
        2. ``anomaly == measurement`` → 剔除集；
        3. 三个目标全有值 → 训练集（``anomaly == response`` 也进，只打标记）；
        4. 其余 → 不完整集。
        """
        train_X, train_Y, train_ids = [], [], []
        flagged: List[str] = []
        excluded: List[str] = []
        fail_X, fail_labels, fail_ids = [], [], []
        incomplete: List[str] = []

        for rec in self.records:
            vals = [rec.y.get(n) for n in self.objective_names]
            if rec.failure is not Failure.NONE:
                fail_ids.append(rec.experiment_id)
                fail_labels.append(rec.failure.value)
                fail_X.append([rec.recipe.get(n, float("nan")) for n in self.space.names])
                continue
            if rec.anomaly is Anomaly.MEASUREMENT:
                excluded.append(rec.experiment_id)
                continue
            if any(v is None or not math.isfinite(float(v)) for v in vals):
                incomplete.append(rec.experiment_id)
                continue
            train_ids.append(rec.experiment_id)
            train_X.append([rec.recipe[n] for n in self.space.names])
            train_Y.append([float(v) for v in vals])
            if rec.anomaly is Anomaly.RESPONSE:
                flagged.append(rec.experiment_id)

        return DataSplit(
            space=self.space,
            train_X=np.asarray(train_X, dtype=float).reshape(-1, len(self.space.names)),
            train_Y=np.asarray(train_Y, dtype=float).reshape(-1, len(self.objective_names)),
            train_ids=train_ids,
            response_flagged_ids=flagged,
            excluded_measurement_ids=excluded,
            failure_X=np.asarray(fail_X, dtype=float).reshape(-1, len(self.space.names)),
            failure_labels=fail_labels,
            failure_ids=fail_ids,
            incomplete_ids=incomplete,
        )

    # -- 统计 ------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        """台账看板用的汇总（模块 6 会消费）。"""
        by_round: Dict[int, int] = {}
        by_anomaly = {a.value: 0 for a in Anomaly}
        by_failure = {f.value: 0 for f in Failure}
        for r in self.records:
            by_round[r.round_id] = by_round.get(r.round_id, 0) + 1
            by_anomaly[r.anomaly.value] += 1
            by_failure[r.failure.value] += 1
        split = self.split()
        out: Dict[str, Any] = {
            "n_records": len(self.records),
            "by_round": dict(sorted(by_round.items())),
            "by_anomaly": by_anomaly,
            "by_failure": by_failure,
            "n_train": split.n_train,
            "spec_hit_rate": (
                float(self.space.spec_hit_rate(split.train_Y)) if split.n_train else 0.0
            ),
            "cost_total_cny": float(
                sum(r.cost_cny for r in self.records if r.cost_cny is not None)
            ),
            "n_batches": len({r.batch_id for r in self.records}),
        }
        return out

    # -- 存取 ------------------------------------------------------------- #
    def to_json(self, path: Any) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": LEDGER_SCHEMA,
            "config_hash": self.space.config_hash(),
            "objective_names": list(self.objective_names),
            "variable_names": list(self.space.names),
            "provenance": self.provenance,
            "records": [r.to_dict() for r in self.records],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p.resolve())

    @classmethod
    def from_json(cls, path: Any, space: CMPRecipeSpace) -> "ExperimentLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema != LEDGER_SCHEMA:
            raise ValueError(f"台账格式不匹配：期望 {LEDGER_SCHEMA}，实际 {schema}")
        if list(payload.get("objective_names", [])) != list(space.objective_names):
            raise ValueError(
                f"目标列不一致：台账 {payload.get('objective_names')} vs "
                f"空间 {list(space.objective_names)}"
            )
        records = [ExperimentRecord.from_dict(d) for d in payload["records"]]
        return cls(space, records, payload.get("provenance"))

if __name__ == "__main__":
    import copy
    import json
    import sys
    from pathlib import Path

    np.random.seed(2026)

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_selfcheck.txt"

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

        from src.config import build_default_config
        from src.experiment import (
            Anomaly,
            ExperimentLedger,
            Failure,
            LEDGER_SCHEMA,
        )

        space = build_default_config()

        # ---------------- 0. 迁移 ---------------- #
        rule("0. 数据迁移：cmp-closed-loop-v1 → cmp-ledger-v2（如实标注派生字段）")
        ledger = ExperimentLedger.from_dataset("data/closed_loop_n34.json", space)
        print(f"  记录数 {len(ledger)}   目标 {ledger.objective_names}")
        print(f"  源格式 {ledger.provenance['source_schema']} → 目标格式 {LEDGER_SCHEMA}")
        print(f"  实测字段：{ledger.provenance['measured_fields']}")
        print(f"  派生字段：{ledger.provenance['derived_fields']}")
        print(f"  缺失字段：{ledger.provenance['missing_fields']}")
        print(f"  原则：{ledger.provenance['note']}")

        # ---------------- 1. 真台账校验 ---------------- #
        rule("1. 真台账校验：缺失元数据应报 warning，而不是 error")
        rep = ledger.validate()
        print(rep.describe())
        print(f"\n  错误码分布：{rep.codes()}")
        print(f"  结论：ok={rep.ok}（缺失协变量/时间戳不阻塞训练，但会降低模型可用维度）")
        assert rep.ok, "迁移后的真台账不应有 error"
        assert rep.codes().get("covariates_missing", 0) == len(ledger)
        assert rep.codes().get("timestamp_missing", 0) == len(ledger)
        # 校验器抓到了一个跨模块漏洞：初始 LHS 只过滤硬约束，不过滤隐式规则库
        n_reject = rep.codes().get("recipe_implicit_reject", 0)
        if n_reject:
            bad = [i.examples[0] for i in rep.issues if i.code == "recipe_implicit_reject"]
            print(f"\n  ⚠ 校验器发现：{n_reject} 条记录命中规则库 REJECT 级规则"
                  f"（{bad}）")
            print("    根因：模块 1 的 sample_lhs_feasible 只保证**硬约束**可行，"
                  "不过滤隐式规则库，")
            print("    所以首批 LHS 里可能出现规则库判定为『会沉降』的配方。"
                  "这类数据被真实测出来了，")
            print("    因此只记 warning、保留在训练集；但**下一批采样前应补上规则过滤**。")

        # ---------------- 2. 三分（真实数据） ---------------- #
        rule("2. 三分：真实数据没有异常标签，应完整进入训练集")
        split = ledger.split()
        print(split.describe())
        payload = json.loads(Path("data/closed_loop_n34.json").read_text(encoding="utf-8"))
        X_src = np.asarray(payload["recipes"], dtype=float)
        Y_src = np.asarray(payload["responses"], dtype=float)
        print(f"\n  源数据 X {X_src.shape} / Y {Y_src.shape}")
        print(f"  三分后 X {split.train_X.shape} / Y {split.train_Y.shape}")
        assert split.train_X.shape == (34, space.dim)
        assert split.train_Y.shape == (34, 3)
        assert np.allclose(split.train_X, X_src), "迁移丢失或错位了配方数据"
        assert np.allclose(split.train_Y, Y_src), "迁移丢失或错位了响应数据"
        print("  ✓ 三分结果与源文件逐元素一致（迁移没有丢数据、没有错位）")

        # ---------------- 3. 故障注入 ---------------- #
        rule("3. 故障注入：构造脏台账来验证校验器（合成数据，仅用于测试）")
        dirty = copy.deepcopy(ledger)
        rng = np.random.default_rng(7)
        # 3.1 补齐一批"看起来正常"的协变量（模拟后续有元数据的新数据）。
        #     环境类协变量按**批次**赋值（同一批共用一套环境），
        #     slurry_age_hours 按**记录**赋值（同批次先后抛光的等待时间不同）。
        batch_ids = sorted({r.batch_id for r in dirty.records})
        batch_env: Dict[str, Dict[str, float]] = {}
        for bi, bid in enumerate(batch_ids):
            batch_env[bid] = {
                "pad_use_count": float((bi * 37) % 380),
                "ambient_temp_c": float(21.5 + rng.normal(0.0, 0.4)),
                "ambient_humidity_pct": float(45.0 + rng.normal(0.0, 2.0)),
                "tool_id": float(1 + (bi % 3)),
            }
        for i, rec in enumerate(dirty.records):
            rec.covariates = dict(batch_env[rec.batch_id])
            rec.covariates["slurry_age_hours"] = float(rng.uniform(0.0, 20.0))
            rec.timestamp = f"2026-11-{(i // 12) + 1:02d}T{(i % 12) + 8:02d}:00:00+08:00"
        dirty.provenance["note"] = "协变量与时间戳为合成值，仅供校验器测试"

        # 3.2 逐类注入缺陷
        dirty.records[7].anomaly = Anomaly.MEASUREMENT          # 仪器故障 → 剔除
        dirty.records[12].anomaly = Anomaly.RESPONSE            # 配方导致 → 保留标记
        dirty.records[19].failure = Failure.SCRATCH             # 划伤 → 失效集
        dirty.records[23].failure = Failure.MIX_FAIL            # 配不出 → 无性能值
        dirty.records[23].y = {n: None for n in space.objective_names}
        dirty.records[5].covariates["ambient_temp_c"] = 41.0     # 超出典型范围
        dirty.records[5].covariates["pad_use_count"] = -5.0      # 负的使用片数
        dirty.records[9].covariates["ambient_humidity_pct"] = 130.0  # 超出物理范围
        dirty.records[9].covariates["tool_id"] = 7.0             # 非法类别
        dirty.records[15].y["Ra"] = 280.0                        # 单位错误（nm 当 Å，10×）
        dirty.records[27].cost_cny = 900.0                       # 成本口径不一致
        dirty.records[30].timestamp = "not-a-timestamp"          # 时间戳无法解析

        print("  注入清单：")
        print("    #07 measurement（仪器故障）      #12 response（配方反常，保留）")
        print("    #19 failure=scratch              #23 failure=mix_fail 且清空性能值")
        print("    #05 ambient_temp=41℃、pad=-5     #09 湿度=130%、tool_id=7")
        print("    #15 Ra=280 Å（10× 单位混用）     #27 成本=900 元（口径不一致）")
        print("    #30 时间戳不可解析")

        dirty_rep = dirty.validate()
        print()
        print(dirty_rep.describe())
        expected = {
            "covariate_negative": 1,
            "covariate_out_of_physical": 1,
            "covariate_bad_category": 1,
            # 4 次：-5 片数、41℃、130% 湿度、tool_id=7 都超出各自典型范围。
            # 前三个里除片数外，其余三个也各自命中一条 error——
            # "一个缺陷同时报 error 与 warning"是有意的：error 说它不合法，
            # warning 说模型会外推。
            "covariate_out_of_typical": 4,
            "y_unit_suspect": 1,
            "cost_inconsistent": 1,
            "timestamp_unparsable": 1,
            "batch_covariate_inconsistent": 2,  # 批次内环境被改动（这正是 R9 的用途）
        }
        got = dirty_rep.codes()
        print(f"\n  预期命中的错误码：{expected}")
        print(f"  实际命中：{ {k: got.get(k, 0) for k in expected} }")
        for code, cnt in expected.items():
            assert got.get(code, 0) == cnt, f"{code} 命中 {got.get(code, 0)} 次，预期 {cnt}"
        assert not dirty_rep.ok, "注入了 error 级缺陷却判定为 ok"
        print("  ✓ 8 类缺陷全部被精确命中，且台账被判为不可直接训练")

        # ---------------- 4. 三分（脏台账） ---------------- #
        rule("4. 三分：measurement 剔除 / response 保留标记 / failure 单独建模")
        dirty_split = dirty.split()
        print(dirty_split.describe())
        print(f"\n  训练集编号：{dirty_split.train_ids[:6]}…（共 {len(dirty_split.train_ids)}）")
        assert "E007" in dirty_split.excluded_measurement_ids, "measurement 未被剔除"
        assert "E012" in dirty_split.train_ids, "response 异常被误删（必须保留）"
        assert "E012" in dirty_split.response_flagged_ids, "response 异常未被标记"
        assert "E019" not in dirty_split.train_ids, "failure 混进了主目标训练集"
        assert "E023" not in dirty_split.train_ids, "无性能值的失败样本混进了训练"
        assert set(dirty_split.failure_ids) == {"E019", "E023"}
        assert dirty_split.train_X.shape[0] == 34 - 1 - 2
        print(f"  ✓ 剔除集 {dirty_split.excluded_measurement_ids}")
        print(f"  ✓ 标记保留 {dirty_split.response_flagged_ids}（仍在训练集里）")
        print(f"  ✓ 失效集 {dirty_split.failure_ids} 标签 {dirty_split.failure_labels}")
        print(f"  ✓ 训练 {dirty_split.n_train} = 34 − 1(measurement) − 2(failure)")

        # ---------------- 5. 协变量结构化校验 ---------------- #
        rule("5. 批次协变量的结构化校验（覆盖率 / 越界 / 用途）")
        cov_rep = dirty.covariate_report()
        print(cov_rep.describe())
        assert cov_rep.coverage["pad_use_count"] == 1.0
        assert cov_rep.out_of_typical["ambient_temp_c"] >= 1
        print(f"\n  处理方式分布：{cov_rep.handling}")
        print("  → gpr_input 的 4 个维度会进 GPR 输入；tool_id 走随机效应，"
              "不占输入维度。")

        # ---------------- 6. 台账统计 ---------------- #
        rule("6. 台账看板：轮次分布 / 异常分布 / 规格满足率 / 成本合计")
        st = dirty.stats()
        for k, v in st.items():
            print(f"  {k:<18}{v}")
        assert st["n_train"] == dirty_split.n_train
        assert st["by_anomaly"]["measurement"] == 1
        assert st["by_failure"]["scratch"] == 1

        # ---------------- 7. 往返一致性 ---------------- #
        rule("7. 读写往返：JSON 序列化后记录数与三分结果必须完全一致")
        clean_path = dirty.to_json("data/experiment_ledger.json")
        print(f"  写入：{clean_path}")
        reloaded = ExperimentLedger.from_json("data/experiment_ledger.json", space)
        print(f"  回读：{len(reloaded)} 条记录（原 {len(dirty)} 条）")
        assert len(reloaded) == len(dirty)
        assert [r.to_dict() for r in reloaded] == [r.to_dict() for r in dirty], \
            "往返后记录内容发生变化"
        r2 = reloaded.split()
        assert r2.train_X.shape == dirty_split.train_X.shape
        assert np.allclose(r2.train_X, dirty_split.train_X)
        assert r2.failure_ids == dirty_split.failure_ids
        print("  ✓ 记录内容、训练矩阵、失效集完全一致")
        rep2 = reloaded.validate()
        assert rep2.n_errors == dirty_rep.n_errors, "往返后错误数变化（枚举/浮点精度问题）"
        print(f"  ✓ 回读后校验结论一致（{rep2.n_errors} error / {rep2.n_warnings} warning）")

        # ---------------- 8. 门槛：脏数据必须挡住 ---------------- #
        rule("8. 训练前门槛：校验不过就不该把数据送进模块 2")
        print(f"  真台账 ok={rep.ok}  → 可以训练")
        print(f"  脏台账 ok={dirty_rep.ok} → 必须拒绝")
        assert rep.ok and not dirty_rep.ok
        print("  → 契约建议：模块 2/3 的入口先调 validate().ok，"
              "或在 split() 之前只接受 !ok 时人工确认过的数据。")

        # ---------------- 9. 汇总 ---------------- #
        rule("9. 汇总")
        print(f"  迁移：34 条 → {LEDGER_SCHEMA}（派生 {len(ledger.provenance['derived_fields'])}"
              f" 个字段，缺失 {len(ledger.provenance['missing_fields'])} 类元数据）")
        print(f"  真台账：{rep.n_errors} error / {rep.n_warnings} warning（缺失元数据）")
        print(f"  脏台账：{dirty_rep.n_errors} error / {dirty_rep.n_warnings} warning"
              f"（8 类缺陷全部命中）")
        print(f"  三分：训练 {dirty_split.n_train} / 剔除 "
              f"{len(dirty_split.excluded_measurement_ids)} / 标记 "
              f"{len(dirty_split.response_flagged_ids)} / 失效 "
              f"{len(dirty_split.failure_ids)} / 不完整 "
              f"{len(dirty_split.incomplete_ids)}")
        print(f"  留档：{out_path.resolve()}")
        print(f"  台账：{clean_path}")
        print("\n全部自检通过（10 条断言：迁移零丢失、真台账无 error、8 类缺陷精确命中、"
              "三分互斥且完备、协变量覆盖率、往返一致、门槛生效）。")

    finally:
        sys.stdout = real_stdout
        log_file.close()
