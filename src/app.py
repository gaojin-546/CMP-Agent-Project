"""CMP 抛光液配方开发 Agent —— 工程师用 Web 界面（Streamlit）。

设计目标（对应本次交付要求）：

1. **多层级物理数据隔离**：每个"基底 + 化学体系"组合拥有独立的目录与数据文件

       data/projects/{substrate_type}/{abrasive}_{oxidizer}_{complexing}/history.csv

   训练、推荐、追加数据**只读当前选中路径**，跨体系不共享任何一行数据。
2. **新建体系时的安全降级**：自动匹配 ``config.FAMILY_PRESETS``；
   匹配不上（全新化学体系）时弹出模态警告，要求用户显式确认"盲跑"。
3. **主工作区**：拖拽上传（CSV/Excel）→ 表头与逐行校验 → 追加；
   一键训练（模块 2，健康报告高亮 + fallback 显眼提示）；
   推荐（模块 3 + 4）→ 配方 / 预测 / 成本 → 一键导出 Excel。

启动：

    streamlit run src/app.py
"""

from __future__ import annotations

import io
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CHEMISTRY_CATALOG,
    CMPRecipeSpace,
    FAMILY_PRESETS,
    RAG_CONFIG,
    build_config,
    build_default_config,
)
from src.experiment import (
    Anomaly,
    ExperimentLedger,
    ExperimentRecord,
    Failure,
)
from src.rag_bridge import SERVICE_HINT, RagServiceError, client as rag
from src.surrogate_model import CMPGPRSurrogate, STATUS_FALLBACK

# --------------------------------------------------------------------------- #
# 0. 常量与路径约定
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "data" / "projects"
HISTORY_NAME = "history.csv"
META_NAME = "meta.json"

#: 界面里的显示名 -> 目录里使用的短代号。
#: 词表来自 config.CHEMISTRY_CATALOG（唯一真源），这里只做展示层包装：
#: 显示成 "氧化铈 Ceria (CeO2)"，用户手输的任意新物质由 slugify_token 规范化。
def _label_map(role: str) -> Dict[str, str]:
    return {
        f"{desc} ({token})": token
        for token, desc in CHEMISTRY_CATALOG.get(role, ())
    }


def _options(role: str) -> List[str]:
    return [label for label in _label_map(role)]


def _options_for(mapping: Dict[str, str]) -> List[str]:
    return list(mapping)


#: 添加剂下拉里的"不添加"哨兵项（比复选框更直观：默认就是不加）
ADD_NONE_LABEL = "无（不添加）"


ABRASIVE_TOKENS = _label_map("abrasive")
OXIDIZER_TOKENS = _label_map("oxidizer")
COMPLEXING_TOKENS = _label_map("complexing")
SURFACTANT_TOKENS = _label_map("surfactant")
INHIBITOR_TOKENS = _label_map("inhibitor")
PH_ADJUSTER_TOKENS = _label_map("ph_adjuster")
SUBSTRATE_TOKENS = _label_map("substrate")

#: 基底：预设用代号显示（Cu / W / Oxide…），同样允许手输新材料
SUBSTRATE_OPTIONS = [t for t, _ in CHEMISTRY_CATALOG["substrate"]]

#: (基底, 磨料代号, 氧化剂代号, 络合剂代号) -> FAMILY_PRESETS 的键
#: 匹配不上就是"全新化学体系"，必须走盲跑确认。
#: 注意：只匹配这 4 个"核心化学"；表面活性剂/缓蚀剂/pH 调节剂是族内添加剂，
#: 不改变机理先验，因此不影响预设匹配（但会写进 meta 与 history）。
FAMILY_MATCH: Dict[Tuple[str, str, str, str], str] = {
    ("Cu", "SiO2", "H2O2", "Glycine"): "cu_peroxide_glycine",
    ("Cu", "SiO2", "H2O2", "CitricAcid"): "cu_alkaline_complex",
    ("Oxide", "CeO2", "H2O2", "None"): "oxide_ceria_anionic",
}

#: 体系级元数据列：随每一行写进 history.csv，供台账溯源（不进 GPR 连续训练）
SYSTEM_META_COLUMNS = [
    "substrate_type",
    "surfactant_type",
    "inhibitor_type",
    "ph_adjuster_type",
    "design_intent",
]

META_COLUMNS = [
    "experiment_id", "round_id", "batch_id", "timestamp", "operator",
    "anomaly", "failure", "notes",
] + SYSTEM_META_COLUMNS


#: Windows 保留设备名（不能作为文件名/目录名，建目录会直接失败）
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def slugify_token(text: str, maxlen: int = 40) -> str:
    """把任意用户输入规范成"路径安全"的目录片段。

    规则（针对 Windows + 我们的目录约定）：

    * 空白、连字符、全角破折号 → 下划线（``Ammonium Persulfate`` → ``Ammonium_Persulfate``）
    * 去掉 Windows 非法字符 ``< > : " / \\ | ? *`` 与控制字符
    * 连续下划线折叠成一个，首尾的下划线与点去掉（Windows 不允许目录名以点结尾）
    * 限制长度，避免路径过长
    * 命中 Windows 保留设备名（CON/NUL/COM1…）时加前导下划线
    * 空输入 → ``NA``

    中文会被保留：NTFS 与 Python 都支持，且比强行转拼音更可读。
    """
    s = unicodedata.normalize("NFKC", str(text or "")).strip()
    s = re.sub(r"[\s\-–—_]+", "_", s)
    s = re.sub(r'[<>:"/\\|?*()（）\[\]{}]', "", s)
    s = re.sub(r"[\x00-\x1f]", "", s)
    s = re.sub(r"_+", "_", s).strip("_.")      # 去掉首尾的 _ 和 .
    s = s[:maxlen].strip("_.")
    if not s:
        return "NA"
    if s.upper() in _WIN_RESERVED:
        s = f"_{s}"
    return s


def to_token(value: str, mapping: Dict[str, str]) -> str:
    """把下拉框的返回值（可能是预设显示名、也可能用户手输）转成目录代号。

    匹配顺序：
    1. 预设显示名完全一致（``胶体二氧化硅 (SiO2)`` → ``SiO2``）；
    2. 与已知代号**大小写不敏感**匹配（``sio2`` / ``SiO2`` → ``SiO2``，
       这样用户手敲小写代码不会被当成"全新体系"而误触发盲跑）；
    3. 其余一律当作自定义化学物质，按 :func:`slugify_token` 规范化。
    """
    v = str(value or "").strip()
    if v in mapping:
        return mapping[v]
    low = v.lower()
    for label, token in mapping.items():
        if low == token.lower() or low == label.lower():
            return token
    return slugify_token(v)


ADDITIVE_ROLES = ("surfactant", "inhibitor", "ph_adjuster")


def family_dir_name(
    abrasive_token: str,
    oxidizer_token: str,
    complexing_token: str,
    surfactant_token: str = "",
    inhibitor_token: str = "",
    ph_adjuster_token: str = "",
) -> str:
    """组合出目录名：``核心三件套`` + 可选 ``__添加剂三件套``。

    为什么添加剂放在 ``__`` 后缀里、而不是直接拼进核心名：

    * **向后兼容**：已有的 ``Cu/SiO2_H2O2_Glycine`` 不能因为这次升级而"改名丢失"；
      没填添加剂时不产生后缀，路径与旧版完全一致。
    * **隔离性**：一旦指定了添加剂（哪怕只是把络合剂换成某种新物质配 PEG），
      就是另一套化学体系，应当各自有目录，避免数据串味。
    * ``__`` 作为分隔符，便于把核心名与添加剂名拆开解析（单个 ``_`` 在代号里
      本来就会出现，比如 ``BTA_OH``）。
    """
    core = "_".join(
        slugify_token(t) for t in (abrasive_token, oxidizer_token, complexing_token)
    )
    add = [slugify_token(t) if str(t or "").strip() else "" for t in
           (surfactant_token, inhibitor_token, ph_adjuster_token)]
    if not any(add):
        return core
    return f"{core}__" + "_".join(t or "None" for t in add)


def split_family_dir(family_dir: str) -> Tuple[str, str]:
    """把目录名拆成 ``(核心三件套, 添加剂后缀)``。"""
    core, _, add = family_dir.partition("__")
    return core, add


@dataclass(frozen=True)
class ProjectRef:
    """一个"基底 + 化学体系"项目的路径引用。"""

    substrate: str
    family_dir: str          # SiO2_H2O2_Glycine 或 SiO2_H2O2_Glycine__PEG_BTA_KOH
    root: Path

    @property
    def label(self) -> str:
        return f"{self.substrate} / {self.family_dir}"

    @property
    def history_path(self) -> Path:
        return self.root / HISTORY_NAME

    @property
    def meta_path(self) -> Path:
        return self.root / META_NAME

    @property
    def rel_label(self) -> str:
        return f"data/projects/{self.substrate}/{self.family_dir}"

    # -- 元数据 ----------------------------------------------------------- #
    def read_meta(self) -> Dict[str, Any]:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def write_meta(self, meta: Dict[str, Any]) -> None:
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @property
    def design_intent(self) -> str:
        """体系的研发目标（选填，留档用）。"""
        return str(self.read_meta().get("design_intent") or "")

    @property
    def additives(self) -> Dict[str, str]:
        """``{角色: 代号}``，缺省为空串（兼容旧项目的 meta）。"""
        add = self.read_meta().get("additives") or {}
        return {role: str(add.get(role, "") or "") for role in ADDITIVE_ROLES}


def _ensure_under_projects(path: Path) -> Path:
    """路径必须落在 data/projects 之下（防止下拉框被篡改造成越权读写）。"""
    resolved = path.resolve()
    base = PROJECTS_ROOT.resolve()
    if base not in resolved.parents and resolved != base:
        raise ValueError(f"拒绝访问工作区外的路径：{resolved}")
    return resolved


# --------------------------------------------------------------------------- #
# 1. 项目发现 / 创建
# --------------------------------------------------------------------------- #
def scan_projects() -> List[ProjectRef]:
    """扫描 ``data/projects/{substrate}/{family_dir}/`` 两层结构。"""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    out: List[ProjectRef] = []
    for sub in sorted(p for p in PROJECTS_ROOT.iterdir() if p.is_dir()):
        for fam in sorted(p for p in sub.iterdir() if p.is_dir()):
            out.append(ProjectRef(sub.name, fam.name, fam))
    return out


def create_project(
    substrate: str,
    abrasive_token: str,
    oxidizer_token: str,
    complexing_token: str,
    *,
    surfactant_token: str = "",
    inhibitor_token: str = "",
    ph_adjuster_token: str = "",
    design_intent: str = "",
) -> ProjectRef:
    """创建多级目录 + 空的 history.csv + meta.json（幂等）。

    传入的可以是原始用户输入（如 ``Ammonium Persulfate``），这里统一
    经 :func:`slugify_token` 规范化；对已规范化的输入是幂等的。

    表面活性剂/缓蚀剂/pH 调节剂是**族内添加剂**：它们改变目录名（保证隔离）、
    写进 ``meta.json`` 的 ``additives``，并在每行 ``history.csv`` 里留档，
    但**不参与 GPR 的连续数值训练**（它们的浓度已经是连续变量
    ``surfactant_wt_pct`` / ``inhibitor_wt_pct``）。
    """
    substrate = slugify_token(substrate)
    abrasive_token = slugify_token(abrasive_token)
    oxidizer_token = slugify_token(oxidizer_token)
    complexing_token = slugify_token(complexing_token)
    surfactant_token = slugify_token(surfactant_token) if str(surfactant_token or "").strip() else ""
    inhibitor_token = slugify_token(inhibitor_token) if str(inhibitor_token or "").strip() else ""
    ph_adjuster_token = slugify_token(ph_adjuster_token) if str(ph_adjuster_token or "").strip() else ""
    family_dir = family_dir_name(
        abrasive_token, oxidizer_token, complexing_token,
        surfactant_token, inhibitor_token, ph_adjuster_token,
    )
    root = _ensure_under_projects(PROJECTS_ROOT / substrate / family_dir)
    root.mkdir(parents=True, exist_ok=True)

    ref = ProjectRef(substrate, family_dir, root)
    space = build_space_for(substrate, abrasive_token, oxidizer_token, complexing_token)
    if not ref.history_path.exists() or ref.history_path.stat().st_size == 0:
        pd.DataFrame(columns=history_columns(space)).to_csv(
            ref.history_path, index=False, encoding="utf-8-sig"
        )
    if not ref.meta_path.exists():
        preset = FAMILY_MATCH.get(
            (substrate, abrasive_token, oxidizer_token, complexing_token)
        )
        ref.write_meta({
            "substrate": substrate,
            "abrasive": abrasive_token,
            "oxidizer": oxidizer_token,
            "complexing": complexing_token,
            "additives": {
                "surfactant": surfactant_token,
                "inhibitor": inhibitor_token,
                "ph_adjuster": ph_adjuster_token,
            },
            "design_intent": str(design_intent or "").strip(),
            "family_preset": preset,
            "blind_run": preset is None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
    return ref


def family_of(ref: ProjectRef) -> Optional[str]:
    """从 meta 或目录名反推预设键（目录名是兜底，meta 优先）。"""
    meta = ref.read_meta()
    if "family_preset" in meta:
        return meta["family_preset"]
    core, _ = split_family_dir(ref.family_dir)
    parts = core.split("_")
    if len(parts) >= 3:
        key = (ref.substrate, parts[0], parts[1], "_".join(parts[2:]))
        return FAMILY_MATCH.get(key)
    return None


def build_space_for(
    substrate: str, abrasive_token: str, oxidizer_token: str, complexing_token: str
) -> CMPRecipeSpace:
    """按"基底 + 化学体系"构造参数空间。

    有预设 → 用预设（继承 Preston 先验、pH 稳定窗口等机理知识）；
    无预设 → 退回默认空间，并标记为盲跑（界面上会强制警告）。
    """
    preset = FAMILY_MATCH.get((substrate, abrasive_token, oxidizer_token, complexing_token))
    if preset and preset in FAMILY_PRESETS:
        return build_config(preset)
    return build_default_config()


def space_for(ref: ProjectRef) -> Tuple[CMPRecipeSpace, bool]:
    """返回 ``(参数空间, 是否盲跑)``。"""
    meta = ref.read_meta()
    preset = meta.get("family_preset", family_of(ref))
    if preset and preset in FAMILY_PRESETS:
        return build_config(preset), False
    return build_default_config(), True


def history_columns(space: CMPRecipeSpace) -> List[str]:
    return (
        list(space.names)
        + list(space.objective_names)
        + [c.name for c in space.covariates]
        + META_COLUMNS
    )


# --------------------------------------------------------------------------- #
# 2. 数据读写与校验
# --------------------------------------------------------------------------- #
def load_history(ref: ProjectRef, space: CMPRecipeSpace) -> pd.DataFrame:
    """只读当前项目路径下的 history.csv（不存在则返回空表）。"""
    path = _ensure_under_projects(ref.history_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=history_columns(space))
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def save_history(ref: ProjectRef, df: pd.DataFrame) -> None:
    path = _ensure_under_projects(ref.history_path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ledger_from_df(df: pd.DataFrame, space: CMPRecipeSpace) -> ExperimentLedger:
    """把 history.csv 转成模块 5 的台账，复用它的三分与校验。"""
    records: List[ExperimentRecord] = []
    for i, row in df.iterrows():
        y = {}
        for name in space.objective_names:
            v = row.get(name, np.nan)
            y[name] = None if pd.isna(v) else float(v)
        cov = {}
        for c in space.covariates:
            v = row.get(c.name, np.nan)
            if not pd.isna(v):
                cov[c.name] = float(v)
        anomaly = str(row.get("anomaly", "none") or "none").strip().lower()
        failure = str(row.get("failure", "none") or "none").strip().lower()
        records.append(
            ExperimentRecord(
                experiment_id=str(row.get("experiment_id") or f"R{i:03d}"),
                round_id=int(row.get("round_id") or 0),
                batch_id=str(row.get("batch_id") or "B00"),
                config_hash=str(row.get("config_hash") or ""),
                recipe={n: float(row[n]) for n in space.names if not pd.isna(row.get(n, np.nan))},
                y=y,
                covariates=cov,
                cost_cny=None if pd.isna(row.get("cost_cny", np.nan)) else float(row["cost_cny"]),
                anomaly=Anomaly(anomaly if anomaly in {"none", "measurement", "response"} else "none"),
                failure=Failure(failure if failure in {f.value for f in Failure} else "none"),
                operator=str(row.get("operator") or ""),
                timestamp=str(row.get("timestamp") or ""),
                notes=str(row.get("notes") or ""),
            )
        )
    return ExperimentLedger(space, records)


@dataclass
class UploadCheck:
    """一次上传的表头 + 逐行校验结果。"""

    ok: bool
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    missing_columns: List[str]
    unknown_columns: List[str]
    messages: List[str]


def validate_upload(
    raw: pd.DataFrame, space: CMPRecipeSpace
) -> UploadCheck:
    """表头校验 + 逐行物理校验（越界 / 硬约束 / 缺值）。

    规则有意"宁严勿松"：一行数据进了历史库就会参与训练，脏数据比没数据更糟。
    """
    messages: List[str] = []
    required = list(space.names) + list(space.objective_names)
    missing = [c for c in required if c not in raw.columns]
    known = set(history_columns(space))
    unknown = [c for c in raw.columns if c not in known]
    if missing:
        messages.append(f"缺少必需列：{missing}")
        return UploadCheck(False, pd.DataFrame(columns=raw.columns), raw.copy(),
                           missing, unknown, messages)
    if unknown:
        messages.append(f"忽略未知列：{unknown}")

    keep = [c for c in history_columns(space) if c in raw.columns]
    df = raw[keep].copy()

    good_rows, bad_rows = [], []
    for idx, row in df.iterrows():
        reasons: List[str] = []
        # 1) 必需列缺值
        for n in required:
            if pd.isna(row[n]):
                reasons.append(f"{n} 缺失")
        # 2) 设计变量越界
        if not any(pd.isna(row[n]) for n in space.names):
            x = np.array([[float(row[n]) for n in space.names]])
            for v in space.variables:
                val = float(row[v.name])
                if not (v.low <= val <= v.high):
                    reasons.append(f"{v.name}={val:.4g} 超出 [{v.low:g}, {v.high:g}]")
            # 3) 硬约束
            if not reasons:
                for con in space.hard_constraints:
                    g = float(np.asarray(con.bind(space.names)(x)).reshape(-1)[0])
                    if g > 1e-6:
                        reasons.append(f"违反 {con.name}（{con.expression}）")
        if reasons:
            bad_rows.append({**row.to_dict(), "_row": int(idx) + 2, "_reasons": "; ".join(reasons)})
        else:
            good_rows.append(row.to_dict())

    accepted = pd.DataFrame(good_rows, columns=df.columns)
    rejected = pd.DataFrame(bad_rows)
    ok = len(accepted) > 0
    if len(rejected):
        messages.append(f"{len(rejected)} 行被拒绝（见下表）")
    if not ok:
        messages.append("没有任何可用行")
    return UploadCheck(ok, accepted, rejected, missing, unknown, messages)


def _system_meta_values(ref: ProjectRef) -> Dict[str, str]:
    """体系级元数据（每行都写一遍，便于 history.csv 单文件溯源）。"""
    meta = ref.read_meta()
    add = meta.get("additives") or {}
    return {
        "substrate_type": ref.substrate,
        "surfactant_type": str(add.get("surfactant", "") or ""),
        "inhibitor_type": str(add.get("inhibitor", "") or ""),
        "ph_adjuster_type": str(add.get("ph_adjuster", "") or ""),
        "design_intent": str(meta.get("design_intent", "") or ""),
    }


def append_rows(ref: ProjectRef, space: CMPRecipeSpace,
                new_rows: pd.DataFrame) -> Tuple[int, int]:
    """把新数据追加进当前项目的 history.csv，返回 ``(新增, 跳过重复)``。

    追加时会补齐体系级元数据列（基底/添加剂/设计意图），这样即使把
    history.csv 单独发给同事，也能看出"这批数据是哪个体系、当初想解决什么问题"。
    """
    current = load_history(ref, space)
    incoming = new_rows.copy()
    for col, val in _system_meta_values(ref).items():
        if col not in incoming.columns:
            incoming[col] = val
        else:
            incoming[col] = incoming[col].where(
                incoming[col].notna() & (incoming[col].astype(str) != ""), val
            )
    incoming = incoming.reindex(columns=list(current.columns) + [
        c for c in incoming.columns if c not in current.columns
    ])
    combined = pd.concat([current, incoming], ignore_index=True)
    key_cols = list(space.names) + list(space.objective_names)
    before = len(combined)
    combined = combined.drop_duplicates(subset=key_cols, keep="first")
    skipped = before - len(combined)
    # 补齐实验编号
    if "experiment_id" in combined.columns:
        for i in range(len(combined)):
            if pd.isna(combined.at[i, "experiment_id"]) or not str(
                combined.at[i, "experiment_id"]
            ).strip():
                combined.at[i, "experiment_id"] = f"E{i:03d}"
    save_history(ref, combined)
    return len(combined) - len(current), skipped


# --------------------------------------------------------------------------- #
# 2b. 数据纠错：删除 / 清空 / 批量打异常标签
# --------------------------------------------------------------------------- #
def _current_anomaly(df: pd.DataFrame) -> pd.Series:
    return df.get("anomaly", pd.Series(["none"] * len(df), index=df.index)).fillna("none")


def delete_rows(df: pd.DataFrame, positions: Sequence[int]) -> pd.DataFrame:
    """删除指定行（按 DataFrame 位置索引），返回新表。

    删除是"硬纠错"：用于录错、重复登记这类**数据本身不该存在**的情况。
    如果只是某次测量可疑但实验确实做过，应该用 :func:`tag_rows` 打
    ``measurement`` 标签——训练时同样会被剔除，但保留追溯。
    """
    if not len(positions):
        return df.copy()
    idx = [int(p) for p in positions if 0 <= int(p) < len(df)]
    return df.drop(index=df.index[idx]).reset_index(drop=True)


def clear_history(df: pd.DataFrame) -> pd.DataFrame:
    """清空所有行，只保留表头（列结构不变）。"""
    return df.iloc[0:0].copy()


def tag_rows(
    df: pd.DataFrame, positions: Sequence[int], value: str = "measurement"
) -> pd.DataFrame:
    """把指定行的 ``anomaly`` 标签改成 ``value``。

    ``measurement`` = 仪器/操作问题：模块 5 的三分会把它**剔除出训练集**，
    但台账保留该行，便于追溯。``none`` = 恢复正常。
    """
    out = df.copy()
    if "anomaly" not in out.columns:
        out["anomaly"] = "none"
    ok = [int(p) for p in positions if 0 <= int(p) < len(out)]
    if ok:
        out.iloc[ok, out.columns.get_loc("anomaly")] = value
    return out


def anomaly_mask(df: pd.DataFrame) -> pd.Series:
    """哪些行被打上了 measurement 标签（界面上给个计数）。"""
    return _current_anomaly(df).astype(str).str.lower().eq("measurement")


# --------------------------------------------------------------------------- #
# 3. 模型 / 推荐
# --------------------------------------------------------------------------- #
def train(space: CMPRecipeSpace, split) -> CMPGPRSurrogate:
    """拟合三个目标的 GPR（模块 2）。

    协变量处理：``history.csv`` 里的批次协变量会被模块 5 完整校验，
    但**当前不送入 GPR**——因为推理时还没做实验，未来批次的
    ``pad_use_count`` / 环境温湿度未知，需要先约定"按什么工况预测"
    （标称值 / 最近一次 / 分布积分），这是建模决策而不是代码问题。
    模块 2 本身支持把协变量当额外输入维度（见 ``X_cov`` 参数）。
    """
    return CMPGPRSurrogate(space, loocv_mode="efficient").fit(
        split.train_X, split.train_Y
    )


def health_table(surrogate: CMPGPRSurrogate) -> pd.DataFrame:
    rows = []
    for name in surrogate.objective_names:
        d = surrogate.diagnostics[name]
        flag = {"ok": "🟢 ok", "degraded": "🟡 degraded", "fallback": "🔴 fallback"}[d.status]
        rows.append({
            "目标": name,
            "状态": flag,
            "LOOCV RMSE": None if not np.isfinite(d.loocv_rmse) else round(d.loocv_rmse, 3),
            "R²": None if not np.isfinite(d.loocv_r2) else round(d.loocv_r2, 3),
            "相对误差": None if not np.isfinite(d.loocv_rel_rmse) else round(d.loocv_rel_rmse, 3),
            "噪声占比": None if not np.isfinite(d.noise_sd_ratio) else round(d.noise_sd_ratio, 3),
            "诊断": "; ".join(d.reasons) if d.reasons else "healthy",
        })
    return pd.DataFrame(rows)


def _style_health(df: pd.DataFrame):
    def color(val: str) -> str:
        if "fallback" in str(val):
            return "background-color:#ffd6d6;color:#8b0000;font-weight:600"
        if "degraded" in str(val):
            return "background-color:#fff2cc;color:#7f6000;font-weight:600"
        if "ok" in str(val):
            return "background-color:#d9ead3;color:#274e13"
        return ""

    return df.style.map(color, subset=["状态"])


def recommend(
    space: CMPRecipeSpace, split, q: int, fast: bool
) -> Tuple[Any, Any]:
    """调用模块 3 推荐 + 模块 4 前沿筛选。"""
    from dataclasses import replace

    from src.acquisition import CMPBatchPlanner
    from src.pareto import ParetoSelector

    cfg = space.acquisition
    if fast:
        # 实测：34 条数据下 (2 重启, 30 迭代) 会让 SLSQP 全部重启不可行、
        # 触发降级路径；(3, 40, 32) 能在 ~11 s 内稳定收敛到可行解。
        cfg = replace(cfg, num_restarts=3, maxiter=40, num_samples=32)
    planner = CMPBatchPlanner(replace(space, acquisition=cfg), verbose=False, seed=7)
    rec = planner.recommend_next_batch(split.train_X, split.train_Y, q=q, round_index=0)
    front = ParetoSelector(space).select(split.train_X, split.train_Y, top_k=5)
    return rec, front


def recommendation_table(rec, space: CMPRecipeSpace) -> pd.DataFrame:
    rows = []
    for i in range(rec.recipes.shape[0]):
        row: Dict[str, Any] = {"序号": i + 1}
        for j, n in enumerate(space.names):
            row[n] = round(float(rec.recipes[i, j]), 4)
        for j, n in enumerate(rec.active_objectives):
            row[f"预测 {n}"] = round(float(rec.predicted_mean[i, j]), 2)
            row[f"±sd {n}"] = round(float(rec.predicted_sd[i, j]), 2)
        row["预测成本(元)"] = round(float(rec.cost_cny[i]), 0)
        rows.append(row)
    return pd.DataFrame(rows)


def front_table(front, space: CMPRecipeSpace) -> pd.DataFrame:
    rows = []
    for c in front.ranked:
        row: Dict[str, Any] = {"名次": c.rank, "数据行": c.data_index}
        for n in space.objective_names:
            row[n] = round(c.objectives[n], 3)
        row["成本(元)"] = round(c.cost_cny, 0)
        row["TOPSIS"] = round(c.topsis_score, 4)
        row["满足规格"] = "✓" if c.on_spec else "✗ " + ",".join(c.spec_violations)
        rows.append(row)
    return pd.DataFrame(rows)


def _round_numeric(df: pd.DataFrame, nd: int = 3) -> pd.DataFrame:
    """把所有浮点列统一保留 ``nd`` 位小数（导出给实验员看，不要 8.914e-16）。"""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(nd)
    return out


def to_excel(rec, front, space: CMPRecipeSpace, ref: ProjectRef) -> bytes:
    """把推荐配方 + 前沿榜单 + 说明写成多 sheet 的 xlsx（直接给实验室）。

    浮点统一保留 3 位小数（``float_format="%.3f"``），避免出现
    ``8.914e-16`` 这种在配液台上毫无意义的科学计数法。
    """
    buf = io.BytesIO()
    add = ref.additives
    meta = pd.DataFrame({
        "项目": [ref.rel_label],
        "基底": [ref.substrate],
        "表面活性剂": [add.get("surfactant") or "—"],
        "缓蚀剂": [add.get("inhibitor") or "—"],
        "pH 调节剂": [add.get("ph_adjuster") or "—"],
        "设计意图": [ref.design_intent or "—"],
        "参数空间": [space.name],
        "配置指纹": [space.config_hash()],
        "采集函数": [rec.acq_name],
        "参考点(归一化)": [str([round(v, 4) for v in rec.ref_point])],
        "参考点(原量纲)": [str([round(v, 2) for v in rec.ref_point_physical])],
        "活跃目标": [", ".join(rec.active_objectives)],
        "剔除目标(后备)": [", ".join(rec.dropped_objectives) or "无"],
        "预测 HV": [round(rec.hv_after_pseudo, 4)],
        "生成时间": [datetime.now().isoformat(timespec="seconds")],
    })
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _round_numeric(recommendation_table(rec, space)).to_excel(
            writer, sheet_name="推荐配方", index=False
        )
        _round_numeric(front_table(front, space)).to_excel(
            writer, sheet_name="前沿榜单", index=False
        )
        meta.T.rename(columns={0: "值"}).to_excel(writer, sheet_name="说明")

        # 显示格式统一成 0.000：值本身仍然是数字（可以在 Excel 里继续算），
        # 只是看起来固定 3 位小数。用 float_format="%.3f" 会把数字写成文本，
        # 实验员就没法在表里做加减了——这是个容易被忽略的坑。
        for sheet_name in ("推荐配方", "前沿榜单", "说明"):
            ws = writer.sheets.get(sheet_name)
            if ws is None:
                continue
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                    min_col=1, max_col=ws.max_column):
                for cell in row:
                    if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                        cell.number_format = "0.000"
    return buf.getvalue()


def template_csv(space: CMPRecipeSpace) -> bytes:
    """给工程师的模板：只含表头 + 一行示例。"""
    row = {n: round((v.low + v.high) / 2, 3) for n, v in zip(space.names, space.variables)}
    for o in space.objectives:
        row[o.name] = round((o.prior_range[0] + o.prior_range[1]) / 2, 2)
    for c in space.covariates:
        row[c.name] = round((c.typical_range[0] + c.typical_range[1]) / 2, 2)
    row.update({"experiment_id": "E000", "round_id": 0, "batch_id": "B00",
                "timestamp": "", "operator": "", "anomaly": "none",
                "failure": "none", "notes": ""})
    df = pd.DataFrame([row], columns=history_columns(space))
    return df.to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------- #
# 4. 界面
# --------------------------------------------------------------------------- #
@st.dialog("⚠️ 全新化学体系：缺少机理先验")
def blind_run_dialog(
    substrate: str,
    abr: str,
    ox: str,
    comp: str,
    *,
    family_dir: str = "",
    additives: Optional[Dict[str, str]] = None,
    design_intent: str = "",
) -> None:
    """模态警告：用户必须显式确认才能创建无预设的体系。"""
    fam = family_dir or f"{abr}_{ox}_{comp}"
    st.error(f"`{substrate} / {fam}` 在 `FAMILY_PRESETS` 中没有匹配项。")
    st.caption(f"将创建目录：`data/projects/{substrate}/{fam}/`")
    add = additives or {}
    extra = [f"{k}: {v}" for k, v in add.items() if v]
    if extra:
        st.caption("族内添加剂：" + "；".join(extra))
    if design_intent:
        st.caption(f"设计意图：{design_intent}")
    st.markdown(
        "创建后将以**默认参数盲跑**：\n\n"
        "- 缺少 Preston 系数等机理先验（MRR 均值函数按默认 k_p=220 起步）\n"
        "- pH 稳定窗口、离子强度等硬约束沿用铜 CMP 体系的定义\n"
        "- 健康检查、约束优化、TOPSIS 全部照常工作，但**预测精度的先验支撑更弱**\n\n"
        "建议：先补 20–25 条该体系的标定实验，再据此重估 `k_p` 与窗口。"
    )
    c1, c2 = st.columns(2)
    if c1.button("取消", width='stretch'):
        st.rerun()
    if c2.button("我已知晓，仍要创建", type="primary", width='stretch'):
        create_project(
            substrate, abr, ox, comp,
            surfactant_token=add.get("surfactant", ""),
            inhibitor_token=add.get("inhibitor", ""),
            ph_adjuster_token=add.get("ph_adjuster", ""),
            design_intent=design_intent,
        )
        st.session_state["flash"] = f"已创建盲跑体系：{substrate} / {fam}"
        st.session_state["project_select"] = f"{substrate} / {fam}"
        st.rerun()


def sidebar() -> Optional[ProjectRef]:
    st.sidebar.title("🧪 项目")
    projects = scan_projects()

    st.sidebar.subheader("① 打开已有项目")
    if not projects:
        st.sidebar.info("data/projects 下还没有项目，请先新建。")
        selected: Optional[ProjectRef] = None
    else:
        labels = [p.label for p in projects]
        by_label = {p.label: p for p in projects}
        # 用显式 key 驱动选择：既让「新建体系」能自动切过去，
        # 也让 Streamlit 的测试框架 AppTest 能稳定地切换（无 key 时 index 参数
        # 会在每轮重算并覆盖用户选择，实测导致切不回来）。
        if st.session_state.get("project_select") not in labels:
            st.session_state["project_select"] = labels[0]
        picked = st.sidebar.selectbox(
            "基底 / 化学体系", labels, key="project_select",
            help="路径：data/projects/{基底}/{磨料}_{氧化剂}_{络合剂}/",
        )
        selected = by_label[picked]
        meta = selected.read_meta()
        st.sidebar.caption(f"📁 `{selected.rel_label}`")
        if meta.get("blind_run"):
            st.sidebar.warning("该体系无预设：当前为盲跑模式", icon="⚠️")
        else:
            st.sidebar.caption(f"预设：`{meta.get('family_preset')}`")

    st.sidebar.divider()
    st.sidebar.subheader("② 新建项目 / 体系")

    # 全部 7 个输入都支持"预设下拉 + 直接手输新物质"。
    # accept_new_options=True 时，不在选项里的字符串会原样返回，
    # 再由 to_token() 规范化成目录代号（大小写不敏感的预设代号会被认出来，
    # 例如手敲 "sio2" 仍归到 SiO2，不会误判成全新体系）。
    def _chem_input(label: str, mapping: Dict[str, str], placeholder: str,
                    help_text: str = "", allow_none: bool = False) -> Tuple[str, str]:
        options = ([ADD_NONE_LABEL] if allow_none else []) + _options_for(mapping)
        raw = st.sidebar.selectbox(
            label, options, accept_new_options=True,
            placeholder=placeholder,
            help=help_text or "支持直接输入新的化学物质；空格会自动转成下划线",
        )
        s = str(raw or "")
        if allow_none and s == ADD_NONE_LABEL:
            return "", ""
        return s, to_token(s, mapping)

    substrate = to_token(
        st.sidebar.selectbox(
            "Substrate Type（基底）", SUBSTRATE_OPTIONS, accept_new_options=True,
            placeholder="选择预设基底，或直接输入新的基底材料",
            help="也支持自定义基底（如 GaN、SiC）；同样会做目录名规范化",
        ),
        SUBSTRATE_TOKENS,
    )
    abr_in, abr = _chem_input(
        "Abrasive（磨料）", ABRASIVE_TOKENS, "选择预设，或直接输入新的磨料名称",
        "空格会自动转成下划线（Ammonium Persulfate → Ammonium_Persulfate）",
    )
    ox_in, ox = _chem_input(
        "Oxidizer（氧化剂）", OXIDIZER_TOKENS, "选择预设，或直接输入新的氧化剂名称"
    )
    comp_in, comp = _chem_input(
        "Complexing（络合剂）", COMPLEXING_TOKENS, "选择预设，或直接输入新的络合剂名称"
    )

    st.sidebar.markdown("**族内添加剂（选填）**")
    surf_in, surf = _chem_input(
        "Surfactant（表面活性剂）", SURFACTANT_TOKENS,
        "默认不添加；选择或手输新物质", allow_none=True,
    )
    inh_in, inh = _chem_input(
        "Inhibitor（缓蚀剂）", INHIBITOR_TOKENS,
        "默认不添加；选择或手输新物质", allow_none=True,
    )
    ph_in, ph = _chem_input(
        "pH Adjuster（pH 调节剂）", PH_ADJUSTER_TOKENS,
        "默认不添加；选择或手输新物质", allow_none=True,
    )

    design_intent = st.sidebar.text_area(
        "设计意图 Design Intent（选填）",
        placeholder="例如：把 Cu 的 Dishing 压到 150 Å 以下，同时守住 MRR ≥ 1500 Å/min",
        height=80,
        help="会写入 meta.json 与 history.csv，供后续追溯'这批数据当初想解决什么问题'",
    )

    family_dir = family_dir_name(abr, ox, comp, surf, inh, ph)

    # 把"用户原文 → 规范化代号"的差异显示出来，避免用户以为自己打错了
    renamed = [
        f"{raw} → {tok}"
        for raw, tok in (
            (abr_in, abr), (ox_in, ox), (comp_in, comp),
            (surf_in, surf), (inh_in, inh), (ph_in, ph),
        )
        if str(raw).strip() and str(raw).strip() != tok
    ]
    if renamed:
        st.sidebar.caption("已规范化目录名：" + "；".join(renamed))

    st.sidebar.caption(f"将创建：`data/projects/{substrate}/{family_dir}/`")
    matched = (substrate, abr, ox, comp) in FAMILY_MATCH
    if not matched:
        st.sidebar.caption("ℹ️ 核心化学（基底+磨料+氧化剂+络合剂）无预设："
                           "创建时会弹窗要求确认盲跑")

    if st.sidebar.button("创建新体系", width='stretch'):
        exists = (PROJECTS_ROOT / substrate / family_dir).exists()
        if exists:
            st.sidebar.warning("该体系已存在，无需重复创建")
        elif matched:
            create_project(
                substrate, abr, ox, comp,
                surfactant_token=surf, inhibitor_token=inh,
                ph_adjuster_token=ph, design_intent=design_intent,
            )
            st.session_state["flash"] = f"已创建：{substrate} / {family_dir}"
            st.session_state["project_select"] = f"{substrate} / {family_dir}"
            st.rerun()
        else:
            blind_run_dialog(
                substrate, abr, ox, comp,
                family_dir=family_dir,
                additives={"surfactant": surf, "inhibitor": inh, "ph_adjuster": ph},
                design_intent=design_intent.strip(),
            )

    st.sidebar.divider()
    st.sidebar.caption(
        "数据隔离：训练 / 推荐 / 追加只读当前选中目录下的 `history.csv`，"
        "不同基底与化学体系之间不共享任何一行数据。"
    )
    rag_status_block()
    return selected


@st.dialog("⚠️ 确认清空当前体系的全部数据")
def clear_confirm_dialog(ref: ProjectRef, n: int) -> None:
    st.error(f"即将清空 `{ref.rel_label}` 下的 **{n}** 条记录，只保留表头。")
    st.markdown(
        "该操作**不可撤销**（不会进回收站）。\n\n"
        "如果只是想排除某几行，请改用数据表里的「标记为 measurement」——"
        "训练时同样会被剔除，但数据留得住。"
    )
    c1, c2 = st.columns(2)
    if c1.button("取消", width='stretch'):
        st.rerun()
    if c2.button("确认清空", type="primary", width='stretch'):
        space, _ = space_for(ref)
        save_history(ref, clear_history(load_history(ref, space)))
        st.session_state["flash"] = f"已清空 {ref.label} 的全部数据"
        st.rerun()


def render_data_tab(ref: ProjectRef, space: CMPRecipeSpace, df: pd.DataFrame) -> None:
    st.subheader("① 历史数据")
    st.caption(f"当前文件：`{ref.rel_label}/{HISTORY_NAME}`（{len(df)} 条记录）")

    if len(df):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("记录数", len(df))
        split_preview = ledger_from_df(df, space).split()
        c2.metric("可训练", split_preview.n_train)
        c3.metric("剔除/失效",
                  len(split_preview.excluded_measurement_ids) + len(split_preview.failure_ids))
        c4.metric("规格满足率", f"{space.spec_hit_rate(split_preview.train_Y):.0%}"
                  if split_preview.n_train else "—")

        # ---- 可勾选的数据表：用于删除 / 打标签 ----
        editor_df = df.copy()
        editor_df.insert(0, "选择", False)
        state = st.data_editor(
            editor_df,
            key=f"editor_{ref.substrate}_{ref.family_dir}",
            hide_index=True,
            height=300,
            width='stretch',
            column_config={
                "选择": st.column_config.CheckboxColumn(
                    "选择", help="勾选后可用下方按钮删除或打异常标签", default=False
                ),
                "anomaly": st.column_config.SelectboxColumn(
                    "anomaly", help="measurement = 训练时剔除；response = 保留并标记",
                    options=["none", "measurement", "response"], disabled=True,
                ),
                "failure": st.column_config.SelectboxColumn(
                    "failure", options=["none", "mix_fail", "scratch",
                                        "agglomeration", "other"], disabled=True,
                ),
            },
            disabled=[c for c in editor_df.columns if c != "选择"],
        )
        picked = [i for i, flag in enumerate(state["选择"].tolist()) if flag]
        n_meas = int(anomaly_mask(df).sum())
        st.caption(f"已勾选 **{len(picked)}** 行 ｜ 当前 measurement 标记 {n_meas} 行"
                   f"（这些行训练时会被剔除，但仍留在台账里）")

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("🗑 删除选中行", disabled=not picked, width='stretch',
                     help="硬删除：用于录错、重复登记。不可撤销"):
            save_history(ref, delete_rows(df, picked))
            st.session_state["flash"] = f"已删除 {len(picked)} 行"
            st.rerun()
        if b2.button("🏷 标记为 measurement", disabled=not picked, width='stretch',
                     help="训练时自动剔除，但保留数据可追溯"):
            save_history(ref, tag_rows(df, picked, "measurement"))
            st.session_state["flash"] = f"已标记 {len(picked)} 行为 measurement"
            st.rerun()
        if b3.button("↩️ 恢复为 none", disabled=not picked, width='stretch',
                     help="取消异常标记，重新纳入训练"):
            save_history(ref, tag_rows(df, picked, "none"))
            st.session_state["flash"] = f"已恢复 {len(picked)} 行"
            st.rerun()
        if b4.button("🗑 清空全部数据", width='stretch',
                     help="二次确认后清空当前体系的所有记录"):
            clear_confirm_dialog(ref, len(df))
    else:
        st.info("该项目还没有数据。请上传 CSV/Excel，或下载模板填写后再上传。")

    st.divider()
    st.subheader("② 追加实验数据")
    left, right = st.columns([2, 1])
    with left:
        up = st.file_uploader(
            "拖拽 CSV / Excel 到此处（表头需与模板一致）",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )
    with right:
        st.download_button(
            "下载数据模板 (CSV)", template_csv(space),
            file_name=f"{ref.substrate}_{ref.family_dir}_template.csv",
            mime="text/csv", width='stretch',
        )

    if up is not None:
        try:
            raw = pd.read_excel(up) if up.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"读取失败：{exc}")
            return
        chk = validate_upload(raw, space)
        for m in chk.messages:
            st.warning(m)
        if chk.accepted.shape[0]:
            st.success(f"通过校验 {chk.accepted.shape[0]} 行 / 共 {raw.shape[0]} 行")
            st.dataframe(chk.accepted, width='stretch', hide_index=True, height=200)
        if chk.rejected.shape[0]:
            st.error(f"{chk.rejected.shape[0]} 行未通过物理校验（不会写入历史库）")
            st.dataframe(chk.rejected, width='stretch', hide_index=True, height=200)
        if chk.ok and st.button("确认追加到当前体系", type="primary"):
            added, skipped = append_rows(ref, space, chk.accepted)
            st.session_state["flash"] = f"已追加 {added} 行，跳过重复 {skipped} 行"
            st.rerun()


def render_train_tab(ref: ProjectRef, space: CMPRecipeSpace, df: pd.DataFrame) -> None:
    st.subheader("一键训练 GPR（模块 2）")
    if not len(df):
        st.info("先上传数据。")
        return
    ledger = ledger_from_df(df, space)
    split = ledger.split()
    st.caption(f"训练集 {split.n_train} 条 / 台账 {len(ledger)} 条"
               f"（measurement 剔除 {len(split.excluded_measurement_ids)}，"
               f"failure 单列 {len(split.failure_ids)}）")

    if split.n_train < 3:
        st.error("可训练样本少于 3 条，无法拟合 GPR。")
        return
    if split.n_train < 16:
        st.error(
            f"⚠️ 当前只有 **{split.n_train}** 条可训练样本（< 16）。"
            "GP 很可能触发 fallback 降级——推荐结果将来自常数模型，**不要直接拿去上机**。"
        )

    if st.button("开始训练", type="primary", width='stretch'):
        with st.spinner("拟合三个目标的 GPR（含 LOOCV 健康检查）…"):
            sur = train(space, split)
        st.session_state["surrogate"] = sur
        st.session_state["split_used"] = split

    sur = st.session_state.get("surrogate")
    if sur is None:
        st.info("点击上方按钮开始训练。")
        return

    table = health_table(sur)
    st.markdown("##### 健康报告（每轮必看）")
    st.dataframe(_style_health(table), width='stretch', hide_index=True)

    bad = [n for n, d in sur.diagnostics.items() if d.status == STATUS_FALLBACK]
    if bad:
        st.error(
            f"🔴 **{bad} 处于常量后备**：这些目标当前学不出结构，预测一律低置信度，"
            "模块 3 会把它们从多目标向量中剔除。请补充样本或检查测量异常。"
        )
    degraded = [n for n, d in sur.diagnostics.items() if d.status == "degraded"]
    if degraded and not bad:
        st.warning(f"🟡 {degraded} 为 degraded：可排序但不宜直接作为交付预测。")
    if not bad and not degraded:
        st.success("🟢 三个目标全部通过健康检查。")

    with st.expander("ARD 长度尺度（机理感知的特征重要性）"):
        st.text(sur.lengthscale_table())


def render_recommend_tab(ref: ProjectRef, space: CMPRecipeSpace, df: pd.DataFrame) -> None:
    st.subheader("推荐与下载（模块 3 + 模块 4）")
    sur = st.session_state.get("surrogate")
    if sur is None:
        st.info("请先在「训练」页完成一次训练。")
        return
    split = st.session_state.get("split_used")
    if split is None or split.n_train < 3:
        st.info("训练集不足。")
        return

    c1, c2 = st.columns([1, 1])
    q = c1.number_input("本轮推荐条数 q", min_value=1, max_value=6, value=3)
    fast = c2.checkbox(
        "快速模式（3 次重启 / 40 迭代，约 11 s；关闭后约 40 s、解更优）", value=True
    )
    if st.button("生成推荐", type="primary", width='stretch'):
        with st.spinner("约束 qNEHVI 优化 + 前沿筛选 …"):
            rec, front = recommend(space, split, int(q), fast)
        st.session_state["rec"] = rec
        st.session_state["front"] = front

    rec = st.session_state.get("rec")
    front = st.session_state.get("front")
    if rec is None:
        st.info("点击「生成推荐」开始。")
        return

    if rec.optimizer_degraded:
        st.warning("优化器降级：SLSQP 未找到可行解，候选来自可行池中采集值最高的批次。")
    if rec.dropped_objectives:
        st.warning(f"以下目标处于后备，已从多目标向量剔除：{rec.dropped_objectives}")

    st.markdown("##### 推荐配方（模块 3 输出，均已满足硬约束）")
    st.dataframe(recommendation_table(rec, space), width='stretch', hide_index=True)
    st.caption(
        f"采集函数 {rec.acq_name} ｜ 参考点(原量纲) "
        f"{[round(v, 1) for v in rec.ref_point_physical]} ｜ "
        f"预测 HV {rec.hv_after_pseudo:.4f} ｜ 耗时 {rec.elapsed_s:.1f}s"
    )

    st.markdown("##### 前沿榜单（模块 4：规格过滤 + 冻结权重 TOPSIS）")
    if front.spec_fallback:
        st.warning(
            "前沿上没有一条满足全部产线规格 → 已降级为按违反度排序。"
            "要交付合格配方，需要把规格当约束，而不只是筛选。"
        )
    st.dataframe(front_table(front, space), width='stretch', hide_index=True)
    st.caption(
        f"权重（冻结）：{front.weights} ｜ 前沿 {front.n_front} 条 / "
        f"合格 {front.n_on_spec} 条 ｜ 熵权敏感性 ρ={front.rank_corr_entropy:+.3f}"
    )

    st.download_button(
        "⬇️ 一键下载为 Excel（推荐 + 前沿 + 说明）",
        to_excel(rec, front, space, ref),
        file_name=f"{ref.substrate}_{ref.family_dir}_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width='stretch',
    )

    st.divider()
    render_explain_block(ref, rec, space)


# --------------------------------------------------------------------------- #
# 5. RAG 知识库（📚 入库 / 💬 问答 / 🤖 配方解释）
# --------------------------------------------------------------------------- #
#: 文献类型（代号 → 中文；与 src/rag_service.py 的 DOC_TYPES 保持一致）
DOC_TYPE_LABELS: Dict[str, str] = {
    "formula": "配方",
    "mechanism": "机理",
    "news": "新闻",
}

#: 服务健康状态的缓存有效期（秒）：避免每次 rerun 都打一次 HTTP
RAG_HEALTH_TTL = 6.0


def _rag_health(force: bool = False):
    """带缓存的健康检查（服务没开时也是一次快速的 connect 失败，不会卡住）。"""
    now = time.time()
    cached = st.session_state.get("_rag_health")
    if (not force and cached and now - cached[0] < RAG_HEALTH_TTL):
        return cached[1]
    health = rag.health()
    st.session_state["_rag_health"] = (now, health)
    return health


def rag_status_block() -> None:
    """侧边栏里的一小块：RAG 服务状态 + 重试。"""
    st.sidebar.divider()
    st.sidebar.subheader("📚 RAG 知识库")
    health = _rag_health()
    if health.ok:
        st.sidebar.success(f"服务{health.label}（PID {health.pid}）", icon="🟢")
        st.sidebar.caption(
            f"构建 `{health.build or '旧版'}` ｜ 切片 {health.chunk_size}/{health.chunk_overlap} "
            f"｜ 模型加载 {health.model_load_s or '?'}s"
        )
        if not health.supports("doc_type"):
            st.sidebar.warning("服务是旧版本，不支持文献类型，请重启 `启动RAG服务.bat`",
                               icon="⚠️")
    elif health.warming:
        st.sidebar.info("BGE-M3 加载中，请稍候…", icon="⏳")
    else:
        st.sidebar.warning(f"服务{health.label}", icon="⚪")
        st.sidebar.caption("双击项目根目录的 `启动RAG服务.bat` 后点下方按钮")
    if st.sidebar.button("🔄 重新检测 RAG 服务", width='stretch'):
        _rag_health(force=True)
        st.rerun()
    return health


def _decode_upload(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def render_ingest_tab(ref: ProjectRef, space: CMPRecipeSpace) -> None:
    st.subheader("📚 知识入库")
    health = _rag_health()
    if not health.ok:
        st.error("RAG 服务未就绪，无法入库。")
        st.markdown(SERVICE_HINT)
        if health.warming:
            st.info("服务正在加载模型，等 20–40 秒后再点「重新检测 RAG 服务」。")
        return
    if not health.supports("doc_type"):
        # 旧进程没有 features 字段 → 一定早于「文献类型」这次改造
        st.error(
            f"RAG 服务是**旧版本**（构建 `{health.build or '未知/旧版'}`，"
            f"PID {health.pid}），不认识「文献类型」参数。\n\n"
            "请关闭那个服务窗口（或双击 `启动RAG服务.bat`，它会自动清理旧进程），"
            "等窗口出现「✅ RAG 服务就绪」后，再点左侧「🔄 重新检测 RAG 服务」。"
        )
        return

    st.caption(
        f"知识库：`{RAG_CONFIG.kb_dir}` ｜ 集合 `{RAG_CONFIG.collection}` ｜ "
        f"模型：`{RAG_CONFIG.model_dir}`（本地加载，未复制）"
    )

    # ---- 库状态 ----
    try:
        stt = rag.stats()
        counts = stt.get("doc_type_counts") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("库内 chunk", stt["chunks"])
        c2.metric("已入库文件", stt["files"])
        c3.metric("当前切片", f"{stt['chunk_size']}/{stt['chunk_overlap']}")
        st.caption(
            f"按文献类型：配方 **{counts.get('formula', 0)}** ｜ 机理 **{counts.get('mechanism', 0)}** ｜ "
            f"新闻 **{counts.get('news', 0)}** ｜ 未标注（早期入库）**{counts.get('untyped', 0)}**　"
            f"—— 检索时配方/机理优先，新闻降权，未标注居中"
        )
    except RagServiceError as exc:
        st.error(exc.user_message())
        return

    st.divider()
    up = st.file_uploader(
        "拖拽 `.md` / `.txt` 文献到此处（可多选）",
        type=["md", "txt", "markdown"], accept_multiple_files=True,
        key=f"rag_up_{ref.substrate}_{ref.family_dir}",
    )
    mode_label = st.radio(
        "入库模式",
        ["⚡ 全文标准入库（大块切片，快）",
         "📝 摘要增强入库（DeepSeek 生成结构化元数据，慢但更可检索）"],
        horizontal=True,
        help="摘要增强会用大模型抽取适用基底/去除速率/缺陷/关键参数，写入 chunk 元数据；"
             "正文仍然完整入库，不会被摘要替代",
    )
    mode = "summary" if "摘要" in mode_label else "raw"

    # ---- 文献类型：强制必填（检索调权的前提）----
    type_labels = {v: k for k, v in DOC_TYPE_LABELS.items()}
    picked_type = st.selectbox(
        "文献类型（必填，用于检索调权）",
        list(DOC_TYPE_LABELS.values()),
        index=None,
        placeholder="请选择：配方 / 机理 / 新闻",
        help="配方=实测配方与工艺窗口；机理=模型与机理分析；新闻=行业动态（检索时会被降权，"
             "避免上万篇新闻淹没技术资料）",
    )
    doc_type = type_labels.get(picked_type or "", "")
    if not doc_type:
        st.warning("请先选择文献类型：它决定这篇文献在检索时的优先级，不能留空。", icon="📌")

    c1, c2, c3, c4 = st.columns(4)
    chunk_size = int(c1.number_input(
        "chunk_size", min_value=300, max_value=4000, value=int(RAG_CONFIG.chunk_size),
        step=100, help="化工文献建议 1500–2000：太碎会把配方参数劈开"))
    chunk_overlap = int(c2.number_input(
        "chunk_overlap", min_value=0, max_value=800, value=int(RAG_CONFIG.chunk_overlap),
        step=50, help="块间重叠，防止配方跨块断裂"))
    substrate_opts = ["（不指定）", *SUBSTRATE_OPTIONS]
    sub_idx = substrate_opts.index(ref.substrate) if ref.substrate in substrate_opts else 0
    substrate = c3.selectbox(
        "基底 metadata", substrate_opts, index=sub_idx,
        help="写入 chunk 元数据，问答时可按基底过滤；老数据没有该字段，会用文本关键词兜底")
    substrate = "" if substrate == "（不指定）" else substrate
    c4.metric("待入库", f"{len(up or [])} 个文件")

    if st.button("开始入库", type="primary", width='stretch',
                 disabled=not up or not doc_type):
        assert up is not None
        results: List[Dict[str, Any]] = []
        log_box = st.empty()
        lines: List[str] = [f"文献类型：{picked_type}（{doc_type}）"]
        for f in up:
            text = _decode_upload(f.read())
            lines.append(f"—— {f.name}（{len(text)} 字符）")
            log_box.code("\n".join(lines), language="text")
            with st.spinner(f"处理 {f.name} …"):
                try:
                    res = rag.ingest(
                        [{"name": f.name, "text": text, "substrate": substrate,
                          "doc_type": doc_type}],
                        mode=mode, substrate=substrate, doc_type=doc_type,
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    )
                    results.extend(res.get("results") or [])
                    lines.extend(res.get("log") or [])
                except RagServiceError as exc:
                    results.append({"name": f.name, "status": "error", "chunks": 0,
                                    "reason": exc.message})
                    lines.append(f"   → error：{exc.message}")
            log_box.code("\n".join(lines), language="text")

        if results:
            st.dataframe(pd.DataFrame(results), width='stretch', hide_index=True)
            imported = sum(1 for r in results if r.get("status") == "imported")
            dups = sum(1 for r in results if r.get("status") == "duplicate")
            errs = sum(1 for r in results if r.get("status") == "error")
            st.success(f"完成：入库 {imported} 个，哈希重复被拦截 {dups} 个，失败 {errs} 个")
        _rag_health(force=True)

    st.divider()
    with st.expander("📋 入库台账（kb_record.json）与维护", expanded=False):
        try:
            rec = rag.record("list")
            items = rec.get("items") or []
            st.caption(f"共 {rec.get('count', len(items))} 条哈希记录："
                       f"用于拦截重复文件，删它不会删向量库。")
            if not items:
                st.info("台账为空：还没有文件走哈希入库，或已被清空。表头与内容都正常，可直接上传文件。")
            else:
                st.dataframe(_record_table(items), width='stretch', hide_index=True,
                             height=220)
                missing = sum(1 for it in items
                              if it.get("chunks") is None or not it.get("substrate"))
                if missing:
                    st.caption(f"其中 {missing} 条是旧版台账记录（没有 `chunks`/`substrate` 字段），"
                               f"已显示为「—」；不影响去重判断。")
            if st.button("🗑 清空入库记录（不删向量库）"):
                rag.record("clear")
                st.success("已清空 kb_record.json")
        except RagServiceError as exc:
            st.warning(exc.user_message())
        except Exception as exc:                    # noqa: BLE001
            # 台账结构再奇怪也不该白屏：降级成一条提示
            st.warning(f"台账读取失败（已降级显示）：{type(exc).__name__}: {exc}")


def _record_table(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """把 ``kb_record.json`` 的条目渲染成表，**缺失字段一律降级为「—」**。

    老版台账只有 ``filename / import_time / mode``，没有 ``chunks / substrate``；
    直接 ``df[["chunks", "substrate"]]`` 会抛 ``KeyError: not in index``。
    这里逐个字段用 ``.get`` 取值并统一成字符串，从根上避免列不存在的问题。
    """
    mode_label = {"raw": "全文标准", "summary": "摘要增强"}
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mode = item.get("mode")
        chunks = item.get("chunks")
        dtype = item.get("doc_type_label") or DOC_TYPE_LABELS.get(
            str(item.get("doc_type") or ""), "")
        rows.append({
            "文件名": str(item.get("filename") or "—"),
            "类型": dtype or "—",
            "入库时间": str(item.get("import_time") or "—"),
            "模式": mode_label.get(str(mode), str(mode) if mode else "—"),
            "切片数": "—" if chunks in (None, "") else str(chunks),
            "基底": str(item.get("substrate") or "—"),
            "哈希": str(item.get("hash") or "")[:12] or "—",
        })
    # 列固定，即使 rows 为空也不会 KeyError
    return pd.DataFrame(rows, columns=["文件名", "类型", "入库时间", "模式",
                                       "切片数", "基底", "哈希"])


def render_qa_tab(ref: ProjectRef, space: CMPRecipeSpace) -> None:
    st.subheader("💬 知识问答")
    health = _rag_health()
    if not health.ok:
        st.error("RAG 服务未就绪，无法问答。")
        st.markdown(SERVICE_HINT)
        if health.warming:
            st.info("服务正在加载模型，稍等 20–40 秒后再试。")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    use_filter = c1.checkbox(f"只看基底 = {ref.substrate} 的文献", value=True,
                             help="老数据没有 substrate 元数据时，自动用文本关键词兜底")
    type_opts = ["全部", *DOC_TYPE_LABELS.values()]
    picked_types = st.multiselect(
        "文献类型过滤（不选 = 全部）", type_opts[1:], default=[],
        help="只看【配方】或【机理】可有效屏蔽行业新闻的干扰",
    )
    top_k = int(c2.number_input("召回条数", min_value=1, max_value=10,
                                value=int(RAG_CONFIG.top_k)))
    if c3.button("🧹 清空对话", width='stretch'):
        st.session_state["rag_chat"] = []
        st.rerun()
    type_codes = [k for k, v in DOC_TYPE_LABELS.items() if v in picked_types] or None

    history: List[Dict[str, Any]] = st.session_state.setdefault("rag_chat", [])
    for msg in history:
        with st.chat_message(msg["role"],
                             avatar="🧑‍🔬" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            hits = msg.get("hits") or []
            if hits:
                with st.expander(f"📎 引用 {len(hits)} 段资料"):
                    for i, h in enumerate(hits, 1):
                        meta = h.get("metadata") or {}
                        st.markdown(
                            f"**{i}. {meta.get('source_file', '未知来源')}** "
                            f"（相似度 {h.get('score', 0):.3f}）"
                        )
                        st.caption(h.get("text", "")[:400].replace("\n", " "))

    question = st.chat_input("基于私有 CMP 知识库提问…")
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("检索 + 生成中…"):
                try:
                    res = rag.ask(
                        question, top_k=top_k,
                        substrate=ref.substrate if use_filter else "",
                        filter_mode="auto",
                        doc_types=type_codes,
                    )
                    ans = res.get("answer") or "（无回答）"
                    hits = res.get("hits") or []
                    st.markdown(ans)
                    if hits:
                        with st.expander(f"📎 引用 {len(hits)} 段资料"):
                            for i, h in enumerate(hits, 1):
                                meta = h.get("metadata") or {}
                                st.markdown(
                                    f"**{i}. {meta.get('source_file', '未知来源')}** "
                                    f"（{h.get('doc_type_label', '未标注')}，"
                                    f"相似度 {h.get('score', 0):.3f}）"
                                )
                                st.caption(h.get("text", "")[:400].replace("\n", " "))
                    st.caption(f"过滤：{res.get('filter_note', '')}")
                    history.append({"role": "assistant", "content": ans, "hits": hits})
                except RagServiceError as exc:
                    st.error(exc.user_message())
                    history.append({"role": "assistant",
                                    "content": f"⚠️ 调用失败：{exc.message}"})
        st.rerun()


def render_explain_block(ref: ProjectRef, rec, space: CMPRecipeSpace) -> None:
    """为每个推荐配方提供"一键解释"按钮（RAG 检索 + 大模型，≤200 字）。"""
    st.markdown("##### 🤖 一键解释配方（基于私有知识库）")
    health = _rag_health()
    if not health.ok:
        st.caption("RAG 服务未连接：启动 `启动RAG服务.bat` 后即可使用该功能。")
        return

    n = rec.recipes.shape[0]
    cols = st.columns(min(n, 3))
    cache: Dict[str, str] = st.session_state.setdefault("rag_explain", {})
    st.caption("检索策略：**优先【配方】与【机理】类文献**，新闻类降权，未标注（早期入库）居中；"
               "回答会逐条标注来源文件名。")
    for i in range(n):
        recipe = {name: round(float(rec.recipes[i, j]), 3)
                  for j, name in enumerate(space.names)}
        metrics = {obj: round(float(rec.predicted_mean[i, j]), 2)
                   for j, obj in enumerate(rec.active_objectives)}
        key = f"{ref.label}|{sorted(recipe.items())}"

        btn = cols[i % len(cols)].button(f"解释配方 #{i + 1}", key=f"explain_{i}",
                                         width='stretch')
        if btn:
            with st.spinner(f"检索并解释配方 #{i + 1} …"):
                try:
                    res = rag.explain(recipe, metrics, substrate=ref.substrate,
                                      prefer_types=["formula", "mechanism"])
                    cache[key] = res.get("answer") or "（无回答）"
                    st.session_state[f"explain_hits_{key}"] = res.get("hits") or []
                    st.session_state[f"explain_note_{key}"] = res.get("filter_note", "")
                except RagServiceError as exc:
                    st.error(exc.user_message())
        if key in cache:
            st.info(cache[key])
            note = st.session_state.get(f"explain_note_{key}")
            if note:
                st.caption(f"检索：{note}")
            hits = st.session_state.get(f"explain_hits_{key}") or []
            if hits:
                with st.expander(f"配方 #{i + 1} 的引用资料（{len(hits)} 段）"):
                    for j, h in enumerate(hits, 1):
                        meta = h.get("metadata") or {}
                        st.markdown(
                            f"**{j}. {meta.get('source_file', '未知')}** "
                            f"（{h.get('doc_type_label', '未标注')}，"
                            f"相似度 {h.get('score', 0):.3f}）"
                        )
                        st.caption(h.get("text", "")[:300].replace("\n", " "))


def main() -> None:
    st.set_page_config(page_title="CMP 配方智能体", page_icon="🧪", layout="wide")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash, icon="✅")

    ref = sidebar()
    if ref is None:
        st.title("🧪 CMP 抛光液配方开发 Agent")
        st.info("左侧新建一个「基底 + 化学体系」项目开始。")
        return

    space, blind = space_for(ref)
    df = load_history(ref, space)

    st.title("🧪 CMP 抛光液配方开发 Agent")
    st.caption(
        f"当前体系：**{ref.label}** ｜ 路径 `{ref.rel_label}` ｜ "
        f"参数空间 `{space.name.split(' / ')[-1]}` ｜ 指纹 `{space.config_hash()}`"
        + (" ｜ ⚠️ 盲跑（无预设）" if blind else "")
    )
    add = ref.additives
    add_txt = "；".join(
        f"{k}={v}" for k, v in
        (("表面活性剂", add.get("surfactant")),
         ("缓蚀剂", add.get("inhibitor")),
         ("pH 调节剂", add.get("ph_adjuster")))
        if v
    ) or "无"
    intent = ref.design_intent
    st.caption(f"族内添加剂：{add_txt}" + (f" ｜ 设计意图：{intent}" if intent else ""))
    if blind:
        st.warning(
            "当前体系不在 `FAMILY_PRESETS` 中：Preston 等机理先验按默认值起步，"
            "预测精度依赖后续标定实验。",
            icon="⚠️",
        )

    tab_data, tab_train, tab_rec, tab_ingest, tab_qa = st.tabs(
        ["① 数据", "② 训练", "③ 推荐", "📚 知识入库", "💬 知识问答"]
    )
    with tab_data:
        render_data_tab(ref, space, df)
    with tab_train:
        render_train_tab(ref, space, df)
    with tab_rec:
        render_recommend_tab(ref, space, df)
    with tab_ingest:
        render_ingest_tab(ref, space)
    with tab_qa:
        render_qa_tab(ref, space)


if __name__ == "__main__":
    # Streamlit 以 __main__ 执行本脚本；导入本模块（如数据播种脚本）时不会启动界面
    main()
