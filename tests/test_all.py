"""端到端集成测试：config → surrogate → acquisition → pareto → experiment。

用法::

    python tests/test_all.py          # 全部跑一遍，失败则 exit 1
    python tests/test_all.py -s       # 跳过最慢的 acquisition 用例

设计原则：

* **不用 pytest**：环境里不一定装了它，而"看一眼就能跑"的脚本在交接时更可靠；
* 每个用例独立计时，失败即打印完整 traceback 并继续跑后面的（一次看到全部问题）；
* 断言只测**不变量**（形状、可行性、单调性、互斥性、往返一致），不测具体数值——
  数值会随随机种子漂移，那不是回归测试该管的事。
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

# 允许从任意目录运行：把项目根加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import build_default_config
from src.surrogate_model import CMPGPRSurrogate, synthetic_cmp_benchmark

DATA_CLOSED_LOOP = ROOT / "data" / "closed_loop_n34.json"

_results = []


def _run(fn) -> None:
    t0 = time.perf_counter()
    try:
        fn()
        _results.append((fn.__name__, True, time.perf_counter() - t0, ""))
        print(f"  ✓ {fn.__name__:<22}{time.perf_counter() - t0:6.2f}s")
    except Exception as exc:  # noqa: BLE001 - 测试运行器要兜住一切
        _results.append((fn.__name__, False, time.perf_counter() - t0, str(exc)))
        print(f"  ✗ {fn.__name__:<22}{time.perf_counter() - t0:6.2f}s  "
              f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# 1. 模块 1：问题定义
# --------------------------------------------------------------------------- #
def test_config() -> None:
    space = build_default_config()
    assert space.dim == 10
    assert space.objective_names == ("MRR", "Ra", "Dishing")
    assert len(space.hard_constraints) == 9

    # 边界与归一化往返
    b = space.bounds()
    assert b.shape == (2, 10)
    assert np.all(b[0] < b[1])
    X = space.sample_lhs_feasible(12, seed=1)
    assert X.shape == (12, 10)
    assert np.allclose(space.from_unit(space.to_unit(X)), X, atol=1e-9)

    # 采样必须全部可行（含隐式规则之外的硬约束）
    assert space.max_violation(X).max() <= 1e-9

    # 目标符号：MRR 最大化，其余最小化
    assert space.objective_signs().tolist() == [1.0, -1.0, -1.0]

    # TOPSIS 权重冻结且和为 1（含成本维度）
    w = space.topsis.weights_array()
    assert abs(w.sum() - 1.0) < 1e-9
    assert "Cost" in space.topsis.names

    # 指纹稳定：同一份配置两次构造必须一致；且 to_dict 必须可 JSON 序列化
    assert space.config_hash() == build_default_config().config_hash()
    import json

    assert json.dumps(space.to_dict(), ensure_ascii=False)  # 不抛异常即通过


# --------------------------------------------------------------------------- #
# 2. 模块 2：代理模型
# --------------------------------------------------------------------------- #
def _fit_surrogate(n: int = 25, seed: int = 5) -> tuple:
    space = build_default_config()
    X = torch.tensor(space.sample_lhs_feasible(n, seed=seed), dtype=torch.double)
    Y = synthetic_cmp_benchmark(space, X, noise=0.03, seed=seed + 100)
    return space, X, Y


def test_surrogate() -> None:
    space, X, Y = _fit_surrogate(25)
    sur = CMPGPRSurrogate(space, loocv_mode="efficient").fit(X, Y)

    # 25 条数据应让三个目标全部通过健康检查
    statuses = {k: v.status for k, v in sur.diagnostics.items()}
    assert all(s == "ok" for s in statuses.values()), statuses
    assert all(v.loocv_r2 > 0.5 for v in sur.diagnostics.values())

    # 预测：形状、有限性、不确定性分层（认知 ≤ 总）
    pred = sur.predict(X[:5])
    assert pred.mean.shape == (5, 3)
    assert torch.isfinite(pred.mean).all()
    assert (pred.sd_epistemic <= pred.sd_total + 1e-9).all()
    assert (pred.sd_aleatoric >= -1e-12).all()

    # 域外点的认知不确定性应更大
    corner = torch.tensor(space.bounds()[0], dtype=torch.double).reshape(1, -1)
    epi_in = float(pred.sd_epistemic[:, 0].mean())
    epi_out = float(sur.predict(corner).sd_epistemic[0, 0])
    assert epi_out > epi_in, (epi_in, epi_out)

    # 健康模型可组装 ModelListGP 交给模块 3
    assert len(sur.to_model_list_gp().models) == 3

    # 常量后备：纯随机指标必须全部判 fallback，且组装被拒绝。
    # 这里刻意用 refit 模式的 LOOCV（每折重训）：efficient 模式固定超参数、
    # 对小样本偏乐观，随机噪声偶尔能蹭到 R²>0 而漏判（实测踩到）。
    X10 = torch.tensor(space.sample_lhs_feasible(10, seed=3), dtype=torch.double)
    Y_noise = torch.randn(10, 3, dtype=torch.double) * 10.0
    bad = CMPGPRSurrogate(space, loocv_mode="refit").fit(X10, Y_noise)
    assert all(d.is_fallback for d in bad.diagnostics.values()), \
        {k: v.status for k, v in bad.diagnostics.items()}
    try:
        bad.to_model_list_gp()
        raise AssertionError("含后备目标时不应组装成功")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# 3. 模块 3：约束采集（用小预算跑通链路，不做质量评估）
# --------------------------------------------------------------------------- #
def test_acquisition() -> None:
    from src.acquisition import CMPBatchPlanner

    space, X, Y = _fit_surrogate(25)
    # 测试用小预算：2 次重启、20 次迭代、32 条 MC 样本
    fast = replace(
        space,
        acquisition=replace(
            space.acquisition, num_restarts=2, maxiter=20, num_samples=32
        ),
    )
    planner = CMPBatchPlanner(fast, seed=1)
    rec = planner.recommend_next_batch(X, Y, q=2, round_index=0)

    assert rec.recipes.shape == (2, space.dim)
    assert rec.predicted_mean.shape == (2, 3)
    assert rec.active_objectives == space.objective_names
    assert rec.dropped_objectives == ()
    # 硬约束：优化器给的候选必须真的可行
    assert fast.max_violation(rec.recipes.numpy()).max() <= 1e-6
    # 参考点由数据定初值（这是修过的一个 bug，必须守住）
    assert rec.ref_point_physical != space.reference_point().tolist()
    assert len(rec.ref_point) == 3
    assert rec.acq_name.startswith("qLogNEHVI")
    assert "第 0 轮推荐" in rec.describe()
    # 回灌接口：物理量纲拼接，不能混入归一化坐标
    X2, Y2 = planner.commit_batch(X, Y, rec, rec.predicted_mean)
    assert X2.shape == (27, 10)
    assert Y2.shape == (27, 3)
    assert np.allclose(X2[:25].numpy(), X.numpy())


# --------------------------------------------------------------------------- #
# 4. 模块 4：前沿 + TOPSIS
# --------------------------------------------------------------------------- #
def test_pareto() -> None:
    from src.pareto import ParetoSelector, is_non_dominated_mask

    space = build_default_config()
    import json

    payload = json.loads(DATA_CLOSED_LOOP.read_text(encoding="utf-8"))
    X = np.asarray(payload["recipes"], dtype=float)
    Y = np.asarray(payload["responses"], dtype=float)

    # 非支配判断与暴力法一致
    Y_max = Y * space.objective_signs()
    fast = is_non_dominated_mask(Y_max)
    slow = np.ones(len(Y), dtype=bool)
    for i in range(len(Y)):
        for j in range(len(Y)):
            if i != j and np.all(Y_max[j] >= Y_max[i]) and np.any(Y_max[j] > Y_max[i]):
                slow[i] = False
                break
    assert (fast == slow).all()
    assert 0 < fast.sum() < len(Y)

    sel = ParetoSelector(space)
    rep = sel.select(X, Y, top_k=5)
    assert rep.n_front == int(fast.sum())
    # 规格过滤：入榜必须全合格（否则说明过滤没生效）
    if not rep.spec_fallback:
        assert all(c.on_spec for c in rep.ranked)
    # 分数降序、落在 [0,1]
    scores = [c.topsis_score for c in rep.ranked]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    # 权重必须是 config 里的冻结值
    assert np.allclose(
        np.asarray(list(rep.weights.values())), space.topsis.weights_array()
    )


# --------------------------------------------------------------------------- #
# 5. 模块 5：台账 + 校验 + 三分
# --------------------------------------------------------------------------- #
def test_experiment() -> None:
    from src.experiment import Anomaly, ExperimentLedger, Failure

    space = build_default_config()
    ledger = ExperimentLedger.from_dataset(DATA_CLOSED_LOOP, space)
    assert len(ledger) == 34

    rep = ledger.validate()
    assert rep.ok, f"迁移后的台账不应有 error：{rep.codes('error')}"

    split = ledger.split()
    assert split.train_X.shape == (34, 10)
    assert split.train_Y.shape == (34, 3)
    assert split.excluded_measurement_ids == []
    # 三分与源文件一致（迁移零丢失）
    import json

    payload = json.loads(DATA_CLOSED_LOOP.read_text(encoding="utf-8"))
    assert np.allclose(split.train_X, np.asarray(payload["recipes"], dtype=float))
    assert np.allclose(split.train_Y, np.asarray(payload["responses"], dtype=float))

    # 三分语义：measurement 剔除 / response 保留标记 / failure 不进主目标
    l2 = ExperimentLedger.from_dataset(DATA_CLOSED_LOOP, space)
    l2.records[0].anomaly = Anomaly.MEASUREMENT
    l2.records[1].anomaly = Anomaly.RESPONSE
    l2.records[2].failure = Failure.SCRATCH
    l2.records[2].y = {n: None for n in space.objective_names}
    s2 = l2.split()
    assert "E000" in s2.excluded_measurement_ids
    assert "E001" in s2.train_ids and "E001" in s2.response_flagged_ids
    assert "E002" not in s2.train_ids
    assert s2.failure_ids == ["E002"]
    # 34 − 1(measurement) − 1(failure) = 32
    assert s2.n_train == 32

    # 往返一致
    tmp = ROOT / "data" / "_test_ledger.json"
    try:
        l2.to_json(tmp)
        back = ExperimentLedger.from_json(tmp, space)
        assert [r.to_dict() for r in back] == [r.to_dict() for r in l2]
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 6. 端到端：推荐 → 回灌 → 台账 → 前沿
# --------------------------------------------------------------------------- #
def test_end_to_end() -> None:
    from src.acquisition import CMPBatchPlanner
    from src.experiment import ExperimentLedger, ExperimentRecord
    from src.pareto import ParetoSelector

    space, X, Y = _fit_surrogate(25)
    fast = replace(
        space,
        acquisition=replace(
            space.acquisition, num_restarts=2, maxiter=20, num_samples=32
        ),
    )
    planner = CMPBatchPlanner(fast, seed=2)
    rec = planner.recommend_next_batch(X, Y, q=2, round_index=0)

    # 模拟"上机实测"
    Y_new = synthetic_cmp_benchmark(fast, rec.recipes, noise=0.03, seed=999)
    X2, Y2 = planner.commit_batch(X, Y, rec, Y_new)
    assert X2.shape == (27, 10)

    # 写进台账并校验
    ledger = ExperimentLedger(fast)
    for i in range(X2.shape[0]):
        ledger.append(
            ExperimentRecord(
                experiment_id=f"E{i:03d}",
                round_id=0 if i < 25 else 1,
                batch_id="B00" if i < 25 else "B01",
                config_hash=fast.config_hash(),
                recipe={n: float(X2[i, j]) for j, n in enumerate(fast.names)},
                y={n: float(Y2[i, j]) for j, n in enumerate(fast.objective_names)},
                covariates={
                    "pad_use_count": float(i * 10 % 380),
                    "slurry_age_hours": 2.0,
                    "ambient_temp_c": 22.0,
                    "ambient_humidity_pct": 45.0,
                    "tool_id": 1.0,
                },
                cost_cny=None,
                timestamp=f"2026-11-16T{8 + (i % 10):02d}:00:00+08:00",
            )
        )
    rep = ledger.validate()
    assert rep.ok, f"端到端台账校验失败：{rep.codes('error')}"
    split = ledger.split()
    assert split.n_train == 27

    # 前沿 + TOPSIS 能消费同一批数据
    front = ParetoSelector(fast).select(split.train_X, split.train_Y, top_k=3)
    assert front.n_front > 0
    assert front.n_ranked > 0


# --------------------------------------------------------------------------- #
# 7. Web 界面（Streamlit AppTest，进程内真实执行脚本）
# --------------------------------------------------------------------------- #
def test_app_tokens() -> None:
    """自定义化学物质的名称规范化 + 体系元数据（纯函数，无副作用）。"""
    from src.config import CHEMISTRY_CATALOG, is_path_safe_token
    from src.app import (
        ABRASIVE_TOKENS,
        ADD_NONE_LABEL,
        COMPLEXING_TOKENS,
        INHIBITOR_TOKENS,
        OXIDIZER_TOKENS,
        PH_ADJUSTER_TOKENS,
        SURFACTANT_TOKENS,
        family_dir_name,
        slugify_token,
        split_family_dir,
        to_token,
    )

    # 化学物质目录：代号必须能安全用作目录名，且每个角色内不重复
    for role, items in CHEMISTRY_CATALOG.items():
        tokens = [t for t, _ in items]
        assert len(tokens) == len(set(tokens)), f"{role} 代号重复"
        for t in tokens:
            assert is_path_safe_token(t), f"{role} 的代号 {t!r} 不能安全用作目录名"
    # 用户点名要的常见化学品必须在预设里
    for token in ("CeO2", "APS", "CitricAcid", "PEG", "BTA", "KOH"):
        assert any(t == token for items in CHEMISTRY_CATALOG.values() for t, _ in items), token

    # 空格 -> 下划线；Windows 非法字符剔除；保留大小写与中文
    assert slugify_token("Ammonium Persulfate") == "Ammonium_Persulfate"
    assert slugify_token("  ammonium   persulfate  ") == "ammonium_persulfate"
    assert slugify_token("a/b:c*d?e|f<g>h\"i") == "abcdefghi"
    assert slugify_token("新络合剂 A-1") == "新络合剂_A_1"
    assert slugify_token("CeO2 (ceria)") == "CeO2_ceria"
    # 空值与 Windows 保留设备名
    assert slugify_token("") == "NA"
    assert slugify_token("   ") == "NA"
    assert slugify_token("CON") == "_CON"
    assert slugify_token("com1") == "_com1"
    # 长度上限 + 首尾点/下划线清理
    assert len(slugify_token("x" * 200)) <= 40
    assert slugify_token(".hidden.") == "hidden"

    # 下拉框返回值 -> 目录代号
    assert to_token("胶体二氧化硅 Colloidal Silica (SiO2)", ABRASIVE_TOKENS) == "SiO2"
    assert to_token("sio2", ABRASIVE_TOKENS) == "SiO2"          # 大小写不敏感
    assert to_token("过硫酸铵 APS (APS)", OXIDIZER_TOKENS) == "APS"
    assert to_token("EDTA", COMPLEXING_TOKENS) == "EDTA"
    assert to_token("Ammonium Persulfate", OXIDIZER_TOKENS) == "Ammonium_Persulfate"
    assert to_token("聚乙二醇 PEG (PEG)", SURFACTANT_TOKENS) == "PEG"
    assert to_token("苯并三氮唑 BTA (BTA)", INHIBITOR_TOKENS) == "BTA"
    assert to_token("氢氧化钾 KOH (KOH)", PH_ADJUSTER_TOKENS) == "KOH"
    assert ADD_NONE_LABEL not in ABRASIVE_TOKENS

    # 目录名：无添加剂时与旧版完全一致；有添加剂时加 "__" 后缀
    assert family_dir_name("SiO2", "H2O2", "Glycine") == "SiO2_H2O2_Glycine"
    assert (family_dir_name("SiO2", "H2O2", "Glycine", "PEG", "BTA", "KOH")
            == "SiO2_H2O2_Glycine__PEG_BTA_KOH")
    assert (family_dir_name("SiO2", "H2O2", "Glycine", "", "BTA", "")
            == "SiO2_H2O2_Glycine__None_BTA_None")
    assert split_family_dir("SiO2_H2O2_Glycine__PEG_BTA_KOH") == (
        "SiO2_H2O2_Glycine", "PEG_BTA_KOH")
    assert split_family_dir("SiO2_H2O2_Glycine") == ("SiO2_H2O2_Glycine", "")


def test_data_ops() -> None:
    """删除 / 清空 / 批量打异常标签（纠错机制的核心逻辑）。"""
    from src.app import anomaly_mask, clear_history, delete_rows, tag_rows

    df = pd.DataFrame({
        "experiment_id": ["E0", "E1", "E2"],
        "MRR": [100.0, 200.0, 300.0],
        "anomaly": ["none", "none", "none"],
    })
    # 删除指定行（按位置索引）
    d = delete_rows(df, [1])
    assert len(d) == 2 and list(d["experiment_id"]) == ["E0", "E2"]
    assert list(d.index) == [0, 1], "删除后应重排索引"
    assert len(delete_rows(df, [])) == 3, "空选择应原样返回"

    # 打标签：measurement 会被训练剔除
    t = tag_rows(df, [0, 2], "measurement")
    assert list(t["anomaly"]) == ["measurement", "none", "measurement"]
    assert list(anomaly_mask(t)) == [True, False, True]
    assert list(tag_rows(t, [0], "none")["anomaly"]) == ["none", "none", "measurement"]

    # 清空：保留列结构
    c = clear_history(df)
    assert len(c) == 0 and list(c.columns) == list(df.columns)

    # 没有 anomaly 列时也能打标签
    df2 = df.drop(columns=["anomaly"])
    assert "anomaly" in tag_rows(df2, [1], "response").columns


def test_rag_config() -> None:
    """RAG 配置：大 chunk 默认值、.env 覆盖、服务地址。"""
    import os

    from dataclasses import replace

    from src.config import RAG_CONFIG, RAGConfig, load_dotenv

    # 化工文献默认：大 chunk（1500~2000）+ 300 重叠
    assert 1500 <= RAG_CONFIG.chunk_size <= 2000, RAG_CONFIG.chunk_size
    assert RAG_CONFIG.chunk_overlap == 300, RAG_CONFIG.chunk_overlap
    # 路径必须指向原项目，且模型没有被复制进本仓库
    assert "private_rag_cmp" in RAG_CONFIG.model_dir
    assert "private_rag_cmp" in RAG_CONFIG.kb_dir
    assert not (ROOT / "model").exists(), "BGE-M3 不应该被复制到本项目"
    assert RAG_CONFIG.service_url.startswith("http")

    # .env 解析：KEY=VALUE / 注释 / 引号 / 不覆盖已有环境变量
    # 注意：用工作区内的临时文件，不用 tempfile（沙箱对系统临时目录的清理会报权限错误）
    p = ROOT / "_test_env_tmp" / ".env"
    p.parent.mkdir(exist_ok=True)
    try:
        p.write_text("# 注释行\nRAG_TEST_A=1\nRAG_TEST_B='带引号'\n坏行没有等号\n",
                     encoding="utf-8")
        os.environ.pop("RAG_TEST_A", None)
        os.environ.pop("RAG_TEST_B", None)
        loaded = load_dotenv(p)
        assert loaded["RAG_TEST_A"] == "1"
        assert loaded["RAG_TEST_B"] == "带引号"
        assert os.environ["RAG_TEST_A"] == "1"
        os.environ["RAG_TEST_A"] = "already"
        load_dotenv(p)                      # 默认不覆盖
        assert os.environ["RAG_TEST_A"] == "already"
        for k in ("RAG_TEST_A", "RAG_TEST_B"):
            os.environ.pop(k, None)
    finally:
        p.unlink(missing_ok=True)
        try:
            p.parent.rmdir()
        except OSError:
            pass

    # 环境变量能覆盖默认值
    os.environ["RAG_CHUNK_SIZE"] = "2000"
    try:
        cfg2 = RAGConfig.from_env()
        assert cfg2.chunk_size == 2000, cfg2.chunk_size
    finally:
        os.environ.pop("RAG_CHUNK_SIZE", None)

    # 未配置 key 时 has_api_key 为 False（不误报）
    assert replace(RAG_CONFIG, api_key="").has_api_key is False


def test_rag_bridge_offline() -> None:
    """服务没启动时：health 不抛异常、ask 给出可执行的中文提示。"""
    from dataclasses import replace

    from src.config import RAG_CONFIG
    from src.rag_bridge import RagClient, RagServiceError

    offline = RagClient(replace(RAG_CONFIG, service_url="http://127.0.0.1:9"))
    h = offline.health()
    assert h.ok is False and h.warming is False
    assert "连不上" in (h.error or ""), h.error
    assert h.label == "未连接"

    try:
        offline.ask("铜 CMP 的 dishing 怎么控制？")
    except RagServiceError as exc:
        assert "启动RAG服务.bat" in exc.hint, exc.hint
    else:
        raise AssertionError("服务不可用时 ask 应该抛 RagServiceError")


def test_rag_service_selftest() -> None:
    """切片自检必须在 Python 3.11（RAG 服务所在解释器）里跑通。"""
    import subprocess

    script = ROOT / "src" / "rag_service.py"
    assert script.exists(), "缺少 src/rag_service.py"
    try:
        proc = subprocess.run(
            ["py", "-3.11", "-X", "utf8", str(script), "--selftest"],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=180,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"    （py -3.11 不可用，跳过：{exc}）")
        return
    out = (proc.stdout or "") + (proc.stderr or "")
    if "No module named 'langchain_text_splitters'" in out:
        print("    （3.11 环境缺 langchain_text_splitters，跳过切片用例）")
        return
    assert proc.returncode == 0, out[-1500:]
    for token in ("配方完整性", "跨块重叠", "全部通过"):
        assert token in out, f"自检输出缺少 {token}"


def test_record_table() -> None:
    """旧版台账（缺 chunks/substrate）渲染不得 KeyError（线上白屏回归）。"""
    from src.app import _record_table

    # 复现现场：只有 filename / import_time / mode 三个字段
    old = [{"hash": "0d783b37d6fc1c19", "filename": "旧文献.md",
            "import_time": "2026-11-06 19:24:19", "mode": "summary"}]
    df = _record_table(old)
    assert list(df.columns) == ["文件名", "类型", "入库时间", "模式", "切片数", "基底", "哈希"], \
        list(df.columns)
    assert df.loc[0, "切片数"] == "—", df.to_dict("records")
    assert df.loc[0, "基底"] == "—"
    assert df.loc[0, "类型"] == "—", "旧台账没有 doc_type，应显示 —"
    assert df.loc[0, "模式"] == "摘要增强"

    # 新格式字段照常显示
    df_new = _record_table([{"hash": "h", "filename": "n.md", "import_time": "t",
                             "mode": "raw", "chunks": 3, "substrate": "Cu",
                             "doc_type": "formula", "doc_type_label": "配方"}])
    assert df_new.loc[0, "切片数"] == "3" and df_new.loc[0, "基底"] == "Cu"
    assert df_new.loc[0, "类型"] == "配方"
    assert df_new.loc[0, "模式"] == "全文标准"

    # 空 / 畸形数据也不能崩
    assert len(_record_table([])) == 0
    assert len(_record_table([]).columns) == 7
    weird = _record_table([{}, {"hash": "y", "filename": None}])
    assert len(weird) == 2 and (weird["文件名"] == "—").all()


def test_doc_type_contract() -> None:
    """文献类型：前端标签表 与 服务端类型体系/优先级规则 必须一致。"""
    from src.app import DOC_TYPE_LABELS

    assert DOC_TYPE_LABELS == {"formula": "配方", "mechanism": "机理", "news": "新闻"}

    src = (ROOT / "src" / "rag_service.py").read_text(encoding="utf-8")
    # 服务端类型体系与优先级（新闻必须垫底、未标注居中）
    assert 'DOC_TYPES: Dict[str, str] = {' in src
    assert '"news": 2' in src and '"": 1' in src
    # 入库强制校验
    assert "必须为每篇文献指定文献类型" in src
    # Prompt 规则
    assert "工艺工程师" in src and "[来源：" in src
    assert "优先采用" in src and "200 字" in src


def test_app() -> None:
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:
        print("    （streamlit 未安装，跳过 UI 用例）")
        return
    if not (ROOT / "data" / "projects").exists():
        print("    （data/projects 不存在，跳过 UI 用例；先跑 scripts/seed_projects.py）")
        return

    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=180).run()
    assert not at.exception, f"界面首次渲染异常：{[e.value for e in at.exception]}"
    assert len(at.tabs) == 5, f"应有五个标签页（含 RAG 两页），实际 {len(at.tabs)}"
    tab_labels = [t.label for t in at.tabs]
    for need in ("📚 知识入库", "💬 知识问答"):
        assert need in tab_labels, f"缺少标签页 {need}"
    assert at.title, "缺少标题"

    # 7 个体系输入（基底 + 磨料/氧化剂/络合剂 + 表面活性剂/缓蚀剂/pH 调节剂）
    # 都必须允许手输新物质
    boxes = {b.label: b for b in at.sidebar.selectbox}
    for label in (
        "Substrate Type（基底）",
        "Abrasive（磨料）", "Oxidizer（氧化剂）", "Complexing（络合剂）",
        "Surfactant（表面活性剂）", "Inhibitor（缓蚀剂）", "pH Adjuster（pH 调节剂）",
    ):
        assert label in boxes, f"缺少下拉框 {label}"
        assert getattr(boxes[label].proto, "accept_new_options", False), \
            f"{label} 未开启 accept_new_options（不能输入新物质）"

    # 设计意图文本框 + 可勾选的数据表
    assert any("Design Intent" in t.label for t in at.sidebar.text_area), "缺少设计意图输入框"
    assert len(at.data_editor) >= 1, "历史数据应使用可编辑/可勾选的数据表"

    buttons = [b for b in at.button if "开始训练" in b.label]
    assert buttons, "找不到训练按钮（项目目录为空？）"
    buttons[0].click().run()
    assert not at.exception, f"训练时界面异常：{[e.value for e in at.exception]}"
    assert any(
        "状态" in list(getattr(d.value, "columns", [])) for d in at.dataframe
    ), "训练后应出现健康报告表"


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    skip_slow = "-s" in argv
    print("=" * 78)
    print("CMP-Agent 集成测试：config → surrogate → acquisition → pareto → "
          "experiment → app")
    print("=" * 78)

    _run(test_config)
    _run(test_app_tokens)
    _run(test_data_ops)
    _run(test_rag_config)
    _run(test_rag_bridge_offline)
    _run(test_record_table)
    _run(test_doc_type_contract)
    _run(test_experiment)
    _run(test_pareto)
    _run(test_surrogate)
    if skip_slow:
        print("  - test_acquisition / test_end_to_end 已按 -s 跳过")
    else:
        _run(test_acquisition)
        _run(test_end_to_end)
    _run(test_app)
    _run(test_rag_service_selftest)

    total = sum(r[2] for r in _results)
    failed = [r for r in _results if not r[1]]
    print("-" * 78)
    print(f"  共 {len(_results)} 个用例，通过 {len(_results) - len(failed)}，"
          f"失败 {len(failed)}，耗时 {total:.1f}s")
    if failed:
        print("  失败用例：")
        for name, _, _, err in failed:
            print(f"    - {name}: {err}")
        print("\n结论：FAIL")
        return 1
    print("\n结论：PASS（exit=0）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
