"""界面自检：用 Streamlit 官方 AppTest 在进程内真实执行 src/app.py。

为什么不用"打开浏览器看看"：脚本是按会话执行的，没有客户端连接时
`streamlit run` 根本不会跑业务代码；AppTest 能把脚本完整执行一遍并
拿到组件树，等价于一次真实的首次访问（还能点按钮）。

运行：python scripts/smoke_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "src" / "app.py")


def main() -> int:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()

    print("=" * 78)
    print("首次访问（默认项目）")
    print("=" * 78)
    if at.exception:
        for e in at.exception:
            print("❌ 异常：", e.value)
        return 1
    print(f"标题        : {[t.value for t in at.title]}")
    print(f"侧边栏下拉框: {[s.label for s in at.sidebar.selectbox]}")
    print(f"下拉当前值  : {[s.value for s in at.sidebar.selectbox][:2]}")
    print(f"按钮        : {[b.label for b in at.button]}")
    print(f"标签页      : {[t.label for t in at.tabs]}")
    print(f"指标卡      : {[(m.label, m.value) for m in at.metric]}")
    print(f"成功/警告/错误: {len(at.success)}/{len(at.warning)}/{len(at.error)}")
    for w in at.warning:
        print("  ⚠", str(w.value)[:100])
    for e in at.error:
        print("  ✖", str(e.value)[:100])

    assert at.title, "缺少标题"
    assert len(at.sidebar.selectbox) >= 1, "侧边栏缺少项目下拉框"
    assert len(at.tabs) == 5, f"应有五个标签页（含 RAG 两页），实际 {len(at.tabs)}"

    # ---- 切到第二个（无预设）项目，验证盲跑警告 ----
    print("\n" + "=" * 78)
    print("切换到无预设体系（应出现盲跑警告）")
    print("=" * 78)
    sb = at.sidebar.selectbox[0]
    sb.set_value("W / Al2O3_APS_CitricAcid").run()
    if at.exception:
        for e in at.exception:
            print("❌ 异常：", e.value)
        return 1
    warns = [str(w.value) for w in at.warning]
    print(f"警告 {len(warns)} 条：")
    for w in warns:
        print("  ⚠", w[:110])
    assert any("盲跑" in w or "FAMILY_PRESETS" in w for w in warns), "无预设体系应有盲跑警告"
    print(f"数据提示：{[str(i.value)[:70] for i in at.info]}")

    # ---- 回到主项目，训练 ----
    print("\n" + "=" * 78)
    print("回到 Cu / SiO2_H2O2_Glycine，点击「开始训练」")
    print("=" * 78)
    at.sidebar.selectbox[0].set_value("Cu / SiO2_H2O2_Glycine").run()
    print(f"当前项目：{at.sidebar.selectbox[0].value}")
    print(f"可见按钮：{[b.label for b in at.button]}")
    print(f"信息条  ：{[str(i.value)[:60] for i in at.info]}")
    train_btn = [b for b in at.button if "开始训练" in b.label]
    assert train_btn, "找不到训练按钮"
    train_btn[0].click().run()
    if at.exception:
        for e in at.exception:
            print("❌ 异常：", e.value)
        return 1
    tables = {d: len(t.value) for d, t in enumerate(at.dataframe)}
    print(f"数据表数量 : {len(at.dataframe)}，行数 {tables}")
    print(f"成功/错误  : {len(at.success)}/{len(at.error)}")
    for s in at.success:
        print("  ✓", str(s.value)[:90])
    for e in at.error:
        print("  ✖", str(e.value)[:90])
    # 健康报告应包含三个目标
    health_found = False
    for d in at.dataframe:
        cols = list(getattr(d.value, "columns", []))
        if "状态" in cols and "目标" in cols:
            health_found = True
            print("健康报告：")
            print(d.value.to_string(index=False))
    assert health_found, "训练后应出现健康报告表"

    # ---- 生成推荐 + 下载按钮 ----
    print("\n" + "=" * 78)
    print("点击「生成推荐」（快速模式），验证推荐表 + Excel 下载按钮")
    print("=" * 78)
    gen_btn = [b for b in at.button if "生成推荐" in b.label]
    assert gen_btn, "找不到『生成推荐』按钮"
    gen_btn[0].click().run()
    if at.exception:
        for e in at.exception:
            print("❌ 异常：", e.value)
        return 1
    print(f"数据表数量 : {len(at.dataframe)}")
    for d in at.dataframe:
        cols = list(getattr(d.value, "columns", []))
        if "预测成本(元)" in cols or "TOPSIS" in cols:
            print(f"  · 表（{len(cols)} 列，{len(d.value)} 行）：{cols[:6]}…")
    dl = [b.label for b in at.download_button]
    print(f"下载按钮   : {dl}")
    assert any("Excel" in x for x in dl), "应出现 Excel 下载按钮"

    # ---- 自定义新化学物质（accept_new_options 的手输路径） ----
    # AppTest 不支持给下拉框塞"不在选项里"的值（会 ValueError），
    # 因此这一段直接走 create_project，验证同一套规范化与盲跑标记逻辑。
    print("\n" + "=" * 78)
    print("自定义新化学物质：Ammonium Persulfate（不在 FAMILY_PRESETS 里）")
    print("=" * 78)
    import shutil

    from src.app import (
        OXIDIZER_TOKENS,
        PROJECTS_ROOT,
        create_project,
        scan_projects,
        slugify_token,
        to_token,
    )

    raw = "Ammonium Persulfate"
    tok = to_token(raw, OXIDIZER_TOKENS)
    print(f"用户输入   : {raw!r}")
    print(f"规范化代号 : {tok!r}（空格 → 下划线）")
    assert tok == "Ammonium_Persulfate", tok
    assert slugify_token("CON") == "_CON", "Windows 保留名未加前缀"

    intent = "验证用：把 Dishing 压到 150 Å 以下，同时守住 MRR ≥ 1500 Å/min"
    ref = create_project(
        "Cu", "SiO2", "H2O2", raw,
        surfactant_token="PEG", inhibitor_token="BTA", ph_adjuster_token="KOH",
        design_intent=intent,
    )
    target = PROJECTS_ROOT / "Cu" / "SiO2_H2O2_Ammonium_Persulfate__PEG_BTA_KOH"
    try:
        print(f"目录       : {ref.rel_label}")
        print(f"history.csv: {ref.history_path.exists()}")
        assert ref.family_dir == "SiO2_H2O2_Ammonium_Persulfate__PEG_BTA_KOH", ref.family_dir
        assert ref.history_path.exists()
        meta = ref.read_meta()
        print(f"meta       : family_preset={meta['family_preset']} blind_run={meta['blind_run']}")
        print(f"additives  : {meta['additives']}")
        print(f"design     : {meta['design_intent']}")
        assert meta["family_preset"] is None and meta["blind_run"] is True, \
            "全新体系必须标记为盲跑（创建时会弹窗要求确认）"
        assert meta["additives"] == {"surfactant": "PEG", "inhibitor": "BTA",
                                     "ph_adjuster": "KOH"}, meta["additives"]
        assert meta["design_intent"] == intent

        # 元数据必须落进 history.csv 的表头，且追加行时自动填充
        from src.app import (append_rows, history_columns, load_history,
                             space_for)
        space_new, _ = space_for(ref)
        cols = history_columns(space_new)
        for c in ("substrate_type", "surfactant_type", "inhibitor_type",
                  "ph_adjuster_type", "design_intent"):
            assert c in cols, f"history.csv 缺少元数据列 {c}"
        # 注意：新建体系按字母序会排到最前，不能再用 scan_projects()[0] 取主项目
        seeded_ref = [p for p in scan_projects()
                      if p.label == "Cu / SiO2_H2O2_Glycine"][0]
        seeded = load_history(seeded_ref, space_new)
        append_rows(ref, space_new, seeded.iloc[[0]])
        hist = load_history(ref, space_new)
        print(f"history 行 : {len(hist)}，添加剂列={hist.loc[0, 'surfactant_type']}/"
              f"{hist.loc[0, 'inhibitor_type']}/{hist.loc[0, 'ph_adjuster_type']}")
        assert len(hist) == 1
        assert hist.loc[0, "surfactant_type"] == "PEG"
        assert hist.loc[0, "design_intent"] == intent

        labels = [p.label for p in scan_projects()]
        print(f"侧栏可见   : {ref.label in labels}（共 {len(labels)} 个项目）")
        assert ref.label in labels

        at2 = AppTest.from_file(APP, default_timeout=180).run()
        assert not at2.exception, f"新增体系后界面异常：{[e.value for e in at2.exception]}"
        opts = list(getattr(at2.sidebar.selectbox[0], "options", []) or [])
        print(f"下拉选项   : {opts}")
        assert ref.label in opts, "新建体系未出现在下拉列表中"
    finally:
        shutil.rmtree(target, ignore_errors=True)
        print("已清理测试目录（不污染 data/projects）")

    # ---- 导出精度：数字单元格统一 0.000 且仍是数字 ----
    print("\n" + "=" * 78)
    print("导出 Excel 的浮点精度（3 位小数、不写成文本）")
    print("=" * 78)
    import io as _io

    import openpyxl

    from src.app import (ledger_from_df, load_history, recommend, space_for,
                         to_excel)

    seeded_ref = [p for p in scan_projects()
                  if p.label == "Cu / SiO2_H2O2_Glycine"][0]
    _space, _ = space_for(seeded_ref)
    _split = ledger_from_df(load_history(seeded_ref, _space), _space).split()
    _rec, _front = recommend(_space, _split, 2, fast=True)
    blob = to_excel(_rec, _front, _space, seeded_ref)
    wb = openpyxl.load_workbook(_io.BytesIO(blob))
    print(f"sheets     : {wb.sheetnames}")
    checked = 0
    for sheet in ("推荐配方", "前沿榜单"):
        ws = wb[sheet]
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    checked += 1
                    assert cell.number_format == "0.000", \
                        f"{sheet}!{cell.coordinate} 格式为 {cell.number_format}"
                    assert cell.value == round(cell.value, 3), \
                        f"{sheet}!{cell.coordinate} 值 {cell.value} 未取到 3 位"
    print(f"数字单元格 : {checked} 个，全部为 0.000 格式且值已四舍五入")
    assert checked > 0

    # ---- RAG：五页签 / 服务联通 / 问答 / 配方解释 ----
    print("\n" + "=" * 78)
    print("RAG 集成：五页签 + 服务状态 + 知识问答 + 一键解释配方")
    print("=" * 78)
    from src.config import RAG_CONFIG
    from src.rag_bridge import RagClient

    judge = RagClient()
    health = judge.health()
    print(f"服务地址   : {RAG_CONFIG.service_url}")
    print(f"服务状态   : ok={health.ok} warming={health.warming} "
          f"model={health.model_load_s}s")
    assert health.ok and not health.warming, health.detail()

    at.run()
    assert not at.exception, [e.value for e in at.exception]
    tabs = [t.label for t in at.tabs]
    print(f"标签页     : {tabs}")
    assert len(tabs) == 5, tabs

    sidebar_ok = [s.value for s in at.sidebar.success]
    print(f"侧栏状态   : {sidebar_ok}")
    assert any("已连接" in s for s in sidebar_ok), sidebar_ok

    stats = judge.stats()
    print(f"库内 chunk : {stats['chunks']}（文件 {stats['files']}）")

    # 问答：发一个真实问题，走通"检索 + 去重 + 基底过滤 + DeepSeek"
    assert len(at.chat_input) == 1, f"问答页应有输入框，实际 {len(at.chat_input)}"
    question = "铜 CMP 中 H2O2 浓度过高会带来什么问题？"
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.value for e in at.exception]
    chat = at.session_state["rag_chat"]
    print(f"对话轮次   : {len(chat)} 条（用户 + 助手）")
    assert len(chat) >= 2, chat
    reply = chat[-1]["content"]
    print(f"回答长度   : {len(reply)} 字")
    print(f"回答摘要   : {reply[:120].replace(chr(10), ' ')}…")
    assert len(reply) > 40, reply
    assert chat[-1].get("hits"), "回答应附带引用片段"

    # 一键解释配方：在推荐页点击第 1 条
    explain_btns = [b for b in at.button if "解释配方" in b.label]
    print(f"解释按钮   : {[b.label for b in explain_btns]}")
    assert explain_btns, "推荐页缺少『解释配方』按钮"
    explain_btns[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    cache = at.session_state.get("rag_explain") or {}
    print(f"解释结果   : {len(cache)} 条")
    assert cache, "点击后应产生解释文本"
    answer = list(cache.values())[0]
    print(f"解释内容   : {answer[:120]}…")
    assert len(answer) > 20

    print("\n" + "=" * 78)
    print("全部通过：三业务页 + RAG 两页、训练/推荐/导出、自定义化学物质、"
          "元数据落盘、导出精度、知识问答、配方解释")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
