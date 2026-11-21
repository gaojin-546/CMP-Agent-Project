"""CMP 抛光液配方开发 Agent 的源码包。

模块规划与实现状态（详见项目 ``README.md``）：

* ``config``          —— 模块 1：问题定义与参数空间（✅ 已实现，仅依赖 numpy）
* ``surrogate_model`` —— 模块 2：GPR 代理模型 + 健康检查 + 常量后备（✅ 已实现）
* ``acquisition``     —— 模块 3：约束 qNEHVI + 成本感知 + 动态参考点（✅ 已实现）
* ``pareto``          —— 模块 4：帕累托前沿 + TOPSIS 筛选（✅ 已实现）
* ``experiment``      —— 模块 5：实验台账 + 校验器 + 数据三分（✅ 已实现）
* ``app``             —— Web 界面（Streamlit，多层级数据隔离 + RAG 两页，✅ 已实现）
* ``rag_service``     —— RAG 微服务（Python 3.11：chromadb + 本地 BGE-M3 + DeepSeek，✅ 已实现）
* ``rag_bridge``      —— 网页侧 RAG HTTP 客户端（Python 3.15，零额外依赖，✅ 已实现）
* ``monitor``         —— 模块 6：收敛判断与可视化（待实现）
* ``confidence``      —— 模块 7：置信度与域外标志（待实现）
* ``registry``        —— 模块 8：知识库与版本管理（待实现）

自检入口（每个模块都能独立跑通并留档）：

    python -m src.config            # 模块 1：参数空间 / 约束 / 成本
    python -m src.surrogate_model   # 模块 2：训练 / LOOCV / 健康检查 / 留出验证
    python -m src.acquisition       # 模块 3：约束 qNEHVI / 参考点 / 闭环三轮
    python -m src.pareto            # 模块 4：前沿提取 / 规格过滤 / TOPSIS
    python -m src.experiment        # 模块 5：台账迁移 / 校验 / 三分 / 缺陷注入
    python tests/test_all.py        # 端到端集成测试（exit=0 表示全绿）
    streamlit run src/app.py        # Web 界面（RAG 需先启动 启动RAG服务.bat）
    py -3.11 src/rag_service.py     # RAG 微服务（也可 --selftest 只测切片）

**RAG 为什么是独立进程**：网页跑在 Python 3.15，而 chromadb 依赖的 onnxruntime
在 3.15 上没有 wheel；且沙箱只允许写工作区，chromadb 连只读查询都会因 sqlite
的读写模式失败。因此知识库能力留在原 Python 3.11 环境，做成 127.0.0.1:8770 的
本地 HTTP 服务，网页通过 ``rag_bridge`` 调用。详细设计与接口见 README §5.14。

端到端最小闭环：

    >>> from src.config import build_default_config
    >>> from src.acquisition import CMPBatchPlanner
    >>> from src.pareto import ParetoSelector
    >>> from src.experiment import ExperimentLedger
    >>> space = build_default_config()
    >>> planner = CMPBatchPlanner(space)
    >>> rec = planner.recommend_next_batch(X_obs, Y_obs, q=3)      # 探索
    >>> X_new, Y_new = planner.commit_batch(X_obs, Y_obs, rec, Y_measured)
    >>> report = ParetoSelector(space).select(X_new, Y_new)        # 筛选
    >>> ledger = ExperimentLedger.from_dataset("data/closed_loop_n34.json", space)
    >>> ledger.validate().ok, ledger.split().n_train               # 审计与三分

分工提醒：模块 3 负责**探索**（信息增益最大化的配方），模块 4 负责**筛选**
（在已知前沿上按冻结偏好排序并剔除不合规格的点），模块 5 负责**记账与审计**
（哪些数据可信、哪些只能进失效集）。三者不要混成一个目标。
"""

__version__ = "0.1.0"
"""与 ``CMPRecipeSpace.version`` 保持一致（整体版本号由模块 8 统一管理）。"""
