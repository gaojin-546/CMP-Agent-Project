# data/projects —— 实验数据仓库（内容不入库）

本目录存放**真实实验数据**，按「基底 / 化学体系」两级隔离：

```text
data/projects/
└── {substrate_type}/
    └── {abrasive}_{oxidizer}_{complexing}[__{surfactant}_{inhibitor}_{ph_adjuster}]/
        ├── history.csv      # 每行一条实验：10 个设计变量 + 3 个目标 + 5 个协变量 + 元数据
        └── meta.json        # 基底/化学体系/添加剂/设计意图/预设族/blind_run 标记
```

## 为什么这个目录被 .gitignore 排除

实验数据属于**产线私有资产**：配方参数、成本、良率线索都在里面，公开即等于泄露工艺。
仓库只保留：

- 本说明文件与该目录占位（`data/projects/.gitkeep`），保证 `git clone` 后目录结构存在；
- `data/template.csv`：可直接用于「📚 知识入库 / ① 数据 → 上传」的空白模板（含一行示例值）。

## 怎么开始用

```bash
# 方式 A：界面里新建（推荐）
#   侧栏「② 新建项目 / 体系」→ 选基底与化学物质 → 创建新体系 → 上传 template.csv

# 方式 B：命令行播种示例项目（生成 34 条闭环数据，用于试跑）
python scripts/seed_projects.py
```

`data/template.csv` 的列与 `src/config.py` 的参数空间严格对应，
其中 `experiment_id / anomaly / failure` 等台账字段说明见 README §6「跨模块数据契约」。
