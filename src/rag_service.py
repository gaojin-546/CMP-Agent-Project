"""RAG 微服务：把原本 Tkinter 里的知识库能力暴露成本地 HTTP 接口。

**为什么是"独立服务"而不是在网页进程里直接 import chromadb**

1. 网页跑在 Python 3.15（torch/botorch/streamlit 都装在这里），而 chromadb 依赖
   的 ``onnxruntime`` 在 3.15 上没有 wheel；老环境的 Python 3.11 里一切就绪。
2. chromadb 打开 sqlite 时是"读写"模式（即使只查询），而 DSH 沙箱只允许写工作区，
   所以从沙箱里连"读"都会报 ``attempt to write a readonly database``。
   本服务由用户双击 ``启动RAG服务.bat`` 启动，不在沙箱进程树内，读写都正常。

因此：本文件用 **Python 3.11** 运行，只依赖标准库 + 原环境已有的
``chromadb / langchain-huggingface / langchain-text-splitters``；
网页端（3.15）通过 ``src/rag_bridge.py`` 走 HTTP 调用。

启动：
    py -3.11 src/rag_service.py            # 或双击 启动RAG服务.bat
自检（不连库、不加载模型，只测切片）：
    py -3.11 src/rag_service.py --selftest
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 0. 环境：必须在 import 第三方库之前设置
# --------------------------------------------------------------------------- #
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # 模型是本地目录，别去联网探测
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAG_CONFIG as CFG, load_dotenv  # noqa: E402

load_dotenv()  # 允许 .env 覆盖已存在的空值

#: 服务端构建号与能力清单：``/health`` 会返回，网页端据此判断"服务是不是旧进程"。
#: 起因：Streamlit 只重跑脚本、不重新 import 已加载模块；RAG 服务同理——
#: 改完代码不重启进程，前端就会拿到旧签名/旧接口，报错信息还南辕北辙。
BUILD = "2026-11-18.3"
FEATURES = ["doc_type", "prefer_types", "exclusive_bind", "record_normalized",
            "health_features"]

TOKENIZERS_PARALLELISM_HINT = True


# --------------------------------------------------------------------------- #
# 1. 子系统状态（延迟加载，加载进度对 /health 可见）
# --------------------------------------------------------------------------- #
STATE: Dict[str, Any] = {
    "ready": False,
    "warming": True,
    "error": None,
    "model_load_s": None,
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "requests": 0,
}
_LOCK = threading.RLock()          # 串行化重活：chromadb / ST 在 CPU 上不是线程安全的
_EMB = None                        # HuggingFaceEmbeddings
_COL = None                        # chroma collection


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 2. 切片：面向化工文献的"大 chunk + 标题锚定"
# --------------------------------------------------------------------------- #
from langchain_core.documents import Document          # noqa: E402
from langchain_text_splitters import (                 # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

#: 化工文献里段落常以这些行开头，属于"参数块"边界，分割时优先保留在一起
_PARAM_LINE = re.compile(
    r"(^\s*[-*•]\s)|(^\s*\d+[\.、]\s)|(磨料|氧化剂|络合|表面活性|缓蚀|pH|压力|转速|"
    r"流量|温度|wt%|wt\.%|Å/min|nm/min|dishing|MRR)",
    re.IGNORECASE,
)


def chunk_markdown(
    text: str,
    chunk_size: int = 1800,
    chunk_overlap: int = 300,
) -> List[Document]:
    """把一篇 Markdown/TXT 切成**大块**，尽量保证配方参数不被劈开。

    策略（三段式）：

    1. ``MarkdownHeaderTextSplitter`` 按 ``#``/``##``/``###`` 切段，标题写进 metadata；
    2. **贪心合并**相邻小节，直到接近 ``chunk_size``——这是关键：文献里
       "磨料 5 wt% / pH 4.2 / 2 psi" 往往分散在连续几段里，合并后才完整；
    3. 单段仍超长时，用 ``RecursiveCharacterTextSplitter`` 在段内按
       ``\\n\\n`` → ``\\n`` → ``。`` 递归切，并带上 ``chunk_overlap`` 重叠。

    最后再在相邻 chunk 之间补一次重叠（把上一块的尾部接到下一块开头），
    保证跨块边界的上下文不断裂。
    """
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 2))

    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,          # 标题留在正文里，chunk 自带上下文
    )
    try:
        sections = md_splitter.split_text(raw)
    except Exception:                 # 非 Markdown 的纯文本
        sections = []
    if not sections:
        sections = [Document(page_content=raw, metadata={})]

    # ---- 第 2 步：贪心合并 ----
    # min_chunk：避免"标题行单独成块"这种垃圾 chunk。合并时若当前缓冲还很小，
    # 宁可略微超过 chunk_size 也把它带上——大 chunk 策略本来就允许一定弹性。
    min_chunk = max(120, chunk_size // 8)
    merged: List[Document] = []
    buf_text = ""
    buf_meta: Dict[str, Any] = {}
    for sec in sections:
        body = sec.page_content.strip()
        if not body:
            continue
        if not buf_text:
            buf_text, buf_meta = body, dict(sec.metadata)
            continue
        if len(buf_text) + 2 + len(body) <= chunk_size:
            buf_text = f"{buf_text}\n\n{body}"
            continue
        if len(buf_text) < min_chunk:
            # 当前缓冲太小（典型：只有一行标题），无条件是并入下一节，
            # 超长部分交给第 3 步的递归切分处理——绝不让"# 标题"单独成块
            buf_text = f"{buf_text}\n\n{body}"
            continue
        merged.append(Document(page_content=buf_text, metadata=buf_meta))
        buf_text, buf_meta = body, dict(sec.metadata)
    if buf_text:
        merged.append(Document(page_content=buf_text, metadata=buf_meta))

    # ---- 第 3 步：超长块内部递归切 ----
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
        keep_separator=True,
    )
    pieces: List[Document] = []
    for doc in merged:
        if len(doc.page_content) <= chunk_size:
            pieces.append(doc)
            continue
        for sub in recursive.split_documents([doc]):
            pieces.append(sub)

    # ---- 第 3.5 步：把递归切分产生的"小尾巴"并掉 ----
    # RecursiveCharacterTextSplitter 遇到 "# 标题\n\n超长正文" 这种结构时，会先
    # 把只有标题那一行单独成块（keep_separator 的行为）。这种 3 字符的块进库
    # 只会污染检索，统一向后并入相邻块。
    merged_small: List[Document] = []
    carry = ""
    for doc in pieces:
        if carry:
            doc = Document(page_content=f"{carry}\n\n{doc.page_content}",
                           metadata=dict(doc.metadata))
            carry = ""
        if len(doc.page_content) < min_chunk:
            carry = doc.page_content
            continue
        merged_small.append(doc)
    if carry:
        if merged_small:
            merged_small[-1].page_content = f"{merged_small[-1].page_content}\n\n{carry}"
        else:
            merged_small.append(Document(page_content=carry, metadata={}))
    pieces = merged_small

    # ---- 相邻块补重叠（跨块上下文）----
    # RecursiveCharacterTextSplitter 自带 overlap，但按标题合并出来的相邻块之间
    # 没有重叠；这里把上一块的尾部接到下一块开头。判重时同时看前缀与就近包含，
    # 避免同一段文字被叠加两次。
    if chunk_overlap > 0 and len(pieces) > 1:
        for i in range(1, len(pieces)):
            prev = pieces[i - 1].page_content
            tail = prev[-chunk_overlap:]
            cut = tail.find("\n")
            if 0 < cut < len(tail) - 1:
                tail = tail[cut + 1:]
            cur = pieces[i].page_content
            head = tail[:40]
            if not head or cur.startswith(head) or head in cur[: len(tail) + 80]:
                continue
            pieces[i].page_content = f"{tail}\n\n{cur}"

    for i, doc in enumerate(pieces):
        doc.metadata = {**doc.metadata, "chunk_index": i, "n_chunks": len(pieces)}
    return pieces


# --------------------------------------------------------------------------- #
# 3. 模型与向量库
# --------------------------------------------------------------------------- #
def _load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_dir = Path(CFG.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"BGE-M3 目录不存在：{model_dir}")
    log(f"加载 BGE-M3（{model_dir}）… 首次约 20–40 s")
    t0 = time.perf_counter()
    emb = HuggingFaceEmbeddings(
        model_name=str(model_dir),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _ = emb.embed_query("维度探测")          # 触发真正的权重加载
    STATE["model_load_s"] = round(time.perf_counter() - t0, 1)
    log(f"BGE-M3 就绪（{STATE['model_load_s']}s，{len(_)} 维）")
    return emb


def _open_collection():
    import chromadb

    kb = Path(CFG.kb_dir)
    kb.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(kb))
    try:
        col = client.get_collection(CFG.collection)
        log(f"已连接现有集合 '{CFG.collection}'，共 {col.count()} 条 chunk")
    except Exception:
        col = client.create_collection(CFG.collection)
        log(f"集合 '{CFG.collection}' 不存在，已新建空集合")
    return col


def warmup() -> None:
    """后台线程：加载模型 + 连接向量库。"""
    global _EMB, _COL
    try:
        with _LOCK:
            _EMB = _load_embeddings()
            _COL = _open_collection()
        STATE.update(ready=True, warming=False, error=None)
        log("✅ RAG 服务就绪")
    except Exception as exc:                       # noqa: BLE001
        STATE.update(ready=False, warming=False, error=f"{type(exc).__name__}: {exc}")
        log(f"❌ 初始化失败：{exc}")
        traceback.print_exc()


def require_ready():
    if not STATE["ready"]:
        if STATE["warming"]:
            raise ServiceError(503, "模型仍在加载，请稍候（首次启动约 20–40 s）")
        raise ServiceError(503, f"RAG 服务未就绪：{STATE['error']}")


class ServiceError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# 4. 入库 / 检索 / 问答
# --------------------------------------------------------------------------- #
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_record() -> Dict[str, Any]:
    p = Path(CFG.record_file)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_record(record: Dict[str, Any]) -> None:
    p = Path(CFG.record_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


#: 基底关键词：老数据没有 substrate 元数据时，用文本命中兜底
SUBSTRATE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Cu": ("Cu", "铜", "copper"),
    "W": ("W ", "钨", "tungsten"),
    "Oxide": ("SiO2", "氧化硅", "二氧化硅", "oxide", "ILD", "介质"),
    "SiN": ("SiN", "氮化硅", "nitride"),
    "Al": ("Al ", "铝", "aluminum", "aluminium"),
    "Co": ("Co ", "钴", "cobalt"),
    "Ru": ("Ru ", "钌", "ruthenium"),
}

# --------------------------------------------------------------------------- #
# 文献类型：入库时强制标注，检索时按类型调权
# --------------------------------------------------------------------------- #
#: 代号 → 中文标签。代号进 metadata（稳定、跨语言），标签只用于展示。
DOC_TYPES: Dict[str, str] = {
    "formula": "配方",
    "mechanism": "机理",
    "news": "新闻",
}

#: 检索优先级：数字越小越优先。
#: 关键设计——**没有类型标注的老 chunk 排在中间（1），不排最后**：
#: 原库 100 条 chunk 都是旧版入库的，没有任何 doc_type；如果"只查配方/机理"，
#: 它们会被 where 过滤全部剔除，等于把已有知识库废掉。所以用"调权"而不是"硬过滤"。
DOC_TYPE_PRIORITY: Dict[str, int] = {
    "formula": 0,
    "mechanism": 0,
    "news": 2,
    "": 1,          # 未标注（老数据）
}


def _normalize_doc_type(value: Any) -> str:
    """接受代号或中文标签，返回代号；未知值原样返回（按"未标注"对待）。"""
    v = str(value or "").strip()
    if not v:
        return ""
    low = v.lower()
    if low in DOC_TYPES:
        return low
    for code, label in DOC_TYPES.items():
        if v == label:
            return code
    return low


def doc_type_label(value: Any) -> str:
    code = _normalize_doc_type(value)
    return DOC_TYPES.get(code, "未标注")


def _type_priority(meta: Optional[Dict[str, Any]]) -> int:
    code = _normalize_doc_type((meta or {}).get("doc_type", ""))
    return DOC_TYPE_PRIORITY.get(code, 1)


def _text_matches_substrate(text: str, substrate: str) -> bool:
    keys = SUBSTRATE_KEYWORDS.get(substrate)
    if not keys:
        return substrate.lower() in text.lower()
    return any(k.lower() in text.lower() for k in keys)


SUMMARY_PROMPT = """你是 CMP（化学机械抛光）领域的科研助理。请阅读下面的文献，只输出一个 JSON 对象，不要任何解释文字。

JSON 字段：
{{"title": "文献标题",
  "substrates": ["适用基底，如 Cu / W / Oxide / SiN"],
  "mrr_range": "去除速率范围，如 800-2000 Å/min；没有就写 unknown",
  "defects": "缺陷情况与局限（dishing / erosion / scratch / 颗粒残留 等）",
  "key_params": "关键配方参数（磨料种类与浓度、氧化剂、络合剂、pH、压力、转速）",
  "summary": "150 字以内的核心机理与结论"}}

文献正文：
{content}
"""


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """从模型输出里抠出 JSON（容忍 ```json 围栏与前后废话）。"""
    raw = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    candidates = [fence.group(1)] if fence else []
    first, last = raw.find("{"), raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def llm_chat(messages: List[Dict[str, str]], *, temperature: Optional[float] = None,
             max_tokens: int = 1200) -> str:
    """直接打 DeepSeek 的 OpenAI 兼容接口（stdlib urllib，避免额外依赖）。"""
    if not CFG.has_api_key:
        raise ServiceError(400, "未配置 DEEPSEEK_API_KEY（见项目根目录 .env）")
    body = json.dumps({
        "model": CFG.llm_model,
        "messages": messages,
        "temperature": CFG.llm_temperature if temperature is None else temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{CFG.llm_base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {CFG.api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=CFG.llm_timeout_s) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ServiceError(502, f"DeepSeek 返回 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise ServiceError(502, f"连不上 DeepSeek：{exc.reason}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise ServiceError(502, f"DeepSeek 返回空结果：{str(data)[:200]}")
    return (choices[0].get("message") or {}).get("content", "") or ""


def summarize(text: str) -> Tuple[Dict[str, Any], str]:
    """生成结构化摘要（用于"摘要增强入库"）。"""
    clipped = text[:12000]
    raw = llm_chat(
        [{"role": "user", "content": SUMMARY_PROMPT.format(content=clipped)}],
        temperature=0.1, max_tokens=900,
    )
    obj = _extract_json(raw)
    if not obj:
        raw2 = llm_chat(
            [{"role": "user", "content": SUMMARY_PROMPT.format(content=clipped)},
             {"role": "assistant", "content": raw},
             {"role": "user", "content": "上面的输出不是合法 JSON。请只输出 JSON 对象，不要任何其它文字。"}],
            temperature=0.0, max_tokens=900,
        )
        obj = _extract_json(raw2)
    if not obj:
        obj = {"title": "", "substrates": [], "mrr_range": "unknown",
               "defects": "", "key_params": "", "summary": raw[:400]}
    return obj, raw


def ingest_file(
    name: str,
    text: str,
    *,
    mode: str,
    substrate: str = "",
    doc_type: str = "",
    chunk_size: int = CFG.chunk_size,
    chunk_overlap: int = CFG.chunk_overlap,
    record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """单个文件入库。返回状态字典（duplicate / imported / error）。

    ``doc_type`` 是**必填**的文献类型（配方/机理/新闻）：它决定检索时的调权，
    也是"1 万篇新闻里捞技术资料"的前提。服务端强制校验，前端只是友好的入口。
    """
    require_ready()
    doc_type = _normalize_doc_type(doc_type)
    if doc_type not in DOC_TYPES:
        raise ServiceError(
            400,
            "必须为每篇文献指定文献类型：formula（配方）/ mechanism（机理）/ news（新闻）",
        )
    file_hash = _sha256(text)
    record = record if record is not None else load_record()
    if file_hash in record:
        prev = record[file_hash]
        return {
            "name": name, "status": "duplicate", "chunks": 0,
            "reason": f"内容哈希已存在（{prev.get('filename')} @ {prev.get('import_time')}），已拦截",
            "hash": file_hash,
        }

    meta_extra: Dict[str, Any] = {}
    summary_text = ""
    if mode == "summary":
        structured, _raw = summarize(text)
        subs = structured.get("substrates") or []
        if isinstance(subs, str):
            subs = [s.strip() for s in re.split(r"[,，/、\s]+", subs) if s.strip()]
        meta_extra = {
            "doc_title": str(structured.get("title") or ""),
            "substrates": ",".join(str(s) for s in subs),
            "mrr_range": str(structured.get("mrr_range") or ""),
            "defects": str(structured.get("defects") or ""),
            "key_params": str(structured.get("key_params") or ""),
            "summary": str(structured.get("summary") or ""),
            "type": "llm_enhanced",
        }
        summary_text = "\n".join(
            f"{k}：{v}" for k, v in meta_extra.items() if v and k != "type"
        )
        substrate = substrate or (subs[0] if subs else "")
    else:
        meta_extra = {"type": "raw", "doc_title": Path(name).stem}

    chunks = chunk_markdown(text, chunk_size, chunk_overlap)
    if summary_text:
        # 摘要本身也作为一个 chunk 放进库：命中"这个体系用什么配方"这类概述性问题
        chunks.insert(0, Document(page_content="【结构化摘要】\n" + summary_text,
                                  metadata={"h1": "结构化摘要", "chunk_index": 0}))
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["n_chunks"] = len(chunks)
    if not chunks:
        return {"name": name, "status": "error", "chunks": 0,
                "reason": "文件没有可入库的文本内容", "hash": file_hash}

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ids, docs, metas = [], [], []
    for i, doc in enumerate(chunks):
        meta = {
            "source_file": name,
            "import_time": now,
            "file_hash": file_hash,
            "chunk_index": i,
            "n_chunks": len(chunks),
            "substrate": substrate or "",
            "doc_type": doc_type,          # 配方 / 机理 / 新闻（入库时强制标注）
            **{k: v for k, v in meta_extra.items() if v not in (None, "")},
        }
        for h in ("h1", "h2", "h3"):
            if h in doc.metadata and doc.metadata[h]:
                meta[h] = str(doc.metadata[h])[:120]
        ids.append(f"{file_hash[:16]}_{i:04d}")
        docs.append(doc.page_content)
        metas.append(meta)

    vectors = _EMB.embed_documents(docs)
    with _LOCK:
        _COL.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

    record[file_hash] = {
        "filename": name,
        "import_time": now,
        "mode": mode,
        "doc_type": doc_type,
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "substrate": substrate,
    }
    save_record(record)
    return {
        "name": name, "status": "imported", "chunks": len(chunks),
        "hash": file_hash, "substrate": substrate, "doc_type": doc_type,
        "doc_type_label": DOC_TYPES[doc_type],
        "chars": len(text), "avg_chunk": round(sum(len(d) for d in docs) / len(docs)),
        "reason": f"{'摘要增强' if mode == 'summary' else '全文标准'}入库完成"
                  f"（类型：{DOC_TYPES[doc_type]}）",
    }


def retrieve(query: str, *, top_k: int = CFG.top_k, substrate: str = "",
             filter_mode: str = "auto", doc_types: Optional[Sequence[str]] = None,
             prefer_types: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """召回 → 去重 → 基底/类型过滤（metadata 优先，老数据用文本兜底）。

    三种过滤语义要分清：

    * ``substrate`` + ``filter_mode``：精确基底过滤，命中不足时用文本关键词兜底；
    * ``doc_types``：**硬过滤**（只要这些类型），用在用户显式选择时；
    * ``prefer_types``：**软排序**（这些类型优先，其余降权但仍可召回），
      用在"解释配方"——因为老数据没有 doc_type，硬过滤会把已有知识库清空。
    """
    require_ready()
    k = max(1, int(top_k))
    mult = max(1, int(CFG.retrieve_multiplier))
    qv = _EMB.embed_query(query)

    def _query(where: Optional[Dict[str, Any]], n: int):
        with _LOCK:
            return _COL.query(
                query_embeddings=[qv], n_results=n,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )

    def _conds(include_substrate: bool, include_types: bool) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if include_substrate and substrate and filter_mode in ("auto", "metadata"):
            out.append({"substrate": substrate})
        if include_types and doc_types:
            out.append({"doc_type": {"$in": list(doc_types)}})
        return out

    def _where(conds: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    used_where = None
    notes: List[str] = []
    if not substrate or filter_mode not in ("auto", "metadata"):
        notes.append("未启用基底过滤")

    conds = _conds(True, True)
    used_where = _where(conds)
    try:
        res = _query(used_where, k * mult)
        if substrate and filter_mode in ("auto", "metadata"):
            notes.append(f"按 metadata substrate={substrate} 过滤")
        if doc_types:
            notes.append("仅取类型 " + "/".join(doc_type_label(t) for t in doc_types))
    except Exception as exc:                      # noqa: BLE001
        used_where = None
        notes.append(f"metadata 过滤不可用（{type(exc).__name__}），回退全库")
        res = _query(None, k * mult)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    def _make(i, doc, meta, dist) -> Dict[str, Any]:
        meta = meta or {}
        return {
            "id": i, "text": doc, "metadata": meta,
            "doc_type": _normalize_doc_type(meta.get("doc_type", "")),
            "doc_type_label": doc_type_label(meta.get("doc_type", "")),
            "distance": round(float(dist), 4),
            "score": round(1.0 / (1.0 + float(dist)), 4),
        }

    # 文本去重：完全相同的 chunk 只留一条（原文逻辑）
    seen, hits = set(), []
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        norm = re.sub(r"\s+", " ", (doc or "")).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hits.append(_make(i, doc, meta, dist))

    # 老数据没有 substrate 元数据 → 若过滤后不足 k 条，用文本关键词补
    fallback_used = False
    if substrate and filter_mode == "auto" and len(hits) < k:
        res2 = _query(_where(_conds(False, True)), k * mult * 2)
        for i, doc, meta, dist in zip(res2.get("ids", [[]])[0],
                                      res2.get("documents", [[]])[0],
                                      res2.get("metadatas", [[]])[0],
                                      res2.get("distances", [[]])[0]):
            norm = re.sub(r"\s+", " ", (doc or "")).strip()
            if not norm or norm in seen:
                continue
            if not _text_matches_substrate(doc or "", substrate):
                continue
            seen.add(norm)
            fallback_used = True
            hits.append(_make(i, doc, meta, dist))
        if fallback_used:
            notes.append("老数据无 substrate 字段，已用文本关键词兜底")

    # ---- 类型优先：**定向查询 + 全库兜底**，而不是在候选池里重排 ----
    #
    # 为什么不能只做重排：候选池只有 k×mult 条，若库里已经有上万篇新闻，
    # 纯向量检索的 top-8 很可能全是新闻，技术文献压根没进池子，再排序也没用。
    # 所以先发一条**只查配方/机理**的定向查询（新闻物理上进不来），再补一条
    # 全类型查询兜住"未标注"的老数据，最后按 (类型优先级, 距离) 合并排序。
    if prefer_types:
        prefer = {_normalize_doc_type(t) for t in prefer_types}
        prefer_cond = {"doc_type": {"$in": sorted(prefer)}}
        targeted = _query(_where(_conds(True, False) + [prefer_cond]), k * mult)
        fallback = _query(_where(_conds(True, False)), k * mult)

        def _absorb(res: Dict[str, Any]) -> int:
            added = 0
            for i, doc, meta, dist in zip(res.get("ids", [[]])[0],
                                          res.get("documents", [[]])[0],
                                          res.get("metadatas", [[]])[0],
                                          res.get("distances", [[]])[0]):
                norm = re.sub(r"\s+", " ", (doc or "")).strip()
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                hits.append(_make(i, doc, meta, dist))
                added += 1
            return added

        n_targeted = len(targeted.get("ids", [[]])[0])
        _absorb(targeted)
        _absorb(fallback)
        # 排序键：先按类型优先级（配方/机理=0，未标注=1，新闻=2），再按距离
        hits.sort(key=lambda h: (_type_priority(h["metadata"]), h["distance"]))
        prefer_label = "/".join(doc_type_label(t) for t in prefer_types)
        news_cnt = sum(1 for h in hits[:k] if h["doc_type"] == "news")
        notes.append(
            f"定向检索 {prefer_label}（返回 {n_targeted} 条）→ 不足再补全库；"
            f"新闻类降权（当前前 {k} 条里 {news_cnt} 条新闻）"
        )
    else:
        hits.sort(key=lambda h: h["distance"])

    return {
        "hits": hits[:k],
        "candidates": len(docs),
        "dedup_removed": len(docs) - len(hits),
        "substrate": substrate,
        "doc_types": list(doc_types) if doc_types else None,
        "prefer_types": list(prefer_types) if prefer_types else None,
        "filter_note": "；".join(notes),
        "used_where": used_where,
    }


#: 问答 Prompt：显式角色 + 来源标注 + 类型可信度分级
RAG_PROMPT = """你是一名 CMP（化学机械抛光）工艺工程师兼文献研究员，服务对象是产线上的配方工程师。

【工作准则】
1. 只依据下面【参考资料】回答；资料未覆盖的内容必须直接写"知识库未覆盖"，禁止臆造机理、公式、数值。
2. **每条关键结论后面用「[来源：文件名]」标注出处**；一句话融合多份资料时，把多个来源都列出。
3. 区分资料类型与可信度：【配方】= 实测配方与工艺窗口，可直接引用；【机理】= 机理与模型，用于解释"为什么"；
   【新闻】= 行业动态/市场信息，只能作背景，**不得当作工艺结论**；【未标注】= 早期入库的文献，需自行判断。
4. 数值必须带单位与条件（wt%、Å/min、psi、rpm、pH）；不同资料的数值冲突时，并列给出并说明差异。
5. 回答用中文，术语严谨；先给结论，再给依据，最后给不确定之处。

【参考资料】
{context}

【问题】{question}
"""

#: 配方解释 Prompt：优先机理/配方类型，严格标源，200 字硬约束
EXPLAIN_PROMPT = """你是一名 CMP 工艺专家，正在为产线评审一条由贝叶斯优化推荐的配方。

【工作准则】
1. **优先采用文献类型为【机理】与【配方】的片段**；只有当这两类资料缺失时，才可参考【新闻】，且必须写明"新闻类来源，仅供参考"。
2. 解释要落到机制上：氧化剂如何生成软反应层、络合剂如何溶解产物、磨料与压力如何提供机械去除、添加剂如何影响 Dishing/Ra。
3. **每一处引用都要标注来源文件名**，格式「[来源：文件名]」；不得把推断说成文献结论。
4. 明确指出风险与资料未覆盖之处（例如缺少该 pH 窗口的实测数据）。
5. 全文不超过 200 字，不要罗列无关配方细节。

【配方】{recipe}
【预测性能】{metrics}

【资料】
{context}
"""


def _format_context(hits: List[Dict[str, Any]], limit: int = 6000) -> str:
    """把召回片段拼成上下文，**每段都带来源与文献类型**。

    类型写进上下文是必须的：模型才能执行"新闻只作背景、配方/机理可作结论"
    这条规则；只靠 system prompt 而不给数据本身贴标签，模型无从判断。
    """
    parts, total = [], 0
    for h in hits:
        src = h["metadata"].get("source_file", "未知")
        label = h.get("doc_type_label") or doc_type_label(h["metadata"].get("doc_type", ""))
        block = f"【来源：{src} ｜ 类型：{label}】\n{h['text']}"
        if total + len(block) > limit:
            block = block[: max(0, limit - total)]
        parts.append(block)
        total += len(block)
        if total >= limit:
            break
    return "\n\n".join(parts)


def answer(question: str, *, top_k: int = CFG.top_k, substrate: str = "",
           filter_mode: str = "auto",
           doc_types: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    r = retrieve(question, top_k=top_k, substrate=substrate, filter_mode=filter_mode,
                 doc_types=doc_types)
    if not r["hits"]:
        return {**r, "answer": "知识库中没有检索到相关内容，无法回答。请先入库相关文献。"}
    ctx = _format_context(r["hits"])
    text = llm_chat([{"role": "user", "content": RAG_PROMPT.format(
        context=ctx, question=question)}], max_tokens=1400)
    return {**r, "answer": text}


def explain(recipe: Dict[str, Any], metrics: Dict[str, Any],
            *, substrate: str = "", top_k: int = CFG.top_k,
            prefer_types: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """解释配方：**优先机理与配方类文献**，新闻类降权但仍可兜底。"""
    recipe_txt = "；".join(f"{k}={v}" for k, v in recipe.items())
    metrics_txt = "；".join(f"{k}={v}" for k, v in (metrics or {}).items()) or "未提供"
    query = f"CMP 配方 {recipe_txt} 机理 去除速率 缺陷 工艺窗口"
    prefer = list(prefer_types) if prefer_types else ["formula", "mechanism"]
    r = retrieve(query, top_k=top_k, substrate=substrate, prefer_types=prefer)
    if not r["hits"]:
        return {**r, "answer": "知识库中没有检索到与该配方相关的资料。"}
    ctx = _format_context(r["hits"], limit=5000)
    text = llm_chat([{"role": "user", "content": EXPLAIN_PROMPT.format(
        recipe=recipe_txt, metrics=metrics_txt, context=ctx)}], max_tokens=600)
    return {**r, "answer": _enforce_length(text.strip(), 200)}


def _enforce_length(text: str, limit: int) -> str:
    """把解释压到 ``limit`` 字以内：优先在句末/分号处收尾，避免截断半个词。"""
    if len(text) <= limit + 10:
        return text
    window = text[: limit + 10]
    for mark in ("。", "；", "！", "?", "！"):
        cut = window.rfind(mark, 0, limit + 1)
        if cut >= limit * 0.6:
            return text[: cut + 1]
    return text[:limit] + "…"


def stats() -> Dict[str, Any]:
    require_ready()
    record = load_record()
    with _LOCK:
        count = _COL.count()

    def _count_where(where: Dict[str, Any]) -> int:
        try:
            with _LOCK:
                return len(_COL.get(where=where, include=[])["ids"])
        except Exception:                          # noqa: BLE001
            return 0

    type_counts = {code: _count_where({"doc_type": code}) for code in DOC_TYPES}
    untyped = max(0, count - sum(type_counts.values()))
    return {
        "chunks": count, "files": len(record),
        "kb_dir": CFG.kb_dir, "collection": CFG.collection,
        "chunk_size": CFG.chunk_size, "chunk_overlap": CFG.chunk_overlap,
        "top_k": CFG.top_k, "model_dir": CFG.model_dir,
        "model_load_s": STATE["model_load_s"],
        "record_file": CFG.record_file,
        "doc_types": dict(DOC_TYPES),
        "doc_type_counts": {**type_counts, "untyped": untyped},
    }


# --------------------------------------------------------------------------- #
# 5. HTTP 层
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "CMPRAG/1.0"

    def log_message(self, fmt, *args):          # 安静一点：只打错误
        if str(args[1] if len(args) > 1 else "") not in ("200", "204"):
            log(f"HTTP {fmt % args}")

    # -- 工具 --
    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:                    # noqa: BLE001
            raise ServiceError(400, f"请求体不是合法 JSON：{exc}") from exc

    def do_OPTIONS(self):                       # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST")
        self.end_headers()

    def do_GET(self):                           # noqa: N802
        path = self.path.split("?")[0]
        try:
            if path == "/health":
                self._send(200, {
                    "ok": bool(STATE["ready"]), "warming": STATE["warming"],
                    "error": STATE["error"], "model_load_s": STATE["model_load_s"],
                    "started_at": STATE["started_at"], "pid": os.getpid(),
                    "chunk_size": CFG.chunk_size, "chunk_overlap": CFG.chunk_overlap,
                    "kb_dir": CFG.kb_dir,
                    "build": BUILD, "features": FEATURES,
                    "doc_types": dict(DOC_TYPES),
                })
            elif path == "/stats":
                self._send(200, stats())
            else:
                self._send(404, {"error": f"未知路径 {path}"})
        except ServiceError as exc:
            self._send(exc.code, {"error": exc.message})
        except Exception as exc:                # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):                          # noqa: N802
        path = self.path.split("?")[0]
        STATE["requests"] += 1
        try:
            if path == "/search":
                b = self._body()
                self._send(200, retrieve(
                    b.get("query", ""), top_k=b.get("top_k", CFG.top_k),
                    substrate=b.get("substrate", ""),
                    filter_mode=b.get("filter_mode", "auto"),
                    doc_types=b.get("doc_types") or None,
                    prefer_types=b.get("prefer_types") or None))
            elif path == "/ask":
                b = self._body()
                q = (b.get("question") or "").strip()
                if not q:
                    raise ServiceError(400, "question 不能为空")
                self._send(200, answer(q, top_k=b.get("top_k", CFG.top_k),
                                       substrate=b.get("substrate", ""),
                                       filter_mode=b.get("filter_mode", "auto"),
                                       doc_types=b.get("doc_types") or None))
            elif path == "/explain":
                b = self._body()
                self._send(200, explain(b.get("recipe") or {}, b.get("metrics") or {},
                                        substrate=b.get("substrate", ""),
                                        top_k=b.get("top_k", CFG.top_k),
                                        prefer_types=b.get("prefer_types") or None))
            elif path == "/ingest":
                b = self._body()
                mode = b.get("mode", "raw")
                if mode not in ("raw", "summary"):
                    raise ServiceError(400, "mode 只能是 raw 或 summary")
                record = load_record()
                results, logs = [], []
                for f in b.get("files") or []:
                    name = f.get("name") or "unknown.md"
                    text = f.get("text") or ""
                    logs.append(f"—— {name}（{len(text)} 字符）")
                    try:
                        r = ingest_file(
                            name, text, mode=mode,
                            substrate=f.get("substrate", "") or b.get("substrate", ""),
                            doc_type=f.get("doc_type", "") or b.get("doc_type", ""),
                            chunk_size=int(b.get("chunk_size") or CFG.chunk_size),
                            chunk_overlap=int(b.get("chunk_overlap") or CFG.chunk_overlap),
                            record=record,
                        )
                    except ServiceError:
                        raise
                    except Exception as exc:        # noqa: BLE001
                        traceback.print_exc()
                        r = {"name": name, "status": "error", "chunks": 0,
                             "reason": f"{type(exc).__name__}: {exc}"}
                    results.append(r)
                    logs.append(f"   → {r['status']}：{r.get('reason','')}")
                self._send(200, {"results": results, "log": logs,
                                 "record_count": len(record),
                                 "chunks_total": (stats()["chunks"] if STATE["ready"] else 0)})
            elif path == "/record":
                b = self._body()
                action = b.get("action", "list")
                record = load_record()
                if action == "list":
                    items = []
                    for h, v in record.items():
                        v = v if isinstance(v, dict) else {}
                        items.append({
                            "hash": h,
                            "filename": v.get("filename") or "",
                            "import_time": v.get("import_time") or "",
                            "mode": v.get("mode") or "",
                            "chunks": v.get("chunks"),
                            "substrate": v.get("substrate") or "",
                            "doc_type": v.get("doc_type") or "",
                            "doc_type_label": doc_type_label(v.get("doc_type", "")),
                        })
                    items.sort(key=lambda x: x.get("import_time", ""), reverse=True)
                    self._send(200, {"count": len(items), "items": items})
                elif action == "clear":
                    save_record({})
                    self._send(200, {"ok": True, "note": "已清空 kb_record.json（向量库未动）"})
                else:
                    raise ServiceError(400, f"未知 action {action}")
            elif path == "/shutdown":
                self._send(200, {"ok": True, "note": "服务即将退出"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send(404, {"error": f"未知路径 {path}"})
        except ServiceError as exc:
            self._send(exc.code, {"error": exc.message})
        except Exception as exc:                # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """禁止端口复用，避免"两个服务同时监听同一端口"。

    Windows 上 ``SO_REUSEADDR`` 的行为是"允许劫持"：默认的
    ``ThreadingHTTPServer.allow_reuse_address = 1`` 会让第二个实例**绑上同一端口
    却互不报错**，于是请求随机落到新旧两个进程，日志里 PID 对不上，还各自占着
    2 GB 内存。实测就踩到了（8770 上有 5384 和 21252 两个进程同时在听）。

    这里改成独占绑定：端口被占时第二个实例直接启动失败并给出可执行提示。
    """

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                )
            except OSError:
                pass
        super().server_bind()


def serve(port: int = CFG.service_port) -> None:
    try:
        httpd = _ExclusiveHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        code = getattr(exc, "winerror", None) or exc.errno
        log(f"❌ 端口 {port} 已被占用，启动中止（WinError/errno={code}）")
        log(f"   说明：本服务只允许单实例运行，避免两个进程抢同一端口、")
        log(f"        请求随机落到旧进程（旧进程可能连着另一套知识库）。")
        log(f"   处理：在另一个窗口执行 taskkill /F /PID <占用PID>，"
            f"或直接双击 启动RAG服务.bat（它会自动清理）。")
        raise SystemExit(2)
    log(f"RAG 服务监听 http://127.0.0.1:{port}（PID {os.getpid()}）")
    log(f"配置：{CFG.describe_paths()}")
    threading.Thread(target=warmup, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("收到 Ctrl+C，正在退出…")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# 6. 自检（不需要模型/向量库，可在任何解释器跑）
# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("=" * 78)
    print("RAG 切片自检：大 chunk 策略 + 配方参数完整性")
    print("=" * 78)
    sample = (
        "# CMP 铜抛光液配方研究\n\n"
        "## 1. 引言\n\n" + "化学机械抛光依靠化学腐蚀与机械磨削的协同作用。" * 40 + "\n\n"
        "## 2. 实验配方\n\n"
        "- 磨料：胶体二氧化硅 5.0 wt%，粒径 50 nm\n"
        "- 氧化剂：H2O2 3.0 wt%\n"
        "- 络合剂：甘氨酸 1.2 wt%\n"
        "- 表面活性剂：PEG 0.05 wt%\n"
        "- pH 调节剂：KOH，pH 10.2\n"
        "- 压力：2.0 psi；转速：93/87 rpm；流量：200 mL/min\n\n"
        "## 3. 结果\n\n" + "MRR 达到 1800 Å/min，Dishing 85 Å。" * 30 + "\n\n"
        "## 4. 结论\n\n" + "该配方在碱性窗口内稳定。" * 30
    )
    for size, overlap in ((1800, 300), (1200, 200), (600, 100)):
        chunks = chunk_markdown(sample, size, overlap)
        lens = [len(c.page_content) for c in chunks]
        print(f"chunk_size={size:<5} overlap={overlap:<4} -> {len(chunks)} 块，"
              f"长度 {lens}，最大 {max(lens)}（上限 {size}）")
        assert chunks, "不应产生空切片"
        # 上限 = chunk_size + 重叠（重叠是刻意加在块首的）
        assert max(lens) <= size + overlap + 20, "单块长度明显超过 chunk_size+overlap"
        assert min(lens) >= 120, f"出现碎片块：{lens}"

    chunks = chunk_markdown(sample, 1800, 300)
    # 配方块必须完整落在同一块里
    hit = [c for c in chunks if "5.0 wt%" in c.page_content]
    assert hit, "配方参数丢失"
    blob = hit[0].page_content
    for token in ("5.0 wt%", "3.0 wt%", "1.2 wt%", "pH 10.2", "2.0 psi", "93/87 rpm"):
        assert token in blob, f"配方参数 {token} 被切碎了"
    print(f"配方完整性：5.0/3.0/1.2 wt%、pH 10.2、2.0 psi、93/87 rpm 同处一块 ✓")
    print(f"标题锚定  ：{ {k: v for k, v in hit[0].metadata.items() if k in ('h1','h2')} }")

    # 重叠必须生效（方向：上一块的尾部要出现在下一块里）
    long_text = "# T\n\n" + ("句子。" * 2000)
    c = chunk_markdown(long_text, 1200, 300)
    assert len(c) >= 2
    overlap_found = c[0].page_content[-60:] in c[1].page_content
    lens = [len(x.page_content) for x in c]
    print(f"跨块重叠  ：{overlap_found}（{len(c)} 块，长度 {lens[:5]}…）")
    assert overlap_found, "相邻块之间没有重叠"
    assert min(lens[:-1]) >= 120, f"出现了过小的 chunk：{lens}"

    # JSON 抽取的健壮性
    for raw in ('```json\n{"a": 1}\n```', '前缀 {"a": 2} 后缀', '{"a": 3}'):
        assert _extract_json(raw)["a"] in (1, 2, 3)
    assert _extract_json("没有 json") is None
    print("JSON 抽取  ：围栏/前后废话/纯对象 均通过 ✓")

    # ---- 文献类型：代号/中文互转 + 优先级（新闻必须排在最后）----
    assert _normalize_doc_type("配方") == "formula"
    assert _normalize_doc_type("NEWS") == "news"
    assert _normalize_doc_type(" 机理 ") == "mechanism"
    assert _normalize_doc_type("") == ""
    assert doc_type_label("mechanism") == "机理"
    assert doc_type_label("") == "未标注"
    assert _type_priority({"doc_type": "formula"}) == 0
    assert _type_priority({"doc_type": "mechanism"}) == 0
    assert _type_priority({"doc_type": "news"}) == 2
    assert _type_priority({}) == 1, "老数据没有 doc_type，应居中而不是垫底"
    print("文献类型  ：配方/机理=0 优先，未标注=1，新闻=2 ✓")

    # ---- Prompt 必须包含角色与类型规则（防止后续改动把规则删掉）----
    assert "工艺工程师" in RAG_PROMPT, "问答 Prompt 缺少角色设定"
    assert "[来源：" in RAG_PROMPT, "问答 Prompt 缺少来源标注要求"
    assert "新闻" in RAG_PROMPT and "不得当作工艺结论" in RAG_PROMPT
    assert "优先采用" in EXPLAIN_PROMPT, "解释 Prompt 缺少类型优先级规则"
    assert "机理" in EXPLAIN_PROMPT and "配方" in EXPLAIN_PROMPT
    assert "200 字" in EXPLAIN_PROMPT and "[来源：" in EXPLAIN_PROMPT
    print("Prompt    ：角色 + 类型优先级 + 强制标源 均已就位 ✓")

    # ---- 上下文必须给每段贴类型标签，否则模型无法执行上面的规则 ----
    fake_hits = [
        {"text": "磨料 5 wt%", "doc_type_label": "配方",
         "metadata": {"source_file": "a.md", "doc_type": "formula"}},
        {"text": "行业新闻……", "doc_type_label": "新闻",
         "metadata": {"source_file": "b.md", "doc_type": "news"}},
    ]
    ctx = _format_context(fake_hits)
    assert "类型：配方" in ctx and "类型：新闻" in ctx, ctx
    print("上下文    ：每段带「类型」标签 ✓")

    print("\n全部通过：切片大小可配、配方参数不被切碎、跨块重叠生效、"
          "文献类型优先级与 Prompt 角色就位")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    port = CFG.service_port
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    serve(port)
