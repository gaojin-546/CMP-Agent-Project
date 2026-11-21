"""网页端（Python 3.15）访问 RAG 微服务的客户端。

**架构**：网页不 import chromadb / sentence-transformers（3.15 上装不上，
且沙箱不允许写 E:\\private_rag_cmp），只通过 HTTP 调
``src/rag_service.py``。那个服务由用户双击 ``启动RAG服务.bat`` 在 Python 3.11
下启动，拥有对原知识库的正常读写权限。

这个模块的职责就三件：
1. 把请求发出去（stdin 之外的唯一 I/O 面）；
2. 把"服务没开 / 正在加载 / 报错"翻译成工程师看得懂的中文提示；
3. 超时控制——BGE-M3 在 CPU 上首次查询会慢，不能沿用默认 30 s。
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import RAG_CONFIG, RAGConfig

__all__ = [
    "RagServiceError", "RagHealth", "RagClient", "client",
    "SERVICE_HINT",
]

SERVICE_HINT = (
    "RAG 服务未启动。请先双击项目根目录的 **启动RAG服务.bat**"
    "（它会在 Python 3.11 下加载本地 BGE-M3 并连接原知识库），"
    "等窗口出现「✅ RAG 服务就绪」后再回来操作。"
)


class RagServiceError(RuntimeError):
    """RAG 服务调用失败（网络、HTTP 状态码或服务端业务错误）。"""

    def __init__(self, message: str, *, hint: str = "", status: Optional[int] = None,
                 detail: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.status = status
        self.detail = detail

    def user_message(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(self.hint)
        if self.detail:
            parts.append(f"底层信息：{self.detail[:300]}")
        return "\n\n".join(parts)


@dataclass
class RagHealth:
    ok: bool = False
    warming: bool = False
    error: Optional[str] = None
    model_load_s: Optional[float] = None
    pid: Optional[int] = None
    chunk_size: int = 0
    chunk_overlap: int = 0
    kb_dir: str = ""
    build: str = ""
    features: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def supports(self, feature: str) -> bool:
        """服务端是否支持某项能力（旧进程没有 ``features``，一律返回 False）。"""
        return feature in self.features

    @property
    def label(self) -> str:
        if self.ok:
            return "已连接"
        if self.warming:
            return "模型加载中"
        return "未连接"

    def detail(self) -> str:
        if self.ok:
            return (f"服务就绪（构建 {self.build or '旧版'}，"
                    f"模型加载 {self.model_load_s or '?'} s，PID {self.pid}）")
        if self.warming:
            return "服务已响应，BGE-M3 正在加载（首次约 20–40 s）"
        return self.error or "服务无响应"


class RagClient:
    """RAG 服务的薄客户端（stdlib urllib，零额外依赖）。"""

    def __init__(self, cfg: Optional[RAGConfig] = None):
        self.cfg = cfg or RAG_CONFIG

    # ------------------------------------------------------------------ #
    # 传输层
    # ------------------------------------------------------------------ #
    @property
    def base_url(self) -> str:
        return self.cfg.service_url.rstrip("/")

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]],
                 timeout: float) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            message = f"RAG 服务返回 HTTP {exc.code}"
            try:
                body = json.loads(detail)
                if isinstance(body, dict) and body.get("error"):
                    message = f"{message}：{body['error']}"
                    if exc.code == 503:
                        return {"__warming__": True, "error": body["error"]}
            except json.JSONDecodeError:
                pass
            raise RagServiceError(message, status=exc.code, detail=detail) from exc
        except urllib.error.URLError as exc:
            raise RagServiceError(
                "连不上 RAG 服务", hint=SERVICE_HINT,
                detail=f"{url}：{exc.reason}",
            ) from exc
        except socket.timeout as exc:
            raise RagServiceError(
                f"RAG 服务响应超时（>{timeout:.0f}s）",
                hint="首次查询要加载模型；若持续超时，请检查服务端窗口是否报错。",
                detail=str(exc),
            ) from exc
        except json.JSONDecodeError as exc:
            raise RagServiceError("RAG 服务返回了非 JSON 内容",
                                  detail=str(exc)) from exc

    def _get(self, path: str, timeout: float) -> Dict[str, Any]:
        return self._request("GET", path, None, timeout)

    def _post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        return self._request("POST", path, payload, timeout)

    # ------------------------------------------------------------------ #
    # 业务接口
    # ------------------------------------------------------------------ #
    def health(self, timeout: Optional[float] = None) -> RagHealth:
        """健康检查：从不抛异常，返回可展示的状态对象。"""
        try:
            raw = self._get("/health", timeout or self.cfg.health_timeout_s)
        except RagServiceError as exc:
            return RagHealth(ok=False, error=exc.message, raw={"detail": exc.detail})
        return RagHealth(
            ok=bool(raw.get("ok")),
            warming=bool(raw.get("warming")),
            error=raw.get("error"),
            model_load_s=raw.get("model_load_s"),
            pid=raw.get("pid"),
            chunk_size=int(raw.get("chunk_size") or 0),
            chunk_overlap=int(raw.get("chunk_overlap") or 0),
            kb_dir=str(raw.get("kb_dir") or ""),
            build=str(raw.get("build") or ""),
            features=list(raw.get("features") or []),
            raw=raw,
        )

    def stats(self) -> Dict[str, Any]:
        return self._get("/stats", 30)

    def search(self, query: str, *, top_k: Optional[int] = None,
               substrate: str = "", filter_mode: str = "auto",
               doc_types: Optional[List[str]] = None,
               prefer_types: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._post("/search", {
            "query": query,
            "top_k": top_k or self.cfg.top_k,
            "substrate": substrate,
            "filter_mode": filter_mode,
            "doc_types": doc_types,
            "prefer_types": prefer_types,
        }, 180)

    def ask(self, question: str, *, top_k: Optional[int] = None,
            substrate: str = "", filter_mode: str = "auto",
            doc_types: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._post("/ask", {
            "question": question,
            "top_k": top_k or self.cfg.top_k,
            "substrate": substrate,
            "filter_mode": filter_mode,
            "doc_types": doc_types,
        }, 240)

    def explain(self, recipe: Dict[str, Any], metrics: Dict[str, Any],
                *, substrate: str = "", top_k: Optional[int] = None,
                prefer_types: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._post("/explain", {
            "recipe": recipe,
            "metrics": metrics,
            "substrate": substrate,
            "top_k": top_k or self.cfg.top_k,
            "prefer_types": prefer_types,
        }, 240)

    def ingest(self, files: List[Dict[str, str]], *, mode: str = "raw",
               substrate: str = "", chunk_size: Optional[int] = None,
               chunk_overlap: Optional[int] = None,
               doc_type: str = "") -> Dict[str, Any]:
        """入库。``doc_type`` 必填（formula/mechanism/news），服务端会强制校验。"""
        payload_files = []
        for f in files:
            item = dict(f)
            item.setdefault("doc_type", doc_type)
            payload_files.append(item)
        return self._post("/ingest", {
            "files": payload_files,
            "mode": mode,
            "substrate": substrate,
            "doc_type": doc_type,
            "chunk_size": chunk_size or self.cfg.chunk_size,
            "chunk_overlap": chunk_overlap if chunk_overlap is not None
            else self.cfg.chunk_overlap,
        }, 900)

    def record(self, action: str = "list") -> Dict[str, Any]:
        return self._post("/record", {"action": action}, 60)

    def shutdown(self) -> Dict[str, Any]:
        return self._post("/shutdown", {}, 10)


#: 进程级单例，界面直接用 ``from src.rag_bridge import client``
client = RagClient()
