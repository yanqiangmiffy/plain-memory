"""AML 外部契约；成功仅在完整写入后返回，错误不回显私密输入。"""

import functools
import secrets
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from starlette.responses import JSONResponse

from plain_memory import __version__, feedback, timeparse
from plain_memory.config import Settings
from plain_memory.embedding import EmbeddingError, HTTPEmbedder, normalize_vectors
from plain_memory.enrich import enrich
from plain_memory.journal import Journal, status_of
from plain_memory.llm import LLM, Reranker
from plain_memory.retrieval import IndexCache, _speaker_prefixed, format_time
from plain_memory.searching import run_search
from plain_memory.store import RequestConflict, Store, StoreMismatch, payload_hash
from plain_memory.text import chunks, lexical_terms

Identifier = Annotated[StrictStr, Field(min_length=1)]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: StrictStr
    timestamp: Annotated[StrictInt, Field(ge=-62135596800000, le=253402300799999)] | None = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: Identifier
    user_id: Identifier
    session_id: Identifier
    messages: Annotated[list[Message], Field(min_length=1)]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[StrictStr, Field(min_length=1)]
    user_id: Identifier
    top_k: Annotated[StrictInt, Field(ge=1)]
    options: list[StrictStr] | None = None


class BodyLimit:
    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        parts, size = [], 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body = event.get("body", b"")
            size += len(body)
            if size > self.maximum:
                return await JSONResponse(
                    {"detail": "Request exceeds the declared body limit."}, 413
                )(scope, receive, send)
            parts.append(body)
            if not event.get("more_body", False):
                break
        consumed = False

        async def buffered_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)


def _printable(value: str, limit: int) -> str:
    """请求方可控的值写日志前去掉控制字符和空白，防止伪造日志行。"""
    return "".join(c if c.isprintable() and c != " " else "?" for c in value[:limit])


class RequestLog:
    """每个请求一行写 stderr：只含 method、路径(无 query)、状态码、耗时、来源 IP。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started, status = time.perf_counter(), None

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        except Exception:
            status = status or 500
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            forwarded = next((v for k, v in scope["headers"] if k == b"cf-connecting-ip"), None)
            client = (
                forwarded.decode("latin-1") if forwarded else (scope.get("client") or ("-",))[0]
            )
            print(
                time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                _printable(scope["method"], 8),
                _printable(scope["path"], 200),
                status if status is not None else "-",
                f"{elapsed_ms:.1f}ms",
                _printable(client, 64),
                file=sys.stderr,
                flush=True,
            )


@contextmanager
def capacity(semaphore):
    if not semaphore.acquire(blocking=False):
        raise HTTPException(429, "Service concurrency limit reached.", headers={"Retry-After": "1"})
    try:
        yield
    finally:
        semaphore.release()


def prepare_units(request, settings: Settings, llm, deadline: float) -> tuple:
    """Build stored units and the texts to embed (message units, then fact units)."""
    cfg = settings.retrieval
    header = next(
        (format_time(m.timestamp) for m in request.messages if m.timestamp is not None), None
    )
    units, embed_texts, previous = [], [], ""
    for message_index, message in enumerate(request.messages):
        line = _speaker_prefixed({"role": message.role, "content": message.content})
        attempt = feedback.parse(message.content, message.role) if cfg.feedback_units else None
        if attempt is not None:  # a graded earlier answer, indexed by its question
            units.append(
                {
                    "message_index": message_index,
                    "chunk_index": 0,
                    "role": message.role,
                    "timestamp": message.timestamp,
                    "content": attempt.render(),
                    "kind": "fb",
                    "note": attempt.outcome,
                    "terms": lexical_terms(f"{attempt.question} {attempt.answer}"),
                }
            )
            embed_texts.append(f"Question: {attempt.question}")
            previous = line[:240]
            continue
        rule_note = None
        if cfg.rule_notes and message.timestamp is not None:
            found = timeparse.relative_notes(
                message.content, timeparse.moment_of(message.timestamp)
            )
            rule_note = "; ".join(found) if found else None
        for chunk_index, text in enumerate(
            chunks(message.content, settings.chunk_tokens, settings.overlap_tokens)
        ):
            units.append(
                {
                    "message_index": message_index,
                    "chunk_index": chunk_index,
                    "role": message.role,
                    "timestamp": message.timestamp,
                    "content": text,
                    "kind": "msg",
                    "note": rule_note if chunk_index == 0 else None,
                }
            )
            context = previous if cfg.context_embedding and chunk_index == 0 else ""
            when = format_time(message.timestamp) or ""
            hint = f"({rule_note})" if rule_note and chunk_index == 0 else ""
            embed_texts.append("\n".join(x for x in (when, context, text, hint) if x))
        previous = line[:240]
    if cfg.add_notes and llm is not None and units:
        lines = [
            _speaker_prefixed({"role": m.role, "content": m.content[:2000]})
            for m in request.messages
        ]
        notes, facts = enrich(
            lines,
            header or "unknown date",
            llm,
            time.monotonic() + (deadline - time.monotonic()) * 0.7,
            settings.llm.group_tokens,
        )
        by_message = {u["message_index"]: u for u in units if u["chunk_index"] == 0}
        for index, note in notes.items():
            if index in by_message and not by_message[index]["note"]:
                by_message[index]["note"] = note
        if cfg.index_facts:
            for number, (index, text) in enumerate(facts):
                if index in by_message:
                    anchor = by_message[index]
                    units.append(
                        {
                            "message_index": index,
                            "chunk_index": -(number + 1),
                            "role": anchor["role"],
                            "timestamp": anchor["timestamp"],
                            "content": text,
                            "kind": "fact",
                            "note": None,
                        }
                    )
                    embed_texts.append(text)
    for unit, text in zip(units, embed_texts):
        unit["terms"] = lexical_terms(f"{text} {unit['note'] or ''}")
    return units, embed_texts


def create_app(settings: Settings, *, embedder=None, llm=None, reranker=None) -> FastAPI:
    token = settings.api_token.get_secret_value()
    if len(token.encode()) < 32 or token == "GENERATE_LOCALLY":
        raise ValueError("Set a random api_token (at least 32 bytes) in the private local config.")
    encoder = embedder if embedder is not None else HTTPEmbedder(settings.embedding)
    try:
        store = Store(settings)
    except Exception:
        encoder.close()
        raise
    if llm is None and settings.llm is not None:
        llm = LLM(settings.llm)
    if reranker is None and settings.rerank is not None and settings.retrieval.rerank:
        reranker = Reranker(settings.rerank)
    cache = IndexCache(settings.index_cache_users, settings.index_cache_units)
    journal = (
        Journal(settings.request_journal, settings.journal_max_bytes)
        if settings.request_journal is not None
        else None
    )
    add_slots = threading.BoundedSemaphore(settings.add_concurrency)
    search_slots = threading.BoundedSemaphore(settings.search_concurrency)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            encoder.close()
            for extra in (llm, reranker):
                if extra is not None:
                    extra.close()

    app = FastAPI(
        title="PlainMemory",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.add_middleware(BodyLimit, maximum=settings.max_body_bytes)
    if settings.request_log:
        app.add_middleware(RequestLog)  # 最后添加=最外层，也能记录 BodyLimit 的 413

    def authorized(authorization: str | None = Header(default=None)):
        expected = f"Bearer {token}".encode()
        if authorization is None or not secrets.compare_digest(authorization.encode(), expected):
            raise HTTPException(
                401, "Invalid or missing Bearer credential.", headers={"WWW-Authenticate": "Bearer"}
            )

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # 默认 FastAPI 错误含 content 等输入；只输出字段位置与错误类型。
        return JSONResponse(
            {"detail": [{"loc": error["loc"], "type": error["type"]} for error in exc.errors()]},
            status_code=422,
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_failure(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(RequestConflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(StoreMismatch)
    async def incompatible(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(sqlite3.Error)
    async def storage_failure(request, exc):
        return JSONResponse({"detail": "Storage temporarily unavailable."}, status_code=503)

    def health():
        store.health()
        return {"status": "ok", "version": __version__}

    for path in ("/", "/health", "/add", "/search"):
        app.add_api_route(path, health, methods=["GET"])

    def journaled(endpoint: str):
        """Record every call (success or failure) of a handler in the optional journal."""

        def wrap(handler):
            if journal is None:
                return handler

            @functools.wraps(handler)
            def inner(request):
                started = time.perf_counter()
                body = request.model_dump()
                try:
                    result = handler(request)
                except Exception as exc:
                    detail = getattr(exc, "detail", None) or str(exc)
                    journal.write(endpoint, started, status_of(exc), body, None, detail)
                    raise
                journal.write(endpoint, started, 200, body, result)
                return result

            return inner

        return wrap

    @app.post("/add", dependencies=[Depends(authorized)])
    @journaled("add")
    def add(request: AddRequest):
        payload = request.model_dump()
        digest = payload_hash(payload)
        response = {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }
        # 已提交的重试不需要占用 embedding 名额。
        if store.completed(payload, digest):
            return response
        with capacity(add_slots):
            deadline = time.monotonic() + settings.add_timeout_seconds
            if store.completed(payload, digest):
                return response
            prepared, embed_texts = prepare_units(request, settings, llm, deadline)
            vectors = None
            if prepared:
                vectors = normalize_vectors(
                    encoder.embed(embed_texts, deadline),
                    len(prepared),
                    settings.embedding.dimensions,
                )
            if time.monotonic() >= deadline:
                raise EmbeddingError("Add exceeded the service deadline; nothing was written.")
            store.commit(payload, digest, prepared, vectors)
        return response

    @app.post("/search", dependencies=[Depends(authorized)])
    @journaled("search")
    def search(request: SearchRequest):
        if not request.query.strip():
            raise HTTPException(422, "query must contain non-whitespace text.")
        with capacity(search_slots):
            deadline = time.monotonic() + settings.search_timeout_seconds
            index = cache.get(store, request.user_id)
            if index is None:
                return {"data": []}
            data = run_search(index, request, settings, encoder, llm, reranker, deadline)
            return {"data": data}

    return app
