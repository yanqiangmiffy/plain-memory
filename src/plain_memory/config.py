"""私密配置独立于源码；相对数据路径基于配置文件所在目录。"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["doubao_multimodal", "openai_compatible"]
    base_url: str
    model: str = Field(min_length=1)
    api_key: SecretStr
    dimensions: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def check_base_url(self):
        normalized = self.base_url.rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include a query or fragment")
        if parsed.path.endswith(("/embeddings", "/embeddings/multimodal")):
            raise ValueError("base_url must not include the embedding endpoint path")
        self.base_url = normalized
        return self


class LLMConfig(BaseModel):
    """gpt-4o-mini via any OpenAI-compatible chat endpoint (open-source track requirement)."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str = "openai/gpt-4o-mini"
    api_key: SecretStr
    concurrency: int = Field(default=8, ge=1, le=64)
    timeout_seconds: float = Field(default=60, gt=0)
    group_tokens: int = Field(default=3000, ge=200)
    proxy: str | None = None  # e.g. an HTTP(S) egress proxy when the endpoint is geo-blocked


class RerankConfig(BaseModel):
    """DashScope text-rerank (the leaderboard does not restrict rerankers)."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str = "qwen3-rerank"
    api_key: SecretStr
    timeout_seconds: float = Field(default=30, gt=0)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_before: int = Field(default=2, ge=0)
    window_after: int = Field(default=2, ge=0)
    max_block_units: int = Field(default=12, ge=1)
    candidate_units: int = Field(default=60, ge=1)
    max_blocks: int = Field(default=100, ge=1, le=100)
    context_tokens: int = Field(default=60000, ge=1000)  # stop adding excerpts past this size
    rrf_k: int = Field(default=60, ge=1)
    context_embedding: bool = True  # embed each message together with its predecessor
    add_notes: bool = False  # LLM: time notes + atomic facts at Add
    index_facts: bool = True
    show_notes: bool = True  # inline resolved-time notes in returned excerpts
    show_facts: bool = False  # append extracted facts to returned excerpts
    query_expansion: bool = False  # LLM: sub-queries at Search
    rerank: bool = False
    rerank_candidates: int = Field(default=50, ge=1, le=200)
    rerank_weight: float = Field(default=0.7, ge=0, le=1)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_token: SecretStr
    database: Path
    embedding: EmbeddingConfig
    llm: LLMConfig | None = None
    rerank: RerankConfig | None = None
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunk_tokens: int = Field(default=512, ge=4)
    overlap_tokens: int = Field(default=64, ge=0)
    max_body_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    add_concurrency: int = Field(default=16, ge=16, le=64)
    search_concurrency: int = Field(default=16, ge=16, le=256)
    add_timeout_seconds: float = Field(default=600, gt=0)
    search_timeout_seconds: float = Field(default=120, gt=0)
    request_log: bool = False  # one stderr line per request: no query, body or credentials

    @model_validator(mode="after")
    def check_overlap(self):
        r = self.retrieval
        if (r.add_notes or r.query_expansion) and self.llm is None:
            raise ValueError("retrieval features add_notes/query_expansion require an llm section")
        if r.rerank and self.rerank is None:
            raise ValueError("retrieval.rerank requires a rerank section")
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self

    def store_identity(self) -> dict:
        return {
            "schema": 3,
            "context_embedding": self.retrieval.context_embedding,
            "add_notes": self.retrieval.add_notes,
            "index_facts": self.retrieval.index_facts,
            "llm_model": self.llm.model if self.llm and self.retrieval.add_notes else None,
            "model": self.embedding.model,
            "base_url": self.embedding.base_url,
            "protocol": self.embedding.protocol,
            "configured_dimensions": self.embedding.dimensions,
            "tokenizer": "cl100k_base",
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "lexical_version": "english-stem-stopword-han-unigrams-bigrams-v2",
        }


def load_settings(path: Path) -> Settings:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = Settings.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError):
        # Pydantic/YAML 的默认错误包含输入值；不得输出配置中的密钥。
        raise ValueError("Cannot load config; check file, field names and types.") from None
    settings.database = (path.resolve().parent / settings.database).resolve()
    return settings
