"""Shared deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

JsonDefault = Callable[[Any], Any]


def canonical_json_text(
    payload: Any,
    *,
    sort_keys: bool = True,
    separators: tuple[str, str] | None = (",", ":"),
    default: JsonDefault | None = str,
    ensure_ascii: bool = True,
) -> str:
    """Return a deterministic JSON representation for hash material."""
    kwargs: dict[str, Any] = {
        "sort_keys": sort_keys,
        "ensure_ascii": ensure_ascii,
    }
    if separators is not None:
        kwargs["separators"] = separators
    if default is not None:
        kwargs["default"] = default
    return json.dumps(payload, **kwargs)


def stable_json_hash(
    payload: Any,
    *,
    sort_keys: bool = True,
    separators: tuple[str, str] | None = (",", ":"),
    default: JsonDefault | None = str,
    ensure_ascii: bool = True,
) -> str:
    material = canonical_json_text(
        payload,
        sort_keys=sort_keys,
        separators=separators,
        default=default,
        ensure_ascii=ensure_ascii,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    read_size = max(1, int(chunk_size))
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(read_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["canonical_json_text", "file_sha256", "stable_json_hash"]
