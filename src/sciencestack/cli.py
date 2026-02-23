"""ScienceStack CLI — query structured scientific papers from the terminal."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import click

from .client import ScienceStackClient, ScienceStackError
from .formatters import (
    format_citations,
    format_json,
    format_nodes,
    format_overview,
    format_references,
    format_search,
)

_SERVICE_NAME = "sciencestack"
_PROTOCOL_VERSION = "1"

_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ok", "service", "protocolVersion", "command"],
    "properties": {
        "ok": {"type": "boolean"},
        "service": {"type": "string"},
        "protocolVersion": {"type": "string"},
        "command": {"type": "string"},
        "data": {},
        "error": {
            "type": "object",
            "required": ["code", "message", "status", "retryable", "exitCode"],
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "status": {"type": "integer"},
                "retryable": {"type": "boolean"},
                "exitCode": {"type": "integer"},
            },
        },
    },
}

_COMMAND_SHAPES = {
    "search": {"type": "object", "required": ["data"], "properties": {"data": {"type": "array"}}},
    "overview": {
        "oneOf": [
            {"type": "object"},
            {"type": "object", "required": ["count", "items"], "properties": {"count": {"type": "integer"}, "items": {"type": "array"}}},
        ]
    },
    "nodes": {
        "oneOf": [
            {"type": "object", "required": ["data"], "properties": {"data": {}}},
            {"type": "object", "required": ["count", "items"], "properties": {"count": {"type": "integer"}, "items": {"type": "array"}}},
        ]
    },
    "content": {
        "oneOf": [
            {"type": "object"},
            {"type": "object", "required": ["count", "items"], "properties": {"count": {"type": "integer"}, "items": {"type": "array"}}},
        ]
    },
    "refs": {
        "oneOf": [
            {"type": "object", "required": ["data"]},
            {"type": "object", "required": ["count", "items"], "properties": {"count": {"type": "integer"}, "items": {"type": "array"}}},
        ]
    },
    "citations": {
        "oneOf": [
            {"type": "object", "required": ["data"]},
            {"type": "object", "required": ["count", "items"], "properties": {"count": {"type": "integer"}, "items": {"type": "array"}}},
        ]
    },
    "authors": {"type": "object", "required": ["data"], "properties": {"data": {"type": "array"}}},
    "getartifacts": {"type": "object", "required": ["data"]},
    "getartifact": {"type": "object"},
    "batch-nodes": {
        "type": "object",
        "required": ["count", "items"],
        "properties": {"count": {"type": "integer"}, "items": {"type": "array"}},
    },
    "health": {
        "type": "object",
        "required": ["status", "probe"],
        "properties": {
            "status": {"type": "string"},
            "probe": {"type": "object"},
            "latencyMs": {"type": "integer"},
        },
    },
    "capabilities": {
        "type": "object",
        "required": ["service", "protocolVersion", "outputs", "commands"],
        "properties": {
            "service": {"type": "string"},
            "protocolVersion": {"type": "string"},
            "outputs": {"type": "array"},
            "commands": {"type": "array"},
        },
    },
    "schema": {
        "type": "object",
        "required": ["service", "protocolVersion", "schemas"],
        "properties": {
            "service": {"type": "string"},
            "protocolVersion": {"type": "string"},
            "schemas": {"type": "object"},
        },
    },
    "config.path": {
        "type": "object",
        "required": ["path", "exists"],
        "properties": {"path": {"type": "string"}, "exists": {"type": "boolean"}},
    },
    "config.init": {
        "type": "object",
        "required": ["path", "created"],
        "properties": {"path": {"type": "string"}, "created": {"type": "boolean"}},
    },
    "doctor": {
        "type": "object",
        "required": ["status", "checks"],
        "properties": {
            "status": {"type": "string"},
            "checks": {"type": "array"},
        },
    },
}

_COMMAND_CONTRACTS = {
    "search": {
        "args": ["query"],
        "options": ["--limit", "--cursor", "--field", "--sort", "--from", "--to"],
        "responseDataShape": _COMMAND_SHAPES["search"],
    },
    "overview": {
        "args": ["paper_id"],
        "options": ["--sections", "--equations", "--tables", "--figures", "--theorems", "--algorithms"],
        "notes": "paper_id accepts comma-separated values for multi-paper fetch",
        "responseDataShape": _COMMAND_SHAPES["overview"],
    },
    "nodes": {
        "args": ["paper_id"],
        "options": ["--type", "--ids", "--format", "--context", "--limit"],
        "notes": "paper_id accepts comma-separated values for multi-paper fetch",
        "responseDataShape": _COMMAND_SHAPES["nodes"],
    },
    "content": {
        "args": ["paper_id"],
        "options": ["--format"],
        "notes": "paper_id accepts comma-separated values for multi-paper fetch",
        "responseDataShape": _COMMAND_SHAPES["content"],
    },
    "refs": {
        "args": ["paper_id"],
        "options": ["--cite-keys", "--limit", "--cursor"],
        "notes": "paper_id accepts comma-separated values for multi-paper fetch",
        "responseDataShape": _COMMAND_SHAPES["refs"],
    },
    "citations": {
        "args": ["paper_id"],
        "options": ["--limit", "--cursor"],
        "notes": "paper_id accepts comma-separated values for multi-paper fetch",
        "responseDataShape": _COMMAND_SHAPES["citations"],
    },
    "authors": {
        "args": ["author_id"],
        "options": ["--sort", "--limit", "--cursor"],
        "responseDataShape": _COMMAND_SHAPES["authors"],
    },
    "getartifacts": {
        "args": [],
        "options": ["--type", "--field", "--limit", "--cursor"],
        "responseDataShape": _COMMAND_SHAPES["getartifacts"],
    },
    "getartifact": {
        "args": ["slug"],
        "options": [],
        "responseDataShape": _COMMAND_SHAPES["getartifact"],
    },
    "batch-nodes": {
        "args": [],
        "options": ["--requests", "--requests-file", "--format"],
        "responseDataShape": _COMMAND_SHAPES["batch-nodes"],
    },
    "health": {
        "args": [],
        "options": ["--probe-paper-id"],
        "responseDataShape": _COMMAND_SHAPES["health"],
    },
    "capabilities": {
        "args": [],
        "options": [],
        "responseDataShape": _COMMAND_SHAPES["capabilities"],
    },
    "schema": {
        "args": ["command_name?"],
        "options": [],
        "responseDataShape": _COMMAND_SHAPES["schema"],
    },
    "config.path": {
        "args": [],
        "options": [],
        "responseDataShape": _COMMAND_SHAPES["config.path"],
    },
    "doctor": {
        "args": [],
        "options": [],
        "responseDataShape": _COMMAND_SHAPES["doctor"],
    },
    "config.init": {
        "args": [],
        "options": ["--force"],
        "responseDataShape": _COMMAND_SHAPES["config.init"],
    },
}


def _get_client(ctx: click.Context) -> ScienceStackClient:
    return ctx.obj["client"]


def _candidate_config_paths() -> list[Path]:
    override = os.environ.get("SCIENCESTACK_CONFIG")
    if override:
        return [Path(override).expanduser()]

    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_home:
        xdg_path = Path(xdg_home).expanduser() / "sciencestack" / "config.json"
    else:
        xdg_path = Path.home() / ".config" / "sciencestack" / "config.json"

    legacy_path = Path.home() / ".sciencestack" / "config.json"
    return [xdg_path, legacy_path]


def _preferred_config_path() -> Path:
    return _candidate_config_paths()[0]


def _check_config_permissions(path: Path, payload: dict) -> None:
    if os.name == "nt":
        return
    if "api_key" not in payload:
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ScienceStackError("CONFIG", f"Insecure config permissions on {path} (expected 600)", 0)


def _read_cli_config() -> tuple[dict, str | None]:
    for path in _candidate_config_paths():
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
            except OSError as exc:
                raise ScienceStackError("CONFIG", f"Unable to read config file: {path}", 0) from exc
            except json.JSONDecodeError as exc:
                raise ScienceStackError("CONFIG", f"Invalid JSON in config file: {path}", 0) from exc

            if not isinstance(parsed, dict):
                raise ScienceStackError("CONFIG", f"Config file must contain a JSON object: {path}", 0)

            _check_config_permissions(path, parsed)
            return parsed, str(path)

    return {}, None


def _write_config_file(path: Path, payload: dict, *, force: bool = False) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists():
        return False
    serialized = json.dumps(payload, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)
    return True


def _resolve_setting(flag_value, env_name: str, config_value, default_value):
    if flag_value is not None:
        return flag_value
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    if config_value is not None:
        return config_value
    return default_value


def _parallel_ordered_map(ctx: click.Context, values: list, fn):
    max_concurrency = ctx.obj.get("max_concurrency", 8)
    if len(values) <= 1 or max_concurrency <= 1:
        return [fn(v) for v in values]

    max_workers = min(len(values), max_concurrency)
    results = [None] * len(values)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(fn, value): idx for idx, value in enumerate(values)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except ScienceStackError:
                raise
            except Exception as exc:
                raise ScienceStackError("INTERNAL_CLIENT", str(exc), 0) from exc
    return results


def _resolve_output_mode(ctx: click.Context) -> str:
    """Resolve output mode."""
    return ctx.obj.get("output", "json")


def _normalize_machine_payload(data: object) -> tuple[object, dict]:
    """Flatten API payloads where useful for agent ergonomics."""
    if not isinstance(data, dict):
        return data, {}
    if "data" not in data:
        return data, {}

    inner = data.get("data")
    version = data.get("_version")
    meta = {"version": version} if version else {}

    # Flatten object payloads: {"arxivId":..., "data": {"title":...}} -> {"arxivId":..., "title":...}
    if isinstance(inner, dict):
        outer_fields = {k: v for k, v in data.items() if k not in {"data", "_version"}}
        flattened = dict(outer_fields)
        flattened.update(inner)
        return flattened, meta

    return data, meta


def _normalize_response_object(raw_response: object) -> dict:
    normalized, response_meta = _normalize_machine_payload(raw_response)
    response_obj = {"response": normalized}
    if response_meta:
        response_obj["responseMeta"] = response_meta
    return response_obj


def _success_envelope(ctx: click.Context, command_name: str, data: dict, meta: dict | None = None) -> dict:
    normalized_data, normalized_meta = _normalize_machine_payload(data)
    payload = {
        "ok": True,
        "service": _SERVICE_NAME,
        "protocolVersion": ctx.obj.get("protocol_version", _PROTOCOL_VERSION),
        "command": command_name,
        "data": normalized_data,
    }
    merged_meta = {}
    if normalized_meta:
        merged_meta.update(normalized_meta)
    if meta is not None:
        merged_meta.update(meta)
    if merged_meta:
        payload["meta"] = merged_meta
    return payload


def _parse_csv_ids(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        parsed = int(cursor)
    except ValueError as exc:
        raise ScienceStackError("VALIDATION", f"Cursor must be an integer offset, got: {cursor}", 0) from exc
    if parsed < 0:
        raise ScienceStackError("VALIDATION", "Cursor must be >= 0", 0)
    return parsed


def _build_pagination_meta(data: dict, *, cursor_offset: int, page_size: int) -> dict:
    pagination = data.get("pagination")
    next_cursor = None
    has_more = None

    if isinstance(pagination, dict):
        has_more = bool(pagination.get("hasMore"))
        current_offset = int(pagination.get("offset", cursor_offset))
        current_limit = int(pagination.get("limit", page_size))
        if has_more:
            next_cursor = str(current_offset + current_limit)
    else:
        rows = data.get("data")
        if isinstance(rows, list) and len(rows) >= page_size:
            next_cursor = str(cursor_offset + len(rows))

    return {
        "pagination": {
            "cursor": str(cursor_offset),
            "nextCursor": next_cursor,
            "pageSize": page_size,
            "hasMore": has_more,
        }
    }


def _schema_type_matches(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate_schema(value: object, schema: dict, *, path: str = "$") -> list[str]:
    if not schema:
        return []

    if "oneOf" in schema:
        options = schema.get("oneOf", [])
        for option in options:
            if not _validate_schema(value, option, path=path):
                return []
        return [f"{path}: does not match any allowed schema variant"]

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _schema_type_matches(value, expected_type):
        return [f"{path}: expected {expected_type}"]

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing required field")

        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in value:
                errors.extend(_validate_schema(value[prop], prop_schema, path=f"{path}.{prop}"))

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_schema(item, item_schema, path=f"{path}[{idx}]"))

    return errors


def _enforce_strict_schema(
    ctx: click.Context,
    command_name: str,
    envelope: dict,
    *,
    validate_command_shape: bool = True,
) -> None:
    if not ctx.obj.get("strict"):
        return

    errors = _validate_schema(envelope, _AGENT_OUTPUT_SCHEMA, path="$")
    if validate_command_shape and envelope.get("ok") and command_name in _COMMAND_SHAPES:
        errors.extend(_validate_schema(envelope.get("data"), _COMMAND_SHAPES[command_name], path="$.data"))

    if errors:
        raise ScienceStackError("CONTRACT_VIOLATION", "; ".join(errors[:3]), 0)


def _emit_multi_paper(
    ctx: click.Context,
    *,
    command_name: str,
    items: list[dict],
    human_sections: list[str] | None = None,
    meta: dict | None = None,
) -> None:
    mode = _resolve_output_mode(ctx)
    if mode == "human":
        if human_sections:
            for idx, section in enumerate(human_sections):
                click.echo(section)
                if idx < len(human_sections) - 1:
                    click.echo("---")
        else:
            click.echo(f"{len(items)} items")
        return

    if mode == "ndjson":
        _emit_data(ctx, {}, command_name=command_name, ndjson_items=items, meta=meta)
        return

    payload = {"count": len(items), "items": items}
    _emit_data(ctx, payload, command_name=command_name, meta=meta)


def _run_multi_paper_command(
    ctx: click.Context,
    *,
    command_name: str,
    paper_ids: list[str],
    fetch_fn,
    human_formatter=None,
    meta: dict | None = None,
) -> None:
    responses = _parallel_ordered_map(ctx, paper_ids, fetch_fn)
    is_human = _resolve_output_mode(ctx) == "human"
    items = []
    human_sections = []
    for pid, data in zip(paper_ids, responses):
        item = {"paperId": pid}
        item.update(_normalize_response_object(data))
        items.append(item)
        if is_human and human_formatter is not None:
            human_sections.append(human_formatter(data))

    _emit_multi_paper(
        ctx,
        command_name=command_name,
        items=items,
        human_sections=human_sections if is_human and human_formatter is not None else None,
        meta=meta,
    )


def _normalize_batch_overview(batch_data: dict, expected_count: int) -> list[dict]:
    papers = batch_data.get("data", [])
    responses = []
    for paper in papers:
        paper_data = paper.get("data", paper)
        responses.append({"data": paper_data})
    if len(responses) != expected_count:
        raise ScienceStackError(
            "INVALID_RESPONSE",
            f"Batch overview returned {len(responses)} items for {expected_count} requested papers",
            0,
        )
    return responses


def _execute_paper_command(
    ctx: click.Context,
    *,
    command_name: str,
    paper_id_arg: str,
    fetch_one,
    fetch_many=None,
    human_formatter=None,
    single_meta_builder=None,
) -> None:
    paper_ids = _parse_csv_ids(paper_id_arg)
    if len(paper_ids) > 1:
        if fetch_many is None:
            _run_multi_paper_command(
                ctx,
                command_name=command_name,
                paper_ids=paper_ids,
                fetch_fn=fetch_one,
                human_formatter=human_formatter,
            )
        else:
            responses = fetch_many(paper_ids)
            is_human = _resolve_output_mode(ctx) == "human"
            items = []
            human_sections = []
            for pid, data in zip(paper_ids, responses):
                item = {"paperId": pid}
                item.update(_normalize_response_object(data))
                items.append(item)
                if is_human and human_formatter is not None:
                    human_sections.append(human_formatter(data))
            _emit_multi_paper(
                ctx,
                command_name=command_name,
                items=items,
                human_sections=human_sections if is_human and human_formatter is not None else None,
            )
        return

    data = fetch_one(paper_ids[0])
    if _resolve_output_mode(ctx) == "human" and human_formatter is not None:
        click.echo(human_formatter(data))
        return

    single_meta = single_meta_builder(data) if single_meta_builder is not None else None
    _emit_data(ctx, data, command_name=command_name, meta=single_meta)


def _emit_data(
    ctx: click.Context,
    data: dict,
    *,
    command_name: str,
    human_text: str | None = None,
    ndjson_items: list[dict] | None = None,
    meta: dict | None = None,
) -> None:
    """Emit command output in requested format."""
    mode = _resolve_output_mode(ctx)
    if mode == "human":
        click.echo(human_text if human_text is not None else format_json(data))
        return

    if mode == "ndjson":
        items = ndjson_items
        if items is None:
            # For list-bearing responses, stream each item by default.
            data_items = data.get("data")
            if isinstance(data_items, list):
                items = data_items
            else:
                items = [data]
        for idx, item in enumerate(items):
            row_meta = {"streamIndex": idx}
            if meta:
                row_meta.update(meta)
            row = _success_envelope(ctx, command_name, item, meta=row_meta)
            _enforce_strict_schema(ctx, command_name, row, validate_command_shape=False)
            click.echo(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return

    payload = _success_envelope(ctx, command_name, data, meta=meta)
    _enforce_strict_schema(ctx, command_name, payload)
    click.echo(format_json(payload))


def _exit_code_for_error(err: ScienceStackError) -> int:
    """Map API failures to deterministic process exit codes."""
    if err.code == "CONTRACT_VIOLATION":
        return 17
    if err.code in {"VALIDATION", "CONFIG"}:
        return 2
    if err.code in {"TIMEOUT", "NETWORK"}:
        return 13
    if err.status_code in {401, 403}:
        return 10
    if err.status_code == 429:
        return 11
    if err.status_code == 404:
        return 12
    if err.status_code >= 500:
        return 15
    return 16


def _exit_with_error(ctx: click.Context, err: ScienceStackError) -> None:
    payload = {
        "ok": False,
        "service": _SERVICE_NAME,
        "protocolVersion": ctx.obj.get("protocol_version", _PROTOCOL_VERSION),
        "command": ctx.info_name or "main",
        "error": {
            "code": err.code,
            "message": err.message,
            "status": err.status_code,
            "retryable": err.code in {"TIMEOUT", "NETWORK"} or err.status_code in {429, 502, 503, 504},
            "exitCode": _exit_code_for_error(err),
        },
    }

    mode = _resolve_output_mode(ctx)
    if mode == "human":
        click.echo(f"Error: {err.code}: {err.message}", err=True)
    elif mode == "ndjson":
        click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")), err=True)
    else:
        click.echo(format_json(payload), err=True)
    sys.exit(payload["error"]["exitCode"])


def _capabilities_payload() -> dict:
    return {
        "service": _SERVICE_NAME,
        "protocolVersion": _PROTOCOL_VERSION,
        "outputs": ["json", "ndjson", "human"],
        "commands": sorted(_COMMAND_CONTRACTS.keys()),
        "globalOptions": [
            "--output",
            "--api-key",
            "--base-url",
            "--protocol-version",
            "--timeout",
            "--retries",
            "--retry-backoff-ms",
            "--max-concurrency",
            "--strict",
        ],
        "multiPaperCommands": ["overview", "nodes", "content", "refs", "citations", "batch-nodes"],
        "config": {
            "env": [
                "SCIENCESTACK_API_KEY",
                "SCIENCESTACK_BASE_URL",
                "SCIENCESTACK_OUTPUT",
                "SCIENCESTACK_PROTOCOL_VERSION",
                "SCIENCESTACK_TIMEOUT",
                "SCIENCESTACK_RETRIES",
                "SCIENCESTACK_RETRY_BACKOFF_MS",
                "SCIENCESTACK_STRICT",
                "SCIENCESTACK_MAX_CONCURRENCY",
                "SCIENCESTACK_CONFIG",
            ],
            "paths": [str(p) for p in _candidate_config_paths()],
            "precedence": "flag > env > config > default",
            "supportedKeys": [
                "api_key",
                "base_url",
                "output",
                "protocol_version",
                "timeout",
                "retries",
                "retry_backoff_ms",
                "max_concurrency",
                "strict",
            ],
        },
        "errorShape": _AGENT_OUTPUT_SCHEMA["properties"]["error"],
    }


def _schemas_payload(command_name: str | None = None) -> dict:
    if command_name:
        if command_name not in _COMMAND_CONTRACTS:
            raise ScienceStackError("VALIDATION", f"Unknown command for schema: {command_name}", 0)
        schemas = {command_name: _COMMAND_CONTRACTS[command_name]}
    else:
        schemas = _COMMAND_CONTRACTS
    return {
        "service": _SERVICE_NAME,
        "protocolVersion": _PROTOCOL_VERSION,
        "schemas": schemas,
        "baseEnvelope": _AGENT_OUTPUT_SCHEMA,
    }


@click.group()
@click.option("--api-key", default=None, help="API key (or set SCIENCESTACK_API_KEY)")
@click.option("--base-url", default=None, help="API base URL")
@click.option(
    "--output",
    default=None,
    type=click.Choice(["json", "human", "ndjson"]),
    help="Output format (json for agents, human for readable text, ndjson for streaming).",
)
@click.option(
    "--protocol-version",
    default=None,
    type=click.Choice([_PROTOCOL_VERSION]),
    help="Agent protocol version contract.",
)
@click.option("--timeout", default=None, type=float, help="HTTP request timeout in seconds.")
@click.option("--retries", default=None, type=int, help="Retries for transient HTTP/network failures.")
@click.option("--retry-backoff-ms", default=None, type=int, help="Base retry backoff in milliseconds.")
@click.option("--max-concurrency", default=None, type=int, help="Max parallel requests for multi-paper commands.")
@click.option("--strict/--no-strict", default=None, help="Validate outputs against declared command schema.")
@click.pass_context
def main(
    ctx,
    api_key: str | None,
    base_url: str,
    output: str,
    protocol_version: str,
    timeout: float | None,
    retries: int | None,
    retry_backoff_ms: int | None,
    max_concurrency: int | None,
    strict: bool,
):
    """Query structured scientific papers via the ScienceStack API."""
    ctx.ensure_object(dict)
    try:
        file_config, config_path = _read_cli_config()
    except ScienceStackError as e:
        # We cannot rely on ctx.obj yet for mode resolution.
        click.echo(
            json.dumps(
                {
                    "ok": False,
                    "service": _SERVICE_NAME,
                    "protocolVersion": _PROTOCOL_VERSION,
                    "command": ctx.invoked_subcommand or "main",
                    "error": {
                        "code": e.code,
                        "message": e.message,
                        "status": e.status_code,
                        "retryable": False,
                        "exitCode": 2,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            err=True,
        )
        sys.exit(2)

    try:
        resolved_api_key = _resolve_setting(api_key, "SCIENCESTACK_API_KEY", file_config.get("api_key"), None)
        resolved_base_url = _resolve_setting(
            base_url,
            "SCIENCESTACK_BASE_URL",
            file_config.get("base_url"),
            "https://sciencestack.ai/api/v1",
        )
        resolved_output = _resolve_setting(output, "SCIENCESTACK_OUTPUT", file_config.get("output"), "json")
        resolved_protocol_version = _resolve_setting(
            protocol_version,
            "SCIENCESTACK_PROTOCOL_VERSION",
            file_config.get("protocol_version"),
            _PROTOCOL_VERSION,
        )
        resolved_timeout = float(_resolve_setting(timeout, "SCIENCESTACK_TIMEOUT", file_config.get("timeout"), 30.0))
        resolved_retries = int(_resolve_setting(retries, "SCIENCESTACK_RETRIES", file_config.get("retries"), 0))
        resolved_retry_backoff_ms = int(
            _resolve_setting(
                retry_backoff_ms,
                "SCIENCESTACK_RETRY_BACKOFF_MS",
                file_config.get("retry_backoff_ms"),
                250,
            )
        )
        resolved_max_concurrency = int(
            _resolve_setting(max_concurrency, "SCIENCESTACK_MAX_CONCURRENCY", file_config.get("max_concurrency"), 8)
        )
        if resolved_output not in {"json", "human", "ndjson"}:
            raise ScienceStackError("CONFIG", f"Invalid output in config/env: {resolved_output}", 0)
        if resolved_protocol_version != _PROTOCOL_VERSION:
            raise ScienceStackError("CONFIG", f"Unsupported protocol version: {resolved_protocol_version}", 0)
        if resolved_timeout <= 0 or resolved_timeout > 300:
            raise ScienceStackError("CONFIG", "timeout must be > 0 and <= 300 seconds", 0)
        if resolved_retries < 0 or resolved_retries > 10:
            raise ScienceStackError("CONFIG", "retries must be between 0 and 10", 0)
        if resolved_retry_backoff_ms < 0 or resolved_retry_backoff_ms > 60000:
            raise ScienceStackError("CONFIG", "retry_backoff_ms must be between 0 and 60000", 0)
        if resolved_max_concurrency < 1 or resolved_max_concurrency > 64:
            raise ScienceStackError("CONFIG", "max_concurrency must be between 1 and 64", 0)

        strict_env = os.environ.get("SCIENCESTACK_STRICT")
        strict_config = file_config.get("strict")
        if strict is not None:
            resolved_strict = strict
        elif strict_env not in (None, ""):
            resolved_strict = strict_env.lower() in {"1", "true", "yes", "on"}
        elif strict_config is not None:
            resolved_strict = bool(strict_config)
        else:
            resolved_strict = False
    except ScienceStackError as e:
        click.echo(
            json.dumps(
                {
                    "ok": False,
                    "service": _SERVICE_NAME,
                    "protocolVersion": _PROTOCOL_VERSION,
                    "command": ctx.invoked_subcommand or "main",
                    "error": {
                        "code": e.code,
                        "message": e.message,
                        "status": e.status_code,
                        "retryable": False,
                        "exitCode": 2,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            err=True,
        )
        sys.exit(2)

    ctx.obj["output"] = resolved_output
    ctx.obj["protocol_version"] = resolved_protocol_version
    ctx.obj["timeout"] = resolved_timeout
    ctx.obj["retries"] = resolved_retries
    ctx.obj["retry_backoff_ms"] = resolved_retry_backoff_ms
    ctx.obj["max_concurrency"] = resolved_max_concurrency
    ctx.obj["strict"] = resolved_strict
    ctx.obj["config_path"] = config_path
    command_without_auth = {"capabilities", "schema", "config", "doctor"}
    if ctx.invoked_subcommand in command_without_auth:
        return

    if not resolved_api_key:
        if resolved_output == "human":
            click.echo("Error: No API key. Set SCIENCESTACK_API_KEY or pass --api-key.", err=True)
        else:
            click.echo(
                json.dumps(
                    {
                        "ok": False,
                        "service": _SERVICE_NAME,
                        "protocolVersion": resolved_protocol_version,
                        "command": ctx.invoked_subcommand or "main",
                        "error": {
                            "code": "CONFIG",
                            "message": "No API key. Set SCIENCESTACK_API_KEY or pass --api-key.",
                            "status": 0,
                            "retryable": False,
                            "exitCode": 2,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                err=True,
            )
        sys.exit(2)
    ctx.obj["client"] = ScienceStackClient(
        resolved_api_key,
        resolved_base_url,
        timeout=resolved_timeout,
        retries=resolved_retries,
        retry_backoff_ms=resolved_retry_backoff_ms,
    )


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


@main.command()
@click.pass_context
def capabilities(ctx):
    """Return machine-readable CLI capabilities for agent bootstrap."""
    payload = _capabilities_payload()
    human = f"protocol={payload['protocolVersion']} outputs={','.join(payload['outputs'])} commands={len(payload['commands'])}"
    _emit_data(ctx, payload, command_name="capabilities", human_text=human)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@main.command()
@click.argument("command_name", required=False)
@click.pass_context
def schema(ctx, command_name):
    """Return JSON schema for all commands or one command."""
    try:
        payload = _schemas_payload(command_name)
        human = f"schemas={len(payload['schemas'])} protocol={payload['protocolVersion']}"
        _emit_data(ctx, payload, command_name="schema", human_text=human)
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@main.group()
def config():
    """Manage local CLI configuration."""


@config.command(name="path")
@click.pass_context
def config_path(ctx):
    """Show effective config path."""
    cfg_path = ctx.obj.get("config_path") or str(_preferred_config_path())
    payload = {"path": cfg_path, "exists": Path(cfg_path).exists()}
    _emit_data(ctx, payload, command_name="config.path", human_text=cfg_path)


@config.command(name="init")
@click.option("--force", is_flag=True, help="Overwrite existing config file")
@click.pass_context
def config_init(ctx, force):
    """Create a local config template."""
    target = _preferred_config_path()
    template = {
        "api_key": "",
        "base_url": "https://sciencestack.ai/api/v1",
        "output": "json",
        "protocol_version": _PROTOCOL_VERSION,
        "timeout": 30.0,
        "retries": 0,
        "retry_backoff_ms": 250,
        "max_concurrency": 8,
        "strict": False,
    }
    try:
        created = _write_config_file(target, template, force=force)
        payload = {"path": str(target), "created": created}
        human = f"{'created' if created else 'exists'}: {target}"
        _emit_data(ctx, payload, command_name="config.init", human_text=human)
    except OSError as exc:
        _exit_with_error(ctx, ScienceStackError("CONFIG", f"Failed to write config: {target}", 0))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--cursor", default=None, help="Cursor offset from previous page (integer string)")
@click.option("--field", "-f", default=None, help="Semantic field filter")
@click.option("--sort", "-s", default="relevance", type=click.Choice(["relevance", "recent", "citations"]))
@click.option("--from", "from_date", default=None, help="Published after (YYYY-MM-DD)")
@click.option("--to", "to_date", default=None, help="Published before (YYYY-MM-DD)")
@click.pass_context
def search(ctx, query, limit, cursor, field, sort, from_date, to_date):
    """Search for papers by topic, author, or arXiv ID.

    Tip: short, specific keywords work best (e.g. "attention mechanism" not
    "papers about attention mechanisms in transformer models").
    """
    client = _get_client(ctx)
    try:
        offset = _parse_cursor(cursor)
        data = client.search(query, limit=limit, offset=offset, field=field, sort=sort, from_date=from_date, to_date=to_date)
        meta = _build_pagination_meta(data, cursor_offset=offset, page_size=limit)
        _emit_data(ctx, data, command_name="search", human_text=format_search(data), meta=meta)
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

@main.command()
@click.argument("paper_id")
@click.option("--sections", is_flag=True, help="Show TOC only")
@click.option("--equations", is_flag=True, help="Show equations (hint: use 'nodes --type equation' for full content)")
@click.option("--tables", is_flag=True, help="Show tables only")
@click.option("--figures", is_flag=True, help="Show figures only")
@click.option("--theorems", is_flag=True, help="Show theorems/lemmas/proofs (mathEnvs) only")
@click.option("--algorithms", is_flag=True, help="Show algorithms only")
@click.pass_context
def overview(ctx, paper_id, sections, equations, tables, figures, theorems, algorithms):
    """Get paper overview (metadata, TOC, element lists).

    Supports comma-separated IDs for batch: sciencestack overview 1706.03762,2305.18290
    """
    client = _get_client(ctx)
    try:
        filter_key = _get_filter_key(sections, equations, tables, figures, theorems, algorithms)
        _execute_paper_command(
            ctx,
            command_name="overview",
            paper_id_arg=paper_id,
            fetch_one=lambda pid: client.overview(pid),
            fetch_many=lambda ids: _normalize_batch_overview(client.batch_overview(ids), len(ids)),
            human_formatter=lambda data: format_overview(data, filter_key),
        )
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


def _get_filter_key(*flags) -> str | None:
    names = ["sections", "equations", "tables", "figures", "theorems", "algorithms"]
    for flag, name in zip(flags, names):
        if flag:
            return name
    return None


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

@main.command()
@click.argument("paper_id")
@click.option("--type", "-t", "node_type", default=None, help="Node type: equation, table, figure, section, math_env, algorithm, code")
@click.option("--ids", "-i", default=None, help="Specific node IDs (comma-separated): eq:1,thm:2,sec:3.2")
@click.option("--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "latex", "raw"]), help="Output format")
@click.option("--context", "-c", "ctx_size", default=None, type=int, help="Include N surrounding nodes")
@click.option("--limit", "-n", default=100, help="Max nodes to return")
@click.pass_context
def nodes(ctx, paper_id, node_type, ids, fmt, ctx_size, limit):
    """Fetch specific paper nodes (equations, sections, figures, etc.)."""
    client = _get_client(ctx)
    try:
        _execute_paper_command(
            ctx,
            command_name="nodes",
            paper_id_arg=paper_id,
            fetch_one=lambda pid: client.nodes(
                pid,
                types=node_type,
                node_ids=ids,
                format=fmt,
                context=ctx_size,
                limit=limit,
            ),
            human_formatter=lambda data: format_nodes(data, fmt),
        )
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------

@main.command()
@click.argument("paper_id")
@click.option("--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "latex", "raw"]))
@click.pass_context
def content(ctx, paper_id, fmt):
    """Get full paper content."""
    client = _get_client(ctx)
    try:
        _execute_paper_command(
            ctx,
            command_name="content",
            paper_id_arg=paper_id,
            fetch_one=lambda pid: client.content(pid, format=fmt),
            human_formatter=lambda data: data.get("data", {}).get("content", ""),
        )
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@main.command()
@click.option("--probe-paper-id", default="1706.03762", show_default=True, help="Known paper ID used for health probe")
@click.pass_context
def health(ctx, probe_paper_id):
    """Probe API health and auth using a stable overview call."""
    client = _get_client(ctx)
    try:
        # A lightweight successful overview proves auth + API availability.
        client.overview(probe_paper_id)
        payload = {
            "status": "up",
            "probe": {"command": "overview", "paperId": probe_paper_id},
        }
        _emit_data(ctx, payload, command_name="health", human_text=f"up (probe={probe_paper_id})")
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _doctor_check_config() -> dict:
    """Check config file existence, validity, and permissions."""
    check = {"name": "config", "status": "pass"}
    paths = _candidate_config_paths()
    found_path = None
    for p in paths:
        if p.exists():
            found_path = p
            break

    if found_path is None:
        check["status"] = "warn"
        check["message"] = "No config file found (optional but recommended)"
        check["hint"] = "Run: sciencestack config init"
        return check

    check["path"] = str(found_path)

    try:
        raw = found_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        check["status"] = "fail"
        check["message"] = f"Cannot read config: {exc}"
        return check

    if not isinstance(parsed, dict):
        check["status"] = "fail"
        check["message"] = "Config must be a JSON object"
        return check

    # Permission check
    if os.name != "nt" and "api_key" in parsed:
        mode = found_path.stat().st_mode & 0o777
        if mode & 0o077:
            check["status"] = "fail"
            check["message"] = f"Insecure permissions: {oct(mode)} (expected 0o600)"
            check["hint"] = f"Run: chmod 600 {found_path}"
            return check

    check["message"] = "Config loaded OK"
    return check


def _doctor_check_api_key(ctx: click.Context) -> dict:
    """Check API key presence and source."""
    check = {"name": "api_key", "status": "pass"}

    flag_key = ctx.params.get("api_key")
    env_key = os.environ.get("SCIENCESTACK_API_KEY")
    file_config, _ = _read_cli_config()
    config_key = file_config.get("api_key") if isinstance(file_config, dict) else None

    if flag_key:
        check["source"] = "flag"
    elif env_key:
        check["source"] = "env"
    elif config_key:
        check["source"] = "config"
    else:
        check["status"] = "fail"
        check["message"] = "No API key found"
        check["hint"] = "Set SCIENCESTACK_API_KEY or add api_key to config"
        return check

    check["message"] = f"API key found (source: {check['source']})"
    return check


def _doctor_check_connectivity(ctx: click.Context) -> dict:
    """Probe upstream API health."""
    check = {"name": "connectivity", "status": "pass"}

    # Resolve key for probe
    env_key = os.environ.get("SCIENCESTACK_API_KEY")
    file_config, _ = _read_cli_config()
    config_key = file_config.get("api_key") if isinstance(file_config, dict) else None
    api_key = ctx.params.get("api_key") or env_key or config_key

    if not api_key:
        check["status"] = "skip"
        check["message"] = "Skipped (no API key)"
        return check

    base_url = ctx.obj.get("base_url", "https://sciencestack.ai/api/v1")
    timeout = ctx.obj.get("timeout", 30.0)
    client = ScienceStackClient(api_key, base_url, timeout=timeout, retries=0, retry_backoff_ms=0)
    try:
        client.overview("1706.03762")
        check["message"] = "API reachable, auth valid"
    except ScienceStackError as e:
        if e.status_code in {401, 403}:
            check["status"] = "fail"
            check["message"] = f"Auth failed: {e.message}"
        elif e.status_code == 429:
            check["status"] = "warn"
            check["message"] = "Rate limited (API is reachable but throttled)"
        elif e.code in {"TIMEOUT", "NETWORK"}:
            check["status"] = "fail"
            check["message"] = f"Cannot reach API: {e.message}"
        elif e.status_code >= 500:
            check["status"] = "warn"
            check["message"] = f"API returned server error: {e.message}"
        else:
            check["status"] = "fail"
            check["message"] = f"{e.code}: {e.message}"

    return check


@main.command()
@click.pass_context
def doctor(ctx):
    """Run diagnostics on config, auth, and API connectivity."""
    checks = [
        _doctor_check_config(),
        _doctor_check_api_key(ctx),
        _doctor_check_connectivity(ctx),
    ]

    all_pass = all(c["status"] == "pass" for c in checks)
    any_fail = any(c["status"] == "fail" for c in checks)
    overall = "pass" if all_pass else ("fail" if any_fail else "warn")

    payload = {"status": overall, "checks": checks}

    # Human output
    lines = []
    icons = {"pass": "+", "warn": "!", "fail": "x", "skip": "-"}
    for c in checks:
        icon = icons.get(c["status"], "?")
        line = f"[{icon}] {c['name']}: {c.get('message', c['status'])}"
        hint = c.get("hint")
        if hint:
            line += f"\n    hint: {hint}"
        lines.append(line)
    lines.append(f"\nOverall: {overall}")
    human_text = "\n".join(lines)

    _emit_data(ctx, payload, command_name="doctor", human_text=human_text)


# ---------------------------------------------------------------------------
# batch-nodes
# ---------------------------------------------------------------------------


def _load_batch_requests(requests: str | None, requests_file: str | None) -> list[dict]:
    if bool(requests) == bool(requests_file):
        raise ScienceStackError("VALIDATION", "Pass exactly one of --requests or --requests-file", 0)

    raw = requests
    if requests_file:
        try:
            if requests_file == "-":
                raw = sys.stdin.read()
            else:
                with open(requests_file, "r", encoding="utf-8") as f:
                    raw = f.read()
        except OSError as exc:
            raise ScienceStackError("VALIDATION", f"Unable to read requests file: {requests_file}", 0) from exc

    try:
        parsed = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ScienceStackError("VALIDATION", "Requests must be valid JSON", 0) from exc

    if not isinstance(parsed, list) or not parsed:
        raise ScienceStackError("VALIDATION", "Requests must be a non-empty JSON array", 0)
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise ScienceStackError("VALIDATION", f"Request at index {i} must be an object", 0)
    return parsed


def _as_csv(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    raise ScienceStackError("VALIDATION", f"{field_name} must be string or array", 0)


def _normalize_batch_node_request(req: dict, idx: int) -> dict:
    paper_id = req.get("paperId") or req.get("arxivId")
    if not paper_id or not isinstance(paper_id, str):
        raise ScienceStackError("VALIDATION", f"Request {idx} missing string paperId/arxivId", 0)

    node_ids = _as_csv(req.get("nodeIds"), field_name="nodeIds")
    node_types = _as_csv(req.get("types"), field_name="types")
    if node_ids is None and node_types is None:
        raise ScienceStackError("VALIDATION", f"Request {idx} must include nodeIds or types", 0)

    return {
        "requestIndex": idx,
        "request": req,
        "paperId": paper_id,
        "nodeIds": node_ids,
        "types": node_types,
        "context": req.get("context"),
        "limit": req.get("limit", 100),
    }


@main.command(name="batch-nodes")
@click.option("--requests", default=None, help="JSON array of requests (MCP-like)")
@click.option("--requests-file", default=None, help="Path to JSON request array file or '-' for stdin")
@click.option("--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "latex", "raw"]), help="Output format")
@click.pass_context
def batch_nodes(ctx, requests, requests_file, fmt):
    """Fetch nodes for multiple papers in one call using request arrays."""
    client = _get_client(ctx)
    try:
        reqs = _load_batch_requests(requests, requests_file)
        normalized = [_normalize_batch_node_request(req, idx) for idx, req in enumerate(reqs)]

        def _fetch_one(item):
            data = client.nodes(
                item["paperId"],
                types=item["types"],
                node_ids=item["nodeIds"],
                format=fmt,
                context=item["context"],
                limit=item["limit"],
            )
            row = {
                "requestIndex": item["requestIndex"],
                "paperId": item["paperId"],
                "request": item["request"],
            }
            row.update(_normalize_response_object(data))
            return row

        items = _parallel_ordered_map(ctx, normalized, _fetch_one)

        _emit_multi_paper(
            ctx,
            command_name="batch-nodes",
            items=items,
            meta={"format": fmt},
        )
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# refs
# ---------------------------------------------------------------------------

@main.command()
@click.argument("paper_id")
@click.option("--cite-keys", "-k", default=None, help="Filter by cite keys (comma-separated)")
@click.option("--limit", "-n", default=100)
@click.option("--cursor", default=None, help="Cursor offset from previous page (integer string)")
@click.pass_context
def refs(ctx, paper_id, cite_keys, limit, cursor):
    """Get paper references (bibliography)."""
    client = _get_client(ctx)
    try:
        offset = _parse_cursor(cursor)
        _execute_paper_command(
            ctx,
            command_name="refs",
            paper_id_arg=paper_id,
            fetch_one=lambda pid: client.references(pid, cite_keys=cite_keys, limit=limit, offset=offset),
            human_formatter=format_references,
            single_meta_builder=lambda data: _build_pagination_meta(data, cursor_offset=offset, page_size=limit),
        )
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------

@main.command()
@click.argument("paper_id")
@click.option("--limit", "-n", default=100)
@click.option("--cursor", default=None, help="Cursor offset from previous page (integer string)")
@click.pass_context
def citations(ctx, paper_id, limit, cursor):
    """Get papers that cite this paper."""
    client = _get_client(ctx)
    try:
        offset = _parse_cursor(cursor)
        _execute_paper_command(
            ctx,
            command_name="citations",
            paper_id_arg=paper_id,
            fetch_one=lambda pid: client.citations(pid, limit=limit, offset=offset),
            human_formatter=format_citations,
            single_meta_builder=lambda data: _build_pagination_meta(data, cursor_offset=offset, page_size=limit),
        )
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# authors
# ---------------------------------------------------------------------------

@main.command()
@click.argument("author_id")
@click.option("--sort", "-s", default="recent", type=click.Choice(["recent", "citations"]))
@click.option("--limit", "-n", default=20)
@click.option("--cursor", default=None, help="Cursor offset from previous page (integer string)")
@click.pass_context
def authors(ctx, author_id, sort, limit, cursor):
    """Get papers by author (Semantic Scholar author ID)."""
    client = _get_client(ctx)
    try:
        offset = _parse_cursor(cursor)
        data = client.author_papers(author_id, sort=sort, limit=limit, offset=offset)
        meta = _build_pagination_meta(data, cursor_offset=offset, page_size=limit)
        _emit_data(ctx, data, command_name="authors", human_text=format_search(data), meta=meta)
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# getartifacts
# ---------------------------------------------------------------------------


@main.command(name="getartifacts")
@click.option("--type", "artifact_type", default=None, help="Artifact type (e.g. weekly_digest, monthly_digest)")
@click.option("--field", default=None, help="Field/tag filter (e.g. rl, nlp, cv)")
@click.option("--limit", "-n", default=20, help="Max artifacts to return")
@click.option("--cursor", default=None, help="Cursor offset from previous page (integer string)")
@click.pass_context
def getartifacts(ctx, artifact_type, field, limit, cursor):
    """List curated research artifacts."""
    client = _get_client(ctx)
    try:
        offset = _parse_cursor(cursor)
        data = client.artifacts(type=artifact_type, field=field, limit=limit, offset=offset)
        meta = _build_pagination_meta(data, cursor_offset=offset, page_size=limit)
        _emit_data(ctx, data, command_name="getartifacts", meta=meta)
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


# ---------------------------------------------------------------------------
# getartifact
# ---------------------------------------------------------------------------


@main.command(name="getartifact")
@click.argument("slug")
@click.pass_context
def getartifact(ctx, slug):
    """Get full artifact payload by slug."""
    client = _get_client(ctx)
    try:
        data = client.artifact(slug)
        _emit_data(ctx, data, command_name="getartifact")
    except ScienceStackError as e:
        _exit_with_error(ctx, e)


if __name__ == "__main__":
    main()
