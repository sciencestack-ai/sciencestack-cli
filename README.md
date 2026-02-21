# sciencestack CLI

Lightweight CLI for querying structured scientific papers (arXiv) via ScienceStack API.

Built for both:
- humans (`--output human`)
- agents (`--output json` default, `--output ndjson` for streams)

## Why this exists

Science is locked in PDFs. Equations, theorems, and figures have no stable addresses - you can't point to "the loss function in Section 3.2" and have a machine fetch it.

This CLI exposes papers as structured objects with node-level IDs (`eq:1`, `thm:3`, `fig:2`, `sec:3.2`). Every equation, figure, theorem, and section is addressable and retrievable.

This enables:
- **Claim graphs**: Link assertions to specific evidence nodes across papers
- **Precise retrieval**: Fetch `thm:4.3` instead of downloading 23 pages
- **Citation traversal**: Follow references at the node level, not paper level
- **Agent workflows**: Stable contracts for automation, not brittle PDF parsing

Agent-first design:
- Stable machine envelope (`ok/service/protocolVersion/command/data`)
- Discoverable contracts (`capabilities`, `schema`)
- Deterministic errors with retry semantics

## Install

```bash
pip install sciencestack
```

Or with pipx (recommended for CLI tools):
```bash
pipx install sciencestack
```

Then run:
```bash
sciencestack --help
```

## Auth

Get an API key at `https://sciencestack.ai`.

Set API key:
```bash
export SCIENCESTACK_API_KEY=your_key_here
```

Or pass on each command:
```bash
sciencestack --api-key your_key_here search "transformers"
```

## Config file (industry-standard)

The CLI supports a user config file in home:

- primary: `~/.config/sciencestack/config.json`
- fallback: `~/.sciencestack/config.json`
- override path: `SCIENCESTACK_CONFIG=/path/to/config.json`

Supported keys:

```json
{
  "api_key": "sk_live_...",
  "base_url": "https://sciencestack.ai/api/v1",
  "output": "json",
  "protocol_version": "1",
  "timeout": 30.0,
  "retries": 0,
  "retry_backoff_ms": 250,
  "max_concurrency": 8,
  "strict": false
}
```

Precedence is:

`flag > env > config > default`

For security, if `api_key` is in config, file permissions should be `600` on Unix/macOS.

CLI helpers:

```bash
sciencestack config path
sciencestack config init
sciencestack config init --force
```

## Quickstart

```bash
sciencestack capabilities
sciencestack schema overview
sciencestack health
sciencestack --strict overview 1706.03762
sciencestack overview 1706.03762
sciencestack search "transformers" --limit 5
```

## 30-second agent bootstrap

Use this sequence in agent runtimes:

```bash
# 1) Discover CLI contract
sciencestack capabilities
sciencestack schema

# 2) Verify auth + upstream health
sciencestack health

# 3) Execute task command
sciencestack --output json nodes 1706.03762 --type equation --limit 5
```

## Agent-first contract

Default output is strict JSON envelope:

```json
{
  "ok": true,
  "service": "sciencestack",
  "protocolVersion": "1",
  "command": "search",
  "data": { "...": "..." },
  "meta": { "...": "..." }
}
```

Payloads are normalized for agent ergonomics:

- API nested object payloads are flattened (e.g. `data.title`, not `data.data.title`)
- API `_version` is surfaced as `meta.version`

Example normalization:

```json
// API-ish shape
{
  "arxivId": "1706.03762v7",
  "_version": "1.0.0",
  "data": {
    "title": "Attention Is All You Need"
  }
}
```

```json
// CLI envelope shape
{
  "ok": true,
  "service": "sciencestack",
  "protocolVersion": "1",
  "command": "overview",
  "data": {
    "arxivId": "1706.03762v7",
    "title": "Attention Is All You Need"
  },
  "meta": {
    "version": "1.0.0"
  }
}
```

Errors are deterministic:

```json
{
  "ok": false,
  "service": "sciencestack",
  "protocolVersion": "1",
  "command": "search",
  "error": {
    "code": "RATE_LIMITED",
    "message": "Try later",
    "status": 429,
    "retryable": true,
    "exitCode": 11
  }
}
```

For batch/stream use:
```bash
sciencestack --output ndjson overview 1706.03762,2301.07041
```

Each line is a full envelope with `meta.streamIndex`.

Cursor-based pagination contract for list commands:
```bash
sciencestack citations 1706.03762 --limit 10 --cursor 0
```

Machine output includes:
- `meta.pagination.cursor`
- `meta.pagination.nextCursor`
- `meta.pagination.pageSize`
- `meta.pagination.hasMore` (when available)

## Main commands

- `search <query>`
- `overview <paper_id>`
- `nodes <paper_id>`
- `content <paper_id>`
- `refs <paper_id>`
- `citations <paper_id>`
- `authors <author_id>`
- `getartifacts [--type ... --field ... --limit ... --cursor ...]`
- `getartifact <slug>`
- `batch-nodes --requests '<json>' --format raw`
- `health`
- `capabilities`
- `schema [command_name]`
- `config path`
- `config init [--force]`

## Output modes

- `--output json` (default): machine-friendly envelope
- `--output ndjson`: one envelope per line (stream-friendly)
- `--output human`: compact text for interactive use

## Transport and performance

- `--timeout N`: request timeout in seconds
- `--retries N`: retries for transient failures
- `--retry-backoff-ms N`: base retry backoff in milliseconds
- `--max-concurrency N`: parallel fan-out for multi-paper and batch commands
- `--strict`: validate output against declared schema (fails with `CONTRACT_VIOLATION` if shape drifts)

## Protocol

- `--protocol-version 1` is supported.
- `capabilities` + `schema` are callable without API key for agent bootstrap.
- Multi-paper fetch is supported for `overview`, `nodes`, `content`, `refs`, `citations`, and `batch-nodes`.

## Stability policy

- Contract changes are versioned by `protocolVersion`.
- Existing `protocolVersion=1` behavior should remain stable.
- Breaking schema changes should introduce a new protocol version rather than silently mutating existing fields.

## Error + retry semantics

| Condition | Exit code | Retryable |
|---|---:|---:|
| Config/validation error | 2 | No |
| Auth error (`401/403`) | 10 | No |
| Rate limit (`429`) | 11 | Yes |
| Not found (`404`) | 12 | No |
| Timeout/network | 13 | Yes |
| Server error (`5xx`) | 15 | Usually |
| Contract violation (`--strict`) | 17 | No |

`error.retryable` in output is the source of truth for automation loops.

## MCP-style batch nodes

For near MCP parity on node fetches, pass request arrays:

```bash
sciencestack --output ndjson batch-nodes \
  --format raw \
  --requests '[{"paperId":"1706.03762","nodeIds":["eq:1"]},{"paperId":"1706.03762","types":["equation"]}]'
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p "test_*.py" -q
```

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e . --no-build-isolation
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p "test_*.py" -q
```

---

This CLI is intentionally small: stable contracts, predictable errors, and practical commands over framework-heavy complexity.
