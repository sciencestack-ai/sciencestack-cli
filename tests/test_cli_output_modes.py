import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

import sciencestack.cli as cli_mod
from sciencestack.client import ScienceStackError


def _parse_first_json_blob(text: str):
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text.lstrip())
    return payload


class FakeClient:
    def __init__(self, api_key: str, base_url: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = kwargs.get("timeout")
        self.retries = kwargs.get("retries")
        self.retry_backoff_ms = kwargs.get("retry_backoff_ms")

    def search(self, query: str, **kwargs):
        return {
            "data": [
                {
                    "arxivId": "1234.5678",
                    "title": "Test Paper",
                    "authors": ["Author A"],
                    "published": "2024-02-03",
                }
            ]
        }

    def batch_overview(self, paper_ids: list[str]):
        return {
            "data": [
                {"data": {"arxivId": paper_ids[0], "title": "Paper One"}},
                {"data": {"arxivId": paper_ids[1], "title": "Paper Two"}},
            ]
        }

    def overview(self, paper_id: str):
        return {"data": {"arxivId": paper_id, "title": "Health Probe"}}

    def nodes(self, paper_id: str, **kwargs):
        return {"data": f"nodes-for-{paper_id}"}

    def content(self, paper_id: str, **kwargs):
        return {"data": {"content": f"content-for-{paper_id}"}}

    def references(self, paper_id: str, **kwargs):
        return {"data": [{"citeKey": f"ref-{paper_id}"}]}

    def citations(self, paper_id: str, **kwargs):
        return {"data": [{"arxivId": f"cite-{paper_id}"}]}

    def author_papers(self, author_id: str, **kwargs):
        return {"data": [{"arxivId": f"author-{author_id}"}]}

    def artifacts(self, **kwargs):
        return {
            "data": [{"slug": "weekly-rl-2026-w05", "title": "Weekly RL Digest"}],
            "pagination": {"hasMore": False, "offset": kwargs.get("offset", 0), "limit": kwargs.get("limit", 20)},
        }

    def artifact(self, slug: str):
        return {"data": {"slug": slug, "breakthroughs": []}}


class ErrorClient(FakeClient):
    def search(self, query: str, **kwargs):
        raise ScienceStackError("RATE_LIMITED", "Try later", 429)


class RecordingClient(FakeClient):
    last_api_key = None
    last_base_url = None
    last_timeout = None
    last_retries = None
    last_retry_backoff_ms = None

    def __init__(self, api_key: str, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        RecordingClient.last_api_key = api_key
        RecordingClient.last_base_url = base_url
        RecordingClient.last_timeout = kwargs.get("timeout")
        RecordingClient.last_retries = kwargs.get("retries")
        RecordingClient.last_retry_backoff_ms = kwargs.get("retry_backoff_ms")


class CursorClient(FakeClient):
    def search(self, query: str, **kwargs):
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 10)
        return {
            "data": [{"arxivId": f"cursor-{offset}"}],
            "pagination": {"hasMore": True, "offset": offset, "limit": limit},
        }


class BadShapeClient(FakeClient):
    def search(self, query: str, **kwargs):
        return {"unexpected": "shape"}


class CliOutputModeTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_search_defaults_to_json(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(cli_mod.main, ["search", "transformers"], env={"SCIENCESTACK_API_KEY": "k"})

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "sciencestack")
        self.assertEqual(payload["protocolVersion"], "1")
        self.assertEqual(payload["command"], "search")
        self.assertEqual(payload["data"]["data"][0]["arxivId"], "1234.5678")

    def test_search_human_output(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["--output", "human", "search", "transformers"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Test Paper", result.output)
        self.assertTrue(result.output.strip().startswith("1234.5678"))

    def test_batch_overview_ndjson(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["--output", "ndjson", "overview", "1111.1111,2222.2222"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        first = _parse_first_json_blob(lines[0])
        second = _parse_first_json_blob(lines[1])
        self.assertEqual(first["command"], "overview")
        self.assertEqual(first["data"]["paperId"], "1111.1111")
        self.assertEqual(second["data"]["paperId"], "2222.2222")

    def test_missing_api_key_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.runner.invoke(
                cli_mod.main,
                ["search", "transformers"],
                env={
                    "SCIENCESTACK_API_KEY": "",
                    "SCIENCESTACK_CONFIG": str(Path(td) / "no_config.json"),
                },
            )

        self.assertEqual(result.exit_code, 2)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["error"]["code"], "CONFIG")
        self.assertFalse(payload["error"]["retryable"])
        self.assertEqual(payload["service"], "sciencestack")
        self.assertEqual(payload["protocolVersion"], "1")

    def test_api_error_maps_to_deterministic_exit_code(self):
        with patch.object(cli_mod, "ScienceStackClient", ErrorClient):
            result = self.runner.invoke(cli_mod.main, ["search", "transformers"], env={"SCIENCESTACK_API_KEY": "k"})

        self.assertEqual(result.exit_code, 11)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["error"]["status"], 429)
        self.assertTrue(payload["error"]["retryable"])

    def test_capabilities_works_without_api_key(self):
        result = self.runner.invoke(cli_mod.main, ["capabilities"], env={"SCIENCESTACK_API_KEY": ""})

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertTrue(payload["ok"])
        self.assertIn("search", payload["data"]["commands"])
        self.assertIn("protocolVersion", payload["data"])
        self.assertIn("overview", payload["data"]["multiPaperCommands"])

    def test_schema_for_single_command(self):
        result = self.runner.invoke(cli_mod.main, ["schema", "overview"], env={"SCIENCESTACK_API_KEY": ""})

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertIn("overview", payload["data"]["schemas"])
        self.assertNotIn("search", payload["data"]["schemas"])

    def test_health_success(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(cli_mod.main, ["health"], env={"SCIENCESTACK_API_KEY": "k"})

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["data"]["status"], "up")
        self.assertEqual(payload["data"]["probe"]["command"], "overview")

    def test_schema_unknown_command_is_validation_error(self):
        result = self.runner.invoke(cli_mod.main, ["schema", "not_a_cmd"], env={"SCIENCESTACK_API_KEY": ""})

        self.assertEqual(result.exit_code, 2)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["error"]["code"], "VALIDATION")

    def test_protocol_version_flag_roundtrip(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["--protocol-version", "1", "search", "transformers"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["protocolVersion"], "1")

    def test_nodes_multi_paper_json(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["nodes", "1111.1111,2222.2222"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["command"], "nodes")
        self.assertEqual(payload["data"]["count"], 2)
        self.assertEqual(payload["data"]["items"][0]["paperId"], "1111.1111")

    def test_content_multi_paper_ndjson(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["--output", "ndjson", "content", "1111.1111,2222.2222"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        row0 = _parse_first_json_blob(lines[0])
        row1 = _parse_first_json_blob(lines[1])
        self.assertEqual(row0["data"]["paperId"], "1111.1111")
        self.assertEqual(row1["data"]["paperId"], "2222.2222")

    def test_search_cursor_metadata(self):
        with patch.object(cli_mod, "ScienceStackClient", CursorClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["search", "transformers", "--limit", "2", "--cursor", "4"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["meta"]["pagination"]["cursor"], "4")
        self.assertEqual(payload["meta"]["pagination"]["nextCursor"], "6")

    def test_strict_mode_contract_violation(self):
        with patch.object(cli_mod, "ScienceStackClient", BadShapeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["--strict", "search", "transformers"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 17)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["error"]["code"], "CONTRACT_VIOLATION")

    def test_batch_nodes_json(self):
        reqs = json.dumps(
            [
                {"paperId": "1111.1111", "nodeIds": ["eq:1"]},
                {"paperId": "2222.2222", "types": ["equation"]},
            ]
        )
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["batch-nodes", "--requests", reqs, "--format", "raw"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["command"], "batch-nodes")
        self.assertEqual(payload["data"]["count"], 2)
        self.assertEqual(payload["data"]["items"][0]["paperId"], "1111.1111")

    def test_reads_api_key_from_xdg_config(self):
        with tempfile.TemporaryDirectory() as td:
            xdg = Path(td) / ".config" / "sciencestack"
            xdg.mkdir(parents=True, exist_ok=True)
            cfg = xdg / "config.json"
            cfg.write_text(json.dumps({"api_key": "cfg-key"}), encoding="utf-8")
            if os.name != "nt":
                cfg.chmod(0o600)

            with patch.object(cli_mod, "ScienceStackClient", RecordingClient):
                result = self.runner.invoke(
                    cli_mod.main,
                    ["search", "transformers"],
                    env={"HOME": td, "XDG_CONFIG_HOME": str(Path(td) / ".config"), "SCIENCESTACK_API_KEY": ""},
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(RecordingClient.last_api_key, "cfg-key")

    def test_flag_overrides_env_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            xdg = Path(td) / ".config" / "sciencestack"
            xdg.mkdir(parents=True, exist_ok=True)
            cfg = xdg / "config.json"
            cfg.write_text(json.dumps({"api_key": "cfg-key"}), encoding="utf-8")
            if os.name != "nt":
                cfg.chmod(0o600)

            with patch.object(cli_mod, "ScienceStackClient", RecordingClient):
                result = self.runner.invoke(
                    cli_mod.main,
                    ["--api-key", "flag-key", "search", "transformers"],
                    env={"HOME": td, "XDG_CONFIG_HOME": str(Path(td) / ".config"), "SCIENCESTACK_API_KEY": "env-key"},
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(RecordingClient.last_api_key, "flag-key")

    def test_insecure_config_permissions_rejected(self):
        if os.name == "nt":
            self.skipTest("Permission mode semantics differ on Windows")

        with tempfile.TemporaryDirectory() as td:
            xdg = Path(td) / ".config" / "sciencestack"
            xdg.mkdir(parents=True, exist_ok=True)
            cfg = xdg / "config.json"
            cfg.write_text(json.dumps({"api_key": "cfg-key"}), encoding="utf-8")
            cfg.chmod(0o644)

            result = self.runner.invoke(
                cli_mod.main,
                ["search", "transformers"],
                env={"HOME": td, "XDG_CONFIG_HOME": str(Path(td) / ".config"), "SCIENCESTACK_API_KEY": ""},
            )

        self.assertEqual(result.exit_code, 2)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["error"]["code"], "CONFIG")

    def test_config_path_without_api_key(self):
        result = self.runner.invoke(cli_mod.main, ["config", "path"], env={"SCIENCESTACK_API_KEY": ""})

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["command"], "config.path")
        self.assertIn("path", payload["data"])

    def test_config_init_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.runner.invoke(
                cli_mod.main,
                ["config", "init"],
                env={"HOME": td, "XDG_CONFIG_HOME": str(Path(td) / ".config"), "SCIENCESTACK_API_KEY": ""},
            )

            self.assertEqual(result.exit_code, 0)
            payload = _parse_first_json_blob(result.output)
            self.assertEqual(payload["command"], "config.init")
            self.assertTrue(payload["data"]["created"])
            cfg_path = Path(payload["data"]["path"])
            self.assertTrue(cfg_path.exists())

    def test_getartifacts_command(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["getartifacts", "--field", "rl", "--limit", "1", "--cursor", "0"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["command"], "getartifacts")
        self.assertEqual(payload["data"]["data"][0]["slug"], "weekly-rl-2026-w05")

    def test_getartifact_command(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["getartifact", "weekly-rl-2026-w05"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["command"], "getartifact")
        self.assertEqual(payload["data"]["slug"], "weekly-rl-2026-w05")

    def test_overview_flattens_inner_data_for_agents(self):
        with patch.object(cli_mod, "ScienceStackClient", FakeClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["overview", "1111.1111"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        payload = _parse_first_json_blob(result.output)
        self.assertEqual(payload["command"], "overview")
        self.assertEqual(payload["data"]["title"], "Health Probe")
        self.assertNotIn("data", payload["data"])

    def test_transport_flags_flow_to_client(self):
        with patch.object(cli_mod, "ScienceStackClient", RecordingClient):
            result = self.runner.invoke(
                cli_mod.main,
                ["--timeout", "12.5", "--retries", "3", "--retry-backoff-ms", "400", "search", "transformers"],
                env={"SCIENCESTACK_API_KEY": "k"},
            )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(RecordingClient.last_timeout, 12.5)
        self.assertEqual(RecordingClient.last_retries, 3)
        self.assertEqual(RecordingClient.last_retry_backoff_ms, 400)


if __name__ == "__main__":
    unittest.main()
