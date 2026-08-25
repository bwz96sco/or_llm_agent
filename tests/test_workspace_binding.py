from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from or_llm_agent.codex_agent import CodexAgentOptions
from or_llm_agent.workspace_binding import build_workspace_binding_prompt, run_workspace_binding


def options() -> CodexAgentOptions:
    return CodexAgentOptions(
        codex_model="gpt-5.6-sol",
        codex_reasoning_effort="high",
        codex_sandbox="workspace-write",
        codex_approval="never",
        max_repair_attempts=0,
        timeout_seconds=30,
    )


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    schema = workspace / "schema.json"
    schema.write_text(json.dumps({"type": "object", "required": ["cost"]}), encoding="utf-8")
    (workspace / "data.csv").write_text("cost\n3\n", encoding="utf-8")
    return workspace, schema


def test_prompt_exposes_only_visible_workspace_and_schema(tmp_path: Path) -> None:
    workspace, schema = make_workspace(tmp_path)
    (tmp_path / "hidden-answer.json").write_text('{"secret": 9173}\n', encoding="utf-8")
    output = tmp_path / "artifacts" / "outputs" / "case" / "bound_instance.json"
    prompt = build_workspace_binding_prompt(
        case_id="case",
        workspace=workspace,
        schema_path=schema,
        output_path=output,
    )
    assert "data.csv" not in prompt
    assert '"required": [' in prompt
    assert "9173" not in prompt
    assert str(output) in prompt


def test_rejects_schema_outside_workspace(tmp_path: Path) -> None:
    workspace, _ = make_workspace(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inside workspace"):
        run_workspace_binding(
            case_id="case",
            workspace=workspace,
            schema_path=schema,
            output_path=tmp_path / "artifacts" / "bound_instance.json",
            artifact_dir=tmp_path / "artifacts",
            options=options(),
        )


def test_run_captures_valid_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, schema = make_workspace(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    output = artifact_dir / "outputs" / "case" / "bound_instance.json"

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"cost": 3}\n', encoding="utf-8")
        last_message = Path(command[command.index("-o") + 1])
        last_message.parent.mkdir(parents=True, exist_ok=True)
        last_message.write_text("done\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"type":"done"}\n', stderr="")

    monkeypatch.setattr("or_llm_agent.workspace_binding.subprocess.run", fake_run)
    result = run_workspace_binding(
        case_id="case",
        workspace=workspace,
        schema_path=schema,
        output_path=output,
        artifact_dir=artifact_dir,
        options=options(),
    )
    assert result.output_status == "valid_json_object"
    assert json.loads(result.status_path.read_text(encoding="utf-8"))["returncode"] == 0
    assert result.events_path.is_file()
