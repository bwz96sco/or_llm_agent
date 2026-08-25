from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from or_llm_agent.codex_agent import CodexAgentOptions
from or_llm_agent.mps_repair import build_mps_repair_prompt, run_mps_repair, validate_mps_repair_payload


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
    schema = workspace / "repair_schema.json"
    schema.write_text(json.dumps({"required": ["operation", "constraint_name"]}), encoding="utf-8")
    (workspace / "problem.md").write_text("Repair infeasible model.\n", encoding="utf-8")
    (workspace / "faulty.mps").write_text("NAME TEST\nENDATA\n", encoding="utf-8")
    (workspace / "solver_state.json").write_text(
        json.dumps({"solver_status": "INFEASIBLE", "iis_constraints": ["root", "symptom"]}),
        encoding="utf-8",
    )
    return workspace, schema


def test_payload_requires_exact_visible_action() -> None:
    visible = ["root", "symptom"]
    assert validate_mps_repair_payload(
        {"operation": "remove_constraint", "constraint_name": "root"}, visible
    ) == "valid_action"
    assert validate_mps_repair_payload(
        {"operation": "remove_constraint", "constraint_name": "other"}, visible
    ) == "constraint_not_in_visible_iis"
    assert validate_mps_repair_payload(
        {"operation": "remove_constraint", "constraint_name": "root", "extra": 1}, visible
    ) == "invalid_action_shape"


def test_prompt_contains_no_hidden_answer(tmp_path: Path) -> None:
    workspace, schema = make_workspace(tmp_path)
    (tmp_path / "oracle.json").write_text('{"target": "root", "objective": 99}\n', encoding="utf-8")
    output = tmp_path / "artifacts" / "outputs" / "repair.json"
    prompt = build_mps_repair_prompt(
        case_id="opaque-1",
        workspace=workspace,
        schema_path=schema,
        output_path=output,
    )
    assert "99" not in prompt
    assert "oracle.json" not in prompt
    assert str(output) in prompt


def test_run_captures_valid_action(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, schema = make_workspace(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    output = artifact_dir / "outputs" / "repair.json"

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"operation":"remove_constraint","constraint_name":"root"}\n', encoding="utf-8")
        last_message = Path(command[command.index("-o") + 1])
        last_message.parent.mkdir(parents=True, exist_ok=True)
        last_message.write_text("done\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"type":"done"}\n', stderr="")

    monkeypatch.setattr("or_llm_agent.mps_repair.subprocess.run", fake_run)
    result = run_mps_repair(
        case_id="opaque-1",
        workspace=workspace,
        schema_path=schema,
        output_path=output,
        artifact_dir=artifact_dir,
        options=options(),
    )
    assert result.output_status == "valid_action"
    assert result.events_path.is_file()
    assert json.loads(result.status_path.read_text(encoding="utf-8"))["returncode"] == 0
