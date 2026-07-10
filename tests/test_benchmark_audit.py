from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from or_llm_agent.audit_cli import main
from or_llm_agent.benchmark_audit import (
    AuditOptions,
    AuditRunnerError,
    POLICY,
    SOURCE_ANCHOR_FIELDS,
    build_codex_command,
    load_source_manifest,
    run_batch,
    sha256_file,
    validate_policy,
)


def _source_manifest(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    campaign_dir = tmp_path / "campaign"
    source_dir = campaign_dir / "source"
    source_dir.mkdir(parents=True)
    statement = source_dir / "row-1.txt"
    statement.write_text("Maximize x subject to x <= 5.\n", encoding="utf-8")
    rows = [
        {
            "row_id": "row-1",
            "split": "pilot_evidence",
            "statement_path": "source/row-1.txt",
            "source_sha256": sha256_file(statement),
        }
    ]
    manifest = source_dir / "manifest.jsonl"
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    return campaign_dir, manifest, rows


def test_policy_requires_frozen_model_and_effort() -> None:
    validate_policy(AuditOptions("terra", *POLICY["terra"], timeout_seconds=10))
    validate_policy(AuditOptions("sol", *POLICY["sol"], timeout_seconds=10))
    with pytest.raises(AuditRunnerError, match="requires model"):
        validate_policy(AuditOptions("terra", "gpt-5.6-sol", "xhigh", timeout_seconds=10))


def test_command_is_one_fresh_non_resume_exec(tmp_path: Path) -> None:
    options = AuditOptions("terra", *POLICY["terra"], timeout_seconds=10)
    command = build_codex_command(
        workspace=tmp_path,
        last_message_path=tmp_path / "last.md",
        options=options,
    )

    assert command.count("exec") == 1
    assert "resume" not in command
    assert "--ignore-user-config" in command
    assert command[command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in command


def test_manifest_rejects_hidden_fields(tmp_path: Path) -> None:
    campaign_dir, manifest, rows = _source_manifest(tmp_path)
    rows[0]["answer"] = "5"
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    with pytest.raises(AuditRunnerError, match="fields must be exactly"):
        load_source_manifest(manifest, campaign_dir)


def _write_success_artifacts(workspace: Path, row_id: str, source_sha256: str) -> None:
    problem = {
        "row_id": row_id,
        "source_sha256": source_sha256,
        "variables": [{"name": "x", "source_quote": "x"}],
        "objective": {"sense": "maximize", "source_quote": "Maximize x"},
        "constraints": [{"expression": "x <= 5", "source_quote": "x <= 5"}],
        "assumptions": [],
        "domain_evidence": [],
    }
    audit = {
        "row_id": row_id,
        "source_sha256": source_sha256,
        "source_status": "ambiguous",
        "chosen_domain": "both",
        "mechanisms": ["domain_unspecified"],
        "material_ambiguities": ["Integrality is not explicit."],
        "summary": "Both domains agree.",
    }
    submission = '''
import json
import gurobipy as gp
from gurobipy import GRB

def build_model(domain: str) -> gp.Model:
    model = gp.Model("fixture")
    model.Params.OutputFlag = 0
    kind = GRB.CONTINUOUS if domain == "continuous" else GRB.INTEGER
    x = model.addVar(lb=0, vtype=kind, name="x")
    model.setObjective(x, GRB.MAXIMIZE)
    model.addConstr(x <= 5, name="capacity")
    return model

def solve(domain: str) -> dict:
    model = build_model(domain)
    model.Params.DualReductions = 0
    model.Params.InfUnbdInfo = 1
    model.optimize()
    return {
        "status": "optimal",
        "objective": float(model.ObjVal),
        "variables": {var.VarName: float(var.X) for var in model.getVars()},
        "max_constraint_violation": float(model.ConstrVio),
        "iis_constraints": [],
    }

if __name__ == "__main__":
    print(json.dumps({"continuous": solve("continuous"), "integer": solve("integer")}))
'''.lstrip()
    report = {
        "continuous": {
            "status": "optimal",
            "objective": 5.0,
            "variables": {"x": 5.0},
            "max_constraint_violation": 0.0,
            "iis_constraints": [],
        },
        "integer": {
            "status": "optimal",
            "objective": 5.0,
            "variables": {"x": 5.0},
            "max_constraint_violation": 0.0,
            "iis_constraints": [],
        },
    }
    (workspace / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    (workspace / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (workspace / "submission.py").write_text(submission, encoding="utf-8")
    (workspace / "solver_report.json").write_text(json.dumps(report), encoding="utf-8")
    with (workspace / "source_anchors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_ANCHOR_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "element_id": "objective",
                "element_type": "objective",
                "generated_element": "maximize x",
                "source_quote": "Maximize x",
                "support_label": "supported",
                "materiality": "major",
                "notes": "",
            }
        )


def test_run_batch_records_single_session_and_parent_reexecution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign_dir, manifest, _ = _source_manifest(tmp_path)
    rows = load_source_manifest(manifest, campaign_dir)
    output_dir = tmp_path / "runs"

    def fake_invoke(command: list[str], prompt: str, timeout_seconds: int) -> tuple[int, bool, str, str]:
        workspace = Path(command[command.index("-C") + 1])
        payload = json.loads((workspace / "run-input.json").read_text(encoding="utf-8"))
        _write_success_artifacts(workspace, payload["row_id"], payload["source_sha256"])
        Path(command[command.index("-o") + 1]).write_text("complete\n", encoding="utf-8")
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-test"}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
            ]
        )
        return 0, False, stdout, ""

    monkeypatch.setattr("or_llm_agent.benchmark_audit._invoke_codex", fake_invoke)
    options = AuditOptions("terra", *POLICY["terra"], timeout_seconds=10)

    results = run_batch(
        rows,
        campaign_dir=campaign_dir,
        output_dir=output_dir,
        project_root=Path(__file__).resolve().parents[1],
        options=options,
        max_sessions=1,
        concurrency=1,
    )

    assert results[0]["terminal_status"] == "success"
    assert results[0]["codex_run_metadata"]["codex_thread_id"] == "thread-test"
    assert results[0]["codex_run_metadata"]["codex_usage"]["output_tokens"] == 5
    assert (output_dir / "rows" / "row-1" / "parent_solver_report.json").is_file()
    with pytest.raises(AuditRunnerError, match="refusing duplicate session"):
        run_batch(
            rows,
            campaign_dir=campaign_dir,
            output_dir=output_dir,
            project_root=Path(__file__).resolve().parents[1],
            options=options,
            max_sessions=1,
            concurrency=1,
        )


def test_cli_dry_run_starts_no_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    campaign_dir, manifest, _ = _source_manifest(tmp_path)
    exit_code = main(
        [
            "run-batch",
            "--manifest",
            str(manifest),
            "--campaign-dir",
            str(campaign_dir),
            "--out-dir",
            str(tmp_path / "runs"),
            "--role",
            "terra",
            "--model",
            "gpt-5.6-terra",
            "--effort",
            "high",
            "--max-sessions",
            "1",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["sessions"] == 1
    assert not (tmp_path / "runs").exists()
