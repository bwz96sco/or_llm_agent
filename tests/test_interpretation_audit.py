from __future__ import annotations

import csv
import json
from pathlib import Path

from or_llm_agent.audit_cli import main
from or_llm_agent.benchmark_audit import execute_submission, sha256_file, write_json, write_jsonl
from or_llm_agent.interpretation_audit import (
    INTERPRETATION_ANCHOR_FIELDS,
    VARIANT_ID_RE,
    InterpretationArtifactError,
    _model_ir_payload,
    build_interpretation_prompt,
    diagnose_interpretation_batch,
    validate_adjudication_artifacts,
    validate_interpretation_artifacts,
)


STATEMENT = "Allocate at least one unit; quantities may be continuous or whole numbers."


def _submission(coefficient: float, *, tuple_return: bool = False) -> str:
    build_return = "return model, {'x': x}" if tuple_return else "return model"
    solve_model = "built = build_model(domain)\n    model = built[0] if isinstance(built, tuple) else built"
    return f'''import json
import gurobipy as gp
from gurobipy import GRB

def build_model(domain: str) -> gp.Model:
    model = gp.Model()
    model.Params.OutputFlag = 0
    vtype = GRB.INTEGER if domain == "integer" else GRB.CONTINUOUS
    x = model.addVar(lb=0.0, vtype=vtype, name="x")
    model.addConstr(x >= 1.0, name="minimum")
    model.setObjective({coefficient} * x, GRB.MINIMIZE)
    {build_return}

def solve(domain: str) -> dict:
    {solve_model}
    model.Params.DualReductions = 0
    model.Params.InfUnbdInfo = 1
    model.optimize()
    result = {{
        "status": "optimal" if model.Status == GRB.OPTIMAL else str(model.Status),
        "objective": float(model.ObjVal) if model.Status == GRB.OPTIMAL else None,
        "variables": {{var.VarName: float(var.X) for var in model.getVars()}} if model.Status == GRB.OPTIMAL else {{}},
        "max_constraint_violation": float(model.ConstrVio) if model.Status == GRB.OPTIMAL else None,
        "iis_constraints": [],
    }}
    return result

if __name__ == "__main__":
    print(json.dumps({{"continuous": solve("continuous"), "integer": solve("integer")}}, sort_keys=True))
'''


def _write_variant(
    workspace: Path,
    *,
    row_id: str,
    source_sha256: str,
    variant_id: str,
    domain: str,
    coefficient: float,
    tuple_return: bool = False,
) -> dict[str, object]:
    variant_dir = workspace / "variants" / variant_id
    variant_dir.mkdir(parents=True)
    write_json(
        variant_dir / "problem.json",
        {
            "row_id": row_id,
            "source_sha256": source_sha256,
            "variant_id": variant_id,
            "variables": [{"name": "x", "source_quote": STATEMENT}],
            "objective": {"coefficient": coefficient, "source_quote": STATEMENT},
            "constraints": [{"name": "minimum", "source_quote": STATEMENT}],
            "assumptions": [],
            "domain_evidence": {"domain": domain, "source_quote": STATEMENT},
        },
    )
    (variant_dir / "submission.py").write_text(
        _submission(coefficient, tuple_return=tuple_return),
        encoding="utf-8",
    )
    report = execute_submission(variant_dir, Path(__file__).resolve().parents[1])
    write_json(variant_dir / "solver_report.json", report)
    return {
        "variant_id": variant_id,
        "domain": domain,
        "changed_assumption": f"Use {domain} quantities",
        "material": True,
        "source_quotes": [STATEMENT],
    }


def _workspace(tmp_path: Path) -> tuple[Path, str, str]:
    workspace = tmp_path / "row"
    workspace.mkdir()
    row_id = "industryor-row-0001"
    source_path = workspace / "statement.txt"
    source_path.write_text(STATEMENT, encoding="utf-8")
    source_sha256 = sha256_file(source_path)
    variants = [
        _write_variant(
            workspace,
            row_id=row_id,
            source_sha256=source_sha256,
            variant_id="continuous",
            domain="continuous",
            coefficient=1.0,
        ),
        _write_variant(
            workspace,
            row_id=row_id,
            source_sha256=source_sha256,
            variant_id="integer",
            domain="integer",
            coefficient=2.0,
        ),
    ]
    write_json(
        workspace / "interpretation_set.json",
        {
            "row_id": row_id,
            "source_sha256": source_sha256,
            "enumeration_status": "within_cap",
            "summary": "Two source-plausible domains.",
            "variants": variants,
        },
    )
    with (workspace / "source_anchors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERPRETATION_ANCHOR_FIELDS)
        writer.writeheader()
        for variant in variants:
            writer.writerow(
                {
                    "variant_id": variant["variant_id"],
                    "element_id": "domain",
                    "element_type": "variable_domain",
                    "generated_element": variant["changed_assumption"],
                    "source_quote": STATEMENT,
                    "support_label": "ambiguous",
                    "materiality": "material",
                    "notes": "",
                }
            )
    return workspace, row_id, source_sha256


def test_interpretation_prompt_is_answer_blind_and_caps_variants(tmp_path: Path) -> None:
    prompt = build_interpretation_prompt(row_id="row-1", source_sha256="abc", project_root=tmp_path)

    assert "at most 4 variants" in prompt
    assert "No benchmark answer is available" in prompt
    assert "Do not create an answer verdict" in prompt
    assert "1 to 64 characters" in prompt
    assert "(model, helper_data)" in prompt


def test_variant_id_contract_accepts_64_safe_characters() -> None:
    assert VARIANT_ID_RE.fullmatch("a" + "b" * 63)
    assert not VARIANT_ID_RE.fullmatch("a" + "b" * 64)
    assert not VARIANT_ID_RE.fullmatch("../variant")


def test_model_ir_accepts_model_tuple_and_records_shape(tmp_path: Path) -> None:
    submission = tmp_path / "submission.py"
    submission.write_text(_submission(1.0, tuple_return=True), encoding="utf-8")

    models, return_shapes = _model_ir_payload(
        submission,
        Path(__file__).resolve().parents[1],
    )

    assert set(models) == {"continuous", "integer"}
    assert return_shapes == {"continuous": "model_tuple", "integer": "model_tuple"}


def test_model_ir_classifies_quadratic_constraint(tmp_path: Path) -> None:
    submission = tmp_path / "submission.py"
    submission.write_text(
        '''import gurobipy as gp

def build_model(domain: str):
    model = gp.Model()
    model.Params.OutputFlag = 0
    x = model.addVar(lb=0.0, name="x")
    y = model.addVar(lb=0.0, name="y")
    model.addQConstr(x * y <= 1.0, name="quadratic")
    model.setObjective(x + y)
    return model
''',
        encoding="utf-8",
    )

    try:
        _model_ir_payload(submission, Path(__file__).resolve().parents[1])
    except InterpretationArtifactError as exc:
        assert exc.code == "unsupported_model_feature"
    else:
        raise AssertionError("expected unsupported quadratic feature")


def test_interpretation_validation_parent_replays_and_fingerprints(tmp_path: Path) -> None:
    workspace, row_id, source_sha256 = _workspace(tmp_path)

    result = validate_interpretation_artifacts(
        workspace,
        row_id=row_id,
        source_sha256=source_sha256,
        statement=STATEMENT,
        project_root=Path(__file__).resolve().parents[1],
    )

    assert result["status"] == "valid"
    assert result["valid_variant_count"] == 2
    assert result["material_multi_variant"] is True
    assert len({variant["model_ir_fingerprint"] for variant in result["variants"]}) == 2
    assert (workspace / "variants" / "integer" / "parent_model_ir.json").is_file()


def test_diagnostic_replay_writes_sidecar_without_mutating_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "source-run"
    (run_dir / "rows").mkdir(parents=True)
    workspace, row_id, source_sha256 = _workspace(run_dir / "rows")
    workspace.rename(workspace.parent / row_id)
    workspace = workspace.parent / row_id
    write_json(
        workspace / "run-input.json",
        {
            "row_id": row_id,
            "split": "diagnostic",
            "statement_path": "source/row.txt",
            "source_sha256": source_sha256,
        },
    )
    write_json(
        workspace / "run-manifest.json",
        {
            "row_id": row_id,
            "returncode": 0,
            "timed_out": False,
            "terminal_status": "success",
            "leakage_findings": [],
        },
    )
    source_files_before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file())

    summary = diagnose_interpretation_batch(
        run_dir,
        tmp_path / "diagnostic",
        Path(__file__).resolve().parents[1],
        expected_valid=1,
        expected_failures={},
    )

    source_files_after = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file())
    assert summary["gate_pass"] is True
    assert summary["source_immutable"] is True
    assert source_files_after == source_files_before


def test_interpretation_validation_rejects_more_than_four_variants(tmp_path: Path) -> None:
    workspace, row_id, source_sha256 = _workspace(tmp_path)
    payload = json.loads((workspace / "interpretation_set.json").read_text(encoding="utf-8"))
    payload["variants"] = payload["variants"] * 3
    write_json(workspace / "interpretation_set.json", payload)

    result = validate_interpretation_artifacts(
        workspace,
        row_id=row_id,
        source_sha256=source_sha256,
        statement=STATEMENT,
        project_root=Path(__file__).resolve().parents[1],
    )

    assert result["status"] == "invalid"
    assert any("1..4" in error for error in result["errors"])


def test_adjudication_validation_requires_exact_variant_accounting(tmp_path: Path) -> None:
    workspace, row_id, source_sha256 = _workspace(tmp_path)
    packet = {"variants": [{"variant_id": "continuous"}, {"variant_id": "integer"}]}
    write_json(
        workspace / "adjudication.json",
        {
            "row_id": row_id,
            "source_sha256": source_sha256,
            "variants": [
                {
                    "variant_id": variant_id,
                    "support_status": "underdetermined",
                    "material": True,
                    "source_quotes": [STATEMENT],
                    "rationale": "The source permits this domain.",
                }
                for variant_id in ("continuous", "integer")
            ],
            "obvious_omission": False,
            "omission_note": "",
            "omission_source_quotes": [],
        },
    )

    result = validate_adjudication_artifacts(
        workspace,
        row_id=row_id,
        source_sha256=source_sha256,
        statement=STATEMENT,
        packet=packet,
    )

    assert result["status"] == "valid"


def test_interpretation_cli_dry_run_starts_no_sessions(tmp_path: Path, capsys) -> None:
    campaign = tmp_path / "campaign"
    source_dir = campaign / "source"
    source_dir.mkdir(parents=True)
    statement = source_dir / "row-1.txt"
    statement.write_text(STATEMENT, encoding="utf-8")
    manifest = source_dir / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "row_id": "row-1",
                "split": "development",
                "statement_path": "source/row-1.txt",
                "source_sha256": sha256_file(statement),
            }
        ],
    )

    exit_code = main(
        [
            "run-interpretation-batch",
            "--manifest",
            str(manifest),
            "--campaign-dir",
            str(campaign),
            "--out-dir",
            str(tmp_path / "runs"),
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
    assert payload["role"] == "terra"
    assert payload["sessions"] == 1
    assert not (tmp_path / "runs").exists()
