from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from or_llm_agent.benchmark_audit import (
    AuditOptions,
    AuditRunnerError,
    SOURCE_ANCHOR_FIELDS,
    _invoke_codex,
    _parse_last_json_line,
    _problem_source_quote_errors,
    _same_solver_report,
    _trace_leakage,
    _validate_solver_result,
    build_codex_command,
    build_run_metadata,
    execute_submission,
    sha256_file,
    source_quote_is_exact,
    validate_policy,
    write_json,
    write_jsonl,
)
from or_llm_agent.redaction import redact_text


MAX_INTERPRETATIONS = 4
INTERPRETATION_ANCHOR_FIELDS = ["variant_id", *SOURCE_ANCHOR_FIELDS]
INTERPRETATION_SET_FIELDS = {
    "row_id",
    "source_sha256",
    "enumeration_status",
    "summary",
    "variants",
}
VARIANT_FIELDS = {
    "variant_id",
    "domain",
    "changed_assumption",
    "material",
    "source_quotes",
}
VARIANT_REQUIRED_FILES = ("problem.json", "submission.py", "solver_report.json")
ADJUDICATION_MANIFEST_FIELDS = {
    "row_id",
    "split",
    "statement_path",
    "source_sha256",
    "packet_path",
    "packet_sha256",
}
ADJUDICATION_FIELDS = {
    "row_id",
    "source_sha256",
    "variants",
    "obvious_omission",
    "omission_note",
    "omission_source_quotes",
}
ADJUDICATION_VARIANT_FIELDS = {
    "variant_id",
    "support_status",
    "material",
    "source_quotes",
    "rationale",
}
TERMINAL_STATES = {
    "success",
    "model_failed",
    "artifact_invalid",
    "solver_failed",
    "leakage_blocked",
    "timed_out",
}
VARIANT_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_interpretation_prompt(*, row_id: str, source_sha256: str, project_root: Path) -> str:
    anchor_header = ",".join(INTERPRETATION_ANCHOR_FIELDS)
    verify_prefix = shlex.join(["uv", "run", "--project", str(project_root), "python"])
    return f"""You are one independent optimization interpretation auditor. Complete the full row in this one Codex session.

Input boundary:
- Read only `statement.txt` and files you create in this workspace.
- Do not inspect parent directories, repositories, benchmark datasets, prior outputs, answer keys, or the internet.
- No benchmark answer is available. Do not guess or search for one.
- Row ID: `{row_id}`
- Required statement SHA-256: `{source_sha256}`

Task:
1. Read the statement and identify only materially different mathematical interpretations directly supported or left open by its wording.
2. Emit one primary interpretation. Add minimal alternatives only when an exact source phrase permits them.
3. Emit at most {MAX_INTERPRETATIONS} variants. Do not create cosmetic rewrites or arbitrary assumptions.
4. Give each variant exactly one selected variable domain: `continuous` or `integer`. If both are plausible, emit separate variants.
5. Implement every variant in Gurobi and solve both continuous and nonnegative-integer forms for parent checking.
6. Preserve ambiguity. If more than four variants are needed, set `enumeration_status` to `truncated`. If coverage is doubtful, use `uncertain`.

Write `interpretation_set.json` with exactly these root fields:
- row_id, source_sha256, enumeration_status, summary, variants
- enumeration_status is one of `within_cap`, `truncated`, `uncertain`.
- variants is a list of one to four objects with exactly: variant_id, domain, changed_assumption, material, source_quotes.
- variant_id uses lowercase letters, digits, `_`, or `-`.
- material is a JSON boolean.
- source_quotes is a non-empty list of character-for-character statement spans.

For each variant `<id>`, write:
- `variants/<id>/problem.json`: row_id, source_sha256, variant_id, variables, objective, constraints, assumptions, domain_evidence. Every modeled element includes an exact source_quote.
- `variants/<id>/submission.py`: self-contained stdlib + gurobipy; define `build_model(domain: str)` for `continuous` and `integer`; solve quietly in `__main__`; print one final JSON object with continuous and integer results.
- `variants/<id>/solver_report.json`: exact parsed output from the final successful submission execution.

Each domain result contains status, objective, variables, max_constraint_violation, and iis_constraints. Use OutputFlag=0, DualReductions=0, and InfUnbdInfo=1. Compute an IIS when infeasible.

Write `source_anchors.csv` with exact header:
`{anchor_header}`
Use one or more rows per variant. support_label is `supported` or `ambiguous`. Every source_quote must be an exact statement span. Preserve backslashes, punctuation, and spaces after JSON or CSV parsing.

Run every variant before finishing, for example:
```bash
{verify_prefix} variants/<id>/submission.py
```

Do not create an answer verdict. The parent compares hidden answers only after all Terra sessions terminate. Finish with a concise status message. Do not ask questions.
"""


def _model_ir_payload(submission: Path, project_root: Path, timeout_seconds: int = 120) -> dict[str, Any]:
    script = """
import importlib.util
import json
import sys
from pathlib import Path
from or_ci.model_ir import extract_model_ir

path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("interpretation_submission", path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load submission")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_model = getattr(module, "build_model", None)
if not callable(build_model):
    raise RuntimeError("submission must define build_model(domain)")
payload = {}
for domain in ("continuous", "integer"):
    model = build_model(domain)
    payload[domain] = extract_model_ir(model).to_dict()
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        ["uv", "run", "--project", str(project_root), "python", "-c", script, str(submission)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise AuditRunnerError(f"parent ModelIR extraction failed: {result.stderr.strip()}")
    payload = _parse_last_json_line(result.stdout)
    if not isinstance(payload, dict) or set(payload) != {"continuous", "integer"}:
        raise AuditRunnerError("parent ModelIR output must have continuous and integer keys")
    return payload


def _selected_result(report: dict[str, Any], domain: str) -> dict[str, Any]:
    result = report.get(domain)
    if not isinstance(result, dict):
        raise AuditRunnerError(f"missing selected domain result: {domain}")
    return result


def _outcome_key(result: dict[str, Any]) -> tuple[str, float | None]:
    status = str(result.get("status", "")).strip().lower()
    objective = result.get("objective")
    return status, float(objective) if isinstance(objective, (int, float)) else None


def _material_multi_variant(variants: list[dict[str, Any]]) -> bool:
    material = [variant for variant in variants if variant["material"]]
    fingerprints = {variant["model_ir_fingerprint"] for variant in material}
    outcomes = {_outcome_key(variant["selected_result"]) for variant in material}
    return len(fingerprints) >= 2 and len(outcomes) >= 2


def validate_interpretation_artifacts(
    workspace: Path,
    *,
    row_id: str,
    source_sha256: str,
    statement: str,
    project_root: Path,
) -> dict[str, Any]:
    set_path = workspace / "interpretation_set.json"
    anchors_path = workspace / "source_anchors.csv"
    if not set_path.is_file() or not anchors_path.is_file():
        return {"status": "invalid", "errors": ["missing interpretation_set.json or source_anchors.csv"]}
    errors: list[str] = []
    try:
        payload = json.loads(set_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "errors": [f"interpretation_set.json: {exc}"]}
    if not isinstance(payload, dict) or set(payload) != INTERPRETATION_SET_FIELDS:
        return {"status": "invalid", "errors": ["interpretation_set.json: invalid root fields"]}
    if payload.get("row_id") != row_id or payload.get("source_sha256") != source_sha256:
        errors.append("interpretation_set.json: row_id or source_sha256 mismatch")
    if payload.get("enumeration_status") not in {"within_cap", "truncated", "uncertain"}:
        errors.append("interpretation_set.json: invalid enumeration_status")
    if not isinstance(payload.get("summary"), str):
        errors.append("interpretation_set.json: summary must be a string")
    variants = payload.get("variants")
    if not isinstance(variants, list) or not 1 <= len(variants) <= MAX_INTERPRETATIONS:
        errors.append(f"interpretation_set.json: variants must contain 1..{MAX_INTERPRETATIONS} rows")
        variants = []
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        variant_error_count = len(errors)
        prefix = f"interpretation_set.json.variants[{index}]"
        if not isinstance(variant, dict) or set(variant) != VARIANT_FIELDS:
            errors.append(f"{prefix}: invalid fields")
            continue
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or not VARIANT_ID_RE.fullmatch(variant_id):
            errors.append(f"{prefix}: invalid variant_id")
            continue
        if variant_id in seen_ids:
            errors.append(f"{prefix}: duplicate variant_id")
            continue
        seen_ids.add(variant_id)
        domain = variant.get("domain")
        if domain not in {"continuous", "integer"}:
            errors.append(f"{prefix}: domain must be continuous or integer")
            continue
        if not isinstance(variant.get("changed_assumption"), str) or not variant["changed_assumption"].strip():
            errors.append(f"{prefix}: changed_assumption must be non-empty")
        if not isinstance(variant.get("material"), bool):
            errors.append(f"{prefix}: material must be boolean")
        quotes = variant.get("source_quotes")
        if not isinstance(quotes, list) or not quotes or any(
            not isinstance(quote, str) or not source_quote_is_exact(quote, statement) or quote == "UNSUPPORTED_BY_SOURCE"
            for quote in quotes
        ):
            errors.append(f"{prefix}: source_quotes must be non-empty exact statement spans")
        variant_dir = workspace / "variants" / variant_id
        missing = [name for name in VARIANT_REQUIRED_FILES if not (variant_dir / name).is_file()]
        if missing:
            errors.append(f"{prefix}: missing files {missing}")
            continue
        try:
            problem = json.loads((variant_dir / "problem.json").read_text(encoding="utf-8"))
            agent_report = json.loads((variant_dir / "solver_report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}: invalid JSON artifact: {exc}")
            continue
        if not isinstance(problem, dict):
            errors.append(f"{prefix}: problem.json root must be an object")
            continue
        for field, expected in (("row_id", row_id), ("source_sha256", source_sha256), ("variant_id", variant_id)):
            if problem.get(field) != expected:
                errors.append(f"{prefix}: problem.json {field} mismatch")
        required_problem = {"variables", "objective", "constraints", "assumptions", "domain_evidence"}
        if not required_problem.issubset(problem):
            errors.append(f"{prefix}: problem.json missing fields {sorted(required_problem - set(problem))}")
        errors.extend(_problem_source_quote_errors(problem, statement, f"{prefix}.problem.json"))
        if not isinstance(agent_report, dict) or set(agent_report) != {"continuous", "integer"}:
            errors.append(f"{prefix}: solver_report.json must have continuous and integer keys")
            continue
        for result_domain in ("continuous", "integer"):
            errors.extend(
                f"{prefix}.{message}" for message in _validate_solver_result(agent_report[result_domain], result_domain)
            )
        if len(errors) > variant_error_count:
            continue
        try:
            parent_report = execute_submission(variant_dir, project_root)
            if not _same_solver_report(agent_report, parent_report):
                raise AuditRunnerError("agent and parent solver reports disagree")
            write_json(variant_dir / "parent_solver_report.json", parent_report)
            model_ir = _model_ir_payload(variant_dir / "submission.py", project_root)
            write_json(variant_dir / "parent_model_ir.json", model_ir)
        except (AuditRunnerError, OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{prefix}: {exc}")
            continue
        normalized.append(
            {
                **variant,
                "model_ir_fingerprint": sha256_json(model_ir[domain]),
                "selected_result": _selected_result(parent_report, domain),
                "paths": {
                    "problem": f"variants/{variant_id}/problem.json",
                    "submission": f"variants/{variant_id}/submission.py",
                    "parent_solver_report": f"variants/{variant_id}/parent_solver_report.json",
                    "parent_model_ir": f"variants/{variant_id}/parent_model_ir.json",
                },
            }
        )
    anchor_ids: set[str] = set()
    try:
        with anchors_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            anchors = list(reader)
            if reader.fieldnames != INTERPRETATION_ANCHOR_FIELDS:
                errors.append("source_anchors.csv: header mismatch")
        if not anchors:
            errors.append("source_anchors.csv: no rows")
        for line_number, anchor in enumerate(anchors, start=2):
            variant_id = str(anchor.get("variant_id", ""))
            anchor_ids.add(variant_id)
            if variant_id not in seen_ids:
                errors.append(f"source_anchors.csv:{line_number}: unknown variant_id")
            quote = str(anchor.get("source_quote", ""))
            if quote == "UNSUPPORTED_BY_SOURCE" or not source_quote_is_exact(quote, statement):
                errors.append(f"source_anchors.csv:{line_number}: source quote is not exact")
            if anchor.get("support_label") not in {"supported", "ambiguous"}:
                errors.append(f"source_anchors.csv:{line_number}: invalid support_label")
    except (OSError, csv.Error) as exc:
        errors.append(f"source_anchors.csv: {exc}")
    missing_anchor_ids = seen_ids - anchor_ids
    if missing_anchor_ids:
        errors.append(f"source_anchors.csv: variants without anchors {sorted(missing_anchor_ids)}")
    if errors:
        return {"status": "invalid", "errors": errors}
    validated = {
        "schema_version": "interpretation_set_validated_v1",
        "row_id": row_id,
        "source_sha256": source_sha256,
        "enumeration_status": payload["enumeration_status"],
        "variants": normalized,
        "valid_variant_count": len(normalized),
        "material_multi_variant": _material_multi_variant(normalized),
    }
    write_json(workspace / "validated-interpretation-set.json", validated)
    return {"status": "valid", "errors": [], **validated}


def _run_generation_row(
    row: dict[str, Any],
    *,
    campaign_dir: Path,
    output_dir: Path,
    project_root: Path,
    options: AuditOptions,
) -> dict[str, Any]:
    row_id = str(row["row_id"])
    workspace = output_dir / "rows" / row_id
    if workspace.exists():
        raise AuditRunnerError(f"{row_id}: output workspace already exists; refusing duplicate session")
    workspace.mkdir(parents=True)
    statement_source = (campaign_dir / str(row["statement_path"])).resolve()
    statement_target = workspace / "statement.txt"
    shutil.copyfile(statement_source, statement_target)
    write_json(workspace / "run-input.json", dict(row))
    prompt = build_interpretation_prompt(
        row_id=row_id,
        source_sha256=str(row["source_sha256"]),
        project_root=project_root,
    )
    prompt_path = workspace / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    last_message = workspace / "last-message.md"
    command = build_codex_command(workspace=workspace, last_message_path=last_message, options=options)
    returncode, timed_out, stdout, stderr = _invoke_codex(command, prompt, options.timeout_seconds)
    (workspace / "codex-events.jsonl").write_text(redact_text(stdout), encoding="utf-8")
    (workspace / "codex-stderr.txt").write_text(redact_text(stderr), encoding="utf-8")
    leakage = _trace_leakage(stdout, stderr, workspace)
    validation: dict[str, Any] = {"status": "not_run", "errors": []}
    if timed_out:
        terminal_status = "timed_out"
    elif returncode != 0:
        terminal_status = "model_failed"
    elif leakage:
        terminal_status = "leakage_blocked"
    else:
        validation = validate_interpretation_artifacts(
            workspace,
            row_id=row_id,
            source_sha256=str(row["source_sha256"]),
            statement=statement_target.read_text(encoding="utf-8"),
            project_root=project_root,
        )
        terminal_status = "success" if validation["status"] == "valid" else "artifact_invalid"
    manifest = {
        "schema_version": "interpretation_audit_session_v1",
        "row_id": row_id,
        "split": row["split"],
        "role": options.role,
        "model": options.model,
        "reasoning_effort": options.effort,
        "session_policy": "one_fresh_codex_exec_per_row_no_resume",
        "source_sha256": row["source_sha256"],
        "prompt_sha256": sha256_file(prompt_path),
        "returncode": returncode,
        "timed_out": timed_out,
        "terminal_status": terminal_status,
        "leakage_findings": leakage,
        "artifact_validation": validation,
        "codex_run_metadata": build_run_metadata(command, options, stdout),
    }
    write_json(workspace / "run-manifest.json", manifest)
    return manifest


def _run_batch(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    options: AuditOptions,
    max_sessions: int,
    concurrency: int,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    schema_version: str,
) -> list[dict[str, Any]]:
    validate_policy(options)
    if max_sessions <= 0 or len(rows) > max_sessions:
        raise AuditRunnerError(f"selected {len(rows)} rows but max_sessions={max_sessions}")
    if concurrency not in {1, 2}:
        raise AuditRunnerError("concurrency must be 1 or 2")
    output_dir.mkdir(parents=True, exist_ok=False)
    if concurrency == 1:
        results = [runner(row) for row in rows]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(runner, row): row["row_id"] for row in rows}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["row_id"])
    write_jsonl(output_dir / "batch-results.jsonl", results)
    write_json(
        output_dir / "batch-summary.json",
        {
            "schema_version": schema_version,
            "role": options.role,
            "model": options.model,
            "reasoning_effort": options.effort,
            "sessions_started": len(results),
            "terminal_counts": {
                status: sum(result["terminal_status"] == status for result in results)
                for status in sorted(TERMINAL_STATES)
            },
        },
    )
    return results


def run_interpretation_batch(
    rows: list[dict[str, Any]],
    *,
    campaign_dir: Path,
    output_dir: Path,
    project_root: Path,
    options: AuditOptions,
    max_sessions: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if options.role != "terra":
        raise AuditRunnerError("interpretation generation requires role=terra")
    return _run_batch(
        rows,
        output_dir=output_dir,
        options=options,
        max_sessions=max_sessions,
        concurrency=concurrency,
        runner=lambda row: _run_generation_row(
            row,
            campaign_dir=campaign_dir,
            output_dir=output_dir,
            project_root=project_root,
            options=options,
        ),
        schema_version="interpretation_audit_batch_v1",
    )


def load_adjudication_manifest(path: Path, campaign_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != ADJUDICATION_MANIFEST_FIELDS:
                raise AuditRunnerError(f"{path}:{line_number}: invalid adjudication manifest fields")
            for field, root in (("statement_path", "source"), ("packet_path", "source")):
                resolved = (campaign_dir / str(row[field])).resolve()
                root_path = (campaign_dir / root).resolve()
                if root_path not in resolved.parents or not resolved.is_file():
                    raise AuditRunnerError(f"{path}:{line_number}: invalid {field}")
            statement = (campaign_dir / str(row["statement_path"])).resolve()
            packet = (campaign_dir / str(row["packet_path"])).resolve()
            if sha256_file(statement) != row["source_sha256"] or sha256_file(packet) != row["packet_sha256"]:
                raise AuditRunnerError(f"{path}:{line_number}: hash mismatch")
            rows.append(row)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise AuditRunnerError("adjudication manifest contains duplicate row IDs")
    return rows


def build_adjudication_prompt(*, row_id: str, source_sha256: str) -> str:
    return f"""You are an independent source-support adjudicator. Review one frozen interpretation packet in one fresh session.

Input boundary:
- Read only `statement.txt` and `candidate_packet.json` in this workspace.
- Do not inspect parent directories, repositories, benchmark datasets, answer keys, prior verdicts, or the internet.
- The packet contains candidate formulations and parent solver evidence, but no benchmark answer.
- Row ID: `{row_id}`
- Required statement SHA-256: `{source_sha256}`

For every candidate variant, decide only whether its changed interpretation is `supported`, `contradicted`, or `underdetermined` by the exact source wording. Do not judge benchmark-answer correctness. Mark material only when the interpretation can alter feasible solutions, solver status, or objective value.

Write `adjudication.json` with exactly:
- row_id, source_sha256, variants, obvious_omission, omission_note, omission_source_quotes
- one variant object per packet variant with exactly: variant_id, support_status, material, source_quotes, rationale
- every source_quotes and omission_source_quotes entry is an exact statement span
- obvious_omission is true only when the packet misses an evident materially different source-plausible interpretation

Do not add or remove candidate variant IDs. Do not create an answer verdict. Finish with a concise status message. Do not ask questions.
"""


def validate_adjudication_artifacts(
    workspace: Path,
    *,
    row_id: str,
    source_sha256: str,
    statement: str,
    packet: dict[str, Any],
) -> dict[str, Any]:
    path = workspace / "adjudication.json"
    if not path.is_file():
        return {"status": "invalid", "errors": ["missing adjudication.json"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "errors": [f"adjudication.json: {exc}"]}
    errors: list[str] = []
    if not isinstance(payload, dict) or set(payload) != ADJUDICATION_FIELDS:
        return {"status": "invalid", "errors": ["adjudication.json: invalid root fields"]}
    if payload.get("row_id") != row_id or payload.get("source_sha256") != source_sha256:
        errors.append("adjudication.json: row_id or source_sha256 mismatch")
    if not isinstance(payload.get("obvious_omission"), bool):
        errors.append("adjudication.json: obvious_omission must be boolean")
    if not isinstance(payload.get("omission_note"), str):
        errors.append("adjudication.json: omission_note must be string")
    omission_quotes = payload.get("omission_source_quotes")
    if not isinstance(omission_quotes, list) or any(
        not isinstance(quote, str) or not source_quote_is_exact(quote, statement)
        for quote in omission_quotes
    ):
        errors.append("adjudication.json: invalid omission_source_quotes")
    expected_ids = {variant["variant_id"] for variant in packet.get("variants", [])}
    reviews = payload.get("variants")
    observed_ids: set[str] = set()
    if not isinstance(reviews, list):
        errors.append("adjudication.json: variants must be a list")
        reviews = []
    for index, review in enumerate(reviews):
        prefix = f"adjudication.json.variants[{index}]"
        if not isinstance(review, dict) or set(review) != ADJUDICATION_VARIANT_FIELDS:
            errors.append(f"{prefix}: invalid fields")
            continue
        variant_id = review.get("variant_id")
        if variant_id not in expected_ids or variant_id in observed_ids:
            errors.append(f"{prefix}: unknown or duplicate variant_id")
        observed_ids.add(str(variant_id))
        if review.get("support_status") not in {"supported", "contradicted", "underdetermined"}:
            errors.append(f"{prefix}: invalid support_status")
        if not isinstance(review.get("material"), bool) or not isinstance(review.get("rationale"), str):
            errors.append(f"{prefix}: invalid material or rationale")
        quotes = review.get("source_quotes")
        if not isinstance(quotes, list) or not quotes or any(
            not isinstance(quote, str) or not source_quote_is_exact(quote, statement)
            for quote in quotes
        ):
            errors.append(f"{prefix}: source_quotes must be exact statement spans")
    if observed_ids != expected_ids:
        errors.append("adjudication.json: variant IDs do not match candidate packet")
    return {"status": "valid", "errors": [], "adjudication": payload} if not errors else {
        "status": "invalid",
        "errors": errors,
    }


def _run_adjudication_row(
    row: dict[str, Any],
    *,
    campaign_dir: Path,
    output_dir: Path,
    options: AuditOptions,
) -> dict[str, Any]:
    row_id = str(row["row_id"])
    workspace = output_dir / "rows" / row_id
    if workspace.exists():
        raise AuditRunnerError(f"{row_id}: output workspace already exists; refusing duplicate session")
    workspace.mkdir(parents=True)
    statement_source = (campaign_dir / str(row["statement_path"])).resolve()
    packet_source = (campaign_dir / str(row["packet_path"])).resolve()
    statement_target = workspace / "statement.txt"
    packet_target = workspace / "candidate_packet.json"
    shutil.copyfile(statement_source, statement_target)
    shutil.copyfile(packet_source, packet_target)
    write_json(workspace / "run-input.json", dict(row))
    prompt = build_adjudication_prompt(row_id=row_id, source_sha256=str(row["source_sha256"]))
    prompt_path = workspace / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    last_message = workspace / "last-message.md"
    command = build_codex_command(workspace=workspace, last_message_path=last_message, options=options)
    returncode, timed_out, stdout, stderr = _invoke_codex(command, prompt, options.timeout_seconds)
    (workspace / "codex-events.jsonl").write_text(redact_text(stdout), encoding="utf-8")
    (workspace / "codex-stderr.txt").write_text(redact_text(stderr), encoding="utf-8")
    leakage = _trace_leakage(stdout, stderr, workspace)
    validation: dict[str, Any] = {"status": "not_run", "errors": []}
    if timed_out:
        terminal_status = "timed_out"
    elif returncode != 0:
        terminal_status = "model_failed"
    elif leakage:
        terminal_status = "leakage_blocked"
    else:
        packet = json.loads(packet_target.read_text(encoding="utf-8"))
        validation = validate_adjudication_artifacts(
            workspace,
            row_id=row_id,
            source_sha256=str(row["source_sha256"]),
            statement=statement_target.read_text(encoding="utf-8"),
            packet=packet,
        )
        terminal_status = "success" if validation["status"] == "valid" else "artifact_invalid"
    manifest = {
        "schema_version": "interpretation_adjudication_session_v1",
        "row_id": row_id,
        "split": row["split"],
        "role": options.role,
        "model": options.model,
        "reasoning_effort": options.effort,
        "session_policy": "one_fresh_codex_exec_per_row_no_resume",
        "source_sha256": row["source_sha256"],
        "packet_sha256": row["packet_sha256"],
        "prompt_sha256": sha256_file(prompt_path),
        "returncode": returncode,
        "timed_out": timed_out,
        "terminal_status": terminal_status,
        "leakage_findings": leakage,
        "artifact_validation": validation,
        "codex_run_metadata": build_run_metadata(command, options, stdout),
    }
    write_json(workspace / "run-manifest.json", manifest)
    return manifest


def run_adjudication_batch(
    rows: list[dict[str, Any]],
    *,
    campaign_dir: Path,
    output_dir: Path,
    options: AuditOptions,
    max_sessions: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if options.role != "sol":
        raise AuditRunnerError("interpretation adjudication requires role=sol")
    return _run_batch(
        rows,
        output_dir=output_dir,
        options=options,
        max_sessions=max_sessions,
        concurrency=concurrency,
        runner=lambda row: _run_adjudication_row(
            row,
            campaign_dir=campaign_dir,
            output_dir=output_dir,
            options=options,
        ),
        schema_version="interpretation_adjudication_batch_v1",
    )


def revalidate_interpretation_batch(run_dir: Path, project_root: Path, *, kind: str) -> list[dict[str, Any]]:
    if kind not in {"generation", "adjudication"}:
        raise AuditRunnerError("revalidation kind must be generation or adjudication")
    row_root = run_dir / "rows"
    if not row_root.is_dir():
        raise AuditRunnerError(f"run has no rows directory: {row_root}")
    results: list[dict[str, Any]] = []
    for workspace in sorted(path for path in row_root.iterdir() if path.is_dir()):
        run_input = json.loads((workspace / "run-input.json").read_text(encoding="utf-8"))
        manifest = json.loads((workspace / "run-manifest.json").read_text(encoding="utf-8"))
        statement = (workspace / "statement.txt").read_text(encoding="utf-8")
        if kind == "generation":
            validation = validate_interpretation_artifacts(
                workspace,
                row_id=str(run_input["row_id"]),
                source_sha256=str(run_input["source_sha256"]),
                statement=statement,
                project_root=project_root,
            )
        else:
            packet = json.loads((workspace / "candidate_packet.json").read_text(encoding="utf-8"))
            validation = validate_adjudication_artifacts(
                workspace,
                row_id=str(run_input["row_id"]),
                source_sha256=str(run_input["source_sha256"]),
                statement=statement,
                packet=packet,
            )
        infrastructure_ok = (
            manifest.get("returncode") == 0
            and not manifest.get("timed_out")
            and not manifest.get("leakage_findings")
        )
        effective = "success" if infrastructure_ok and validation["status"] == "valid" else manifest.get(
            "terminal_status", "unknown"
        )
        result = {
            "row_id": run_input["row_id"],
            "initial_terminal_status": manifest.get("terminal_status", "unknown"),
            "effective_terminal_status": effective,
            "model_sessions_added": 0,
            "validation": validation,
        }
        write_json(workspace / "revalidation.json", result)
        results.append(result)
    write_json(
        run_dir / "revalidation-summary.json",
        {
            "schema_version": "interpretation_revalidation_v1",
            "kind": kind,
            "rows": len(results),
            "effective_success": sum(row["effective_terminal_status"] == "success" for row in results),
            "model_sessions_added": 0,
            "results": results,
        },
    )
    return results
