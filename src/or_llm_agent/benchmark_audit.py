from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from or_llm_agent.redaction import redact_text


class AuditRunnerError(ValueError):
    pass


SOURCE_MANIFEST_FIELDS = {"row_id", "split", "statement_path", "source_sha256"}
SOURCE_ANCHOR_FIELDS = [
    "element_id",
    "element_type",
    "generated_element",
    "source_quote",
    "support_label",
    "materiality",
    "notes",
]
REQUIRED_ARTIFACTS = [
    "problem.json",
    "submission.py",
    "solver_report.json",
    "source_anchors.csv",
    "audit.json",
]
POLICY = {
    "terra": ("gpt-5.6-terra", "high"),
    "sol": ("gpt-5.6-sol", "xhigh"),
}
TERMINAL_STATES = {
    "success",
    "model_failed",
    "artifact_invalid",
    "solver_failed",
    "leakage_blocked",
    "timed_out",
}
FORBIDDEN_TRACE_MARKERS = (
    "/hidden/",
    "answer-key.jsonl",
    "nl4opt_with_optimal_solution",
    '"en_answer"',
)


@dataclass(frozen=True)
class AuditOptions:
    role: str
    model: str
    effort: str
    timeout_seconds: int
    codex_sandbox: str = "workspace-write"
    codex_approval: str = "never"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditRunnerError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise AuditRunnerError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_policy(options: AuditOptions) -> None:
    if options.role not in POLICY:
        raise AuditRunnerError(f"unsupported role: {options.role!r}")
    expected_model, expected_effort = POLICY[options.role]
    if (options.model, options.effort) != (expected_model, expected_effort):
        raise AuditRunnerError(
            f"{options.role} policy requires model={expected_model} effort={expected_effort}; "
            f"received model={options.model} effort={options.effort}"
        )
    if options.timeout_seconds <= 0:
        raise AuditRunnerError("timeout must be positive")


def load_source_manifest(path: Path, campaign_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if set(row) != SOURCE_MANIFEST_FIELDS:
            raise AuditRunnerError(
                f"{path}:{line_number}: source manifest fields must be exactly {sorted(SOURCE_MANIFEST_FIELDS)}"
            )
        row_id = str(row["row_id"])
        if row_id in seen:
            raise AuditRunnerError(f"{path}:{line_number}: duplicate row_id {row_id}")
        seen.add(row_id)
        statement_path = (campaign_dir / str(row["statement_path"])).resolve()
        source_root = (campaign_dir / "source").resolve()
        if source_root not in statement_path.parents:
            raise AuditRunnerError(f"{path}:{line_number}: statement path escapes campaign source root")
        if not statement_path.is_file():
            raise AuditRunnerError(f"{path}:{line_number}: statement missing: {statement_path}")
        if sha256_file(statement_path) != row["source_sha256"]:
            raise AuditRunnerError(f"{path}:{line_number}: statement hash mismatch")
    return rows


def build_prompt(*, row_id: str, source_sha256: str, project_root: Path) -> str:
    anchor_header = ",".join(SOURCE_ANCHOR_FIELDS)
    verify_command = shlex.join(["uv", "run", "--project", str(project_root), "python", "submission.py"])
    return f"""You are one independent optimization benchmark auditor. Complete the full workflow in this one Codex session.

Input boundary:
- Read only `statement.txt` and files you create in the current row workspace.
- Do not inspect parent directories, repositories, home-directory data, benchmark datasets, answer keys, prior model outputs, or the internet.
- No benchmark answer is available in this workspace. Do not guess or search for one.
- Row ID: `{row_id}`
- Required statement SHA-256: `{source_sha256}`

Task:
1. Read `statement.txt`.
2. Derive a source-backed optimization specification.
3. Implement and execute the model in Gurobi.
4. Check both continuous and nonnegative-integer interpretations when meaningful. Do not silently choose integrality.
5. Inspect infeasible/unbounded status and calculate an IIS for infeasible interpretations.
6. Repair your specification or code inside this same session if self-checks fail.
7. Record ambiguity instead of inventing missing facts.

Write these exact files in the current workspace:

`problem.json`
- JSON object with: row_id, source_sha256, variables, objective, constraints, assumptions, domain_evidence.
- Every variable/objective/constraint must include an exact source quote.
- Exact means character-for-character after JSON parsing. Preserve every source backslash, escape marker, punctuation mark, and space. If the statement contains a backslash before a currency symbol, the parsed source_quote must retain that backslash, so escape it correctly in JSON.

`submission.py`
- Self-contained Python using only Python stdlib and gurobipy.
- Define `build_model(domain: str) -> gurobipy.Model` for `continuous` and `integer`.
- Use nonnegative variables unless the statement explicitly says otherwise.
- In `__main__`, solve both domains with OutputFlag=0, DualReductions=0, and InfUnbdInfo=1.
- Print exactly one JSON object as the last stdout line with keys `continuous` and `integer`.
- Each domain result must include status, objective, variables, max_constraint_violation, and iis_constraints.

`solver_report.json`
- The exact parsed JSON object produced by the final successful execution of `submission.py`.

`source_anchors.csv`
- Exact header: `{anchor_header}`
- One row per material modeling element.
- support_label must be one of supported, contradicted, unsupported, ambiguous.
- source_quote must be one contiguous exact statement span, or multiple complete exact sentences kept intact and in source order, or `UNSUPPORTED_BY_SOURCE`.
- Exact means character-for-character after CSV parsing. Never remove source backslashes or normalize escaped currency text.
- Never stitch partial clauses, paraphrase, or omit words inside an anchored sentence. Use separate rows when separate clauses are needed.

`audit.json`
- JSON object with: row_id, source_sha256, source_status, chosen_domain, mechanisms, material_ambiguities, summary.
- source_status must be one of well_posed, ambiguous, inconsistent.
- chosen_domain must be continuous, integer, both, or unresolved.
- mechanisms and material_ambiguities must be JSON lists.

Run this exact verification command before finishing:
```bash
{verify_command}
```

Do not create a benchmark-answer verdict. The parent process compares answers only after this session terminates.
Finish with a concise status message. Do not ask questions.
"""


def build_codex_command(*, workspace: Path, last_message_path: Path, options: AuditOptions) -> list[str]:
    return [
        "codex",
        "-a",
        options.codex_approval,
        "exec",
        "--json",
        "--ignore-user-config",
        "-C",
        str(workspace),
        "--skip-git-repo-check",
        "-s",
        options.codex_sandbox,
        "-o",
        str(last_message_path),
        "-c",
        f"model_reasoning_effort={json.dumps(options.effort)}",
        "-m",
        options.model,
        "-",
    ]


def _coerce_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _codex_version() -> str:
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return redact_text((result.stdout or result.stderr).strip())


def _event_metadata(stdout: str) -> tuple[str, dict[str, Any]]:
    thread_id = ""
    usage: dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        candidate_thread = event.get("thread_id")
        if isinstance(candidate_thread, str) and candidate_thread:
            thread_id = candidate_thread
        thread = event.get("thread")
        if isinstance(thread, dict) and isinstance(thread.get("id"), str):
            thread_id = thread["id"]
        candidate_usage = event.get("usage")
        if not isinstance(candidate_usage, dict):
            turn = event.get("turn")
            candidate_usage = turn.get("usage") if isinstance(turn, dict) else None
        if isinstance(candidate_usage, dict):
            usage = candidate_usage
    return thread_id, usage


def build_run_metadata(command: list[str], options: AuditOptions, stdout: str) -> dict[str, Any]:
    thread_id, usage = _event_metadata(stdout)
    return {
        "schema_version": "benchmark_audit_codex_metadata_v1",
        "adapter": "codex-cli",
        "codex_model_arg": options.model,
        "codex_reasoning_effort_arg": options.effort,
        "codex_effective_model": options.model,
        "codex_effective_model_source": "arg",
        "codex_effective_reasoning_effort": options.effort,
        "codex_effective_reasoning_effort_source": "arg",
        "codex_cli_version": _codex_version(),
        "codex_command": [redact_text(item) for item in command],
        "codex_command_string": redact_text(shlex.join(command)),
        "codex_thread_id": thread_id,
        "codex_usage": usage,
    }


def _invoke_codex(command: list[str], prompt: str, timeout_seconds: int) -> tuple[int, bool, str, str]:
    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        return result.returncode, False, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, True, _coerce_stream(exc.stdout), _coerce_stream(exc.stderr)


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AuditRunnerError("submission stdout contains no JSON object")


def execute_submission(workspace: Path, project_root: Path, timeout_seconds: int = 120) -> dict[str, Any]:
    result = subprocess.run(
        ["uv", "run", "--project", str(project_root), "python", "submission.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise AuditRunnerError(f"parent submission execution failed: {redact_text(result.stderr.strip())}")
    payload = _parse_last_json_line(result.stdout)
    if set(payload) != {"continuous", "integer"}:
        raise AuditRunnerError("submission report must have continuous and integer keys")
    return payload


def _validate_solver_result(result: Any, domain: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return [f"{domain}: result must be an object"]
    required = {"status", "objective", "variables", "max_constraint_violation", "iis_constraints"}
    missing = required - set(result)
    if missing:
        errors.append(f"{domain}: missing fields {sorted(missing)}")
    if not isinstance(result.get("variables"), dict):
        errors.append(f"{domain}: variables must be an object")
    if not isinstance(result.get("iis_constraints"), list):
        errors.append(f"{domain}: iis_constraints must be a list")
    return errors


def _same_solver_report(agent: dict[str, Any], parent: dict[str, Any]) -> bool:
    for domain in ("continuous", "integer"):
        left = agent.get(domain, {})
        right = parent.get(domain, {})
        if left.get("status") != right.get("status"):
            return False
        left_objective = left.get("objective")
        right_objective = right.get("objective")
        if left_objective is None or right_objective is None:
            if left_objective != right_objective:
                return False
        elif not math.isclose(float(left_objective), float(right_objective), rel_tol=1e-6, abs_tol=1e-6):
            return False
    return True


def source_quote_is_exact(quote: str, statement: str) -> bool:
    if quote == "UNSUPPORTED_BY_SOURCE" or quote in statement:
        return True
    fragments = [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+", quote) if fragment.strip()]
    if len(fragments) < 2 or any(fragment not in statement for fragment in fragments):
        return False
    positions = [statement.index(fragment) for fragment in fragments]
    return positions == sorted(positions)


def _problem_source_quote_errors(payload: Any, statement: str, path: str = "problem.json") -> list[str]:
    errors: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key == "source_quote" and isinstance(value, str) and not source_quote_is_exact(value, statement):
                errors.append(f"{child_path}: source quote is not exact")
            else:
                errors.extend(_problem_source_quote_errors(value, statement, child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            errors.extend(_problem_source_quote_errors(value, statement, f"{path}[{index}]"))
    return errors


def validate_artifacts(
    workspace: Path,
    *,
    row_id: str,
    source_sha256: str,
    statement: str,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    for filename in REQUIRED_ARTIFACTS:
        if not (workspace / filename).is_file():
            errors.append(f"missing artifact: {filename}")
    if errors:
        return {"status": "invalid", "errors": errors}

    try:
        problem = json.loads((workspace / "problem.json").read_text(encoding="utf-8"))
        audit = json.loads((workspace / "audit.json").read_text(encoding="utf-8"))
        agent_report = json.loads((workspace / "solver_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "errors": [f"JSON artifact error: {exc}"]}

    for name, payload in (("problem.json", problem), ("audit.json", audit)):
        if not isinstance(payload, dict):
            errors.append(f"{name}: root must be an object")
            continue
        if payload.get("row_id") != row_id:
            errors.append(f"{name}: row_id mismatch")
        if payload.get("source_sha256") != source_sha256:
            errors.append(f"{name}: source_sha256 mismatch")
    problem_fields = {"variables", "objective", "constraints", "assumptions", "domain_evidence"}
    if isinstance(problem, dict):
        missing = problem_fields - set(problem)
        if missing:
            errors.append(f"problem.json: missing fields {sorted(missing)}")
        errors.extend(_problem_source_quote_errors(problem, statement))
    if isinstance(audit, dict):
        if audit.get("source_status") not in {"well_posed", "ambiguous", "inconsistent"}:
            errors.append("audit.json: invalid source_status")
        if audit.get("chosen_domain") not in {"continuous", "integer", "both", "unresolved"}:
            errors.append("audit.json: invalid chosen_domain")
        for field in ("mechanisms", "material_ambiguities"):
            if not isinstance(audit.get(field), list):
                errors.append(f"audit.json: {field} must be a list")

    try:
        with (workspace / "source_anchors.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            anchors = list(reader)
            if reader.fieldnames != SOURCE_ANCHOR_FIELDS:
                errors.append("source_anchors.csv: header mismatch")
        if not anchors:
            errors.append("source_anchors.csv: no rows")
        for line_number, anchor in enumerate(anchors, start=2):
            quote = str(anchor.get("source_quote", ""))
            if not source_quote_is_exact(quote, statement):
                errors.append(f"source_anchors.csv:{line_number}: source quote is not exact")
            if anchor.get("support_label") not in {"supported", "contradicted", "unsupported", "ambiguous"}:
                errors.append(f"source_anchors.csv:{line_number}: invalid support_label")
    except (OSError, csv.Error) as exc:
        errors.append(f"source_anchors.csv: {exc}")

    if not isinstance(agent_report, dict) or set(agent_report) != {"continuous", "integer"}:
        errors.append("solver_report.json: expected continuous and integer keys")
    else:
        for domain in ("continuous", "integer"):
            errors.extend(_validate_solver_result(agent_report.get(domain), domain))
    if errors:
        return {"status": "invalid", "errors": errors}

    try:
        parent_report = execute_submission(workspace, project_root)
    except (AuditRunnerError, OSError, subprocess.SubprocessError) as exc:
        return {"status": "solver_failed", "errors": [str(exc)]}
    write_json(workspace / "parent_solver_report.json", parent_report)
    if not _same_solver_report(agent_report, parent_report):
        return {"status": "invalid", "errors": ["agent and parent solver reports disagree"]}
    return {"status": "valid", "errors": [], "parent_solver_report": "parent_solver_report.json"}


def _trace_leakage(stdout: str, stderr: str, workspace: Path) -> list[str]:
    findings: list[str] = []
    trace = f"{stdout}\n{stderr}".lower()
    findings.extend(marker for marker in FORBIDDEN_TRACE_MARKERS if marker in trace)
    for path in workspace.rglob("*"):
        if path.is_file() and path.name.lower() in {"answer-key.jsonl", "answers.jsonl"}:
            findings.append(str(path))
    return sorted(set(findings))


def run_row(
    row: dict[str, Any],
    *,
    campaign_dir: Path,
    output_dir: Path,
    project_root: Path,
    options: AuditOptions,
) -> dict[str, Any]:
    row_id = str(row["row_id"])
    workspace = output_dir / "rows" / row_id
    run_manifest_path = workspace / "run-manifest.json"
    if run_manifest_path.exists():
        existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        raise AuditRunnerError(
            f"{row_id}: run manifest already exists with status={existing.get('terminal_status')}; refusing duplicate session"
        )
    workspace.mkdir(parents=True, exist_ok=False)
    statement_source = (campaign_dir / str(row["statement_path"])).resolve()
    statement_target = workspace / "statement.txt"
    shutil.copyfile(statement_source, statement_target)
    input_payload = dict(row)
    write_json(workspace / "run-input.json", input_payload)
    prompt = build_prompt(row_id=row_id, source_sha256=str(row["source_sha256"]), project_root=project_root)
    prompt_path = workspace / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    last_message_path = workspace / "last-message.md"
    command = build_codex_command(workspace=workspace, last_message_path=last_message_path, options=options)
    returncode, timed_out, stdout, stderr = _invoke_codex(command, prompt, options.timeout_seconds)
    (workspace / "codex-events.jsonl").write_text(redact_text(stdout), encoding="utf-8")
    (workspace / "codex-stderr.txt").write_text(redact_text(stderr), encoding="utf-8")
    metadata = build_run_metadata(command, options, stdout)
    leakage = _trace_leakage(stdout, stderr, workspace)
    validation: dict[str, Any] = {"status": "not_run", "errors": []}
    if timed_out:
        terminal_status = "timed_out"
    elif returncode != 0:
        terminal_status = "model_failed"
    elif leakage:
        terminal_status = "leakage_blocked"
    else:
        validation = validate_artifacts(
            workspace,
            row_id=row_id,
            source_sha256=str(row["source_sha256"]),
            statement=statement_target.read_text(encoding="utf-8"),
            project_root=project_root,
        )
        if validation["status"] == "valid":
            terminal_status = "success"
        elif validation["status"] == "solver_failed":
            terminal_status = "solver_failed"
        else:
            terminal_status = "artifact_invalid"
    manifest = {
        "schema_version": "benchmark_audit_session_v1",
        "row_id": row_id,
        "split": row["split"],
        "role": options.role,
        "model": options.model,
        "reasoning_effort": options.effort,
        "session_policy": "one_fresh_codex_exec_per_row_no_resume",
        "source_sha256": row["source_sha256"],
        "prompt_sha256": sha256_file(prompt_path),
        "command": [redact_text(part) for part in command],
        "returncode": returncode,
        "timed_out": timed_out,
        "terminal_status": terminal_status,
        "leakage_findings": leakage,
        "artifact_validation": validation,
        "codex_run_metadata": metadata,
        "artifacts": {
            "events": "codex-events.jsonl",
            "stderr": "codex-stderr.txt",
            "last_message": "last-message.md",
            "prompt": "prompt.md",
            "input": "run-input.json",
        },
    }
    write_json(run_manifest_path, manifest)
    return manifest


def run_batch(
    rows: list[dict[str, Any]],
    *,
    campaign_dir: Path,
    output_dir: Path,
    project_root: Path,
    options: AuditOptions,
    max_sessions: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    validate_policy(options)
    if max_sessions <= 0:
        raise AuditRunnerError("max_sessions must be positive")
    if len(rows) > max_sessions:
        raise AuditRunnerError(f"manifest has {len(rows)} rows but max_sessions={max_sessions}")
    if concurrency not in {1, 2}:
        raise AuditRunnerError("concurrency must be 1 or 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if concurrency == 1:
        for row in rows:
            results.append(
                run_row(
                    row,
                    campaign_dir=campaign_dir,
                    output_dir=output_dir,
                    project_root=project_root,
                    options=options,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    run_row,
                    row,
                    campaign_dir=campaign_dir,
                    output_dir=output_dir,
                    project_root=project_root,
                    options=options,
                ): row["row_id"]
                for row in rows
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["row_id"])
    write_jsonl(output_dir / "batch-results.jsonl", results)
    summary = {
        "schema_version": "benchmark_audit_batch_v1",
        "role": options.role,
        "model": options.model,
        "reasoning_effort": options.effort,
        "sessions_started": len(results),
        "terminal_counts": {
            status: sum(result["terminal_status"] == status for result in results)
            for status in sorted(TERMINAL_STATES)
        },
    }
    write_json(output_dir / "batch-summary.json", summary)
    return results


def revalidate_batch(run_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    row_root = run_dir / "rows"
    if not row_root.is_dir():
        raise AuditRunnerError(f"run has no rows directory: {row_root}")
    results: list[dict[str, Any]] = []
    for workspace in sorted(path for path in row_root.iterdir() if path.is_dir()):
        manifest_path = workspace / "run-manifest.json"
        input_path = workspace / "run-input.json"
        statement_path = workspace / "statement.txt"
        if not manifest_path.is_file() or not input_path.is_file() or not statement_path.is_file():
            raise AuditRunnerError(f"{workspace.name}: missing recorded run input, statement, or manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_input = json.loads(input_path.read_text(encoding="utf-8"))
        statement = statement_path.read_text(encoding="utf-8")
        validation = validate_artifacts(
            workspace,
            row_id=str(run_input["row_id"]),
            source_sha256=str(run_input["source_sha256"]),
            statement=statement,
            project_root=project_root,
        )
        initial_status = manifest.get("terminal_status", "unknown")
        infrastructure_ok = (
            manifest.get("returncode") == 0
            and not manifest.get("timed_out")
            and not manifest.get("leakage_findings")
        )
        effective_status = "success" if infrastructure_ok and validation["status"] == "valid" else initial_status
        result = {
            "row_id": run_input["row_id"],
            "initial_terminal_status": initial_status,
            "effective_terminal_status": effective_status,
            "model_sessions_added": 0,
            "validation": validation,
        }
        write_json(workspace / "revalidation.json", result)
        results.append(result)
    summary = {
        "schema_version": "benchmark_audit_revalidation_v1",
        "rows": len(results),
        "effective_success": sum(row["effective_terminal_status"] == "success" for row in results),
        "model_sessions_added": 0,
        "results": results,
    }
    write_json(run_dir / "revalidation-summary.json", summary)
    return results
