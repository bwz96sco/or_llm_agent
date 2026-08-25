from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from or_llm_agent.codex_agent import CodexAgentOptions, codex_exec_model_args, neutral_work_dir
from or_llm_agent.or_ci import or_ci_command, run_or_ci_verify
from or_llm_agent.redaction import redact_text


@dataclass(frozen=True)
class RevisionResult:
    problem_id: str
    returncode: int
    timed_out: bool
    submission_path: Path
    report_path: Path
    events_path: Path
    last_message_path: Path
    status_path: Path
    nominal_status: str
    nominal_classification: str


def build_revision_command(artifact_dir: Path, work_dir: Path, last_message_path: Path, options: CodexAgentOptions) -> list[str]:
    command = [
        "codex",
        "-a",
        options.codex_approval,
        "exec",
        "--json",
        "-C",
        str(work_dir),
        "--add-dir",
        str(artifact_dir),
        "--skip-git-repo-check",
        "-s",
        options.codex_sandbox,
        "-o",
        str(last_message_path),
    ]
    command.extend(codex_exec_model_args(options))
    command.append("-")
    return command


def build_revision_prompt(
    *,
    problem_id: str,
    base_submission: str,
    base_statement: str,
    change_log: str,
    problem: dict,
    problem_path: Path,
    submission_path: Path,
    report_path: Path,
    max_repair_attempts: int,
    disclosed_witness: str | None = None,
) -> str:
    verify_command, _ = or_ci_command()
    verify = shlex.join(
        [
            *verify_command,
            "verify",
            "--problem",
            str(problem_path),
            "--submission",
            str(submission_path),
            "--out",
            str(report_path),
        ]
    )
    witness_section = ""
    witness_rule = "- Do not read benchmark answers, hidden tests, witnesses, ledgers, or reference implementations."
    if disclosed_witness:
        witness_section = f"""
One disclosed counterexample from an earlier evaluation:
```json
{disclosed_witness}
```
"""
        witness_rule = "- You may use only disclosed counterexample above; do not seek other hidden tests, ledgers, or reference implementations."
    return f"""You are revising one inherited optimization model for `{problem_id}`.

Write revised Python only to:
{submission_path}

Visible requirement history:
{base_statement}

Current change request:
{change_log}
{witness_section}

Base inherited submission:
```python
{base_submission}
```

Current public instance schema and nominal values:
```json
{json.dumps(problem.get("instance", {}), indent=2, sort_keys=True)}
```

Requirements:
- Preserve every active earlier requirement unless current change explicitly revokes it.
- Implement current change. Remove only explicitly revoked behavior.
- Expose `build_model(data: dict) -> gurobipy.Model`; return unoptimized model.
- Expose `semantic_variables(model) -> dict[str, gurobipy.Var]` mapping public semantic keys to decision variables.
- Use values from `data`; do not hard-code nominal parameter values.
{witness_rule}
- Do not fetch external information or edit repository files.
- Do not run cleanup/removal commands.

Nominal verification command:
```bash
{verify}
```

Write submission, run verification, inspect report, repair at most {max_repair_attempts} times. Nominal PASS is necessary, not proof of semantic correctness. Finish without asking questions.
"""


def run_revision(
    *,
    problem_id: str,
    base_submission_path: Path,
    base_statement_path: Path,
    change_log_path: Path,
    problem_path: Path,
    artifact_dir: Path,
    options: CodexAgentOptions,
    witness_path: Path | None = None,
) -> RevisionResult:
    for path in (base_submission_path, base_statement_path, change_log_path, problem_path):
        if not path.is_file():
            raise ValueError(f"required revision input missing: {path}")
    if witness_path is not None and not witness_path.is_file():
        raise ValueError(f"disclosed witness missing: {witness_path}")

    artifact_dir = artifact_dir.resolve()
    submission_path = artifact_dir / "submissions" / f"{problem_id}.py"
    report_path = artifact_dir / "reports" / f"{problem_id}.json"
    events_path = artifact_dir / "sessions" / problem_id / "codex-events.jsonl"
    last_message_path = artifact_dir / "sessions" / problem_id / "last-message.md"
    status_path = artifact_dir / "agent-status" / f"{problem_id}.json"
    for path in (submission_path, report_path, events_path, last_message_path, status_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = neutral_work_dir(artifact_dir, problem_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    problem = json.loads(problem_path.read_text(encoding="utf-8"))
    prompt = build_revision_prompt(
        problem_id=problem_id,
        base_submission=base_submission_path.read_text(encoding="utf-8"),
        base_statement=base_statement_path.read_text(encoding="utf-8"),
        change_log=change_log_path.read_text(encoding="utf-8"),
        problem=problem,
        problem_path=problem_path.resolve(),
        submission_path=submission_path,
        report_path=report_path,
        max_repair_attempts=options.max_repair_attempts,
        disclosed_witness=witness_path.read_text(encoding="utf-8") if witness_path else None,
    )
    command = build_revision_command(artifact_dir, work_dir, last_message_path, options)
    timed_out = False
    try:
        process = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=options.timeout_seconds if options.timeout_seconds and options.timeout_seconds > 0 else None,
        )
        returncode = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

    events_path.write_text(redact_text(stdout), encoding="utf-8")
    if not last_message_path.exists():
        last_message_path.write_text("Nested Codex produced no final message.\n", encoding="utf-8")

    nominal_status = "FAIL"
    nominal_classification = "SUBMISSION_MISSING"
    if submission_path.is_file():
        verification = run_or_ci_verify(
            problem_path=problem_path,
            submission_path=submission_path,
            report_path=report_path,
            cwd=work_dir,
        )
        nominal_status = verification.status
        nominal_classification = verification.classification

    status = {
        "schema_version": "or_llm_agent_revision_v1",
        "problem_id": problem_id,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "submission_path": str(submission_path),
        "report_path": str(report_path),
        "events_path": str(events_path),
        "last_message_path": str(last_message_path),
        "nominal_status": nominal_status,
        "nominal_classification": nominal_classification,
        "stderr": redact_text(stderr),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = RevisionResult(
        problem_id=problem_id,
        returncode=returncode,
        timed_out=timed_out,
        submission_path=submission_path,
        report_path=report_path,
        events_path=events_path,
        last_message_path=last_message_path,
        status_path=status_path,
        nominal_status=nominal_status,
        nominal_classification=nominal_classification,
    )
    (artifact_dir / "revision-result.json").write_text(
        json.dumps({**asdict(result), **{key: str(value) for key, value in asdict(result).items() if isinstance(value, Path)}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
