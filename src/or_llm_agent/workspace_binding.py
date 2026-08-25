from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from or_llm_agent.codex_agent import CodexAgentOptions, codex_exec_model_args
from or_llm_agent.redaction import redact_text


@dataclass(frozen=True)
class WorkspaceBindingResult:
    case_id: str
    returncode: int
    timed_out: bool
    output_path: Path
    events_path: Path
    last_message_path: Path
    status_path: Path
    output_status: str


def build_workspace_binding_command(
    *,
    workspace: Path,
    artifact_dir: Path,
    last_message_path: Path,
    options: CodexAgentOptions,
) -> list[str]:
    command = [
        "codex",
        "-a",
        options.codex_approval,
        "exec",
        "--json",
        "-C",
        str(workspace),
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


def build_workspace_binding_prompt(
    *,
    case_id: str,
    workspace: Path,
    schema_path: Path,
    output_path: Path,
) -> str:
    relative_schema = schema_path.relative_to(workspace)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return f"""You are binding operational data for OR case `{case_id}`.

Read visible files under this workspace only:
{workspace}

Output normalized JSON only to:
{output_path}

Required output schema is visible at `{relative_schema}` and reproduced below:
```json
{json.dumps(schema, indent=2, sort_keys=True)}
```

Rules:
- Read business instructions, tables, units, missing-value conventions, joins, and version rules from visible workspace files.
- Preserve source values exactly after documented normalization or unit conversion.
- Do not formulate or solve an optimization model.
- Do not fetch external information.
- Do not inspect parent directories, hidden tests, reference answers, gold ledgers, or evaluator artifacts.
- Write one JSON object matching required schema; no Markdown or extra files.
- Finish without asking questions.
"""


def run_workspace_binding(
    *,
    case_id: str,
    workspace: Path,
    schema_path: Path,
    output_path: Path,
    artifact_dir: Path,
    options: CodexAgentOptions,
) -> WorkspaceBindingResult:
    workspace = workspace.resolve()
    schema_path = schema_path.resolve()
    output_path = output_path.resolve()
    artifact_dir = artifact_dir.resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace missing: {workspace}")
    if not schema_path.is_file() or not schema_path.is_relative_to(workspace):
        raise ValueError("schema must be a file inside workspace")
    if not output_path.is_relative_to(artifact_dir):
        raise ValueError("output must be inside artifact directory")

    events_path = artifact_dir / "sessions" / case_id / "codex-events.jsonl"
    last_message_path = artifact_dir / "sessions" / case_id / "last-message.md"
    status_path = artifact_dir / "agent-status" / f"{case_id}.json"
    raw_path = artifact_dir / "raw" / f"{case_id}.json"
    for path in (output_path, events_path, last_message_path, status_path, raw_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_workspace_binding_prompt(
        case_id=case_id,
        workspace=workspace,
        schema_path=schema_path,
        output_path=output_path,
    )
    command = build_workspace_binding_command(
        workspace=workspace,
        artifact_dir=artifact_dir,
        last_message_path=last_message_path,
        options=options,
    )
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

    output_status = "missing"
    if output_path.is_file():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            output_status = "valid_json_object" if isinstance(payload, dict) else "json_not_object"
        except json.JSONDecodeError:
            output_status = "invalid_json"

    status = {
        "schema_version": "or_llm_workspace_binding_v1",
        "case_id": case_id,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "workspace": str(workspace),
        "schema_path": str(schema_path),
        "output_path": str(output_path),
        "output_status": output_status,
        "events_path": str(events_path),
        "last_message_path": str(last_message_path),
        "stderr": redact_text(stderr),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps({"prompt": prompt, **status}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = WorkspaceBindingResult(
        case_id=case_id,
        returncode=returncode,
        timed_out=timed_out,
        output_path=output_path,
        events_path=events_path,
        last_message_path=last_message_path,
        status_path=status_path,
        output_status=output_status,
    )
    (artifact_dir / "workspace-binding-result.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(result).items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
