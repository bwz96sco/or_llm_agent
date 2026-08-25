from __future__ import annotations

import argparse
from pathlib import Path

from or_llm_agent.codex_agent import CodexAgentOptions
from or_llm_agent.workspace_binding import run_workspace_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="or-llm-workspace-bind")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument("--codex-reasoning-effort", default="high")
    parser.add_argument("--codex-sandbox", choices=("read-only", "workspace-write", "danger-full-access"), default="workspace-write")
    parser.add_argument("--codex-approval", choices=("untrusted", "on-failure", "on-request", "never"), default="never")
    parser.add_argument("--timeout-seconds", default=900, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_workspace_binding(
            case_id=args.case_id,
            workspace=args.workspace,
            schema_path=args.schema,
            output_path=args.output,
            artifact_dir=args.artifact_dir,
            options=CodexAgentOptions(
                codex_model=args.codex_model,
                codex_reasoning_effort=args.codex_reasoning_effort,
                codex_sandbox=args.codex_sandbox,
                codex_approval=args.codex_approval,
                max_repair_attempts=0,
                timeout_seconds=args.timeout_seconds if args.timeout_seconds > 0 else None,
            ),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"workspace binding {result.case_id}: {result.output_status}")
    return 0 if result.returncode == 0 and result.output_status == "valid_json_object" else 1


if __name__ == "__main__":
    raise SystemExit(main())
