from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from or_llm_agent.benchmark_audit import (
    AuditOptions,
    AuditRunnerError,
    load_source_manifest,
    run_batch,
    validate_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated one-session benchmark audits")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run-batch")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--campaign-dir", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--role", choices=["terra", "sol"], required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--effort", required=True)
    run.add_argument("--max-sessions", type=int, required=True)
    run.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--ids", nargs="*")
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run-batch":
        parser.print_help()
        return 2
    try:
        options = AuditOptions(
            role=args.role,
            model=args.model,
            effort=args.effort,
            timeout_seconds=args.timeout_seconds,
        )
        validate_policy(options)
        rows = load_source_manifest(args.manifest, args.campaign_dir)
        if args.ids:
            requested = set(args.ids)
            rows = [row for row in rows if row["row_id"] in requested]
            missing = requested - {row["row_id"] for row in rows}
            if missing:
                raise AuditRunnerError(f"unknown requested row IDs: {sorted(missing)}")
        if len(rows) > args.max_sessions:
            raise AuditRunnerError(f"selected {len(rows)} rows but max_sessions={args.max_sessions}")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "rows": [row["row_id"] for row in rows],
                        "sessions": len(rows),
                        "role": args.role,
                        "model": args.model,
                        "effort": args.effort,
                    },
                    sort_keys=True,
                )
            )
            return 0
        project_root = Path(__file__).resolve().parents[2]
        results = run_batch(
            rows,
            campaign_dir=args.campaign_dir,
            output_dir=args.out_dir,
            project_root=project_root,
            options=options,
            max_sessions=args.max_sessions,
            concurrency=args.concurrency,
        )
        failures = sum(result["terminal_status"] != "success" for result in results)
        print(json.dumps({"sessions": len(results), "success": len(results) - failures, "failures": failures}))
        return 1 if failures else 0
    except (AuditRunnerError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
