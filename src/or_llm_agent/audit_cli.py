from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from or_llm_agent.benchmark_audit import (
    AuditOptions,
    AuditRunnerError,
    load_source_manifest,
    revalidate_batch,
    run_batch,
    validate_policy,
)
from or_llm_agent.interpretation_audit import (
    diagnose_interpretation_batch,
    load_adjudication_manifest,
    revalidate_interpretation_batch,
    run_adjudication_batch,
    run_interpretation_batch,
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
    revalidate = subparsers.add_parser("revalidate-batch")
    revalidate.add_argument("--run-dir", type=Path, required=True)
    interpretation = subparsers.add_parser("run-interpretation-batch")
    interpretation.add_argument("--manifest", type=Path, required=True)
    interpretation.add_argument("--campaign-dir", type=Path, required=True)
    interpretation.add_argument("--out-dir", type=Path, required=True)
    interpretation.add_argument("--model", required=True)
    interpretation.add_argument("--effort", required=True)
    interpretation.add_argument("--max-sessions", type=int, required=True)
    interpretation.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    interpretation.add_argument("--timeout-seconds", type=int, default=1800)
    interpretation.add_argument("--ids", nargs="*")
    interpretation.add_argument("--dry-run", action="store_true")
    adjudicate = subparsers.add_parser("adjudicate-interpretation-batch")
    adjudicate.add_argument("--manifest", type=Path, required=True)
    adjudicate.add_argument("--campaign-dir", type=Path, required=True)
    adjudicate.add_argument("--out-dir", type=Path, required=True)
    adjudicate.add_argument("--model", required=True)
    adjudicate.add_argument("--effort", required=True)
    adjudicate.add_argument("--max-sessions", type=int, required=True)
    adjudicate.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    adjudicate.add_argument("--timeout-seconds", type=int, default=1800)
    adjudicate.add_argument("--ids", nargs="*")
    adjudicate.add_argument("--dry-run", action="store_true")
    revalidate_interpretation = subparsers.add_parser("revalidate-interpretation-batch")
    revalidate_interpretation.add_argument("--run-dir", type=Path, required=True)
    revalidate_interpretation.add_argument("--kind", choices=["generation", "adjudication"], required=True)
    diagnose_interpretation = subparsers.add_parser("diagnose-interpretation-batch")
    diagnose_interpretation.add_argument("--run-dir", type=Path, required=True)
    diagnose_interpretation.add_argument("--out-dir", type=Path, required=True)
    diagnose_interpretation.add_argument("--expected-valid", type=int, required=True)
    diagnose_interpretation.add_argument("--expected-failure", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in {
        "run-batch",
        "revalidate-batch",
        "run-interpretation-batch",
        "adjudicate-interpretation-batch",
        "revalidate-interpretation-batch",
        "diagnose-interpretation-batch",
    }:
        parser.print_help()
        return 2
    try:
        if args.command == "revalidate-batch":
            project_root = Path(__file__).resolve().parents[2]
            results = revalidate_batch(args.run_dir, project_root)
            failures = sum(result["effective_terminal_status"] != "success" for result in results)
            print(json.dumps({"rows": len(results), "success": len(results) - failures, "failures": failures}))
            return 1 if failures else 0
        if args.command == "revalidate-interpretation-batch":
            project_root = Path(__file__).resolve().parents[2]
            results = revalidate_interpretation_batch(args.run_dir, project_root, kind=args.kind)
            failures = sum(result["effective_terminal_status"] != "success" for result in results)
            print(json.dumps({"rows": len(results), "success": len(results) - failures, "failures": failures}))
            return 1 if failures else 0
        if args.command == "diagnose-interpretation-batch":
            expected_failures: dict[str, str] = {}
            for value in args.expected_failure:
                row_id, separator, code = value.partition("=")
                if not separator or not row_id or not code or row_id in expected_failures:
                    raise AuditRunnerError(f"invalid --expected-failure value: {value}")
                expected_failures[row_id] = code
            project_root = Path(__file__).resolve().parents[2]
            summary = diagnose_interpretation_batch(
                args.run_dir,
                args.out_dir,
                project_root,
                expected_valid=args.expected_valid,
                expected_failures=expected_failures,
            )
            print(json.dumps(summary, sort_keys=True))
            return 0 if summary["gate_pass"] else 1
        is_adjudication = args.command == "adjudicate-interpretation-batch"
        role = "sol" if is_adjudication else "terra" if args.command == "run-interpretation-batch" else args.role
        options = AuditOptions(
            role=role,
            model=args.model,
            effort=args.effort,
            timeout_seconds=args.timeout_seconds,
        )
        validate_policy(options)
        rows = (
            load_adjudication_manifest(args.manifest, args.campaign_dir)
            if is_adjudication
            else load_source_manifest(args.manifest, args.campaign_dir)
        )
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
                        "role": role,
                        "model": args.model,
                        "effort": args.effort,
                    },
                    sort_keys=True,
                )
            )
            return 0
        project_root = Path(__file__).resolve().parents[2]
        if args.command == "run-interpretation-batch":
            results = run_interpretation_batch(
                rows,
                campaign_dir=args.campaign_dir,
                output_dir=args.out_dir,
                project_root=project_root,
                options=options,
                max_sessions=args.max_sessions,
                concurrency=args.concurrency,
            )
        elif is_adjudication:
            results = run_adjudication_batch(
                rows,
                campaign_dir=args.campaign_dir,
                output_dir=args.out_dir,
                options=options,
                max_sessions=args.max_sessions,
                concurrency=args.concurrency,
            )
        else:
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
