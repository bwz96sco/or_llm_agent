from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from or_llm_agent.bwor import (
    default_bwor_dataset,
    default_bwor_run_dataset,
    default_or_ci_root,
    default_problem_path,
    load_bwor_record,
    load_bwor_run_record,
    load_problem,
    repo_root,
)
from or_llm_agent.code_blocks import extract_python_module, has_build_model_contract
from or_llm_agent.codex_agent import (
    CodexAgentOptions,
    CodexAgentPaths,
    build_agent_paths,
    build_codex_run_metadata,
    codex_exec_model_args,
    codex_run_metadata_summary_fields,
    neutral_work_dir,
    proof_anchor_ledger_path,
    run_codex_agent,
)
from or_llm_agent.json_blocks import extract_json_object
from or_llm_agent.or_ci import SpecValidationResult, VerificationResult, or_ci_command, run_or_ci_validate_spec, run_or_ci_verify
from or_llm_agent.prompts import (
    CAPABILITY_SYSTEM_PROMPT,
    CLARIFICATION_SYSTEM_PROMPT,
    OR_CI_SYSTEM_PROMPT,
    PROBLEM_SPEC_SYSTEM_PROMPT,
    build_clarification_question_prompt,
    build_clarified_problem_spec_prompt,
    build_or_ci_prompt,
    build_problem_spec_prompt,
    build_statement_capability_prompt,
)
from or_llm_agent.provider import query_llm, required_env_names
from or_llm_agent.redaction import env_status, redact_text
from or_llm_agent.revision import run_revision


class CLIError(Exception):
    pass


@dataclass(frozen=True)
class GenerationInputs:
    problem_id: str
    record: dict[str, Any]
    problem_path: Path
    problem: dict[str, Any]


@dataclass(frozen=True)
class ProblemSpecAgentResult:
    raw_text: str
    returncode: int
    timed_out: bool
    events_path: Path
    last_message_path: Path
    stderr: str
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FidelityReviewAgentResult:
    raw_text: str
    returncode: int
    timed_out: bool
    events_path: Path
    last_message_path: Path
    stderr: str
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityAgentResult:
    raw_text: str
    returncode: int
    timed_out: bool
    events_path: Path
    last_message_path: Path
    stderr: str
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveBatchCase:
    problem_id: str
    statement_path: Path
    case_dir: Path


@dataclass(frozen=True)
class SolveBatchCaseResult:
    problem_id: str
    exit_code: int
    statement_path: Path
    case_dir: Path
    error: str = ""


@dataclass(frozen=True)
class SolveClarifiedBatchCase:
    problem_id: str
    case_dir: Path
    clarification_path: Path
    resolution_dir: Path


@dataclass(frozen=True)
class SolveClarifiedBatchCaseResult:
    problem_id: str
    row: dict[str, Any]


@dataclass(frozen=True)
class ClarificationAgentResult:
    raw_text: str
    returncode: int
    timed_out: bool
    events_path: Path
    last_message_path: Path
    stderr: str
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClarificationQuestion:
    id: str
    issue_type: str
    prompt: str
    source_evidence: str
    allowed_answer_type: str
    options: tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class ClarificationAnswer:
    question_id: str
    answer: Any
    reviewer: str
    rationale: str
    source: str


FIDELITY_ACCEPTED_STATUSES = {"accepted", "llm_accepted"}
FIDELITY_REJECTED_STATUSES = {"rejected", "llm_rejected"}
SOURCE_FIDELITY_RUBRIC_VERSION = "source_fidelity_v1"
LEGACY_SOURCE_FIDELITY_RUBRIC_VERSION = "legacy-flat"
SOURCE_FIDELITY_DIMENSIONS = (
    "source_suitability",
    "data_completeness",
    "sets_and_indices",
    "numeric_parameters",
    "action_space",
    "objective",
    "units_and_scaling",
    "constraint_families",
    "metamorphic_coverage",
    "clarification_dependency",
    "materiality",
)
SOURCE_FIDELITY_DIMENSION_STATUSES = {"pass", "warn", "fail", "not_applicable"}
SOURCE_FIDELITY_SEVERITIES = {"none", "minor", "major", "critical"}
SOURCE_FIDELITY_BLOCKING_SEVERITIES = {"major", "critical"}
SOURCE_FIDELITY_HARD_BLOCK_DIMENSIONS = {
    "data_completeness",
    "action_space",
    "objective",
    "units_and_scaling",
    "constraint_families",
}
CAPABILITY_STATUSES = {"supported", "needs_human", "unsupported"}
CAPABILITY_BLOCKING_STATUSES = {"needs_human", "unsupported"}
CLARIFICATION_ISSUE_TYPES = {
    "missing_numeric_data",
    "ambiguous_objective",
    "unit_conflict",
    "domain_choice",
    "timing_convention",
    "data_conflict",
    "modeling_convention",
}
CLARIFICATION_ANSWER_TYPES = {"free_text", "single_choice", "multi_choice", "number", "boolean"}
CLARIFICATION_RESOLUTION_STATUSES = {"answered", "partially_answered", "rejected", "unresolved"}


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except CLIError as exc:
        print(redact_text(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "health":
        return health_command(args)
    if args.command == "classify-statement":
        return classify_statement_command(args)
    if args.command == "spec":
        return spec_command(args)
    if args.command == "generate":
        return generate_command(args)
    if args.command == "revise":
        return revise_command(args)
    if args.command == "verify":
        return verify_command(args)
    if args.command == "pilot":
        return pilot_command(args)
    if args.command == "solve":
        return solve_command(args)
    if args.command == "solve-batch":
        return solve_batch_command(args)
    if args.command == "prepare-clarification":
        return prepare_clarification_command(args)
    if args.command == "answer-clarification":
        return answer_clarification_command(args)
    if args.command == "solve-clarified":
        return solve_clarified_command(args)
    if args.command == "prepare-clarification-batch":
        return prepare_clarification_batch_command(args)
    if args.command == "solve-clarified-batch":
        return solve_clarified_batch_command(args)
    if args.command == "review-fidelity":
        return review_fidelity_command(args)
    if args.command == "review-fidelity-batch":
        return review_fidelity_batch_command(args)
    if args.command == "resolve-fidelity":
        return resolve_fidelity_command(args)
    if args.command == "resolve-fidelity-batch":
        return resolve_fidelity_batch_command(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="or-llm-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="check local OR-LLM-Agent and OR-CI readiness")
    health.add_argument("--model", default="o3-mini")
    health.add_argument("--live", action="store_true", help="perform a minimal provider request")
    health.add_argument("--agent", action="store_true", help="check Codex agent-mode readiness")

    classify = subparsers.add_parser("classify-statement", help="classify source statement support before ProblemSpec generation")
    classify.add_argument("--mode", choices=("agent",), default="agent")
    classify.add_argument("--statement-file", required=True, type=Path)
    classify.add_argument("--problem-id", required=True)
    classify.add_argument("--out", required=True, type=Path)
    classify.add_argument("--raw", type=Path, help="raw classifier output; defaults next to --out")
    classify.add_argument("--artifact-dir", type=Path, help="agent-mode artifact root; defaults to classifier output directory")
    _add_agent_options(classify)

    spec = subparsers.add_parser("spec", help="generate and validate an OR-CI problem metadata spec")
    spec.add_argument("--mode", choices=("agent",), default="agent")
    spec.add_argument("--statement-file", required=True, type=Path)
    spec.add_argument("--problem-id", required=True)
    spec.add_argument("--out", required=True, type=Path)
    spec.add_argument("--raw", required=True, type=Path)
    spec.add_argument("--status", type=Path, help="status JSON path; defaults to the spec output directory")
    spec.add_argument("--artifact-dir", type=Path, help="agent-mode artifact root; defaults to spec output directory")
    spec.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    _add_agent_options(spec)

    generate = subparsers.add_parser("generate", help="generate an OR-CI build_model submission")
    generate_source = generate.add_mutually_exclusive_group(required=True)
    generate_source.add_argument("--bwor-id")
    generate_source.add_argument("--problem", type=Path)
    generate.add_argument("--statement-file", type=Path, help="natural-language statement for explicit --problem runs")
    generate.add_argument("--model", default="o3-mini")
    generate.add_argument("--mode", choices=("api", "agent"), default="api")
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--raw", required=True, type=Path)
    generate.add_argument("--artifact-dir", type=Path, help="agent-mode artifact root; defaults to raw output parent")
    generate.add_argument("--dataset", default=default_bwor_dataset(), type=Path)
    generate.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    _add_agent_options(generate)

    revise = subparsers.add_parser("revise", help="revise an inherited OR model under a visible requirement change")
    revise.add_argument("--problem-id", required=True)
    revise.add_argument("--base-submission", required=True, type=Path)
    revise.add_argument("--base-statement", required=True, type=Path)
    revise.add_argument("--change-log", required=True, type=Path)
    revise.add_argument("--problem", required=True, type=Path)
    revise.add_argument("--artifact-dir", required=True, type=Path)
    revise.add_argument("--witness-file", type=Path, help="optional single disclosed counterexample for one repair attempt")
    _add_agent_options(revise)

    verify = subparsers.add_parser("verify", help="verify a generated submission with standalone OR-CI")
    verify.add_argument("--problem", required=True, type=Path)
    verify.add_argument("--submission", required=True, type=Path)
    verify.add_argument("--out", required=True, type=Path)

    pilot = subparsers.add_parser("pilot", help="run generate plus OR-CI verify for a BWOR batch")
    pilot.add_argument("--ids", required=True, nargs="+")
    pilot.add_argument("--model", default="o3-mini")
    pilot.add_argument("--mode", choices=("api", "agent"), default="api")
    pilot.add_argument("--artifact-dir", required=True, type=Path)
    pilot.add_argument("--dataset", default=default_bwor_dataset(), type=Path)
    pilot.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    pilot.add_argument("--reuse-submissions", action="store_true")
    _add_agent_options(pilot)

    solve = subparsers.add_parser("solve", help="run statement-to-spec-to-model-to-OR-CI verification")
    solve.add_argument("--mode", choices=("agent",), default="agent")
    solve.add_argument("--statement-file", required=True, type=Path)
    solve.add_argument("--problem-id", required=True)
    solve.add_argument("--artifact-dir", required=True, type=Path)
    solve.add_argument("--model", default="o3-mini")
    solve.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    solve.add_argument(
        "--source-anchor-contract",
        default="none",
        choices=("none", "proof_carrying"),
        help="agent mode: require proof-anchor ledger output for model generation",
    )
    _add_agent_options(solve)

    solve_batch = subparsers.add_parser("solve-batch", help="run statement-to-spec-to-model solves for a batch")
    solve_batch.add_argument("--mode", choices=("agent",), default="agent")
    solve_batch.add_argument("--ids", required=True, nargs="+")
    solve_batch.add_argument("--artifact-dir", required=True, type=Path)
    solve_batch.add_argument("--statements-dir", type=Path, help="directory containing <problem-id>.txt statement files")
    solve_batch.add_argument(
        "--dataset",
        default=default_bwor_run_dataset(),
        type=Path,
        help="clean BWOR run JSONL with exactly id, en_question, answer",
    )
    solve_batch.add_argument("--model", default="o3-mini")
    solve_batch.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    solve_batch.add_argument(
        "--source-anchor-contract",
        default="none",
        choices=("none", "proof_carrying"),
        help="agent mode: require proof-anchor ledger output for each model generation",
    )
    solve_batch.add_argument(
        "--agent-concurrency",
        default=2,
        type=int,
        help="agent mode: maximum number of solve cases to run concurrently; 1 preserves serial execution",
    )
    _add_agent_options(solve_batch)

    prepare_clarification = subparsers.add_parser(
        "prepare-clarification",
        help="generate human clarification questions for a needs_human solve artifact",
    )
    prepare_clarification.add_argument("--artifact-dir", required=True, type=Path, help="single blocked solve artifact directory")
    prepare_clarification.add_argument("--out", required=True, type=Path, help="question artifact JSON path")
    _add_agent_options(prepare_clarification)

    answer_clarification = subparsers.add_parser(
        "answer-clarification",
        help="validate and stamp human clarification answers for a blocked solve artifact",
    )
    answer_clarification.add_argument("--artifact-dir", required=True, type=Path, help="single blocked solve artifact directory")
    answer_clarification.add_argument("--answers", required=True, type=Path, help="answer artifact JSON path")
    answer_clarification.add_argument("--reviewer", required=True)

    solve_clarified = subparsers.add_parser(
        "solve-clarified",
        help="solve a needs_human case using approved clarification answers",
    )
    solve_clarified.add_argument("--artifact-dir", required=True, type=Path, help="source blocked solve artifact directory")
    solve_clarified.add_argument("--clarification", required=True, type=Path, help="answer artifact JSON path")
    solve_clarified.add_argument("--resolution-dir", required=True, type=Path, help="separate artifact directory for the clarified run")
    solve_clarified.add_argument("--mode", choices=("agent",), default="agent")
    solve_clarified.add_argument("--model", default="o3-mini")
    solve_clarified.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    _add_agent_options(solve_clarified)

    prepare_clarification_batch = subparsers.add_parser(
        "prepare-clarification-batch",
        help="generate clarification questions for needs_human cases in a solve-batch",
    )
    prepare_clarification_batch.add_argument("--artifact-dir", required=True, type=Path, help="solve-batch artifact directory")
    prepare_clarification_batch.add_argument("--ids", nargs="+")
    _add_agent_options(prepare_clarification_batch)

    solve_clarified_batch = subparsers.add_parser(
        "solve-clarified-batch",
        help="solve clarified needs_human cases in a solve-batch",
    )
    solve_clarified_batch.add_argument("--artifact-dir", required=True, type=Path, help="solve-batch artifact directory")
    solve_clarified_batch.add_argument("--clarifications-dir", required=True, type=Path, help="directory containing answer artifacts")
    solve_clarified_batch.add_argument("--ids", nargs="+")
    solve_clarified_batch.add_argument("--mode", choices=("agent",), default="agent")
    solve_clarified_batch.add_argument("--model", default="o3-mini")
    solve_clarified_batch.add_argument("--or-ci-root", default=default_or_ci_root(), type=Path)
    solve_clarified_batch.add_argument(
        "--agent-concurrency",
        default=2,
        type=int,
        help="agent mode: maximum number of clarified solve cases to run concurrently; 1 preserves serial execution",
    )
    _add_agent_options(solve_clarified_batch)

    review = subparsers.add_parser("review-fidelity", help="record a source-statement fidelity review")
    review.add_argument("--artifact-dir", required=True, type=Path, help="single solve artifact directory")
    review.add_argument("--mode", choices=("manual", "agent"), default="manual")
    review.add_argument("--status", choices=("accepted", "rejected"))
    review.add_argument("--reviewer")
    review.add_argument("--note")
    review.add_argument("--evidence", action="append", default=[])
    _add_agent_options(review)

    review_batch = subparsers.add_parser("review-fidelity-batch", help="record fidelity reviews for a solve-batch")
    review_batch.add_argument("--artifact-dir", required=True, type=Path, help="solve-batch artifact directory")
    review_batch.add_argument("--ids", nargs="+")
    review_batch.add_argument("--mode", choices=("manual", "agent"), default="manual")
    review_batch.add_argument("--status", choices=("accepted", "rejected"))
    review_batch.add_argument("--reviewer")
    review_batch.add_argument("--note")
    review_batch.add_argument("--evidence", action="append", default=[])
    _add_agent_options(review_batch)

    resolve = subparsers.add_parser("resolve-fidelity", help="repair and classify rejected source-statement fidelity")
    resolve.add_argument("--artifact-dir", required=True, type=Path, help="single solve artifact directory")
    resolve.add_argument("--resolution-dir", type=Path, help="directory for repaired solve artifacts")
    resolve.add_argument("--mode", choices=("agent",), default="agent")
    resolve.add_argument("--force", action="store_true", help="run even when source fidelity is not rejected")
    resolve.add_argument("--impact-tolerance-abs", default=1e-6, type=float)
    resolve.add_argument("--impact-tolerance-rel", default=1e-6, type=float)
    _add_agent_options(resolve)

    resolve_batch = subparsers.add_parser("resolve-fidelity-batch", help="repair and classify rejected fidelity cases in a solve-batch")
    resolve_batch.add_argument("--artifact-dir", required=True, type=Path, help="solve-batch artifact directory")
    resolve_batch.add_argument("--ids", nargs="+")
    resolve_batch.add_argument("--resolution-dir", type=Path, help="root directory for repaired solve artifacts")
    resolve_batch.add_argument("--mode", choices=("agent",), default="agent")
    resolve_batch.add_argument("--force", action="store_true", help="run even when source fidelity is not rejected")
    resolve_batch.add_argument("--impact-tolerance-abs", default=1e-6, type=float)
    resolve_batch.add_argument("--impact-tolerance-rel", default=1e-6, type=float)
    _add_agent_options(resolve_batch)
    return parser


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-model", help="agent mode: model for nested codex exec")
    parser.add_argument("--codex-reasoning-effort", help="agent mode: pass model_reasoning_effort to nested codex exec")
    parser.add_argument("--codex-sandbox", default="workspace-write", choices=("read-only", "workspace-write", "danger-full-access"))
    parser.add_argument("--codex-approval", default="never", choices=("untrusted", "on-failure", "on-request", "never"))
    parser.add_argument("--max-repair-attempts", default=3, type=int)
    parser.add_argument("--codex-timeout-seconds", default=900, type=int, help="agent mode: timeout for one nested codex exec run; <=0 disables")


def revise_command(args: argparse.Namespace) -> int:
    try:
        result = run_revision(
            problem_id=args.problem_id,
            base_submission_path=args.base_submission.resolve(),
            base_statement_path=args.base_statement.resolve(),
            change_log_path=args.change_log.resolve(),
            problem_path=args.problem.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            options=_codex_options_from_args(args),
            witness_path=args.witness_file.resolve() if args.witness_file else None,
        )
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
    print(f"revision {result.problem_id}: {result.nominal_status}/{result.nominal_classification}")
    return 0 if result.returncode == 0 and result.nominal_status == "PASS" else 1


def _codex_options_from_args(args: argparse.Namespace) -> CodexAgentOptions:
    return CodexAgentOptions(
        codex_model=args.codex_model,
        codex_sandbox=args.codex_sandbox,
        codex_approval=args.codex_approval,
        max_repair_attempts=args.max_repair_attempts,
        timeout_seconds=args.codex_timeout_seconds,
        codex_reasoning_effort=args.codex_reasoning_effort,
        source_anchor_contract=getattr(args, "source_anchor_contract", "none"),
    )


def health_command(args: argparse.Namespace) -> int:
    load_dotenv()
    checks: list[tuple[str, bool, str]] = []

    if args.agent:
        checks.append(_check_codex_cli())
        checks.append(_check_codex_exec())
    else:
        env_names = required_env_names(args.model)
        if len(env_names) == 1:
            checks.append((env_names[0], bool(os.getenv(env_names[0])), env_status(*env_names)))
        else:
            checks.append(("Claude provider key", any(os.getenv(name) for name in env_names), env_status(*env_names)))
        checks.append(("OPENAI_API_BASE", True, env_status("OPENAI_API_BASE") if not args.model.lower().startswith("claude") else "not used"))

    checks.append(_check_gurobi())
    checks.append(_check_or_ci_import())
    checks.append(_check_or_ci_cli())
    if args.live and not args.agent:
        checks.append(_check_live_provider(args.model))

    for name, ok, detail in checks:
        label = "OK" if ok else "FAIL"
        print(f"{label} {name}: {redact_text(detail)}")

    return 0 if all(ok for _, ok, _ in checks) else 1


def classify_statement_command(args: argparse.Namespace) -> int:
    statement = _read_statement(args.statement_file)
    raw_path = args.raw or args.out.with_name(f"{args.out.stem}-raw.txt")
    artifact_dir = args.artifact_dir or args.out.parent
    result = classify_statement(
        problem_id=args.problem_id,
        statement=statement,
        out_path=args.out,
        raw_path=raw_path,
        artifact_dir=artifact_dir,
        args=args,
    )
    print(
        f"{args.problem_id}: capability_status={result['status']} "
        f"generation={result['capability_generation_status']}"
    )
    if result.get("agent_stderr"):
        print(redact_text(str(result["agent_stderr"])), file=sys.stderr)
    return 0 if result["capability_generation_status"] == "classified" else 1


def spec_command(args: argparse.Namespace) -> int:
    statement = _read_statement(args.statement_file)
    status_path = args.status or _default_spec_status_path(args.out)
    artifact_dir = args.artifact_dir or args.out.parent
    result = generate_problem_spec(
        problem_id=args.problem_id,
        statement=statement,
        out_path=args.out,
        raw_path=args.raw,
        status_path=status_path,
        artifact_dir=artifact_dir,
        args=args,
    )
    print(
        f"{args.problem_id}: spec_generation={result['spec_generation_status']} "
        f"spec_validation={result['spec_validation_status']}"
    )
    if result.get("generation_error"):
        print(redact_text(str(result["generation_error"])), file=sys.stderr)
    if result.get("validation_stderr"):
        print(redact_text(str(result["validation_stderr"])), file=sys.stderr)
    return 0 if _spec_is_ready(result) else 1


def generate_command(args: argparse.Namespace) -> int:
    inputs = _generation_inputs_from_args(args)
    if args.mode == "agent":
        paths = build_agent_paths(
            problem_id=inputs.problem_id,
            artifact_root=(args.artifact_dir or args.raw.parent),
            submission_path=args.out,
            raw_path=args.raw,
        )
        result = generate_agent_submission(
            inputs=inputs,
            paths=paths,
            args=args,
        )
        verification = run_or_ci_verify(
            problem_path=inputs.problem_path,
            submission_path=paths.submission_path,
            report_path=paths.report_path,
            cwd=repo_root(),
        )
        print(f"{inputs.problem_id}: final_classification={verification.classification} status={verification.status}")
    else:
        result = generate_submission(
            inputs=inputs,
            model=args.model,
            out_path=args.out,
            raw_path=args.raw,
        )
    print(f"{inputs.problem_id}: generation={result['generation_status']}")
    if result["generation_error"]:
        print(redact_text(result["generation_error"]), file=sys.stderr)
    return 0 if result["generation_status"] == "generated" else 1


def verify_command(args: argparse.Namespace) -> int:
    verification = run_or_ci_verify(
        problem_path=args.problem,
        submission_path=args.submission,
        report_path=args.out,
        cwd=repo_root(),
    )
    _print_verification(verification)
    return verification.returncode


def pilot_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    submissions_dir = artifact_dir / "submissions"
    raw_dir = artifact_dir / "raw"
    reports_dir = artifact_dir / "reports"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for problem_id in args.ids:
        submission_path = submissions_dir / f"{problem_id}.py"
        raw_path = raw_dir / f"{problem_id}.txt"
        report_path = reports_dir / f"{problem_id}.json"
        inputs = _bwor_generation_inputs(problem_id, args.dataset, args.or_ci_root)

        if args.reuse_submissions and submission_path.exists():
            generation = {
                "generation_status": "reused",
                "generation_error": "",
                "raw_response": raw_path,
                "submission": submission_path,
                "generation_mode": args.mode,
                "agent_returncode": None,
            }
        elif args.mode == "agent":
            generation = generate_agent_submission(
                inputs=inputs,
                paths=build_agent_paths(
                    problem_id=problem_id,
                    artifact_root=artifact_dir,
                    submission_path=submission_path,
                    raw_path=raw_path,
                    report_path=report_path,
                ),
                args=args,
            )
        else:
            generation = generate_submission(
                inputs=inputs,
                model=args.model,
                out_path=submission_path,
                raw_path=raw_path,
            )

        verification = run_or_ci_verify(
            problem_path=inputs.problem_path,
            submission_path=submission_path,
            report_path=report_path,
            cwd=repo_root(),
        )
        row = build_pilot_row(
            artifact_dir=artifact_dir,
            problem_id=problem_id,
            model=args.model,
            generation=generation,
            verification=verification,
            report_path=report_path,
        )
        rows.append(row)
        print(f"{problem_id}: generation={row['generation_status']} classification={row['classification']}")

    summary = summarize(rows)
    (artifact_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(artifact_dir / "report.md", args, summary, rows)
    print(f"wrote {artifact_dir / 'report.md'}")
    return 0


def solve_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    spec_dir = artifact_dir / "spec"
    submissions_dir = artifact_dir / "submissions"
    raw_dir = artifact_dir / "raw"
    reports_dir = artifact_dir / "reports"
    for directory in (spec_dir, submissions_dir, raw_dir, reports_dir, artifact_dir / "sessions"):
        directory.mkdir(parents=True, exist_ok=True)

    statement = _read_statement(args.statement_file)
    problem_path = spec_dir / "problem.json"
    spec_raw_path = raw_dir / "spec.txt"
    spec_status_path = spec_dir / "status.json"
    spec_fidelity_review_path = spec_dir / "fidelity-review.md"
    spec_fidelity_report_path = spec_dir / "fidelity-review.json"
    capability_path = spec_dir / "capability.json"
    capability_raw_path = raw_dir / "capability.txt"
    capability_result = classify_statement(
        problem_id=args.problem_id,
        statement=statement,
        out_path=capability_path,
        raw_path=capability_raw_path,
        artifact_dir=artifact_dir,
        args=args,
    )

    summary: dict[str, Any] = {
        "problem_id": args.problem_id,
        "statement_file": str(args.statement_file),
        "capability": _relative(capability_path, artifact_dir),
        "capability_raw": _relative(capability_raw_path, artifact_dir),
        **_capability_summary_fields(capability_result),
        "spec": _relative(problem_path, artifact_dir),
        "spec_raw": _relative(spec_raw_path, artifact_dir),
        "spec_status": _relative(spec_status_path, artifact_dir),
        "spec_generation_status": "skipped",
        "spec_validation_status": "skipped",
        "spec_generation_error": "",
        "spec_validation_returncode": None,
        "spec_attempt_count": 0,
        "spec_repair_status": "skipped",
        "spec_fidelity_status": "not_reviewed",
        "spec_fidelity_review": _relative(spec_fidelity_review_path, artifact_dir),
        "spec_fidelity_report": _relative(spec_fidelity_report_path, artifact_dir),
        "model_generation_status": "skipped",
        "verification_status": "skipped",
        "classification": "skipped",
        "terminal_state": "excluded_with_reason",
        "accounting_state": "excluded_with_reason",
        "exclusion_reason": "not_generated",
        "blocking_reason": "not_generated",
    }

    if not _capability_is_supported(capability_result):
        summary["classification"] = "blocked_capability"
        summary["reason"] = (
            f"capability routing returned {capability_result['status']}; "
            "ProblemSpec generation skipped"
        )
        _apply_solve_terminal_accounting(summary)
        _write_summary(
            spec_status_path,
            {
                "problem_id": args.problem_id,
                "spec_generation_status": "skipped",
                "spec_validation_status": "skipped",
                "reason": summary["reason"],
            },
        )
        _write_spec_fidelity_review(
            spec_fidelity_review_path,
            report_path=spec_fidelity_report_path,
            summary=summary,
            statement=statement,
            problem_path=problem_path,
        )
        _write_summary(artifact_dir / "summary.json", summary)
        print(
            f"{args.problem_id}: capability={summary['capability_status']} "
            "spec_generation=skipped model_generation=skipped"
        )
        return 1

    spec_result = generate_problem_spec(
        problem_id=args.problem_id,
        statement=statement,
        out_path=problem_path,
        raw_path=spec_raw_path,
        status_path=spec_status_path,
        artifact_dir=artifact_dir,
        args=args,
    )
    summary.update(
        {
            "spec_generation_status": spec_result["spec_generation_status"],
            "spec_validation_status": spec_result["spec_validation_status"],
            "spec_generation_error": spec_result.get("generation_error", ""),
            "spec_validation_returncode": spec_result.get("validation_returncode"),
            "spec_attempt_count": spec_result.get("spec_attempt_count", 1),
            "spec_repair_status": spec_result.get("spec_repair_status", "not_needed"),
            **codex_run_metadata_summary_fields(spec_result.get("codex_run_metadata"), prefix="spec"),
        }
    )

    if not _spec_is_ready(spec_result):
        summary["reason"] = "spec validation failed; model generation skipped"
        _apply_solve_terminal_accounting(summary)
        _write_spec_fidelity_review(
            spec_fidelity_review_path,
            report_path=spec_fidelity_report_path,
            summary=summary,
            statement=statement,
            problem_path=problem_path,
        )
        _write_summary(artifact_dir / "summary.json", summary)
        print(f"{args.problem_id}: spec_validation={summary['spec_validation_status']} model_generation=skipped")
        return 1

    inputs = GenerationInputs(
        problem_id=args.problem_id,
        record={"id": args.problem_id, "en_question": statement},
        problem_path=problem_path,
        problem=load_problem(problem_path),
    )
    submission_path = submissions_dir / f"{args.problem_id}.py"
    model_raw_path = raw_dir / f"{args.problem_id}.txt"
    report_path = reports_dir / f"{args.problem_id}.json"
    generation = generate_agent_submission(
        inputs=inputs,
        paths=build_agent_paths(
            problem_id=args.problem_id,
            artifact_root=artifact_dir,
            submission_path=submission_path,
            raw_path=model_raw_path,
            report_path=report_path,
        ),
        args=args,
    )
    verification = run_or_ci_verify(
        problem_path=problem_path,
        submission_path=submission_path,
        report_path=report_path,
        cwd=repo_root(),
    )
    summary.update(
        {
            "submission": _relative(submission_path, artifact_dir),
            "model_raw": _relative(model_raw_path, artifact_dir),
            "report": _relative(report_path, artifact_dir),
            "model_generation_status": generation["generation_status"],
            "model_generation_error": generation["generation_error"],
            "verification_status": verification.status,
            "classification": verification.classification,
            "verification_note": "passed generated spec" if verification.status == "PASS" else "failed generated spec",
            "verify_returncode": verification.returncode,
            "verify_stdout": verification.stdout,
            "verify_stderr": verification.stderr,
            "source_anchor_contract": generation.get("source_anchor_contract", "none"),
            "proof_anchor_ledger": _relative(generation["proof_anchor_ledger"], artifact_dir)
            if generation.get("proof_anchor_ledger")
            else "",
            "proof_anchor_ledger_schema_status": generation.get("proof_anchor_ledger_schema_status", ""),
            "proof_anchor_ledger_schema_validation": generation.get("proof_anchor_ledger_schema_validation", {}),
            **codex_run_metadata_summary_fields(generation.get("codex_run_metadata"), prefix="generation"),
        }
    )
    _apply_solve_terminal_accounting(summary)
    _write_spec_fidelity_review(
        spec_fidelity_review_path,
        report_path=spec_fidelity_report_path,
        summary=summary,
        statement=statement,
        problem_path=problem_path,
    )
    _write_summary(artifact_dir / "summary.json", summary)
    print(
        f"{args.problem_id}: spec_validation={summary['spec_validation_status']} "
        f"model_generation={summary['model_generation_status']} verification={summary['verification_status']} "
        f"classification={summary['classification']}"
    )
    return 0 if generation["generation_status"] == "generated" and verification.returncode == 0 else 1


def solve_batch_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    concurrency = _validate_agent_concurrency(args.agent_concurrency)
    _validate_unique_problem_ids(args.ids)
    cases = _prepare_solve_batch_cases(args, artifact_dir)
    results = _run_solve_batch_cases(args, cases, concurrency=concurrency)
    result_by_id = {result.problem_id: result for result in results}

    rows: list[dict[str, Any]] = []
    for case in cases:
        result = result_by_id[case.problem_id]
        row = _solve_batch_row(
            problem_id=case.problem_id,
            exit_code=result.exit_code,
            artifact_dir=artifact_dir,
            case_dir=case.case_dir,
            statement_path=case.statement_path,
        )
        if result.error:
            row["batch_error"] = result.error
            _apply_solve_terminal_accounting(row, batch_error=result.error)
        rows.append(row)

    summary = summarize_solve_batch(rows)
    payload = {"summary": summary, "rows": rows}
    _write_summary(artifact_dir / "summary.json", payload)
    write_solve_batch_report(artifact_dir / "report.md", args, summary, rows)
    print(f"wrote {artifact_dir / 'report.md'}")
    return 0 if all(row["exit_code"] == 0 for row in rows) else 1


def _validate_agent_concurrency(value: int) -> int:
    if value < 1:
        raise CLIError("--agent-concurrency must be >= 1")
    return value


def _validate_unique_problem_ids(problem_ids: list[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for problem_id in problem_ids:
        if problem_id in seen and problem_id not in duplicates:
            duplicates.append(problem_id)
        seen.add(problem_id)
    if duplicates:
        joined = ", ".join(duplicates)
        raise CLIError(f"solve-batch problem ids must be unique; duplicate id(s): {joined}")


def _prepare_solve_batch_cases(args: argparse.Namespace, artifact_dir: Path) -> list[SolveBatchCase]:
    cases: list[SolveBatchCase] = []
    for problem_id in args.ids:
        statement_path = _batch_statement_file(args, artifact_dir, problem_id)
        cases.append(
            SolveBatchCase(
                problem_id=problem_id,
                statement_path=statement_path,
                case_dir=artifact_dir / problem_id,
            )
        )
    return cases


def _run_solve_batch_cases(
    args: argparse.Namespace,
    cases: list[SolveBatchCase],
    *,
    concurrency: int,
) -> list[SolveBatchCaseResult]:
    if concurrency == 1 or len(cases) <= 1:
        return [_run_solve_batch_case(args, case) for case in cases]

    results: list[SolveBatchCaseResult] = []
    max_workers = min(concurrency, len(cases))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_solve_batch_case, args, case) for case in cases]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_solve_batch_case(args: argparse.Namespace, case: SolveBatchCase) -> SolveBatchCaseResult:
    solve_args = argparse.Namespace(**vars(args))
    solve_args.command = "solve"
    solve_args.problem_id = case.problem_id
    solve_args.statement_file = case.statement_path
    solve_args.artifact_dir = case.case_dir
    try:
        exit_code = solve_command(solve_args)
        return SolveBatchCaseResult(
            problem_id=case.problem_id,
            exit_code=exit_code,
            statement_path=case.statement_path,
            case_dir=case.case_dir,
        )
    except Exception as exc:
        error = redact_text(f"{type(exc).__name__}: {exc}")
        print(f"{case.problem_id}: solve failed with unexpected error: {error}", file=sys.stderr)
        return SolveBatchCaseResult(
            problem_id=case.problem_id,
            exit_code=1,
            statement_path=case.statement_path,
            case_dir=case.case_dir,
            error=error,
        )


def prepare_clarification_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    question_artifact = prepare_clarification_artifact(
        artifact_dir=artifact_dir,
        out_path=args.out.resolve(),
        args=args,
    )
    print(
        f"{question_artifact['problem_id']}: clarification_questions={len(question_artifact['questions'])} "
        f"wrote={args.out}"
    )
    return 0


def answer_clarification_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    questions = load_clarification_questions(_canonical_clarification_questions_path(artifact_dir))
    answers = load_clarification_answers(args.answers.resolve(), questions=questions, reviewer=args.reviewer)
    canonical_answers = _canonical_clarification_answers_path(artifact_dir)
    _write_clarification_artifact(canonical_answers, answers)
    gate_status = _clarification_gate_status(questions, answers)
    print(
        f"{answers['problem_id']}: clarification_status={answers['resolution_status']} "
        f"gate={gate_status} answers={len(answers['answers'])}"
    )
    return 0 if gate_status == "passed" else 1


def solve_clarified_command(args: argparse.Namespace) -> int:
    result = solve_clarified_artifact(
        artifact_dir=args.artifact_dir.resolve(),
        clarification_path=args.clarification.resolve(),
        resolution_dir=args.resolution_dir.resolve(),
        args=args,
    )
    print(
        f"{result['problem_id']}: clarification_gate={result['clarification_gate_status']} "
        f"spec_validation={result['spec_validation_status']} verification={result['verification_status']} "
        f"classification={result['classification']}"
    )
    return 0 if result["model_generation_status"] == "generated" and result["verification_status"] == "PASS" else 1


def prepare_clarification_batch_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    ids = args.ids or _needs_human_ids_from_batch_summary(artifact_dir)
    rows: list[dict[str, Any]] = []
    failures = 0
    for problem_id in ids:
        case_dir = artifact_dir / problem_id
        out_path = _canonical_clarification_questions_path(case_dir)
        try:
            question_artifact = prepare_clarification_artifact(artifact_dir=case_dir, out_path=out_path, args=args)
            rows.append(_clarification_row_from_questions(problem_id, case_dir, artifact_dir, question_artifact))
        except CLIError as exc:
            failures += 1
            rows.append(_failed_prepare_batch_row(problem_id, case_dir, artifact_dir, error=redact_text(str(exc))))
        except Exception as exc:
            failures += 1
            error = redact_text(f"{type(exc).__name__}: {exc}")
            print(f"{problem_id}: clarification prepare failed with unexpected error: {error}", file=sys.stderr)
            rows.append(_failed_prepare_batch_row(problem_id, case_dir, artifact_dir, error=error))

    summary = summarize_clarification_rows(rows)
    _write_summary(artifact_dir / "clarification-summary.json", {"summary": summary, "rows": rows})
    write_clarification_report(artifact_dir / "clarification-report.md", summary, rows)
    print(f"prepared clarification for {len(rows)} case(s); wrote {artifact_dir / 'clarification-report.md'}")
    return 0 if failures == 0 else 1


def solve_clarified_batch_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    ids = args.ids or _needs_human_ids_from_batch_summary(artifact_dir)
    concurrency = _validate_agent_concurrency(args.agent_concurrency)
    _validate_unique_problem_ids(ids)
    cases = [
        SolveClarifiedBatchCase(
            problem_id=problem_id,
            case_dir=artifact_dir / problem_id,
            clarification_path=_clarification_answers_path(args.clarifications_dir.resolve(), problem_id),
            resolution_dir=_next_attempt_dir(artifact_dir / "clarified" / problem_id),
        )
        for problem_id in ids
    ]
    results = _run_solve_clarified_batch_cases(args, cases, artifact_dir=artifact_dir, concurrency=concurrency)
    result_by_id = {result.problem_id: result for result in results}
    rows = [result_by_id[case.problem_id].row for case in cases]

    summary = summarize_clarification_rows(rows)
    _write_summary(artifact_dir / "clarification-summary.json", {"summary": summary, "rows": rows})
    write_clarification_report(artifact_dir / "clarification-report.md", summary, rows)
    print(f"solved clarified {len(rows)} case(s); wrote {artifact_dir / 'clarification-report.md'}")
    return 0 if all(row.get("clarification_gate_status") == "passed" and row.get("verification_status") == "PASS" for row in rows) else 1


def _run_solve_clarified_batch_cases(
    args: argparse.Namespace,
    cases: list[SolveClarifiedBatchCase],
    *,
    artifact_dir: Path,
    concurrency: int,
) -> list[SolveClarifiedBatchCaseResult]:
    if concurrency == 1 or len(cases) <= 1:
        return [_run_solve_clarified_batch_case(args, case, artifact_dir=artifact_dir) for case in cases]

    results: list[SolveClarifiedBatchCaseResult] = []
    max_workers = min(concurrency, len(cases))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_solve_clarified_batch_case, args, case, artifact_dir=artifact_dir)
            for case in cases
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_solve_clarified_batch_case(
    args: argparse.Namespace,
    case: SolveClarifiedBatchCase,
    *,
    artifact_dir: Path,
) -> SolveClarifiedBatchCaseResult:
    case_args = argparse.Namespace(**vars(args))
    try:
        result = solve_clarified_artifact(
            artifact_dir=case.case_dir,
            clarification_path=case.clarification_path,
            resolution_dir=case.resolution_dir,
            args=case_args,
        )
        row = _clarification_row_from_solve(result, artifact_dir)
    except CLIError as exc:
        row = _blocked_clarified_batch_row(case, artifact_dir=artifact_dir, error=redact_text(str(exc)))
    except Exception as exc:
        error = redact_text(f"{type(exc).__name__}: {exc}")
        print(f"{case.problem_id}: clarified solve failed with unexpected error: {error}", file=sys.stderr)
        row = _blocked_clarified_batch_row(case, artifact_dir=artifact_dir, error=error)
    return SolveClarifiedBatchCaseResult(problem_id=case.problem_id, row=row)


def _blocked_clarified_batch_row(
    case: SolveClarifiedBatchCase,
    *,
    artifact_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "problem_id": case.problem_id,
        "source_artifact": _relative(case.case_dir, artifact_dir),
        "clarification_status": "unresolved",
        "clarification_gate_status": "blocked",
        "error": error,
        "resolution_artifact": _relative(case.resolution_dir, artifact_dir),
        "classification": "blocked_clarification",
        "verification_status": "skipped",
        "spec_fidelity_status": "not_reviewed",
    }


def _failed_prepare_batch_row(
    problem_id: str,
    case_dir: Path,
    artifact_dir: Path,
    *,
    error: str,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "source_artifact": _relative(case_dir, artifact_dir),
        "baseline_block_reason": "",
        "clarification_status": "prepare_failed",
        "clarification_question_count": 0,
        "clarification_answer_count": 0,
        "clarification_source": "",
        "clarification_gate_status": "blocked",
        "clarification_questions": "",
        "generated_questions": [],
        "answer_provenance": [],
        "classification": "blocked_clarification",
        "verification_status": "skipped",
        "spec_fidelity_status": "not_reviewed",
        "error": error,
    }


def review_fidelity_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    review = _fidelity_review_payload_for_artifact(artifact_dir, args)
    summary = apply_fidelity_review(artifact_dir=artifact_dir, review=review)
    print(
        f"{summary['problem_id']}: spec_fidelity_status={summary['spec_fidelity_status']} "
        f"gate={summary['spec_fidelity_gate_status']}"
    )
    return 0


def review_fidelity_batch_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    review_ids = args.ids or _batch_ids_from_summary(artifact_dir)
    for problem_id in review_ids:
        case_dir = artifact_dir / problem_id
        review = _fidelity_review_payload_for_artifact(case_dir, args)
        apply_fidelity_review(artifact_dir=case_dir, review=review)

    aggregate_ids = _batch_ids_for_aggregate(artifact_dir, fallback_ids=review_ids)
    rows = [
        _solve_batch_row(
            problem_id=problem_id,
            exit_code=_case_exit_code(artifact_dir / problem_id),
            artifact_dir=artifact_dir,
            case_dir=artifact_dir / problem_id,
            statement_path=artifact_dir / "statements" / f"{problem_id}.txt",
        )
        for problem_id in aggregate_ids
    ]
    summary = summarize_solve_batch(rows)
    payload = {"summary": summary, "rows": rows}
    _write_summary(artifact_dir / "summary.json", payload)
    rubric_summary = summarize_source_fidelity_rubric(rows)
    _write_summary(
        artifact_dir / "fidelity-rubric-summary.json",
        {"summary": rubric_summary, "rows": _source_fidelity_rubric_rows(rows)},
    )
    write_fidelity_rubric_report(artifact_dir / "fidelity-rubric-report.md", rubric_summary, rows)
    report_args = argparse.Namespace(ids=aggregate_ids, mode="agent")
    write_solve_batch_report(artifact_dir / "report.md", report_args, summary, rows)
    print(f"reviewed {len(review_ids)} case(s); wrote {artifact_dir / 'report.md'}")
    return 0


def resolve_fidelity_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    resolution_dir = args.resolution_dir.resolve() if args.resolution_dir else _next_attempt_dir(artifact_dir / "fidelity-resolution")
    result = resolve_fidelity_artifact(artifact_dir=artifact_dir, resolution_dir=resolution_dir, args=args)
    print(
        f"{result['problem_id']}: fidelity_resolution={result['resolution_status']} "
        f"impact={result['impact_analysis']['classification']}"
    )
    return 0 if result["resolution_status"] != "repair_failed" else 1


def resolve_fidelity_batch_command(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    all_ids = args.ids or _batch_ids_from_summary(artifact_dir)
    ids = list(all_ids)
    if args.ids is None and not args.force:
        ids = [
            problem_id
            for problem_id in ids
            if _artifact_fidelity_status(artifact_dir / problem_id) in FIDELITY_REJECTED_STATUSES
        ]
    resolution_root = (args.resolution_dir.resolve() if args.resolution_dir else artifact_dir / "fidelity-resolution")
    results: list[dict[str, Any]] = []
    for problem_id in ids:
        case_dir = artifact_dir / problem_id
        case_resolution_dir = _next_attempt_dir(resolution_root / problem_id)
        results.append(resolve_fidelity_artifact(artifact_dir=case_dir, resolution_dir=case_resolution_dir, args=args))

    summary = summarize_fidelity_resolution(results)
    _write_summary(artifact_dir / "fidelity-resolution-summary.json", {"summary": summary, "rows": results})
    write_fidelity_resolution_report(artifact_dir / "fidelity-resolution-report.md", args, summary, results)

    rows = [
        _solve_batch_row(
            problem_id=problem_id,
            exit_code=_case_exit_code(artifact_dir / problem_id),
            artifact_dir=artifact_dir,
            case_dir=artifact_dir / problem_id,
            statement_path=artifact_dir / "statements" / f"{problem_id}.txt",
        )
        for problem_id in all_ids
    ]
    batch_summary = summarize_solve_batch(rows)
    _write_summary(artifact_dir / "summary.json", {"summary": batch_summary, "rows": rows})
    report_args = argparse.Namespace(ids=all_ids, mode="agent")
    write_solve_batch_report(artifact_dir / "report.md", report_args, batch_summary, rows)
    print(f"resolved {len(results)} case(s); wrote {artifact_dir / 'fidelity-resolution-report.md'}")
    return 0


def classify_statement(
    *,
    problem_id: str,
    statement: str,
    out_path: Path,
    raw_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    agent_result = _run_capability_agent(
        problem_id=problem_id,
        statement=statement,
        artifact_dir=artifact_dir.resolve(),
        args=args,
    )
    raw_text = redact_text(agent_result.raw_text).rstrip() + "\n"
    raw_path.write_text(raw_text, encoding="utf-8")

    parsed = extract_json_object(agent_result.raw_text)
    payload = _normalize_capability_payload(
        problem_id=problem_id,
        parsed=parsed,
        raw_path=raw_path,
        agent_result=agent_result,
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def prepare_clarification_artifact(
    *,
    artifact_dir: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = _read_json_object(artifact_dir / "summary.json")
    if not summary:
        raise CLIError(f"solve summary does not exist or is not valid JSON: {artifact_dir / 'summary.json'}")
    if summary.get("capability_status") != "needs_human":
        raise CLIError(
            f"clarification can only be prepared for capability_status=needs_human; "
            f"found {summary.get('capability_status', 'unknown')}"
        )

    problem_id = str(summary.get("problem_id") or artifact_dir.name)
    statement_path = _artifact_path(artifact_dir, summary.get("statement_file"), "statement.txt")
    statement = _read_statement(statement_path)
    capability_path = _artifact_path(artifact_dir, summary.get("capability"), "spec/capability.json")
    capability = _read_json_object(capability_path)
    raw_path = artifact_dir / "raw" / "clarification-questions.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    agent_result = _run_clarification_question_agent(
        problem_id=problem_id,
        statement=statement,
        capability=capability,
        artifact_dir=artifact_dir,
        args=args,
    )
    raw_text = redact_text(agent_result.raw_text).rstrip() + "\n"
    raw_path.write_text(raw_text, encoding="utf-8")

    if agent_result.timed_out or agent_result.returncode != 0:
        status_path = _write_clarification_agent_failure_status(
            artifact_dir=artifact_dir,
            status="agent_failed",
            raw_path=raw_path,
            agent_result=agent_result,
        )
        raise CLIError(
            f"clarification question planner failed; status={status_path}; raw_response={raw_path}"
        )

    parsed = extract_json_object(agent_result.raw_text)
    if isinstance(parsed, dict):
        parsed["problem_id"] = problem_id
        parsed["source_artifact"] = _portable_source_artifact(artifact_dir)
        parsed["blocking_status"] = "needs_human"
        payload = normalize_clarification_questions(parsed, source_path=out_path)
    else:
        status_path = _write_clarification_agent_failure_status(
            artifact_dir=artifact_dir,
            status="agent_no_json",
            raw_path=raw_path,
            agent_result=agent_result,
        )
        raise CLIError(
            f"clarification question JSON is malformed; status={status_path}; raw_response={raw_path}"
        )

    payload["raw_response"] = str(raw_path)
    payload["agent_returncode"] = agent_result.returncode
    payload["agent_timed_out"] = agent_result.timed_out
    payload["codex_events"] = str(agent_result.events_path)
    payload["last_message"] = str(agent_result.last_message_path)
    payload["agent_stderr"] = redact_text(agent_result.stderr)
    payload.update(codex_run_metadata_summary_fields(agent_result.run_metadata))
    _write_clarification_artifact(out_path, payload)
    canonical = _canonical_clarification_questions_path(artifact_dir)
    if canonical.resolve() != out_path.resolve():
        _write_clarification_artifact(canonical, payload)
    return payload


def solve_clarified_artifact(
    *,
    artifact_dir: Path,
    clarification_path: Path,
    resolution_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if resolution_dir.resolve() == artifact_dir.resolve():
        raise CLIError(
            "solve-clarified requires --resolution-dir to be different from --artifact-dir "
            f"so the blocked source artifact is preserved: {artifact_dir}"
        )

    source_summary = _read_json_object(artifact_dir / "summary.json")
    if not source_summary:
        raise CLIError(f"solve summary does not exist or is not valid JSON: {artifact_dir / 'summary.json'}")
    if source_summary.get("capability_status") != "needs_human":
        raise CLIError(
            f"solve-clarified requires source capability_status=needs_human; "
            f"found {source_summary.get('capability_status', 'unknown')}"
        )

    questions = load_clarification_questions(_canonical_clarification_questions_path(artifact_dir))
    answers = load_clarification_answers(clarification_path, questions=questions)
    gate_status = _clarification_gate_status(questions, answers)
    if gate_status != "passed":
        missing = _required_unanswered_question_ids(questions, answers)
        suffix = f"; missing required answers: {', '.join(missing)}" if missing else ""
        raise CLIError(f"clarification gate is {gate_status}{suffix}")

    problem_id = str(source_summary.get("problem_id") or artifact_dir.name)
    statement_path = _artifact_path(artifact_dir, source_summary.get("statement_file"), "statement.txt")
    statement = _read_statement(statement_path)
    clarification_context = _clarification_context(
        source_artifact_dir=artifact_dir,
        resolution_dir=resolution_dir,
        questions=questions,
        answers=answers,
        question_path=_canonical_clarification_questions_path(artifact_dir),
        answer_path=clarification_path,
        gate_status=gate_status,
    )
    clarified_statement = _statement_with_clarification(statement, clarification_context)

    spec_dir = resolution_dir / "spec"
    submissions_dir = resolution_dir / "submissions"
    raw_dir = resolution_dir / "raw"
    reports_dir = resolution_dir / "reports"
    clarification_dir = resolution_dir / "clarification"
    for directory in (spec_dir, submissions_dir, raw_dir, reports_dir, resolution_dir / "sessions", clarification_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied_questions = clarification_dir / "questions.json"
    copied_answers = clarification_dir / "answers.json"
    _write_clarification_artifact(copied_questions, questions)
    _write_clarification_artifact(copied_answers, answers)
    clarification_context["question_artifact"] = _relative(copied_questions, resolution_dir)
    clarification_context["answer_artifact"] = _relative(copied_answers, resolution_dir)

    problem_path = spec_dir / "problem.json"
    spec_raw_path = raw_dir / "spec.txt"
    spec_status_path = spec_dir / "status.json"
    spec_fidelity_review_path = spec_dir / "fidelity-review.md"
    spec_fidelity_report_path = spec_dir / "fidelity-review.json"
    spec_result = generate_problem_spec(
        problem_id=problem_id,
        statement=statement,
        out_path=problem_path,
        raw_path=spec_raw_path,
        status_path=spec_status_path,
        artifact_dir=resolution_dir,
        args=args,
        clarification_context=clarification_context,
    )

    summary: dict[str, Any] = {
        "problem_id": problem_id,
        "statement_file": str(statement_path),
        "source_artifact": str(artifact_dir),
        "resolution_artifact": str(resolution_dir),
        "clarified_from": str(artifact_dir),
        "clarification_status": answers["resolution_status"],
        "clarification_question_count": len(questions["questions"]),
        "clarification_answer_count": len(answers["answers"]),
        "clarification_source": _clarification_source(answers),
        "clarification_gate_status": gate_status,
        "clarification_questions": _relative(copied_questions, resolution_dir),
        "clarification_answers": _relative(copied_answers, resolution_dir),
        "spec": _relative(problem_path, resolution_dir),
        "spec_raw": _relative(spec_raw_path, resolution_dir),
        "spec_status": _relative(spec_status_path, resolution_dir),
        "spec_generation_status": spec_result["spec_generation_status"],
        "spec_validation_status": spec_result["spec_validation_status"],
        "spec_generation_error": spec_result.get("generation_error", ""),
        "spec_validation_returncode": spec_result.get("validation_returncode"),
        "spec_attempt_count": spec_result.get("spec_attempt_count", 1),
        "spec_repair_status": spec_result.get("spec_repair_status", "not_needed"),
        **codex_run_metadata_summary_fields(spec_result.get("codex_run_metadata"), prefix="spec"),
        "spec_fidelity_status": "not_reviewed",
        "spec_fidelity_review": _relative(spec_fidelity_review_path, resolution_dir),
        "spec_fidelity_report": _relative(spec_fidelity_report_path, resolution_dir),
        "model_generation_status": "skipped",
        "verification_status": "skipped",
        "classification": "skipped",
    }

    if not _spec_is_ready(spec_result):
        summary["reason"] = "clarified spec validation failed; model generation skipped"
        _write_spec_fidelity_review(
            spec_fidelity_review_path,
            report_path=spec_fidelity_report_path,
            summary=summary,
            statement=statement,
            problem_path=problem_path,
            clarification_context=clarification_context,
        )
        _write_summary(resolution_dir / "summary.json", summary)
        return summary

    inputs = GenerationInputs(
        problem_id=problem_id,
        record={"id": problem_id, "en_question": clarified_statement},
        problem_path=problem_path,
        problem=load_problem(problem_path),
    )
    submission_path = submissions_dir / f"{problem_id}.py"
    model_raw_path = raw_dir / f"{problem_id}.txt"
    report_path = reports_dir / f"{problem_id}.json"
    generation = generate_agent_submission(
        inputs=inputs,
        paths=build_agent_paths(
            problem_id=problem_id,
            artifact_root=resolution_dir,
            submission_path=submission_path,
            raw_path=model_raw_path,
            report_path=report_path,
        ),
        args=args,
    )
    verification = run_or_ci_verify(
        problem_path=problem_path,
        submission_path=submission_path,
        report_path=report_path,
        cwd=repo_root(),
    )
    summary.update(
        {
            "submission": _relative(submission_path, resolution_dir),
            "model_raw": _relative(model_raw_path, resolution_dir),
            "report": _relative(report_path, resolution_dir),
            "model_generation_status": generation["generation_status"],
            "model_generation_error": generation["generation_error"],
            "verification_status": verification.status,
            "classification": verification.classification,
            "verification_note": "passed clarified generated spec" if verification.status == "PASS" else "failed clarified generated spec",
            "verify_returncode": verification.returncode,
            "verify_stdout": verification.stdout,
            "verify_stderr": verification.stderr,
            **codex_run_metadata_summary_fields(generation.get("codex_run_metadata"), prefix="generation"),
        }
    )
    _write_spec_fidelity_review(
        spec_fidelity_review_path,
        report_path=spec_fidelity_report_path,
        summary=summary,
        statement=statement,
        problem_path=problem_path,
        clarification_context=clarification_context,
    )
    _write_summary(resolution_dir / "summary.json", summary)
    return summary


def generate_problem_spec(
    *,
    problem_id: str,
    statement: str,
    out_path: Path,
    raw_path: Path,
    status_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    initial_previous_problem: dict[str, Any] | None = None,
    initial_previous_response: str = "",
    initial_repair_error: str = "",
    clarification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    max_attempts = 1 + max(args.max_repair_attempts, 0)
    attempts: list[dict[str, Any]] = []
    previous_problem: dict[str, Any] | None = initial_previous_problem
    previous_response = initial_previous_response
    repair_error = initial_repair_error
    initial_repair_requested = bool(initial_repair_error)

    for attempt in range(1, max_attempts + 1):
        agent_result = _run_problem_spec_agent(
            problem_id=problem_id,
            statement=statement,
            artifact_dir=artifact_dir.resolve(),
            args=args,
            attempt=attempt,
            previous_problem=previous_problem,
            previous_response=previous_response,
            repair_error=repair_error,
            clarification_context=clarification_context,
        )
        raw_text = redact_text(agent_result.raw_text).rstrip() + "\n"
        attempt_raw_path = _spec_attempt_raw_path(raw_path, attempt)
        attempt_raw_path.write_text(raw_text, encoding="utf-8")
        raw_path.write_text(raw_text, encoding="utf-8")

        attempt_payload: dict[str, Any] = {
            "attempt": attempt,
            "mode": args.mode,
            "spec_generation_status": "generated",
            "spec_validation_status": "skipped",
            "generation_error": "",
            "raw_response": str(attempt_raw_path),
            "agent_returncode": agent_result.returncode,
            "agent_timed_out": agent_result.timed_out,
            "codex_events": str(agent_result.events_path),
            "last_message": str(agent_result.last_message_path),
            "agent_stderr": redact_text(agent_result.stderr),
            "repair_input_error": repair_error,
        }
        attempt_payload.update(codex_run_metadata_summary_fields(agent_result.run_metadata))

        problem = extract_json_object(agent_result.raw_text)
        if problem is None:
            out_path.write_text("{}\n", encoding="utf-8")
            generation_error = "response did not contain one JSON object"
            attempt_payload.update(
                {
                    "spec_generation_status": "no_json",
                    "generation_error": generation_error,
                }
            )
            attempts.append(attempt_payload)
            previous_problem = None
            previous_response = raw_text
            repair_error = generation_error
            continue

        if clarification_context is not None:
            _mark_problem_as_clarified(problem, clarification_context)
        out_path.write_text(json.dumps(problem, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validation = run_or_ci_validate_spec(problem_path=out_path, cwd=repo_root())
        attempt_payload.update(_spec_validation_payload(validation))
        attempts.append(attempt_payload)
        if validation.status == "passed":
            break

        previous_problem = problem
        previous_response = raw_text
        repair_error = _format_spec_repair_error(validation)

    final_attempt = attempts[-1]
    payload: dict[str, Any] = {
        "problem_id": problem_id,
        "mode": args.mode,
        "spec_generation_status": final_attempt["spec_generation_status"],
        "spec_validation_status": final_attempt["spec_validation_status"],
        "generation_error": final_attempt["generation_error"],
        "raw_response": str(raw_path),
        "problem": str(out_path),
        "status": str(status_path),
        "agent_returncode": final_attempt["agent_returncode"],
        "agent_timed_out": final_attempt["agent_timed_out"],
        "codex_events": final_attempt["codex_events"],
        "last_message": final_attempt["last_message"],
        "agent_stderr": final_attempt["agent_stderr"],
        "spec_attempt_count": len(attempts),
        "spec_repair_status": _spec_repair_status(attempts, initial_repair_requested=initial_repair_requested),
        "spec_attempts": attempts,
    }
    payload.update(codex_run_metadata_summary_fields(final_attempt.get("codex_run_metadata")))
    for key in ("validation_returncode", "validation_stdout", "validation_stderr"):
        if key in final_attempt:
            payload[key] = final_attempt[key]
    if clarification_context is not None:
        payload.update(
            {
                "clarified": True,
                "clarified_from": clarification_context.get("clarified_from", ""),
                "clarification_status": clarification_context.get("clarification_status", ""),
                "clarification_gate_status": clarification_context.get("clarification_gate_status", ""),
                "clarification_question_count": clarification_context.get("clarification_question_count", 0),
                "clarification_answer_count": clarification_context.get("clarification_answer_count", 0),
            }
        )
    _write_summary(status_path, payload)
    return payload


def generate_submission(
    *,
    inputs: GenerationInputs,
    model: str,
    out_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": OR_CI_SYSTEM_PROMPT},
        {"role": "user", "content": build_or_ci_prompt(inputs.problem_id, inputs.record, inputs.problem)},
    ]

    try:
        response = query_llm(messages, model_name=model)
    except Exception as exc:
        error = f"PROVIDER_ERROR: {redact_text(exc)}"
        raw_path.write_text(error + "\n", encoding="utf-8")
        out_path.write_text("# generation failed before Python code was produced\n", encoding="utf-8")
        return _generation_result("failed", error, raw_path, out_path)

    raw_path.write_text(redact_text(response).rstrip() + "\n", encoding="utf-8")
    code = extract_python_module(response)
    if code is None:
        out_path.write_text("# no fenced Python code block found\n", encoding="utf-8")
        return _generation_result("no_python_code", "response did not contain a fenced python code block", raw_path, out_path)

    out_path.write_text(code.rstrip() + "\n", encoding="utf-8")
    if not has_build_model_contract(code):
        return _generation_result("generated_without_build_model", "response did not contain def build_model", raw_path, out_path)
    return _generation_result("generated", "", raw_path, out_path)


def generate_agent_submission(
    *,
    inputs: GenerationInputs,
    paths: CodexAgentPaths,
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = _codex_options_from_args(args)
    result = run_codex_agent(
        problem_id=inputs.problem_id,
        record=inputs.record,
        problem=inputs.problem,
        problem_path=inputs.problem_path.resolve(),
        paths=paths,
        options=options,
        verify_command=or_ci_command()[0],
    )

    if not paths.submission_path.exists():
        error = f"codex exec did not write submission; returncode={result.returncode}"
        paths.submission_path.write_text("# agent mode did not produce a submission\n", encoding="utf-8")
        status = "agent_failed"
    else:
        code = paths.submission_path.read_text(encoding="utf-8")
        if has_build_model_contract(code):
            status = "generated"
            error = "" if result.returncode == 0 else f"codex exec returned nonzero status: {result.returncode}"
        else:
            status = "generated_without_build_model"
            error = "agent submission did not contain def build_model"

    payload = _generation_result(status, error, paths.raw_path, paths.submission_path)
    payload.update(
        {
            "generation_mode": "agent",
            "agent_returncode": result.returncode,
            "agent_timed_out": result.timed_out,
            "agent_status": paths.status_path,
            "agent_manifest": paths.manifest_path,
            "last_message": paths.last_message_path,
            "codex_events": paths.events_path,
            "source_anchor_contract": options.source_anchor_contract,
            "proof_anchor_ledger": proof_anchor_ledger_path(paths),
            "proof_anchor_ledger_schema_status": result.proof_anchor_ledger_validation.get("status", ""),
            "proof_anchor_ledger_schema_validation": result.proof_anchor_ledger_validation,
        }
    )
    payload.update(codex_run_metadata_summary_fields(result.run_metadata))
    return payload


def resolve_fidelity_artifact(*, artifact_dir: Path, resolution_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_summary_path = artifact_dir / "summary.json"
    source_summary = _read_json_object(source_summary_path)
    if not source_summary:
        raise CLIError(f"solve summary does not exist or is not valid JSON: {source_summary_path}")

    problem_id = str(source_summary.get("problem_id") or artifact_dir.name)
    source_fidelity_status = str(source_summary.get("spec_fidelity_status", "not_reviewed"))
    if source_fidelity_status not in FIDELITY_REJECTED_STATUSES and not args.force:
        result = {
            "problem_id": problem_id,
            "resolution_status": "skipped_not_rejected",
            "source_artifact": str(artifact_dir),
            "resolution_artifact": str(resolution_dir),
            "source_fidelity_status": source_fidelity_status,
            "repaired_fidelity_status": "",
            "created_at": _utc_now(),
            "impact_analysis": {
                "classification": "not_needed",
                "reason": f"source fidelity status is {source_fidelity_status}",
            },
        }
        report_path = resolution_dir / "fidelity-resolution.json"
        _write_summary(report_path, result)
        _apply_fidelity_resolution_result(source_summary_path, source_summary, result, artifact_dir, resolution_dir, report_path)
        return result

    statement_path = _artifact_path(artifact_dir, source_summary.get("statement_file"), "statement.txt")
    if not statement_path.is_file():
        raise CLIError(f"statement file does not exist: {statement_path}")
    statement = _read_statement(statement_path)
    previous_problem_path = _artifact_path(artifact_dir, source_summary.get("spec"), "spec/problem.json")
    previous_problem = _read_json_object(previous_problem_path)
    if not previous_problem:
        raise CLIError(f"generated spec does not exist or is not valid JSON: {previous_problem_path}")
    fidelity_report_path = _artifact_path(artifact_dir, source_summary.get("spec_fidelity_report"), "spec/fidelity-review.json")
    fidelity_report = _read_json_object(fidelity_report_path)
    repair_issue = _format_fidelity_repair_issue(source_summary=source_summary, fidelity_report=fidelity_report)

    exit_code, repaired_summary = _run_repaired_solve_from_fidelity(
        problem_id=problem_id,
        statement=statement,
        statement_path=statement_path,
        source_artifact_dir=artifact_dir,
        resolution_dir=resolution_dir,
        previous_problem=previous_problem,
        repair_issue=repair_issue,
        args=args,
    )

    if exit_code == 0:
        review_args = argparse.Namespace(**vars(args))
        review_args.mode = "agent"
        review = _agent_review_payload(resolution_dir, review_args)
        repaired_summary = apply_fidelity_review(artifact_dir=resolution_dir, review=review)

    repaired_fidelity_status = str(repaired_summary.get("spec_fidelity_status", "not_reviewed"))
    if repaired_fidelity_status in FIDELITY_ACCEPTED_STATUSES:
        impact_analysis = {"classification": "not_needed", "reason": "repaired artifact passed source-statement fidelity review"}
        resolution_status = "repaired_accepted"
    elif exit_code != 0:
        impact_analysis = {"classification": "unresolved", "reason": "repaired solve did not complete successfully"}
        resolution_status = "repair_failed"
    else:
        impact_analysis = analyze_fidelity_resolution_impact(
            source_artifact_dir=artifact_dir,
            repaired_artifact_dir=resolution_dir,
            source_summary=source_summary,
            repaired_summary=repaired_summary,
            tolerance_abs=args.impact_tolerance_abs,
            tolerance_rel=args.impact_tolerance_rel,
        )
        impact_classification = impact_analysis["classification"]
        if impact_classification == "harmless_equivalent":
            resolution_status = "residual_harmless_equivalent"
        elif impact_classification == "material":
            resolution_status = "residual_material"
        else:
            resolution_status = "residual_unresolved"

    result = {
        "problem_id": problem_id,
        "resolution_status": resolution_status,
        "source_artifact": str(artifact_dir),
        "resolution_artifact": str(resolution_dir),
        "source_fidelity_status": source_fidelity_status,
        "repaired_fidelity_status": repaired_fidelity_status,
        "repair_issue": repair_issue,
        "created_at": _utc_now(),
        "impact_analysis": impact_analysis,
    }
    report_path = resolution_dir / "fidelity-resolution.json"
    _write_summary(report_path, result)
    _apply_fidelity_resolution_result(source_summary_path, source_summary, result, artifact_dir, resolution_dir, report_path)
    return result


def _run_repaired_solve_from_fidelity(
    *,
    problem_id: str,
    statement: str,
    statement_path: Path,
    source_artifact_dir: Path,
    resolution_dir: Path,
    previous_problem: dict[str, Any],
    repair_issue: str,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    spec_dir = resolution_dir / "spec"
    submissions_dir = resolution_dir / "submissions"
    raw_dir = resolution_dir / "raw"
    reports_dir = resolution_dir / "reports"
    for directory in (spec_dir, submissions_dir, raw_dir, reports_dir, resolution_dir / "sessions"):
        directory.mkdir(parents=True, exist_ok=True)

    problem_path = spec_dir / "problem.json"
    spec_raw_path = raw_dir / "spec.txt"
    spec_status_path = spec_dir / "status.json"
    spec_fidelity_review_path = spec_dir / "fidelity-review.md"
    spec_fidelity_report_path = spec_dir / "fidelity-review.json"
    spec_result = generate_problem_spec(
        problem_id=problem_id,
        statement=statement,
        out_path=problem_path,
        raw_path=spec_raw_path,
        status_path=spec_status_path,
        artifact_dir=resolution_dir,
        args=args,
        initial_previous_problem=previous_problem,
        initial_previous_response=json.dumps(previous_problem, ensure_ascii=False, indent=2),
        initial_repair_error=repair_issue,
    )

    summary: dict[str, Any] = {
        "problem_id": problem_id,
        "statement_file": str(statement_path),
        "source_artifact": str(source_artifact_dir),
        "spec": _relative(problem_path, resolution_dir),
        "spec_raw": _relative(spec_raw_path, resolution_dir),
        "spec_status": _relative(spec_status_path, resolution_dir),
        "spec_generation_status": spec_result["spec_generation_status"],
        "spec_validation_status": spec_result["spec_validation_status"],
        "spec_generation_error": spec_result.get("generation_error", ""),
        "spec_validation_returncode": spec_result.get("validation_returncode"),
        "spec_attempt_count": spec_result.get("spec_attempt_count", 1),
        "spec_repair_status": spec_result.get("spec_repair_status", "not_needed"),
        **codex_run_metadata_summary_fields(spec_result.get("codex_run_metadata"), prefix="spec"),
        "spec_fidelity_status": "not_reviewed",
        "spec_fidelity_review": _relative(spec_fidelity_review_path, resolution_dir),
        "spec_fidelity_report": _relative(spec_fidelity_report_path, resolution_dir),
        "model_generation_status": "skipped",
        "verification_status": "skipped",
        "classification": "skipped",
        "fidelity_repair_issue": repair_issue,
    }

    if not _spec_is_ready(spec_result):
        summary["reason"] = "repaired spec validation failed; model generation skipped"
        _write_spec_fidelity_review(
            spec_fidelity_review_path,
            report_path=spec_fidelity_report_path,
            summary=summary,
            statement=statement,
            problem_path=problem_path,
        )
        _write_summary(resolution_dir / "summary.json", summary)
        return 1, summary

    inputs = GenerationInputs(
        problem_id=problem_id,
        record={"id": problem_id, "en_question": statement},
        problem_path=problem_path,
        problem=load_problem(problem_path),
    )
    submission_path = submissions_dir / f"{problem_id}.py"
    model_raw_path = raw_dir / f"{problem_id}.txt"
    report_path = reports_dir / f"{problem_id}.json"
    generation = generate_agent_submission(
        inputs=inputs,
        paths=build_agent_paths(
            problem_id=problem_id,
            artifact_root=resolution_dir,
            submission_path=submission_path,
            raw_path=model_raw_path,
            report_path=report_path,
        ),
        args=args,
    )
    verification = run_or_ci_verify(
        problem_path=problem_path,
        submission_path=submission_path,
        report_path=report_path,
        cwd=repo_root(),
    )
    summary.update(
        {
            "submission": _relative(submission_path, resolution_dir),
            "model_raw": _relative(model_raw_path, resolution_dir),
            "report": _relative(report_path, resolution_dir),
            "model_generation_status": generation["generation_status"],
            "model_generation_error": generation["generation_error"],
            "verification_status": verification.status,
            "classification": verification.classification,
            "verification_note": "passed generated spec" if verification.status == "PASS" else "failed generated spec",
            "verify_returncode": verification.returncode,
            "verify_stdout": verification.stdout,
            "verify_stderr": verification.stderr,
            **codex_run_metadata_summary_fields(generation.get("codex_run_metadata"), prefix="generation"),
        }
    )
    _write_spec_fidelity_review(
        spec_fidelity_review_path,
        report_path=spec_fidelity_report_path,
        summary=summary,
        statement=statement,
        problem_path=problem_path,
    )
    _write_summary(resolution_dir / "summary.json", summary)
    exit_code = 0 if generation["generation_status"] == "generated" and verification.returncode == 0 else 1
    return exit_code, summary


def analyze_fidelity_resolution_impact(
    *,
    source_artifact_dir: Path,
    repaired_artifact_dir: Path,
    source_summary: dict[str, Any],
    repaired_summary: dict[str, Any],
    tolerance_abs: float,
    tolerance_rel: float,
) -> dict[str, Any]:
    source_report = _read_json_object(_artifact_path(source_artifact_dir, source_summary.get("report"), "reports/report.json"))
    repaired_report = _read_json_object(_artifact_path(repaired_artifact_dir, repaired_summary.get("report"), "reports/report.json"))
    source_objective = _report_objective_value(source_report)
    repaired_objective = _report_objective_value(repaired_report)
    analysis: dict[str, Any] = {
        "source_verification_status": source_summary.get("verification_status"),
        "repaired_verification_status": repaired_summary.get("verification_status"),
        "source_classification": source_summary.get("classification"),
        "repaired_classification": repaired_summary.get("classification"),
        "source_objective": source_objective,
        "repaired_objective": repaired_objective,
        "tolerance_abs": tolerance_abs,
        "tolerance_rel": tolerance_rel,
    }
    if source_summary.get("verification_status") != "PASS" or repaired_summary.get("verification_status") != "PASS":
        analysis.update({"classification": "unresolved", "reason": "both artifacts must verify before impact can be compared"})
        return analysis
    if source_objective is None or repaired_objective is None:
        analysis.update({"classification": "unresolved", "reason": "missing original_solver_status objective value"})
        return analysis
    delta = abs(repaired_objective - source_objective)
    threshold = max(tolerance_abs, tolerance_rel * max(1.0, abs(source_objective), abs(repaired_objective)))
    analysis["objective_delta"] = delta
    analysis["objective_threshold"] = threshold
    if delta <= threshold:
        analysis.update({"classification": "harmless_equivalent", "reason": "verified objective value is unchanged within tolerance"})
    else:
        analysis.update({"classification": "material", "reason": "verified objective value changed after fidelity repair"})
    return analysis


def _report_objective_value(report: dict[str, Any]) -> float | None:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, dict) or check.get("name") != "original_solver_status":
            continue
        details = check.get("details")
        if not isinstance(details, dict):
            return None
        value = details.get("objective_value")
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _format_fidelity_repair_issue(*, source_summary: dict[str, Any], fidelity_report: dict[str, Any]) -> str:
    review = fidelity_report.get("review") if isinstance(fidelity_report.get("review"), dict) else {}
    issues = review.get("issues") if isinstance(review.get("issues"), list) else []
    lines = [
        "Source-statement fidelity review rejected the generated ProblemSpec.",
        f"Review status: {source_summary.get('spec_fidelity_status', 'unknown')}",
    ]
    note = review.get("note") or source_summary.get("spec_fidelity_review_note")
    if isinstance(note, str) and note.strip():
        lines.append(f"Reviewer note: {note.strip()}")
    if issues:
        lines.append("Issues:")
        for issue in issues:
            if isinstance(issue, dict):
                field = issue.get("field", "unknown")
                severity = issue.get("severity", "unknown")
                message = issue.get("message", "")
                lines.append(f"- {severity} {field}: {message}")
            else:
                lines.append(f"- {issue}")
    lines.append("Return a complete corrected OR-CI ProblemSpec JSON object. Preserve faithful fields and repair the mismatch.")
    return "\n".join(lines)


def summarize_fidelity_resolution(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    impacts: dict[str, int] = {}
    for result in results:
        status = str(result.get("resolution_status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        impact = result.get("impact_analysis", {}).get("classification", "unknown")
        impacts[str(impact)] = impacts.get(str(impact), 0) + 1
    return {"total": len(results), "resolution_statuses": statuses, "impact_classifications": impacts}


def write_fidelity_resolution_report(path: Path, args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    matrix = [
        "| Problem | Resolution | Source Fidelity | Repaired Fidelity | Impact | Repaired Artifact |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        impact = row.get("impact_analysis", {}).get("classification", "unknown")
        matrix.append(
            f"| {row['problem_id']} | `{row['resolution_status']}` | `{row.get('source_fidelity_status', '')}` | "
            f"`{row.get('repaired_fidelity_status', '')}` | `{impact}` | `{row.get('resolution_artifact', '')}` |"
        )
    path.write_text(
        f"""# Fidelity Resolution Report

## Scope

- Producer: `or_llm_agent resolve-fidelity-batch --mode {args.mode}`

## Summary

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Matrix

{chr(10).join(matrix)}

## Interpretation Notes

- `repaired_accepted` means the source artifact was repaired into a new solve artifact that passed fidelity review.
- `residual_harmless_equivalent` means fidelity review still did not accept the repaired artifact, but objective impact analysis found no result change.
- `residual_material` means the repaired artifact changed the verified objective value.
- `residual_unresolved` means the CLI could not determine impact from deterministic artifacts.
""",
        encoding="utf-8",
    )


def _apply_fidelity_resolution_result(
    source_summary_path: Path,
    source_summary: dict[str, Any],
    result: dict[str, Any],
    source_artifact_dir: Path,
    resolution_dir: Path,
    report_path: Path,
) -> None:
    impact = result.get("impact_analysis", {})
    source_summary.update(
        {
            "fidelity_resolution_status": result["resolution_status"],
            "fidelity_resolution_artifact": _relative(resolution_dir, source_artifact_dir),
            "fidelity_resolution_report": _relative(report_path, source_artifact_dir),
            "fidelity_resolution_repaired_fidelity_status": result.get("repaired_fidelity_status", ""),
            "fidelity_resolution_impact_classification": impact.get("classification", "unknown"),
        }
    )
    _write_summary(source_summary_path, source_summary)


def _artifact_fidelity_status(artifact_dir: Path) -> str:
    return str(_read_json_object(artifact_dir / "summary.json").get("spec_fidelity_status", "missing_summary"))


def _next_attempt_dir(root: Path) -> Path:
    for index in range(1, 1000):
        candidate = root / f"attempt-{index}"
        if not candidate.exists():
            return candidate
    raise CLIError(f"could not find available attempt directory under {root}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _generation_result(status: str, error: str, raw_path: Path, out_path: Path) -> dict[str, Any]:
    return {
        "generation_status": status,
        "generation_error": redact_text(error),
        "raw_response": raw_path,
        "submission": out_path,
        "generation_mode": "api",
        "agent_returncode": None,
    }


def _capability_is_supported(result: dict[str, Any]) -> bool:
    return result.get("status") == "supported" and result.get("capability_generation_status") == "classified"


def _capability_summary_fields(result: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "capability_status": result.get("status", "unknown"),
        "capability_generation_status": result.get("capability_generation_status", "unknown"),
        "problem_family": result.get("problem_family", "unknown"),
        "capability_supported_features": result.get("supported_features", []),
        "capability_unsupported_features": result.get("unsupported_features", []),
        "capability_missing_information": result.get("missing_information", []),
        "capability_recommended_next_action": result.get("recommended_next_action", ""),
        "capability_review_note": result.get("review_note", ""),
        "capability_agent_returncode": result.get("agent_returncode"),
        "capability_agent_timed_out": result.get("agent_timed_out"),
        "capability_agent_stderr": result.get("agent_stderr", ""),
        **codex_run_metadata_summary_fields(result.get("codex_run_metadata"), prefix="capability"),
    }
    if "confidence" in result:
        fields["capability_confidence"] = result["confidence"]
    return fields


def load_clarification_questions(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CLIError(f"clarification question artifact does not exist: {path}; run prepare-clarification first")
    return normalize_clarification_questions(_read_json_object(path), source_path=path)


def normalize_clarification_questions(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise CLIError(f"clarification question artifact is not a JSON object: {source_path}")
    problem_id = _required_text(payload, "problem_id", context=str(source_path))
    source_artifact = _required_text(payload, "source_artifact", context=str(source_path))
    blocking_status = _required_text(payload, "blocking_status", context=str(source_path))
    if blocking_status != "needs_human":
        raise CLIError(f"clarification question artifact blocking_status must be needs_human: {source_path}")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise CLIError(f"clarification question artifact must contain a non-empty questions list: {source_path}")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_question in enumerate(raw_questions, 1):
        if not isinstance(raw_question, dict):
            raise CLIError(f"clarification question #{index} must be an object: {source_path}")
        question = _normalize_clarification_question(raw_question, source_path=source_path)
        if question.id in seen_ids:
            raise CLIError(f"duplicate clarification question id {question.id!r}: {source_path}")
        seen_ids.add(question.id)
        questions.append(
            {
                "id": question.id,
                "issue_type": question.issue_type,
                "prompt": question.prompt,
                "source_evidence": question.source_evidence,
                "allowed_answer_type": question.allowed_answer_type,
                "options": list(question.options),
                "required": question.required,
            }
        )

    normalized = {
        "problem_id": problem_id,
        "source_artifact": source_artifact,
        "blocking_status": blocking_status,
        "questions": questions,
    }
    for key in (
        "raw_response",
        "agent_returncode",
        "agent_timed_out",
        "codex_events",
        "last_message",
        "agent_stderr",
        "codex_run_metadata",
        "codex_effective_model",
        "codex_effective_reasoning_effort",
        "codex_cli_version",
        "codex_usage",
    ):
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def _normalize_clarification_question(raw_question: dict[str, Any], *, source_path: Path) -> ClarificationQuestion:
    question_id = _required_text(raw_question, "id", context=str(source_path))
    issue_type = _required_text(raw_question, "issue_type", context=str(source_path))
    if issue_type not in CLARIFICATION_ISSUE_TYPES:
        raise CLIError(f"unsupported clarification issue_type {issue_type!r}: {source_path}")
    prompt = _required_text(raw_question, "prompt", context=str(source_path))
    source_evidence = _required_text(raw_question, "source_evidence", context=str(source_path))
    allowed_answer_type = _required_text(raw_question, "allowed_answer_type", context=str(source_path))
    if allowed_answer_type not in CLARIFICATION_ANSWER_TYPES:
        raise CLIError(f"unsupported clarification allowed_answer_type {allowed_answer_type!r}: {source_path}")
    options_value = raw_question.get("options", [])
    if not isinstance(options_value, list) or not all(isinstance(item, str) and item for item in options_value):
        raise CLIError(f"clarification question {question_id!r} options must be a list of strings: {source_path}")
    if allowed_answer_type in {"single_choice", "multi_choice"} and not options_value:
        raise CLIError(f"clarification question {question_id!r} requires non-empty options: {source_path}")
    required_value = raw_question.get("required", True)
    if not isinstance(required_value, bool):
        raise CLIError(f"clarification question {question_id!r} required must be boolean: {source_path}")
    return ClarificationQuestion(
        id=question_id,
        issue_type=issue_type,
        prompt=prompt,
        source_evidence=source_evidence,
        allowed_answer_type=allowed_answer_type,
        options=tuple(options_value),
        required=required_value,
    )


def load_clarification_answers(
    path: Path,
    *,
    questions: dict[str, Any],
    reviewer: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise CLIError(f"clarification answer artifact does not exist: {path}")
    payload = _read_json_object(path)
    if not payload:
        raise CLIError(f"clarification answer artifact is not a JSON object: {path}")
    problem_id = _required_text(payload, "problem_id", context=str(path))
    if problem_id != questions["problem_id"]:
        raise CLIError(
            f"clarification answer problem_id {problem_id!r} does not match questions problem_id "
            f"{questions['problem_id']!r}: {path}"
        )
    raw_status = payload.get("resolution_status", "answered")
    resolution_status = str(raw_status).strip() if isinstance(raw_status, str) else ""
    if resolution_status not in CLARIFICATION_RESOLUTION_STATUSES:
        raise CLIError(f"unsupported clarification resolution_status {resolution_status!r}: {path}")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list):
        raise CLIError(f"clarification answer artifact must contain an answers list: {path}")

    questions_by_id = {question["id"]: question for question in questions["questions"]}
    answers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_answer in enumerate(raw_answers, 1):
        if not isinstance(raw_answer, dict):
            raise CLIError(f"clarification answer #{index} must be an object: {path}")
        answer = _normalize_clarification_answer(
            raw_answer,
            questions_by_id=questions_by_id,
            reviewer=reviewer,
            source_path=path,
        )
        if answer.question_id in seen:
            raise CLIError(f"duplicate clarification answer for question {answer.question_id!r}: {path}")
        seen.add(answer.question_id)
        answers.append(
            {
                "question_id": answer.question_id,
                "answer": answer.answer,
                "reviewer": answer.reviewer,
                "rationale": answer.rationale,
                "source": answer.source,
            }
        )

    normalized = {
        "problem_id": problem_id,
        "answers": answers,
        "resolution_status": resolution_status,
    }
    missing = _required_unanswered_question_ids(questions, normalized)
    if missing and resolution_status == "answered":
        normalized["resolution_status"] = "partially_answered"
    return normalized


def _normalize_clarification_answer(
    raw_answer: dict[str, Any],
    *,
    questions_by_id: dict[str, dict[str, Any]],
    reviewer: str | None,
    source_path: Path,
) -> ClarificationAnswer:
    question_id = _required_text(raw_answer, "question_id", context=str(source_path))
    if question_id not in questions_by_id:
        raise CLIError(f"clarification answer references unknown question_id {question_id!r}: {source_path}")
    if "answer" not in raw_answer or _is_empty_answer(raw_answer.get("answer")):
        raise CLIError(f"clarification answer for {question_id!r} is empty: {source_path}")
    question = questions_by_id[question_id]
    answer_value = raw_answer.get("answer")
    answer_type = question["allowed_answer_type"]
    options = question.get("options", [])
    if answer_type == "single_choice" and answer_value not in options:
        raise CLIError(f"clarification answer for {question_id!r} must be one of {options}: {source_path}")
    if answer_type == "multi_choice":
        if not isinstance(answer_value, list) or not answer_value or not all(item in options for item in answer_value):
            raise CLIError(f"clarification answer for {question_id!r} must be a non-empty subset of {options}: {source_path}")
    if answer_type == "number" and (isinstance(answer_value, bool) or not isinstance(answer_value, (int, float))):
        raise CLIError(f"clarification answer for {question_id!r} must be numeric: {source_path}")
    if answer_type == "boolean" and not isinstance(answer_value, bool):
        raise CLIError(f"clarification answer for {question_id!r} must be boolean: {source_path}")

    reviewer_value = raw_answer.get("reviewer") or reviewer
    if not isinstance(reviewer_value, str) or not reviewer_value.strip():
        raise CLIError(f"clarification answer for {question_id!r} requires a reviewer: {source_path}")
    rationale = raw_answer.get("rationale", "")
    if not isinstance(rationale, str):
        raise CLIError(f"clarification answer rationale for {question_id!r} must be a string: {source_path}")
    source = raw_answer.get("source", "manual_review")
    if not isinstance(source, str) or not source.strip():
        raise CLIError(f"clarification answer source for {question_id!r} must be a non-empty string: {source_path}")
    return ClarificationAnswer(
        question_id=question_id,
        answer=answer_value,
        reviewer=reviewer_value.strip(),
        rationale=rationale.strip(),
        source=source.strip(),
    )


def _required_text(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CLIError(f"{key} must be a non-empty string in {context}")
    return value.strip()


def _is_empty_answer(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not value
    return False


def _required_unanswered_question_ids(questions: dict[str, Any], answers: dict[str, Any]) -> list[str]:
    answered = {
        str(answer.get("question_id"))
        for answer in answers.get("answers", [])
        if isinstance(answer, dict) and not _is_empty_answer(answer.get("answer"))
    }
    return [
        str(question["id"])
        for question in questions.get("questions", [])
        if isinstance(question, dict) and question.get("required", True) and question.get("id") not in answered
    ]


def _clarification_gate_status(questions: dict[str, Any], answers: dict[str, Any]) -> str:
    status = answers.get("resolution_status")
    if status == "rejected":
        return "blocked_rejected"
    if status == "unresolved":
        return "blocked_unresolved"
    if _required_unanswered_question_ids(questions, answers):
        return "blocked_unanswered_required"
    return "passed"


def _clarification_source(answers: dict[str, Any]) -> str:
    sources = sorted(
        {
            str(answer.get("source")).strip()
            for answer in answers.get("answers", [])
            if isinstance(answer, dict) and str(answer.get("source", "")).strip()
        }
    )
    return ", ".join(sources) if sources else "unknown"


def _canonical_clarification_questions_path(artifact_dir: Path) -> Path:
    return artifact_dir / "clarification" / "questions.json"


def _canonical_clarification_answers_path(artifact_dir: Path) -> Path:
    return artifact_dir / "clarification" / "answers.json"


def _write_clarification_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_clarification_agent_failure_status(
    *,
    artifact_dir: Path,
    status: str,
    raw_path: Path,
    agent_result: ClarificationAgentResult,
) -> Path:
    status_path = artifact_dir / "clarification" / "status.json"
    _write_clarification_artifact(
        status_path,
        {
            "status": status,
            "agent_returncode": agent_result.returncode,
            "agent_timed_out": agent_result.timed_out,
            "raw_response": str(raw_path),
            "codex_events": str(agent_result.events_path),
            "last_message": str(agent_result.last_message_path),
            "agent_stderr": redact_text(agent_result.stderr),
            **codex_run_metadata_summary_fields(agent_result.run_metadata),
        },
    )
    return status_path


def _clarification_context(
    *,
    source_artifact_dir: Path,
    resolution_dir: Path,
    questions: dict[str, Any],
    answers: dict[str, Any],
    question_path: Path,
    answer_path: Path,
    gate_status: str,
) -> dict[str, Any]:
    return {
        "clarified": True,
        "clarified_from": str(source_artifact_dir),
        "clarification_status": answers["resolution_status"],
        "clarification_gate_status": gate_status,
        "clarification_question_count": len(questions["questions"]),
        "clarification_answer_count": len(answers["answers"]),
        "clarification_source": _clarification_source(answers),
        "question_artifact": _relative(question_path, resolution_dir),
        "answer_artifact": _relative(answer_path, resolution_dir),
        "questions": questions["questions"],
        "answers": answers["answers"],
    }


def _statement_with_clarification(statement: str, clarification_context: dict[str, Any]) -> str:
    return (
        f"{statement.strip()}\n\n"
        "Approved clarification context for this run:\n"
        f"{json.dumps(clarification_context, ensure_ascii=False, indent=2)}"
    )


def _mark_problem_as_clarified(problem: dict[str, Any], clarification_context: dict[str, Any]) -> None:
    source_context = problem.get("source_context") if isinstance(problem.get("source_context"), dict) else {}
    source_context.update(
        {
            "clarified": True,
            "clarified_from": clarification_context.get("clarified_from", ""),
            "clarification_status": clarification_context.get("clarification_status", ""),
            "clarification_gate_status": clarification_context.get("clarification_gate_status", ""),
            "clarification_question_count": clarification_context.get("clarification_question_count", 0),
            "clarification_answer_count": clarification_context.get("clarification_answer_count", 0),
        }
    )
    problem["source_context"] = source_context


def build_pilot_row(
    *,
    artifact_dir: Path,
    problem_id: str,
    model: str,
    generation: dict[str, Any],
    verification: VerificationResult,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "model": model,
        "generation_mode": generation.get("generation_mode", "api"),
        "generation_status": generation["generation_status"],
        "generation_error": generation["generation_error"],
        "submission": _relative(generation["submission"], artifact_dir),
        "raw_response": _relative(generation["raw_response"], artifact_dir),
        "report": _relative(report_path, artifact_dir),
        "agent_returncode": generation.get("agent_returncode"),
        "agent_timed_out": generation.get("agent_timed_out"),
        "agent_manifest": _relative(generation["agent_manifest"], artifact_dir)
        if isinstance(generation.get("agent_manifest"), Path)
        else "",
        "verify_returncode": verification.returncode,
        "verify_stdout": verification.stdout,
        "verify_stderr": verification.stderr,
        "classification": verification.classification,
        "status": verification.status,
        "failure_check": verification.failure_check,
        "checks": verification.checks,
        **codex_run_metadata_summary_fields(generation.get("codex_run_metadata")),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifications: dict[str, int] = {}
    generation_statuses: dict[str, int] = {}
    generation_modes: dict[str, int] = {}
    for row in rows:
        classifications[row["classification"]] = classifications.get(row["classification"], 0) + 1
        generation_statuses[row["generation_status"]] = generation_statuses.get(row["generation_status"], 0) + 1
        generation_modes[row.get("generation_mode", "api")] = generation_modes.get(row.get("generation_mode", "api"), 0) + 1
    return {
        "total": len(rows),
        "classifications": classifications,
        "generation_statuses": generation_statuses,
        "generation_modes": generation_modes,
    }


def _solve_exclusion_reason(row: dict[str, Any], *, batch_error: str = "") -> str:
    if batch_error:
        return "batch_error"
    capability_status = str(row.get("capability_status", ""))
    if capability_status in {"needs_human", "unsupported"}:
        return f"capability_{capability_status}"
    if row.get("classification") == "blocked_capability":
        return "capability_blocked"
    if row.get("classification") == "missing_summary":
        return "missing_summary"
    spec_status = str(row.get("spec_validation_status", ""))
    if spec_status and spec_status not in {"passed", "missing_summary"}:
        return f"spec_validation_{spec_status}"
    model_status = str(row.get("model_generation_status", ""))
    if model_status and model_status != "generated":
        return f"model_generation_{model_status}"
    return "not_generated"


def _apply_solve_terminal_accounting(row: dict[str, Any], *, batch_error: str = "") -> None:
    if row.get("model_generation_status") == "generated":
        row["terminal_state"] = "generated"
        row["accounting_state"] = "generated"
        row["exclusion_reason"] = ""
        row["blocking_reason"] = ""
        return

    reason = _solve_exclusion_reason(row, batch_error=batch_error)
    row["terminal_state"] = "excluded_with_reason"
    row["accounting_state"] = "excluded_with_reason"
    row["exclusion_reason"] = reason
    row["blocking_reason"] = reason


def summarize_solve_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifications: dict[str, int] = {}
    capability_statuses: dict[str, int] = {}
    spec_statuses: dict[str, int] = {}
    model_statuses: dict[str, int] = {}
    terminal_states: dict[str, int] = {}
    accounting_states: dict[str, int] = {}
    exclusion_reasons: dict[str, int] = {}
    fidelity_statuses: dict[str, int] = {}
    fidelity_gate_statuses: dict[str, int] = {}
    fidelity_resolution_statuses: dict[str, int] = {}
    fidelity_resolution_impacts: dict[str, int] = {}
    clarification_statuses: dict[str, int] = {}
    clarification_gate_statuses: dict[str, int] = {}
    exit_codes: dict[str, int] = {}
    clarification_question_count = 0
    clarification_answer_count = 0
    for row in rows:
        classifications[row["classification"]] = classifications.get(row["classification"], 0) + 1
        capability_status = row.get("capability_status", "unknown")
        capability_statuses[capability_status] = capability_statuses.get(capability_status, 0) + 1
        spec_statuses[row["spec_validation_status"]] = spec_statuses.get(row["spec_validation_status"], 0) + 1
        model_statuses[row["model_generation_status"]] = model_statuses.get(row["model_generation_status"], 0) + 1
        terminal_state = str(row.get("terminal_state", "missing_terminal_state"))
        terminal_states[terminal_state] = terminal_states.get(terminal_state, 0) + 1
        accounting_state = str(row.get("accounting_state", terminal_state))
        accounting_states[accounting_state] = accounting_states.get(accounting_state, 0) + 1
        if terminal_state == "excluded_with_reason":
            reason = str(row.get("exclusion_reason") or row.get("blocking_reason") or "missing_reason")
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        fidelity_status = row.get("spec_fidelity_status", "unknown")
        fidelity_statuses[fidelity_status] = fidelity_statuses.get(fidelity_status, 0) + 1
        gate = row.get("spec_fidelity_gate_status", "unknown")
        fidelity_gate_statuses[gate] = fidelity_gate_statuses.get(gate, 0) + 1
        if row.get("fidelity_resolution_status"):
            status = str(row["fidelity_resolution_status"])
            fidelity_resolution_statuses[status] = fidelity_resolution_statuses.get(status, 0) + 1
        if row.get("fidelity_resolution_impact_classification"):
            impact = str(row["fidelity_resolution_impact_classification"])
            fidelity_resolution_impacts[impact] = fidelity_resolution_impacts.get(impact, 0) + 1
        if row.get("clarification_status"):
            status = str(row["clarification_status"])
            clarification_statuses[status] = clarification_statuses.get(status, 0) + 1
        if row.get("clarification_gate_status"):
            gate = str(row["clarification_gate_status"])
            clarification_gate_statuses[gate] = clarification_gate_statuses.get(gate, 0) + 1
        clarification_question_count += int(row.get("clarification_question_count") or 0)
        clarification_answer_count += int(row.get("clarification_answer_count") or 0)
        exit_key = str(row["exit_code"])
        exit_codes[exit_key] = exit_codes.get(exit_key, 0) + 1
    return {
        "total": len(rows),
        "succeeded": sum(1 for row in rows if row["exit_code"] == 0),
        "failed": sum(1 for row in rows if row["exit_code"] != 0),
        "classifications": classifications,
        "capability_statuses": capability_statuses,
        "spec_validation_statuses": spec_statuses,
        "model_generation_statuses": model_statuses,
        "terminal_states": terminal_states,
        "accounting_states": accounting_states,
        "accounted": sum(1 for row in rows if row.get("terminal_state") in {"generated", "excluded_with_reason"}),
        "unaccounted": sum(1 for row in rows if row.get("terminal_state") not in {"generated", "excluded_with_reason"}),
        "exclusion_reasons": exclusion_reasons,
        "spec_fidelity_statuses": fidelity_statuses,
        "spec_fidelity_gate_statuses": fidelity_gate_statuses,
        "fidelity_resolution_statuses": fidelity_resolution_statuses,
        "fidelity_resolution_impacts": fidelity_resolution_impacts,
        "clarification_statuses": clarification_statuses,
        "clarification_gate_statuses": clarification_gate_statuses,
        "clarification_question_count": clarification_question_count,
        "clarification_answer_count": clarification_answer_count,
        "exit_codes": exit_codes,
    }


def summarize_source_fidelity_rubric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    gate_statuses: dict[str, int] = {}
    rubric_versions: dict[str, int] = {}
    dimension_failures: dict[str, int] = {}
    dimension_warnings: dict[str, int] = {}
    excluded: list[str] = []
    reviewed_count = 0
    rubric_complete_count = 0
    legacy_flat_count = 0
    provisional_count = 0
    or_ci_verified_count = 0
    for row in rows:
        problem_id = str(row.get("problem_id", "unknown"))
        status = str(row.get("spec_fidelity_status", "unknown"))
        gate = str(row.get("spec_fidelity_gate_status", "unknown"))
        version = str(row.get("spec_fidelity_rubric_version", "missing"))
        statuses[status] = statuses.get(status, 0) + 1
        gate_statuses[gate] = gate_statuses.get(gate, 0) + 1
        rubric_versions[version] = rubric_versions.get(version, 0) + 1
        if row.get("verification_status") == "PASS":
            or_ci_verified_count += 1
        if status not in {"not_reviewed", "unknown", "missing_summary"}:
            reviewed_count += 1
        if row.get("spec_fidelity_rubric_complete") is True:
            rubric_complete_count += 1
        if version == LEGACY_SOURCE_FIDELITY_RUBRIC_VERSION:
            legacy_flat_count += 1
        if row.get("spec_fidelity_provisional") is True:
            provisional_count += 1
        for dimension in _string_list(row.get("spec_fidelity_failed_dimensions")):
            dimension_failures[dimension] = dimension_failures.get(dimension, 0) + 1
        for dimension in _string_list(row.get("spec_fidelity_warned_dimensions")):
            dimension_warnings[dimension] = dimension_warnings.get(dimension, 0) + 1
        if status not in FIDELITY_ACCEPTED_STATUSES or row.get("spec_fidelity_provisional") is True or row.get("verification_status") != "PASS":
            excluded.append(problem_id)

    accepted_count = sum(statuses.get(status, 0) for status in FIDELITY_ACCEPTED_STATUSES)
    rejected_count = sum(statuses.get(status, 0) for status in FIDELITY_REJECTED_STATUSES)
    return {
        "total": len(rows),
        "or_ci_verified_count": or_ci_verified_count,
        "reviewed_count": reviewed_count,
        "rubric_complete_count": rubric_complete_count,
        "legacy_flat_count": legacy_flat_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "not_reviewed_count": statuses.get("not_reviewed", 0),
        "provisional_count": provisional_count,
        "spec_fidelity_statuses": statuses,
        "spec_fidelity_gate_statuses": gate_statuses,
        "rubric_versions": rubric_versions,
        "dimension_failures": dimension_failures,
        "dimension_warnings": dimension_warnings,
        "excluded_from_headline_claim": excluded,
        "strongest_defensible_claim": _source_fidelity_claim_sentence(
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            provisional_count=provisional_count,
        ),
    }


def _source_fidelity_claim_sentence(*, accepted_count: int, rejected_count: int, provisional_count: int) -> str:
    return (
        "Among OR-CI-verified generated models, source-fidelity review accepted "
        f"{accepted_count} case(s), rejected {rejected_count} case(s), and marked "
        f"{provisional_count} case(s) as provisional; use this as a layered "
        "acceptance/false-accept exposure claim, not a raw solve-count claim."
    )


def _source_fidelity_rubric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "problem_id": row.get("problem_id", "unknown"),
            "verification_status": row.get("verification_status", "unknown"),
            "classification": row.get("classification", "unknown"),
            "spec_fidelity_status": row.get("spec_fidelity_status", "unknown"),
            "spec_fidelity_gate_status": row.get("spec_fidelity_gate_status", "unknown"),
            "spec_fidelity_rubric_version": row.get("spec_fidelity_rubric_version", "missing"),
            "spec_fidelity_rubric_complete": row.get("spec_fidelity_rubric_complete", False),
            "spec_fidelity_failed_dimensions": _string_list(row.get("spec_fidelity_failed_dimensions")),
            "spec_fidelity_warned_dimensions": _string_list(row.get("spec_fidelity_warned_dimensions")),
            "spec_fidelity_provisional": bool(row.get("spec_fidelity_provisional")),
            "spec_fidelity_materiality": row.get("spec_fidelity_materiality", ""),
            "artifact_dir": row.get("artifact_dir", ""),
        }
        for row in rows
    ]


def summarize_clarification_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    gates: dict[str, int] = {}
    classifications: dict[str, int] = {}
    or_ci_statuses: dict[str, int] = {}
    fidelity_statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("clarification_status", "questions_prepared"))
        gate = str(row.get("clarification_gate_status", "awaiting_answers"))
        classification = str(row.get("classification", "not_run"))
        or_ci_status = str(row.get("verification_status", "not_run"))
        fidelity_status = str(row.get("spec_fidelity_status", "not_reviewed"))
        statuses[status] = statuses.get(status, 0) + 1
        gates[gate] = gates.get(gate, 0) + 1
        classifications[classification] = classifications.get(classification, 0) + 1
        or_ci_statuses[or_ci_status] = or_ci_statuses.get(or_ci_status, 0) + 1
        fidelity_statuses[fidelity_status] = fidelity_statuses.get(fidelity_status, 0) + 1
    return {
        "attempted_needs_human_count": len(rows),
        "generated_question_count": sum(int(row.get("clarification_question_count") or 0) for row in rows),
        "answered_question_count": sum(int(row.get("clarification_answer_count") or 0) for row in rows),
        "clarified_supported_case_count": sum(
            1
            for row in rows
            if row.get("clarification_gate_status") == "passed"
            and row.get("verification_status") == "PASS"
            and row.get("classification") == "SUCCESS"
        ),
        "unresolved_case_count": sum(
            1
            for row in rows
            if row.get("clarification_gate_status") not in {"passed", "awaiting_answers"}
            or row.get("classification") == "blocked_clarification"
        ),
        "or_ci_statuses": or_ci_statuses,
        "fidelity_statuses": fidelity_statuses,
        "clarification_statuses": statuses,
        "clarification_gate_statuses": gates,
        "classifications": classifications,
    }


def write_markdown_report(path: Path, args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    matrix = [
        "| Problem | Mode | Generation | Agent RC | OR-CI Classification | Failure Check | Submission | Report |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        matrix.append(
            f"| {row['problem_id']} | `{row.get('generation_mode', 'api')}` | `{row['generation_status']}` | "
            f"`{row.get('agent_returncode') if row.get('agent_returncode') is not None else '-'}` | `{row['classification']}` | "
            f"`{row['failure_check'] or '-'}` | `{row['submission']}` | `{row['report']}` |"
        )

    any_generated = any(row["generation_status"] in {"generated", "reused"} for row in rows)
    if any_generated:
        current_result = "COMPLETED: at least one generated or reused submission reached OR-CI verification."
    elif summary["generation_statuses"].get("failed") == summary["total"]:
        current_result = (
            "BLOCKED: provider generation failed for every requested problem. "
            "See `summary.json` and `raw/*.txt` for sanitized provider errors."
        )
    else:
        current_result = (
            "BLOCKED: generation did not produce a valid submission for any requested problem. "
            "See `summary.json` and the raw or agent status artifacts for details."
        )

    path.write_text(
        f"""# OR-CI Integration Pilot Report

## Scope

- Producer: `or_llm_agent`
- Verifier: standalone OR-CI CLI through the editable `or-ci` dependency
- Model: `{args.model}`
- Problems: {", ".join(args.ids)}

## Summary

{current_result}

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Matrix

{chr(10).join(matrix)}

## Interpretation Notes

- `SUCCESS` means the generated submission passed the configured OR-CI invariants, not full model correctness.
- Semantic failures distinguish which verifier invariant failed first.
- Generation failures are reported separately from OR-CI classifications.
""",
        encoding="utf-8",
    )


def write_solve_batch_report(path: Path, args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    matrix = [
        "| Problem | Exit | Capability | Spec Validation | Attempts | Repair | Model Generation | Verification | Classification | Fidelity | Gate | Resolution | Impact | Artifact |",
        "|---|---:|---|---|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        matrix.append(
            f"| {row['problem_id']} | `{row['exit_code']}` | `{row.get('capability_status', 'unknown')}` | "
            f"`{row['spec_validation_status']}` | "
            f"`{row['spec_attempt_count']}` | `{row['spec_repair_status']}` | "
            f"`{row['model_generation_status']}` | `{row['verification_status']}` | "
            f"`{row['classification']}` | `{row.get('spec_fidelity_status', 'unknown')}` | "
            f"`{row.get('spec_fidelity_gate_status', 'unknown')}` | "
            f"`{row.get('fidelity_resolution_status', '-')}` | "
            f"`{row.get('fidelity_resolution_impact_classification', '-')}` | "
            f"`{row['artifact_dir']}` |"
        )

    capstone_section = ""
    if (path.parent / "fidelity-rubric-summary.json").is_file() or (path.parent / "fidelity-rubric-report.md").is_file():
        capstone_section = """

## Source-Fidelity Capstone

- Machine summary: `fidelity-rubric-summary.json`
- Report output: `fidelity-rubric-report.md`
"""

    path.write_text(
        f"""# Statement-Only Solve Batch Report

## Scope

- Producer: `or_llm_agent solve-batch --mode {args.mode}`
- Problems: {", ".join(args.ids)}
- Source: `--statements-dir` if supplied, otherwise `--dataset`

## Summary

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Matrix

{chr(10).join(matrix)}
{capstone_section}

## Interpretation Notes

- `classification=SUCCESS` means the generated submission passed OR-CI checks against the generated spec.
- `capability_status=supported` means statement routing allowed ProblemSpec generation; `needs_human` and `unsupported` block generation before OR-CI verification.
- `spec_fidelity_gate_status=manual_review_required` means source-statement fidelity has not been certified.
- `spec_fidelity_gate_status=accepted` means a reviewer accepted source-statement fidelity for this run artifact.
- `spec_fidelity_gate_status=rejected` means a reviewer rejected source-statement fidelity for this run artifact.
- `spec_fidelity_gate_status=llm_accepted` means the nested Codex reviewer accepted source-statement fidelity; keep it separate from human certification.
- `spec_fidelity_gate_status=llm_rejected` means the nested Codex reviewer rejected or could not certify source-statement fidelity.
- `fidelity_resolution_status=repaired_accepted` means the rejected source artifact was repaired into a new solve artifact that passed review.
- `fidelity_resolution_impact_classification=harmless_equivalent` means a residual mismatch remained but deterministic objective comparison found no result change.
- Inspect each case's `spec/fidelity-review.md` and `spec/fidelity-review.json` before treating generated specs as benchmark metadata.
""",
        encoding="utf-8",
    )


def write_fidelity_rubric_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    matrix = [
        "| Problem | OR-CI | Fidelity | Rubric | Failed Dimensions | Warned Dimensions | Provisional | Materiality | Artifact |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        failed = ", ".join(_string_list(row.get("spec_fidelity_failed_dimensions"))) or "-"
        warned = ", ".join(_string_list(row.get("spec_fidelity_warned_dimensions"))) or "-"
        matrix.append(
            f"| {row.get('problem_id', 'unknown')} | `{row.get('verification_status', 'unknown')}` | "
            f"`{row.get('spec_fidelity_status', 'unknown')}` | "
            f"`{row.get('spec_fidelity_rubric_version', 'missing')}` | {failed} | {warned} | "
            f"`{row.get('spec_fidelity_provisional', False)}` | "
            f"`{row.get('spec_fidelity_materiality', '') or '-'}` | `{row.get('artifact_dir', '')}` |"
        )

    boundary_lines = _fidelity_boundary_taxonomy_lines(rows)
    excluded = _string_list(summary.get("excluded_from_headline_claim"))
    excluded_line = ", ".join(excluded) if excluded else "None."
    path.write_text(
        f"""# Source-Fidelity Rubric Capstone

## Goal

Review generated OR-CI `ProblemSpec` artifacts against their source problem statements so report claims separate generated-spec/code verification from source-statement fidelity.

## Claim Boundary

OR-CI `PASS` means the generated submission passed checks against the generated spec. It does not by itself certify that the generated spec preserved the original statement.

## Aggregate Metrics

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Case Matrix

{chr(10).join(matrix)}

## Boundary Taxonomy

{chr(10).join(boundary_lines)}

## Capstone Conclusion

{summary.get('strongest_defensible_claim', '')}

- Cases excluded from headline source-fidelity claims: {excluded_line}
- Weakest evidence categories are dimensions with nonzero warnings or failures.
- Recommended next action: inspect rejected, provisional, and legacy-flat cases before using this batch in a paper-facing claim.
""",
        encoding="utf-8",
    )


def _fidelity_boundary_taxonomy_lines(rows: list[dict[str, Any]]) -> list[str]:
    categories = {
        "source/data invalid": 0,
        "unsupported modeling scope": 0,
        "source mismatch": 0,
        "verifier weakness": 0,
        "provisional clarification": 0,
    }
    for row in rows:
        failed = set(_string_list(row.get("spec_fidelity_failed_dimensions")))
        warned = set(_string_list(row.get("spec_fidelity_warned_dimensions")))
        dimensions = failed | warned
        if "data_completeness" in dimensions:
            categories["source/data invalid"] += 1
        if "source_suitability" in dimensions:
            categories["unsupported modeling scope"] += 1
        if dimensions & {"sets_and_indices", "numeric_parameters", "action_space", "objective", "units_and_scaling", "constraint_families"}:
            categories["source mismatch"] += 1
        if "metamorphic_coverage" in dimensions:
            categories["verifier weakness"] += 1
        if row.get("spec_fidelity_provisional") is True or "clarification_dependency" in dimensions:
            categories["provisional clarification"] += 1
    return [f"- {name}: {count}" for name, count in categories.items()]


def write_clarification_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    matrix = [
        "| Problem | Baseline Block | Questions | Answers | Answer Source | Gate | Clarified Status | OR-CI | Fidelity | Artifact |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        matrix.append(
            f"| {row['problem_id']} | `{row.get('baseline_block_reason', '')}` | "
            f"`{row.get('clarification_question_count', 0)}` | `{row.get('clarification_answer_count', 0)}` | "
            f"`{row.get('clarification_source', '')}` | `{row.get('clarification_gate_status', '')}` | "
            f"`{row.get('classification', 'not_run')}` | "
            f"`{row.get('verification_status', 'not_run')}` | `{row.get('spec_fidelity_status', 'not_reviewed')}` | "
            f"`{row.get('resolution_artifact') or row.get('source_artifact', '')}` |"
        )
    unresolved = [
        row["problem_id"]
        for row in rows
        if row.get("clarification_gate_status") not in {"passed", "awaiting_answers"}
        or row.get("classification") in {"blocked_clarification", "skipped"}
    ]
    moved = [
        row["problem_id"]
        for row in rows
        if row.get("clarification_gate_status") == "passed"
        and row.get("verification_status") == "PASS"
        and row.get("classification") == "SUCCESS"
    ]
    detail_sections = [_clarification_report_detail(row) for row in rows]
    path.write_text(
        f"""# Clarification Report

## Summary

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Matrix

{chr(10).join(matrix)}

## Unresolved Cases

{", ".join(unresolved) if unresolved else "None."}

## Moved From `needs_human` To Supported

{", ".join(moved) if moved else "None."}

## Case Details

{chr(10).join(detail_sections) if detail_sections else "None."}

## Interpretation Notes

- Baseline block reason comes from the original capability gate.
- Clarified runs are written to separate resolution artifacts; source blocked artifacts are not overwritten.
- OR-CI `PASS` still verifies the generated clarified ProblemSpec, not the source statement by itself.
- Fidelity must be reviewed against the original statement plus approved clarification answers.
""",
        encoding="utf-8",
    )


def _clarification_report_detail(row: dict[str, Any]) -> str:
    question_lines = [
        f"- `{item.get('id', '')}` `{item.get('issue_type', '')}`: {item.get('prompt', '')} "
        f"(evidence: {item.get('source_evidence', '')})"
        for item in row.get("generated_questions", [])
        if isinstance(item, dict)
    ] or ["- None recorded."]
    answer_lines = [
        f"- `{item.get('question_id', '')}` reviewer=`{item.get('reviewer', '')}` "
        f"source=`{item.get('source', '')}` rationale={item.get('rationale', '')}"
        for item in row.get("answer_provenance", [])
        if isinstance(item, dict)
    ] or ["- None recorded."]
    return f"""### {row.get('problem_id', 'unknown')}

- Baseline block reason: {row.get('baseline_block_reason', '')}
- Clarified rerun status: `{row.get('classification', 'not_run')}`
- OR-CI verification status: `{row.get('verification_status', 'not_run')}`
- Fidelity status: `{row.get('spec_fidelity_status', 'not_reviewed')}`

Generated question set:

{chr(10).join(question_lines)}

Answer provenance:

{chr(10).join(answer_lines)}
"""


def _batch_statement_file(args: argparse.Namespace, artifact_dir: Path, problem_id: str) -> Path:
    if args.statements_dir:
        candidate = args.statements_dir / f"{problem_id}.txt"
        if candidate.is_file():
            return candidate

    try:
        record = load_bwor_run_record(args.dataset, problem_id)
    except (KeyError, ValueError) as exc:
        raise CLIError(str(exc)) from exc
    statement = record.get("en_question") or record.get("cn_question")
    if not isinstance(statement, str) or not statement.strip():
        raise CLIError(f"no statement text found for {problem_id} in {args.dataset}")
    statement_path = artifact_dir / "statements" / f"{problem_id}.txt"
    statement_path.parent.mkdir(parents=True, exist_ok=True)
    statement_path.write_text(statement.strip() + "\n", encoding="utf-8")
    return statement_path


def _solve_batch_row(
    *,
    problem_id: str,
    exit_code: int,
    artifact_dir: Path,
    case_dir: Path,
    statement_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "problem_id": problem_id,
        "exit_code": exit_code,
        "artifact_dir": _relative(case_dir, artifact_dir),
        "statement_file": _relative(statement_path, artifact_dir),
        "capability_status": "missing_summary",
        "capability_generation_status": "missing_summary",
        "problem_family": "missing_summary",
        "spec_validation_status": "missing_summary",
        "spec_attempt_count": 0,
        "spec_repair_status": "missing_summary",
        "model_generation_status": "missing_summary",
        "verification_status": "missing_summary",
        "classification": "missing_summary",
        "terminal_state": "excluded_with_reason",
        "accounting_state": "excluded_with_reason",
        "exclusion_reason": "missing_summary",
        "blocking_reason": "missing_summary",
    }
    summary_path = case_dir / "summary.json"
    if not summary_path.is_file():
        return row
    case_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "capability_status",
        "capability_generation_status",
        "problem_family",
        "capability_supported_features",
        "capability_unsupported_features",
        "capability_missing_information",
        "capability_recommended_next_action",
        "capability_review_note",
        "capability_confidence",
        "capability_codex_effective_model",
        "capability_codex_effective_reasoning_effort",
        "capability_codex_cli_version",
        "capability_codex_usage",
        "capability_codex_run_metadata",
        "capability",
        "capability_raw",
        "spec_validation_status",
        "spec_attempt_count",
        "spec_repair_status",
        "spec_codex_effective_model",
        "spec_codex_effective_reasoning_effort",
        "spec_codex_cli_version",
        "spec_codex_usage",
        "spec_codex_run_metadata",
        "spec_fidelity_status",
        "spec_fidelity_gate_status",
        "spec_fidelity_reviewed_at",
        "spec_fidelity_reviewed_by",
        "spec_fidelity_codex_effective_model",
        "spec_fidelity_codex_effective_reasoning_effort",
        "spec_fidelity_codex_cli_version",
        "spec_fidelity_codex_usage",
        "spec_fidelity_codex_run_metadata",
        "model_generation_status",
        "generation_codex_effective_model",
        "generation_codex_effective_reasoning_effort",
        "generation_codex_cli_version",
        "generation_codex_usage",
        "generation_codex_run_metadata",
        "verification_status",
        "classification",
        "spec",
        "spec_fidelity_review",
        "spec_fidelity_report",
        "spec_fidelity_review_mode",
        "spec_fidelity_confidence",
        "spec_fidelity_issue_count",
        "spec_fidelity_rubric_version",
        "spec_fidelity_rubric_complete",
        "spec_fidelity_rubric_error",
        "spec_fidelity_failed_dimensions",
        "spec_fidelity_warned_dimensions",
        "spec_fidelity_blocking_dimension_count",
        "spec_fidelity_warning_dimension_count",
        "spec_fidelity_provisional",
        "spec_fidelity_materiality",
        "fidelity_resolution_status",
        "fidelity_resolution_artifact",
        "fidelity_resolution_report",
        "fidelity_resolution_repaired_fidelity_status",
        "fidelity_resolution_impact_classification",
        "clarified_from",
        "clarification_status",
        "clarification_question_count",
        "clarification_answer_count",
        "clarification_source",
        "clarification_gate_status",
        "clarification_questions",
        "clarification_answers",
        "source_anchor_contract",
        "proof_anchor_ledger",
        "proof_anchor_ledger_schema_status",
        "proof_anchor_ledger_schema_validation",
        "submission",
        "report",
    ):
        if key in case_summary:
            row[key] = case_summary[key]
    _apply_solve_terminal_accounting(row)
    return row


def apply_fidelity_review(*, artifact_dir: Path, review: dict[str, Any]) -> dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    if not summary_path.is_file():
        raise CLIError(f"solve summary does not exist: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise CLIError(f"solve summary must be a JSON object: {summary_path}")
    review = _normalize_fidelity_review(review, summary=summary)
    if review["status"] in FIDELITY_ACCEPTED_STATUSES:
        _ensure_acceptance_is_allowed(summary, artifact_dir)

    review_path = _artifact_path(artifact_dir, summary.get("spec_fidelity_review"), "spec/fidelity-review.md")
    report_path = _artifact_path(artifact_dir, summary.get("spec_fidelity_report"), "spec/fidelity-review.json")
    report = _read_json_object(report_path)
    report.update(
        {
            "problem_id": summary.get("problem_id"),
            "fidelity_status": review["status"],
            "gate_status": review["status"],
            "review": review,
            "rubric_version": review.get("rubric_version", LEGACY_SOURCE_FIDELITY_RUBRIC_VERSION),
            "rubric_complete": review.get("rubric_complete", False),
            "dimensions": review.get("dimensions", {}),
        }
    )
    _update_source_fidelity_check(report, review)

    summary.update(
        {
            "spec_fidelity_status": review["status"],
            "spec_fidelity_gate_status": review["status"],
            "spec_fidelity_reviewed_at": review["reviewed_at"],
            "spec_fidelity_reviewed_by": review["reviewer"],
            "spec_fidelity_review_note": review["note"],
            "spec_fidelity_evidence": review["evidence"],
            "spec_fidelity_review_mode": review.get("mode", "manual"),
            "spec_fidelity_rubric_version": review.get("rubric_version", LEGACY_SOURCE_FIDELITY_RUBRIC_VERSION),
            "spec_fidelity_rubric_complete": review.get("rubric_complete", False),
            "spec_fidelity_rubric_error": review.get("rubric_error", ""),
            "spec_fidelity_failed_dimensions": _review_failed_dimensions(review),
            "spec_fidelity_warned_dimensions": _review_warned_dimensions(review),
            "spec_fidelity_blocking_dimension_count": len(_review_blocking_dimensions(review)),
            "spec_fidelity_warning_dimension_count": len(_review_warned_dimensions(review)),
            "spec_fidelity_provisional": _review_is_provisional(review, summary),
            "spec_fidelity_materiality": _review_materiality(review),
            **codex_run_metadata_summary_fields(review.get("codex_run_metadata"), prefix="spec_fidelity"),
        }
    )
    if "confidence" in review:
        summary["spec_fidelity_confidence"] = review["confidence"]
    if isinstance(review.get("issues"), list):
        summary["spec_fidelity_issue_count"] = len(review["issues"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_reviewed_fidelity_markdown(review_path, summary=summary, report=report, artifact_dir=artifact_dir)
    _write_summary(summary_path, summary)
    return summary


def _fidelity_review_payload_for_artifact(artifact_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "agent":
        return _agent_review_payload(artifact_dir, args)
    return _review_payload(args)


def _review_payload(args: argparse.Namespace) -> dict[str, Any]:
    missing = [name for name in ("status", "reviewer", "note") if not getattr(args, name, None)]
    if missing:
        raise CLIError(f"manual fidelity review requires: {', '.join('--' + name for name in missing)}")
    return {
        "mode": "manual",
        "status": args.status,
        "reviewer": args.reviewer,
        "note": args.note,
        "evidence": list(args.evidence or []),
        "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _agent_review_payload(artifact_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    summary = _read_json_object(summary_path)
    if not summary:
        raise CLIError(f"solve summary does not exist or is not valid JSON: {summary_path}")

    result = _run_fidelity_review_agent(artifact_dir=artifact_dir, summary=summary, args=args)
    parsed = extract_json_object(result.raw_text)

    status = "llm_rejected"
    note = "fidelity reviewer did not return a JSON object; treating source-statement fidelity as rejected"
    confidence: float | int | None = None
    issues: list[Any] = []
    dimensions: dict[str, dict[str, Any]] = {}
    rubric_error = "missing dimensions"
    evidence: list[str] = [
        f"agent events: {_relative(result.events_path, artifact_dir)}",
        f"agent final message: {_relative(result.last_message_path, artifact_dir)}",
    ]

    if isinstance(parsed, dict):
        decision = str(parsed.get("status", "")).strip().lower()
        if decision == "accepted":
            status = "llm_accepted"
        elif decision == "rejected":
            status = "llm_rejected"
        else:
            status = "llm_rejected"
            note = f"fidelity reviewer returned unknown status {decision!r}; treating as rejected"

        review_note = parsed.get("review_note") or parsed.get("note")
        if isinstance(review_note, str) and review_note.strip():
            note = review_note.strip()
        elif decision in {"accepted", "rejected"}:
            note = f"agent fidelity reviewer returned {decision}"

        parsed_confidence = parsed.get("confidence")
        if isinstance(parsed_confidence, (int, float)):
            confidence = parsed_confidence

        parsed_issues = parsed.get("issues")
        if isinstance(parsed_issues, list):
            issues = parsed_issues

        parsed_evidence = parsed.get("evidence")
        if isinstance(parsed_evidence, list):
            evidence.extend(_stringify_review_evidence(item) for item in parsed_evidence)

        dimensions, dimension_errors = _normalize_source_fidelity_dimensions(parsed.get("dimensions"))
        rubric_error = "; ".join(dimension_errors)
        if dimension_errors:
            status = "llm_rejected"
            note = f"fidelity reviewer returned incomplete source_fidelity_v1 rubric: {rubric_error}; {note}"

    if result.timed_out:
        status = "llm_rejected"
        note = f"fidelity reviewer timed out; {note}"
    elif result.returncode != 0 and status == "llm_accepted":
        status = "llm_rejected"
        note = f"fidelity reviewer exited with returncode={result.returncode}; {note}"

    if status == "llm_accepted":
        block_reason = _acceptance_block_reason(summary, artifact_dir)
        if block_reason:
            status = "llm_rejected"
            note = f"agent attempted to accept, but parent gate rejected acceptance: {block_reason}; {note}"

    payload: dict[str, Any] = {
        "mode": "agent",
        "status": status,
        "reviewer": "codex-agent",
        "note": redact_text(note),
        "evidence": evidence,
        "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "issues": issues,
        "rubric_version": SOURCE_FIDELITY_RUBRIC_VERSION,
        "rubric_complete": not rubric_error,
        "rubric_error": rubric_error,
        "dimensions": dimensions,
        "agent_returncode": result.returncode,
        "agent_timed_out": result.timed_out,
        "agent_stderr": redact_text(result.stderr),
    }
    payload.update(codex_run_metadata_summary_fields(result.run_metadata))
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


def _run_fidelity_review_agent(
    *,
    artifact_dir: Path,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> FidelityReviewAgentResult:
    problem_id = str(summary.get("problem_id") or artifact_dir.name)
    session_name = f"{problem_id}-fidelity-review"
    session_dir = artifact_dir / "sessions" / session_name
    events_path = session_dir / "codex-events.jsonl"
    last_message_path = session_dir / "last-message.md"
    work_dir = neutral_work_dir(artifact_dir, session_name)
    options = _codex_options_from_args(args)
    for path in (artifact_dir, session_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)
    last_message_path.unlink(missing_ok=True)

    command = [
        "codex",
        "-a",
        args.codex_approval,
        "exec",
        "--json",
        "-C",
        str(work_dir),
        "--add-dir",
        str(artifact_dir),
        "--skip-git-repo-check",
        "-s",
        args.codex_sandbox,
        "-o",
        str(last_message_path),
    ]
    command.extend(codex_exec_model_args(options))
    command.append("-")

    timed_out = False
    try:
        result = subprocess.run(
            command,
            input=_build_fidelity_review_agent_prompt(artifact_dir=artifact_dir, summary=summary),
            capture_output=True,
            text=True,
            check=False,
            timeout=args.codex_timeout_seconds if args.codex_timeout_seconds and args.codex_timeout_seconds > 0 else None,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _coerce_timeout_stream(exc.stdout)
        stderr = _coerce_timeout_stream(exc.stderr)
        timeout_message = f"codex exec timed out after {args.codex_timeout_seconds} seconds"
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr else timeout_message + "\n"

    run_metadata = build_codex_run_metadata(command=command, options=options, stdout=stdout)
    events_path.write_text(redact_text(stdout), encoding="utf-8")
    if last_message_path.exists():
        raw_text = last_message_path.read_text(encoding="utf-8")
    else:
        raw_text = stdout
        last_message_path.write_text(raw_text, encoding="utf-8")
    return FidelityReviewAgentResult(
        raw_text=raw_text,
        returncode=returncode,
        timed_out=timed_out,
        events_path=events_path,
        last_message_path=last_message_path,
        stderr=redact_text(stderr),
        run_metadata=run_metadata,
    )


def _build_fidelity_review_agent_prompt(*, artifact_dir: Path, summary: dict[str, Any]) -> str:
    statement_path = _artifact_path(artifact_dir, summary.get("statement_file"), "statement.txt")
    problem_path = _artifact_path(artifact_dir, summary.get("spec"), "spec/problem.json")
    report_path = _artifact_path(artifact_dir, summary.get("report"), "reports/report.json")
    fidelity_report_path = _artifact_path(artifact_dir, summary.get("spec_fidelity_report"), "spec/fidelity-review.json")
    clarification_questions_path = _artifact_path(artifact_dir, summary.get("clarification_questions"), "clarification/questions.json")
    clarification_answers_path = _artifact_path(artifact_dir, summary.get("clarification_answers"), "clarification/answers.json")
    statement = _read_optional_text(statement_path)
    problem_text = _read_optional_text(problem_path)
    report_text = _read_optional_text(report_path)
    fidelity_text = _read_optional_text(fidelity_report_path)
    clarification_questions_text = _read_optional_text(clarification_questions_path)
    clarification_answers_text = _read_optional_text(clarification_answers_path)

    return f"""You are running as a nested Codex agent for OR-LLM-Agent `review-fidelity --mode agent`.

Goal: decide whether the generated OR-CI ProblemSpec faithfully represents the original natural-language problem statement.

Do not edit repository source files or artifact files. Read the materials below and return exactly one JSON object.

Required JSON shape:
{{
  "rubric_version": "source_fidelity_v1",
  "status": "accepted" | "rejected",
  "confidence": 0.0,
  "dimensions": {{
    "source_suitability": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "data_completeness": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "sets_and_indices": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "numeric_parameters": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "action_space": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "objective": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "units_and_scaling": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "constraint_families": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}},
    "metamorphic_coverage": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "reports/report.json"]}},
    "clarification_dependency": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["clarification/answers.json"]}},
    "materiality": {{"status": "pass|warn|fail|not_applicable", "severity": "none|minor|major|critical", "finding": "one sentence", "evidence": ["source-backed observation"], "artifact_refs": ["statement", "spec/problem.json"]}}
  }},
  "issues": [
    {{"severity": "critical|major|minor", "field": "field or concept", "message": "specific mismatch or risk"}}
  ],
  "review_note": "short decision rationale",
  "evidence": ["short source-backed observations"]
}}

Decision rules:
- Accept only when the generated `instance`, objective sense and coefficients, constraint families and bounds, and metamorphic paths are faithful to the source statement.
- If clarification artifacts are present, evaluate fidelity against the original statement plus approved human/source-backed clarification answers.
- Do not treat agent-generated assumptions as approved clarification. Mark `clarification_dependency` as `warn` or `fail` if assumptions supply objective, action-space, unit, or data facts.
- OR-CI `PASS` proves only that the generated submission passed the generated spec. It does not prove that the spec matches the original statement.
- Reject if a value, set, action, objective, constraint, unit/scaling convention, or important metamorphic path is missing, ambiguous, invented, or materially changed.
- Reject if any hard-block dimension (`data_completeness`, `action_space`, `objective`, `units_and_scaling`, `constraint_families`) has `status=fail` and `severity=major` or `critical`.
- Reject if the generated spec validation or model verification did not pass.
- Keep the answer to one JSON object and no surrounding prose.

Solve summary:
```json
{_trim_for_prompt(json.dumps(summary, ensure_ascii=False, indent=2), 6000)}
```

Original statement from `{statement_path}`:
```text
{_trim_for_prompt(statement, 8000)}
```

Generated ProblemSpec from `{problem_path}`:
```json
{_trim_for_prompt(problem_text, 12000)}
```

OR-CI verification report from `{report_path}`:
```json
{_trim_for_prompt(report_text, 6000)}
```

Existing fidelity report from `{fidelity_report_path}`:
```json
{_trim_for_prompt(fidelity_text, 6000)}
```

Clarification questions from `{clarification_questions_path}`:
```json
{_trim_for_prompt(clarification_questions_text, 6000)}
```

Clarification answers from `{clarification_answers_path}`:
```json
{_trim_for_prompt(clarification_answers_text, 6000)}
```
"""


def _trim_for_prompt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _stringify_review_evidence(item: Any) -> str:
    if isinstance(item, str):
        return item
    try:
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(item)


def _normalize_fidelity_review(review: dict[str, Any], *, summary: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(review)
    dimensions, dimension_errors = _normalize_source_fidelity_dimensions(normalized.get("dimensions"))
    if dimensions:
        normalized["dimensions"] = dimensions
        normalized["rubric_version"] = SOURCE_FIDELITY_RUBRIC_VERSION
        normalized["rubric_complete"] = not dimension_errors
        normalized["rubric_error"] = "; ".join(dimension_errors)
    elif normalized.get("mode") == "agent":
        normalized["dimensions"] = {}
        normalized["rubric_version"] = SOURCE_FIDELITY_RUBRIC_VERSION
        normalized["rubric_complete"] = False
        normalized["rubric_error"] = normalized.get("rubric_error") or "missing dimensions"
        if normalized.get("status") in FIDELITY_ACCEPTED_STATUSES:
            normalized["status"] = "llm_rejected"
            normalized["note"] = f"rubric incomplete: {normalized['rubric_error']}; {normalized.get('note', '')}"
    else:
        normalized["dimensions"] = {}
        normalized["rubric_version"] = LEGACY_SOURCE_FIDELITY_RUBRIC_VERSION
        normalized["rubric_complete"] = False
        normalized["rubric_error"] = ""

    blocking = _review_blocking_dimensions(normalized)
    if normalized.get("status") in FIDELITY_ACCEPTED_STATUSES and blocking:
        normalized["status"] = "llm_rejected" if normalized.get("status") == "llm_accepted" else "rejected"
        issue = {
            "severity": "critical",
            "field": "source_fidelity_rubric",
            "message": f"hard-block dimension failure: {', '.join(blocking)}",
        }
        issues = normalized.get("issues")
        normalized["issues"] = [*(issues if isinstance(issues, list) else []), issue]
        normalized["note"] = f"rubric hard-block dimension failure ({', '.join(blocking)}); {normalized.get('note', '')}"

    if _summary_has_provisional_clarification(summary) and normalized.get("dimensions"):
        clarification = normalized["dimensions"].get("clarification_dependency")
        if isinstance(clarification, dict) and clarification.get("status") in {"pass", "not_applicable"}:
            clarification["status"] = "warn"
            clarification["severity"] = "major"
            clarification["finding"] = "Clarification source is provisional and cannot count as human/source-backed evidence."
            clarification.setdefault("evidence", []).append(f"clarification_source={summary.get('clarification_source', '')}")
    return normalized


def _normalize_source_fidelity_dimensions(value: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not isinstance(value, dict):
        return {}, ["dimensions must be a JSON object"]
    normalized: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name in SOURCE_FIDELITY_DIMENSIONS:
        raw = value.get(name)
        if not isinstance(raw, dict):
            errors.append(f"{name} is missing or not an object")
            continue
        status = str(raw.get("status", "")).strip().lower()
        severity = str(raw.get("severity", "")).strip().lower()
        if status not in SOURCE_FIDELITY_DIMENSION_STATUSES:
            errors.append(f"{name}.status must be one of {sorted(SOURCE_FIDELITY_DIMENSION_STATUSES)}")
            status = "fail"
        if severity not in SOURCE_FIDELITY_SEVERITIES:
            errors.append(f"{name}.severity must be one of {sorted(SOURCE_FIDELITY_SEVERITIES)}")
            severity = "critical" if status == "fail" else "minor"
        normalized[name] = {
            "status": status,
            "severity": severity,
            "finding": str(raw.get("finding", "")).strip(),
            "evidence": [_stringify_review_evidence(item) for item in raw.get("evidence", [])]
            if isinstance(raw.get("evidence"), list)
            else [],
            "artifact_refs": [_stringify_review_evidence(item) for item in raw.get("artifact_refs", [])]
            if isinstance(raw.get("artifact_refs"), list)
            else [],
        }
    return normalized, errors


def _review_failed_dimensions(review: dict[str, Any]) -> list[str]:
    return [
        name
        for name, dimension in _review_dimensions(review).items()
        if dimension.get("status") == "fail"
    ]


def _review_warned_dimensions(review: dict[str, Any]) -> list[str]:
    warned: list[str] = []
    for name, dimension in _review_dimensions(review).items():
        status = dimension.get("status")
        severity = dimension.get("severity")
        if status == "warn" or (status == "fail" and name not in _review_blocking_dimensions(review)):
            warned.append(name)
        elif status == "pass" and severity in {"minor", "major", "critical"}:
            warned.append(name)
    return warned


def _review_blocking_dimensions(review: dict[str, Any]) -> list[str]:
    return [
        name
        for name, dimension in _review_dimensions(review).items()
        if name in SOURCE_FIDELITY_HARD_BLOCK_DIMENSIONS
        and dimension.get("status") == "fail"
        and dimension.get("severity") in SOURCE_FIDELITY_BLOCKING_SEVERITIES
    ]


def _review_dimensions(review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dimensions = review.get("dimensions")
    return dimensions if isinstance(dimensions, dict) else {}


def _review_materiality(review: dict[str, Any]) -> str:
    dimension = _review_dimensions(review).get("materiality")
    if not isinstance(dimension, dict):
        return ""
    status = str(dimension.get("status", "")).strip()
    severity = str(dimension.get("severity", "")).strip()
    if status and severity:
        return f"{status}/{severity}"
    return status or severity


def _review_is_provisional(review: dict[str, Any], summary: dict[str, Any]) -> bool:
    if review.get("provisional") is True or _summary_has_provisional_clarification(summary):
        return True
    dimension = _review_dimensions(review).get("clarification_dependency")
    if not isinstance(dimension, dict):
        return False
    text = " ".join(
        [
            str(dimension.get("finding", "")),
            *[str(item) for item in dimension.get("evidence", []) if isinstance(item, str)],
        ]
    ).lower()
    return dimension.get("status") in {"warn", "fail"} and ("provisional" in text or "agent" in text or "assumption" in text)


def _summary_has_provisional_clarification(summary: dict[str, Any]) -> bool:
    source = str(summary.get("clarification_source", "")).lower()
    return "provisional" in source or "assumption" in source or "agent" in source


def _ensure_acceptance_is_allowed(summary: dict[str, Any], artifact_dir: Path) -> None:
    block_reason = _acceptance_block_reason(summary, artifact_dir)
    if block_reason:
        raise CLIError(f"cannot accept fidelity review: {block_reason}")


def _acceptance_block_reason(summary: dict[str, Any], artifact_dir: Path) -> str:
    if summary.get("spec_validation_status") != "passed":
        return "spec validation did not pass"
    if summary.get("verification_status") != "PASS":
        return "OR-CI verification did not pass"
    problem_path = _artifact_path(artifact_dir, summary.get("spec"), "spec/problem.json")
    if not problem_path.is_file():
        return f"generated spec is missing: {problem_path}"
    return ""


def _update_source_fidelity_check(report: dict[str, Any], review: dict[str, Any]) -> None:
    checks = report.get("automatic_checks")
    if not isinstance(checks, list):
        checks = []
    status = "PASS" if review["status"] in FIDELITY_ACCEPTED_STATUSES else "FAIL"
    detail = f"{review.get('mode', 'manual')} review {review['status']}: {review['note']}"
    failed_dimensions = _review_failed_dimensions(review)
    if failed_dimensions:
        detail = f"{detail}; failed_dimensions={', '.join(failed_dimensions)}"
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "source_statement_fidelity":
            check["status"] = status
            check["detail"] = detail
            break
    else:
        checks.append({"name": "source_statement_fidelity", "status": status, "detail": detail})
    report["automatic_checks"] = checks


def _write_reviewed_fidelity_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    report: dict[str, Any],
    artifact_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    statement = _read_optional_text(_artifact_path(artifact_dir, summary.get("statement_file"), "statement.txt"))
    if not statement:
        statement = str(report.get("statement_excerpt", "")).strip()
    statement_excerpt = statement.replace("\r\n", "\n").strip()
    if len(statement_excerpt) > 800:
        statement_excerpt = statement_excerpt[:797].rstrip() + "..."
    check_lines = [
        f"- `{check.get('status', 'UNKNOWN')}` {check.get('name', 'unknown')}: {check.get('detail', '')}"
        for check in report.get("automatic_checks", [])
        if isinstance(check, dict)
    ]
    risk_lines = [
        f"- `{flag.get('severity', 'unknown')}` {flag.get('code', 'unknown')}: {flag.get('message', '')}"
        for flag in report.get("risk_flags", [])
        if isinstance(flag, dict)
    ] or ["- None detected by automatic checks."]
    review = report.get("review") if isinstance(report.get("review"), dict) else {}
    evidence_lines = [f"- {item}" for item in review.get("evidence", [])] or ["- None recorded."]
    rubric_table = _rubric_markdown_table(review.get("dimensions"))

    path.write_text(
        f"""# ProblemSpec Fidelity Review

## Run

- Problem: `{summary.get('problem_id', 'unknown')}`
- Statement file: `{summary.get('statement_file', '')}`
- Generated spec: `{summary.get('spec', '')}`
- Spec raw output: `{summary.get('spec_raw', '')}`
- Spec generation: `{summary.get('spec_generation_status', '')}`
- Spec validation: `{summary.get('spec_validation_status', '')}`
- Model generation: `{summary.get('model_generation_status', '')}`
- Verification: `{summary.get('verification_status', '')}`
- Classification: `{summary.get('classification', '')}`
- Fidelity status: `{summary.get('spec_fidelity_status', '')}`
- Fidelity gate: `{summary.get('spec_fidelity_gate_status', '')}`
- Rubric version: `{summary.get('spec_fidelity_rubric_version', '')}`
- Rubric complete: `{summary.get('spec_fidelity_rubric_complete', False)}`
- Provisional: `{summary.get('spec_fidelity_provisional', False)}`
- Failed dimensions: `{", ".join(_string_list(summary.get('spec_fidelity_failed_dimensions'))) or "-"}`
- Warned dimensions: `{", ".join(_string_list(summary.get('spec_fidelity_warned_dimensions'))) or "-"}`
- Materiality: `{summary.get('spec_fidelity_materiality', '') or "-"}`
- Structured report: `{summary.get('spec_fidelity_report', '')}`

## Review Decision

- Status: `{review.get('status', 'unknown')}`
- Reviewer: `{review.get('reviewer', '')}`
- Reviewed at: `{review.get('reviewed_at', '')}`
- Note: {review.get('note', '')}

## Evidence

{chr(10).join(evidence_lines)}

## Rubric Dimensions

{rubric_table}

## Statement Excerpt

```text
{statement_excerpt}
```

## Automatic Checks

{chr(10).join(check_lines)}

## Risk Flags

{chr(10).join(risk_lines)}
""",
        encoding="utf-8",
    )


def _rubric_markdown_table(dimensions: Any) -> str:
    if not isinstance(dimensions, dict) or not dimensions:
        return "No structured rubric dimensions recorded."
    lines = [
        "| Dimension | Status | Severity | Finding | Evidence |",
        "|---|---|---|---|---|",
    ]
    for name in SOURCE_FIDELITY_DIMENSIONS:
        dimension = dimensions.get(name)
        if not isinstance(dimension, dict):
            continue
        evidence = "; ".join(_string_list(dimension.get("evidence"))) or "-"
        lines.append(
            f"| `{name}` | `{dimension.get('status', '')}` | `{dimension.get('severity', '')}` | "
            f"{str(dimension.get('finding', '')).replace('|', '/')} | {evidence.replace('|', '/')} |"
        )
    return "\n".join(lines)


def _artifact_path(artifact_dir: Path, value: Any, fallback: str) -> Path:
    if isinstance(value, str) and value:
        path = Path(value)
        return path if path.is_absolute() else artifact_dir / path
    return artifact_dir / fallback


def _read_optional_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _batch_ids_from_summary(artifact_dir: Path) -> list[str]:
    summary_path = artifact_dir / "summary.json"
    if not summary_path.is_file():
        raise CLIError(f"batch summary does not exist: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise CLIError(f"batch summary contains no rows: {summary_path}")
    ids = [str(row.get("problem_id")) for row in rows if isinstance(row, dict) and row.get("problem_id")]
    if not ids:
        raise CLIError(f"batch summary contains no problem ids: {summary_path}")
    return ids


def _batch_ids_for_aggregate(artifact_dir: Path, *, fallback_ids: list[str]) -> list[str]:
    summary_ids: list[str] = []
    try:
        summary_ids = _batch_ids_from_summary(artifact_dir)
    except CLIError:
        summary_ids = []

    statement_dir = artifact_dir / "statements"
    statement_ids = [path.stem for path in sorted(statement_dir.glob("*.txt"))] if statement_dir.is_dir() else []
    if statement_ids:
        if summary_ids and set(summary_ids) == set(statement_ids):
            return summary_ids
        return statement_ids
    if summary_ids:
        return summary_ids

    case_ids = [path.name for path in sorted(artifact_dir.iterdir()) if (path / "summary.json").is_file()]
    return case_ids or fallback_ids


def _needs_human_ids_from_batch_summary(artifact_dir: Path) -> list[str]:
    summary_path = artifact_dir / "summary.json"
    if not summary_path.is_file():
        raise CLIError(f"batch summary does not exist: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise CLIError(f"batch summary contains no rows: {summary_path}")
    ids = [
        str(row.get("problem_id"))
        for row in rows
        if isinstance(row, dict) and row.get("problem_id") and row.get("capability_status") == "needs_human"
    ]
    if not ids:
        raise CLIError(f"batch summary contains no needs_human cases: {summary_path}")
    return ids


def _clarification_answers_path(clarifications_dir: Path, problem_id: str) -> Path:
    candidates = [
        clarifications_dir / f"{problem_id}.json",
        clarifications_dir / problem_id / "answers.json",
        clarifications_dir / problem_id / "clarification" / "answers.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _clarification_row_from_questions(
    problem_id: str,
    case_dir: Path,
    artifact_dir: Path,
    question_artifact: dict[str, Any],
) -> dict[str, Any]:
    source_summary = _read_json_object(case_dir / "summary.json")
    return {
        "problem_id": problem_id,
        "source_artifact": _relative(case_dir, artifact_dir),
        "baseline_block_reason": source_summary.get("reason", source_summary.get("capability_review_note", "")),
        "clarification_status": "questions_prepared",
        "clarification_question_count": len(question_artifact.get("questions", [])),
        "clarification_answer_count": 0,
        "clarification_source": "",
        "clarification_gate_status": "awaiting_answers",
        "clarification_questions": _relative(_canonical_clarification_questions_path(case_dir), artifact_dir),
        "generated_questions": _question_report_items(question_artifact),
        "answer_provenance": [],
        "classification": "not_run",
        "verification_status": "not_run",
        "spec_fidelity_status": "not_reviewed",
    }


def _clarification_row_from_solve(result: dict[str, Any], batch_artifact_dir: Path) -> dict[str, Any]:
    source_artifact = str(result.get("source_artifact", result.get("clarified_from", "")))
    row = {
        "problem_id": result["problem_id"],
        "source_artifact": source_artifact,
        "baseline_block_reason": "",
        "clarified_from": result.get("clarified_from", ""),
        "clarification_status": result.get("clarification_status", ""),
        "clarification_question_count": result.get("clarification_question_count", 0),
        "clarification_answer_count": result.get("clarification_answer_count", 0),
        "clarification_source": result.get("clarification_source", ""),
        "clarification_gate_status": result.get("clarification_gate_status", ""),
        "clarification_questions": result.get("clarification_questions", ""),
        "clarification_answers": result.get("clarification_answers", ""),
        "classification": result.get("classification", ""),
        "verification_status": result.get("verification_status", ""),
        "spec_fidelity_status": result.get("spec_fidelity_status", "not_reviewed"),
        "spec_fidelity_gate_status": result.get("spec_fidelity_gate_status", ""),
        "resolution_artifact": result.get("resolution_artifact", ""),
    }
    summary_path = batch_artifact_dir / str(result["problem_id"]) / "summary.json"
    source_summary = _read_json_object(summary_path)
    if source_summary:
        row["baseline_block_reason"] = source_summary.get("reason", source_summary.get("capability_review_note", ""))
    resolution_dir = Path(str(result.get("resolution_artifact", "")))
    if resolution_dir.is_dir():
        question_artifact = _read_json_object(_artifact_path(resolution_dir, result.get("clarification_questions"), "clarification/questions.json"))
        answer_artifact = _read_json_object(_artifact_path(resolution_dir, result.get("clarification_answers"), "clarification/answers.json"))
        row["generated_questions"] = _question_report_items(question_artifact)
        row["answer_provenance"] = _answer_provenance_items(answer_artifact)
    return row


def _question_report_items(question_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    questions = question_artifact.get("questions") if isinstance(question_artifact, dict) else None
    if not isinstance(questions, list):
        return []
    items: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        items.append(
            {
                "id": question.get("id", ""),
                "issue_type": question.get("issue_type", ""),
                "prompt": question.get("prompt", ""),
                "source_evidence": question.get("source_evidence", ""),
            }
        )
    return items


def _answer_provenance_items(answer_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    answers = answer_artifact.get("answers") if isinstance(answer_artifact, dict) else None
    if not isinstance(answers, list):
        return []
    items: list[dict[str, Any]] = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        items.append(
            {
                "question_id": answer.get("question_id", ""),
                "reviewer": answer.get("reviewer", ""),
                "source": answer.get("source", ""),
                "rationale": answer.get("rationale", ""),
            }
        )
    return items


def _case_exit_code(case_dir: Path) -> int:
    summary = _read_json_object(case_dir / "summary.json")
    if summary.get("model_generation_status") == "generated" and summary.get("verification_status") == "PASS":
        return 0
    return 1


def _generation_inputs_from_args(args: argparse.Namespace) -> GenerationInputs:
    if args.problem:
        return _problem_generation_inputs(args.problem, args.statement_file)
    return _bwor_generation_inputs(args.bwor_id, args.dataset, args.or_ci_root)


def _bwor_generation_inputs(problem_id: str, dataset_path: Path, or_ci_root: Path) -> GenerationInputs:
    problem_path = default_problem_path(problem_id, or_ci_root)
    if not problem_path.is_file():
        raise CLIError(f"problem file does not exist: {problem_path}")
    return GenerationInputs(
        problem_id=problem_id,
        record=load_bwor_record(dataset_path, problem_id),
        problem_path=problem_path,
        problem=load_problem(problem_path),
    )


def _problem_generation_inputs(problem_path: Path, statement_file: Path | None) -> GenerationInputs:
    if not problem_path.is_file():
        raise CLIError(f"problem file does not exist: {problem_path}")
    problem = load_problem(problem_path)
    problem_id = str(problem.get("id") or problem_path.stem)
    statement = _read_statement(statement_file) if statement_file else ""
    return GenerationInputs(
        problem_id=problem_id,
        record={"id": problem_id, "en_question": statement},
        problem_path=problem_path,
        problem=problem,
    )


def _read_statement(path: Path) -> str:
    if not path.is_file():
        raise CLIError(f"statement file does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()


def _default_spec_status_path(out_path: Path) -> Path:
    return out_path.parent / "status.json"


def _run_capability_agent(
    *,
    problem_id: str,
    statement: str,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> CapabilityAgentResult:
    session_name = f"{problem_id}-capability"
    session_dir = artifact_dir / "sessions" / session_name
    events_path = session_dir / "codex-events.jsonl"
    last_message_path = session_dir / "last-message.md"
    work_dir = neutral_work_dir(artifact_dir, session_name)
    options = _codex_options_from_args(args)
    for path in (artifact_dir, session_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)
    last_message_path.unlink(missing_ok=True)

    command = [
        "codex",
        "-a",
        args.codex_approval,
        "exec",
        "--json",
        "-C",
        str(work_dir),
        "--add-dir",
        str(artifact_dir),
        "--skip-git-repo-check",
        "-s",
        args.codex_sandbox,
        "-o",
        str(last_message_path),
    ]
    command.extend(codex_exec_model_args(options))
    command.append("-")

    timed_out = False
    try:
        result = subprocess.run(
            command,
            input=_build_capability_agent_prompt(problem_id=problem_id, statement=statement),
            capture_output=True,
            text=True,
            check=False,
            timeout=args.codex_timeout_seconds if args.codex_timeout_seconds and args.codex_timeout_seconds > 0 else None,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _coerce_timeout_stream(exc.stdout)
        stderr = _coerce_timeout_stream(exc.stderr)
        timeout_message = f"codex exec timed out after {args.codex_timeout_seconds} seconds"
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr else timeout_message + "\n"

    run_metadata = build_codex_run_metadata(command=command, options=options, stdout=stdout)
    events_path.write_text(redact_text(stdout), encoding="utf-8")
    if last_message_path.exists():
        raw_text = last_message_path.read_text(encoding="utf-8")
    else:
        raw_text = stdout
        last_message_path.write_text(raw_text, encoding="utf-8")
    return CapabilityAgentResult(
        raw_text=raw_text,
        returncode=returncode,
        timed_out=timed_out,
        events_path=events_path,
        last_message_path=last_message_path,
        stderr=redact_text(stderr),
        run_metadata=run_metadata,
    )


def _build_capability_agent_prompt(*, problem_id: str, statement: str) -> str:
    return f"""{CAPABILITY_SYSTEM_PROMPT}

You are running as a nested Codex agent for OR-LLM-Agent `classify-statement --mode agent`.
Do not edit repository source files or artifact files. Return only the capability
classification JSON object as your final answer.

{build_statement_capability_prompt(problem_id, statement)}

Start now. Complete the classification without asking for confirmation.
"""


def _run_clarification_question_agent(
    *,
    problem_id: str,
    statement: str,
    capability: dict[str, Any],
    artifact_dir: Path,
    args: argparse.Namespace,
) -> ClarificationAgentResult:
    session_name = f"{problem_id}-clarification-questions"
    session_dir = artifact_dir / "sessions" / session_name
    events_path = session_dir / "codex-events.jsonl"
    last_message_path = session_dir / "last-message.md"
    work_dir = neutral_work_dir(artifact_dir, session_name)
    options = _codex_options_from_args(args)
    for path in (artifact_dir, session_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)
    last_message_path.unlink(missing_ok=True)

    command = [
        "codex",
        "-a",
        args.codex_approval,
        "exec",
        "--json",
        "-C",
        str(work_dir),
        "--add-dir",
        str(artifact_dir),
        "--skip-git-repo-check",
        "-s",
        args.codex_sandbox,
        "-o",
        str(last_message_path),
    ]
    command.extend(codex_exec_model_args(options))
    command.append("-")

    timed_out = False
    try:
        result = subprocess.run(
            command,
            input=_build_clarification_question_agent_prompt(
                problem_id=problem_id,
                statement=statement,
                capability=capability,
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=args.codex_timeout_seconds if args.codex_timeout_seconds and args.codex_timeout_seconds > 0 else None,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _coerce_timeout_stream(exc.stdout)
        stderr = _coerce_timeout_stream(exc.stderr)
        timeout_message = f"codex exec timed out after {args.codex_timeout_seconds} seconds"
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr else timeout_message + "\n"

    run_metadata = build_codex_run_metadata(command=command, options=options, stdout=stdout)
    events_path.write_text(redact_text(stdout), encoding="utf-8")
    if last_message_path.exists():
        raw_text = last_message_path.read_text(encoding="utf-8")
    else:
        raw_text = stdout
        last_message_path.write_text(raw_text, encoding="utf-8")
    return ClarificationAgentResult(
        raw_text=raw_text,
        returncode=returncode,
        timed_out=timed_out,
        events_path=events_path,
        last_message_path=last_message_path,
        stderr=redact_text(stderr),
        run_metadata=run_metadata,
    )


def _build_clarification_question_agent_prompt(
    *,
    problem_id: str,
    statement: str,
    capability: dict[str, Any],
) -> str:
    return f"""{CLARIFICATION_SYSTEM_PROMPT}

You are running as a nested Codex agent for OR-LLM-Agent `prepare-clarification`.
Do not edit repository source files or artifact files. Return only the
clarification question JSON object as your final answer.

{build_clarification_question_prompt(problem_id, statement, capability)}

Start now. Complete the question set without asking for confirmation.
"""


def _normalize_capability_payload(
    *,
    problem_id: str,
    parsed: dict[str, Any] | None,
    raw_path: Path,
    agent_result: CapabilityAgentResult,
) -> dict[str, Any]:
    status = "needs_human"
    generation_status = "no_json"
    note = "capability classifier did not return a JSON object; stopping before ProblemSpec generation"
    problem_family = "unknown"
    supported_features: list[str] = []
    unsupported_features: list[str] = []
    missing_information: list[str] = []
    recommended_next_action = "ask_human"
    confidence: float | int | None = None

    if isinstance(parsed, dict):
        requested_status = str(parsed.get("status", "")).strip().lower()
        if requested_status in CAPABILITY_STATUSES:
            status = requested_status
            generation_status = "classified"
        else:
            generation_status = "invalid_status"
            unsupported_features.append(f"unknown capability status: {requested_status!r}")

        problem_family_value = parsed.get("problem_family")
        if isinstance(problem_family_value, str) and problem_family_value.strip():
            problem_family = problem_family_value.strip()

        supported_features = _string_list(parsed.get("supported_features"))
        unsupported_features.extend(_string_list(parsed.get("unsupported_features")))
        missing_information = _string_list(parsed.get("missing_information"))

        action = parsed.get("recommended_next_action")
        if isinstance(action, str) and action.strip():
            recommended_next_action = action.strip()

        review_note = parsed.get("review_note") or parsed.get("note")
        if isinstance(review_note, str) and review_note.strip():
            note = review_note.strip()
        elif generation_status == "classified":
            note = f"capability classifier returned {status}"

        parsed_confidence = parsed.get("confidence")
        if isinstance(parsed_confidence, (int, float)) and not isinstance(parsed_confidence, bool):
            confidence = parsed_confidence

    if agent_result.timed_out:
        status = "needs_human"
        generation_status = "agent_timeout"
        note = f"capability classifier timed out; {note}"
    elif agent_result.returncode != 0:
        if status == "supported":
            status = "needs_human"
            unsupported_features.append("classifier process exited nonzero after returning supported")
            recommended_next_action = "ask_human"
        if generation_status == "classified":
            generation_status = "classified_with_agent_error"
        note = f"capability classifier exited with returncode={agent_result.returncode}; {note}"

    payload: dict[str, Any] = {
        "problem_id": problem_id,
        "mode": "agent",
        "status": status,
        "capability_generation_status": generation_status,
        "problem_family": problem_family,
        "supported_features": supported_features,
        "unsupported_features": unsupported_features,
        "missing_information": missing_information,
        "recommended_next_action": recommended_next_action,
        "review_note": redact_text(note),
        "raw_response": str(raw_path),
        "agent_returncode": agent_result.returncode,
        "agent_timed_out": agent_result.timed_out,
        "codex_events": str(agent_result.events_path),
        "last_message": str(agent_result.last_message_path),
        "agent_stderr": redact_text(agent_result.stderr),
    }
    payload.update(codex_run_metadata_summary_fields(agent_result.run_metadata))
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_stringify_review_evidence(item) for item in value]


def _run_problem_spec_agent(
    *,
    problem_id: str,
    statement: str,
    artifact_dir: Path,
    args: argparse.Namespace,
    attempt: int = 1,
    previous_problem: dict[str, Any] | None = None,
    previous_response: str = "",
    repair_error: str = "",
    clarification_context: dict[str, Any] | None = None,
) -> ProblemSpecAgentResult:
    session_name = f"{problem_id}-spec" if attempt == 1 else f"{problem_id}-spec-repair-{attempt - 1}"
    session_dir = artifact_dir / "sessions" / session_name
    events_path = session_dir / "codex-events.jsonl"
    last_message_path = session_dir / "last-message.md"
    work_dir = neutral_work_dir(artifact_dir, session_name)
    options = _codex_options_from_args(args)
    for path in (artifact_dir, session_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)

    command = [
        "codex",
        "-a",
        args.codex_approval,
        "exec",
        "--json",
        "-C",
        str(work_dir),
        "--add-dir",
        str(artifact_dir),
        "--skip-git-repo-check",
        "-s",
        args.codex_sandbox,
        "-o",
        str(last_message_path),
    ]
    command.extend(codex_exec_model_args(options))
    command.append("-")

    timed_out = False
    try:
        result = subprocess.run(
            command,
            input=_build_problem_spec_agent_prompt(
                problem_id,
                statement,
                previous_problem=previous_problem,
                previous_response=previous_response,
                repair_error=repair_error,
                clarification_context=clarification_context,
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=args.codex_timeout_seconds if args.codex_timeout_seconds and args.codex_timeout_seconds > 0 else None,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _coerce_timeout_stream(exc.stdout)
        stderr = _coerce_timeout_stream(exc.stderr)
        timeout_message = f"codex exec timed out after {args.codex_timeout_seconds} seconds"
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr else timeout_message + "\n"

    run_metadata = build_codex_run_metadata(command=command, options=options, stdout=stdout)
    events_path.write_text(redact_text(stdout), encoding="utf-8")
    if last_message_path.exists():
        raw_text = last_message_path.read_text(encoding="utf-8")
    else:
        raw_text = stdout
        last_message_path.write_text(raw_text, encoding="utf-8")
    return ProblemSpecAgentResult(
        raw_text=raw_text,
        returncode=returncode,
        timed_out=timed_out,
        events_path=events_path,
        last_message_path=last_message_path,
        stderr=redact_text(stderr),
        run_metadata=run_metadata,
    )


def _build_problem_spec_agent_prompt(
    problem_id: str,
    statement: str,
    *,
    previous_problem: dict[str, Any] | None = None,
    previous_response: str = "",
    repair_error: str = "",
    clarification_context: dict[str, Any] | None = None,
) -> str:
    repair_context = _build_problem_spec_repair_context(
        previous_problem=previous_problem,
        previous_response=previous_response,
        repair_error=repair_error,
    )
    prompt = (
        build_clarified_problem_spec_prompt(problem_id, statement, clarification_context)
        if clarification_context is not None
        else build_problem_spec_prompt(problem_id, statement)
    )
    return f"""{PROBLEM_SPEC_SYSTEM_PROMPT}

You are running as a nested Codex agent for OR-LLM-Agent `spec --mode agent`.
Do not edit repository source files. Return the generated problem metadata as
your final answer.

{prompt}
{repair_context}

Start now. Complete the workflow without asking for confirmation.
"""


def _build_problem_spec_repair_context(
    *,
    previous_problem: dict[str, Any] | None,
    previous_response: str,
    repair_error: str,
) -> str:
    if not repair_error:
        return ""

    previous = ""
    if previous_problem is not None:
        previous = json.dumps(previous_problem, ensure_ascii=False, indent=2)
    elif previous_response:
        previous = previous_response.strip()
    if len(previous) > 4000:
        previous = previous[:3997].rstrip() + "..."

    return f"""

Previous generated metadata needs repair. Repair it now.

Repair issue:
```text
{repair_error.strip()}
```

Previous generated content:
```json
{previous}
```

Return a complete corrected metadata JSON object. Do not return a patch.
"""


def _spec_validation_payload(validation: SpecValidationResult) -> dict[str, Any]:
    return {
        "spec_validation_status": validation.status,
        "validation_returncode": validation.returncode,
        "validation_stdout": validation.stdout,
        "validation_stderr": validation.stderr,
    }


def _spec_attempt_raw_path(raw_path: Path, attempt: int) -> Path:
    return raw_path.with_name(f"{raw_path.stem}-attempt-{attempt}{raw_path.suffix}")


def _format_spec_repair_error(validation: SpecValidationResult) -> str:
    detail = validation.stderr.strip() or validation.stdout.strip()
    if detail:
        return detail
    return f"or-ci validate-spec failed with returncode={validation.returncode}"


def _spec_repair_status(attempts: list[dict[str, Any]], *, initial_repair_requested: bool = False) -> str:
    if not attempts:
        return "failed"
    ready = (
        attempts[-1].get("spec_generation_status") == "generated"
        and attempts[-1].get("spec_validation_status") == "passed"
    )
    if not ready:
        return "failed"
    return "repaired" if initial_repair_requested or len(attempts) > 1 else "not_needed"


def _spec_is_ready(result: dict[str, Any]) -> bool:
    return result.get("spec_generation_status") == "generated" and result.get("spec_validation_status") == "passed"


def _write_spec_fidelity_review(
    path: Path,
    *,
    report_path: Path,
    summary: dict[str, Any],
    statement: str,
    problem_path: Path,
    clarification_context: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fidelity = _build_spec_fidelity_payload(
        summary=summary,
        statement=statement,
        problem_path=problem_path,
        clarification_context=clarification_context,
    )
    summary["spec_fidelity_gate_status"] = fidelity["gate_status"]
    summary["spec_fidelity_risk_flags"] = [flag["code"] for flag in fidelity["risk_flags"]]
    summary["spec_fidelity_rubric_version"] = SOURCE_FIDELITY_RUBRIC_VERSION
    summary["spec_fidelity_rubric_complete"] = False
    summary["spec_fidelity_rubric_error"] = "not_reviewed"
    summary["spec_fidelity_failed_dimensions"] = []
    summary["spec_fidelity_warned_dimensions"] = []
    summary["spec_fidelity_blocking_dimension_count"] = 0
    summary["spec_fidelity_warning_dimension_count"] = 0
    summary["spec_fidelity_provisional"] = _summary_has_provisional_clarification(summary)
    summary["spec_fidelity_materiality"] = ""
    report_path.write_text(json.dumps(fidelity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    statement_excerpt = statement.replace("\r\n", "\n").strip()
    if len(statement_excerpt) > 800:
        statement_excerpt = statement_excerpt[:797].rstrip() + "..."
    check_lines = [
        f"- `{check['status']}` {check['name']}: {check['detail']}"
        for check in fidelity["automatic_checks"]
    ]
    risk_lines = [
        f"- `{flag['severity']}` {flag['code']}: {flag['message']}"
        for flag in fidelity["risk_flags"]
    ] or ["- None detected by automatic checks."]
    clarification_section = ""
    if clarification_context is not None:
        clarification_section = f"""

## Clarification Context

- Clarified from: `{summary.get('clarified_from', '')}`
- Clarification status: `{summary.get('clarification_status', '')}`
- Clarification gate: `{summary.get('clarification_gate_status', '')}`
- Question artifact: `{summary.get('clarification_questions', '')}`
- Answer artifact: `{summary.get('clarification_answers', '')}`
- Question count: `{summary.get('clarification_question_count', 0)}`
- Answer count: `{summary.get('clarification_answer_count', 0)}`
"""

    path.write_text(
        f"""# ProblemSpec Fidelity Review

## Run

- Problem: `{summary['problem_id']}`
- Statement file: `{summary['statement_file']}`
- Generated spec: `{summary['spec']}`
- Spec raw output: `{summary['spec_raw']}`
- Spec generation: `{summary['spec_generation_status']}`
- Spec validation: `{summary['spec_validation_status']}`
- Model generation: `{summary['model_generation_status']}`
- Verification: `{summary['verification_status']}`
- Classification: `{summary['classification']}`
- Fidelity status: `{summary['spec_fidelity_status']}`
- Fidelity gate: `{summary['spec_fidelity_gate_status']}`
- Structured report: `{summary['spec_fidelity_report']}`

## Statement Excerpt

```text
{statement_excerpt}
```

## Automatic Checks

{chr(10).join(check_lines)}

## Risk Flags

{chr(10).join(risk_lines)}
{clarification_section}

## Manual Checklist

- [ ] Sets and indices in `instance` match the statement.
- [ ] Parameters and numeric values in `instance` match the statement.
- [ ] Objective direction and coefficients match the statement.
- [ ] Constraint families and bounds match the statement.
- [ ] Metamorphic checks touch objective and constraint data paths, where available.
- [ ] OR-CI result is interpreted as verification against the generated spec, not proof of original-statement correctness.
- [ ] If clarification is present, the generated model matches the original statement plus approved clarification answers.

## Reviewer Note

TODO
""",
        encoding="utf-8",
    )


def _build_spec_fidelity_payload(
    *,
    summary: dict[str, Any],
    statement: str,
    problem_path: Path,
    clarification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problem = _read_json_object(problem_path)
    capability_status = str(summary.get("capability_status", "unknown"))
    if capability_status in CAPABILITY_BLOCKING_STATUSES:
        gate_status = "blocked_capability"
    else:
        gate_status = "manual_review_required" if summary.get("spec_validation_status") == "passed" else "blocked_spec_invalid"
    automatic_checks = [
        {
            "name": "or_ci_spec_validation",
            "status": "PASS" if summary.get("spec_validation_status") == "passed" else "FAIL",
            "detail": f"spec_validation_status={summary.get('spec_validation_status')}",
        },
        {
            "name": "model_verified_against_generated_spec",
            "status": _fidelity_check_status(summary.get("verification_status")),
            "detail": f"verification_status={summary.get('verification_status')}",
        },
        {
            "name": "source_statement_fidelity",
            "status": "PENDING",
            "detail": "manual review is required before treating generated spec as ground truth",
        },
    ]
    if "capability_status" in summary:
        capability_check_status = "PASS" if capability_status == "supported" else "FAIL"
        automatic_checks.append(
            {
                "name": "capability_routing",
                "status": capability_check_status,
                "detail": (
                    f"capability_status={capability_status}; "
                    f"action={summary.get('capability_recommended_next_action', '')}; "
                    f"note={summary.get('capability_review_note', '')}"
                ),
            }
        )
    if clarification_context is not None:
        automatic_checks.append(
            {
                "name": "clarification_gate",
                "status": "PASS" if summary.get("clarification_gate_status") == "passed" else "FAIL",
                "detail": (
                    f"clarification_status={summary.get('clarification_status', '')}; "
                    f"gate={summary.get('clarification_gate_status', '')}; "
                    f"questions={summary.get('clarification_question_count', 0)}; "
                    f"answers={summary.get('clarification_answer_count', 0)}"
                ),
            }
        )
    manual_checklist = [
        "sets and indices match the statement",
        "numeric parameters match the statement",
        "objective direction and coefficients match the statement",
        "constraint families and bounds match the statement",
        "metamorphic paths touch objective and constraint data paths where available",
        "OR-CI result is interpreted only against the generated spec",
    ]
    if clarification_context is not None:
        manual_checklist.append("generated spec matches the original statement plus approved clarification answers")
    return {
        "problem_id": summary["problem_id"],
        "gate_status": gate_status,
        "fidelity_status": summary["spec_fidelity_status"],
        "statement_file": summary["statement_file"],
        "statement_excerpt": statement[:800],
        "spec": summary["spec"],
        "spec_validation_status": summary["spec_validation_status"],
        "model_generation_status": summary["model_generation_status"],
        "verification_status": summary["verification_status"],
        "classification": summary["classification"],
        "automatic_checks": automatic_checks,
        "rubric_version": SOURCE_FIDELITY_RUBRIC_VERSION,
        "required_dimensions": list(SOURCE_FIDELITY_DIMENSIONS),
        "risk_flags": [*_spec_fidelity_risk_flags(problem, summary), *_capability_risk_flags(summary)],
        "manual_checklist": manual_checklist,
        "clarification": clarification_context or {},
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _fidelity_check_status(verification_status: Any) -> str:
    if verification_status == "PASS":
        return "PASS"
    if verification_status == "skipped":
        return "SKIP"
    return "FAIL"


def _spec_fidelity_risk_flags(problem: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, str]]:
    instance = problem.get("instance") if isinstance(problem.get("instance"), dict) else {}
    metamorphic = problem.get("metamorphic") if isinstance(problem.get("metamorphic"), dict) else {}
    cost_scaling = metamorphic.get("cost_scaling") if isinstance(metamorphic.get("cost_scaling"), dict) else {}
    coefficient_paths = cost_scaling.get("coefficient_paths") if isinstance(cost_scaling.get("coefficient_paths"), list) else []
    searchable = " ".join(
        [
            str(problem.get("problem_type", "")),
            str(summary.get("problem_family", "")),
            *(_flatten_keys(instance)),
            *(str(path) for path in coefficient_paths),
        ]
    ).lower()
    flags: list[dict[str, str]] = []
    has_profit = "profit" in searchable or "revenue" in searchable
    has_cost = "cost" in searchable or "price" in searchable
    has_derived_net = "net" in searchable or "benefit" in searchable
    if has_derived_net and not (has_profit and has_cost):
        flags.append(
            {
                "code": "derived_objective_without_primitives",
                "severity": "warning",
                "message": "objective data appears derived; preserve primitive profit/revenue and cost fields when available",
            }
        )
    if "constraint_relaxation" not in metamorphic:
        flags.append(
            {
                "code": "no_constraint_relaxation",
                "severity": "warning",
                "message": "metadata has no configured constraint relaxation check",
            }
        )
    if _summary_has_provisional_clarification(summary):
        flags.append(
            {
                "code": "provisional_clarification_dependency",
                "severity": "warning",
                "message": f"clarification_source={summary.get('clarification_source', '')} is not human/source-backed evidence",
            }
        )
    if str(problem.get("problem_type", "")).upper() == "MULTI_SCENARIO":
        missing_objective = [
            str(scenario.get("name", index))
            for index, scenario in enumerate(problem.get("scenarios", []))
            if isinstance(scenario, dict)
            and scenario.get("expected_solver_status", "OPTIMAL") == "OPTIMAL"
            and ("objective" not in scenario or "metamorphic" not in scenario)
        ]
        if missing_objective:
            flags.append(
                {
                    "code": "multi_scenario_missing_objective_check",
                    "severity": "warning",
                    "message": f"scenario(s) need source-fidelity objective review: {', '.join(missing_objective)}",
                }
            )
    if any(token in searchable for token in ("tsp", "routing", "route", "distance", "travel")) and "constraint_relaxation" not in metamorphic:
        flags.append(
            {
                "code": "routing_or_tsp_cost_scaling_only",
                "severity": "warning",
                "message": "routing/TSP-like metadata relies on cost scaling without an additional structural invariant",
            }
        )
    if str(problem.get("problem_type", "")).upper() in {"QP", "MIQP"} or any(
        token in searchable for token in ("quadratic", "pairwise")
    ):
        flags.append(
            {
                "code": "complex_unit_scaling_requires_review",
                "severity": "warning",
                "message": "quadratic or pairwise objective terms require explicit unit/scaling fidelity review",
            }
        )
    return flags


def _capability_risk_flags(summary: dict[str, Any]) -> list[dict[str, str]]:
    status = summary.get("capability_status")
    if status not in CAPABILITY_BLOCKING_STATUSES:
        return []
    flags: list[dict[str, str]] = []
    for feature in _string_list(summary.get("capability_unsupported_features")):
        flags.append(
            {
                "code": "unsupported_stochastic_policy"
                if any(token in feature.lower() for token in ("stochastic", "dynamic", "policy", "robust"))
                else "unsupported_feature",
                "severity": "critical" if status == "unsupported" else "warning",
                "message": feature,
            }
        )
    for item in _string_list(summary.get("capability_missing_information")):
        flags.append(
            {
                "code": "dataset_missing_numeric_objective"
                if any(token in item.lower() for token in ("numeric", "cost", "coefficient", "objective"))
                else "missing_or_ambiguous_information",
                "severity": "warning",
                "message": item,
            }
        )
    if not flags:
        flags.append(
            {
                "code": f"capability_{status}",
                "severity": "critical" if status == "unsupported" else "warning",
                "message": str(summary.get("capability_review_note", "")),
            }
        )
    return flags


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            keys.append(child)
            keys.extend(_flatten_keys(item, child))
        return keys
    if isinstance(value, list):
        keys = []
        for index, item in enumerate(value):
            keys.extend(_flatten_keys(item, f"{prefix}[{index}]"))
        return keys
    return []


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _check_gurobi() -> tuple[str, bool, str]:
    try:
        import gurobipy as gp

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.dispose()
        env.dispose()
        return "Gurobi model creation", True, "available"
    except Exception as exc:
        return "Gurobi model creation", False, redact_text(exc)


def _check_or_ci_import() -> tuple[str, bool, str]:
    try:
        import or_ci.cli  # noqa: F401

        return "or-ci import", True, "available"
    except Exception as exc:
        return "or-ci import", False, redact_text(exc)


def _check_or_ci_cli() -> tuple[str, bool, str]:
    try:
        command, mode = or_ci_command()
        result = subprocess.run([*command, "--help"], capture_output=True, text=True, check=False)
    except OSError as exc:
        return "or-ci CLI", False, redact_text(exc)
    detail = redact_text((result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else "")
    return "or-ci CLI", result.returncode == 0, f"{mode}: {detail or f'returncode={result.returncode}'}"


def _check_codex_cli() -> tuple[str, bool, str]:
    return _check_command("Codex CLI", ["codex", "--help"])


def _check_codex_exec() -> tuple[str, bool, str]:
    return _check_command("Codex exec", ["codex", "exec", "--help"])


def _check_command(name: str, command: list[str]) -> tuple[str, bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return name, False, redact_text(exc)
    detail = (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else ""
    return name, result.returncode == 0, redact_text(detail or f"returncode={result.returncode}")


def _check_live_provider(model: str) -> tuple[str, bool, str]:
    try:
        response = query_llm(
            [
                {"role": "system", "content": "You are a provider health check. Reply briefly."},
                {"role": "user", "content": "Reply with exactly: ok"},
            ],
            model_name=model,
        )
    except Exception as exc:
        return "provider live request", False, redact_text(exc)
    return "provider live request", bool(response.strip()), "response received"


def _print_verification(verification: VerificationResult) -> None:
    if verification.stdout:
        print(verification.stdout, end="" if verification.stdout.endswith("\n") else "\n")
    if verification.stderr:
        print(verification.stderr, file=sys.stderr, end="" if verification.stderr.endswith("\n") else "\n")
    print(f"classification={verification.classification} status={verification.status}")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _portable_source_artifact(path: Path) -> str:
    return _relative(path.resolve(), repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
