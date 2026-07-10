from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from or_llm_agent.codex_agent import (
    CodexAgentOptions,
    CodexAgentPaths,
    build_agent_paths,
    build_agent_prompt,
    build_codex_command,
    run_codex_agent,
    validate_proof_anchor_ledger,
)


class CodexAgentModeTests(unittest.TestCase):
    def test_command_uses_neutral_work_dir_and_artifact_add_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = Path(temp_dir) / "or-ci" / "artifacts" / "pilot" / "run"
            paths = build_agent_paths(problem_id="BWOR-001", artifact_root=artifact_root)
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=3,
                timeout_seconds=900,
            )

            command = build_codex_command(paths, options)

            self.assertEqual(command[:5], ["codex", "-a", "never", "exec", "--json"])
            self.assertEqual(command[command.index("-C") + 1], str(paths.work_dir))
            self.assertEqual(command[command.index("--add-dir") + 1], str(paths.artifact_root))
            self.assertNotEqual(paths.work_dir, paths.artifact_root)
            self.assertFalse(paths.work_dir.is_relative_to(paths.artifact_root))
            self.assertIn("codex-work", str(paths.work_dir))
            self.assertEqual(
                paths.manifest_path,
                paths.artifact_root / "agent-status" / "BWOR-001.agent-run-manifest.json",
            )
            self.assertEqual(command[-1], "-")

    def test_command_passes_reasoning_effort_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-001")
            options = CodexAgentOptions(
                codex_model="gpt-5.5",
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=3,
                timeout_seconds=900,
                codex_reasoning_effort="xhigh",
            )

            command = build_codex_command(paths, options)

            self.assertEqual(command[command.index("-c") + 1], 'model_reasoning_effort="xhigh"')
            self.assertEqual(command[command.index("-m") + 1], "gpt-5.5")
            self.assertLess(command.index("-c"), command.index("-m"))

    def test_prompt_documents_fallback_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-001")
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=2,
                timeout_seconds=30,
            )

            prompt = build_agent_prompt(
                problem_id="BWOR-001",
                record={"en_question": "Build a tiny LP."},
                problem={"instance": {"sets": ["x"]}, "metamorphic": {}},
                problem_path=Path("/tmp/problem.json"),
                paths=paths,
                options=options,
                verify_command=["python", "-m", "or_ci.cli"],
            )

            self.assertIn(str(paths.artifact_root), prompt)
            self.assertIn(str(paths.work_dir), prompt)
            self.assertIn("submissions/BWOR-001.py", prompt)
            self.assertIn("reports/BWOR-001.json", prompt)
            self.assertIn("agent-status/BWOR-001.json", prompt)
            self.assertIn("sessions/BWOR-001/last-message.md", prompt)
            self.assertIn("The parent CLI will harvest those fallback files", prompt)
            self.assertIn("Start now. Complete the workflow without asking for confirmation.", prompt)
            self.assertIn("Do not run cleanup or removal commands", prompt)
            self.assertNotIn("Proof anchor ledger CSV", prompt)

    def test_prompt_documents_proof_anchor_contract_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-001")
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=2,
                timeout_seconds=30,
                source_anchor_contract="proof_carrying",
            )

            prompt = build_agent_prompt(
                problem_id="BWOR-001",
                record={"en_question": "Build a tiny LP."},
                problem={"instance": {"sets": ["x"]}, "metamorphic": {}},
                problem_path=Path("/tmp/problem.json"),
                paths=paths,
                options=options,
                verify_command=["python", "-m", "or_ci.cli"],
            )

            self.assertIn(f"Proof anchor ledger CSV: {paths.proof_anchor_ledger_path}", prompt)
            self.assertIn("row_id,variant,element_id,element_type", prompt)
            self.assertIn("Allowed `element_type` values are singular only", prompt)
            self.assertIn("Do not write plural category labels", prompt)
            self.assertIn("objective, variable, domain, constraint", prompt)
            self.assertIn("variant` must be `proof_carrying`", prompt)
            self.assertIn("proof-anchor-ledgers/BWOR-001.csv", prompt)
            self.assertIn("support_label=ambiguous", prompt)

    def test_prompt_does_not_leak_dataset_labels_or_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-001")
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=2,
                timeout_seconds=30,
            )

            prompt = build_agent_prompt(
                problem_id="BWOR-001",
                record={
                    "en_question": "Build a tiny LP.",
                    "answer": 123456789,
                    "problem_type": "LP",
                    "difficulty": "Medium",
                    "domain": "hidden_domain",
                    "solution_status": "optimal",
                },
                problem={"instance": {"sets": ["x"]}, "metamorphic": {}},
                problem_path=Path("/tmp/problem.json"),
                paths=paths,
                options=options,
                verify_command=["python", "-m", "or_ci.cli"],
            )

            self.assertIn("Build a tiny LP.", prompt)
            self.assertNotIn("123456789", prompt)
            self.assertNotIn("hidden_domain", prompt)
            self.assertNotIn("Medium", prompt)
            self.assertNotIn("solution_status", prompt)

    def test_prompt_documents_multi_scenario_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-032")
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=2,
                timeout_seconds=30,
            )

            prompt = build_agent_prompt(
                problem_id="BWOR-032",
                record={"en_question": "Check base infeasible and repair feasible scenarios."},
                problem={
                    "problem_type": "MULTI_SCENARIO",
                    "scenarios": [
                        {
                            "name": "base_infeasible",
                            "instance": {"case": "base"},
                            "expected_solver_status": "INFEASIBLE",
                        },
                        {
                            "name": "repair_feasible",
                            "instance": {"case": "repair", "objective": 10.0},
                            "expected_solver_status": "OPTIMAL",
                            "objective": {"value": 10.0},
                        },
                    ],
                },
                problem_path=Path("/tmp/problem.json"),
                paths=paths,
                options=options,
                verify_command=["python", "-m", "or_ci.cli"],
            )

            self.assertIn("OR-CI multi-scenario metadata", prompt)
            self.assertIn("base_infeasible", prompt)
            self.assertIn("calls `build_model(data)` once per scenario", prompt)

    def test_run_codex_agent_harvests_fallback_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-001")
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=3,
                timeout_seconds=900,
            )

            def fake_run(
                command: list[str],
                input: str | None = None,
                capture_output: bool = True,
                text: bool = True,
                check: bool = False,
                timeout: int | None = None,
            ) -> subprocess.CompletedProcess[str]:
                if command == ["codex", "--version"]:
                    return subprocess.CompletedProcess(command, 0, stdout="codex-cli test\n", stderr="")
                self.assertEqual(command[command.index("-C") + 1], str(paths.work_dir))
                self.assertIsNotNone(input)
                self.assertIn("Start now.", input)
                _write_fallback_artifacts(paths, "BWOR-001")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        '{"type":"thread.started","thread_id":"thread-test"}\n'
                        '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3,'
                        '"reasoning_output_tokens":2}}\n'
                    ),
                    stderr="",
                )

            with (
                patch("or_llm_agent.codex_agent.subprocess.run", fake_run),
                patch("or_llm_agent.codex_agent.shutil.which", return_value="/usr/local/bin/codex"),
            ):
                result = run_codex_agent(
                    problem_id="BWOR-001",
                    record={"en_question": "Build a tiny LP."},
                    problem={"instance": {"sets": ["x"]}, "metamorphic": {}},
                    problem_path=Path("/tmp/problem.json"),
                    paths=paths,
                    options=options,
                    verify_command=["python", "-m", "or_ci.cli"],
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.run_metadata["codex_thread_id"], "thread-test")
            self.assertEqual(result.run_metadata["codex_usage"]["input_tokens"], 12)
            self.assertEqual(result.run_metadata["codex_cli_version"], "codex-cli test")
            self.assertTrue(paths.submission_path.is_file())
            self.assertTrue(paths.report_path.is_file())
            self.assertTrue(paths.status_path.is_file())
            self.assertTrue(paths.last_message_path.is_file())
            self.assertTrue(paths.manifest_path.is_file())
            self.assertIn("def build_model", paths.submission_path.read_text(encoding="utf-8"))

            status = json.loads(paths.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["final_classification"], "SUCCESS")
            self.assertEqual(status["generation_mode"], "agent")
            self.assertEqual(status["work_dir"], str(paths.work_dir))
            self.assertEqual(status["agent_manifest"], str(paths.manifest_path))
            self.assertEqual(status["codex_run_metadata"]["codex_thread_id"], "thread-test")
            self.assertEqual(status["codex_usage"]["reasoning_output_tokens"], 2)

            raw = json.loads(paths.raw_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["returncode"], 0)
            self.assertEqual(raw["work_dir"], str(paths.work_dir))
            self.assertEqual(raw["manifest_path"], str(paths.manifest_path))
            self.assertEqual(raw["codex_run_metadata"]["codex_usage"]["output_tokens"], 3)
            self.assertEqual(
                set(raw["harvested_artifacts"]),
                {
                    str(paths.submission_path),
                    str(paths.report_path),
                    str(paths.status_path),
                    str(paths.last_message_path),
                },
            )

            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "swe_agent_run_manifest_v1")
            self.assertEqual(manifest["adapter"], "codex-cli")
            self.assertEqual(manifest["problem_id"], "BWOR-001")
            self.assertEqual(manifest["command"], result.command)
            self.assertEqual(manifest["codex_run_metadata"]["codex_cli_version"], "codex-cli test")
            self.assertEqual(manifest["options"]["codex_sandbox"], "workspace-write")
            self.assertIsNone(manifest["options"]["codex_reasoning_effort"])
            self.assertEqual(manifest["paths"]["work_dir"], str(paths.work_dir))
            self.assertEqual(manifest["outputs"]["submission_path"], str(paths.submission_path))
            self.assertEqual(manifest["result"]["returncode"], 0)
            self.assertFalse(manifest["result"]["timed_out"])
            self.assertEqual(
                set(manifest["harvested_artifacts"]),
                {
                    str(paths.submission_path),
                    str(paths.report_path),
                    str(paths.status_path),
                    str(paths.last_message_path),
                },
            )

    def test_run_codex_agent_harvests_and_validates_proof_anchor_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir), "BWOR-001")
            options = CodexAgentOptions(
                codex_model=None,
                codex_sandbox="workspace-write",
                codex_approval="never",
                max_repair_attempts=3,
                timeout_seconds=900,
                source_anchor_contract="proof_carrying",
            )

            def fake_run(
                command: list[str],
                input: str | None = None,
                capture_output: bool = True,
                text: bool = True,
                check: bool = False,
                timeout: int | None = None,
            ) -> subprocess.CompletedProcess[str]:
                if command == ["codex", "--version"]:
                    return subprocess.CompletedProcess(command, 0, stdout="codex-cli test\n", stderr="")
                self.assertIsNotNone(input)
                self.assertIn("Proof anchor ledger CSV", input)
                _write_fallback_artifacts(paths, "BWOR-001")
                _write_fallback_proof_anchor_ledger(paths, "BWOR-001")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch("or_llm_agent.codex_agent.subprocess.run", fake_run),
                patch("or_llm_agent.codex_agent.shutil.which", return_value="/usr/local/bin/codex"),
            ):
                result = run_codex_agent(
                    problem_id="BWOR-001",
                    record={"en_question": "Build a tiny LP."},
                    problem={"instance": {"sets": ["x"]}, "metamorphic": {}},
                    problem_path=Path("/tmp/problem.json"),
                    paths=paths,
                    options=options,
                    verify_command=["python", "-m", "or_ci.cli"],
                )

            self.assertEqual(result.proof_anchor_ledger_validation["status"], "valid")
            self.assertTrue(paths.proof_anchor_ledger_path.is_file())
            status = json.loads(paths.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["source_anchor_contract"], "proof_carrying")
            self.assertEqual(status["proof_anchor_ledger_schema_status"], "valid")
            self.assertEqual(status["proof_anchor_ledger"], str(paths.proof_anchor_ledger_path))
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["options"]["source_anchor_contract"], "proof_carrying")
            self.assertEqual(manifest["proof_anchor_ledger_schema_validation"]["row_count"], 1)

    def test_validate_proof_anchor_ledger_rejects_plural_element_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.csv"
            path.write_text(
                "\n".join(
                    [
                        "row_id,variant,element_id,element_type,generated_element_text,source_anchor_text,source_anchor_location,support_label,severity,materiality_reason,reviewer_id,review_prompt_path,review_trace_path,notes",
                        "BWOR-001,proof_carrying,c-1,constraints,Capacity <= limit,capacity limit,prompt:1,ambiguous,none,capacity constraint,codex,parent_agent_prompt,sessions/BWOR-001/codex-events.jsonl,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = validate_proof_anchor_ledger(path, required=True)

            self.assertEqual(result["status"], "invalid")
            self.assertIn("element_type='constraints'", "\n".join(result["errors"]))


def _paths(root: Path, problem_id: str) -> CodexAgentPaths:
    artifact_root = root / "artifact"
    return CodexAgentPaths(
        artifact_root=artifact_root,
        work_dir=root / "neutral-work" / problem_id,
        session_dir=artifact_root / "sessions" / problem_id,
        events_path=artifact_root / "sessions" / problem_id / "codex-events.jsonl",
        last_message_path=artifact_root / "sessions" / problem_id / "last-message.md",
        submission_path=artifact_root / "submissions" / f"{problem_id}.py",
        report_path=artifact_root / "reports" / f"{problem_id}.json",
        raw_path=artifact_root / "raw" / f"{problem_id}.txt",
        status_path=artifact_root / "agent-status" / f"{problem_id}.json",
        manifest_path=artifact_root / "agent-status" / f"{problem_id}.agent-run-manifest.json",
        proof_anchor_ledger_path=artifact_root / "proof-anchor-ledgers" / f"{problem_id}.csv",
    )


def _write_fallback_artifacts(paths: CodexAgentPaths, problem_id: str) -> None:
    fallback_submission = paths.work_dir / "submissions" / f"{problem_id}.py"
    fallback_report = paths.work_dir / "reports" / f"{problem_id}.json"
    fallback_status = paths.work_dir / "agent-status" / f"{problem_id}.json"
    fallback_message = paths.work_dir / "sessions" / problem_id / "last-message.md"

    fallback_submission.parent.mkdir(parents=True, exist_ok=True)
    fallback_report.parent.mkdir(parents=True, exist_ok=True)
    fallback_status.parent.mkdir(parents=True, exist_ok=True)
    fallback_message.parent.mkdir(parents=True, exist_ok=True)

    fallback_submission.write_text("def build_model(data: dict):\n    return None\n", encoding="utf-8")
    fallback_report.write_text(
        json.dumps({"classification": "SUCCESS", "status": "PASS"}) + "\n",
        encoding="utf-8",
    )
    fallback_status.write_text(
        json.dumps({"attempts": 1, "final_classification": "SUCCESS", "final_status": "PASS"}) + "\n",
        encoding="utf-8",
    )
    fallback_message.write_text("Generated and verified.\n", encoding="utf-8")


def _write_fallback_proof_anchor_ledger(paths: CodexAgentPaths, problem_id: str) -> None:
    fallback_ledger = paths.work_dir / "proof-anchor-ledgers" / f"{problem_id}.csv"
    fallback_ledger.parent.mkdir(parents=True, exist_ok=True)
    fallback_ledger.write_text(
        "\n".join(
            [
                "row_id,variant,element_id,element_type,generated_element_text,source_anchor_text,source_anchor_location,support_label,severity,materiality_reason,reviewer_id,review_prompt_path,review_trace_path,notes",
                "BWOR-001,proof_carrying,obj-1,objective,Maximize profit,profit statement,prompt:1,ambiguous,none,objective term,codex,parent_agent_prompt,sessions/BWOR-001/codex-events.jsonl,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
