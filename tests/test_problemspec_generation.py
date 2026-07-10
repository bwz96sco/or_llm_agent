from __future__ import annotations

import json
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from or_llm_agent.bwor import (
    BWOR_RUN_DATASET_KEYS,
    build_bwor_run_records,
    load_bwor_run_record,
)
from or_llm_agent.cli import (
    CapabilityAgentResult,
    CLIError,
    ClarificationAgentResult,
    FidelityReviewAgentResult,
    ProblemSpecAgentResult,
    load_clarification_answers,
    load_clarification_questions,
    main,
)
from or_llm_agent.or_ci import VerificationResult
from or_llm_agent.prompts import (
    build_clarification_question_prompt,
    build_clarified_problem_spec_prompt,
    build_or_ci_prompt,
    build_problem_metadata_template,
    build_problem_spec_prompt,
    build_statement_capability_prompt,
)


class ProblemSpecGenerationTests(unittest.TestCase):
    def test_classify_statement_writes_capability_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            capability_path = root / "spec" / "capability.json"
            raw_path = root / "raw" / "capability.txt"
            statement_path.write_text("Maximize profit subject to capacity.", encoding="utf-8")
            payload = _supported_capability("linear production planning")

            with patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, payload)):
                exit_code = main(
                    [
                        "classify-statement",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-001",
                        "--out",
                        str(capability_path),
                        "--raw",
                        str(raw_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(capability_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "supported")
            self.assertEqual(report["capability_generation_status"], "classified")
            self.assertEqual(report["problem_family"], "linear production planning")
            self.assertIn('"status": "supported"', raw_path.read_text(encoding="utf-8"))

    def test_capability_prompt_documents_blocking_semantics(self) -> None:
        prompt = build_statement_capability_prompt("CASE-001", "Goal programming with symbolic c_j.")

        self.assertIn("needs_human", prompt)
        self.assertIn("unsupported", prompt)
        self.assertIn("symbolic coefficients", prompt)
        self.assertIn("goal-programming", prompt)
        self.assertIn("QP/MIQP", prompt)
        self.assertIn("multi-scenario", prompt)
        self.assertIn("quadratic constraints", prompt)
        self.assertIn("Do not generate a ProblemSpec", prompt)

    def test_problem_spec_prompt_documents_constraint_relaxation_schema(self) -> None:
        prompt = build_problem_spec_prompt("CASE-001", "A capacity-constrained LP.")

        self.assertIn('"paths": [', prompt)
        self.assertIn('"instance.<resource_or_requirement_field>"', prompt)
        self.assertIn('"factor": 1.2', prompt)
        self.assertIn("do not use `path`, `amount`, or `direction`", prompt)
        self.assertIn("Preserve primitive statement quantities", prompt)
        self.assertIn("profit and transport cost", prompt)

    def test_problem_spec_prompt_documents_feature_family_schema(self) -> None:
        prompt = build_problem_spec_prompt("CASE-001", "A quadratic objective goal-programming case.")

        self.assertIn("`LP`, `MILP`, `QP`, `MIQP`, `MULTI_SCENARIO`", prompt)
        self.assertIn("metamorphic.goal_programming", prompt)
        self.assertIn('mode: "weighted"', prompt)
        self.assertIn('mode: "lexicographic"', prompt)
        self.assertIn("scenarios[]", prompt)
        self.assertIn("do not collapse those outcomes into one blended model", prompt)
        self.assertIn("expected_solver_status", prompt)
        self.assertIn("quadratic constraints remain unsupported", prompt)
        self.assertIn("Do not include dataset labels", prompt)

    def test_or_ci_prompt_uses_only_statement_from_dataset_record(self) -> None:
        prompt = build_or_ci_prompt(
            "CASE-001",
            {
                "id": "CASE-001",
                "en_question": "Visible statement text.",
                "answer": 999999,
                "problem_type": "NLP",
                "difficulty": "Hard",
                "domain": "secret_domain",
                "solution_status": "optimal",
            },
            _valid_problem(),
        )

        self.assertIn("Visible statement text.", prompt)
        self.assertNotIn("999999", prompt)
        self.assertNotIn("secret_domain", prompt)
        self.assertNotIn("Hard", prompt)
        self.assertNotIn("solution_status", prompt)

    def test_or_ci_prompt_documents_multi_scenario_metadata(self) -> None:
        prompt = build_or_ci_prompt(
            "CASE-SCENARIO",
            {"id": "CASE-SCENARIO", "en_question": "Check base and repair scenarios."},
            _multi_scenario_problem(),
        )

        self.assertIn("OR-CI multi-scenario metadata", prompt)
        self.assertIn("base_infeasible", prompt)
        self.assertIn("For `MULTI_SCENARIO`, OR-CI calls `build_model(data)` once per scenario", prompt)

    def test_problem_metadata_template_uses_requested_problem_id(self) -> None:
        template = json.loads(build_problem_metadata_template("CASE-123"))

        self.assertEqual(template["id"], "CASE-123")
        self.assertEqual(template["problem_type"], "LP")
        self.assertIn("cost_scaling", template["metamorphic"])
        relaxation = template["metamorphic"]["constraint_relaxation"]["relaxations"][0]
        self.assertIn("paths", relaxation)
        self.assertIn("factor", relaxation)

    def test_generate_accepts_explicit_problem_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            problem_path = root / "problem.json"
            statement_path = root / "problem.txt"
            submission_path = root / "submission.py"
            raw_path = root / "raw.txt"
            problem_path.write_text(json.dumps(_valid_problem()), encoding="utf-8")
            statement_path.write_text("Explicit statement context.", encoding="utf-8")

            def fake_query(messages, model_name):
                self.assertEqual(model_name, "mock-model")
                self.assertIn("Explicit statement context.", messages[1]["content"])
                return "```python\nimport gurobipy as gp\n\ndef build_model(data: dict):\n    return gp.Model()\n```"

            with patch("or_llm_agent.cli.query_llm", fake_query):
                exit_code = main(
                    [
                        "generate",
                        "--mode",
                        "api",
                        "--problem",
                        str(problem_path),
                        "--statement-file",
                        str(statement_path),
                        "--model",
                        "mock-model",
                        "--out",
                        str(submission_path),
                        "--raw",
                        str(raw_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("def build_model", submission_path.read_text(encoding="utf-8"))
            self.assertIn("def build_model", raw_path.read_text(encoding="utf-8"))

    def test_spec_writes_problem_raw_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            problem_path = root / "spec" / "problem.json"
            raw_path = root / "raw" / "spec.txt"
            status_path = root / "spec" / "status.json"
            statement_path.write_text("Minimize one production cost.", encoding="utf-8")

            with patch("or_llm_agent.cli._run_problem_spec_agent", return_value=_agent_result(root, _valid_problem())):
                exit_code = main(
                    [
                        "spec",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-001",
                        "--out",
                        str(problem_path),
                        "--raw",
                        str(raw_path),
                        "--status",
                        str(status_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(problem_path.read_text(encoding="utf-8"))["id"], "CASE-001")
            self.assertIn('"id": "CASE-001"', raw_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["spec_generation_status"], "generated")
            self.assertEqual(status["spec_validation_status"], "passed")
            self.assertEqual(status["validation_returncode"], 0)
            self.assertEqual(status["spec_attempt_count"], 1)
            self.assertEqual(status["spec_repair_status"], "not_needed")

    def test_spec_repairs_metadata_after_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            problem_path = root / "spec" / "problem.json"
            raw_path = root / "raw" / "spec.txt"
            status_path = root / "spec" / "status.json"
            statement_path.write_text("Maximize profit subject to capacity.", encoding="utf-8")
            calls = []

            def fake_run(**kwargs):
                calls.append(kwargs)
                payload = _invalid_constraint_relaxation_problem() if len(calls) == 1 else _valid_problem()
                return _agent_result(root, payload)

            with patch("or_llm_agent.cli._run_problem_spec_agent", fake_run):
                exit_code = main(
                    [
                        "spec",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-001",
                        "--out",
                        str(problem_path),
                        "--raw",
                        str(raw_path),
                        "--status",
                        str(status_path),
                        "--max-repair-attempts",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["attempt"], 1)
            self.assertEqual(calls[1]["attempt"], 2)
            self.assertIn("paths must be a non-empty list", calls[1]["repair_error"])
            self.assertEqual(json.loads(problem_path.read_text(encoding="utf-8"))["id"], "CASE-001")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["spec_validation_status"], "passed")
            self.assertEqual(status["spec_attempt_count"], 2)
            self.assertEqual(status["spec_repair_status"], "repaired")
            self.assertEqual(status["spec_attempts"][0]["spec_validation_status"], "failed")
            self.assertEqual(status["spec_attempts"][1]["spec_validation_status"], "passed")
            self.assertTrue((root / "raw" / "spec-attempt-1.txt").exists())
            self.assertTrue((root / "raw" / "spec-attempt-2.txt").exists())

    def test_solve_stops_before_model_generation_when_spec_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            statement_path.write_text("This statement will yield an invalid spec.", encoding="utf-8")

            invalid_problem = {
                "id": "CASE-FAIL",
                "problem_type": "LP",
                "metamorphic": {
                    "cost_scaling": {
                        "coefficient_paths": ["instance.price"],
                        "factors": [2.0],
                    }
                },
            }

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, _supported_capability())),
                patch("or_llm_agent.cli._run_problem_spec_agent", return_value=_agent_result(root, invalid_problem)),
                patch("or_llm_agent.cli.generate_agent_submission") as generate_agent_submission,
            ):
                exit_code = main(
                    [
                        "solve",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-FAIL",
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            generate_agent_submission.assert_not_called()
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["capability_status"], "supported")
            self.assertEqual(summary["spec_generation_status"], "generated")
            self.assertEqual(summary["spec_validation_status"], "failed")
            self.assertEqual(summary["spec_repair_status"], "failed")
            self.assertEqual(summary["spec_fidelity_status"], "not_reviewed")
            self.assertEqual(summary["spec_fidelity_review"], "spec/fidelity-review.md")
            self.assertEqual(summary["spec_fidelity_report"], "spec/fidelity-review.json")
            self.assertEqual(summary["spec_fidelity_gate_status"], "blocked_spec_invalid")
            self.assertEqual(summary["model_generation_status"], "skipped")
            self.assertFalse((artifact_dir / "submissions" / "CASE-FAIL.py").exists())
            review = (artifact_dir / "spec" / "fidelity-review.md").read_text(encoding="utf-8")
            self.assertIn("Fidelity status: `not_reviewed`", review)
            self.assertIn("Fidelity gate: `blocked_spec_invalid`", review)
            self.assertIn("OR-CI result is interpreted", review)
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["gate_status"], "blocked_spec_invalid")
            self.assertEqual(report["automatic_checks"][0]["status"], "FAIL")

    def test_solve_stops_before_problem_spec_when_capability_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            statement_path.write_text("Minimize cost and cholesterol as a goal programming model.", encoding="utf-8")
            capability = {
                "status": "unsupported",
                "problem_family": "goal programming",
                "supported_features": [],
                "unsupported_features": ["multi-objective goal-programming semantics"],
                "missing_information": [],
                "recommended_next_action": "extend_schema_or_verifier",
                "confidence": 0.94,
                "review_note": "Current ProblemSpec lacks goal-programming support.",
            }

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, capability)),
                patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent,
                patch("or_llm_agent.cli.generate_agent_submission") as generate_agent_submission,
            ):
                exit_code = main(
                    [
                        "solve",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-GOAL",
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            generate_agent_submission.assert_not_called()
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["capability_status"], "unsupported")
            self.assertEqual(summary["problem_family"], "goal programming")
            self.assertEqual(summary["spec_generation_status"], "skipped")
            self.assertEqual(summary["spec_validation_status"], "skipped")
            self.assertEqual(summary["model_generation_status"], "skipped")
            self.assertEqual(summary["classification"], "blocked_capability")
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["gate_status"], "blocked_capability")
            self.assertIn("unsupported_feature", [flag["code"] for flag in report["risk_flags"]])
            self.assertTrue((artifact_dir / "spec" / "capability.json").exists())

    def test_solve_stops_before_problem_spec_when_capability_needs_human(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            statement_path.write_text("Use trucks with symbolic costs c_j.", encoding="utf-8")
            capability = {
                "status": "needs_human",
                "problem_family": "assignment",
                "supported_features": ["linear assignment structure"],
                "unsupported_features": [],
                "missing_information": ["numeric truck costs c_j are not provided"],
                "recommended_next_action": "ask_human",
                "confidence": 0.9,
                "review_note": "Symbolic objective coefficients require clarification.",
            }

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, capability)),
                patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent,
            ):
                exit_code = main(
                    [
                        "solve",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-SYMBOLIC",
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["capability_status"], "needs_human")
            self.assertEqual(summary["capability_missing_information"], ["numeric truck costs c_j are not provided"])
            self.assertEqual(summary["classification"], "blocked_capability")
            self.assertEqual(summary["terminal_state"], "excluded_with_reason")
            self.assertEqual(summary["accounting_state"], "excluded_with_reason")
            self.assertEqual(summary["exclusion_reason"], "capability_needs_human")

    def test_clarification_question_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "questions.json"
            path.write_text(json.dumps(_clarification_questions()), encoding="utf-8")

            artifact = load_clarification_questions(path)

            self.assertEqual(artifact["problem_id"], "CASE-SYMBOLIC")
            self.assertEqual(artifact["blocking_status"], "needs_human")
            self.assertEqual(artifact["questions"][0]["issue_type"], "missing_numeric_data")

    def test_clarification_question_schema_rejects_unknown_issue_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "questions.json"
            payload = _clarification_questions()
            payload["questions"][0]["issue_type"] = "guess_anyway"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(CLIError, "unsupported clarification issue_type"):
                load_clarification_questions(path)

    def test_clarification_question_schema_rejects_malformed_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = _clarification_questions()
            cases = [
                ("empty-questions.json", {**base, "questions": []}, "non-empty questions list"),
                ("bad-status.json", {**base, "blocking_status": "supported"}, "blocking_status must be needs_human"),
                (
                    "bad-required.json",
                    {**base, "questions": [{**base["questions"][0], "required": "true"}]},
                    "required must be boolean",
                ),
            ]

            for filename, payload, pattern in cases:
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaisesRegex(CLIError, pattern):
                        load_clarification_questions(path)

    def test_clarification_loaders_report_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaisesRegex(CLIError, "run prepare-clarification first"):
                load_clarification_questions(root / "missing-questions.json")

            with self.assertRaisesRegex(CLIError, "clarification answer artifact does not exist"):
                load_clarification_answers(root / "missing-answers.json", questions=_clarification_questions())

    def test_clarification_answer_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            questions_path = root / "questions.json"
            answers_path = root / "answers.json"
            questions_path.write_text(json.dumps(_clarification_questions()), encoding="utf-8")
            answers_path.write_text(json.dumps(_clarification_answers(reviewer="")), encoding="utf-8")

            questions = load_clarification_questions(questions_path)
            answers = load_clarification_answers(answers_path, questions=questions, reviewer="unit-test")

            self.assertEqual(answers["resolution_status"], "answered")
            self.assertEqual(answers["answers"][0]["reviewer"], "unit-test")
            self.assertEqual(answers["answers"][0]["source"], "manual_review")

    def test_clarification_answer_schema_marks_missing_required_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            questions_path = root / "questions.json"
            answers_path = root / "answers.json"
            questions_path.write_text(json.dumps(_clarification_questions()), encoding="utf-8")
            answers_path.write_text(
                json.dumps({"problem_id": "CASE-SYMBOLIC", "answers": [], "resolution_status": "answered"}),
                encoding="utf-8",
            )

            questions = load_clarification_questions(questions_path)
            answers = load_clarification_answers(answers_path, questions=questions, reviewer="unit-test")

            self.assertEqual(answers["resolution_status"], "partially_answered")

    def test_clarification_answer_schema_rejects_malformed_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            questions_path = root / "questions.json"
            questions_path.write_text(json.dumps(_clarification_questions()), encoding="utf-8")
            questions = load_clarification_questions(questions_path)
            valid_answer = _clarification_answers()["answers"][0]
            cases = [
                (
                    "unknown-question.json",
                    {"problem_id": "CASE-SYMBOLIC", "answers": [{**valid_answer, "question_id": "q-missing"}]},
                    "unknown question_id",
                ),
                (
                    "missing-answer.json",
                    {"problem_id": "CASE-SYMBOLIC", "answers": [{"question_id": "q1", "reviewer": "unit-test"}]},
                    "is empty",
                ),
                (
                    "bad-status.json",
                    {"problem_id": "CASE-SYMBOLIC", "answers": [], "resolution_status": "guessed"},
                    "unsupported clarification resolution_status",
                ),
                (
                    "wrong-problem.json",
                    {"problem_id": "CASE-OTHER", "answers": [valid_answer], "resolution_status": "answered"},
                    "does not match questions problem_id",
                ),
            ]

            for filename, payload, pattern in cases:
                with self.subTest(filename=filename):
                    answers_path = root / filename
                    answers_path.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaisesRegex(CLIError, pattern):
                        load_clarification_answers(answers_path, questions=questions)

    def test_clarification_prompts_document_required_contracts(self) -> None:
        question_prompt = build_clarification_question_prompt(
            "CASE-SYMBOLIC",
            "Use trucks with symbolic costs c_j.",
            {"status": "needs_human", "missing_information": ["numeric truck costs c_j are not provided"]},
        )
        spec_prompt = build_clarified_problem_spec_prompt(
            "CASE-SYMBOLIC",
            "Use trucks with symbolic costs c_j.",
            {"answers": [{"question_id": "q1", "answer": "costs are 1, 2, 3"}]},
        )

        self.assertIn("missing_numeric_data", question_prompt)
        self.assertIn("Do not solve the problem or invent the answer", question_prompt)
        self.assertIn("Approved clarification context", spec_prompt)
        self.assertIn("source_context", spec_prompt)
        self.assertIn("Do not use unapproved assumptions", spec_prompt)

    def test_prepare_clarification_writes_question_artifact_for_needs_human(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            _write_needs_human_case(root, artifact_dir)
            out_path = root / "questions.json"
            payload = _clarification_questions()

            with (
                patch("or_llm_agent.cli.repo_root", return_value=root.resolve()),
                patch(
                    "or_llm_agent.cli._run_clarification_question_agent",
                    return_value=_clarification_agent_result(root, payload),
                ),
            ):
                exit_code = main(
                    [
                        "prepare-clarification",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written["questions"][0]["id"], "q1")
            self.assertEqual(written["source_artifact"], "run")
            self.assertTrue((artifact_dir / "clarification" / "questions.json").exists())
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["classification"], "blocked_capability")

    def test_prepare_clarification_refuses_non_needs_human_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            out_path = root / "questions.json"
            _write_blocked_case(
                root,
                artifact_dir,
                problem_id="CASE-UNSUPPORTED",
                capability={
                    "status": "unsupported",
                    "problem_family": "goal programming",
                    "supported_features": [],
                    "unsupported_features": ["goal programming"],
                    "missing_information": [],
                    "recommended_next_action": "extend_schema_or_verifier",
                    "confidence": 0.9,
                    "review_note": "Unsupported.",
                },
            )

            with patch("or_llm_agent.cli._run_clarification_question_agent") as run_agent:
                exit_code = main(
                    [
                        "prepare-clarification",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_agent.assert_not_called()
            self.assertFalse(out_path.exists())

    def test_prepare_clarification_golden_bwor_ambiguity_classes(self) -> None:
        cases = [
            (
                "BWOR-013",
                "At least 10% of 1000 thousand yuan, i.e. 1 million yuan, must be reserved.",
                "The statement says 10% of 1000 thousand yuan but also says 1 million yuan.",
                "unit_conflict",
            ),
            (
                "BWOR-020",
                "The cost of using truck j is c_j.",
                "numeric truck costs c_j are not provided",
                "missing_numeric_data",
            ),
            (
                "BWOR-027",
                "Find the shortest route that visits each city exactly once.",
                "Clarify whether the route must return to the starting city and which distance convention to use.",
                "timing_convention",
            ),
            (
                "BWOR-046",
                "Factory supply is 270 tons and regional demand totals 290 tons.",
                "capacity-demand conflict makes the transportation problem infeasible unless unmet demand or extra supply is clarified",
                "data_conflict",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for problem_id, statement, missing_info, expected_issue_type in cases:
                artifact_dir = root / problem_id
                _write_blocked_case(
                    root,
                    artifact_dir,
                    problem_id=problem_id,
                    capability={
                        "status": "needs_human",
                        "problem_family": "clarification golden",
                        "supported_features": ["linear structure after clarification"],
                        "unsupported_features": [],
                        "missing_information": [missing_info],
                        "recommended_next_action": "ask_human",
                        "confidence": 0.9,
                        "review_note": missing_info,
                    },
                    statement=statement,
                )
                payload = {
                    "problem_id": problem_id,
                    "source_artifact": str(artifact_dir),
                    "blocking_status": "needs_human",
                    "questions": [
                        {
                            "id": "q1",
                            "issue_type": expected_issue_type,
                            "prompt": "What human clarification resolves this blocking issue?",
                            "source_evidence": missing_info,
                            "allowed_answer_type": "free_text",
                            "options": [],
                            "required": True,
                        }
                    ],
                }
                result = _clarification_agent_result(root, payload)
                with patch("or_llm_agent.cli._run_clarification_question_agent", return_value=result):
                    exit_code = main(
                        [
                            "prepare-clarification",
                            "--artifact-dir",
                            str(artifact_dir),
                            "--out",
                            str(root / f"{problem_id}-questions.json"),
                        ]
                    )

                self.assertEqual(exit_code, 0, problem_id)
                questions = json.loads((artifact_dir / "clarification" / "questions.json").read_text(encoding="utf-8"))
                self.assertEqual(questions["problem_id"], problem_id)
                self.assertEqual(questions["questions"][0]["issue_type"], expected_issue_type)
                self.assertTrue(questions["questions"][0]["required"])

    def test_prepare_clarification_fails_loud_on_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            _write_needs_human_case(root, artifact_dir)
            out_path = root / "questions.json"
            result = ClarificationAgentResult(
                raw_text="not json",
                returncode=0,
                timed_out=False,
                events_path=root / "clarification-events.jsonl",
                last_message_path=root / "clarification-last.md",
                stderr="",
            )

            with patch("or_llm_agent.cli._run_clarification_question_agent", return_value=result):
                exit_code = main(
                    [
                        "prepare-clarification",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse(out_path.exists())
            self.assertFalse((artifact_dir / "clarification" / "questions.json").exists())
            status = json.loads((artifact_dir / "clarification" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "agent_no_json")
            self.assertEqual(status["agent_returncode"], 0)
            self.assertFalse(status["agent_timed_out"])
            expected_raw_path = artifact_dir.resolve() / "raw" / "clarification-questions.txt"
            self.assertEqual(status["raw_response"], str(expected_raw_path))
            self.assertTrue(Path(status["raw_response"]).exists())
            self.assertEqual(expected_raw_path.read_text(encoding="utf-8"), "not json\n")

    def test_prepare_clarification_fails_loud_on_agent_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            _write_needs_human_case(root, artifact_dir)
            out_path = root / "questions.json"
            result = ClarificationAgentResult(
                raw_text=json.dumps(_clarification_questions()),
                returncode=1,
                timed_out=False,
                events_path=root / "clarification-events.jsonl",
                last_message_path=root / "clarification-last.md",
                stderr="planner failed",
            )

            with patch("or_llm_agent.cli._run_clarification_question_agent", return_value=result):
                exit_code = main(
                    [
                        "prepare-clarification",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--out",
                        str(out_path),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse(out_path.exists())
            self.assertFalse((artifact_dir / "clarification" / "questions.json").exists())
            status = json.loads((artifact_dir / "clarification" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "agent_failed")
            self.assertEqual(status["agent_returncode"], 1)
            self.assertFalse(status["agent_timed_out"])
            self.assertEqual(status["agent_stderr"], "planner failed")
            expected_raw_path = artifact_dir.resolve() / "raw" / "clarification-questions.txt"
            self.assertEqual(status["raw_response"], str(expected_raw_path))
            self.assertTrue(Path(status["raw_response"]).exists())

    def test_answer_clarification_blocks_when_required_unanswered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            _write_needs_human_case(root, artifact_dir)
            (artifact_dir / "clarification").mkdir(parents=True)
            (artifact_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            answers_path = root / "answers.json"
            answers_path.write_text(
                json.dumps({"problem_id": "CASE-SYMBOLIC", "answers": [], "resolution_status": "answered"}),
                encoding="utf-8",
            )
            original_answers = answers_path.read_text(encoding="utf-8")

            exit_code = main(
                [
                    "answer-clarification",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--answers",
                    str(answers_path),
                    "--reviewer",
                    "unit-test",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(answers_path.read_text(encoding="utf-8"), original_answers)
            answers = json.loads((artifact_dir / "clarification" / "answers.json").read_text(encoding="utf-8"))
            self.assertEqual(answers["resolution_status"], "partially_answered")

    def test_solve_clarified_refuses_unanswered_required_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            resolution_dir = root / "resolved"
            _write_needs_human_case(root, artifact_dir)
            (artifact_dir / "clarification").mkdir(parents=True)
            (artifact_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            answers_path = root / "answers.json"
            answers_path.write_text(
                json.dumps({"problem_id": "CASE-SYMBOLIC", "answers": [], "resolution_status": "answered"}),
                encoding="utf-8",
            )

            with patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent:
                exit_code = main(
                    [
                        "solve-clarified",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--clarification",
                        str(answers_path),
                        "--resolution-dir",
                        str(resolution_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            self.assertFalse((resolution_dir / "summary.json").exists())

    def test_solve_clarified_refuses_rejected_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            resolution_dir = root / "resolved"
            _write_needs_human_case(root, artifact_dir)
            (artifact_dir / "clarification").mkdir(parents=True)
            (artifact_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            answers = _clarification_answers()
            answers["resolution_status"] = "rejected"
            answers_path = root / "answers.json"
            answers_path.write_text(json.dumps(answers), encoding="utf-8")

            with patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent:
                exit_code = main(
                    [
                        "solve-clarified",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--clarification",
                        str(answers_path),
                        "--resolution-dir",
                        str(resolution_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            self.assertFalse((resolution_dir / "summary.json").exists())

    def test_solve_clarified_refuses_unresolved_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            resolution_dir = root / "resolved"
            _write_needs_human_case(root, artifact_dir)
            (artifact_dir / "clarification").mkdir(parents=True)
            (artifact_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            answers = _clarification_answers()
            answers["resolution_status"] = "unresolved"
            answers_path = root / "answers.json"
            answers_path.write_text(json.dumps(answers), encoding="utf-8")

            with patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent:
                exit_code = main(
                    [
                        "solve-clarified",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--clarification",
                        str(answers_path),
                        "--resolution-dir",
                        str(resolution_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            self.assertFalse((resolution_dir / "summary.json").exists())

    def test_solve_clarified_refuses_source_artifact_as_resolution_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            _write_needs_human_case(root, artifact_dir)
            original_summary = (artifact_dir / "summary.json").read_text(encoding="utf-8")
            original_capability = (artifact_dir / "spec" / "capability.json").read_text(encoding="utf-8")
            original_raw = (artifact_dir / "raw" / "capability.txt").read_text(encoding="utf-8")
            (artifact_dir / "clarification").mkdir(parents=True)
            (artifact_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            answers_path = root / "answers.json"
            answers_path.write_text(json.dumps(_clarification_answers()), encoding="utf-8")

            with patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent:
                exit_code = main(
                    [
                        "solve-clarified",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--clarification",
                        str(answers_path),
                        "--resolution-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            self.assertEqual((artifact_dir / "summary.json").read_text(encoding="utf-8"), original_summary)
            self.assertEqual((artifact_dir / "spec" / "capability.json").read_text(encoding="utf-8"), original_capability)
            self.assertEqual((artifact_dir / "raw" / "capability.txt").read_text(encoding="utf-8"), original_raw)

    def test_solve_clarified_uses_answers_and_does_not_overwrite_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "run"
            resolution_dir = root / "resolved"
            _write_needs_human_case(root, artifact_dir)
            original_summary = (artifact_dir / "summary.json").read_text(encoding="utf-8")
            original_capability = (artifact_dir / "spec" / "capability.json").read_text(encoding="utf-8")
            original_raw = (artifact_dir / "raw" / "capability.txt").read_text(encoding="utf-8")
            (artifact_dir / "clarification").mkdir(parents=True)
            (artifact_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            answers_path = root / "answers.json"
            answers_path.write_text(json.dumps(_clarification_answers()), encoding="utf-8")
            spec_calls = []

            def fake_spec_agent(**kwargs):
                spec_calls.append(kwargs)
                return _agent_result(root, _valid_problem(kwargs["problem_id"]))

            verification = VerificationResult(
                returncode=0,
                stdout="",
                stderr="",
                report={"classification": "SUCCESS", "status": "PASS", "checks": []},
            )

            with (
                patch("or_llm_agent.cli._run_problem_spec_agent", fake_spec_agent),
                patch("or_llm_agent.cli.generate_agent_submission", _fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", return_value=verification),
            ):
                exit_code = main(
                    [
                        "solve-clarified",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--clarification",
                        str(answers_path),
                        "--resolution-dir",
                        str(resolution_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual((artifact_dir / "summary.json").read_text(encoding="utf-8"), original_summary)
            self.assertEqual((artifact_dir / "spec" / "capability.json").read_text(encoding="utf-8"), original_capability)
            self.assertEqual((artifact_dir / "raw" / "capability.txt").read_text(encoding="utf-8"), original_raw)
            source_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(source_summary["classification"], "blocked_capability")
            clarified_summary = json.loads((resolution_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(clarified_summary["clarified_from"], str(artifact_dir.resolve()))
            self.assertEqual(clarified_summary["clarification_status"], "answered")
            self.assertEqual(clarified_summary["clarification_question_count"], 1)
            self.assertEqual(clarified_summary["clarification_answer_count"], 1)
            self.assertEqual(clarified_summary["clarification_gate_status"], "passed")
            problem = json.loads((resolution_dir / "spec" / "problem.json").read_text(encoding="utf-8"))
            self.assertTrue(problem["source_context"]["clarified"])
            self.assertIn("clarification_context", spec_calls[0])
            review = json.loads((resolution_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(review["clarification"]["clarification_gate_status"], "passed")
            model_raw = (resolution_dir / "raw" / "CASE-SYMBOLIC.txt").read_text(encoding="utf-8")
            self.assertEqual(model_raw, "generated model")

    def test_prepare_clarification_batch_defaults_to_needs_human_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch_dir = root / "batch"
            _write_needs_human_case(root, batch_dir / "CASE-SYMBOLIC", problem_id="CASE-SYMBOLIC")
            _write_blocked_case(
                root,
                batch_dir / "CASE-UNSUPPORTED",
                problem_id="CASE-UNSUPPORTED",
                capability={
                    "status": "unsupported",
                    "problem_family": "goal programming",
                    "supported_features": [],
                    "unsupported_features": ["goal programming"],
                    "missing_information": [],
                    "recommended_next_action": "extend_schema_or_verifier",
                    "confidence": 0.9,
                    "review_note": "Unsupported.",
                },
            )
            rows = [
                {
                    "problem_id": "CASE-SYMBOLIC",
                    "capability_status": "needs_human",
                    "classification": "blocked_capability",
                },
                {
                    "problem_id": "CASE-UNSUPPORTED",
                    "capability_status": "unsupported",
                    "classification": "blocked_capability",
                },
            ]
            (batch_dir / "summary.json").write_text(json.dumps({"summary": {"total": 2}, "rows": rows}), encoding="utf-8")
            calls = []

            def fake_prepare(*, artifact_dir, out_path, args):
                calls.append(artifact_dir.name)
                payload = _clarification_questions(problem_id=artifact_dir.name)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch("or_llm_agent.cli.prepare_clarification_artifact", fake_prepare):
                exit_code = main(["prepare-clarification-batch", "--artifact-dir", str(batch_dir)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["CASE-SYMBOLIC"])
            self.assertTrue((batch_dir / "clarification-report.md").exists())
            summary = json.loads((batch_dir / "clarification-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["attempted_needs_human_count"], 1)

    def test_prepare_clarification_batch_records_failures_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch_dir = root / "batch"
            _write_needs_human_case(root, batch_dir / "CASE-OK", problem_id="CASE-OK")
            _write_needs_human_case(root, batch_dir / "CASE-BAD", problem_id="CASE-BAD")
            rows = [
                {"problem_id": "CASE-OK", "capability_status": "needs_human", "classification": "blocked_capability"},
                {"problem_id": "CASE-BAD", "capability_status": "needs_human", "classification": "blocked_capability"},
            ]
            (batch_dir / "summary.json").write_text(json.dumps({"summary": {"total": 2}, "rows": rows}), encoding="utf-8")
            calls = []

            def fake_prepare(*, artifact_dir, out_path, args):
                calls.append(artifact_dir.name)
                if artifact_dir.name == "CASE-BAD":
                    raise CLIError("planner failed for CASE-BAD")
                payload = _clarification_questions(problem_id=artifact_dir.name)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch("or_llm_agent.cli.prepare_clarification_artifact", fake_prepare):
                exit_code = main(["prepare-clarification-batch", "--artifact-dir", str(batch_dir)])

            self.assertEqual(exit_code, 1)
            self.assertEqual(calls, ["CASE-OK", "CASE-BAD"])
            self.assertTrue((batch_dir / "clarification-summary.json").exists())
            self.assertTrue((batch_dir / "clarification-report.md").exists())
            summary = json.loads((batch_dir / "clarification-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["attempted_needs_human_count"], 2)
            statuses = summary["summary"]["clarification_statuses"]
            self.assertEqual(statuses.get("questions_prepared"), 1)
            self.assertEqual(statuses.get("prepare_failed"), 1)
            self.assertGreaterEqual(summary["summary"]["unresolved_case_count"], 1)
            rows_by_id = {row["problem_id"]: row for row in summary["rows"]}
            self.assertEqual(rows_by_id["CASE-OK"]["clarification_status"], "questions_prepared")
            self.assertEqual(rows_by_id["CASE-BAD"]["clarification_status"], "prepare_failed")
            self.assertEqual(rows_by_id["CASE-BAD"]["clarification_gate_status"], "blocked")
            self.assertIn("planner failed for CASE-BAD", rows_by_id["CASE-BAD"]["error"])

    def test_solve_clarified_batch_defaults_to_needs_human_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch_dir = root / "batch"
            clarifications_dir = root / "answers"
            clarifications_dir.mkdir()
            _write_needs_human_case(root, batch_dir / "CASE-SYMBOLIC", problem_id="CASE-SYMBOLIC")
            _write_blocked_case(
                root,
                batch_dir / "CASE-UNSUPPORTED",
                problem_id="CASE-UNSUPPORTED",
                capability={
                    "status": "unsupported",
                    "problem_family": "goal programming",
                    "supported_features": [],
                    "unsupported_features": ["goal programming"],
                    "missing_information": [],
                    "recommended_next_action": "extend_schema_or_verifier",
                    "confidence": 0.9,
                    "review_note": "Unsupported.",
                },
            )
            rows = [
                {
                    "problem_id": "CASE-SYMBOLIC",
                    "capability_status": "needs_human",
                    "classification": "blocked_capability",
                },
                {
                    "problem_id": "CASE-UNSUPPORTED",
                    "capability_status": "unsupported",
                    "classification": "blocked_capability",
                },
            ]
            (batch_dir / "summary.json").write_text(json.dumps({"summary": {"total": 2}, "rows": rows}), encoding="utf-8")
            (clarifications_dir / "CASE-SYMBOLIC.json").write_text(json.dumps(_clarification_answers()), encoding="utf-8")
            calls = []

            def fake_solve(*, artifact_dir, clarification_path, resolution_dir, args):
                calls.append(artifact_dir.name)
                return {
                    "problem_id": artifact_dir.name,
                    "source_artifact": str(artifact_dir),
                    "resolution_artifact": str(resolution_dir),
                    "clarified_from": str(artifact_dir),
                    "clarification_status": "answered",
                    "clarification_question_count": 1,
                    "clarification_answer_count": 1,
                    "clarification_source": "manual_review",
                    "clarification_gate_status": "passed",
                    "classification": "SUCCESS",
                    "verification_status": "PASS",
                    "spec_fidelity_status": "not_reviewed",
                }

            with patch("or_llm_agent.cli.solve_clarified_artifact", fake_solve):
                exit_code = main(
                    [
                        "solve-clarified-batch",
                        "--artifact-dir",
                        str(batch_dir),
                        "--clarifications-dir",
                        str(clarifications_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["CASE-SYMBOLIC"])
            summary = json.loads((batch_dir / "clarification-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["clarified_supported_case_count"], 1)

    def test_solve_clarified_batch_runs_cases_concurrently_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch_dir = root / "batch"
            clarifications_dir = root / "answers"
            clarifications_dir.mkdir()
            problem_ids = ["CASE-001", "CASE-002"]
            for problem_id in problem_ids:
                _write_needs_human_case(root, batch_dir / problem_id, problem_id=problem_id)
                (clarifications_dir / f"{problem_id}.json").write_text(
                    json.dumps(_clarification_answers(problem_id=problem_id)),
                    encoding="utf-8",
                )
            rows = [
                {
                    "problem_id": problem_id,
                    "capability_status": "needs_human",
                    "classification": "blocked_capability",
                }
                for problem_id in problem_ids
            ]
            (batch_dir / "summary.json").write_text(json.dumps({"summary": {"total": 2}, "rows": rows}), encoding="utf-8")
            barrier = threading.Barrier(2, timeout=5)
            active_workers = 0
            max_active_workers = 0
            lock = threading.Lock()

            def fake_solve(*, artifact_dir, clarification_path, resolution_dir, args):
                nonlocal active_workers, max_active_workers
                with lock:
                    active_workers += 1
                    max_active_workers = max(max_active_workers, active_workers)
                try:
                    barrier.wait()
                    return {
                        "problem_id": artifact_dir.name,
                        "source_artifact": str(artifact_dir),
                        "resolution_artifact": str(resolution_dir),
                        "clarified_from": str(artifact_dir),
                        "clarification_status": "answered",
                        "clarification_question_count": 1,
                        "clarification_answer_count": 1,
                        "clarification_source": "manual_review",
                        "clarification_gate_status": "passed",
                        "classification": "SUCCESS",
                        "verification_status": "PASS",
                        "spec_fidelity_status": "not_reviewed",
                    }
                finally:
                    with lock:
                        active_workers -= 1

            with patch("or_llm_agent.cli.solve_clarified_artifact", fake_solve):
                exit_code = main(
                    [
                        "solve-clarified-batch",
                        "--artifact-dir",
                        str(batch_dir),
                        "--clarifications-dir",
                        str(clarifications_dir),
                        "--ids",
                        *problem_ids,
                        "--agent-concurrency",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(max_active_workers, 2)
            summary = json.loads((batch_dir / "clarification-summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["problem_id"] for row in summary["rows"]], problem_ids)
            self.assertEqual(summary["summary"]["clarified_supported_case_count"], 2)

    def test_solve_clarified_batch_records_unresolved_row_when_answer_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            batch_dir = root / "batch"
            clarifications_dir = root / "answers"
            clarifications_dir.mkdir()
            case_dir = batch_dir / "CASE-SYMBOLIC"
            _write_needs_human_case(root, case_dir, problem_id="CASE-SYMBOLIC")
            (case_dir / "clarification").mkdir(parents=True)
            (case_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            rows = [
                {
                    "problem_id": "CASE-SYMBOLIC",
                    "capability_status": "needs_human",
                    "classification": "blocked_capability",
                }
            ]
            (batch_dir / "summary.json").write_text(json.dumps({"summary": {"total": 1}, "rows": rows}), encoding="utf-8")

            exit_code = main(
                [
                    "solve-clarified-batch",
                    "--artifact-dir",
                    str(batch_dir),
                    "--clarifications-dir",
                    str(clarifications_dir),
                ]
            )

            self.assertEqual(exit_code, 1)
            payload = json.loads((batch_dir / "clarification-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["unresolved_case_count"], 1)
            self.assertEqual(payload["summary"]["classifications"]["blocked_clarification"], 1)
            self.assertEqual(payload["summary"]["clarification_statuses"]["unresolved"], 1)
            self.assertEqual(payload["rows"][0]["clarification_gate_status"], "blocked")
            self.assertIn("does not exist", payload["rows"][0]["error"])
            report = (batch_dir / "clarification-report.md").read_text(encoding="utf-8")
            self.assertIn("CASE-SYMBOLIC", report)

    def test_clarification_report_includes_question_set_and_answer_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "batch"
            resolution_dir = artifact_dir / "clarified" / "CASE-SYMBOLIC" / "attempt-1"
            (resolution_dir / "clarification").mkdir(parents=True)
            (resolution_dir / "clarification" / "questions.json").write_text(
                json.dumps(_clarification_questions()),
                encoding="utf-8",
            )
            (resolution_dir / "clarification" / "answers.json").write_text(
                json.dumps(_clarification_answers()),
                encoding="utf-8",
            )
            row = {
                "problem_id": "CASE-SYMBOLIC",
                "source_artifact": str(artifact_dir / "CASE-SYMBOLIC"),
                "resolution_artifact": str(resolution_dir),
                "clarified_from": str(artifact_dir / "CASE-SYMBOLIC"),
                "clarification_status": "answered",
                "clarification_question_count": 1,
                "clarification_answer_count": 1,
                "clarification_source": "manual_review",
                "clarification_gate_status": "passed",
                "clarification_questions": "clarification/questions.json",
                "clarification_answers": "clarification/answers.json",
                "classification": "SUCCESS",
                "verification_status": "PASS",
                "spec_fidelity_status": "not_reviewed",
            }

            from or_llm_agent.cli import _clarification_row_from_solve, summarize_clarification_rows, write_clarification_report

            report_row = _clarification_row_from_solve(row, artifact_dir)
            summary = summarize_clarification_rows([report_row])
            report_path = root / "clarification-report.md"
            write_clarification_report(report_path, summary, [report_row])

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Generated question set", report)
            self.assertIn("What numeric truck costs should replace c_j?", report)
            self.assertIn("Answer provenance", report)
            self.assertIn("reviewer=`unit-test`", report)
            self.assertEqual(summary["attempted_needs_human_count"], 1)
            self.assertEqual(summary["generated_question_count"], 1)
            self.assertEqual(summary["answered_question_count"], 1)
            self.assertEqual(summary["clarified_supported_case_count"], 1)
            self.assertEqual(summary["or_ci_statuses"]["PASS"], 1)
            self.assertEqual(summary["fidelity_statuses"]["not_reviewed"], 1)

    def test_solve_writes_spec_fidelity_review_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            statement_path.write_text("Minimize one production cost.", encoding="utf-8")

            def fake_generate_agent_submission(inputs, paths, args):
                paths.submission_path.parent.mkdir(parents=True, exist_ok=True)
                paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
                paths.submission_path.write_text("def build_model(data):\n    return None\n", encoding="utf-8")
                paths.raw_path.write_text("generated model", encoding="utf-8")
                return {
                    "generation_status": "generated",
                    "generation_error": "",
                    "raw_response": paths.raw_path,
                    "submission": paths.submission_path,
                    "generation_mode": "agent",
                    "agent_returncode": 0,
                    "agent_timed_out": False,
                }

            verification = VerificationResult(
                returncode=0,
                stdout="",
                stderr="",
                report={"classification": "SUCCESS", "status": "PASS", "checks": []},
            )

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, _supported_capability())),
                patch("or_llm_agent.cli._run_problem_spec_agent", return_value=_agent_result(root, _valid_problem())),
                patch("or_llm_agent.cli.generate_agent_submission", fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", return_value=verification),
            ):
                exit_code = main(
                    [
                        "solve",
                        "--mode",
                        "agent",
                        "--statement-file",
                        str(statement_path),
                        "--problem-id",
                        "CASE-001",
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["capability_status"], "supported")
            self.assertEqual(summary["spec_validation_status"], "passed")
            self.assertEqual(summary["spec_repair_status"], "not_needed")
            self.assertEqual(summary["model_generation_status"], "generated")
            self.assertEqual(summary["verification_status"], "PASS")
            self.assertEqual(summary["classification"], "SUCCESS")
            self.assertEqual(summary["spec_fidelity_status"], "not_reviewed")
            self.assertEqual(summary["spec_fidelity_review"], "spec/fidelity-review.md")
            self.assertEqual(summary["spec_fidelity_report"], "spec/fidelity-review.json")
            self.assertEqual(summary["spec_fidelity_gate_status"], "manual_review_required")
            self.assertEqual(summary["spec_fidelity_rubric_version"], "source_fidelity_v1")
            self.assertFalse(summary["spec_fidelity_rubric_complete"])
            review = (artifact_dir / "spec" / "fidelity-review.md").read_text(encoding="utf-8")
            self.assertIn("Model generation: `generated`", review)
            self.assertIn("Verification: `PASS`", review)
            self.assertIn("Fidelity status: `not_reviewed`", review)
            self.assertIn("Fidelity gate: `manual_review_required`", review)
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["gate_status"], "manual_review_required")
            self.assertEqual(report["rubric_version"], "source_fidelity_v1")
            self.assertEqual(report["automatic_checks"][1]["status"], "PASS")

    def test_source_fidelity_risk_flags_cover_feature_boundaries(self) -> None:
        from or_llm_agent.cli import _spec_fidelity_risk_flags

        multi_problem = _multi_scenario_problem()
        del multi_problem["scenarios"][1]["objective"]
        multi_flags = _spec_fidelity_risk_flags(multi_problem, {})
        self.assertIn("multi_scenario_missing_objective_check", [flag["code"] for flag in multi_flags])

        routing_problem = _valid_problem()
        routing_problem["instance"] = {"distance": [1, 2, 3]}
        routing_flags = _spec_fidelity_risk_flags(routing_problem, {"problem_family": "TSP routing"})
        self.assertIn("routing_or_tsp_cost_scaling_only", [flag["code"] for flag in routing_flags])

        miqp_problem = _valid_problem()
        miqp_problem["problem_type"] = "MIQP"
        miqp_flags = _spec_fidelity_risk_flags(miqp_problem, {})
        self.assertIn("complex_unit_scaling_requires_review", [flag["code"] for flag in miqp_flags])

    def test_solve_batch_writes_aggregate_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            for problem_id in ("CASE-001", "CASE-002"):
                (statements_dir / f"{problem_id}.txt").write_text(f"Statement for {problem_id}.", encoding="utf-8")
            artifact_dir = root / "batch"

            def fake_run(**kwargs):
                return _agent_result(root, _valid_problem(kwargs["problem_id"]))

            def fake_generate_agent_submission(inputs, paths, args):
                paths.submission_path.parent.mkdir(parents=True, exist_ok=True)
                paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
                paths.submission_path.write_text("def build_model(data):\n    return None\n", encoding="utf-8")
                paths.raw_path.write_text("generated model", encoding="utf-8")
                return {
                    "generation_status": "generated",
                    "generation_error": "",
                    "raw_response": paths.raw_path,
                    "submission": paths.submission_path,
                    "generation_mode": "agent",
                    "agent_returncode": 0,
                    "agent_timed_out": False,
                }

            verification = VerificationResult(
                returncode=0,
                stdout="",
                stderr="",
                report={"classification": "SUCCESS", "status": "PASS", "checks": []},
            )

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, _supported_capability())),
                patch("or_llm_agent.cli._run_problem_spec_agent", fake_run),
                patch("or_llm_agent.cli.generate_agent_submission", fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", return_value=verification),
            ):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-001",
                        "CASE-002",
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["summary"]["total"], 2)
            self.assertEqual(batch["summary"]["succeeded"], 2)
            self.assertEqual(batch["summary"]["capability_statuses"]["supported"], 2)
            self.assertEqual(batch["summary"]["terminal_states"]["generated"], 2)
            self.assertEqual(batch["summary"]["accounted"], 2)
            self.assertEqual(batch["summary"]["unaccounted"], 0)
            self.assertEqual(batch["summary"]["spec_fidelity_statuses"]["not_reviewed"], 2)
            self.assertEqual(batch["summary"]["spec_fidelity_gate_statuses"]["manual_review_required"], 2)
            self.assertEqual(len(batch["rows"]), 2)
            self.assertEqual({row["terminal_state"] for row in batch["rows"]}, {"generated"})
            self.assertTrue((artifact_dir / "report.md").exists())
            self.assertTrue((artifact_dir / "CASE-001" / "spec" / "fidelity-review.json").exists())

    def test_solve_batch_marks_capability_block_as_excluded_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            (statements_dir / "CASE-SYMBOLIC.txt").write_text("Use trucks with symbolic costs c_j.", encoding="utf-8")
            artifact_dir = root / "batch"
            capability = {
                "status": "needs_human",
                "problem_family": "assignment",
                "supported_features": ["linear assignment structure"],
                "unsupported_features": [],
                "missing_information": ["numeric truck costs c_j are not provided"],
                "recommended_next_action": "ask_human",
                "confidence": 0.9,
                "review_note": "Symbolic objective coefficients require clarification.",
            }

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, capability)),
                patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent,
            ):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-SYMBOLIC",
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            run_problem_spec_agent.assert_not_called()
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["summary"]["terminal_states"]["excluded_with_reason"], 1)
            self.assertEqual(batch["summary"]["exclusion_reasons"]["capability_needs_human"], 1)
            self.assertEqual(batch["summary"]["accounted"], 1)
            self.assertEqual(batch["summary"]["unaccounted"], 0)
            row = batch["rows"][0]
            self.assertEqual(row["capability_status"], "needs_human")
            self.assertEqual(row["classification"], "blocked_capability")
            self.assertEqual(row["terminal_state"], "excluded_with_reason")
            self.assertEqual(row["accounting_state"], "excluded_with_reason")
            self.assertEqual(row["exclusion_reason"], "capability_needs_human")

    def test_solve_batch_preserves_serial_behavior_with_concurrency_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            for problem_id in ("CASE-001", "CASE-002"):
                (statements_dir / f"{problem_id}.txt").write_text(f"Statement for {problem_id}.", encoding="utf-8")
            artifact_dir = root / "batch"
            calls: list[str] = []

            def fake_solve(solve_args):
                calls.append(solve_args.problem_id)
                _write_minimal_case_summary(solve_args.artifact_dir, solve_args.problem_id)
                return 0

            with patch("or_llm_agent.cli.solve_command", fake_solve):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-001",
                        "CASE-002",
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--agent-concurrency",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls, ["CASE-001", "CASE-002"])
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["problem_id"] for row in batch["rows"]], ["CASE-001", "CASE-002"])

    def test_solve_batch_runs_cases_concurrently_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            for problem_id in ("CASE-001", "CASE-002"):
                (statements_dir / f"{problem_id}.txt").write_text(f"Statement for {problem_id}.", encoding="utf-8")
            artifact_dir = root / "batch"
            barrier = threading.Barrier(2, timeout=5)
            seen: set[str] = set()
            lock = threading.Lock()

            def fake_run(**kwargs):
                with lock:
                    seen.add(kwargs["problem_id"])
                barrier.wait()
                return _agent_result(root, _valid_problem(kwargs["problem_id"]))

            verification = VerificationResult(
                returncode=0,
                stdout="",
                stderr="",
                report={"classification": "SUCCESS", "status": "PASS", "checks": []},
            )

            with (
                patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, _supported_capability())),
                patch("or_llm_agent.cli._run_problem_spec_agent", fake_run),
                patch("or_llm_agent.cli.generate_agent_submission", _fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", return_value=verification),
            ):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-001",
                        "CASE-002",
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--agent-concurrency",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen, {"CASE-001", "CASE-002"})

    def test_solve_batch_handles_larger_concurrent_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            problem_ids = [f"CASE-{index:03d}" for index in range(1, 13)]
            for problem_id in problem_ids:
                (statements_dir / f"{problem_id}.txt").write_text(f"Statement for {problem_id}.", encoding="utf-8")
            artifact_dir = root / "batch"
            barrier = threading.Barrier(4, timeout=5)
            active_workers = 0
            max_active_workers = 0
            lock = threading.Lock()

            def fake_solve(solve_args):
                nonlocal active_workers, max_active_workers
                with lock:
                    active_workers += 1
                    max_active_workers = max(max_active_workers, active_workers)
                try:
                    barrier.wait()
                    _write_minimal_case_summary(solve_args.artifact_dir, solve_args.problem_id)
                    return 0
                finally:
                    with lock:
                        active_workers -= 1

            with patch("or_llm_agent.cli.solve_command", fake_solve):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        *problem_ids,
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--agent-concurrency",
                        "4",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(max_active_workers, 4)
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["summary"]["total"], 12)
            self.assertEqual(batch["summary"]["succeeded"], 12)
            self.assertEqual([row["problem_id"] for row in batch["rows"]], problem_ids)

    def test_bwor_run_dataset_builder_keeps_only_statement_and_answer_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "bwor.jsonl"
            source_path.write_text(
                json.dumps(
                    {
                        "id": "BWOR-001",
                        "en_question": "Statement only.",
                        "cn_question": "Chinese statement.",
                        "answer": 12.5,
                        "solution_status": "optimal",
                        "domain": "finance",
                        "problem_type": "LP",
                        "difficulty": "Medium",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = build_bwor_run_records(source_path)

            self.assertEqual(len(records), 1)
            self.assertEqual(tuple(records[0].keys()), BWOR_RUN_DATASET_KEYS)
            self.assertEqual(records[0]["id"], "BWOR-001")
            self.assertEqual(records[0]["en_question"], "Statement only.")
            self.assertEqual(records[0]["answer"], 12.5)

    def test_bwor_run_dataset_rejects_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_path = root / "bwor_run.jsonl"
            run_path.write_text(
                json.dumps(
                    {
                        "id": "BWOR-001",
                        "en_question": "Statement only.",
                        "answer": 12.5,
                        "problem_type": "LP",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "extra keys: problem_type"):
                load_bwor_run_record(run_path, "BWOR-001")

    def test_solve_batch_rejects_non_run_dataset_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "bwor.jsonl"
            artifact_dir = root / "batch"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "BWOR-001",
                        "en_question": "Statement only.",
                        "answer": 12.5,
                        "problem_type": "LP",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("or_llm_agent.cli.solve_command") as solve_command:
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "BWOR-001",
                        "--dataset",
                        str(dataset_path),
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            solve_command.assert_not_called()
            self.assertFalse((artifact_dir / "summary.json").exists())

    def test_solve_batch_rejects_invalid_agent_concurrency_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "batch"

            with patch("or_llm_agent.cli.solve_command") as solve_command:
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-001",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--agent-concurrency",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 1)
            solve_command.assert_not_called()
            self.assertFalse((artifact_dir / "summary.json").exists())

    def test_solve_batch_rejects_duplicate_ids_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "batch"

            with patch("or_llm_agent.cli.solve_command") as solve_command:
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-001",
                        "CASE-001",
                        "--artifact-dir",
                        str(artifact_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            solve_command.assert_not_called()
            self.assertFalse((artifact_dir / "summary.json").exists())

    def test_solve_batch_preserves_input_order_when_workers_finish_out_of_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            for problem_id in ("CASE-SLOW", "CASE-FAST"):
                (statements_dir / f"{problem_id}.txt").write_text(f"Statement for {problem_id}.", encoding="utf-8")
            artifact_dir = root / "batch"

            def fake_solve(solve_args):
                if solve_args.problem_id == "CASE-SLOW":
                    time.sleep(0.2)
                _write_minimal_case_summary(solve_args.artifact_dir, solve_args.problem_id)
                return 0

            with patch("or_llm_agent.cli.solve_command", fake_solve):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-SLOW",
                        "CASE-FAST",
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--agent-concurrency",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["problem_id"] for row in batch["rows"]], ["CASE-SLOW", "CASE-FAST"])

    def test_solve_batch_records_case_exception_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statements_dir = root / "statements"
            statements_dir.mkdir()
            for problem_id in ("CASE-FAIL", "CASE-OK"):
                (statements_dir / f"{problem_id}.txt").write_text(f"Statement for {problem_id}.", encoding="utf-8")
            artifact_dir = root / "batch"

            def fake_solve(solve_args):
                if solve_args.problem_id == "CASE-FAIL":
                    raise RuntimeError("synthetic failure")
                _write_minimal_case_summary(solve_args.artifact_dir, solve_args.problem_id)
                return 0

            with patch("or_llm_agent.cli.solve_command", fake_solve):
                exit_code = main(
                    [
                        "solve-batch",
                        "--mode",
                        "agent",
                        "--ids",
                        "CASE-FAIL",
                        "CASE-OK",
                        "--statements-dir",
                        str(statements_dir),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--agent-concurrency",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 1)
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            row_by_id = {row["problem_id"]: row for row in batch["rows"]}
            self.assertEqual(row_by_id["CASE-FAIL"]["exit_code"], 1)
            self.assertIn("RuntimeError", row_by_id["CASE-FAIL"]["batch_error"])
            self.assertEqual(row_by_id["CASE-FAIL"]["terminal_state"], "excluded_with_reason")
            self.assertEqual(row_by_id["CASE-FAIL"]["exclusion_reason"], "batch_error")
            self.assertEqual(row_by_id["CASE-OK"]["exit_code"], 0)
            self.assertEqual(row_by_id["CASE-OK"]["terminal_state"], "generated")
            self.assertEqual(batch["summary"]["terminal_states"]["excluded_with_reason"], 1)
            self.assertEqual(batch["summary"]["terminal_states"]["generated"], 1)
            self.assertEqual(batch["summary"]["accounted"], 2)
            self.assertEqual(batch["summary"]["unaccounted"], 0)

    def test_review_fidelity_accepts_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir)

            exit_code = main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--status",
                    "accepted",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Numbers and constraints match the statement.",
                    "--evidence",
                    "manual checklist",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "accepted")
            self.assertEqual(summary["spec_fidelity_gate_status"], "accepted")
            self.assertEqual(summary["spec_fidelity_reviewed_by"], "unit-test")
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fidelity_status"], "accepted")
            self.assertEqual(report["gate_status"], "accepted")
            self.assertEqual(report["review"]["status"], "accepted")
            self.assertEqual(report["rubric_version"], "legacy-flat")
            self.assertFalse(summary["spec_fidelity_rubric_complete"])
            self.assertEqual(report["automatic_checks"][2]["status"], "PASS")
            review = (artifact_dir / "spec" / "fidelity-review.md").read_text(encoding="utf-8")
            self.assertIn("## Review Decision", review)
            self.assertIn("Status: `accepted`", review)

    def test_review_fidelity_agent_accepts_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir)

            payload = {
                "status": "accepted",
                "confidence": 0.92,
                "rubric_version": "source_fidelity_v1",
                "dimensions": _rubric_dimensions(),
                "issues": [],
                "review_note": "Source statement and generated spec match.",
                "evidence": ["objective and price field match"],
            }
            run_metadata = {
                "schema_version": "codex_run_metadata_v1",
                "adapter": "codex-cli",
                "codex_effective_model": "gpt-5.5",
                "codex_effective_reasoning_effort": "xhigh",
                "codex_cli_version": "codex-cli test",
                "codex_usage": {"input_tokens": 10, "reasoning_output_tokens": 4},
            }
            with patch(
                "or_llm_agent.cli._run_fidelity_review_agent",
                return_value=_review_agent_result(root, payload, run_metadata=run_metadata),
            ):
                exit_code = main(["review-fidelity", "--mode", "agent", "--artifact-dir", str(artifact_dir)])

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "llm_accepted")
            self.assertEqual(summary["spec_fidelity_gate_status"], "llm_accepted")
            self.assertEqual(summary["spec_fidelity_review_mode"], "agent")
            self.assertEqual(summary["spec_fidelity_confidence"], 0.92)
            self.assertEqual(summary["spec_fidelity_issue_count"], 0)
            self.assertEqual(summary["spec_fidelity_rubric_version"], "source_fidelity_v1")
            self.assertTrue(summary["spec_fidelity_rubric_complete"])
            self.assertEqual(summary["spec_fidelity_blocking_dimension_count"], 0)
            self.assertEqual(summary["spec_fidelity_codex_effective_model"], "gpt-5.5")
            self.assertEqual(summary["spec_fidelity_codex_effective_reasoning_effort"], "xhigh")
            self.assertEqual(summary["spec_fidelity_codex_usage"]["reasoning_output_tokens"], 4)
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["review"]["reviewer"], "codex-agent")
            self.assertEqual(report["review"]["codex_effective_model"], "gpt-5.5")
            self.assertEqual(report["dimensions"]["objective"]["status"], "pass")
            self.assertEqual(report["automatic_checks"][2]["status"], "PASS")

    def test_review_fidelity_agent_rejects_missing_rubric_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir)

            payload = {
                "status": "accepted",
                "confidence": 0.92,
                "issues": [],
                "review_note": "Source statement and generated spec match.",
                "evidence": ["objective and price field match"],
            }
            with patch(
                "or_llm_agent.cli._run_fidelity_review_agent",
                return_value=_review_agent_result(root, payload),
            ):
                exit_code = main(["review-fidelity", "--mode", "agent", "--artifact-dir", str(artifact_dir)])

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "llm_rejected")
            self.assertFalse(summary["spec_fidelity_rubric_complete"])
            self.assertIn("incomplete source_fidelity_v1 rubric", summary["spec_fidelity_review_note"])
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["automatic_checks"][2]["status"], "FAIL")

    def test_review_fidelity_agent_hard_block_dimension_overrides_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir)

            payload = {
                "status": "accepted",
                "confidence": 0.8,
                "rubric_version": "source_fidelity_v1",
                "dimensions": _rubric_dimensions(
                    objective=_rubric_dimension(
                        status="fail",
                        severity="critical",
                        finding="Generated objective omits a source cost term.",
                    )
                ),
                "issues": [],
                "review_note": "Attempted acceptance should be blocked.",
                "evidence": ["objective mismatch"],
            }
            with patch(
                "or_llm_agent.cli._run_fidelity_review_agent",
                return_value=_review_agent_result(root, payload),
            ):
                exit_code = main(["review-fidelity", "--mode", "agent", "--artifact-dir", str(artifact_dir)])

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "llm_rejected")
            self.assertEqual(summary["spec_fidelity_failed_dimensions"], ["objective"])
            self.assertEqual(summary["spec_fidelity_blocking_dimension_count"], 1)
            self.assertEqual(summary["spec_fidelity_issue_count"], 1)

    def test_review_fidelity_marks_provisional_clarification_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir)
            summary_path = artifact_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["clarification_source"] = "provisional_agent_assumption"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

            payload = {
                "status": "accepted",
                "confidence": 0.9,
                "rubric_version": "source_fidelity_v1",
                "dimensions": _rubric_dimensions(),
                "issues": [],
                "review_note": "Accepted with provisional clarification warning.",
                "evidence": ["reviewed generated clarification"],
            }
            with patch(
                "or_llm_agent.cli._run_fidelity_review_agent",
                return_value=_review_agent_result(root, payload),
            ):
                exit_code = main(["review-fidelity", "--mode", "agent", "--artifact-dir", str(artifact_dir)])

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "llm_accepted")
            self.assertTrue(summary["spec_fidelity_provisional"])
            self.assertIn("clarification_dependency", summary["spec_fidelity_warned_dimensions"])
            review = (artifact_dir / "spec" / "fidelity-review.md").read_text(encoding="utf-8")
            self.assertIn("Provisional: `True`", review)

    def test_review_fidelity_agent_rejects_no_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir)

            result = FidelityReviewAgentResult(
                raw_text="not json",
                returncode=0,
                timed_out=False,
                events_path=root / "review-events.jsonl",
                last_message_path=root / "review-last-message.md",
                stderr="",
            )
            with patch("or_llm_agent.cli._run_fidelity_review_agent", return_value=result):
                exit_code = main(["review-fidelity", "--mode", "agent", "--artifact-dir", str(artifact_dir)])

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "llm_rejected")
            self.assertEqual(summary["spec_fidelity_gate_status"], "llm_rejected")
            self.assertIn("did not return a JSON object", summary["spec_fidelity_review_note"])
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["automatic_checks"][2]["status"], "FAIL")

    def test_review_fidelity_rejects_case_even_when_verification_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir, verification=VerificationResult(1, "", "", None))

            exit_code = main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--status",
                    "rejected",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Verification did not pass.",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "rejected")
            self.assertEqual(summary["spec_fidelity_gate_status"], "rejected")
            report = json.loads((artifact_dir / "spec" / "fidelity-review.json").read_text(encoding="utf-8"))
            self.assertEqual(report["automatic_checks"][2]["status"], "FAIL")

    def test_review_fidelity_does_not_accept_failed_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            _write_solved_case(root, statement_path, artifact_dir, verification=VerificationResult(1, "", "", None))

            exit_code = main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--status",
                    "accepted",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Should fail.",
                ]
            )

            self.assertEqual(exit_code, 1)
            summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["spec_fidelity_status"], "not_reviewed")
            self.assertEqual(summary["spec_fidelity_gate_status"], "manual_review_required")

    def test_review_fidelity_batch_updates_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "batch"
            statements_dir = artifact_dir / "statements"
            statements_dir.mkdir(parents=True)
            for problem_id in ("CASE-001", "CASE-002"):
                _write_solved_case(
                    root,
                    statements_dir / f"{problem_id}.txt",
                    artifact_dir / problem_id,
                    problem_id=problem_id,
                )
            rows = [
                {
                    "problem_id": problem_id,
                    "exit_code": 0,
                    "artifact_dir": problem_id,
                    "statement_file": f"statements/{problem_id}.txt",
                    "spec_validation_status": "passed",
                    "spec_attempt_count": 1,
                    "spec_repair_status": "not_needed",
                    "model_generation_status": "generated",
                    "verification_status": "PASS",
                    "classification": "SUCCESS",
                    "spec_fidelity_status": "not_reviewed",
                    "spec_fidelity_gate_status": "manual_review_required",
                }
                for problem_id in ("CASE-001", "CASE-002")
            ]
            (artifact_dir / "summary.json").write_text(
                json.dumps({"summary": {"total": 2}, "rows": rows}, indent=2) + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "review-fidelity-batch",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--status",
                    "accepted",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Manual batch review accepted.",
                ]
            )

            self.assertEqual(exit_code, 0)
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["summary"]["spec_fidelity_statuses"]["accepted"], 2)
            self.assertEqual(batch["summary"]["spec_fidelity_gate_statuses"]["accepted"], 2)
            self.assertEqual(batch["rows"][0]["spec_fidelity_status"], "accepted")
            self.assertIn("accepted", (artifact_dir / "report.md").read_text(encoding="utf-8"))
            self.assertTrue((artifact_dir / "fidelity-rubric-summary.json").exists())
            self.assertTrue((artifact_dir / "fidelity-rubric-report.md").exists())
            rubric = json.loads((artifact_dir / "fidelity-rubric-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(rubric["summary"]["legacy_flat_count"], 2)
            self.assertIn("Source-Fidelity Capstone", (artifact_dir / "report.md").read_text(encoding="utf-8"))
            self.assertIn("Claim Boundary", (artifact_dir / "fidelity-rubric-report.md").read_text(encoding="utf-8"))

    def test_review_fidelity_batch_with_ids_preserves_unreviewed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "batch"
            statements_dir = artifact_dir / "statements"
            statements_dir.mkdir(parents=True)
            for problem_id in ("CASE-001", "CASE-002"):
                _write_solved_case(
                    root,
                    statements_dir / f"{problem_id}.txt",
                    artifact_dir / problem_id,
                    problem_id=problem_id,
                )
            rows = [
                {
                    "problem_id": problem_id,
                    "exit_code": 0,
                    "artifact_dir": problem_id,
                    "statement_file": f"statements/{problem_id}.txt",
                    "spec_validation_status": "passed",
                    "spec_attempt_count": 1,
                    "spec_repair_status": "not_needed",
                    "model_generation_status": "generated",
                    "verification_status": "PASS",
                    "classification": "SUCCESS",
                    "spec_fidelity_status": "not_reviewed",
                    "spec_fidelity_gate_status": "manual_review_required",
                }
                for problem_id in ("CASE-001", "CASE-002")
            ]
            (artifact_dir / "summary.json").write_text(
                json.dumps({"summary": {"total": 2}, "rows": rows}, indent=2) + "\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "review-fidelity-batch",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--ids",
                    "CASE-001",
                    "--status",
                    "accepted",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Manual subset review accepted.",
                ]
            )

            self.assertEqual(exit_code, 0)
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(batch["summary"]["total"], 2)
            self.assertEqual(batch["summary"]["spec_fidelity_statuses"]["accepted"], 1)
            self.assertEqual(batch["summary"]["spec_fidelity_statuses"]["not_reviewed"], 1)
            self.assertEqual([row["problem_id"] for row in batch["rows"]], ["CASE-001", "CASE-002"])
            row_by_id = {row["problem_id"]: row for row in batch["rows"]}
            self.assertEqual(row_by_id["CASE-001"]["spec_fidelity_status"], "accepted")
            self.assertEqual(row_by_id["CASE-002"]["spec_fidelity_status"], "not_reviewed")

    def test_resolve_fidelity_repairs_rejected_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            resolution_dir = root / "resolved"
            _write_solved_case(root, statement_path, artifact_dir, verification=_verification_with_objective(10.0))
            main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--status",
                    "rejected",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Missing action in generated spec.",
                ]
            )
            repair_calls = []

            def fake_run(**kwargs):
                repair_calls.append(kwargs)
                return _agent_result(root, _valid_problem(kwargs["problem_id"]))

            accepted_review = {
                "status": "accepted",
                "confidence": 0.91,
                "rubric_version": "source_fidelity_v1",
                "dimensions": _rubric_dimensions(),
                "issues": [],
                "review_note": "Repaired spec now matches the statement.",
                "evidence": ["missing action restored"],
            }
            with (
                patch("or_llm_agent.cli._run_problem_spec_agent", fake_run),
                patch("or_llm_agent.cli.generate_agent_submission", _fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", _fake_verify_with_objective(10.0)),
                patch("or_llm_agent.cli._run_fidelity_review_agent", return_value=_review_agent_result(root, accepted_review)),
            ):
                exit_code = main(
                    [
                        "resolve-fidelity",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--resolution-dir",
                        str(resolution_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(repair_calls)
            self.assertIn("Missing action", repair_calls[0]["repair_error"])
            source_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(source_summary["fidelity_resolution_status"], "repaired_accepted")
            self.assertEqual(source_summary["fidelity_resolution_repaired_fidelity_status"], "llm_accepted")
            self.assertEqual(source_summary["fidelity_resolution_impact_classification"], "not_needed")
            repaired_summary = json.loads((resolution_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(repaired_summary["spec_repair_status"], "repaired")
            self.assertEqual(repaired_summary["spec_fidelity_status"], "llm_accepted")
            resolution = json.loads((resolution_dir / "fidelity-resolution.json").read_text(encoding="utf-8"))
            self.assertEqual(resolution["resolution_status"], "repaired_accepted")

    def test_resolve_fidelity_classifies_harmless_equivalent_when_review_still_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            statement_path = root / "problem.txt"
            artifact_dir = root / "run"
            resolution_dir = root / "resolved"
            _write_solved_case(root, statement_path, artifact_dir, verification=_verification_with_objective(10.0))
            main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--status",
                    "rejected",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Missing dominated terminal action.",
                ]
            )
            rejected_review = {
                "status": "rejected",
                "confidence": 0.8,
                "rubric_version": "source_fidelity_v1",
                "dimensions": _rubric_dimensions(
                    action_space=_rubric_dimension(
                        status="warn",
                        severity="minor",
                        finding="Residual dominated action-space issue remains.",
                    )
                ),
                "issues": [{"severity": "minor", "field": "action_space", "message": "still incomplete"}],
                "review_note": "Residual action-space issue remains.",
                "evidence": [],
            }
            with (
                patch("or_llm_agent.cli._run_problem_spec_agent", return_value=_agent_result(root, _valid_problem())),
                patch("or_llm_agent.cli.generate_agent_submission", _fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", _fake_verify_with_objective(10.0)),
                patch("or_llm_agent.cli._run_fidelity_review_agent", return_value=_review_agent_result(root, rejected_review)),
            ):
                exit_code = main(
                    [
                        "resolve-fidelity",
                        "--artifact-dir",
                        str(artifact_dir),
                        "--resolution-dir",
                        str(resolution_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            source_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(source_summary["fidelity_resolution_status"], "residual_harmless_equivalent")
            self.assertEqual(source_summary["fidelity_resolution_impact_classification"], "harmless_equivalent")
            resolution = json.loads((resolution_dir / "fidelity-resolution.json").read_text(encoding="utf-8"))
            self.assertEqual(resolution["impact_analysis"]["classification"], "harmless_equivalent")
            self.assertEqual(resolution["repaired_fidelity_status"], "llm_rejected")

    def test_resolve_fidelity_batch_processes_only_rejected_cases_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "batch"
            statements_dir = artifact_dir / "statements"
            statements_dir.mkdir(parents=True)
            for problem_id in ("CASE-001", "CASE-002"):
                _write_solved_case(
                    root,
                    statements_dir / f"{problem_id}.txt",
                    artifact_dir / problem_id,
                    problem_id=problem_id,
                    verification=_verification_with_objective(10.0),
                )
            main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir / "CASE-001"),
                    "--status",
                    "rejected",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Needs repair.",
                ]
            )
            main(
                [
                    "review-fidelity",
                    "--artifact-dir",
                    str(artifact_dir / "CASE-002"),
                    "--status",
                    "accepted",
                    "--reviewer",
                    "unit-test",
                    "--note",
                    "Accepted.",
                ]
            )
            rows = [
                {
                    "problem_id": problem_id,
                    "exit_code": 0,
                    "artifact_dir": problem_id,
                    "statement_file": f"statements/{problem_id}.txt",
                    "spec_validation_status": "passed",
                    "spec_attempt_count": 1,
                    "spec_repair_status": "not_needed",
                    "model_generation_status": "generated",
                    "verification_status": "PASS",
                    "classification": "SUCCESS",
                    "spec_fidelity_status": "rejected" if problem_id == "CASE-001" else "accepted",
                    "spec_fidelity_gate_status": "rejected" if problem_id == "CASE-001" else "accepted",
                }
                for problem_id in ("CASE-001", "CASE-002")
            ]
            (artifact_dir / "summary.json").write_text(
                json.dumps({"summary": {"total": 2}, "rows": rows}, indent=2) + "\n",
                encoding="utf-8",
            )
            accepted_review = {
                "status": "accepted",
                "confidence": 0.9,
                "rubric_version": "source_fidelity_v1",
                "dimensions": _rubric_dimensions(),
                "issues": [],
                "review_note": "Resolved.",
                "evidence": [],
            }
            with (
                patch("or_llm_agent.cli._run_problem_spec_agent", return_value=_agent_result(root, _valid_problem("CASE-001"))),
                patch("or_llm_agent.cli.generate_agent_submission", _fake_generate_agent_submission),
                patch("or_llm_agent.cli.run_or_ci_verify", _fake_verify_with_objective(10.0)),
                patch("or_llm_agent.cli._run_fidelity_review_agent", return_value=_review_agent_result(root, accepted_review)),
            ):
                exit_code = main(["resolve-fidelity-batch", "--artifact-dir", str(artifact_dir)])

            self.assertEqual(exit_code, 0)
            resolution_summary = json.loads((artifact_dir / "fidelity-resolution-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(resolution_summary["summary"]["total"], 1)
            self.assertEqual(resolution_summary["rows"][0]["problem_id"], "CASE-001")
            batch = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
            row_by_id = {row["problem_id"]: row for row in batch["rows"]}
            self.assertEqual(row_by_id["CASE-001"]["fidelity_resolution_status"], "repaired_accepted")
            self.assertNotIn("fidelity_resolution_status", row_by_id["CASE-002"])


def _agent_result(root: Path, payload: dict) -> ProblemSpecAgentResult:
    return ProblemSpecAgentResult(
        raw_text=json.dumps(payload, indent=2),
        returncode=0,
        timed_out=False,
        events_path=root / "events.jsonl",
        last_message_path=root / "last-message.md",
        stderr="",
    )


def _review_agent_result(root: Path, payload: dict, run_metadata: dict | None = None) -> FidelityReviewAgentResult:
    return FidelityReviewAgentResult(
        raw_text=json.dumps(payload, indent=2),
        returncode=0,
        timed_out=False,
        events_path=root / "review-events.jsonl",
        last_message_path=root / "review-last-message.md",
        stderr="",
        run_metadata=run_metadata or {},
    )


def _clarification_agent_result(root: Path, payload: dict) -> ClarificationAgentResult:
    return ClarificationAgentResult(
        raw_text=json.dumps(payload, indent=2),
        returncode=0,
        timed_out=False,
        events_path=root / "clarification-events.jsonl",
        last_message_path=root / "clarification-last-message.md",
        stderr="",
    )


def _capability_agent_result(root: Path, payload: dict) -> CapabilityAgentResult:
    return CapabilityAgentResult(
        raw_text=json.dumps(payload, indent=2),
        returncode=0,
        timed_out=False,
        events_path=root / "capability-events.jsonl",
        last_message_path=root / "capability-last-message.md",
        stderr="",
    )


def _supported_capability(problem_family: str = "linear programming") -> dict:
    return {
        "status": "supported",
        "problem_family": problem_family,
        "supported_features": ["single-objective linear model with numeric data"],
        "unsupported_features": [],
        "missing_information": [],
        "recommended_next_action": "continue_to_problemspec",
        "confidence": 0.95,
        "review_note": "Statement is within current OR-CI LP/MILP support.",
    }


def _clarification_questions(problem_id: str = "CASE-SYMBOLIC") -> dict:
    return {
        "problem_id": problem_id,
        "source_artifact": "run",
        "blocking_status": "needs_human",
        "questions": [
            {
                "id": "q1",
                "issue_type": "missing_numeric_data",
                "prompt": "What numeric truck costs should replace c_j?",
                "source_evidence": "The statement uses symbolic truck costs c_j.",
                "allowed_answer_type": "free_text",
                "options": [],
                "required": True,
            }
        ],
    }


def _clarification_answers(problem_id: str = "CASE-SYMBOLIC", reviewer: str = "unit-test") -> dict:
    return {
        "problem_id": problem_id,
        "answers": [
            {
                "question_id": "q1",
                "answer": "Use costs 10, 12, and 14 for trucks 1, 2, and 3.",
                "reviewer": reviewer,
                "rationale": "Manual review supplied the omitted numeric costs.",
                "source": "manual_review",
            }
        ],
        "resolution_status": "answered",
    }


def _rubric_dimension(
    *,
    status: str = "pass",
    severity: str = "none",
    finding: str = "Source and generated spec are aligned for this dimension.",
    evidence: list[str] | None = None,
    artifact_refs: list[str] | None = None,
) -> dict:
    return {
        "status": status,
        "severity": severity,
        "finding": finding,
        "evidence": evidence if evidence is not None else ["unit-test evidence"],
        "artifact_refs": artifact_refs if artifact_refs is not None else ["statement", "spec/problem.json"],
    }


def _rubric_dimensions(**overrides: dict) -> dict:
    dimensions = {
        "source_suitability": _rubric_dimension(),
        "data_completeness": _rubric_dimension(),
        "sets_and_indices": _rubric_dimension(),
        "numeric_parameters": _rubric_dimension(),
        "action_space": _rubric_dimension(),
        "objective": _rubric_dimension(),
        "units_and_scaling": _rubric_dimension(),
        "constraint_families": _rubric_dimension(),
        "metamorphic_coverage": _rubric_dimension(),
        "clarification_dependency": _rubric_dimension(status="not_applicable", finding="No clarification dependency."),
        "materiality": _rubric_dimension(finding="No material mismatch detected."),
    }
    dimensions.update(overrides)
    return dimensions


def _fake_generate_agent_submission(inputs, paths, args):
    paths.submission_path.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    paths.submission_path.write_text("def build_model(data):\n    return None\n", encoding="utf-8")
    paths.raw_path.write_text("generated model", encoding="utf-8")
    return {
        "generation_status": "generated",
        "generation_error": "",
        "raw_response": paths.raw_path,
        "submission": paths.submission_path,
        "generation_mode": "agent",
        "agent_returncode": 0,
        "agent_timed_out": False,
    }


def _verification_with_objective(objective: float) -> VerificationResult:
    return VerificationResult(
        returncode=0,
        stdout="",
        stderr="",
        report={
            "classification": "SUCCESS",
            "status": "PASS",
            "checks": [
                {
                    "name": "original_solver_status",
                    "status": "PASS",
                    "details": {"objective_value": objective},
                }
            ],
        },
    )


def _fake_verify_with_objective(objective: float):
    verification = _verification_with_objective(objective)

    def fake_verify(problem_path, submission_path, report_path, cwd):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(verification.report, indent=2) + "\n", encoding="utf-8")
        return verification

    return fake_verify


def _write_solved_case(
    root: Path,
    statement_path: Path,
    artifact_dir: Path,
    *,
    problem_id: str = "CASE-001",
    verification: VerificationResult | None = None,
) -> None:
    statement_path.parent.mkdir(parents=True, exist_ok=True)
    statement_path.write_text(f"Statement for {problem_id}.", encoding="utf-8")
    verification = verification or VerificationResult(
        returncode=0,
        stdout="",
        stderr="",
        report={"classification": "SUCCESS", "status": "PASS", "checks": []},
    )

    def fake_verify(problem_path, submission_path, report_path, cwd):
        if verification.report is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(verification.report, indent=2) + "\n", encoding="utf-8")
        return verification

    with (
        patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, _supported_capability())),
        patch("or_llm_agent.cli._run_problem_spec_agent", return_value=_agent_result(root, _valid_problem(problem_id))),
        patch("or_llm_agent.cli.generate_agent_submission", _fake_generate_agent_submission),
        patch("or_llm_agent.cli.run_or_ci_verify", fake_verify),
    ):
        exit_code = main(
            [
                "solve",
                "--mode",
                "agent",
                "--statement-file",
                str(statement_path),
                "--problem-id",
                problem_id,
                "--artifact-dir",
                str(artifact_dir),
            ]
        )
    expected = 0 if verification.returncode == 0 else 1
    if exit_code != expected:
        raise AssertionError(f"unexpected solve exit code: {exit_code}")


def _write_minimal_case_summary(artifact_dir: Path, problem_id: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(
            {
                "problem_id": problem_id,
                "capability_status": "supported",
                "capability_generation_status": "classified",
                "problem_family": "linear programming",
                "spec_validation_status": "passed",
                "spec_attempt_count": 1,
                "spec_repair_status": "not_needed",
                "model_generation_status": "generated",
                "verification_status": "PASS",
                "classification": "SUCCESS",
                "spec_fidelity_status": "not_reviewed",
                "spec_fidelity_gate_status": "manual_review_required",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_needs_human_case(root: Path, artifact_dir: Path, *, problem_id: str = "CASE-SYMBOLIC") -> None:
    capability = {
        "status": "needs_human",
        "problem_family": "assignment",
        "supported_features": ["linear assignment structure"],
        "unsupported_features": [],
        "missing_information": ["numeric truck costs c_j are not provided"],
        "recommended_next_action": "ask_human",
        "confidence": 0.9,
        "review_note": "Symbolic objective coefficients require clarification.",
    }
    _write_blocked_case(root, artifact_dir, problem_id=problem_id, capability=capability)


def _write_blocked_case(
    root: Path,
    artifact_dir: Path,
    *,
    problem_id: str,
    capability: dict,
    statement: str = "Use trucks with symbolic costs c_j.",
) -> None:
    statement_path = root / f"{problem_id}.txt"
    statement_path.write_text(statement, encoding="utf-8")
    with (
        patch("or_llm_agent.cli._run_capability_agent", return_value=_capability_agent_result(root, capability)),
        patch("or_llm_agent.cli._run_problem_spec_agent") as run_problem_spec_agent,
    ):
        exit_code = main(
            [
                "solve",
                "--mode",
                "agent",
                "--statement-file",
                str(statement_path),
                "--problem-id",
                problem_id,
                "--artifact-dir",
                str(artifact_dir),
            ]
        )
    if exit_code != 1:
        raise AssertionError(f"unexpected blocked solve exit code: {exit_code}")
    run_problem_spec_agent.assert_not_called()


def _valid_problem(problem_id: str = "CASE-001") -> dict:
    return {
        "id": problem_id,
        "problem_type": "LP",
        "instance": {"price": 4.0},
        "metamorphic": {
            "cost_scaling": {
                "coefficient_paths": ["instance.price"],
                "factors": [2.0],
                "tolerance_abs": 1e-6,
                "tolerance_rel": 1e-6,
            }
        },
    }


def _multi_scenario_problem(problem_id: str = "CASE-SCENARIO") -> dict:
    return {
        "id": problem_id,
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
                "metamorphic": {
                    "cost_scaling": {
                        "coefficient_paths": ["instance.objective"],
                        "factors": [2.0],
                    }
                },
            },
        ],
    }


def _invalid_constraint_relaxation_problem() -> dict:
    problem = _valid_problem()
    problem["metamorphic"]["constraint_relaxation"] = {
        "relaxations": [
            {
                "path": "instance.capacity",
                "direction": "increase",
                "amount": 1.0,
                "objective_relation": "non_decrease",
            }
        ],
        "tolerance_abs": 1e-6,
        "tolerance_rel": 1e-6,
    }
    return problem


if __name__ == "__main__":
    unittest.main()
