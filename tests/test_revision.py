from __future__ import annotations

import tempfile
from pathlib import Path

from or_llm_agent.cli import build_parser
from or_llm_agent.codex_agent import CodexAgentOptions
from or_llm_agent.revision import build_revision_command, build_revision_prompt


def test_revise_parser_contract() -> None:
    args = build_parser().parse_args(
        [
            "revise",
            "--problem-id",
            "feed-v1",
            "--base-submission",
            "base.py",
            "--base-statement",
            "base.md",
            "--change-log",
            "change.md",
            "--problem",
            "problem.json",
            "--artifact-dir",
            "artifacts",
        ]
    )
    assert args.command == "revise"
    assert args.problem_id == "feed-v1"


def test_revision_command_uses_requested_model_and_effort() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        options = CodexAgentOptions(
            codex_model="gpt-5.6-sol",
            codex_sandbox="workspace-write",
            codex_approval="never",
            max_repair_attempts=2,
            timeout_seconds=900,
            codex_reasoning_effort="high",
        )
        command = build_revision_command(root, root / "work", root / "last.md", options)
        assert command[command.index("-m") + 1] == "gpt-5.6-sol"
        assert command[command.index("-c") + 1] == 'model_reasoning_effort="high"'
        assert command[-1] == "-"


def test_revision_prompt_contains_visible_history_but_no_hidden_surfaces() -> None:
    prompt = build_revision_prompt(
        problem_id="feed-v1",
        base_submission="def build_model(data): pass",
        base_statement="Need protein and vitamins.",
        change_log="Cap feed 5.",
        problem={"instance": {"feeds": ["feed_1", "feed_5"]}},
        problem_path=Path("/tmp/problem.json"),
        submission_path=Path("/tmp/revised.py"),
        report_path=Path("/tmp/report.json"),
        max_repair_attempts=3,
    )
    assert "Need protein and vitamins." in prompt
    assert "Cap feed 5." in prompt
    assert "semantic_variables" in prompt
    assert "hidden tests" in prompt
    assert "witness instance" not in prompt
    assert "F-FEED5-CAP" not in prompt


def test_revision_prompt_can_disclose_one_repair_witness() -> None:
    prompt = build_revision_prompt(
        problem_id="feed-v2-repair",
        base_submission="def build_model(data): pass",
        base_statement="Preserve active rules.",
        change_log="Repair current model.",
        problem={"instance": {"feeds": ["feed_1"]}},
        problem_path=Path("/tmp/problem.json"),
        submission_path=Path("/tmp/revised.py"),
        report_path=Path("/tmp/report.json"),
        max_repair_attempts=1,
        disclosed_witness='{"requirements": {"protein": 900}}',
    )
    assert "One disclosed counterexample" in prompt
    assert '"protein": 900' in prompt
    assert "You may use only disclosed counterexample" in prompt
