from __future__ import annotations

from main import build_parser, main


def test_cli_contract_parses_all_three_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["similarity"]).command == "similarity"
    nearest = parser.parse_args(["nearest", "--word", "sinh_viên", "--k", "5"])
    assert nearest.word == "sinh_viên"
    assert nearest.k == 5
    assert parser.parse_args(["classify"]).command == "classify"


def test_completed_task_command_returns_success(monkeypatch) -> None:
    captured = {}

    def fake_dispatch(args):
        captured["command"] = args.command
        return 0

    monkeypatch.setattr("main._dispatch", fake_dispatch)

    assert main(["classify"]) == 0
    assert captured["command"] == "classify"
