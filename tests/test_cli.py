from __future__ import annotations

from main import build_parser, main


def test_cli_contract_parses_all_three_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["similarity"]).command == "similarity"
    nearest = parser.parse_args(["nearest", "--word", "sinh_viên", "--k", "5"])
    assert nearest.word == "sinh_viên"
    assert nearest.k == 5
    assert parser.parse_args(["classify"]).command == "classify"


def test_stage0_task_command_is_an_explicit_placeholder(capsys) -> None:
    assert main(["classify"]) == 2
    assert "intentionally not implemented" in capsys.readouterr().out
