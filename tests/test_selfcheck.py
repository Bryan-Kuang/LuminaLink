from luminalink.selfcheck import run_self_check


def test_self_check_runs() -> None:
    """Ensure the scaffolding pipeline runs end-to-end without optional ML deps."""

    run_self_check()

