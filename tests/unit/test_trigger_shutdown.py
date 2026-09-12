"""The trigger stops when Kubernetes asks it to.

Pins the behaviour missing on 2026-09-12, when a sibling service ignored
SIGTERM for the full 30-minute drain grace because nothing registered a
handler and PID 1 gets no default signal action.
"""
import signal

import pytest

from src.consolidator.trigger.shutdown import STOP_SIGNALS, install_stop_handlers


def test_registers_a_handler_for_sigterm():
    # The registration itself is the fix: PID 1 ignores any signal whose
    # disposition is still the default, so an unregistered SIGTERM never
    # reaches the process at all.
    registered = {}
    install_stop_handlers(register=registered.setdefault)

    assert signal.SIGTERM in registered


def test_registers_a_handler_for_sigint_too():
    registered = {}
    install_stop_handlers(register=registered.setdefault)

    assert signal.SIGINT in registered
    assert set(registered) == set(STOP_SIGNALS)


def test_the_handler_exits_zero():
    # A clean stop, not a crash: the drain and the restart both read this
    # as an orderly exit.
    registered = {}
    install_stop_handlers(register=registered.setdefault)

    with pytest.raises(SystemExit) as excinfo:
        registered[signal.SIGTERM](signal.SIGTERM, None)

    assert excinfo.value.code == 0


def test_the_handler_is_not_swallowed_by_the_loops_error_handling():
    # run_forever guards its body with `except Exception` and retries.
    # If the handler raised anything derived from Exception the consumer
    # would log it, sleep, and carry on running — ignoring the signal in
    # a new and more confusing way. SystemExit derives from
    # BaseException, so it passes straight through.
    registered = {}
    install_stop_handlers(register=registered.setdefault)

    try:
        registered[signal.SIGTERM](signal.SIGTERM, None)
    except Exception:  # pylint: disable=broad-exception-caught
        pytest.fail("the stop signal was catchable as Exception; run_forever would swallow it")
    except SystemExit:
        pass


def test_install_uses_the_real_signal_module_by_default():
    # Restore whatever pytest had installed, so this test cannot leak a
    # handler into the rest of the run.
    previous = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
    try:
        install_stop_handlers()
        for sig in STOP_SIGNALS:
            assert signal.getsignal(sig) not in (signal.SIG_DFL, signal.SIG_IGN)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
