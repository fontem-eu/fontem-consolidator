"""The chart must honour an explicit replicas: 0.

`{{ .Values.replicas | default 1 }}` looks right and is not: Helm's
`default` substitutes for any EMPTY value, and 0 is empty. So
`replicas: 0` rendered as 1, and fontem-shared ran this API for weeks
while its values file said the tier was paused at zero.

That mattered beyond tidiness. The pause was how the data tier was meant
to be held still on clean stores, and believing it was held still while
it was not is the kind of thing you only discover by reading the
cluster. The sweeper and trigger templates already guarded with hasKey;
only this one was wrong, which is exactly why nobody noticed.

Skipped when helm is unavailable rather than silently passing.
"""

import re
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm not installed"
)

_DEPLOYMENTS = {
    "fontem-consolidator": "replicas",
    "consolidator-sweeper": "sweeperReplicas",
    "consolidator-trigger": "triggerReplicas",
}


def _render(**values) -> dict[str, int]:
    cmd = ["helm", "template", "t", "deployment"]
    for k, v in values.items():
        cmd += ["--set", f"{k}={v}"]
    out = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=".",
    ).stdout
    found: dict[str, int] = {}
    for doc in out.split("---"):
        if "kind: Deployment" not in doc:
            continue
        name = re.search(r"^\s+name:\s*(\S+)\s*$", doc, re.M)
        reps = re.search(r"^\s+replicas:\s*(\d+)\s*$", doc, re.M)
        if name and reps and name.group(1) in _DEPLOYMENTS:
            found[name.group(1)] = int(reps.group(1))
    return found


@pytest.mark.parametrize("deployment,key", list(_DEPLOYMENTS.items()))
def test_zero_replicas_is_honoured(deployment, key):
    """The regression. Every one of the three must be able to scale to
    zero from values, or 'paused' is a comment rather than a state."""
    assert _render(**{key: 0})[deployment] == 0


@pytest.mark.parametrize("deployment,key", list(_DEPLOYMENTS.items()))
def test_explicit_counts_pass_through(deployment, key):
    assert _render(**{key: 3})[deployment] == 3


def test_defaults_when_unset():
    """Unset keeps the API and trigger serving and the sweeper off — the
    sweeper is a continuous background rotation and opting into it should
    be deliberate."""
    rendered = _render()
    assert rendered["fontem-consolidator"] == 1
    assert rendered["consolidator-trigger"] == 1
    assert rendered["consolidator-sweeper"] == 0
