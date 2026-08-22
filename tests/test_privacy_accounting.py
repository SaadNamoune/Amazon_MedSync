import sys
from pathlib import Path

from opacus.accountants.utils import get_noise_multiplier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.federation.privacy_accounting import CumulativePrivacyTracker  # noqa: E402


def test_no_rounds_recorded_gives_zero_epsilon():
    tracker = CumulativePrivacyTracker(target_delta=1e-5)
    assert tracker.cumulative_epsilon() == 0.0


def test_cumulative_epsilon_grows_with_more_rounds():
    tracker = CumulativePrivacyTracker(target_delta=1e-5)

    tracker.record_round(noise_multiplier=1.0, sample_rate=0.1, steps=10)
    eps_after_one_round = tracker.cumulative_epsilon()
    assert eps_after_one_round > 0

    tracker.record_round(noise_multiplier=1.0, sample_rate=0.1, steps=10)
    eps_after_two_rounds = tracker.cumulative_epsilon()

    # Composing a second round of the same (noise_multiplier, sample_rate)
    # queries must cost strictly more privacy than stopping after the first --
    # this is the whole point of tracking cumulative spend instead of just
    # reporting each round's own isolated epsilon.
    assert eps_after_two_rounds > eps_after_one_round


def test_cumulative_epsilon_exceeds_any_single_round_target():
    # Mirrors LocalClient's real calibration: each round independently hits
    # target_epsilon=1.0 in isolation. The true cost of querying the same
    # data repeatedly across rounds must compose to something higher than
    # that single-round target once there's more than one round -- this is
    # the exact real-world gap CumulativePrivacyTracker exists to surface.
    target_delta = 1e-5
    sample_rate, epochs = 0.1, 1
    noise_multiplier = get_noise_multiplier(
        target_epsilon=1.0, target_delta=target_delta, sample_rate=sample_rate, epochs=epochs,
    )
    steps_per_round = round(epochs / sample_rate)

    tracker = CumulativePrivacyTracker(target_delta=target_delta)
    for _ in range(5):
        tracker.record_round(noise_multiplier, sample_rate, steps_per_round)
    assert tracker.cumulative_epsilon() > 1.0
