"""
Cross-round privacy composition. LocalClient.train_round() reports a
per-round epsilon (a fresh accountant, calibrated to hit target_epsilon
for that round in isolation) -- that answers "how much privacy did THIS
round's query cost", which is what the target_epsilon knob controls, and
it's the number the rest of the pipeline (README, monitoring dashboard)
has been reporting throughout this project's experiment history.

It does NOT answer "how much privacy has this node spent in total over
every round so far" -- since every round queries the same underlying
patient data, that total keeps composing round over round, and it's the
number that actually matters for a real privacy guarantee to a data
subject. This module tracks that: one CumulativePrivacyTracker per node,
fed one (noise_multiplier, sample_rate, steps) entry per round, producing
the properly RDP-composed total epsilon over all rounds so far.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from opacus.accountants import RDPAccountant


@dataclass
class CumulativePrivacyTracker:
    """One per hospital node. Call record_round() after each of that
    node's training rounds, then cumulative_epsilon() for the properly
    composed total spent so far (not just the latest round's own figure)."""

    target_delta: float = 1e-5
    _history: List[Tuple[float, float, int]] = field(default_factory=list)

    def record_round(self, noise_multiplier: float, sample_rate: float, steps: int) -> None:
        self._history.append((noise_multiplier, sample_rate, steps))

    def cumulative_epsilon(self) -> float:
        if not self._history:
            return 0.0
        accountant = RDPAccountant()
        for noise_multiplier, sample_rate, steps in self._history:
            for _ in range(steps):
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return accountant.get_epsilon(delta=self.target_delta)
