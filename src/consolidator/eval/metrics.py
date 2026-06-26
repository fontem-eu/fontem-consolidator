"""Confusion matrix + standard metrics for matcher evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int  # correctly linked (same real entity)
    fp: int  # falsely linked (different real entities — over-merge)
    fn: int  # missed link (same entity, matcher didn't link)
    tn: int  # correctly kept apart

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }
