"""
ml/anomaly_detection.py
------------------------
AI/ML layer for the MALE UAV Digital Twin (problem statement Section D).

Approach (plain-language explanation)
--------------------------------------
We do NOT train this on labeled "fault" examples, because in real life you
rarely have enough labeled failure data -- failures are, by definition, rare.

Instead we use an UNSUPERVISED anomaly detector (Isolation Forest): we show
it only NORMAL engine telemetry, and it learns the "shape" of normal
operation across all sensors at once (RPM, CHT, EGT, oil pressure, etc.).
When new telemetry comes in, the model checks how well it fits that learned
shape. A poor fit = anomaly, even if it's a type of fault the model has
never explicitly seen before -- which is exactly why this generalizes
better than fixed thresholds ("alert if CHT > 220C"), the problem
statement's whole point about moving beyond threshold-based monitoring.

Isolation Forest specifically works by randomly partitioning the data
repeatedly; points that are "few and different" (anomalies) get isolated
in fewer partitioning steps than normal points. It needs no fault labels,
trains fast, and handles multiple correlated sensors naturally -- a good
fit for a hackathon timeline.

What this module provides
--------------------------
- AnomalyDetector: fit on normal data, score new samples -> health_score (0-100) + anomaly flag
- RULEstimator: watches the health_score trend over time and extrapolates
  to estimate Remaining Useful Life (RUL) in hours
- A CLI demo: trains on a clean cruise run, then evaluates a run with an
  injected fault, and reports how quickly the model catches it
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulator.engine_simulator import EngineSimulator, MissionProfile, FaultType


# The sensor readings the model looks at. Order doesn't matter, but it must
# be consistent between training and scoring.
FEATURE_COLUMNS = [
    "rpm", "cylinder_head_temp_c", "exhaust_gas_temp_c",
    "oil_pressure_kpa", "oil_temp_c", "fuel_flow_lph",
    "vibration_g", "battery_voltage_v",
]


class AnomalyDetector:
    """Wraps an Isolation Forest + feature scaling into a simple fit/score API."""

    def __init__(self, contamination: float = 0.02):
        # contamination = the expected fraction of "weird" points even in
        # normal data (sensor noise, brief transients). 2% is a reasonable
        # starting point; tune it down if you get too many false alarms.
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        self.is_fitted = False

    def _extract(self, sample: dict) -> list[float]:
        return [sample[col] for col in FEATURE_COLUMNS]

    def fit(self, normal_samples: list[dict]) -> None:
        """Train on telemetry samples known to be from normal operation."""
        X = np.array([self._extract(s) for s in normal_samples])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

    def score(self, sample: dict) -> dict:
        """Scores one telemetry sample. Returns health_score (0-100, higher
        = healthier) and a boolean anomaly flag."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score() -- train on normal data first.")

        X = np.array([self._extract(sample)])
        X_scaled = self.scaler.transform(X)

        # decision_function: positive = looks normal, negative = looks anomalous.
        # predict: -1 means the model flags it as an outlier, 1 means normal.
        raw_score = float(self.model.decision_function(X_scaled)[0])
        is_anomaly = bool(self.model.predict(X_scaled)[0] == -1)

        return {
            "health_score": self._to_health_score(raw_score),
            "is_anomaly": is_anomaly,
            "raw_score": round(raw_score, 4),
        }

    @staticmethod
    def _to_health_score(raw_score: float) -> float:
        """Squashes the raw anomaly score into an intuitive 0-100 scale
        for the dashboard (100 = perfectly normal, 0 = severely anomalous)."""
        clipped = max(-0.5, min(0.5, raw_score))
        return round((clipped + 0.5) * 100, 1)

    def save(self, path: str) -> None:
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.is_fitted = True


# --------------------------------------------------------------------------
# RUL (Remaining Useful Life) estimation
# --------------------------------------------------------------------------

@dataclass
class RULEstimator:
    """
    Watches health_score over time and extrapolates the decline trend to
    estimate how long until the engine hits a critical health threshold.

    This is a simple linear-trend estimator -- good enough for a hackathon
    demo and easy to explain to judges. A production system would use
    something more sophisticated (e.g. a learned degradation curve per
    fault type), which is worth mentioning as a "future work" point.
    """
    window_size: int = 60          # how many recent readings to trend on
    critical_threshold: float = 20.0  # health_score considered "failure imminent"
    history: deque = field(default_factory=lambda: deque(maxlen=60))

    def update(self, sim_time_s: float, health_score: float) -> Optional[float]:
        """Feed in one new (time, health_score) point. Returns estimated
        RUL in seconds, or None if there's not enough data yet or health
        isn't declining."""
        self.history.append((sim_time_s, health_score))
        if len(self.history) < 10:
            return None

        times = np.array([t for t, _ in self.history])
        scores = np.array([s for _, s in self.history])

        # Fit a straight line: health_score = slope * time + intercept
        slope, intercept = np.polyfit(times, scores, 1)

        if slope >= 0:
            # health is flat or improving -- no meaningful RUL to report
            return None

        # Solve for the time at which the line crosses the critical threshold
        time_at_threshold = (self.critical_threshold - intercept) / slope
        rul_seconds = time_at_threshold - times[-1]
        return max(0.0, rul_seconds)


def summarize_degradation(points: list[tuple[float, float]], window_size: int = 60) -> dict:
    """Summarize recent health decline from ``(time_seconds, health_score)`` points."""
    if len(points) < 2:
        return {
            "state": "insufficient_data",
            "sample_count": len(points),
            "health_score": points[-1][1] if points else None,
            "degradation_rate_per_hour": None,
            "health_change": None,
            "predicted_health_1h": None,
            "predicted_health_6h": None,
            "trend_confidence": None,
        }

    recent = points[-window_size:]
    times = np.array([point[0] for point in recent])
    scores = np.array([point[1] for point in recent])
    slope_per_second = float(np.polyfit(times, scores, 1)[0])
    intercept = float(np.polyfit(times, scores, 1)[1])
    rate_per_hour = slope_per_second * 3600.0
    baseline_count = min(10, len(points))
    baseline = float(np.mean([score for _, score in points[:baseline_count]]))
    current = float(scores[-1])
    health_change = current - baseline
    predictions = {
        horizon: round(max(0.0, min(100.0, intercept + slope_per_second * (times[-1] + horizon * 3600))), 1)
        for horizon in (1, 6)
    }
    fitted = intercept + slope_per_second * times
    variance = float(np.sum((scores - np.mean(scores)) ** 2))
    residual_variance = float(np.sum((scores - fitted) ** 2))
    trend_confidence = 1.0 - (residual_variance / variance) if variance else 1.0

    if current < 40 or rate_per_hour <= -20:
        state = "critical"
    elif rate_per_hour <= -5:
        state = "degrading"
    else:
        state = "stable"

    return {
        "state": state,
        "sample_count": len(points),
        "health_score": round(current, 1),
        "degradation_rate_per_hour": round(rate_per_hour, 2),
        "health_change": round(health_change, 1),
        "predicted_health_1h": predictions[1],
        "predicted_health_6h": predictions[6],
        "trend_confidence": round(max(0.0, min(1.0, trend_confidence)), 2),
    }


# --------------------------------------------------------------------------
@dataclass
class AlertDebouncer:
    """
    Requires several CONSECUTIVE anomaly flags before raising a confirmed
    alert, instead of trusting any single noisy reading. This is standard
    practice in real fault-detection systems -- a single flickering sensor
    reading shouldn't trigger a maintenance alert, but a sustained pattern
    should. Dramatically cuts false positives with only a small increase
    in detection lag.
    """
    required_consecutive: int = 3
    _streak: int = 0

    def update(self, is_anomaly: bool) -> bool:
        """Feed in one raw anomaly flag. Returns True only when the
        required number of consecutive flags has been reached."""
        self._streak = self._streak + 1 if is_anomaly else 0
        return self._streak >= self.required_consecutive
# CLI demo: train on clean data, then evaluate a faulty run
# --------------------------------------------------------------------------

def generate_samples(mission: MissionProfile, duration_s: float,
                      fault: Optional[FaultType] = None, fault_onset: float = 60.0,
                      seed: int = 1, warmup_s: float = 150.0) -> list[dict]:
    """
    Runs the simulator and returns telemetry samples.

    `warmup_s` seconds are run FIRST but not returned/recorded -- this lets
    the engine reach steady-state cruise temperature before we start
    collecting data. Without this, the cold-start transient (CHT/EGT/oil
    temp all climbing from ambient) gets mixed into "normal" training data
    and confuses the anomaly detector, since a cold engine looks very
    different from a warm one even with zero faults present.
    """
    sim = EngineSimulator(mission=mission, dt=1.0, seed=seed)
    for _ in range(int(warmup_s)):
        sim.step()

    samples = []
    for _ in range(int(duration_s)):
        if fault and sim.sim_time >= (warmup_s + fault_onset) and not sim.active_faults:
            sim.inject_fault(fault)
        samples.append(sim.step())
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train + demo the anomaly detector")
    parser.add_argument("--train-duration", type=float, default=600.0,
                         help="seconds of normal data to train on")
    parser.add_argument("--test-duration", type=float, default=300.0)
    parser.add_argument("--fault", choices=[f.value for f in FaultType], default="overheating")
    parser.add_argument("--fault-onset", type=float, default=100.0)
    parser.add_argument("--model-out", type=str, default="ml/anomaly_model.joblib")
    args = parser.parse_args()

    print(f"Generating {args.train_duration:.0f}s of NORMAL telemetry to train on...")
    normal_samples = generate_samples(MissionProfile.CRUISE, args.train_duration, seed=1)

    detector = AnomalyDetector(contamination=0.02)
    detector.fit(normal_samples)
    detector.save(args.model_out)
    print(f"Model trained and saved to {args.model_out}\n")

    print(f"Generating {args.test_duration:.0f}s of test telemetry with a "
          f"'{args.fault}' fault injected at t={args.fault_onset:.0f}s...\n")
    test_samples = generate_samples(
        MissionProfile.CRUISE, args.test_duration,
        fault=FaultType(args.fault), fault_onset=args.fault_onset, seed=2,
    )

    rul_estimator = RULEstimator()
    actual_onset_time = test_samples[0]["timestamp"] + args.fault_onset - 1
    first_detection_after_onset = None
    false_positives_before_onset = 0

    for sample in test_samples:
        result = detector.score(sample)
        rul_seconds = rul_estimator.update(sample["timestamp"], result["health_score"])

        flag = " <-- ANOMALY DETECTED" if result["is_anomaly"] else ""
        if result["is_anomaly"]:
            if sample["timestamp"] >= actual_onset_time:
                if first_detection_after_onset is None:
                    first_detection_after_onset = sample["timestamp"]
            else:
                false_positives_before_onset += 1

        rul_str = f"RUL~{rul_seconds/60:.1f}min" if rul_seconds is not None else "RUL=n/a"
        print(f"t={sample['timestamp']:>5.0f}s  health={result['health_score']:>5.1f}  "
              f"{rul_str:<12} true_faults={sample['active_faults']}{flag}")

    print("\n--- Summary ---")
    print(f"Fault injected at:            t={actual_onset_time:.0f}s")
    if first_detection_after_onset is not None:
        detection_lag = first_detection_after_onset - actual_onset_time
        print(f"First post-fault anomaly at:  t={first_detection_after_onset:.0f}s "
              f"(detection lag: {detection_lag:.0f}s)")
    else:
        print("No anomaly was flagged after the fault -- try a longer test duration, "
              "or lower `contamination` for higher sensitivity.")
    print(f"False-positive flags before fault onset: {false_positives_before_onset} "
          f"(some baseline noise is expected -- that's what `contamination` controls)")


if __name__ == "__main__":
    main()
    