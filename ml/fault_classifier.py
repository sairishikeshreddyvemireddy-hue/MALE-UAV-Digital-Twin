"""
ml/fault_classifier.py
------------------------
A SECOND ML model, complementing anomaly_detection.py.

Why two models instead of one
-------------------------------
The Isolation Forest in anomaly_detection.py answers one question: "does
this look normal or not?" It's unsupervised on purpose -- it never needs
labeled fault examples, which is realistic since real engines rarely have
enough labeled failure history.

But once something IS flagged as abnormal, a maintenance engineer's next
question is "abnormal HOW?" -- which fault, how confident are we, and
which sensors are driving that conclusion? That needs a different kind of
model: a SUPERVISED CLASSIFIER, trained on labeled examples of each fault
type. We can only afford to do this because we have a physics-informed
SIMULATOR that can generate as much labeled data as we want for every
fault type -- something you normally can't do with a real engine.

This is a fair thing to say plainly in a demo: the anomaly detector proves
the "we don't need labels" case; the classifier proves the "if a digital
twin CAN generate labels, use them" case. Using both together is good
practice, not a shortcut.

What this module provides
--------------------------
- FaultClassifier: trained on many labeled examples (normal + every fault
  type), predicts which fault is most likely, with a confidence score
- Root-cause explanation: for any prediction, reports which specific
  sensors deviate most from normal (in standard deviations), translated
  into a human-readable sentence -- e.g. "driven by elevated EGT (+3.1sigma)
  and elevated CHT (+2.6sigma)". This is the "explainable AI" piece.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulator.engine_simulator import EngineSimulator, MissionProfile, FaultType

FEATURE_COLUMNS = [
    "rpm", "cylinder_head_temp_c", "exhaust_gas_temp_c",
    "oil_pressure_kpa", "oil_temp_c", "fuel_flow_lph",
    "vibration_g", "battery_voltage_v",
]

# Human-readable labels + direction word, used when building root-cause text
FEATURE_LABELS = {
    "rpm": "RPM",
    "cylinder_head_temp_c": "Cylinder Head Temp",
    "exhaust_gas_temp_c": "Exhaust Gas Temp",
    "oil_pressure_kpa": "Oil Pressure",
    "oil_temp_c": "Oil Temp",
    "fuel_flow_lph": "Fuel Flow",
    "vibration_g": "Vibration",
    "battery_voltage_v": "Battery Voltage",
}


class FaultClassifier:
    """Supervised multi-class classifier: normal + each fault type."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        self.is_fitted = False
        # mean/std of EACH feature under NORMAL operation only -- used to
        # compute how many standard deviations a live reading has drifted,
        # which is what makes the root-cause explanation meaningful.
        self.normal_mean = None
        self.normal_std = None

    def _extract(self, sample: dict) -> list[float]:
        return [sample[col] for col in FEATURE_COLUMNS]

    def fit(self, samples: list[dict], labels: list[str]) -> None:
        X = np.array([self._extract(s) for s in samples])
        y = np.array(labels)
        self.model.fit(X, y)

        normal_X = X[y == "normal"]
        self.normal_mean = normal_X.mean(axis=0)
        self.normal_std = normal_X.std(axis=0) + 1e-6  # avoid divide-by-zero
        self.is_fitted = True

    def predict(self, sample: dict) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        x = np.array([self._extract(sample)])
        predicted = self.model.predict(x)[0]
        proba = self.model.predict_proba(x)[0]
        confidence = float(proba.max())

        # class-by-class confidence, sorted, for anyone who wants the full picture
        class_confidences = {
            cls: round(float(p), 3)
            for cls, p in sorted(zip(self.model.classes_, proba), key=lambda t: -t[1])
        }

        root_cause = self._explain(x[0]) if predicted != "normal" else "No significant deviation from normal operation."

        return {
            "predicted_fault": predicted,
            "confidence": round(confidence, 3),
            "class_confidences": class_confidences,
            "root_cause": root_cause,
        }

    def _explain(self, x_row: np.ndarray) -> str:
        """Builds a plain-English root-cause sentence from the sensors that
        deviate most from normal, in standard deviations (z-scores)."""
        z_scores = (x_row - self.normal_mean) / self.normal_std
        # rank features by |z-score|, take the top 2 as the "reason"
        ranked = sorted(
            zip(FEATURE_COLUMNS, z_scores), key=lambda t: -abs(t[1])
        )[:2]

        parts = []
        for col, z in ranked:
            direction = "elevated" if z > 0 else "reduced"
            parts.append(f"{direction} {FEATURE_LABELS[col]} ({z:+.1f}\u03c3)")

        return "Primarily driven by " + " and ".join(parts) + "."

    def save(self, path: str) -> None:
        joblib.dump({
            "model": self.model,
            "normal_mean": self.normal_mean,
            "normal_std": self.normal_std,
        }, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.normal_mean = data["normal_mean"]
        self.normal_std = data["normal_std"]
        self.is_fitted = True


# --------------------------------------------------------------------------
# Training data generation -- this is only possible because we HAVE a
# simulator. A real engine wouldn't give you this many labeled examples.
# --------------------------------------------------------------------------

def generate_labeled_dataset(
    per_class_duration: float = 400.0, warmup_s: float = 150.0, seed_base: int = 100
) -> tuple[list[dict], list[str]]:
    samples, labels = [], []

    # "normal" class: draw from every mission profile so the classifier
    # doesn't confuse a legitimate hot-weather or high-altitude reading
    # with an actual fault.
    normal_profiles = [MissionProfile.CRUISE, MissionProfile.HIGH_ALTITUDE,
                        MissionProfile.HOT_WEATHER, MissionProfile.RAPID_THROTTLE]
    for i, profile in enumerate(normal_profiles):
        sim = EngineSimulator(mission=profile, dt=1.0, seed=seed_base + i)
        for _ in range(int(warmup_s)):
            sim.step()
        for _ in range(int(per_class_duration)):
            samples.append(sim.step())
            labels.append("normal")

    # each fault class: warm up, inject the fault, then only keep samples
    # from AFTER it's fully developed (severity ~1) -- we want the
    # classifier to learn the fault's steady signature, not its onset ramp.
    for i, fault in enumerate(FaultType):
        if fault == FaultType.NONE:
            continue
        sim = EngineSimulator(mission=MissionProfile.CRUISE, dt=1.0, seed=seed_base + 50 + i)
        for _ in range(int(warmup_s)):
            sim.step()
        sim.inject_fault(fault, ramp_seconds=60.0)
        for _ in range(90):  # let it fully develop before recording
            sim.step()
        for _ in range(int(per_class_duration)):
            samples.append(sim.step())
            labels.append(fault.value)

    return samples, labels


def main():
    parser = argparse.ArgumentParser(description="Train + demo the fault classifier")
    parser.add_argument("--model-out", type=str, default="ml/fault_classifier.joblib")
    args = parser.parse_args()

    print("Generating labeled training data (normal + all fault types)...")
    samples, labels = generate_labeled_dataset()
    print(f"  {len(samples)} labeled samples across {len(set(labels))} classes\n")

    clf = FaultClassifier()
    clf.fit(samples, labels)
    clf.save(args.model_out)
    print(f"Model trained and saved to {args.model_out}\n")

    print("--- Quick validation: one fresh example per fault type ---")
    correct = 0
    total = 0
    for fault in FaultType:
        if fault == FaultType.NONE:
            continue
        sim = EngineSimulator(mission=MissionProfile.CRUISE, dt=1.0, seed=999)
        for _ in range(150):
            sim.step()
        sim.inject_fault(fault, ramp_seconds=60.0)
        for _ in range(120):
            sample = sim.step()

        result = clf.predict(sample)
        hit = "✓" if result["predicted_fault"] == fault.value else "✗"
        correct += (result["predicted_fault"] == fault.value)
        total += 1
        print(f"{hit} true={fault.value:<24} predicted={result['predicted_fault']:<24} "
              f"confidence={result['confidence']:.2f}")
        print(f"    reason: {result['root_cause']}")

    print(f"\nValidation accuracy: {correct}/{total}")


if __name__ == "__main__":
    main()