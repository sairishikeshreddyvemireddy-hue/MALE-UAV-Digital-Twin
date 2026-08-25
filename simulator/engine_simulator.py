"""
engine_simulator.py
--------------------
Synthetic telemetry generator for an aero piston engine used in a MALE UAV.

Purpose
-------
Real engine/CAN-bus data is not available during development, so this module
produces physically-plausible, time-stepped telemetry that the rest of the
Digital Twin stack (backend ingestion, ML anomaly detection, dashboard) can
consume exactly as if it came from a real ECU/FADEC over SocketCAN.

Design notes
------------
- The engine is modeled as a small set of coupled first-order dynamical
  systems (thermal lag, RPM response, vibration) rather than a full
  thermodynamic cycle model -- enough physical realism to produce believable
  trends and fault signatures without needing a full combustion simulation.
- Faults are injected as *state perturbations*, not just noise, so the
  resulting telemetry has the same kind of signature a real fault would
  produce (e.g. an overheating trend has a time constant, not a step jump).
- The simulator is a generator: `for sample in sim.run(): ...` yields one
  telemetry dict per timestep. This makes it trivial to plug into an asyncio
  queue, a websocket broadcaster, a CSV logger, or a CAN-frame packer later.
"""

from __future__ import annotations

import json
import math
import random
import time
import argparse
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Iterator, Optional


# --------------------------------------------------------------------------
# Fault model
# --------------------------------------------------------------------------

class FaultType(str, Enum):
    NONE = "none"
    MISFIRE = "misfire"
    INJECTOR_ABNORMALITY = "injector_abnormality"
    COOLING_DEGRADATION = "cooling_degradation"
    LUBRICATION_ISSUE = "lubrication_issue"
    SENSOR_DRIFT = "sensor_drift"
    COMBUSTION_INSTABILITY = "combustion_instability"
    OVERHEATING = "overheating"
    ABNORMAL_VIBRATION = "abnormal_vibration"


@dataclass
class ActiveFault:
    fault_type: FaultType
    severity: float          # 0.0 (just started) -> 1.0 (fully developed)
    onset_time: float        # simulation time (s) the fault started
    ramp_seconds: float = 60.0  # how long it takes to reach full severity

    def progression(self, sim_time: float) -> float:
        """Returns 0..1 severity based on elapsed time since onset."""
        elapsed = max(0.0, sim_time - self.onset_time)
        return min(1.0, elapsed / self.ramp_seconds)


# --------------------------------------------------------------------------
# Mission profile (drives ambient conditions + throttle demand over time)
# --------------------------------------------------------------------------

class MissionProfile(str, Enum):
    CRUISE = "cruise"
    HIGH_ALTITUDE = "high_altitude"
    ENDURANCE = "endurance"
    HOT_WEATHER = "hot_weather"
    RAPID_THROTTLE = "rapid_throttle_transitions"


@dataclass
class MissionConditions:
    ambient_temp_c: float
    altitude_m: float
    throttle_pct: float  # 0-100


def mission_conditions(profile: MissionProfile, sim_time: float) -> MissionConditions:
    """Computes ambient/throttle conditions for a given mission profile at time t (s)."""
    if profile == MissionProfile.CRUISE:
        return MissionConditions(ambient_temp_c=15.0, altitude_m=3000, throttle_pct=65.0)

    if profile == MissionProfile.HIGH_ALTITUDE:
        # Thinner air -> lower ambient temp, engine has to work harder for same power
        return MissionConditions(ambient_temp_c=-10.0, altitude_m=7500, throttle_pct=78.0)

    if profile == MissionProfile.ENDURANCE:
        # Long steady cruise, very slow throttle drift as fuel burns off / weight drops
        drift = 2.0 * math.sin(sim_time / 1800.0)
        return MissionConditions(ambient_temp_c=10.0, altitude_m=4000, throttle_pct=60.0 + drift)

    if profile == MissionProfile.HOT_WEATHER:
        return MissionConditions(ambient_temp_c=42.0, altitude_m=500, throttle_pct=70.0)

    if profile == MissionProfile.RAPID_THROTTLE:
        # Square-wave-ish throttle transitions to stress thermal/RPM response
        period = 40.0
        phase = (sim_time % period) / period
        throttle = 85.0 if phase < 0.5 else 35.0
        return MissionConditions(ambient_temp_c=20.0, altitude_m=2000, throttle_pct=throttle)

    raise ValueError(f"Unknown mission profile: {profile}")


# --------------------------------------------------------------------------
# Engine simulator
# --------------------------------------------------------------------------

@dataclass
class EngineParams:
    """Nominal steady-state engine characteristics (tune to your real engine)."""
    idle_rpm: float = 1800.0
    max_rpm: float = 6500.0
    nominal_cht_c: float = 175.0     # cylinder head temp at cruise throttle
    nominal_egt_c: float = 780.0
    nominal_oil_press_kpa: float = 350.0
    nominal_oil_temp_c: float = 95.0
    nominal_fuel_flow_lph: float = 14.0  # litres/hour at cruise
    nominal_vibration_g: float = 0.35    # RMS g at cruise
    battery_voltage_v: float = 28.0


@dataclass
class EngineSimulator:
    params: EngineParams = field(default_factory=EngineParams)
    dt: float = 1.0                       # seconds per timestep
    mission: MissionProfile = MissionProfile.CRUISE
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self.sim_time = 0.0

        # Internal state (what actually gets integrated step to step)
        self.rpm = self.params.idle_rpm
        self.cht_c = 60.0     # start cold
        self.egt_c = 60.0
        self.oil_temp_c = 40.0
        self.oil_press_kpa = self.params.nominal_oil_press_kpa
        self.fuel_flow_lph = 0.0
        self.vibration_g = 0.05
        self.injection_timing_deg = 22.0  # deg BTDC nominal

        self.active_faults: list[ActiveFault] = []

    # ---- fault injection API, called externally (e.g. from a test harness
    # or an operator UI) to simulate a developing failure ------------------
    def inject_fault(self, fault_type: FaultType, ramp_seconds: float = 90.0) -> None:
        self.active_faults.append(
            ActiveFault(fault_type=fault_type, severity=0.0,
                        onset_time=self.sim_time, ramp_seconds=ramp_seconds)
        )

    def clear_faults(self) -> None:
        self.active_faults = []

    def _fault_severity(self, fault_type: FaultType) -> float:
        return max(
            (f.progression(self.sim_time) for f in self.active_faults if f.fault_type == fault_type),
            default=0.0,
        )

    # ---- core step function -----------------------------------------------
    def step(self) -> dict:
        cond = mission_conditions(self.mission, self.sim_time)

        # --- RPM: first-order approach toward a throttle-commanded target ---
        target_rpm = self.params.idle_rpm + (cond.throttle_pct / 100.0) * (
            self.params.max_rpm - self.params.idle_rpm
        )
        misfire_sev = self._fault_severity(FaultType.MISFIRE)
        combustion_sev = self._fault_severity(FaultType.COMBUSTION_INSTABILITY)
        rpm_tau = 3.0  # seconds, response lag
        self.rpm += (target_rpm - self.rpm) * (self.dt / rpm_tau)
        # misfire causes periodic RPM sag
        if misfire_sev > 0:
            self.rpm -= misfire_sev * 250.0 * (0.5 + 0.5 * math.sin(self.sim_time * 4.0))
        self.rpm += self._rng.gauss(0, 15 + 40 * combustion_sev)
        self.rpm = max(0.0, self.rpm)

        # --- Fuel flow: scales with RPM/throttle; injector faults distort it ---
        injector_sev = self._fault_severity(FaultType.INJECTOR_ABNORMALITY)
        base_fuel = self.params.nominal_fuel_flow_lph * (cond.throttle_pct / 65.0)
        self.fuel_flow_lph = base_fuel * (1.0 + injector_sev * self._rng.uniform(-0.3, 0.5))
        self.fuel_flow_lph = max(0.0, self.fuel_flow_lph + self._rng.gauss(0, 0.15))

        # --- Thermal: CHT/EGT track RPM & ambient with lag; cooling fault slows shedding ---
        cooling_sev = self._fault_severity(FaultType.COOLING_DEGRADATION)
        overheat_sev = self._fault_severity(FaultType.OVERHEATING)
        cht_target = cond.ambient_temp_c + self.params.nominal_cht_c * (self.rpm / self.params.max_rpm) * 1.3
        cht_target += overheat_sev * 60.0
        cht_tau = 45.0 * (1.0 + cooling_sev * 1.5)  # degraded cooling = slower to shed heat = higher effective temp
        self.cht_c += (cht_target - self.cht_c) * (self.dt / cht_tau)
        self.cht_c += self._rng.gauss(0, 0.4)

        egt_target = cond.ambient_temp_c + self.params.nominal_egt_c * (self.rpm / self.params.max_rpm) * 1.2
        egt_target += injector_sev * 80.0 + combustion_sev * 100.0
        self.egt_c += (egt_target - self.egt_c) * (self.dt / 20.0)
        self.egt_c += self._rng.gauss(0, 2.0)

        # --- Oil system: lubrication fault drops pressure, raises temp ---
        lube_sev = self._fault_severity(FaultType.LUBRICATION_ISSUE)
        oil_press_target = self.params.nominal_oil_press_kpa * (0.6 + 0.4 * (self.rpm / self.params.max_rpm))
        oil_press_target *= (1.0 - 0.7 * lube_sev)
        self.oil_press_kpa += (oil_press_target - self.oil_press_kpa) * (self.dt / 10.0)
        self.oil_press_kpa += self._rng.gauss(0, 3.0)

        oil_temp_target = self.cht_c * 0.55 + 10.0 + lube_sev * 25.0
        self.oil_temp_c += (oil_temp_target - self.oil_temp_c) * (self.dt / 60.0)
        self.oil_temp_c += self._rng.gauss(0, 0.3)

        # --- Vibration: baseline scales with RPM; several faults raise it ---
        vib_sev = self._fault_severity(FaultType.ABNORMAL_VIBRATION)
        vibration_target = self.params.nominal_vibration_g * (self.rpm / self.params.max_rpm) * 1.1
        vibration_target += (misfire_sev + combustion_sev + vib_sev + 0.5 * lube_sev) * 0.8
        self.vibration_g += (vibration_target - self.vibration_g) * (self.dt / 5.0)
        self.vibration_g = max(0.0, self.vibration_g + self._rng.gauss(0, 0.02))

        # --- Injection timing + sensor drift (affects reported value, not truth) ---
        drift_sev = self._fault_severity(FaultType.SENSOR_DRIFT)
        true_timing = 22.0 - 4.0 * (self.rpm / self.params.max_rpm)
        self.injection_timing_deg = true_timing

        # --- Battery/alternator: dips slightly under high electrical/thermal load ---
        battery_v = self.params.battery_voltage_v - 0.4 * (self.oil_temp_c > 110) + self._rng.gauss(0, 0.05)

        self.sim_time += self.dt

        sample = {
            "timestamp": round(self.sim_time, 2),
            "mission_profile": self.mission.value,
            "ambient_temp_c": round(cond.ambient_temp_c, 2),
            "altitude_m": cond.altitude_m,
            "throttle_pct": round(cond.throttle_pct, 1),
            "rpm": round(self.rpm, 1),
            "cylinder_head_temp_c": round(self.cht_c, 2),
            "exhaust_gas_temp_c": round(self.egt_c, 2),
            "oil_pressure_kpa": round(self.oil_press_kpa, 2),
            "oil_temp_c": round(self.oil_temp_c, 2),
            "fuel_flow_lph": round(self.fuel_flow_lph, 3),
            "vibration_g": round(self.vibration_g, 4),
            "injection_timing_deg": round(
                self.injection_timing_deg + (drift_sev * self._rng.uniform(2, 6)), 2
            ),
            "battery_voltage_v": round(battery_v, 2),
            "active_faults": [f.fault_type.value for f in self.active_faults],
        }
        return sample

    def run(self, duration_s: Optional[float] = None, realtime: bool = False) -> Iterator[dict]:
        """
        Yields telemetry samples. If duration_s is None, runs forever
        (useful for a live streaming backend). If realtime=True, sleeps
        `dt` seconds between samples so the stream mirrors wall-clock time;
        otherwise it runs as fast as possible (useful for batch data
        generation / ML training sets).
        """
        elapsed = 0.0
        while duration_s is None or elapsed < duration_s:
            sample = self.step()
            yield sample
            elapsed += self.dt
            if realtime:
                time.sleep(self.dt)


# --------------------------------------------------------------------------
# CLI entry point: quick standalone smoke test / data generation
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aero piston engine telemetry simulator")
    parser.add_argument("--mission", choices=[m.value for m in MissionProfile], default="cruise")
    parser.add_argument("--duration", type=float, default=120.0, help="simulated seconds")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--realtime", action="store_true", help="sleep dt seconds between samples")
    parser.add_argument("--inject-fault", choices=[f.value for f in FaultType], default=None)
    parser.add_argument("--fault-onset", type=float, default=30.0, help="sim time (s) to start the fault")
    parser.add_argument("--out", type=str, default=None, help="write JSONL to this path instead of stdout")
    args = parser.parse_args()

    sim = EngineSimulator(mission=MissionProfile(args.mission), dt=args.dt, seed=42)

    out_fh = open(args.out, "w") if args.out else None
    fault_injected = False

    for sample in sim.run(duration_s=args.duration, realtime=args.realtime):
        if args.inject_fault and not fault_injected and sim.sim_time >= args.fault_onset:
            sim.inject_fault(FaultType(args.inject_fault))
            fault_injected = True

        line = json.dumps(sample)
        if out_fh:
            out_fh.write(line + "\n")
        else:
            print(line)

    if out_fh:
        out_fh.close()


if __name__ == "__main__":
    main()