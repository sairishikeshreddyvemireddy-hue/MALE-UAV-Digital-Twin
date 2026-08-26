"""
backend/main.py
----------------
FastAPI backend for the MALE UAV Digital Twin.

What this file does, in plain terms:
1. Lets you START a "mission" (a simulated engine run).
2. Runs the engine simulator in the background, once per second.
3. SAVES every reading into the database (via database/models.py).
4. STREAMS every reading live to any connected dashboard over a WebSocket,
   so a browser can show real-time charts without constantly re-asking
   the server "any new data yet?" (that's what WebSockets are for).
5. Lets you inject a fault into a running mission to test the pipeline
   end-to-end (simulator -> database -> live dashboard -> later, ML).

Beginner notes on the concepts used here:
- "async def" functions are ones that can pause (e.g. `await asyncio.sleep`)
  without blocking the whole server -- this is how we run the simulator
  loop and handle web requests at the same time.
- A WebSocket is a persistent two-way connection (unlike a normal web
  request, which opens, gets one answer, and closes). The dashboard opens
  one WebSocket connection and just keeps receiving messages as they arrive.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# NOTE: if your database file is named database/model.py (singular) rather
# than database/models.py, change the import below to:
#     from database.model import ...
from database.models import (
    init_db, get_db, SessionLocal,
    Engine, Mission, TelemetryReading, FaultEvent, HealthIndex,
)
from simulator.engine_simulator import EngineSimulator, MissionProfile, FaultType
from ml.anomaly_detection import AnomalyDetector, RULEstimator, AlertDebouncer, summarize_degradation
from ml.fault_classifier import FaultClassifier


app = FastAPI(title="MALE UAV Digital Twin API")

# Loaded once at startup below. If a model file doesn't exist yet, the
# corresponding `_ready` flag stays False and the backend simply skips
# that piece of ML -- the rest of the system keeps working without it.
ml_detector = AnomalyDetector()
ml_ready = False
fault_clf = FaultClassifier()
clf_ready = False

# Allows a browser-based dashboard (running on a different port/origin) to
# call this API. Wide open here for hackathon convenience -- you'd lock
# this down to specific origins in a production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

@app.on_event("startup")
def on_startup():
    init_db()

    global ml_ready, clf_ready
    try:
        ml_detector.load("ml/anomaly_model.joblib")
        ml_ready = True
        print("[startup] ML anomaly model loaded -- live health scoring is ACTIVE.")
    except Exception as e:
        ml_ready = False
        print(f"[startup] No anomaly model loaded ({e}) -- health scoring will be skipped. "
              f"Run `python ml/anomaly_detection.py` first to train one.")

    try:
        fault_clf.load("ml/fault_classifier.joblib")
        clf_ready = True
        print("[startup] Fault classifier loaded -- fault diagnosis is ACTIVE.")
    except Exception as e:
        clf_ready = False
        print(f"[startup] No fault classifier loaded ({e}) -- diagnosis will be skipped. "
              f"Run `python ml/fault_classifier.py` first to train one.")



# --------------------------------------------------------------------------
# In-memory state: tracks currently-running simulations and connected
# dashboard clients. A hackathon demo doesn't need Redis/Celery for this --
# plain Python dicts are enough since it's a single backend process.
# --------------------------------------------------------------------------

class RunningMission:
    def __init__(self, mission_id: int, simulator: EngineSimulator):
        self.mission_id = mission_id
        self.simulator = simulator
        self.task: asyncio.Task | None = None
        self.active = True
        # Per-mission ML state -- each running mission gets its own RUL
        # trend tracker and its own "how many anomalies in a row" counter,
        # since mixing missions together would produce nonsense trends.
        self.rul_estimator = RULEstimator()
        self.debouncer = AlertDebouncer(required_consecutive=3)
        self.health_history: list[tuple[float, float]] = []


active_missions: Dict[int, RunningMission] = {}


class ConnectionManager:
    """Keeps track of which dashboard WebSocket clients are watching which mission."""

    def __init__(self):
        self.connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, mission_id: int, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(mission_id, []).append(ws)

    def disconnect(self, mission_id: int, ws: WebSocket):
        if mission_id in self.connections and ws in self.connections[mission_id]:
            self.connections[mission_id].remove(ws)

    async def broadcast(self, mission_id: int, message: dict):
        for ws in list(self.connections.get(mission_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                # client disconnected mid-broadcast; drop it silently
                self.disconnect(mission_id, ws)


manager = ConnectionManager()


# --------------------------------------------------------------------------
# The background loop: runs the simulator, saves to DB, broadcasts live
# --------------------------------------------------------------------------

FAULT_LOG_THRESHOLD = 0.05  # log a FaultEvent once severity crosses this

async def run_simulation_loop(mission_id: int, running: RunningMission):
    db: Session = SessionLocal()
    logged_faults: set[str] = set()

    try:
        while running.active:
            sample = running.simulator.step()

            # --- persist to database ---
            reading = TelemetryReading(
                mission_id=mission_id,
                sim_time_s=sample["timestamp"],
                rpm=sample["rpm"],
                cht_c=sample["cylinder_head_temp_c"],
                egt_c=sample["exhaust_gas_temp_c"],
                oil_pressure_kpa=sample["oil_pressure_kpa"],
                oil_temp_c=sample["oil_temp_c"],
                fuel_flow_lph=sample["fuel_flow_lph"],
                vibration_g=sample["vibration_g"],
                injection_timing_deg=sample["injection_timing_deg"],
                battery_voltage_v=sample["battery_voltage_v"],
            )
            db.add(reading)

            # log any newly-active fault once, the first time we see it
            for fault_type in sample["active_faults"]:
                if fault_type not in logged_faults:
                    db.add(FaultEvent(
                        mission_id=mission_id,
                        fault_type=fault_type,
                        onset_time_s=sample["timestamp"],
                        severity=0.0,
                        detected_at=datetime.utcnow(),
                    ))
                    logged_faults.add(fault_type)

            # --- ML scoring: run the trained anomaly detector on this ---
            # --- exact reading, live, the moment it's generated         ---
            ml_payload = {}
            if ml_ready:
                result = ml_detector.score(sample)
                rul_seconds = running.rul_estimator.update(sample["timestamp"], result["health_score"])
                running.health_history.append((sample["timestamp"], result["health_score"]))
                degradation = summarize_degradation(running.health_history)
                confirmed_alert = running.debouncer.update(result["is_anomaly"])
                rul_hours = (rul_seconds / 3600.0) if rul_seconds is not None else None

                db.add(HealthIndex(
                    mission_id=mission_id,
                    sim_time_s=sample["timestamp"],
                    health_score=result["health_score"],
                    rul_hours=rul_hours,
                ))

                ml_payload = {
                    "health_score": result["health_score"],
                    "is_anomaly": result["is_anomaly"],
                    "confirmed_alert": confirmed_alert,
                    "rul_hours": rul_hours,
                    "degradation": degradation,
                }

                # Only run the fault classifier once an alert is CONFIRMED
                # (not on every single flickering raw flag) -- diagnosing
                # "which fault" only makes sense once we're confident
                # something is actually wrong.
                if confirmed_alert and clf_ready:
                    diagnosis = fault_clf.predict(sample)
                    ml_payload["predicted_fault"] = diagnosis["predicted_fault"]
                    ml_payload["diagnosis_confidence"] = diagnosis["confidence"]
                    ml_payload["root_cause"] = diagnosis["root_cause"]

            db.commit()

            # --- stream live to any connected dashboards ---
            # (ML fields are merged in here, alongside the raw sensor
            # reading, so the dashboard gets everything in one message)
            await manager.broadcast(mission_id, {**sample, **ml_payload})

            await asyncio.sleep(running.simulator.dt)
    finally:
        db.close()


# --------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------

@app.post("/engines")
def create_engine(serial_number: str, model: str = "", uav_id: str = "", db: Session = Depends(get_db)):
    """Register an engine before you can start missions for it."""
    engine = Engine(serial_number=serial_number, model=model, uav_id=uav_id)
    db.add(engine)
    db.commit()
    db.refresh(engine)
    return {"id": engine.id, "serial_number": engine.serial_number}

@app.get("/engines")
def list_engines(db: Session = Depends(get_db)):
    """Lists all registered engines -- used by the dashboard to populate
    the engine selector when starting a new mission."""
    rows = db.query(Engine).all()
    return [
        {"id": e.id, "serial_number": e.serial_number, "model": e.model, "uav_id": e.uav_id}
        for e in rows
    ]
@app.post("/missions/start")
async def start_mission(engine_id: int, profile: str = "cruise", db: Session = Depends(get_db)):
    """
    Starts a new simulated mission: creates a Mission row and kicks off the
    background loop that generates, saves, and streams telemetry.
    """
    engine_row = db.query(Engine).filter(Engine.id == engine_id).first()
    if not engine_row:
        raise HTTPException(status_code=404, detail="Engine not found. Create one via POST /engines first.")

    try:
        mission_profile = MissionProfile(profile)
    except ValueError:
        valid = [p.value for p in MissionProfile]
        raise HTTPException(status_code=400, detail=f"Invalid profile. Choose one of: {valid}")

    mission = Mission(engine_id=engine_id, profile=profile, start_time=datetime.utcnow())
    db.add(mission)
    db.commit()
    db.refresh(mission)

    simulator = EngineSimulator(mission=mission_profile, dt=1.0)
    running = RunningMission(mission_id=mission.id, simulator=simulator)
    running.task = asyncio.create_task(run_simulation_loop(mission.id, running))
    active_missions[mission.id] = running

    return {"mission_id": mission.id, "profile": profile, "status": "running"}


@app.post("/missions/{mission_id}/stop")
def stop_mission(mission_id: int, db: Session = Depends(get_db)):
    running = active_missions.get(mission_id)
    if not running:
        raise HTTPException(status_code=404, detail="No active simulation for this mission_id.")
    running.active = False

    mission = db.query(Mission).filter(Mission.id == mission_id).first()
    if mission:
        mission.end_time = datetime.utcnow()
        db.commit()

    del active_missions[mission_id]
    return {"mission_id": mission_id, "status": "stopped"}


@app.post("/missions/{mission_id}/inject-fault")
def inject_fault(mission_id: int, fault_type: str):
    """Triggers a fault in a currently-running mission -- useful for demoing
    the fault detection pipeline live to judges."""
    running = active_missions.get(mission_id)
    if not running:
        raise HTTPException(status_code=404, detail="No active simulation for this mission_id.")
    try:
        ft = FaultType(fault_type)
    except ValueError:
        valid = [f.value for f in FaultType]
        raise HTTPException(status_code=400, detail=f"Invalid fault_type. Choose one of: {valid}")

    running.simulator.inject_fault(ft)
    return {"mission_id": mission_id, "fault_injected": fault_type}


@app.get("/missions/{mission_id}/telemetry")
def get_telemetry(mission_id: int, limit: int = 100, db: Session = Depends(get_db)):
    """Returns the most recent stored telemetry rows for a mission."""
    rows = (
        db.query(TelemetryReading)
        .filter(TelemetryReading.mission_id == mission_id)
        .order_by(TelemetryReading.sim_time_s.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "sim_time_s": r.sim_time_s, "rpm": r.rpm, "cht_c": r.cht_c, "egt_c": r.egt_c,
            "oil_pressure_kpa": r.oil_pressure_kpa, "oil_temp_c": r.oil_temp_c,
            "fuel_flow_lph": r.fuel_flow_lph, "vibration_g": r.vibration_g,
            "battery_voltage_v": r.battery_voltage_v,
        }
        for r in reversed(rows)
    ]


@app.get("/missions/{mission_id}/health")
def get_health_history(mission_id: int, limit: int = 200, db: Session = Depends(get_db)):
    """Returns the most recent ML-computed health scores for a mission."""
    rows = (
        db.query(HealthIndex)
        .filter(HealthIndex.mission_id == mission_id)
        .order_by(HealthIndex.sim_time_s.desc())
        .limit(limit)
        .all()
    )
    return [
        {"sim_time_s": r.sim_time_s, "health_score": r.health_score, "rul_hours": r.rul_hours}
        for r in reversed(rows)
    ]


@app.get("/missions/{mission_id}/degradation")
def get_degradation_summary(mission_id: int, limit: int = 200, db: Session = Depends(get_db)):
    """Returns the current health-degradation state for a completed or active mission."""
    rows = (
        db.query(HealthIndex)
        .filter(HealthIndex.mission_id == mission_id)
        .order_by(HealthIndex.sim_time_s.desc())
        .limit(limit)
        .all()
    )
    points = [(row.sim_time_s, row.health_score) for row in reversed(rows)]
    return summarize_degradation(points)


@app.get("/missions/{mission_id}/degradation/trend")
def get_degradation_trend(mission_id: int, limit: int = 200, db: Session = Depends(get_db)):
    """Returns projected health scores based on the recent degradation trend."""
    rows = (
        db.query(HealthIndex)
        .filter(HealthIndex.mission_id == mission_id)
        .order_by(HealthIndex.sim_time_s.desc())
        .limit(limit)
        .all()
    )
    points = [(row.sim_time_s, row.health_score) for row in reversed(rows)]
    summary = summarize_degradation(points)
    return {
        "mission_id": mission_id,
        "current_health_score": summary["health_score"],
        "degradation_rate_per_hour": summary["degradation_rate_per_hour"],
        "predicted_health_1h": summary["predicted_health_1h"],
        "predicted_health_6h": summary["predicted_health_6h"],
        "trend_confidence": summary["trend_confidence"],
        "state": summary["state"],
        "sample_count": summary["sample_count"],
    }


@app.get("/missions/{mission_id}/faults")
def get_faults(mission_id: int, db: Session = Depends(get_db)):
    rows = db.query(FaultEvent).filter(FaultEvent.mission_id == mission_id).all()
    return [
        {"fault_type": f.fault_type, "onset_time_s": f.onset_time_s, "detected_at": f.detected_at.isoformat()}
        for f in rows
    ]


# --------------------------------------------------------------------------
# WebSocket endpoint: dashboard connects here to receive live telemetry
# --------------------------------------------------------------------------

@app.websocket("/ws/telemetry/{mission_id}")
async def telemetry_ws(websocket: WebSocket, mission_id: int):
    await manager.connect(mission_id, websocket)
    try:
        while True:
            # We don't expect messages from the client, but this keeps the
            # connection open and lets us detect disconnects promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(mission_id, websocket)


@app.get("/")
def root():
    return {"status": "ok", "message": "MALE UAV Digital Twin API is running"}