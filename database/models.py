"""
database/models.py
-------------------
SQLAlchemy models for the MALE UAV Digital Twin project.

Beginner note: SQLAlchemy is a Python library that lets you define database
tables as normal Python classes (this is called an "ORM" -- Object Relational
Mapper). You never have to write raw SQL like `CREATE TABLE ...` yourself;
SQLAlchemy generates it for you from these class definitions.

By default this uses SQLite (a database that's just a single file on disk,
no server needed) so it works immediately with zero setup -- great for a
hackathon demo. To switch to Postgres later, you only need to change the
DATABASE_URL below and `pip install psycopg2-binary`.
"""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# --------------------------------------------------------------------------
# Connection setup
# --------------------------------------------------------------------------

# Reads from an environment variable if set (so you can switch to Postgres
# in production without touching code), otherwise falls back to a local
# SQLite file called digital_twin.db in the database/ folder.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./digital_twin.db")

# `connect_args` is only needed for SQLite (lets multiple threads share the
# connection, which FastAPI needs). Postgres doesn't need this argument.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

class Engine(Base):
    """One row per physical aero piston engine (or per UAV, for the demo)."""
    __tablename__ = "engines"

    id = Column(Integer, primary_key=True, index=True)
    serial_number = Column(String, unique=True, nullable=False)
    model = Column(String, nullable=True)
    uav_id = Column(String, nullable=True)

    missions = relationship("Mission", back_populates="engine")


class Mission(Base):
    """One row per simulator run / flight. Everything else hangs off this."""
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(Integer, ForeignKey("engines.id"), nullable=False)
    profile = Column(String, nullable=False)          # e.g. "cruise", "high_altitude"
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    engine = relationship("Engine", back_populates="missions")
    telemetry_readings = relationship("TelemetryReading", back_populates="mission")
    fault_events = relationship("FaultEvent", back_populates="mission")
    health_indices = relationship("HealthIndex", back_populates="mission")


class TelemetryReading(Base):
    """One row per simulator timestep. This table grows fast -- that's normal."""
    __tablename__ = "telemetry_readings"

    id = Column(Integer, primary_key=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=False, index=True)

    sim_time_s = Column(Float, nullable=False)         # matches "timestamp" from the simulator
    rpm = Column(Float)
    cht_c = Column(Float)                               # cylinder head temp
    egt_c = Column(Float)                                # exhaust gas temp
    oil_pressure_kpa = Column(Float)
    oil_temp_c = Column(Float)
    fuel_flow_lph = Column(Float)
    vibration_g = Column(Float)
    injection_timing_deg = Column(Float)
    battery_voltage_v = Column(Float)

    mission = relationship("Mission", back_populates="telemetry_readings")


class FaultEvent(Base):
    """One row per fault that started during a mission."""
    __tablename__ = "fault_events"

    id = Column(Integer, primary_key=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=False, index=True)

    fault_type = Column(String, nullable=False)          # e.g. "overheating", "misfire"
    onset_time_s = Column(Float, nullable=False)          # sim_time_s when it started
    severity = Column(Float, default=0.0)                 # 0.0 - 1.0, updated as it progresses
    detected_at = Column(DateTime, default=datetime.utcnow)  # when the ML layer flagged it
    resolved_at = Column(DateTime, nullable=True)

    mission = relationship("Mission", back_populates="fault_events")


class HealthIndex(Base):
    """One row per health-score computation -- the ML layer's output over time."""
    __tablename__ = "health_indices"

    id = Column(Integer, primary_key=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=False, index=True)

    sim_time_s = Column(Float, nullable=False)
    health_score = Column(Float, nullable=False)          # 0-100, higher = healthier
    rul_hours = Column(Float, nullable=True)               # estimated remaining useful life

    mission = relationship("Mission", back_populates="health_indices")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def init_db() -> None:
    """Creates all tables if they don't already exist. Call this once at startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency: gives each request its own database session and
    closes it afterward. Usage in a route:

        @app.get("/telemetry")
        def read_telemetry(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Running this file directly (python database/models.py) creates the
    # digital_twin.db SQLite file with all tables -- a quick way to verify
    # everything is defined correctly before wiring up the backend.
    init_db()
    print(f"Database initialized at: {DATABASE_URL}")