from __future__ import annotations
from datetime import datetime, date
from sqlalchemy import String, Integer, Float, Date, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Building(Base):
    __tablename__ = "buildings"
    building_id: Mapped[str] = mapped_column(String, primary_key=True)
    site: Mapped[str | None] = mapped_column(String, nullable=True)
    usage: Mapped[str | None] = mapped_column(String, nullable=True)
    sub_usage: Mapped[str | None] = mapped_column(String, nullable=True)
    sqm: Mapped[float | None] = mapped_column(Float, nullable=True)
    yearbuilt: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mean_consumption: Mapped[float | None] = mapped_column(Float, nullable=True)

    metrics: Mapped[list["Metric"]] = relationship(back_populates="building")
    days: Mapped[list["TestDay"]] = relationship(back_populates="building")


class Model(Base):
    __tablename__ = "models"
    name: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    checkpoint_path: Mapped[str | None] = mapped_column(String, nullable=True)
    n_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lookback: Mapped[int | None] = mapped_column(Integer, nullable=True)
    forecast: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hyperparams: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Metric(Base):
    __tablename__ = "metrics"
    building_id: Mapped[str] = mapped_column(ForeignKey("buildings.building_id"), primary_key=True)
    model_name: Mapped[str] = mapped_column(ForeignKey("models.name"), primary_key=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    wape: Mapped[float | None] = mapped_column(Float, nullable=True)

    building: Mapped["Building"] = relationship(back_populates="metrics")


class TestDay(Base):
    __tablename__ = "test_days"
    building_id: Mapped[str] = mapped_column(ForeignKey("buildings.building_id"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)

    building: Mapped["Building"] = relationship(back_populates="days")


class InferenceRun(Base):
    __tablename__ = "inference_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    building_id: Mapped[str] = mapped_column(String)
    day: Mapped[date] = mapped_column(Date)
    model_names: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
