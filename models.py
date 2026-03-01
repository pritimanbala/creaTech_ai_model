#!/usr/bin/env python3
"""Concrete strength and cost/time models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostRates:
    cement_rate: float = 7.0
    scm_rate: float = 3.0
    water_rate: float = 0.2
    aggregates_rate: float = 1.2


def maturity_index_nurse_saul(temp_c: float, curing_days: float, t0_c: float = -10.0) -> float:
    """Simplified maturity index in degree-hours."""
    hours = curing_days * 24.0
    return max(0.0, (temp_c - t0_c) * hours)


class StrengthModelA:
    """Strength = f(maturity, temperature, RH)."""

    def predict(self, rec: Dict[str, float]) -> float:
        maturity = maturity_index_nurse_saul(rec["ambient_temp_c"], rec["curing_days"])
        humidity_factor = 0.7 + 0.003 * rec["relative_humidity_pct"]
        temp_factor = 0.9 + 0.01 * (rec["ambient_temp_c"] - 20.0)
        strength = 4.0 + 0.06 * math.sqrt(max(maturity, 0.0)) * humidity_factor * temp_factor
        return float(max(0.0, strength))


class StrengthModelB:
    """Strength = f(material proportions)."""

    def predict(self, rec: Dict[str, float]) -> float:
        cement = rec["cement_kg_m3"]
        scm = rec["scm_kg_m3"]
        water = rec["water_kg_m3"]
        aggregates = rec["aggregates_kg_m3"]
        binder = max(cement + scm, 1e-6)
        wc = water / max(cement, 1e-6)

        strength = (
            8.0
            + 0.06 * cement
            + 0.03 * scm
            - 18.0 * wc
            + 0.004 * aggregates
            + 2.0 * math.log1p(binder / 100.0)
        )
        return float(max(0.0, strength))


@dataclass
class CombinedStrengthPredictor:
    model_a: Callable[[Dict[str, float]], float] | StrengthModelA
    model_b: Callable[[Dict[str, float]], float] | StrengthModelB
    weight_a: float = 0.5
    weight_b: float = 0.5

    def predict(self, rec: Dict[str, float]) -> float:
        a = self.model_a.predict(rec) if hasattr(self.model_a, "predict") else float(self.model_a(rec))
        b = self.model_b.predict(rec) if hasattr(self.model_b, "predict") else float(self.model_b(rec))
        strength = self.weight_a * a + self.weight_b * b
        logger.debug("Strength model outputs: A=%.3f, B=%.3f, combined=%.3f", a, b, strength)
        return float(strength)


def compute_cost(rec: Dict[str, float], rates: CostRates | None = None) -> float:
    rates = rates or CostRates()
    cost = (
        rec["cement_kg_m3"] * rates.cement_rate
        + rec["scm_kg_m3"] * rates.scm_rate
        + rec["water_kg_m3"] * rates.water_rate
        + rec["aggregates_kg_m3"] * rates.aggregates_rate
    )
    return float(cost)


def compute_time(rec: Dict[str, float]) -> float:
    return float(rec["curing_days"])
