# nsga_engine.py
# -*- coding: utf-8 -*-
"""
NSGA-II Dashboard engine
========================

This file is a full nsga_engine.py integration build that preserves the public
interface used by app.py while applying the requested upgrade scope only:

CHANGE 1. NSGAConfig extension for the latest 260527 Ground model settings.
CHANGE 2. _prepare_engine_dataframe() extension to recognize and preserve:
          QPA, Repairable_Flag, Replace_Time_H, Repair_TAT_H.
CHANGE 3. _build_engine_context() calculation upgrade:
          Failure_Rate -> annual lambda, QPA, Fleet Size, p_need, repairable time split.
CHANGE 4. Candidate selection upgrade:
          p_need_period, long lead, Ao impact, and long-core protection logic.
CHANGE 5. Residual shortage exposure helpers/objective upgrade:
          Poisson EBO based residual risk is added to F2 while preserving UI keys.
CHANGE 6. Additive result-table upgrade:
          final_spares_df, repairable_audit_df, diagnostic_df are returned without
          removing existing app.py result keys.
CHANGE 7. PERIOD Ao/DT consistency fix:
          config.years is the single analysis-period control; annual DT is expanded
          to the full analysis period before Ao = T_obs / (T_obs + DT_period).
          Canonical period names replace hard-coded 2y internals while legacy output
          aliases are retained only where needed for app.py backward compatibility.

STAGE 2 MODEL C ROBUST INTEGRATION
----------------------------------
- Activates Model C robust candidate-pool expansion using the configured K stress grid.
- Evaluates each inventory alternative over all K scenarios and calculates:
  Tail Risk, Worst Ao, Worst Shortage and Robustness Gap.
- Replaces the former random-search + single-elite loop with an explicit NSGA-II
  loop: non-dominated sorting, crowding distance, tournament selection,
  crossover, mutation and elitist environmental replacement.
- Monte Carlo remains disabled in Stage 2 and will be connected in a later stage.

STAGE 1 ARCHITECTURE PREPARATION (retained)
-------------------------------------------
- Adds Model-C/Robust configuration placeholders that are NOT yet connected to
  the objective function. This deliberately preserves current NSGA outputs.
- Adds Monte-Carlo configuration placeholders that are NOT yet executed.
- Preserves shared cold-start input columns needed by both future Model C and
  future Monte Carlo (Annual_Rad_Hours, QPA, Lead Time, Repair TAT, etc.).
- Adds readiness validation and metadata outputs so app.py can audit whether the
  uploaded data/config are structurally ready for the later stages.
- Existing optimization equations, candidate rules, objective values and result
  keys remain unchanged in Stage 1.

STAGE 3 REPRESENTATIVE BRIDGE
-----------------------------
- Selects representative Model-C Pareto alternatives across the unique stock spectrum.
- Default strategy labels: 최소재고형 / 경량 대응형 / 균형 대응형 / 안정형 / 최대안정형.
- Same-stock ties are resolved by lower Tail Risk, higher Worst Ao, lower Cost.
- Produces representative_alternatives_df, representative_inventory_df,
  representative_selection_audit_df, and monte_carlo_bridge_df.
- Monte Carlo is NOT executed in Stage 3; only its deterministic input bridge is created.

STAGE 4 COLD-START MONTE CARLO INTEGRATION
-------------------------------------------
- Monte Carlo execution is implemented in a separate simulation_engine.py module.
- nsga_engine.py remains responsible for Model-C alternative generation and bridge data.
- monte_carlo_bridge_df is the contractual interface between optimization and simulation.
- app.py may execute simulation_engine.py after NSGA completion and merge MC outputs.
- The NSGA objectives are NOT recalibrated using Monte Carlo results in Stage 4.

STAGE 5 COST / RISK DECISION INTEGRATION
------------------------------------------
- decision_engine.py consumes representative alternatives + Monte Carlo summaries.
- No Stage-5 score is fed back into NSGA-II.
- No automatic "best" solution is selected.
- Decision grades are relative descriptive grades for visualization/audit only.
- NSGA-II remains the alternative generator; Monte Carlo remains the uncertainty validator.

STAGE 6 FINAL INTEGRATION / AUDIT
---------------------------------
- audit_engine.py validates cross-stage consistency after Decision Layer execution.
- Stage 6 does not change Model-C objectives, representative selection, or MC distributions.
- Final audit checks are read-only and cannot feed back into optimization.
- Final UI consolidates Executive summary, strategy drill-down, and regression/audit outputs.

FINAL DEBUG / MULTI-ECHELON ALIGNMENT
---------------------------------------
- Removes forced D-level-only inventory.
- Each candidate is stocked at its own Maint_Echelon (O / I / D).
- Restores lead-time / Repair-TAT exposure-based shortage probability.
- Restores canonical cost = purchase + holding(5%) + period maintenance spend.
- Removes duplicate maintenance-cost addition.
- Robust F2 is evaluated over the fixed hard-eligible spare universe.
- Tail Risk = mean of the worst two normalized shortage-exposure scenario risks.

SUPPLY-STRUCTURE OPTION (ENGINE)
-------------------------------
This engine supports multiple PBL supply structures without maintaining separate code:
- MAINT_ECHELON: stock at each item's Maint_Echelon (O/I/D).
- DEPOT_DIRECT: concentrate all PBL stock at D-level for direct depot-to-unit support.

The selected structure propagates through population generation, crossover/mutation,
Ao waiting-time effects, Robust F2 diagnostics, policy output, and Monte-Carlo bridge.

The public interface and runtime output keys used by the current Streamlit app.py
are preserved:
summary, prepared_df, candidate_df, history_df, pareto_df, policy_df,
sweep_summary_df, all_target_runs_df.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_RANDOM_SEED = 42
DEFAULT_TARGET_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94]
DEFAULT_REPRESENTATIVE_TARGET = 0.94

# Analysis period master setting.
# Change this one value (or NSGAConfig.years) to run 3-year, 5-year, etc.
YEARS = 5.0
HOURS_PER_YEAR = 365.0 * 24.0
T_OBS_HOURS = YEARS * HOURS_PER_YEAR

AO_COST_PENALTY = 8.0e8
ITEM_COUNT_WEIGHT = 1.0
STOCK_UNIT_WEIGHT = 0.05
H_PRE = 0.05

ABS_CAP_SMALL = 3
ABS_CAP_MED = 4
ABS_CAP_LARGE = 6

BASELINE_WAIT_FRACTION_BY_ECHELON = {
    "O": 0.75,
    "I": 0.90,
    "D": 1.00,
}
MIN_WAIT_FLOOR_H_BY_ECHELON = {
    "O": 1.0,
    "I": 6.0,
    "D": 24.0,
}

PM_EFFECT_ON_FAILURE = 0.03
PM_EFFECT_ON_RESTORE = 0.18


@dataclass
class NSGAConfig:
    # Existing app.py fields: do not rename/remove.
    population_size: int = 80
    n_generations: int = 200
    random_seed: int = DEFAULT_RANDOM_SEED
    target_ao: float = DEFAULT_REPRESENTATIVE_TARGET

    # Existing UI knobs.
    pmin: float = 0.01
    long_lead_percentile: float = 0.90
    ao_impact_percentile: float = 0.70

    # Existing sweep / standardization.
    ao_target_grid: List[float] = field(default_factory=lambda: list(DEFAULT_TARGET_GRID))
    representative_target: float = DEFAULT_REPRESENTATIVE_TARGET
    long_core_impact_percentile: float = 0.70
    item_count_weight: float = ITEM_COUNT_WEIGHT
    stock_unit_weight: float = STOCK_UNIT_WEIGHT
    ao_cost_penalty: float = AO_COST_PENALTY
    # Analysis period; app.py may override this through the Analysis Years slider.
    years: float = YEARS

    # CHANGE 1: latest 260527 Ground model setting fields.
    hours_per_year: float = HOURS_PER_YEAR
    failure_rate_unit: str = "per_million_hours"
    annual_operating_hours: float = 8760.0
    apply_qpa_to_failure_rate: bool = True
    default_qpa: float = 1.0
    fleet_size: float = 1.0

    # --------------------------------------------------------
    # PBL supply / stocking structure
    # --------------------------------------------------------
    # MAINT_ECHELON:
    #   each item is stocked at its own Maint_Echelon (O / I / D)
    # DEPOT_DIRECT:
    #   all PBL stock is concentrated at D (depot) and supports the unit directly
    #
    # d_only_pbl_stock is retained as a deprecated backward-compatible alias.
    supply_structure_mode: str = "MAINT_ECHELON"
    d_only_pbl_stock: bool = False

    holding_cost_rate: float = H_PRE
    shortage_prob_mode: str = "lead_time_demand"
    lead_time_shortage_floor_h: float = 1.0
    use_lead_time_for_stock_cap: bool = True
    apply_repairable_time_split: bool = True
    apply_repairable_stock_cap: bool = True
    stock_cap_safety_z: float = 4.0
    stock_cap_buffer_units: float = 2.0
    use_residual_shortage_exposure_f2: bool = True
    residual_shortage_exposure_weight: float = 0.10
    residual_shortage_exposure_scale_eps: float = 1e-12

    # --------------------------------------------------------
    # STAGE 1: Model C / future Monte-Carlo interface settings
    # IMPORTANT:
    # These settings are placeholders/readiness metadata only in Stage 1.
    # They are intentionally NOT connected to current candidate/objective
    # calculations so the existing NSGA outputs remain unchanged.
    # --------------------------------------------------------
    analysis_model: str = "MODEL_C"
    robust_enabled: bool = True
    robust_candidate_expansion_enabled: bool = True
    robust_k_grid: List[float] = field(
        default_factory=lambda: [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    )
    robust_tail_quantile: float = 0.80
    nsga_crossover_prob: float = 0.90
    nsga_mutation_prob: float = 0.02

    monte_carlo_enabled: bool = False
    monte_carlo_iterations: int = 3000
    monte_carlo_seed: int = DEFAULT_RANDOM_SEED
    monte_carlo_k_distribution: str = "lognormal"
    monte_carlo_k_median: float = 1.0
    monte_carlo_k_p05: float = 0.5
    monte_carlo_k_p95: float = 2.0
    monte_carlo_period_months: List[int] = field(default_factory=lambda: [12, 24, 36])
    monte_carlo_common_random_numbers: bool = True

    # --------------------------------------------------------
    # STAGE 3: Representative-solution Bridge settings
    # --------------------------------------------------------
    representative_bridge_enabled: bool = True
    representative_strategy_count: int = 5
    representative_stock_quantiles: List[float] = field(
        default_factory=lambda: [0.00, 0.25, 0.50, 0.75, 1.00]
    )


def find_best_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def coerce_numeric(series, fill_value: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        idx = series.index
        s = series
    else:
        try:
            idx = series.index  # type: ignore[attr-defined]
        except Exception:
            idx = None
        s = pd.Series(series)
        if idx is not None:
            s.index = idx
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return out.astype(float)


def _coerce_repairable_flag(series: pd.Series) -> pd.Series:
    """
    Convert repairable flag-like values to 0/1 safely.

    Truthy examples: 1, true, y, yes, repairable, 수리가능.
    Falsy examples: 0, false, n, no, consumable, expendable, 폐기.
    Unknown or missing values default to 0 so older input files remain compatible.
    """
    s = series.copy()
    numeric = pd.to_numeric(s, errors="coerce")
    text = s.astype(str).str.strip().str.lower()

    truthy = text.isin([
        "1", "1.0", "true", "t", "y", "yes", "repairable", "reparable",
        "r", "수리가능", "수리", "가능", "가능품", "repair", "returnable",
    ])
    falsy = text.isin([
        "0", "0.0", "false", "f", "n", "no", "non-repairable", "nonrepairable",
        "consumable", "expendable", "폐기", "불가", "소모품", "discard", "condemn",
    ])

    out = pd.Series(0, index=series.index, dtype=int)
    out.loc[numeric.fillna(0) > 0] = 1
    out.loc[truthy] = 1
    out.loc[falsy] = 0
    return out.astype(int)


def minmax(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr.copy()
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def parse_level_to_int(v) -> int:
    if pd.isna(v):
        return 0
    s = str(v)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def map_echelon(v) -> str:
    if pd.isna(v):
        return "X"
    s = str(v).strip().lower()
    if any(k in s for k in ["organizational", "operational", "operator", "field", "unit", "line", "o-level", "o level", "현장", "운용", "부대", "일선"]):
        return "O"
    if any(k in s for k in ["intermediate", "support", "shop", "base", "i-level", "i level", "direct support", "중간", "야전정비", "지원", "정비대"]):
        return "I"
    if any(k in s for k in ["depot", "factory", "overhaul", "sustainment", "d-level", "d level", "창정비", "공창", "후방"]):
        return "D"
    if s in ["o", "org", "op"]:
        return "O"
    if s in ["i", "int"]:
        return "I"
    if s in ["d", "dep"]:
        return "D"
    return "X"

SUPPLY_STRUCTURE_MAINT_ECHELON = "MAINT_ECHELON"
SUPPLY_STRUCTURE_DEPOT_DIRECT = "DEPOT_DIRECT"
SUPPORTED_SUPPLY_STRUCTURE_MODES = (
    SUPPLY_STRUCTURE_MAINT_ECHELON,
    SUPPLY_STRUCTURE_DEPOT_DIRECT,
)


def normalize_supply_structure_mode(config) -> str:
    """
    Normalize user/config aliases into a stable internal supply-structure mode.

    MAINT_ECHELON
        Stock an item at its own maintenance echelon (O/I/D).

    DEPOT_DIRECT
        Concentrate all PBL stock at D-level and treat the depot as the direct
        supply source to the operating unit.

    Backward compatibility
    ----------------------
    If legacy d_only_pbl_stock=True, DEPOT_DIRECT takes precedence.
    """
    if bool(getattr(config, "d_only_pbl_stock", False)):
        return SUPPLY_STRUCTURE_DEPOT_DIRECT

    raw = str(
        getattr(config, "supply_structure_mode", SUPPLY_STRUCTURE_MAINT_ECHELON)
    ).strip().upper().replace("-", "_").replace(" ", "_")

    aliases = {
        "MAINT_ECHELON": SUPPLY_STRUCTURE_MAINT_ECHELON,
        "MULTI_ECHELON": SUPPLY_STRUCTURE_MAINT_ECHELON,
        "ECHELON": SUPPLY_STRUCTURE_MAINT_ECHELON,
        "O_I_D": SUPPLY_STRUCTURE_MAINT_ECHELON,
        "OID": SUPPLY_STRUCTURE_MAINT_ECHELON,
        "계층형": SUPPLY_STRUCTURE_MAINT_ECHELON,
        "정비계층": SUPPLY_STRUCTURE_MAINT_ECHELON,

        "DEPOT_DIRECT": SUPPLY_STRUCTURE_DEPOT_DIRECT,
        "D_DIRECT": SUPPLY_STRUCTURE_DEPOT_DIRECT,
        "D_ONLY": SUPPLY_STRUCTURE_DEPOT_DIRECT,
        "DEPOT": SUPPLY_STRUCTURE_DEPOT_DIRECT,
        "창직보급": SUPPLY_STRUCTURE_DEPOT_DIRECT,
        "창_직보급": SUPPLY_STRUCTURE_DEPOT_DIRECT,
    }
    return aliases.get(raw, raw)


def derive_stocking_echelon(maint_echelon, supply_structure_mode: str) -> np.ndarray:
    """
    Convert maintenance-echelon classification into the actual stocking location
    implied by the selected PBL supply structure.
    """
    maint = np.asarray(maint_echelon, dtype=object).astype(str)
    mode = str(supply_structure_mode).strip().upper()

    if mode == SUPPLY_STRUCTURE_DEPOT_DIRECT:
        return np.full(len(maint), "D", dtype=object)

    if mode == SUPPLY_STRUCTURE_MAINT_ECHELON:
        # Unknown values are conservatively routed to D so inventory never
        # disappears from the optimization decision space.
        return np.where(
            np.isin(maint, ["O", "I", "D"]),
            maint,
            "D",
        ).astype(object)

    raise ValueError(
        f"Unsupported supply_structure_mode={supply_structure_mode!r}. "
        f"Supported: {SUPPORTED_SUPPLY_STRUCTURE_MODES}"
    )



def long_dynamic_cap_from_lambda(lam_period: float) -> int:
    if lam_period <= 0.30:
        return ABS_CAP_SMALL
    if lam_period <= 0.80:
        return ABS_CAP_MED
    return ABS_CAP_LARGE


def poisson_tail_prob(mu: float, s: int) -> float:
    mu = max(float(mu), 0.0)
    s = max(int(s), 0)
    if mu <= 1e-12:
        return 0.0
    term = math.exp(-mu)
    cdf = term
    for k in range(1, s + 1):
        term *= mu / k
        cdf += term
    tail = 1.0 - cdf
    return float(min(max(tail, 0.0), 1.0))


def poisson_sf_ge(mu: float, k: int) -> float:
    """Return P[Poisson(mu) >= k] with small-mu and integer guards."""
    mu = max(float(mu), 0.0)
    k = int(k)
    if k <= 0:
        return 1.0
    return poisson_tail_prob(mu, k - 1)


def poisson_ebo_scalar(mu: float, stock: int) -> float:
    """Expected backorders E[(D-stock)+] for D ~ Poisson(mu)."""
    mu = max(float(mu), 0.0)
    s = max(int(stock), 0)
    if mu <= 1e-12:
        return 0.0
    ebo = mu * poisson_sf_ge(mu, s) - s * poisson_sf_ge(mu, s + 1)
    return float(max(ebo, 0.0))


def poisson_ebo_array(mu_vec: np.ndarray, stock_vec: np.ndarray) -> np.ndarray:
    """Vectorized expected backorder calculation using the stable scalar routine."""
    mu_arr = np.asarray(mu_vec, dtype=float)
    stock_arr = np.asarray(stock_vec, dtype=int)
    if mu_arr.shape != stock_arr.shape:
        mu_arr, stock_arr = np.broadcast_arrays(mu_arr, stock_arr)
    out = np.zeros(mu_arr.shape, dtype=float)
    it = np.nditer([mu_arr, stock_arr, out], flags=['multi_index'], op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for mu, st, dest in it:
        dest[...] = poisson_ebo_scalar(float(mu), int(st))
    return out


def node_diag_contrib(level_value: int) -> float:
    if level_value <= 1:
        return 0.10
    if level_value == 2:
        return 0.18
    if level_value == 3:
        return 0.35
    if level_value == 4:
        return 0.55
    if level_value == 5:
        return 0.80
    return 1.00


def solution_signature(y: np.ndarray, sO: np.ndarray, sI: np.ndarray, sD: np.ndarray) -> str:
    y = np.asarray(y, dtype=int)
    sO = np.asarray(sO, dtype=int)
    sI = np.asarray(sI, dtype=int)
    sD = np.asarray(sD, dtype=int)
    return (
        "Y:" + ",".join(map(str, y.tolist()))
        + "|O:" + ",".join(map(str, sO.tolist()))
        + "|I:" + ",".join(map(str, sI.tolist()))
        + "|D:" + ",".join(map(str, sD.tolist()))
    )


# ------------------------------------------------------------------
# STAGE 1 readiness validation
# ------------------------------------------------------------------
def validate_stage1_config(config: NSGAConfig) -> Dict[str, object]:
    """Validate future Model-C/Monte-Carlo interface settings without using them."""
    issues: List[str] = []

    k_grid = [float(x) for x in getattr(config, "robust_k_grid", [])]
    if not k_grid or any((not np.isfinite(x)) or x <= 0 for x in k_grid):
        issues.append("robust_k_grid must contain positive finite values.")

    q = float(getattr(config, "robust_tail_quantile", 0.80))
    if not (0.0 < q < 1.0):
        issues.append("robust_tail_quantile must be between 0 and 1.")
    if bool(getattr(config, "robust_enabled", False)) and not any(abs(x - 1.0) < 1e-9 for x in k_grid):
        issues.append("robust_k_grid must include K=1.0 when robust mode is enabled.")
    pc = float(getattr(config, "nsga_crossover_prob", 0.90))
    pm = float(getattr(config, "nsga_mutation_prob", 0.02))
    if not (0.0 <= pc <= 1.0):
        issues.append("nsga_crossover_prob must be between 0 and 1.")
    if not (0.0 <= pm <= 1.0):
        issues.append("nsga_mutation_prob must be between 0 and 1.")

    mc_iter = int(getattr(config, "monte_carlo_iterations", 3000))
    if mc_iter <= 0:
        issues.append("monte_carlo_iterations must be > 0.")

    mc_periods = [int(x) for x in getattr(config, "monte_carlo_period_months", [12, 24, 36])]
    if not mc_periods or any(x <= 0 for x in mc_periods):
        issues.append("monte_carlo_period_months must contain positive month values.")

    rep_count = int(getattr(config, "representative_strategy_count", 5))
    rep_quantiles = [float(x) for x in getattr(config, "representative_stock_quantiles", [0.0, 0.25, 0.5, 0.75, 1.0])]
    if rep_count <= 0:
        issues.append("representative_strategy_count must be > 0.")
    if not rep_quantiles or any((not np.isfinite(x)) or x < 0.0 or x > 1.0 for x in rep_quantiles):
        issues.append("representative_stock_quantiles must contain finite values between 0 and 1.")
    if bool(getattr(config, "representative_bridge_enabled", True)) and len(rep_quantiles) != rep_count:
        issues.append("representative_stock_quantiles length must equal representative_strategy_count.")

    supply_mode = normalize_supply_structure_mode(config)
    if supply_mode not in SUPPORTED_SUPPLY_STRUCTURE_MODES:
        issues.append(
            f"supply_structure_mode must be one of {SUPPORTED_SUPPLY_STRUCTURE_MODES}; "
            f"got {supply_mode!r}."
        )

    p05 = float(getattr(config, "monte_carlo_k_p05", 0.5))
    med = float(getattr(config, "monte_carlo_k_median", 1.0))
    p95 = float(getattr(config, "monte_carlo_k_p95", 2.0))
    if not (0 < p05 <= med <= p95):
        issues.append("Monte-Carlo K anchors must satisfy 0 < P05 <= median <= P95.")

    return {
        "stage1_ready": len(issues) == 0,
        "issues": issues,
        "analysis_model": str(getattr(config, "analysis_model", "MODEL_C_READY")),
        "robust_k_grid": k_grid,
        "monte_carlo_period_months": mc_periods,
        "monte_carlo_iterations": mc_iter,
        "representative_strategy_count": rep_count,
        "representative_stock_quantiles": rep_quantiles,
        "representative_bridge_enabled": bool(getattr(config, "representative_bridge_enabled", True)),
        "supply_structure_mode": supply_mode,
    }


def _prepare_engine_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    col_part = find_best_column(df, ["Part_ID", "part_id", "part", "item", "item_id", "niin", "nsn", "품번", "부품명"])
    col_parent = find_best_column(df, ["Parent_ID", "parent_id", "parent", "상위품번", "상위부품"])
    col_level = find_best_column(df, ["Level", "level", "레벨"])
    col_echelon = find_best_column(df, ["Maint_Echelon", "maint_echelon", "echelon", "정비단계", "정비계층"])
    col_maint_action = find_best_column(df, ["Maint_Action", "maint_action", "Maintenance_Action", "정비행위", "정비조치"])
    col_fr = find_best_column(df, ["Failure_Rate", "failure_rate", "fail_rate", "lambda", "annual_failure_rate", "고장률"])
    col_price = find_best_column(df, ["Unit_Price_KRW", "unit_cost", "unit_price", "cost", "price", "단가"])
    col_transport_cost = find_best_column(df, ["Transport_Cost_KRW", "transport_cost"])
    col_lead = find_best_column(df, ["Total_Lead_Time_H", "lead_time", "leadtime", "lt_days", "lt", "리드타임"])
    col_transport_time = find_best_column(df, ["Transport_Time_H", "transport_time"])
    col_cm_cost = find_best_column(df, ["CM_Cost_KRW", "cm_cost", "corrective_cost"])
    col_cm_time = find_best_column(df, ["CM_Time_Hours", "cm_time", "repair_time", "maintenance_time"])
    col_condemn = find_best_column(df, ["Condemnation_Rate_Pct", "condemnation_rate_pct", "condemn", "폐기율"])
    col_pm = find_best_column(df, ["PM_Cycle", "pm_cycle", "pm", "maintenance_cycle", "예방정비주기"])
    col_rad_idx = find_best_column(df, ["Rad_Degradation_Index", "rad_degradation_index", "soh_score", "SOH_SCORE"])
    col_ann_rad = find_best_column(df, ["Annual_Rad_Hours", "annual_rad_hours", "Annual_Operating_Hours", "annual_operating_hours"])
    col_env_res = find_best_column(df, ["Env_Resistance", "env_resistance", "환경저항성", "env_res"])

    # Optional shared reporting/classification column for later decision/drill-down.
    col_category = find_best_column(df, ["Category", "category", "Part_Category", "part_category", "품목구분", "분류"])

    # CHANGE 2: latest 260527 Ground optional columns.
    col_qpa = find_best_column(df, [
        "QPA", "Quantity_Per_Equipment", "Qty_Per_Equipment", "Quantity_Per_System",
        "Qty_Per_System", "Qty_Per_Assembly", "qpa", "qty_per_equipment",
        "qty_per_system", "대당수량", "장착수량",
    ])
    col_repairable = find_best_column(df, [
        "Repairable_Flag", "Is_Repairable", "Repairable", "Repair_Return_Flag",
        "repairable_flag", "is_repairable", "repairable", "수리가능여부", "수리가능",
    ])
    col_replace_time = find_best_column(df, [
        "Replace_Time_H", "Replacement_Time_H", "Swap_Time_H", "Remove_Install_Time_H",
        "replace_time_h", "replacement_time_h", "swap_time_h", "교체시간", "장탈착시간",
    ])
    col_repair_tat = find_best_column(df, [
        "Repair_TAT_H", "Repair_Turnaround_Time_H", "Repair_Turn_Around_Time_H",
        "Repair_Time_H", "Depot_Repair_Time_H", "repair_tat_h",
        "repair_turnaround_time_h", "repair_time_h", "수리TAT", "수리기간", "창정비시간",
    ])

    if col_part is None:
        df["part_id"] = [f"PART_{i + 1:05d}" for i in range(len(df))]
        col_part = "part_id"

    out = pd.DataFrame(index=df.index)
    out["part_id"] = df[col_part].astype(str)
    out["parent_id"] = df[col_parent].astype(str) if col_parent is not None else ""
    out["level_raw"] = df[col_level] if col_level is not None else 0
    out["level_num"] = out["level_raw"].apply(parse_level_to_int)
    out["maint_echelon_raw"] = df[col_echelon] if col_echelon is not None else "O"
    out["maint_echelon"] = out["maint_echelon_raw"].apply(map_echelon)
    out["maint_action_raw"] = df[col_maint_action].astype(str) if col_maint_action is not None else ""
    out["category"] = df[col_category].astype(str) if col_category is not None else ""

    out["failure_rate"] = coerce_numeric(df[col_fr], 0.01) if col_fr is not None else 0.01
    out["unit_cost"] = coerce_numeric(df[col_price], 1000.0) if col_price is not None else 1000.0
    out["transport_cost"] = coerce_numeric(df[col_transport_cost], 0.0) if col_transport_cost is not None else 0.0
    out["lead_time"] = coerce_numeric(df[col_lead], 24.0) if col_lead is not None else 24.0
    out["transport_time"] = coerce_numeric(df[col_transport_time], 8.0) if col_transport_time is not None else np.maximum(1.0, out["lead_time"] * 0.20)
    out["cm_cost"] = coerce_numeric(df[col_cm_cost], 0.3 * out["unit_cost"].mean() if len(out) else 1000.0) if col_cm_cost is not None else np.maximum(200.0, out["unit_cost"] * 0.25)
    out["cm_time"] = coerce_numeric(df[col_cm_time], np.maximum(1.0, out["lead_time"] * 0.10)) if col_cm_time is not None else np.maximum(1.0, out["lead_time"] * 0.10)
    out["condemn_pct"] = coerce_numeric(df[col_condemn], 10.0) if col_condemn is not None else 10.0
    out["pm_cycle"] = coerce_numeric(df[col_pm], 365.0) if col_pm is not None else 365.0
    out["rad_deg_index"] = coerce_numeric(df[col_rad_idx], 0.0) if col_rad_idx is not None else 0.0
    out["annual_rad_hours"] = coerce_numeric(df[col_ann_rad], 0.0) if col_ann_rad is not None else 0.0
    out["env_resistance"] = coerce_numeric(df[col_env_res], 3.0) if col_env_res is not None else 3.0

    out["qpa"] = coerce_numeric(df[col_qpa], 1.0) if col_qpa is not None else 1.0
    if col_repairable is not None:
        out["repairable_flag"] = _coerce_repairable_flag(df[col_repairable])
        out["repairable_source"] = "Repairable_Flag"
    elif col_maint_action is not None:
        # Legacy radar BOM fallback: infer only the repairable/non-repairable class.
        # Do NOT invent Repair_TAT_H from CM time; missing TAT continues to use lead-time exposure.
        action = df[col_maint_action].astype(str).str.strip().str.lower()
        repair_mask = action.str.contains("repair|수리", regex=True, na=False)
        replace_mask = action.str.contains("replace|교체", regex=True, na=False)
        out["repairable_flag"] = np.where(repair_mask & ~replace_mask, 1, 0).astype(int)
        out["repairable_source"] = "Maint_Action_Fallback"
    else:
        out["repairable_flag"] = 0
        out["repairable_source"] = "Default_NonRepairable"
    out["replace_time_h"] = coerce_numeric(df[col_replace_time], 0.0) if col_replace_time is not None else 0.0
    out["repair_tat_h"] = coerce_numeric(df[col_repair_tat], 0.0) if col_repair_tat is not None else 0.0

    out["failure_rate"] = out["failure_rate"].clip(lower=0.0)
    out["unit_cost"] = out["unit_cost"].clip(lower=0.0)
    out["lead_time"] = out["lead_time"].clip(lower=1.0)
    out["transport_time"] = out["transport_time"].clip(lower=0.1)
    out["cm_cost"] = out["cm_cost"].clip(lower=0.0)
    out["cm_time"] = out["cm_time"].clip(lower=0.1)
    out["condemn_pct"] = out["condemn_pct"].clip(lower=0.0, upper=100.0)
    out["pm_cycle"] = out["pm_cycle"].replace(0, 365).clip(lower=1.0)
    out["env_resistance"] = out["env_resistance"].clip(lower=1.0, upper=5.0)
    out["annual_rad_hours"] = out["annual_rad_hours"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    out["qpa"] = out["qpa"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
    out["repairable_flag"] = out["repairable_flag"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=1).astype(int)
    out["replace_time_h"] = out["replace_time_h"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    out["repair_tat_h"] = out["repair_tat_h"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    # display/compatibility columns for existing app preview
    fr_norm = minmax(out["failure_rate"].to_numpy())
    out["criticality"] = np.where(fr_norm >= 0.80, 1.00, np.where(fr_norm >= 0.50, 0.45, 0.15))
    out["stock"] = 0
    out["holding_cost"] = H_PRE * out["unit_cost"]
    out["base_score"] = (
        out["failure_rate"] * 0.45
        + (out["lead_time"] / max(float(out["lead_time"].max()), 1.0)) * 0.20
        + (out["criticality"] / max(float(out["criticality"].max()), 1.0)) * 0.35
    )
    return out.reset_index(drop=True)


def convert_failure_rate_to_annual_lambda(
    failure_rate: np.ndarray,
    failure_rate_unit: str = "per_million_hours",
    annual_operating_hours: float = 8760.0,
) -> np.ndarray:
    """
    Convert Failure_Rate into annual item-level failure intensity.
    """
    fr = np.asarray(failure_rate, dtype=float)
    fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
    fr = np.clip(fr, 0.0, None)
    unit = str(failure_rate_unit or "per_million_hours").strip().lower()
    hours = max(float(annual_operating_hours or 8760.0), 1.0)
    if unit in ["per_million_hours", "per_million_hour", "pmh", "failures_per_million_hours", "per_1e6_hours"]:
        return fr * hours / 1_000_000.0
    if unit in ["per_hour", "hourly", "lambda_h", "lambda_per_hour"]:
        return fr * hours
    if unit in ["per_year", "annual", "yearly", "lambda_y", "lambda_per_year"]:
        return fr
    return fr


def percentile_rank_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    arr2 = np.nan_to_num(arr, nan=np.nanmedian(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0.0)
    order = np.argsort(arr2, kind="mergesort")
    ranks = np.empty_like(arr2, dtype=float)
    if arr2.size == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, arr2.size)
    return ranks


def apply_repairable_time_split(
    restore_base_h: np.ndarray,
    lead_h: np.ndarray,
    repairable_flag: np.ndarray,
    replace_time_h: np.ndarray,
    repair_tat_h: np.ndarray,
    config: NSGAConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    restore_base = np.asarray(restore_base_h, dtype=float)
    lead = np.asarray(lead_h, dtype=float)
    repairable = np.asarray(repairable_flag, dtype=int) > 0
    replace_t = np.asarray(replace_time_h, dtype=float)
    tat = np.asarray(repair_tat_h, dtype=float)

    restore_used = np.nan_to_num(restore_base, nan=0.0, posinf=0.0, neginf=0.0).copy()
    exposure = np.maximum(np.nan_to_num(lead, nan=0.0, posinf=0.0, neginf=0.0), float(getattr(config, "lead_time_shortage_floor_h", 1.0)))

    restore_split = np.zeros_like(restore_used, dtype=bool)
    exposure_split = np.zeros_like(exposure, dtype=bool)

    if bool(getattr(config, "apply_repairable_time_split", True)):
        use_replace = repairable & np.isfinite(replace_t) & (replace_t > 0)
        restore_used[use_replace] = replace_t[use_replace]
        restore_split[use_replace] = True

        use_tat = repairable & np.isfinite(tat) & (tat > 0)
        exposure[use_tat] = tat[use_tat]
        exposure_split[use_tat] = True

    restore_used = np.maximum(np.nan_to_num(restore_used, nan=0.0, posinf=0.0, neginf=0.0), 0.1)
    exposure = np.maximum(np.nan_to_num(exposure, nan=0.0, posinf=0.0, neginf=0.0), float(getattr(config, "lead_time_shortage_floor_h", 1.0)))
    return restore_used, exposure, restore_split, exposure_split


def stock_cap_from_pipeline_mu(
    mu_pipe: np.ndarray,
    lambda_period: np.ndarray,
    is_long: np.ndarray,
    repairable_flag: np.ndarray,
    config: NSGAConfig,
) -> np.ndarray:
    mu = np.asarray(mu_pipe, dtype=float)
    lam = np.asarray(lambda_period, dtype=float)
    long_mask = np.asarray(is_long, dtype=bool)
    repairable = np.asarray(repairable_flag, dtype=int) > 0
    z = float(getattr(config, "stock_cap_safety_z", 4.0))
    buf = float(getattr(config, "stock_cap_buffer_units", 2.0))
    cap = np.ceil(mu + z * np.sqrt(np.maximum(mu, 1e-9)) + buf).astype(int)
    small = int(getattr(config, "abs_cap_small", ABS_CAP_SMALL) if hasattr(config, "abs_cap_small") else ABS_CAP_SMALL)
    med = int(getattr(config, "abs_cap_med", ABS_CAP_MED) if hasattr(config, "abs_cap_med") else ABS_CAP_MED)
    large = int(getattr(config, "abs_cap_large", ABS_CAP_LARGE) if hasattr(config, "abs_cap_large") else ABS_CAP_LARGE)
    abs_cap = np.where(lam <= 0.30, small, np.where(lam <= 0.80, med, large))
    cap = np.where(long_mask, np.maximum(cap, abs_cap), cap)
    if bool(getattr(config, "apply_repairable_stock_cap", True)):
        cap = np.where(repairable, np.maximum(cap, np.ceil(mu + 1.0).astype(int)), cap)
    cap = np.nan_to_num(cap, nan=1, posinf=999, neginf=1).astype(int)
    return np.clip(cap, 1, 999)


def _build_engine_context(df_prepared: pd.DataFrame, config: NSGAConfig) -> Dict[str, object]:
    df = df_prepared.copy()
    n_all = len(df)
    if n_all == 0:
        raise ValueError("Input dataframe is empty.")

    part_id_all = df["part_id"].astype(str).to_numpy()
    parent_id_all = df["parent_id"].astype(str).replace("nan", "").replace("None", "").to_numpy()
    level_num_all = df["level_num"].to_numpy(dtype=int)
    echelon_cat_all = df["maint_echelon"].to_numpy(dtype=object)

    FR_raw_all = np.nan_to_num(df["failure_rate"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    unit_price_all = df["unit_cost"].to_numpy(dtype=float)
    transport_cost_all = df["transport_cost"].to_numpy(dtype=float)
    lead_all = df["lead_time"].to_numpy(dtype=float)
    transport_time_all = df["transport_time"].to_numpy(dtype=float)
    cm_cost_all = df["cm_cost"].to_numpy(dtype=float)
    cm_time_all = df["cm_time"].to_numpy(dtype=float)
    condemn_all = np.clip(df["condemn_pct"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    pm_cycle_all = df["pm_cycle"].to_numpy(dtype=float)

    qpa_all = np.nan_to_num(df.get("qpa", pd.Series(float(getattr(config, "default_qpa", 1.0)), index=df.index)).to_numpy(dtype=float), nan=float(getattr(config, "default_qpa", 1.0)), posinf=float(getattr(config, "default_qpa", 1.0)), neginf=float(getattr(config, "default_qpa", 1.0)))
    qpa_all = np.clip(qpa_all, 0.0, None)
    repairable_all = np.nan_to_num(df.get("repairable_flag", pd.Series(0, index=df.index)).to_numpy(dtype=int), nan=0).astype(int)
    repairable_all = np.clip(repairable_all, 0, 1)
    replace_time_all = np.nan_to_num(df.get("replace_time_h", pd.Series(0.0, index=df.index)).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    repair_tat_all = np.nan_to_num(df.get("repair_tat_h", pd.Series(0.0, index=df.index)).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    lambda_item_all = convert_failure_rate_to_annual_lambda(
        FR_raw_all,
        getattr(config, "failure_rate_unit", "per_million_hours"),
        getattr(config, "annual_operating_hours", 8760.0),
    )
    if bool(getattr(config, "apply_qpa_to_failure_rate", True)):
        lambda_system_all = lambda_item_all * qpa_all
    else:
        lambda_system_all = lambda_item_all.copy()
    lambda_demand_all = lambda_system_all * max(float(getattr(config, "fleet_size", 1.0)), 0.0)

    period_years = float(getattr(config, "years", YEARS))
    if (not np.isfinite(period_years)) or period_years <= 0.0:
        raise ValueError("NSGAConfig.years must be a positive finite number.")
    hours_per_year = max(float(getattr(config, "hours_per_year", HOURS_PER_YEAR)), 1e-9)
    t_obs_hours = period_years * hours_per_year
    lambda_period_all = lambda_demand_all * period_years

    pm_norm_all = 1.0 / np.sqrt(np.maximum(pm_cycle_all, 1.0) / 365.0)
    pm_norm_all = np.clip(pm_norm_all, 0.0, 1.0)
    gamma_all = PM_EFFECT_ON_FAILURE * pm_norm_all
    delta_all = PM_EFFECT_ON_RESTORE * pm_norm_all
    lambda_system_pm_all = lambda_system_all * (1.0 - gamma_all)
    lambda_demand_pm_all = lambda_demand_all * (1.0 - gamma_all)
    lambda_period_pm_all = lambda_demand_pm_all * period_years

    proc_cost_all = unit_price_all + transport_cost_all

    part_to_idx = {pid: i for i, pid in enumerate(part_id_all)}
    children_map = {i: [] for i in range(n_all)}
    for i in range(n_all):
        p = parent_id_all[i]
        if p and p in part_to_idx:
            children_map[part_to_idx[p]].append(i)
    is_leaf_all = np.array([len(children_map[i]) == 0 for i in range(n_all)], dtype=bool)

    def get_ancestor_chain(idx: int) -> List[int]:
        chain_rev = [idx]
        seen = {idx}
        cur_parent = parent_id_all[idx]
        hop = 0
        while cur_parent not in ["", "nan", "None", "null"]:
            cur_parent = str(cur_parent)
            if cur_parent not in part_to_idx:
                break
            pidx = part_to_idx[cur_parent]
            if pidx in seen:
                break
            chain_rev.append(pidx)
            seen.add(pidx)
            cur_parent = parent_id_all[pidx]
            hop += 1
            if hop > 50:
                break
        return list(reversed(chain_rev))

    all_ancestor_chains = [get_ancestor_chain(i) for i in range(n_all)]
    depth_all = np.array([len(ch) for ch in all_ancestor_chains], dtype=int)

    has_physical_attr = ((unit_price_all > 0) | (cm_time_all > 0) | (lead_all > 0))
    spare_demand_mask = ((level_num_all >= 3) & is_leaf_all & (lambda_system_all > 0) & has_physical_attr)
    if int(spare_demand_mask.sum()) < 20:
        spare_demand_mask = ((level_num_all >= 3) & (lambda_system_all > 0) & has_physical_attr)
    spare_idx = np.where(spare_demand_mask)[0]
    if len(spare_idx) == 0:
        raise ValueError("No real spare-demand parts found from uploaded data.")

    fr_raw_sp = FR_raw_all[spare_idx]
    lambda_item_sp = lambda_item_all[spare_idx]
    lambda_system_sp = lambda_system_pm_all[spare_idx]
    lambda_demand_sp = lambda_demand_pm_all[spare_idx]
    lambda_period_sp = lambda_period_pm_all[spare_idx]
    proc_cost_sp = proc_cost_all[spare_idx]
    lead_sp = lead_all[spare_idx]
    transport_sp = transport_time_all[spare_idx]
    cm_cost_sp = cm_cost_all[spare_idx]
    cm_time_sp = cm_time_all[spare_idx]
    condemn_sp = condemn_all[spare_idx]
    delta_sp = delta_all[spare_idx]
    level_sp = level_num_all[spare_idx]
    depth_sp = depth_all[spare_idx]
    echelon_sp = echelon_cat_all[spare_idx]
    supply_structure_mode = normalize_supply_structure_mode(config)
    stocking_echelon_sp = derive_stocking_echelon(
        echelon_sp,
        supply_structure_mode,
    )
    part_id_sp = part_id_all[spare_idx]
    parent_id_sp = parent_id_all[spare_idx]
    qpa_sp = qpa_all[spare_idx]
    repairable_sp = repairable_all[spare_idx]
    replace_time_sp = replace_time_all[spare_idx]
    repair_tat_sp = repair_tat_all[spare_idx]
    chains_sp = [all_ancestor_chains[i] for i in spare_idx]

    parent_group_map: Dict[str, List[int]] = {}
    for local_idx, global_idx in enumerate(spare_idx):
        p = parent_id_all[global_idx]
        p = str(p) if p not in ["", "nan", "None", "null"] else f"ROOT_{part_id_all[global_idx]}"
        parent_group_map.setdefault(p, []).append(local_idx)

    lambda_eff_sp = lambda_system_sp.copy()
    for _, members in parent_group_map.items():
        if len(members) <= 1:
            continue
        fr_sub = lambda_system_sp[members]
        total = fr_sub.sum()
        if total > 1e-12:
            shares = fr_sub / total
            lambda_eff_sp[members] = total * shares / len(members) * 1.2

    T_diag_sp = np.zeros(len(spare_idx), dtype=float)
    for k, chain in enumerate(chains_sp):
        T_diag_sp[k] = sum(node_diag_contrib(int(level_num_all[node_idx])) for node_idx in chain)

    T_reminst_sp = 0.35 * np.maximum(cm_time_sp, 1.0)
    T_repair_base_sp = 0.65 * cm_time_sp
    T_repair_sp = T_repair_base_sp * (1.0 - delta_sp)
    for i, e in enumerate(echelon_sp):
        if e == "O":
            T_reminst_sp[i] *= 0.85
            T_repair_sp[i] *= 0.75
        elif e == "D":
            T_reminst_sp[i] *= 1.15
            T_repair_sp[i] *= 1.25
    T_restore_base_sp = T_reminst_sp + (1.0 - condemn_sp) * T_repair_sp

    T_restore_intrinsic_sp, shortage_exposure_sp, restore_split_sp, exposure_split_sp = apply_repairable_time_split(
        T_restore_base_sp,
        lead_sp,
        repairable_sp,
        replace_time_sp,
        repair_tat_sp,
        config,
    )

    T_issue_O_sp = np.full(len(spare_idx), 0.3, dtype=float)
    T_issue_I_sp = np.maximum(0.8, np.minimum(4.0, 0.25 * np.maximum(transport_sp, 1.0)))
    T_issue_D_sp = np.maximum(1.5, np.minimum(8.0, 0.45 * np.maximum(transport_sp, 1.0)))

    baseline_wait_sp = np.zeros(len(spare_idx), dtype=float)
    for i, e in enumerate(echelon_sp):
        frac = BASELINE_WAIT_FRACTION_BY_ECHELON.get(e, 0.35)
        floor = MIN_WAIT_FLOOR_H_BY_ECHELON.get(e, 2.0)
        baseline_wait_sp[i] = max(floor, frac * lead_sp[i])

    p_need_sp = 1.0 - np.exp(-np.clip(lambda_period_sp, 0.0, None))
    ll_pct = float(config.long_lead_percentile * 100.0 if config.long_lead_percentile <= 1.0 else config.long_lead_percentile)
    ai_pct = float(config.ao_impact_percentile * 100.0 if config.ao_impact_percentile <= 1.0 else config.ao_impact_percentile)
    long_core_pct = float(config.long_core_impact_percentile * 100.0 if config.long_core_impact_percentile <= 1.0 else config.long_core_impact_percentile)
    ll_pct = float(np.clip(ll_pct, 0.0, 100.0))
    ai_pct = float(np.clip(ai_pct, 0.0, 100.0))
    long_core_pct = float(np.clip(long_core_pct, 0.0, 100.0))

    thr_lead = float(np.nanpercentile(shortage_exposure_sp, ll_pct)) if len(shortage_exposure_sp) else 0.0
    is_long_sp = shortage_exposure_sp >= thr_lead
    impact_sp = lambda_eff_sp * np.maximum(T_diag_sp + baseline_wait_sp + T_restore_intrinsic_sp, 1e-9) / hours_per_year
    thr_impact = float(np.nanpercentile(impact_sp, ai_pct)) if len(impact_sp) else 0.0
    is_high_impact_sp = impact_sp >= thr_impact

    pmin_mask_sp = p_need_sp >= float(config.pmin)
    long_core_mask_sp = is_long_sp & is_high_impact_sp

    # STAGE 2 / MODEL C candidate expansion:
    # Keep every legacy candidate and add items whose probability of at least one
    # demand crosses Pmin at the maximum configured reliability multiplier K.
    # This is a direct uncertainty-envelope rule, not an arbitrary extra percentile.
    robust_k_grid = np.asarray(getattr(config, "robust_k_grid", [1.0]), dtype=float)
    robust_k_grid = robust_k_grid[np.isfinite(robust_k_grid) & (robust_k_grid > 0)]
    if robust_k_grid.size == 0:
        robust_k_grid = np.asarray([1.0], dtype=float)
    robust_k_max = float(np.max(robust_k_grid))
    robust_p_need_max_sp = 1.0 - np.exp(-np.clip(lambda_period_sp * robust_k_max, 0.0, None))
    robust_expand_mask_sp = robust_p_need_max_sp >= float(config.pmin)

    base_candidate_mask_sp = pmin_mask_sp | is_high_impact_sp | is_long_sp | long_core_mask_sp
    if bool(getattr(config, "robust_enabled", False)) and bool(getattr(config, "robust_candidate_expansion_enabled", False)):
        candidate_mask = base_candidate_mask_sp | robust_expand_mask_sp
    else:
        candidate_mask = base_candidate_mask_sp
    robust_added_mask_sp = candidate_mask & (~base_candidate_mask_sp)
    candidate_idx_local = np.where(candidate_mask)[0]
    if len(candidate_idx_local) == 0:
        candidate_idx_local = np.argsort(-impact_sp)[: min(len(spare_idx), max(1, int(config.population_size)))]
    if len(candidate_idx_local) == 0:
        candidate_idx_local = np.arange(len(spare_idx), dtype=int)

    part_id_c = part_id_sp[candidate_idx_local]
    parent_id_c = parent_id_sp[candidate_idx_local]
    level_c = level_sp[candidate_idx_local]
    depth_c = depth_sp[candidate_idx_local]
    echelon_c = echelon_sp[candidate_idx_local]
    stocking_echelon_c = stocking_echelon_sp[candidate_idx_local]
    fr_raw_c = fr_raw_sp[candidate_idx_local]
    lambda_item_c = lambda_item_sp[candidate_idx_local]
    annual_fr_c = lambda_system_sp[candidate_idx_local]
    lambda_eff_c = lambda_eff_sp[candidate_idx_local]
    lambda_demand_c = lambda_demand_sp[candidate_idx_local]
    lam_period_c = lambda_period_sp[candidate_idx_local]
    proc_cost_c = proc_cost_sp[candidate_idx_local]
    lead_c = lead_sp[candidate_idx_local]
    cm_cost_c = cm_cost_sp[candidate_idx_local]
    T_diag_c = T_diag_sp[candidate_idx_local]
    T_reminst_c = T_reminst_sp[candidate_idx_local]
    T_repair_c = T_repair_sp[candidate_idx_local]
    T_restore_intrinsic_c = T_restore_intrinsic_sp[candidate_idx_local]
    T_issue_O_c = T_issue_O_sp[candidate_idx_local]
    T_issue_I_c = T_issue_I_sp[candidate_idx_local]
    T_issue_D_c = T_issue_D_sp[candidate_idx_local]
    baseline_wait_c = baseline_wait_sp[candidate_idx_local]
    shortage_exposure_c = shortage_exposure_sp[candidate_idx_local]
    p_need_c = p_need_sp[candidate_idx_local]
    impact_c = impact_sp[candidate_idx_local]
    is_long_c = is_long_sp[candidate_idx_local]
    is_high_impact_c = is_high_impact_sp[candidate_idx_local]
    is_pmin_over_c = pmin_mask_sp[candidate_idx_local]
    long_core_mask_c = long_core_mask_sp[candidate_idx_local]
    robust_p_need_max_c = robust_p_need_max_sp[candidate_idx_local]
    robust_added_c = robust_added_mask_sp[candidate_idx_local]
    qpa_c = qpa_sp[candidate_idx_local]
    repairable_c = repairable_sp[candidate_idx_local]
    replace_time_c = replace_time_sp[candidate_idx_local]
    repair_tat_c = repair_tat_sp[candidate_idx_local]
    restore_split_c = restore_split_sp[candidate_idx_local]
    exposure_split_c = exposure_split_sp[candidate_idx_local]

    m_cand = len(candidate_idx_local)
    long_idx_c = np.where(is_long_c)[0]
    long_core_thr = float(np.nanpercentile(impact_c[long_idx_c], long_core_pct)) if len(long_idx_c) > 0 else 0.0

    prebuy_flag_c = np.zeros(m_cand, dtype=bool)
    protection_flag_c = np.zeros(m_cand, dtype=bool)
    for j in range(m_cand):
        if is_long_c[j] and (is_pmin_over_c[j] or is_high_impact_c[j]):
            prebuy_flag_c[j] = True
        if is_long_c[j] and (impact_c[j] >= long_core_thr):
            protection_flag_c[j] = True

    impact_norm_c = minmax(impact_c)
    critical_weight_c = np.where(
        impact_norm_c >= 0.80,
        1.00,
        np.where(impact_norm_c >= 0.50, 0.45, 0.15),
    )
    impact_rank_c = percentile_rank_array(impact_c)

    # Canonical Model-C fixed evaluation-universe weights.
    fixed_eval_impact_norm_sp = minmax(impact_sp)
    fixed_eval_critical_weight_sp = np.where(
        fixed_eval_impact_norm_sp >= 0.80,
        1.00,
        np.where(fixed_eval_impact_norm_sp >= 0.50, 0.45, 0.15),
    )
    fixed_eval_impact_rank_sp = percentile_rank_array(impact_sp)

    # Fleet demand intensity corresponding to the sibling-normalized system intensity.
    lambda_eff_demand_sp = lambda_eff_sp * max(float(getattr(config, "fleet_size", 1.0)), 0.0)
    lambda_eff_demand_c = lambda_eff_demand_sp[candidate_idx_local]

    mu_pipe_nom_c = lambda_eff_demand_c * shortage_exposure_c / HOURS_PER_YEAR
    stock_cap_c = stock_cap_from_pipeline_mu(mu_pipe_nom_c, lam_period_c, is_long_c, repairable_c, config)
    prot_floor_c = np.where(protection_flag_c, 1, 0).astype(int)

    # Canonical cost basis: fleet-demand-driven maintenance spend for the full analysis period.
    base_maint_spend = float(np.sum(lambda_demand_c * cm_cost_c * period_years))

    prepared_out = df_prepared.copy()
    prepared_out["is_spare_demand"] = False
    prepared_out.loc[spare_idx, "is_spare_demand"] = True
    prepared_out["candidate"] = False
    prepared_out.loc[spare_idx[candidate_idx_local], "candidate"] = True
    prepared_out["lambda_item_annual"] = 0.0
    prepared_out.loc[spare_idx, "lambda_item_annual"] = lambda_item_sp
    prepared_out["lambda_system_annual"] = 0.0
    prepared_out.loc[spare_idx, "lambda_system_annual"] = lambda_system_sp
    prepared_out["lambda_demand_annual"] = 0.0
    prepared_out.loc[spare_idx, "lambda_demand_annual"] = lambda_demand_sp
    prepared_out["lambda_period_demand"] = 0.0
    prepared_out.loc[spare_idx, "lambda_period_demand"] = lambda_period_sp
    prepared_out["p_need_period"] = 0.0
    prepared_out.loc[spare_idx, "p_need_period"] = p_need_sp
    prepared_out["p_need_2y"] = prepared_out["p_need_period"]
    prepared_out["impact_score"] = 0.0
    prepared_out.loc[spare_idx, "impact_score"] = impact_sp
    prepared_out["shortage_exposure_h"] = 0.0
    prepared_out.loc[spare_idx, "shortage_exposure_h"] = shortage_exposure_sp
    prepared_out["restore_used_h"] = 0.0
    prepared_out.loc[spare_idx, "restore_used_h"] = T_restore_intrinsic_sp
    prepared_out["is_long_lead"] = False
    prepared_out.loc[spare_idx, "is_long_lead"] = is_long_sp
    prepared_out["is_high_ao_impact"] = False
    prepared_out.loc[spare_idx, "is_high_ao_impact"] = is_high_impact_sp
    prepared_out["is_pmin_over"] = False
    prepared_out.loc[spare_idx, "is_pmin_over"] = pmin_mask_sp
    prepared_out["restore_split_applied"] = False
    prepared_out.loc[spare_idx, "restore_split_applied"] = restore_split_sp
    prepared_out["exposure_split_applied"] = False
    prepared_out.loc[spare_idx, "exposure_split_applied"] = exposure_split_sp
    prepared_out["robust_p_need_kmax"] = 0.0
    prepared_out.loc[spare_idx, "robust_p_need_kmax"] = robust_p_need_max_sp
    prepared_out["robust_candidate_added"] = False
    prepared_out.loc[spare_idx, "robust_candidate_added"] = robust_added_mask_sp
    prepared_out["supply_structure_mode"] = supply_structure_mode
    prepared_out["stocking_echelon"] = ""
    prepared_out.loc[spare_idx, "stocking_echelon"] = stocking_echelon_sp

    candidate_reason = []
    for pmin_ok, long_ok, impact_ok, long_core_ok, robust_added in zip(
        is_pmin_over_c, is_long_c, is_high_impact_c, long_core_mask_c, robust_added_c
    ):
        reasons = []
        if pmin_ok:
            reasons.append("p_need")
        if long_ok:
            reasons.append("long_lead")
        if impact_ok:
            reasons.append("ao_impact")
        if long_core_ok:
            reasons.append("long_core")
        if robust_added:
            reasons.append("robust_kmax_p_need")
        candidate_reason.append("+".join(reasons) if reasons else "fallback")

    candidate_df = pd.DataFrame({
        "part_id": part_id_c,
        "parent_id": parent_id_c,
        "level_num": level_c,
        "depth": depth_c,
        "maint_echelon": echelon_c,
        "stocking_echelon": stocking_echelon_c,
        "supply_structure_mode": supply_structure_mode,
        "failure_rate_raw": fr_raw_c,
        "failure_rate_adj": annual_fr_c,
        "lambda_item_annual": lambda_item_c,
        "lambda_system_annual": annual_fr_c,
        "lambda_demand_annual": lambda_demand_c,
        "lambda_period_demand": lam_period_c,
        "lambda_eff": lambda_eff_c,
        "lambda_period": lam_period_c,
        "proc_cost": proc_cost_c,
        "lead_time": lead_c,
        "cm_cost": cm_cost_c,
        "T_diag_h": T_diag_c,
        "T_restore_intrinsic_h": T_restore_intrinsic_c,
        "restore_used_h": T_restore_intrinsic_c,
        "baseline_wait_h": baseline_wait_c,
        "shortage_exposure_h": shortage_exposure_c,
        "p_need_period": p_need_c,
        "p_need_2y": p_need_c,
        "impact_score": impact_c,
        "ao_impact_rank_weight": impact_rank_c,
        "is_long_lead": is_long_c,
        "is_high_ao_impact": is_high_impact_c,
        "is_pmin_over": is_pmin_over_c,
        "candidate_reason": candidate_reason,
        "robust_p_need_kmax": robust_p_need_max_c,
        "robust_candidate_added": robust_added_c,
        "prebuy_flag": prebuy_flag_c,
        "protection_flag": protection_flag_c,
        "critical_weight": critical_weight_c,
        "stock_cap": stock_cap_c,
        "prot_floor": prot_floor_c,
        "qpa": qpa_c,
        "repairable_flag": repairable_c,
        "replace_time_h": replace_time_c,
        "repair_tat_h": repair_tat_c,
        "restore_split_applied": restore_split_c,
        "exposure_split_applied": exposure_split_c,
    })

    return {
        "config": config,
        "prepared_df": prepared_out.reset_index(drop=True),
        "candidate_df": candidate_df.reset_index(drop=True),
        "m_cand": m_cand,
        "part_id_c": part_id_c,
        "parent_id_c": parent_id_c,
        "level_c": level_c,
        "depth_c": depth_c,
        "echelon_c": echelon_c,
        "stocking_echelon_c": stocking_echelon_c,
        "supply_structure_mode": supply_structure_mode,
        "annual_fr_c": annual_fr_c,
        "lambda_eff_c": lambda_eff_c,
        "lambda_eff_demand_c": lambda_eff_demand_c,
        "lambda_demand_c": lambda_demand_c,
        "lambda_item_c": lambda_item_c,
        "lambda_period_c": lam_period_c,
        "proc_cost_c": proc_cost_c,
        "lead_c": lead_c,
        "cm_cost_c": cm_cost_c,
        "T_diag_c": T_diag_c,
        "T_reminst_c": T_reminst_c,
        "T_repair_c": T_repair_c,
        "T_restore_intrinsic_c": T_restore_intrinsic_c,
        "T_issue_O_c": T_issue_O_c,
        "T_issue_I_c": T_issue_I_c,
        "T_issue_D_c": T_issue_D_c,
        "baseline_wait_c": baseline_wait_c,
        "shortage_exposure_c": shortage_exposure_c,
        "p_need_c": p_need_c,
        "impact_c": impact_c,
        "impact_rank_c": impact_rank_c,
        "is_long_c": is_long_c,
        "prebuy_flag_c": prebuy_flag_c,
        "protection_flag_c": protection_flag_c,
        "critical_weight_c": critical_weight_c,
        "stock_cap_c": stock_cap_c,
        "prot_floor_c": prot_floor_c,
        "base_maint_spend": base_maint_spend,
        "period_years": period_years,
        "hours_per_year": hours_per_year,
        "t_obs_hours": t_obs_hours,
        "rep_target": float(config.representative_target),
        "target_grid": [float(x) for x in config.ao_target_grid],
        "robust_k_grid": robust_k_grid.astype(float),
        "robust_k_max": robust_k_max,
        "robust_added_count": int(np.sum(robust_added_c)),

        # Canonical fixed evaluation universe for robust F2.
        "spare_idx": spare_idx,
        "candidate_idx_local": candidate_idx_local,
        "part_id_sp": part_id_sp,
        "echelon_sp": echelon_sp,
        "stocking_echelon_sp": stocking_echelon_sp,
        "lambda_eff_system_sp": lambda_eff_sp,
        "lambda_eff_demand_sp": lambda_eff_demand_sp,
        "lambda_period_sp": lambda_period_sp,
        "T_diag_sp": T_diag_sp,
        "T_restore_intrinsic_sp": T_restore_intrinsic_sp,
        "T_issue_O_sp": T_issue_O_sp,
        "T_issue_I_sp": T_issue_I_sp,
        "T_issue_D_sp": T_issue_D_sp,
        "baseline_wait_sp": baseline_wait_sp,
        "shortage_exposure_sp": shortage_exposure_sp,
        "fixed_eval_critical_weight_sp": fixed_eval_critical_weight_sp,
        "fixed_eval_impact_rank_sp": fixed_eval_impact_rank_sp,
    }


def prepare_input_dataframe(
    df_raw: pd.DataFrame,
    config: Optional[NSGAConfig] = None,
) -> pd.DataFrame:
    df_prepared = _prepare_engine_dataframe(df_raw)
    preview_config = config if config is not None else NSGAConfig()
    try:
        ctx = _build_engine_context(df_prepared, preview_config)
        return ctx["prepared_df"]
    except Exception:
        df_prepared["candidate"] = False
        df_prepared["is_spare_demand"] = False
        df_prepared["p_need_2y"] = 0.0
        df_prepared["impact_score"] = 0.0
        df_prepared["is_long_lead"] = False
        df_prepared["is_high_ao_impact"] = False
        df_prepared["is_pmin_over"] = False
        return df_prepared


def _build_poisson_tail_table(mu_vec: np.ndarray, stock_cap_vec: np.ndarray) -> np.ndarray:
    mu_arr = np.asarray(mu_vec, dtype=float)
    cap_arr = np.asarray(stock_cap_vec, dtype=int)
    if mu_arr.size == 0:
        return np.zeros((0, 1), dtype=float)
    max_cap = int(np.max(cap_arr)) if cap_arr.size else 0
    table = np.zeros((mu_arr.size, max_cap + 1), dtype=float)
    for i, mu in enumerate(mu_arr):
        cap_i = int(cap_arr[i])
        for s in range(cap_i + 1):
            table[i, s] = poisson_tail_prob(float(mu), int(s))
        if cap_i < max_cap:
            table[i, cap_i + 1:] = table[i, cap_i]
    return table


def _build_poisson_ebo_table(mu_vec: np.ndarray, stock_cap_vec: np.ndarray) -> np.ndarray:
    mu_arr = np.asarray(mu_vec, dtype=float)
    cap_arr = np.asarray(stock_cap_vec, dtype=int)
    if mu_arr.size == 0:
        return np.zeros((0, 1), dtype=float)
    max_cap = int(np.max(cap_arr)) if cap_arr.size else 0
    table = np.zeros((mu_arr.size, max_cap + 1), dtype=float)
    for i, mu in enumerate(mu_arr):
        cap_i = int(cap_arr[i])
        for s_i in range(max_cap + 1):
            st = min(int(s_i), cap_i)
            table[i, s_i] = poisson_ebo_scalar(float(mu), st)
    return table


def _ensure_sim_cache(ctx: Dict[str, object]) -> None:
    """
    Model-C multi-echelon simulation caches.

    Inventory rule
    --------------
    Each candidate is stocked at its own Maint_Echelon:
      O item -> Stock_O
      I item -> Stock_I
      D item -> Stock_D

    Shortage probability still uses replenishment exposure
    (Lead Time for non-repairable / Repair TAT for repairable).
    Robust F2 is evaluated over the fixed hard-eligible spare universe.
    """
    if ctx.get("_sim_cache_ready", False):
        return

    stock_cap_c = np.asarray(ctx["stock_cap_c"], dtype=int)
    lambda_eff_system_c = np.asarray(ctx["lambda_eff_c"], dtype=float)
    lambda_eff_demand_c = np.asarray(ctx["lambda_eff_demand_c"], dtype=float)
    shortage_exposure_c = np.asarray(ctx["shortage_exposure_c"], dtype=float)
    critical_weight_c = np.asarray(ctx["critical_weight_c"], dtype=float)
    T_diag_c = np.asarray(ctx["T_diag_c"], dtype=float)
    T_restore_c = np.asarray(ctx["T_restore_intrinsic_c"], dtype=float)
    stocking_echelon_c = np.asarray(ctx["stocking_echelon_c"], dtype=object)

    ctx["_dt_diag_const"] = float(np.sum(lambda_eff_system_c * T_diag_c * critical_weight_c))
    ctx["_dt_restore_const"] = float(np.sum(lambda_eff_system_c * T_restore_c * critical_weight_c))
    ctx["_prebuy_flag_bool"] = np.asarray(ctx["prebuy_flag_c"], dtype=bool)
    ctx["_protection_flag_bool"] = np.asarray(ctx["protection_flag_c"], dtype=bool)

    ctx["_echelon_is_O"] = stocking_echelon_c == "O"
    ctx["_echelon_is_I"] = stocking_echelon_c == "I"
    ctx["_echelon_is_D"] = stocking_echelon_c == "D"

    issue_local_c = np.where(
        ctx["_echelon_is_O"],
        np.asarray(ctx["T_issue_O_c"], dtype=float),
        np.where(
            ctx["_echelon_is_I"],
            np.asarray(ctx["T_issue_I_c"], dtype=float),
            np.asarray(ctx["T_issue_D_c"], dtype=float),
        ),
    )
    ctx["_issue_local_c"] = issue_local_c.astype(float)

    hours_per_year = max(float(ctx.get("hours_per_year", HOURS_PER_YEAR)), 1e-9)
    period_years = max(float(ctx.get("period_years", YEARS)), 1e-9)

    # One local stock quantity per item. Probability uses the engineering
    # replenishment exposure, not the echelon issue time.
    mu_exposure_c = lambda_eff_demand_c * shortage_exposure_c / hours_per_year
    ctx["_p_short_table_local"] = _build_poisson_tail_table(mu_exposure_c, stock_cap_c)

    lambda_period_c = np.asarray(ctx["lambda_period_c"], dtype=float)
    ctx["_ebo_table_period"] = _build_poisson_ebo_table(lambda_period_c, stock_cap_c)
    residual_weight_c = shortage_exposure_c * np.asarray(ctx["impact_rank_c"], dtype=float)
    ctx["_residual_weight_c"] = residual_weight_c.astype(float)
    baseline_ebo_c = poisson_ebo_array(lambda_period_c, np.zeros_like(stock_cap_c, dtype=int))
    ctx["_baseline_residual_shortage_exposure"] = max(
        float(np.sum(baseline_ebo_c * residual_weight_c)),
        1e-12,
    )

    # Fixed robust evaluation universe.
    lambda_eff_demand_sp = np.asarray(ctx["lambda_eff_demand_sp"], dtype=float)
    lambda_period_sp = np.asarray(ctx["lambda_period_sp"], dtype=float)
    shortage_exposure_sp = np.asarray(ctx["shortage_exposure_sp"], dtype=float)
    impact_rank_sp = np.asarray(ctx["fixed_eval_impact_rank_sp"], dtype=float)
    stocking_echelon_sp = np.asarray(ctx["stocking_echelon_sp"], dtype=object)

    issue_local_sp = np.where(
        stocking_echelon_sp == "O",
        np.asarray(ctx["T_issue_O_sp"], dtype=float),
        np.where(
            stocking_echelon_sp == "I",
            np.asarray(ctx["T_issue_I_sp"], dtype=float),
            np.asarray(ctx["T_issue_D_sp"], dtype=float),
        ),
    )
    ctx["_issue_local_sp"] = issue_local_sp.astype(float)

    exposure_reference = poisson_ebo_array(
        lambda_period_sp,
        np.zeros_like(lambda_period_sp, dtype=int),
    ) * shortage_exposure_sp * impact_rank_sp
    ctx["_fixed_eval_exposure_reference"] = max(float(np.sum(exposure_reference)), 1e-12)

    max_cap = int(np.max(stock_cap_c)) if stock_cap_c.size else 0
    candidate_idx_local = np.asarray(ctx["candidate_idx_local"], dtype=int)
    robust_tables = {}
    for kval in np.asarray(ctx.get("robust_k_grid", [1.0]), dtype=float):
        k = float(kval)
        p_table = np.zeros((len(lambda_period_sp), max_cap + 1), dtype=float)
        ebo_table = np.zeros((len(lambda_period_sp), max_cap + 1), dtype=float)
        mu_exp_u = lambda_eff_demand_sp * k * shortage_exposure_sp / hours_per_year
        lam_period_u = lambda_period_sp * k
        for i in range(len(lambda_period_sp)):
            for s in range(max_cap + 1):
                p_table[i, s] = poisson_tail_prob(float(mu_exp_u[i]), int(s))
                ebo_table[i, s] = poisson_ebo_scalar(float(lam_period_u[i]), int(s))
        robust_tables[k] = {"p_local_u": p_table, "ebo_u": ebo_table}

    ctx["_robust_tables_fixed_universe"] = robust_tables
    ctx["_candidate_idx_local_arr"] = candidate_idx_local
    ctx["_sim_cache_ready"] = True

def _simulate_population(
    ctx: Dict[str, object],
    Y_in: np.ndarray,
    SO_in: np.ndarray,
    SI_in: np.ndarray,
    SD_in: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Nominal Model-C evaluator with O/I/D inventory restored.

    A part may hold stock only at its own Maint_Echelon. This prevents artificial
    duplication of the same item's inventory across multiple maintenance levels.
    """
    _ensure_sim_cache(ctx)

    Y = np.asarray(Y_in, dtype=int).copy()
    SO = np.asarray(SO_in, dtype=int).copy()
    SI = np.asarray(SI_in, dtype=int).copy()
    SD = np.asarray(SD_in, dtype=int).copy()
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
        SO = SO.reshape(1, -1)
        SI = SI.reshape(1, -1)
        SD = SD.reshape(1, -1)

    n_pop, m_cand = Y.shape
    prot_floor_c = np.asarray(ctx["prot_floor_c"], dtype=int)
    stock_cap_c = np.asarray(ctx["stock_cap_c"], dtype=int)
    is_O = np.asarray(ctx["_echelon_is_O"], dtype=bool).reshape(1, -1)
    is_I = np.asarray(ctx["_echelon_is_I"], dtype=bool).reshape(1, -1)
    is_D = np.asarray(ctx["_echelon_is_D"], dtype=bool).reshape(1, -1)

    # Force each item to its declared maintenance echelon only.
    SO = np.where(is_O, Y * SO, 0)
    SI = np.where(is_I, Y * SI, 0)
    SD = np.where(is_D, Y * SD, 0)

    total_stock = SO + SI + SD

    # Protection floor: at least one unit at the item's own echelon.
    floor_mask = (
        (Y > 0)
        & (prot_floor_c.reshape(1, -1) > 0)
        & (total_stock < prot_floor_c.reshape(1, -1))
    )
    SO = SO + (floor_mask & is_O).astype(int)
    SI = SI + (floor_mask & is_I).astype(int)
    SD = SD + (floor_mask & is_D).astype(int)

    total_stock = SO + SI + SD

    # Cap the one local-echelon stock quantity.
    total_stock = np.minimum(total_stock, stock_cap_c.reshape(1, -1))
    SO = np.where(is_O, total_stock, 0)
    SI = np.where(is_I, total_stock, 0)
    SD = np.where(is_D, total_stock, 0)

    item_idx = np.arange(m_cand)
    p_short = ctx["_p_short_table_local"][
        item_idx.reshape(1, -1),
        np.clip(total_stock, 0, ctx["_p_short_table_local"].shape[1] - 1),
    ]
    managed = Y > 0
    p_short = np.where(managed, p_short, 1.0)

    baseline_wait = np.asarray(ctx["baseline_wait_c"], dtype=float).reshape(1, -1)
    issue_local = np.asarray(ctx["_issue_local_c"], dtype=float).reshape(1, -1)
    improved_wait = (1.0 - p_short) * issue_local + p_short * baseline_wait
    wait_per_failure = np.where(
        managed,
        np.minimum(baseline_wait, improved_wait),
        baseline_wait,
    )

    lambda_eff_system = np.asarray(ctx["lambda_eff_c"], dtype=float).reshape(1, -1)
    critical_weight = np.asarray(ctx["critical_weight_c"], dtype=float).reshape(1, -1)

    dt_diag_annual = np.full(n_pop, float(ctx["_dt_diag_const"]), dtype=float)
    dt_restore_annual = np.full(n_pop, float(ctx["_dt_restore_const"]), dtype=float)
    dt_wait_annual = np.sum(
        lambda_eff_system * wait_per_failure * critical_weight,
        axis=1,
    )
    dt_total_annual = dt_diag_annual + dt_wait_annual + dt_restore_annual

    period_years = max(float(ctx.get("period_years", YEARS)), 1e-9)
    t_obs_hours = max(float(ctx.get("t_obs_hours", period_years * HOURS_PER_YEAR)), 1e-9)

    dt_diag = dt_diag_annual * period_years
    dt_wait = dt_wait_annual * period_years
    dt_restore = dt_restore_annual * period_years
    dt_total = dt_total_annual * period_years
    ao = np.clip(t_obs_hours / (t_obs_hours + dt_total), 0.0, 1.0)

    proc_cost = np.asarray(ctx["proc_cost_c"], dtype=float).reshape(1, -1)
    stock_cost = np.sum(total_stock * proc_cost, axis=1)
    hold_rate = float(
        getattr(ctx.get("config", None), "holding_cost_rate", H_PRE)
        if ctx.get("config", None) is not None
        else H_PRE
    )
    hold_cost = hold_rate * stock_cost
    total_cost = stock_cost + hold_cost + float(ctx.get("base_maint_spend", 0.0))

    managed_parts = np.sum((Y > 0) & (total_stock > 0), axis=1).astype(float)
    total_stock_sum = np.sum(total_stock, axis=1).astype(float)

    ebo_table = np.asarray(ctx["_ebo_table_period"], dtype=float)
    residual_weight = np.asarray(ctx["_residual_weight_c"], dtype=float).reshape(1, -1)
    ebo_selected = ebo_table[
        item_idx.reshape(1, -1),
        np.clip(total_stock, 0, ebo_table.shape[1] - 1),
    ]
    residual_shortage_exposure = np.sum(ebo_selected * residual_weight, axis=1)
    baseline_residual = float(ctx["_baseline_residual_shortage_exposure"])
    residual_scaled = residual_shortage_exposure / max(baseline_residual, 1e-12)

    return {
        "Y": Y,
        "SO": SO,
        "SI": SI,
        "SD": SD,
        "total_stock_matrix": total_stock,
        "total_stock": total_stock_sum,
        "managed_parts": managed_parts,
        "stock_cost": stock_cost,
        "hold_cost": hold_cost,
        "total_cost": total_cost,
        "Ao": ao,
        "DT_total_h": dt_total,
        "DT_diag_h": dt_diag,
        "DT_wait_h": dt_wait,
        "DT_restore_h": dt_restore,
        "DT_total_annual_h": dt_total_annual,
        "DT_diag_annual_h": dt_diag_annual,
        "DT_wait_annual_h": dt_wait_annual,
        "DT_restore_annual_h": dt_restore_annual,
        "residual_shortage_exposure": residual_shortage_exposure,
        "residual_shortage_exposure_scaled": residual_scaled,
        "p_short_local": p_short,
    }

def _random_population(
    ctx: Dict[str, object],
    config: NSGAConfig,
    rng: np.random.Generator,
    n_pop: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = int(ctx["m_cand"])
    stock_cap = np.asarray(ctx["stock_cap_c"], dtype=int)
    prebuy = np.asarray(ctx["prebuy_flag_c"], dtype=bool)
    protection = np.asarray(ctx["protection_flag_c"], dtype=bool)
    p_need = np.asarray(ctx["p_need_c"], dtype=float)
    impact = minmax(np.asarray(ctx["impact_c"], dtype=float))
    echelon = np.asarray(ctx["stocking_echelon_c"], dtype=object)

    score = np.clip(
        0.35
        + 0.40 * minmax(p_need)
        + 0.25 * impact
        + 0.20 * prebuy.astype(float)
        + 0.25 * protection.astype(float),
        0.05,
        0.95,
    )
    Y = (rng.random((n_pop, m)) < score.reshape(1, -1)).astype(int)
    SO = np.zeros((n_pop, m), dtype=int)
    SI = np.zeros((n_pop, m), dtype=int)
    SD = np.zeros((n_pop, m), dtype=int)

    for j in range(m):
        cap = max(1, int(stock_cap[j]))
        q = rng.integers(0, cap + 1, size=n_pop)
        if protection[j]:
            q = np.maximum(q, 1)
        q = np.where(Y[:, j] > 0, q, 0)

        if echelon[j] == "O":
            SO[:, j] = q
        elif echelon[j] == "I":
            SI[:, j] = q
        else:
            SD[:, j] = q

    return Y, SO, SI, SD

def _extract_solution(sim: Dict[str, np.ndarray], idx: int) -> Dict[str, object]:
    return {
        "Y": sim["Y"][idx].copy(),
        "SO": sim["SO"][idx].copy(),
        "SI": sim["SI"][idx].copy(),
        "SD": sim["SD"][idx].copy(),
        "Ao": float(sim["Ao"][idx]),
        "total_cost": float(sim["total_cost"][idx]),
        "DT_total_h": float(sim["DT_total_h"][idx]),
        "DT_diag_h": float(sim["DT_diag_h"][idx]),
        "DT_wait_h": float(sim["DT_wait_h"][idx]),
        "DT_restore_h": float(sim["DT_restore_h"][idx]),
        "DT_total_annual_h": float(sim["DT_total_annual_h"][idx]),
        "DT_diag_annual_h": float(sim["DT_diag_annual_h"][idx]),
        "DT_wait_annual_h": float(sim["DT_wait_annual_h"][idx]),
        "DT_restore_annual_h": float(sim["DT_restore_annual_h"][idx]),
        "managed_parts": float(sim["managed_parts"][idx]),
        "total_stock": float(sim["total_stock"][idx]),
        "residual_shortage_exposure": float(sim.get("residual_shortage_exposure", np.zeros_like(sim["total_stock"]))[idx]),
        "residual_shortage_exposure_scaled": float(sim.get("residual_shortage_exposure_scaled", np.zeros_like(sim["total_stock"]))[idx]),
    }


def _ensure_robust_cache(ctx: Dict[str, object], config: NSGAConfig) -> None:
    """Precompute K-scenario shortage/EBO lookup tables for Model C."""
    cache_key = tuple(round(float(x), 8) for x in np.asarray(ctx.get("robust_k_grid", [1.0]), dtype=float))
    if ctx.get("_robust_cache_key") == cache_key:
        return

    _ensure_sim_cache(ctx)
    k_grid = np.asarray(ctx.get("robust_k_grid", [1.0]), dtype=float)
    stock_cap = np.asarray(ctx["stock_cap_c"], dtype=int)
    lambda_eff = np.asarray(ctx["lambda_eff_c"], dtype=float)
    lambda_period = np.asarray(ctx["lambda_period_c"], dtype=float)
    hours_per_year = max(float(ctx.get("hours_per_year", HOURS_PER_YEAR)), 1e-9)
    issue_o = np.asarray(ctx["T_issue_O_c"], dtype=float)
    issue_i = np.asarray(ctx["T_issue_I_c"], dtype=float)
    issue_d = np.asarray(ctx["T_issue_D_c"], dtype=float)

    robust_tables = {}
    for k in k_grid:
        kk = float(k)
        robust_tables[kk] = {
            "pO": _build_poisson_tail_table(lambda_eff * kk * issue_o / hours_per_year, stock_cap),
            "pI": _build_poisson_tail_table(lambda_eff * kk * issue_i / hours_per_year, stock_cap),
            "pD": _build_poisson_tail_table(lambda_eff * kk * issue_d / hours_per_year, stock_cap),
            "ebo": _build_poisson_ebo_table(lambda_period * kk, stock_cap),
        }
    ctx["_robust_tables"] = robust_tables
    ctx["_robust_cache_key"] = cache_key


def _evaluate_robust_population(
    ctx: Dict[str, object],
    config: NSGAConfig,
    sim: Dict[str, np.ndarray],
    ao_target: float,
) -> Dict[str, np.ndarray]:
    """
    Robust F2 over the fixed hard-eligible spare universe with O/I/D stock restored.

    Each candidate's total local stock (SO+SI+SD) is mapped to the fixed universe.
    The stock location is its Maint_Echelon, and the corresponding O/I/D issue time
    is used in Ao diagnostics.
    """
    _ensure_sim_cache(ctx)

    Y = np.asarray(sim["Y"], dtype=int)
    total_stock_c = (
        np.asarray(sim["SO"], dtype=int)
        + np.asarray(sim["SI"], dtype=int)
        + np.asarray(sim["SD"], dtype=int)
    )
    n_pop = Y.shape[0]

    k_grid = np.asarray(ctx.get("robust_k_grid", [1.0]), dtype=float)
    k_grid = k_grid[np.isfinite(k_grid) & (k_grid > 0)]
    if k_grid.size == 0:
        k_grid = np.asarray([1.0], dtype=float)

    candidate_idx = np.asarray(ctx["_candidate_idx_local_arr"], dtype=int)
    n_u = len(np.asarray(ctx["lambda_period_sp"], dtype=float))
    n_scen = len(k_grid)

    lambda_system_u = np.asarray(ctx["lambda_eff_system_sp"], dtype=float)
    T_diag_u = np.asarray(ctx["T_diag_sp"], dtype=float)
    T_restore_u = np.asarray(ctx["T_restore_intrinsic_sp"], dtype=float)
    issue_local_u = np.asarray(ctx["_issue_local_sp"], dtype=float)
    baseline_wait_u = np.asarray(ctx["baseline_wait_sp"], dtype=float)
    shortage_exposure_u = np.asarray(ctx["shortage_exposure_sp"], dtype=float)
    impact_rank_u = np.asarray(ctx["fixed_eval_impact_rank_sp"], dtype=float)
    critical_weight_u = np.asarray(ctx["fixed_eval_critical_weight_sp"], dtype=float)
    exposure_reference = max(float(ctx["_fixed_eval_exposure_reference"]), 1e-12)

    period_years = max(float(ctx.get("period_years", YEARS)), 1e-9)
    t_obs_hours = max(float(ctx.get("t_obs_hours", period_years * HOURS_PER_YEAR)), 1e-9)

    ao_mat = np.zeros((n_pop, n_scen), dtype=float)
    shortage_qty_mat = np.zeros((n_pop, n_scen), dtype=float)
    exposure_risk_mat = np.zeros((n_pop, n_scen), dtype=float)

    total_stock_u = np.zeros((n_pop, n_u), dtype=int)
    managed_u = np.zeros((n_pop, n_u), dtype=bool)
    total_stock_u[:, candidate_idx] = total_stock_c
    managed_u[:, candidate_idx] = (Y > 0)

    for sidx, kval in enumerate(k_grid):
        k = float(kval)
        table = ctx["_robust_tables_fixed_universe"][k]
        max_s = table["p_local_u"].shape[1] - 1
        stock_idx = np.clip(total_stock_u, 0, max_s)
        uidx = np.arange(n_u).reshape(1, -1)

        p_short = table["p_local_u"][uidx, stock_idx]
        p_short = np.where(managed_u, p_short, 1.0)

        improved_wait = (
            (1.0 - p_short) * issue_local_u.reshape(1, -1)
            + p_short * baseline_wait_u.reshape(1, -1)
        )
        wait_per_failure = np.where(
            managed_u,
            np.minimum(baseline_wait_u.reshape(1, -1), improved_wait),
            baseline_wait_u.reshape(1, -1),
        )

        lam_sys = lambda_system_u.reshape(1, -1) * k
        cw = critical_weight_u.reshape(1, -1)
        dt_diag_ann = np.sum(lam_sys * T_diag_u.reshape(1, -1) * cw, axis=1)
        dt_restore_ann = np.sum(lam_sys * T_restore_u.reshape(1, -1) * cw, axis=1)
        dt_wait_ann = np.sum(lam_sys * wait_per_failure * cw, axis=1)
        dt_total = (dt_diag_ann + dt_restore_ann + dt_wait_ann) * period_years
        ao_mat[:, sidx] = np.clip(t_obs_hours / (t_obs_hours + dt_total), 0.0, 1.0)

        ebo = table["ebo_u"][uidx, stock_idx]
        shortage_qty_mat[:, sidx] = np.sum(ebo, axis=1)
        exposure_total = np.sum(
            ebo
            * shortage_exposure_u.reshape(1, -1)
            * impact_rank_u.reshape(1, -1),
            axis=1,
        )
        exposure_risk_mat[:, sidx] = exposure_total / exposure_reference

    tail_count = min(2, n_scen)
    sorted_risk = np.sort(exposure_risk_mat, axis=1)
    tail_risk = np.mean(sorted_risk[:, -tail_count:], axis=1)

    worst_ao = np.min(ao_mat, axis=1)
    worst_shortage = np.max(shortage_qty_mat, axis=1)
    worst_idx = np.argmax(exposure_risk_mat, axis=1)
    worst_k = k_grid[worst_idx]
    nominal_idx = int(np.argmin(np.abs(k_grid - 1.0)))
    nominal_ao = ao_mat[:, nominal_idx]
    robustness_gap = np.maximum(0.0, nominal_ao - worst_ao)

    return {
        "k_grid": k_grid,
        "ao_matrix": ao_mat,
        "shortage_matrix": shortage_qty_mat,
        "residual_scaled_matrix": exposure_risk_mat,
        "scenario_loss_matrix": exposure_risk_mat,
        "tail_risk": tail_risk,
        "worst_Ao": worst_ao,
        "worst_shortage": worst_shortage,
        "robustness_gap": robustness_gap,
        "worst_k": worst_k,
        "nominal_Ao": nominal_ao,
        "tail_count": np.full(n_pop, tail_count, dtype=int),
    }

def _constraint_dominates(i: int, j: int, objectives: np.ndarray, violation: np.ndarray) -> bool:
    vi, vj = float(violation[i]), float(violation[j])
    if vi <= 1e-12 and vj > 1e-12:
        return True
    if vi > 1e-12 and vj <= 1e-12:
        return False
    if vi > 1e-12 and vj > 1e-12:
        return vi < vj - 1e-12
    return bool(np.all(objectives[i] <= objectives[j] + 1e-12) and np.any(objectives[i] < objectives[j] - 1e-12))


def _fast_non_dominated_sort(
    objectives: np.ndarray,
    violation: np.ndarray,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Vectorized constraint-aware non-dominated sorting.

    The previous implementation performed Python-level p×q comparisons every
    generation. With POP=80 and parent+offspring=160 this became the dominant
    runtime cost on the real 1,000-part BOM. This version builds the same
    dominance relation with NumPy broadcasting, then performs only front peeling
    in Python.
    """
    obj = np.asarray(objectives, dtype=float)
    vio = np.asarray(violation, dtype=float)
    n = len(obj)
    if n == 0:
        return [], np.empty(0, dtype=int)

    tol = 1e-12
    feasible = vio <= tol

    # dom[i, j] == True means i constraint-dominates j.
    dom = np.zeros((n, n), dtype=bool)

    # Feasible always dominates infeasible.
    dom |= feasible[:, None] & (~feasible[None, :])

    # Both infeasible: smaller violation dominates.
    both_infeasible = (~feasible[:, None]) & (~feasible[None, :])
    dom |= both_infeasible & (vio[:, None] < (vio[None, :] - tol))

    # Both feasible: standard Pareto dominance.
    fi = np.flatnonzero(feasible)
    if fi.size:
        fobj = obj[fi]
        le = np.all(fobj[:, None, :] <= fobj[None, :, :] + tol, axis=2)
        lt = np.any(fobj[:, None, :] < fobj[None, :, :] - tol, axis=2)
        fdom = le & lt
        dom[np.ix_(fi, fi)] |= fdom

    np.fill_diagonal(dom, False)

    dominated_count = dom.sum(axis=0).astype(int)
    rank = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)
    fronts: List[List[int]] = []
    level = 0

    while remaining.any():
        front_arr = np.flatnonzero(remaining & (dominated_count == 0))
        if front_arr.size == 0:
            # Numerical safeguard: should never happen for a strict dominance DAG.
            front_arr = np.flatnonzero(remaining)
        rank[front_arr] = level
        fronts.append(front_arr.astype(int).tolist())
        remaining[front_arr] = False

        # Remove outgoing dominance edges from this front.
        if remaining.any():
            dominated_count -= dom[front_arr].sum(axis=0).astype(int)
            dominated_count[~remaining] = -1
        level += 1

    return fronts, rank


def _crowding_distance(objectives: np.ndarray, front: List[int]) -> Dict[int, float]:
    if not front:
        return {}
    dist = {int(i): 0.0 for i in front}
    if len(front) <= 2:
        return {int(i): float("inf") for i in front}
    sub = objectives[np.asarray(front, dtype=int)]
    nobj = sub.shape[1]
    for m in range(nobj):
        order_local = np.argsort(sub[:, m], kind="mergesort")
        vals = sub[order_local, m]
        dist[int(front[order_local[0]])] = float("inf")
        dist[int(front[order_local[-1]])] = float("inf")
        span = float(vals[-1] - vals[0])
        if abs(span) < 1e-12:
            continue
        for pos in range(1, len(order_local) - 1):
            idx = int(front[order_local[pos]])
            if np.isinf(dist[idx]):
                continue
            dist[idx] += float((vals[pos + 1] - vals[pos - 1]) / span)
    return dist


def _evaluate_model_c_population(ctx: Dict[str, object], config: NSGAConfig, ao_target: float, Y, SO, SI, SD) -> Dict[str, object]:
    sim = _simulate_population(ctx, Y, SO, SI, SD)
    robust = _evaluate_robust_population(ctx, config, sim, ao_target)
    complexity = float(config.item_count_weight) * sim["managed_parts"] + float(config.stock_unit_weight) * sim["total_stock"]
    objectives = np.column_stack([
        sim["total_cost"].astype(float),
        robust["tail_risk"].astype(float),
        complexity.astype(float),
    ])
    # Preserve the original target-Ao interpretation as the feasibility constraint.
    violation = np.maximum(0.0, float(ao_target) - sim["Ao"])
    return {"sim": sim, "robust": robust, "objectives": objectives, "violation": violation}


def _extract_model_c_solution(evaluation: Dict[str, object], idx: int) -> Dict[str, object]:
    sol = _extract_solution(evaluation["sim"], idx)
    r = evaluation["robust"]
    obj = evaluation["objectives"]
    sol.update({
        "tail_risk": float(r["tail_risk"][idx]),
        "worst_Ao": float(r["worst_Ao"][idx]),
        "worst_shortage": float(r["worst_shortage"][idx]),
        "robustness_gap": float(r["robustness_gap"][idx]),
        "worst_k": float(r["worst_k"][idx]),
        "nominal_Ao_robust_grid": float(r["nominal_Ao"][idx]),
        "F1_cost": float(obj[idx, 0]),
        "F2_tail_risk": float(obj[idx, 1]),
        "F3_complexity": float(obj[idx, 2]),
        "constraint_violation": float(evaluation["violation"][idx]),
    })
    return sol


def _tournament_index(rng: np.random.Generator, rank: np.ndarray, crowd: np.ndarray) -> int:
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]:
        return int(a)
    if rank[b] < rank[a]:
        return int(b)
    if crowd[a] > crowd[b]:
        return int(a)
    if crowd[b] > crowd[a]:
        return int(b)
    return int(a if rng.random() < 0.5 else b)


def _make_offspring(
    ctx: Dict[str, object],
    config: NSGAConfig,
    rng: np.random.Generator,
    Y, SO, SI, SD, rank, crowd,
):
    n, m = Y.shape
    cY = np.empty_like(Y)
    cO = np.zeros_like(SO)
    cI = np.zeros_like(SI)
    cD = np.zeros_like(SD)

    cap = np.asarray(ctx["stock_cap_c"], dtype=int)
    protection = np.asarray(ctx["protection_flag_c"], dtype=bool)
    echelon = np.asarray(ctx["stocking_echelon_c"], dtype=object)
    pc = float(np.clip(getattr(config, "nsga_crossover_prob", 0.90), 0.0, 1.0))
    pm = max(
        float(np.clip(getattr(config, "nsga_mutation_prob", 0.02), 0.0, 1.0)),
        1.0 / max(m, 1),
    )

    parent_qty = SO + SI + SD

    for out_i in range(0, n, 2):
        p1 = _tournament_index(rng, rank, crowd)
        p2 = _tournament_index(rng, rank, crowd)
        y1, q1 = Y[p1].copy(), parent_qty[p1].copy()
        y2, q2 = Y[p2].copy(), parent_qty[p2].copy()

        if rng.random() < pc:
            mask = rng.random(m) < 0.5
            yy1, yy2 = y1.copy(), y2.copy()
            qq1, qq2 = q1.copy(), q2.copy()
            y1[mask], y2[mask] = yy2[mask], yy1[mask]
            q1[mask], q2[mask] = qq2[mask], qq1[mask]

        for child_pos, (y, q) in enumerate([(y1, q1), (y2, q2)]):
            target = out_i + child_pos
            if target >= n:
                break

            mut_mask = rng.random(m) < pm
            for j in np.where(mut_mask)[0]:
                if rng.random() < 0.35:
                    y[j] = 1 - int(y[j])
                qj = int(rng.integers(0, max(1, int(cap[j])) + 1))
                if protection[j] and y[j] > 0:
                    qj = max(qj, 1)
                q[j] = qj

            q = np.where(y > 0, q, 0)
            q = np.minimum(q, cap)
            floor_mask = (y > 0) & protection & (q < 1)
            q = np.where(floor_mask, 1, q)

            cY[target] = y
            for j in range(m):
                if echelon[j] == "O":
                    cO[target, j] = q[j]
                elif echelon[j] == "I":
                    cI[target, j] = q[j]
                else:
                    cD[target, j] = q[j]

    return cY, cO, cI, cD

def _environmental_select(evaluation: Dict[str, object], n_keep: int) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    fronts, rank = _fast_non_dominated_sort(evaluation["objectives"], evaluation["violation"])
    selected = []
    crowd_all = np.zeros(len(rank), dtype=float)
    for front in fronts:
        cd = _crowding_distance(evaluation["objectives"], front)
        for i, v in cd.items():
            crowd_all[i] = v
        if len(selected) + len(front) <= n_keep:
            selected.extend(front)
        else:
            remain = n_keep - len(selected)
            chosen = sorted(front, key=lambda i: cd.get(int(i), 0.0), reverse=True)[:remain]
            selected.extend(chosen)
            break
    return np.asarray(selected, dtype=int), crowd_all, fronts


def _run_nsga_single_target(
    ctx: Dict[str, object],
    config: NSGAConfig,
    ao_target: float,
    seed: int,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    """Stage-2 explicit NSGA-II for Model C robust optimization."""
    rng = np.random.default_rng(seed)
    n_pop = max(int(config.population_size), 8)
    n_gen = max(int(config.n_generations), 1)

    Y, SO, SI, SD = _random_population(ctx, config, rng, n_pop)
    history_rows: List[Dict[str, float]] = []

    for gen in range(1, n_gen + 1):
        parent_eval = _evaluate_model_c_population(ctx, config, ao_target, Y, SO, SI, SD)
        fronts, rank = _fast_non_dominated_sort(parent_eval["objectives"], parent_eval["violation"])
        crowd = np.zeros(n_pop, dtype=float)
        for front in fronts:
            for i, v in _crowding_distance(parent_eval["objectives"], front).items():
                crowd[i] = v

        cY, cO, cI, cD = _make_offspring(ctx, config, rng, Y, SO, SI, SD, rank, crowd)
        combY = np.vstack([Y, cY])
        combO = np.vstack([SO, cO])
        combI = np.vstack([SI, cI])
        combD = np.vstack([SD, cD])
        comb_eval = _evaluate_model_c_population(ctx, config, ao_target, combY, combO, combI, combD)
        keep, _, _ = _environmental_select(comb_eval, n_pop)
        Y, SO, SI, SD = combY[keep], combO[keep], combI[keep], combD[keep]

        current_eval = _evaluate_model_c_population(ctx, config, ao_target, Y, SO, SI, SD)
        feasible = np.where(current_eval["violation"] <= 1e-12)[0]
        if feasible.size:
            best_idx = int(feasible[np.lexsort((current_eval["robust"]["tail_risk"][feasible], current_eval["sim"]["total_cost"][feasible]))[0]])
        else:
            best_idx = int(np.argmin(current_eval["violation"]))
        best_sol = _extract_model_c_solution(current_eval, best_idx)
        history_rows.append({
            "generation": gen,
            "target_ao": float(ao_target),
            "best_Ao": float(best_sol["Ao"]),
            "best_cost": float(best_sol["total_cost"]),
            "best_total_stock": float(best_sol["total_stock"]),
            "best_managed_parts": float(best_sol["managed_parts"]),
            "best_tail_risk": float(best_sol["tail_risk"]),
            "best_worst_Ao": float(best_sol["worst_Ao"]),
            "best_worst_shortage": float(best_sol["worst_shortage"]),
            "best_robustness_gap": float(best_sol["robustness_gap"]),
        })
        if progress_callback is not None:
            progress_callback(gen, n_gen, history_rows[-1])

    final_eval = _evaluate_model_c_population(ctx, config, ao_target, Y, SO, SI, SD)
    fronts, rank = _fast_non_dominated_sort(final_eval["objectives"], final_eval["violation"])
    front0 = fronts[0] if fronts and fronts[0] else list(range(n_pop))

    unique = {}
    for idx in front0:
        sol = _extract_model_c_solution(final_eval, int(idx))
        sig = solution_signature(sol["Y"], sol["SO"], sol["SI"], sol["SD"])
        if sig not in unique:
            unique[sig] = sol
    pareto_solutions = list(unique.values())

    feasible = [s for s in pareto_solutions if float(s.get("constraint_violation", 0.0)) <= 1e-12]
    if feasible:
        best = min(feasible, key=lambda s: (float(s["total_cost"]), float(s["tail_risk"]), float(s["total_stock"])))
    else:
        best = min(pareto_solutions, key=lambda s: (float(s.get("constraint_violation", 1e9)), float(s["tail_risk"]), float(s["total_cost"])))

    return {
        "target_ao": float(ao_target),
        "history_df": pd.DataFrame(history_rows),
        "pareto_solutions": pareto_solutions,
        "best_solution": best,
    }

def _collect_run_dataframe(ctx: Dict[str, object], res_t: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    target = float(res_t.get("target_ao", np.nan))
    for sol in res_t.get("pareto_solutions", []):
        rows.append({
            "target_ao": target,
            "Ao": float(sol["Ao"]),
            "total_cost": float(sol["total_cost"]),
            "DT_total_h": float(sol["DT_total_h"]),
            "DT_diag_h": float(sol["DT_diag_h"]),
            "DT_wait_h": float(sol["DT_wait_h"]),
            "DT_restore_h": float(sol["DT_restore_h"]),
            "DT_total_annual_h": float(sol.get("DT_total_annual_h", 0.0)),
            "DT_diag_annual_h": float(sol.get("DT_diag_annual_h", 0.0)),
            "DT_wait_annual_h": float(sol.get("DT_wait_annual_h", 0.0)),
            "DT_restore_annual_h": float(sol.get("DT_restore_annual_h", 0.0)),
            "managed_parts": float(sol["managed_parts"]),
            "total_stock": float(sol["total_stock"]),
            "residual_shortage_exposure": float(sol.get("residual_shortage_exposure", 0.0)),
            "residual_shortage_exposure_scaled": float(sol.get("residual_shortage_exposure_scaled", 0.0)),
            "Tail_Risk": float(sol.get("tail_risk", 0.0)),
            "Worst_Ao": float(sol.get("worst_Ao", sol.get("Ao", 0.0))),
            "Worst_Shortage": float(sol.get("worst_shortage", 0.0)),
            "Robustness_Gap": float(sol.get("robustness_gap", 0.0)),
            "Worst_K": float(sol.get("worst_k", 1.0)),
            "F1_Cost": float(sol.get("F1_cost", sol.get("total_cost", 0.0))),
            "F2_Tail_Risk": float(sol.get("F2_tail_risk", sol.get("tail_risk", 0.0))),
            "F3_Complexity": float(sol.get("F3_complexity", 0.0)),
            "Constraint_Violation": float(sol.get("constraint_violation", 0.0)),
        })
    return pd.DataFrame(rows)

def _build_policy_df(ctx: Dict[str, object], sol: Dict[str, object]) -> pd.DataFrame:
    Y = np.asarray(sol["Y"], dtype=int)
    SO = np.asarray(sol["SO"], dtype=int)
    SI = np.asarray(sol["SI"], dtype=int)
    SD = np.asarray(sol["SD"], dtype=int)
    total_stock = SO + SI + SD
    prebuy_bool = np.asarray(ctx["prebuy_flag_c"], dtype=bool)
    protection_bool = np.asarray(ctx["protection_flag_c"], dtype=bool)
    priority_score = minmax(np.asarray(ctx["impact_c"], dtype=float)) * 0.7 + minmax(np.asarray(ctx["p_need_c"], dtype=float)) * 0.3
    manage_flag = (Y > 0).astype(int)
    reorder_signal = np.where((manage_flag > 0) & (total_stock <= np.asarray(ctx["prot_floor_c"], dtype=int)), "CHECK", "")
    df_policy = pd.DataFrame({
        "part_id": ctx["part_id_c"],
        "parent_id": ctx["parent_id_c"],
        "level_num": ctx["level_c"],
        "maint_echelon": ctx["echelon_c"],
        "stocking_echelon": ctx["stocking_echelon_c"],
        "supply_structure_mode": ctx["supply_structure_mode"],
        "manage_flag": manage_flag,
        "recommended_stock": total_stock,
        "stock_O": SO,
        "stock_I": SI,
        "stock_D": SD,
        "total_stock": total_stock,
        "unit_cost": ctx["proc_cost_c"],
        "expected_demand_period": ctx["lambda_period_c"],
        "p_need_period": ctx["p_need_c"],
        "expected_demand_2y": ctx["lambda_period_c"],
        "p_need_2y": ctx["p_need_c"],
        "lead_time": ctx["lead_c"],
        "impact_score": ctx["impact_c"],
        "priority_score": priority_score,
        "prebuy_flag": prebuy_bool,
        "protection_flag": protection_bool,
        "selected_prebuy_unit": np.where((prebuy_bool) & (total_stock > 0), total_stock, 0),
        "selected_protection_unit": np.where((protection_bool) & (total_stock > 0), total_stock, 0),
        "normal_stock_unit": np.where((~prebuy_bool) & (~protection_bool) & (total_stock > 0), total_stock, 0),
        "qpa": ctx["candidate_df"]["qpa"].to_numpy() if "qpa" in ctx["candidate_df"].columns else 1.0,
        "repairable_flag": ctx["candidate_df"]["repairable_flag"].to_numpy() if "repairable_flag" in ctx["candidate_df"].columns else 0,
        "replace_time_h": ctx["candidate_df"]["replace_time_h"].to_numpy() if "replace_time_h" in ctx["candidate_df"].columns else 0.0,
        "repair_tat_h": ctx["candidate_df"]["repair_tat_h"].to_numpy() if "repair_tat_h" in ctx["candidate_df"].columns else 0.0,
        "reorder_signal": reorder_signal,
    })
    return df_policy.sort_values(by=["manage_flag", "priority_score", "recommended_stock"], ascending=[False, False, False]).reset_index(drop=True)


def _build_final_spares_df(ctx: Dict[str, object], sol: Dict[str, object]) -> pd.DataFrame:
    policy = _build_policy_df(ctx, sol).copy()
    policy["final_initial_spare_qty"] = policy["recommended_stock"].astype(int)
    candidate_df = ctx.get("candidate_df", pd.DataFrame()).copy()
    keep_cols = [
        "part_id", "lambda_item_annual", "lambda_system_annual", "lambda_demand_annual",
        "lambda_period_demand", "p_need_period", "shortage_exposure_h", "stock_cap",
        "candidate_reason",
    ]
    merge_cols = [c for c in keep_cols if c in candidate_df.columns]
    if merge_cols and "part_id" in merge_cols:
        policy = policy.merge(candidate_df[merge_cols].drop_duplicates("part_id"), on="part_id", how="left")
    return policy.reset_index(drop=True)


def _build_repairable_audit_df(ctx: Dict[str, object]) -> pd.DataFrame:
    cand = ctx.get("candidate_df", pd.DataFrame()).copy()
    cols = [
        "part_id", "parent_id", "level_num", "maint_echelon", "repairable_flag",
        "replace_time_h", "repair_tat_h", "restore_used_h", "shortage_exposure_h",
        "restore_split_applied", "exposure_split_applied", "stock_cap",
    ]
    existing = [c for c in cols if c in cand.columns]
    if not existing:
        return pd.DataFrame(columns=cols)
    return cand[existing].reset_index(drop=True)


def _build_diagnostic_df(ctx: Dict[str, object]) -> pd.DataFrame:
    cand = ctx.get("candidate_df", pd.DataFrame()).copy()
    cols = [
        "part_id", "parent_id", "level_num", "maint_echelon", "failure_rate", "qpa",
        "lambda_item_annual", "lambda_system_annual", "lambda_demand_annual",
        "lambda_period_demand", "p_need_period", "impact_score", "impact_rank",
        "is_long_lead", "is_high_ao_impact", "is_pmin_over", "long_core_flag",
        "candidate_reason", "robust_p_need_kmax", "robust_candidate_added", "stock_cap",
    ]
    existing = [c for c in cols if c in cand.columns]
    if not existing:
        return pd.DataFrame(columns=cols)
    return cand[existing].reset_index(drop=True)


def _build_stage1_readiness_df(
    df_prepared: pd.DataFrame,
    config: NSGAConfig,
    readiness: Dict[str, object],
) -> pd.DataFrame:
    """Additive audit table for future Model C + Monte-Carlo integration."""
    cols = set(df_prepared.columns)

    def present(name: str) -> bool:
        return name in cols

    def positive_count(name: str) -> int:
        if name not in cols:
            return 0
        return int(pd.to_numeric(df_prepared[name], errors="coerce").fillna(0).gt(0).sum())

    rows = [
        ("Part_ID", present("part_id"), int(df_prepared["part_id"].notna().sum()) if present("part_id") else 0, "NSGA/MC common key"),
        ("Failure_Rate", present("failure_rate"), positive_count("failure_rate"), "base failure intensity"),
        ("QPA", present("qpa"), positive_count("qpa"), "quantity per equipment"),
        ("Annual_Rad_Hours", present("annual_rad_hours"), positive_count("annual_rad_hours"), "future MC per-row operating-hours input; zero is allowed in Stage 1"),
        ("Lead_Time", present("lead_time"), positive_count("lead_time"), "non-repairable replenishment exposure"),
        ("Repairable_Flag", present("repairable_flag"), int(pd.to_numeric(df_prepared.get("repairable_flag", 0), errors="coerce").fillna(0).gt(0).sum()) if present("repairable_flag") else 0, "repairable/non-repairable split"),
        ("Replace_Time_H", present("replace_time_h"), positive_count("replace_time_h"), "equipment restoration time"),
        ("Repair_TAT_H", present("repair_tat_h"), positive_count("repair_tat_h"), "repair-pipeline exposure"),
        ("Unit_Cost", present("unit_cost"), positive_count("unit_cost"), "cost/risk decision layer"),
        ("Maint_Echelon", present("maint_echelon"), int(df_prepared["maint_echelon"].notna().sum()) if present("maint_echelon") else 0, "O/I/D logistics layer"),
    ]
    out = pd.DataFrame(rows, columns=["field", "available", "positive_or_nonnull_count", "future_role"])
    out["stage1_config_ready"] = bool(readiness.get("stage1_ready", False))
    out["config_issue_count"] = len(readiness.get("issues", []))
    return out



REPRESENTATIVE_STRATEGY_NAMES = [
    "최소재고형",
    "경량 대응형",
    "균형 대응형",
    "안정형",
    "최대안정형",
]


def _strategy_names_for_count(n: int) -> List[str]:
    """Return interpretable strategy labels even when fewer than five unique stock levels exist."""
    n = max(int(n), 0)
    if n <= 0:
        return []
    if n == 1:
        return ["균형 대응형"]
    if n == 2:
        return ["최소재고형", "최대안정형"]
    if n == 3:
        return ["최소재고형", "균형 대응형", "최대안정형"]
    if n == 4:
        return ["최소재고형", "경량 대응형", "안정형", "최대안정형"]
    return list(REPRESENTATIVE_STRATEGY_NAMES[:n])


def _candidate_tiebreak_key(sol: Dict[str, object], source_solution_id: int) -> Tuple[float, float, float, float, int]:
    """
    Tie-break within the same total-stock level.
    Lower Tail Risk -> higher Worst Ao -> lower Cost -> lower Robustness Gap -> stable source id.
    """
    return (
        float(sol.get("tail_risk", np.inf)),
        -float(sol.get("worst_Ao", sol.get("Ao", 0.0))),
        float(sol.get("total_cost", np.inf)),
        float(sol.get("robustness_gap", np.inf)),
        int(source_solution_id),
    )


def _select_representative_solutions(
    ctx: Dict[str, object],
    result_map: Dict[float, Dict[str, object]],
    representative_target: float,
    config: NSGAConfig,
) -> Dict[str, object]:
    """
    STAGE 3 Bridge — common-basis representative strategy selection.

    Why the entire Target Sweep is used
    -----------------------------------
    A single target's Pareto front can contain fewer than five distinct stock levels.
    To preserve the intended cost/risk strategy spectrum without inventing solutions,
    the bridge first pools Pareto solutions generated across the entire Target Sweep.

    Common-basis re-evaluation
    --------------------------
    Tail Risk contains Target-Ao shortfall, so values created under different targets
    are not directly comparable. Every pooled solution is therefore re-evaluated under
    ONE common representative target (default 0.94) before representative selection.

    Selection
    ---------
    1) Pool and de-duplicate all Target-Sweep Pareto solutions by inventory signature.
    2) Re-evaluate every pooled solution under the common representative target.
    3) Compute the common-basis non-dominated set using Cost / Tail Risk / Complexity.
       For representative-spectrum construction, target violation is shown but is not
       used as a dominance gate; this permits an explicit low-cost/risk-accepting option.
    4) If the common non-dominated set has too few unique stock levels, add common-basis
       evaluated solutions only as a transparent stock-coverage fallback.
    5) At the same stock level choose: lower Tail Risk -> higher Worst Ao ->
       lower Cost -> lower Robustness Gap.
    6) Select stock-spectrum positions at 0/25/50/75/100% by default.

    Strategy labels are descriptive display labels, never an optimality ranking.
    """
    supply_structure_mode = str(
        ctx.get("supply_structure_mode", normalize_supply_structure_mode(config))
    )
    pooled = []
    seen_signature = set()

    for source_target in sorted(result_map.keys()):
        res = result_map[source_target]
        for source_solution_id, sol in enumerate(res.get("pareto_solutions", [])):
            sig = solution_signature(sol["Y"], sol["SO"], sol["SI"], sol["SD"])
            if sig in seen_signature:
                continue
            seen_signature.add(sig)
            pooled.append({
                "source_target_ao": float(source_target),
                "source_solution_id": int(source_solution_id),
                "solution": sol,
                "signature": sig,
            })

    if not pooled:
        return {
            "representative_alternatives_df": pd.DataFrame(),
            "representative_inventory_df": pd.DataFrame(),
            "monte_carlo_bridge_df": pd.DataFrame(),
            "representative_selection_audit_df": pd.DataFrame(),
            "selected_solutions": [],
        }

    Y = np.vstack([np.asarray(x["solution"]["Y"], dtype=int) for x in pooled])
    SO = np.vstack([np.asarray(x["solution"]["SO"], dtype=int) for x in pooled])
    SI = np.vstack([np.asarray(x["solution"]["SI"], dtype=int) for x in pooled])
    SD = np.vstack([np.asarray(x["solution"]["SD"], dtype=int) for x in pooled])

    common_eval = _evaluate_model_c_population(
        ctx, config, float(representative_target), Y, SO, SI, SD
    )

    # Common-basis trade-off non-dominance: do not gate by target feasibility here.
    zero_violation = np.zeros(len(pooled), dtype=float)
    common_fronts, _ = _fast_non_dominated_sort(common_eval["objectives"], zero_violation)
    common_front0 = set(common_fronts[0] if common_fronts else range(len(pooled)))

    common_records = []
    for i, meta in enumerate(pooled):
        common_sol = _extract_model_c_solution(common_eval, i)
        common_records.append({
            **meta,
            "common_solution": common_sol,
            "common_nondominated": bool(i in common_front0),
            "common_target_feasible": bool(float(common_sol.get("Ao", 0.0)) >= float(representative_target) - 1e-12),
        })

    desired = max(int(getattr(config, "representative_strategy_count", 5)), 1)

    nondom_records = [x for x in common_records if x["common_nondominated"]]
    nondom_stock_levels = {int(round(float(x["common_solution"].get("total_stock", 0.0)))) for x in nondom_records}

    # Prefer the common non-dominated set. If stock-level coverage is insufficient,
    # widen only enough to the common-evaluated pool to produce an interpretable spectrum.
    if len(nondom_stock_levels) >= min(desired, 5):
        selection_pool = nondom_records
        pool_mode = "COMMON_NONDOMINATED"
    else:
        selection_pool = common_records
        pool_mode = "COMMON_NONDOMINATED_PLUS_STOCK_COVERAGE_FALLBACK"

    # Best solution at each unique stock level under the common target.
    by_stock: Dict[int, Dict[str, object]] = {}
    for rec in selection_pool:
        sol = rec["common_solution"]
        stock_units = int(round(float(sol.get("total_stock", 0.0))))
        cur = by_stock.get(stock_units)
        key = (
            float(sol.get("tail_risk", np.inf)),
            -float(sol.get("worst_Ao", sol.get("Ao", 0.0))),
            float(sol.get("total_cost", np.inf)),
            float(sol.get("robustness_gap", np.inf)),
            float(rec["source_target_ao"]),
            int(rec["source_solution_id"]),
        )
        if cur is None:
            by_stock[stock_units] = rec
        else:
            csol = cur["common_solution"]
            ckey = (
                float(csol.get("tail_risk", np.inf)),
                -float(csol.get("worst_Ao", csol.get("Ao", 0.0))),
                float(csol.get("total_cost", np.inf)),
                float(csol.get("robustness_gap", np.inf)),
                float(cur["source_target_ao"]),
                int(cur["source_solution_id"]),
            )
            if key < ckey:
                by_stock[stock_units] = rec

    stock_levels = sorted(by_stock.keys())
    n_levels = len(stock_levels)
    n_target = min(desired, n_levels)

    configured_q = np.asarray(
        getattr(config, "representative_stock_quantiles", [0.0, 0.25, 0.5, 0.75, 1.0]),
        dtype=float,
    )
    if configured_q.size != desired:
        configured_q = np.linspace(0.0, 1.0, desired)

    if n_target == desired:
        use_q = configured_q
    elif n_target == 1:
        use_q = np.asarray([0.5])
    else:
        use_q = np.linspace(0.0, 1.0, n_target)

    raw_idx = np.rint(use_q * max(n_levels - 1, 0)).astype(int)
    chosen_idx: List[int] = []
    for idx in raw_idx.tolist():
        idx = int(np.clip(idx, 0, max(n_levels - 1, 0)))
        if idx not in chosen_idx:
            chosen_idx.append(idx)

    while len(chosen_idx) < n_target:
        unused = [i for i in range(n_levels) if i not in chosen_idx]
        if not unused:
            break
        best_i = max(
            unused,
            key=lambda i: (
                min(abs(i - j) for j in chosen_idx) if chosen_idx else n_levels,
                -i,
            ),
        )
        chosen_idx.append(int(best_i))

    chosen_idx = sorted(chosen_idx)[:n_target]
    strategy_names = _strategy_names_for_count(len(chosen_idx))

    selected_solutions = []
    rep_rows = []
    selected_stock_levels = set()

    for rank, (level_idx, strategy_name) in enumerate(zip(chosen_idx, strategy_names), start=1):
        stock_units = int(stock_levels[level_idx])
        rec = by_stock[stock_units]
        sol = rec["common_solution"]
        alt_id = f"C-REP-{rank:02d}"

        selected_solutions.append({
            "alternative_id": alt_id,
            "strategy_name": strategy_name,
            "source_target_ao": float(rec["source_target_ao"]),
            "source_solution_id": int(rec["source_solution_id"]),
            "stock_level_index": int(level_idx),
            "stock_level_percentile": float(level_idx / max(n_levels - 1, 1)),
            "solution": sol,
        })
        selected_stock_levels.add(stock_units)

        rep_rows.append({
            "Alternative_ID": alt_id,
            "Strategy": strategy_name,
            "Source_Target_Ao": float(rec["source_target_ao"]),
            "Source_Solution_ID": int(rec["source_solution_id"]),
            "Common_Evaluation_Target_Ao": float(representative_target),
            "Ao": float(sol.get("Ao", 0.0)),
            "Worst_Ao": float(sol.get("worst_Ao", sol.get("Ao", 0.0))),
            "Tail_Risk": float(sol.get("tail_risk", 0.0)),
            "Worst_Shortage": float(sol.get("worst_shortage", 0.0)),
            "Robustness_Gap": float(sol.get("robustness_gap", 0.0)),
            "Worst_K": float(sol.get("worst_k", 1.0)),
            "NSGA_Cost_KRW": float(sol.get("total_cost", 0.0)),
            "Managed_Items": int(round(float(sol.get("managed_parts", 0.0)))),
            "Stock_Units": stock_units,
            "Stock_Level_Index": int(level_idx),
            "Stock_Level_Percentile": float(level_idx / max(n_levels - 1, 1)),
            "Common_Target_Feasible": bool(rec["common_target_feasible"]),
            "Common_Nondominated": bool(rec["common_nondominated"]),
            "Selection_Pool_Mode": pool_mode,
            "Selection_Basis": "common_target_revaluation_then_unique_stock_spectrum",
        })

    representative_alternatives_df = pd.DataFrame(rep_rows)

    # Audit every stock-level candidate used by the selection pool.
    audit_rows = []
    for level_idx, stock_units in enumerate(stock_levels):
        rec = by_stock[stock_units]
        sol = rec["common_solution"]
        audit_rows.append({
            "Stock_Level_Index": int(level_idx),
            "Stock_Units": int(stock_units),
            "Source_Target_Ao": float(rec["source_target_ao"]),
            "Source_Solution_ID": int(rec["source_solution_id"]),
            "Selected": bool(stock_units in selected_stock_levels),
            "Common_Evaluation_Target_Ao": float(representative_target),
            "Ao": float(sol.get("Ao", 0.0)),
            "Worst_Ao": float(sol.get("worst_Ao", sol.get("Ao", 0.0))),
            "Tail_Risk": float(sol.get("tail_risk", 0.0)),
            "Worst_Shortage": float(sol.get("worst_shortage", 0.0)),
            "Robustness_Gap": float(sol.get("robustness_gap", 0.0)),
            "NSGA_Cost_KRW": float(sol.get("total_cost", 0.0)),
            "Common_Target_Feasible": bool(rec["common_target_feasible"]),
            "Common_Nondominated": bool(rec["common_nondominated"]),
            "Selection_Pool_Mode": pool_mode,
        })
    representative_selection_audit_df = pd.DataFrame(audit_rows)

    # Candidate-level inventory audit for each representative.
    inv_frames = []

    # Future Monte-Carlo bridge covers ALL spare-demand parts; non-stocked items = 0.
    prepared = ctx.get("prepared_df", pd.DataFrame()).copy()
    if isinstance(prepared, pd.DataFrame) and not prepared.empty and "is_spare_demand" in prepared.columns:
        spare_base = prepared.loc[prepared["is_spare_demand"].astype(bool)].copy()
    else:
        spare_base = pd.DataFrame()

    bridge_frames = []
    for meta in selected_solutions:
        sol = meta["solution"]
        alt_id = meta["alternative_id"]
        strategy_name = meta["strategy_name"]
        source_target = meta["source_target_ao"]
        source_id = meta["source_solution_id"]

        policy = _build_policy_df(ctx, sol).copy()
        policy.insert(0, "Strategy", strategy_name)
        policy.insert(0, "Alternative_ID", alt_id)
        policy["Source_Target_Ao"] = float(source_target)
        policy["Source_Solution_ID"] = int(source_id)
        policy["Stock_Qty"] = pd.to_numeric(policy["recommended_stock"], errors="coerce").fillna(0).astype(int)
        inv_frames.append(policy)

        if not spare_base.empty:
            b = spare_base.copy()
            pidx = policy.set_index("part_id")
            b["Stock_Qty"] = b["part_id"].map(pidx["Stock_Qty"]).fillna(0).astype(int)
            b["Stock_O"] = b["part_id"].map(pidx["stock_O"]).fillna(0).astype(int)
            b["Stock_I"] = b["part_id"].map(pidx["stock_I"]).fillna(0).astype(int)
            b["Stock_D"] = b["part_id"].map(pidx["stock_D"]).fillna(0).astype(int)

            # Stocking location is a property of the selected business supply structure,
            # not of whether the part happened to be a candidate in a particular solution.
            if "stocking_echelon" in b.columns:
                b["Stocking_Echelon"] = b["stocking_echelon"].astype(str)
            else:
                b["Stocking_Echelon"] = derive_stocking_echelon(
                    b["maint_echelon"].astype(str).to_numpy(),
                    supply_structure_mode,
                )
            b["Supply_Structure_Mode"] = supply_structure_mode

            input_hours = pd.to_numeric(
                b.get("annual_rad_hours", pd.Series(0.0, index=b.index)),
                errors="coerce",
            ).fillna(0.0)
            # NSGA and Monte Carlo must use the same operating-hour basis.
            # Input Annual_Rad_Hours is preserved for audit but does not silently override
            # NSGAConfig.annual_operating_hours.
            common_hours = max(float(getattr(config, "annual_operating_hours", 8760.0)), 1.0)
            b["MC_Annual_Operating_Hours"] = common_hours
            b["Input_Annual_Rad_Hours"] = input_hours

            raw_fr = pd.to_numeric(
                b.get("failure_rate", pd.Series(0.0, index=b.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
            op_h = b["MC_Annual_Operating_Hours"].to_numpy(dtype=float)
            unit = str(getattr(config, "failure_rate_unit", "per_million_hours")).strip().lower()
            if unit in ["per_million_hours", "per_million_hour", "pmh", "failures_per_million_hours", "per_1e6_hours"]:
                annual_lambda = raw_fr * op_h / 1_000_000.0
            elif unit in ["per_hour", "hourly", "lambda_h", "lambda_per_hour"]:
                annual_lambda = raw_fr * op_h
            elif unit in ["per_year", "annual", "yearly", "lambda_y", "lambda_per_year"]:
                annual_lambda = raw_fr
            else:
                annual_lambda = raw_fr

            qpa = pd.to_numeric(
                b.get("qpa", pd.Series(float(getattr(config, "default_qpa", 1.0)), index=b.index)),
                errors="coerce",
            ).fillna(float(getattr(config, "default_qpa", 1.0))).to_numpy(dtype=float)
            if bool(getattr(config, "apply_qpa_to_failure_rate", True)):
                annual_lambda = annual_lambda * qpa
            annual_lambda = annual_lambda * max(float(getattr(config, "fleet_size", 1.0)), 0.0)

            bridge_frames.append(pd.DataFrame({
                "Alternative_ID": alt_id,
                "Strategy": strategy_name,
                "Source_Target_Ao": float(source_target),
                "Source_Solution_ID": int(source_id),
                "Common_Evaluation_Target_Ao": float(representative_target),
                "Part_ID": b["part_id"].astype(str).to_numpy(),
                "Stock_Qty": b["Stock_Qty"].to_numpy(dtype=int),
                "Stock_O": b["Stock_O"].to_numpy(dtype=int),
                "Stock_I": b["Stock_I"].to_numpy(dtype=int),
                "Stock_D": b["Stock_D"].to_numpy(dtype=int),
                "Failure_Rate": raw_fr,
                "QPA": qpa,
                "Annual_Operating_Hours": op_h,
                "Input_Annual_Rad_Hours": b["Input_Annual_Rad_Hours"].to_numpy(dtype=float),
                "Lambda_Annual_MC_Base": annual_lambda,
                "Lead_Time_H": pd.to_numeric(b.get("lead_time", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "Repairable_Flag": pd.to_numeric(b.get("repairable_flag", 0), errors="coerce").fillna(0).astype(int).to_numpy(),
                "Replace_Time_H": pd.to_numeric(b.get("replace_time_h", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "Repair_TAT_H": pd.to_numeric(b.get("repair_tat_h", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "Unit_Price_KRW": pd.to_numeric(b.get("unit_cost", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "Maint_Echelon": b.get("maint_echelon", pd.Series("", index=b.index)).astype(str).to_numpy(),
                "Stocking_Echelon": b["Stocking_Echelon"].astype(str).to_numpy(),
                "Supply_Structure_Mode": b["Supply_Structure_Mode"].astype(str).to_numpy(),
                "Is_Candidate": b.get("candidate", pd.Series(False, index=b.index)).astype(bool).to_numpy(),
            }))

    representative_inventory_df = pd.concat(inv_frames, ignore_index=True) if inv_frames else pd.DataFrame()
    monte_carlo_bridge_df = pd.concat(bridge_frames, ignore_index=True) if bridge_frames else pd.DataFrame()

    return {
        "representative_alternatives_df": representative_alternatives_df,
        "representative_inventory_df": representative_inventory_df,
        "monte_carlo_bridge_df": monte_carlo_bridge_df,
        "representative_selection_audit_df": representative_selection_audit_df,
        "selected_solutions": selected_solutions,
    }


def run_nsga2(
    df_input: pd.DataFrame,
    config: Optional[NSGAConfig] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    start_time = time.time()
    if config is None:
        config = NSGAConfig()

    # Stage 1 validation is intentionally non-analytic.
    # It checks only future-interface sanity and does not alter current calculations.
    stage1_readiness = validate_stage1_config(config)
    if not stage1_readiness["stage1_ready"]:
        raise ValueError(
            "Stage 1 future-architecture configuration is invalid: "
            + "; ".join(stage1_readiness["issues"])
        )

    df_prepared_base = _prepare_engine_dataframe(df_input)
    ctx = _build_engine_context(df_prepared_base, config)

    result_map: Dict[float, Dict[str, object]] = {}
    all_runs: List[pd.DataFrame] = []
    histories: List[pd.DataFrame] = []

    rep_target = float(config.representative_target)
    target_grid = [float(t) for t in config.ao_target_grid]
    if rep_target not in target_grid:
        target_grid.append(rep_target)
        target_grid = sorted(set(target_grid))

    for idx, target in enumerate(target_grid):
        run_seed = int(config.random_seed + idx * 17)
        wrapped_callback = None
        if progress_callback is not None:
            stage_idx = int(idx + 1)
            stage_total = int(len(target_grid))
            current_target = float(target)
            is_rep = abs(current_target - rep_target) < 1e-12
            total_gen = int(config.n_generations)
            progress_callback(0, total_gen, {
                "event": "stage_start",
                "current_target_ao": current_target,
                "sweep_stage_idx": stage_idx,
                "sweep_stage_total": stage_total,
                "representative_target": rep_target,
                "is_representative_target": is_rep,
                "stage_progress": 0.0,
                "overall_progress": float((stage_idx - 1) / max(stage_total, 1)),
            })

            def wrapped_callback(gen, total_gen, best_summary, *, _stage_idx=stage_idx, _stage_total=stage_total, _current_target=current_target, _rep_target=rep_target, _is_rep=is_rep):
                payload = dict(best_summary)
                frac = float(gen / max(total_gen, 1))
                payload.update({
                    "event": "generation",
                    "current_target_ao": _current_target,
                    "sweep_stage_idx": _stage_idx,
                    "sweep_stage_total": _stage_total,
                    "representative_target": _rep_target,
                    "is_representative_target": _is_rep,
                    "stage_progress": frac,
                    "overall_progress": float(((_stage_idx - 1) + frac) / max(_stage_total, 1)),
                })
                progress_callback(gen, total_gen, payload)

        res_t = _run_nsga_single_target(ctx=ctx, config=config, ao_target=target, seed=run_seed, progress_callback=wrapped_callback)
        result_map[round(target, 6)] = res_t
        run_df = _collect_run_dataframe(ctx, res_t)
        all_runs.append(run_df)
        hist = res_t["history_df"].copy()
        histories.append(hist)

    rep_key = round(rep_target, 6)
    if rep_key not in result_map:
        rep_key = sorted(result_map.keys())[-1]
    rep_res = result_map[rep_key]
    best_sol = rep_res["best_solution"]

    all_target_runs_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    pareto_df = _collect_run_dataframe(ctx, rep_res)
    policy_df = _build_policy_df(ctx, best_sol)

    # STAGE 3: deterministic representative-solution bridge for future Monte Carlo.
    if bool(getattr(config, "representative_bridge_enabled", True)):
        _rep_bridge = _select_representative_solutions(
            ctx=ctx,
            result_map=result_map,
            representative_target=rep_target,
            config=config,
        )
    else:
        _rep_bridge = {
            "representative_alternatives_df": pd.DataFrame(),
            "representative_inventory_df": pd.DataFrame(),
            "monte_carlo_bridge_df": pd.DataFrame(),
            "representative_selection_audit_df": pd.DataFrame(),
            "selected_solutions": [],
        }
    representative_alternatives_df = _rep_bridge["representative_alternatives_df"]
    representative_inventory_df = _rep_bridge["representative_inventory_df"]
    monte_carlo_bridge_df = _rep_bridge["monte_carlo_bridge_df"]
    representative_selection_audit_df = _rep_bridge["representative_selection_audit_df"]

    sweep_rows = []
    for target, res in result_map.items():
        best = res["best_solution"]
        sweep_rows.append({
            "target_ao": float(target),
            "best_Ao": float(best["Ao"]),
            "best_cost": float(best["total_cost"]),
            "best_total_stock": float(best["total_stock"]),
            "best_managed_parts": float(best["managed_parts"]),
            "best_residual_shortage_exposure": float(best.get("residual_shortage_exposure", 0.0)),
            "best_residual_shortage_exposure_scaled": float(best.get("residual_shortage_exposure_scaled", 0.0)),
            "Tail_Risk": float(best.get("tail_risk", 0.0)),
            "Worst_Ao": float(best.get("worst_Ao", best.get("Ao", 0.0))),
            "Worst_Shortage": float(best.get("worst_shortage", 0.0)),
            "Robustness_Gap": float(best.get("robustness_gap", 0.0)),
            "Worst_K": float(best.get("worst_k", 1.0)),
            "n_pareto": int(len(res.get("pareto_solutions", []))),
        })
    sweep_summary_df = pd.DataFrame(sweep_rows).sort_values("target_ao").reset_index(drop=True)

    summary = {
        "target_ao": float(rep_target),
        "best_Ao": float(best_sol["Ao"]),
        "best_cost": float(best_sol["total_cost"]),
        "DT_total_h": float(best_sol["DT_total_h"]),
        "DT_diag_h": float(best_sol["DT_diag_h"]),
        "DT_wait_h": float(best_sol["DT_wait_h"]),
        "DT_restore_h": float(best_sol["DT_restore_h"]),
        "DT_total_annual_h": float(best_sol.get("DT_total_annual_h", 0.0)),
        "DT_diag_annual_h": float(best_sol.get("DT_diag_annual_h", 0.0)),
        "DT_wait_annual_h": float(best_sol.get("DT_wait_annual_h", 0.0)),
        "DT_restore_annual_h": float(best_sol.get("DT_restore_annual_h", 0.0)),
        "analysis_years": float(ctx.get("period_years", config.years)),
        "hours_per_year": float(ctx.get("hours_per_year", config.hours_per_year)),
        "T_obs_hours": float(ctx.get("t_obs_hours", config.years * config.hours_per_year)),
        "managed_parts": float(best_sol["managed_parts"]),
        "total_stock": float(best_sol["total_stock"]),
        "residual_shortage_exposure": float(best_sol.get("residual_shortage_exposure", 0.0)),
        "residual_shortage_exposure_scaled": float(best_sol.get("residual_shortage_exposure_scaled", 0.0)),
        "candidate_parts": int(ctx["m_cand"]),
        "elapsed_sec": float(time.time() - start_time),
        "engine_step": "MODEL_C_ROBUST_STAGE2_NS_GA_II",
        "stage1_ready": bool(stage1_readiness["stage1_ready"]),
        "analysis_model": str(config.analysis_model),
        "supply_structure_mode": str(ctx.get("supply_structure_mode", normalize_supply_structure_mode(config))),
        "future_robust_connected": bool(config.robust_enabled),
        "future_monte_carlo_connected": False,
        "monte_carlo_bridge_ready": bool(not monte_carlo_bridge_df.empty),
        "monte_carlo_execution_module": "simulation_engine.py",
        "decision_layer_module": "decision_engine.py",
        "decision_layer_feedback_to_nsga": False,
        "automatic_best_selection": False,
        "final_audit_module": "audit_engine.py",
        "final_audit_feedback_to_analysis": False,
        "Tail_Risk": float(best_sol.get("tail_risk", 0.0)),
        "Worst_Ao": float(best_sol.get("worst_Ao", best_sol.get("Ao", 0.0))),
        "Worst_Shortage": float(best_sol.get("worst_shortage", 0.0)),
        "Robustness_Gap": float(best_sol.get("robustness_gap", 0.0)),
        "Worst_K": float(best_sol.get("worst_k", 1.0)),
        "robust_candidate_added_count": int(ctx.get("robust_added_count", 0)),
        "representative_bridge_connected": bool(getattr(config, "representative_bridge_enabled", True)),
        "representative_alternative_count": int(len(representative_alternatives_df)),
        "representative_strategy_names": " | ".join(representative_alternatives_df["Strategy"].astype(str).tolist()) if not representative_alternatives_df.empty else "",
    }

    final_spares_df = _build_final_spares_df(ctx, best_sol)
    repairable_audit_df = _build_repairable_audit_df(ctx)
    diagnostic_df = _build_diagnostic_df(ctx)
    stage1_readiness_df = _build_stage1_readiness_df(ctx["prepared_df"], config, stage1_readiness)

    stage1_config_df = pd.DataFrame([
        {
            "analysis_model": config.analysis_model,
            "supply_structure_mode": normalize_supply_structure_mode(config),
            "legacy_d_only_pbl_stock": bool(config.d_only_pbl_stock),
            "robust_enabled": config.robust_enabled,
            "robust_candidate_expansion_enabled": config.robust_candidate_expansion_enabled,
            "robust_k_grid": ",".join(map(str, config.robust_k_grid)),
            "robust_tail_quantile": config.robust_tail_quantile,
            "nsga_crossover_prob": config.nsga_crossover_prob,
            "nsga_mutation_prob": config.nsga_mutation_prob,
            "monte_carlo_enabled": config.monte_carlo_enabled,
            "monte_carlo_iterations": config.monte_carlo_iterations,
            "monte_carlo_seed": config.monte_carlo_seed,
            "monte_carlo_k_distribution": config.monte_carlo_k_distribution,
            "monte_carlo_k_median": config.monte_carlo_k_median,
            "monte_carlo_k_p05": config.monte_carlo_k_p05,
            "monte_carlo_k_p95": config.monte_carlo_k_p95,
            "monte_carlo_period_months": ",".join(map(str, config.monte_carlo_period_months)),
            "monte_carlo_common_random_numbers": config.monte_carlo_common_random_numbers,
            "representative_bridge_enabled": config.representative_bridge_enabled,
            "representative_strategy_count": config.representative_strategy_count,
            "representative_stock_quantiles": ",".join(map(str, config.representative_stock_quantiles)),
            "stage1_only_no_objective_change": False,
        }
    ])

    # Auditable K-by-K detail for the representative best solution.
    _best_eval = _evaluate_model_c_population(
        ctx, config, rep_target,
        np.asarray(best_sol["Y"], dtype=int).reshape(1, -1),
        np.asarray(best_sol["SO"], dtype=int).reshape(1, -1),
        np.asarray(best_sol["SI"], dtype=int).reshape(1, -1),
        np.asarray(best_sol["SD"], dtype=int).reshape(1, -1),
    )
    _br = _best_eval["robust"]
    _kgrid = np.asarray(_br["k_grid"], dtype=float)
    _loss = np.asarray(_br["scenario_loss_matrix"][0], dtype=float)
    _tail_n = int(_br["tail_count"][0])
    _tail_order = np.argsort(_loss)[-max(_tail_n, 1):]
    _tail_flag = np.zeros(len(_kgrid), dtype=bool)
    _tail_flag[_tail_order] = True
    robust_scenario_detail_df = pd.DataFrame({
        "K": _kgrid,
        "Ao": np.asarray(_br["ao_matrix"][0], dtype=float),
        "Expected_Shortage_Units": np.asarray(_br["shortage_matrix"][0], dtype=float),
        "Residual_Shortage_Exposure_Scaled": np.asarray(_br["residual_scaled_matrix"][0], dtype=float),
        "Scenario_Loss": _loss,
        "Tail_Used": _tail_flag,
    })

    robust_scenario_summary_df = pd.DataFrame([
        {
            "analysis_model": str(config.analysis_model),
            "supply_structure_mode": normalize_supply_structure_mode(config),
            "robust_enabled": bool(config.robust_enabled),
            "K_Grid": ",".join(map(str, config.robust_k_grid)),
            "Tail_Quantile": float(config.robust_tail_quantile),
            "Tail_Risk": float(best_sol.get("tail_risk", 0.0)),
            "Worst_Ao": float(best_sol.get("worst_Ao", best_sol.get("Ao", 0.0))),
            "Worst_Shortage": float(best_sol.get("worst_shortage", 0.0)),
            "Robustness_Gap": float(best_sol.get("robustness_gap", 0.0)),
            "Worst_K": float(best_sol.get("worst_k", 1.0)),
            "Nominal_Ao": float(best_sol.get("Ao", 0.0)),
            "Robust_Candidate_Added": int(ctx.get("robust_added_count", 0)),
        }
    ])

    result = {
        "summary": summary,
        "prepared_df": ctx["prepared_df"],
        "candidate_df": ctx["candidate_df"],
        "history_df": history_df,
        "pareto_df": pareto_df,
        "policy_df": policy_df,
        "sweep_summary_df": sweep_summary_df,
        "all_target_runs_df": all_target_runs_df,
        "final_spares_df": final_spares_df,
        "repairable_audit_df": repairable_audit_df,
        "diagnostic_df": diagnostic_df,
        # Stage 1 additive audit outputs.
        "stage1_readiness": stage1_readiness,
        "stage1_readiness_df": stage1_readiness_df,
        "stage1_config_df": stage1_config_df,
        "robust_scenario_summary_df": robust_scenario_summary_df,
        "robust_scenario_detail_df": robust_scenario_detail_df,
        "representative_alternatives_df": representative_alternatives_df,
        "representative_inventory_df": representative_inventory_df,
        "monte_carlo_bridge_df": monte_carlo_bridge_df,
        "representative_selection_audit_df": representative_selection_audit_df,
        "logs": [
            "CHANGE 1 applied: NSGAConfig extended for latest Ground model settings.",
            "CHANGE 2 applied: QPA/Repairable/Replace/TAT columns preserved in prepared/candidate data.",
            "CHANGE 3 applied: _build_engine_context now uses annual lambda, QPA, Fleet Size, p_need, and repairable time split.",
            "CHANGE 4 applied: candidate selection uses p_need_period, long lead, Ao impact, and long-core rules.",
            "CHANGE 5 applied: F2 includes Poisson EBO based residual shortage exposure when enabled.",
            "CHANGE 6 applied: final_spares_df, repairable_audit_df, diagnostic_df are returned additively.",
            "STAGE 1 applied: future Model C/Monte-Carlo config interface and readiness audit added.",
            "STAGE 2 applied: MODEL C robust candidate expansion is active using Kmax p_need crossing.",
            "STAGE 2 applied: Tail Risk, Worst Ao, Worst Shortage and Robustness Gap are calculated across the K grid.",
            "STAGE 2 applied: explicit NSGA-II operators replace the former random-search + single-elite loop.",
            "STAGE 2 safety: Monte Carlo remains disabled and is not yet part of optimization.",
            "STAGE 3 applied: representative Model-C Pareto solutions are selected across unique stock levels.",
            "STAGE 3 applied: minimum/light/balanced/stable/maximum-stable strategy labels are display labels, not optimality scores.",
            "STAGE 3 applied: monte_carlo_bridge_df contains all spare-demand parts with representative initial stock quantities.",
            "STAGE 3 safety: Monte Carlo is still not executed inside nsga_engine.py; Stage 3 creates a deterministic simulation bridge.",
            "STAGE 4 interface ready: simulation_engine.py consumes monte_carlo_bridge_df after NSGA execution.",
            "STAGE 4 separation principle: Monte Carlo validates representative alternatives but does not tune NSGA objectives.",
            "STAGE 5 interface ready: decision_engine.py translates MC results into transparent cost/risk decision indicators.",
            "STAGE 5 safety: decision grades never feed back into NSGA and no automatic best alternative is selected.",
            "STAGE 6 interface ready: audit_engine.py performs read-only cross-stage consistency and regression checks.",
            "STAGE 6 safety: audit results never modify NSGA, Monte Carlo, representative strategies, or decision grades.",
        ],
    }
    return result


__all__ = [
    "NSGAConfig",
    "prepare_input_dataframe",
    "run_nsga2",
    "validate_stage1_config",
]
