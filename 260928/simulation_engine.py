# -*- coding: utf-8 -*-
"""
simulation_engine.py
====================

FINAL DEBUGGED · Cold-start Monte Carlo engine

Canonical logic restored
------------------------
1. K is generated PER PART × PERIOD × ITERATION.
2. The same part-level K and same demand event times are shared across all
   representative strategies (Common Random Numbers).
3. Demand count:
      D_i ~ Poisson(Lambda_Annual_MC_Base_i × K_i × years)
4. Event times are uniform over the analysis horizon conditional on demand count.
5. If stock is available:
      immediate fulfillment, stock -1, replenishment arrival scheduled after
      Repair_TAT_H (repairable) or Lead_Time_H (non-repairable).
6. If stock is unavailable:
      shortage event; delay equals the engineering replenishment delay.
      The backorder is satisfied by that delivery itself, so no extra inventory
      increment is created.
7. Three service aggregation methods are returned:
      - 월 동일 가중
      - 건수 가중
      - 가중평균 (50% 월 동일 가중 + 50% 건수 가중)
8. Decision Layer should use "건수 가중" as the primary physical comparison.

No historical demand is used to tune the alternatives.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


Z_95 = 1.6448536269514722
DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24.0
WEIGHTED_BLEND_ALPHA = 0.50


@dataclass
class MonteCarloConfig:
    iterations: int = 3000
    random_seed: int = 42
    period_months: List[int] = field(default_factory=lambda: [12, 24, 36])

    k_distribution: str = "lognormal"
    k_median: float = 1.0
    k_p05: float = 0.5
    k_p95: float = 2.0

    common_random_numbers: bool = True
    min_cycle_time_h: float = 1.0
    store_iteration_detail: bool = True
    weighted_blend_alpha: float = WEIGHTED_BLEND_ALPHA


def validate_monte_carlo_config(config: MonteCarloConfig) -> Dict[str, object]:
    issues: List[str] = []
    if int(config.iterations) <= 0:
        issues.append("iterations must be > 0.")
    periods = [int(x) for x in config.period_months]
    if not periods or any(x <= 0 for x in periods):
        issues.append("period_months must contain positive month values.")
    if str(config.k_distribution).strip().lower() != "lognormal":
        issues.append("Only lognormal K is supported.")
    if not (0 < float(config.k_p05) <= float(config.k_median) <= float(config.k_p95)):
        issues.append("K anchors must satisfy 0 < P05 <= median <= P95.")
    if float(config.min_cycle_time_h) <= 0:
        issues.append("min_cycle_time_h must be > 0.")
    if not (0.0 <= float(config.weighted_blend_alpha) <= 1.0):
        issues.append("weighted_blend_alpha must be between 0 and 1.")
    return {"ready": len(issues) == 0, "issues": issues}


def _lognormal_parameters(config: MonteCarloConfig) -> tuple[float, float]:
    med = max(float(config.k_median), 1e-12)
    p05 = max(float(config.k_p05), 1e-12)
    p95 = max(float(config.k_p95), p05)
    mu = math.log(med)
    sigma_low = abs(math.log(med / p05)) / Z_95 if p05 < med else 0.0
    sigma_high = abs(math.log(p95 / med)) / Z_95 if p95 > med else 0.0
    vals = [x for x in (sigma_low, sigma_high) if x > 0]
    sigma = float(np.mean(vals)) if vals else 1e-9
    return mu, max(sigma, 1e-9)


def _validate_bridge_df(df: pd.DataFrame) -> None:
    required = [
        "Alternative_ID", "Strategy", "Part_ID", "Stock_Qty",
        "Lambda_Annual_MC_Base", "Lead_Time_H",
        "Repairable_Flag", "Repair_TAT_H",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"monte_carlo_bridge_df missing required columns: {missing}")
    if df.empty:
        raise ValueError("monte_carlo_bridge_df is empty.")


def _prepare_bridge(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    _validate_bridge_df(df)
    out = df.copy()
    out["Alternative_ID"] = out["Alternative_ID"].astype(str)
    out["Strategy"] = out["Strategy"].astype(str)
    out["Part_ID"] = out["Part_ID"].astype(str)

    for c, default in {
        "Stock_Qty": 0,
        "Lambda_Annual_MC_Base": 0.0,
        "Lead_Time_H": 0.0,
        "Repairable_Flag": 0,
        "Repair_TAT_H": 0.0,
    }.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)

    out["Stock_Qty"] = out["Stock_Qty"].clip(lower=0).astype(int)
    out["Lambda_Annual_MC_Base"] = out["Lambda_Annual_MC_Base"].clip(lower=0.0)
    out["Lead_Time_H"] = out["Lead_Time_H"].clip(lower=0.0)
    out["Repairable_Flag"] = out["Repairable_Flag"].clip(lower=0, upper=1).astype(int)
    out["Repair_TAT_H"] = out["Repair_TAT_H"].clip(lower=0.0)

    # Physical parameters must be invariant across representative alternatives.
    for pid, sub in out.groupby("Part_ID", sort=False):
        for col in ["Lambda_Annual_MC_Base", "Lead_Time_H", "Repairable_Flag", "Repair_TAT_H"]:
            if pd.to_numeric(sub[col], errors="coerce").fillna(0.0).nunique(dropna=False) > 1:
                raise ValueError(
                    f"Bridge physical parameter differs across alternatives: Part_ID={pid}, column={col}"
                )

    return (
        out,
        out["Alternative_ID"].drop_duplicates().tolist(),
        sorted(out["Part_ID"].drop_duplicates().tolist()),
    )


def _simulate_part(
    event_times_days: np.ndarray,
    initial_stock: int,
    replenishment_delay_days: float,
) -> tuple[int, int, float, list[float]]:
    """
    Canonical Stage-4 part event logic.

    Returns:
      immediate_count, shortage_count, total_shortage_delay_days, delays_per_demand
    """
    on_hand = max(int(initial_stock), 0)
    delay = max(float(replenishment_delay_days), 0.0)
    arrivals: List[float] = []

    immediate = 0
    shortage = 0
    shortage_delay = 0.0
    delays: List[float] = []

    for t in event_times_days:
        t = float(t)
        while arrivals and arrivals[0] <= t:
            heapq.heappop(arrivals)
            on_hand += 1

        if on_hand > 0:
            on_hand -= 1
            immediate += 1
            delays.append(0.0)
            heapq.heappush(arrivals, t + delay)
        else:
            shortage += 1
            shortage_delay += delay
            delays.append(delay)
            # Backorder is satisfied by its own delivery; no inventory increment.

    return immediate, shortage, shortage_delay, delays


def _pct(values, q):
    a = np.asarray(values, dtype=float)
    return float(np.quantile(a, q)) if a.size else 0.0


def run_coldstart_monte_carlo(
    bridge_df: pd.DataFrame,
    config: Optional[MonteCarloConfig] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    if config is None:
        config = MonteCarloConfig()

    check = validate_monte_carlo_config(config)
    if not check["ready"]:
        raise ValueError("Invalid MonteCarloConfig: " + "; ".join(check["issues"]))

    bridge, alt_ids, part_ids = _prepare_bridge(bridge_df)
    periods = [int(x) for x in config.period_months]
    n_iter = int(config.iterations)

    alt_meta = (
        bridge[["Alternative_ID", "Strategy"]]
        .drop_duplicates("Alternative_ID")
        .set_index("Alternative_ID")
    )

    physical = (
        bridge.sort_values(["Part_ID", "Alternative_ID"])
        .drop_duplicates("Part_ID")
        .set_index("Part_ID")
        .reindex(part_ids)
    )
    lam_y = pd.to_numeric(physical["Lambda_Annual_MC_Base"], errors="coerce").fillna(0.0).to_numpy(float)
    lead_h = pd.to_numeric(physical["Lead_Time_H"], errors="coerce").fillna(0.0).to_numpy(float)
    repairable = pd.to_numeric(physical["Repairable_Flag"], errors="coerce").fillna(0).astype(int).to_numpy()
    tat_h = pd.to_numeric(physical["Repair_TAT_H"], errors="coerce").fillna(0.0).to_numpy(float)

    cycle_h = np.where((repairable > 0) & (tat_h > 0), tat_h, lead_h)
    cycle_h = np.maximum(cycle_h, float(config.min_cycle_time_h))
    cycle_days = cycle_h / HOURS_PER_DAY

    part_pos = {p: i for i, p in enumerate(part_ids)}
    alt_pos = {a: i for i, a in enumerate(alt_ids)}
    stock = np.zeros((len(alt_ids), len(part_ids)), dtype=int)
    for _, r in bridge[["Alternative_ID", "Part_ID", "Stock_Qty"]].iterrows():
        stock[alt_pos[str(r["Alternative_ID"])], part_pos[str(r["Part_ID"])]] = int(r["Stock_Qty"])

    mu_k, sigma_k = _lognormal_parameters(config)

    iteration_rows: List[Dict[str, object]] = []
    part_agg: Dict[tuple[int, int], Dict[str, np.ndarray]] = {}
    total_work = len(periods) * n_iter
    done = 0

    for months in periods:
        years = float(months) / 12.0
        horizon_days = years * DAYS_PER_YEAR
        month_days = horizon_days / float(months)

        for ai in range(len(alt_ids)):
            key = (months, ai)
            part_agg[key] = {
                "demand": np.zeros(len(part_ids), float),
                "shortage": np.zeros(len(part_ids), float),
                "immediate": np.zeros(len(part_ids), float),
                "delay": np.zeros(len(part_ids), float),
            }

        # Period-specific RNG. Part-level draws are generated once and shared across alternatives.
        rng = np.random.default_rng(int(config.random_seed + months * 100003))

        for it in range(1, n_iter + 1):
            k_by_part = rng.lognormal(mean=mu_k, sigma=sigma_k, size=len(part_ids))
            events_by_part: List[np.ndarray] = []
            demand_counts = np.zeros(len(part_ids), dtype=int)

            for pi, base_lam in enumerate(lam_y):
                mu = max(float(base_lam) * float(k_by_part[pi]) * years, 0.0)
                n = int(rng.poisson(mu))
                demand_counts[pi] = n
                if n > 0:
                    ev = np.sort(rng.uniform(0.0, horizon_days, size=n)).astype(float)
                else:
                    ev = np.empty(0, float)
                events_by_part.append(ev)

            total_demand = int(demand_counts.sum())

            for ai, aid in enumerate(alt_ids):
                immediate_total = 0
                shortage_total = 0
                shortage_delay_total = 0.0
                all_delays: List[float] = []
                monthly_demand = np.zeros(months, dtype=int)
                monthly_immediate = np.zeros(months, dtype=int)

                for pi, events in enumerate(events_by_part):
                    if demand_counts[pi] <= 0:
                        continue

                    imm, sh, sh_delay, delays = _simulate_part(
                        events,
                        int(stock[ai, pi]),
                        float(cycle_days[pi]),
                    )
                    immediate_total += imm
                    shortage_total += sh
                    shortage_delay_total += sh_delay
                    all_delays.extend(delays)

                    for t, dly in zip(events, delays):
                        mi = min(months - 1, int(float(t) / max(month_days, 1e-12)))
                        monthly_demand[mi] += 1
                        if dly <= 1e-12:
                            monthly_immediate[mi] += 1

                    key = (months, ai)
                    part_agg[key]["demand"][pi] += int(demand_counts[pi])
                    part_agg[key]["shortage"][pi] += sh
                    part_agg[key]["immediate"][pi] += imm
                    part_agg[key]["delay"][pi] += sh_delay

                case_weighted = 1.0 if total_demand <= 0 else float(immediate_total / total_demand)
                active = monthly_demand > 0
                if active.any():
                    monthly_equal = float(np.mean(monthly_immediate[active] / monthly_demand[active]))
                else:
                    monthly_equal = 1.0
                blend = (
                    float(config.weighted_blend_alpha) * monthly_equal
                    + (1.0 - float(config.weighted_blend_alpha)) * case_weighted
                )

                mean_event_delay = float(np.mean(all_delays)) if all_delays else 0.0
                p95_event_delay = float(np.percentile(all_delays, 95)) if all_delays else 0.0

                if config.store_iteration_detail:
                    iteration_rows.append({
                        "Alternative_ID": aid,
                        "Strategy": str(alt_meta.loc[aid, "Strategy"]),
                        "Period_Months": months,
                        "Iteration": it,
                        "Mean_K": float(np.mean(k_by_part)) if len(k_by_part) else 1.0,
                        "P95_K": float(np.percentile(k_by_part, 95)) if len(k_by_part) else 1.0,
                        "Demand_Total": total_demand,
                        "Immediate_Fulfilled": int(immediate_total),
                        "Shortage_Count": int(shortage_total),
                        "Case_Weighted_Service": case_weighted,
                        "Monthly_Equal_Service": monthly_equal,
                        "Bridge_Weighted_Service": blend,
                        "Zero_Shortage": int(shortage_total == 0),
                        "Mean_Event_Delay_Days": mean_event_delay,
                        "P95_Event_Delay_Days": p95_event_delay,
                        "Total_Shortage_Delay_Days": float(shortage_delay_total),
                    })

            done += 1
            if progress_callback is not None and (
                it == 1 or it == n_iter or it % max(1, n_iter // 30) == 0
            ):
                progress_callback({
                    "Period_Months": months,
                    "Iteration": it,
                    "Iterations": n_iter,
                    "Progress": float(done / max(total_work, 1)),
                })

    iteration_df = pd.DataFrame(iteration_rows)

    method_map = {
        "월 동일 가중": "Monthly_Equal_Service",
        "건수 가중": "Case_Weighted_Service",
        "가중평균": "Bridge_Weighted_Service",
    }
    summary_rows: List[Dict[str, object]] = []
    for (aid, strategy, months), sub in iteration_df.groupby(
        ["Alternative_ID", "Strategy", "Period_Months"], sort=False
    ):
        for method, score_col in method_map.items():
            scores = pd.to_numeric(sub[score_col], errors="coerce").fillna(1.0).to_numpy(float)
            demand = pd.to_numeric(sub["Demand_Total"], errors="coerce").fillna(0.0).to_numpy(float)
            shortage = pd.to_numeric(sub["Shortage_Count"], errors="coerce").fillna(0.0).to_numpy(float)
            zero = pd.to_numeric(sub["Zero_Shortage"], errors="coerce").fillna(0.0).to_numpy(float)
            mean_delay = pd.to_numeric(sub["Mean_Event_Delay_Days"], errors="coerce").fillna(0.0).to_numpy(float)
            p95_delay = pd.to_numeric(sub["P95_Event_Delay_Days"], errors="coerce").fillna(0.0).to_numpy(float)

            summary_rows.append({
                "Alternative_ID": aid,
                "Strategy": strategy,
                "Period_Months": int(months),
                "Evaluation_Method": method,
                "Iterations": int(len(sub)),
                "Service_Mean": float(np.mean(scores)),
                "Service_P05": _pct(scores, 0.05),
                "Service_P50": _pct(scores, 0.50),
                "Service_P95": _pct(scores, 0.95),
                "Probability_Zero_Shortage": float(np.mean(zero)),
                "Demand_Mean": float(np.mean(demand)),
                "Demand_P05": _pct(demand, 0.05),
                "Demand_P50": _pct(demand, 0.50),
                "Demand_P95": _pct(demand, 0.95),
                "Shortage_Mean": float(np.mean(shortage)),
                "Shortage_P95": _pct(shortage, 0.95),
                "Mean_Event_Delay_Days": float(np.mean(mean_delay)),
                "P95_Event_Delay_Days": _pct(p95_delay, 0.95),
                "Total_Shortage_Delay_Mean": float(
                    pd.to_numeric(sub["Total_Shortage_Delay_Days"], errors="coerce").fillna(0.0).mean()
                ),
            })

    summary_df = pd.DataFrame(summary_rows)

    part_rows: List[Dict[str, object]] = []
    for months in periods:
        for ai, aid in enumerate(alt_ids):
            a = part_agg[(months, ai)]
            for pi, pid in enumerate(part_ids):
                d = a["demand"][pi]
                sh = a["shortage"][pi]
                imm = a["immediate"][pi]
                part_rows.append({
                    "Alternative_ID": aid,
                    "Strategy": str(alt_meta.loc[aid, "Strategy"]),
                    "Period_Months": months,
                    "Part_ID": pid,
                    "Initial_Stock": int(stock[ai, pi]),
                    "Lambda_Annual_MC_Base": float(lam_y[pi]),
                    "Replenishment_Cycle_H": float(cycle_h[pi]),
                    "Repairable_Flag": int(repairable[pi]),
                    "Demand_Mean": float(d / n_iter),
                    "Shortage_Mean": float(sh / n_iter),
                    "Service_Rate_Aggregated": 1.0 if d <= 0 else float(imm / d),
                    "Mean_Shortage_Delay_Days": 0.0 if sh <= 0 else float(a["delay"][pi] / sh),
                })
    part_df = pd.DataFrame(part_rows)

    config_df = pd.DataFrame([{
        "Iterations": n_iter,
        "Random_Seed": int(config.random_seed),
        "Period_Months": ",".join(map(str, periods)),
        "K_Distribution": config.k_distribution,
        "K_Granularity": "PART x PERIOD x ITERATION",
        "K_Median_Input": float(config.k_median),
        "K_P05_Input": float(config.k_p05),
        "K_P95_Input": float(config.k_p95),
        "Common_Random_Numbers": bool(config.common_random_numbers),
        "CRN_Shared_Across": "Representative alternatives for same part/period/iteration",
        "Evaluation_Methods": "월 동일 가중 | 건수 가중 | 가중평균",
        "Decision_Primary_Method": "건수 가중",
        "Weighted_Blend_Alpha": float(config.weighted_blend_alpha),
        "Replenishment_Rule": "Repairable=>Repair_TAT_H; NonRepairable=>Lead_Time_H",
        "Historical_Data_Used": False,
    }])

    return {
        "mc_summary_df": summary_df,
        "mc_iteration_df": iteration_df,
        "mc_part_summary_df": part_df,
        "mc_config_df": config_df,
        "mc_validation": {
            "ready": True,
            "alternative_count": len(alt_ids),
            "part_count": len(part_ids),
            "period_count": len(periods),
            "method_count": len(method_map),
            "iterations": n_iter,
            "common_random_numbers": bool(config.common_random_numbers),
        },
    }


__all__ = ["MonteCarloConfig", "validate_monte_carlo_config", "run_coldstart_monte_carlo"]
