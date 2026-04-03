
# nsga_engine.py
# -*- coding: utf-8 -*-

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

YEARS = 2.0
T_OBS_HOURS = YEARS * 365.0 * 24.0
HOURS_PER_YEAR = 365.0 * 24.0

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

# Baseline failure-rate policy
# This version intentionally uses the raw failure_rate values without any
# aging-index-based amplification. The PM reduction terms below remain intact
# because they are part of the existing model logic, not the removed aging correction.


@dataclass
class NSGAConfig:
    population_size: int = 60
    n_generations: int = 40
    random_seed: int = DEFAULT_RANDOM_SEED
    target_ao: float = DEFAULT_REPRESENTATIVE_TARGET

    # existing UI knobs
    pmin: float = 0.05
    long_lead_percentile: float = 0.80
    ao_impact_percentile: float = 0.75

    # internal sweep / standardization
    ao_target_grid: List[float] = field(default_factory=lambda: list(DEFAULT_TARGET_GRID))
    representative_target: float = DEFAULT_REPRESENTATIVE_TARGET
    long_core_impact_percentile: float = 0.70
    item_count_weight: float = ITEM_COUNT_WEIGHT
    stock_unit_weight: float = STOCK_UNIT_WEIGHT
    ao_cost_penalty: float = AO_COST_PENALTY
    years: float = YEARS


def find_best_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


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


def long_dynamic_cap_from_lambda(lam_2y: float) -> int:
    if lam_2y <= 0.30:
        return ABS_CAP_SMALL
    if lam_2y <= 0.80:
        return ABS_CAP_MED
    return ABS_CAP_LARGE


def poisson_tail_prob(mu: float, s: int) -> float:
    mu = max(float(mu), 0.0)
    s = max(int(s), 0)
    if mu <= 1e-12:
        return 0.0

    # P(N > s) = 1 - sum_{k=0}^s exp(-mu) mu^k / k!
    term = math.exp(-mu)
    cdf = term
    for k in range(1, s + 1):
        term *= mu / k
        cdf += term
    tail = 1.0 - cdf
    return float(min(max(tail, 0.0), 1.0))


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


def _prepare_engine_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    col_part = find_best_column(df, ["Part_ID", "part_id", "part", "item", "item_id", "niin", "nsn", "품번", "부품명"])
    col_parent = find_best_column(df, ["Parent_ID", "parent_id", "parent", "상위품번", "상위부품"])
    col_level = find_best_column(df, ["Level", "level", "레벨"])
    col_echelon = find_best_column(df, ["Maint_Echelon", "maint_echelon", "echelon", "정비단계", "정비계층"])
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
    col_ann_rad = find_best_column(df, ["Annual_Rad_Hours", "annual_rad_hours"])
    col_env_res = find_best_column(df, ["Env_Resistance", "env_resistance", "환경저항성", "env_res"])

    if col_part is None:
        df["part_id"] = [f"PART_{i+1:05d}" for i in range(len(df))]
        col_part = "part_id"

    out = pd.DataFrame(index=df.index)
    out["part_id"] = df[col_part].astype(str)
    out["parent_id"] = df[col_parent].astype(str) if col_parent is not None else ""
    out["level_raw"] = df[col_level] if col_level is not None else 0
    out["level_num"] = out["level_raw"].apply(parse_level_to_int)
    out["maint_echelon_raw"] = df[col_echelon] if col_echelon is not None else "O"
    out["maint_echelon"] = out["maint_echelon_raw"].apply(map_echelon)

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

    out["failure_rate"] = out["failure_rate"].clip(lower=0.0)
    out["unit_cost"] = out["unit_cost"].clip(lower=0.0)
    out["lead_time"] = out["lead_time"].clip(lower=1.0)
    out["transport_time"] = out["transport_time"].clip(lower=0.1)
    out["cm_cost"] = out["cm_cost"].clip(lower=0.0)
    out["cm_time"] = out["cm_time"].clip(lower=0.1)
    out["condemn_pct"] = out["condemn_pct"].clip(lower=0.0, upper=100.0)
    out["pm_cycle"] = out["pm_cycle"].replace(0, 365).clip(lower=1.0)
    out["env_resistance"] = out["env_resistance"].clip(lower=1.0, upper=5.0)

    # display/compatibility columns for existing app preview
    out["criticality"] = np.where(
        minmax(out["failure_rate"].to_numpy()) >= 0.80, 1.00,
        np.where(minmax(out["failure_rate"].to_numpy()) >= 0.50, 0.45, 0.15)
    )
    out["stock"] = 0
    out["holding_cost"] = H_PRE * out["unit_cost"]
    out["base_score"] = (
        out["failure_rate"] * 0.45
        + (out["lead_time"] / max(float(out["lead_time"].max()), 1.0)) * 0.20
        + (out["criticality"] / max(float(out["criticality"].max()), 1.0)) * 0.35
    )

    return out.reset_index(drop=True)


def _build_engine_context(df_prepared: pd.DataFrame, config: NSGAConfig) -> Dict[str, object]:
    df = df_prepared.copy()
    n_all = len(df)

    part_id_all = df["part_id"].astype(str).to_numpy()
    parent_id_all = df["parent_id"].astype(str).replace("nan", "").replace("None", "").to_numpy()
    level_num_all = df["level_num"].to_numpy(dtype=int)
    echelon_cat_all = df["maint_echelon"].to_numpy(dtype=object)

    FR_all = df["failure_rate"].to_numpy(dtype=float)
    unit_price_all = df["unit_cost"].to_numpy(dtype=float)
    transport_cost_all = df["transport_cost"].to_numpy(dtype=float)
    lead_all = df["lead_time"].to_numpy(dtype=float)
    transport_time_all = df["transport_time"].to_numpy(dtype=float)
    cm_cost_all = df["cm_cost"].to_numpy(dtype=float)
    cm_time_all = df["cm_time"].to_numpy(dtype=float)
    condemn_all = (df["condemn_pct"].to_numpy(dtype=float) / 100.0).clip(0.0, 1.0)
    pm_cycle_all = df["pm_cycle"].to_numpy(dtype=float)
    rad_idx_all = df["rad_deg_index"].to_numpy(dtype=float)
    annual_rad_hours_all = df["annual_rad_hours"].to_numpy(dtype=float)
    env_res_all = df["env_resistance"].to_numpy(dtype=float)

    pm_norm_all = 1.0 / np.sqrt(np.maximum(pm_cycle_all, 1.0) / 365.0)
    pm_norm_all = np.clip(pm_norm_all, 0.0, 1.0)

    # Keep legacy columns in the prepared dataframe for compatibility with the
    # existing app, but do not amplify failure rate with any aging correction.
    aging_multiplier_all = np.ones_like(FR_all, dtype=float)

    gamma_all = PM_EFFECT_ON_FAILURE * pm_norm_all
    delta_all = PM_EFFECT_ON_RESTORE * pm_norm_all

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
    spare_demand_mask = ((level_num_all >= 3) & is_leaf_all & (FR_all > 0) & has_physical_attr)
    if int(spare_demand_mask.sum()) < 20:
        spare_demand_mask = ((level_num_all >= 3) & (FR_all > 0) & has_physical_attr)

    spare_idx = np.where(spare_demand_mask)[0]
    if len(spare_idx) == 0:
        raise ValueError("No real spare-demand parts found from uploaded data.")

    annual_fr_sp = FR_all[spare_idx] * (1.0 - gamma_all[spare_idx])
    lam_2y_sp = annual_fr_sp * config.years

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
    part_id_sp = part_id_all[spare_idx]
    parent_id_sp = parent_id_all[spare_idx]
    chains_sp = [all_ancestor_chains[i] for i in spare_idx]

    parent_group_map: Dict[str, List[int]] = {}
    for local_idx, global_idx in enumerate(spare_idx):
        p = parent_id_all[global_idx]
        p = str(p) if p not in ["", "nan", "None", "null"] else f"ROOT_{part_id_all[global_idx]}"
        parent_group_map.setdefault(p, []).append(local_idx)

    lambda_eff_sp = annual_fr_sp.copy()
    for _, members in parent_group_map.items():
        if len(members) <= 1:
            continue
        fr_sub = annual_fr_sp[members]
        total = fr_sub.sum()
        if total > 1e-12:
            shares = fr_sub / total
            lambda_eff_sp[members] = total * shares / len(members) * 1.2
        else:
            lambda_eff_sp[members] = annual_fr_sp[members]

    T_diag_sp = np.zeros(len(spare_idx), dtype=float)
    for k, chain in enumerate(chains_sp):
        t = 0.0
        for node_idx in chain:
            t += node_diag_contrib(int(level_num_all[node_idx]))
        T_diag_sp[k] = t

    T_reminst_sp = 0.35 * np.maximum(cm_time_sp, 1.0)
    T_repair_base_sp = 0.65 * cm_time_sp
    T_repair_sp = T_repair_base_sp * (1.0 - delta_sp)

    for i, e in enumerate(echelon_sp):
        if e == "O":
            T_reminst_sp[i] *= 0.85
            T_repair_sp[i] *= 0.75
        elif e == "I":
            T_reminst_sp[i] *= 1.00
            T_repair_sp[i] *= 1.00
        elif e == "D":
            T_reminst_sp[i] *= 1.15
            T_repair_sp[i] *= 1.25

    T_restore_intrinsic_sp = T_reminst_sp + (1.0 - condemn_sp) * T_repair_sp

    T_issue_O_sp = np.full(len(spare_idx), 0.3, dtype=float)
    T_issue_I_sp = np.maximum(0.8, np.minimum(4.0, 0.25 * np.maximum(transport_sp, 1.0)))
    T_issue_D_sp = np.maximum(1.5, np.minimum(8.0, 0.45 * np.maximum(transport_sp, 1.0)))

    baseline_wait_sp = np.zeros(len(spare_idx), dtype=float)
    for i, e in enumerate(echelon_sp):
        frac = BASELINE_WAIT_FRACTION_BY_ECHELON.get(e, 0.35)
        floor = MIN_WAIT_FLOOR_H_BY_ECHELON.get(e, 2.0)
        baseline_wait_sp[i] = max(floor, frac * lead_sp[i])

    p_need_sp = 1.0 - np.exp(-lam_2y_sp)
    ll_pct = float(config.long_lead_percentile * 100.0 if config.long_lead_percentile <= 1.0 else config.long_lead_percentile)
    ai_pct = float(config.ao_impact_percentile * 100.0 if config.ao_impact_percentile <= 1.0 else config.ao_impact_percentile)
    long_core_pct = float(config.long_core_impact_percentile * 100.0 if config.long_core_impact_percentile <= 1.0 else config.long_core_impact_percentile)

    thr_lead = float(np.nanpercentile(lead_sp, ll_pct))
    is_long_sp = (lead_sp >= thr_lead)

    impact_sp = lambda_eff_sp * np.maximum(T_diag_sp + baseline_wait_sp + T_restore_intrinsic_sp, 1e-9) / HOURS_PER_YEAR
    thr_impact = float(np.nanpercentile(impact_sp, ai_pct))
    is_high_impact_sp = (impact_sp >= thr_impact)

    candidate_mask = (p_need_sp >= config.pmin) | is_high_impact_sp | is_long_sp
    candidate_idx_local = np.where(candidate_mask)[0]
    if len(candidate_idx_local) == 0:
        candidate_idx_local = np.arange(len(spare_idx), dtype=int)

    # candidate-local arrays
    part_id_c = part_id_sp[candidate_idx_local]
    parent_id_c = parent_id_sp[candidate_idx_local]
    level_c = level_sp[candidate_idx_local]
    depth_c = depth_sp[candidate_idx_local]
    echelon_c = echelon_sp[candidate_idx_local]

    annual_fr_c = annual_fr_sp[candidate_idx_local]
    lambda_eff_c = lambda_eff_sp[candidate_idx_local]
    lam_2y_c = lam_2y_sp[candidate_idx_local]
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

    p_need_c = p_need_sp[candidate_idx_local]
    impact_c = impact_sp[candidate_idx_local]
    is_long_c = is_long_sp[candidate_idx_local]

    m_cand = len(candidate_idx_local)
    long_idx_c = np.where(is_long_c)[0]
    long_core_thr = float(np.nanpercentile(impact_c[long_idx_c], long_core_pct)) if len(long_idx_c) > 0 else 0.0

    prebuy_flag_c = np.zeros(m_cand, dtype=bool)
    protection_flag_c = np.zeros(m_cand, dtype=bool)
    for j in range(m_cand):
        if is_long_c[j] and ((p_need_c[j] >= config.pmin) or (impact_c[j] >= thr_impact)):
            prebuy_flag_c[j] = True
        if is_long_c[j] and (impact_c[j] >= long_core_thr):
            protection_flag_c[j] = True

    impact_norm_c = minmax(impact_c)
    critical_weight_c = np.where(impact_norm_c >= 0.80, 1.00, np.where(impact_norm_c >= 0.50, 0.45, 0.15))

    mu_pipe_nom_c = lambda_eff_c * (baseline_wait_c + T_restore_intrinsic_c) / HOURS_PER_YEAR
    stock_cap_c = np.zeros(m_cand, dtype=int)
    for i in range(m_cand):
        if is_long_c[i]:
            stock_cap_c[i] = long_dynamic_cap_from_lambda(lam_2y_c[i])
        else:
            stock_cap_c[i] = max(1, int(np.ceil(mu_pipe_nom_c[i] + 4.0 * np.sqrt(max(mu_pipe_nom_c[i], 1e-9)) + 2.0)))

    prot_floor_c = np.where(protection_flag_c, 1, 0).astype(int)
    base_maint_spend = float(np.sum(annual_fr_c * cm_cost_c * config.years))

    prepared_out = df_prepared.copy()
    prepared_out["is_spare_demand"] = False
    prepared_out.loc[spare_idx, "is_spare_demand"] = True
    prepared_out["candidate"] = False
    prepared_out.loc[spare_idx[candidate_idx_local], "candidate"] = True
    prepared_out["p_need_2y"] = 0.0
    prepared_out.loc[spare_idx, "p_need_2y"] = p_need_sp
    prepared_out["impact_score"] = 0.0
    prepared_out.loc[spare_idx, "impact_score"] = impact_sp
    prepared_out["is_long_lead"] = False
    prepared_out.loc[spare_idx, "is_long_lead"] = is_long_sp
    prepared_out["is_high_ao_impact"] = False
    prepared_out.loc[spare_idx, "is_high_ao_impact"] = is_high_impact_sp
    prepared_out["is_pmin_over"] = False
    prepared_out.loc[spare_idx, "is_pmin_over"] = (p_need_sp >= config.pmin)

    candidate_df = pd.DataFrame({
        "part_id": part_id_c,
        "parent_id": parent_id_c,
        "level_num": level_c,
        "depth": depth_c,
        "maint_echelon": echelon_c,
        "failure_rate_adj": annual_fr_c,
        "lambda_eff": lambda_eff_c,
        "lam_2y": lam_2y_c,
        "proc_cost": proc_cost_c,
        "lead_time": lead_c,
        "cm_cost": cm_cost_c,
        "T_diag_h": T_diag_c,
        "T_restore_intrinsic_h": T_restore_intrinsic_c,
        "baseline_wait_h": baseline_wait_c,
        "p_need_2y": p_need_c,
        "impact_score": impact_c,
        "is_long_lead": is_long_c,
        "prebuy_flag": prebuy_flag_c,
        "protection_flag": protection_flag_c,
        "critical_weight": critical_weight_c,
        "stock_cap": stock_cap_c,
        "prot_floor": prot_floor_c,
    })

    return {
        "prepared_df": prepared_out.reset_index(drop=True),
        "candidate_df": candidate_df.reset_index(drop=True),
        "m_cand": m_cand,
        "part_id_c": part_id_c,
        "parent_id_c": parent_id_c,
        "level_c": level_c,
        "depth_c": depth_c,
        "echelon_c": echelon_c,
        "annual_fr_c": annual_fr_c,
        "lambda_eff_c": lambda_eff_c,
        "lam_2y_c": lam_2y_c,
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
        "p_need_c": p_need_c,
        "impact_c": impact_c,
        "is_long_c": is_long_c,
        "prebuy_flag_c": prebuy_flag_c,
        "protection_flag_c": protection_flag_c,
        "critical_weight_c": critical_weight_c,
        "stock_cap_c": stock_cap_c,
        "prot_floor_c": prot_floor_c,
        "base_maint_spend": base_maint_spend,
        "rep_target": float(config.representative_target),
        "target_grid": [float(x) for x in config.ao_target_grid],
    }


def prepare_input_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_prepared = _prepare_engine_dataframe(df_raw)
    # preview should show candidate-related fields even before actual run
    preview_config = NSGAConfig()
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


def _ensure_sim_cache(ctx: Dict[str, object]) -> None:
    if ctx.get("_sim_cache_ready", False):
        return

    lambda_eff_c = np.asarray(ctx["lambda_eff_c"], dtype=float)
    T_issue_O_c = np.asarray(ctx["T_issue_O_c"], dtype=float)
    T_issue_I_c = np.asarray(ctx["T_issue_I_c"], dtype=float)
    T_issue_D_c = np.asarray(ctx["T_issue_D_c"], dtype=float)
    stock_cap_c = np.asarray(ctx["stock_cap_c"], dtype=int)
    echelon_c = np.asarray(ctx["echelon_c"], dtype=object)
    critical_weight_c = np.asarray(ctx["critical_weight_c"], dtype=float)
    T_diag_c = np.asarray(ctx["T_diag_c"], dtype=float)
    T_restore_intrinsic_c = np.asarray(ctx["T_restore_intrinsic_c"], dtype=float)

    ctx["_lambda_cw"] = (lambda_eff_c * critical_weight_c).astype(float)
    ctx["_dt_diag_const"] = float(np.sum(lambda_eff_c * T_diag_c * critical_weight_c))
    ctx["_dt_restore_const"] = float(np.sum(lambda_eff_c * T_restore_intrinsic_c * critical_weight_c))
    ctx["_prebuy_flag_bool"] = np.asarray(ctx["prebuy_flag_c"], dtype=bool)
    ctx["_protection_flag_bool"] = np.asarray(ctx["protection_flag_c"], dtype=bool)
    ctx["_echelon_is_O"] = np.asarray(echelon_c, dtype=object) == "O"
    ctx["_echelon_is_I"] = np.asarray(echelon_c, dtype=object) == "I"
    ctx["_echelon_is_D"] = np.asarray(echelon_c, dtype=object) == "D"
    ctx["_p_short_table_O"] = _build_poisson_tail_table(lambda_eff_c * T_issue_O_c / HOURS_PER_YEAR, stock_cap_c)
    ctx["_p_short_table_I"] = _build_poisson_tail_table(lambda_eff_c * T_issue_I_c / HOURS_PER_YEAR, stock_cap_c)
    ctx["_p_short_table_D"] = _build_poisson_tail_table(lambda_eff_c * T_issue_D_c / HOURS_PER_YEAR, stock_cap_c)
    ctx["_sim_cache_ready"] = True


def _simulate_population(
    ctx: Dict[str, object],
    Y_in: np.ndarray,
    SO_in: np.ndarray,
    SI_in: np.ndarray,
    SD_in: np.ndarray,
) -> Dict[str, np.ndarray]:
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

    m_cand = int(ctx["m_cand"])
    n_pop = Y.shape[0]

    prot_floor_c = np.asarray(ctx["prot_floor_c"], dtype=int)
    stock_cap_c = np.asarray(ctx["stock_cap_c"], dtype=int)

    SO = Y * SO
    SI = Y * SI
    SD = Y * SD

    total_stock = SO + SI + SD
    need_floor = (Y > 0) & (prot_floor_c.reshape(1, -1) > 0) & (total_stock < 1)
    if np.any(need_floor):
        SO = SO + (need_floor & ctx["_echelon_is_O"].reshape(1, -1)).astype(int)
        SI = SI + (need_floor & ctx["_echelon_is_I"].reshape(1, -1)).astype(int)
        SD = SD + (need_floor & ctx["_echelon_is_D"].reshape(1, -1)).astype(int)

    total_stock = SO + SI + SD
    excess = np.maximum(total_stock - stock_cap_c.reshape(1, -1), 0)

    reduc_D = np.minimum(SD, excess)
    SD = SD - reduc_D
    excess = excess - reduc_D

    reduc_I = np.minimum(SI, excess)
    SI = SI - reduc_I
    excess = excess - reduc_I

    reduc_O = np.minimum(SO, excess)
    SO = SO - reduc_O

    total_stock = SO + SI + SD

    item_idx = np.arange(m_cand)
    p_short_O = ctx["_p_short_table_O"][item_idx.reshape(1, -1), np.clip(SO, 0, ctx["_p_short_table_O"].shape[1] - 1)]
    p_short_I = ctx["_p_short_table_I"][item_idx.reshape(1, -1), np.clip(SI, 0, ctx["_p_short_table_I"].shape[1] - 1)]
    p_short_D = ctx["_p_short_table_D"][item_idx.reshape(1, -1), np.clip(SD, 0, ctx["_p_short_table_D"].shape[1] - 1)]

    unmanaged_mask = (Y <= 0)
    p_short_O = np.where(unmanaged_mask, 1.0, p_short_O)
    p_short_I = np.where(unmanaged_mask, 1.0, p_short_I)
    p_short_D = np.where(unmanaged_mask, 1.0, p_short_D)

    T_issue_O_c = np.asarray(ctx["T_issue_O_c"], dtype=float).reshape(1, -1)
    T_issue_I_c = np.asarray(ctx["T_issue_I_c"], dtype=float).reshape(1, -1)
    T_issue_D_c = np.asarray(ctx["T_issue_D_c"], dtype=float).reshape(1, -1)
    baseline_wait_c = np.asarray(ctx["baseline_wait_c"], dtype=float).reshape(1, -1)

    improved_wait = (
        (1.0 - p_short_O) * T_issue_O_c
        + p_short_O * (1.0 - p_short_I) * T_issue_I_c
        + p_short_O * p_short_I * (1.0 - p_short_D) * T_issue_D_c
        + p_short_O * p_short_I * p_short_D * baseline_wait_c
    )
    T_wait_per_failure = np.where(Y > 0, np.minimum(baseline_wait_c, improved_wait), baseline_wait_c)

    dt_wait = np.sum(T_wait_per_failure * ctx["_lambda_cw"].reshape(1, -1), axis=1)
    dt_diag = np.full(n_pop, ctx["_dt_diag_const"], dtype=float)
    dt_restore = np.full(n_pop, ctx["_dt_restore_const"], dtype=float)
    dt_total = dt_diag + dt_wait + dt_restore

    Ao = T_OBS_HOURS / (T_OBS_HOURS + dt_total)

    proc_cost_c = np.asarray(ctx["proc_cost_c"], dtype=float)
    stock_cost = np.sum(total_stock * proc_cost_c.reshape(1, -1), axis=1).astype(float)
    hold_cost = (H_PRE * stock_cost).astype(float)
    total_cost = stock_cost + hold_cost + float(ctx["base_maint_spend"])

    prebuy_mask = ctx["_prebuy_flag_bool"].reshape(1, -1)
    protection_mask = ctx["_protection_flag_bool"].reshape(1, -1) & (~prebuy_mask)
    stocked_mask = total_stock > 0
    normal_mask = stocked_mask & (~prebuy_mask) & (~protection_mask)

    raw_managed_items = np.sum(Y > 0, axis=1).astype(int)
    stocked_managed_parts = np.sum(stocked_mask, axis=1).astype(int)
    total_stock_units = np.sum(total_stock, axis=1).astype(int)
    prebuy_selected = np.sum(stocked_mask & prebuy_mask, axis=1).astype(int)
    protection_selected = np.sum(stocked_mask & protection_mask, axis=1).astype(int)
    normal_stock_items = np.sum(normal_mask, axis=1).astype(int)
    selected_prebuy_units = np.sum(total_stock * prebuy_mask.astype(int), axis=1).astype(int)
    selected_protection_units = np.sum(total_stock * protection_mask.astype(int), axis=1).astype(int)
    normal_stock_units = np.sum(total_stock * normal_mask.astype(int), axis=1).astype(int)

    return {
        "Ao": np.asarray(Ao, dtype=float),
        "DT_diag_h": np.asarray(dt_diag, dtype=float),
        "DT_wait_h": np.asarray(dt_wait, dtype=float),
        "DT_restore_h": np.asarray(dt_restore, dtype=float),
        "DT_total_h": np.asarray(dt_total, dtype=float),
        "stock_cost": np.asarray(stock_cost, dtype=float),
        "hold_cost": np.asarray(hold_cost, dtype=float),
        "total_cost": np.asarray(total_cost, dtype=float),
        "managed_items": np.asarray(stocked_managed_parts, dtype=int),
        "raw_managed_items": np.asarray(raw_managed_items, dtype=int),
        "stocked_managed_parts": np.asarray(stocked_managed_parts, dtype=int),
        "total_stock_units": np.asarray(total_stock_units, dtype=int),
        "prebuy_selected": np.asarray(prebuy_selected, dtype=int),
        "protection_selected": np.asarray(protection_selected, dtype=int),
        "normal_stock_items": np.asarray(normal_stock_items, dtype=int),
        "selected_prebuy_units": np.asarray(selected_prebuy_units, dtype=int),
        "selected_protection_units": np.asarray(selected_protection_units, dtype=int),
        "normal_stock_units": np.asarray(normal_stock_units, dtype=int),
        "sO": np.asarray(SO, dtype=int),
        "sI": np.asarray(SI, dtype=int),
        "sD": np.asarray(SD, dtype=int),
        "total_stock": np.asarray(total_stock, dtype=int),
        "Y": np.asarray(Y, dtype=int),
    }

def _evaluate_one_solution(ctx: Dict[str, object], y_manage: np.ndarray, sO: np.ndarray, sI: np.ndarray, sD: np.ndarray) -> Dict[str, object]:
    sim = _simulate_population(ctx, y_manage, sO, sI, sD)
    return {
        "Ao": float(sim["Ao"][0]),
        "DT_diag_h": float(sim["DT_diag_h"][0]),
        "DT_wait_h": float(sim["DT_wait_h"][0]),
        "DT_restore_h": float(sim["DT_restore_h"][0]),
        "DT_total_h": float(sim["DT_total_h"][0]),
        "stock_cost": float(sim["stock_cost"][0]),
        "hold_cost": float(sim["hold_cost"][0]),
        "total_cost": float(sim["total_cost"][0]),
        "managed_items": int(sim["managed_items"][0]),
        "raw_managed_items": int(sim["raw_managed_items"][0]),
        "stocked_managed_parts": int(sim["stocked_managed_parts"][0]),
        "total_stock_units": int(sim["total_stock_units"][0]),
        "prebuy_selected": int(sim["prebuy_selected"][0]),
        "protection_selected": int(sim["protection_selected"][0]),
        "normal_stock_items": int(sim["normal_stock_items"][0]),
        "selected_prebuy_units": int(sim["selected_prebuy_units"][0]),
        "selected_protection_units": int(sim["selected_protection_units"][0]),
        "normal_stock_units": int(sim["normal_stock_units"][0]),
        "sO": np.asarray(sim["sO"][0], dtype=int),
        "sI": np.asarray(sim["sI"][0], dtype=int),
        "sD": np.asarray(sim["sD"][0], dtype=int),
        "total_stock": np.asarray(sim["total_stock"][0], dtype=int),
    }

def _eval_nsga(ctx: Dict[str, object], pop_mat: np.ndarray, ao_target: float, config: NSGAConfig) -> Tuple[np.ndarray, np.ndarray]:
    m_cand = int(ctx["m_cand"])
    Y = np.clip(np.rint(pop_mat[:, :m_cand]), 0, 1).astype(int)
    SO = np.clip(np.rint(pop_mat[:, m_cand:m_cand + m_cand]), 0, None).astype(int)
    SI = np.clip(np.rint(pop_mat[:, m_cand + m_cand:m_cand + 2 * m_cand]), 0, None).astype(int)
    SD = np.clip(np.rint(pop_mat[:, m_cand + 2 * m_cand:m_cand + 3 * m_cand]), 0, None).astype(int)

    sim = _simulate_population(ctx, Y, SO, SI, SD)
    Ao = np.asarray(sim["Ao"], dtype=float)

    ao_shortfall = np.maximum(float(ao_target) - Ao, 0.0)
    F = np.zeros((pop_mat.shape[0], 3), dtype=float)
    F[:, 0] = np.asarray(sim["total_cost"], dtype=float) + float(config.ao_cost_penalty) * (ao_shortfall ** 2)
    F[:, 1] = ao_shortfall
    F[:, 2] = float(config.item_count_weight) * np.asarray(sim["managed_items"], dtype=float) + float(config.stock_unit_weight) * np.asarray(sim["total_stock_units"], dtype=float)
    return F, Ao

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n == 0:
        return []

    le = F[:, None, :] <= F[None, :, :]
    lt = F[:, None, :] < F[None, :, :]
    dom_matrix = np.all(le, axis=2) & np.any(lt, axis=2)
    np.fill_diagonal(dom_matrix, False)

    S = [np.flatnonzero(dom_matrix[p]).tolist() for p in range(n)]
    n_dom = dom_matrix.sum(axis=0).astype(int)

    first_front = np.flatnonzero(n_dom == 0).tolist()
    fronts: List[List[int]] = [first_front]

    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(int(q))
        i += 1
        if nxt:
            fronts.append(nxt)
        else:
            break
    return fronts




def _pareto_front_2d_min(F2d: np.ndarray) -> np.ndarray:
    F2d = np.asarray(F2d, dtype=float)
    n = F2d.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    order = np.lexsort((F2d[:, 1], F2d[:, 0]))
    sorted_pts = F2d[order]

    keep = []
    best_second = np.inf
    for pos, pt in enumerate(sorted_pts):
        second = float(pt[1])
        if second < best_second - 1e-12:
            keep.append(pos)
            best_second = second
    return order[np.asarray(keep, dtype=int)]

def crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    mobj = F.shape[1]
    dist = np.zeros(len(front), dtype=float)
    if len(front) == 0:
        return dist
    if len(front) <= 2:
        dist[:] = np.inf
        return dist

    idx = np.array(front, dtype=int)
    for j in range(mobj):
        vals = F[idx, j]
        order = np.argsort(vals)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf

        vmin = vals[order[0]]
        vmax = vals[order[-1]]
        denom = vmax - vmin
        if denom < 1e-12:
            continue

        mid_order = order[1:-1]
        if mid_order.size > 0:
            prev_vals = vals[order[:-2]]
            next_vals = vals[order[2:]]
            dist[mid_order] += (next_vals - prev_vals) / denom
    return dist

    idx = np.array(front, dtype=int)
    for j in range(mobj):
        vals = F[idx, j]
        order = np.argsort(vals)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf

        vmin, vmax = vals[order[0]], vals[order[-1]]
        if vmax - vmin < 1e-12:
            continue
        for k in range(1, len(front) - 1):
            dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (vmax - vmin)
    return dist


def rank_and_crowd(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    fronts = fast_non_dominated_sort(F)
    rank = np.empty(F.shape[0], dtype=int)
    crowd = np.zeros(F.shape[0], dtype=float)

    for rnk, front in enumerate(fronts):
        for i in front:
            rank[i] = rnk
        cd = crowding_distance(F, front)
        for t, i in enumerate(front):
            crowd[i] = cd[t]
    return rank, crowd, fronts


def tournament_select(rank: np.ndarray, crowd: np.ndarray, rng: np.random.Generator) -> int:
    i, j = rng.integers(0, len(rank), size=2)
    if rank[i] < rank[j]:
        return int(i)
    if rank[j] < rank[i]:
        return int(j)
    return int(i if crowd[i] > crowd[j] else j)


def _tournament_select_batch(rank: np.ndarray, crowd: np.ndarray, n_pick: int, rng: np.random.Generator) -> np.ndarray:
    n = int(len(rank))
    pairs = rng.integers(0, n, size=(max(int(n_pick), 1), 2))
    i = pairs[:, 0]
    j = pairs[:, 1]

    rank_i = rank[i]
    rank_j = rank[j]
    crowd_i = crowd[i]
    crowd_j = crowd[j]

    choose_i = (rank_i < rank_j) | ((rank_i == rank_j) & (crowd_i > crowd_j))
    winners = np.where(choose_i, i, j).astype(int)
    return winners


def uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(p1.shape[0]) < 0.5
    c1, c2 = p1.copy(), p2.copy()
    c1[mask] = p2[mask]
    c2[mask] = p1[mask]
    return c1, c2



def _build_offspring_population(
    pop: np.ndarray,
    rank: np.ndarray,
    crowd: np.ndarray,
    pop_size: int,
    rng: np.random.Generator,
    mutate_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    n_vars = int(pop.shape[1])
    off = np.empty((pop_size, n_vars), dtype=float)

    parent_pick_count = int(np.ceil(pop_size / 2.0)) * 2
    parents = _tournament_select_batch(rank, crowd, parent_pick_count, rng)

    p1_idx = parents[0::2]
    p2_idx = parents[1::2]
    pair_count = int(len(p1_idx))

    parent1 = pop[p1_idx]
    parent2 = pop[p2_idx]
    masks = rng.random((pair_count, n_vars)) < 0.5

    child1 = np.where(masks, parent2, parent1)
    child2 = np.where(masks, parent1, parent2)

    write_idx = 0
    for k in range(pair_count):
        off[write_idx] = mutate_fn(child1[k])
        write_idx += 1
        if write_idx < pop_size:
            off[write_idx] = mutate_fn(child2[k])
            write_idx += 1
        else:
            break
    return off


def _select_representative_solution_index(
    F: np.ndarray,
    Ao: np.ndarray,
    ao_target: float,
) -> int:
    F_arr = np.asarray(F, dtype=float)
    Ao_arr = np.asarray(Ao, dtype=float)
    n = int(len(Ao_arr))
    if n == 0:
        return 0

    ao_target_f = float(ao_target)
    cost_obj = F_arr[:, 0]
    shortfall = np.maximum(ao_target_f - Ao_arr, 0.0)
    burden = F_arr[:, 2]

    feasible_idx = np.where(Ao_arr >= (ao_target_f - 1e-12))[0]
    if feasible_idx.size > 0:
        order = np.lexsort((
            burden[feasible_idx],
            -Ao_arr[feasible_idx],
            cost_obj[feasible_idx],
        ))
        return int(feasible_idx[order[0]])

    order = np.lexsort((
        burden,
        cost_obj,
        shortfall,
    ))
    return int(order[0])


def _run_nsga_single_target(
    ctx: Dict[str, object],
    config: NSGAConfig,
    ao_target: float,
    seed: int,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    m_cand = int(ctx["m_cand"])

    n_vars = m_cand + 3 * m_cand
    mut_p = float(np.clip(20.0 / max(n_vars, 1), 0.01, 0.06))
    POP = int(config.population_size)
    GEN = int(config.n_generations)

    pop = np.zeros((POP, n_vars), dtype=float)

    p_need_c = np.asarray(ctx["p_need_c"], dtype=float)
    impact_c = np.asarray(ctx["impact_c"], dtype=float)
    is_long_c = np.asarray(ctx["is_long_c"], dtype=bool)
    stock_cap_c = np.asarray(ctx["stock_cap_c"], dtype=int)
    prot_floor_c = np.asarray(ctx["prot_floor_c"], dtype=int)
    echelon_c = np.asarray(ctx["echelon_c"], dtype=object)

    seed_prob = np.clip(
        0.12 + 0.55 * (0.45 * minmax(p_need_c) + 0.35 * minmax(impact_c) + 0.20 * is_long_c.astype(float)),
        0.08, 0.85
    )

    pop[:, :m_cand] = (rng.random((POP, m_cand)) < seed_prob.reshape(1, -1)).astype(float)

    for j in range(m_cand):
        active_rows = pop[:, j] > 0
        if not np.any(active_rows):
            continue
        ub = int(stock_cap_c[j])
        lo = int(min(int(prot_floor_c[j]), ub))
        hi = int(min(ub, 1))
        if hi < lo:
            hi = lo
        draw = rng.integers(lo, hi + 1, size=int(np.sum(active_rows)))
        if echelon_c[j] == "O":
            pop[active_rows, m_cand + j] = draw
        elif echelon_c[j] == "I":
            pop[active_rows, m_cand + m_cand + j] = draw
        else:
            pop[active_rows, m_cand + 2 * m_cand + j] = draw

    def repair_solution(v: np.ndarray) -> np.ndarray:
        y = np.clip(np.rint(v[:m_cand]), 0, 1).astype(int)
        sO = np.clip(np.rint(v[m_cand:m_cand + m_cand]), 0, None).astype(int)
        sI = np.clip(np.rint(v[m_cand + m_cand:m_cand + 2 * m_cand]), 0, None).astype(int)
        sD = np.clip(np.rint(v[m_cand + 2 * m_cand:m_cand + 3 * m_cand]), 0, None).astype(int)

        sO = y * sO
        sI = y * sI
        sD = y * sD

        for j in range(m_cand):
            if y[j] > 0 and prot_floor_c[j] > 0 and (sO[j] + sI[j] + sD[j] < 1):
                if echelon_c[j] == "O":
                    sO[j] = 1
                elif echelon_c[j] == "I":
                    sI[j] = 1
                else:
                    sD[j] = 1

            total = sO[j] + sI[j] + sD[j]
            cap = int(stock_cap_c[j])
            if total > cap:
                excess = total - cap
                reduc = min(sD[j], excess)
                sD[j] -= reduc
                excess -= reduc
                if excess > 0:
                    reduc = min(sI[j], excess)
                    sI[j] -= reduc
                    excess -= reduc
                if excess > 0:
                    reduc = min(sO[j], excess)
                    sO[j] -= reduc
        return np.concatenate([y, sO, sI, sD]).astype(float)

    stock_cap_plus1_c = np.asarray(ctx.get("stock_cap_plus1_c", np.asarray(stock_cap_c, dtype=int) + 1), dtype=int)

    def mutate(child: np.ndarray) -> np.ndarray:
        y = child.copy()
        flip_idx = np.where(rng.random(m_cand) < mut_p)[0]
        if flip_idx.size > 0:
            y[flip_idx] = 1.0 - np.rint(y[flip_idx])

        for block in range(3):
            start = m_cand + block * m_cand
            idxs = np.where(rng.random(m_cand) < mut_p)[0]
            if idxs.size > 0:
                draw = np.fromiter(
                    (rng.integers(0, int(stock_cap_plus1_c[j])) for j in idxs),
                    dtype=float,
                    count=int(idxs.size),
                )
                y[start + idxs] = draw
        return repair_solution(y)

    hist_gen = np.arange(1, GEN + 1, dtype=int)
    hist_best_ao = np.zeros(GEN, dtype=float)
    hist_mean_ao = np.zeros(GEN, dtype=float)
    hist_best_cost = np.zeros(GEN, dtype=float)
    hist_mean_cost = np.zeros(GEN, dtype=float)
    hist_best_penalty = np.zeros(GEN, dtype=float)
    hist_mean_burden = np.zeros(GEN, dtype=float)

    F, Ao = _eval_nsga(ctx, pop, ao_target, config)
    for gen in range(1, GEN + 1):
        rank, crowd, fronts = rank_and_crowd(F)

        cost_proxy = F[:, 0]
        penalty_vals = F[:, 1]
        burden_vals = F[:, 2]

        best_ao_val = float(np.max(Ao))
        mean_ao_val = float(np.mean(Ao))
        best_cost_val = float(np.min(cost_proxy))
        mean_cost_val = float(np.mean(cost_proxy))
        best_penalty_val = float(np.min(penalty_vals))
        mean_burden_val = float(np.mean(burden_vals))

        row_idx = gen - 1
        hist_best_ao[row_idx] = best_ao_val
        hist_mean_ao[row_idx] = mean_ao_val
        hist_best_cost[row_idx] = best_cost_val
        hist_mean_cost[row_idx] = mean_cost_val
        hist_best_penalty[row_idx] = best_penalty_val
        hist_mean_burden[row_idx] = mean_burden_val

        best_summary = {
            "gen": gen,
            "best_Ao": best_ao_val,
            "mean_Ao": mean_ao_val,
            "best_cost": best_cost_val,
            "mean_cost": mean_cost_val,
            "best_penalty": best_penalty_val,
            "mean_burden": mean_burden_val,
        }

        if progress_callback is not None:
            progress_callback(gen, GEN, best_summary)

        if gen == GEN:
            break

        off = _build_offspring_population(
            pop=pop,
            rank=rank,
            crowd=crowd,
            pop_size=POP,
            rng=rng,
            mutate_fn=mutate,
        )

        comb = np.vstack([pop, off])
        F_c, Ao_c = _eval_nsga(ctx, comb, ao_target, config)
        _, crowd_c, fronts_c = rank_and_crowd(F_c)

        new_idx: List[int] = []
        for front in fronts_c:
            if len(new_idx) + len(front) <= POP:
                new_idx.extend(front)
            else:
                front_np = np.array(front, dtype=int)
                order = np.argsort(-crowd_c[front_np])
                need = POP - len(new_idx)
                new_idx.extend(front_np[order[:need]].tolist())
                break

        new_idx_np = np.array(new_idx, dtype=int)
        pop = comb[new_idx_np]
        F = F_c[new_idx_np]
        Ao = Ao_c[new_idx_np]

    cost = F[:, 0]
    F2d = np.column_stack([cost, -Ao])
    pareto_idx = _pareto_front_2d_min(F2d)
    if pareto_idx.size == 0:
        pareto_idx = np.arange(len(cost), dtype=int)
    dominated_idx = np.setdiff1d(np.arange(len(cost), dtype=int), pareto_idx, assume_unique=True)

    order = np.argsort(cost[pareto_idx], kind="mergesort")
    p_idx = pareto_idx[order]

    Y = np.clip(np.rint(pop[:, :m_cand]), 0, 1).astype(int)
    SO = np.clip(np.rint(pop[:, m_cand:m_cand + m_cand]), 0, None).astype(int)
    SI = np.clip(np.rint(pop[:, m_cand + m_cand:m_cand + 2 * m_cand]), 0, None).astype(int)
    SD = np.clip(np.rint(pop[:, m_cand + 2 * m_cand:m_cand + 3 * m_cand]), 0, None).astype(int)
    sim_pop = _simulate_population(ctx, Y, SO, SI, SD)

    best_idx = _select_representative_solution_index(F, Ao, ao_target)
    best_sim = {
        "Ao": float(sim_pop["Ao"][best_idx]),
        "DT_diag_h": float(sim_pop["DT_diag_h"][best_idx]),
        "DT_wait_h": float(sim_pop["DT_wait_h"][best_idx]),
        "DT_restore_h": float(sim_pop["DT_restore_h"][best_idx]),
        "DT_total_h": float(sim_pop["DT_total_h"][best_idx]),
        "stock_cost": float(sim_pop["stock_cost"][best_idx]),
        "hold_cost": float(sim_pop["hold_cost"][best_idx]),
        "total_cost": float(sim_pop["total_cost"][best_idx]),
        "managed_items": int(sim_pop["managed_items"][best_idx]),
        "raw_managed_items": int(sim_pop["raw_managed_items"][best_idx]),
        "stocked_managed_parts": int(sim_pop["stocked_managed_parts"][best_idx]),
        "total_stock_units": int(sim_pop["total_stock_units"][best_idx]),
        "prebuy_selected": int(sim_pop["prebuy_selected"][best_idx]),
        "protection_selected": int(sim_pop["protection_selected"][best_idx]),
        "normal_stock_items": int(sim_pop["normal_stock_items"][best_idx]),
        "selected_prebuy_units": int(sim_pop["selected_prebuy_units"][best_idx]),
        "selected_protection_units": int(sim_pop["selected_protection_units"][best_idx]),
        "normal_stock_units": int(sim_pop["normal_stock_units"][best_idx]),
        "sO": np.asarray(sim_pop["sO"][best_idx], dtype=int),
        "sI": np.asarray(sim_pop["sI"][best_idx], dtype=int),
        "sD": np.asarray(sim_pop["sD"][best_idx], dtype=int),
        "total_stock": np.asarray(sim_pop["total_stock"][best_idx], dtype=int),
    }

    return {
        "target_ao": float(ao_target),
        "pop": pop,
        "F": F,
        "Ao": Ao,
        "cost": cost,
        "pareto_idx": pareto_idx,
        "dominated_idx": dominated_idx,
        "p_cost": cost[p_idx],
        "p_Ao": Ao[p_idx],
        "best_Ao": float(sim_pop["Ao"][best_idx]),
        "share_ge_target": float(np.mean(Ao >= ao_target)),
        "best_sim": best_sim,
        "best_y": np.asarray(sim_pop["Y"][best_idx], dtype=int),
        "best_sO": np.asarray(sim_pop["sO"][best_idx], dtype=int),
        "best_sI": np.asarray(sim_pop["sI"][best_idx], dtype=int),
        "best_sD": np.asarray(sim_pop["sD"][best_idx], dtype=int),
        "history_df": pd.DataFrame({
            "gen": hist_gen,
            "best_Ao": hist_best_ao,
            "mean_Ao": hist_mean_ao,
            "best_cost": hist_best_cost,
            "mean_cost": hist_mean_cost,
            "best_penalty": hist_best_penalty,
            "mean_burden": hist_mean_burden,
        }),
        "sim_pop": sim_pop,
    }

def _collect_run_dataframe(ctx: Dict[str, object], res_single: Dict[str, object]) -> pd.DataFrame:
    pop = np.asarray(res_single["pop"], dtype=float)
    target_ao = float(res_single["target_ao"])
    sim_pop = res_single.get("sim_pop")

    if not isinstance(sim_pop, dict):
        m_cand = int(ctx["m_cand"])
        Y = np.clip(np.rint(pop[:, :m_cand]), 0, 1).astype(int)
        SO = np.clip(np.rint(pop[:, m_cand:m_cand + m_cand]), 0, None).astype(int)
        SI = np.clip(np.rint(pop[:, m_cand + m_cand:m_cand + 2 * m_cand]), 0, None).astype(int)
        SD = np.clip(np.rint(pop[:, m_cand + 2 * m_cand:m_cand + 3 * m_cand]), 0, None).astype(int)
        sim_pop = _simulate_population(ctx, Y, SO, SI, SD)

    n = int(pop.shape[0])
    pareto_mask = np.zeros(n, dtype=int)
    pareto_idx = np.asarray(res_single.get("pareto_idx", np.array([], dtype=int)), dtype=int)
    pareto_idx = pareto_idx[(pareto_idx >= 0) & (pareto_idx < n)]
    pareto_mask[pareto_idx] = 1

    Y_sig = np.asarray(sim_pop["Y"], dtype=int)
    sO_sig = np.asarray(sim_pop["sO"], dtype=int)
    sI_sig = np.asarray(sim_pop["sI"], dtype=int)
    sD_sig = np.asarray(sim_pop["sD"], dtype=int)
    signatures = [solution_signature(Y_sig[i], sO_sig[i], sI_sig[i], sD_sig[i]) for i in range(n)]

    ao_arr = np.asarray(sim_pop["Ao"], dtype=float)
    total_cost_arr = np.asarray(sim_pop["total_cost"], dtype=float)
    stock_cost_arr = np.asarray(sim_pop["stock_cost"], dtype=float)
    hold_cost_arr = np.asarray(sim_pop["hold_cost"], dtype=float)
    dt_diag_arr = np.asarray(sim_pop["DT_diag_h"], dtype=float)
    dt_wait_arr = np.asarray(sim_pop["DT_wait_h"], dtype=float)
    dt_restore_arr = np.asarray(sim_pop["DT_restore_h"], dtype=float)
    dt_total_arr = np.asarray(sim_pop["DT_total_h"], dtype=float)
    managed_items_arr = np.asarray(sim_pop["managed_items"], dtype=int)
    raw_managed_items_arr = np.asarray(sim_pop.get("raw_managed_items", managed_items_arr), dtype=int)
    stocked_managed_parts_arr = np.asarray(sim_pop.get("stocked_managed_parts", managed_items_arr), dtype=int)
    total_stock_units_arr = np.asarray(sim_pop["total_stock_units"], dtype=int)
    prebuy_selected_arr = np.asarray(sim_pop.get("prebuy_selected", np.zeros(n, dtype=int)), dtype=int)
    protection_selected_arr = np.asarray(sim_pop.get("protection_selected", np.zeros(n, dtype=int)), dtype=int)
    normal_stock_items_arr = np.asarray(sim_pop.get("normal_stock_items", np.maximum(stocked_managed_parts_arr - prebuy_selected_arr - protection_selected_arr, 0)), dtype=int)
    selected_prebuy_units_arr = np.asarray(sim_pop.get("selected_prebuy_units", np.zeros(n, dtype=int)), dtype=int)
    selected_protection_units_arr = np.asarray(sim_pop.get("selected_protection_units", np.zeros(n, dtype=int)), dtype=int)
    normal_stock_units_arr = np.asarray(sim_pop.get("normal_stock_units", np.maximum(total_stock_units_arr - selected_prebuy_units_arr - selected_protection_units_arr, 0)), dtype=int)

    return pd.DataFrame({
        "target_ao": np.full(n, target_ao, dtype=float),
        "solution_id_within_target": np.arange(n, dtype=int),
        "Ao": ao_arr,
        "total_cost": total_cost_arr,
        "stock_cost": stock_cost_arr,
        "hold_cost": hold_cost_arr,
        "DT_diag_h": dt_diag_arr,
        "DT_wait_h": dt_wait_arr,
        "DT_restore_h": dt_restore_arr,
        "DT_total_h": dt_total_arr,
        "managed_items": managed_items_arr,
        "raw_managed_items": raw_managed_items_arr,
        "stocked_managed_parts": stocked_managed_parts_arr,
        "total_stock_units": total_stock_units_arr,
        "prebuy_selected": prebuy_selected_arr,
        "protection_selected": protection_selected_arr,
        "normal_stock_items": normal_stock_items_arr,
        "selected_prebuy_units": selected_prebuy_units_arr,
        "selected_protection_units": selected_protection_units_arr,
        "normal_stock_units": normal_stock_units_arr,
        "is_pareto_2d": pareto_mask.astype(int),
        "signature": signatures,
    })

def _build_pareto_df(ctx: Dict[str, object], res_single: Dict[str, object], config: NSGAConfig) -> pd.DataFrame:
    F = np.asarray(res_single["F"], dtype=float)
    Ao = np.asarray(res_single["Ao"], dtype=float)
    pareto_idx = np.asarray(res_single["pareto_idx"], dtype=int)
    sim_pop = res_single.get("sim_pop")

    if not isinstance(sim_pop, dict):
        pop = np.asarray(res_single["pop"], dtype=float)
        m_cand = int(ctx["m_cand"])
        Y = np.clip(np.rint(pop[:, :m_cand]), 0, 1).astype(int)
        SO = np.clip(np.rint(pop[:, m_cand:m_cand + m_cand]), 0, None).astype(int)
        SI = np.clip(np.rint(pop[:, m_cand + m_cand:m_cand + 2 * m_cand]), 0, None).astype(int)
        SD = np.clip(np.rint(pop[:, m_cand + 2 * m_cand:m_cand + 3 * m_cand]), 0, None).astype(int)
        sim_pop = _simulate_population(ctx, Y, SO, SI, SD)

    if pareto_idx.size == 0:
        return pd.DataFrame()

    ao_shortfall = np.maximum(float(res_single["target_ao"]) - np.asarray(sim_pop["Ao"], dtype=float)[pareto_idx], 0.0)

    df_pareto = pd.DataFrame({
        "solution_id": [f"SOL_{int(idx)+1:03d}" for idx in pareto_idx.tolist()],
        "Ao": Ao[pareto_idx].astype(float),
        "Ao_gap": ao_shortfall.astype(float),
        "total_cost": np.asarray(sim_pop["total_cost"], dtype=float)[pareto_idx],
        "stock_cost": np.asarray(sim_pop["stock_cost"], dtype=float)[pareto_idx],
        "hold_cost": np.asarray(sim_pop["hold_cost"], dtype=float)[pareto_idx],
        "DT_diag_h": np.asarray(sim_pop["DT_diag_h"], dtype=float)[pareto_idx],
        "DT_wait_h": np.asarray(sim_pop["DT_wait_h"], dtype=float)[pareto_idx],
        "DT_restore_h": np.asarray(sim_pop["DT_restore_h"], dtype=float)[pareto_idx],
        "DT_total_h": np.asarray(sim_pop["DT_total_h"], dtype=float)[pareto_idx],
        "F1_objective": F[pareto_idx, 0].astype(float),
        "F2_shortfall": F[pareto_idx, 1].astype(float),
        "F3_burden": F[pareto_idx, 2].astype(float),
        "managed_parts": np.asarray(sim_pop.get("stocked_managed_parts", sim_pop["managed_items"]), dtype=int)[pareto_idx],
        "raw_managed_parts": np.asarray(sim_pop.get("raw_managed_items", sim_pop["managed_items"]), dtype=int)[pareto_idx],
        "total_stock_units": np.asarray(sim_pop["total_stock_units"], dtype=int)[pareto_idx],
        "selected_prebuy": np.asarray(sim_pop["prebuy_selected"], dtype=int)[pareto_idx],
        "selected_protection": np.asarray(sim_pop["protection_selected"], dtype=int)[pareto_idx],
        "normal_stock_items": np.asarray(sim_pop.get("normal_stock_items", np.maximum(np.asarray(sim_pop.get("stocked_managed_parts", sim_pop["managed_items"]), dtype=int) - np.asarray(sim_pop["prebuy_selected"], dtype=int) - np.asarray(sim_pop["protection_selected"], dtype=int), 0)), dtype=int)[pareto_idx],
        "selected_prebuy_units": np.asarray(sim_pop.get("selected_prebuy_units", np.zeros_like(np.asarray(sim_pop["total_stock_units"], dtype=int))), dtype=int)[pareto_idx],
        "selected_protection_units": np.asarray(sim_pop.get("selected_protection_units", np.zeros_like(np.asarray(sim_pop["total_stock_units"], dtype=int))), dtype=int)[pareto_idx],
        "normal_stock_units": np.asarray(sim_pop.get("normal_stock_units", np.maximum(np.asarray(sim_pop["total_stock_units"], dtype=int) - np.asarray(sim_pop.get("selected_prebuy_units", np.zeros_like(np.asarray(sim_pop["total_stock_units"], dtype=int))), dtype=int) - np.asarray(sim_pop.get("selected_protection_units", np.zeros_like(np.asarray(sim_pop["total_stock_units"], dtype=int))), dtype=int), 0)), dtype=int)[pareto_idx],
    })
    if not df_pareto.empty:
        df_pareto = df_pareto.sort_values(by=["Ao", "total_cost"], ascending=[False, True]).reset_index(drop=True)
    return df_pareto

def _build_policy_df(ctx: Dict[str, object], res_single: Dict[str, object]) -> pd.DataFrame:
    best_sim = res_single["best_sim"]
    y = np.asarray(res_single["best_y"], dtype=int)
    sO = np.asarray(best_sim["sO"], dtype=int)
    sI = np.asarray(best_sim["sI"], dtype=int)
    sD = np.asarray(best_sim["sD"], dtype=int)
    total_stock = np.asarray(best_sim["total_stock"], dtype=int)

    impact_c = np.asarray(ctx["impact_c"], dtype=float)
    p_need_c = np.asarray(ctx["p_need_c"], dtype=float)
    priority_score = np.asarray(ctx.get("priority_score_c"), dtype=float)
    if priority_score.size == 0:
        lead_c = np.asarray(ctx["lead_c"], dtype=float)
        priority_score = 0.45 * minmax(impact_c) + 0.35 * minmax(p_need_c) + 0.20 * minmax(lead_c)

    reorder_signal = np.where(
        total_stock <= 0, "즉시 재주문",
        np.where(total_stock == 1, "긴급 점검", np.where(total_stock <= 3, "단기 모니터링", "정상"))
    )

    prebuy_bool = np.asarray(ctx.get("prebuy_flag_c_bool_cached", ctx["prebuy_flag_c"]), dtype=bool)
    protection_bool = np.asarray(ctx.get("protection_flag_c_bool_cached", ctx["protection_flag_c"]), dtype=bool)
    policy_type = np.where(
        prebuy_bool & (y > 0), "PRE-BUY",
        np.where(protection_bool & (total_stock > 0), "PROTECTION", np.where(y > 0, "MANAGED", "NORMAL"))
    )

    df_policy = pd.DataFrame({
        "part_id": np.asarray(ctx["part_id_c"], dtype=object),
        "parent_id": np.asarray(ctx["parent_id_c"], dtype=object),
        "level_num": np.asarray(ctx["level_c"], dtype=int),
        "depth": np.asarray(ctx["depth_c"], dtype=int),
        "maint_echelon": np.asarray(ctx["echelon_c"], dtype=object),
        "failure_rate": np.asarray(ctx["annual_fr_c"], dtype=float),
        "failure_rate_adj": np.asarray(ctx["annual_fr_c"], dtype=float),
        "lambda_eff": np.asarray(ctx["lambda_eff_c"], dtype=float),
        "lead_time": np.asarray(ctx["lead_c"], dtype=float),
        "proc_cost": np.asarray(ctx["proc_cost_c"], dtype=float),
        "impact_score": impact_c,
        "p_need_2y": p_need_c,
        "stock_cap": np.asarray(ctx["stock_cap_c"], dtype=int),
        "prebuy_flag": np.asarray(ctx["prebuy_flag_c"], dtype=bool),
        "protection_flag": np.asarray(ctx["protection_flag_c"], dtype=bool),
        "manage_flag": y > 0,
        "stock_O": sO,
        "stock_I": sI,
        "stock_D": sD,
        "recommended_stock": total_stock,
        "priority_score": priority_score,
        "policy_type": policy_type,
        "stock_bucket": policy_type,
        "stocked_manage_flag": total_stock > 0,
        "selected_prebuy_unit": np.where(policy_type == "PRE-BUY", total_stock, 0),
        "selected_protection_unit": np.where(policy_type == "PROTECTION", total_stock, 0),
        "normal_stock_unit": np.where(policy_type == "MANAGED", total_stock, 0),
        "reorder_signal": reorder_signal,
    })
    return df_policy.sort_values(by=["manage_flag", "priority_score", "recommended_stock"], ascending=[False, False, False]).reset_index(drop=True)


def run_nsga2(
    df_input: pd.DataFrame,
    config: NSGAConfig,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    start_time = time.time()

    df_prepared_base = _prepare_engine_dataframe(df_input)
    ctx = _build_engine_context(df_prepared_base, config)

    result_map: Dict[float, Dict[str, object]] = {}
    all_runs: List[pd.DataFrame] = []

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

            progress_callback(
                0,
                total_gen,
                {
                    "event": "stage_start",
                    "current_target_ao": current_target,
                    "sweep_stage_idx": stage_idx,
                    "sweep_stage_total": stage_total,
                    "representative_target": rep_target,
                    "is_representative_target": is_rep,
                    "stage_progress": 0.0,
                    "overall_progress": float((stage_idx - 1) / max(stage_total, 1)),
                },
            )

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

        res_t = _run_nsga_single_target(
            ctx=ctx,
            config=config,
            ao_target=target,
            seed=run_seed,
            progress_callback=wrapped_callback,
        )
        result_map[round(target, 6)] = res_t
        all_runs.append(_collect_run_dataframe(ctx, res_t))

    df_all_runs = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    rep_key = round(rep_target, 6)
    if rep_key not in result_map:
        raise ValueError(f"Representative target {rep_target:.2f} is not available in result map.")

    res = result_map[rep_key]
    best_sim = res["best_sim"]

    if not df_all_runs.empty:
        ao_num = pd.to_numeric(df_all_runs["Ao"], errors="coerce").fillna(0.0)
        target_num = pd.to_numeric(df_all_runs["target_ao"], errors="coerce").fillna(0.0)
        df_all_runs["_ge_target"] = (ao_num >= target_num).astype(float)

        sweep_summary_df = (
            df_all_runs.groupby("target_ao", as_index=False)
            .agg(
                n_solutions=("Ao", "count"),
                ao_min=("Ao", "min"),
                ao_max=("Ao", "max"),
                cost_min=("total_cost", "min"),
                cost_max=("total_cost", "max"),
                wait_min=("DT_wait_h", "min"),
                wait_max=("DT_wait_h", "max"),
                managed_min=("stocked_managed_parts", "min"),
                managed_max=("stocked_managed_parts", "max"),
                prebuy_min=("prebuy_selected", "min"),
                prebuy_max=("prebuy_selected", "max"),
                protection_min=("protection_selected", "min"),
                protection_max=("protection_selected", "max"),
                share_ge_target=("_ge_target", "mean"),
            )
            .sort_values("target_ao")
            .reset_index(drop=True)
        )
        df_all_runs = df_all_runs.drop(columns=["_ge_target"], errors="ignore")
    else:
        sweep_summary_df = pd.DataFrame()

    pareto_df = _build_pareto_df(ctx, res, config)
    policy_df = _build_policy_df(ctx, res)
    candidate_df = ctx["candidate_df"].copy()
    prepared_df = ctx["prepared_df"].copy()
    history_df = res["history_df"].copy()

    summary = {
        "target_ao": rep_target,
        "requested_target_ao": float(config.target_ao),
        "representative_target": rep_target,
        "best_Ao": float(best_sim["Ao"]),
        "best_cost": float(best_sim["total_cost"]),
        "best_stock_cost": float(best_sim["stock_cost"]),
        "best_hold_cost": float(best_sim["hold_cost"]),
        "best_DT_diag_h": float(best_sim["DT_diag_h"]),
        "best_DT_wait_h": float(best_sim["DT_wait_h"]),
        "best_DT_restore_h": float(best_sim["DT_restore_h"]),
        "best_DT_total_h": float(best_sim["DT_total_h"]),
        "best_managed_parts": int(best_sim.get("stocked_managed_parts", best_sim["managed_items"])),
        "best_raw_managed_parts": int(best_sim.get("raw_managed_items", best_sim["managed_items"])),
        "best_total_stock_units": int(best_sim["total_stock_units"]),
        "best_selected_prebuy": int(best_sim["prebuy_selected"]),
        "best_selected_protection": int(best_sim["protection_selected"]),
        "best_normal_stock_items": int(best_sim.get("normal_stock_items", max(int(best_sim.get("stocked_managed_parts", best_sim["managed_items"])) - int(best_sim["prebuy_selected"]) - int(best_sim["protection_selected"]), 0))),
        "best_selected_prebuy_units": int(best_sim.get("selected_prebuy_units", 0)),
        "best_selected_protection_units": int(best_sim.get("selected_protection_units", 0)),
        "best_normal_stock_units": int(best_sim.get("normal_stock_units", max(int(best_sim["total_stock_units"]) - int(best_sim.get("selected_prebuy_units", 0)) - int(best_sim.get("selected_protection_units", 0)), 0))),
        "managed_parts": int(best_sim.get("stocked_managed_parts", best_sim["managed_items"])),
        "raw_managed_parts": int(best_sim.get("raw_managed_items", best_sim["managed_items"])),
        "total_stock_units": int(best_sim["total_stock_units"]),
        "selected_prebuy": int(best_sim["prebuy_selected"]),
        "selected_protection": int(best_sim["protection_selected"]),
        "normal_stock_items": int(best_sim.get("normal_stock_items", max(int(best_sim.get("stocked_managed_parts", best_sim["managed_items"])) - int(best_sim["prebuy_selected"]) - int(best_sim["protection_selected"]), 0))),
        "selected_prebuy_units": int(best_sim.get("selected_prebuy_units", 0)),
        "selected_protection_units": int(best_sim.get("selected_protection_units", 0)),
        "normal_stock_units": int(best_sim.get("normal_stock_units", max(int(best_sim["total_stock_units"]) - int(best_sim.get("selected_prebuy_units", 0)) - int(best_sim.get("selected_protection_units", 0)), 0))),
        "stock_category_parts_sum": int(best_sim["prebuy_selected"]) + int(best_sim["protection_selected"]) + int(best_sim.get("normal_stock_items", max(int(best_sim.get("stocked_managed_parts", best_sim["managed_items"])) - int(best_sim["prebuy_selected"]) - int(best_sim["protection_selected"]), 0))),
        "stock_category_units_sum": int(best_sim.get("selected_prebuy_units", 0)) + int(best_sim.get("selected_protection_units", 0)) + int(best_sim.get("normal_stock_units", max(int(best_sim["total_stock_units"]) - int(best_sim.get("selected_prebuy_units", 0)) - int(best_sim.get("selected_protection_units", 0)), 0))),
        "candidate_count": int(ctx["m_cand"]),
        "share_ge_target": float(res["share_ge_target"]),
        "run_population_size": int(config.population_size),
        "run_n_generations": int(config.n_generations),
        "run_pmin": float(config.pmin),
        "run_long_lead_percentile": float(config.long_lead_percentile),
        "run_ao_impact_percentile": float(config.ao_impact_percentile),
        "run_random_seed": int(config.random_seed),
        "run_seconds": float(time.time() - start_time),
        "engine_mode": "target_sweep_standardized_to_0.94",
    }

    return {
        "summary": summary,
        "prepared_df": prepared_df,
        "candidate_df": candidate_df,
        "history_df": history_df,
        "pareto_df": pareto_df,
        "policy_df": policy_df,
        "sweep_summary_df": sweep_summary_df,
        "all_target_runs_df": df_all_runs,
        "result_map": result_map,
    }


# ------------------------------------------------------------
# Backward compatibility exports
# ------------------------------------------------------------
evaluate_one_solution = _evaluate_one_solution

__all__ = [
    "NSGAConfig",
    "prepare_input_dataframe",
    "run_nsga2",
    "evaluate_one_solution",
]
