# -*- coding: utf-8 -*-
"""
audit_engine.py
===============

PBL SUPPLY STRUCTURE OPTION · Final integration / audit / regression layer.

Purpose
-------
This module validates the integrity of the already-computed pipeline:

    Model-C NSGA-II
        -> Representative Bridge
        -> Cold-start Monte Carlo
        -> Cost/Risk Decision Layer

It does NOT modify or re-optimize any result.

Core checks
-----------
1. Required output tables exist.
2. Representative alternative IDs are unique.
3. Representative Stock_Units reconcile with MC Bridge Stock_Qty sums.
4. MC periods/iteration counts are structurally complete.
5. Common Random Numbers are preserved across alternatives.
6. Probability metrics are in [0, 1].
7. Decision formulas reproduce from raw Monte Carlo metrics.
8. Minimum-stock baseline is used for cost-efficiency.
9. No overall weighted score / no automatic best selection.
10. Decision grades remain descriptive/relative only.

Outputs
-------
audit_summary_df
audit_checks_df
audit_strategy_df
audit_part_risk_df
audit_config_df
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AuditConfig:
    probability_tolerance: float = 1e-9
    numeric_tolerance: float = 1e-8
    expected_periods: tuple[int, ...] = (12, 24, 36)


def validate_audit_config(config: AuditConfig) -> Dict[str, object]:
    issues: List[str] = []
    if float(config.probability_tolerance) < 0:
        issues.append("probability_tolerance must be >= 0.")
    if float(config.numeric_tolerance) < 0:
        issues.append("numeric_tolerance must be >= 0.")
    periods = tuple(int(x) for x in config.expected_periods)
    if not periods or any(x <= 0 for x in periods):
        issues.append("expected_periods must contain positive integers.")
    return {"ready": len(issues) == 0, "issues": issues, "expected_periods": periods}


def _check_row(name: str, status: bool, detail: str, severity: str = "ERROR") -> Dict[str, object]:
    return {
        "Check": name,
        "Status": "PASS" if bool(status) else "FAIL",
        "Severity": severity,
        "Detail": detail,
    }


def _safe_df(result: dict, key: str) -> pd.DataFrame:
    value = result.get(key, pd.DataFrame())
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def build_final_audit_package(
    result: dict,
    config: Optional[AuditConfig] = None,
) -> Dict[str, object]:
    if config is None:
        config = AuditConfig()

    cfg_check = validate_audit_config(config)
    if not cfg_check["ready"]:
        raise ValueError("Invalid AuditConfig: " + "; ".join(cfg_check["issues"]))

    checks: List[Dict[str, object]] = []
    tol = float(config.numeric_tolerance)
    ptol = float(config.probability_tolerance)
    expected_periods = tuple(int(x) for x in config.expected_periods)

    reps = _safe_df(result, "representative_alternatives_df")
    bridge = _safe_df(result, "monte_carlo_bridge_df")
    mc_summary = _safe_df(result, "mc_summary_df")
    mc_iter = _safe_df(result, "mc_iteration_df")
    mc_part = _safe_df(result, "mc_part_summary_df")
    decision = _safe_df(result, "decision_table_df")
    decision_audit = _safe_df(result, "decision_grade_audit_df")
    decision_cfg = _safe_df(result, "decision_config_df")

    required_tables = {
        "representative_alternatives_df": reps,
        "monte_carlo_bridge_df": bridge,
        "mc_summary_df": mc_summary,
        "mc_iteration_df": mc_iter,
        "decision_table_df": decision,
    }
    for key, df in required_tables.items():
        checks.append(_check_row(
            f"required_table::{key}",
            isinstance(df, pd.DataFrame) and not df.empty,
            f"rows={len(df) if isinstance(df, pd.DataFrame) else 0}",
        ))

    # Representative IDs
    rep_unique_ok = (
        not reps.empty
        and "Alternative_ID" in reps.columns
        and reps["Alternative_ID"].astype(str).is_unique
    )
    checks.append(_check_row(
        "representative_id_unique",
        rep_unique_ok,
        f"representative_count={len(reps)}",
    ))

    # Bridge stock reconciliation
    bridge_stock_ok = True
    bridge_stock_details = []
    if reps.empty or bridge.empty or not {"Alternative_ID", "Stock_Units"}.issubset(reps.columns) \
            or not {"Alternative_ID", "Stock_Qty"}.issubset(bridge.columns):
        bridge_stock_ok = False
        bridge_stock_details.append("required columns missing")
    else:
        for _, r in reps.iterrows():
            aid = str(r["Alternative_ID"])
            expected_stock = int(round(float(pd.to_numeric(r["Stock_Units"], errors="coerce") or 0)))
            actual_stock = int(
                pd.to_numeric(
                    bridge.loc[bridge["Alternative_ID"].astype(str) == aid, "Stock_Qty"],
                    errors="coerce",
                ).fillna(0).sum()
            )
            ok = expected_stock == actual_stock
            bridge_stock_ok = bridge_stock_ok and ok
            bridge_stock_details.append(f"{aid}:{actual_stock}/{expected_stock}")
    checks.append(_check_row(
        "bridge_stock_reconciliation",
        bridge_stock_ok,
        " | ".join(bridge_stock_details[:12]),
    ))

    supply_stock_ok = False
    supply_detail = []
    required_supply_cols = {
        "Stock_O", "Stock_I", "Stock_D", "Stock_Qty",
        "Maint_Echelon", "Stocking_Echelon", "Supply_Structure_Mode"
    }
    if not bridge.empty and required_supply_cols.issubset(bridge.columns):
        so = pd.to_numeric(bridge["Stock_O"], errors="coerce").fillna(0.0)
        si = pd.to_numeric(bridge["Stock_I"], errors="coerce").fillna(0.0)
        sd = pd.to_numeric(bridge["Stock_D"], errors="coerce").fillna(0.0)
        sq = pd.to_numeric(bridge["Stock_Qty"], errors="coerce").fillna(0.0)

        maint = bridge["Maint_Echelon"].astype(str).str.upper().str.strip()
        stocking = bridge["Stocking_Echelon"].astype(str).str.upper().str.strip()
        modes = bridge["Supply_Structure_Mode"].astype(str).str.upper().str.strip()

        mode_values = sorted(modes.dropna().unique().tolist())
        one_mode_ok = len(mode_values) == 1
        mode = mode_values[0] if one_mode_ok else ""

        total_ok = bool(np.allclose(so + si + sd, sq, atol=tol, rtol=0.0))

        if mode == "MAINT_ECHELON":
            stocking_rule_ok = bool((stocking == maint.where(maint.isin(["O","I","D"]), "D")).all())
            o_ok = bool(((stocking != "O") | ((si.abs() <= tol) & (sd.abs() <= tol))).all())
            i_ok = bool(((stocking != "I") | ((so.abs() <= tol) & (sd.abs() <= tol))).all())
            d_ok = bool(((stocking != "D") | ((so.abs() <= tol) & (si.abs() <= tol))).all())
            supply_stock_ok = bool(one_mode_ok and total_ok and stocking_rule_ok and o_ok and i_ok and d_ok)
            supply_detail = [
                f"mode={mode}",
                f"total_equals_qty={total_ok}",
                f"stocking_matches_maint={stocking_rule_ok}",
                f"O_stock_alignment={o_ok}",
                f"I_stock_alignment={i_ok}",
                f"D_stock_alignment={d_ok}",
            ]

        elif mode == "DEPOT_DIRECT":
            stocking_rule_ok = bool((stocking == "D").all())
            o_zero = bool((so.abs() <= tol).all())
            i_zero = bool((si.abs() <= tol).all())
            d_equals_total = bool(np.allclose(sd, sq, atol=tol, rtol=0.0))
            supply_stock_ok = bool(
                one_mode_ok and total_ok and stocking_rule_ok
                and o_zero and i_zero and d_equals_total
            )
            supply_detail = [
                f"mode={mode}",
                f"total_equals_qty={total_ok}",
                f"all_stocking_echelon_D={stocking_rule_ok}",
                f"Stock_O_zero={o_zero}",
                f"Stock_I_zero={i_zero}",
                f"Stock_D_equals_total={d_equals_total}",
            ]
        else:
            supply_detail = [
                f"mode_values={mode_values}",
                "unsupported_or_mixed_supply_structure_mode",
            ]

    checks.append(_check_row(
        "pbl_supply_structure_stock_alignment",
        supply_stock_ok,
        " | ".join(supply_detail) if supply_detail else "required columns missing",
    ))

    bridge_lambda_ok = False
    if not bridge.empty and {
        "Failure_Rate", "Annual_Operating_Hours", "QPA", "Lambda_Annual_MC_Base"
    }.issubset(bridge.columns):
        fr = pd.to_numeric(bridge["Failure_Rate"], errors="coerce").fillna(0.0).to_numpy(float)
        oh = pd.to_numeric(bridge["Annual_Operating_Hours"], errors="coerce").fillna(0.0).to_numpy(float)
        qpa = pd.to_numeric(bridge["QPA"], errors="coerce").fillna(1.0).to_numpy(float)
        lam = pd.to_numeric(bridge["Lambda_Annual_MC_Base"], errors="coerce").fillna(0.0).to_numpy(float)
        expected_lam = fr * oh / 1_000_000.0 * qpa
        bridge_lambda_ok = bool(np.allclose(expected_lam, lam, atol=tol, rtol=1e-10))
    checks.append(_check_row(
        "mc_bridge_lambda_consistency",
        bridge_lambda_ok,
        "Lambda_Annual_MC_Base == Failure_Rate × Annual_Operating_Hours / 1e6 × QPA",
    ))

    # MC summary period coverage.
    periods_actual = set()
    if not mc_summary.empty and "Period_Months" in mc_summary.columns:
        periods_actual = set(
            pd.to_numeric(mc_summary["Period_Months"], errors="coerce").dropna().astype(int).tolist()
        )
    period_ok = set(expected_periods).issubset(periods_actual)
    checks.append(_check_row(
        "mc_period_coverage",
        period_ok,
        f"expected={list(expected_periods)}, actual={sorted(periods_actual)}",
    ))

    # Summary structural completeness: alternatives x periods x evaluation methods.
    alt_count = int(reps["Alternative_ID"].nunique()) if not reps.empty and "Alternative_ID" in reps.columns else 0
    if "Evaluation_Method" in mc_summary.columns:
        methods_actual = sorted(mc_summary["Evaluation_Method"].astype(str).dropna().unique().tolist())
    else:
        methods_actual = []
    expected_methods = ["가중평균", "건수 가중", "월 동일 가중"]
    methods_ok = set(expected_methods).issubset(set(methods_actual))
    checks.append(_check_row(
        "mc_evaluation_methods",
        methods_ok,
        f"expected={expected_methods}, actual={methods_actual}",
    ))

    expected_summary_rows = alt_count * len(expected_periods) * len(expected_methods)
    summary_shape_ok = len(mc_summary) == expected_summary_rows if expected_summary_rows > 0 else False
    checks.append(_check_row(
        "mc_summary_shape",
        summary_shape_ok,
        f"actual={len(mc_summary)}, expected={expected_summary_rows}",
    ))

    # Probability bounds.
    probability_cols = ["Service_Mean", "Service_P05", "Probability_Zero_Shortage"]
    prob_ok = True
    prob_detail = []
    for col in probability_cols:
        if col not in mc_summary.columns:
            prob_ok = False
            prob_detail.append(f"{col}:missing")
            continue
        v = pd.to_numeric(mc_summary[col], errors="coerce")
        ok = v.notna().all() and ((v >= -ptol) & (v <= 1.0 + ptol)).all()
        prob_ok = prob_ok and bool(ok)
        prob_detail.append(f"{col}:{'OK' if ok else 'BAD'}")
    checks.append(_check_row(
        "mc_probability_bounds",
        prob_ok,
        ", ".join(prob_detail),
    ))

    # CRN integrity.
    # K is part-specific, so iteration output stores Mean_K/P95_K summaries.
    # These summaries and total physical demand must be identical across alternatives.
    crn_k_ok = False
    crn_demand_ok = False
    if not mc_iter.empty and {
        "Period_Months", "Iteration", "Mean_K", "P95_K", "Demand_Total"
    }.issubset(mc_iter.columns):
        g = mc_iter.groupby(["Period_Months", "Iteration"], dropna=False).agg(
            MeanK_nunique=("Mean_K", "nunique"),
            P95K_nunique=("P95_K", "nunique"),
            Demand_nunique=("Demand_Total", "nunique"),
        )
        crn_k_ok = bool(
            (g["MeanK_nunique"] <= 1).all()
            and (g["P95K_nunique"] <= 1).all()
        )
        crn_demand_ok = bool((g["Demand_nunique"] <= 1).all())
    checks.append(_check_row(
        "mc_common_random_numbers_K",
        crn_k_ok,
        "same period/iteration must share identical K across alternatives",
    ))
    checks.append(_check_row(
        "mc_common_random_numbers_demand",
        crn_demand_ok,
        "same period/iteration must share identical Demand_Total across alternatives",
    ))

    # Decision formula checks.
    stability_ok = False
    shortage_prob_ok = False
    baseline_ok = False
    no_auto_best_ok = False
    no_weighted_ok = False

    if not decision.empty:
        if {"Service_P05", "Probability_Zero_Shortage", "Stability_Core"}.issubset(decision.columns):
            expected_stability = np.minimum(
                pd.to_numeric(decision["Service_P05"], errors="coerce").fillna(0.0).to_numpy(float),
                pd.to_numeric(decision["Probability_Zero_Shortage"], errors="coerce").fillna(0.0).to_numpy(float),
            )
            actual_stability = pd.to_numeric(
                decision["Stability_Core"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            stability_ok = bool(np.allclose(expected_stability, actual_stability, atol=tol, rtol=0.0))

        if {"Probability_Zero_Shortage", "Probability_Any_Shortage"}.issubset(decision.columns):
            expected_risk = 1.0 - pd.to_numeric(
                decision["Probability_Zero_Shortage"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            actual_risk = pd.to_numeric(
                decision["Probability_Any_Shortage"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            shortage_prob_ok = bool(np.allclose(expected_risk, actual_risk, atol=tol, rtol=0.0))

        if {"Stock_Units", "Alternative_ID", "Cost_Efficiency_Baseline_ID"}.issubset(decision.columns):
            min_stock = float(pd.to_numeric(decision["Stock_Units"], errors="coerce").min())
            baseline_id = str(decision["Cost_Efficiency_Baseline_ID"].iloc[0])
            hit = decision[decision["Alternative_ID"].astype(str) == baseline_id]
            baseline_ok = (
                not hit.empty
                and abs(float(pd.to_numeric(hit["Stock_Units"], errors="coerce").iloc[0]) - min_stock) <= tol
            )

    checks.append(_check_row(
        "decision_stability_formula",
        stability_ok,
        "Stability_Core == min(Service_P05, Probability_Zero_Shortage)",
    ))
    checks.append(_check_row(
        "decision_shortage_probability_formula",
        shortage_prob_ok,
        "Probability_Any_Shortage == 1 - Probability_Zero_Shortage",
    ))
    checks.append(_check_row(
        "decision_min_stock_baseline",
        baseline_ok,
        "cost-efficiency baseline must be minimum-stock representative alternative",
    ))

    decision_method_ok = False
    if not decision.empty and "Decision_Evaluation_Method" in decision.columns:
        decision_method_ok = bool(
            (decision["Decision_Evaluation_Method"].astype(str) == "건수 가중").all()
        )
    checks.append(_check_row(
        "decision_primary_method_case_weighted",
        decision_method_ok,
        "Decision Layer must use 건수 가중 as the primary physical comparison",
    ))

    if not decision_cfg.empty:
        if "Automatic_Best_Selection_Used" in decision_cfg.columns:
            no_auto_best_ok = bool(
                (decision_cfg["Automatic_Best_Selection_Used"].astype(bool) == False).all()
            )
        if "Overall_Weighted_Score_Used" in decision_cfg.columns:
            no_weighted_ok = bool(
                (decision_cfg["Overall_Weighted_Score_Used"].astype(bool) == False).all()
            )
    checks.append(_check_row(
        "decision_no_automatic_best",
        no_auto_best_ok,
        "automatic best solution selection must remain disabled",
    ))
    checks.append(_check_row(
        "decision_no_weighted_overall_score",
        no_weighted_ok,
        "single weighted overall score must remain disabled",
    ))

    relative_grade_ok = False
    if not decision_audit.empty and "Absolute_Certification" in decision_audit.columns:
        relative_grade_ok = bool(
            (decision_audit["Absolute_Certification"].astype(bool) == False).all()
        )
    checks.append(_check_row(
        "decision_grades_are_relative",
        relative_grade_ok,
        "A~E grades must be descriptive/relative, not absolute certification",
    ))

    audit_checks_df = pd.DataFrame(checks)
    total_checks = int(len(audit_checks_df))
    pass_count = int((audit_checks_df["Status"] == "PASS").sum()) if total_checks else 0
    fail_count = total_checks - pass_count
    overall_status = "PASS" if fail_count == 0 else "CHECK"

    audit_summary_df = pd.DataFrame([{
        "Overall_Status": overall_status,
        "Total_Checks": total_checks,
        "Pass_Count": pass_count,
        "Fail_Count": fail_count,
        "Representative_Alternatives": alt_count,
        "MC_Periods": ",".join(map(str, sorted(periods_actual))),
        "Decision_Alternatives": int(len(decision)),
        "Automatic_Best_Selection": False,
        "Weighted_Overall_Score": False,
    }])

    # Strategy-level final consolidated view.
    if not decision.empty:
        strategy_cols = [c for c in [
            "Alternative_ID", "Strategy", "Decision_Period_Months",
            "Stock_Units", "NSGA_Cost_KRW",
            "Service_P05", "Probability_Zero_Shortage",
            "Shortage_P95", "Mean_Event_Delay_Days",
            "Future_Stability_Grade", "Shortage_Risk_Grade",
            "Cost_Efficiency_Grade",
        ] if c in decision.columns]
        audit_strategy_df = decision[strategy_cols].copy()
    else:
        audit_strategy_df = pd.DataFrame()

    # Part-level final risk drill-down.
    if not mc_part.empty:
        part = mc_part.copy()
        for c in ["Shortage_Mean", "Demand_Mean", "Mean_Shortage_Delay_Days"]:
            if c not in part.columns:
                part[c] = 0.0
            part[c] = pd.to_numeric(part[c], errors="coerce").fillna(0.0)
        part["Risk_Priority_Index"] = (
            part["Shortage_Mean"]
            * (1.0 + part["Mean_Shortage_Delay_Days"])
        )
        audit_part_risk_df = part.sort_values(
            ["Period_Months", "Risk_Priority_Index", "Demand_Mean"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    else:
        audit_part_risk_df = pd.DataFrame()

    audit_config_df = pd.DataFrame([{
        "Expected_Periods": ",".join(map(str, expected_periods)),
        "Probability_Tolerance": ptol,
        "Numeric_Tolerance": tol,
        "Audit_Changes_Analysis_Results": False,
        "Audit_Only": True,
    }])

    return {
        "audit_summary_df": audit_summary_df,
        "audit_checks_df": audit_checks_df,
        "audit_strategy_df": audit_strategy_df,
        "audit_part_risk_df": audit_part_risk_df,
        "audit_config_df": audit_config_df,
        "audit_summary": {
            "overall_status": overall_status,
            "total_checks": total_checks,
            "pass_count": pass_count,
            "fail_count": fail_count,
        },
    }


__all__ = [
    "AuditConfig",
    "validate_audit_config",
    "build_final_audit_package",
]
