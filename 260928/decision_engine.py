# -*- coding: utf-8 -*-
"""
decision_engine.py
==================

FINAL DEBUGGED · Cost / Risk decision layer for Model-C + Cold-start Monte Carlo.

Design principle
----------------
This module does NOT alter NSGA-II alternatives and does NOT rerun Monte Carlo.
It translates already-computed outputs into transparent decision-support indicators.

No hidden "best solution" is created.
No single weighted overall score is used.

Primary Monte Carlo comparison method
-------------------------------------
The Decision Layer uses '건수 가중' (total immediate fulfilled / total demand)
as the primary physical service comparison. Monthly-equal and blended results
remain available in the simulation output for sensitivity review.

Three independent decision axes
-------------------------------
1) Future Stability
   Stability_Core = min(Service_P05, Probability_Zero_Shortage)

   Rationale:
   Both values are probabilities on the same 0~1 scale.
   Using the minimum is deliberately non-compensatory: a very strong metric cannot
   hide a weak one. If Stability_Core ties (common in sparse-demand systems),
   Probability_Zero_Shortage is used only as a transparent tie-break.

2) Shortage Risk
   Primary physical indicator = Probability_Any_Shortage
                              = 1 - Probability_Zero_Shortage
   Secondary tie-break       = Shortage_P95

3) Cost Efficiency
   Baseline = minimum-stock representative alternative for the same period.
              If more than one solution has the same minimum stock, lower cost is used.
   Risk_Reduction_vs_MinStock = baseline Probability_Any_Shortage
                                - alternative Probability_Any_Shortage
   Additional_Cost_vs_MinStock = Cost - baseline Cost
   Risk_Reduction_per_100M_KRW =
       Risk_Reduction_vs_MinStock / (Additional_Cost_vs_MinStock / 100,000,000)

   The minimum-stock baseline is labeled "기준" because an incremental efficiency
   ratio is not mathematically defined when additional cost is zero. If another
   inventory composition is cheaper and also lowers shortage risk, it is labeled
   "비용감소" rather than forcing an invalid positive-cost efficiency grade.

Relative grades
---------------
A/B/C/D/E are RELATIVE grades among the representative alternatives in the SAME
simulation period. They are not regulatory certification or absolute safety levels.

- Future Stability: higher Stability_Core is better.
- Shortage Risk: lower Probability_Any_Shortage is better; Shortage_P95 is tie-break.
- Cost Efficiency: higher Risk_Reduction_per_100M_KRW is better among alternatives
  with positive additional cost and positive risk reduction.

Grades are based on deterministic sorted order with tolerance-aware ties.
If two values are numerically equal within tolerance, they receive the same grade.

The raw values and grade basis are always returned for audit.

Stage 6 interface
-----------------
decision_table_df and audit metadata are consumed by audit_engine.py.
Audit is read-only; no grade or decision result is recalculated by the audit layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


GRADE_LABELS = ["A", "B", "C", "D", "E"]


@dataclass
class DecisionConfig:
    decision_period_months: int = 36
    primary_evaluation_method: str = "건수 가중"
    tie_tolerance: float = 1e-9
    cost_unit_krw: float = 100_000_000.0
    grade_labels: Tuple[str, str, str, str, str] = ("A", "B", "C", "D", "E")


def validate_decision_config(config: DecisionConfig) -> Dict[str, object]:
    issues: List[str] = []
    if int(config.decision_period_months) <= 0:
        issues.append("decision_period_months must be > 0.")
    if not str(config.primary_evaluation_method).strip():
        issues.append("primary_evaluation_method must be non-empty.")
    if float(config.tie_tolerance) < 0:
        issues.append("tie_tolerance must be >= 0.")
    if float(config.cost_unit_krw) <= 0:
        issues.append("cost_unit_krw must be > 0.")
    if len(tuple(config.grade_labels)) != 5:
        issues.append("grade_labels must contain exactly five labels.")
    return {"ready": len(issues) == 0, "issues": issues}


def _require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _stable_relative_grades(
    values: pd.Series,
    higher_is_better: bool,
    labels: Tuple[str, str, str, str, str],
    tolerance: float,
    secondary: Optional[pd.Series] = None,
    secondary_higher_is_better: bool = False,
) -> pd.Series:
    """
    Deterministic A~E relative grading with tolerance-aware ties.

    Rank positions are mapped across available grade labels. For five distinct
    alternatives this becomes A/B/C/D/E. When values tie within tolerance they
    receive the same grade.
    """
    idx = values.index
    primary = pd.to_numeric(values, errors="coerce")
    if secondary is None:
        secondary_num = pd.Series(0.0, index=idx)
    else:
        secondary_num = pd.to_numeric(secondary, errors="coerce").fillna(0.0)

    valid = primary.notna()
    if not valid.any():
        return pd.Series(["-"] * len(values), index=idx, dtype=object)

    work = pd.DataFrame({
        "primary": primary,
        "secondary": secondary_num,
        "_idx": np.arange(len(idx)),
    }, index=idx)
    work = work.loc[valid].copy()

    work = work.sort_values(
        ["primary", "secondary", "_idx"],
        ascending=[not higher_is_better, not secondary_higher_is_better, True],
        kind="mergesort",
    )

    # Dense group number with tolerance-aware primary ties and secondary ties.
    groups: List[int] = []
    g = 0
    prev_p = None
    prev_s = None
    for _, row in work.iterrows():
        p = float(row["primary"])
        s = float(row["secondary"])
        if prev_p is None:
            groups.append(g)
        else:
            same_p = abs(p - prev_p) <= tolerance
            same_s = abs(s - prev_s) <= tolerance
            if not (same_p and same_s):
                g += 1
            groups.append(g)
        prev_p = p
        prev_s = s
    work["_group"] = groups

    n_groups = int(work["_group"].max()) + 1 if len(work) else 1
    label_list = list(labels)
    group_to_label: Dict[int, str] = {}
    for group in range(n_groups):
        if n_groups <= 1:
            label_idx = 0
        else:
            label_idx = int(round(group * (len(label_list) - 1) / (n_groups - 1)))
        group_to_label[group] = label_list[min(max(label_idx, 0), len(label_list) - 1)]

    out = pd.Series(["-"] * len(values), index=idx, dtype=object)
    for row_idx, row in work.iterrows():
        out.loc[row_idx] = group_to_label[int(row["_group"])]
    return out


def _cost_efficiency_grades(
    efficiency: pd.Series,
    additional_cost: pd.Series,
    risk_reduction: pd.Series,
    labels: Tuple[str, str, str, str, str],
    tolerance: float,
) -> pd.Series:
    out = pd.Series(["-"] * len(efficiency), index=efficiency.index, dtype=object)

    base_mask = pd.to_numeric(additional_cost, errors="coerce").fillna(0.0).abs() <= tolerance
    out.loc[base_mask] = "기준"

    add_cost = pd.to_numeric(additional_cost, errors="coerce").fillna(0.0)
    risk_red = pd.to_numeric(risk_reduction, errors="coerce").fillna(0.0)

    # If a higher-stock composition is actually cheaper than the minimum-stock baseline
    # and also reduces shortage risk, a positive incremental-cost efficiency ratio is
    # not the right concept. Mark it transparently as cost-decreasing instead.
    cost_decrease_gain = (~base_mask) & (add_cost < -tolerance) & (risk_red > tolerance)
    out.loc[cost_decrease_gain] = "비용감소"

    valid = (
        ~base_mask
        & (add_cost > tolerance)
        & pd.to_numeric(efficiency, errors="coerce").notna()
        & (risk_red > tolerance)
    )
    if valid.any():
        out.loc[valid] = _stable_relative_grades(
            pd.to_numeric(efficiency.loc[valid], errors="coerce"),
            higher_is_better=True,
            labels=labels,
            tolerance=tolerance,
        )

    # Positive additional spending without risk reduction is explicitly E.
    no_gain = (~base_mask) & (add_cost > tolerance) & (risk_red <= tolerance)
    out.loc[no_gain] = labels[-1]

    # Lower cost but not lower risk: no efficiency grade; show comparison only.
    cost_decrease_no_gain = (~base_mask) & (add_cost < -tolerance) & (risk_red <= tolerance)
    out.loc[cost_decrease_no_gain] = "비교"

    return out


def build_decision_package(
    representative_alternatives_df: pd.DataFrame,
    mc_summary_df: pd.DataFrame,
    config: Optional[DecisionConfig] = None,
) -> Dict[str, object]:
    if config is None:
        config = DecisionConfig()

    check = validate_decision_config(config)
    if not check["ready"]:
        raise ValueError("Invalid DecisionConfig: " + "; ".join(check["issues"]))

    reps = representative_alternatives_df.copy()
    mc = mc_summary_df.copy()

    _require_columns(
        reps,
        ["Alternative_ID", "Strategy", "NSGA_Cost_KRW", "Stock_Units"],
        "representative_alternatives_df",
    )
    _require_columns(
        mc,
        [
            "Alternative_ID", "Strategy", "Period_Months",
            "Service_P05", "Probability_Zero_Shortage",
            "Shortage_P95", "Shortage_Mean", "Mean_Event_Delay_Days",
        ],
        "mc_summary_df",
    )

    reps["Alternative_ID"] = reps["Alternative_ID"].astype(str)
    mc["Alternative_ID"] = mc["Alternative_ID"].astype(str)

    periods = sorted(
        pd.to_numeric(mc["Period_Months"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if not periods:
        raise ValueError("mc_summary_df contains no valid Period_Months.")

    requested_period = int(config.decision_period_months)
    decision_period = requested_period if requested_period in periods else max(periods)

    period_mask = (
        pd.to_numeric(mc["Period_Months"], errors="coerce").astype(int) == decision_period
    )
    if "Evaluation_Method" in mc.columns:
        method_mask = (
            mc["Evaluation_Method"].astype(str)
            == str(config.primary_evaluation_method)
        )
    else:
        method_mask = pd.Series(True, index=mc.index)

    period_mc = mc.loc[period_mask & method_mask].copy()
    if period_mc.empty:
        raise ValueError(
            f"No Monte Carlo summary rows for period={decision_period}, "
            f"method={config.primary_evaluation_method!r}."
        )

    merged = reps.merge(
        period_mc,
        on=["Alternative_ID", "Strategy"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching representative alternatives and Monte Carlo results.")

    numeric_cols = [
        "NSGA_Cost_KRW", "Stock_Units",
        "Service_Mean", "Service_P05", "Probability_Zero_Shortage",
        "Shortage_P95", "Shortage_Mean", "Mean_Event_Delay_Days",
    ]
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    merged["Stability_Core"] = np.minimum(
        merged["Service_P05"].clip(0.0, 1.0),
        merged["Probability_Zero_Shortage"].clip(0.0, 1.0),
    )
    merged["Probability_Any_Shortage"] = (
        1.0 - merged["Probability_Zero_Shortage"].clip(0.0, 1.0)
    )

    # Baseline is the minimum-stock representative strategy.
    # If stock ties exist, use the lower-cost alternative deterministically.
    baseline_idx = (
        merged.sort_values(
            ["Stock_Units", "NSGA_Cost_KRW", "Alternative_ID"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .index[0]
    )
    baseline_cost = float(merged.loc[baseline_idx, "NSGA_Cost_KRW"])
    baseline_risk = float(merged.loc[baseline_idx, "Probability_Any_Shortage"])
    baseline_alt = str(merged.loc[baseline_idx, "Alternative_ID"])

    merged["Additional_Cost_vs_MinStock_KRW"] = (
        merged["NSGA_Cost_KRW"] - baseline_cost
    )
    merged["Risk_Reduction_vs_MinStock"] = (
        baseline_risk - merged["Probability_Any_Shortage"]
    )

    denom = merged["Additional_Cost_vs_MinStock_KRW"] / float(config.cost_unit_krw)
    merged["Risk_Reduction_per_100M_KRW"] = np.where(
        denom > float(config.tie_tolerance),
        merged["Risk_Reduction_vs_MinStock"] / denom,
        np.nan,
    )

    labels = tuple(config.grade_labels)
    tol = float(config.tie_tolerance)

    merged["Future_Stability_Grade"] = _stable_relative_grades(
        merged["Stability_Core"],
        higher_is_better=True,
        labels=labels,
        tolerance=tol,
        secondary=merged["Probability_Zero_Shortage"],
        secondary_higher_is_better=True,
    )

    merged["Shortage_Risk_Grade"] = _stable_relative_grades(
        merged["Probability_Any_Shortage"],
        higher_is_better=False,
        labels=labels,
        tolerance=tol,
        secondary=merged["Shortage_P95"],
        secondary_higher_is_better=False,
    )

    merged["Cost_Efficiency_Grade"] = _cost_efficiency_grades(
        merged["Risk_Reduction_per_100M_KRW"],
        merged["Additional_Cost_vs_MinStock_KRW"],
        merged["Risk_Reduction_vs_MinStock"],
        labels=labels,
        tolerance=tol,
    )

    merged["Decision_Period_Months"] = int(decision_period)
    merged["Decision_Evaluation_Method"] = str(config.primary_evaluation_method)
    merged["Cost_Efficiency_Baseline_ID"] = baseline_alt

    # Human-readable descriptions without recommending a winner.
    merged["Decision_Interpretation"] = merged.apply(
        lambda r: (
            f"{r['Strategy']} | 미래 안정성 상대등급 {r['Future_Stability_Grade']} | "
            f"부족위험 상대등급 {r['Shortage_Risk_Grade']} | "
            f"비용효율 상대등급 {r['Cost_Efficiency_Grade']} | "
            f"Service P05 {float(r['Service_P05'])*100:.1f}% | "
            f"재고부족 0회 확률 {float(r['Probability_Zero_Shortage'])*100:.1f}%"
        ),
        axis=1,
    )

    decision_table_cols = [
        "Alternative_ID", "Strategy", "Decision_Period_Months", "Decision_Evaluation_Method",
        "Stock_Units", "NSGA_Cost_KRW",
        "Service_Mean", "Service_P05", "Probability_Zero_Shortage",
        "Stability_Core", "Future_Stability_Grade",
        "Probability_Any_Shortage", "Shortage_P95", "Shortage_Mean",
        "Shortage_Risk_Grade",
        "Additional_Cost_vs_MinStock_KRW", "Risk_Reduction_vs_MinStock",
        "Risk_Reduction_per_100M_KRW", "Cost_Efficiency_Grade",
        "Mean_Event_Delay_Days",
        "Cost_Efficiency_Baseline_ID",
        "Decision_Interpretation",
    ]
    decision_table_df = merged[decision_table_cols].sort_values(
        ["Stock_Units", "NSGA_Cost_KRW"],
        ascending=[True, True],
    ).reset_index(drop=True)

    grade_audit_df = pd.DataFrame([
        {
            "Axis": "미래 안정성 상대등급",
            "Raw_Metric": "Stability_Core = min(Service_P05, Probability_Zero_Shortage)",
            "Direction": "높을수록 좋음",
            "Grade_Method": "Stability_Core 우선, 동률 시 Probability_Zero_Shortage로만 tie-break; 동일하면 동일 등급",
            "Absolute_Certification": False,
        },
        {
            "Axis": "부족위험 상대등급",
            "Raw_Metric": "Probability_Any_Shortage = 1 - Probability_Zero_Shortage; Shortage_P95 tie-break",
            "Direction": "낮을수록 좋음",
            "Grade_Method": "동일 기간 대표대안 간 상대 A~E, 수치 동률은 동일 등급",
            "Absolute_Certification": False,
        },
        {
            "Axis": "비용효율 상대등급",
            "Raw_Metric": "Risk_Reduction_vs_MinStock / (Additional_Cost_vs_MinStock / 100M KRW)",
            "Direction": "높을수록 좋음",
            "Grade_Method": "최소비용안은 '기준'; 추가비용 대비 부족확률 감소가 있는 대안끼리 상대 A~E",
            "Absolute_Certification": False,
        },
    ])

    decision_config_df = pd.DataFrame([{
        "Requested_Decision_Period_Months": requested_period,
        "Applied_Decision_Period_Months": decision_period,
        "Primary_Evaluation_Method": str(config.primary_evaluation_method),
        "Tie_Tolerance": float(config.tie_tolerance),
        "Cost_Unit_KRW": float(config.cost_unit_krw),
        "Baseline_Alternative_ID": baseline_alt,
        "Baseline_Type": "Minimum stock representative alternative",
        "Baseline_Cost_KRW": baseline_cost,
        "Baseline_Probability_Any_Shortage": baseline_risk,
        "Overall_Weighted_Score_Used": False,
        "Automatic_Best_Selection_Used": False,
        "Grade_Type": "Relative descriptive grade within same period",
    }])

    # Positioning map data; no composite score.
    positioning_df = decision_table_df[[
        "Alternative_ID", "Strategy", "Stock_Units", "NSGA_Cost_KRW",
        "Service_Mean", "Stability_Core", "Future_Stability_Grade",
        "Probability_Any_Shortage", "Shortage_Risk_Grade",
        "Risk_Reduction_per_100M_KRW", "Cost_Efficiency_Grade",
    ]].copy()

    return {
        "decision_table_df": decision_table_df,
        "decision_positioning_df": positioning_df,
        "decision_grade_audit_df": grade_audit_df,
        "decision_config_df": decision_config_df,
        "decision_summary": {
            "decision_period_months": int(decision_period),
            "primary_evaluation_method": str(config.primary_evaluation_method),
            "baseline_alternative_id": baseline_alt,
            "alternative_count": int(len(decision_table_df)),
            "overall_weighted_score_used": False,
            "automatic_best_selection_used": False,
        },
    }


__all__ = [
    "DecisionConfig",
    "validate_decision_config",
    "build_decision_package",
]
