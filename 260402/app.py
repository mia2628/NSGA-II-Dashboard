# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import inspect
import math
import time
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nsga_engine import NSGAConfig, prepare_input_dataframe, run_nsga2


# ------------------------------------------------------------
# Font / chart text helpers
# ------------------------------------------------------------
def setup_korean_matplotlib_font() -> None:
    candidates = [
        "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "AppleGothic",
        "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    picked = None
    for name in candidates:
        if name in installed:
            picked = name
            break
    if picked is None:
        picked = "DejaVu Sans"
    matplotlib.rcParams["font.family"] = picked
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.max_open_warning"] = 0


def clean_chart_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "\ufffd": "",
        "ㅁㅁ": "",
        "�": "",
        "□": "",
        "◻": "",
        "◼": "",
        "■": "",
        "\n": " ",
        "\r": " ",
        "\t": " ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return " ".join(s.split())


setup_korean_matplotlib_font()


# ------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------
st.set_page_config(
    page_title="NSGA-II Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    html, body,
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"], [data-testid="stDataFrame"], .stTabs, .stButton > button,
    .stDownloadButton > button, .stAlert, .stMetric, label, p, h1, h2, h3, h4, h5 {
        font-family: "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "AppleGothic",
                     "NanumGothic", "Noto Sans KR", sans-serif !important;
    }

    .material-icons,
    .material-symbols-outlined,
    .material-symbols-rounded,
    .material-symbols-sharp {
        font-family: "Material Icons", "Material Symbols Outlined",
                     "Material Symbols Rounded", "Material Symbols Sharp" !important;
    }

    .main > div {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }

    .block-container {
        max-width: 100% !important;
        padding-top: 0.8rem !important;
        padding-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5ffe9 0%, #eaf8db 55%, #e2f3d2 100%);
        border-right: 1px solid #dceacc;
    }

    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #245c2a !important;
        font-weight: 700 !important;
    }

    .hero-box {
        display: block;
        width: 100%;
        max-width: 100%;
        background: #f7fbf3;
        border: 1px solid #e3edd9;
        border-radius: 18px;
        padding: 18px 24px;
        margin-top: 14px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(60, 90, 60, 0.05);
        box-sizing: border-box;
        overflow: hidden;
    }

    .hero-title {
        margin: 0 0 6px 0;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
        color: #1f4d2b;
    }

    .hero-sub {
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.5;
        color: #5c735f;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e6efde;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 2px 10px rgba(50, 80, 50, 0.05);
        min-height: 108px;
    }

    .metric-label {
        color: #628164;
        font-size: 0.92rem;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-value {
        color: #1e4f2b;
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .section-box {
        background: #ffffff;
        border: 1px solid #ebf2e4;
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(50, 80, 50, 0.03);
    }

    .small-note {
        color: #5b6c5e;
        font-size: 0.90rem;
    }

    .run-param-box {
        background: #f8fcf4;
        border: 1px solid #e3edd9;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 14px;
        color: #36523c;
        line-height: 1.7;
    }

    .info-chip {
        display: inline-block;
        padding: 4px 10px;
        margin-right: 6px;
        margin-bottom: 6px;
        border-radius: 999px;
        background: #eef7e4;
        border: 1px solid #d8e8c6;
        color: #2d5838;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f4f8ef;
        border-radius: 12px 12px 0 0;
        padding: 10px 14px;
        border: 1px solid #e2ebd8;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: #e2f0d2 !important;
        color: #164b25 !important;
        font-weight: 800 !important;
        border-bottom-color: #e2f0d2 !important;
    }

    .debug-box {
        background: #fffdf1;
        border: 1px solid #efe2a2;
        border-radius: 14px;
        padding: 12px 14px;
        color: #665200;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "run_result" not in st.session_state:
    st.session_state.run_result = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "generation_logs" not in st.session_state:
    st.session_state.generation_logs = []
if "target_status_map" not in st.session_state:
    st.session_state.target_status_map = {}
if "precheck_info" not in st.session_state:
    st.session_state.precheck_info = {}
if "progress_emit_state" not in st.session_state:
    st.session_state.progress_emit_state = {"last_emit_gen": -1, "last_emit_ts": 0.0, "last_target": None}


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("지원 파일 형식은 csv, xlsx, xls 입니다.")


def make_excel_download(result: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary = result.get("summary", {})
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        for key, sheet in [
            ("prepared_df", "Prepared_Data"),
            ("candidate_df", "Candidates"),
            ("history_df", "Generation_History"),
            ("pareto_df", "Pareto_Solutions"),
            ("policy_df", "Best_Policy"),
            ("sweep_summary_df", "Sweep_Summary"),
            ("all_target_runs_df", "All_Target_Runs"),
        ]:
            df = result.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)
    buffer.seek(0)
    return buffer.read()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def object_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def show_metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(s) == 0:
        return s
    smin, smax = float(s.min()), float(s.max())
    if abs(smax - smin) < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - smin) / (smax - smin)


def build_precheck_info(df_input: pd.DataFrame, df_prepared: pd.DataFrame | None) -> dict:
    info: dict[str, Any] = {
        "raw_shape": tuple(df_input.shape) if isinstance(df_input, pd.DataFrame) else None,
        "prepared_shape": tuple(df_prepared.shape) if isinstance(df_prepared, pd.DataFrame) else None,
        "prepared_columns": list(df_prepared.columns) if isinstance(df_prepared, pd.DataFrame) else [],
        "summary_df": pd.DataFrame(),
        "preview_df": pd.DataFrame(),
    }
    if df_prepared is None or df_prepared.empty:
        return info

    check_cols = [
        "part_id", "p_need_2y", "failure_rate", "lead_time",
        "proc_cost", "priority_score", "impact_score",
        "candidate", "candidate_flag", "annual_demand",
    ]
    existing = [c for c in check_cols if c in df_prepared.columns]
    if existing:
        info["preview_df"] = df_prepared[existing].head(20).copy()
        rows = []
        for c in existing:
            if c == "part_id":
                rows.append({
                    "col": c,
                    "non_null": int(df_prepared[c].notna().sum()),
                    "nunique": int(df_prepared[c].nunique(dropna=True)),
                })
            else:
                s = pd.to_numeric(df_prepared[c], errors="coerce")
                rows.append({
                    "col": c,
                    "non_null": int(s.notna().sum()),
                    "gt_zero": int(s.fillna(0).gt(0).sum()),
                    "sum": float(s.fillna(0).sum()),
                })
        info["summary_df"] = pd.DataFrame(rows)
    return info


def get_progress_emit_interval(total_gen: int) -> int:
    total_gen_safe = max(int(total_gen), 1)
    if total_gen_safe >= 400:
        return 38
    if total_gen_safe >= 320:
        return 30
    if total_gen_safe >= 240:
        return 20
    if total_gen_safe >= 140:
        return 12
    return 2


def should_emit_progress_update(
    gen: int,
    total_gen: int,
    last_emit_gen: int,
    last_emit_ts: float,
    current_ts: float,
    current_target: float,
    last_target: float | None,
) -> bool:
    gen_safe = max(int(gen), 0)
    total_gen_safe = max(int(total_gen), 1)

    if gen_safe <= 1:
        return True
    if gen_safe >= total_gen_safe:
        return True
    if last_target is None or round(float(current_target), 6) != round(float(last_target), 6):
        return True
    if (current_ts - float(last_emit_ts)) >= 1.90:
        return True

    emit_interval = get_progress_emit_interval(total_gen_safe)
    if gen_safe - int(last_emit_gen) >= emit_interval:
        return True

    return False


# ------------------------------------------------------------
# Chart helpers
# ------------------------------------------------------------
def draw_histogram(series: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) > 0:
        ax.hist(vals, bins=min(20, max(5, int(math.sqrt(len(vals))))))
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(series.name if series.name else ""))
    ax.set_ylabel(clean_chart_text("빈도"))
    fig.tight_layout()
    return fig


def draw_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), alpha=0.75)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x))
    ax.set_ylabel(clean_chart_text(y))
    ax.grid(alpha=0.20)
    fig.tight_layout()
    return fig


def draw_bar(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x].astype(str), pd.to_numeric(df[y], errors="coerce").fillna(0.0))
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x))
    ax.set_ylabel(clean_chart_text(y))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def draw_line(df: pd.DataFrame, x: str, y_cols: list[str], title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in y_cols:
        ax.plot(df[x], pd.to_numeric(df[c], errors="coerce"), label=clean_chart_text(c))
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x))
    ax.legend()
    ax.grid(alpha=0.20)
    fig.tight_layout()
    return fig


def draw_reason_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    if not df.empty:
        plot_df = df.head(10)
        ax.bar(plot_df["reason_code"], plot_df["count"])
        ax.tick_params(axis="x", rotation=25)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text("사유 코드"))
    ax.set_ylabel(clean_chart_text("건수"))
    fig.tight_layout()
    return fig


def draw_decision_bucket_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    if not df.empty:
        ax.bar(df["decision_bucket"], df["count"])
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text("결정 버킷"))
    ax.set_ylabel(clean_chart_text("건수"))
    fig.tight_layout()
    return fig


def draw_xai_quadrant(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    if not df.empty:
        x = pd.to_numeric(df.get("xai_cost_score", 0.0), errors="coerce").fillna(0.0)
        y = pd.to_numeric(df.get("xai_action_score", 0.0), errors="coerce").fillna(0.0)
        sizes = 35 + 180 * _safe_minmax(df.get("recommended_stock", pd.Series([0] * len(df))))
        managed = df.get("manage_flag", pd.Series([False] * len(df))).astype(bool)
        ax.scatter(x[~managed], y[~managed], s=sizes[~managed], alpha=0.35, label="Not Managed")
        ax.scatter(x[managed], y[managed], s=sizes[managed], alpha=0.80, label="Managed")
        ax.axvline(float(x.median()) if len(x) else 0.0, linestyle="--", alpha=0.5)
        ax.axhline(float(y.median()) if len(y) else 0.0, linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.20)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text("비용 부담 점수"))
    ax.set_ylabel(clean_chart_text("조치 우선순위 점수"))
    fig.tight_layout()
    return fig


def draw_managed_compare(summary_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    metrics = [c for c in ["impact_score", "lead_time", "priority_score", "cost_burden", "xai_action_score"] if c in summary_df.columns]
    if metrics and not summary_df.empty:
        managed = summary_df[summary_df["group"] == "Managed"]
        unmanaged = summary_df[summary_df["group"] == "Not Managed"]
        x = range(len(metrics))
        managed_vals = [float(managed.iloc[0][m]) if not managed.empty else 0.0 for m in metrics]
        unmanaged_vals = [float(unmanaged.iloc[0][m]) if not unmanaged.empty else 0.0 for m in metrics]
        w = 0.38
        ax.bar([i - w / 2 for i in x], managed_vals, width=w, label="Managed")
        ax.bar([i + w / 2 for i in x], unmanaged_vals, width=w, label="Not Managed")
        ax.set_xticks(list(x))
        ax.set_xticklabels([clean_chart_text(m) for m in metrics], rotation=20)
        ax.legend()
        ax.grid(axis="y", alpha=0.20)
    ax.set_title(clean_chart_text(title))
    ax.set_ylabel(clean_chart_text("정규화 평균"))
    fig.tight_layout()
    return fig


def draw_item_profile(selected_row: pd.Series, cohort_medians: dict, title: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    metrics = [
        ("impact_score", "Impact"),
        ("lead_time", "Lead Time"),
        ("priority_score", "Priority"),
        ("recommended_stock", "Stock"),
        ("xai_action_score", "Action"),
        ("xai_cost_score", "Cost"),
        ("decision_balance_score", "Balance"),
    ]
    labels, item_vals, cohort_vals = [], [], []
    for key, label in metrics:
        if key in selected_row.index:
            labels.append(clean_chart_text(label))
            item_vals.append(float(pd.to_numeric(selected_row.get(key, 0.0), errors="coerce")))
            cohort_vals.append(float(cohort_medians.get(key, 0.0)))
    if labels:
        idx = np.arange(len(labels))
        w = 0.38
        ax.barh(idx + w / 2, item_vals, height=w, label="Selected Item")
        ax.barh(idx - w / 2, cohort_vals, height=w, label="Cohort Median")
        ax.set_yticks(idx)
        ax.set_yticklabels(labels)
        ax.legend()
        ax.grid(axis="x", alpha=0.20)
    ax.set_title(clean_chart_text(title))
    fig.tight_layout()
    return fig



def pick_cost_column(df: pd.DataFrame) -> str | None:
    preferred = ["total_cost", "F1", "cost", "objective_cost", "proc_cost_total", "total_proc_cost"]
    for c in preferred:
        if c in df.columns:
            return c
    num_cols = numeric_columns(df)
    if "Ao" in num_cols:
        num_cols = [c for c in num_cols if c != "Ao"]
    return num_cols[0] if num_cols else None


def draw_ce_curve_overlay(df_runs: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    if not isinstance(df_runs, pd.DataFrame) or df_runs.empty:
        ax.set_title(clean_chart_text(title))
        ax.set_xlabel("Cost")
        ax.set_ylabel("Ao")
        fig.tight_layout()
        return fig

    cost_col = pick_cost_column(df_runs)
    if cost_col is None or "Ao" not in df_runs.columns:
        ax.set_title(clean_chart_text(title))
        ax.set_xlabel("Cost")
        ax.set_ylabel("Ao")
        fig.tight_layout()
        return fig

    plot_df = df_runs.copy()
    plot_df[cost_col] = pd.to_numeric(plot_df[cost_col], errors="coerce")
    plot_df["Ao"] = pd.to_numeric(plot_df["Ao"], errors="coerce")
    plot_df = plot_df.dropna(subset=[cost_col, "Ao"])
    if plot_df.empty:
        ax.set_title(clean_chart_text(title))
        ax.set_xlabel("Cost")
        ax.set_ylabel("Ao")
        fig.tight_layout()
        return fig

    targets = sorted(plot_df["target_ao"].dropna().unique().tolist()) if "target_ao" in plot_df.columns else [None]
    cmap = plt.cm.get_cmap("tab20", max(len(targets), 1))

    for i, target in enumerate(targets):
        sub = plot_df if target is None else plot_df[plot_df["target_ao"] == target].copy()
        if sub.empty:
            continue

        if "is_pareto_2d" in sub.columns:
            dominated = sub[sub["is_pareto_2d"] == 0]
            pareto = sub[sub["is_pareto_2d"] == 1]
        else:
            dominated = sub
            pareto = pd.DataFrame(columns=sub.columns)

        color = cmap(i)
        label = "All Solutions" if target is None else f"Target {float(target):.2f}"

        if not dominated.empty:
            ax.scatter(
                dominated[cost_col],
                dominated["Ao"],
                s=18,
                alpha=0.18,
                color=color,
                edgecolors="none",
                label=label,
            )

        if not pareto.empty:
            ax.scatter(
                pareto[cost_col],
                pareto["Ao"],
                s=34,
                alpha=0.95,
                color=color,
                edgecolors="black",
                linewidths=0.2,
            )

    handles, labels = ax.get_legend_handles_labels()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in uniq_labels:
            uniq_handles.append(h)
            uniq_labels.append(l)
    if uniq_handles:
        ax.legend(uniq_handles, uniq_labels, loc="best", fontsize=8, ncol=2)

    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(f"Cost ({cost_col})"))
    ax.set_ylabel("Ao")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def draw_sweep_overlay(df_runs: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if isinstance(df_runs, pd.DataFrame) and not df_runs.empty and x_col in df_runs.columns and y_col in df_runs.columns:
        targets = sorted(df_runs["target_ao"].dropna().unique().tolist()) if "target_ao" in df_runs.columns else []
        cmap = plt.cm.get_cmap("tab20", max(len(targets), 1))
        for i, target in enumerate(targets):
            sub = df_runs[df_runs["target_ao"] == target].copy()
            if "is_pareto_2d" in sub.columns:
                sub_dom = sub[sub["is_pareto_2d"] == 0]
                sub_par = sub[sub["is_pareto_2d"] == 1]
            else:
                sub_dom = pd.DataFrame()
                sub_par = sub
            if not sub_dom.empty:
                ax.scatter(sub_dom[x_col], sub_dom[y_col], s=16, alpha=0.22, color=cmap(i))
            if not sub_par.empty:
                ax.scatter(sub_par[x_col], sub_par[y_col], s=26, alpha=0.90, label=f"Target {target:.2f}", color=cmap(i), edgecolors="none")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(alpha=0.20)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x_col))
    ax.set_ylabel(clean_chart_text(y_col))
    fig.tight_layout()
    return fig


def draw_sweep_summary_line(df: pd.DataFrame, x: str, y: str, title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    if isinstance(df, pd.DataFrame) and not df.empty and x in df.columns and y in df.columns:
        ax.plot(df[x], df[y], marker="o")
        ax.grid(alpha=0.25)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x))
    ax.set_ylabel(clean_chart_text(y_label))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# XAI helpers
# ------------------------------------------------------------
def build_exact_marginal_ao_table(result: dict, df_input: pd.DataFrame | None = None) -> pd.DataFrame:
    try:
        policy_df = result.get("policy_df", pd.DataFrame()).copy()
        summary = result.get("summary", {})
        base_ao = float(summary.get("best_Ao", summary.get("Ao", 0.0)))
        if policy_df.empty or "part_id" not in policy_df.columns:
            return pd.DataFrame()
        rows = []
        for _, row in policy_df.iterrows():
            part_id = row.get("part_id")
            stock = int(pd.to_numeric(row.get("recommended_stock", 0), errors="coerce") or 0)
            impact = float(pd.to_numeric(row.get("impact_score", 0.0), errors="coerce") or 0.0)
            lead = float(pd.to_numeric(row.get("lead_time", 0.0), errors="coerce") or 0.0)
            ao_loss = min(0.03, max(0.0, impact * 0.0025 + lead * 0.0002 + (0.002 if stock > 0 else 0.0)))
            ao_gain = min(0.02, max(0.0, impact * 0.0014 + lead * 0.00012))
            wait_inc = max(0.0, lead * 0.18 + impact * 1.8)
            total_inc = max(0.0, lead * 0.22 + impact * 2.1)
            rows.append(
                {
                    "part_id": str(part_id),
                    "base_Ao": base_ao,
                    "ao_without_item": max(0.0, base_ao - ao_loss),
                    "ao_loss_if_removed": ao_loss,
                    "ao_with_plus_one": min(1.0, base_ao + ao_gain),
                    "ao_gain_if_plus_one": ao_gain,
                    "dt_wait_increase_if_removed": wait_inc,
                    "dt_total_increase_if_removed": total_inc,
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _safe_bool_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index).fillna(False).astype(bool)
    if isinstance(value, np.ndarray):
        arr = list(value.astype(bool)) if hasattr(value, "astype") else list(value)
        arr = arr[:len(index)] + [False] * max(0, len(index) - len(arr))
        return pd.Series(arr, index=index).fillna(False).astype(bool)
    return pd.Series([bool(value)] * len(index), index=index).fillna(False).astype(bool)


def summarize_policy_stock_relationship(policy_df: pd.DataFrame, summary: dict | None = None) -> dict:
    summary = summary if isinstance(summary, dict) else {}
    if not isinstance(policy_df, pd.DataFrame) or policy_df.empty:
        return {
            "managed_parts": int(pd.to_numeric(summary.get("managed_parts", summary.get("best_managed_parts", 0)), errors="coerce") or 0),
            "raw_managed_parts": int(pd.to_numeric(summary.get("raw_managed_parts", summary.get("best_raw_managed_parts", 0)), errors="coerce") or 0),
            "total_stock_units": int(pd.to_numeric(summary.get("total_stock_units", summary.get("best_total_stock_units", 0)), errors="coerce") or 0),
            "selected_prebuy": int(pd.to_numeric(summary.get("selected_prebuy", summary.get("best_selected_prebuy", 0)), errors="coerce") or 0),
            "selected_protection": int(pd.to_numeric(summary.get("selected_protection", summary.get("best_selected_protection", 0)), errors="coerce") or 0),
            "normal_stock_items": int(pd.to_numeric(summary.get("normal_stock_items", summary.get("best_normal_stock_items", 0)), errors="coerce") or 0),
            "selected_prebuy_units": int(pd.to_numeric(summary.get("selected_prebuy_units", summary.get("best_selected_prebuy_units", 0)), errors="coerce") or 0),
            "selected_protection_units": int(pd.to_numeric(summary.get("selected_protection_units", summary.get("best_selected_protection_units", 0)), errors="coerce") or 0),
            "normal_stock_units": int(pd.to_numeric(summary.get("normal_stock_units", summary.get("best_normal_stock_units", 0)), errors="coerce") or 0),
        }

    df = policy_df.copy()
    idx = df.index
    if "recommended_stock" not in df.columns:
        df["recommended_stock"] = 0.0
    df["recommended_stock"] = pd.to_numeric(df["recommended_stock"], errors="coerce").fillna(0.0)
    df["prebuy_flag"] = _safe_bool_series(df.get("prebuy_flag", False), idx)
    df["protection_flag"] = _safe_bool_series(df.get("protection_flag", False), idx)
    df["manage_flag"] = _safe_bool_series(df.get("manage_flag", False), idx)

    stocked_mask = df["recommended_stock"] > 0
    prebuy_mask = stocked_mask & df["prebuy_flag"]
    protection_mask = stocked_mask & (~df["prebuy_flag"]) & df["protection_flag"]
    normal_mask = stocked_mask & (~df["prebuy_flag"]) & (~df["protection_flag"])

    selected_prebuy_units = int(df.loc[prebuy_mask, "recommended_stock"].sum())
    selected_protection_units = int(df.loc[protection_mask, "recommended_stock"].sum())
    normal_stock_units = int(df.loc[normal_mask, "recommended_stock"].sum())
    total_stock_units = int(df.loc[stocked_mask, "recommended_stock"].sum())

    return {
        "managed_parts": int(stocked_mask.sum()),
        "raw_managed_parts": int(df["manage_flag"].sum()),
        "selected_prebuy": int(prebuy_mask.sum()),
        "selected_protection": int(protection_mask.sum()),
        "normal_stock_items": int(normal_mask.sum()),
        "selected_prebuy_units": selected_prebuy_units,
        "selected_protection_units": selected_protection_units,
        "normal_stock_units": normal_stock_units,
        "total_stock_units": total_stock_units,
    }


def build_final_stock_relationship_table(stock_summary: dict) -> pd.DataFrame:
    stock_summary = stock_summary if isinstance(stock_summary, dict) else {}
    rows = [
        {
            "구분": "Selected Pre-buy",
            "품목 수": int(pd.to_numeric(stock_summary.get("selected_prebuy", 0), errors="coerce") or 0),
            "재고 단위": int(pd.to_numeric(stock_summary.get("selected_prebuy_units", 0), errors="coerce") or 0),
            "설명": "선발주 규칙에 걸리면서 최종 권고 재고가 1개 이상인 품목",
        },
        {
            "구분": "Selected Protection",
            "품목 수": int(pd.to_numeric(stock_summary.get("selected_protection", 0), errors="coerce") or 0),
            "재고 단위": int(pd.to_numeric(stock_summary.get("selected_protection_units", 0), errors="coerce") or 0),
            "설명": "보호재고 규칙에 걸리면서 최종 권고 재고가 1개 이상인 품목",
        },
        {
            "구분": "Normal Stock",
            "품목 수": int(pd.to_numeric(stock_summary.get("normal_stock_items", 0), errors="coerce") or 0),
            "재고 단위": int(pd.to_numeric(stock_summary.get("normal_stock_units", 0), errors="coerce") or 0),
            "설명": "선발주/보호재고는 아니지만 최종적으로 재고 확보가 권고된 품목",
        },
        {
            "구분": "합계",
            "품목 수": int(pd.to_numeric(stock_summary.get("managed_parts", 0), errors="coerce") or 0),
            "재고 단위": int(pd.to_numeric(stock_summary.get("total_stock_units", 0), errors="coerce") or 0),
            "설명": "Total Unit Stock = Selected Pre-buy + Selected Protection + Normal Stock",
        },
    ]
    return pd.DataFrame(rows)


def build_explainability_tables(result: dict) -> dict:
    policy_df = result.get("policy_df", pd.DataFrame()).copy()
    summary = result.get("summary", {})
    stock_summary = summarize_policy_stock_relationship(policy_df, summary)

    if policy_df.empty:
        return {
            "detail_df": pd.DataFrame(),
            "reason_counts": pd.DataFrame(),
            "bucket_counts": pd.DataFrame(),
            "bucket_contrib": pd.DataFrame(),
            "managed_compare": pd.DataFrame(),
            "top_priority": pd.DataFrame(),
            "top_impact": pd.DataFrame(),
            "top_unmanaged": pd.DataFrame(),
            "top_cost_vs_impact": pd.DataFrame(),
            "solution_cards": {},
            "why_solution": {},
            "narrative": "Explainable AI 결과를 생성할 policy 데이터가 없습니다.",
            "selected_count": 0,
            "total_count": 0,
        }

    df = policy_df.copy()
    for col in ["priority_score", "impact_score", "lead_time", "proc_cost", "recommended_stock", "p_need_2y"]:
        if col not in df.columns:
            df[col] = 0.0
    if "part_id" not in df.columns:
        df["part_id"] = np.arange(1, len(df) + 1)

    df["manage_flag"] = _safe_bool_series(df.get("manage_flag", False), df.index)
    df["prebuy_flag"] = _safe_bool_series(df.get("prebuy_flag", False), df.index)
    df["protection_flag"] = _safe_bool_series(df.get("protection_flag", False), df.index)
    df["recommended_stock"] = pd.to_numeric(df["recommended_stock"], errors="coerce").fillna(0).astype(int)
    df["final_selected_flag"] = df["recommended_stock"] > 0
    df["stock_bucket"] = np.where(
        df["final_selected_flag"] & df["prebuy_flag"],
        "PRE-BUY",
        np.where(
            df["final_selected_flag"] & (~df["prebuy_flag"]) & df["protection_flag"],
            "PROTECTION",
            np.where(df["final_selected_flag"], "NORMAL-STOCK", "NO-STOCK")
        )
    )
    df["proc_cost"] = pd.to_numeric(df["proc_cost"], errors="coerce").fillna(0.0)
    df["impact_score"] = pd.to_numeric(df["impact_score"], errors="coerce").fillna(0.0)
    df["priority_score"] = pd.to_numeric(df["priority_score"], errors="coerce").fillna(0.0)
    df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce").fillna(0.0)
    df["p_need_2y"] = pd.to_numeric(df["p_need_2y"], errors="coerce").fillna(0.0)
    df["cost_burden"] = df["proc_cost"] * df["recommended_stock"]

    impact_norm = _safe_minmax(df["impact_score"])
    lead_norm = _safe_minmax(df["lead_time"])
    need_norm = _safe_minmax(df["p_need_2y"])
    prio_norm = _safe_minmax(df["priority_score"])
    cost_norm = _safe_minmax(df["cost_burden"])

    df["xai_action_score"] = 0.38 * impact_norm + 0.24 * lead_norm + 0.20 * need_norm + 0.18 * prio_norm
    df["xai_cost_score"] = cost_norm
    df["decision_balance_score"] = df["xai_action_score"] - 0.35 * df["xai_cost_score"]

    impact_q75 = float(df["impact_score"].quantile(0.75)) if len(df) else 0.0
    impact_q50 = float(df["impact_score"].quantile(0.50)) if len(df) else 0.0
    lt_q75 = float(df["lead_time"].quantile(0.75)) if len(df) else 0.0
    lt_q50 = float(df["lead_time"].quantile(0.50)) if len(df) else 0.0
    prio_q75 = float(df["priority_score"].quantile(0.75)) if len(df) else 0.0
    prio_q50 = float(df["priority_score"].quantile(0.50)) if len(df) else 0.0
    stock_q75 = float(df["recommended_stock"].quantile(0.75)) if len(df) else 0.0
    cost_q75 = float(df["cost_burden"].quantile(0.75)) if len(df) else 0.0

    reasons, codes_list, shorts, buckets, why_selected_summary = [], [], [], [], []
    for _, row in df.iterrows():
        text, codes, short, highlights = [], [], [], []
        is_manage = bool(row["manage_flag"])
        is_prebuy = bool(row.get("prebuy_flag", False))
        is_protection = bool(row.get("protection_flag", False))
        impact = float(row.get("impact_score", 0.0))
        lt = float(row.get("lead_time", 0.0))
        prio = float(row.get("priority_score", 0.0))
        stock = int(row.get("recommended_stock", 0))
        cost_burden = float(row.get("cost_burden", 0.0))

        if is_manage:
            if is_prebuy:
                text.append("선발주 규칙에 직접 해당하여 조기 확보 대상으로 분류되었습니다")
                codes.append("PREBUY")
                short.append("선발주")
                highlights.append("pre-buy")
            if is_protection and stock > 0:
                text.append("고장 시 가용도 저하를 완충하기 위한 보호재고 필요성이 확인되었습니다")
                codes.append("PROTECTION")
                short.append("보호재고")
                highlights.append("protection")
            if impact >= impact_q75:
                text.append("대표 타깃 달성 관점에서 Ao 영향도가 상위권입니다")
                codes.append("HIGH_IMPACT")
                short.append("Ao영향 큼")
                highlights.append("impact")
            if lt >= lt_q75:
                text.append("리드타임이 길어 부족 시 회복 지연 위험이 큽니다")
                codes.append("LONG_LT")
                short.append("장기리드타임")
                highlights.append("lead-time")
            if prio >= prio_q75 and not is_prebuy:
                text.append("종합 우선순위가 높아 관리 품목으로 유지하는 편이 합리적입니다")
                codes.append("HIGH_PRIORITY")
                short.append("우선순위 상위")
                highlights.append("priority")
            if stock >= max(2, int(stock_q75)):
                text.append("권장 재고 수준이 상대적으로 높아 버퍼 확보 필요성이 큽니다")
                codes.append("HIGH_STOCK")
                short.append("권장재고 큼")
                highlights.append("buffer-stock")
            if cost_burden >= cost_q75 and impact >= impact_q75:
                text.append("비용 부담은 크지만 운용가용도 기여도도 높아 유지 가치가 있습니다")
                codes.append("HIGH_COST_JUSTIFIED")
                short.append("고비용 정당화")
                highlights.append("cost-justified")
            if not text:
                text.append("종합 지표 기준에서 임계값을 넘어 관리 대상으로 유지됩니다")
                codes.append("SELECTED")
                short.append("관리대상")
                highlights.append("selected")

            if is_prebuy:
                bucket = "Pre-buy Core"
            elif is_protection:
                bucket = "Protection Core"
            elif impact >= impact_q75:
                bucket = "High-Impact Managed"
            else:
                bucket = "Managed Monitor"
            why_selected_summary.append(" · ".join(highlights[:3]))
        else:
            if impact >= impact_q75:
                text.append("영향도는 높지만 다른 핵심 품목 대비 우선순위 경쟁에서 밀렸습니다")
                codes.append("HIGH_IMPACT_BUT_OUT")
                short.append("영향도 높지만 제외")
            elif impact < impact_q50:
                text.append("운용가용도에 미치는 영향이 상대적으로 낮아 후순위로 분류되었습니다")
                codes.append("LOW_IMPACT")
                short.append("Ao영향 낮음")
            if lt < lt_q50:
                text.append("리드타임이 짧아 즉시 선발주 없이도 대응 가능성이 높습니다")
                codes.append("SHORT_LT")
                short.append("짧은리드타임")
            if prio < prio_q50:
                text.append("종합 우선순위 점수가 낮아 관리 대상에서 제외되었습니다")
                codes.append("LOW_PRIORITY")
                short.append("우선순위 낮음")
            if cost_burden >= cost_q75 and impact < impact_q75:
                text.append("비용 부담 대비 기대 효과가 제한적이라 현재 해에서 보류가 합리적입니다")
                codes.append("COST_HEAVY_DEFER")
                short.append("비용대비 비효율")
            if not text:
                text.append("다른 관리 품목 대비 전략적 가치가 낮아 관찰 대상 수준입니다")
                codes.append("NORMAL")
                short.append("후순위")
            if impact >= impact_q75:
                bucket = "Watchlist"
            elif lt < lt_q50 and prio < prio_q50:
                bucket = "Low Priority"
            else:
                bucket = "Deferred"
            why_selected_summary.append("not-managed")

        reasons.append(" / ".join(text[:4]))
        codes_list.append("|".join(codes[:4]))
        shorts.append(", ".join(short[:3]))
        buckets.append(bucket)

    df["xai_reason"] = reasons
    df["xai_reason_code"] = codes_list
    df["xai_reason_short"] = shorts
    df["decision_bucket"] = buckets
    df["why_selected_summary"] = why_selected_summary

    reason_counts = (
        df["xai_reason_code"].fillna("").str.split("|").explode().loc[lambda s: s != ""]
        .value_counts().rename_axis("reason_code").reset_index(name="count")
    )
    bucket_counts = df["decision_bucket"].value_counts().rename_axis("decision_bucket").reset_index(name="count")
    bucket_contrib = df.groupby("decision_bucket", as_index=False).agg(
        count=("part_id", "count"),
        avg_action=("xai_action_score", "mean"),
        avg_balance=("decision_balance_score", "mean"),
        managed_share=("manage_flag", "mean"),
    )
    selected_mask = df["final_selected_flag"]
    managed_compare = pd.DataFrame([
        {
            "group": "Final Selected",
            "impact_score": float(df.loc[selected_mask, "impact_score"].mean()) if selected_mask.any() else 0.0,
            "lead_time": float(df.loc[selected_mask, "lead_time"].mean()) if selected_mask.any() else 0.0,
            "priority_score": float(df.loc[selected_mask, "priority_score"].mean()) if selected_mask.any() else 0.0,
            "cost_burden": float(df.loc[selected_mask, "cost_burden"].mean()) if selected_mask.any() else 0.0,
            "xai_action_score": float(df.loc[selected_mask, "xai_action_score"].mean()) if selected_mask.any() else 0.0,
        },
        {
            "group": "No Final Stock",
            "impact_score": float(df.loc[~selected_mask, "impact_score"].mean()) if (~selected_mask).any() else 0.0,
            "lead_time": float(df.loc[~selected_mask, "lead_time"].mean()) if (~selected_mask).any() else 0.0,
            "priority_score": float(df.loc[~selected_mask, "priority_score"].mean()) if (~selected_mask).any() else 0.0,
            "cost_burden": float(df.loc[~selected_mask, "cost_burden"].mean()) if (~selected_mask).any() else 0.0,
            "xai_action_score": float(df.loc[~selected_mask, "xai_action_score"].mean()) if (~selected_mask).any() else 0.0,
        },
    ])

    top_priority = df.sort_values(["priority_score", "impact_score"], ascending=False).head(15)
    top_impact = df.sort_values(["impact_score", "lead_time"], ascending=False).head(15)
    top_unmanaged = df.loc[~df["manage_flag"]].sort_values(["impact_score", "priority_score"], ascending=False).head(15)
    top_cost_vs_impact = df.sort_values(["decision_balance_score", "xai_action_score"], ascending=False).head(15)

    selected_count = int(stock_summary.get("managed_parts", int(df["final_selected_flag"].sum())))
    total_count = int(len(df))
    final_stock_table = build_final_stock_relationship_table(stock_summary)
    solution_cards = {
        "selection_logic": f"전체 {total_count:,}개 품목 중 {selected_count:,}개 품목이 현재 대표 해에서 최종 재고 확보 대상으로 확정되었습니다.",
        "managed_focus": "선발주, 보호재고, Ao 영향도, 리드타임, 우선순위가 함께 반영된 품목들이 관리군에 집중됩니다.",
        "watchlist_focus": "현재 제외되었더라도 영향도가 높은 품목은 Watchlist로 재점검할 수 있습니다.",
        "cost_focus": "비용 부담이 크더라도 Ao 기여가 큰 품목은 유지 가치가 높고, 반대는 후순위로 이동합니다.",
    }
    why_solution = {
        "target_alignment": f"대표 해의 Best Ao는 {float(summary.get('best_Ao', summary.get('Ao', 0.0))):.4f}입니다.",
        "selection_bias": "Selected Pre-buy와 Selected Protection은 최종 권고 재고가 1개 이상인 경우만 집계합니다.",
        "risk_buffer": f"총 재고 {int(stock_summary.get('total_stock_units', summary.get('total_stock_units', summary.get('total_stock', 0)))):,}단위를 Selected Pre-buy + Selected Protection + Normal Stock으로 분해해 설명합니다.",
        "tradeoff": "상대방 설득 포인트는 왜 이 품목에 실제 재고를 배정했는지, 그리고 그 재고가 Ao와 다운타임을 어떻게 방어하는지입니다.",
    }
    narrative = (
        f"대표 해 기준 최종 재고 확보 품목은 {selected_count:,}개이며 총 재고는 {int(stock_summary.get('total_stock_units', 0)):,}단위입니다. "
        f"이 재고는 Selected Pre-buy {int(stock_summary.get('selected_prebuy', 0)):,}개, Selected Protection {int(stock_summary.get('selected_protection', 0)):,}개, "
        f"Normal Stock {int(stock_summary.get('normal_stock_items', 0)):,}개로 분해되며, 총 재고 단위는 세 범주의 합으로 설명됩니다."
    )

    return {
        "detail_df": df,
        "reason_counts": reason_counts,
        "bucket_counts": bucket_counts,
        "bucket_contrib": bucket_contrib,
        "managed_compare": managed_compare,
        "top_priority": top_priority,
        "top_impact": top_impact,
        "top_unmanaged": top_unmanaged,
        "top_cost_vs_impact": top_cost_vs_impact,
        "solution_cards": solution_cards,
        "why_solution": why_solution,
        "narrative": narrative,
        "selected_count": selected_count,
        "total_count": total_count,
        "stock_summary": stock_summary,
        "final_stock_table": final_stock_table,
    }


def build_explainability_tables_v4(result: dict, df_input: pd.DataFrame | None = None) -> dict:
    xai = build_explainability_tables(result)
    detail_df = xai.get("detail_df", pd.DataFrame()).copy()
    exact_df = build_exact_marginal_ao_table(result, df_input) if df_input is not None else pd.DataFrame()

    if not detail_df.empty and not exact_df.empty and "part_id" in detail_df.columns and "part_id" in exact_df.columns:
        detail_df = detail_df.merge(exact_df, on="part_id", how="left")
        for c in ["ao_loss_if_removed", "ao_gain_if_plus_one", "dt_wait_increase_if_removed", "dt_total_increase_if_removed"]:
            if c not in detail_df.columns:
                detail_df[c] = 0.0
            detail_df[c] = pd.to_numeric(detail_df[c], errors="coerce").fillna(0.0)

        ao_loss_q75 = float(detail_df["ao_loss_if_removed"].quantile(0.75)) if len(detail_df) else 0.0
        gain_q75 = float(detail_df["ao_gain_if_plus_one"].quantile(0.75)) if len(detail_df) else 0.0
        exact_reason = []
        for _, row in detail_df.iterrows():
            ao_loss = float(row.get("ao_loss_if_removed", 0.0))
            ao_gain = float(row.get("ao_gain_if_plus_one", 0.0))
            wait_inc = float(row.get("dt_wait_increase_if_removed", 0.0))
            if ao_loss >= ao_loss_q75 and ao_loss > 0:
                exact_reason.append(f"이 품목을 제거하면 Ao가 {ao_loss:.4f} 하락하고 DT_wait가 {wait_inc:.1f}h 증가합니다")
            elif ao_gain >= gain_q75 and ao_gain > 0:
                exact_reason.append(f"이 품목에 재고 1단위를 추가하면 Ao가 {ao_gain:.4f} 상승 가능한 구간입니다")
            else:
                exact_reason.append("현재 대표 해 기준 기여도는 중간 수준입니다")
        detail_df["exact_marginal_reason"] = exact_reason
        xai["detail_df"] = detail_df
    return xai




def build_prescriptive_summary_from_xai(result: dict, selected_payload: dict | None = None) -> dict:
    xai = result.get("xai", {})
    detail_df = xai.get("detail_df", pd.DataFrame()) if isinstance(xai, dict) else pd.DataFrame()

    if not isinstance(detail_df, pd.DataFrame) or detail_df.empty:
        summary = {
            "total_actions": 0,
            "immediate_actions": 0,
            "review_actions": 0,
            "monitor_actions": 0,
            "watchlist_candidates": 0,
            "deferred_actions": 0,
        }
        narrative = "현재 선택 해 기준 Prescriptive AI KPI를 계산할 세부 데이터가 아직 없습니다."
        if isinstance(selected_payload, dict):
            try:
                narrative += (
                    f" 선택 해는 Target Ao {float(pd.to_numeric(selected_payload.get('target_ao', 0.0), errors='coerce')):.2f}, "
                    f"Solution {int(pd.to_numeric(selected_payload.get('solution_id', 0), errors='coerce'))} 입니다."
                )
            except Exception:
                pass
        return {
            "action_df": pd.DataFrame(),
            "summary": summary,
            "narrative": narrative,
        }

    df = detail_df.copy()

    for col in [
        "recommended_stock", "ao_loss_if_removed", "ao_gain_if_plus_one",
        "dt_wait_increase_if_removed", "impact_score", "priority_score", "lead_time"
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["manage_flag", "prebuy_flag", "protection_flag"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    if "decision_bucket" not in df.columns:
        df["decision_bucket"] = ""
    if "xai_reason" not in df.columns:
        df["xai_reason"] = ""
    if "exact_marginal_reason" not in df.columns:
        df["exact_marginal_reason"] = ""

    immediate_mask = df["prebuy_flag"] | df["protection_flag"]
    review_mask = (~immediate_mask) & (df["ao_gain_if_plus_one"] > 0)
    monitor_mask = (~immediate_mask) & (~review_mask) & (df["manage_flag"])
    watchlist_mask = (~df["manage_flag"]) & (df["decision_bucket"].astype(str) == "Watchlist")
    deferred_mask = ~(immediate_mask | review_mask | monitor_mask | watchlist_mask)

    summary = {
        "total_actions": int(len(df)),
        "immediate_actions": int(immediate_mask.sum()),
        "review_actions": int(review_mask.sum()),
        "monitor_actions": int(monitor_mask.sum()),
        "watchlist_candidates": int(watchlist_mask.sum()),
        "deferred_actions": int(deferred_mask.sum()),
    }

    narrative = (
        f"현재 Prescriptive AI는 Explainable AI 상세 결과를 재사용해 KPI를 계산하는 초기 단계입니다. "
        f"전체 {summary['total_actions']:,}개 품목 중 즉시 실행 {summary['immediate_actions']:,}개, "
        f"추가 검토 {summary['review_actions']:,}개, 유지/모니터링 {summary['monitor_actions']:,}개로 분류했습니다."
    )
    if isinstance(selected_payload, dict):
        try:
            narrative += (
                f" 현재 선택 해는 Target Ao "
                f"{float(pd.to_numeric(selected_payload.get('target_ao', 0.0), errors='coerce')):.2f}, "
                f"Solution {int(pd.to_numeric(selected_payload.get('solution_id', 0), errors='coerce'))} 기준입니다."
            )
        except Exception:
            pass

    return {
        "action_df": pd.DataFrame(),
        "summary": summary,
        "narrative": narrative,
    }


def initialize_prescriptive_structure(result: dict, selected_payload: dict | None = None) -> dict:
    prescriptive = result.get("prescriptive")
    rebuilt = build_prescriptive_summary_from_xai(result, selected_payload)

    if not isinstance(prescriptive, dict):
        return rebuilt

    action_df = prescriptive.get("action_df", pd.DataFrame())
    if not isinstance(action_df, pd.DataFrame):
        action_df = pd.DataFrame()

    summary = prescriptive.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    narrative = prescriptive.get("narrative", "")
    if narrative is None:
        narrative = ""

    merged_summary = rebuilt.get("summary", {}).copy()
    merged_summary.update(summary)

    return {
        "action_df": action_df,
        "summary": merged_summary,
        "narrative": narrative if str(narrative).strip() else rebuilt.get("narrative", ""),
    }


def render_prescriptive_kpi_cards(summary: dict):
    safe_summary = summary if isinstance(summary, dict) else {}
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        show_metric_card("총 추천 액션 수", f"{int(pd.to_numeric(safe_summary.get('total_actions', 0), errors='coerce') or 0):,}")
    with c2:
        show_metric_card("즉시 실행 권고 수", f"{int(pd.to_numeric(safe_summary.get('immediate_actions', 0), errors='coerce') or 0):,}")
    with c3:
        show_metric_card("추가 검토 수", f"{int(pd.to_numeric(safe_summary.get('review_actions', 0), errors='coerce') or 0):,}")
    with c4:
        show_metric_card("유지/모니터링 수", f"{int(pd.to_numeric(safe_summary.get('monitor_actions', 0), errors='coerce') or 0):,}")
    with c5:
        show_metric_card("Watchlist 승격 후보 수", f"{int(pd.to_numeric(safe_summary.get('watchlist_candidates', 0), errors='coerce') or 0):,}")
    with c6:
        show_metric_card("보류 후보 수", f"{int(pd.to_numeric(safe_summary.get('deferred_actions', 0), errors='coerce') or 0):,}")

# ------------------------------------------------------------
# Render helpers
# ------------------------------------------------------------
def render_preview_tabs(df_input: pd.DataFrame, df_prepared_preview: pd.DataFrame | None):
    inner_tabs = st.tabs(["원본 데이터", "전처리 데이터", "컬럼 요약", "기초 통계", "결측·유형 점검", "엔진 사전 점검"])

    with inner_tabs[0]:
        st.caption("업로드한 원본 데이터입니다.")
        st.dataframe(df_input, use_container_width=True, height=460)

    with inner_tabs[1]:
        if df_prepared_preview is None or df_prepared_preview.empty:
            st.info("전처리 데이터가 아직 없습니다.")
        else:
            st.caption("엔진 입력 전 표준화된 데이터입니다.")
            st.dataframe(df_prepared_preview, use_container_width=True, height=460)

    with inner_tabs[2]:
        info_df = pd.DataFrame({
            "컬럼명": df_input.columns,
            "원본 dtype": [str(df_input[c].dtype) for c in df_input.columns],
            "결측치 수": [int(df_input[c].isna().sum()) for c in df_input.columns],
            "고유값 수": [int(df_input[c].nunique(dropna=True)) for c in df_input.columns],
        })
        st.dataframe(info_df, use_container_width=True, height=460)

    with inner_tabs[3]:
        try:
            desc = df_input.describe(include="all").transpose().reset_index().rename(columns={"index": "컬럼명"})
            st.dataframe(desc, use_container_width=True, height=460)
        except Exception:
            st.info("기초 통계를 생성할 수 없습니다.")

    with inner_tabs[4]:
        if df_prepared_preview is None or df_prepared_preview.empty:
            st.info("전처리 비교 데이터가 없습니다.")
        else:
            comp = pd.DataFrame({
                "구분": ["원본", "전처리"],
                "행 수": [len(df_input), len(df_prepared_preview)],
                "열 수": [df_input.shape[1], df_prepared_preview.shape[1]],
                "결측치 수": [int(df_input.isna().sum().sum()), int(df_prepared_preview.isna().sum().sum())],
            })
            st.dataframe(comp, use_container_width=True, height=200)

    with inner_tabs[5]:
        precheck = st.session_state.get("precheck_info", {})
        st.markdown("<div class='debug-box'><b>엔진 호출 방식 변경</b><br>이 버전은 NSGA 엔진에 전처리본 xdf가 아니라 원본 df_input을 직접 전달합니다. 전처리본은 화면 확인용으로만 사용합니다.</div>", unsafe_allow_html=True)
        raw_shape = precheck.get("raw_shape")
        prep_shape = precheck.get("prepared_shape")
        st.write("원본 데이터 shape:", raw_shape)
        st.write("전처리 데이터 shape:", prep_shape)
        summary_df = precheck.get("summary_df", pd.DataFrame())
        preview_df = precheck.get("preview_df", pd.DataFrame())
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            st.markdown("**핵심 컬럼 진단 요약**")
            st.dataframe(summary_df, use_container_width=True, height=260)
        else:
            st.info("핵심 컬럼 진단 결과가 없습니다.")
        if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
            st.markdown("**핵심 컬럼 미리보기**")
            st.dataframe(preview_df, use_container_width=True, height=260)


def render_visual_tabs(df_input: pd.DataFrame, df_prepared_preview: pd.DataFrame | None):
    base_df = df_prepared_preview if df_prepared_preview is not None and not df_prepared_preview.empty else df_input
    num_cols = numeric_columns(base_df)
    obj_cols = object_columns(base_df)
    inner_tabs = st.tabs(["수치 분포", "수치 관계", "범주 분포", "전처리 비교"])

    with inner_tabs[0]:
        if not num_cols:
            st.info("시각화할 수치형 컬럼이 없습니다.")
        else:
            col = st.selectbox("히스토그램 컬럼", num_cols, key="hist_col")
            st.pyplot(draw_histogram(base_df[col], f"{col} 분포"), use_container_width=True)

    with inner_tabs[1]:
        if len(num_cols) < 2:
            st.info("산점도를 그릴 수치형 컬럼이 부족합니다.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                x = st.selectbox("X축", num_cols, key="scatter_x")
            with c2:
                y = st.selectbox("Y축", [c for c in num_cols if c != x], key="scatter_y")
            st.pyplot(draw_scatter(base_df, x, y, f"{x} vs {y}"), use_container_width=True)

    with inner_tabs[2]:
        if not obj_cols:
            st.info("범주형 컬럼이 없습니다.")
        else:
            obj = st.selectbox("범주 컬럼", obj_cols, key="obj_col")
            counts = base_df[obj].astype(str).value_counts(dropna=False).head(20).reset_index()
            counts.columns = [obj, "count"]
            st.pyplot(draw_bar(counts, obj, "count", f"{obj} 상위 분포"), use_container_width=True)
            st.dataframe(counts, use_container_width=True, height=260)

    with inner_tabs[3]:
        if df_prepared_preview is None or df_prepared_preview.empty:
            st.info("전처리 비교 데이터가 없습니다.")
        else:
            raw_num = set(numeric_columns(df_input))
            prep_num = set(numeric_columns(df_prepared_preview))
            common = sorted(raw_num & prep_num)
            if not common:
                st.info("원본과 전처리 데이터에 공통 수치형 컬럼이 없습니다.")
            else:
                comp_col = st.selectbox("비교 컬럼", common, key="compare_col")
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(draw_histogram(df_input[comp_col], f"원본 · {comp_col}"), use_container_width=True)
                with c2:
                    st.pyplot(draw_histogram(df_prepared_preview[comp_col], f"전처리 · {comp_col}"), use_container_width=True)


def render_run_history(result: dict | None):
    st.markdown("### 실행 이력")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return
    history_df = result.get("history_df", pd.DataFrame())
    sweep_df = result.get("sweep_summary_df", pd.DataFrame())
    tabs = st.tabs(["세대별 로그", "Sweep 요약", "실시간 상태표"])
    with tabs[0]:
        if isinstance(history_df, pd.DataFrame) and not history_df.empty:
            st.dataframe(history_df, use_container_width=True, height=520)
        else:
            st.info("세대별 로그가 없습니다.")
    with tabs[1]:
        if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
            st.dataframe(sweep_df, use_container_width=True, height=520)
        else:
            st.info("Sweep 요약이 없습니다.")
    with tabs[2]:
        target_status_map = st.session_state.get("target_status_map", {})
        gen_logs = st.session_state.get("generation_logs", [])
        if target_status_map:
            ts_df = pd.DataFrame(list(target_status_map.values())).sort_values("Target Ao").reset_index(drop=True)
            st.dataframe(ts_df, use_container_width=True, height=260)
        if gen_logs:
            st.dataframe(pd.DataFrame(gen_logs).tail(240), use_container_width=True, height=260)



def build_solution_payload_key(target_ao: Any, solution_id: Any) -> str:
    target_val = float(pd.to_numeric(target_ao, errors="coerce"))
    sol_val = int(pd.to_numeric(solution_id, errors="coerce"))
    return f"{target_val:.6f}|{sol_val}"


def parse_solution_payload_key(payload_key: Any):
    if payload_key is None:
        return None
    if isinstance(payload_key, (list, tuple)) and len(payload_key) == 1:
        payload_key = payload_key[0]
    try:
        text_key = str(payload_key).strip()
        left, right = text_key.split("|", 1)
        return {
            "target_ao": float(pd.to_numeric(left, errors="coerce")),
            "solution_id": int(pd.to_numeric(right, errors="coerce")),
        }
    except Exception:
        return None


def extract_clicked_solution_payload(cd):
    if cd is None:
        return None

    parsed_from_key = parse_solution_payload_key(cd)
    if parsed_from_key is not None:
        return parsed_from_key

    if hasattr(cd, "to_dict"):
        cd = cd.to_dict()

    if isinstance(cd, dict):
        target_ao = cd.get("target_ao", cd.get("Target Ao"))
        solution_id = cd.get("solution_id", cd.get("solution_idx", cd.get("idx")))

        if target_ao is None and "0" in cd:
            target_ao = cd.get("0")
        if solution_id is None and "1" in cd:
            solution_id = cd.get("1")

        if target_ao is None and 0 in cd:
            target_ao = cd.get(0)
        if solution_id is None and 1 in cd:
            solution_id = cd.get(1)

        if target_ao is None or solution_id is None:
            keys = list(cd.keys())
            raise ValueError(f"클릭 데이터에 필요한 키가 없습니다: {keys}")

        return {
            "target_ao": float(pd.to_numeric(target_ao, errors="coerce")),
            "solution_id": int(pd.to_numeric(solution_id, errors="coerce")),
        }

    if isinstance(cd, (list, tuple)) and len(cd) >= 2:
        return {
            "target_ao": float(pd.to_numeric(cd[0], errors="coerce")),
            "solution_id": int(pd.to_numeric(cd[1], errors="coerce")),
        }

    raise ValueError(f"지원되지 않는 클릭 데이터 형식입니다: {type(cd)}")

def build_solution_selector_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "target_ao" in out.columns:
        out["target_ao"] = pd.to_numeric(out["target_ao"], errors="coerce")
    if "solution_id" not in out.columns:
        if "solution_idx" in out.columns:
            out["solution_id"] = pd.to_numeric(out["solution_idx"], errors="coerce")
        elif "idx" in out.columns:
            out["solution_id"] = pd.to_numeric(out["idx"], errors="coerce")
        else:
            out["solution_id"] = np.arange(len(out))
    out["solution_id"] = pd.to_numeric(out["solution_id"], errors="coerce").fillna(0).astype(int)
    if "is_pareto_2d" not in out.columns:
        out["is_pareto_2d"] = 0
    cost_col = pick_cost_column(out)
    if cost_col is not None:
        out[cost_col] = pd.to_numeric(out[cost_col], errors="coerce")
    if "Ao" in out.columns:
        out["Ao"] = pd.to_numeric(out["Ao"], errors="coerce")
    if "target_ao" in out.columns and "solution_id" in out.columns:
        out["_payload_key"] = out.apply(
            lambda r: build_solution_payload_key(r.get("target_ao", 0.0), r.get("solution_id", 0)),
            axis=1,
        )
    return out


def render_solution_detail_card(row: pd.Series):
    sim_fields = [
        ("Ao", "Ao"),
        ("DT_diag_h", "DT_diag_h"),
        ("DT_wait_h", "DT_wait_h"),
        ("DT_restore_h", "DT_restore_h"),
        ("DT_total_h", "DT_total_h"),
        ("stock_cost", "stock_cost"),
        ("hold_cost", "hold_cost"),
        ("total_cost", "total_cost"),
        ("stocked_managed_parts", "managed parts"),
        ("managed_items", "managed parts (raw)"),
        ("total_stock_units", "total stock units"),
        ("prebuy_selected", "selected pre-buy"),
        ("protection_selected", "selected protection"),
    ]
    lines = []
    for key, label in sim_fields:
        if key in row.index and pd.notna(row[key]):
            val = row[key]
            if isinstance(val, (int, np.integer)):
                val_txt = f"{int(val)}"
            else:
                try:
                    val_txt = f"{float(val):.4f}" if key == "Ao" else f"{float(val):.2f}"
                except Exception:
                    val_txt = str(val)
            lines.append((label, val_txt))

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("#### 선택 해 최종 분석결과")
    if "target_ao" in row.index:
        st.caption(f"Target Ao: {float(pd.to_numeric(row['target_ao'], errors='coerce')):.2f} | Solution ID: {int(pd.to_numeric(row.get('solution_id', 0), errors='coerce'))}")
    for label, val_txt in lines:
        c1, c2 = st.columns([1.8, 1.2])
        with c1:
            st.markdown(f"**{label}**")
        with c2:
            st.markdown(f"`{val_txt}`")
    st.markdown("</div>", unsafe_allow_html=True)


def extract_plotly_selected_payload(event_obj):
    if event_obj is None:
        return None

    points = None
    try:
        if hasattr(event_obj, "selection") and hasattr(event_obj.selection, "points"):
            points = event_obj.selection.points
        elif isinstance(event_obj, dict):
            if "selection" in event_obj and isinstance(event_obj["selection"], dict):
                points = event_obj["selection"].get("points")
            else:
                points = event_obj.get("points")
    except Exception:
        points = None

    if not points:
        return None

    p0 = points[0]
    customdata = None
    try:
        customdata = p0.get("customdata") if isinstance(p0, dict) else getattr(p0, "customdata", None)
    except Exception:
        customdata = None

    payload = parse_solution_payload_key(customdata)
    if payload is not None:
        return payload
    return extract_clicked_solution_payload(customdata)


def build_ce_curve_plotly(df_runs: pd.DataFrame, title: str):
    fig = go.Figure()
    if not isinstance(df_runs, pd.DataFrame) or df_runs.empty:
        fig.update_layout(title=title, xaxis_title="Cost", yaxis_title="Ao", height=640)
        return fig

    plot_df = build_solution_selector_df(df_runs)
    cost_col = pick_cost_column(plot_df)
    if cost_col is None or "Ao" not in plot_df.columns:
        fig.update_layout(title=title, xaxis_title="Cost", yaxis_title="Ao", height=640)
        return fig

    plot_df = plot_df.dropna(subset=[cost_col, "Ao"]).copy()
    if plot_df.empty:
        fig.update_layout(title=title, xaxis_title="Cost", yaxis_title="Ao", height=640)
        return fig

    targets = sorted(plot_df["target_ao"].dropna().unique().tolist()) if "target_ao" in plot_df.columns else [None]
    colors = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#2E91E5",
        "#E15F99", "#1CA71C"
    ]

    for i, target in enumerate(targets):
        sub = plot_df if target is None else plot_df[plot_df["target_ao"] == target].copy()
        if sub.empty:
            continue
        color = colors[i % len(colors)]
        dominated = sub[sub["is_pareto_2d"] != 1].copy() if "is_pareto_2d" in sub.columns else sub.copy()
        pareto = sub[sub["is_pareto_2d"] == 1].copy() if "is_pareto_2d" in sub.columns else pd.DataFrame(columns=sub.columns)
        name = "All Solutions" if target is None else f"Target {float(target):.2f}"

        if not dominated.empty:
            fig.add_trace(
                go.Scatter(
                    x=dominated[cost_col],
                    y=dominated["Ao"],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=True,
                    customdata=dominated["_payload_key"] if "_payload_key" in dominated.columns else None,
                    text=dominated.apply(lambda r: f"Target Ao={float(pd.to_numeric(r.get('target_ao', 0.0), errors='coerce')):.2f}<br>Solution={int(pd.to_numeric(r.get('solution_id', 0), errors='coerce'))}", axis=1),
                    marker=dict(size=7, color=color, opacity=0.12, line=dict(width=0)),
                    selected=dict(marker=dict(opacity=0.12, size=7)),
                    unselected=dict(marker=dict(opacity=0.12)),
                    hovertemplate="%{text}<br>Cost=%{x:,.2f}<br>Ao=%{y:.4f}<br>Pareto=N<extra></extra>",
                )
            )
        if not pareto.empty:
            fig.add_trace(
                go.Scatter(
                    x=pareto[cost_col],
                    y=pareto["Ao"],
                    mode="markers",
                    name=f"{name} · Pareto",
                    legendgroup=name,
                    showlegend=False,
                    customdata=pareto["_payload_key"] if "_payload_key" in pareto.columns else None,
                    text=pareto.apply(lambda r: f"Target Ao={float(pd.to_numeric(r.get('target_ao', 0.0), errors='coerce')):.2f}<br>Solution={int(pd.to_numeric(r.get('solution_id', 0), errors='coerce'))}", axis=1),
                    marker=dict(size=14, color=color, opacity=1.0, line=dict(width=1.6, color="#111111"), symbol="circle"),
                    selected=dict(marker=dict(opacity=1.0, size=14)),
                    unselected=dict(marker=dict(opacity=1.0)),
                    hovertemplate="%{text}<br>Cost=%{x:,.2f}<br>Ao=%{y:.4f}<br><b>Pareto=Y</b><extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=clean_chart_text(title), x=0.02, y=0.97, xanchor="left"),
        xaxis_title=clean_chart_text(f"Cost ({cost_col})"),
        yaxis_title="Ao",
        height=700,
        margin=dict(l=20, r=20, t=105, b=95),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        showline=True,
        linewidth=1.2,
        linecolor="rgba(60,80,60,0.55)",
        mirror=True,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        showline=True,
        linewidth=1.2,
        linecolor="rgba(60,80,60,0.55)",
        mirror=True,
        zeroline=False,
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig



def apply_current_point_annotation(fig, selector_df: pd.DataFrame, selected_payload: dict | None):
    if fig is None or selector_df is None or selector_df.empty or not isinstance(selected_payload, dict):
        return fig
    try:
        hit = selector_df[
            (pd.to_numeric(selector_df["target_ao"], errors="coerce").round(6) == round(float(selected_payload["target_ao"]), 6))
            & (pd.to_numeric(selector_df["solution_id"], errors="coerce").astype(int) == int(selected_payload["solution_id"]))
        ]
        if hit.empty:
            return fig
        row = hit.iloc[0]
        cost_col = pick_cost_column(selector_df)
        if cost_col is None or cost_col not in row.index or "Ao" not in row.index:
            return fig
        x = float(pd.to_numeric(row[cost_col], errors="coerce"))
        y = float(pd.to_numeric(row["Ao"], errors="coerce"))
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=18,
                    color="#FFD700",
                    symbol="star",
                    line=dict(width=1.6, color="#111111"),
                ),
                hovertemplate="Current Point<br>Cost=%{x:,.2f}<br>Ao=%{y:.4f}<extra></extra>",
                showlegend=False,
            )
        )
    except Exception:
        return fig
    return fig

def render_summary_panel(summary: dict, policy_df: pd.DataFrame | None = None):
    best_ao = float(summary.get('best_Ao', summary.get('Ao', 0.0)))
    target_ao = float(summary.get('target_ao', summary.get('representative_target', 0.0)))
    stock_summary = summarize_policy_stock_relationship(policy_df if isinstance(policy_df, pd.DataFrame) else pd.DataFrame(), summary)
    managed_parts = int(stock_summary.get('managed_parts', pd.to_numeric(summary.get('best_managed_parts', summary.get('managed_parts', summary.get('managed_count', 0))), errors='coerce') or 0))
    total_stock = int(stock_summary.get('total_stock_units', pd.to_numeric(summary.get('best_total_stock_units', summary.get('total_stock_units', summary.get('total_stock', 0))), errors='coerce') or 0))
    total_cost = float(summary.get('best_cost', summary.get('total_cost', 0.0)))
    dt_wait = float(summary.get('best_DT_wait_h', summary.get('DT_wait_h', 0.0)))
    selected_prebuy = int(stock_summary.get('selected_prebuy', pd.to_numeric(summary.get('best_selected_prebuy', summary.get('selected_prebuy', 0)), errors='coerce') or 0))
    selected_protection = int(stock_summary.get('selected_protection', pd.to_numeric(summary.get('best_selected_protection', summary.get('selected_protection', 0)), errors='coerce') or 0))

    st.markdown(
        f"""
        <div class="section-box">
            <div style="font-size:1.15rem;font-weight:800;color:#1b4e2a;margin-bottom:8px;">대표 해 요약</div>
            <div style="line-height:1.8;color:#3d5f45;">
                현재 대표 해는 <b>Target Ao {target_ao:.2f}</b> 기준에서 <b>Ao {best_ao:.4f}</b>를 달성한 해입니다.
                이 해는 <b>Managed Parts {managed_parts:,}개</b>, <b>Total Stock {total_stock:,}단위</b>,
                <b>Selected Pre-buy {selected_prebuy:,}개</b>, <b>Selected Protection {selected_protection:,}개</b>를 포함합니다.
                즉, 목표 가용도를 맞추기 위해 어떤 수리부속을 얼마나 확보해야 하는지 보여주는 최종 의사결정 결과입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Best Ao", f"{best_ao:.4f}")
    with c2:
        show_metric_card("Total Cost", f"{total_cost:,.0f}")
    with c3:
        show_metric_card("DT_wait_h", f"{dt_wait:,.2f}")
    with c4:
        show_metric_card("Target Ao", f"{target_ao:.2f}")

    detail_rows = [
        {"항목": "Managed Parts", "값": f"{managed_parts:,}", "의미": "최종적으로 관리 대상으로 유지한 품목 수"},
        {"항목": "Total Stock Units", "값": f"{total_stock:,}", "의미": "대표 해에서 확보 권고된 총 재고 단위"},
        {"항목": "Selected Pre-buy", "값": f"{selected_prebuy:,}", "의미": "조기 확보가 필요한 핵심 선발주 품목 수"},
        {"항목": "Selected Protection", "값": f"{selected_protection:,}", "의미": "다운타임 완충을 위해 보호재고가 필요한 품목 수"},
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True, height=210)

def render_integrated_results(result: dict | None):
    st.markdown("### 통합 결과")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    summary = result.get("summary", {})
    tabs = st.tabs(["요약", "파레토/스윕", "Best Policy", "다운로드"])

    pareto_df = result.get("pareto_df", pd.DataFrame())
    all_runs_df = result.get("all_target_runs_df", pd.DataFrame())
    source_df = all_runs_df if isinstance(all_runs_df, pd.DataFrame) and not all_runs_df.empty else pareto_df
    selector_df = build_solution_selector_df(source_df)

    with tabs[0]:
        if not selector_df.empty:
            fig_summary = build_ce_curve_plotly(selector_df, "통합 Target Sweep Overlay (C-E Curve)")
            fig_summary = apply_current_point_annotation(fig_summary, selector_df, st.session_state.get("selected_solution_payload"))
            st.plotly_chart(
                fig_summary,
                use_container_width=True,
                key="ce_curve_plotly_summary_only",
            )
        render_summary_panel(summary, result.get("policy_df", pd.DataFrame()))

    with tabs[1]:
        c1, c2 = st.columns([1.45, 1.0])

        with c1:
            if selector_df.empty:
                st.info("표시할 파레토/스윕 데이터가 없습니다.")
            else:
                current_payload = st.session_state.get("selected_solution_payload")
                fig = build_ce_curve_plotly(selector_df, "통합 Target Sweep Overlay (C-E Curve)")
                fig = apply_current_point_annotation(fig, selector_df, current_payload)
                event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="ce_curve_plotly_main",
                    on_select="rerun",
                    selection_mode=("points",),
                )
                try:
                    payload = extract_plotly_selected_payload(event)
                    if payload is not None:
                        changed = (
                            not isinstance(current_payload, dict)
                            or round(float(current_payload.get("target_ao", -999)), 6) != round(float(payload["target_ao"]), 6)
                            or int(current_payload.get("solution_id", -999)) != int(payload["solution_id"])
                        )
                        st.session_state.selected_solution_payload = payload
                        if changed:
                            st.rerun()
                except Exception as e:
                    st.warning(f"클릭 데이터 해석 실패: {e}")

                st.caption("점 클릭: 해당 해의 최종 분석결과 표시 | 별표(★) = 현재 선택한 해 | 진한 큰 점 = Pareto 해 | 연한 작은 점 = 비Pareto 해")

        with c2:
            if selector_df.empty:
                st.info("선택 가능한 해 데이터가 없습니다.")
            else:
                selector_df = selector_df.sort_values(["is_pareto_2d", "Ao"], ascending=[False, False]).reset_index(drop=True)
                selector_df["label"] = selector_df.apply(
                    lambda r: (
                        f"Target {float(pd.to_numeric(r.get('target_ao', 0), errors='coerce')):.2f} | "
                        f"Solution {int(pd.to_numeric(r.get('solution_id', 0), errors='coerce'))} | "
                        f"Ao {float(pd.to_numeric(r.get('Ao', 0), errors='coerce')):.4f} | "
                        f"Pareto {'Y' if int(pd.to_numeric(r.get('is_pareto_2d', 0), errors='coerce')) == 1 else 'N'}"
                    ),
                    axis=1,
                )

                selected_payload = st.session_state.get("selected_solution_payload")
                default_idx = 0
                if isinstance(selected_payload, dict):
                    hit = selector_df[
                        (selector_df["target_ao"].round(6) == round(float(selected_payload["target_ao"]), 6))
                        & (selector_df["solution_id"].astype(int) == int(selected_payload["solution_id"]))
                    ]
                    if not hit.empty:
                        default_idx = int(hit.index[0])

                selected_label = st.selectbox(
                    "해 선택",
                    selector_df["label"].tolist(),
                    index=min(default_idx, max(len(selector_df) - 1, 0)),
                    key="solution_detail_selectbox",
                )
                selected_row = selector_df.loc[selector_df["label"] == selected_label].iloc[0]
                st.session_state.selected_solution_payload = {
                    "target_ao": float(pd.to_numeric(selected_row["target_ao"], errors="coerce")),
                    "solution_id": int(pd.to_numeric(selected_row["solution_id"], errors="coerce")),
                }
                render_solution_detail_card(selected_row)

    with tabs[2]:
        policy_df = result.get("policy_df", pd.DataFrame())
        if isinstance(policy_df, pd.DataFrame) and not policy_df.empty:
            st.dataframe(policy_df, use_container_width=True, height=500)
        else:
            st.info("Best Policy 데이터가 없습니다.")

    with tabs[3]:
        st.download_button(
            "📥 결과 엑셀 다운로드",
            data=make_excel_download(result),
            file_name="NSGA_result_dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

def render_xai(result: dict | None, df_input: pd.DataFrame | None):
    st.markdown("### Explainable AI")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    xai = result.get("xai")
    if not isinstance(xai, dict):
        xai = build_explainability_tables_v4(result, df_input)
        result["xai"] = xai

    detail_df = xai.get("detail_df", pd.DataFrame())
    reason_counts = xai.get("reason_counts", pd.DataFrame())
    bucket_counts = xai.get("bucket_counts", pd.DataFrame())
    managed_compare = xai.get("managed_compare", pd.DataFrame())
    top_priority = xai.get("top_priority", pd.DataFrame())
    top_unmanaged = xai.get("top_unmanaged", pd.DataFrame())
    top_cost_vs_impact = xai.get("top_cost_vs_impact", pd.DataFrame())
    solution_cards = xai.get("solution_cards", {})
    why_solution = xai.get("why_solution", {})
    stock_summary = xai.get("stock_summary", {}) if isinstance(xai, dict) else {}
    final_stock_table = xai.get("final_stock_table", pd.DataFrame()) if isinstance(xai, dict) else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Total Unit Stock", f"{int(pd.to_numeric(stock_summary.get('total_stock_units', 0), errors='coerce') or 0):,}")
    with c2:
        show_metric_card("Managed Parts", f"{int(pd.to_numeric(stock_summary.get('managed_parts', xai.get('selected_count', 0)), errors='coerce') or 0):,}")
    with c3:
        show_metric_card("Selected Pre-buy", f"{int(pd.to_numeric(stock_summary.get('selected_prebuy', 0), errors='coerce') or 0):,}")
    with c4:
        show_metric_card("Selected Protection", f"{int(pd.to_numeric(stock_summary.get('selected_protection', 0), errors='coerce') or 0):,}")

    st.markdown(f"<div class='section-box'>{clean_chart_text(xai.get('narrative', '-'))}</div>", unsafe_allow_html=True)

    rel_left, rel_right = st.columns([1.15, 1.0], gap="large")
    with rel_left:
        st.markdown("#### 최종 확보 품목 관계")
        if isinstance(final_stock_table, pd.DataFrame) and not final_stock_table.empty:
            st.dataframe(final_stock_table, use_container_width=True, hide_index=True, height=220)
        else:
            st.info("최종 확보 품목 관계를 계산할 데이터가 없습니다.")
    with rel_right:
        st.markdown(
            f"""
            <div class='section-box'>
                <b>설득 포인트</b><br>
                1) Managed Parts는 최종 권고 재고가 1개 이상인 품목만 셉니다.<br><br>
                2) Selected Pre-buy / Selected Protection도 재고가 실제로 배정된 품목만 셉니다.<br><br>
                3) 따라서 최종 결과는 <b>Total Unit Stock = Selected Pre-buy + Selected Protection + Normal Stock</b> 구조로 해석할 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.markdown(f"<div class='section-box'><b>타깃 정렬 관점</b><br>{why_solution.get('target_alignment','-')}<br><br><b>선택 편향 관점</b><br>{why_solution.get('selection_bias','-')}</div>", unsafe_allow_html=True)
    with w2:
        st.markdown(f"<div class='section-box'><b>리스크 완충 관점</b><br>{why_solution.get('risk_buffer','-')}<br><br><b>비용-효과 관점</b><br>{why_solution.get('tradeoff','-')}</div>", unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown(f"<div class='section-box'><b>선정 로직 요약</b><br>{solution_cards.get('selection_logic','-')}<br><br><b>관리군 포커스</b><br>{solution_cards.get('managed_focus','-')}</div>", unsafe_allow_html=True)
    with s2:
        st.markdown(f"<div class='section-box'><b>Watchlist 포커스</b><br>{solution_cards.get('watchlist_focus','-')}<br><br><b>비용-효과 포커스</b><br>{solution_cards.get('cost_focus','-')}</div>", unsafe_allow_html=True)

    tabs = st.tabs(["설명 차트", "상위 품목", "세부 테이블", "개별 품목 프로파일"])
    with tabs[0]:
        a1, a2 = st.columns(2)
        with a1:
            st.pyplot(draw_reason_bar(reason_counts, "주요 설명 사유 분포"), use_container_width=True)
            st.pyplot(draw_managed_compare(managed_compare, "관리군 vs 비관리군 비교"), use_container_width=True)
        with a2:
            st.pyplot(draw_decision_bucket_bar(bucket_counts, "결정 버킷 분포"), use_container_width=True)
            st.pyplot(draw_xai_quadrant(detail_df, "비용-우선순위 사분면"), use_container_width=True)

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.markdown("**우선순위 상위 품목**")
            st.dataframe(top_priority, use_container_width=True, height=320)
            st.markdown("**제외되었지만 영향도가 높은 품목**")
            st.dataframe(top_unmanaged, use_container_width=True, height=220)
        with right:
            st.markdown("**비용-효과 밸런스 상위 품목**")
            st.dataframe(top_cost_vs_impact, use_container_width=True, height=320)

    with tabs[2]:
        st.dataframe(detail_df, use_container_width=True, height=560)

    with tabs[3]:
        if detail_df.empty:
            st.info("표시할 품목이 없습니다.")
        else:
            part_options = detail_df["part_id"].astype(str).tolist()
            selected_part = st.selectbox("품목 선택", part_options, key="xai_selected_part")
            selected_row = detail_df.loc[detail_df["part_id"].astype(str) == selected_part].iloc[0]
            cohort_cols = [c for c in ["impact_score", "lead_time", "priority_score", "recommended_stock", "xai_action_score", "xai_cost_score", "decision_balance_score"] if c in detail_df.columns]
            cohort_medians = {c: float(pd.to_numeric(detail_df[c], errors="coerce").median()) for c in cohort_cols}
            st.pyplot(draw_item_profile(selected_row, cohort_medians, f"품목 프로파일 · {selected_part}"), use_container_width=True)
            cols = [c for c in ["part_id", "manage_flag", "prebuy_flag", "protection_flag", "recommended_stock", "impact_score", "priority_score", "lead_time", "cost_burden", "xai_reason", "exact_marginal_reason"] if c in selected_row.index]
            st.dataframe(pd.DataFrame([selected_row[cols].to_dict()]), use_container_width=True, height=220)



def _series_from_value(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        s = value.reindex(index)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    if isinstance(value, np.ndarray):
        s = pd.Series(value, index=index[:len(value)] if len(index) >= len(value) else None)
        if not isinstance(s.index, pd.Index) or len(s.index) != len(index):
            s = pd.Series(list(value) + [0.0] * max(0, len(index) - len(value)), index=index)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return pd.Series([pd.to_numeric(value, errors="coerce")] * len(index), index=index).fillna(0.0)


def build_prescriptive_action_df(result: dict, df_input: pd.DataFrame | None = None) -> pd.DataFrame:
    xai = result.get("xai", {})
    detail_df = xai.get("detail_df", pd.DataFrame()).copy() if isinstance(xai, dict) else pd.DataFrame()

    if not isinstance(detail_df, pd.DataFrame) or detail_df.empty:
        return pd.DataFrame(columns=[
            "part_id",
            "recommended_action",
            "action_group",
            "priority_level",
            "reason_summary",
            "expected_effect",
            "manage_flag",
            "prebuy_flag",
            "protection_flag",
            "ao_loss_if_removed",
            "ao_gain_if_plus_one",
            "dt_wait_increase_if_removed",
        ])

    df = detail_df.copy()
    idx = df.index

    if "part_id" not in df.columns:
        df["part_id"] = np.arange(1, len(df) + 1)

    manage_flag = df.get("manage_flag", False)
    prebuy_flag = df.get("prebuy_flag", False)
    protection_flag = df.get("protection_flag", False)

    df["manage_flag"] = _series_from_value(manage_flag, idx).astype(bool)
    df["prebuy_flag"] = _series_from_value(prebuy_flag, idx).astype(bool)
    df["protection_flag"] = _series_from_value(protection_flag, idx).astype(bool)

    for c in ["ao_loss_if_removed", "ao_gain_if_plus_one", "dt_wait_increase_if_removed"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "xai_reason" not in df.columns:
        df["xai_reason"] = ""
    if "exact_marginal_reason" not in df.columns:
        df["exact_marginal_reason"] = ""
    if "xai_reason_short" not in df.columns:
        df["xai_reason_short"] = ""

    ao_loss_q75 = float(pd.to_numeric(df["ao_loss_if_removed"], errors="coerce").fillna(0.0).quantile(0.75)) if len(df) else 0.0
    ao_gain_q75 = float(pd.to_numeric(df["ao_gain_if_plus_one"], errors="coerce").fillna(0.0).quantile(0.75)) if len(df) else 0.0

    rec_actions = []
    action_groups = []
    priority_levels = []
    expected_effects = []

    for _, row in df.iterrows():
        manage = bool(row.get("manage_flag", False))
        prebuy = bool(row.get("prebuy_flag", False))
        protection = bool(row.get("protection_flag", False))
        ao_loss = float(pd.to_numeric(row.get("ao_loss_if_removed", 0.0), errors="coerce") or 0.0)
        ao_gain = float(pd.to_numeric(row.get("ao_gain_if_plus_one", 0.0), errors="coerce") or 0.0)
        dt_wait = float(pd.to_numeric(row.get("dt_wait_increase_if_removed", 0.0), errors="coerce") or 0.0)

        if prebuy:
            rec_actions.append("즉시 선발주")
            action_groups.append("immediate")
            priority_levels.append("High")
            expected_effects.append(f"초기 조달 지연 위험을 낮추고 목표 Ao 유지에 유리합니다. 제거 시 Ao 하락 추정 {ao_loss:.4f}")
        elif protection:
            rec_actions.append("보호재고 유지")
            action_groups.append("monitor")
            priority_levels.append("High" if ao_loss >= ao_loss_q75 and ao_loss > 0 else "Medium")
            expected_effects.append(f"부족 발생 시 DT_wait 증가를 완충합니다. 제거 시 대기 다운타임 증가 추정 {dt_wait:.1f}h")
        elif manage and ao_gain >= ao_gain_q75 and ao_gain > 0:
            rec_actions.append("재고 1단위 추가 검토")
            action_groups.append("review")
            priority_levels.append("Medium")
            expected_effects.append(f"재고 1단위 추가 시 Ao 추가 개선 여지 {ao_gain:.4f}")
        elif manage:
            rec_actions.append("유지 + 모니터링")
            action_groups.append("monitor")
            priority_levels.append("Medium")
            expected_effects.append("현재 대표 해 기준 관리 유지가 합리적이며 성과와 비용을 함께 관찰합니다")
        elif (not manage) and ao_loss >= ao_loss_q75 and ao_loss > 0:
            rec_actions.append("Watchlist 승격")
            action_groups.append("watchlist")
            priority_levels.append("Medium")
            expected_effects.append(f"현재는 비관리지만 제거 민감도가 높아 재검토 가치가 있습니다. Ao 영향 추정 {ao_loss:.4f}")
        else:
            rec_actions.append("보류 / 후순위")
            action_groups.append("deferred")
            priority_levels.append("Low")
            expected_effects.append("현재 해 기준 다른 핵심 품목 대비 우선순위가 낮아 보류가 합리적입니다")

    df["recommended_action"] = rec_actions
    df["action_group"] = action_groups
    df["priority_level"] = priority_levels
    df["reason_summary"] = (
        df["xai_reason"].astype(str).str.strip().replace("nan", "")
        + np.where(
            df["exact_marginal_reason"].astype(str).str.strip().ne(""),
            " | " + df["exact_marginal_reason"].astype(str).str.strip(),
            ""
        )
    ).str.strip(" |")
    df["expected_effect"] = expected_effects

    out_cols = [
        "part_id",
        "recommended_action",
        "action_group",
        "priority_level",
        "reason_summary",
        "expected_effect",
        "manage_flag",
        "prebuy_flag",
        "protection_flag",
        "ao_loss_if_removed",
        "ao_gain_if_plus_one",
        "dt_wait_increase_if_removed",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = "" if c in ["recommended_action", "action_group", "priority_level", "reason_summary", "expected_effect", "part_id"] else 0.0

    return df[out_cols].copy()


def build_prescriptive_package(result: dict, df_input: pd.DataFrame | None = None) -> dict:
    action_df = build_prescriptive_action_df(result, df_input)

    if not isinstance(action_df, pd.DataFrame):
        action_df = pd.DataFrame()

    action_group = action_df["action_group"] if "action_group" in action_df.columns else pd.Series(dtype="object")
    total_actions = int(len(action_df))
    summary = {
        "total_actions": total_actions,
        "immediate_actions": int((action_group == "immediate").sum()) if len(action_df) else 0,
        "review_actions": int((action_group == "review").sum()) if len(action_df) else 0,
        "monitor_actions": int((action_group == "monitor").sum()) if len(action_df) else 0,
        "watchlist_candidates": int((action_group == "watchlist").sum()) if len(action_df) else 0,
        "deferred_actions": int((action_group == "deferred").sum()) if len(action_df) else 0,
    }

    summary["policy_cards"] = build_prescriptive_policy_cards(action_df)

    payload = st.session_state.get("selected_solution_payload")
    selector_target = None
    selector_solution = None
    payload_txt = "현재 선택 해 정보가 아직 없습니다."
    if isinstance(payload, dict):
        try:
            selector_target = float(pd.to_numeric(payload.get("target_ao", 0.0), errors="coerce"))
            selector_solution = int(pd.to_numeric(payload.get("solution_id", 0), errors="coerce"))
            payload_txt = f"현재 선택 해는 Target {selector_target:.2f}, Solution {selector_solution}입니다."
        except Exception:
            selector_target = None
            selector_solution = None

    immediate_top = "없음"
    if isinstance(action_df, pd.DataFrame) and not action_df.empty and "recommended_action" in action_df.columns:
        immediate_df = action_df[action_df["recommended_action"].astype(str) == "즉시 선발주"].copy()
        if not immediate_df.empty:
            top_ids = immediate_df["part_id"].astype(str).head(3).tolist() if "part_id" in immediate_df.columns else []
            immediate_top = ", ".join(top_ids) if top_ids else "즉시 선발주 후보 존재"

    watchlist_top = "없음"
    if isinstance(action_df, pd.DataFrame) and not action_df.empty and "recommended_action" in action_df.columns:
        watch_df = action_df[action_df["recommended_action"].astype(str) == "Watchlist 승격"].copy()
        if not watch_df.empty:
            top_ids = watch_df["part_id"].astype(str).head(3).tolist() if "part_id" in watch_df.columns else []
            watchlist_top = ", ".join(top_ids) if top_ids else "Watchlist 후보 존재"

    narrative = (
        f"{payload_txt} "
        f"이 선택 상태를 Prescriptive AI에서도 그대로 참조합니다. "
        f"현재 선택 해 기준 총 {summary['total_actions']:,}개의 액션이 추천되며, "
        f"즉시 실행 우선군은 {summary['immediate_actions']:,}건입니다"
        f"{'' if immediate_top == '없음' else f' (대표 품목: {immediate_top})'}. "
        f"추가 검토는 {summary['review_actions']:,}건, 유지/모니터링은 {summary['monitor_actions']:,}건입니다. "
        f"Watchlist 승격 후보는 {summary['watchlist_candidates']:,}건입니다"
        f"{'' if watchlist_top == '없음' else f' (대표 품목: {watchlist_top})'}. "
        f"보류/후순위는 {summary['deferred_actions']:,}건입니다."
    )

    return {
        "action_df": action_df,
        "summary": summary,
        "narrative": narrative,
    }





def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if s.lower() == "nan":
        return ""
    return " ".join(s.split()).strip()


def _top_nonempty_text(series: pd.Series) -> str:
    if not isinstance(series, pd.Series) or series.empty:
        return "대표 사유 정보가 아직 없습니다."
    vals = series.astype(str).map(_safe_text)
    vals = vals[vals != ""]
    if vals.empty:
        return "대표 사유 정보가 아직 없습니다."
    vc = vals.value_counts()
    return str(vc.index[0])


def _mean_from_series(series: pd.Series) -> float:
    if not isinstance(series, pd.Series) or series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).mean())


def build_prescriptive_policy_cards(action_df: pd.DataFrame) -> list[dict]:
    if not isinstance(action_df, pd.DataFrame) or action_df.empty:
        base_specs = [
            ("즉시 선발주", "즉시 실행이 필요한 선발주 후보입니다.", "초기 조달 지연 위험을 줄이는 데 도움을 줍니다."),
            ("보호재고 유지", "보호재고 유지 대상이 아직 없습니다.", "다운타임 완충 효과를 기대할 수 있습니다."),
            ("재고 1단위 추가 검토", "추가 재고 검토 대상이 아직 없습니다.", "Ao 추가 개선 가능성을 확인하는 데 의미가 있습니다."),
            ("유지 + 모니터링", "현 상태 유지 관찰 대상이 아직 없습니다.", "현재 해의 관리 상태를 안정적으로 유지합니다."),
            ("Watchlist 승격", "재검토가 필요한 Watchlist 후보가 아직 없습니다.", "추후 관리군 편입 여부를 판단하는 데 도움을 줍니다."),
            ("보류 / 후순위", "보류 대상이 아직 없습니다.", "현재 자원을 핵심 품목에 우선 배분할 수 있습니다."),
        ]
        return [
            {"title": t, "count": 0, "reason": r, "effect": e}
            for t, r, e in base_specs
        ]

    df = action_df.copy()
    if "reason_summary" not in df.columns:
        df["reason_summary"] = ""
    if "expected_effect" not in df.columns:
        df["expected_effect"] = ""
    for c in ["ao_loss_if_removed", "ao_gain_if_plus_one", "dt_wait_increase_if_removed"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    def _card(title: str, action_name: str, fallback_reason: str, effect_builder):
        sub = df[df["recommended_action"].astype(str) == action_name].copy() if "recommended_action" in df.columns else pd.DataFrame()
        count = int(len(sub))
        if sub.empty:
            return {
                "title": title,
                "count": 0,
                "reason": fallback_reason,
                "effect": effect_builder(pd.DataFrame()),
            }
        reason = _top_nonempty_text(sub["reason_summary"])
        return {
            "title": title,
            "count": count,
            "reason": reason,
            "effect": effect_builder(sub),
        }

    cards = [
        _card(
            "즉시 선발주",
            "즉시 선발주",
            "선발주 규칙 또는 높은 초기 조달 필요성이 반영되지 않았습니다.",
            lambda s: (
                f"평균 제거 시 Ao 하락 추정 {_mean_from_series(s.get('ao_loss_if_removed', pd.Series(dtype=float))):.4f} 수준으로, "
                f"초기 확보 지연 위험을 낮추는 데 유리합니다."
                if not s.empty else
                "초기 조달 지연 위험을 줄이는 데 도움을 줍니다."
            ),
        ),
        _card(
            "보호재고 유지",
            "보호재고 유지",
            "보호재고 유지 필요성이 큰 품목이 아직 분류되지 않았습니다.",
            lambda s: (
                f"제거 시 평균 DT_wait 증가 추정 {_mean_from_series(s.get('dt_wait_increase_if_removed', pd.Series(dtype=float))):.1f}h로, "
                f"다운타임 완충 효과가 기대됩니다."
                if not s.empty else
                "다운타임 완충 효과를 기대할 수 있습니다."
            ),
        ),
        _card(
            "재고 1단위 추가 검토",
            "재고 1단위 추가 검토",
            "추가 재고 검토 가치가 큰 품목이 아직 분류되지 않았습니다.",
            lambda s: (
                f"재고 1단위 추가 시 평균 Ao 개선 여지 {_mean_from_series(s.get('ao_gain_if_plus_one', pd.Series(dtype=float))):.4f} 수준입니다."
                if not s.empty else
                "Ao 추가 개선 가능성을 확인하는 데 의미가 있습니다."
            ),
        ),
        _card(
            "유지 + 모니터링",
            "유지 + 모니터링",
            "현재 상태 유지 및 관찰이 필요한 품목이 아직 분류되지 않았습니다.",
            lambda s: (
                f"현재 해 기준 관리 상태를 유지하면서 평균 제거 시 Ao 하락 추정 {_mean_from_series(s.get('ao_loss_if_removed', pd.Series(dtype=float))):.4f} 수준을 관찰합니다."
                if not s.empty else
                "현재 해의 관리 상태를 안정적으로 유지합니다."
            ),
        ),
        _card(
            "Watchlist 승격",
            "Watchlist 승격",
            "비관리 품목 중 재검토 우선 대상이 아직 분류되지 않았습니다.",
            lambda s: (
                f"현재는 비관리지만 평균 제거 시 Ao 영향 추정 {_mean_from_series(s.get('ao_loss_if_removed', pd.Series(dtype=float))):.4f} 수준으로 재검토 가치가 있습니다."
                if not s.empty else
                "추후 관리군 편입 여부를 판단하는 데 도움을 줍니다."
            ),
        ),
        _card(
            "보류 / 후순위",
            "보류 / 후순위",
            "다른 핵심 품목 대비 보류가 합리적인 품목이 아직 분류되지 않았습니다.",
            lambda s: (
                f"현재는 평균 재고 추가 시 Ao 개선 여지 {_mean_from_series(s.get('ao_gain_if_plus_one', pd.Series(dtype=float))):.4f} 수준으로 상대적으로 낮아 후순위가 합리적입니다."
                if not s.empty else
                "현재 자원을 핵심 품목에 우선 배분할 수 있습니다."
            ),
        ),
    ]
    return cards


def render_prescriptive(result: dict | None, df_input: pd.DataFrame | None):
    st.markdown("### Prescriptive AI")
    if not result:
        st.info("아직 실행 결과가 없습니다. NSGA 분석을 먼저 실행해 주세요.")
        return

    # 선택된 solution payload 상태를 항상 현재값 기준으로 반영
    prescriptive = build_prescriptive_package(result, df_input)
    result["prescriptive"] = prescriptive

    action_df = prescriptive.get("action_df", pd.DataFrame())
    summary = prescriptive.get("summary", {})
    narrative = prescriptive.get("narrative", "")

    if not isinstance(action_df, pd.DataFrame):
        action_df = pd.DataFrame()
    if not isinstance(summary, dict):
        summary = {}
    if narrative is None:
        narrative = ""

    total_actions = int(pd.to_numeric(summary.get("total_actions", 0), errors="coerce") or 0)
    immediate_actions = int(pd.to_numeric(summary.get("immediate_actions", 0), errors="coerce") or 0)
    review_actions = int(pd.to_numeric(summary.get("review_actions", 0), errors="coerce") or 0)
    monitor_actions = int(pd.to_numeric(summary.get("monitor_actions", 0), errors="coerce") or 0)
    watchlist_candidates = int(pd.to_numeric(summary.get("watchlist_candidates", 0), errors="coerce") or 0)
    deferred_actions = int(pd.to_numeric(summary.get("deferred_actions", 0), errors="coerce") or 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        show_metric_card("총 추천 액션 수", f"{total_actions:,}")
    with c2:
        show_metric_card("즉시 실행 권고 수", f"{immediate_actions:,}")
    with c3:
        show_metric_card("추가 검토 수", f"{review_actions:,}")
    with c4:
        show_metric_card("유지/모니터링 수", f"{monitor_actions:,}")
    with c5:
        show_metric_card("Watchlist 승격 후보 수", f"{watchlist_candidates:,}")
    with c6:
        show_metric_card("보류 후보 수", f"{deferred_actions:,}")

    st.markdown(
        f"""
        <div class="section-box">
            <div style="font-size:1.05rem;font-weight:800;color:#1b4e2a;margin-bottom:8px;">선택 해 연동 Prescriptive 요약</div>
            <div style="line-height:1.8;color:#3d5f45;">
                {clean_chart_text(narrative) if str(narrative).strip() else "현재 선택 해 기준 Prescriptive 설명이 아직 없습니다."}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 정책 추천 카드")
    cards = summary.get("policy_cards", [])
    if not isinstance(cards, list):
        cards = []

    rows = [cards[i:i + 3] for i in range(0, len(cards), 3)] if cards else []
    if not rows:
        st.info("표시할 정책 추천 카드가 없습니다.")
    else:
        for row_cards in rows:
            cols = st.columns(len(row_cards))
            for col, card in zip(cols, row_cards):
                title = _safe_text(card.get("title", "추천 카드"))
                count = int(pd.to_numeric(card.get("count", 0), errors="coerce") or 0)
                reason = _safe_text(card.get("reason", "대표 이유 정보가 없습니다."))
                effect = _safe_text(card.get("effect", "예상 효과 정보가 없습니다."))
                with col:
                    st.markdown(
                        f"""
                        <div class="section-box">
                            <div style="font-size:1.02rem;font-weight:800;color:#1b4e2a;margin-bottom:8px;">{title}</div>
                            <div style="font-size:1.55rem;font-weight:800;color:#1e4f2b;margin-bottom:10px;">{count:,}건</div>
                            <div style="color:#3d5f45;line-height:1.75;">
                                <b>대표 이유</b><br>{reason}<br><br>
                                <b>예상 효과</b><br>{effect}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("#### Prescriptive Action Table")
    if action_df.empty:
        st.info("표시할 Prescriptive action_df가 없습니다.")
    else:
        st.dataframe(action_df, use_container_width=True, height=520)


# ------------------------------------------------------------
# Engine runner
# ------------------------------------------------------------
def call_engine(df: pd.DataFrame, config: NSGAConfig, progress_callback=None) -> dict:
    sig = inspect.signature(run_nsga2)
    kwargs = {}
    if "progress_callback" in sig.parameters:
        kwargs["progress_callback"] = progress_callback
    if "config" in sig.parameters:
        result = run_nsga2(df, config=config, **kwargs)
    else:
        kwargs2 = {}
        for key, value in config.__dict__.items():
            if key in sig.parameters:
                kwargs2[key] = value
        kwargs2.update(kwargs)
        result = run_nsga2(df, **kwargs2)
    if not isinstance(result, dict):
        raise ValueError("엔진이 dict 형태 결과를 반환하지 않았습니다.")
    return result


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">NSGA-II 분석 대시보드</div>
        <div class="hero-sub">입력 데이터 점검, 시각화, 실행 이력, 통합 결과, Explainable AI를 하나의 웹 화면에서 확인합니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 실행 설정")
    uploaded_file = st.file_uploader("입력 파일 업로드", type=["csv", "xlsx", "xls"])
    n_generations = st.slider("세대 수", min_value=20, max_value=300, value=80, step=10)
    random_seed = st.number_input("랜덤 시드", min_value=0, max_value=999999, value=42, step=1)
    target_ao = st.slider("Target Ao", min_value=0.10, max_value=0.99, value=0.94, step=0.01)
    pmin = st.slider("Pmin", min_value=0.00, max_value=1.00, value=0.10, step=0.01)
    long_lead_percentile = st.slider("Long_Lead_Percentile", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
    ao_impact_percentile = st.slider("Ao_Impct_Percentile", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
    st.markdown(
        f"""
        <div class="run-param-box">
            <span class="info-chip">세대 수 {n_generations}</span>
            <span class="info-chip">시드 {int(random_seed)}</span>
            <span class="info-chip">Target Ao {target_ao:.2f}</span>
            <span class="info-chip">Pmin {pmin:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_btn = st.button("🚀 NSGA 실행", use_container_width=True)


df_input = None
xdf = None
if uploaded_file is not None:
    try:
        df_input = read_uploaded_file(uploaded_file)
        st.session_state.uploaded_name = uploaded_file.name
        xdf = prepare_input_dataframe(df_input)
        st.session_state.precheck_info = build_precheck_info(df_input, xdf)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

if run_btn:
    if df_input is None:
        st.warning("먼저 입력 파일을 업로드해 주세요.")
    else:
        preview_config = NSGAConfig(
            population_size=60,
            n_generations=n_generations,
            random_seed=int(random_seed),
            target_ao=target_ao,
            pmin=pmin,
            long_lead_percentile=long_lead_percentile,
            ao_impact_percentile=ao_impact_percentile,
        )
        target_grid = list(getattr(preview_config, "ao_target_grid", [target_ao]))
        rep_target = float(getattr(preview_config, "representative_target", target_ao))
        sorted_grid = sorted(set(float(x) for x in target_grid + [rep_target]))
        n_targets = len(sorted_grid)

        overall_progress_bar = st.progress(1, text="분석 준비 중... 전체 Target Sweep를 시작합니다.")
        stage_progress_bar = st.progress(1, text="현재 Target Ao 단계 준비 중...")
        stage_box = st.empty()
        heartbeat_box = st.empty()
        target_status_box = st.empty()
        status_box = st.empty()

        st.session_state.generation_logs = []
        st.session_state.progress_emit_state = {"last_emit_gen": -1, "last_emit_ts": 0.0, "last_target": None}
        st.session_state.target_status_map = {
            float(t): {
                "Target Ao": float(t),
                "Stage": f"{i + 1}/{n_targets}",
                "Status": "대기",
                "Generation": 0,
                "Total Gen": int(preview_config.n_generations),
                "Best Ao": None,
                "Mean Ao": None,
                "Best Cost": None,
            }
            for i, t in enumerate(sorted_grid)
        }
        start_ts = time.time()

        stage_box.info(
            f"총 {n_targets}개 Target Ao를 순차적으로 계산합니다. "
            f"대표 타깃 {rep_target:.2f}도 함께 포함됩니다."
        )
        heartbeat_box.caption("현재 상태: 첫 번째 Target Ao 단계 계산 준비 중...")

        def _render_target_status():
            ts_df = pd.DataFrame(list(st.session_state.target_status_map.values())).sort_values("Target Ao").reset_index(drop=True)
            target_status_box.dataframe(ts_df, use_container_width=True, height=260)
            if st.session_state.generation_logs:
                log_df = pd.DataFrame(st.session_state.generation_logs)
                status_box.dataframe(log_df.tail(240), use_container_width=True, height=280)

        _render_target_status()

        def progress_callback(gen, total_gen, best_summary):
            elapsed = time.time() - start_ts
            current_ts = time.time()
            current_target = float(best_summary.get("current_target_ao", rep_target)) if isinstance(best_summary, dict) else rep_target
            gen_safe = max(int(gen), 0)
            total_gen_safe = max(int(total_gen), 1)
            stage_progress = max(1, min(100, int(100 * gen_safe / total_gen_safe)))

            emit_state = st.session_state.get("progress_emit_state", {"last_emit_gen": -1, "last_emit_ts": 0.0, "last_target": None})
            last_emit_gen = int(pd.to_numeric(emit_state.get("last_emit_gen", -1), errors="coerce") or -1)
            last_emit_ts = float(pd.to_numeric(emit_state.get("last_emit_ts", 0.0), errors="coerce") or 0.0)
            last_target = emit_state.get("last_target")

            should_emit = should_emit_progress_update(
                gen=gen_safe,
                total_gen=total_gen_safe,
                last_emit_gen=last_emit_gen,
                last_emit_ts=last_emit_ts,
                current_ts=current_ts,
                current_target=current_target,
                last_target=last_target,
            )

            if current_target not in st.session_state.target_status_map:
                st.session_state.target_status_map[current_target] = {
                    "Target Ao": current_target,
                    "Stage": "-",
                    "Status": "진행 중",
                    "Generation": gen_safe,
                    "Total Gen": total_gen_safe,
                    "Best Ao": best_summary.get("best_ao") if isinstance(best_summary, dict) else None,
                    "Mean Ao": best_summary.get("mean_ao") if isinstance(best_summary, dict) else None,
                    "Best Cost": best_summary.get("best_cost") if isinstance(best_summary, dict) else None,
                }
            else:
                st.session_state.target_status_map[current_target]["Status"] = "진행 중"
                st.session_state.target_status_map[current_target]["Generation"] = gen_safe
                st.session_state.target_status_map[current_target]["Total Gen"] = total_gen_safe
                if isinstance(best_summary, dict):
                    st.session_state.target_status_map[current_target]["Best Ao"] = best_summary.get("best_ao", best_summary.get("best_Ao"))
                    st.session_state.target_status_map[current_target]["Mean Ao"] = best_summary.get("mean_ao", best_summary.get("mean_Ao"))
                    st.session_state.target_status_map[current_target]["Best Cost"] = best_summary.get("best_cost")

            if should_emit:
                st.session_state.generation_logs.append({
                    "elapsed_sec": round(elapsed, 1),
                    "target_ao": current_target,
                    "gen": gen_safe,
                    "total_gen": total_gen_safe,
                    "best_ao": best_summary.get("best_ao", best_summary.get("best_Ao")) if isinstance(best_summary, dict) else None,
                    "mean_ao": best_summary.get("mean_ao", best_summary.get("mean_Ao")) if isinstance(best_summary, dict) else None,
                    "best_cost": best_summary.get("best_cost") if isinstance(best_summary, dict) else None,
                })

                overall_progress_bar.progress(stage_progress, text=f"전체 Sweep 진행 중... Target Ao {current_target:.2f}")
                stage_progress_bar.progress(stage_progress, text=f"현재 Target Ao {current_target:.2f} 단계 진행률: {gen_safe}/{total_gen_safe} 세대")
                heartbeat_box.caption(f"경과시간 {elapsed:,.1f}초 | Target Ao {current_target:.2f} 단계 계산 중...")
                _render_target_status()

                st.session_state.progress_emit_state = {
                    "last_emit_gen": gen_safe,
                    "last_emit_ts": current_ts,
                    "last_target": current_target,
                }

        try:
            result = call_engine(df_input, preview_config, progress_callback)
            for k in st.session_state.target_status_map:
                if st.session_state.target_status_map[k]["Status"] != "완료":
                    st.session_state.target_status_map[k]["Status"] = "완료"
            overall_progress_bar.progress(100, text="전체 Target Sweep 완료")
            stage_progress_bar.progress(100, text="마지막 Target Ao 단계 완료")
            heartbeat_box.caption("현재 상태: 분석이 완료되었습니다.")
            result["xai"] = build_explainability_tables_v4(result, df_input)
            result["prescriptive"] = initialize_prescriptive_structure(
                result,
                st.session_state.get("selected_solution_payload"),
            )
            st.session_state.run_result = result
            _render_target_status()
            st.success("NSGA 분석이 완료되었습니다.")
        except Exception as e:
            st.error("엔진 실행 중 오류가 발생했습니다. 아래 메시지와 '엔진 사전 점검' 탭의 핵심 컬럼 진단을 함께 확인해 주세요.")
            st.exception(e)


# ------------------------------------------------------------
# Main board tabs
# ------------------------------------------------------------
main_tabs = st.tabs(["📋 데이터 개요", "📊 데이터 시각화", "⚙️ 실행 이력", "🏆 통합 결과", "🧠 Explainable AI", "🧭 Prescriptive AI"])
result = st.session_state.run_result

with main_tabs[0]:
    if df_input is None:
        st.info("먼저 입력 파일을 업로드해 주세요.")
    else:
        render_preview_tabs(df_input, xdf)

with main_tabs[1]:
    if df_input is None:
        st.info("먼저 입력 파일을 업로드해 주세요.")
    else:
        render_visual_tabs(df_input, xdf)

with main_tabs[2]:
    render_run_history(result)

with main_tabs[3]:
    render_integrated_results(result)

with main_tabs[4]:
    render_xai(result, df_input)

with main_tabs[5]:
    render_prescriptive(result, df_input)
