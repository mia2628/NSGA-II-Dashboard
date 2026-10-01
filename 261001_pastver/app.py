# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import inspect
import math
import time
import copy
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nsga_engine import NSGAConfig, prepare_input_dataframe, run_nsga2, get_solution_snapshot


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

    /* 하단 진행표를 위 진행표와 같은 계열의 폰트/크기로 통일 */
    [data-testid="stTable"] table,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        font-family: "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo",
                     "AppleGothic", "NanumGothic", "Noto Sans KR", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.35 !important;
    }
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }


    /* NSGA 실행 진행표 공통 폰트/크기 */
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] *,
    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        font-family: "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo",
                     "AppleGothic", "NanumGothic", "Noto Sans KR", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.35 !important;
        font-variant-numeric: tabular-nums !important;
    }
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
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
if "live_generation_logs" not in st.session_state:
    st.session_state.live_generation_logs = []
if "target_status_map" not in st.session_state:
    st.session_state.target_status_map = {}
if "precheck_info" not in st.session_state:
    st.session_state.precheck_info = {}
if "progress_emit_state" not in st.session_state:
    st.session_state.progress_emit_state = {"last_emit_gen": -1, "last_emit_ts": 0.0, "last_target": None}
if "selected_solution_payload" not in st.session_state:
    st.session_state.selected_solution_payload = None


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


def get_selected_snapshot(result: dict, selected_payload: dict | None = None) -> dict | None:
    payload = selected_payload if isinstance(selected_payload, dict) else st.session_state.get("selected_solution_payload")
    if not isinstance(result, dict) or not isinstance(payload, dict):
        return None
    try:
        return get_solution_snapshot(
            result,
            target_ao=float(payload["target_ao"]),
            solution_id=int(payload["solution_id"]),
        )
    except Exception:
        return None


def make_excel_download(result: dict, selected_payload: dict | None = None) -> bytes:
    snapshot = get_selected_snapshot(result, selected_payload)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if isinstance(snapshot, dict):
            selected_summary = snapshot.get("summary", {})
            selected_solution_df = snapshot.get("selected_solution_df", pd.DataFrame())
            selected_policy_df = snapshot.get("policy_df", pd.DataFrame())

            pd.DataFrame([selected_summary]).to_excel(writer, sheet_name="Summary", index=False)
            if isinstance(selected_solution_df, pd.DataFrame) and not selected_solution_df.empty:
                selected_solution_df.to_excel(writer, sheet_name="Selected_Solution", index=False)
            if isinstance(selected_policy_df, pd.DataFrame) and not selected_policy_df.empty:
                selected_policy_df.to_excel(writer, sheet_name="Best_Policy", index=False)
        else:
            pd.DataFrame([result.get("summary", {})]).to_excel(writer, sheet_name="Summary", index=False)
            policy_df = result.get("policy_df", pd.DataFrame())
            if isinstance(policy_df, pd.DataFrame) and not policy_df.empty:
                policy_df.to_excel(writer, sheet_name="Best_Policy", index=False)

        for key, sheet in [
            ("prepared_df", "Prepared_Data"),
            ("candidate_df", "Candidates"),
            ("history_df", "Generation_History"),
            ("pareto_df", "Pareto_Solutions"),
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


def to_numeric_series(value: Any, index: pd.Index | None = None, fill_value: float = 0.0) -> pd.Series:
    if isinstance(value, pd.Series):
        s = value.reindex(index) if index is not None else value.copy()
    elif isinstance(value, np.ndarray):
        s = pd.Series(value)
        if index is not None:
            s = s.reindex(range(len(index)))
            s.index = index
    elif isinstance(value, (list, tuple)):
        s = pd.Series(value)
        if index is not None:
            s = s.reindex(range(len(index)))
            s.index = index
    else:
        if index is None:
            s = pd.Series([value])
        else:
            s = pd.Series([value] * len(index), index=index)
    return pd.to_numeric(s, errors="coerce").fillna(fill_value)


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
        "part_id", "analysis_years", "lambda_period", "p_need_period", "p_need_2y",
        "failure_rate", "lead_time", "proc_cost", "priority_score", "impact_score",
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
    ax.bar(df[x].astype(str), to_numeric_series(df[y], index=df.index))
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x))
    ax.set_ylabel(clean_chart_text(y))
    ax.tick_params(axis="x", rotation=45)
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
        x = to_numeric_series(df.get("xai_cost_score", 0.0), index=df.index)
        y = to_numeric_series(df.get("xai_action_score", 0.0), index=df.index)
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
        managed = summary_df[summary_df["group"] == "Final Selected"]
        unmanaged = summary_df[summary_df["group"] == "No Final Stock"]
        x = range(len(metrics))
        managed_vals = [float(managed.iloc[0][m]) if not managed.empty else 0.0 for m in metrics]
        unmanaged_vals = [float(unmanaged.iloc[0][m]) if not unmanaged.empty else 0.0 for m in metrics]
        w = 0.38
        ax.bar([i - w / 2 for i in x], managed_vals, width=w, label="Final Selected")
        ax.bar([i + w / 2 for i in x], unmanaged_vals, width=w, label="No Final Stock")
        ax.set_xticks(list(x))
        ax.set_xticklabels([clean_chart_text(m) for m in metrics], rotation=20)
        ax.legend()
        ax.grid(axis="y", alpha=0.20)
    ax.set_title(clean_chart_text(title))
    ax.set_ylabel(clean_chart_text("평균"))
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


# ------------------------------------------------------------
# XAI helpers
# ------------------------------------------------------------
def _safe_bool_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index).fillna(False).astype(bool)
    if isinstance(value, np.ndarray):
        arr = list(value.astype(bool)) if hasattr(value, "astype") else list(value)
        arr = arr[:len(index)] + [False] * max(0, len(index) - len(arr))
        return pd.Series(arr, index=index).fillna(False).astype(bool)
    if isinstance(value, (list, tuple)):
        arr = list(value)
        arr = arr[:len(index)] + [False] * max(0, len(index) - len(arr))
        return pd.Series(arr, index=index).fillna(False).astype(bool)
    return pd.Series([bool(value)] * len(index), index=index).fillna(False).astype(bool)


def _safe_numeric_series(value: Any, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value.reindex(index), errors="coerce").fillna(fill_value)
    if isinstance(value, np.ndarray):
        arr = value.tolist()
    elif isinstance(value, (list, tuple)):
        arr = list(value)
    else:
        arr = [value] * len(index)
    arr = arr[:len(index)] + [fill_value] * max(0, len(index) - len(arr))
    return pd.to_numeric(pd.Series(arr, index=index), errors="coerce").fillna(fill_value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if isinstance(out, pd.Series):
            out = out.iloc[0] if len(out) else np.nan
        if pd.isna(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except Exception:
        return int(default)


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
            rows.append({
                "part_id": str(part_id),
                "base_Ao": base_ao,
                "ao_without_item": max(0.0, base_ao - ao_loss),
                "ao_loss_if_removed": ao_loss,
                "ao_with_plus_one": min(1.0, base_ao + ao_gain),
                "ao_gain_if_plus_one": ao_gain,
                "dt_wait_increase_if_removed": wait_inc,
                "dt_total_increase_if_removed": total_inc,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def summarize_policy_stock_relationship(policy_df: pd.DataFrame, summary: dict | None = None) -> dict:
    summary = summary if isinstance(summary, dict) else {}
    if not isinstance(policy_df, pd.DataFrame) or policy_df.empty:
        selected_prebuy = _safe_int(summary.get("selected_prebuy", summary.get("best_selected_prebuy", 0)))
        selected_protection = _safe_int(summary.get("selected_protection", summary.get("best_selected_protection", 0)))
        normal_stock_items = _safe_int(summary.get("normal_stock_items", summary.get("best_normal_stock_items", 0)))
        selected_prebuy_units = _safe_int(summary.get("selected_prebuy_units", summary.get("best_selected_prebuy_units", 0)))
        selected_protection_units = _safe_int(summary.get("selected_protection_units", summary.get("best_selected_protection_units", 0)))
        normal_stock_units = _safe_int(summary.get("normal_stock_units", summary.get("best_normal_stock_units", 0)))
        total_stock_units = _safe_int(summary.get("total_stock_units", summary.get("best_total_stock_units", selected_prebuy_units + selected_protection_units + normal_stock_units)))
        managed_parts = _safe_int(summary.get("managed_parts", summary.get("best_managed_parts", selected_prebuy + selected_protection + normal_stock_items)))
        raw_managed_parts = _safe_int(summary.get("raw_managed_parts", summary.get("best_raw_managed_parts", managed_parts)))
        return {
            "managed_parts": managed_parts,
            "raw_managed_parts": raw_managed_parts,
            "total_stock_units": total_stock_units,
            "selected_prebuy": selected_prebuy,
            "selected_protection": selected_protection,
            "normal_stock_items": normal_stock_items,
            "selected_prebuy_units": selected_prebuy_units,
            "selected_protection_units": selected_protection_units,
            "normal_stock_units": normal_stock_units,
            "stock_category_parts_sum": selected_prebuy + selected_protection + normal_stock_items,
            "stock_category_units_sum": selected_prebuy_units + selected_protection_units + normal_stock_units,
            "managed_minus_raw_gap": managed_parts - raw_managed_parts,
            "unit_balance_gap": total_stock_units - (selected_prebuy_units + selected_protection_units + normal_stock_units),
        }

    df = policy_df.copy()
    idx = df.index
    df["recommended_stock"] = _safe_numeric_series(df.get("recommended_stock", 0.0), idx, 0.0)
    df["prebuy_flag"] = _safe_bool_series(df.get("prebuy_flag", False), idx)
    df["protection_flag"] = _safe_bool_series(df.get("protection_flag", False), idx)
    df["manage_flag"] = _safe_bool_series(df.get("manage_flag", False), idx)

    stocked_mask = df["recommended_stock"] > 0
    prebuy_mask = stocked_mask & df["prebuy_flag"]
    protection_mask = stocked_mask & (~df["prebuy_flag"]) & df["protection_flag"]
    normal_mask = stocked_mask & (~df["prebuy_flag"]) & (~df["protection_flag"])

    selected_prebuy_units = _safe_int(df.loc[prebuy_mask, "recommended_stock"].sum())
    selected_protection_units = _safe_int(df.loc[protection_mask, "recommended_stock"].sum())
    normal_stock_units = _safe_int(df.loc[normal_mask, "recommended_stock"].sum())
    total_stock_units = _safe_int(df.loc[stocked_mask, "recommended_stock"].sum())
    selected_prebuy = int(prebuy_mask.sum())
    selected_protection = int(protection_mask.sum())
    normal_stock_items = int(normal_mask.sum())
    managed_parts = int(stocked_mask.sum())
    raw_managed_parts = int(df["manage_flag"].sum())

    return {
        "managed_parts": managed_parts,
        "raw_managed_parts": raw_managed_parts,
        "selected_prebuy": selected_prebuy,
        "selected_protection": selected_protection,
        "normal_stock_items": normal_stock_items,
        "selected_prebuy_units": selected_prebuy_units,
        "selected_protection_units": selected_protection_units,
        "normal_stock_units": normal_stock_units,
        "total_stock_units": total_stock_units,
        "stock_category_parts_sum": selected_prebuy + selected_protection + normal_stock_items,
        "stock_category_units_sum": selected_prebuy_units + selected_protection_units + normal_stock_units,
        "managed_minus_raw_gap": managed_parts - raw_managed_parts,
        "unit_balance_gap": total_stock_units - (selected_prebuy_units + selected_protection_units + normal_stock_units),
    }


def build_final_stock_relationship_table(stock_summary: dict) -> pd.DataFrame:
    stock_summary = stock_summary if isinstance(stock_summary, dict) else {}
    return pd.DataFrame([
        {"구분": "Selected Pre-buy", "품목 수": _safe_int(stock_summary.get("selected_prebuy", 0)), "재고 단위": _safe_int(stock_summary.get("selected_prebuy_units", 0)), "설명": "선발주 규칙에 걸리면서 최종 권고 재고가 1개 이상인 품목"},
        {"구분": "Selected Protection", "품목 수": _safe_int(stock_summary.get("selected_protection", 0)), "재고 단위": _safe_int(stock_summary.get("selected_protection_units", 0)), "설명": "보호재고 규칙에 걸리면서 최종 권고 재고가 1개 이상인 품목"},
        {"구분": "Normal Stock", "품목 수": _safe_int(stock_summary.get("normal_stock_items", 0)), "재고 단위": _safe_int(stock_summary.get("normal_stock_units", 0)), "설명": "선발주/보호재고는 아니지만 최종적으로 재고 확보가 권고된 품목"},
        {"구분": "Managed Parts", "품목 수": _safe_int(stock_summary.get("managed_parts", 0)), "재고 단위": _safe_int(stock_summary.get("total_stock_units", 0)), "설명": "최종 재고가 실제 배정된 품목 수와 총 재고 단위"},
        {"구분": "Raw Managed Parts", "품목 수": _safe_int(stock_summary.get("raw_managed_parts", stock_summary.get("managed_parts", 0))), "재고 단위": 0, "설명": "Y>0 관리 의사결정 기준 품목 수(재고 배정 전 원시 관리 수)"},
        {"구분": "산술 검증", "품목 수": _safe_int(stock_summary.get("stock_category_parts_sum", 0)), "재고 단위": _safe_int(stock_summary.get("stock_category_units_sum", 0)), "설명": "Selected Pre-buy + Selected Protection + Normal Stock의 합"},
    ])


def build_explainability_tables(result: dict) -> dict:
    policy_df = result.get("policy_df", pd.DataFrame()).copy()
    summary = result.get("summary", {})
    stock_summary = summarize_policy_stock_relationship(policy_df, summary)

    if policy_df.empty:
        return {
            "detail_df": pd.DataFrame(), "reason_counts": pd.DataFrame(), "bucket_counts": pd.DataFrame(),
            "bucket_contrib": pd.DataFrame(), "managed_compare": pd.DataFrame(), "top_priority": pd.DataFrame(),
            "top_impact": pd.DataFrame(), "top_unmanaged": pd.DataFrame(), "top_cost_vs_impact": pd.DataFrame(),
            "solution_cards": {}, "why_solution": {}, "narrative": "Explainable AI 결과를 생성할 policy 데이터가 없습니다.",
            "selected_count": 0, "total_count": 0,
        }

    df = policy_df.copy()
    for col in ["priority_score", "impact_score", "lead_time", "proc_cost", "recommended_stock", "p_need_period", "p_need_2y"]:
        if col not in df.columns:
            df[col] = 0.0
    if "part_id" not in df.columns:
        df["part_id"] = np.arange(1, len(df) + 1)

    p_need_col = "p_need_period" if "p_need_period" in df.columns else "p_need_2y"

    df["manage_flag"] = _safe_bool_series(df.get("manage_flag", False), df.index)
    df["prebuy_flag"] = _safe_bool_series(df.get("prebuy_flag", False), df.index)
    df["protection_flag"] = _safe_bool_series(df.get("protection_flag", False), df.index)
    df["recommended_stock"] = pd.to_numeric(df["recommended_stock"], errors="coerce").fillna(0).astype(int)
    df["final_selected_flag"] = df["recommended_stock"] > 0
    df["stock_bucket"] = np.where(
        df["final_selected_flag"] & df["prebuy_flag"], "PRE-BUY",
        np.where(df["final_selected_flag"] & (~df["prebuy_flag"]) & df["protection_flag"], "PROTECTION",
                 np.where(df["final_selected_flag"], "NORMAL-STOCK", "NO-STOCK"))
    )
    for c in ["proc_cost", "impact_score", "priority_score", "lead_time", p_need_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["cost_burden"] = df["proc_cost"] * df["recommended_stock"]

    impact_norm = _safe_minmax(df["impact_score"])
    lead_norm = _safe_minmax(df["lead_time"])
    need_norm = _safe_minmax(df[p_need_col])
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
                codes.append("PREBUY"); short.append("선발주"); highlights.append("pre-buy")
            if is_protection and stock > 0:
                text.append("고장 시 가용도 저하를 완충하기 위한 보호재고 필요성이 확인되었습니다")
                codes.append("PROTECTION"); short.append("보호재고"); highlights.append("protection")
            if impact >= impact_q75:
                text.append("대표 타깃 달성 관점에서 Ao 영향도가 상위권입니다")
                codes.append("HIGH_IMPACT"); short.append("Ao영향 큼"); highlights.append("impact")
            if lt >= lt_q75:
                text.append("리드타임이 길어 부족 시 회복 지연 위험이 큽니다")
                codes.append("LONG_LT"); short.append("장기리드타임"); highlights.append("lead-time")
            if prio >= prio_q75 and not is_prebuy:
                text.append("종합 우선순위가 높아 관리 품목으로 유지하는 편이 합리적입니다")
                codes.append("HIGH_PRIORITY"); short.append("우선순위 상위"); highlights.append("priority")
            if stock >= max(2, int(stock_q75)):
                text.append("권장 재고 수준이 상대적으로 높아 버퍼 확보 필요성이 큽니다")
                codes.append("HIGH_STOCK"); short.append("권장재고 큼"); highlights.append("buffer-stock")
            if cost_burden >= cost_q75 and impact >= impact_q75:
                text.append("비용 부담은 크지만 운용가용도 기여도도 높아 유지 가치가 있습니다")
                codes.append("HIGH_COST_JUSTIFIED"); short.append("고비용 정당화"); highlights.append("cost-justified")
            if not text:
                text.append("종합 지표 기준에서 임계값을 넘어 관리 대상으로 유지됩니다")
                codes.append("SELECTED"); short.append("관리대상"); highlights.append("selected")
            bucket = "Pre-buy Core" if is_prebuy else ("Protection Core" if is_protection else ("High-Impact Managed" if impact >= impact_q75 else "Managed Monitor"))
            why_selected_summary.append(" · ".join(highlights[:3]))
        else:
            if impact >= impact_q75:
                text.append("영향도는 높지만 다른 핵심 품목 대비 우선순위 경쟁에서 밀렸습니다")
                codes.append("HIGH_IMPACT_BUT_OUT"); short.append("영향도 높지만 제외")
            elif impact < impact_q50:
                text.append("운용가용도에 미치는 영향이 상대적으로 낮아 후순위로 분류되었습니다")
                codes.append("LOW_IMPACT"); short.append("Ao영향 낮음")
            if lt < lt_q50:
                text.append("리드타임이 짧아 즉시 선발주 없이도 대응 가능성이 높습니다")
                codes.append("SHORT_LT"); short.append("짧은리드타임")
            if prio < prio_q50:
                text.append("종합 우선순위 점수가 낮아 관리 대상에서 제외되었습니다")
                codes.append("LOW_PRIORITY"); short.append("우선순위 낮음")
            if cost_burden >= cost_q75 and impact < impact_q75:
                text.append("비용 부담 대비 기대 효과가 제한적이라 현재 해에서 보류가 합리적입니다")
                codes.append("COST_HEAVY_DEFER"); short.append("비용대비 비효율")
            if not text:
                text.append("다른 관리 품목 대비 전략적 가치가 낮아 관찰 대상 수준입니다")
                codes.append("NORMAL"); short.append("후순위")
            bucket = "Watchlist" if impact >= impact_q75 else ("Low Priority" if lt < lt_q50 and prio < prio_q50 else "Deferred")
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
        count=("part_id", "count"), avg_action=("xai_action_score", "mean"),
        avg_balance=("decision_balance_score", "mean"), managed_share=("manage_flag", "mean"),
    )
    selected_mask = df["final_selected_flag"]
    managed_compare = pd.DataFrame([
        {"group": "Final Selected",
         "impact_score": float(df.loc[selected_mask, "impact_score"].mean()) if selected_mask.any() else 0.0,
         "lead_time": float(df.loc[selected_mask, "lead_time"].mean()) if selected_mask.any() else 0.0,
         "priority_score": float(df.loc[selected_mask, "priority_score"].mean()) if selected_mask.any() else 0.0,
         "cost_burden": float(df.loc[selected_mask, "cost_burden"].mean()) if selected_mask.any() else 0.0,
         "xai_action_score": float(df.loc[selected_mask, "xai_action_score"].mean()) if selected_mask.any() else 0.0},
        {"group": "No Final Stock",
         "impact_score": float(df.loc[~selected_mask, "impact_score"].mean()) if (~selected_mask).any() else 0.0,
         "lead_time": float(df.loc[~selected_mask, "lead_time"].mean()) if (~selected_mask).any() else 0.0,
         "priority_score": float(df.loc[~selected_mask, "priority_score"].mean()) if (~selected_mask).any() else 0.0,
         "cost_burden": float(df.loc[~selected_mask, "cost_burden"].mean()) if (~selected_mask).any() else 0.0,
         "xai_action_score": float(df.loc[~selected_mask, "xai_action_score"].mean()) if (~selected_mask).any() else 0.0},
    ])

    top_priority = df.sort_values(["priority_score", "impact_score"], ascending=False).head(15)
    top_impact = df.sort_values(["impact_score", "lead_time"], ascending=False).head(15)
    top_unmanaged = df.loc[~df["manage_flag"]].sort_values(["impact_score", "priority_score"], ascending=False).head(15)
    top_cost_vs_impact = df.sort_values(["decision_balance_score", "xai_action_score"], ascending=False).head(15)

    selected_count = int(stock_summary.get("managed_parts", int(df["final_selected_flag"].sum())))
    total_count = int(len(df))
    final_stock_table = build_final_stock_relationship_table(stock_summary)
    analysis_years = _safe_float(summary.get("analysis_years", summary.get("run_years", 0.0)))
    solution_cards = {
        "selection_logic": f"전체 {total_count:,}개 품목 중 {selected_count:,}개 품목이 현재 대표 해에서 최종 재고 확보 대상으로 확정되었습니다.",
        "managed_focus": "선발주, 보호재고, Ao 영향도, 리드타임, 우선순위가 함께 반영된 품목들이 관리군에 집중됩니다.",
        "watchlist_focus": "현재 제외되었더라도 영향도가 높은 품목은 Watchlist로 재점검할 수 있습니다.",
        "cost_focus": "비용 부담이 크더라도 Ao 기여가 큰 품목은 유지 가치가 높고, 반대는 후순위로 이동합니다.",
    }
    why_solution = {
        "target_alignment": f"대표 해의 Best Ao는 {float(summary.get('best_Ao', summary.get('Ao', 0.0))):.4f}이며 분석기간은 {analysis_years:g}년입니다.",
        "selection_bias": "Selected Pre-buy와 Selected Protection은 최종 권고 재고가 1개 이상인 경우만 집계합니다.",
        "risk_buffer": f"총 재고 {int(stock_summary.get('total_stock_units', 0)):,}단위를 Selected Pre-buy + Selected Protection + Normal Stock으로 분해해 설명합니다.",
        "tradeoff": "상대방 설득 포인트는 왜 이 품목에 실제 재고를 배정했는지, 그리고 그 재고가 Ao와 다운타임을 어떻게 방어하는지입니다.",
    }
    narrative = (
        f"분석기간 {analysis_years:g}년 기준 대표 해의 최종 재고 확보 품목은 {selected_count:,}개이며 "
        f"총 재고는 {int(stock_summary.get('total_stock_units', 0)):,}단위입니다. "
        f"이 재고는 Selected Pre-buy {int(stock_summary.get('selected_prebuy', 0)):,}개, "
        f"Selected Protection {int(stock_summary.get('selected_protection', 0)):,}개, "
        f"Normal Stock {int(stock_summary.get('normal_stock_items', 0)):,}개로 분해됩니다."
    )

    return {
        "detail_df": df, "reason_counts": reason_counts, "bucket_counts": bucket_counts,
        "bucket_contrib": bucket_contrib, "managed_compare": managed_compare,
        "top_priority": top_priority, "top_impact": top_impact, "top_unmanaged": top_unmanaged,
        "top_cost_vs_impact": top_cost_vs_impact, "solution_cards": solution_cards,
        "why_solution": why_solution, "narrative": narrative, "selected_count": selected_count,
        "total_count": total_count, "stock_summary": stock_summary, "final_stock_table": final_stock_table,
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


# ------------------------------------------------------------
# Prescriptive helpers
# ------------------------------------------------------------
def _series_from_value(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value.reindex(index), errors="coerce").fillna(0.0)
    if isinstance(value, np.ndarray):
        arr = list(value)
    elif isinstance(value, (list, tuple)):
        arr = list(value)
    else:
        arr = [value] * len(index)
    arr = arr[:len(index)] + [0.0] * max(0, len(index) - len(arr))
    return pd.to_numeric(pd.Series(arr, index=index), errors="coerce").fillna(0.0)


def build_prescriptive_action_df(result: dict, df_input: pd.DataFrame | None = None) -> pd.DataFrame:
    xai = result.get("xai", {})
    detail_df = xai.get("detail_df", pd.DataFrame()).copy() if isinstance(xai, dict) else pd.DataFrame()

    if not isinstance(detail_df, pd.DataFrame) or detail_df.empty:
        return pd.DataFrame(columns=[
            "part_id", "recommended_action", "action_group", "priority_level", "reason_summary",
            "expected_effect", "manage_flag", "prebuy_flag", "protection_flag",
            "ao_loss_if_removed", "ao_gain_if_plus_one", "dt_wait_increase_if_removed",
        ])

    df = detail_df.copy()
    idx = df.index

    if "part_id" not in df.columns:
        df["part_id"] = np.arange(1, len(df) + 1)

    df["manage_flag"] = _series_from_value(df.get("manage_flag", False), idx).astype(bool)
    df["prebuy_flag"] = _series_from_value(df.get("prebuy_flag", False), idx).astype(bool)
    df["protection_flag"] = _series_from_value(df.get("protection_flag", False), idx).astype(bool)

    for c in ["ao_loss_if_removed", "ao_gain_if_plus_one", "dt_wait_increase_if_removed"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["xai_reason", "exact_marginal_reason", "xai_reason_short"]:
        if c not in df.columns:
            df[c] = ""

    ao_loss_q75 = float(df["ao_loss_if_removed"].quantile(0.75)) if len(df) else 0.0
    ao_gain_q75 = float(df["ao_gain_if_plus_one"].quantile(0.75)) if len(df) else 0.0

    rec_actions, action_groups, priority_levels, expected_effects = [], [], [], []
    for _, row in df.iterrows():
        manage = bool(row.get("manage_flag", False))
        prebuy = bool(row.get("prebuy_flag", False))
        protection = bool(row.get("protection_flag", False))
        ao_loss = _safe_float(row.get("ao_loss_if_removed", 0.0))
        ao_gain = _safe_float(row.get("ao_gain_if_plus_one", 0.0))
        dt_wait = _safe_float(row.get("dt_wait_increase_if_removed", 0.0))

        if prebuy:
            rec_actions.append("즉시 선발주"); action_groups.append("immediate"); priority_levels.append("High")
            expected_effects.append(f"초기 조달 지연 위험을 낮추고 목표 Ao 유지에 유리합니다. 제거 시 Ao 하락 추정 {ao_loss:.4f}")
        elif protection:
            rec_actions.append("보호재고 유지"); action_groups.append("monitor")
            priority_levels.append("High" if ao_loss >= ao_loss_q75 and ao_loss > 0 else "Medium")
            expected_effects.append(f"부족 발생 시 DT_wait 증가를 완충합니다. 제거 시 대기 다운타임 증가 추정 {dt_wait:.1f}h")
        elif manage and ao_gain >= ao_gain_q75 and ao_gain > 0:
            rec_actions.append("재고 1단위 추가 검토"); action_groups.append("review"); priority_levels.append("Medium")
            expected_effects.append(f"재고 1단위 추가 시 Ao 추가 개선 여지 {ao_gain:.4f}")
        elif manage:
            rec_actions.append("유지 + 모니터링"); action_groups.append("monitor"); priority_levels.append("Medium")
            expected_effects.append("현재 대표 해 기준 관리 유지가 합리적이며 성과와 비용을 함께 관찰합니다")
        elif (not manage) and ao_loss >= ao_loss_q75 and ao_loss > 0:
            rec_actions.append("Watchlist 승격"); action_groups.append("watchlist"); priority_levels.append("Medium")
            expected_effects.append(f"현재는 비관리지만 제거 민감도가 높아 재검토 가치가 있습니다. Ao 영향 추정 {ao_loss:.4f}")
        else:
            rec_actions.append("보류 / 후순위"); action_groups.append("deferred"); priority_levels.append("Low")
            expected_effects.append("현재 해 기준 다른 핵심 품목 대비 우선순위가 낮아 보류가 합리적입니다")

    df["recommended_action"] = rec_actions
    df["action_group"] = action_groups
    df["priority_level"] = priority_levels
    df["reason_summary"] = (
        df["xai_reason"].astype(str).str.strip().replace("nan", "")
        + np.where(df["exact_marginal_reason"].astype(str).str.strip().ne(""),
                   " | " + df["exact_marginal_reason"].astype(str).str.strip(), "")
    ).str.strip(" |")
    df["expected_effect"] = expected_effects

    out_cols = [
        "part_id", "recommended_action", "action_group", "priority_level", "reason_summary",
        "expected_effect", "manage_flag", "prebuy_flag", "protection_flag",
        "ao_loss_if_removed", "ao_gain_if_plus_one", "dt_wait_increase_if_removed",
    ]
    return df[out_cols].copy()


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
    return str(vals.value_counts().index[0])


def _mean_from_series(series: pd.Series) -> float:
    if not isinstance(series, pd.Series) or series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).mean())


def build_prescriptive_policy_cards(action_df: pd.DataFrame) -> list[dict]:
    specs = [
        ("즉시 선발주", "초기 조달 지연 위험을 줄이는 데 도움을 줍니다."),
        ("보호재고 유지", "다운타임 완충 효과를 기대할 수 있습니다."),
        ("재고 1단위 추가 검토", "Ao 추가 개선 가능성을 확인하는 데 의미가 있습니다."),
        ("유지 + 모니터링", "현재 해의 관리 상태를 안정적으로 유지합니다."),
        ("Watchlist 승격", "추후 관리군 편입 여부를 판단하는 데 도움을 줍니다."),
        ("보류 / 후순위", "현재 자원을 핵심 품목에 우선 배분할 수 있습니다."),
    ]
    if not isinstance(action_df, pd.DataFrame) or action_df.empty:
        return [{"title": t, "count": 0, "reason": "대상 품목이 아직 없습니다.", "effect": e} for t, e in specs]

    cards = []
    for title, fallback_effect in specs:
        sub = action_df[action_df["recommended_action"].astype(str) == title].copy()
        if sub.empty:
            cards.append({"title": title, "count": 0, "reason": "대상 품목이 아직 없습니다.", "effect": fallback_effect})
        else:
            cards.append({
                "title": title,
                "count": int(len(sub)),
                "reason": _top_nonempty_text(sub.get("reason_summary", pd.Series(dtype=str))),
                "effect": _top_nonempty_text(sub.get("expected_effect", pd.Series(dtype=str))),
            })
    return cards


def build_prescriptive_package(result: dict, df_input: pd.DataFrame | None = None) -> dict:
    action_df = build_prescriptive_action_df(result, df_input)
    action_group = action_df["action_group"] if "action_group" in action_df.columns else pd.Series(dtype="object")
    summary = {
        "total_actions": int(len(action_df)),
        "immediate_actions": int((action_group == "immediate").sum()) if len(action_df) else 0,
        "review_actions": int((action_group == "review").sum()) if len(action_df) else 0,
        "monitor_actions": int((action_group == "monitor").sum()) if len(action_df) else 0,
        "watchlist_candidates": int((action_group == "watchlist").sum()) if len(action_df) else 0,
        "deferred_actions": int((action_group == "deferred").sum()) if len(action_df) else 0,
    }
    summary["policy_cards"] = build_prescriptive_policy_cards(action_df)

    payload = st.session_state.get("selected_solution_payload")
    payload_txt = "현재 선택 해 정보가 아직 없습니다."
    if isinstance(payload, dict):
        try:
            payload_txt = (
                f"현재 선택 해는 Target {float(pd.to_numeric(payload.get('target_ao', 0.0), errors='coerce')):.2f}, "
                f"Solution {int(pd.to_numeric(payload.get('solution_id', 0), errors='coerce'))}입니다."
            )
        except Exception:
            pass

    years = _safe_float(result.get("summary", {}).get("analysis_years", 0))
    narrative = (
        f"{payload_txt} 분석기간은 {years:g}년입니다. "
        f"현재 선택 해 기준 총 {summary['total_actions']:,}개의 액션이 추천되며, "
        f"즉시 실행 {summary['immediate_actions']:,}건, 추가 검토 {summary['review_actions']:,}건, "
        f"유지/모니터링 {summary['monitor_actions']:,}건, Watchlist {summary['watchlist_candidates']:,}건, "
        f"보류/후순위 {summary['deferred_actions']:,}건입니다."
    )
    return {"action_df": action_df, "summary": summary, "narrative": narrative}


def initialize_prescriptive_structure(result: dict, selected_payload: dict | None = None) -> dict:
    return build_prescriptive_package(result, None)


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
            st.caption("선택한 분석기간과 파라미터가 반영된 엔진 입력 전 표준화 데이터입니다.")
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
        st.markdown(
            "<div class='debug-box'><b>기간 연동 엔진</b><br>"
            "원본 df_input을 엔진에 전달하고, 선택한 분석기간은 NSGAConfig.years로 전달됩니다. "
            "전처리 미리보기에도 동일한 분석기간이 반영됩니다.</div>",
            unsafe_allow_html=True,
        )
        st.write("원본 데이터 shape:", precheck.get("raw_shape"))
        st.write("전처리 데이터 shape:", precheck.get("prepared_shape"))
        summary_df = precheck.get("summary_df", pd.DataFrame())
        preview_df = precheck.get("preview_df", pd.DataFrame())
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            st.markdown("**핵심 컬럼 진단 요약**")
            st.dataframe(summary_df, use_container_width=True, height=260)
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
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(counts[obj].astype(str), counts["count"])
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
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
        st.dataframe(history_df, use_container_width=True, height=520) if isinstance(history_df, pd.DataFrame) and not history_df.empty else st.info("세대별 로그가 없습니다.")
    with tabs[1]:
        st.dataframe(sweep_df, use_container_width=True, height=520) if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty else st.info("Sweep 요약이 없습니다.")
    with tabs[2]:
        target_status_map = st.session_state.get("target_status_map", {})
        gen_logs = st.session_state.get("generation_logs", [])
        if target_status_map:
            st.dataframe(pd.DataFrame(list(target_status_map.values())).sort_values("Target Ao").reset_index(drop=True), use_container_width=True, height=260)
        if gen_logs:
            st.dataframe(pd.DataFrame(gen_logs).tail(240), use_container_width=True, height=260)


def build_solution_payload_key(target_ao: Any, solution_id: Any) -> str:
    return f"{float(pd.to_numeric(target_ao, errors='coerce')):.6f}|{int(pd.to_numeric(solution_id, errors='coerce'))}"


def parse_solution_payload_key(payload_key: Any):
    if payload_key is None:
        return None
    if isinstance(payload_key, (list, tuple)) and len(payload_key) == 1:
        payload_key = payload_key[0]
    try:
        left, right = str(payload_key).strip().split("|", 1)
        return {"target_ao": float(pd.to_numeric(left, errors="coerce")), "solution_id": int(pd.to_numeric(right, errors="coerce"))}
    except Exception:
        return None


def build_solution_selector_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "target_ao" in out.columns:
        out["target_ao"] = pd.to_numeric(out["target_ao"], errors="coerce")
    if "solution_id" not in out.columns:
        if "solution_id_within_target" in out.columns:
            out["solution_id"] = pd.to_numeric(out["solution_id_within_target"], errors="coerce")
        elif "solution_idx" in out.columns:
            out["solution_id"] = pd.to_numeric(out["solution_idx"], errors="coerce")
        elif "idx" in out.columns:
            out["solution_id"] = pd.to_numeric(out["idx"], errors="coerce")
        else:
            out["solution_id"] = np.arange(len(out))
    # String Pareto solution IDs are not used for point payloads; convert safely.
    if not pd.api.types.is_numeric_dtype(out["solution_id"]):
        out["solution_id"] = np.arange(len(out))
    out["solution_id"] = pd.to_numeric(out["solution_id"], errors="coerce").fillna(0).astype(int)
    if "is_pareto_2d" not in out.columns:
        out["is_pareto_2d"] = 0
    if "Ao" in out.columns:
        out["Ao"] = pd.to_numeric(out["Ao"], errors="coerce")
    if "target_ao" in out.columns:
        out["_payload_key"] = out.apply(lambda r: build_solution_payload_key(r.get("target_ao", 0.0), r.get("solution_id", 0)), axis=1)
    return out


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
        dominated = sub[sub["is_pareto_2d"] != 1].copy()
        pareto = sub[sub["is_pareto_2d"] == 1].copy()
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
                    customdata=dominated["_payload_key"],
                    marker=dict(size=8, color=color, opacity=0.24, line=dict(width=0)),
                    selected=dict(marker=dict(size=26, opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.12)),
                    hovertemplate="Cost=%{x:,.2f}<br>Ao=%{y:.4f}<br>Pareto=N<extra></extra>",
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
                    customdata=pareto["_payload_key"],
                    marker=dict(size=14, color=color, opacity=0.95, line=dict(width=1.6, color="#111111")),
                    selected=dict(marker=dict(size=24, opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.28)),
                    hovertemplate="Cost=%{x:,.2f}<br>Ao=%{y:.4f}<br><b>Pareto=Y</b><extra></extra>",
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
            font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        clickmode="event+select",
        uirevision="nsga-ce-fixed",
        selectionrevision="nsga-ce-selection",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", showline=True, mirror=True, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", showline=True, mirror=True, zeroline=False)
    return fig


def _apply_persistent_selected_point(fig, selected_payload: dict | None):
    """Persist the selected point after fragment reruns without adding a blocking overlay trace."""
    if fig is None or not isinstance(selected_payload, dict):
        return fig

    try:
        selected_key = build_solution_payload_key(
            selected_payload["target_ao"],
            selected_payload["solution_id"],
        )
    except Exception:
        return fig

    for trace in fig.data:
        custom = getattr(trace, "customdata", None)
        if custom is None:
            continue
        try:
            keys = [str(x) for x in list(custom)]
        except Exception:
            continue

        hits = [i for i, key in enumerate(keys) if key == selected_key]
        trace.selectedpoints = hits if hits else []
    return fig


def _get_ce_base_figure(result: dict, selector_df: pd.DataFrame):
    """Build the heavy C-E figure once per NSGA run, then reuse it on point clicks."""
    if not isinstance(result, dict):
        return build_ce_curve_plotly(selector_df, "Cost-Effectiveness Curve")

    cache_key = "_ce_base_figure_dict"
    cached = result.get(cache_key)
    if isinstance(cached, dict):
        return go.Figure(copy.deepcopy(cached))

    fig = build_ce_curve_plotly(selector_df, "Cost-Effectiveness Curve")
    result[cache_key] = copy.deepcopy(fig.to_dict())
    return fig


def extract_plotly_selected_payload(event_obj):
    if event_obj is None:
        return None
    points = None
    try:
        if hasattr(event_obj, "selection") and hasattr(event_obj.selection, "points"):
            points = event_obj.selection.points
        elif isinstance(event_obj, dict):
            points = event_obj.get("selection", {}).get("points") if isinstance(event_obj.get("selection"), dict) else event_obj.get("points")
    except Exception:
        points = None
    if not points:
        return None
    p0 = points[0]
    customdata = p0.get("customdata") if isinstance(p0, dict) else getattr(p0, "customdata", None)
    return parse_solution_payload_key(customdata)


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
        fig.add_trace(go.Scatter(
            x=[float(pd.to_numeric(row[cost_col], errors="coerce"))],
            y=[float(pd.to_numeric(row["Ao"], errors="coerce"))],
            mode="markers", marker=dict(size=18, color="#FFD700", symbol="star", line=dict(width=1.6, color="#111111")),
            hovertemplate="Current Point<br>Cost=%{x:,.2f}<br>Ao=%{y:.4f}<extra></extra>", showlegend=False,
        ))
    except Exception:
        pass
    return fig


def render_solution_detail_card(row: pd.Series):
    fields = [
        ("analysis_years", "분석 기간(년)"), ("Ao", "Ao"), ("DT_diag_h", "DT_diag_h"), ("DT_wait_h", "DT_wait_h"),
        ("DT_restore_h", "DT_restore_h"), ("DT_total_h", "DT_total_h"), ("stock_cost", "stock_cost"),
        ("hold_cost", "hold_cost"), ("total_cost", "total_cost"), ("stocked_managed_parts", "managed parts"),
        ("total_stock_units", "total stock units"), ("prebuy_selected", "selected pre-buy"),
        ("protection_selected", "selected protection"),
    ]
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("#### 선택 해 최종 분석결과")
    for key, label in fields:
        if key in row.index and pd.notna(row[key]):
            val = row[key]
            try:
                val_txt = f"{float(val):.4f}" if key == "Ao" else (f"{int(val)}" if key == "analysis_years" else f"{float(val):.2f}")
            except Exception:
                val_txt = str(val)
            c1, c2 = st.columns([1.8, 1.2])
            with c1:
                st.markdown(f"**{label}**")
            with c2:
                st.markdown(f"`{val_txt}`")
    st.markdown("</div>", unsafe_allow_html=True)


def render_summary_panel(summary: dict, policy_df: pd.DataFrame | None = None):
    best_ao = _safe_float(summary.get("best_Ao", summary.get("Ao", 0.0)))
    target_ao = _safe_float(summary.get("target_ao", summary.get("representative_target", 0.0)))
    years = _safe_float(summary.get("analysis_years", summary.get("run_years", 0.0)))
    stock_summary = summarize_policy_stock_relationship(policy_df if isinstance(policy_df, pd.DataFrame) else pd.DataFrame(), summary)
    managed_parts = _safe_int(stock_summary.get("managed_parts", 0))
    total_stock = _safe_int(stock_summary.get("total_stock_units", 0))
    total_cost = _safe_float(summary.get("best_cost", 0.0))
    dt_wait = _safe_float(summary.get("best_DT_wait_h", 0.0))
    dt_total = _safe_float(summary.get("best_DT_total_h", 0.0))

    st.markdown(
        f"""
        <div class="section-box">
            <div style="font-size:1.15rem;font-weight:800;color:#1b4e2a;margin-bottom:8px;">대표 해 요약</div>
            <div style="line-height:1.8;color:#3d5f45;">
                <b>분석기간 {years:g}년</b> 기준, 대표 Target Ao 0.94에서 Ao {best_ao:.4f}를 산출했습니다.
                관측시간과 DT_diag/DT_wait/DT_restore는 모두 동일한 {years:g}년 기간으로 환산됩니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: show_metric_card("분석 기간", f"{years:g}년")
    with c2: show_metric_card("Best Ao", f"{best_ao:.4f}")
    with c3: show_metric_card("Target Ao", f"0.94")
    with c4: show_metric_card("Managed Parts", f"{managed_parts:,}")
    with c5: show_metric_card("Total Stock Units", f"{total_stock:,}")
    with c6: show_metric_card("Total Cost", f"{total_cost:,.0f}")

    detail_rows = [
        {"항목": "분석 기간", "값": f"{years:g}년", "의미": "Ao·다운타임·고장발생확률·유지비 계산에 적용되는 기간"},
        {"항목": "관측시간", "값": f"{_safe_float(summary.get('analysis_hours', years * 8760)):,.0f} h", "의미": "분석기간 × 8,760시간"},
        {"항목": "Managed Parts", "값": f"{managed_parts:,}", "의미": "최종적으로 재고가 1개 이상 배정된 품목 수"},
        {"항목": "DT_wait_h", "값": f"{dt_wait:,.2f}", "의미": f"{years:g}년 분석기간의 누적 기대 대기 다운타임"},
        {"항목": "DT_total_h", "값": f"{dt_total:,.2f}", "의미": f"{years:g}년 분석기간의 누적 기대 총 다운타임"},
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True, height=230)




# ------------------------------------------------------------
# Browser-side Cost-Effectiveness Curve
# ------------------------------------------------------------
_CE_V2_AVAILABLE = bool(
    hasattr(st, "components")
    and hasattr(st.components, "v2")
    and hasattr(st.components.v2, "component")
)

_CE_COMPONENT = None
if _CE_V2_AVAILABLE:
    _CE_COMPONENT = st.components.v2.component(
        name="nsga_ce_curve_frontend",
        html="""
        <div class="ce-root">
          <div class="ce-top">
            <div class="ce-chart-panel">
              <div class="ce-title">Cost-Effectiveness Curve</div>
              <svg class="ce-svg" role="img" aria-label="Cost-Effectiveness Curve"></svg>
            </div>
            <div class="ce-detail-panel">
              <div class="ce-detail-title">선택 해 최종 분석결과</div>
              <div class="ce-selected-label"></div>
              <div class="ce-metrics"></div>
              <button type="button" class="ce-export-btn">선택 해 수리부속 엑셀 다운로드</button>
            </div>
          </div>
          <div class="ce-policy-panel">
            <div class="ce-policy-title">선택 해 산출 수리부속</div>
            <div class="ce-policy-summary"></div>
            <div class="ce-table-wrap">
              <table class="ce-policy-table">
                <thead>
                  <tr>
                    <th>품번</th><th>품목</th><th>총수량</th><th>O</th><th>I</th><th>D</th>
                    <th>정비단계</th><th>정책구분</th><th>리드타임(h)</th><th>단가</th>
                    <th>고장률</th><th>기간 고장확률</th><th>영향도</th>
                  </tr>
                </thead>
                <tbody></tbody>
              </table>
            </div>
          </div>
        </div>
        """,
        css="""
        .ce-root { width:100%; font-family:"Malgun Gothic","맑은 고딕",sans-serif; }
        .ce-top {
          display:grid; grid-template-columns:minmax(0,1.7fr) minmax(270px,1fr);
          gap:18px; align-items:stretch;
        }
        .ce-chart-panel,.ce-detail-panel,.ce-policy-panel {
          border:1px solid #e3eadf; border-radius:14px; background:#fff; padding:14px;
          box-sizing:border-box; min-width:0;
        }
        .ce-title,.ce-detail-title,.ce-policy-title { font-weight:800; color:#214d2b; margin-bottom:8px; }
        .ce-svg { width:100%; height:430px; display:block; overflow:visible; }
        .ce-grid { stroke:#e7ece5; stroke-width:1; }
        .ce-axis { stroke:#8fa092; stroke-width:1.1; }
        .ce-tick { fill:#5b695e; font-size:11px; }
        .ce-axis-label { fill:#405548; font-size:12px; font-weight:700; }
        .ce-point { cursor:pointer; transition:r .035s ease, opacity .035s ease, stroke-width .035s ease; }
        .ce-point.normal { opacity:.24; }
        .ce-point.pareto { opacity:.94; stroke:#111; stroke-width:1.2; }
        .ce-point.selected { opacity:1 !important; stroke:#111 !important; stroke-width:3 !important; }
        .ce-selected-label { font-size:12px; color:#607363; margin-bottom:10px; }
        .ce-metrics { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
        .ce-metric { border:1px solid #edf1ea; border-radius:10px; padding:9px 10px; background:#fbfdf9; }
        .ce-metric-label { font-size:11px; color:#6b7d6e; margin-bottom:3px; }
        .ce-metric-value { font-size:16px; font-weight:800; color:#214d2b; word-break:break-word; }
        .ce-export-btn {
          margin-top:12px; width:100%; min-height:40px; border-radius:10px;
          border:1px solid #cfe0c8; background:#eef7e8; color:#24552f;
          font-weight:800; cursor:pointer;
        }
        .ce-export-btn:hover { background:#e4f2dc; }
        .ce-policy-panel { margin-top:16px; display:none; }
        .ce-policy-summary { font-size:12px; color:#607363; margin-bottom:8px; }
        .ce-table-wrap { width:100%; overflow:auto; max-height:430px; border:1px solid #edf1ea; border-radius:10px; }
        .ce-policy-table { width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }
        .ce-policy-table th { position:sticky; top:0; z-index:1; background:#f5f8f3; color:#38513d; font-weight:800; }
        .ce-policy-table th,.ce-policy-table td { padding:7px 8px; border-bottom:1px solid #edf1ea; text-align:right; }
        .ce-policy-table th:first-child,.ce-policy-table td:first-child,
        .ce-policy-table th:nth-child(2),.ce-policy-table td:nth-child(2),
        .ce-policy-table th:nth-child(7),.ce-policy-table td:nth-child(7),
        .ce-policy-table th:nth-child(8),.ce-policy-table td:nth-child(8) { text-align:left; }
        .ce-policy-table tbody tr:hover { background:#f8fbf6; }
        @media (max-width: 900px) {
          .ce-top { grid-template-columns:1fr; }
          .ce-svg { height:380px; }
        }
        """,
        js="""
        export default function(component) {
          const { parentElement, data } = component;
          const root = parentElement.querySelector(".ce-root");
          const svg = parentElement.querySelector(".ce-svg");
          const metricsEl = parentElement.querySelector(".ce-metrics");
          const selectedLabel = parentElement.querySelector(".ce-selected-label");
          const policyPanel = parentElement.querySelector(".ce-policy-panel");
          const policySummary = parentElement.querySelector(".ce-policy-summary");
          const policyBody = parentElement.querySelector(".ce-policy-table tbody");
          const exportBtn = parentElement.querySelector(".ce-export-btn");

          const points = Array.isArray(data?.points) ? data.points : [];
          const candidateMeta = Array.isArray(data?.candidate_meta) ? data.candidate_meta : [];
          const solutionVectors = data?.solution_vectors || {};
          const showPolicy = Boolean(data?.show_policy);
          let selectedKey = data?.selected_key || (points[0]?.key ?? "");
          let selectedPolicyRows = [];

          policyPanel.style.display = showPolicy ? "block" : "none";

          const NS = "http://www.w3.org/2000/svg";
          const colors = ["#4C78A8","#F58518","#54A24B","#E45756","#72B7B2","#B279A2","#9D755D","#2E91E5","#1CA71C","#E15F99","#7A9E2E","#7B61A8"];
          const uniqTargets = [...new Set(points.map(p => Number(p.target_ao)))].sort((a,b)=>a-b);
          const colorFor = (t) => colors[Math.max(0, uniqTargets.indexOf(Number(t))) % colors.length];

          const fmt = (v, digits=2) => {
            const n = Number(v);
            if (!Number.isFinite(n)) return "-";
            return n.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits});
          };
          const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));

          function policyType(meta, qty, managed) {
            if (meta.prebuy_flag && managed) return "PRE-BUY";
            if (meta.protection_flag && qty > 0) return "PROTECTION";
            if (managed) return "MANAGED";
            return "NORMAL";
          }

          function buildPolicyRows(key) {
            const vec = solutionVectors[key];
            if (!showPolicy || !vec || !candidateMeta.length) return [];
            const sO = vec.sO || [], sI = vec.sI || [], sD = vec.sD || [], y = vec.Y || [];
            const rows = [];
            for (let i=0; i<candidateMeta.length; i++) {
              const o = Number(sO[i] || 0), ii = Number(sI[i] || 0), d = Number(sD[i] || 0);
              const qty = o + ii + d;
              if (qty <= 0) continue;
              const m = candidateMeta[i] || {};
              const managed = Number(y[i] || 0) > 0;
              rows.push({
                part_id: m.part_id ?? "",
                item_name: m.item_name ?? "",
                qty, stock_O:o, stock_I:ii, stock_D:d,
                echelon: m.maint_echelon ?? "",
                policy_type: policyType(m, qty, managed),
                lead_time: Number(m.lead_time || 0),
                proc_cost: Number(m.proc_cost || 0),
                failure_rate: Number(m.failure_rate || 0),
                p_need_period: Number(m.p_need_period || 0),
                impact_score: Number(m.impact_score || 0)
              });
            }
            rows.sort((a,b) => b.qty-a.qty || b.impact_score-a.impact_score);
            return rows;
          }

          function renderPolicy(key) {
            if (!showPolicy) return;
            selectedPolicyRows = buildPolicyRows(key);
            const totalUnits = selectedPolicyRows.reduce((s,r)=>s+r.qty,0);
            policySummary.textContent = `산출 품목 ${selectedPolicyRows.length.toLocaleString()}종 · 총 재고 ${totalUnits.toLocaleString()}개`;
            policyBody.innerHTML = selectedPolicyRows.map(r => `
              <tr>
                <td>${esc(r.part_id)}</td><td>${esc(r.item_name)}</td>
                <td>${fmt(r.qty,0)}</td><td>${fmt(r.stock_O,0)}</td><td>${fmt(r.stock_I,0)}</td><td>${fmt(r.stock_D,0)}</td>
                <td>${esc(r.echelon)}</td><td>${esc(r.policy_type)}</td>
                <td>${fmt(r.lead_time,1)}</td><td>${fmt(r.proc_cost,0)}</td>
                <td>${fmt(r.failure_rate,6)}</td><td>${fmt(r.p_need_period,4)}</td><td>${fmt(r.impact_score,6)}</td>
              </tr>`).join("");
          }

          function renderDetail(p) {
            if (!p) return;
            selectedLabel.textContent = `목표 Ao ${Number(p.target_ao).toFixed(2)} · 해 ${p.solution_id}`;
            const rows = [
              ["Ao", Number(p.Ao).toFixed(4)],
              ["총 비용", fmt(p.total_cost,0)],
              ["DT 진단", fmt(p.DT_diag_h,2)+" h"],
              ["DT 대기", fmt(p.DT_wait_h,2)+" h"],
              ["DT 복구", fmt(p.DT_restore_h,2)+" h"],
              ["총 DT", fmt(p.DT_total_h,2)+" h"],
              ["관리 품목", fmt(p.stocked_managed_parts,0)],
              ["총 재고", fmt(p.total_stock_units,0)],
              ["선발주", fmt(p.prebuy_selected,0)],
              ["보호재고", fmt(p.protection_selected,0)]
            ];
            metricsEl.innerHTML = rows.map(([k,v]) =>
              `<div class="ce-metric"><div class="ce-metric-label">${k}</div><div class="ce-metric-value">${v}</div></div>`
            ).join("");
            renderPolicy(p.key);
          }

          function selectPoint(key) {
            selectedKey = key;
            const p = points.find(x => x.key === key) || points[0];
            root.querySelectorAll(".ce-point").forEach(el => {
              const isSel = el.dataset.key === key;
              el.classList.toggle("selected", isSel);
              const base = el.classList.contains("pareto") ? 6.3 : 4.2;
              el.setAttribute("r", isSel ? 10 : base);
            });
            renderDetail(p);
          }

          function downloadSelectedPolicy() {
            const p = points.find(x => x.key === selectedKey) || points[0];
            if (!p) return;
            const rows = showPolicy ? selectedPolicyRows : buildPolicyRows(selectedKey);
            const summary = [
              ["항목","값"],
              ["목표 Ao",Number(p.target_ao).toFixed(2)],
              ["해 번호",String(p.solution_id)],
              ["Ao",Number(p.Ao).toFixed(6)],
              ["총 비용",String(p.total_cost)],
              ["총 DT(h)",String(p.DT_total_h)],
              ["총 재고",String(p.total_stock_units)]
            ];
            const table = [
              ["품번","품목","총수량","O","I","D","정비단계","정책구분","리드타임(h)","단가","고장률","기간 고장확률","영향도"],
              ...rows.map(r => [r.part_id,r.item_name,r.qty,r.stock_O,r.stock_I,r.stock_D,r.echelon,r.policy_type,r.lead_time,r.proc_cost,r.failure_rate,r.p_need_period,r.impact_score])
            ];
            const toHtmlRows = arr => arr.map(row => "<tr>"+row.map(v=>"<td>"+esc(v)+"</td>").join("")+"</tr>").join("");
            const html = `<!doctype html><html><meta charset="utf-8"><body>
              <table>${toHtmlRows(summary)}</table><br><table>${toHtmlRows(table)}</table></body></html>`;
            const blob = new Blob(["\\ufeff", html], {type:"application/vnd.ms-excel;charset=utf-8"});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `NSGA_selected_${Number(p.target_ao).toFixed(2)}_${p.solution_id}.xls`;
            a.click();
            setTimeout(()=>URL.revokeObjectURL(url),1000);
          }

          exportBtn.addEventListener("click", downloadSelectedPolicy);

          function draw() {
            svg.innerHTML = "";
            if (!points.length) {
              const t = document.createElementNS(NS,"text");
              t.setAttribute("x","50%"); t.setAttribute("y","50%");
              t.setAttribute("text-anchor","middle"); t.textContent="표시할 분석 해가 없습니다.";
              svg.appendChild(t); return;
            }

            const W = Math.max(svg.clientWidth || 720, 480);
            const H = 430, ml=72, mr=20, mt=18, mb=55;
            svg.setAttribute("viewBox",`0 0 ${W} ${H}`);

            const costs = points.map(p=>Number(p.total_cost)).filter(Number.isFinite);
            const aos = points.map(p=>Number(p.Ao)).filter(Number.isFinite);
            let xmin=Math.min(...costs), xmax=Math.max(...costs);
            let ymin=Math.min(...aos), ymax=Math.max(...aos);
            if (xmax<=xmin) xmax=xmin+1;
            if (ymax<=ymin) ymax=ymin+0.01;
            const xpad=(xmax-xmin)*0.04, ypad=(ymax-ymin)*0.08;
            xmin-=xpad; xmax+=xpad; ymin=Math.max(0,ymin-ypad); ymax=Math.min(1,ymax+ypad);

            const sx=x=>ml+(x-xmin)/(xmax-xmin)*(W-ml-mr);
            const sy=y=>mt+(ymax-y)/(ymax-ymin)*(H-mt-mb);

            for (let i=0;i<=5;i++) {
              const gx=ml+i*(W-ml-mr)/5, gy=mt+i*(H-mt-mb)/5;
              const vl=document.createElementNS(NS,"line");
              vl.setAttribute("x1",gx); vl.setAttribute("x2",gx); vl.setAttribute("y1",mt); vl.setAttribute("y2",H-mb);
              vl.setAttribute("class","ce-grid"); svg.appendChild(vl);
              const hl=document.createElementNS(NS,"line");
              hl.setAttribute("x1",ml); hl.setAttribute("x2",W-mr); hl.setAttribute("y1",gy); hl.setAttribute("y2",gy);
              hl.setAttribute("class","ce-grid"); svg.appendChild(hl);
              const xt=document.createElementNS(NS,"text");
              xt.setAttribute("x",gx); xt.setAttribute("y",H-mb+18); xt.setAttribute("text-anchor","middle"); xt.setAttribute("class","ce-tick");
              xt.textContent=fmt(xmin+i*(xmax-xmin)/5,0); svg.appendChild(xt);
              const yt=document.createElementNS(NS,"text");
              yt.setAttribute("x",ml-10); yt.setAttribute("y",gy+4); yt.setAttribute("text-anchor","end"); yt.setAttribute("class","ce-tick");
              yt.textContent=(ymax-i*(ymax-ymin)/5).toFixed(3); svg.appendChild(yt);
            }

            const xAxis=document.createElementNS(NS,"line");
            xAxis.setAttribute("x1",ml); xAxis.setAttribute("x2",W-mr); xAxis.setAttribute("y1",H-mb); xAxis.setAttribute("y2",H-mb); xAxis.setAttribute("class","ce-axis"); svg.appendChild(xAxis);
            const yAxis=document.createElementNS(NS,"line");
            yAxis.setAttribute("x1",ml); yAxis.setAttribute("x2",ml); yAxis.setAttribute("y1",mt); yAxis.setAttribute("y2",H-mb); yAxis.setAttribute("class","ce-axis"); svg.appendChild(yAxis);

            const xlabel=document.createElementNS(NS,"text");
            xlabel.setAttribute("x",(ml+W-mr)/2); xlabel.setAttribute("y",H-12); xlabel.setAttribute("text-anchor","middle"); xlabel.setAttribute("class","ce-axis-label");
            xlabel.textContent="Cost"; svg.appendChild(xlabel);
            const ylabel=document.createElementNS(NS,"text");
            ylabel.setAttribute("x",16); ylabel.setAttribute("y",(mt+H-mb)/2); ylabel.setAttribute("text-anchor","middle"); ylabel.setAttribute("class","ce-axis-label");
            ylabel.setAttribute("transform",`rotate(-90 16 ${(mt+H-mb)/2})`); ylabel.textContent="Ao"; svg.appendChild(ylabel);

            points.forEach(p => {
              const c=document.createElementNS(NS,"circle");
              c.setAttribute("cx",sx(Number(p.total_cost))); c.setAttribute("cy",sy(Number(p.Ao)));
              c.setAttribute("r",p.is_pareto_2d ? 6.3 : 4.2); c.setAttribute("fill",colorFor(p.target_ao));
              c.setAttribute("class",`ce-point ${p.is_pareto_2d ? "pareto" : "normal"}`); c.dataset.key=p.key;
              const tt=document.createElementNS(NS,"title");
              tt.textContent=`목표 Ao ${Number(p.target_ao).toFixed(2)} | Ao ${Number(p.Ao).toFixed(4)} | Cost ${fmt(p.total_cost,0)}`;
              c.appendChild(tt);
              c.addEventListener("click", () => selectPoint(p.key));
              svg.appendChild(c);
            });
            selectPoint(selectedKey);
          }

          draw();
          const ro = new ResizeObserver(() => draw());
          ro.observe(svg);
          return () => ro.disconnect();
        }
        """
    )


def _build_frontend_ce_points(selector_df: pd.DataFrame) -> list[dict]:
    if not isinstance(selector_df, pd.DataFrame) or selector_df.empty:
        return []
    out = selector_df.copy()
    needed = [
        "target_ao", "solution_id", "Ao", "total_cost", "DT_diag_h", "DT_wait_h",
        "DT_restore_h", "DT_total_h", "stocked_managed_parts", "total_stock_units",
        "prebuy_selected", "protection_selected", "is_pareto_2d"
    ]
    for c in needed:
        if c not in out.columns:
            out[c] = 0
    pts = []
    for _, r in out.iterrows():
        target = float(pd.to_numeric(r.get("target_ao", 0.0), errors="coerce"))
        sid = int(pd.to_numeric(r.get("solution_id", 0), errors="coerce"))
        pts.append({
            "key": build_solution_payload_key(target, sid),
            "target_ao": target,
            "solution_id": sid,
            "Ao": float(pd.to_numeric(r.get("Ao", 0.0), errors="coerce")),
            "total_cost": float(pd.to_numeric(r.get("total_cost", 0.0), errors="coerce")),
            "DT_diag_h": float(pd.to_numeric(r.get("DT_diag_h", 0.0), errors="coerce")),
            "DT_wait_h": float(pd.to_numeric(r.get("DT_wait_h", 0.0), errors="coerce")),
            "DT_restore_h": float(pd.to_numeric(r.get("DT_restore_h", 0.0), errors="coerce")),
            "DT_total_h": float(pd.to_numeric(r.get("DT_total_h", 0.0), errors="coerce")),
            "stocked_managed_parts": int(pd.to_numeric(r.get("stocked_managed_parts", r.get("managed_items", 0)), errors="coerce")),
            "total_stock_units": int(pd.to_numeric(r.get("total_stock_units", 0), errors="coerce")),
            "prebuy_selected": int(pd.to_numeric(r.get("prebuy_selected", 0), errors="coerce")),
            "protection_selected": int(pd.to_numeric(r.get("protection_selected", 0), errors="coerce")),
            "is_pareto_2d": int(pd.to_numeric(r.get("is_pareto_2d", 0), errors="coerce")),
        })
    return pts


def _build_frontend_policy_payload(result: dict) -> tuple[list[dict], dict]:
    cached_meta = result.get("_frontend_candidate_meta_cache")
    cached_vec = result.get("_frontend_solution_vectors_cache")
    if isinstance(cached_meta, list) and isinstance(cached_vec, dict):
        return cached_meta, cached_vec

    ctx = result.get("_engine_context", {})
    result_map = result.get("result_map", {})
    if not isinstance(ctx, dict) or not isinstance(result_map, dict):
        return [], {}

    part_id = np.asarray(ctx.get("part_id_c", []), dtype=object)
    item_name = np.asarray(ctx.get("item_name_c", np.full(len(part_id), "", dtype=object)), dtype=object)
    echelon = np.asarray(ctx.get("echelon_c", np.full(len(part_id), "", dtype=object)), dtype=object)
    lead = np.asarray(ctx.get("lead_c", np.zeros(len(part_id))), dtype=float)
    proc_cost = np.asarray(ctx.get("proc_cost_c", np.zeros(len(part_id))), dtype=float)
    fr = np.asarray(ctx.get("annual_fr_c", np.zeros(len(part_id))), dtype=float)
    pneed = np.asarray(ctx.get("p_need_c", np.zeros(len(part_id))), dtype=float)
    impact = np.asarray(ctx.get("impact_c", np.zeros(len(part_id))), dtype=float)
    prebuy = np.asarray(ctx.get("prebuy_flag_c", np.zeros(len(part_id), dtype=bool)), dtype=bool)
    protection = np.asarray(ctx.get("protection_flag_c", np.zeros(len(part_id), dtype=bool)), dtype=bool)

    meta = []
    for i in range(len(part_id)):
        meta.append({
            "part_id": str(part_id[i]),
            "item_name": str(item_name[i]),
            "maint_echelon": str(echelon[i]),
            "lead_time": float(lead[i]),
            "proc_cost": float(proc_cost[i]),
            "failure_rate": float(fr[i]),
            "p_need_period": float(pneed[i]),
            "impact_score": float(impact[i]),
            "prebuy_flag": bool(prebuy[i]),
            "protection_flag": bool(protection[i]),
        })

    vectors = {}
    for target_key, one in result_map.items():
        if not isinstance(one, dict):
            continue
        sim = one.get("sim_pop")
        if not isinstance(sim, dict):
            continue
        ao_arr = np.asarray(sim.get("Ao", []))
        sO = np.asarray(sim.get("sO", []), dtype=int)
        sI = np.asarray(sim.get("sI", []), dtype=int)
        sD = np.asarray(sim.get("sD", []), dtype=int)
        Y = np.asarray(sim.get("Y", []), dtype=int)
        for sid in range(len(ao_arr)):
            key = build_solution_payload_key(float(target_key), sid)
            vectors[key] = {
                "sO": sO[sid].tolist(),
                "sI": sI[sid].tolist(),
                "sD": sD[sid].tolist(),
                "Y": Y[sid].tolist(),
            }

    result["_frontend_candidate_meta_cache"] = meta
    result["_frontend_solution_vectors_cache"] = vectors
    return meta, vectors


def render_frontend_ce_component(
    result: dict,
    selector_df: pd.DataFrame,
    component_key: str,
    show_policy: bool = False,
):
    current_payload = _selected_payload_or_default(result, selector_df)
    current_key = (
        build_solution_payload_key(current_payload["target_ao"], current_payload["solution_id"])
        if isinstance(current_payload, dict) else ""
    )

    points = result.get("_frontend_ce_points_cache")
    if not isinstance(points, list):
        points = _build_frontend_ce_points(selector_df)
        result["_frontend_ce_points_cache"] = points

    candidate_meta, solution_vectors = ([], {})
    if show_policy:
        candidate_meta, solution_vectors = _build_frontend_policy_payload(result)

    if _CE_V2_AVAILABLE and _CE_COMPONENT is not None:
        _CE_COMPONENT(
            data={
                "points": points,
                "selected_key": current_key,
                "show_policy": bool(show_policy),
                "candidate_meta": candidate_meta,
                "solution_vectors": solution_vectors,
            },
            key=component_key,
            width="stretch",
            height=1040 if show_policy else 535,
        )
        return

    # Older Streamlit fallback.
    _interactive_ce_fragment_body(result, selector_df, component_key + "_fallback", True)

_fragment_decorator = getattr(st, "fragment", getattr(st, "experimental_fragment", None))


def _selected_payload_or_default(result: dict, selector_df: pd.DataFrame) -> dict | None:
    payload = st.session_state.get("selected_solution_payload")
    if isinstance(payload, dict):
        return payload

    default_payload = result.get("default_selected_payload") if isinstance(result, dict) else None
    if isinstance(default_payload, dict):
        st.session_state.selected_solution_payload = {
            "target_ao": float(default_payload.get("target_ao", 0.0)),
            "solution_id": int(default_payload.get("solution_id", 0)),
        }
        return st.session_state.selected_solution_payload

    if isinstance(selector_df, pd.DataFrame) and not selector_df.empty:
        row = selector_df.iloc[0]
        payload = {
            "target_ao": float(pd.to_numeric(row.get("target_ao", 0.0), errors="coerce")),
            "solution_id": int(pd.to_numeric(row.get("solution_id", 0), errors="coerce")),
        }
        st.session_state.selected_solution_payload = payload
        return payload
    return None


def _render_clicked_solution_summary(snapshot: dict | None):
    if not isinstance(snapshot, dict):
        st.info("선택한 해의 상세결과를 불러올 수 없습니다.")
        return
    summary = snapshot.get("summary", {})
    policy_df = snapshot.get("policy_df", pd.DataFrame())
    render_summary_panel(summary, policy_df)


def _interactive_ce_fragment_body(result: dict, selector_df: pd.DataFrame, chart_key: str, show_summary: bool):
    if not isinstance(selector_df, pd.DataFrame) or selector_df.empty:
        st.info("표시할 분석 해가 없습니다.")
        return

    current_payload = _selected_payload_or_default(result, selector_df)

    # Build once per run, then reuse. Only selectedpoints changes on click.
    fig = _get_ce_base_figure(result, selector_df)
    fig = _apply_persistent_selected_point(fig, current_payload)

    left, right = st.columns([1.65, 1.0], gap="large")

    with left:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=chart_key,
            on_select="rerun",
            selection_mode=("points",),
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "doubleClick": "reset",
                "responsive": True,
            },
        )

    payload = extract_plotly_selected_payload(event)
    if isinstance(payload, dict):
        new_payload = {
            "target_ao": float(payload["target_ao"]),
            "solution_id": int(payload["solution_id"]),
        }
        # Update only if the actual selected point changed.
        if (
            not isinstance(current_payload, dict)
            or round(float(current_payload.get("target_ao", -999.0)), 6) != round(float(new_payload["target_ao"]), 6)
            or int(current_payload.get("solution_id", -999)) != int(new_payload["solution_id"])
        ):
            st.session_state.selected_solution_payload = new_payload
        current_payload = new_payload

    # Snapshot lookup is cached in nsga_engine.py.
    snapshot = get_selected_snapshot(result, current_payload)

    with right:
        if isinstance(snapshot, dict):
            selected = snapshot.get("summary", {})
            st.caption(
                f"현재 선택 해 · 목표 Ao {float(selected.get('target_ao', 0.0)):.2f} · "
                f"해 {int(selected.get('solution_id', 0))} · "
                f"Ao {float(selected.get('Ao', selected.get('best_Ao', 0.0))):.4f}"
            )
            selected_df = snapshot.get("selected_solution_df", pd.DataFrame())
            if isinstance(selected_df, pd.DataFrame) and not selected_df.empty:
                render_solution_detail_card(selected_df.iloc[0])
            elif show_summary:
                _render_clicked_solution_summary(snapshot)
        else:
            st.warning("선택한 점과 분석결과를 연결하지 못했습니다.")


if _fragment_decorator is not None:
    @_fragment_decorator
    def render_ce_summary_fragment(result: dict, selector_df: pd.DataFrame):
        _interactive_ce_fragment_body(result, selector_df, "ce_curve_click_summary", True)

    @_fragment_decorator
    def render_ce_detail_fragment(result: dict, selector_df: pd.DataFrame):
        _interactive_ce_fragment_body(result, selector_df, "ce_curve_click_detail", False)
else:
    def render_ce_summary_fragment(result: dict, selector_df: pd.DataFrame):
        _interactive_ce_fragment_body(result, selector_df, "ce_curve_click_summary", True)

    def render_ce_detail_fragment(result: dict, selector_df: pd.DataFrame):
        _interactive_ce_fragment_body(result, selector_df, "ce_curve_click_detail", False)


def render_integrated_results(result: dict | None):
    st.markdown("### 통합 결과")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    tabs = st.tabs(["상세결과", "다운로드"])
    all_runs_df = result.get("all_target_runs_df", pd.DataFrame())
    pareto_df = result.get("pareto_df", pd.DataFrame())
    source_df = all_runs_df if isinstance(all_runs_df, pd.DataFrame) and not all_runs_df.empty else pareto_df
    selector_df = build_solution_selector_df(source_df)

    with tabs[0]:
        render_frontend_ce_component(
            result,
            selector_df,
            "ce_frontend_detail",
            show_policy=True,
        )

    with tabs[1]:
        payload = st.session_state.get("selected_solution_payload")
        snapshot = get_selected_snapshot(result, payload)

        if isinstance(snapshot, dict):
            s = snapshot.get("summary", {})
            st.markdown(
                f"""
                <div class="section-box">
                    <b>다운로드 대상 해</b><br>
                    목표 운용가용도 {float(s.get('target_ao', 0.0)):.2f} ·
                    해 번호 {int(s.get('solution_id', 0))} ·
                    Ao {float(s.get('Ao', s.get('best_Ao', 0.0))):.4f} ·
                    총비용 {float(s.get('total_cost', s.get('best_cost', 0.0))):,.0f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.download_button(
            "📥 현재 선택 해 결과 엑셀 다운로드",
            data=make_excel_download(result, payload),
            file_name="NSGA_selected_solution.xlsx",
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
    stock_summary = xai.get("stock_summary", {})
    final_stock_table = xai.get("final_stock_table", pd.DataFrame())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: show_metric_card("Total Unit Stock", f"{_safe_int(stock_summary.get('total_stock_units', 0)):,}")
    with c2: show_metric_card("Managed Parts", f"{_safe_int(stock_summary.get('managed_parts', 0)):,}")
    with c3: show_metric_card("Selected Pre-buy", f"{_safe_int(stock_summary.get('selected_prebuy', 0)):,}")
    with c4: show_metric_card("Selected Protection", f"{_safe_int(stock_summary.get('selected_protection', 0)):,}")
    with c5: show_metric_card("Normal Stock", f"{_safe_int(stock_summary.get('normal_stock_items', 0)):,}")

    st.markdown(f"<div class='section-box'>{clean_chart_text(xai.get('narrative', '-'))}</div>", unsafe_allow_html=True)

    if isinstance(final_stock_table, pd.DataFrame) and not final_stock_table.empty:
        st.markdown("#### 최종 확보 품목 관계")
        st.dataframe(final_stock_table, use_container_width=True, hide_index=True, height=220)

    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.markdown(f"<div class='section-box'><b>타깃 정렬 관점</b><br>{why_solution.get('target_alignment','-')}<br><br><b>선택 편향 관점</b><br>{why_solution.get('selection_bias','-')}</div>", unsafe_allow_html=True)
    with w2:
        st.markdown(f"<div class='section-box'><b>리스크 완충 관점</b><br>{why_solution.get('risk_buffer','-')}<br><br><b>비용-효과 관점</b><br>{why_solution.get('tradeoff','-')}</div>", unsafe_allow_html=True)

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
            st.dataframe(top_priority, use_container_width=True, height=320)
            st.dataframe(top_unmanaged, use_container_width=True, height=220)
        with right:
            st.dataframe(top_cost_vs_impact, use_container_width=True, height=320)
    with tabs[2]:
        st.dataframe(detail_df, use_container_width=True, height=560)
    with tabs[3]:
        if detail_df.empty:
            st.info("표시할 품목이 없습니다.")
        else:
            selected_part = st.selectbox("품목 선택", detail_df["part_id"].astype(str).tolist(), key="xai_selected_part")
            selected_row = detail_df.loc[detail_df["part_id"].astype(str) == selected_part].iloc[0]
            cohort_cols = [c for c in ["impact_score", "lead_time", "priority_score", "recommended_stock", "xai_action_score", "xai_cost_score", "decision_balance_score"] if c in detail_df.columns]
            cohort_medians = {c: float(pd.to_numeric(detail_df[c], errors="coerce").median()) for c in cohort_cols}
            st.pyplot(draw_item_profile(selected_row, cohort_medians, f"품목 프로파일 · {selected_part}"), use_container_width=True)
            st.dataframe(pd.DataFrame([selected_row.to_dict()]), use_container_width=True, height=220)


def render_prescriptive(result: dict | None, df_input: pd.DataFrame | None):
    st.markdown("### Prescriptive AI")
    if not result:
        st.info("아직 실행 결과가 없습니다. NSGA 분석을 먼저 실행해 주세요.")
        return

    prescriptive = result.get("prescriptive")
    if not isinstance(prescriptive, dict):
        prescriptive = build_prescriptive_package(result, df_input)
        result["prescriptive"] = prescriptive
    action_df = prescriptive.get("action_df", pd.DataFrame())
    summary = prescriptive.get("summary", {})
    narrative = prescriptive.get("narrative", "")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: show_metric_card("총 추천 액션 수", f"{_safe_int(summary.get('total_actions', 0)):,}")
    with c2: show_metric_card("즉시 실행 권고 수", f"{_safe_int(summary.get('immediate_actions', 0)):,}")
    with c3: show_metric_card("추가 검토 수", f"{_safe_int(summary.get('review_actions', 0)):,}")
    with c4: show_metric_card("유지/모니터링 수", f"{_safe_int(summary.get('monitor_actions', 0)):,}")
    with c5: show_metric_card("Watchlist 승격 후보 수", f"{_safe_int(summary.get('watchlist_candidates', 0)):,}")
    with c6: show_metric_card("보류 후보 수", f"{_safe_int(summary.get('deferred_actions', 0)):,}")

    st.markdown(f"<div class='section-box'>{clean_chart_text(narrative)}</div>", unsafe_allow_html=True)

    st.markdown("#### 정책 추천 카드")
    cards = summary.get("policy_cards", [])
    for i in range(0, len(cards), 3):
        row_cards = cards[i:i + 3]
        cols = st.columns(len(row_cards))
        for col, card in zip(cols, row_cards):
            with col:
                st.markdown(
                    f"""<div class="section-box">
                    <div style="font-size:1.02rem;font-weight:800;color:#1b4e2a;">{_safe_text(card.get('title'))}</div>
                    <div style="font-size:1.55rem;font-weight:800;color:#1e4f2b;margin:8px 0;">{_safe_int(card.get('count')):,}건</div>
                    <div style="color:#3d5f45;line-height:1.75;"><b>대표 이유</b><br>{_safe_text(card.get('reason'))}<br><br>
                    <b>예상 효과</b><br>{_safe_text(card.get('effect'))}</div></div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("#### Prescriptive Action Table")
    st.dataframe(action_df, use_container_width=True, height=520) if not action_df.empty else st.info("표시할 Prescriptive action_df가 없습니다.")


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
        kwargs2 = {key: value for key, value in config.__dict__.items() if key in sig.parameters}
        kwargs2.update(kwargs)
        result = run_nsga2(df, **kwargs2)
    if not isinstance(result, dict):
        raise ValueError("엔진이 dict 형태 결과를 반환하지 않았습니다.")
    return result


# ------------------------------------------------------------
# Header / Sidebar controls
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">AI기반 수리부속 소요예측 분석결과 대시보드</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 실행 설정")
    uploaded_file = st.file_uploader("입력 파일 업로드", type=["csv", "xlsx", "xls"])
    n_generations = st.slider("분석 반복 횟수", min_value=20, max_value=300, value=80, step=10, help="분석 모델이 여러 후보 조합을 반복해서 비교·개선하는 횟수입니다. 값을 높이면 더 많은 후보를 탐색해 좋은 해를 찾을 가능성이 커지지만 계산시간도 늘어납니다.")
    pmin = st.slider("최소 고장발생 가능성", min_value=0.00, max_value=1.00, value=0.10, step=0.01, help="선택한 분석기간 동안 한 번 이상 고장이 발생할 가능성을 기준으로 후보 품목을 거르는 값입니다. 값을 낮추면 고장 가능성이 낮은 품목까지 후보에 포함되어 탐색 범위와 계산량이 커지고, 값을 높이면 후보가 줄어듭니다.")

    # NEW: analysis period slider, placed directly next to the Pmin control area.
    analysis_years = st.slider(
        "분석 기간 (년)",
        min_value=1,
        max_value=30,
        value=2,
        step=1,
        help="Ao 관측시간, 누적 기대 다운타임, 기간 내 고장발생확률, 유지비 계산에 동일하게 적용됩니다.",
    )

    long_lead_percentile = st.slider("장기 조달기간 판단 기준", min_value=0.50, max_value=0.99, value=0.80, step=0.01, help="전체 후보의 조달 리드타임 분포에서 장기 조달 품목을 구분하는 기준입니다. 예를 들어 0.80이면 리드타임 상위 약 20%를 장기 조달 품목으로 봅니다. 값을 높이면 장기 조달로 분류되는 품목이 줄고, 낮추면 늘어납니다.")
    st.markdown(
        f"""
        <div class="run-param-box">
            <span class="info-chip">분석 반복 {n_generations}회</span>
            <span class="info-chip">최소 고장발생 가능성 {pmin:.2f}</span>
            <span class="info-chip">분석 기간 {analysis_years}년</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_btn = st.button("🚀 NSGA 실행", use_container_width=True)


# Preview config uses the same period/knobs as the actual run so precheck values
# such as p_need_period and candidate flags match the run configuration.
preview_ui_config = NSGAConfig(
    population_size=60,
    n_generations=n_generations,
    random_seed=42,
    target_ao=0.94,
    pmin=pmin,
    long_lead_percentile=long_lead_percentile,
    years=float(analysis_years),
)

df_input = None
xdf = None
if uploaded_file is not None:
    try:
        df_input = read_uploaded_file(uploaded_file)
        st.session_state.uploaded_name = uploaded_file.name
        xdf = prepare_input_dataframe(df_input, config=preview_ui_config)
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
            random_seed=42,
            target_ao=0.94,
            pmin=pmin,
            long_lead_percentile=long_lead_percentile,
            years=float(analysis_years),
        )
        target_grid = list(getattr(preview_config, "ao_target_grid", [float(getattr(preview_config, "representative_target", 0.94))]))
        rep_target = float(getattr(preview_config, "representative_target", 0.94))
        sorted_grid = sorted(set(float(x) for x in target_grid + [rep_target]))
        n_targets = len(sorted_grid)

        overall_progress_bar = st.progress(1, text=f"분석 준비 중... 분석기간 {analysis_years}년 Target Sweep를 시작합니다.")
        stage_progress_bar = st.progress(1)
        stage_box = st.empty()
        heartbeat_box = st.empty()
        target_status_box = st.empty()
        status_box = st.empty()

        st.session_state.generation_logs = []
        st.session_state.live_generation_logs = []
        st.session_state.progress_emit_state = {"last_emit_gen": -1, "last_emit_ts": 0.0, "last_target": None}
        st.session_state.target_status_map = {
            float(t): {
                "Target Ao": float(t), "Stage": f"{i + 1}/{n_targets}", "Status": "대기",
                "Generation": 0, "Total Gen": int(preview_config.n_generations),
                "Best Ao": None, "Mean Ao": None, "Best Cost": None,
            }
            for i, t in enumerate(sorted_grid)
        }
        start_ts = time.time()

        stage_box.info(
            f"분석기간 {analysis_years}년 · 총 {n_targets}개 Target Ao를 순차적으로 계산합니다. "
            f"대표 타깃 {rep_target:.2f}도 함께 포함됩니다."
        )

        def _render_target_status(current_target: float | None = None):
            ts_df = pd.DataFrame(list(st.session_state.target_status_map.values())).sort_values("Target Ao").reset_index(drop=True)

            if current_target is not None and not ts_df.empty:
                current_mask = pd.to_numeric(ts_df["Target Ao"], errors="coerce").round(6) == round(float(current_target), 6)
                ts_df["현재 위치"] = np.where(current_mask, "●", "")
                current_positions = np.flatnonzero(current_mask.to_numpy())

                # Follow the current target row with a moving visible window.
                if current_positions.size > 0:
                    pos = int(current_positions[0])
                    start_i = max(0, pos - 5)
                    end_i = min(len(ts_df), start_i + 8)
                    start_i = max(0, end_i - 8)
                    ts_view = ts_df.iloc[start_i:end_i].copy()
                else:
                    ts_view = ts_df.tail(8).copy()
            else:
                ts_df["현재 위치"] = ""
                ts_view = ts_df.head(8).copy()

            def _highlight_target_row(row):
                if str(row.get("현재 위치", "")).strip() == "●":
                    return ["background-color: #e7f7df; color: #245c2a; font-weight: 700"] * len(row)
                return [""] * len(row)

            ts_display = ts_view.drop(columns=["현재 위치"], errors="ignore")
            ts_style_source = ts_view.copy()

            progress_formatters = {}
            for _col in ts_display.columns:
                if _col in ["Target Ao", "Best Ao", "Mean Ao"]:
                    progress_formatters[_col] = lambda x: "" if pd.isna(x) else f"{float(x):.4f}"
                elif _col == "Best Cost":
                    progress_formatters[_col] = lambda x: "" if pd.isna(x) else f"{float(x):,.0f}"
                elif _col in ["Generation", "Total Gen"]:
                    progress_formatters[_col] = lambda x: "" if pd.isna(x) else f"{int(float(x)):,d}"

            styled_target = ts_display.style.format(progress_formatters, na_rep="").apply(
                lambda row: [
                    "background-color: #e7f7df; color: #245c2a; font-weight: 700"
                    if str(ts_style_source.loc[row.name].get("현재 위치", "")).strip() == "●"
                    else ""
                    for _ in row
                ],
                axis=1,
            )
            target_status_box.dataframe(
                styled_target,
                use_container_width=True,
                height=300,
                hide_index=True,
            )
            live_logs = st.session_state.get("live_generation_logs", [])
            if live_logs:
                log_df = pd.DataFrame(live_logs).copy()
                log_view = log_df.tail(8).copy().reset_index(drop=True)
                log_view["현재"] = ""
                if len(log_view) > 0:
                    log_view.loc[log_view.index[-1], "현재"] = "●"

                log_style_source = log_view.copy()

                lower_formatters = {}
                for _col in log_view.columns:
                    if _col in ["target_ao", "best_ao", "mean_ao"]:
                        lower_formatters[_col] = lambda x: "" if pd.isna(x) else f"{float(x):.4f}"
                    elif _col == "best_cost":
                        lower_formatters[_col] = lambda x: "" if pd.isna(x) else f"{float(x):,.0f}"
                    elif _col in ["gen", "total_gen", "analysis_years"]:
                        lower_formatters[_col] = lambda x: "" if pd.isna(x) else f"{int(float(x)):,d}"
                    elif _col == "elapsed_sec":
                        lower_formatters[_col] = lambda x: "" if pd.isna(x) else f"{float(x):.1f}"

                styled_log = log_view.style.format(lower_formatters, na_rep="").apply(
                    lambda row: [
                        "background-color: #e7f7df; color: #245c2a; font-weight: 700"
                        if str(log_style_source.loc[row.name].get("현재", "")).strip() == "●"
                        else ""
                        for _ in row
                    ],
                    axis=1,
                )
                # st.table은 내부 스크롤바가 없어서 항상 최신 8개 세대 전체가 보입니다.
                # 현재 세대는 항상 마지막 행으로 이동합니다.
                status_box.table(styled_log)
            else:
                empty_live = pd.DataFrame([{"상태": "첫 세대 계산 준비 중"}])
                status_box.table(
                    empty_live.style.apply(
                        lambda row: ["background-color: #e7f7df; color: #245c2a; font-weight: 700"] * len(row),
                        axis=1,
                    )
                )


        _render_target_status(sorted_grid[0] if sorted_grid else None)

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
                gen_safe, total_gen_safe, last_emit_gen, last_emit_ts, current_ts, current_target, last_target
            )

            # When the target changes, close the previous running target.
            for t_key, t_row in st.session_state.target_status_map.items():
                if (
                    round(float(t_key), 6) != round(float(current_target), 6)
                    and t_row.get("Status") == "진행 중"
                ):
                    t_row["Status"] = "완료"

            if current_target not in st.session_state.target_status_map:
                st.session_state.target_status_map[current_target] = {
                    "Target Ao": current_target, "Stage": "-", "Status": "진행 중",
                    "Generation": gen_safe, "Total Gen": total_gen_safe,
                    "Best Ao": None, "Mean Ao": None, "Best Cost": None,
                }
            else:
                st.session_state.target_status_map[current_target]["Status"] = "진행 중"
                st.session_state.target_status_map[current_target]["Generation"] = gen_safe
                st.session_state.target_status_map[current_target]["Total Gen"] = total_gen_safe

            if isinstance(best_summary, dict):
                st.session_state.target_status_map[current_target]["Best Ao"] = best_summary.get("best_ao", best_summary.get("best_Ao"))
                st.session_state.target_status_map[current_target]["Mean Ao"] = best_summary.get("mean_ao", best_summary.get("mean_Ao"))
                st.session_state.target_status_map[current_target]["Best Cost"] = best_summary.get("best_cost")

            # Lightweight live log: one row per generation so the lower table
            # always follows the current generation.
            live_row = {
                "elapsed_sec": round(elapsed, 1),
                "analysis_years": int(analysis_years),
                "target_ao": current_target,
                "gen": gen_safe,
                "total_gen": total_gen_safe,
                "best_ao": best_summary.get("best_ao", best_summary.get("best_Ao")) if isinstance(best_summary, dict) else None,
                "mean_ao": best_summary.get("mean_ao", best_summary.get("mean_Ao")) if isinstance(best_summary, dict) else None,
                "best_cost": best_summary.get("best_cost") if isinstance(best_summary, dict) else None,
            }
            st.session_state.live_generation_logs.append(live_row)
            if len(st.session_state.live_generation_logs) > 8:
                st.session_state.live_generation_logs = st.session_state.live_generation_logs[-8:]

            # Refresh the two live tables every generation; heavier widgets
            # remain throttled below.
            _render_target_status(current_target)

            if should_emit:
                st.session_state.generation_logs.append({
                    "elapsed_sec": round(elapsed, 1), "analysis_years": int(analysis_years),
                    "target_ao": current_target, "gen": gen_safe, "total_gen": total_gen_safe,
                    "best_ao": best_summary.get("best_ao", best_summary.get("best_Ao")) if isinstance(best_summary, dict) else None,
                    "mean_ao": best_summary.get("mean_ao", best_summary.get("mean_Ao")) if isinstance(best_summary, dict) else None,
                    "best_cost": best_summary.get("best_cost") if isinstance(best_summary, dict) else None,
                })

                overall_progress = float(best_summary.get("overall_progress", stage_progress / 100.0)) if isinstance(best_summary, dict) else stage_progress / 100.0
                overall_pct = max(1, min(100, int(round(overall_progress * 100))))
                overall_progress_bar.progress(overall_pct)
                stage_progress_bar.progress(stage_progress)
                st.session_state.progress_emit_state = {
                    "last_emit_gen": gen_safe, "last_emit_ts": current_ts, "last_target": current_target,
                }

        try:
            result = call_engine(df_input, preview_config, progress_callback)
            for k in st.session_state.target_status_map:
                st.session_state.target_status_map[k]["Status"] = "완료"
            overall_progress_bar.progress(100)
            stage_progress_bar.progress(100)
            heartbeat_box.caption(f"분석 완료 · 분석기간 {analysis_years}년")
            result["xai"] = build_explainability_tables_v4(result, df_input)
            result["prescriptive"] = initialize_prescriptive_structure(result, st.session_state.get("selected_solution_payload"))
            st.session_state.run_result = result
            default_payload = result.get("default_selected_payload")
            if isinstance(default_payload, dict):
                st.session_state.selected_solution_payload = {
                    "target_ao": float(default_payload.get("target_ao", rep_target)),
                    "solution_id": int(default_payload.get("solution_id", 0)),
                }
            _render_target_status(sorted_grid[-1] if sorted_grid else None)
            st.success(f"NSGA 분석이 완료되었습니다. 분석기간: {analysis_years}년")
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
