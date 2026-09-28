# -*- coding: utf-8 -*-
from __future__ import annotations

APP_BUILD_VERSION = "SUPPLY_DECISION_DASHBOARD_V5_20260928"

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
import plotly.graph_objects as go
import streamlit as st

from nsga_engine import (
    NSGAConfig,
    prepare_input_dataframe,
    run_nsga2,
    validate_stage1_config,
)
from simulation_engine import (
    MonteCarloConfig,
    run_coldstart_monte_carlo,
    validate_monte_carlo_config,
)
from decision_engine import (
    DecisionConfig,
    build_decision_package,
    validate_decision_config,
)
from audit_engine import (
    AuditConfig,
    build_final_audit_package,
    validate_audit_config,
)


# ------------------------------------------------------------
# Font / chart helpers
# ------------------------------------------------------------
def setup_korean_matplotlib_font() -> None:
    candidates = [
        "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "AppleGothic",
        "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    picked = next((x for x in candidates if x in installed), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = picked
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.max_open_warning"] = 0


def clean_chart_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    for old, new in {
        "\ufffd": "", "ㅁㅁ": "", "�": "", "□": "", "◻": "",
        "◼": "", "■": "", "\n": " ", "\r": " ", "\t": " ",
    }.items():
        s = s.replace(old, new)
    return " ".join(s.split())


setup_korean_matplotlib_font()

STREAMLIT_FRAGMENT_AVAILABLE = hasattr(st, "fragment")
_fragment = st.fragment if STREAMLIT_FRAGMENT_AVAILABLE else (lambda f: f)


# ------------------------------------------------------------
# Page / style
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
[data-testid="stMarkdownContainer"], [data-testid="stDataFrame"],
.stTabs, .stButton > button, .stDownloadButton > button, .stAlert,
.stMetric, label, p, h1, h2, h3, h4, h5 {
    font-family: "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo",
                 "NanumGothic", "Noto Sans KR", sans-serif !important;
}
.block-container {max-width:100% !important; padding-top:.8rem !important;}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#f5ffe9 0%,#eaf8db 55%,#e2f3d2 100%);
    border-right:1px solid #dceacc;
}
.hero-box {
    width:100%; background:#f7fbf3; border:1px solid #e3edd9;
    border-radius:18px; padding:18px 24px; margin:14px 0 16px 0;
    box-shadow:0 2px 10px rgba(60,90,60,.05); box-sizing:border-box;
}
.hero-title {font-size:1.9rem;font-weight:800;color:#1f4d2b;margin-bottom:6px;}
.hero-sub {font-size:.98rem;line-height:1.5;color:#5c735f;}
.metric-card {
    background:#fff;border:1px solid #e6efde;border-radius:16px;
    padding:14px 16px;box-shadow:0 2px 10px rgba(50,80,50,.05);min-height:108px;
}
.metric-label {color:#628164;font-size:.92rem;margin-bottom:8px;font-weight:600;}
.metric-value {color:#1e4f2b;font-size:1.6rem;font-weight:800;line-height:1.1;}
.section-box {
    background:#fff;border:1px solid #ebf2e4;border-radius:18px;
    padding:16px 18px;margin-bottom:14px;box-shadow:0 2px 8px rgba(50,80,50,.03);
}
.run-param-box {
    background:#f8fcf4;border:1px solid #e3edd9;border-radius:16px;
    padding:14px 16px;margin-bottom:14px;color:#36523c;line-height:1.7;
}
.info-chip {
    display:inline-block;padding:4px 10px;margin:0 6px 6px 0;border-radius:999px;
    background:#eef7e4;border:1px solid #d8e8c6;color:#2d5838;
    font-size:.85rem;font-weight:600;
}
.debug-box {
    background:#fffdf1;border:1px solid #efe2a2;border-radius:14px;
    padding:12px 14px;color:#665200;margin-bottom:10px;
}
.stage1-box {
    background:#f2f8ff;border:1px solid #cfe0f5;border-radius:14px;
    padding:12px 14px;color:#25496b;margin-bottom:10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
defaults = {
    "run_result": None,
    "uploaded_name": None,
    "generation_logs": [],
    "target_status_map": {},
    "precheck_info": {},
    "progress_emit_state": {"last_emit_gen": -1, "last_emit_ts": 0.0, "last_target": None},
    "selected_solution_payload": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
        pd.DataFrame([result.get("summary", {})]).to_excel(writer, sheet_name="Summary", index=False)
        for key, sheet in [
            ("prepared_df", "Prepared_Data"),
            ("candidate_df", "Candidates"),
            ("history_df", "Generation_History"),
            ("pareto_df", "Pareto_Solutions"),
            ("policy_df", "Best_Policy"),
            ("sweep_summary_df", "Sweep_Summary"),
            ("all_target_runs_df", "All_Target_Runs"),
            ("final_spares_df", "Final_Spares"),
            ("repairable_audit_df", "Repairable_Audit"),
            ("diagnostic_df", "Diagnostics"),
            ("stage1_readiness_df", "Stage1_Readiness"),
            ("stage1_config_df", "Stage1_Config"),
            ("robust_scenario_summary_df", "ModelC_Robust"),
            ("robust_scenario_detail_df", "ModelC_K_Scenarios"),
            ("representative_alternatives_df", "Representative_Alternatives"),
            ("representative_inventory_df", "Representative_Inventory"),
            ("monte_carlo_bridge_df", "MC_Bridge"),
            ("representative_selection_audit_df", "Representative_Audit"),
            ("mc_summary_df", "MC_Summary"),
            ("mc_iteration_df", "MC_Iterations"),
            ("mc_part_summary_df", "MC_Part_Summary"),
            ("mc_config_df", "MC_Config"),
            ("decision_table_df", "Decision_Table"),
            ("decision_positioning_df", "Decision_Positioning"),
            ("decision_grade_audit_df", "Decision_Grade_Audit"),
            ("decision_config_df", "Decision_Config"),
            ("audit_summary_df", "Final_Audit_Summary"),
            ("audit_checks_df", "Final_Audit_Checks"),
            ("audit_strategy_df", "Final_Strategy_View"),
            ("audit_part_risk_df", "Final_Part_Risk"),
            ("audit_config_df", "Final_Audit_Config"),
        ]:
            df = result.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
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


def build_precheck_info(df_input: pd.DataFrame, df_prepared: pd.DataFrame | None) -> dict:
    info = {
        "raw_shape": tuple(df_input.shape),
        "prepared_shape": tuple(df_prepared.shape) if isinstance(df_prepared, pd.DataFrame) else None,
        "summary_df": pd.DataFrame(),
        "preview_df": pd.DataFrame(),
    }
    if not isinstance(df_prepared, pd.DataFrame) or df_prepared.empty:
        return info

    check_cols = [
        "part_id", "failure_rate", "qpa", "annual_rad_hours", "lead_time",
        "repairable_flag", "replace_time_h", "repair_tat_h",
        "p_need_2y", "impact_score", "candidate",
    ]
    existing = [c for c in check_cols if c in df_prepared.columns]
    info["preview_df"] = df_prepared[existing].head(20).copy()

    rows = []
    for c in existing:
        s = df_prepared[c]
        if c == "part_id":
            rows.append({"col": c, "non_null": int(s.notna().sum()), "nunique": int(s.nunique(dropna=True))})
        else:
            n = pd.to_numeric(s, errors="coerce")
            rows.append({
                "col": c,
                "non_null": int(n.notna().sum()),
                "gt_zero": int(n.fillna(0).gt(0).sum()),
                "sum": float(n.fillna(0).sum()),
            })
    info["summary_df"] = pd.DataFrame(rows)
    return info


def get_progress_emit_interval(total_gen: int) -> int:
    if total_gen >= 400:
        return 38
    if total_gen >= 320:
        return 30
    if total_gen >= 240:
        return 20
    if total_gen >= 140:
        return 12
    return 2


def should_emit_progress_update(gen, total_gen, last_emit_gen, last_emit_ts, current_ts, current_target, last_target):
    if int(gen) <= 1 or int(gen) >= max(int(total_gen), 1):
        return True
    if last_target is None or round(float(current_target), 6) != round(float(last_target), 6):
        return True
    if current_ts - float(last_emit_ts) >= 1.9:
        return True
    return int(gen) - int(last_emit_gen) >= get_progress_emit_interval(max(int(total_gen), 1))


# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
def draw_histogram(series: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals):
        ax.hist(vals, bins=min(20, max(5, int(math.sqrt(len(vals))))))
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(series.name or ""))
    ax.set_ylabel("빈도")
    fig.tight_layout()
    return fig


def draw_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), alpha=.75)
    ax.set_title(clean_chart_text(title))
    ax.set_xlabel(clean_chart_text(x))
    ax.set_ylabel(clean_chart_text(y))
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def pick_cost_column(df: pd.DataFrame) -> str | None:
    for c in ["total_cost", "F1", "cost", "objective_cost", "proc_cost_total", "total_proc_cost"]:
        if c in df.columns:
            return c
    nums = [c for c in numeric_columns(df) if c != "Ao"]
    return nums[0] if nums else None


def build_solution_selector_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "target_ao" in out.columns:
        out["target_ao"] = pd.to_numeric(out["target_ao"], errors="coerce")
    if "solution_id" not in out.columns:
        out["solution_id"] = np.arange(len(out))
    if "is_pareto_2d" not in out.columns:
        out["is_pareto_2d"] = 0
    if "Ao" in out.columns:
        out["Ao"] = pd.to_numeric(out["Ao"], errors="coerce")
    return out


def build_ce_curve_plotly(df_runs: pd.DataFrame, title: str):
    fig = go.Figure()
    plot_df = build_solution_selector_df(df_runs)
    cost_col = pick_cost_column(plot_df)
    if plot_df.empty or cost_col is None or "Ao" not in plot_df.columns:
        fig.update_layout(title=title, xaxis_title="Cost", yaxis_title="Ao", height=620)
        return fig

    plot_df[cost_col] = pd.to_numeric(plot_df[cost_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[cost_col, "Ao"])
    targets = sorted(plot_df["target_ao"].dropna().unique()) if "target_ao" in plot_df.columns else [None]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#2E91E5"]

    for i, target in enumerate(targets):
        sub = plot_df if target is None else plot_df[plot_df["target_ao"] == target]
        if sub.empty:
            continue
        color = colors[i % len(colors)]
        custom = np.stack(
            [
                sub["target_ao"] if "target_ao" in sub.columns else np.zeros(len(sub)),
                sub["solution_id"],
            ], axis=1
        )
        fig.add_trace(go.Scatter(
            x=sub[cost_col], y=sub["Ao"], mode="markers",
            name="All Solutions" if target is None else f"Target {float(target):.2f}",
            customdata=custom,
            marker=dict(size=8, color=color, opacity=.55),
            hovertemplate="Target=%{customdata[0]:.2f}<br>Solution=%{customdata[1]}<br>Cost=%{x:,.2f}<br>Ao=%{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=clean_chart_text(title),
        xaxis_title=f"Cost ({cost_col})",
        yaxis_title="Ao",
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-.2),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.08)")
    return fig


# ------------------------------------------------------------
# Stage 1 UI
# ------------------------------------------------------------
def render_stage1_readiness(result: dict | None = None, config: NSGAConfig | None = None):
    if result:
        readiness = result.get("stage1_readiness", {})
        rdf = result.get("stage1_readiness_df", pd.DataFrame())
        cdf = result.get("stage1_config_df", pd.DataFrame())
    else:
        cfg = config or NSGAConfig()
        readiness = validate_stage1_config(cfg)
        rdf = pd.DataFrame()
        cdf = pd.DataFrame()

    status = "준비 완료" if readiness.get("stage1_ready") else "점검 필요"
    st.markdown(
        f"""
        <div class="stage1-box">
            <b>Stage 2 · Model C Robust 상태: {status}</b><br>
            Model C Robust는 실제 계산에 연결되어 있으며 K 시나리오 기반 Tail Risk와 Worst 지표를 산출합니다.<br>
            Monte Carlo 미래 시뮬레이션은 아직 계산에 연결하지 않았습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if readiness.get("issues"):
        for issue in readiness["issues"]:
            st.warning(issue)
    if isinstance(rdf, pd.DataFrame) and not rdf.empty:
        st.dataframe(rdf, use_container_width=True, hide_index=True, height=380)
    if isinstance(cdf, pd.DataFrame) and not cdf.empty:
        st.markdown("**향후 Model C / Monte Carlo 설정 인터페이스**")
        st.dataframe(cdf, use_container_width=True, hide_index=True, height=180)


# ------------------------------------------------------------
# Render helpers
# ------------------------------------------------------------
def render_preview_tabs(df_input: pd.DataFrame, df_prepared: pd.DataFrame | None):
    tabs = st.tabs(["원본 데이터", "전처리 데이터", "컬럼 요약", "기초 통계", "결측·유형 점검", "엔진 사전 점검", "Stage 1 준비상태"])
    with tabs[0]:
        st.dataframe(df_input, use_container_width=True, height=460)
    with tabs[1]:
        if isinstance(df_prepared, pd.DataFrame) and not df_prepared.empty:
            st.dataframe(df_prepared, use_container_width=True, height=460)
        else:
            st.info("전처리 데이터가 없습니다.")
    with tabs[2]:
        info_df = pd.DataFrame({
            "컬럼명": df_input.columns,
            "dtype": [str(df_input[c].dtype) for c in df_input.columns],
            "결측치 수": [int(df_input[c].isna().sum()) for c in df_input.columns],
            "고유값 수": [int(df_input[c].nunique(dropna=True)) for c in df_input.columns],
        })
        st.dataframe(info_df, use_container_width=True, height=460)
    with tabs[3]:
        try:
            st.dataframe(df_input.describe(include="all").transpose().reset_index(), use_container_width=True, height=460)
        except Exception:
            st.info("기초 통계를 생성할 수 없습니다.")
    with tabs[4]:
        if isinstance(df_prepared, pd.DataFrame) and not df_prepared.empty:
            comp = pd.DataFrame({
                "구분": ["원본", "전처리"],
                "행 수": [len(df_input), len(df_prepared)],
                "열 수": [df_input.shape[1], df_prepared.shape[1]],
                "결측치 수": [int(df_input.isna().sum().sum()), int(df_prepared.isna().sum().sum())],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)
        else:
            st.info("전처리 비교 데이터가 없습니다.")
    with tabs[5]:
        precheck = st.session_state.get("precheck_info", {})
        st.markdown(
            "<div class='debug-box'><b>엔진 호출 원칙</b><br>NSGA 엔진에는 원본 df_input을 전달하고, 전처리본은 점검/표시용으로 사용합니다.</div>",
            unsafe_allow_html=True,
        )
        st.write("원본 shape:", precheck.get("raw_shape"))
        st.write("전처리 shape:", precheck.get("prepared_shape"))
        sdf = precheck.get("summary_df", pd.DataFrame())
        pdf = precheck.get("preview_df", pd.DataFrame())
        if isinstance(sdf, pd.DataFrame) and not sdf.empty:
            st.dataframe(sdf, use_container_width=True, height=280)
        if isinstance(pdf, pd.DataFrame) and not pdf.empty:
            st.dataframe(pdf, use_container_width=True, height=300)
    with tabs[6]:
        render_stage1_readiness(st.session_state.get("run_result"), NSGAConfig())


def render_visual_tabs(df_input: pd.DataFrame, df_prepared: pd.DataFrame | None):
    base = df_prepared if isinstance(df_prepared, pd.DataFrame) and not df_prepared.empty else df_input
    nums = numeric_columns(base)
    objs = object_columns(base)
    tabs = st.tabs(["수치 분포", "수치 관계", "범주 분포"])
    with tabs[0]:
        if nums:
            col = st.selectbox("히스토그램 컬럼", nums, key="hist_col")
            st.pyplot(draw_histogram(base[col], f"{col} 분포"), use_container_width=True)
        else:
            st.info("수치형 컬럼이 없습니다.")
    with tabs[1]:
        if len(nums) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x = st.selectbox("X축", nums, key="scatter_x")
            with c2:
                y = st.selectbox("Y축", [c for c in nums if c != x], key="scatter_y")
            st.pyplot(draw_scatter(base, x, y, f"{x} vs {y}"), use_container_width=True)
        else:
            st.info("산점도용 수치형 컬럼이 부족합니다.")
    with tabs[2]:
        if objs:
            obj = st.selectbox("범주 컬럼", objs, key="obj_col")
            counts = base[obj].astype(str).value_counts(dropna=False).head(20)
            st.bar_chart(counts)
        else:
            st.info("범주형 컬럼이 없습니다.")


def render_run_history(result: dict | None):
    st.markdown("### 실행 이력")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return
    tabs = st.tabs(["세대별 로그", "Sweep 요약", "실시간 상태표"])
    with tabs[0]:
        h = result.get("history_df", pd.DataFrame())
        st.dataframe(h, use_container_width=True, height=520) if isinstance(h, pd.DataFrame) and not h.empty else st.info("세대별 로그가 없습니다.")
    with tabs[1]:
        s = result.get("sweep_summary_df", pd.DataFrame())
        st.dataframe(s, use_container_width=True, height=520) if isinstance(s, pd.DataFrame) and not s.empty else st.info("Sweep 요약이 없습니다.")
    with tabs[2]:
        tmap = st.session_state.get("target_status_map", {})
        logs = st.session_state.get("generation_logs", [])
        if tmap:
            st.dataframe(pd.DataFrame(list(tmap.values())).sort_values("Target Ao"), use_container_width=True, height=280)
        if logs:
            st.dataframe(pd.DataFrame(logs).tail(240), use_container_width=True, height=280)


def render_summary_panel(summary: dict):
    best_ao = float(summary.get("best_Ao", 0.0))
    target = float(summary.get("target_ao", 0.0))
    total_cost = float(summary.get("best_cost", summary.get("total_cost", 0.0)))
    stock = int(float(summary.get("total_stock", 0)))
    managed = int(float(summary.get("managed_parts", 0)))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Best Ao", f"{best_ao:.4f}")
    with c2:
        show_metric_card("Total Cost", f"{total_cost:,.0f}")
    with c3:
        show_metric_card("Total Stock", f"{stock:,}")
    with c4:
        show_metric_card("Managed Parts", f"{managed:,}")
    st.caption(f"대표 Target Ao: {target:.2f}")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        show_metric_card("Tail Risk", f"{float(summary.get('Tail_Risk', 0.0)):.4f}")
    with r2:
        show_metric_card("Worst Ao", f"{float(summary.get('Worst_Ao', 0.0)):.4f}")
    with r3:
        show_metric_card("Worst Shortage", f"{float(summary.get('Worst_Shortage', 0.0)):,.3f}")
    with r4:
        show_metric_card("Robustness Gap", f"{float(summary.get('Robustness_Gap', 0.0)):.4f}")


def render_model_c_robust_panel(result: dict | None):
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return
    summary = result.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Tail Risk", f"{float(summary.get('Tail_Risk', 0.0)):.4f}")
    with c2:
        show_metric_card("Worst Ao", f"{float(summary.get('Worst_Ao', 0.0)):.4f}")
    with c3:
        show_metric_card("Worst Shortage", f"{float(summary.get('Worst_Shortage', 0.0)):,.3f}")
    with c4:
        show_metric_card("Robustness Gap", f"{float(summary.get('Robustness_Gap', 0.0)):.4f}")

    st.markdown(
        f"""
        <div class="section-box">
            <b>Model C Robust 해석</b><br>
            K Grid: <b>{', '.join(map(str, result.get('stage1_readiness', {}).get('robust_k_grid', [])))}</b><br>
            Worst K: <b>{float(summary.get('Worst_K', 1.0)):.2f}</b> ·
            Robust 후보 추가: <b>{int(summary.get('robust_candidate_added_count', 0)):,}개</b><br><br>
            <b>Tail Risk</b>: K 시나리오별 손실 중 불리한 꼬리구간의 평균<br>
            <b>Worst Ao</b>: 모든 K 시나리오 중 가장 낮은 Ao<br>
            <b>Worst Shortage</b>: 모든 K 시나리오 중 가장 큰 기대 부족수량<br>
            <b>Robustness Gap</b>: 기준(K≈1.0) Ao와 Worst Ao의 차이
        </div>
        """,
        unsafe_allow_html=True,
    )

    rdf = result.get("robust_scenario_summary_df", pd.DataFrame())
    if isinstance(rdf, pd.DataFrame) and not rdf.empty:
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    detail = result.get("robust_scenario_detail_df", pd.DataFrame())
    if isinstance(detail, pd.DataFrame) and not detail.empty:
        st.markdown("**대표해 K 시나리오별 산출근거**")
        st.caption("Tail_Used=True인 시나리오들의 Scenario Loss 평균이 Tail Risk입니다.")
        st.dataframe(detail, use_container_width=True, hide_index=True, height=300)

    p = result.get("pareto_df", pd.DataFrame())
    robust_cols = [
        "target_ao", "Ao", "total_cost", "total_stock", "managed_parts",
        "Tail_Risk", "Worst_Ao", "Worst_Shortage", "Robustness_Gap", "Worst_K",
        "F1_Cost", "F2_Tail_Risk", "F3_Complexity", "Constraint_Violation",
    ]
    robust_cols = [c for c in robust_cols if isinstance(p, pd.DataFrame) and c in p.columns]
    if robust_cols:
        st.markdown("**Model C Pareto Robust 지표**")
        st.dataframe(p[robust_cols], use_container_width=True, height=480)



def render_representative_bridge_panel(result: dict | None):
    """Stage 3 representative-strategy bridge view."""
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    reps = result.get("representative_alternatives_df", pd.DataFrame())
    inv = result.get("representative_inventory_df", pd.DataFrame())
    bridge = result.get("monte_carlo_bridge_df", pd.DataFrame())
    audit = result.get("representative_selection_audit_df", pd.DataFrame())

    if not isinstance(reps, pd.DataFrame) or reps.empty:
        st.info("대표해 Bridge 데이터가 없습니다. Model C Pareto 해의 재고수준 분포를 확인해 주세요.")
        return

    st.markdown(
        """
        <div class="section-box">
            <b>Stage 3 · 대표해 Bridge</b><br>
            Model C Pareto 해 전체를 미래 시뮬레이션에 모두 넣지 않고,
            <b>서로 다른 재고수준의 분포를 대표하는 대안</b>으로 압축합니다.<br>
            기본 5개 전략명은 <b>최소재고형 → 경량 대응형 → 균형 대응형 → 안정형 → 최대안정형</b>이며,
            이름은 이해를 위한 표시값일 뿐 우열·추천점수가 아닙니다.<br>
            같은 재고수량의 해가 여러 개이면 <b>Tail Risk가 낮고 → Worst Ao가 높고 → 비용이 낮은 해</b>를 우선 사용합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(reps), 5))
    for col, (_, row) in zip(cols, reps.iterrows()):
        with col:
            show_metric_card(str(row.get("Strategy", "-")), f"{int(row.get('Stock_Units', 0)):,} EA")
            st.caption(
                f"Cost {float(row.get('NSGA_Cost_KRW', 0.0)):,.0f} | "
                f"Worst Ao {float(row.get('Worst_Ao', 0.0)):.4f} | "
                f"Tail {float(row.get('Tail_Risk', 0.0)):.4f}"
            )

    st.markdown("**대표 대안 비교**")
    st.dataframe(reps, use_container_width=True, hide_index=True, height=310)

    selected_alt = st.selectbox(
        "대표 대안 상세 보기",
        reps["Alternative_ID"].astype(str).tolist(),
        format_func=lambda x: (
            f"{x} · "
            + str(reps.loc[reps["Alternative_ID"].astype(str) == str(x), "Strategy"].iloc[0])
        ),
        key="stage3_rep_alt_select",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**선택 대안의 품목별 재고 구성**")
        if isinstance(inv, pd.DataFrame) and not inv.empty:
            sub = inv[inv["Alternative_ID"].astype(str) == str(selected_alt)].copy()
            view_cols = [c for c in [
                "part_id", "maint_echelon", "manage_flag", "Stock_Qty",
                "stock_O", "stock_I", "stock_D", "lead_time",
                "qpa", "repairable_flag", "repair_tat_h",
            ] if c in sub.columns]
            st.dataframe(sub[view_cols], use_container_width=True, height=430)
        else:
            st.info("대표해 재고 상세가 없습니다.")

    with c2:
        st.markdown("**향후 Monte Carlo 전달 데이터**")
        if isinstance(bridge, pd.DataFrame) and not bridge.empty:
            b = bridge[bridge["Alternative_ID"].astype(str) == str(selected_alt)].copy()
            st.caption(
                f"전체 spare-demand 품목 {len(b):,}개 · "
                f"재고 보유 품목 {(pd.to_numeric(b['Stock_Qty'], errors='coerce').fillna(0) > 0).sum():,}개"
            )
            bridge_cols = [c for c in [
                "Part_ID", "Stock_Qty", "Failure_Rate", "QPA",
                "Annual_Operating_Hours", "Lambda_Annual_MC_Base",
                "Lead_Time_H", "Repairable_Flag", "Repair_TAT_H",
            ] if c in b.columns]
            st.dataframe(b[bridge_cols], use_container_width=True, height=430)
        else:
            st.info("Monte Carlo Bridge 데이터가 없습니다.")

    with st.expander("대표해 선정 Audit"):
        if isinstance(audit, pd.DataFrame) and not audit.empty:
            st.dataframe(audit, use_container_width=True, hide_index=True, height=420)
        else:
            st.info("선정 Audit 데이터가 없습니다.")




def draw_probability_ring(value: float, title: str, subtitle: str = ""):
    """Full circular ring for Stage 4 probability/service visualization."""
    v = float(np.clip(pd.to_numeric(value, errors="coerce") if value is not None else 0.0, 0.0, 1.0))
    fig = go.Figure(
        data=[
            go.Pie(
                values=[v, max(1.0 - v, 0.0)],
                labels=[title, "Remaining"],
                hole=0.72,
                sort=False,
                direction="clockwise",
                textinfo="none",
                hoverinfo="skip",
                marker=dict(line=dict(width=0)),
                showlegend=False,
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>{v*100:.1f}%</b><br><span style='font-size:13px'>{clean_chart_text(title)}</span>",
        x=0.5, y=0.53, showarrow=False, align="center",
        font=dict(size=23),
    )
    if subtitle:
        fig.add_annotation(
            text=f"<span style='font-size:11px'>{clean_chart_text(subtitle)}</span>",
            x=0.5, y=0.36, showarrow=False, align="center",
            font=dict(size=11),
        )
    fig.update_layout(
        height=285,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="white",
    )
    return fig


@_fragment
def render_future_simulation_panel(result: dict | None):
    """Stage 4 Cold-start Monte Carlo decision-support view."""
    st.markdown("### 미래 시뮬레이션 · Cold-start Monte Carlo")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    summary_df = result.get("mc_summary_df", pd.DataFrame())
    part_df = result.get("mc_part_summary_df", pd.DataFrame())
    cfg_df = result.get("mc_config_df", pd.DataFrame())

    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        st.info("미래 시뮬레이션 결과가 없습니다. 사이드바에서 'NSGA 완료 후 Monte Carlo 실행'을 선택해 주세요.")
        return

    st.markdown(
        """
        <div class="section-box">
            <b>미래 시뮬레이션은 어떻게 계산하나?</b><br>
            ① NSGA-II가 먼저 <b>비용·재고·가용성 관점의 초기수리부속 대안</b>을 만듭니다.<br>
            ② 미래 고장률은 입력 RAM 값을 하나의 고정된 정답으로 두지 않고,
            품목별로 <b>K 불확실성 계수</b>를 곱해 반복마다 흔듭니다.
            K는 LogNormal 분포를 사용하며 대략 <b>P05=0.5 · 중앙값=1.0 · P95=2.0</b> 범위를 갖습니다.<br>
            ③ 각 반복에서 품목 i의 미래 수요는
            <b>Poisson(기준 연간고장수요 × K × 기간)</b>으로 생성하고,
            같은 미래수요를 모든 전략에 공통으로 적용하여 전략 간 비교조건을 맞춥니다.<br>
            ④ 수요가 발생했을 때 재고가 있으면 즉시충족,
            없으면 shortage로 기록하며, 수리가능품은 <b>Repair TAT</b>,
            비수리가능품은 <b>Lead Time</b>을 기준으로 보충주기를 반영합니다.<br>
            ⑤ 이 과정을 기본 <b>3,000회</b> 반복하여 평균 즉시충족률,
            하위 5% 서비스수준, 재고부족 발생확률, Shortage P95 등을 계산합니다.<br><br>
            즉, 하나의 고장률·하나의 수요량을 정답처럼 사용하는 것이 아니라
            <b>고장률과 미래수요가 달라질 수 있는 여러 상황에서 각 재고전략이 얼마나 견디는지</b>를 비교하는 분석입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    strategies = summary_df["Strategy"].drop_duplicates().astype(str).tolist()
    periods = sorted(pd.to_numeric(summary_df["Period_Months"], errors="coerce").dropna().astype(int).unique().tolist())
    methods = (
        summary_df["Evaluation_Method"].drop_duplicates().astype(str).tolist()
        if "Evaluation_Method" in summary_df.columns
        else ["건수 가중"]
    )

    st.markdown("#### 결과 조회 조건")
    default_method_idx = methods.index("건수 가중") if "건수 가중" in methods else 0

    if STREAMLIT_FRAGMENT_AVAILABLE:
        selected_strategy = st.radio(
            "전략 선택",
            strategies,
            horizontal=True,
            key="mc_strategy_radio",
        )
        c1, c2 = st.columns(2)
        with c1:
            selected_period = st.selectbox(
                "미래 기간",
                periods,
                index=max(len(periods) - 1, 0),
                format_func=lambda x: f"{int(x)}개월",
                key="mc_period_select",
            )
        with c2:
            selected_method = st.selectbox(
                "평가방식",
                methods,
                index=default_method_idx,
                key="mc_method_select",
            )
    else:
        with st.form("mc_result_filter_form", border=True):
            selected_strategy = st.radio(
                "전략 선택",
                strategies,
                horizontal=True,
                key="mc_strategy_radio",
            )
            c1, c2 = st.columns(2)
            with c1:
                selected_period = st.selectbox(
                    "미래 기간",
                    periods,
                    index=max(len(periods) - 1, 0),
                    format_func=lambda x: f"{int(x)}개월",
                    key="mc_period_select",
                )
            with c2:
                selected_method = st.selectbox(
                    "평가방식",
                    methods,
                    index=default_method_idx,
                    key="mc_method_select",
                )
            st.form_submit_button("조회 적용", use_container_width=True)

    hit = summary_df[
        (summary_df["Strategy"].astype(str) == str(selected_strategy))
        & (pd.to_numeric(summary_df["Period_Months"], errors="coerce").astype(int) == int(selected_period))
        & (
            (summary_df["Evaluation_Method"].astype(str) == str(selected_method))
            if "Evaluation_Method" in summary_df.columns
            else True
        )
    ]
    if hit.empty:
        st.warning("선택한 전략/기간의 시뮬레이션 결과가 없습니다.")
        return
    row = hit.iloc[0]

    service_mean = float(row.get("Service_Mean", 0.0))
    service_p05 = float(row.get("Service_P05", 0.0))
    zero_shortage_prob = float(row.get("Probability_Zero_Shortage", 0.0))
    any_shortage_prob = float(np.clip(1.0 - zero_shortage_prob, 0.0, 1.0))

    st.caption(
        "※ Ao는 시간기준 운용가용도이고, 아래 즉시충족률은 수요건 기준 서비스 수준입니다. "
        "서로 분모와 의미가 다르므로 같은 %로 움직이지 않습니다."
    )
    if service_p05 <= 1e-12 and service_mean > 1e-12:
        st.info(
            "즉시충족률(P05)=0%는 평균 즉시충족률이 0%라는 뜻이 아닙니다. "
            "Monte Carlo 하위 5% 시나리오에 즉시충족률 0%인 경우가 포함된다는 뜻입니다."
        )

    g1, g2, g3, g4 = st.columns([1.05, 1.05, 1.0, 1.0])
    with g1:
        st.plotly_chart(
            draw_probability_ring(
                service_mean,
                "평균 수요 즉시충족률",
                "Monte Carlo 반복별 즉시충족률의 평균",
            ),
            use_container_width=True,
            key=f"mc_service_mean_ring_{selected_strategy}_{selected_period}_{selected_method}",
        )
    with g2:
        st.plotly_chart(
            draw_probability_ring(
                any_shortage_prob,
                "재고부족 발생확률",
                f"{int(selected_period)}개월 중 1회 이상 shortage가 발생할 확률 · 낮을수록 좋음",
            ),
            use_container_width=True,
            key=f"mc_any_shortage_ring_{selected_strategy}_{selected_period}_{selected_method}",
        )
    with g3:
        show_metric_card("즉시충족률(P05)", f"{service_p05 * 100:,.1f}%")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        show_metric_card("재고부족 건수(P95)", f"{float(row.get('Shortage_P95', 0.0)):,.2f}")
    with g4:
        show_metric_card("재고부족 0회 확률", f"{zero_shortage_prob * 100:,.1f}%")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        show_metric_card("평균 미래수요", f"{float(row.get('Demand_Mean', 0.0)):,.2f}")

    st.markdown("#### 대표대안 비교")
    compare = summary_df[
        (pd.to_numeric(summary_df["Period_Months"], errors="coerce").astype(int) == int(selected_period))
        & (
            (summary_df["Evaluation_Method"].astype(str) == str(selected_method))
            if "Evaluation_Method" in summary_df.columns
            else True
        )
    ].copy()
    if "Probability_Zero_Shortage" in compare.columns:
        compare["Probability_Any_Shortage"] = (
            1.0 - pd.to_numeric(
                compare["Probability_Zero_Shortage"], errors="coerce"
            ).fillna(0.0)
        ).clip(0.0, 1.0)

    view_cols = [c for c in [
        "Alternative_ID", "Strategy", "Period_Months",
        "Demand_Mean", "Demand_P95",
        "Service_Mean", "Service_P05",
        "Probability_Any_Shortage", "Probability_Zero_Shortage",
        "Shortage_Mean", "Shortage_P95",
        "Mean_Event_Delay_Days",
    ] if c in compare.columns]
    st.dataframe(compare[view_cols], use_container_width=True, hide_index=True, height=300)

    tabs = st.tabs(["기간별 변화", "품목 Drill-down", "시뮬레이션 가정"])
    with tabs[0]:
        trend = summary_df[
            (summary_df["Strategy"].astype(str) == str(selected_strategy))
            & (
                (summary_df["Evaluation_Method"].astype(str) == str(selected_method))
                if "Evaluation_Method" in summary_df.columns
                else True
            )
        ].copy()
        trend = trend.sort_values("Period_Months")
        if not trend.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend["Period_Months"], y=trend["Service_P05"],
                mode="lines+markers", name="Service P05",
            ))
            fig.add_trace(go.Scatter(
                x=trend["Period_Months"], y=trend["Probability_Zero_Shortage"],
                mode="lines+markers", name="Zero Shortage Probability",
            ))
            fig.update_layout(
                height=390,
                xaxis_title="기간(개월)",
                yaxis_title="확률 / 비율",
                yaxis=dict(range=[0, 1.02]),
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.18),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,.08)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.08)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if isinstance(part_df, pd.DataFrame) and not part_df.empty:
            psub = part_df[
                (part_df["Strategy"].astype(str) == str(selected_strategy))
                & (pd.to_numeric(part_df["Period_Months"], errors="coerce").astype(int) == int(selected_period))
            ].copy()
            psub = psub.sort_values(
                ["Shortage_Mean", "Demand_Mean"],
                ascending=[False, False],
            )
            st.caption("Shortage Mean이 큰 품목부터 표시합니다.")
            st.dataframe(psub, use_container_width=True, hide_index=True, height=470)
        else:
            st.info("품목별 시뮬레이션 진단 데이터가 없습니다.")

    with tabs[2]:
        if isinstance(cfg_df, pd.DataFrame) and not cfg_df.empty:
            st.dataframe(cfg_df, use_container_width=True, hide_index=True, height=220)
        st.markdown(
            """
            - `K`: 미래 고장률 불확실성 배수
            - `Demand`: Poisson 확률수요
            - 수리가능품 재고 복귀: `Repair_TAT_H`
            - 비수리가능품 재고 복귀: `Lead_Time_H`
            - 5개 전략은 같은 K와 같은 수요사건을 공유하여 비교합니다(Common Random Numbers).
            """
        )




def draw_grade_badge(
    grade: str,
    title: str,
    raw_value_text: str,
    subtitle: str = "",
):
    """Energy-label-like circular badge for transparent relative grades."""
    grade_text = clean_chart_text(grade if grade is not None else "-")
    # Ring is visual only; the grade itself comes from decision_engine.py.
    fig = go.Figure(
        data=[
            go.Pie(
                values=[1.0],
                labels=[title],
                hole=0.72,
                sort=False,
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
                marker=dict(line=dict(width=0)),
            )
        ]
    )
    fig.add_annotation(
        text=(
            f"<span style='font-size:42px'><b>{grade_text}</b></span><br>"
            f"<span style='font-size:13px'>{clean_chart_text(title)}</span><br>"
            f"<span style='font-size:12px'>{clean_chart_text(raw_value_text)}</span>"
        ),
        x=0.5, y=0.53, showarrow=False, align="center",
    )
    if subtitle:
        fig.add_annotation(
            text=f"<span style='font-size:10px'>{clean_chart_text(subtitle)}</span>",
            x=0.5, y=0.26, showarrow=False, align="center",
        )
    fig.update_layout(
        height=300,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="white",
    )
    return fig


@_fragment

def _build_projected_pareto_cloud(result: dict, decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a visualization-only background cloud from all NSGA Pareto alternatives.

    The 5 representative alternatives have actual Monte Carlo metrics.
    Other Pareto points do NOT receive a second full Monte Carlo run; their service/risk
    values are linearly interpolated by stock level from the 5 MC-confirmed anchors.
    This is for decision-map context only, never for grading or automatic selection.
    """
    source = result.get("all_target_runs_df", pd.DataFrame())
    if not isinstance(source, pd.DataFrame) or source.empty:
        source = result.get("pareto_df", pd.DataFrame())
    if not isinstance(source, pd.DataFrame) or source.empty:
        return pd.DataFrame()

    need = {"total_cost", "total_stock"}
    if not need.issubset(source.columns):
        return pd.DataFrame()

    cloud = source.copy()
    cloud["total_cost"] = pd.to_numeric(cloud["total_cost"], errors="coerce")
    cloud["total_stock"] = pd.to_numeric(cloud["total_stock"], errors="coerce")
    if "Constraint_Violation" in cloud.columns:
        cv = pd.to_numeric(cloud["Constraint_Violation"], errors="coerce").fillna(0.0)
        cloud = cloud[cv <= 1e-9].copy()

    cloud = cloud[np.isfinite(cloud["total_cost"]) & np.isfinite(cloud["total_stock"])].copy()
    cloud = cloud.drop_duplicates(["total_cost", "total_stock"]).reset_index(drop=True)
    if cloud.empty:
        return cloud

    anchors = decision_df.copy()
    for c in ["Stock_Units", "Service_Mean", "Stability_Core", "Probability_Any_Shortage", "NSGA_Cost_KRW"]:
        anchors[c] = pd.to_numeric(anchors[c], errors="coerce")
    anchors = anchors.dropna(subset=["Stock_Units", "NSGA_Cost_KRW"]).sort_values("Stock_Units")
    anchors = anchors.drop_duplicates("Stock_Units", keep="first")
    if anchors.empty:
        return pd.DataFrame()

    x = anchors["Stock_Units"].to_numpy(float)
    q = cloud["total_stock"].to_numpy(float)

    def interp_col(col, default=0.0):
        y = pd.to_numeric(anchors[col], errors="coerce").fillna(default).to_numpy(float)
        if len(x) == 1:
            return np.full(len(q), y[0], dtype=float)
        return np.interp(q, x, y, left=y[0], right=y[-1])

    cloud["Projected_Service_Mean"] = np.clip(interp_col("Service_Mean"), 0.0, 1.0)
    cloud["Projected_Stability_Core"] = np.clip(interp_col("Stability_Core"), 0.0, 1.0)
    cloud["Projected_Any_Shortage"] = np.clip(interp_col("Probability_Any_Shortage"), 0.0, 1.0)

    baseline = anchors.sort_values(["Stock_Units", "NSGA_Cost_KRW"]).iloc[0]
    baseline_risk = float(baseline["Probability_Any_Shortage"])
    baseline_cost = float(baseline["NSGA_Cost_KRW"])
    cloud["Additional_Cost_KRW"] = cloud["total_cost"] - baseline_cost
    cloud["Projected_Risk_Reduction"] = baseline_risk - cloud["Projected_Any_Shortage"]

    # Assign descriptive region by nearest MC-confirmed representative anchor
    # in normalized cost / service plane. This is not a recommendation.
    ax_cost = anchors["NSGA_Cost_KRW"].to_numpy(float)
    ax_serv = anchors["Service_Mean"].to_numpy(float)
    labels = anchors["Strategy"].astype(str).to_numpy()
    cmin, cmax = float(np.min(ax_cost)), float(np.max(ax_cost))
    smin, smax = float(np.min(ax_serv)), float(np.max(ax_serv))
    cr = max(cmax - cmin, 1e-9)
    sr = max(smax - smin, 1e-9)
    pc = cloud["total_cost"].to_numpy(float)
    ps = cloud["Projected_Service_Mean"].to_numpy(float)

    d = (
        ((pc[:, None] - ax_cost[None, :]) / cr) ** 2
        + ((ps[:, None] - ax_serv[None, :]) / sr) ** 2
    )
    cloud["Projected_Strategy_Region"] = labels[np.argmin(d, axis=1)]
    return cloud


def _add_strategy_region_background(fig: go.Figure, anchors: pd.DataFrame, x_col: str, y_col: str):
    """Add low-opacity nearest-anchor regions for descriptive strategy zones."""
    if anchors is None or anchors.empty or len(anchors) < 2:
        return

    a = anchors.copy()
    a[x_col] = pd.to_numeric(a[x_col], errors="coerce")
    a[y_col] = pd.to_numeric(a[y_col], errors="coerce")
    a = a.dropna(subset=[x_col, y_col]).reset_index(drop=True)
    if len(a) < 2:
        return

    xs = a[x_col].to_numpy(float)
    ys = a[y_col].to_numpy(float)
    xpad = max((xs.max() - xs.min()) * 0.08, 1e-9)
    ypad = max((ys.max() - ys.min()) * 0.12, 0.01)
    gx = np.linspace(xs.min() - xpad, xs.max() + xpad, 90)
    gy = np.linspace(max(0.0, ys.min() - ypad), min(1.05, ys.max() + ypad), 70)
    xx, yy = np.meshgrid(gx, gy)

    xr = max(xs.max() - xs.min(), 1e-9)
    yr = max(ys.max() - ys.min(), 1e-9)
    dist = (
        ((xx[..., None] - xs[None, None, :]) / xr) ** 2
        + ((yy[..., None] - ys[None, None, :]) / yr) ** 2
    )
    zz = np.argmin(dist, axis=2)

    colors = [
        "rgba(76,175,80,0.035)",
        "rgba(33,150,243,0.030)",
        "rgba(255,193,7,0.035)",
        "rgba(156,39,176,0.028)",
        "rgba(244,67,54,0.025)",
    ]
    n = max(len(a), 1)
    colorscale = []
    for i in range(n):
        lo = i / n
        hi = (i + 1) / n
        color = colors[i % len(colors)]
        colorscale.extend([(lo, color), (max(lo, hi - 1e-6), color)])

    fig.add_trace(
        go.Heatmap(
            x=gx, y=gy, z=zz,
            zmin=0, zmax=max(n - 1, 1),
            colorscale=colorscale,
            showscale=False,
            hoverinfo="skip",
            opacity=0.24,
            name="전략 근접영역",
        )
    )


@_fragment
def render_cost_risk_decision_panel(result: dict | None):
    """Stage 5 transparent cost-risk decision view."""
    st.markdown("### 비용 · Risk 의사결정")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    decision_df = result.get("decision_table_df", pd.DataFrame())
    positioning_df = result.get("decision_positioning_df", pd.DataFrame())
    audit_df = result.get("decision_grade_audit_df", pd.DataFrame())
    config_df = result.get("decision_config_df", pd.DataFrame())

    if not isinstance(decision_df, pd.DataFrame) or decision_df.empty:
        st.info(
            "Decision Layer 결과가 없습니다. "
            "Monte Carlo 실행을 활성화한 후 NSGA + 미래 시뮬레이션을 실행해 주세요."
        )
        return

    st.markdown(
        """
        <div class="section-box">
            <b>Stage 5 해석 원칙</b><br>
            아래 A~E는 <b>같은 기간의 대표대안끼리 비교한 상대등급</b>이며,
            규격 인증이나 절대적인 안전등급이 아닙니다.<br>
            <b>단일 종합점수와 자동 최적안은 사용하지 않습니다.</b>
            미래 안정성, 부족위험, 비용효율을 각각 분리하여 보고
            원자료(Service P05, 재고부족 0회 확률, Shortage P95, 비용)를 함께 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    strategies = decision_df["Strategy"].astype(str).tolist()
    if STREAMLIT_FRAGMENT_AVAILABLE:
        selected_strategy = st.radio(
            "의사결정 전략 선택",
            strategies,
            horizontal=True,
            key="decision_strategy_radio",
        )
    else:
        with st.form("decision_strategy_form", border=True):
            selected_strategy = st.radio(
                "의사결정 전략 선택",
                strategies,
                horizontal=True,
                key="decision_strategy_radio",
            )
            st.form_submit_button("전략 조회 적용", use_container_width=True)
    hit = decision_df[decision_df["Strategy"].astype(str) == str(selected_strategy)]
    if hit.empty:
        st.warning("선택 전략의 Decision Layer 결과가 없습니다.")
        return
    row = hit.iloc[0]

    b1, b2, b3 = st.columns(3)
    with b1:
        st.plotly_chart(
            draw_grade_badge(
                str(row.get("Future_Stability_Grade", "-")),
                "미래 안정성 상대등급",
                f"Stability {float(row.get('Stability_Core', 0.0))*100:.1f}%",
                "min(Service P05, 재고부족 0회 확률) · 동률 시 Zero-Shortage tie-break",
            ),
            use_container_width=True,
            key=f"grade_stability_{selected_strategy}",
        )
    with b2:
        st.plotly_chart(
            draw_grade_badge(
                str(row.get("Shortage_Risk_Grade", "-")),
                "부족위험 상대등급",
                f"부족발생 {float(row.get('Probability_Any_Shortage', 0.0))*100:.1f}%",
                f"Shortage P95 {float(row.get('Shortage_P95', 0.0)):.2f}",
            ),
            use_container_width=True,
            key=f"grade_shortage_{selected_strategy}",
        )
    with b3:
        eff = pd.to_numeric(row.get("Risk_Reduction_per_100M_KRW", np.nan), errors="coerce")
        eff_txt = (
            "최소재고형 기준"
            if pd.isna(eff)
            else f"1억원당 Risk↓ {float(eff)*100:.3f}%p"
        )
        st.plotly_chart(
            draw_grade_badge(
                str(row.get("Cost_Efficiency_Grade", "-")),
                "비용효율 상대등급",
                eff_txt,
                "최소재고형 대비 추가비용 / 부족확률 감소",
            ),
            use_container_width=True,
            key=f"grade_efficiency_{selected_strategy}",
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("NSGA 비용", f"{float(row.get('NSGA_Cost_KRW', 0.0)):,.0f}원")
    with c2:
        show_metric_card("총 재고", f"{int(float(row.get('Stock_Units', 0))):,} EA")
    with c3:
        show_metric_card("평균 즉시충족률", f"{float(row.get('Service_Mean', 0.0))*100:.2f}%")
    with c4:
        show_metric_card(
            "재고부족 발생확률",
            f"{float(row.get('Probability_Any_Shortage', 0.0))*100:.2f}%"
        )

    st.markdown("#### 비용–안정성 · 위험감소 의사결정 지도")
    st.caption(
        "큰 점은 Monte Carlo로 직접 검증한 대표 5개 대안입니다. "
        "작은 점은 전체 NSGA Pareto 대안을 대표 5개 MC 결과 사이에서 재고수준 기준으로 보간한 "
        "시각화용 후보군이며, 등급·자동선택에는 사용하지 않습니다. "
        "배경 음영은 가장 가까운 대표전략의 설명용 영역으로 추천구간이 아닙니다."
    )

    map_mode = st.radio(
        "지도 관점",
        ["비용–평균 즉시충족률", "비용–보수 안정성", "추가비용–위험감소"],
        horizontal=True,
        key="decision_map_mode",
    )

    if isinstance(positioning_df, pd.DataFrame) and not positioning_df.empty:
        cloud = _build_projected_pareto_cloud(result, decision_df)
        fig = go.Figure()

        if map_mode == "비용–평균 즉시충족률":
            anchors = positioning_df.copy()
            anchors["X_View"] = pd.to_numeric(anchors["NSGA_Cost_KRW"], errors="coerce") / 100_000_000.0
            anchors["Y_View"] = pd.to_numeric(anchors["Service_Mean"], errors="coerce").fillna(0.0)
            x_title = "총 비용 (억원)"
            y_title = "평균 수요 즉시충족률"
            y_range = [0, 1.02]
            if not cloud.empty:
                cloud["X_View"] = cloud["total_cost"] / 100_000_000.0
                cloud["Y_View"] = cloud["Projected_Service_Mean"]

        elif map_mode == "비용–보수 안정성":
            anchors = positioning_df.copy()
            anchors["X_View"] = pd.to_numeric(anchors["NSGA_Cost_KRW"], errors="coerce") / 100_000_000.0
            anchors["Y_View"] = pd.to_numeric(anchors["Stability_Core"], errors="coerce").fillna(0.0)
            x_title = "총 비용 (억원)"
            y_title = "보수 안정성 = min(Service P05, Zero Shortage)"
            y_range = [0, 1.02]
            if not cloud.empty:
                cloud["X_View"] = cloud["total_cost"] / 100_000_000.0
                cloud["Y_View"] = cloud["Projected_Stability_Core"]

        else:
            anchors = positioning_df.copy()
            baseline_row = decision_df.sort_values(
                ["Stock_Units", "NSGA_Cost_KRW", "Alternative_ID"],
                ascending=[True, True, True],
            ).iloc[0]
            baseline_cost = float(baseline_row["NSGA_Cost_KRW"])
            baseline_risk = float(baseline_row["Probability_Any_Shortage"])

            anchors["X_View"] = (
                pd.to_numeric(anchors["NSGA_Cost_KRW"], errors="coerce") - baseline_cost
            ) / 100_000_000.0
            anchors["Y_View"] = (
                baseline_risk
                - pd.to_numeric(anchors["Probability_Any_Shortage"], errors="coerce").fillna(baseline_risk)
            ) * 100.0
            x_title = "최소재고형 대비 추가비용 (억원)"
            y_title = "재고부족 발생확률 감소 (%p)"
            ymin = min(-1.0, float(anchors["Y_View"].min()) - 1.0)
            ymax = max(1.0, float(anchors["Y_View"].max()) + 1.0)
            y_range = [ymin, ymax]
            if not cloud.empty:
                cloud["X_View"] = cloud["Additional_Cost_KRW"] / 100_000_000.0
                cloud["Y_View"] = cloud["Projected_Risk_Reduction"] * 100.0

        anchors = anchors.dropna(subset=["X_View", "Y_View"]).reset_index(drop=True)

        # Background strategy regions only for 0~1 vertical axes.
        if map_mode != "추가비용–위험감소":
            _add_strategy_region_background(fig, anchors, "X_View", "Y_View")

        # Other Pareto alternatives: small projected points.
        if isinstance(cloud, pd.DataFrame) and not cloud.empty:
            fig.add_trace(
                go.Scatter(
                    x=cloud["X_View"],
                    y=cloud["Y_View"],
                    mode="markers",
                    name="기타 Pareto 대안(보간)",
                    marker=dict(size=5, opacity=0.30),
                    customdata=np.column_stack([
                        cloud["total_stock"].to_numpy(),
                        cloud["Projected_Strategy_Region"].astype(str).to_numpy(),
                    ]),
                    hovertemplate=(
                        "Pareto 후보<br>"
                        "X=%{x:.3f}<br>"
                        "Y=%{y:.3f}<br>"
                        "Stock=%{customdata[0]:.0f}<br>"
                        "근접전략=%{customdata[1]}<extra></extra>"
                    ),
                )
            )

            # Candidate-cloud trend by X bins.
            try:
                qn = min(12, max(4, int(cloud["X_View"].nunique() // 5)))
                cloud["_bin"] = pd.qcut(cloud["X_View"], q=qn, duplicates="drop")
                trend = cloud.groupby("_bin", observed=True).agg(
                    X_View=("X_View", "median"),
                    Y_View=("Y_View", "median"),
                ).dropna().sort_values("X_View")
                if len(trend) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=trend["X_View"],
                            y=trend["Y_View"],
                            mode="lines",
                            name="Pareto 후보군 추세",
                            line=dict(width=2, dash="dot"),
                            hoverinfo="skip",
                        )
                    )
            except Exception:
                pass

        # MC-confirmed representative anchors.
        anchors_sorted = anchors.sort_values("X_View")
        fig.add_trace(
            go.Scatter(
                x=anchors_sorted["X_View"],
                y=anchors_sorted["Y_View"],
                mode="lines",
                name="대표전략 연결선",
                line=dict(width=2),
                hoverinfo="skip",
            )
        )

        for _, r in anchors.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[float(r["X_View"])],
                    y=[float(r["Y_View"])],
                    mode="markers+text",
                    text=[str(r["Strategy"])],
                    textposition="top center",
                    marker=dict(
                        size=max(16, min(48, 16 + float(r["Stock_Units"]) ** 0.5)),
                        line=dict(width=1.5),
                    ),
                    customdata=[[
                        str(r["Future_Stability_Grade"]),
                        str(r["Shortage_Risk_Grade"]),
                        str(r["Cost_Efficiency_Grade"]),
                        int(r["Stock_Units"]),
                        float(r.get("Probability_Any_Shortage", 0.0)) * 100.0,
                    ]],
                    hovertemplate=(
                        "%{text}<br>"
                        "X=%{x:.3f}<br>"
                        "Y=%{y:.3f}<br>"
                        "안정성=%{customdata[0]} / 부족위험=%{customdata[1]} / 비용효율=%{customdata[2]}<br>"
                        "Stock=%{customdata[3]} EA<br>"
                        "재고부족 발생확률=%{customdata[4]:.1f}%<extra></extra>"
                    ),
                    name=str(r["Strategy"]),
                    showlegend=False,
                )
            )

        fig.update_layout(
            height=790,
            xaxis_title=x_title,
            yaxis_title=y_title,
            yaxis=dict(range=y_range),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=35, r=30, t=40, b=60),
            legend=dict(orientation="h", y=-0.16),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.08)")
        st.plotly_chart(fig, use_container_width=True, key=f"decision_map_{map_mode}")

        if map_mode == "추가비용–위험감소":
            st.caption(
                "이 관점은 '얼마를 더 쓰면 재고부족 발생확률이 몇 %p 줄어드는가'를 직접 보여줍니다. "
                "우상향하면서 같은 비용에서 더 높은 점일수록 비용 대비 위험감소가 큽니다."
            )

    st.markdown("#### PBL 사업 착수 전 필수 관계 점검")
    st.caption(
        "아래 차트는 '얼마를 사야 하는가'보다, 비용을 늘렸을 때 위험·재고·가용성이 "
        "어떻게 바뀌는지와 추가투자의 한계효과가 어디서 둔화되는지를 보기 위한 의사결정 보조자료입니다."
    )

    reps_df = result.get("representative_alternatives_df", pd.DataFrame())
    rel = decision_df.copy()
    if isinstance(reps_df, pd.DataFrame) and not reps_df.empty and "Strategy" in reps_df.columns:
        rep_cols = [c for c in ["Strategy", "Worst_Ao", "Nominal_Ao", "Tail_Risk", "Worst_Shortage", "Robustness_Gap"] if c in reps_df.columns]
        if len(rep_cols) > 1:
            rel = rel.merge(
                reps_df[rep_cols].drop_duplicates("Strategy"),
                on="Strategy",
                how="left",
                suffixes=("", "_NSGA"),
            )

    for c in [
        "NSGA_Cost_KRW", "Stock_Units", "Probability_Any_Shortage",
        "Service_Mean", "Worst_Ao", "Nominal_Ao", "Tail_Risk",
        "Worst_Shortage", "Robustness_Gap",
    ]:
        if c in rel.columns:
            rel[c] = pd.to_numeric(rel[c], errors="coerce")

    rel["Cost_100M"] = rel["NSGA_Cost_KRW"] / 100_000_000.0
    rel["Any_Shortage_Pct"] = rel["Probability_Any_Shortage"] * 100.0
    rel["Service_Mean_Pct"] = rel["Service_Mean"] * 100.0

    r1, r2 = st.columns(2)

    with r1:
        fig_cost_risk = go.Figure()
        fig_cost_risk.add_trace(
            go.Scatter(
                x=rel["Cost_100M"],
                y=rel["Any_Shortage_Pct"],
                mode="lines+markers+text",
                text=rel["Strategy"].astype(str),
                textposition="top center",
                marker=dict(size=14),
                customdata=np.column_stack([rel["Stock_Units"].fillna(0).to_numpy()]),
                hovertemplate=(
                    "%{text}<br>비용=%{x:.2f}억원<br>"
                    "재고부족 발생확률=%{y:.2f}%<br>"
                    "재고=%{customdata[0]:.0f} EA<extra></extra>"
                ),
            )
        )
        fig_cost_risk.update_layout(
            title="비용 ↔ 재고부족 발생확률",
            height=500,
            xaxis_title="총 비용 (억원)",
            yaxis_title="재고부족 발생확률 (%) · 낮을수록 좋음",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=25, t=65, b=55),
            showlegend=False,
        )
        fig_cost_risk.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
        fig_cost_risk.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
        st.plotly_chart(fig_cost_risk, use_container_width=True, key="pbl_cost_shortage_chart")

    with r2:
        fig_stock_risk = go.Figure()
        fig_stock_risk.add_trace(
            go.Scatter(
                x=rel["Stock_Units"],
                y=rel["Any_Shortage_Pct"],
                mode="lines+markers+text",
                text=rel["Strategy"].astype(str),
                textposition="top center",
                marker=dict(size=14),
                customdata=np.column_stack([rel["Cost_100M"].fillna(0).to_numpy()]),
                hovertemplate=(
                    "%{text}<br>재고=%{x:.0f} EA<br>"
                    "재고부족 발생확률=%{y:.2f}%<br>"
                    "비용=%{customdata[0]:.2f}억원<extra></extra>"
                ),
            )
        )
        fig_stock_risk.update_layout(
            title="재고수량 ↔ 재고부족 발생확률",
            height=500,
            xaxis_title="대표전략 총 재고 (EA)",
            yaxis_title="재고부족 발생확률 (%) · 낮을수록 좋음",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=25, t=65, b=55),
            showlegend=False,
        )
        fig_stock_risk.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
        fig_stock_risk.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
        st.plotly_chart(fig_stock_risk, use_container_width=True, key="pbl_stock_shortage_chart")

    r3, r4 = st.columns(2)

    with r3:
        if "Worst_Ao" in rel.columns and rel["Worst_Ao"].notna().any():
            ao_plot = rel.dropna(subset=["Worst_Ao"]).sort_values("Cost_100M")
            fig_cost_ao = go.Figure()
            fig_cost_ao.add_trace(
                go.Scatter(
                    x=ao_plot["Cost_100M"],
                    y=ao_plot["Worst_Ao"] * 100.0,
                    mode="lines+markers+text",
                    text=ao_plot["Strategy"].astype(str),
                    textposition="top center",
                    marker=dict(size=14),
                    hovertemplate=(
                        "%{text}<br>비용=%{x:.2f}억원<br>"
                        "Worst Ao=%{y:.3f}%<extra></extra>"
                    ),
                )
            )
            fig_cost_ao.update_layout(
                title="비용 ↔ 불리한 고장률 시나리오의 Worst Ao",
                height=500,
                xaxis_title="총 비용 (억원)",
                yaxis_title="Worst Ao (%) · 높을수록 좋음",
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=40, r=25, t=65, b=55),
                showlegend=False,
            )
            fig_cost_ao.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
            fig_cost_ao.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
            st.plotly_chart(fig_cost_ao, use_container_width=True, key="pbl_cost_worstao_chart")
        else:
            st.info("Worst Ao 데이터가 없어 비용–Worst Ao 차트를 표시하지 못했습니다.")

    with r4:
        # Incremental / marginal value of moving from one representative strategy
        # to the next more expensive strategy.
        marginal = rel.sort_values(["NSGA_Cost_KRW", "Stock_Units"]).reset_index(drop=True).copy()
        marginal["Prev_Strategy"] = marginal["Strategy"].shift(1)
        marginal["Delta_Cost_100M"] = marginal["Cost_100M"].diff()
        marginal["Delta_Risk_Reduction_PctPt"] = (
            marginal["Probability_Any_Shortage"].shift(1)
            - marginal["Probability_Any_Shortage"]
        ) * 100.0
        marginal["Risk_Reduction_per_100M"] = np.where(
            marginal["Delta_Cost_100M"] > 1e-12,
            marginal["Delta_Risk_Reduction_PctPt"] / marginal["Delta_Cost_100M"],
            np.nan,
        )
        marginal = marginal.iloc[1:].copy()
        marginal["Transition"] = (
            marginal["Prev_Strategy"].astype(str) + " → " + marginal["Strategy"].astype(str)
        )

        fig_marginal = go.Figure()
        fig_marginal.add_trace(
            go.Bar(
                x=marginal["Transition"],
                y=marginal["Risk_Reduction_per_100M"],
                customdata=np.column_stack([
                    marginal["Delta_Cost_100M"].fillna(0).to_numpy(),
                    marginal["Delta_Risk_Reduction_PctPt"].fillna(0).to_numpy(),
                ]),
                hovertemplate=(
                    "%{x}<br>"
                    "추가비용=%{customdata[0]:.2f}억원<br>"
                    "Risk 감소=%{customdata[1]:.2f}%p<br>"
                    "1억원당 Risk 감소=%{y:.3f}%p<extra></extra>"
                ),
            )
        )
        fig_marginal.update_layout(
            title="전략 상향 시 한계 Risk 감소효과",
            height=500,
            xaxis_title="전략 전환 구간",
            yaxis_title="추가 1억원당 재고부족 확률 감소 (%p/억원)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=25, t=65, b=90),
            showlegend=False,
        )
        fig_marginal.update_xaxes(tickangle=-20, showgrid=False)
        fig_marginal.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.07)")
        st.plotly_chart(fig_marginal, use_container_width=True, key="pbl_marginal_value_chart")
        st.caption(
            "막대가 높을수록 같은 추가비용으로 재고부족 위험을 더 많이 줄이는 구간입니다. "
            "뒤쪽 전략으로 갈수록 막대가 급격히 낮아지면 추가투자의 한계효과가 둔화되는 신호입니다."
        )

    # Numeric transition table supports the executive decision discussion.
    if not marginal.empty:
        show_cols = [
            "Transition", "Delta_Cost_100M",
            "Delta_Risk_Reduction_PctPt", "Risk_Reduction_per_100M",
        ]
        marginal_view = marginal[show_cols].copy()
        marginal_view = marginal_view.rename(columns={
            "Transition": "전략 전환",
            "Delta_Cost_100M": "추가비용(억원)",
            "Delta_Risk_Reduction_PctPt": "재고부족확률 감소(%p)",
            "Risk_Reduction_per_100M": "1억원당 Risk 감소(%p)",
        })
        st.dataframe(
            marginal_view,
            use_container_width=True,
            hide_index=True,
            height=min(260, 48 + 36 * len(marginal_view)),
        )


    st.markdown("#### 대표대안 비용·Risk 비교표")
    view_cols = [c for c in [
        "Alternative_ID", "Strategy", "Decision_Period_Months",
        "Stock_Units", "NSGA_Cost_KRW",
        "Service_Mean", "Service_P05", "Probability_Zero_Shortage",
        "Stability_Core", "Future_Stability_Grade",
        "Probability_Any_Shortage", "Shortage_P95", "Shortage_Risk_Grade",
        "Additional_Cost_vs_MinStock_KRW", "Risk_Reduction_vs_MinStock",
        "Risk_Reduction_per_100M_KRW", "Cost_Efficiency_Grade",
        "Mean_Event_Delay_Days",
    ] if c in decision_df.columns]
    st.dataframe(
        decision_df[view_cols],
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    with st.expander("등급 산식·기준 Audit"):
        if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, hide_index=True, height=220)
        if isinstance(config_df, pd.DataFrame) and not config_df.empty:
            st.markdown("**Decision Layer 설정**")
            st.dataframe(config_df, use_container_width=True, hide_index=True, height=180)




def render_final_executive_panel(result: dict | None):
    """Stage 6 final executive summary without selecting a political-like 'winner'."""
    st.markdown("### 최종 Executive 요약")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    summary = result.get("summary", {})
    decision_df = result.get("decision_table_df", pd.DataFrame())
    audit_summary_df = result.get("audit_summary_df", pd.DataFrame())

    audit_status = "-"
    pass_text = "-"
    if isinstance(audit_summary_df, pd.DataFrame) and not audit_summary_df.empty:
        audit_status = str(audit_summary_df.iloc[0].get("Overall_Status", "-"))
        pass_count = int(pd.to_numeric(audit_summary_df.iloc[0].get("Pass_Count", 0), errors="coerce") or 0)
        total_checks = int(pd.to_numeric(audit_summary_df.iloc[0].get("Total_Checks", 0), errors="coerce") or 0)
        pass_text = f"{pass_count}/{total_checks}"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        show_metric_card("대표 대안 수", f"{int(summary.get('representative_alternative_count', 0)):,}")
    with c2:
        show_metric_card("MC 반복 수", f"{int(summary.get('monte_carlo_iterations', 0)):,}")
    with c3:
        show_metric_card("Decision 기간", f"{int(summary.get('decision_period_months', 0))}개월")
    with c4:
        supply_mode = str(summary.get("supply_structure_mode", "-"))
        supply_text = (
            "정비계층 연계형"
            if supply_mode == "MAINT_ECHELON"
            else ("창 직보급형" if supply_mode == "DEPOT_DIRECT" else supply_mode)
        )
        show_metric_card("보급구조", supply_text)
    with c5:
        show_metric_card("최종 Audit", f"{audit_status} · {pass_text}")

    st.markdown(
        """
        <div class="section-box">
            <b>최종 해석 프레임</b><br>
            이 웹은 하나의 “정답 재고량”을 고르는 도구가 아니라,
            <b>비용을 더 투입할수록 미래 수요 충족·재고부족 Risk가 어떻게 변하는지</b>를
            대표전략별로 비교하는 의사결정 지원체계입니다.<br>
            NSGA-II는 대안을 만들고, Monte Carlo는 미래 불확실성을 검증하며,
            Decision Layer는 비용·안정성·부족위험을 분리해 보여주고,
            Final Audit은 각 단계가 서로 일관되게 연결됐는지 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if isinstance(decision_df, pd.DataFrame) and not decision_df.empty:
        view_cols = [c for c in [
            "Strategy", "Stock_Units", "NSGA_Cost_KRW",
            "Service_P05", "Probability_Zero_Shortage",
            "Shortage_P95",
            "Future_Stability_Grade", "Shortage_Risk_Grade",
            "Cost_Efficiency_Grade",
        ] if c in decision_df.columns]
        st.markdown("#### 대표전략 한눈에 비교")
        st.dataframe(
            decision_df[view_cols],
            use_container_width=True,
            hide_index=True,
            height=300,
        )


def render_final_strategy_drilldown(result: dict | None):
    """Unified Stage 6 strategy drill-down across NSGA, MC and Decision outputs."""
    st.markdown("### 전략별 통합 Drill-down")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    reps = result.get("representative_alternatives_df", pd.DataFrame())
    mc = result.get("mc_summary_df", pd.DataFrame())
    decision = result.get("decision_table_df", pd.DataFrame())
    inv = result.get("representative_inventory_df", pd.DataFrame())
    part = result.get("mc_part_summary_df", pd.DataFrame())

    if not isinstance(reps, pd.DataFrame) or reps.empty:
        st.info("대표 대안 데이터가 없습니다.")
        return

    strategies = reps["Strategy"].astype(str).tolist()
    strategy = st.selectbox("통합 Drill-down 전략", strategies, key="stage6_strategy_drilldown")
    rep_hit = reps[reps["Strategy"].astype(str) == strategy]
    if rep_hit.empty:
        st.warning("선택 전략이 없습니다.")
        return
    rep = rep_hit.iloc[0]

    period_options = [12, 24, 36]
    if isinstance(mc, pd.DataFrame) and not mc.empty:
        period_options = sorted(
            pd.to_numeric(mc["Period_Months"], errors="coerce").dropna().astype(int).unique().tolist()
        )
    period = st.selectbox(
        "Drill-down 기간",
        period_options,
        index=max(len(period_options)-1, 0),
        format_func=lambda x: f"{int(x)}개월",
        key="stage6_drill_period",
    )

    mrow = pd.Series(dtype=object)
    if isinstance(mc, pd.DataFrame) and not mc.empty:
        mhit = mc[
            (mc["Strategy"].astype(str) == strategy)
            & (pd.to_numeric(mc["Period_Months"], errors="coerce").astype(int) == int(period))
            & (
                (mc["Evaluation_Method"].astype(str) == "건수 가중")
                if "Evaluation_Method" in mc.columns
                else True
            )
        ]
        if not mhit.empty:
            mrow = mhit.iloc[0]

    drow = pd.Series(dtype=object)
    if isinstance(decision, pd.DataFrame) and not decision.empty:
        dhit = decision[decision["Strategy"].astype(str) == strategy]
        if not dhit.empty:
            drow = dhit.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("재고", f"{int(float(rep.get('Stock_Units', 0))):,} EA")
    with c2:
        show_metric_card("NSGA 비용", f"{float(rep.get('NSGA_Cost_KRW', 0.0)):,.0f}원")
    with c3:
        show_metric_card("Worst Ao", f"{float(rep.get('Worst_Ao', 0.0)):.4f}")
    with c4:
        show_metric_card("Tail Risk", f"{float(rep.get('Tail_Risk', 0.0)):.4f}")

    if not mrow.empty:
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            show_metric_card("Service P05", f"{float(mrow.get('Service_P05', 0.0))*100:.2f}%")
        with q2:
            show_metric_card("재고부족 0회 확률", f"{float(mrow.get('Probability_Zero_Shortage', 0.0))*100:.2f}%")
        with q3:
            show_metric_card("Shortage P95", f"{float(mrow.get('Shortage_P95', 0.0)):.2f}")
        with q4:
            show_metric_card("평균 지연일", f"{float(mrow.get('Mean_Event_Delay_Days', 0.0)):.2f}")

    if not drow.empty:
        st.markdown(
            f"""
            <div class="section-box">
                <b>Decision Layer · {clean_chart_text(strategy)}</b><br>
                미래 안정성 상대등급: <b>{clean_chart_text(drow.get('Future_Stability_Grade','-'))}</b> &nbsp;·&nbsp;
                부족위험 상대등급: <b>{clean_chart_text(drow.get('Shortage_Risk_Grade','-'))}</b> &nbsp;·&nbsp;
                비용효율 상대등급: <b>{clean_chart_text(drow.get('Cost_Efficiency_Grade','-'))}</b><br>
                ※ 상대등급은 같은 기간 대표대안끼리의 비교표시이며 절대 인증등급이 아닙니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("#### 품목별 초기재고 구성")
        st.caption(
            "Maint_Echelon은 정비계층, Stocking_Echelon은 선택한 PBL 보급구조에 따른 실제 재고배치 계층입니다."
        )
        if isinstance(inv, pd.DataFrame) and not inv.empty:
            sub = inv[inv["Strategy"].astype(str) == strategy].copy()
            cols = [c for c in [
                "part_id", "maint_echelon", "Stock_Qty", "stock_O", "stock_I", "stock_D",
                "lead_time", "qpa", "repairable_flag", "repair_tat_h",
            ] if c in sub.columns]
            st.dataframe(sub[cols], use_container_width=True, height=430)
        else:
            st.info("재고 구성 데이터가 없습니다.")

    with right:
        st.markdown("#### 품목별 미래 Risk")
        if isinstance(part, pd.DataFrame) and not part.empty:
            psub = part[
                (part["Strategy"].astype(str) == strategy)
                & (pd.to_numeric(part["Period_Months"], errors="coerce").astype(int) == int(period))
            ].copy()
            if not psub.empty:
                psub["Risk_Priority_Index"] = (
                    pd.to_numeric(psub.get("Shortage_Mean", 0.0), errors="coerce").fillna(0.0)
                    * (
                        1.0
                        + pd.to_numeric(
                            psub.get("Mean_Shortage_Delay_Days", 0.0),
                            errors="coerce",
                        ).fillna(0.0)
                    )
                )
                psub = psub.sort_values(
                    ["Risk_Priority_Index", "Demand_Mean"],
                    ascending=[False, False],
                )
                st.dataframe(psub, use_container_width=True, height=430)
            else:
                st.info("해당 전략/기간의 품목 Risk 데이터가 없습니다.")
        else:
            st.info("품목별 미래 Risk 데이터가 없습니다.")


def render_final_audit_panel(result: dict | None):
    """Stage 6 regression / consistency audit."""
    st.markdown("### 최종 Audit · 회귀검증")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return

    summary_df = result.get("audit_summary_df", pd.DataFrame())
    checks_df = result.get("audit_checks_df", pd.DataFrame())
    strategy_df = result.get("audit_strategy_df", pd.DataFrame())
    part_df = result.get("audit_part_risk_df", pd.DataFrame())
    cfg_df = result.get("audit_config_df", pd.DataFrame())

    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        st.info("최종 Audit 결과가 없습니다.")
        return

    s = summary_df.iloc[0]
    status = str(s.get("Overall_Status", "-"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Audit 상태", status)
    with c2:
        show_metric_card("PASS", f"{int(s.get('Pass_Count', 0)):,}")
    with c3:
        show_metric_card("FAIL", f"{int(s.get('Fail_Count', 0)):,}")
    with c4:
        show_metric_card("전체 검사", f"{int(s.get('Total_Checks', 0)):,}")

    if status != "PASS":
        st.warning(
            "Audit에서 확인이 필요한 항목이 있습니다. "
            "아래 FAIL 행을 확인한 뒤 원자료/설정을 검토해 주세요."
        )
    else:
        st.success("NSGA → Bridge → Monte Carlo → Decision Layer 간 자동 일관성 검사를 통과했습니다.")

    tabs = st.tabs(["검사결과", "전략 통합표", "품목 Risk", "Audit 설정"])
    with tabs[0]:
        st.dataframe(checks_df, use_container_width=True, hide_index=True, height=470)
    with tabs[1]:
        st.dataframe(strategy_df, use_container_width=True, hide_index=True, height=360)
    with tabs[2]:
        st.dataframe(part_df, use_container_width=True, hide_index=True, height=470)
    with tabs[3]:
        st.dataframe(cfg_df, use_container_width=True, hide_index=True, height=180)



def render_integrated_results(result: dict | None):
    st.markdown("### 통합 결과")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return
    tabs = st.tabs(["요약", "파레토/스윕", "Model C Robust", "대표해 Bridge", "Best Policy", "Stage 3 Audit", "다운로드"])
    pareto = result.get("pareto_df", pd.DataFrame())
    all_runs = result.get("all_target_runs_df", pd.DataFrame())
    source = all_runs if isinstance(all_runs, pd.DataFrame) and not all_runs.empty else pareto

    with tabs[0]:
        if isinstance(source, pd.DataFrame) and not source.empty:
            st.plotly_chart(build_ce_curve_plotly(source, "통합 Target Sweep Overlay (C-E Curve)"), use_container_width=True)
        render_summary_panel(result.get("summary", {}))
    with tabs[1]:
        st.dataframe(source, use_container_width=True, height=520) if isinstance(source, pd.DataFrame) else st.info("데이터가 없습니다.")
    with tabs[2]:
        render_model_c_robust_panel(result)
    with tabs[3]:
        render_representative_bridge_panel(result)
    with tabs[4]:
        p = result.get("policy_df", pd.DataFrame())
        st.dataframe(p, use_container_width=True, height=520) if isinstance(p, pd.DataFrame) and not p.empty else st.info("Best Policy가 없습니다.")
    with tabs[5]:
        render_stage1_readiness(result)
        audit = result.get("representative_selection_audit_df", pd.DataFrame())
        if isinstance(audit, pd.DataFrame) and not audit.empty:
            st.markdown("**Stage 3 Representative Selection Audit**")
            st.dataframe(audit, use_container_width=True, hide_index=True, height=360)
    with tabs[6]:
        st.download_button(
            "📥 결과 엑셀 다운로드",
            data=make_excel_download(result),
            file_name="NSGA_result_dashboard_SUPPLY_STRUCTURE_OPTION.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def render_xai(result: dict | None):
    st.markdown("### Explainable AI")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return
    policy = result.get("policy_df", pd.DataFrame())
    if not isinstance(policy, pd.DataFrame) or policy.empty:
        st.info("설명 가능한 policy 데이터가 없습니다.")
        return
    df = policy.copy()
    if "recommended_stock" not in df.columns:
        df["recommended_stock"] = 0
    if "impact_score" not in df.columns:
        df["impact_score"] = 0.0
    if "priority_score" not in df.columns:
        df["priority_score"] = 0.0
    if "lead_time" not in df.columns:
        df["lead_time"] = 0.0
    df["manage_flag"] = pd.to_numeric(df.get("manage_flag", 0), errors="coerce").fillna(0).astype(int)
    df["xai_reason"] = np.where(
        df["manage_flag"] > 0,
        "현재 대표 해에서 관리 대상으로 선택됨",
        "현재 대표 해에서 비관리 대상으로 분류됨",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        show_metric_card("Selected", f"{int((df['manage_flag']>0).sum()):,}")
    with c2:
        show_metric_card("Total", f"{len(df):,}")
    with c3:
        show_metric_card("Stock Units", f"{int(pd.to_numeric(df['recommended_stock'], errors='coerce').fillna(0).sum()):,}")
    st.dataframe(df, use_container_width=True, height=560)


def render_prescriptive(result: dict | None):
    st.markdown("### Prescriptive AI")
    if not result:
        st.info("아직 실행 결과가 없습니다.")
        return
    policy = result.get("policy_df", pd.DataFrame())
    if not isinstance(policy, pd.DataFrame) or policy.empty:
        st.info("추천 가능한 policy 데이터가 없습니다.")
        return

    df = policy.copy()
    manage = pd.to_numeric(df.get("manage_flag", 0), errors="coerce").fillna(0).astype(int) > 0
    prebuy = df.get("prebuy_flag", False)
    protection = df.get("protection_flag", False)
    prebuy = pd.Series(prebuy, index=df.index).fillna(False).astype(bool)
    protection = pd.Series(protection, index=df.index).fillna(False).astype(bool)

    action = np.where(
        prebuy, "즉시 선발주",
        np.where(protection, "보호재고 유지",
                 np.where(manage, "유지 + 모니터링", "보류 / 후순위"))
    )
    df["recommended_action"] = action

    counts = df["recommended_action"].value_counts()
    cols = st.columns(4)
    for col, label in zip(cols, ["즉시 선발주", "보호재고 유지", "유지 + 모니터링", "보류 / 후순위"]):
        with col:
            show_metric_card(label, f"{int(counts.get(label,0)):,}")
    st.dataframe(df, use_container_width=True, height=520)


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
        kwargs2 = {k: v for k, v in config.__dict__.items() if k in sig.parameters}
        kwargs2.update(kwargs)
        result = run_nsga2(df, **kwargs2)
    if not isinstance(result, dict):
        raise ValueError("엔진이 dict 형태 결과를 반환하지 않았습니다.")
    return result


# ------------------------------------------------------------
# Hero
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">NSGA-II 분석 대시보드</div>
        <div class="hero-sub">
            사업별 PBL 보급구조를 선택해 O/I/D 계층형 또는 D 직접지원형으로 재고·Ao·Risk를 일관되게 평가하고,
            품목별 K 기반 Cold-start Monte Carlo와 3개 평가방식까지 재검증한 FINAL DEBUGGED 통합본입니다.
            자동 최적안은 선택하지 않으며, 원자료·산식·Final Audit를 함께 확인할 수 있습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 실행 설정")
    uploaded_file = st.file_uploader("입력 파일 업로드", type=["csv", "xlsx", "xls"])
    n_generations = st.slider("세대 수", 20, 300, 200, 10)
    random_seed = st.number_input("랜덤 시드", 0, 999999, 42, 1)
    target_ao = st.slider("Target Ao", 0.10, 0.99, 0.94, 0.01)
    analysis_years = st.slider(
        "Analysis Years", 1, 20, 5, 1,
        help="분석기간(년). 기간 수요, DT 누적, 비용 및 Ao 산출에 반영됩니다.",
    )
    pmin = st.slider("Pmin", 0.00, 1.00, 0.01, 0.01)
    long_lead_percentile = st.slider("Long_Lead_Percentile", 0.50, 0.99, 0.90, 0.01)
    ao_impact_percentile = st.slider("Ao_Impct_Percentile", 0.50, 0.99, 0.70, 0.01)

    supply_structure_label = st.selectbox(
        "PBL 보급구조",
        options=[
            "정비계층 연계형 (O/I/D)",
            "창 직보급형 (D 직접지원)",
        ],
        index=0,
        help=(
            "정비계층 연계형: 품목의 Maint_Echelon에 따라 O/I/D 재고를 배치합니다. "
            "창 직보급형: 모든 PBL 재고를 D-level에 집중하여 창에서 운용부대를 직접 지원하는 구조로 평가합니다."
        ),
    )
    supply_structure_mode = (
        "MAINT_ECHELON"
        if supply_structure_label.startswith("정비계층")
        else "DEPOT_DIRECT"
    )

    st.markdown(
        f"""
        <div class="run-param-box">
            <span class="info-chip">세대 수 {n_generations}</span>
            <span class="info-chip">시드 {int(random_seed)}</span>
            <span class="info-chip">Target Ao {target_ao:.2f}</span>
            <span class="info-chip">분석기간 {int(analysis_years)}년</span>
            <span class="info-chip">Pmin {pmin:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage1_preview_cfg = NSGAConfig(supply_structure_mode=str(supply_structure_mode))
    stage1_preview = validate_stage1_config(stage1_preview_cfg)
    with st.expander("🔮 미래 시뮬레이션 설정", expanded=False):
        run_mc_after_nsga = st.checkbox(
            "NSGA 완료 후 Monte Carlo 실행",
            value=True,
            help="대표해 Bridge 5개 대안을 동일 미래수요 조건에서 비교합니다.",
        )
        mc_iterations = st.number_input(
            "Monte Carlo 반복 수",
            min_value=100,
            max_value=10000,
            value=3000,
            step=100,
        )
        decision_period_months = st.selectbox(
            "비용·Risk 의사결정 기준 기간",
            options=[12, 24, 36],
            index=2,
            format_func=lambda x: f"{int(x)}개월",
            help="등급과 비용효율 비교는 선택한 동일 기간의 대표대안끼리만 수행합니다.",
        )
        st.caption("기간: 12 / 24 / 36개월 · K~LogNormal(P05≈0.5, Median=1.0, P95≈2.0)")
    st.caption(f"현재 보급구조: **{supply_structure_label}**")
    run_btn = st.button("🚀 NSGA + 미래 시뮬레이션 실행", use_container_width=True)


# ------------------------------------------------------------
# Load / preview
# ------------------------------------------------------------
df_input = None
xdf = None
if uploaded_file is not None:
    try:
        df_input = read_uploaded_file(uploaded_file)
        st.session_state.uploaded_name = uploaded_file.name
        preview_input_config = NSGAConfig(
            population_size=80,
            n_generations=int(n_generations),
            random_seed=int(random_seed),
            target_ao=float(target_ao),
            representative_target=float(target_ao),
            years=float(analysis_years),
            pmin=float(pmin),
            long_lead_percentile=float(long_lead_percentile),
            ao_impact_percentile=float(ao_impact_percentile),
            supply_structure_mode=str(supply_structure_mode),
            analysis_model="MODEL_C",
            robust_enabled=True,
            robust_candidate_expansion_enabled=True,
            robust_k_grid=[0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
        )
        xdf = prepare_input_dataframe(df_input, config=preview_input_config)
        st.session_state.precheck_info = build_precheck_info(df_input, xdf)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if run_btn:
    if df_input is None:
        st.warning("먼저 입력 파일을 업로드해 주세요.")
    else:
        preview_config = NSGAConfig(
            population_size=80,
            n_generations=n_generations,
            random_seed=int(random_seed),
            target_ao=target_ao,
            representative_target=float(target_ao),
            years=float(analysis_years),
            pmin=pmin,
            long_lead_percentile=long_lead_percentile,
            ao_impact_percentile=ao_impact_percentile,
            supply_structure_mode=str(supply_structure_mode),

            # STAGE 2 / MODEL C robust settings.
            analysis_model="MODEL_C",
            robust_enabled=True,
            robust_candidate_expansion_enabled=True,
            robust_k_grid=[0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
            robust_tail_quantile=0.80,
            monte_carlo_enabled=bool(run_mc_after_nsga),
            monte_carlo_iterations=int(mc_iterations),
            monte_carlo_seed=int(random_seed),
            representative_bridge_enabled=True,
            representative_strategy_count=5,
            representative_stock_quantiles=[0.00, 0.25, 0.50, 0.75, 1.00],
        )

        readiness = validate_stage1_config(preview_config)
        if not readiness["stage1_ready"]:
            st.error("Stage 1 설정 검증에 실패했습니다.")
            for issue in readiness["issues"]:
                st.error(issue)
            st.stop()

        target_grid = list(getattr(preview_config, "ao_target_grid", [target_ao]))
        rep_target = float(getattr(preview_config, "representative_target", target_ao))
        sorted_grid = sorted(set(float(x) for x in target_grid + [rep_target]))
        n_targets = len(sorted_grid)

        overall_progress_bar = st.progress(1, text="분석 준비 중...")
        stage_progress_bar = st.progress(1, text="현재 Target Ao 단계 준비 중...")
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

        def _render_target_status(active_target=None):
            ts_df = (
                pd.DataFrame(list(st.session_state.target_status_map.values()))
                .sort_values("Target Ao")
                .reset_index(drop=True)
            )

            if active_target is not None and not ts_df.empty:
                def _highlight_active(row):
                    try:
                        is_active = (
                            str(row.get("Status", "")) == "진행 중"
                            and abs(float(row.get("Target Ao")) - float(active_target)) < 1e-12
                        )
                    except Exception:
                        is_active = False
                    style = (
                        "background-color: #E8F5E9; font-weight: 700;"
                        if is_active else ""
                    )
                    return [style] * len(row)

                # Auto-follow viewport: keep the active Target Ao row visible
                # without requiring the user to drag the dataframe scrollbar.
                active_matches = ts_df.index[
                    np.isclose(
                        pd.to_numeric(ts_df["Target Ao"], errors="coerce"),
                        float(active_target),
                        atol=1e-12,
                    )
                ].tolist()
                if active_matches:
                    active_pos = int(active_matches[0])
                    window_size = 7
                    start_pos = max(0, active_pos - window_size // 2)
                    end_pos = min(len(ts_df), start_pos + window_size)
                    start_pos = max(0, end_pos - window_size)
                    ts_view = ts_df.iloc[start_pos:end_pos].copy()
                else:
                    ts_view = ts_df.copy()

                styled = ts_view.style.apply(_highlight_active, axis=1)
                target_status_box.dataframe(
                    styled,
                    use_container_width=True,
                    height=min(260, 42 + 35 * max(len(ts_view), 1)),
                )
            else:
                target_status_box.dataframe(
                    ts_df,
                    use_container_width=True,
                    height=260,
                )

            if st.session_state.generation_logs:
                log_df = pd.DataFrame(st.session_state.generation_logs).tail(240).reset_index(drop=True)

                def _highlight_active_log(row):
                    try:
                        same_target = (
                            active_target is not None
                            and abs(float(row.get("target_ao")) - float(active_target)) < 1e-12
                        )
                        if not same_target:
                            return [""] * len(row)

                        active_rows = ts_df[
                            np.isclose(
                                pd.to_numeric(ts_df["Target Ao"], errors="coerce"),
                                float(active_target),
                                atol=1e-12,
                            )
                        ]
                        active_gen = (
                            int(active_rows.iloc[0]["Generation"])
                            if not active_rows.empty
                            else None
                        )
                        is_active = active_gen is not None and int(row.get("gen", -1)) == active_gen
                    except Exception:
                        is_active = False

                    style = "background-color: #E8F5E9; font-weight: 700;" if is_active else ""
                    return [style] * len(row)

                # Auto-follow viewport: the newest/current generation stays in view.
                # We intentionally render a moving window instead of trying to manipulate
                # Streamlit's internal browser scrollbar, which is not a stable public API.
                active_indices = []
                try:
                    active_indices = log_df.index[
                        (
                            np.isclose(
                                pd.to_numeric(log_df["target_ao"], errors="coerce"),
                                float(active_target),
                                atol=1e-12,
                            )
                        )
                        & (
                            pd.to_numeric(log_df["gen"], errors="coerce")
                            == float(active_gen)
                        )
                    ].tolist()
                except Exception:
                    active_indices = []

                if active_indices:
                    active_log_pos = int(active_indices[-1])
                    log_window = 11
                    start_log = max(0, active_log_pos - log_window + 1)
                    log_view = log_df.iloc[start_log:active_log_pos + 1].copy()
                else:
                    log_view = log_df.tail(11).copy()

                styled_logs = log_view.style.apply(_highlight_active_log, axis=1)
                status_box.dataframe(
                    styled_logs,
                    use_container_width=True,
                    height=min(420, 42 + 35 * max(len(log_view), 1)),
                )

        _render_target_status()

        def progress_callback(gen, total_gen, best_summary):
            elapsed = time.time() - start_ts
            current_ts = time.time()
            current_target = float(best_summary.get("current_target_ao", rep_target)) if isinstance(best_summary, dict) else rep_target
            gen_safe = max(int(gen), 0)
            total_gen_safe = max(int(total_gen), 1)
            stage_progress = max(1, min(100, int(100 * gen_safe / total_gen_safe)))

            emit_state = st.session_state.progress_emit_state
            should_emit = should_emit_progress_update(
                gen_safe, total_gen_safe,
                int(emit_state.get("last_emit_gen", -1)),
                float(emit_state.get("last_emit_ts", 0.0)),
                current_ts,
                current_target,
                emit_state.get("last_target"),
            )

            if current_target not in st.session_state.target_status_map:
                st.session_state.target_status_map[current_target] = {
                    "Target Ao": current_target, "Stage": "-", "Status": "진행 중",
                    "Generation": gen_safe, "Total Gen": total_gen_safe,
                    "Best Ao": None, "Mean Ao": None, "Best Cost": None,
                }

            row = st.session_state.target_status_map[current_target]
            row["Status"] = "진행 중"
            row["Generation"] = gen_safe
            row["Total Gen"] = total_gen_safe
            if isinstance(best_summary, dict):
                row["Best Ao"] = best_summary.get("best_ao", best_summary.get("best_Ao"))
                row["Mean Ao"] = best_summary.get("mean_ao", best_summary.get("mean_Ao"))
                row["Best Cost"] = best_summary.get("best_cost")

            if should_emit:
                st.session_state.generation_logs.append({
                    "elapsed_sec": round(elapsed, 1),
                    "target_ao": current_target,
                    "gen": gen_safe,
                    "total_gen": total_gen_safe,
                    "best_ao": row["Best Ao"],
                    "mean_ao": row["Mean Ao"],
                    "best_cost": row["Best Cost"],
                })
                overall_progress_bar.progress(stage_progress, text=f"전체 Sweep 진행 중... Target Ao {current_target:.2f}")
                stage_progress_bar.progress(stage_progress, text=f"Target Ao {current_target:.2f}: {gen_safe}/{total_gen_safe} 세대")
                heartbeat_box.caption(f"경과시간 {elapsed:,.1f}초 | Target Ao {current_target:.2f}")
                _render_target_status(current_target)
                st.session_state.progress_emit_state = {
                    "last_emit_gen": gen_safe,
                    "last_emit_ts": current_ts,
                    "last_target": current_target,
                }

        try:
            result = call_engine(df_input, preview_config, progress_callback)

            if bool(run_mc_after_nsga):
                bridge_df = result.get("monte_carlo_bridge_df", pd.DataFrame())
                if not isinstance(bridge_df, pd.DataFrame) or bridge_df.empty:
                    raise ValueError("Stage 4 Monte Carlo를 실행할 대표해 Bridge 데이터가 없습니다.")

                mc_config = MonteCarloConfig(
                    iterations=int(mc_iterations),
                    random_seed=int(random_seed),
                    period_months=[12, 24, 36],
                    k_distribution="lognormal",
                    k_median=1.0,
                    k_p05=0.5,
                    k_p95=2.0,
                    common_random_numbers=True,
                    store_iteration_detail=True,
                )
                mc_check = validate_monte_carlo_config(mc_config)
                if not mc_check.get("ready", False):
                    raise ValueError("Monte Carlo 설정 오류: " + "; ".join(mc_check.get("issues", [])))

                mc_progress = st.progress(1, text="Cold-start Monte Carlo 준비 중...")

                def _mc_progress_callback(payload):
                    frac = float(payload.get("Progress", 0.0))
                    period_m = int(payload.get("Period_Months", 0))
                    iteration_now = int(payload.get("Iteration", 0))
                    total_now = int(payload.get("Iterations", mc_iterations))
                    mc_progress.progress(
                        max(1, min(100, int(frac * 100))),
                        text=f"미래 시뮬레이션 {period_m}개월 · {iteration_now}/{total_now}회",
                    )

                mc_result = run_coldstart_monte_carlo(
                    bridge_df=bridge_df,
                    config=mc_config,
                    progress_callback=_mc_progress_callback,
                )
                result.update(mc_result)
                result.setdefault("summary", {})["future_monte_carlo_connected"] = True
                result["summary"]["monte_carlo_iterations"] = int(mc_iterations)
                result["summary"]["monte_carlo_period_months"] = "12,24,36"

                decision_config = DecisionConfig(
                    decision_period_months=int(decision_period_months),
                    primary_evaluation_method="건수 가중",
                    tie_tolerance=1e-9,
                    cost_unit_krw=100_000_000.0,
                )
                decision_check = validate_decision_config(decision_config)
                if not decision_check.get("ready", False):
                    raise ValueError(
                        "Decision Layer 설정 오류: "
                        + "; ".join(decision_check.get("issues", []))
                    )

                decision_result = build_decision_package(
                    representative_alternatives_df=result.get(
                        "representative_alternatives_df", pd.DataFrame()
                    ),
                    mc_summary_df=result.get("mc_summary_df", pd.DataFrame()),
                    config=decision_config,
                )
                result.update(decision_result)
                result["summary"]["decision_layer_connected"] = True
                result["summary"]["decision_period_months"] = int(
                    decision_result["decision_summary"]["decision_period_months"]
                )
                result["summary"]["automatic_best_selection"] = False

                audit_config = AuditConfig(
                    probability_tolerance=1e-9,
                    numeric_tolerance=1e-8,
                    expected_periods=(12, 24, 36),
                )
                audit_check = validate_audit_config(audit_config)
                if not audit_check.get("ready", False):
                    raise ValueError(
                        "Final Audit 설정 오류: "
                        + "; ".join(audit_check.get("issues", []))
                    )

                audit_result = build_final_audit_package(
                    result=result,
                    config=audit_config,
                )
                result.update(audit_result)
                result["summary"]["final_audit_connected"] = True
                result["summary"]["final_audit_status"] = str(
                    audit_result["audit_summary"]["overall_status"]
                )
                result["summary"]["final_audit_pass_count"] = int(
                    audit_result["audit_summary"]["pass_count"]
                )
                result["summary"]["final_audit_fail_count"] = int(
                    audit_result["audit_summary"]["fail_count"]
                )
                mc_progress.progress(
                    100,
                    text="Cold-start Monte Carlo + Decision Layer + Final Audit 완료",
                )
            else:
                result.setdefault("summary", {})["future_monte_carlo_connected"] = False
                result["summary"]["decision_layer_connected"] = False
                result["summary"]["automatic_best_selection"] = False
                result["summary"]["final_audit_connected"] = False
                result["summary"]["final_audit_status"] = "NOT_RUN"

            for k in st.session_state.target_status_map:
                st.session_state.target_status_map[k]["Status"] = "완료"
            overall_progress_bar.progress(100, text="전체 Target Sweep 완료")
            stage_progress_bar.progress(100, text="마지막 Target Ao 단계 완료")
            heartbeat_box.caption("현재 상태: 분석 완료")
            st.session_state.run_result = result
            _render_target_status()
            st.success("NSGA 분석이 완료되었습니다.")
        except Exception as e:
            st.error("엔진 실행 중 오류가 발생했습니다.")
            st.exception(e)


# ------------------------------------------------------------
# Main tabs
# ------------------------------------------------------------
main_tabs = st.tabs([
    "📌 최종 요약",
    "📋 데이터 개요",
    "📊 데이터 시각화",
    "⚙️ 실행 이력",
    "🏆 통합 결과",
    "🔮 미래 시뮬레이션",
    "🎯 비용·Risk 의사결정",
    "🔎 전략 Drill-down",
    "🧾 최종 Audit",
    "🧠 Explainable AI",
    "🧭 Prescriptive AI",
])
result = st.session_state.run_result


with main_tabs[0]:
    render_final_executive_panel(result)

with main_tabs[1]:
    if df_input is None:
        st.info("먼저 입력 파일을 업로드해 주세요.")
    else:
        render_preview_tabs(df_input, xdf)

with main_tabs[2]:
    if df_input is None:
        st.info("먼저 입력 파일을 업로드해 주세요.")
    else:
        render_visual_tabs(df_input, xdf)

with main_tabs[3]:
    render_run_history(result)

with main_tabs[4]:
    render_integrated_results(result)

with main_tabs[5]:
    render_future_simulation_panel(result)

with main_tabs[6]:
    render_cost_risk_decision_panel(result)

with main_tabs[7]:
    render_final_strategy_drilldown(result)

with main_tabs[8]:
    render_final_audit_panel(result)

with main_tabs[9]:
    render_xai(result)

with main_tabs[10]:
    render_prescriptive(result)
