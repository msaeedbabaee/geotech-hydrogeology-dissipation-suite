"""
Hydrogeological Aquifer Testing & CPTu Dissipation Analysis Suite.
Features Theis & Cooper-Jacob pumping test interpretation, boundary detection,
and Houlsby & Teh (1988) CPTu pore pressure dissipation modeling.
Strictly compliant with the Canadian Foundation Engineering Manual (CFEM Ch 5).
"""

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, root_scalar
from scipy.special import exp1
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CORE HYDROGEOLOGICAL & DISSIPATION COMPUTATIONAL ENGINES (OOP)
# -----------------------------------------------------------------------------

class PumpingTestEngine:
    """Analytical evaluation of transient drawdown using Theis and Cooper-Jacob methods."""

    @staticmethod
    def theis_well_function(u: np.ndarray) -> np.ndarray:
        """Calculates exponential integral W(u) using infinite series convergence."""
        u_arr = np.asarray(u, dtype=float)
        w = np.zeros_like(u_arr)
        valid = u_arr > 0
        w[valid] = exp1(u_arr[valid])
        return w

    @classmethod
    def calculate_theis_drawdown(
        cls,
        time_days: np.ndarray,
        q_rate_m3_day: float,
        transmissivity: float,
        storativity: float,
        radius_m: float
    ) -> np.ndarray:
        """Computes transient drawdown profile s(r, t) via Theis formulation."""
        t_clean = np.maximum(time_days, 1e-6)
        u = (radius_m ** 2 * storativity) / (4.0 * transmissivity * t_clean)
        w_u = cls.theis_well_function(u)
        drawdown = (q_rate_m3_day / (4.0 * np.pi * transmissivity)) * w_u
        return np.maximum(drawdown, 0.0)

    @classmethod
    def fit_cooper_jacob(
        cls,
        time_min: np.ndarray,
        drawdown_m: np.ndarray,
        q_m3_day: float,
        radius_m: float,
        t_min_threshold: float = 10.0
    ) -> Dict:
        """
        Fits straight line to semi-logarithmic drawdown data (s vs log10(t)) for t >= 10 min.
        Computes T, S, and detects hydraulic recharge/barrier boundaries.
        """
        mask = time_min >= t_min_threshold
        if np.sum(mask) < 4:
            mask = np.ones_like(time_min, dtype=bool)

        t_fit = time_min[mask]
        s_fit = drawdown_m[mask]
        log10_t = np.log10(t_fit)

        # Linear regression: s = slope * log10(t) + intercept
        coeffs = np.polyfit(log10_t, s_fit, 1)
        slope = coeffs[0]      # Drawdown per log cycle (Delta s)
        intercept = coeffs[1]

        delta_s = max(float(slope), 1e-4)

        # Transmissivity T (m2/day) [CFEM Eq. 5.84]
        transmissivity = (2.303 * q_m3_day) / (4.0 * np.pi * delta_s)

        # Intercept t0 where drawdown = 0: log10(t0) = -intercept / slope
        log10_t0 = -intercept / slope
        t0_min = 10.0 ** log10_t0
        t0_day = max(t0_min / 1440.0, 1e-7)

        # Storativity S [CFEM Eq. 5.85]
        storativity = (2.25 * transmissivity * t0_day) / (radius_m ** 2)
        storativity = float(np.clip(storativity, 1e-6, 0.35))

        # Boundary condition detection via late-time derivative
        boundary_diagnosis = "Infinite Acting Aquifer (Homogeneous Flow)"
        if len(t_fit) >= 6:
            mid_idx = len(t_fit) // 2
            slope_early = np.polyfit(log10_t[:mid_idx], s_fit[:mid_idx], 1)[0]
            slope_late = np.polyfit(log10_t[mid_idx:], s_fit[mid_idx:], 1)[0]
            ratio = slope_late / max(slope_early, 1e-4)

            if ratio < 0.65:
                boundary_diagnosis = "Constant-Head Recharge Boundary Detected (River/Lake Infiltration)"
            elif ratio > 1.45:
                boundary_diagnosis = "Impermeable Barrier / Negative Boundary Detected (Aquifer Thinning/Fault)"

        return {
            "Transmissivity_m2_day": round(transmissivity, 2),
            "Transmissivity_m2_s": transmissivity / 86400.0,
            "Storativity": storativity,
            "Delta_s_per_cycle_m": round(delta_s, 3),
            "t0_min": round(t0_min, 4),
            "Slope": slope,
            "Intercept": intercept,
            "Boundary_Type": boundary_diagnosis
        }


class CPTuDissipationEngine:
    """Analyzes pore pressure dissipation (PPD) records per Houlsby & Teh (1988)."""

    @staticmethod
    def identify_dissipation_behavior(time_s: np.ndarray, u2_kpa: np.ndarray) -> str:
        """Determines whether dissipation curve exhibits monotonic or dilatory response."""
        if len(u2_kpa) < 3:
            return "Monotonic"
        u_initial = u2_kpa[0]
        u_max = np.max(u2_kpa)
        if (u_max - u_initial) > 0.05 * u_initial and np.argmax(u2_kpa) > 1:
            return "Dilatory (Overconsolidated / Fissured Fabric)"
        return "Monotonic (Normally Consolidated / Soft Clay)"

    @classmethod
    def evaluate_dissipation_properties(
        cls,
        time_s: np.ndarray,
        u2_kpa: np.ndarray,
        u0_static_kpa: float,
        cone_area_cm2: float = 10.0,
        rigidity_index: float = 100.0,
        liquid_limit: Optional[float] = None
    ) -> Dict:
        """
        Computes t50, horizontal consolidation coefficient ch, and permeability k.
        Implements Houlsby & Teh (1988) [CFEM Eq. 5.45] and empirical correlations.
        """
        behavior = cls.identify_dissipation_behavior(time_s, u2_kpa)
        u_max = np.max(u2_kpa)
        delta_u_max = max(u_max - u0_static_kpa, 1.0)
        target_u = u0_static_kpa + 0.5 * delta_u_max

        # Interpolate t50
        idx_peak = np.argmax(u2_kpa)
        t_decay = time_s[idx_peak:]
        u_decay = u2_kpa[idx_peak:]

        if np.min(u_decay) > target_u:
            # Extrapolate roughly if test was terminated prior to 50% consolidation
            t50 = float(np.max(time_s) * 1.5)
        else:
            t50 = float(np.interp(target_u, u_decay[::-1], t_decay[::-1]))

        t50 = max(t50, 1.0)

        # Probe radius ac (cm)
        ac_cm = 1.78 if cone_area_cm2 <= 10.0 else 2.22

        # Modified time factor for u2 shoulder filter: T*50 = 0.245 [CFEM Eq. 5.45]
        t_star_50 = 0.245
        ir = max(rigidity_index, 10.0)

        # Horizontal consolidation coefficient ch (cm2/s) [CFEM Eq. 5.45]
        ch_cm2_s = (t_star_50 * (ac_cm ** 2) * np.sqrt(ir)) / t50
        ch_m2_yr = ch_cm2_s * (0.0001 * 86400.0 * 365.25)

        # Horizontal permeability k_h (cm/s) from t50 [CFEM Eq. 5.47]
        k_t50_cm_s = (251.0 * t50) ** (-1.25)
        k_t50_m_s = k_t50_cm_s * 0.01

        # Kozeny-Carman based on Liquid Limit (Chapuis & Aubertin 2003) [CFEM Eq. 5.75 & 5.77]
        k_kozeny_m_s = np.nan
        if liquid_limit is not None and liquid_limit > 15.0:
            ll = min(liquid_limit, 120.0)
            spec_surface = 1.0 / max(1.3513 - 0.0089 * ll, 0.1)
            # Representative estimate of clay permeability
            k_kozeny_m_s = 1e-9 * (100.0 / spec_surface) ** 2

        return {
            "Behavior": behavior,
            "t50_s": round(t50, 1),
            "Target_u50_kPa": round(target_u, 1),
            "ch_cm2_s": round(ch_cm2_s, 4),
            "ch_m2_yr": round(ch_m2_yr, 2),
            "k_h_from_t50_m_s": k_t50_m_s,
            "k_h_from_t50_cm_s": k_t50_cm_s,
            "k_Kozeny_Carman_m_s": k_kozeny_m_s,
            "Radius_ac_cm": ac_cm,
            "Rigidity_Index": ir
        }


# -----------------------------------------------------------------------------
# 2. SYNTHETIC BENCHMARK DATASET GENERATORS
# -----------------------------------------------------------------------------

def generate_benchmark_pumping_dataset() -> pd.DataFrame:
    """Generates synthetic transient drawdown data with a known constant-head recharge boundary."""
    np.random.seed(42)
    # Logarithmically spaced time readings (1 min to 1440 min = 24 hrs)
    time_min = np.array([
        1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60,
        90, 120, 180, 240, 360, 480, 720, 960, 1200, 1440
    ], dtype=float)

    # Reservoir properties: Q = 1800 m3/day, T = 350 m2/day, S = 0.0008, r = 25 m
    q_rate = 1800.0
    t_true = 350.0
    s_true = 0.0008
    r_well = 25.0

    t_days = time_min / 1440.0
    s_theis = PumpingTestEngine.calculate_theis_drawdown(t_days, q_rate, t_true, s_true, r_well)

    # Add realistic noise and simulate recharge boundary flattening after 360 min
    drawdown_obs = []
    for t_m, s_val in zip(time_min, s_theis):
        noise = np.random.normal(0, 0.015)
        if t_m > 300.0:
            flattening = 0.35 * np.log10(t_m / 300.0)
            s_val = max(s_val - flattening, 0.1)
        drawdown_obs.append(round(max(s_val + noise, 0.02), 3))

    return pd.DataFrame({
        "Time_min": time_min,
        "Time_day": np.round(t_days, 5),
        "Drawdown_m": drawdown_obs
    })


def generate_benchmark_dissipation_dataset(behavior_type: str = "Monotonic") -> pd.DataFrame:
    """Generates continuous CPTu pore pressure dissipation sounding records."""
    np.random.seed(123)
    time_s = np.array([
        0.5, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 900, 1500, 2400
    ], dtype=float)

    u0 = 80.0  # Static hydrostatic pore pressure (kPa)
    u_init = 420.0

    u2_profile = []
    if behavior_type == "Monotonic":
        for t in time_s:
            # Monotonic decay
            degree = 1.0 / (1.0 + (t / 110.0) ** 0.85)
            u_current = u0 + (u_init - u0) * degree + np.random.normal(0, 1.2)
            u2_profile.append(round(u_current, 1))
    else:  # Dilatory
        for t in time_s:
            if t <= 15.0:
                # Pore pressure buildup during stress redistribution
                u_current = u_init + 45.0 * (1.0 - np.exp(-t / 4.0)) + np.random.normal(0, 1.5)
            else:
                degree = 1.0 / (1.0 + ((t - 15.0) / 160.0) ** 0.9)
                u_current = u0 + (u_init + 40.0 - u0) * degree + np.random.normal(0, 1.5)
            u2_profile.append(round(u_current, 1))

    return pd.DataFrame({
        "Time_s": time_s,
        "Pore_Pressure_u2_kPa": u2_profile
    })


# -----------------------------------------------------------------------------
# 3. INTERACTIVE VISUALIZATION DASHBOARDS (PLOTLY)
# -----------------------------------------------------------------------------

class HydroVisualizer:
    """Builds interactive synchronized charts for pumping tests and dissipation profiles."""

    @staticmethod
    def create_pumping_test_dashboard(
        df_pump: pd.DataFrame,
        fit_results: Dict,
        radius_m: float
    ) -> go.Figure:
        """Draws Cooper-Jacob semi-log fit and diagnostic boundary analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Cooper-Jacob Semi-Log Analysis (s vs log₁₀ t)",
                "Diagnostic Drawdown Rate (ds / d(ln t))"
            )
        )

        t_min = df_pump["Time_min"].values
        s_obs = df_pump["Drawdown_m"].values

        # Subplot 1: Semi-log observation points
        fig.add_trace(
            go.Scatter(
                x=t_min, y=s_obs, mode='markers', name='Observed Drawdown',
                marker=dict(size=8, color='#1f77b4', symbol='circle'),
                hovertemplate='Time: %{x} min<br>Drawdown: %{y:.3f} m'
            ),
            row=1, col=1
        )

        # Draw linear Cooper-Jacob model
        t_model = np.logspace(np.log10(max(fit_results["t0_min"] * 0.8, 0.5)), np.log10(np.max(t_min)), 100)
        s_model = fit_results["Slope"] * np.log10(t_model) + fit_results["Intercept"]
        fig.add_trace(
            go.Scatter(
                x=t_model, y=s_model, mode='lines', name='Cooper-Jacob Straight Line Fit',
                line=dict(color='#d62728', dash='dash', width=2),
                hovertemplate='Fitted Drawdown: %{y:.3f} m'
            ),
            row=1, col=1
        )

        # Zero-intercept marker t0
        fig.add_trace(
            go.Scatter(
                x=[fit_results["t0_min"]], y=[0.0], mode='markers+text',
                name=f't₀ Intercept ({fit_results["t0_min"]:.2f} min)',
                text=[f't₀ = {fit_results["t0_min"]:.2f} min'],
                textposition='bottom right',
                marker=dict(size=11, color='#2ca02c', symbol='star')
            ),
            row=1, col=1
        )

        # Subplot 2: Derivative analysis for boundaries
        ln_t = np.log(t_min)
        ds_dln_t = np.gradient(s_obs, ln_t)
        fig.add_trace(
            go.Scatter(
                x=t_min, y=ds_dln_t, mode='lines+markers', name='Log Derivative ds/d(ln t)',
                line=dict(color='#9467bd', width=2), marker=dict(size=6),
                hovertemplate='Time: %{x} min<br>Derivative: %{y:.3f} m'
            ),
            row=1, col=2
        )

        fig.update_xaxes(type="log", title_text="Elapsed Time (min)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown s (m)", autorange="reversed", row=1, col=1)
        fig.update_xaxes(type="log", title_text="Elapsed Time (min)", row=1, col=2)
        fig.update_yaxes(title_text="Derivative ds / d(ln t) (m)", row=1, col=2)

        fig.update_layout(
            height=580,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            margin=dict(l=60, r=40, t=60, b=80)
        )
        return fig

    @staticmethod
    def create_dissipation_dashboard(
        df_diss: pd.DataFrame,
        diss_results: Dict,
        u0_static: float
    ) -> go.Figure:
        """Constructs interactive CPTu pore pressure dissipation curve and t50 threshold."""
        fig = go.Figure()

        time_s = df_diss["Time_s"].values
        u2 = df_diss["Pore_Pressure_u2_kPa"].values

        # Measured dissipation profile
        fig.add_trace(
            go.Scatter(
                x=time_s, y=u2, mode='lines+markers', name='Measured u₂(t)',
                line=dict(color='#ff7f0e', width=2.5),
                marker=dict(size=7, color='#d62728'),
                hovertemplate='Time: %{x:.1f} s<br>u₂: %{y:.1f} kPa'
            )
        )

        # Static pore pressure reference u0
        fig.add_hline(
            y=u0_static, line=dict(color='#2ca02c', width=1.8, dash='dash'),
            annotation_text=f"Static Hydrostatic u₀ = {u0_static:.1f} kPa",
            annotation_position="bottom right"
        )

        # 50% target dissipation horizontal line
        u50_val = diss_results["Target_u50_kPa"]
        fig.add_hline(
            y=u50_val, line=dict(color='#37474f', width=1.5, dash='dot'),
            annotation_text=f"50% Consolidation u₅₀ = {u50_val:.1f} kPa",
            annotation_position="top right"
        )

        # Vertical line at t50
        fig.add_vline(
            x=diss_results["t50_s"], line=dict(color='#9467bd', width=1.8, dash='dashdot'),
            annotation_text=f"t₅₀ = {diss_results['t50_s']:.1f} s",
            annotation_position="top left"
        )

        fig.update_xaxes(type="log", title_text="Dissipation Time (s) [Log Scale]", showgrid=True)
        fig.update_yaxes(title_text="Piezocone Pore Pressure u₂ (kPa)", showgrid=True)

        fig.update_layout(
            title=f"<b>CPTu Dissipation Response: {diss_results['Behavior']}</b>",
            height=540,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            margin=dict(l=60, r=40, t=60, b=80)
        )
        return fig

    @staticmethod
    def generate_static_publication_figure(
        df_pump: pd.DataFrame,
        pump_results: Dict,
        df_diss: pd.DataFrame,
        diss_results: Dict,
        u0_static: float
    ) -> io.BytesIO:
        """Produces a 300 DPI publication-grade dual-panel figure."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # Panel 1: Pumping Test Semi-Log Cooper-Jacob
        t_min = df_pump["Time_min"].values
        s_obs = df_pump["Drawdown_m"].values
        axes[0].scatter(t_min, s_obs, color='#1f77b4', s=35, label='Measured Drawdown', zorder=3)

        t_fit = np.logspace(np.log10(max(pump_results["t0_min"] * 0.8, 0.5)), np.log10(np.max(t_min)), 100)
        s_fit = pump_results["Slope"] * np.log10(t_fit) + pump_results["Intercept"]
        axes[0].plot(t_fit, s_fit, '--', color='#d62728', lw=1.8,
                     label=f'Cooper-Jacob Fit ($T={pump_results["Transmissivity_m2_day"]:.1f}$ m$^2$/day)')

        axes[0].scatter([pump_results["t0_min"]], [0.0], color='#2ca02c', marker='*', s=120, zorder=4,
                        label=f'$t_0={pump_results["t0_min"]:.2f}$ min')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Elapsed Pumping Time $t$ (min)', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Drawdown $s$ (m)', fontsize=10, fontweight='bold')
        axes[0].set_title('(a) Cooper-Jacob Pumping Test Analysis', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].legend(loc='lower left', frameon=True, fontsize=8)
        axes[0].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 2: CPTu Dissipation Profile
        t_s = df_diss["Time_s"].values
        u2 = df_diss["Pore_Pressure_u2_kPa"].values
        axes[1].plot(t_s, u2, 'o-', color='#ff7f0e', lw=1.8, ms=4, label='Measured $u_2(t)$')
        axes[1].axhline(u0_static, color='#2ca02c', ls='--', lw=1.4, label=f'Static $u_0={u0_static:.1f}$ kPa')
        axes[1].axhline(diss_results["Target_u50_kPa"], color='#37474f', ls=':', lw=1.4,
                        label=f'$u_{{50}}={diss_results["Target_u50_kPa"]:.1f}$ kPa')
        axes[1].axvline(diss_results["t50_s"], color='#9467bd', ls='-.', lw=1.4,
                        label=f'$t_{{50}}={diss_results["t50_s"]:.1f}$ s')

        axes[1].set_xscale('log')
        axes[1].set_xlabel('Dissipation Time $t$ (s)', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('Pore Pressure $u_2$ (kPa)', fontsize=10, fontweight='bold')
        axes[1].set_title(f'(b) CPTu Dissipation Response ({diss_results["Behavior"]})', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right', frameon=True, fontsize=8)
        axes[1].grid(True, which='both', ls='--', alpha=0.6)

        fig.suptitle('Hydrogeological Characterization Suite (CFEM Chapter 5 Standards)',
                     fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf


# -----------------------------------------------------------------------------
# 4. EXCEL EXPORT ENGINE (OPENPYXL)
# -----------------------------------------------------------------------------

class HydroExcelExporter:
    """Exports interpreted hydrogeological results into an executive-styled Excel workbook."""

    @staticmethod
    def export(
        df_pump: pd.DataFrame,
        pump_results: Dict,
        df_diss: pd.DataFrame,
        diss_results: Dict,
        meta_params: Dict
    ) -> io.BytesIO:
        wb = openpyxl.Workbook()

        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        sub_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
        font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        font_sub = Font(name="Calibri", size=11, bold=True, color="1F497D")
        font_data = Font(name="Calibri", size=9)
        thin_border = Border(
            left=Side(style='thin', color='B0B0B0'),
            right=Side(style='thin', color='B0B0B0'),
            top=Side(style='thin', color='B0B0B0'),
            bottom=Side(style='thin', color='B0B0B0')
        )

        # Sheet 1: Executive Interpretation Summary
        ws_sum = wb.active
        ws_sum.title = "Executive_Summary"
        ws_sum.views.sheetView[0].showGridLines = True

        ws_sum["A1"] = "SITE HYDROGEOLOGICAL CHARACTERIZATION REPORT"
        ws_sum["A1"].font = Font(name="Calibri", size=14, bold=True, color="1F497D")
        ws_sum["A2"] = "Compliant with Canadian Foundation Engineering Manual (CFEM Ch 5 & 22)"
        ws_sum["A2"].font = Font(name="Calibri", size=10, italic=True)

        # Block 1: Pumping Test Parameters
        ws_sum["A4"] = "1. PUMPING TEST (COOPER-JACOB) INTERPRETATION"
        ws_sum["A4"].font = font_sub
        ws_sum["A4"].fill = sub_fill

        pump_items = [
            ("Pumping Flow Rate Q (m3/day)", meta_params["Q_m3_day"]),
            ("Observation Well Radial Distance r (m)", meta_params["radius_m"]),
            ("Transmissivity T (m2/day)", pump_results["Transmissivity_m2_day"]),
            ("Transmissivity T (m2/s)", f"{pump_results['Transmissivity_m2_s']:.2e}"),
            ("Dimensionless Storativity S", f"{pump_results['Storativity']:.2e}"),
            ("Zero-Drawdown Intercept t0 (min)", pump_results["t0_min"]),
            ("Slope Delta s per Log Cycle (m)", pump_results["Delta_s_per_cycle_m"]),
            ("Diagnosed Boundary Condition", pump_results["Boundary_Type"]),
        ]
        r = 5
        for k, v in pump_items:
            ws_sum[f"A{r}"] = k
            ws_sum[f"B{r}"] = v
            ws_sum[f"A{r}"].font = font_data
            ws_sum[f"B{r}"].font = font_data
            ws_sum[f"A{r}"].border = thin_border
            ws_sum[f"B{r}"].border = thin_border
            r += 1

        r += 1
        # Block 2: CPTu Dissipation Parameters
        ws_sum[f"A{r}"] = "2. CPTu DISSIPATION & PERMEABILITY INTERPRETATION"
        ws_sum[f"A{r}"].font = font_sub
        ws_sum[f"A{r}"].fill = sub_fill
        r += 1

        diss_items = [
            ("Dissipation Curve Behavior", diss_results["Behavior"]),
            ("Measured Time to 50% Consolidation t50 (s)", diss_results["t50_s"]),
            ("Calculated Target Pore Pressure u50 (kPa)", diss_results["Target_u50_kPa"]),
            ("Assumed Static Hydrostatic u0 (kPa)", meta_params["u0_static_kPa"]),
            ("Horizontal Consolidation Coefficient ch (cm2/s)", diss_results["ch_cm2_s"]),
            ("Horizontal Consolidation Coefficient ch (m2/year)", diss_results["ch_m2_yr"]),
            ("Permeability from t50 kh (m/s) [CFEM Eq. 5.47]", f"{diss_results['k_h_from_t50_m_s']:.2e}"),
            ("Permeability from Kozeny-Carman (m/s) [CFEM Eq. 5.75]",
             f"{diss_results['k_Kozeny_Carman_m_s']:.2e}" if not np.isnan(diss_results['k_Kozeny_Carman_m_s']) else "N/A"),
        ]
        for k, v in diss_items:
            ws_sum[f"A{r}"] = k
            ws_sum[f"B{r}"] = v
            ws_sum[f"A{r}"].font = font_data
            ws_sum[f"B{r}"].font = font_data
            ws_sum[f"A{r}"].border = thin_border
            ws_sum[f"B{r}"].border = thin_border
            r += 1

        # Sheet 2: Raw Pumping Test Data
        ws_pdata = wb.create_sheet(title="Pumping_Test_Data")
        ws_pdata.views.sheetView[0].showGridLines = True
        p_cols = ["Time (min)", "Time (day)", "Measured Drawdown (m)"]
        ws_pdata.append(p_cols)
        for c_idx in range(1, 4):
            c = ws_pdata.cell(row=1, column=c_idx)
            c.fill = header_fill
            c.font = font_header
            c.alignment = Alignment(horizontal="center")

        for _, r_val in df_pump.iterrows():
            ws_pdata.append(list(r_val))

        # Sheet 3: Raw Dissipation Data
        ws_ddata = wb.create_sheet(title="Dissipation_Data")
        ws_ddata.views.sheetView[0].showGridLines = True
        d_cols = ["Time (s)", "Porewater Pressure u2 (kPa)"]
        ws_ddata.append(d_cols)
        for c_idx in range(1, 3):
            c = ws_ddata.cell(row=1, column=c_idx)
            c.fill = header_fill
            c.font = font_header
            c.alignment = Alignment(horizontal="center")

        for _, r_val in df_diss.iterrows():
            ws_ddata.append(list(r_val))

        # Auto-fit columns
        for sheet in wb.worksheets:
            for col in sheet.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                col_letter = get_column_letter(col[0].column)
                sheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf


# -----------------------------------------------------------------------------
# 5. STREAMLIT WEB APPLICATION
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="CFEM Aquifer & CPTu Dissipation Suite",
        page_icon="💧",
        layout="wide"
    )

    st.title("💧 Hydrogeological Aquifer Testing & CPTu Dissipation Suite")
    st.markdown(
        """
        **Advanced In-Situ Aquifer Parameter Estimation & Boundary Diagnostics**  
        *Compliant with the Canadian Foundation Engineering Manual (CFEM Ch 5 & 22), Theis (1935), Cooper-Jacob (1946), and Houlsby & Teh (1988).*
        """
    )
    st.write("---")

    # Sidebar: Engineering Controls
    st.sidebar.header("🚜 1. Pumping Test Parameters")
    q_rate_m3_day = st.sidebar.number_input(
        "Pumping Rate Q (m³/day)", min_value=10.0, max_value=20000.0, value=1800.0, step=100.0
    )
    radius_m = st.sidebar.number_input(
        "Observation Radial Distance r (m)", min_value=1.0, max_value=1000.0, value=25.0, step=5.0
    )

    st.sidebar.header("⚡ 2. CPTu Dissipation Parameters")
    cone_type = st.sidebar.selectbox("Cone Penetrometer Area", ["10 cm² (ac = 1.78 cm)", "15 cm² (ac = 2.22 cm)"])
    cone_area = 10.0 if "10" in cone_type else 15.0

    rigidity_ir = st.sidebar.slider(
        "Rigidity Index IR = G / su", min_value=10.0, max_value=500.0, value=100.0, step=10.0,
        help="Default value IR = 100 per CFEM Section 5.4.4.6."
    )
    u0_static = st.sidebar.number_input(
        "Assumed Hydrostatic Equilibrium u₀ (kPa)", min_value=0.0, max_value=500.0, value=80.0, step=5.0
    )
    liquid_limit = st.sidebar.number_input(
        "Liquid Limit LL (%) [Optional for Kozeny-Carman]", min_value=0.0, max_value=150.0, value=45.0, step=5.0
    )

    # File Uploaders
    st.sidebar.header("📂 3. Data File Uploads")
    pump_file = st.sidebar.file_uploader("Upload Pumping Test (CSV or Excel)", type=["csv", "xlsx"])
    diss_file = st.sidebar.file_uploader("Upload CPTu Dissipation (CSV or Excel)", type=["csv", "xlsx"])

    # Load or synthesize Pumping Data
    if pump_file is not None:
        try:
            df_pump = pd.read_csv(pump_file) if pump_file.name.endswith(".csv") else pd.read_excel(pump_file)
            st.sidebar.success("Custom pumping test dataset loaded!")
        except Exception as e:
            st.sidebar.error(f"Error parsing pumping file: {e}. Fallback to synthetic benchmark.")
            df_pump = generate_benchmark_pumping_dataset()
    else:
        df_pump = generate_benchmark_pumping_dataset()

    # Load or synthesize Dissipation Data
    if diss_file is not None:
        try:
            df_diss = pd.read_csv(diss_file) if diss_file.name.endswith(".csv") else pd.read_excel(diss_file)
            st.sidebar.success("Custom CPTu dissipation dataset loaded!")
        except Exception as e:
            st.sidebar.error(f"Error parsing dissipation file: {e}. Fallback to synthetic benchmark.")
            df_diss = generate_benchmark_dissipation_dataset("Monotonic")
    else:
        diss_mode = st.sidebar.radio("Synthetic Dissipation Response Type", ["Monotonic", "Dilatory"])
        df_diss = generate_benchmark_dissipation_dataset(diss_mode)

    # Execute Analytical Pipelines
    pump_results = PumpingTestEngine.fit_cooper_jacob(
        time_min=df_pump["Time_min"].values,
        drawdown_m=df_pump["Drawdown_m"].values,
        q_m3_day=q_rate_m3_day,
        radius_m=radius_m
    )

    diss_results = CPTuDissipationEngine.evaluate_dissipation_properties(
        time_s=df_diss["Time_s"].values,
        u2_kpa=df_diss["Pore_Pressure_u2_kPa"].values,
        u0_static_kpa=u0_static,
        cone_area_cm2=cone_area,
        rigidity_index=rigidity_ir,
        liquid_limit=liquid_limit
    )

    # Top-Level Summary KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Transmissivity (T)", f"{pump_results['Transmissivity_m2_day']:.1f} m²/day", delta=f"{pump_results['Transmissivity_m2_s']:.2e} m²/s")
    with kpi2:
        st.metric("Storativity (S)", f"{pump_results['Storativity']:.2e}", delta="Confined Aquifer" if pump_results['Storativity'] < 0.005 else "Unconfined")
    with kpi3:
        st.metric("50% Dissipation (t₅₀)", f"{diss_results['t50_s']:.1f} s", delta=f"Target u₅₀ = {diss_results['Target_u50_kPa']:.0f} kPa")
    with kpi4:
        st.metric("Horiz. Permeability (kh)", f"{diss_results['k_h_from_t50_m_s']:.2e} m/s", delta=f"ch = {diss_results['ch_m2_yr']:.1f} m²/yr")

    st.write("")

    # Application Tabs
    tab_pump, tab_diss, tab_export = st.tabs([
        "📊 Transient Pumping Analysis (Theis / Cooper-Jacob)",
        "⚡ CPTu Dissipation & Consolidation (Houlsby & Teh)",
        "📥 Professional Geotechnical Deliverables"
    ])

    with tab_pump:
        st.subheader("Cooper-Jacob Semi-Logarithmic Regression & Boundary Diagnostics")
        col_p1, col_p2 = st.columns([3, 1])

        with col_p1:
            fig_pump = HydroVisualizer.create_pumping_test_dashboard(df_pump, pump_results, radius_m)
            st.plotly_chart(fig_pump, use_container_width=True)

        with col_p2:
            st.markdown("#### 🧭 Diagnostic Summary")
            st.info(f"**Boundary Condition:**\n{pump_results['Boundary_Type']}")
            st.success(
                f"- **Δs per Log Cycle:** {pump_results['Delta_s_per_cycle_m']:.3f} m\n"
                f"- **Zero-Drawdown t₀:** {pump_results['t0_min']:.3f} min\n"
                f"- **Transmissivity T:** {pump_results['Transmissivity_m2_day']:.1f} m²/day"
            )
            st.markdown(
                """
                **Interpretation Rules (CFEM Ch 5.8.6):**
                - A downward curvature in late-time drawdown indicates a **recharge boundary** (e.g., river connection).
                - An upward curvature indicates an **impermeable barrier** or aquifer pinching.
                """
            )

    with tab_diss:
        st.subheader("Houlsby & Teh (1988) Strain Path Dissipation Modeling")
        col_d1, col_d2 = st.columns([3, 1])

        with col_d1:
            fig_diss = HydroVisualizer.create_dissipation_dashboard(df_diss, diss_results, u0_static)
            st.plotly_chart(fig_diss, use_container_width=True)

        with col_d2:
            st.markdown("#### 🔬 Soil Parameters")
            st.info(f"**Response:** {diss_results['Behavior']}")
            st.success(
                f"- **t₅₀:** {diss_results['t50_s']} s\n"
                f"- **c_h:** {diss_results['ch_cm2_s']} cm²/s ({diss_results['ch_m2_yr']} m²/yr)\n"
                f"- **k_h (t₅₀):** {diss_results['k_h_from_t50_m_s']:.2e} m/s\n"
                f"- **Kozeny-Carman k:** {diss_results['k_Kozeny_Carman_m_s']:.2e} m/s" if not np.isnan(diss_results['k_Kozeny_Carman_m_s']) else "- **Kozeny-Carman:** N/A"
            )
            st.markdown(
                """
                **CFEM Eq. 5.45 & 5.47 Formulations:**
                - Modified time factor $T^*_{50} = 0.245$ for $u_2$ shoulder position.
                - Permeability reflects predominantly the horizontal direction ($k_h$) due to macro-stratification.
                """
            )

    with tab_export:
        st.subheader("Executive Hydrogeological Deliverables")
        st.markdown("Download high-resolution vector figures and fully formatted Excel interpretation workbooks.")

        meta_dict = {
            "Q_m3_day": q_rate_m3_day,
            "radius_m": radius_m,
            "u0_static_kPa": u0_static
        }

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("##### 📄 Multi-Panel Report Graphic (300 DPI)")
            fig_buf = HydroVisualizer.generate_static_publication_figure(
                df_pump, pump_results, df_diss, diss_results, u0_static
            )
            st.image(fig_buf, caption="Print Preview: Hydrogeological Characterization", use_container_width=True)
            st.download_button(
                label="⬇️ Download Publication Figure (PNG - 300 DPI)",
                data=fig_buf,
                file_name="CFEM_Hydrogeology_Aquifer_Suite.png",
                mime="image/png"
            )

        with col_ex2:
            st.markdown("##### 📊 Full Hydrogeological Calculation Schedule (.xlsx)")
            st.markdown(
                """
                Includes:
                - **Executive Summary:** Pumping test fit parameters, storativity, transmissivity, boundary diagnosis, and consolidation properties.
                - **Pumping_Test_Data:** Continuous drawdown vs. time observations.
                - **Dissipation_Data:** In-situ piezocone decay records.
                """
            )
            excel_buf = HydroExcelExporter.export(
                df_pump, pump_results, df_diss, diss_results, meta_dict
            )
            st.download_button(
                label="⬇️ Download Excel Workbook (.xlsx)",
                data=excel_buf,
                file_name="CFEM_Hydrogeological_Analysis_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
