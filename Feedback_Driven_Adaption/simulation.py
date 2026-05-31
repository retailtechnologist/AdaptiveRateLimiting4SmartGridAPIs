#!/usr/bin/env python3
"""
simulation.py
=============
Smart Grid API rate-limiting simulation harness.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh
Section: IV.A -- Feedback Driven Adaptation

Simulation Phases (1 000 steps each, 5 000 total)
--------------------------------------------------
    Phase 1   0-999    Normal operation
    Phase 2   1000-1999  Peak load  (EV charging / renewable surge)
    Phase 3   2000-2999  DDoS attack
    Phase 4   3000-3999  Grid emergency  (frequency deviation)
    Phase 5   4000-4999  Recovery / stabilisation

Algorithm
---------
    Imported from pid_rate_limiter.PIDRateLimiter

Outputs
-------
    smart_grid_rate_limiter_simulation.csv  -- 5 000-row test dataset
    adaptive_rate_limiter_results.png       -- 7-panel diagnostic chart
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # write PNG without a display server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime, timedelta

from Feedback_Driven_Adaption import PIDRateLimiter


# ---------------------------------------------------------------------------
# Constants shared across simulation and plot helpers
# ---------------------------------------------------------------------------

SCENARIO_ORDER  = ["Normal", "Peak Load", "DDoS Attack", "Grid Emergency", "Recovery"]
SCENARIO_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#3498db"]
PHASE_BOUNDS    = [0, 1000, 2000, 3000, 4000, 5000]

# PID controller defaults (mirrors PIDRateLimiter.__init__ defaults)
R_BASE       = 1000.0
KP           = 1500.0
KI           = 10.0
KD           = 300.0
TARGET_LOAD  = 0.60


# ---------------------------------------------------------------------------
# 1. Traffic Generator
# ---------------------------------------------------------------------------

def generate_smart_grid_traffic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate n rows of synthetic smart-grid API traffic covering 5 phases.

    Each metric is drawn from a Gaussian distribution whose mean and
    standard deviation reflect the operational characteristics of that phase.
    Values are clipped to physically plausible ranges before return.

    Parameters
    ----------
    n    : total number of time-steps (rows)
    seed : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        scenario, cpu_pct, memory_pct, latency_ms, queue_depth, incoming_req
    """
    rng = np.random.default_rng(seed)

    cpu      = np.empty(n)
    memory   = np.empty(n)
    latency  = np.empty(n)
    queue    = np.empty(n, dtype=float)
    incoming = np.empty(n, dtype=float)
    scenario = np.empty(n, dtype=object)

    for i in range(n):

        # -- Phase 1: Normal operation --
        if i < 1000:
            scenario[i] = "Normal"
            cpu[i]      = rng.normal(35,    5)
            memory[i]   = rng.normal(40,    5)
            latency[i]  = rng.normal(80,   15)
            queue[i]    = rng.normal(50,   20)
            incoming[i] = rng.normal(800, 100)

        # -- Phase 2: Peak load (linear ramp from Normal to peak) --
        elif i < 2000:
            ph = (i - 1000) / 999       # 0.0 -> 1.0
            scenario[i] = "Peak Load"
            cpu[i]      = rng.normal(38 + 42 * ph,    8)
            memory[i]   = rng.normal(43 + 32 * ph,    7)
            latency[i]  = rng.normal(85 + 265 * ph,  30)
            queue[i]    = rng.normal(55 + 620 * ph,  60)
            incoming[i] = rng.normal(850 + 1700 * ph, 200)

        # -- Phase 3: DDoS attack (sustained high load) --
        elif i < 3000:
            scenario[i] = "DDoS Attack"
            cpu[i]      = rng.normal(88,    4)
            memory[i]   = rng.normal(82,    5)
            latency[i]  = rng.normal(425,  50)
            queue[i]    = rng.normal(930,  60)
            incoming[i] = rng.normal(4600, 350)

        # -- Phase 4: Grid emergency (frequency deviation) --
        elif i < 4000:
            scenario[i] = "Grid Emergency"
            cpu[i]      = rng.normal(72,   10)
            memory[i]   = rng.normal(68,    8)
            latency[i]  = rng.normal(310,  45)
            queue[i]    = rng.normal(720, 100)
            incoming[i] = rng.normal(2100, 250)

        # -- Phase 5: Recovery (linear ramp back toward Normal) --
        else:
            ph = (i - 4000) / 999       # 0.0 -> 1.0
            scenario[i] = "Recovery"
            cpu[i]      = rng.normal(88 - 53 * ph,    6)
            memory[i]   = rng.normal(82 - 42 * ph,    6)
            latency[i]  = rng.normal(425 - 345 * ph, 30)
            queue[i]    = rng.normal(930 - 880 * ph,  50)
            incoming[i] = rng.normal(4600 - 3800 * ph, 200)

    # Clip to physically plausible ranges
    cpu      = np.clip(cpu,       5.0,  99.0)
    memory   = np.clip(memory,   10.0,  98.0)
    latency  = np.clip(latency,  10.0, 900.0)
    queue    = np.clip(queue,     0.0, 2000.0).astype(int)
    incoming = np.clip(incoming, 50.0, 6500.0).astype(int)

    return pd.DataFrame({
        "scenario"   : scenario,
        "cpu_pct"    : np.round(cpu,     2),
        "memory_pct" : np.round(memory,  2),
        "latency_ms" : np.round(latency, 2),
        "queue_depth": queue,
        "incoming_req": incoming,
    })


# ---------------------------------------------------------------------------
# 2. Simulation Runner
# ---------------------------------------------------------------------------

def run_simulation(n: int = 5000) -> pd.DataFrame:
    """
    Feed the generated traffic through the PIDRateLimiter and record results.

    For each time-step the function computes:
        accepted_req  = min(incoming_req, rate_limit)
        rejected_req  = max(0, incoming_req - rate_limit)
        rejection_rate = rejected_req / incoming_req

    Parameters
    ----------
    n : number of simulation steps

    Returns
    -------
    pd.DataFrame with 18 columns covering inputs, controller outputs,
    and derived metrics (one row per time-step).
    """
    traffic = generate_smart_grid_traffic(n)
    pid     = PIDRateLimiter(
        r_base=R_BASE, Kp=KP, Ki=KI, Kd=KD, target_load=TARGET_LOAD
    )
    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    rows    = []

    for i, row in traffic.iterrows():
        ts     = base_ts + timedelta(seconds=int(i))
        result = pid.step(
            row.cpu_pct, row.memory_pct, row.latency_ms, row.queue_depth
        )

        rl       = result["rate_limit"]
        accepted = int(min(row.incoming_req, rl))
        rejected = int(max(0, row.incoming_req - rl))
        rej_rate = rejected / row.incoming_req if row.incoming_req > 0 else 0.0

        rows.append({
            "timestamp"     : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "step"          : i,
            "scenario"      : row.scenario,
            # -- Infrastructure metrics --
            "cpu_pct"       : row.cpu_pct,
            "memory_pct"    : row.memory_pct,
            "latency_ms"    : row.latency_ms,
            "queue_depth"   : row.queue_depth,
            # -- Traffic --
            "incoming_req"  : row.incoming_req,
            # -- PID outputs --
            "rate_limit"    : result["rate_limit"],
            "observed_load" : result["observed_load"],
            "error"         : result["error"],
            "p_term"        : result["p_term"],
            "i_term"        : result["i_term"],
            "d_term"        : result["d_term"],
            "integral"      : result["integral"],
            # -- Derived metrics --
            "accepted_req"  : accepted,
            "rejected_req"  : rejected,
            "rejection_rate": round(rej_rate, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Summary Table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by scenario and print a formatted summary table.

    Parameters
    ----------
    df : output of run_simulation()

    Returns
    -------
    per-scenario summary as a pd.DataFrame
    """
    summary = (
        df.groupby("scenario")
          .agg(
              Steps            = ("step",           "count"),
              Avg_CPU_pct      = ("cpu_pct",         "mean"),
              Avg_Memory_pct   = ("memory_pct",      "mean"),
              Avg_Latency_ms   = ("latency_ms",      "mean"),
              Avg_Queue        = ("queue_depth",      "mean"),
              Avg_Incoming_rpm = ("incoming_req",     "mean"),
              Avg_RateLimit    = ("rate_limit",       "mean"),
              Min_RateLimit    = ("rate_limit",       "min"),
              Max_RateLimit    = ("rate_limit",       "max"),
              Avg_ObsLoad      = ("observed_load",    "mean"),
              Avg_Error        = ("error",            "mean"),
              Avg_Rej_pct      = ("rejection_rate",   "mean"),
              Total_Accepted   = ("accepted_req",     "sum"),
              Total_Rejected   = ("rejected_req",     "sum"),
          )
          .round(2)
          .reindex(SCENARIO_ORDER)
    )
    summary["Avg_Rej_pct"] = (summary["Avg_Rej_pct"] * 100).round(2)

    LINE = "=" * 130
    print(f"\n{LINE}")
    print("  PID FEEDBACK-DRIVEN ADAPTIVE RATE LIMITER -- SIMULATION RESULTS")
    print("  Paper Section IV.A  |  5000-step Smart Grid Simulation")
    print(f"  Controller: r_base={R_BASE:.0f} | Kp={KP:.0f} | Ki={KI:.0f} "
          f"| Kd={KD:.0f} | target_load={TARGET_LOAD}")
    print(LINE)
    print(summary.to_string())
    print(LINE)

    total_in  = int(df["incoming_req"].sum())
    total_acc = int(df["accepted_req"].sum())
    total_rej = int(df["rejected_req"].sum())
    print(
        f"\n  OVERALL  Incoming: {total_in:>12,}  "
        f"Accepted: {total_acc:>12,}  "
        f"Rejected: {total_rej:>12,}  "
        f"Rejection Rate: {total_rej / total_in * 100:.2f}%"
    )
    print(
        f"           Rate-limit range: [{df.rate_limit.min():.0f}, "
        f"{df.rate_limit.max():.0f}] req/min  "
        f"  PID target load: {TARGET_LOAD}  "
        f"  Mean observed load: {df.observed_load.mean():.3f}"
    )
    print(LINE)
    return summary


# ---------------------------------------------------------------------------
# 4. Plot Helpers (private)
# ---------------------------------------------------------------------------

def _shade_phases(ax, alpha: float = 0.07) -> None:
    """Colour the background of each operating phase."""
    for j, color in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j + 1], alpha=alpha, color=color)
    ax.set_xlim(0, 5000)


def _phase_labels(ax, y_frac: float = 0.97) -> None:
    """Overlay phase name at the top of each shaded region."""
    ylo, yhi = ax.get_ylim()
    y_pos = ylo + (yhi - ylo) * y_frac
    for j, (name, color) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j + 1]) / 2
        ax.text(mid, y_pos, name, ha="center", va="top",
                fontsize=7, color=color, fontweight="bold")


# ---------------------------------------------------------------------------
# 5. Results Plot
# ---------------------------------------------------------------------------

def plot_results(
    df: pd.DataFrame,
    out: str = "adaptive_rate_limiter_results.png",
) -> None:
    """
    Produce a 7-panel diagnostic figure and save it as a PNG.

    Panels
    ------
    A  Rate limit vs incoming request volume
    B  CPU and memory utilisation
    C  Request latency and queue depth (dual y-axis)
    D  PID error signal with fill
    E  PID component breakdown (P, I, D terms)
    F  Rejection rate distribution by scenario (box-plot)
    G  Observed system load vs PID target

    Parameters
    ----------
    df  : output of run_simulation()
    out : output file path for the PNG
    """
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "Smart Grid -- PID Feedback-Driven Adaptive Rate Limiter\n"
        "r(t) = r_base + Kp*e(t) + Ki*integral(e)dt + Kd*de(t)/dt"
        "     [5000-step simulation, 5 phases]",
        fontsize=12, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        hspace=0.48, wspace=0.30,
        top=0.96, bottom=0.06, left=0.07, right=0.97,
    )

    s = df["step"]

    # -- Panel A: Rate limit vs incoming requests --------------------------
    ax = fig.add_subplot(gs[0, :])
    _shade_phases(ax)
    ax.fill_between(s, df["incoming_req"], alpha=0.22,
                    color="#7f8c8d", label="Incoming Requests")
    ax.plot(s, df["rate_limit"], color="#e74c3c", lw=1.4, zorder=3,
            label="Adaptive Rate Limit r(t)")
    ax.axhline(R_BASE, color="navy", ls="--", lw=1.2, alpha=0.70,
               label=f"Baseline r_base = {R_BASE:.0f} req/min")
    ax.set_ylabel("Requests / min", fontsize=10)
    ax.set_title("(A)  Adaptive Rate Limit vs Incoming Request Volume",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.30, lw=0.6)
    _phase_labels(ax)

    # -- Panel B: CPU & Memory --------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _shade_phases(ax)
    ax.plot(s, df["cpu_pct"],    color="#e74c3c", lw=0.8, alpha=0.85, label="CPU %")
    ax.plot(s, df["memory_pct"], color="#3498db", lw=0.8, alpha=0.85, label="Memory %")
    ax.axhline(60, color="black", ls=":", lw=1.0, alpha=0.5, label="60% target")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Utilisation (%)", fontsize=9)
    ax.set_title("(B)  CPU & Memory Utilisation",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)

    # -- Panel C: Latency & Queue Depth (dual y-axis) ---------------------
    ax = fig.add_subplot(gs[1, 1])
    _shade_phases(ax)
    ax2 = ax.twinx()
    ax.plot( s, df["latency_ms"],  color="#e67e22", lw=0.8, alpha=0.85, label="Latency (ms)")
    ax2.plot(s, df["queue_depth"], color="#8e44ad", lw=0.8, alpha=0.85, label="Queue Depth")
    ax.set_ylabel("Latency (ms)",  color="#e67e22", fontsize=9)
    ax2.set_ylabel("Queue Depth",  color="#8e44ad", fontsize=9)
    ax.set_title("(C)  Request Latency & Queue Depth",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend( loc="upper left",  fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)
    ax.set_xlim(0, 5000)

    # -- Panel D: PID Error Signal ----------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    _shade_phases(ax)
    ax.plot(s, df["error"], color="#2c3e50", lw=0.9, label="Error e(t)")
    ax.axhline(0, color="red", ls="--", lw=1.0, alpha=0.70)
    ax.fill_between(s, df["error"], 0,
                    where=(df["error"] >= 0), alpha=0.18, color="#27ae60",
                    label="Under-loaded: relax limit")
    ax.fill_between(s, df["error"], 0,
                    where=(df["error"] <  0), alpha=0.18, color="#e74c3c",
                    label="Over-loaded: tighten limit")
    ax.set_ylabel("e(t) = target_load - observed_load", fontsize=9)
    ax.set_title("(D)  PID Error Signal", fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)

    # -- Panel E: PID Component Breakdown ---------------------------------
    ax = fig.add_subplot(gs[2, 1])
    _shade_phases(ax)
    ax.plot(s, df["p_term"], color="#e74c3c", lw=0.8, alpha=0.85, label="P = Kp*e(t)")
    ax.plot(s, df["i_term"], color="#27ae60", lw=0.8, alpha=0.85, label="I = Ki*integral(e)")
    ax.plot(s, df["d_term"], color="#3498db", lw=0.8, alpha=0.85, label="D = Kd*de/dt")
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_ylabel("Rate Adjustment (req/min)", fontsize=9)
    ax.set_title("(E)  PID Controller Components",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)

    # -- Panel F: Rejection Rate Box-plot ---------------------------------
    ax = fig.add_subplot(gs[3, 0])
    box_data = [
        df[df["scenario"] == sc]["rejection_rate"].values * 100
        for sc in SCENARIO_ORDER
    ]
    bp = ax.boxplot(
        box_data, labels=SCENARIO_ORDER, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], SCENARIO_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for i, sc in enumerate(SCENARIO_ORDER):
        med = np.median(df[df["scenario"] == sc]["rejection_rate"].values * 100)
        ax.text(i + 1, med + 1.0, f"{med:.1f}%",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_ylabel("Rejection Rate (%)", fontsize=9)
    ax.set_title("(F)  Request Rejection Rate by Scenario",
                 fontweight="bold", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.grid(True, alpha=0.30, lw=0.6, axis="y")

    # -- Panel G: Observed Load vs Target ---------------------------------
    ax = fig.add_subplot(gs[3, 1])
    _shade_phases(ax)
    ax.plot(s, df["observed_load"], color="#e74c3c", lw=0.9, label="Observed Load")
    ax.axhline(TARGET_LOAD, color="navy", ls="--", lw=1.4,
               label=f"Target Load = {TARGET_LOAD}")
    ax.fill_between(s, df["observed_load"], TARGET_LOAD,
                    where=(df["observed_load"] >  TARGET_LOAD),
                    alpha=0.18, color="#e74c3c", label="Over target")
    ax.fill_between(s, df["observed_load"], TARGET_LOAD,
                    where=(df["observed_load"] <= TARGET_LOAD),
                    alpha=0.12, color="#27ae60", label="Under target")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Composite Load Index [0, 1]", fontsize=9)
    ax.set_title("(G)  System Load vs PID Target",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)

    # -- Bottom legend: operating phases ----------------------------------
    legend_handles = [
        Patch(facecolor=color, alpha=0.65, label=name)
        for name, color in zip(SCENARIO_ORDER, SCENARIO_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=5, fontsize=9,
        title="Operating Phases", title_fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plots saved  ->  {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N        = 5000
    CSV_FILE = "smart_grid_rate_limiter_simulation.csv"
    PNG_FILE = "adaptive_rate_limiter_results.png"

    print("=" * 65)
    print("  Smart Grid Adaptive Rate Limiter -- PID Simulation")
    print("  Section IV.A: Feedback Driven Adaptation")
    print("=" * 65)
    print(f"\n  Generating {N:,} steps of synthetic smart-grid traffic ...")

    df = run_simulation(N)

    df.to_csv(CSV_FILE, index=False)
    print(f"  CSV saved    ->  {CSV_FILE}  ({N:,} rows, {len(df.columns)} columns)")

    print_summary(df)

    print("\n  Generating plots ...")
    plot_results(df, out=PNG_FILE)

    print("\n  Done.")
