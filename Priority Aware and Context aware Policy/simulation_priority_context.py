#!/usr/bin/env python3
"""
simulation_priority_context.py
================================
Simulation harness for Section IV.C: Priority Aware and Context Aware Policy.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh

Structure
---------
5 clients x 1300 steps = 6 500 records in CSV

Clients & Roles
---------------
Client                  Role              Typical operations
--------------------    --------------    -------------------------------------------
GridOps_Station         GRID_OPERATOR     Fault isolation, load shedding, DER control
Utility_API_01          UTILITY           Demand response, meter data, market ops
DER_Provider_01         DER_PROVIDER      Frequency regulation, DER control, PMU sync
AMI_Network_01          SMART_METER       Meter data, telemetry (floods during attack)
ThirdParty_Market_01    THIRD_PARTY       Market operations, telemetry (abusive in crisis)

Simulation Phases (260 steps each)
------------------------------------
Phase 1  (0-259)     NORMAL     Baseline, routine operations
Phase 2  (260-519)   ELEVATED   Renewable variability, increased demand
Phase 3  (520-779)   EMERGENCY  DDoS flooding + frequency deviation event
Phase 4  (780-1039)  CRITICAL   Blackout risk; critical commands take priority
Phase 5  (1040-1299) Recovery   Grid stabilises from ALERT to NORMAL

Outputs
-------
    priority_context_simulation.csv           6 500-row test dataset (23 columns)
    priority_context_results.png              8-panel diagnostic chart
    priority_context_scatter.png              Priority x Rate-limit scatter
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime, timedelta

from priority_context_rate_limiter import PriorityContextRateLimiter


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STEPS_PER_SIM = 1300
CLIENT_IDS = [
    "GridOps_Station",
    "Utility_API_01",
    "DER_Provider_01",
    "AMI_Network_01",
    "ThirdParty_Market_01",
]

CLIENT_ROLES = {
    "GridOps_Station"    : "GRID_OPERATOR",
    "Utility_API_01"     : "UTILITY",
    "DER_Provider_01"    : "DER_PROVIDER",
    "AMI_Network_01"     : "SMART_METER",
    "ThirdParty_Market_01": "THIRD_PARTY",
}

CLIENT_COLORS = {
    "GridOps_Station"    : "#2ecc71",
    "Utility_API_01"     : "#3498db",
    "DER_Provider_01"    : "#9b59b6",
    "AMI_Network_01"     : "#e74c3c",
    "ThirdParty_Market_01": "#f39c12",
}

TIER_COLORS = {
    "CRITICAL": "#e74c3c",
    "HIGH"    : "#e67e22",
    "MEDIUM"  : "#3498db",
    "LOW"     : "#95a5a6",
}

SCENARIO_ORDER  = ["NORMAL", "ELEVATED", "EMERGENCY", "CRITICAL", "Recovery"]
SCENARIO_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c", "#8e44ad", "#3498db"]
PHASE_BOUNDS    = [0, 260, 520, 780, 1040, 1300]

# Limiter configuration
L_BASE = 100.0
BETA   = 3.0

# Grid conditions per phase
PHASE_GRID = {
    "NORMAL"   : {"grid_state": "NORMAL",    "freq_mean": 0.05, "freq_std": 0.03,
                  "ren_mean": 20,  "ren_std": 5,  "emg": False},
    "ELEVATED" : {"grid_state": "ELEVATED",  "freq_mean": 0.15, "freq_std": 0.05,
                  "ren_mean": 45,  "ren_std": 10, "emg": False},
    "EMERGENCY": {"grid_state": "EMERGENCY", "freq_mean": 0.35, "freq_std": 0.08,
                  "ren_mean": 65,  "ren_std": 12, "emg": True},
    "CRITICAL" : {"grid_state": "CRITICAL",  "freq_mean": 0.45, "freq_std": 0.05,
                  "ren_mean": 70,  "ren_std": 10, "emg": True},
    "Recovery" : {"grid_state": "ALERT",     "freq_mean": 0.20, "freq_std": 0.08,
                  "ren_mean": 40,  "ren_std": 10, "emg": False},
}

# Message type distributions per (client, phase)
MSG_DIST = {
    ("GridOps_Station", "NORMAL")   : (["PMU_SYNC", "DER_CONTROL"],                       [0.5, 0.5]),
    ("GridOps_Station", "ELEVATED") : (["FREQUENCY_REGULATION", "DER_CONTROL", "PMU_SYNC"],[0.4, 0.4, 0.2]),
    ("GridOps_Station", "EMERGENCY"): (["FAULT_ISOLATION", "LOAD_SHEDDING", "DEMAND_RESPONSE", "PROTECTION_RELAY"], [0.3, 0.3, 0.2, 0.2]),
    ("GridOps_Station", "CRITICAL") : (["FAULT_ISOLATION", "LOAD_SHEDDING", "PROTECTION_RELAY"], [0.4, 0.35, 0.25]),
    ("GridOps_Station", "Recovery") : (["DER_CONTROL", "DEMAND_RESPONSE", "PMU_SYNC"],    [0.4, 0.3, 0.3]),

    ("Utility_API_01", "NORMAL")    : (["METER_DATA", "MARKET_OPERATIONS"],               [0.6, 0.4]),
    ("Utility_API_01", "ELEVATED")  : (["METER_DATA", "DEMAND_RESPONSE", "MARKET_OPERATIONS"], [0.4, 0.4, 0.2]),
    ("Utility_API_01", "EMERGENCY") : (["DEMAND_RESPONSE", "DER_CONTROL", "METER_DATA"],  [0.5, 0.3, 0.2]),
    ("Utility_API_01", "CRITICAL")  : (["DEMAND_RESPONSE", "LOAD_SHEDDING", "DER_CONTROL"], [0.5, 0.3, 0.2]),
    ("Utility_API_01", "Recovery")  : (["METER_DATA", "DEMAND_RESPONSE", "MARKET_OPERATIONS"], [0.5, 0.3, 0.2]),

    ("DER_Provider_01", "NORMAL")   : (["DER_CONTROL", "TELEMETRY"],                      [0.6, 0.4]),
    ("DER_Provider_01", "ELEVATED") : (["FREQUENCY_REGULATION", "DER_CONTROL"],           [0.5, 0.5]),
    ("DER_Provider_01", "EMERGENCY"): (["FREQUENCY_REGULATION", "DER_CONTROL", "PMU_SYNC"], [0.5, 0.35, 0.15]),
    ("DER_Provider_01", "CRITICAL") : (["FREQUENCY_REGULATION", "DER_CONTROL"],           [0.6, 0.4]),
    ("DER_Provider_01", "Recovery") : (["DER_CONTROL", "FREQUENCY_REGULATION", "TELEMETRY"], [0.4, 0.3, 0.3]),

    ("AMI_Network_01", "NORMAL")    : (["METER_DATA", "TELEMETRY"],                       [0.7, 0.3]),
    ("AMI_Network_01", "ELEVATED")  : (["METER_DATA", "TELEMETRY"],                       [0.7, 0.3]),
    ("AMI_Network_01", "EMERGENCY") : (["METER_DATA", "TELEMETRY"],                       [0.8, 0.2]),   # DDoS flood
    ("AMI_Network_01", "CRITICAL")  : (["METER_DATA", "TELEMETRY"],                       [0.8, 0.2]),
    ("AMI_Network_01", "Recovery")  : (["METER_DATA", "TELEMETRY"],                       [0.7, 0.3]),

    ("ThirdParty_Market_01", "NORMAL")   : (["MARKET_OPERATIONS", "TELEMETRY"],           [0.6, 0.4]),
    ("ThirdParty_Market_01", "ELEVATED") : (["MARKET_OPERATIONS", "METER_DATA"],          [0.5, 0.5]),
    ("ThirdParty_Market_01", "EMERGENCY"): (["TELEMETRY", "METER_DATA"],                  [0.5, 0.5]),   # spam
    ("ThirdParty_Market_01", "CRITICAL") : (["TELEMETRY", "METER_DATA", "MARKET_OPERATIONS"], [0.5, 0.3, 0.2]),
    ("ThirdParty_Market_01", "Recovery") : (["MARKET_OPERATIONS", "TELEMETRY"],           [0.6, 0.4]),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Phase & Scenario helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_scenario(step: int) -> str:
    if step < 260:   return "NORMAL"
    if step < 520:   return "ELEVATED"
    if step < 780:   return "EMERGENCY"
    if step < 1040:  return "CRITICAL"
    return "Recovery"


def get_grid_conditions(scenario: str, rng) -> tuple:
    """Return (grid_state, freq_deviation, renewable_pct, emergency_flag) for a step."""
    cfg   = PHASE_GRID[scenario]
    freq  = float(np.clip(rng.normal(cfg["freq_mean"], cfg["freq_std"]), 0.0, 0.5))
    ren   = float(np.clip(rng.normal(cfg["ren_mean"],  cfg["ren_std"]),  0.0, 100.0))
    return cfg["grid_state"], freq, ren, cfg["emg"]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-client traffic generators
# ─────────────────────────────────────────────────────────────────────────────

def get_message_type(client_id: str, scenario: str, rng) -> str:
    key   = (client_id, scenario)
    types, probs = MSG_DIST.get(key, (["TELEMETRY"], [1.0]))
    return rng.choice(types, p=probs)


def get_urgency(client_id: str, message_type: str, scenario: str) -> bool:
    """Urgency flag: True for safety-critical message types or operators during crises."""
    critical_msgs = {"FAULT_ISOLATION", "PROTECTION_RELAY", "LOAD_SHEDDING"}
    if message_type in critical_msgs:
        return True
    if client_id == "GridOps_Station" and scenario in ("EMERGENCY", "CRITICAL"):
        return True
    if client_id == "Utility_API_01" and message_type == "DEMAND_RESPONSE" and scenario == "CRITICAL":
        return True
    return False


def get_incoming(client_id: str, scenario: str, rng) -> int:
    """Requests generated by this client in one time-step."""
    if client_id == "GridOps_Station":
        if scenario in ("EMERGENCY", "CRITICAL"):
            return max(1, int(rng.normal(55, 10)))    # surge of critical commands
        return max(1, int(rng.normal(25, 5)))

    if client_id == "Utility_API_01":
        if scenario == "CRITICAL":
            return max(1, int(rng.normal(80, 15)))
        return max(1, int(rng.normal(45, 10)))

    if client_id == "DER_Provider_01":
        if scenario in ("ELEVATED", "EMERGENCY", "CRITICAL"):
            return max(1, int(rng.normal(70, 12)))
        return max(1, int(rng.normal(35, 7)))

    if client_id == "AMI_Network_01":
        if scenario in ("EMERGENCY", "CRITICAL"):
            return max(1, int(rng.normal(420, 50)))   # DDoS flood
        return max(1, int(rng.normal(90, 18)))

    if client_id == "ThirdParty_Market_01":
        if scenario in ("EMERGENCY", "CRITICAL"):
            return max(1, int(rng.normal(320, 40)))   # opportunistic spam
        return max(1, int(rng.normal(60, 12)))

    return 10


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(steps: int = STEPS_PER_SIM, seed: int = 42) -> pd.DataFrame:
    """
    Run the Priority-Context rate limiter simulation.

    One row is written per (client, time-step) pair:
        5 clients x 1300 steps = 6500 rows

    Parameters
    ----------
    steps : number of time-steps (default 1300)
    seed  : random seed

    Returns
    -------
    pd.DataFrame with 23 columns
    """
    rng     = np.random.default_rng(seed)
    limiter = PriorityContextRateLimiter(L_base=L_BASE, beta=BETA)
    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    rows    = []

    for step in range(steps):
        ts       = base_ts + timedelta(seconds=step)
        scenario = get_scenario(step)
        gs, freq, ren, emg = get_grid_conditions(scenario, rng)

        for cid in CLIENT_IDS:
            msg      = get_message_type(cid, scenario, rng)
            urgency  = get_urgency(cid, msg, scenario)
            incoming = get_incoming(cid, scenario, rng)
            role     = CLIENT_ROLES[cid]

            result = limiter.process(
                source_role    = role,
                message_type   = msg,
                urgency        = urgency,
                incoming       = incoming,
                grid_state     = gs,
                freq_deviation = freq,
                renewable_pct  = ren,
                emergency_flag = emg,
            )

            rows.append({
                "timestamp"     : ts.strftime("%Y-%m-%d %H:%M:%S"),
                "step"          : step,
                "scenario"      : scenario,
                "client_id"     : cid,
                "source_role"   : role,
                "message_type"  : msg,
                "urgency"       : urgency,
                # -- Grid conditions --
                "grid_state"    : gs,
                "freq_deviation": round(freq, 4),
                "renewable_pct" : round(ren, 2),
                "emergency_flag": emg,
                # -- Priority --
                "p_message"     : result["p_message"],
                "p_role"        : result["p_role"],
                "p_effective"   : result["p_effective"],
                "priority_tier" : result["priority_tier"],
                # -- Context --
                "gamma"         : result["gamma"],
                "base_gamma"    : result["base_gamma"],
                "adj_factor"    : result["adj_factor"],
                # -- Rate limit formula --
                "rate_limit"    : result["rate_limit"],
                # -- Decision --
                "incoming_req"  : incoming,
                "accepted_req"  : result["accepted"],
                "rejected_req"  : result["rejected"],
                "rejection_rate": result["rejection_rate"],
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> tuple:
    """Print three summary tables: scenario, client, and priority tier."""
    LINE  = "=" * 130
    LINE2 = "-" * 130

    print(f"\n{LINE}")
    print("  PRIORITY-AWARE & CONTEXT-AWARE RATE LIMITER -- SIMULATION RESULTS")
    print("  Paper Section IV.C  |  6500-record Smart Grid Simulation (5 clients x 1300 steps)")
    print(f"  Formula: L(p,c) = L_base*(1 + beta*p*gamma(c))   "
          f"L_base={L_BASE:.0f}  beta={BETA:.1f}")
    print(LINE)

    # ── Table 1: per-scenario ────────────────────────────────────────────────
    sc = (
        df.groupby("scenario")
          .agg(
              Records       = ("step",           "count"),
              Avg_Incoming  = ("incoming_req",   "mean"),
              Avg_p         = ("p_effective",    "mean"),
              Avg_gamma     = ("gamma",          "mean"),
              Avg_RateLimit = ("rate_limit",     "mean"),
              Max_RateLimit = ("rate_limit",     "max"),
              Min_RateLimit = ("rate_limit",     "min"),
              Avg_Rej_pct   = ("rejection_rate", "mean"),
              Total_Accept  = ("accepted_req",   "sum"),
              Total_Reject  = ("rejected_req",   "sum"),
          )
          .round(2)
          .reindex(SCENARIO_ORDER)
    )
    sc["Avg_Rej_pct"] = (sc["Avg_Rej_pct"] * 100).round(2)

    print("\n  [TABLE 1]  Per-Scenario Summary")
    print(LINE2)
    print(sc.to_string())

    # ── Table 2: per-client ──────────────────────────────────────────────────
    cl = (
        df.groupby("client_id")
          .agg(
              Records       = ("step",           "count"),
              Avg_Incoming  = ("incoming_req",   "mean"),
              Max_Incoming  = ("incoming_req",   "max"),
              Avg_p         = ("p_effective",    "mean"),
              Avg_gamma     = ("gamma",          "mean"),
              Avg_RateLimit = ("rate_limit",     "mean"),
              Max_RateLimit = ("rate_limit",     "max"),
              Avg_Rej_pct   = ("rejection_rate", "mean"),
              Total_Accept  = ("accepted_req",   "sum"),
              Total_Reject  = ("rejected_req",   "sum"),
          )
          .round(2)
          .reindex(CLIENT_IDS)
    )
    cl["Avg_Rej_pct"] = (cl["Avg_Rej_pct"] * 100).round(2)

    print(f"\n  [TABLE 2]  Per-Client Summary")
    print(LINE2)
    print(cl.to_string())

    # ── Table 3: per-priority-tier ───────────────────────────────────────────
    pt = (
        df.groupby("priority_tier")
          .agg(
              Records       = ("step",           "count"),
              Avg_p         = ("p_effective",    "mean"),
              Avg_gamma     = ("gamma",          "mean"),
              Avg_RateLimit = ("rate_limit",     "mean"),
              Max_RateLimit = ("rate_limit",     "max"),
              Avg_Incoming  = ("incoming_req",   "mean"),
              Avg_Rej_pct   = ("rejection_rate", "mean"),
              Total_Accept  = ("accepted_req",   "sum"),
              Total_Reject  = ("rejected_req",   "sum"),
          )
          .round(2)
          .reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    )
    pt["Avg_Rej_pct"] = (pt["Avg_Rej_pct"] * 100).round(2)

    print(f"\n  [TABLE 3]  Per-Priority-Tier Summary")
    print(LINE2)
    print(pt.to_string())
    print(LINE2)

    total_in  = int(df["incoming_req"].sum())
    total_acc = int(df["accepted_req"].sum())
    total_rej = int(df["rejected_req"].sum())
    print(
        f"\n  OVERALL  Records: {len(df):,}  |  "
        f"Incoming: {total_in:,}  Accepted: {total_acc:,}  "
        f"Rejected: {total_rej:,}  Rejection Rate: {total_rej/total_in*100:.2f}%"
    )
    print(
        f"           Avg p: {df.p_effective.mean():.3f}  "
        f"  Avg gamma: {df.gamma.mean():.3f}  "
        f"  Avg rate-limit: {df.rate_limit.mean():.1f}  "
        f"  Peak rate-limit: {df.rate_limit.max():.1f}"
    )
    print(LINE)
    return sc, cl, pt


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shade(ax, alpha: float = 0.07) -> None:
    for j, color in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j + 1], alpha=alpha, color=color)
    ax.set_xlim(0, STEPS_PER_SIM)


def _phase_labels(ax, y_frac: float = 0.96) -> None:
    ylo, yhi = ax.get_ylim()
    y = ylo + (yhi - ylo) * y_frac
    for j, (name, color) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j + 1]) / 2
        ax.text(mid, y, name, ha="center", va="top",
                fontsize=6.5, color=color, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main results plot (8 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, out: str = "priority_context_results.png") -> None:
    """
    8-panel diagnostic figure.

    A  Incoming request rate per client              (full width)
    B  Computed rate limit L(p,c) per client
    C  Context multiplier gamma(c) over time
    D  Effective priority p per client over time
    E  Accepted vs Rejected stacked (all clients)
    F  Grid conditions: freq deviation & renewable % (dual axis)
    G  Rejection rate by scenario x client           (grouped bar)
    H  Avg rate limit by priority tier x scenario    (grouped bar)
    """
    piv_in    = df.pivot_table(index="step", columns="client_id", values="incoming_req",   aggfunc="mean")
    piv_rl    = df.pivot_table(index="step", columns="client_id", values="rate_limit",     aggfunc="mean")
    piv_p     = df.pivot_table(index="step", columns="client_id", values="p_effective",    aggfunc="mean")
    piv_acc   = df.pivot_table(index="step", columns="client_id", values="accepted_req",   aggfunc="sum")
    piv_rej   = df.pivot_table(index="step", columns="client_id", values="rejected_req",   aggfunc="sum")
    piv_gamma = df.pivot_table(index="step", columns="client_id", values="gamma",          aggfunc="mean")
    piv_freq  = df.pivot_table(index="step", values="freq_deviation", aggfunc="mean")
    piv_ren   = df.pivot_table(index="step", values="renewable_pct",  aggfunc="mean")

    steps = piv_in.index

    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "Smart Grid -- Priority-Aware & Context-Aware Rate Limiter  (Section IV.C)\n"
        "L(p, c) = L_base * (1 + beta * p * gamma(c))"
        "     [6500-record simulation | 5 clients x 1300 steps | 5 phases]",
        fontsize=12, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        hspace=0.50, wspace=0.30,
        top=0.960, bottom=0.05, left=0.07, right=0.97,
    )

    # ── A: Incoming request rate ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    _shade(ax)
    for cid in CLIENT_IDS:
        ax.plot(steps, piv_in[cid], color=CLIENT_COLORS[cid], lw=0.9, alpha=0.85, label=cid)
    ax.set_ylabel("Incoming Requests / step", fontsize=9)
    ax.set_title("(A)  Per-Client Incoming Request Rate", fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=5)
    ax.grid(True, alpha=0.30, lw=0.6)
    _phase_labels(ax)

    # ── B: Rate limit L(p,c) per client ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _shade(ax)
    for cid in CLIENT_IDS:
        ax.plot(steps, piv_rl[cid], color=CLIENT_COLORS[cid], lw=0.9, alpha=0.85, label=cid)
    ax.axhline(L_BASE, color="navy", ls="--", lw=1.2, alpha=0.7, label=f"L_base={L_BASE:.0f}")
    ax.set_ylabel("Rate Limit L(p,c)  [req/step]", fontsize=9)
    ax.set_title("(B)  Adaptive Rate Limit  L(p,c) = L_base*(1 + beta*p*gamma)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.30, lw=0.6)

    # ── C: Context multiplier gamma(c) ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _shade(ax)
    gamma_all = df.groupby("step")["gamma"].mean()
    ax.fill_between(gamma_all.index, gamma_all, alpha=0.25, color="#e74c3c")
    ax.plot(gamma_all.index, gamma_all, color="#c0392b", lw=1.2, label="gamma(c) [avg]")
    for j, (name, color) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j + 1]) / 2
        gval = gamma_all.iloc[
            (gamma_all.index >= PHASE_BOUNDS[j]) & (gamma_all.index < PHASE_BOUNDS[j+1])
        ].mean() if hasattr(gamma_all.index, '__len__') else 1.0
        gval = gamma_all[
            (gamma_all.index >= PHASE_BOUNDS[j]) & (gamma_all.index < PHASE_BOUNDS[j+1])
        ].mean()
        ax.annotate(f"avg={gval:.2f}", xy=(mid, gval),
                    fontsize=7, ha="center", color=color, fontweight="bold",
                    xytext=(0, 12), textcoords="offset points")
    ax.set_ylabel("Context Multiplier gamma(c)", fontsize=9)
    ax.set_title("(C)  Grid Context Multiplier gamma(c)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)
    ax.set_xlim(0, STEPS_PER_SIM)

    # ── D: Effective priority p per client ────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    _shade(ax)
    for cid in CLIENT_IDS:
        ax.plot(steps, piv_p[cid], color=CLIENT_COLORS[cid], lw=0.9, alpha=0.85, label=cid)
    # Tier boundary lines
    for thresh, label, clr in [(0.85, "CRITICAL", "#e74c3c"), (0.65, "HIGH", "#e67e22"),
                                (0.45, "MEDIUM", "#3498db")]:
        ax.axhline(thresh, color=clr, ls=":", lw=1.0, alpha=0.6, label=f"p>={thresh} ({label})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Effective Priority Weight p", fontsize=9)
    ax.set_title("(D)  Effective Priority p per Client",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.30, lw=0.6)

    # ── E: Accepted vs Rejected stacked ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    _shade(ax)
    total_acc = piv_acc.sum(axis=1)
    total_rej = piv_rej.sum(axis=1)
    ax.stackplot(steps, total_acc, total_rej,
                 labels=["Accepted", "Rejected"],
                 colors=["#27ae60", "#e74c3c"], alpha=0.72)
    ax.set_ylabel("Requests / step", fontsize=9)
    ax.set_title("(E)  Total Accepted vs Rejected (All Clients)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.30, lw=0.6)
    ax.set_xlim(0, STEPS_PER_SIM)

    # ── F: Grid conditions (freq + renewable) ─────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    _shade(ax)
    ax2 = ax.twinx()
    ax.plot(piv_freq.index, piv_freq.values, color="#e74c3c", lw=0.9,
            alpha=0.85, label="Freq Deviation (Hz)")
    ax.axhline(0.5, color="#e74c3c", ls="--", lw=1.0, alpha=0.5, label="0.5 Hz threshold")
    ax2.plot(piv_ren.index, piv_ren.values, color="#27ae60", lw=0.9,
             alpha=0.85, label="Renewable %")
    ax.set_ylabel("Freq Deviation from 60 Hz (Hz)", color="#e74c3c", fontsize=9)
    ax2.set_ylabel("Renewable Penetration (%)",      color="#27ae60", fontsize=9)
    ax.set_title("(F)  Grid Conditions: Frequency Deviation & Renewable Penetration",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(loc="upper left",  fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.30, lw=0.6)
    ax.set_xlim(0, STEPS_PER_SIM)

    # ── G: Rejection rate by scenario x client ────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    rej_piv = (
        df.groupby(["scenario", "client_id"])["rejection_rate"]
          .mean()
          .unstack("client_id")
          .reindex(SCENARIO_ORDER) * 100
    )
    x     = np.arange(len(SCENARIO_ORDER))
    n_cl  = len(CLIENT_IDS)
    width = 0.15
    for k, cid in enumerate(CLIENT_IDS):
        offset = (k - n_cl / 2 + 0.5) * width
        bars = ax.bar(x + offset, rej_piv[cid].fillna(0), width,
                      color=CLIENT_COLORS[cid], alpha=0.78, label=cid)
        for bar in bars:
            h = bar.get_height()
            if h > 8:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.0f}%", ha="center", va="bottom",
                        fontsize=5.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_ORDER, fontsize=8, rotation=10)
    ax.set_ylabel("Avg Rejection Rate (%)", fontsize=9)
    ax.set_title("(G)  Rejection Rate by Scenario x Client",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.30, lw=0.6, axis="y")

    # Legend strips
    phase_handles  = [Patch(facecolor=c, alpha=0.65, label=n)
                      for n, c in zip(SCENARIO_ORDER, SCENARIO_COLORS)]
    client_handles = [Patch(facecolor=CLIENT_COLORS[cid], alpha=0.80, label=cid)
                      for cid in CLIENT_IDS]
    fig.legend(
        handles=phase_handles + client_handles,
        loc="lower center", ncol=10, fontsize=8,
        title="Phases and Clients",
        title_fontsize=8,
        bbox_to_anchor=(0.5, 0.001),
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plots saved  ->  {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Supplementary scatter + tier bar plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_supplementary(df: pd.DataFrame, out: str = "priority_context_scatter.png") -> None:
    """
    Two supplementary panels:

    H  Scatter: p_effective vs rate_limit coloured by grid_state
       (visually confirms the L(p,c) formula)
    I  Grouped bar: avg rate limit by priority tier x scenario
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "Priority-Aware Rate Limiter -- Supplementary Analysis  (Section IV.C)",
        fontsize=12, fontweight="bold",
    )

    gs_colors = {
        "NORMAL"   : "#2ecc71",
        "ELEVATED" : "#f1c40f",
        "ALERT"    : "#e67e22",
        "EMERGENCY": "#e74c3c",
        "CRITICAL" : "#8e44ad",
    }

    # ── H: p vs L(p,c) scatter ───────────────────────────────────────────────
    ax = axes[0]
    sample = df.sample(min(3000, len(df)), random_state=42)
    for gs_label, gs_color in gs_colors.items():
        sub = sample[sample["grid_state"] == gs_label]
        if not sub.empty:
            ax.scatter(sub["p_effective"], sub["rate_limit"],
                       c=gs_color, s=14, alpha=0.45, label=gs_label)
    # Overlay formula curves
    p_line = np.linspace(0, 1, 200)
    for gs_label, gs_color in gs_colors.items():
        avg_gamma = df[df["grid_state"] == gs_label]["gamma"].mean()
        if np.isnan(avg_gamma):
            continue
        l_line = L_BASE * (1 + BETA * p_line * avg_gamma)
        l_line = np.clip(l_line, 50, 2000)
        ax.plot(p_line, l_line, color=gs_color, lw=1.8, ls="--", alpha=0.85)
    ax.set_xlabel("Effective Priority Weight p", fontsize=10)
    ax.set_ylabel("Rate Limit L(p,c)  [req/step]", fontsize=10)
    ax.set_title("(H)  p_effective vs Rate Limit L(p,c)\n"
                 "(dashed = formula curve per grid state, dots = simulation data)",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(title="Grid State", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6)

    # ── I: Avg rate limit by tier x scenario (grouped bar) ───────────────────
    ax = axes[1]
    tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    rl_piv = (
        df.groupby(["scenario", "priority_tier"])["rate_limit"]
          .mean()
          .unstack("priority_tier")
          .reindex(SCENARIO_ORDER)[tier_order]
    )
    x     = np.arange(len(SCENARIO_ORDER))
    width = 0.20
    for k, tier in enumerate(tier_order):
        offset = (k - 2 + 0.5) * width
        bars = ax.bar(x + offset, rl_piv[tier].fillna(0), width,
                      color=TIER_COLORS[tier], alpha=0.78, label=tier)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                        f"{h:.0f}", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_ORDER, fontsize=9, rotation=10)
    ax.set_ylabel("Avg Rate Limit L(p,c)  [req/step]", fontsize=10)
    ax.set_title("(I)  Avg Rate Limit by Priority Tier x Scenario\n"
                 "(shows how gamma amplifies high-priority limits during crises)",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(title="Priority Tier", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.30, lw=0.6, axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Plots saved  ->  {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_FILE  = "priority_context_simulation.csv"
    PNG_MAIN  = "priority_context_results.png"
    PNG_SUPP  = "priority_context_scatter.png"

    print("=" * 70)
    print("  Smart Grid -- Priority-Aware & Context-Aware Rate Limiter")
    print("  Section IV.C: Priority Aware and Context Aware Policy")
    print("=" * 70)
    print(f"\n  Running: {len(CLIENT_IDS)} clients x {STEPS_PER_SIM} steps "
          f"= {len(CLIENT_IDS)*STEPS_PER_SIM:,} records ...")

    df = run_simulation(steps=STEPS_PER_SIM, seed=42)

    df.to_csv(CSV_FILE, index=False)
    print(f"  CSV saved    ->  {CSV_FILE}  ({len(df):,} rows, {len(df.columns)} columns)")

    print_summary(df)

    print("\n  Generating plots ...")
    plot_results(df, out=PNG_MAIN)
    plot_supplementary(df, out=PNG_SUPP)

    print("\n  Done.")
