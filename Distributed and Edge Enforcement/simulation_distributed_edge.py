#!/usr/bin/env python3
"""
simulation_distributed_edge.py
================================
Simulation harness for Section IV.E: Distributed and Edge Enforcement.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh

Structure  --  6 edge nodes x 1500 time steps = 9 000 records
-----------------------------------------------------------------
Node                    Type                Region      L_base  Proximity
--------------------    ----------------    ---------   ------  ---------
DER_Hub_North           DER Controller      NORTH       2000    0.95
PMU_Collector_West      PMU Sync            WEST        1800    0.90
AMI_Gateway_East        AMI Data Gateway    EAST        1500    0.70
EV_Station_South        EV Charging Hub     SOUTH       1200    0.65
Smart_Meter_Cluster_1   Telemetry Leaf      EAST        800     0.40
Smart_Meter_Cluster_2   Telemetry Leaf      WEST        800     0.40

Simulation Phases (250 steps each, 6 phases x 250 = 1500 total)
----------------------------------------------------------------
Phase 1  (0-249)    NORMAL          Balanced baseline operation
Phase 2  (250-499)  LOAD_SPIKE      Uneven surge (South + East nodes)
Phase 3  (500-749)  DDOS_ATTACK     Flood targeting leaf nodes (SM1, SM2)
Phase 4  (750-999)  NODE_FAILURE    Smart_Meter_Cluster_1 goes offline
Phase 5  (1000-1249)MTD_ACTIVE      MTD engine rotates L_base parameters
Phase 6  (1250-1499)RECOVERY        System stabilises after crisis

Global formula (paper):
    L_global = sum_e L_e * phi(e)
    phi(e)   = (0.60*(L_base_e/L_base_max) + 0.40*proximity_e) / normaliser

Outputs
-------
    distributed_edge_simulation.csv         9 000-row dataset (22 columns)
    distributed_edge_main_results.png       8-panel main chart
    distributed_edge_supplementary.png      4-panel supplementary chart
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

from distributed_edge_rate_limiter import (
    EdgeNodeConfig, DistributedEdgeRateLimiter
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Edge Node Definitions
# ─────────────────────────────────────────────────────────────────────────────

NODE_CONFIGS = [
    EdgeNodeConfig(
        node_id="DER_Hub_North",
        L_base=2000, proximity_score=0.95,
        neighbors=["PMU_Collector_West", "AMI_Gateway_East"],
        node_type="DER_CONTROLLER", region="NORTH",
    ),
    EdgeNodeConfig(
        node_id="PMU_Collector_West",
        L_base=1800, proximity_score=0.90,
        neighbors=["DER_Hub_North", "Smart_Meter_Cluster_2"],
        node_type="PMU_SYNC", region="WEST",
    ),
    EdgeNodeConfig(
        node_id="AMI_Gateway_East",
        L_base=1500, proximity_score=0.70,
        neighbors=["DER_Hub_North", "EV_Station_South", "Smart_Meter_Cluster_1"],
        node_type="AMI_GATEWAY", region="EAST",
    ),
    EdgeNodeConfig(
        node_id="EV_Station_South",
        L_base=1200, proximity_score=0.65,
        neighbors=["AMI_Gateway_East", "Smart_Meter_Cluster_1"],
        node_type="EV_HUB", region="SOUTH",
    ),
    EdgeNodeConfig(
        node_id="Smart_Meter_Cluster_1",
        L_base=800, proximity_score=0.40,
        neighbors=["AMI_Gateway_East", "EV_Station_South"],
        node_type="TELEMETRY_LEAF", region="EAST",
    ),
    EdgeNodeConfig(
        node_id="Smart_Meter_Cluster_2",
        L_base=800, proximity_score=0.40,
        neighbors=["PMU_Collector_West"],
        node_type="TELEMETRY_LEAF", region="WEST",
    ),
]

NODE_IDS     = [c.node_id for c in NODE_CONFIGS]
NODE_COLORS  = {
    "DER_Hub_North"         : "#e74c3c",
    "PMU_Collector_West"    : "#9b59b6",
    "AMI_Gateway_East"      : "#3498db",
    "EV_Station_South"      : "#f39c12",
    "Smart_Meter_Cluster_1" : "#2ecc71",
    "Smart_Meter_Cluster_2" : "#1abc9c",
}

STEPS_PER_SIM  = 1500
SCENARIO_ORDER = ["NORMAL", "LOAD_SPIKE", "DDOS_ATTACK",
                  "NODE_FAILURE", "MTD_ACTIVE", "RECOVERY"]
SCENARIO_COLORS= ["#2ecc71", "#f1c40f", "#e74c3c",
                  "#8e44ad", "#e67e22", "#3498db"]
PHASE_BOUNDS   = [0, 250, 500, 750, 1000, 1250, 1500]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Scenario helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_scenario(step: int) -> str:
    if step < 250:   return "NORMAL"
    if step < 500:   return "LOAD_SPIKE"
    if step < 750:   return "DDOS_ATTACK"
    if step < 1000:  return "NODE_FAILURE"
    if step < 1250:  return "MTD_ACTIVE"
    return "RECOVERY"


def is_mtd_active(step: int) -> bool:
    return 1000 <= step < 1250


def get_availability(node_id: str, step: int) -> bool:
    """Smart_Meter_Cluster_1 is offline during NODE_FAILURE phase."""
    if node_id == "Smart_Meter_Cluster_1" and 750 <= step < 1000:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Traffic & load generators
# ─────────────────────────────────────────────────────────────────────────────

# Per-(node, scenario) incoming request parameters  (mu, sigma)
_INCOMING_PARAMS = {
    # NORMAL: all nodes near their normal operating range
    ("DER_Hub_North",         "NORMAL")      : (180,  25),
    ("PMU_Collector_West",    "NORMAL")      : (220,  30),
    ("AMI_Gateway_East",      "NORMAL")      : (160,  20),
    ("EV_Station_South",      "NORMAL")      : (130,  18),
    ("Smart_Meter_Cluster_1", "NORMAL")      : (90,   12),
    ("Smart_Meter_Cluster_2", "NORMAL")      : (85,   12),
    # LOAD_SPIKE: south + east surge (EV charging + AMI burst)
    ("DER_Hub_North",         "LOAD_SPIKE")  : (200,  30),
    ("PMU_Collector_West",    "LOAD_SPIKE")  : (230,  30),
    ("AMI_Gateway_East",      "LOAD_SPIKE")  : (900, 120),   # spike
    ("EV_Station_South",      "LOAD_SPIKE")  : (850, 100),   # spike
    ("Smart_Meter_Cluster_1", "LOAD_SPIKE")  : (700,  80),
    ("Smart_Meter_Cluster_2", "LOAD_SPIKE")  : (110,  15),
    # DDOS_ATTACK: flood on leaf nodes
    ("DER_Hub_North",         "DDOS_ATTACK") : (190,  25),
    ("PMU_Collector_West",    "DDOS_ATTACK") : (210,  28),
    ("AMI_Gateway_East",      "DDOS_ATTACK") : (250,  35),
    ("EV_Station_South",      "DDOS_ATTACK") : (220,  30),
    ("Smart_Meter_Cluster_1", "DDOS_ATTACK") : (1600, 200),  # DDoS target
    ("Smart_Meter_Cluster_2", "DDOS_ATTACK") : (1400, 180),  # DDoS target
    # NODE_FAILURE: SM1 offline; traffic redistributes
    ("DER_Hub_North",         "NODE_FAILURE"): (200,  25),
    ("PMU_Collector_West",    "NODE_FAILURE"): (220,  30),
    ("AMI_Gateway_East",      "NODE_FAILURE"): (380,  50),   # absorbs SM1 traffic
    ("EV_Station_South",      "NODE_FAILURE"): (320,  45),
    ("Smart_Meter_Cluster_1", "NODE_FAILURE"): (0,     0),   # offline
    ("Smart_Meter_Cluster_2", "NODE_FAILURE"): (90,   12),
    # MTD_ACTIVE: normal traffic but limits rotate
    ("DER_Hub_North",         "MTD_ACTIVE")  : (185,  25),
    ("PMU_Collector_West",    "MTD_ACTIVE")  : (215,  28),
    ("AMI_Gateway_East",      "MTD_ACTIVE")  : (165,  22),
    ("EV_Station_South",      "MTD_ACTIVE")  : (140,  18),
    ("Smart_Meter_Cluster_1", "MTD_ACTIVE")  : (95,   12),
    ("Smart_Meter_Cluster_2", "MTD_ACTIVE")  : (90,   12),
    # RECOVERY: gradual return to normal
    ("DER_Hub_North",         "RECOVERY")    : (183,  22),
    ("PMU_Collector_West",    "RECOVERY")    : (212,  28),
    ("AMI_Gateway_East",      "RECOVERY")    : (170,  22),
    ("EV_Station_South",      "RECOVERY")    : (135,  18),
    ("Smart_Meter_Cluster_1", "RECOVERY")    : (88,   12),
    ("Smart_Meter_Cluster_2", "RECOVERY")    : (86,   11),
}

# Per-(node, scenario) base CPU load [fraction]
_LOAD_PARAMS = {
    ("DER_Hub_North",         "NORMAL")      : (0.25, 0.04),
    ("PMU_Collector_West",    "NORMAL")      : (0.28, 0.04),
    ("AMI_Gateway_East",      "NORMAL")      : (0.22, 0.04),
    ("EV_Station_South",      "NORMAL")      : (0.20, 0.04),
    ("Smart_Meter_Cluster_1", "NORMAL")      : (0.18, 0.03),
    ("Smart_Meter_Cluster_2", "NORMAL")      : (0.18, 0.03),

    ("DER_Hub_North",         "LOAD_SPIKE")  : (0.35, 0.05),
    ("PMU_Collector_West",    "LOAD_SPIKE")  : (0.38, 0.05),
    ("AMI_Gateway_East",      "LOAD_SPIKE")  : (0.82, 0.06),
    ("EV_Station_South",      "LOAD_SPIKE")  : (0.78, 0.06),
    ("Smart_Meter_Cluster_1", "LOAD_SPIKE")  : (0.85, 0.05),
    ("Smart_Meter_Cluster_2", "LOAD_SPIKE")  : (0.30, 0.04),

    ("DER_Hub_North",         "DDOS_ATTACK") : (0.30, 0.04),
    ("PMU_Collector_West",    "DDOS_ATTACK") : (0.32, 0.04),
    ("AMI_Gateway_East",      "DDOS_ATTACK") : (0.45, 0.05),
    ("EV_Station_South",      "DDOS_ATTACK") : (0.40, 0.05),
    ("Smart_Meter_Cluster_1", "DDOS_ATTACK") : (0.92, 0.04),
    ("Smart_Meter_Cluster_2", "DDOS_ATTACK") : (0.88, 0.04),

    ("DER_Hub_North",         "NODE_FAILURE"): (0.38, 0.05),
    ("PMU_Collector_West",    "NODE_FAILURE"): (0.42, 0.05),
    ("AMI_Gateway_East",      "NODE_FAILURE"): (0.75, 0.06),
    ("EV_Station_South",      "NODE_FAILURE"): (0.68, 0.06),
    ("Smart_Meter_Cluster_1", "NODE_FAILURE"): (0.00, 0.00),
    ("Smart_Meter_Cluster_2", "NODE_FAILURE"): (0.22, 0.03),

    ("DER_Hub_North",         "MTD_ACTIVE")  : (0.28, 0.04),
    ("PMU_Collector_West",    "MTD_ACTIVE")  : (0.30, 0.04),
    ("AMI_Gateway_East",      "MTD_ACTIVE")  : (0.25, 0.04),
    ("EV_Station_South",      "MTD_ACTIVE")  : (0.22, 0.04),
    ("Smart_Meter_Cluster_1", "MTD_ACTIVE")  : (0.20, 0.03),
    ("Smart_Meter_Cluster_2", "MTD_ACTIVE")  : (0.20, 0.03),

    ("DER_Hub_North",         "RECOVERY")    : (0.27, 0.04),
    ("PMU_Collector_West",    "RECOVERY")    : (0.30, 0.04),
    ("AMI_Gateway_East",      "RECOVERY")    : (0.28, 0.04),
    ("EV_Station_South",      "RECOVERY")    : (0.24, 0.04),
    ("Smart_Meter_Cluster_1", "RECOVERY")    : (0.19, 0.03),
    ("Smart_Meter_Cluster_2", "RECOVERY")    : (0.19, 0.03),
}


def get_incoming(node_id: str, scenario: str, rng, available: bool) -> int:
    if not available:
        return 0
    mu, sigma = _INCOMING_PARAMS.get((node_id, scenario), (100, 15))
    return max(0, int(rng.normal(mu, sigma)))


def get_load(node_id: str, scenario: str, rng, available: bool) -> float:
    if not available:
        return 0.0
    mu, sigma = _LOAD_PARAMS.get((node_id, scenario), (0.25, 0.04))
    return float(np.clip(rng.normal(mu, sigma), 0.0, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(steps: int = STEPS_PER_SIM, seed: int = 42) -> pd.DataFrame:
    """
    Run the Distributed Edge Rate Limiting simulation.

    One CSV row per (node, step) pair:
        6 nodes x 1500 steps = 9 000 rows

    Parameters
    ----------
    steps : number of time steps (default 1500)
    seed  : random seed

    Returns
    -------
    pd.DataFrame with 22 columns
    """
    rng     = np.random.default_rng(seed)
    limiter = DistributedEdgeRateLimiter(NODE_CONFIGS, seed=seed)
    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    rows    = []

    for step in range(steps):
        ts       = base_ts + timedelta(seconds=step)
        scenario = get_scenario(step)
        mtd      = is_mtd_active(step)

        # Build per-node dicts for this step
        avail_dict    = {nid: get_availability(nid, step) for nid in NODE_IDS}
        load_dict     = {nid: get_load(nid, scenario, rng, avail_dict[nid])
                         for nid in NODE_IDS}
        incoming_dict = {nid: get_incoming(nid, scenario, rng, avail_dict[nid])
                         for nid in NODE_IDS}

        # Process through the distributed limiter
        results = limiter.step(
            step             = step,
            incoming_by_node = incoming_dict,
            load_by_node     = load_dict,
            availability     = avail_dict,
            mtd_active       = mtd,
        )

        for nid in NODE_IDS:
            cfg = next(c for c in NODE_CONFIGS if c.node_id == nid)
            res = results[nid]

            rows.append({
                "timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
                "step"            : step,
                "scenario"        : scenario,
                "node_id"         : nid,
                "node_type"       : cfg.node_type,
                "region"          : cfg.region,
                "L_base"          : cfg.L_base,
                "proximity_score" : cfg.proximity_score,
                # -- Algorithm outputs --
                "phi_e"           : res["phi_e"],
                "load_e"          : res["load_e"],
                "availability"    : int(res["availability"]),
                "mtd_factor"      : res["mtd_factor"],
                "gossip_adj"      : res["gossip_adj"],
                "is_gossip_step"  : int(res["is_gossip_step"]),
                "L_e"             : res["L_e"],
                "L_global"        : res["L_global"],
                # -- Traffic --
                "incoming_req"    : incoming_dict[nid],
                "accepted_req"    : res["accepted"],
                "rejected_req"    : res["rejected"],
                "rejection_rate"  : res["rejection_rate"],
                "tokens"          : res["tokens"],
                "mtd_active"      : int(mtd),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> tuple:
    """Print three summary tables: scenario, node, and phase x node."""
    LINE  = "=" * 130
    LINE2 = "-" * 130

    print(f"\n{LINE}")
    print("  DISTRIBUTED & EDGE ENFORCEMENT RATE LIMITER -- SIMULATION RESULTS")
    print("  Paper Section IV.E  |  9000-record Simulation (6 nodes x 1500 steps)")
    print("  Formula: L_global = sum_e L_e * phi(e)")
    print(LINE)

    # ── Table 1: per-scenario ────────────────────────────────────────────────
    sc = (
        df.groupby("scenario")
          .agg(
              Records        = ("step",            "count"),
              Avg_L_global   = ("L_global",        "mean"),
              Min_L_global   = ("L_global",        "min"),
              Max_L_global   = ("L_global",        "max"),
              Avg_L_e        = ("L_e",             "mean"),
              Avg_Load       = ("load_e",          "mean"),
              Avg_phi        = ("phi_e",           "mean"),
              Avg_MTD        = ("mtd_factor",      "mean"),
              Avg_Gossip_Adj = ("gossip_adj",      "mean"),
              Avg_Rej_pct    = ("rejection_rate",  "mean"),
              Total_Accept   = ("accepted_req",    "sum"),
              Total_Reject   = ("rejected_req",    "sum"),
          )
          .round(3)
          .reindex(SCENARIO_ORDER)
    )
    sc["Avg_Rej_pct"] = (sc["Avg_Rej_pct"] * 100).round(2)

    print("\n  [TABLE 1]  Per-Scenario Summary  (all 6 nodes aggregated)")
    print(LINE2)
    print(sc.to_string())

    # ── Table 2: per-node ────────────────────────────────────────────────────
    nd = (
        df.groupby("node_id")
          .agg(
              Records        = ("step",            "count"),
              L_base         = ("L_base",          "first"),
              Proximity      = ("proximity_score", "first"),
              Avg_phi        = ("phi_e",           "mean"),
              Avg_L_e        = ("L_e",             "mean"),
              Min_L_e        = ("L_e",             "min"),
              Max_L_e        = ("L_e",             "max"),
              Avg_Load       = ("load_e",          "mean"),
              Avg_Gossip_Adj = ("gossip_adj",      "mean"),
              Avg_MTD        = ("mtd_factor",      "mean"),
              Avg_Rej_pct    = ("rejection_rate",  "mean"),
              Total_Accept   = ("accepted_req",    "sum"),
              Total_Reject   = ("rejected_req",    "sum"),
          )
          .round(3)
          .reindex(NODE_IDS)
    )
    nd["Avg_Rej_pct"] = (nd["Avg_Rej_pct"] * 100).round(2)

    print(f"\n  [TABLE 2]  Per-Node Summary (all phases)")
    print(LINE2)
    print(nd.to_string())

    # ── Table 3: L_global pivot per scenario (unique per step) ──────────────
    lg = (
        df.drop_duplicates(subset=["step"])
          .groupby("scenario")["L_global"]
          .agg(["mean", "min", "max", "std"])
          .round(2)
          .reindex(SCENARIO_ORDER)
    )
    lg.columns = ["Mean_L_global", "Min_L_global", "Max_L_global", "Std_L_global"]

    print(f"\n  [TABLE 3]  Global Rate Limit L_global per Scenario")
    print(LINE2)
    print(lg.to_string())
    print(LINE2)

    total_in  = int(df["incoming_req"].sum())
    total_acc = int(df["accepted_req"].sum())
    total_rej = int(df["rejected_req"].sum())
    print(
        f"\n  OVERALL  Records: {len(df):,}  |  "
        f"Incoming: {total_in:,}  Accepted: {total_acc:,}  "
        f"Rejected: {total_rej:,}  "
        f"Rejection Rate: {total_rej / max(total_in,1) * 100:.2f}%"
    )
    print(
        f"           Avg phi range: [{df.phi_e.min():.4f}, {df.phi_e.max():.4f}]  "
        f"  Avg L_global: {df.L_global.mean():.1f}  "
        f"  MTD steps: 250 / 1500"
    )
    print(LINE)
    return sc, nd, lg


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shade(ax, alpha=0.07):
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=alpha, color=c)
    ax.set_xlim(0, STEPS_PER_SIM)


def _phase_labels(ax, y_frac=0.96):
    ylo, yhi = ax.get_ylim()
    y = ylo + (yhi - ylo) * y_frac
    for j, (n, c) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j+1]) / 2
        ax.text(mid, y, n, ha="center", va="top",
                fontsize=5.5, color=c, fontweight="bold", rotation=0)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main results figure (8 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_results(df: pd.DataFrame, out: str = "distributed_edge_main_results.png") -> None:
    """
    8-panel main diagnostic figure.

    A  L_global over time + per-node L_e contribution fill  (full width)
    B  Per-node L_e over time
    C  Node availability timeline (heatmap)
    D  Incoming traffic per node over time
    E  Accepted vs Rejected stacked (all nodes)
    F  Rejection rate per node over time
    G  phi(e) weighting factor per node per phase (grouped bar)
    H  Load distribution heatmap (node x scenario)
    """
    piv_L_e   = df.pivot_table(index="step", columns="node_id", values="L_e",            aggfunc="mean")
    piv_in    = df.pivot_table(index="step", columns="node_id", values="incoming_req",    aggfunc="sum")
    piv_acc   = df.pivot_table(index="step", columns="node_id", values="accepted_req",    aggfunc="sum")
    piv_rej   = df.pivot_table(index="step", columns="node_id", values="rejected_req",    aggfunc="sum")
    piv_rr    = df.pivot_table(index="step", columns="node_id", values="rejection_rate",  aggfunc="mean")
    piv_avail = df.pivot_table(index="step", columns="node_id", values="availability",    aggfunc="mean")
    piv_load  = df.pivot_table(index="step", columns="node_id", values="load_e",          aggfunc="mean")
    piv_phi   = df.pivot_table(index="scenario", columns="node_id", values="phi_e",       aggfunc="mean").reindex(SCENARIO_ORDER)
    L_global  = df.drop_duplicates("step").set_index("step")["L_global"]

    steps = piv_L_e.index
    roll  = 20

    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "Smart Grid -- Distributed & Edge Enforcement Rate Limiter  (Section IV.E)\n"
        "L_global = sum_e L_e * phi(e)     phi(e) = (0.60*capacity_norm + 0.40*proximity) / normaliser\n"
        "[9000-record simulation | 6 edge nodes x 1500 steps | 6 operating phases]",
        fontsize=11, fontweight="bold", y=0.997,
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.52, wspace=0.30,
                           top=0.960, bottom=0.045,
                           left=0.07, right=0.97)

    # ── A: L_global + per-node L_e*phi contribution (full width) ────────────────
    ax = fig.add_subplot(gs[0, :])
    _shade(ax)
    valid_nodes = [n for n in NODE_IDS if n in piv_L_e.columns]
    colors_list = [NODE_COLORS[n] for n in valid_nodes]

    # Per-step phi from simulation data (already computed per row)
    step_phi = df.groupby(["step", "node_id"])["phi_e"].first().unstack("node_id")
    contrib  = [
        (piv_L_e[n].fillna(0) * step_phi[n].fillna(0)).values
        for n in valid_nodes
    ]
    ax.stackplot(steps, contrib,
                 labels=[f"{n.split('_')[0]} contribution" for n in valid_nodes],
                 colors=colors_list, alpha=0.60)
    ax.plot(L_global.index, L_global.values,
            color="black", lw=1.8, zorder=5, label="L_global (aggregate)")
    ax.set_ylabel("Rate Limit (req/step)", fontsize=9)
    ax.set_title("(A)  Global Rate Limit L_global = sum_e [L_e * phi(e)]  --  Per-Node Weighted Contributions",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3, lw=0.6)
    _phase_labels(ax)

    # ── B: Per-node L_e over time ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _shade(ax)
    for nid in valid_nodes:
        ax.plot(steps, piv_L_e[nid].rolling(roll, min_periods=1).mean(),
                color=NODE_COLORS[nid], lw=0.9, alpha=0.85, label=nid.split("_")[0])
    ax.set_ylabel("Local Rate Limit L_e (req/step)", fontsize=9)
    ax.set_title("(B)  Per-Node Local Rate Limit L_e Over Time",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── C: Node availability heatmap ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    avail_matrix = piv_avail[valid_nodes].T.values
    # Resample to 150 columns for readability
    resample = max(1, len(steps) // 150)
    av_resamp = avail_matrix[:, ::resample]
    step_resamp = steps[::resample]
    im = ax.imshow(av_resamp, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1,
                   extent=[steps[0], steps[-1], -0.5, len(valid_nodes)-0.5])
    ax.set_yticks(range(len(valid_nodes)))
    ax.set_yticklabels([n.replace("_", "\n") for n in valid_nodes], fontsize=6.5)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_title("(C)  Node Availability Timeline  (green=online, red=offline)",
                 fontweight="bold", loc="left", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=0.08, color=c)
    ax.set_xlim(0, STEPS_PER_SIM)

    # ── D: Incoming traffic per node ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    _shade(ax)
    for nid in valid_nodes:
        ax.plot(steps, piv_in[nid].rolling(roll, min_periods=1).mean(),
                color=NODE_COLORS[nid], lw=0.85, alpha=0.85,
                label=nid.split("_")[0])
    ax.set_ylabel("Incoming Requests / step", fontsize=9)
    ax.set_title("(D)  Per-Node Incoming Traffic",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── E: Accepted vs Rejected stacked (all nodes) ───────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    _shade(ax)
    total_acc = piv_acc.sum(axis=1)
    total_rej = piv_rej.sum(axis=1)
    ax.stackplot(steps, total_acc, total_rej,
                 labels=["Accepted", "Rejected"],
                 colors=["#27ae60", "#e74c3c"], alpha=0.72)
    ax.set_ylabel("Requests / step (all nodes)", fontsize=9)
    ax.set_title("(E)  Total Accepted vs Rejected (All Nodes)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_xlim(0, STEPS_PER_SIM)

    # ── F: Rejection rate per node ────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    _shade(ax)
    for nid in valid_nodes:
        rr_smooth = (piv_rr[nid] * 100).rolling(roll, min_periods=1).mean()
        ax.plot(steps, rr_smooth,
                color=NODE_COLORS[nid], lw=0.85, alpha=0.85,
                label=nid.split("_")[0])
    ax.set_ylabel("Rejection Rate (%)", fontsize=9)
    ax.set_title("(F)  Per-Node Rejection Rate Over Time",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── G: phi(e) grouped bar by scenario ────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    x     = np.arange(len(SCENARIO_ORDER))
    n_nd  = len(valid_nodes)
    width = 0.13
    for k, nid in enumerate(valid_nodes):
        offset = (k - n_nd / 2 + 0.5) * width
        vals   = [piv_phi.loc[sc, nid] if sc in piv_phi.index else 0
                  for sc in SCENARIO_ORDER]
        ax.bar(x + offset, vals, width,
               color=NODE_COLORS[nid], alpha=0.78,
               label=nid.split("_")[0])
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_ORDER, fontsize=7, rotation=15)
    ax.set_ylabel("phi(e) Weighting Factor", fontsize=9)
    ax.set_title("(G)  phi(e) Distribution by Scenario x Node",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6, axis="y")

    # Phase legend
    phase_handles  = [Patch(facecolor=c, alpha=0.65, label=n)
                      for n, c in zip(SCENARIO_ORDER, SCENARIO_COLORS)]
    node_handles   = [Patch(facecolor=NODE_COLORS[nid], alpha=0.80,
                            label=nid.replace("_", " "))
                      for nid in valid_nodes]
    fig.legend(handles=phase_handles + node_handles,
               loc="lower center", ncol=6, fontsize=7.5,
               title="Phases and Edge Nodes",
               title_fontsize=8,
               bbox_to_anchor=(0.5, 0.002))

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved  ->  {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Supplementary figure (4 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_supplementary(df: pd.DataFrame,
                        out: str = "distributed_edge_supplementary.png") -> None:
    """
    4-panel supplementary analysis.

    S1  Load distribution heatmap (node x phase)
    S2  Gossip adjustment factor per node over time
    S3  MTD factor evolution for each node (active phase highlighted)
    S4  Rejection rate heatmap (node x scenario)
    """
    piv_load   = df.pivot_table(index="step",     columns="node_id",
                                values="load_e",   aggfunc="mean")
    piv_gossip = df.pivot_table(index="step",     columns="node_id",
                                values="gossip_adj", aggfunc="mean")
    piv_mtd    = df.pivot_table(index="step",     columns="node_id",
                                values="mtd_factor", aggfunc="mean")
    valid_nodes = [n for n in NODE_IDS if n in piv_load.columns]
    steps       = piv_load.index

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "Distributed & Edge Enforcement -- Supplementary Analysis  (Section IV.E)",
        fontsize=12, fontweight="bold",
    )

    # ── S1: Load heatmap (node x step) ────────────────────────────────────────
    ax = axes[0, 0]
    load_matrix = piv_load[valid_nodes].T.values
    resamp = max(1, len(steps) // 200)
    im = ax.imshow(load_matrix[:, ::resamp], aspect="auto",
                   cmap="RdYlGn_r", vmin=0, vmax=1,
                   extent=[steps[0], steps[-1], -0.5, len(valid_nodes)-0.5])
    ax.set_yticks(range(len(valid_nodes)))
    ax.set_yticklabels([n.replace("_", "\n") for n in valid_nodes], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Load Fraction")
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=0.08, color=c)
    for j, (n, c) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j+1]) / 2
        ax.text(mid, len(valid_nodes)-0.3, n, ha="center", va="top",
                fontsize=5.5, color=c, fontweight="bold")
    ax.set_xlim(0, STEPS_PER_SIM)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_title("(S1)  Node Load Heatmap Over Time\n"
                 "(dark red = high load, green = low/offline)",
                 fontweight="bold", loc="left", fontsize=9)

    # ── S2: Gossip adjustment per node ────────────────────────────────────────
    ax = axes[0, 1]
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=0.07, color=c)
    for nid in valid_nodes:
        ax.plot(steps, piv_gossip[nid], color=NODE_COLORS[nid],
                lw=0.8, alpha=0.80, label=nid.split("_")[0])
    ax.axhline(1.00, color="black", ls=":", lw=1.0, alpha=0.5, label="Hold (1.0)")
    ax.axhline(0.90, color="red",   ls="--", lw=0.8, alpha=0.5, label="Back-off (0.90)")
    ax.axhline(1.05, color="green", ls="--", lw=0.8, alpha=0.5, label="Ramp-up (1.05)")
    ax.set_ylim(0.82, 1.10)
    ax.set_xlim(0, STEPS_PER_SIM)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Gossip Adjustment Factor", fontsize=9)
    ax.set_title("(S2)  Gossip Protocol: Per-Node Adjustment Factors\n"
                 "(sync every 10 steps; based on neighbour load)",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── S3: MTD factor per node ───────────────────────────────────────────────
    ax = axes[1, 0]
    ax.axvspan(PHASE_BOUNDS[4], PHASE_BOUNDS[5], alpha=0.18, color="#e67e22",
               label="MTD Active Phase")
    for nid in valid_nodes:
        ax.plot(steps, piv_mtd[nid], color=NODE_COLORS[nid],
                lw=0.85, alpha=0.85, label=nid.split("_")[0])
    ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.5, label="Nominal (1.0)")
    ax.axhline(0.70, color="red",   ls=":", lw=0.8, alpha=0.5)
    ax.axhline(1.30, color="green", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlim(0, STEPS_PER_SIM)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("MTD Factor", fontsize=9)
    ax.set_title("(S3)  Moving Target Defense: Parameter Rotation\n"
                 "(rotates L_base by +/-30% every 50 steps during MTD phase)",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── S4: Rejection rate heatmap (node x scenario) ──────────────────────────
    ax = axes[1, 1]
    rej_heat = (
        df.groupby(["node_id", "scenario"])["rejection_rate"]
          .mean()
          .unstack("scenario")
          .reindex(valid_nodes)[SCENARIO_ORDER]
          .fillna(0) * 100
    )
    im2 = ax.imshow(rej_heat.values, aspect="auto",
                    cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(SCENARIO_ORDER, fontsize=7.5, rotation=18)
    ax.set_yticks(range(len(valid_nodes)))
    ax.set_yticklabels([n.replace("_", " ") for n in valid_nodes], fontsize=7.5)
    for i in range(len(valid_nodes)):
        for j in range(len(SCENARIO_ORDER)):
            val = rej_heat.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white" if val > 50 else "black",
                    fontweight="bold")
    plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.02, label="Avg Rejection %")
    ax.set_title("(S4)  Rejection Rate Heatmap  (Node x Scenario)",
                 fontweight="bold", loc="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved  ->  {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_FILE  = "distributed_edge_simulation.csv"
    PNG_MAIN  = "distributed_edge_main_results.png"
    PNG_SUPP  = "distributed_edge_supplementary.png"

    print("=" * 70)
    print("  Smart Grid -- Distributed & Edge Enforcement Rate Limiter")
    print("  Section IV.E: Distributed and Edge Enforcement")
    print("=" * 70)
    print(f"\n  Running {len(NODE_IDS)} nodes x {STEPS_PER_SIM} steps "
          f"= {len(NODE_IDS)*STEPS_PER_SIM:,} records ...")

    df = run_simulation(steps=STEPS_PER_SIM, seed=42)

    df.to_csv(CSV_FILE, index=False)
    print(f"  CSV saved    ->  {CSV_FILE}  ({len(df):,} rows, {len(df.columns)} columns)")

    print_summary(df)

    print("\n  Generating plots ...")
    plot_main_results(df, out=PNG_MAIN)
    plot_supplementary(df, out=PNG_SUPP)

    print("\n  Done.")
