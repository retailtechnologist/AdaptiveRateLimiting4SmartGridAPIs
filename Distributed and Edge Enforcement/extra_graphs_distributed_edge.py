#!/usr/bin/env python3
"""
extra_graphs_distributed_edge.py
==================================
Generates two additional standalone graphs for Section IV.E simulation:

    Graph 1 -- Per-Client (Node) Incoming Request Rate over time
    Graph 2 -- Overall Accepted vs Rejection Rate (time-series + per-scenario bar)

Reads from the existing CSV produced by simulation_distributed_edge.py.
Does NOT modify any other file.

Output
------
    extra_graphs_distributed_edge.png
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (must match simulation constants)
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILE  = "distributed_edge_simulation.csv"
OUT_FILE  = "extra_graphs_distributed_edge.png"
ROLL      = 20          # rolling-average window for smoothing

NODE_IDS = [
    "DER_Hub_North",
    "PMU_Collector_West",
    "AMI_Gateway_East",
    "EV_Station_South",
    "Smart_Meter_Cluster_1",
    "Smart_Meter_Cluster_2",
]

NODE_COLORS = {
    "DER_Hub_North"         : "#e74c3c",
    "PMU_Collector_West"    : "#9b59b6",
    "AMI_Gateway_East"      : "#3498db",
    "EV_Station_South"      : "#f39c12",
    "Smart_Meter_Cluster_1" : "#2ecc71",
    "Smart_Meter_Cluster_2" : "#1abc9c",
}

NODE_LABELS = {
    "DER_Hub_North"         : "DER Hub North",
    "PMU_Collector_West"    : "PMU Collector West",
    "AMI_Gateway_East"      : "AMI Gateway East",
    "EV_Station_South"      : "EV Station South",
    "Smart_Meter_Cluster_1" : "Smart Meter Cluster 1",
    "Smart_Meter_Cluster_2" : "Smart Meter Cluster 2",
}

SCENARIO_ORDER  = ["NORMAL", "LOAD_SPIKE", "DDOS_ATTACK",
                   "NODE_FAILURE", "MTD_ACTIVE", "RECOVERY"]
SCENARIO_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c",
                   "#8e44ad", "#e67e22", "#3498db"]
PHASE_BOUNDS    = [0, 250, 500, 750, 1000, 1250, 1500]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shade(ax: plt.Axes, alpha: float = 0.08) -> None:
    """Shade each operating phase on the given axis."""
    for j, color in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j + 1], alpha=alpha, color=color)
    ax.set_xlim(0, PHASE_BOUNDS[-1])


def _phase_labels(ax: plt.Axes, y_frac: float = 0.97) -> None:
    """Annotate phase names at the top of each shaded band."""
    ylo, yhi = ax.get_ylim()
    y = ylo + (yhi - ylo) * y_frac
    for j, (name, color) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j + 1]) / 2
        ax.text(mid, y, name, ha="center", va="top",
                fontsize=7, color=color, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def pivot_by_node(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a step x node_id pivot table for the given column."""
    return df.pivot_table(index="step", columns="node_id",
                          values=col, aggfunc="sum")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1  --  Per-Client (Node) Incoming Request Rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_incoming_request_rate(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Line chart showing the incoming request rate for each edge node over
    all 1500 simulation steps.

    A rolling average (window=ROLL) is overlaid on the raw step-level
    values to reduce noise while preserving trend visibility.
    Nodes that go offline (availability=0) naturally show zero requests.
    """
    piv = pivot_by_node(df, "incoming_req")
    steps = piv.index
    valid = [n for n in NODE_IDS if n in piv.columns]

    _shade(ax)

    for nid in valid:
        raw    = piv[nid].fillna(0)
        smooth = raw.rolling(ROLL, min_periods=1).mean()

        # Faint raw line
        ax.plot(steps, raw, color=NODE_COLORS[nid],
                lw=0.4, alpha=0.25)
        # Bold smoothed line
        ax.plot(steps, smooth, color=NODE_COLORS[nid],
                lw=1.6, alpha=0.95, label=NODE_LABELS[nid])

    # Annotate notable events
    ax.annotate("EV & AMI surge",
                xy=(375, 1550), xytext=(430, 1700),
                fontsize=7.5, color="#f39c12", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.0))
    ax.annotate("DDoS flood\n(leaf nodes)",
                xy=(620, 1500), xytext=(560, 1750),
                fontsize=7.5, color="#2ecc71", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.0))
    ax.annotate("SM1 offline",
                xy=(875, 0), xytext=(820, 300),
                fontsize=7.5, color="#8e44ad", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.0))

    ax.set_xlabel("Simulation Step", fontsize=10)
    ax.set_ylabel("Incoming Requests per Step", fontsize=10)
    ax.set_title(
        "Graph 1  --  Per-Client (Node) Incoming Request Rate\n"
        "Faint lines = raw per-step values  |  Bold lines = "
        f"{ROLL}-step rolling average",
        fontweight="bold", loc="left", fontsize=11,
    )
    ax.legend(
        fontsize=8.5, loc="upper right", ncol=2,
        framealpha=0.92, edgecolor="#cccccc",
    )
    ax.grid(True, alpha=0.30, lw=0.6)
    _phase_labels(ax)

    # Per-node summary annotation in lower-left corner
    summary_lines = []
    for nid in valid:
        peak = int(piv[nid].fillna(0).max())
        avg  = piv[nid].fillna(0).mean()
        summary_lines.append(f"{NODE_LABELS[nid]:<28}: peak={peak:>5}  avg={avg:>6.1f}")
    summary_text = "Per-node stats (all steps):\n" + "\n".join(summary_lines)
    ax.text(0.01, 0.03, summary_text, transform=ax.transAxes,
            fontsize=6.8, family="monospace", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.80, edgecolor="#cccccc"))


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2  --  Overall Accepted vs Rejection Rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_accepted_vs_rejection(df: pd.DataFrame,
                                ax_ts: plt.Axes,
                                ax_bar: plt.Axes) -> None:
    """
    Two-part panel:

    Left  (ax_ts)  -- Time-series stacked area of total accepted / rejected
                       requests across all nodes, with per-node rejection rate
                       lines overlaid on the right twin axis.

    Right (ax_bar) -- Per-scenario grouped bar chart comparing total accepted
                      vs total rejected (absolute counts), with rejection rate
                      percentage annotated above each rejected bar.
    """
    piv_acc = pivot_by_node(df, "accepted_req")
    piv_rej = pivot_by_node(df, "rejected_req")
    piv_in  = pivot_by_node(df, "incoming_req")
    valid   = [n for n in NODE_IDS if n in piv_acc.columns]
    steps   = piv_acc.index

    total_acc = piv_acc[valid].sum(axis=1)
    total_rej = piv_rej[valid].sum(axis=1)
    total_in  = piv_in[valid].sum(axis=1).clip(lower=1)
    rej_pct   = (total_rej / total_in * 100).rolling(ROLL, min_periods=1).mean()

    # ── Left: time-series stacked area ─────────────────────────────────────
    _shade(ax_ts)
    ax_ts.stackplot(steps, total_acc, total_rej,
                    labels=["Accepted (all nodes)", "Rejected (all nodes)"],
                    colors=["#27ae60", "#e74c3c"], alpha=0.72)

    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(steps, rej_pct,
                color="navy", lw=1.4, ls="--",
                label=f"Overall Rejection Rate % (roll-{ROLL})")
    ax_ts2.set_ylabel("Rejection Rate (%)", fontsize=9, color="navy")
    ax_ts2.tick_params(axis="y", colors="navy")
    ax_ts2.set_ylim(0, 110)

    # Per-node rejection rate (thin lines on right axis)
    piv_rr = df.pivot_table(index="step", columns="node_id",
                             values="rejection_rate", aggfunc="mean")
    for nid in valid:
        rr = (piv_rr[nid] * 100).rolling(ROLL, min_periods=1).mean()
        ax_ts2.plot(steps, rr, color=NODE_COLORS[nid],
                    lw=0.75, alpha=0.55, ls=":")

    ax_ts.set_xlabel("Simulation Step", fontsize=10)
    ax_ts.set_ylabel("Requests per Step (all nodes)", fontsize=10)
    ax_ts.set_title(
        "Graph 2a  --  Overall Accepted vs Rejected Traffic (Time-Series)\n"
        "Stacked area = absolute counts  |  Dashed navy = overall rejection %  "
        "|  Dotted = per-node rejection %",
        fontweight="bold", loc="left", fontsize=10,
    )
    handles_ts,  labels_ts  = ax_ts.get_legend_handles_labels()
    handles_ts2, labels_ts2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(handles_ts + handles_ts2, labels_ts + labels_ts2,
                 fontsize=8, loc="upper left", framealpha=0.90)
    ax_ts.grid(True, alpha=0.30, lw=0.6)
    _phase_labels(ax_ts)

    # ── Right: per-scenario grouped bar ────────────────────────────────────
    sc_grp = (
        df.groupby("scenario")
          .agg(Total_Accept=("accepted_req", "sum"),
               Total_Reject=("rejected_req", "sum"),
               Total_In    =("incoming_req", "sum"))
          .reindex(SCENARIO_ORDER)
    )
    sc_grp["Rej_pct"] = (sc_grp["Total_Reject"] /
                          sc_grp["Total_In"].clip(lower=1) * 100).round(1)

    x     = np.arange(len(SCENARIO_ORDER))
    width = 0.38

    bars_acc = ax_bar.bar(x - width / 2, sc_grp["Total_Accept"] / 1000,
                          width, color="#27ae60", alpha=0.82, label="Accepted (k req)")
    bars_rej = ax_bar.bar(x + width / 2, sc_grp["Total_Reject"] / 1000,
                          width, color="#e74c3c", alpha=0.82, label="Rejected (k req)")

    # Annotate rejection % above rejected bars
    for bar, pct in zip(bars_rej, sc_grp["Rej_pct"]):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#c0392b")

    # Annotate accepted count above accepted bars
    for bar, val in zip(bars_acc, sc_grp["Total_Accept"]):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{val/1000:.0f}k", ha="center", va="bottom",
                    fontsize=7.5, color="#1a7a40")

    # Color x-tick labels by scenario
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(SCENARIO_ORDER, fontsize=8.5, rotation=14)
    for tick, color in zip(ax_bar.get_xticklabels(), SCENARIO_COLORS):
        tick.set_color(color)
        tick.set_fontweight("bold")

    ax_bar.set_ylabel("Requests (thousands)", fontsize=10)
    ax_bar.set_title(
        "Graph 2b  --  Accepted vs Rejected by Scenario\n"
        "(% annotations = rejection rate; label above accepted bars = accepted count)",
        fontweight="bold", loc="left", fontsize=10,
    )
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.grid(True, alpha=0.30, lw=0.6, axis="y")

    # Overall stats text box
    total_a = int(df["accepted_req"].sum())
    total_r = int(df["rejected_req"].sum())
    total_i = int(df["incoming_req"].sum())
    overall_text = (
        f"Overall  (all phases, all nodes)\n"
        f"Incoming : {total_i:>10,}\n"
        f"Accepted : {total_a:>10,}  ({total_a/total_i*100:.1f}%)\n"
        f"Rejected : {total_r:>10,}  ({total_r/total_i*100:.1f}%)"
    )
    ax_bar.text(0.02, 0.97, overall_text, transform=ax_bar.transAxes,
                fontsize=8.5, family="monospace", va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          alpha=0.90, edgecolor="#999999"))


# ─────────────────────────────────────────────────────────────────────────────
# Main figure assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(df: pd.DataFrame, out_path: str) -> None:
    """
    Assemble a three-panel figure:
        Row 0  (full width)  --  Graph 1: Per-Client Incoming Request Rate
        Row 1  left          --  Graph 2a: Time-series accepted vs rejected
        Row 1  right         --  Graph 2b: Per-scenario bar chart
    """
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#f4f6f8")

    fig.suptitle(
        "Distributed & Edge Enforcement  --  Additional Graphs  (Section IV.E)\n"
        "Graph 1: Per-Client Incoming Request Rate     "
        "Graph 2: Overall Accepted vs Rejection Rate",
        fontsize=13, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.46, wspace=0.28,
        top=0.960, bottom=0.06,
        left=0.07, right=0.97,
    )

    # Row 0: Graph 1 spans both columns
    ax_graph1 = fig.add_subplot(gs[0, :])
    plot_incoming_request_rate(df, ax_graph1)

    # Row 1: Graph 2 split into time-series (left) and bar chart (right)
    ax_ts  = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    plot_accepted_vs_rejection(df, ax_ts, ax_bar)

    # Shared phase legend at the bottom
    phase_handles = [
        Patch(facecolor=c, alpha=0.65, label=n)
        for n, c in zip(SCENARIO_ORDER, SCENARIO_COLORS)
    ]
    node_handles = [
        Patch(facecolor=NODE_COLORS[nid], alpha=0.85, label=NODE_LABELS[nid])
        for nid in NODE_IDS
    ]
    fig.legend(
        handles=phase_handles + node_handles,
        loc="lower center", ncol=6, fontsize=8,
        title="Operating Phases (shading)  |  Edge Nodes (colored lines)",
        title_fontsize=8.5,
        framealpha=0.92, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.002),
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved  ->  {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Extra Graphs -- Distributed & Edge Enforcement")
    print("  Section IV.E: Additional standalone visualisations")
    print("=" * 60)
    print(f"\n  Loading data from  {CSV_FILE} ...")

    df = load_data(CSV_FILE)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")

    print("  Building figure ...")
    build_figure(df, OUT_FILE)

    print("\n  Done.")
