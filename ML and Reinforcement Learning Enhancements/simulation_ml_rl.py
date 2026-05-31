#!/usr/bin/env python3
"""
simulation_ml_rl.py
====================
Simulation harness for Section IV.D:
Machine Learning and Reinforcement Learning Enhancements.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh

Structure  --  8 500 time-step simulation
------------------------------------------
Phase 1  NORMAL        steps 0-1699     Baseline grid operation
Phase 2  PEAK_LOAD     steps 1700-3399  EV surge + renewable variability
Phase 3  DDOS_ATTACK   steps 3400-5099  Sustained DDoS flood
Phase 4  GRID_EMERGENCY steps 5100-6799  Frequency deviation / blackout risk
Phase 5  RECOVERY      steps 6800-8499  Stabilisation after crisis

Agent training schedule
------------------------
Steps   0-2999  :  Training mode (epsilon decays 1.0 -> ~0.10)
Steps 3000-8499 :  Evaluation mode (epsilon -> 0.05, near-greedy)

Baseline comparison
-------------------
A static rate limiter fixed at 500 req/step is run in parallel
to demonstrate the adaptive advantage of the DRL agent.

Outputs
-------
    ml_rl_simulation.csv               8 500-row dataset (31 columns)
    ml_rl_main_results.png             8-panel main diagnostic chart
    ml_rl_training_analysis.png        4-panel training / learning curve chart
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

from ml_rl_rate_limiter import DQNAgent, SmartGridMDP, LoadPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STEPS           = 8500
STATIC_LIMIT    = 500      # fixed baseline rate limit (req/step)
TRAIN_STEPS     = 3000     # agent trains during first TRAIN_STEPS steps

SCENARIO_ORDER  = ["NORMAL", "PEAK_LOAD", "DDOS_ATTACK", "GRID_EMERGENCY", "RECOVERY"]
SCENARIO_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6", "#3498db"]
PHASE_BOUNDS    = [0, 1700, 3400, 5100, 6800, 8500]

ACTION_LIMITS   = SmartGridMDP.RATE_LIMITS     # [100, 250, 500, 750, 1000, 1500, 2000, 3000]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Phase / Scenario helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_scenario(step: int) -> str:
    if step < 1700:  return "NORMAL"
    if step < 3400:  return "PEAK_LOAD"
    if step < 5100:  return "DDOS_ATTACK"
    if step < 6800:  return "GRID_EMERGENCY"
    return "RECOVERY"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Traffic and state generators
# ─────────────────────────────────────────────────────────────────────────────

# Per-phase parameter tables
_PHASE_PARAMS = {
    "NORMAL"        : dict(legit_mu=250,  legit_std=40,  atk_mu=0,    atk_std=0,
                           cpu_base=0.30, lat_base=0.20, threat_mu=0.05, threat_std=0.03,
                           freq_mu=0.04,  freq_std=0.02, ren_mu=0.20, ren_std=0.05,
                           ev_mu=0.15,    ev_std=0.04),
    "PEAK_LOAD"     : dict(legit_mu=900,  legit_std=120, atk_mu=30,   atk_std=20,
                           cpu_base=0.55, lat_base=0.40, threat_mu=0.15, threat_std=0.05,
                           freq_mu=0.12,  freq_std=0.04, ren_mu=0.55, ren_std=0.10,
                           ev_mu=0.70,    ev_std=0.10),
    "DDOS_ATTACK"   : dict(legit_mu=300,  legit_std=50,  atk_mu=3800, atk_std=400,
                           cpu_base=0.85, lat_base=0.75, threat_mu=0.90, threat_std=0.05,
                           freq_mu=0.08,  freq_std=0.03, ren_mu=0.35, ren_std=0.08,
                           ev_mu=0.30,    ev_std=0.07),
    "GRID_EMERGENCY": dict(legit_mu=600,  legit_std=80,  atk_mu=500,  atk_std=100,
                           cpu_base=0.70, lat_base=0.60, threat_mu=0.60, threat_std=0.10,
                           freq_mu=0.38,  freq_std=0.06, ren_mu=0.72, ren_std=0.10,
                           ev_mu=0.50,    ev_std=0.08),
    "RECOVERY"      : dict(legit_mu=400,  legit_std=60,  atk_mu=200,  atk_std=80,
                           cpu_base=0.45, lat_base=0.35, threat_mu=0.25, threat_std=0.08,
                           freq_mu=0.14,  freq_std=0.04, ren_mu=0.40, ren_std=0.08,
                           ev_mu=0.35,    ev_std=0.06),
}

MAX_LEGIT  = 1500.0
MAX_ATK    = 5000.0
MAX_FREQ   = 0.5


def generate_step_data(step: int, rng) -> dict:
    """
    Generate raw environment metrics for one simulation step.

    Returns a dict with all raw and normalised features plus traffic counts.
    """
    sc  = get_scenario(step)
    p   = _PHASE_PARAMS[sc]
    tof = (step % 1440) / 1440.0       # time-of-day (24-hr cycle at 1 step/min)

    # Traffic
    legit = int(np.clip(rng.normal(p["legit_mu"], p["legit_std"]), 1, MAX_LEGIT))
    atk   = int(np.clip(rng.normal(p["atk_mu"],   p["atk_std"]),  0, MAX_ATK))
    total = legit + atk

    # Grid / system metrics
    threat  = float(np.clip(rng.normal(p["threat_mu"], p["threat_std"]), 0, 1))
    freq_d  = float(np.clip(rng.normal(p["freq_mu"],   p["freq_std"]),   0, MAX_FREQ))
    ren     = float(np.clip(rng.normal(p["ren_mu"],    p["ren_std"]),     0, 1))
    ev      = float(np.clip(rng.normal(p["ev_mu"],     p["ev_std"]),      0, 1))

    # System load (depends on total traffic trying to enter)
    traffic_pressure = min(total / 2000.0, 1.0)
    cpu     = float(np.clip(p["cpu_base"] + 0.40 * traffic_pressure
                            + rng.normal(0, 0.03), 0, 1))
    mem     = float(np.clip(0.25 + 0.50 * cpu + rng.normal(0, 0.03), 0, 1))
    queue   = float(np.clip(total / 5000.0, 0, 1))
    lat     = float(np.clip(p["lat_base"] + 0.50 * cpu
                            + rng.normal(0, 0.02), 0, 1))
    req_r   = float(np.clip(total / (MAX_LEGIT + MAX_ATK), 0, 1))

    return dict(
        scenario=sc, legit=legit, atk=atk, total=total,
        cpu=cpu, mem=mem, queue=queue, lat=lat, req_r=req_r,
        threat=threat, freq_d=freq_d, ren=ren, ev=ev, tof=tof,
    )


def build_state(d: dict, predicted_load: float) -> np.ndarray:
    """Pack raw metrics into the normalised 10-D state vector."""
    return np.array([
        d["cpu"], d["mem"], d["queue"], d["lat"], d["req_r"],
        d["threat"], d["freq_d"] / MAX_FREQ, d["ren"],
        d["ev"], predicted_load,
    ], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Post-action system update (feedback model)
# ─────────────────────────────────────────────────────────────────────────────

def apply_rate_limit(
    rate_limit: int,
    legit: int,
    atk:   int,
) -> tuple:
    """
    Apply rate limit to incoming traffic.
    Legitimate traffic is served first (priority model).
    Returns (accepted_total, accepted_legit, rejected_legit, rejected_atk).
    """
    total            = legit + atk
    accepted_legit   = int(min(legit, rate_limit))
    remaining_budget = max(0, rate_limit - accepted_legit)
    accepted_atk     = int(min(atk, remaining_budget))
    rejected_legit   = legit  - accepted_legit
    rejected_atk     = atk    - accepted_atk
    return (accepted_legit + accepted_atk, accepted_legit,
            rejected_legit, rejected_atk)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(steps: int = STEPS, seed: int = 42) -> pd.DataFrame:
    """
    Run the DRL rate-limiting simulation.

    Returns a pd.DataFrame with 31 columns and `steps` rows.
    """
    rng       = np.random.default_rng(seed)
    agent     = DQNAgent(seed=seed)
    predictor = LoadPredictor(lr=0.01)
    mdp       = SmartGridMDP()
    base_ts   = datetime(2024, 1, 1, 0, 0, 0)
    rows: list = []

    # Initialise state
    d0             = generate_step_data(0, rng)
    prev_pred_load = 0.5
    prev_state     = build_state(d0, prev_pred_load)

    for step in range(steps):
        ts = base_ts + timedelta(seconds=step)

        # -- Generate environment data for this step ----------------------
        d   = generate_step_data(step, rng)

        # -- ML predictor: predict load for next step ---------------------
        pred_load = predictor.predict(
            d["ren"], d["ev"], d["tof"], d["req_r"]
        )
        state = build_state(d, pred_load)

        # -- RL agent selects action --------------------------------------
        rl_action_idx = agent.select_action(state)
        rl_limit      = mdp.action_to_limit(rl_action_idx)
        q_vals        = agent.get_q_values(state)

        # -- Apply rate limits --------------------------------------------
        rl_acc, rl_acc_l, rl_rej_l, rl_rej_a = apply_rate_limit(
            rl_limit, d["legit"], d["atk"]
        )
        st_acc, st_acc_l, st_rej_l, st_rej_a = apply_rate_limit(
            STATIC_LIMIT, d["legit"], d["atk"]
        )

        rl_rej_total = rl_rej_l + rl_rej_a
        rl_rej_rate  = rl_rej_total / max(d["total"], 1)
        st_rej_rate  = (st_rej_l + st_rej_a) / max(d["total"], 1)

        # -- Compute RL reward -------------------------------------------
        reward, comps = mdp.compute_reward(
            action_idx      = rl_action_idx,
            legit_incoming  = d["legit"],
            total_incoming  = d["total"],
            cpu_util        = d["cpu"],
            latency_norm    = d["lat"],
            threat_level    = d["threat"],
        )

        # -- Static baseline reward (for comparison) ----------------------
        _, st_comps = mdp.compute_reward(
            action_idx     = mdp.RATE_LIMITS.index(STATIC_LIMIT),
            legit_incoming = d["legit"],
            total_incoming = d["total"],
            cpu_util       = d["cpu"],
            latency_norm   = d["lat"],
            threat_level   = d["threat"],
        )
        static_reward = sum(st_comps.values())

        # -- Next state (uses same-step data; next step generates new) ----
        next_pred = predictor.predict(d["ren"], d["ev"], d["tof"], d["req_r"])
        next_state = build_state(d, next_pred)

        # -- Store experience and update agent ----------------------------
        agent.store(state, rl_action_idx, reward, next_state, False)
        loss = agent.update()

        # -- Update load predictor with actual load -----------------------
        actual_load = d["cpu"] * 0.5 + d["req_r"] * 0.5
        predictor.update(d["ren"], d["ev"], d["tof"], d["req_r"], actual_load)

        # -- Record row ---------------------------------------------------
        rows.append({
            "timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "step"            : step,
            "scenario"        : d["scenario"],
            "mode"            : "Training" if step < TRAIN_STEPS else "Evaluation",
            # State features
            "cpu_util"        : round(d["cpu"],     3),
            "memory_util"     : round(d["mem"],     3),
            "queue_norm"      : round(d["queue"],   3),
            "latency_norm"    : round(d["lat"],     3),
            "req_rate_norm"   : round(d["req_r"],   3),
            "threat_level"    : round(d["threat"],  3),
            "freq_dev_norm"   : round(d["freq_d"] / MAX_FREQ, 3),
            "renewable_norm"  : round(d["ren"],     3),
            "ev_demand_norm"  : round(d["ev"],      3),
            "predicted_load"  : round(pred_load,    3),
            # Traffic
            "legit_incoming"  : d["legit"],
            "atk_incoming"    : d["atk"],
            "total_incoming"  : d["total"],
            # RL agent
            "rl_action_idx"   : rl_action_idx,
            "rl_rate_limit"   : rl_limit,
            "rl_accepted"     : rl_acc,
            "rl_rejected"     : rl_rej_total,
            "rl_rej_legit"    : rl_rej_l,
            "rl_rejection_rate": round(rl_rej_rate, 4),
            "rl_reward"       : round(reward,        4),
            "rl_legit_term"   : comps["legit_term"],
            "rl_overload_term": comps["overload_term"],
            "rl_latency_term" : comps["latency_term"],
            "rl_security_term": comps["security_term"],
            "q_max"           : round(float(q_vals.max()), 4),
            "epsilon"         : round(agent.epsilon,  4),
            "train_loss"      : round(loss, 6) if loss is not None else None,
            # Static baseline
            "static_rate_limit": STATIC_LIMIT,
            "static_accepted"  : st_acc,
            "static_rejected"  : st_rej_l + st_rej_a,
            "static_rej_legit" : st_rej_l,
            "static_rej_rate"  : round(st_rej_rate, 4),
            "static_reward"    : round(static_reward, 4),
        })

        prev_state = next_state

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> tuple:
    """Print three tables: scenario, mode (train/eval), and action distribution."""
    LINE  = "=" * 130
    LINE2 = "-" * 130

    print(f"\n{LINE}")
    print("  DRL ADAPTIVE RATE LIMITER -- SIMULATION RESULTS  (Section IV.D)")
    print("  Paper: R(s,a) = w1*(1-reject_legit) - w2*overload - w3*latency + w4*security")
    print(f"  8500-step simulation  |  DQN agent vs Static baseline ({STATIC_LIMIT} req/step)")
    print(LINE)

    # ── Table 1: per-scenario ────────────────────────────────────────────────
    sc = (
        df.groupby("scenario")
          .agg(
              Steps            = ("step",            "count"),
              Avg_Total_In     = ("total_incoming",  "mean"),
              Avg_Threat       = ("threat_level",    "mean"),
              Avg_RL_Limit     = ("rl_rate_limit",   "mean"),
              Avg_RL_Reward    = ("rl_reward",       "mean"),
              Avg_Static_Reward= ("static_reward",   "mean"),
              RL_Rej_Legit_pct = ("rl_rej_legit",   "sum"),
              ST_Rej_Legit_pct = ("static_rej_legit","sum"),
              Avg_RL_Rej_pct   = ("rl_rejection_rate","mean"),
              Avg_ST_Rej_pct   = ("static_rej_rate", "mean"),
          )
          .round(2)
          .reindex(SCENARIO_ORDER)
    )
    sc["RL_Rej_L%"] = (sc["RL_Rej_Legit_pct"] / df.groupby("scenario")["legit_incoming"].sum() * 100).round(1)
    sc["ST_Rej_L%"] = (sc["ST_Rej_Legit_pct"] / df.groupby("scenario")["legit_incoming"].sum() * 100).round(1)
    sc.drop(columns=["RL_Rej_Legit_pct", "ST_Rej_Legit_pct"], inplace=True)
    sc["Avg_RL_Rej_pct"]  = (sc["Avg_RL_Rej_pct"]  * 100).round(2)
    sc["Avg_ST_Rej_pct"]  = (sc["Avg_ST_Rej_pct"]  * 100).round(2)

    print("\n  [TABLE 1]  Per-Scenario Summary  (RL Agent vs Static Baseline)")
    print(LINE2)
    print(sc.to_string())

    # ── Table 2: training mode vs evaluation mode ────────────────────────────
    mode = (
        df.groupby("mode")
          .agg(
              Steps             = ("step",             "count"),
              Avg_RL_Reward     = ("rl_reward",        "mean"),
              Avg_Static_Reward = ("static_reward",    "mean"),
              Avg_RL_Limit      = ("rl_rate_limit",    "mean"),
              Avg_RL_Rej_pct    = ("rl_rejection_rate","mean"),
              Avg_ST_Rej_pct    = ("static_rej_rate",  "mean"),
              Avg_Epsilon       = ("epsilon",           "mean"),
              Avg_Loss          = ("train_loss",        "mean"),
          )
          .round(4)
    )
    mode["Avg_RL_Rej_pct"] = (mode["Avg_RL_Rej_pct"] * 100).round(2)
    mode["Avg_ST_Rej_pct"] = (mode["Avg_ST_Rej_pct"] * 100).round(2)

    print(f"\n  [TABLE 2]  Training Mode vs Evaluation Mode")
    print(LINE2)
    print(mode.to_string())

    # ── Table 3: action distribution per scenario ────────────────────────────
    action_labels = [f"L{v}" for v in ACTION_LIMITS]
    act_dist = (
        df.groupby(["scenario", "rl_action_idx"])["step"]
          .count()
          .unstack("rl_action_idx")
          .fillna(0)
          .astype(int)
          .reindex(SCENARIO_ORDER)
    )
    act_dist.columns = [f"A{i}({ACTION_LIMITS[i]})" for i in range(len(ACTION_LIMITS))]

    print(f"\n  [TABLE 3]  RL Action Distribution by Scenario (count of steps per action)")
    print(LINE2)
    print(act_dist.to_string())
    print(LINE2)

    # ── Overall ─────────────────────────────────────────────────────────────
    rl_cum   = df["rl_reward"].sum()
    st_cum   = df["static_reward"].sum()
    rl_rl    = df["rl_rej_legit"].sum()
    st_rl    = df["static_rej_legit"].sum()
    total_l  = df["legit_incoming"].sum()
    print(
        f"\n  OVERALL  Steps: {len(df):,}  |  "
        f"RL Cumulative Reward: {rl_cum:.1f}  "
        f"Static Cumulative Reward: {st_cum:.1f}  "
        f"Advantage: {rl_cum - st_cum:+.1f}"
    )
    print(
        f"           RL Legit Rejection: {rl_rl:,} / {total_l:,} "
        f"({rl_rl/total_l*100:.2f}%)  "
        f"Static Legit Rejection: {st_rl:,} / {total_l:,} "
        f"({st_rl/total_l*100:.2f}%)"
    )
    print(LINE)
    return sc, mode, act_dist


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shade(ax, alpha=0.07):
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=alpha, color=c)
    ax.set_xlim(0, STEPS)


def _phase_labels(ax, y_frac=0.96):
    ylo, yhi = ax.get_ylim()
    y = ylo + (yhi - ylo) * y_frac
    for j, (n, c) in enumerate(zip(SCENARIO_ORDER, SCENARIO_COLORS)):
        mid = (PHASE_BOUNDS[j] + PHASE_BOUNDS[j+1]) / 2
        ax.text(mid, y, n, ha="center", va="top",
                fontsize=6, color=c, fontweight="bold")


def _train_eval_vline(ax):
    ax.axvline(TRAIN_STEPS, color="navy", ls="--", lw=1.2, alpha=0.70,
               label=f"Train->Eval boundary (step {TRAIN_STEPS})")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main 8-panel results figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_results(df: pd.DataFrame, out: str = "ml_rl_main_results.png") -> None:
    """
    8-panel main diagnostic figure.

    A  RL vs Static rate limit over time (full width)
    B  Reward: RL vs Static over time
    C  Incoming traffic (legit vs attack) + threat level
    D  Accepted vs Rejected (RL) stacked area
    E  Cumulative reward: RL vs Static
    F  Rejection rate of legit traffic: RL vs Static
    G  Avg rate limit per scenario per mode (grouped bar)
    H  Reward components over time (RL)
    """
    roll = 50   # rolling window for smoothing
    s    = df["step"]

    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "Smart Grid -- DRL Adaptive Rate Limiter  (Section IV.D)\n"
        "R(s,a) = w1*(1-reject_legit) - w2*overload - w3*latency + w4*security\n"
        f"[8500-step simulation | DQN agent vs Static {STATIC_LIMIT} req/step | "
        f"Training: steps 0-{TRAIN_STEPS}  Evaluation: steps {TRAIN_STEPS}-8499]",
        fontsize=11, fontweight="bold", y=0.997,
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.50, wspace=0.30,
                           top=0.963, bottom=0.04,
                           left=0.07, right=0.97)

    # ── A: Rate limit comparison (full width) ─────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    _shade(ax)
    ax.plot(s, df["rl_rate_limit"].rolling(roll, min_periods=1).mean(),
            color="#e74c3c", lw=1.3, label=f"RL Rate Limit (roll-{roll})")
    ax.axhline(STATIC_LIMIT, color="navy", ls="--", lw=1.5,
               label=f"Static Baseline = {STATIC_LIMIT} req/step")
    _train_eval_vline(ax)
    ax.set_ylabel("Rate Limit (req/step)", fontsize=9)
    ax.set_title("(A)  RL Adaptive Rate Limit vs Static Baseline",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.grid(True, alpha=0.3, lw=0.6)
    _phase_labels(ax)

    # ── B: Reward comparison ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _shade(ax)
    ax.plot(s, df["rl_reward"].rolling(roll, min_periods=1).mean(),
            color="#e74c3c", lw=0.9, label="RL Reward")
    ax.plot(s, df["static_reward"].rolling(roll, min_periods=1).mean(),
            color="navy", lw=0.9, ls="--", label="Static Reward")
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.6)
    _train_eval_vline(ax)
    ax.set_ylabel("Step Reward (smoothed)", fontsize=9)
    ax.set_title("(B)  Step Reward: RL vs Static", fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── C: Traffic + threat ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _shade(ax)
    ax2 = ax.twinx()
    ax.fill_between(s, df["legit_incoming"], alpha=0.40, color="#27ae60", label="Legit Traffic")
    ax.fill_between(s, df["atk_incoming"],  alpha=0.35, color="#e74c3c", label="Attack Traffic")
    ax2.plot(s, df["threat_level"].rolling(roll, min_periods=1).mean(),
             color="#8e44ad", lw=1.0, label="Threat Level")
    ax.set_ylabel("Requests / step", fontsize=9)
    ax2.set_ylabel("Threat Level", color="#8e44ad", fontsize=9)
    ax.set_title("(C)  Incoming Traffic + Threat Level",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(loc="upper left",  fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_xlim(0, STEPS)

    # ── D: RL accepted vs rejected ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    _shade(ax)
    ax.stackplot(s, df["rl_accepted"], df["rl_rejected"],
                 labels=["RL Accepted", "RL Rejected"],
                 colors=["#27ae60", "#e74c3c"], alpha=0.72)
    ax.set_ylabel("Requests / step", fontsize=9)
    ax.set_title("(D)  RL Agent: Accepted vs Rejected Traffic",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── E: Cumulative reward ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    _shade(ax)
    ax.plot(s, df["rl_reward"].cumsum(),     color="#e74c3c", lw=1.3, label="RL Cumulative")
    ax.plot(s, df["static_reward"].cumsum(), color="navy",    lw=1.3, ls="--",
            label="Static Cumulative")
    _train_eval_vline(ax)
    ax.set_ylabel("Cumulative Reward", fontsize=9)
    ax.set_title("(E)  Cumulative Reward: RL vs Static",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_xlim(0, STEPS)

    # ── F: Legit rejection rate comparison ────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    _shade(ax)
    rl_lr = (df["rl_rej_legit"] / df["legit_incoming"].clip(lower=1) * 100)
    st_lr = (df["static_rej_legit"] / df["legit_incoming"].clip(lower=1) * 100)
    ax.plot(s, rl_lr.rolling(roll, min_periods=1).mean(),
            color="#e74c3c", lw=0.9, label="RL Legit Rejection %")
    ax.plot(s, st_lr.rolling(roll, min_periods=1).mean(),
            color="navy", lw=0.9, ls="--", label="Static Legit Rejection %")
    _train_eval_vline(ax)
    ax.set_ylabel("Legit Rejection Rate (%)", fontsize=9)
    ax.set_title("(F)  Legitimate Traffic Rejection: RL vs Static",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── G: Action distribution heatmap ────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    heat = np.zeros((len(SCENARIO_ORDER), len(ACTION_LIMITS)))
    for i, sc in enumerate(SCENARIO_ORDER):
        sub = df[df["scenario"] == sc]["rl_action_idx"].values
        for a in range(len(ACTION_LIMITS)):
            heat[i, a] = (sub == a).sum()
        total = heat[i].sum()
        if total > 0:
            heat[i] /= total
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(ACTION_LIMITS)))
    ax.set_xticklabels([str(v) for v in ACTION_LIMITS], fontsize=8, rotation=30)
    ax.set_yticks(range(len(SCENARIO_ORDER)))
    ax.set_yticklabels(SCENARIO_ORDER, fontsize=8)
    for i in range(len(SCENARIO_ORDER)):
        for j in range(len(ACTION_LIMITS)):
            ax.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if heat[i,j] < 0.5 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Action Frequency")
    ax.set_xlabel("Rate Limit (req/step)", fontsize=9)
    ax.set_title("(G)  RL Action Distribution Heatmap by Scenario",
                 fontweight="bold", loc="left", fontsize=10)

    # Phase legend
    phase_handles = [Patch(facecolor=c, alpha=0.65, label=n)
                     for n, c in zip(SCENARIO_ORDER, SCENARIO_COLORS)]
    fig.legend(handles=phase_handles, loc="lower center", ncol=5,
               fontsize=8, title="Operating Phases", title_fontsize=8,
               bbox_to_anchor=(0.5, 0.002))

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved  ->  {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Training analysis figure (4 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_analysis(df: pd.DataFrame, out: str = "ml_rl_training_analysis.png") -> None:
    """
    4-panel training / learning analysis.

    T1  Training loss over time (smoothed)
    T2  Epsilon decay schedule
    T3  Reward components breakdown (RL agent, full run)
    T4  Q-value (max) evolution over time
    """
    roll = 100
    s    = df["step"]
    train_df = df[df["mode"] == "Training"]
    eval_df  = df[df["mode"] == "Evaluation"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#f4f6f8")
    fig.suptitle(
        "DRL Rate Limiter -- Training Analysis  (Section IV.D)\n"
        "DQN: 2-layer neural network | Double-DQN targets | Experience Replay",
        fontsize=12, fontweight="bold",
    )

    # ── T1: Training loss ────────────────────────────────────────────────────
    ax = axes[0, 0]
    loss_series = df["train_loss"].dropna()
    loss_smooth = loss_series.rolling(roll, min_periods=1).mean()
    ax.plot(loss_series.index, loss_series.values,  color="#e74c3c", alpha=0.20, lw=0.5)
    ax.plot(loss_smooth.index, loss_smooth.values,  color="#c0392b", lw=1.5,
            label=f"Loss (roll-{roll})")
    ax.axvline(TRAIN_STEPS, color="navy", ls="--", lw=1.2,
               label=f"Step {TRAIN_STEPS}: eval mode")
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=0.06, color=c)
    ax.set_xlim(0, STEPS)
    ax.set_xlabel("Training Step", fontsize=9)
    ax.set_ylabel("MSE Loss", fontsize=9)
    ax.set_title("(T1)  Neural Network Training Loss (Bellman MSE)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── T2: Epsilon decay ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(s, df["epsilon"], color="#f39c12", lw=1.2, label="Epsilon")
    ax.axhline(0.10, color="gray", ls=":", lw=1.0, alpha=0.7, label="eps=0.10")
    ax.axhline(0.05, color="red",  ls=":",  lw=1.0, alpha=0.7, label="eps=0.05 (final)")
    ax.axvline(TRAIN_STEPS, color="navy", ls="--", lw=1.2)
    ax.fill_between(s, df["epsilon"], alpha=0.15, color="#f39c12")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, STEPS)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Epsilon (Exploration Rate)", fontsize=9)
    ax.set_title("(T2)  Epsilon-Greedy Exploration Decay",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── T3: Reward component breakdown ────────────────────────────────────────
    ax = axes[1, 0]
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=0.06, color=c)
    comps = {
        "Legit Term (+)"   : ("rl_legit_term",    "#27ae60"),
        "Security Term (+)": ("rl_security_term",  "#3498db"),
        "Overload Term (-)": ("rl_overload_term",  "#e74c3c"),
        "Latency Term (-)" : ("rl_latency_term",   "#e67e22"),
    }
    for label, (col, clr) in comps.items():
        ax.plot(s, df[col].rolling(roll, min_periods=1).mean(),
                color=clr, lw=0.9, alpha=0.85, label=label)
    ax.axhline(0, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlim(0, STEPS)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Reward Component (smoothed)", fontsize=9)
    ax.set_title("(T3)  RL Reward Components Over Time",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3, lw=0.6)

    # ── T4: Q-value max evolution ─────────────────────────────────────────────
    ax = axes[1, 1]
    for j, c in enumerate(SCENARIO_COLORS):
        ax.axvspan(PHASE_BOUNDS[j], PHASE_BOUNDS[j+1], alpha=0.06, color=c)
    ax.plot(s, df["q_max"].rolling(roll, min_periods=1).mean(),
            color="#9b59b6", lw=1.2, label=f"Max Q-value (roll-{roll})")
    ax.fill_between(s, df["q_max"].rolling(roll, min_periods=1).mean(),
                    alpha=0.15, color="#9b59b6")
    ax.axvline(TRAIN_STEPS, color="navy", ls="--", lw=1.2,
               label=f"Step {TRAIN_STEPS}")
    ax.set_xlim(0, STEPS)
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Max Q-value", fontsize=9)
    ax.set_title("(T4)  Max Q-Value Evolution (confidence of agent)",
                 fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.6)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved  ->  {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_FILE = "ml_rl_simulation.csv"
    PNG_MAIN = "ml_rl_main_results.png"
    PNG_TRAIN= "ml_rl_training_analysis.png"

    print("=" * 70)
    print("  Smart Grid -- DRL Adaptive Rate Limiter Simulation")
    print("  Section IV.D: Machine Learning & Reinforcement Learning")
    print("=" * 70)
    print(f"\n  Running {STEPS:,}-step simulation ...")
    print(f"  DQN agent trains for first {TRAIN_STEPS:,} steps, "
          f"then evaluates for {STEPS-TRAIN_STEPS:,} steps.")
    print(f"  Static baseline: fixed {STATIC_LIMIT} req/step throughout.\n")

    df = run_simulation(steps=STEPS, seed=42)

    df.to_csv(CSV_FILE, index=False)
    print(f"  CSV saved    ->  {CSV_FILE}  ({len(df):,} rows, {len(df.columns)} columns)")

    print_summary(df)

    print("\n  Generating plots ...")
    plot_main_results(df,      out=PNG_MAIN)
    plot_training_analysis(df, out=PNG_TRAIN)

    print("\n  Done.")
