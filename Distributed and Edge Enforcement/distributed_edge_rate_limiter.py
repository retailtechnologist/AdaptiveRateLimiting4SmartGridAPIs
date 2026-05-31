#!/usr/bin/env python3
"""
distributed_edge_rate_limiter.py
==================================
Algorithm: Hierarchical Distributed and Edge Enforcement Rate Limiter.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh
Section: IV.E -- Distributed and Edge Enforcement

Architecture
------------
Three-tier hierarchy:

    Central Policy Controller (CPC)
            |
    Regional Coordinators (RC)
        /   |   |   \\
    Edge Nodes  (local enforcement + local token buckets)

Global rate limit formula (paper)
----------------------------------
    L_global = sum_{e in E}  L_e * phi(e)

where
    L_e      local rate limit at edge node e (adaptive, load-adjusted)
    phi(e)   dynamic weighting factor:
                 phi_raw(e) = W_CAP * (L_base_e / L_base_max)
                            + W_PROX * proximity_score_e
                 phi(e)     = phi_raw(e) / sum_e phi_raw(e)  (normalised)

Per-node local rate limit
--------------------------
    L_e(t) = L_base_e * (1 - load_e) * availability_e * mtd_factor_e
             * gossip_adjustment_e

Gossip Protocol
---------------
Every GOSSIP_INTERVAL steps, each node exchanges current load with its
topological neighbours (partial mesh).  Based on the average neighbour
load, a node applies a small multiplicative adjustment to its L_e:
    avg_nbr_load > 0.75  -->  adjustment = 0.90  (back off to avoid cascade)
    avg_nbr_load < 0.30  -->  adjustment = 1.05  (exploit spare capacity)
    otherwise            -->  adjustment = 1.00  (hold)

Moving Target Defense (MTD)
----------------------------
When MTD is active the engine randomly rotates each node's effective
L_base by +/-MTD_VARIATION (default 30%) on a ROTATION_PERIOD cycle,
making the enforcement surface difficult for attackers to profile.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Edge node configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeNodeConfig:
    """
    Static configuration for one edge node.

    Attributes
    ----------
    node_id         : unique identifier string
    L_base          : nominal maximum rate limit (req / step)
    proximity_score : closeness to critical grid assets [0, 1]
                      (1.0 = directly managing DER / PMU control)
    neighbors       : list of node_id strings (gossip topology)
    node_type       : descriptive role label
    region          : regional coordinator assignment
    """
    node_id        : str
    L_base         : float
    proximity_score: float
    neighbors      : list = field(default_factory=list)
    node_type      : str  = "GENERIC"
    region         : str  = "REGION_A"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Local token bucket  (per-node enforcement)
# ─────────────────────────────────────────────────────────────────────────────

class LocalTokenBucket:
    """
    Token bucket enforcing the per-node local rate limit L_e.

    Tokens refill each step up to L_e (the adaptive local limit).
    Requests are accepted up to available tokens; excess is rejected.
    """

    def __init__(self, initial_capacity: float = 500.0) -> None:
        self._tokens   = initial_capacity
        self._capacity = initial_capacity

    def step(self, incoming: int, L_e: float) -> tuple:
        """
        Process incoming requests under limit L_e.

        Returns (accepted, rejected, tokens_remaining).
        """
        self._capacity = L_e
        self._tokens   = min(L_e, self._tokens + L_e)   # refill up to L_e

        accepted = int(min(incoming, self._tokens))
        rejected = int(max(0, incoming - accepted))
        self._tokens -= accepted

        return accepted, rejected, round(self._tokens, 2)

    def reset(self, capacity: float) -> None:
        self._tokens   = capacity
        self._capacity = capacity


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Weight calculator  (phi(e) + global limit aggregation)
# ─────────────────────────────────────────────────────────────────────────────

class WeightCalculator:
    """
    Computes dynamic weighting factors phi(e) for all edge nodes and
    aggregates them into the global rate limit via the paper's formula.

        phi_raw(e) = W_CAP * (L_base_e / L_base_max)
                   + W_PROX * proximity_score_e

        phi(e)     = phi_raw(e) / sum_e phi_raw(e)

        L_global   = sum_e  L_e(t) * phi(e)

    Weights W_CAP and W_PROX control the relative importance of node
    capacity vs. proximity to critical grid infrastructure.
    """

    W_CAP  = 0.60   # weight of capacity contribution
    W_PROX = 0.40   # weight of proximity contribution

    def __init__(self, configs: list[EdgeNodeConfig]) -> None:
        self._configs = configs
        L_bases       = [c.L_base for c in configs]
        self._L_max   = max(L_bases)

        # Precompute raw weights (static part; changes only if topology changes)
        raw = np.array([
            self.W_CAP  * (c.L_base / self._L_max) +
            self.W_PROX * c.proximity_score
            for c in configs
        ])
        self._phi = raw / raw.sum()      # normalise to sum=1

    def phi(self) -> np.ndarray:
        """Return current phi(e) array (one value per node, sums to 1)."""
        return self._phi.copy()

    def recompute_phi(self, availability: np.ndarray) -> np.ndarray:
        """
        Recompute phi excluding failed/offline nodes.

        Parameters
        ----------
        availability : (N,) bool/float array; 0 = offline, 1 = online

        Returns
        -------
        Adjusted phi array (offline nodes get phi=0, rest renormalised).
        """
        raw = np.array([
            self.W_CAP  * (c.L_base / self._L_max) +
            self.W_PROX * c.proximity_score
            for c in self._configs
        ]) * availability

        total = raw.sum()
        if total == 0:
            return np.zeros(len(self._configs))
        phi_adj     = raw / total
        self._phi   = phi_adj
        return phi_adj

    def global_limit(self, L_e_arr: np.ndarray) -> float:
        """
        Apply the paper's formula:  L_global = sum_e L_e * phi(e)
        """
        return float(np.dot(L_e_arr, self._phi))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Gossip protocol  (state synchronisation)
# ─────────────────────────────────────────────────────────────────────────────

class GossipProtocol:
    """
    Lightweight gossip protocol for edge-node state synchronisation.

    Each node maintains a local view of its neighbours' load levels.
    Every GOSSIP_INTERVAL steps, nodes exchange load information and
    compute a multiplicative adjustment to their own L_e.

    Adjustment rules
    ----------------
    avg_neighbour_load > HIGH_THRESHOLD  -->  adjustment = BACK_OFF
        (neighbours overloaded; reduce own rate to prevent cascade)
    avg_neighbour_load < LOW_THRESHOLD   -->  adjustment = RAMP_UP
        (neighbours have spare capacity; increase own rate)
    otherwise                            -->  adjustment = 1.00
    """

    GOSSIP_INTERVAL  = 10    # steps between gossip rounds
    HIGH_THRESHOLD   = 0.75  # neighbour avg load -> back-off
    LOW_THRESHOLD    = 0.30  # neighbour avg load -> ramp-up
    BACK_OFF         = 0.90
    RAMP_UP          = 1.05
    HOLD             = 1.00

    def __init__(self, configs: list[EdgeNodeConfig]) -> None:
        self._idx  = {c.node_id: i for i, c in enumerate(configs)}
        self._nbrs = {c.node_id: c.neighbors for c in configs}
        # Per-node adjustments (persist between gossip rounds)
        self._adj  = {c.node_id: 1.0 for c in configs}

    def sync(self, step: int, load_by_node: dict) -> dict:
        """
        Perform a gossip round if step is a sync step; otherwise return
        last cached adjustments.

        Parameters
        ----------
        step         : current simulation step
        load_by_node : {node_id: load_fraction [0,1]}

        Returns
        -------
        {node_id: adjustment_multiplier}
        """
        if step % self.GOSSIP_INTERVAL != 0:
            return self._adj.copy()

        for nid, nbrs in self._nbrs.items():
            if not nbrs:
                continue
            nbr_loads = [load_by_node.get(n, 0.0) for n in nbrs]
            avg       = float(np.mean(nbr_loads))

            if avg > self.HIGH_THRESHOLD:
                self._adj[nid] = self.BACK_OFF
            elif avg < self.LOW_THRESHOLD:
                self._adj[nid] = self.RAMP_UP
            else:
                self._adj[nid] = self.HOLD

        return self._adj.copy()

    def is_gossip_step(self, step: int) -> bool:
        return step % self.GOSSIP_INTERVAL == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Moving Target Defense engine
# ─────────────────────────────────────────────────────────────────────────────

class MTDEngine:
    """
    Moving Target Defense: periodically rotates each node's effective
    L_base within a bounded range, making the enforcement surface
    unpredictable for profiling-based DDoS attacks.

    On each ROTATION_PERIOD the engine draws a new MTD factor for every
    node from Uniform(1-VARIATION, 1+VARIATION).  Between rotations
    the factor is held constant (deterministic window for legitimate traffic).
    """

    ROTATION_PERIOD = 50    # steps between parameter rotations
    VARIATION       = 0.30  # +/- 30% of nominal L_base

    def __init__(self, node_ids: list[str], seed: int = 99) -> None:
        self._ids     = node_ids
        self._rng     = np.random.default_rng(seed)
        self._factors = {nid: 1.0 for nid in node_ids}
        self._round   = -1

    def get_factor(self, node_id: str, step: int, active: bool) -> float:
        """
        Return the MTD multiplier for node_id at the current step.

        If MTD is not active, always returns 1.0.
        Rotates the factor every ROTATION_PERIOD steps.
        """
        if not active:
            return 1.0

        current_round = step // self.ROTATION_PERIOD
        if current_round != self._round:
            self._round = current_round
            for nid in self._ids:
                lo = 1.0 - self.VARIATION
                hi = 1.0 + self.VARIATION
                self._factors[nid] = float(self._rng.uniform(lo, hi))

        return self._factors.get(node_id, 1.0)

    def get_all_factors(self) -> dict:
        return self._factors.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Distributed Edge Rate Limiter  (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class DistributedEdgeRateLimiter:
    """
    Hierarchical Distributed Rate Limiter orchestrating all edge nodes.

    Implements the paper's global aggregation formula:
        L_global = sum_{e in E} L_e * phi(e)

    Per-step pipeline
    -----------------
    1.  Receive per-node availability and load from environment.
    2.  Recompute phi(e) based on current availability.
    3.  Run gossip protocol; obtain per-node adjustment multipliers.
    4.  Compute per-node adaptive local limit:
            L_e = L_base_e * (1-load_e) * availability_e
                           * mtd_factor_e * gossip_adj_e
        clipped to [L_min, L_base_e].
    5.  Aggregate:  L_global = sum_e L_e * phi(e).
    6.  Apply local token bucket per node; record results.
    """

    L_MIN_FRAC = 0.05   # minimum fraction of L_base a node can drop to

    def __init__(
        self,
        configs:        list[EdgeNodeConfig],
        gossip_interval: int   = 10,
        mtd_variation:  float  = 0.30,
        mtd_period:     int    = 50,
        seed:           int    = 42,
    ) -> None:
        self._configs = configs
        self._ids     = [c.node_id for c in configs]
        self._idx     = {c.node_id: i for i, c in enumerate(configs)}

        self._weights = WeightCalculator(configs)
        self._gossip  = GossipProtocol(configs)
        self._mtd     = MTDEngine(self._ids, seed=seed)
        self._buckets = {c.node_id: LocalTokenBucket(c.L_base) for c in configs}

    # ── Main step ──────────────────────────────────────────────────────────

    def step(
        self,
        step:              int,
        incoming_by_node:  dict,       # {node_id: int}
        load_by_node:      dict,       # {node_id: float 0-1}
        availability:      dict,       # {node_id: bool}
        mtd_active:        bool = False,
    ) -> dict:
        """
        Process one time-step across all edge nodes.

        Returns a dict: {node_id: per_node_result_dict}
        Each result contains L_e, phi_e, L_global, gossip_adj, mtd_factor,
        accepted, rejected, tokens, rejection_rate.
        """
        avail_arr = np.array([float(availability.get(nid, True))
                              for nid in self._ids])

        # Step 2: recompute phi
        phi_arr = self._weights.recompute_phi(avail_arr)

        # Step 3: gossip adjustment
        gossip_adj = self._gossip.sync(step, load_by_node)
        is_gossip  = self._gossip.is_gossip_step(step)

        # Steps 4-5: per-node L_e and global aggregation
        L_e_arr = np.zeros(len(self._ids))
        for i, c in enumerate(self._configs):
            nid   = c.node_id
            load  = float(load_by_node.get(nid, 0.0))
            avail = float(availability.get(nid, True))
            mtd_f = self._mtd.get_factor(nid, step, mtd_active)
            g_adj = gossip_adj.get(nid, 1.0)

            L_e = c.L_base * (1.0 - load) * avail * mtd_f * g_adj
            L_e = float(np.clip(L_e, self.L_MIN_FRAC * c.L_base, c.L_base))
            L_e_arr[i] = L_e

        L_global = self._weights.global_limit(L_e_arr)

        # Step 6: per-node token bucket
        results = {}
        for i, c in enumerate(self._configs):
            nid  = c.node_id
            L_e  = float(L_e_arr[i])
            incoming = incoming_by_node.get(nid, 0)
            acc, rej, tokens = self._buckets[nid].step(incoming, L_e)
            rej_rate = rej / incoming if incoming > 0 else 0.0

            results[nid] = {
                "L_e"           : round(L_e,         2),
                "phi_e"         : round(phi_arr[i],  4),
                "L_global"      : round(L_global,    2),
                "mtd_factor"    : round(self._mtd.get_factor(nid, step, mtd_active), 4),
                "gossip_adj"    : round(gossip_adj.get(nid, 1.0), 4),
                "is_gossip_step": is_gossip,
                "load_e"        : round(float(load_by_node.get(nid, 0.0)), 4),
                "availability"  : float(availability.get(nid, True)),
                "accepted"      : acc,
                "rejected"      : rej,
                "tokens"        : tokens,
                "rejection_rate": round(rej_rate, 4),
            }

        return results

    @property
    def node_ids(self) -> list:
        return self._ids

    @property
    def configs(self) -> list:
        return self._configs

    def __repr__(self) -> str:
        return (
            f"DistributedEdgeRateLimiter("
            f"nodes={self._ids}, "
            f"phi={np.round(self._weights.phi(), 3).tolist()})"
        )
