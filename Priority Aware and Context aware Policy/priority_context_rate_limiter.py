#!/usr/bin/env python3
"""
priority_context_rate_limiter.py
=================================
Algorithm: Multi-Level Priority-Aware and Context-Aware Rate Limiter.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh
Section: IV.C -- Priority Aware and Context Aware Policy

Core formula
------------
    L(p, c) = L_base * (1 + beta * p * gamma(c))

where
    p          effective priority weight in [0, 1]
                 = 0.50 * p_message  +  0.30 * p_role  +  0.20 * p_urgency
    gamma(c)   context multiplier derived from current grid conditions
                 = base_gamma(grid_state) * continuous_adjustment(freq, renewable, flags)
    beta       priority sensitivity factor  (controls how much priority amplifies the limit)
    L_base     baseline rate (requests allowed when p = 0)

Priority tiers
--------------
    CRITICAL   p >= 0.85   -- fault isolation, protection relay, load shedding
    HIGH       p in [0.65, 0.85)  -- demand response, frequency regulation, DER control
    MEDIUM     p in [0.45, 0.65)  -- PMU sync
    LOW        p < 0.45    -- market operations, meter data, telemetry

Grid state -> base gamma
------------------------
    NORMAL     1.0      ELEVATED   1.5      ALERT      2.0
    EMERGENCY  3.0      CRITICAL   4.5

Context adjustment (continuous)
-------------------------------
    gamma(c) = base_gamma * (1 + 0.30*norm_freq + 0.20*norm_ren + 0.50*emg_flag)
    clipped to [1.0, 6.0]

Effect on rate limits (L_base=100, beta=3)
------------------------------------------
    CRITICAL message  +  CRITICAL grid   ->  L up to  1900 req/step
    LOW message       +  NORMAL  grid    ->  L ~  160 req/step
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Priority Classifier
# ─────────────────────────────────────────────────────────────────────────────

class MessagePriorityClassifier:
    """
    Assigns a priority weight p in [0, 1] to each incoming API request
    based on three factors: message type, source role, and urgency flag.

        p = W_MSG * p_message + W_ROLE * p_role + W_URG * p_urgency

    Message type weights reflect the criticality of the operation in the
    smart grid context.  Source role weights reflect the authority and
    accountability of the requesting entity.  Urgency is a binary flag
    that can be set by the client or auto-assigned during grid events.
    """

    # Weights for the three contributing factors
    W_MSG  = 0.50
    W_ROLE = 0.30
    W_URG  = 0.20

    # Message-type priority scores
    MESSAGE_PRIORITY: dict = {
        "FAULT_ISOLATION"     : 1.00,   # highest -- safety-critical
        "PROTECTION_RELAY"    : 0.95,
        "LOAD_SHEDDING"       : 0.95,
        "DEMAND_RESPONSE"     : 0.90,
        "FREQUENCY_REGULATION": 0.85,
        "DER_CONTROL"         : 0.80,
        "PMU_SYNC"            : 0.70,
        "MARKET_OPERATIONS"   : 0.40,
        "METER_DATA"          : 0.30,
        "TELEMETRY"           : 0.20,   # lowest -- routine telemetry
    }

    # Source role factors
    ROLE_FACTOR: dict = {
        "GRID_OPERATOR" : 1.00,
        "UTILITY"       : 0.85,
        "DER_PROVIDER"  : 0.75,
        "EV_CHARGER"    : 0.55,
        "SMART_METER"   : 0.45,
        "THIRD_PARTY"   : 0.35,
    }

    # Priority tier thresholds
    TIER_THRESHOLDS: dict = {
        "CRITICAL": 0.85,
        "HIGH"    : 0.65,
        "MEDIUM"  : 0.45,
        "LOW"     : 0.00,
    }

    def classify(
        self,
        message_type: str,
        source_role:  str,
        urgency:      bool,
    ) -> dict:
        """
        Compute the effective priority weight for a single request.

        Parameters
        ----------
        message_type : one of MESSAGE_PRIORITY keys
        source_role  : one of ROLE_FACTOR keys
        urgency      : True if the request is flagged as time-critical

        Returns
        -------
        dict with keys: p_message, p_role, p_urgency, p_effective, priority_tier
        """
        p_msg  = self.MESSAGE_PRIORITY.get(message_type, 0.30)
        p_role = self.ROLE_FACTOR.get(source_role, 0.35)
        p_urg  = 1.0 if urgency else 0.0

        p_eff  = float(np.clip(
            self.W_MSG * p_msg + self.W_ROLE * p_role + self.W_URG * p_urg,
            0.0, 1.0,
        ))

        tier = "LOW"
        for t, threshold in self.TIER_THRESHOLDS.items():
            if p_eff >= threshold:
                tier = t
                break

        return {
            "p_message"    : round(p_msg,  4),
            "p_role"       : round(p_role, 4),
            "p_urgency"    : round(p_urg,  4),
            "p_effective"  : round(p_eff,  4),
            "priority_tier": tier,
        }

    @classmethod
    def list_message_types(cls) -> list:
        return list(cls.MESSAGE_PRIORITY.keys())

    @classmethod
    def list_roles(cls) -> list:
        return list(cls.ROLE_FACTOR.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Grid Context Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class GridContextEvaluator:
    """
    Computes the context multiplier gamma(c) from real-time grid conditions.

    gamma(c) = base_gamma(grid_state) * continuous_adjustment(freq, ren, emg)

    The base gamma captures the discrete operating regime of the grid.
    The continuous adjustment provides fine-grained sensitivity to:
        - Frequency deviation from nominal (60 Hz) -- grid stress indicator
        - Renewable penetration percentage       -- volatility indicator
        - Emergency flag                          -- binary crisis signal

    gamma is clipped to [GAMMA_MIN, GAMMA_MAX].
    """

    # Discrete grid-state gamma values
    GRID_STATE_BASE_GAMMA: dict = {
        "NORMAL"   : 1.0,
        "ELEVATED" : 1.5,
        "ALERT"    : 2.0,
        "EMERGENCY": 3.0,
        "CRITICAL" : 4.5,
    }

    # Continuous adjustment weights
    W_FREQ = 0.30   # weight of frequency deviation contribution
    W_REN  = 0.20   # weight of renewable penetration contribution
    W_EMG  = 0.50   # weight of emergency flag contribution

    FREQ_MAX  = 0.5   # Hz -- deviation beyond this is treated as maximum stress
    GAMMA_MIN = 1.0
    GAMMA_MAX = 6.0

    def compute_gamma(
        self,
        grid_state:     str,
        freq_deviation: float = 0.0,   # Hz from nominal
        renewable_pct:  float = 20.0,  # % of generation from renewables
        emergency_flag: bool  = False,
    ) -> dict:
        """
        Compute the context multiplier for the current grid conditions.

        Parameters
        ----------
        grid_state     : one of GRID_STATE_BASE_GAMMA keys
        freq_deviation : absolute Hz deviation from 60 Hz nominal
        renewable_pct  : renewable energy as % of total generation (0-100)
        emergency_flag : True if a grid emergency has been declared

        Returns
        -------
        dict with keys: grid_state, base_gamma, adj_factor, gamma, freq_norm, ren_norm
        """
        base_gamma = self.GRID_STATE_BASE_GAMMA.get(grid_state, 1.0)

        freq_norm = float(np.clip(abs(freq_deviation) / self.FREQ_MAX, 0.0, 1.0))
        ren_norm  = float(np.clip(renewable_pct / 100.0, 0.0, 1.0))
        emg_val   = 1.0 if emergency_flag else 0.0

        adj_factor = 1.0 + self.W_FREQ * freq_norm + self.W_REN * ren_norm + self.W_EMG * emg_val
        gamma      = float(np.clip(base_gamma * adj_factor, self.GAMMA_MIN, self.GAMMA_MAX))

        return {
            "grid_state" : grid_state,
            "base_gamma" : round(base_gamma,  4),
            "adj_factor" : round(adj_factor,  4),
            "gamma"      : round(gamma,       4),
            "freq_norm"  : round(freq_norm,   4),
            "ren_norm"   : round(ren_norm,    4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-Level Priority-Context Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class PriorityContextRateLimiter:
    """
    Multi-Level Context-Aware Rate Limiter implementing the paper's formula:

        L(p, c) = L_base * (1 + beta * p * gamma(c))

    Critical commands (fault isolation, load shedding during blackouts) receive
    near-unrestricted passage through the high p*gamma product.  Routine
    telemetry faces conservative limits even during emergencies because its
    low p cannot be amplified sufficiently by gamma alone.

    A simple per-role token bucket enforces the computed limit each step,
    ensuring that bursty clients accumulate deficit over time.

    Parameters
    ----------
    L_base : baseline rate -- requests allowed when p = 0  (req / step)
    beta   : priority sensitivity factor
    L_min  : hard floor on computed rate limit
    L_max  : hard ceiling on computed rate limit
    """

    def __init__(
        self,
        L_base: float = 100.0,
        beta:   float = 3.0,
        L_min:  float = 50.0,
        L_max:  float = 2000.0,
    ) -> None:
        self.L_base = L_base
        self.beta   = beta
        self.L_min  = L_min
        self.L_max  = L_max

        self._classifier = MessagePriorityClassifier()
        self._evaluator  = GridContextEvaluator()
        self._tokens: dict = {}   # per-role token accumulator

    # ── Core formula ─────────────────────────────────────────────────────────

    def compute_rate_limit(self, p: float, gamma: float) -> float:
        """
        Apply the paper's formula:  L(p, c) = L_base * (1 + beta * p * gamma)

        Returns the computed limit clamped to [L_min, L_max].
        """
        raw = self.L_base * (1.0 + self.beta * p * gamma)
        return round(float(np.clip(raw, self.L_min, self.L_max)), 2)

    # ── Main processing step ─────────────────────────────────────────────────

    def process(
        self,
        source_role:    str,
        message_type:   str,
        urgency:        bool,
        incoming:       int,
        grid_state:     str,
        freq_deviation: float = 0.0,
        renewable_pct:  float = 20.0,
        emergency_flag: bool  = False,
    ) -> dict:
        """
        Evaluate one batch of API requests and return the rate-limit decision.

        Steps
        -----
        1. Classify request -> effective priority p.
        2. Evaluate grid context -> gamma(c).
        3. Compute L(p, c).
        4. Enforce limit via per-role token bucket.
        5. Return accepted / rejected counts and all intermediate values.

        Parameters
        ----------
        source_role    : client role identifier
        message_type   : API operation type
        urgency        : True if time-critical flag is set
        incoming       : number of requests in this time-step
        grid_state     : current operating regime
        freq_deviation : Hz deviation from 60 Hz nominal
        renewable_pct  : renewable energy as % of generation
        emergency_flag : True if grid emergency declared

        Returns
        -------
        dict containing all intermediate values and the final decision
        """
        # Step 1: priority classification
        prio  = self._classifier.classify(message_type, source_role, urgency)
        p     = prio["p_effective"]

        # Step 2: context evaluation
        ctx   = self._evaluator.compute_gamma(
            grid_state, freq_deviation, renewable_pct, emergency_flag
        )
        gamma = ctx["gamma"]

        # Step 3: adaptive rate limit
        L = self.compute_rate_limit(p, gamma)

        # Step 4: per-role token bucket enforcement
        bucket = self._tokens.get(source_role, self.L_base)
        bucket = min(L, bucket + L)           # refill up to computed limit
        accepted = int(min(incoming, bucket))
        rejected = int(max(0, incoming - accepted))
        bucket  -= accepted
        self._tokens[source_role] = bucket

        rej_rate = rejected / incoming if incoming > 0 else 0.0

        return {
            # -- Priority breakdown --
            "p_message"    : prio["p_message"],
            "p_role"       : prio["p_role"],
            "p_urgency"    : prio["p_urgency"],
            "p_effective"  : p,
            "priority_tier": prio["priority_tier"],
            # -- Context --
            "grid_state"   : grid_state,
            "base_gamma"   : ctx["base_gamma"],
            "adj_factor"   : ctx["adj_factor"],
            "gamma"        : gamma,
            "freq_norm"    : ctx["freq_norm"],
            "ren_norm"     : ctx["ren_norm"],
            # -- Rate limit formula --
            "rate_limit"   : L,
            # -- Decision --
            "accepted"     : accepted,
            "rejected"     : rejected,
            "rejection_rate": round(rej_rate, 4),
            "tokens"       : round(bucket, 2),
        }

    def reset(self) -> None:
        """Clear all per-role token accumulators."""
        self._tokens.clear()

    def __repr__(self) -> str:
        return (
            f"PriorityContextRateLimiter("
            f"L_base={self.L_base}, beta={self.beta}, "
            f"L_min={self.L_min}, L_max={self.L_max})"
        )
