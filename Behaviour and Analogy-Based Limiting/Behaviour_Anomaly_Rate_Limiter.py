#!/usr/bin/env python3
"""
Behaviour_Anomaly_Rate_Limiter.py
==================================
Algorithm: Anomaly-Weighted Token Bucket for Behaviour and Anomaly-Based Rate Limiting.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh
Section: IV.B -- Behaviour and Anomaly-Based Limiting

Core equations
--------------
Anomaly score (EWMA of normalised z-score deviation):
    alpha_i(t) = lambda * |x_i(t) - mu_i| / sigma_i  +  (1 - lambda) * alpha_i(t-1)

Adaptive token refill rate:
    refill_i(t) = r_base * (1 - alpha_i(t))

where
    x_i(t)     current request rate for client i at time t
    mu_i        learned mean of client i historical behaviour
    sigma_i     learned std-dev of client i historical behaviour
    lambda      EWMA smoothing factor  (0 < lambda < 1)
    alpha_i     anomaly score in [0, 1]  (0 = fully normal, 1 = fully anomalous)

The z-score is normalised by z_threshold (default 3.0, the 3-sigma rule) before
being blended into alpha, so deviations beyond z_threshold standard deviations
contribute the maximum penalty (1.0) to the anomaly score.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-client behavioural baseline
# ─────────────────────────────────────────────────────────────────────────────

class ClientProfile:
    """
    Per-client behavioural baseline using Welford's online algorithm.

    Tracks the running mean and standard deviation of observed request
    rates without storing the entire history, making it suitable for
    long-running smart grid deployments.

    The profile enters a "warmed-up" state after warmup_steps observations,
    at which point anomaly scoring is considered reliable.
    """

    def __init__(self, warmup_steps: int = 50) -> None:
        """
        Parameters
        ----------
        warmup_steps : minimum observations before anomaly scoring begins.
                       During warm-up, alpha stays at 0.
        """
        self.warmup_steps = warmup_steps
        self._n    = 0
        self._mean = 0.0
        self._M2   = 0.0    # accumulated squared deviations (Welford's method)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Number of observations recorded so far."""
        return self._n

    @property
    def mean(self) -> float:
        """Running sample mean."""
        return self._mean

    @property
    def std(self) -> float:
        """
        Running sample standard deviation.
        Returns 1.0 until at least two observations have been collected
        to avoid division-by-zero in the anomaly score computation.
        """
        if self._n < 2:
            return 1.0
        return float(max(np.sqrt(self._M2 / (self._n - 1)), 1e-6))

    @property
    def is_warmed_up(self) -> bool:
        """True once warmup_steps observations have been collected."""
        return self._n >= self.warmup_steps

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, x: float) -> None:
        """
        Incorporate one new observation using Welford's numerically stable
        online algorithm for computing running mean and variance.
        """
        self._n   += 1
        delta      = x - self._mean
        self._mean += delta / self._n
        delta2     = x - self._mean
        self._M2  += delta * delta2

    def reset(self) -> None:
        """Clear all accumulated statistics and restart from scratch."""
        self._n    = 0
        self._mean = 0.0
        self._M2   = 0.0

    def __repr__(self) -> str:
        return (
            f"ClientProfile(n={self._n}, mean={self._mean:.2f}, "
            f"std={self.std:.2f}, warmed_up={self.is_warmed_up})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Anomaly-Weighted Token Bucket (single client)
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyWeightedTokenBucket:
    """
    Anomaly-Weighted Token Bucket for a single API client.

    Augments a classical token bucket with per-client behavioural profiling.
    As a client's request rate deviates from its learned baseline, its anomaly
    score alpha rises and its token refill rate is proportionally reduced:

        refill_i(t) = r_base * (1 - alpha_i(t))

    The anomaly score is an EWMA of the normalised z-score deviation:

        alpha_i(t) = lambda * clip(|x - mu| / (sigma * z_thresh), 0, 1)
                   + (1 - lambda) * alpha_i(t - 1)

    Clients exhibiting anomalous behaviour (e.g. compromised IoT devices
    conducting DDoS) receive progressively stricter refill rates, while
    legitimate traffic patterns preserve near-full throughput.

    Token Bucket mechanics
    ----------------------
    Tokens accumulate at the adaptive refill rate up to a fixed capacity.
    Each accepted request consumes one token.  Requests exceeding the
    available token count are rejected and a 429-equivalent is implied.
    """

    def __init__(
        self,
        client_id: str,
        r_base: float              = 100.0,  # req / time-step when alpha = 0
        lambda_ewma: float         = 0.30,   # EWMA smoothing factor
        z_threshold: float         = 3.0,    # sigma multiples = full anomaly
        warmup_steps: int          = 50,
        capacity_multiplier: float = 1.5,    # capacity = r_base * multiplier
    ) -> None:
        self.client_id          = client_id
        self.r_base             = r_base
        self.lambda_ewma        = lambda_ewma
        self.z_threshold        = z_threshold
        self.capacity           = r_base * capacity_multiplier

        # Mutable state
        self._tokens  = float(self.capacity)
        self._alpha   = 0.0
        self._profile = ClientProfile(warmup_steps=warmup_steps)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Current anomaly score in [0, 1]."""
        return self._alpha

    @property
    def tokens(self) -> float:
        """Current token count."""
        return self._tokens

    @property
    def profile(self) -> ClientProfile:
        """Reference to this client's behavioural profile."""
        return self._profile

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self, incoming: float, dt: float = 1.0) -> dict:
        """
        Process one time-step for this client.

        Algorithm steps
        ---------------
        1. Update baseline statistics (mean, std) with current request count.
        2. Compute z-score deviation and update anomaly score alpha via EWMA.
        3. Compute adaptive refill rate  =  r_base * (1 - alpha).
        4. Refill token bucket (capped at capacity).
        5. Accept min(incoming, tokens); reject the remainder.

        Parameters
        ----------
        incoming : observed request count in this time-step
        dt       : time-step length in seconds (default 1)

        Returns
        -------
        dict with keys:
            accepted, rejected, rejection_rate,
            alpha, refill_rate, tokens,
            z_score, z_score_norm, mu, sigma
        """
        # Step 1: update behavioural profile
        self._profile.update(incoming)
        mu    = self._profile.mean
        sigma = self._profile.std

        # Step 2: anomaly score update
        if self._profile.is_warmed_up:
            z_raw  = abs(incoming - mu) / sigma
            z_norm = float(np.clip(z_raw / self.z_threshold, 0.0, 1.0))
            self._alpha = float(np.clip(
                self.lambda_ewma * z_norm + (1.0 - self.lambda_ewma) * self._alpha,
                0.0, 1.0,
            ))
        else:
            z_raw  = 0.0
            z_norm = 0.0

        # Step 3: adaptive refill rate
        refill_rate = self.r_base * (1.0 - self._alpha)

        # Step 4: refill tokens
        self._tokens = min(self.capacity, self._tokens + refill_rate * dt)

        # Step 5: accept / reject requests
        accepted = float(min(incoming, self._tokens))
        rejected = float(max(0.0, incoming - accepted))
        self._tokens -= accepted

        rej_rate = rejected / incoming if incoming > 0 else 0.0

        return {
            "accepted"      : int(accepted),
            "rejected"      : int(rejected),
            "rejection_rate": round(rej_rate, 4),
            "alpha"         : round(self._alpha, 4),
            "refill_rate"   : round(refill_rate, 2),
            "tokens"        : round(self._tokens, 2),
            "z_score"       : round(z_raw,  4),
            "z_score_norm"  : round(z_norm, 4),
            "mu"            : round(mu,     2),
            "sigma"         : round(sigma,  2),
        }

    def reset(self) -> None:
        """Reset bucket tokens, anomaly score, and behavioural profile."""
        self._tokens  = float(self.capacity)
        self._alpha   = 0.0
        self._profile.reset()

    def __repr__(self) -> str:
        return (
            f"AnomalyWeightedTokenBucket(client_id='{self.client_id}', "
            f"r_base={self.r_base}, lambda={self.lambda_ewma}, "
            f"alpha={self._alpha:.4f}, tokens={self._tokens:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-client orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyWeightedRateLimiter:
    """
    Multi-client rate limiter orchestrating one AnomalyWeightedTokenBucket
    per registered client.

    Accepts a configuration dict so that each client can be tuned
    independently (different r_base, lambda, z_threshold, etc.).

    Usage
    -----
        configs = {
            "AMI_Meter"  : {"r_base": 100, "lambda_ewma": 0.3},
            "EV_Charger" : {"r_base": 100, "lambda_ewma": 0.3},
        }
        limiter  = AnomalyWeightedRateLimiter(configs)
        results  = limiter.step({"AMI_Meter": 30, "EV_Charger": 420})
    """

    def __init__(
        self,
        client_configs: dict,
        default_r_base: float          = 100.0,
        default_lambda_ewma: float     = 0.30,
        default_z_threshold: float     = 3.0,
        default_warmup_steps: int      = 50,
        default_capacity_mult: float   = 1.5,
    ) -> None:
        self._buckets: dict = {}
        for cid, cfg in client_configs.items():
            self._buckets[cid] = AnomalyWeightedTokenBucket(
                client_id           = cid,
                r_base              = cfg.get("r_base",              default_r_base),
                lambda_ewma         = cfg.get("lambda_ewma",         default_lambda_ewma),
                z_threshold         = cfg.get("z_threshold",         default_z_threshold),
                warmup_steps        = cfg.get("warmup_steps",        default_warmup_steps),
                capacity_multiplier = cfg.get("capacity_multiplier", default_capacity_mult),
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self, incoming_by_client: dict, dt: float = 1.0) -> dict:
        """
        Process one time-step for all registered clients.

        Parameters
        ----------
        incoming_by_client : {client_id: request_count}
        dt                 : time-step length in seconds

        Returns
        -------
        {client_id: step_result_dict}   (same keys as AnomalyWeightedTokenBucket.step)
        """
        return {
            cid: bucket.step(incoming_by_client.get(cid, 0.0), dt)
            for cid, bucket in self._buckets.items()
        }

    @property
    def clients(self) -> list:
        """Ordered list of registered client IDs."""
        return list(self._buckets.keys())

    def get_bucket(self, client_id: str) -> AnomalyWeightedTokenBucket:
        """Return the token bucket for a specific client."""
        return self._buckets[client_id]

    def reset_all(self) -> None:
        """Reset all client buckets and profiles."""
        for bucket in self._buckets.values():
            bucket.reset()

    def __repr__(self) -> str:
        return (
            f"AnomalyWeightedRateLimiter("
            f"clients={list(self._buckets.keys())})"
        )
