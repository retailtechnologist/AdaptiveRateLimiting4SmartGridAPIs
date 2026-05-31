#!/usr/bin/env python3
"""
Feedback_Driven_Adaption.py
===========================
Algorithm implementation: PID-inspired Feedback-Driven Adaptive Rate Limiter.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh
Section: IV.A -- Feedback Driven Adaptation

PID formula
-----------
    r(t) = r_base + Kp*e(t) + Ki * integral_0^t e(tau) dtau + Kd * de(t)/dt

where
    r(t)        instantaneous allowed request rate (req/min)
    r_base      nominal baseline rate
    e(t)        error signal  =  target_load - observed_load
    Kp, Ki, Kd  tuned controller gains

Composite load metric (normalised to [0, 1])
--------------------------------------------
    load = 0.35*(cpu/100) + 0.25*(mem/100) + 0.25*(lat/500) + 0.15*(queue/1000)
"""

import numpy as np


class PIDRateLimiter:
    """
    PID-inspired feedback controller for smart grid API rate limiting.

    The controller continuously monitors four infrastructure metrics --
    CPU utilisation, memory pressure, request latency, and queue depth --
    and combines them into a single composite load index.  The PID loop
    drives that index toward a configurable target (default 0.60) by
    adjusting the allowed request rate at every time-step.

    Anti-windup
    -----------
    The integral accumulator is clamped to +/-integral_clamp to prevent
    the I-term from winding up during prolonged overload or underload and
    causing large overshoots upon condition change.
    """

    # ------------------------------------------------------------------
    # Load-metric normalisation constants
    # ------------------------------------------------------------------
    CPU_MAX   = 100.0   # percent
    MEM_MAX   = 100.0   # percent
    LAT_MAX   = 500.0   # ms  (500 ms = fully saturated latency)
    QUEUE_MAX = 1000.0  # requests in queue

    # ------------------------------------------------------------------
    # Weights for the composite load metric  (must sum to 1.0)
    # ------------------------------------------------------------------
    W_CPU   = 0.35
    W_MEM   = 0.25
    W_LAT   = 0.25
    W_QUEUE = 0.15

    def __init__(
        self,
        r_base: float         = 1000.0,  # req/min  baseline rate
        Kp: float             = 1500.0,  # proportional gain
        Ki: float             = 10.0,    # integral gain
        Kd: float             = 300.0,   # derivative gain
        r_min: float          = 100.0,   # req/min  hard floor
        r_max: float          = 5000.0,  # req/min  hard ceiling
        target_load: float    = 0.60,    # desired composite load [0, 1]
        dt: float             = 1.0,     # time-step duration (seconds)
        integral_clamp: float = 100.0,   # anti-windup bound on integrator
    ) -> None:
        self.r_base         = r_base
        self.Kp             = Kp
        self.Ki             = Ki
        self.Kd             = Kd
        self.r_min          = r_min
        self.r_max          = r_max
        self.target_load    = target_load
        self.dt             = dt
        self.integral_clamp = integral_clamp

        # Mutable controller state
        self._integral   = 0.0
        self._prev_error = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        cpu: float,
        memory: float,
        latency: float,
        queue: float,
    ) -> dict:
        """
        Advance one time-step and return the new rate-limit decision.

        Parameters
        ----------
        cpu     : CPU utilisation  (0-100 %)
        memory  : Memory pressure  (0-100 %)
        latency : Request latency  (ms)
        queue   : Queue depth      (request count)

        Returns
        -------
        dict with keys:
            rate_limit    -- new allowed request rate (req/min)
            observed_load -- composite load index [0, 1]
            error         -- e(t) = target_load - observed_load
            p_term        -- Kp * e(t)
            i_term        -- Ki * integral
            d_term        -- Kd * de(t)/dt
            integral      -- accumulated error after anti-windup
        """
        observed = self._load_metric(cpu, memory, latency, queue)
        error    = self.target_load - observed   # e(t)

        # --- Integral term with anti-windup ---
        self._integral += error * self.dt
        self._integral  = float(np.clip(
            self._integral, -self.integral_clamp, self.integral_clamp
        ))

        # --- Derivative term (backward finite difference) ---
        derivative       = (error - self._prev_error) / self.dt
        self._prev_error = error

        p_term = self.Kp * error
        i_term = self.Ki * self._integral
        d_term = self.Kd * derivative

        # --- PID output clamped to operating range ---
        r_t = float(np.clip(
            self.r_base + p_term + i_term + d_term,
            self.r_min,
            self.r_max,
        ))

        return {
            "rate_limit"   : round(r_t,              2),
            "observed_load": round(observed,          4),
            "error"        : round(error,             4),
            "p_term"       : round(p_term,            4),
            "i_term"       : round(i_term,            4),
            "d_term"       : round(d_term,            4),
            "integral"     : round(self._integral,    4),
        }

    def reset(self) -> None:
        """Reset controller state (integral accumulator and previous error)."""
        self._integral   = 0.0
        self._prev_error = 0.0

    @property
    def state(self) -> dict:
        """Read-only snapshot of internal controller state."""
        return {
            "integral"  : self._integral,
            "prev_error": self._prev_error,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metric(
        self,
        cpu: float,
        memory: float,
        latency: float,
        queue: float,
    ) -> float:
        """Return the weighted composite system load index in [0, 1]."""
        cpu_n = float(np.clip(cpu     / self.CPU_MAX,   0.0, 1.0))
        mem_n = float(np.clip(memory  / self.MEM_MAX,   0.0, 1.0))
        lat_n = float(np.clip(latency / self.LAT_MAX,   0.0, 1.0))
        que_n = float(np.clip(queue   / self.QUEUE_MAX, 0.0, 1.0))
        return (
            self.W_CPU   * cpu_n +
            self.W_MEM   * mem_n +
            self.W_LAT   * lat_n +
            self.W_QUEUE * que_n
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PIDRateLimiter("
            f"r_base={self.r_base}, Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
            f"target_load={self.target_load}, "
            f"r_min={self.r_min}, r_max={self.r_max})"
        )
