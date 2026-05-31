#!/usr/bin/env python3
"""
ml_rl_rate_limiter.py
======================
Algorithm: Deep Reinforcement Learning (DQN) Adaptive Rate Limiter.

Paper  : "Adaptive Rate Limiting Strategies for Smart Grid Critical APIs"
Author : Bhanu Pratap Singh
Section: IV.D -- Machine Learning and Reinforcement Learning Enhancements

MDP Formulation
---------------
The paper treats rate limiting as a Markov Decision Process (MDP) where:

    State  s  -- system snapshot (load, grid health, threat level, forecasts)
    Action a  -- chosen rate limit level (req / step)
    Reward R  -- balances security, availability, efficiency

Reward function (from paper)
-----------------------------
    R(s, a) = w1*(1 - reject_legit) - w2*overload_penalty
            - w3*latency_cost       + w4*security_gain

    w1 = 0.30  (legitimate traffic preservation)
    w2 = 0.35  (overload penalty)
    w3 = 0.15  (latency cost)
    w4 = 0.20  (security gain)

Algorithm
---------
Deep Q-Network (DQN) with:
    - 2-layer neural network Q-function (numpy, no external ML lib)
    - Experience replay buffer
    - Periodic target-network synchronisation
    - Epsilon-greedy exploration with exponential decay

ML Load Predictor
-----------------
An online linear regression model predicts next-step load from:
weather, EV demand, renewable output, time-of-day, and recent history.
This predicted load is injected into the RL agent's state vector,
enabling proactive (anticipatory) rate adjustment.

State vector (10 features, all normalised to [0, 1])
------------------------------------------------------
    [0] cpu_util          [1] memory_util       [2] queue_norm
    [3] latency_norm      [4] req_rate_norm      [5] threat_level
    [6] freq_dev_norm     [7] renewable_norm     [8] ev_demand_norm
    [9] predicted_load    (ML predictor output)

Action space (8 discrete rate limits, req / step)
--------------------------------------------------
    0:100  1:250  2:500  3:750  4:1000  5:1500  6:2000  7:3000
"""

from __future__ import annotations
import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ML Load Predictor
# ─────────────────────────────────────────────────────────────────────────────

class LoadPredictor:
    """
    Online linear regression that predicts next-step normalised load.

    Features (5-D): renewable_norm, ev_demand_norm, time_norm,
                    req_rate_norm, recent_avg_load

    Trained incrementally with stochastic gradient descent so it adapts
    to evolving grid patterns without storing history.
    """

    N_FEATURES = 5

    def __init__(self, lr: float = 0.01) -> None:
        self.lr      = lr
        self.weights = np.zeros(self.N_FEATURES)
        self.bias    = 0.5
        self._n      = 0
        self._ema    = 0.5   # exponential moving average of recent load

    def _feature_vec(
        self,
        renewable_norm: float,
        ev_demand_norm: float,
        time_norm:      float,
        req_rate_norm:  float,
    ) -> np.ndarray:
        return np.array([
            renewable_norm,
            ev_demand_norm,
            time_norm,
            req_rate_norm,
            self._ema,
        ])

    def predict(
        self,
        renewable_norm: float,
        ev_demand_norm: float,
        time_norm:      float,
        req_rate_norm:  float,
    ) -> float:
        """Return predicted normalised load in [0, 1]."""
        x = self._feature_vec(renewable_norm, ev_demand_norm, time_norm, req_rate_norm)
        return float(np.clip(self.weights @ x + self.bias, 0.0, 1.0))

    def update(
        self,
        renewable_norm: float,
        ev_demand_norm: float,
        time_norm:      float,
        req_rate_norm:  float,
        actual_load:    float,
    ) -> float:
        """Online SGD update; returns prediction error."""
        x     = self._feature_vec(renewable_norm, ev_demand_norm, time_norm, req_rate_norm)
        pred  = float(np.clip(self.weights @ x + self.bias, 0.0, 1.0))
        error = actual_load - pred
        self.weights += self.lr * error * x
        self.bias    += self.lr * error
        self._ema     = 0.95 * self._ema + 0.05 * actual_load
        self._n      += 1
        return abs(error)

    def __repr__(self) -> str:
        return f"LoadPredictor(n={self._n}, lr={self.lr}, ema={self._ema:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Neural Q-Network  (pure numpy, no ML framework required)
# ─────────────────────────────────────────────────────────────────────────────

class NeuralQNetwork:
    """
    Two-hidden-layer feedforward Q-network:
        Input(10) -> ReLU(64) -> ReLU(32) -> Linear(n_actions)

    Trained via mini-batch MSE on Bellman targets with manual
    backpropagation (no automatic differentiation dependency).

    He initialisation is used for ReLU layers.
    Gradients are clipped to [-1, 1] to prevent explosion.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr:        float = 1e-3,
        h1:        int   = 64,
        h2:        int   = 32,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr        = lr

        rng = np.random.default_rng(0)
        # He initialisation (fan-in)
        self.W1 = rng.standard_normal((state_dim, h1)) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.standard_normal((h2, n_actions)) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(n_actions)

        # Cache for backprop
        self._x = self._z1 = self._h1 = self._z2 = self._h2 = None

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, state_dim)  ->  Q: (B, n_actions)
        Caches intermediates for subsequent backward() call.
        """
        self._x  = x
        self._z1 = x  @ self.W1 + self.b1
        self._h1 = np.maximum(0.0, self._z1)
        self._z2 = self._h1 @ self.W2 + self.b2
        self._h2 = np.maximum(0.0, self._z2)
        return self._h2 @ self.W3 + self.b3

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a single state (1-D or 2-D)."""
        if state.ndim == 1:
            state = state[np.newaxis, :]
        return self.forward(state)[0]

    # ── Backward ────────────────────────────────────────────────────────────

    def _clip(self, g: np.ndarray) -> np.ndarray:
        return np.clip(g, -1.0, 1.0)

    def train_on_batch(
        self,
        states:  np.ndarray,   # (B, state_dim)
        actions: np.ndarray,   # (B,) int
        targets: np.ndarray,   # (B,) float  Bellman targets
    ) -> float:
        """
        One SGD step minimising MSE between Q(s,a) and Bellman target.
        Returns the scalar loss.
        """
        B     = len(states)
        q     = self.forward(states)                    # (B, n_actions)

        # Build target matrix -- only the taken action differs
        q_tgt = q.copy()
        q_tgt[np.arange(B), actions] = targets          # replace taken-action Q

        # MSE gradient
        dq = (2.0 / B) * (q - q_tgt)                   # (B, n_actions)
        loss = float(np.mean((q - q_tgt) ** 2))

        # Layer 3
        dW3 = self._clip(self._h2.T @ dq)
        db3 = dq.sum(axis=0)
        dh2 = dq @ self.W3.T

        # Layer 2 (ReLU)
        dz2 = dh2 * (self._z2 > 0)
        dW2 = self._clip(self._h1.T @ dz2)
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T

        # Layer 1 (ReLU)
        dz1 = dh1 * (self._z1 > 0)
        dW1 = self._clip(self._x.T  @ dz1)
        db1 = dz1.sum(axis=0)

        # SGD update
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3

        return loss

    # ── Weight copy (for target network) ────────────────────────────────────

    def copy_weights_from(self, source: "NeuralQNetwork") -> None:
        self.W1 = source.W1.copy();  self.b1 = source.b1.copy()
        self.W2 = source.W2.copy();  self.b2 = source.b2.copy()
        self.W3 = source.W3.copy();  self.b3 = source.b3.copy()

    def __repr__(self) -> str:
        return (
            f"NeuralQNetwork(in={self.state_dim}, hidden=(64,32), "
            f"out={self.n_actions}, lr={self.lr})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done) tuples.
    Uniform random sampling is used for mini-batch extraction.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._buf: list  = [None] * capacity
        self._pos: int   = 0
        self._size: int  = 0

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf[self._pos] = (state, action, reward, next_state, float(done))
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Return (states, actions, rewards, next_states, dones) as arrays."""
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        s, a, r, ns, d = zip(*[self._buf[i] for i in idx])
        return (np.array(s),  np.array(a, dtype=int),
                np.array(r),  np.array(ns), np.array(d))

    def __len__(self) -> int:
        return self._size


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MDP Environment (reward + state processing)
# ─────────────────────────────────────────────────────────────────────────────

class SmartGridMDP:
    """
    MDP specification for the smart-grid rate-limiting problem.

    Defines the action space, reward function weights, and helper
    methods for computing reward components from transition data.
    The actual state transitions (traffic simulation) live in the
    simulation harness, keeping algorithm and environment separate.

    Reward weights (paper section IV.D)
    ------------------------------------
        w1 = 0.30  legitimate traffic preservation  (1 - reject_legit)
        w2 = 0.35  overload penalty                  (cpu over 80%)
        w3 = 0.15  latency cost
        w4 = 0.20  security gain                     (restricting under threat)
    """

    # Discrete action space: allowed rate limits (req / step)
    RATE_LIMITS: list = [100, 250, 500, 750, 1000, 1500, 2000, 3000]
    N_ACTIONS:   int  = len(RATE_LIMITS)
    STATE_DIM:   int  = 10

    # Reward weights
    W1 = 0.30   # legitimate preservation
    W2 = 0.35   # overload penalty
    W3 = 0.15   # latency cost
    W4 = 0.20   # security gain

    CPU_OVERLOAD_THRESHOLD = 0.80   # fraction
    MAX_RATE = float(max(RATE_LIMITS))

    def action_to_limit(self, action_idx: int) -> int:
        """Map discrete action index to req/step rate limit."""
        return self.RATE_LIMITS[int(action_idx)]

    def compute_reward(
        self,
        action_idx:       int,
        legit_incoming:   int,
        total_incoming:   int,
        cpu_util:         float,
        latency_norm:     float,
        threat_level:     float,
    ) -> tuple:
        """
        Compute scalar reward and its four components.

        R(s, a) = w1*(1 - reject_legit)
                - w2*overload_penalty
                - w3*latency_cost
                + w4*security_gain

        Parameters
        ----------
        action_idx     : chosen action index
        legit_incoming : legitimate request count this step
        total_incoming : total (legit + attack) request count
        cpu_util       : normalised CPU utilisation [0, 1]
        latency_norm   : normalised latency [0, 1]
        threat_level   : anomaly / threat score [0, 1]

        Returns
        -------
        (reward, components_dict)
        """
        L = self.action_to_limit(action_idx)

        # -- Rejected legitimate traffic -----------------------------------
        # Priority ordering: legitimate traffic is served first up to L
        legit_accepted = int(min(legit_incoming, L))
        reject_legit   = max(0.0, (legit_incoming - legit_accepted) /
                             max(legit_incoming, 1))

        # -- Overload penalty (CPU > 80%) ----------------------------------
        overload = max(0.0, (cpu_util - self.CPU_OVERLOAD_THRESHOLD) /
                       (1.0 - self.CPU_OVERLOAD_THRESHOLD))

        # -- Latency cost --------------------------------------------------
        latency_cost = latency_norm

        # -- Security gain (restricting rate when threat is high) ----------
        restrictiveness = 1.0 - (L / self.MAX_RATE)
        security_gain   = float(threat_level * restrictiveness)

        reward = (self.W1 * (1.0 - reject_legit)
                  - self.W2 * overload
                  - self.W3 * latency_cost
                  + self.W4 * security_gain)

        return float(reward), {
            "legit_term"   : round(self.W1 * (1.0 - reject_legit), 4),
            "overload_term": round(-self.W2 * overload,              4),
            "latency_term" : round(-self.W3 * latency_cost,          4),
            "security_term": round(self.W4 * security_gain,          4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent implementing the paper's DRL-based rate limiter.

    Architecture
    ------------
    Implements Double DQN with:
        - Online network   : trained every step (after warm-up)
        - Target network   : copied from online every target_update steps
        - Experience replay: uniform sampling from circular buffer
        - Epsilon-greedy   : exponential decay from eps_start -> eps_end

    The agent wraps both the NeuralQNetwork (value function approximator)
    and the SmartGridMDP (reward / action semantics).

    Usage
    -----
        agent  = DQNAgent()
        action = agent.select_action(state)
        # ... environment step ...
        agent.store(state, action, reward, next_state, done)
        loss = agent.update()
    """

    def __init__(
        self,
        state_dim:     int   = SmartGridMDP.STATE_DIM,
        n_actions:     int   = SmartGridMDP.N_ACTIONS,
        lr:            float = 1e-3,
        gamma:         float = 0.95,
        eps_start:     float = 1.00,
        eps_end:       float = 0.05,
        eps_decay:     float = 0.999,
        batch_size:    int   = 64,
        buffer_size:   int   = 10_000,
        target_update: int   = 100,
        seed:          int   = 42,
    ) -> None:
        np.random.seed(seed)

        self.gamma        = gamma
        self.epsilon      = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.batch_size   = batch_size
        self.target_update= target_update
        self._step        = 0

        self.online_net = NeuralQNetwork(state_dim, n_actions, lr)
        self.target_net = NeuralQNetwork(state_dim, n_actions, lr)
        self.target_net.copy_weights_from(self.online_net)

        self.replay = ReplayBuffer(buffer_size)
        self.mdp    = SmartGridMDP()

        # Training history
        self.losses:   list = []
        self.epsilons: list = []

    # ── Decision ────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.mdp.N_ACTIONS))
        q = self.online_net.predict(state)
        return int(np.argmax(q))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return all Q-values for a state (for logging)."""
        return self.online_net.predict(state)

    # ── Experience storage ───────────────────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool = False,
    ) -> None:
        self.replay.push(state, action, reward, next_state, done)

    # ── Learning update ──────────────────────────────────────────────────────

    def update(self) -> float | None:
        """
        One training step:
            1. Sample mini-batch from replay buffer.
            2. Compute Double-DQN Bellman targets.
            3. Train online network.
            4. Decay epsilon.
            5. Periodic target-network sync.

        Returns scalar loss, or None if buffer not yet warm.
        """
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay.sample(self.batch_size)

        # Double DQN: online net selects best action, target net evaluates it
        q_online_next  = self.online_net.forward(next_states)
        best_actions   = np.argmax(q_online_next, axis=1)
        q_target_next  = self.target_net.forward(next_states)
        q_next_vals    = q_target_next[np.arange(self.batch_size), best_actions]

        bellman_targets = (rewards
                           + self.gamma * q_next_vals * (1.0 - dones))

        loss = self.online_net.train_on_batch(states, actions, bellman_targets)

        # Epsilon decay
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        # Periodic target sync
        self._step += 1
        if self._step % self.target_update == 0:
            self.target_net.copy_weights_from(self.online_net)

        self.losses.append(loss)
        self.epsilons.append(self.epsilon)
        return loss

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        """True while in exploration-heavy phase (epsilon > 0.10)."""
        return self.epsilon > 0.10

    def __repr__(self) -> str:
        return (
            f"DQNAgent(epsilon={self.epsilon:.3f}, "
            f"steps={self._step}, buffer={len(self.replay)})"
        )
