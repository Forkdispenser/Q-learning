"""Tabular Q-learning for Gymnasium's Taxi-v4 environment.

Everything that the experiments need lives here:
  * QConfig            - all hyperparameters in one place (alpha, gamma, epsilon schedule, episodes)
  * QLearningAgent     - Q-table, epsilon-greedy policy and the Q-learning update
  * train / evaluate   - training loop with terminal-state handling, and greedy evaluation
  * NoisyActionWrapper - controllable action noise for the stochastic-environment experiment (part H)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np

ENV_ID = "Taxi-v4"


@dataclass
class QConfig:
    alpha: float = 0.1          # learning rate
    gamma: float = 0.99         # discount factor
    epsilon_start: float = 1.0  # initial exploration rate
    epsilon_min: float = 0.01   # floor of the exploration rate
    epsilon_decay: float = 0.995  # multiplicative decay applied after every episode
    episodes: int = 2000
    q_init: float = 0.0         # initial value of every Q(s, a)
    max_steps: int = 200        # Taxi-v4 truncates episodes at 200 steps

    def to_dict(self) -> dict:
        return asdict(self)


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = rng
        self.q = np.full((n_states, n_actions), cfg.q_init, dtype=np.float64)
        self.epsilon = cfg.epsilon_start

    def greedy_action(self, state: int) -> int:
        # Break ties randomly so an all-zero row does not always pick action 0.
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def act(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)].

        A terminal next state has no future value, so the bootstrap term is dropped.
        Truncation (time limit) is NOT terminal: the state still has future value.
        """
        future = 0.0 if terminated else self.cfg.gamma * np.max(self.q[s_next])
        td_target = r + future
        self.q[s, a] += self.cfg.alpha * (td_target - self.q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


class NoisyActionWrapper(gym.ActionWrapper):
    """With probability p, replace the agent's intended action with a *different* random valid action."""

    def __init__(self, env: gym.Env, p: float, seed: int | None = None):
        super().__init__(env)
        self.p = p
        self.noise_rng = np.random.default_rng(seed)

    def action(self, action):
        if self.p > 0 and self.noise_rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            return int(self.noise_rng.choice(others))
        return action


def make_env(noise_p: float = 0.0, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if noise_p > 0:
        env = NoisyActionWrapper(env, noise_p, seed=None if seed is None else seed + 10_000)
    return env


def train(cfg: QConfig, seed: int = 0, noise_p: float = 0.0) -> tuple[QLearningAgent, dict]:
    """Train a Q-learning agent. Returns the agent and per-episode training history."""
    env = make_env(noise_p, seed)
    rng = np.random.default_rng(seed)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, cfg, rng)

    rewards = np.zeros(cfg.episodes)
    steps = np.zeros(cfg.episodes, dtype=int)
    successes = np.zeros(cfg.episodes, dtype=bool)
    epsilons = np.zeros(cfg.episodes)

    for ep in range(cfg.episodes):
        state, _ = env.reset(seed=seed if ep == 0 else None)
        total, done, t = 0.0, False, 0
        terminated = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total += reward
            t += 1
            done = terminated or truncated or t >= cfg.max_steps
        rewards[ep], steps[ep], successes[ep] = total, t, terminated
        epsilons[ep] = agent.epsilon
        agent.decay_epsilon()

    env.close()
    return agent, {"rewards": rewards, "steps": steps, "successes": successes, "epsilons": epsilons}


def evaluate(q: np.ndarray, episodes: int = 100, seed: int = 12345, noise_p: float = 0.0,
             max_steps: int = 200) -> dict:
    """Run the greedy policy (exploration disabled) and report reward / success / step metrics."""
    env = make_env(noise_p, seed)
    rewards, steps, successes, illegal = [], [], [], []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total, t, n_illegal, terminated, done = 0.0, 0, 0, False, False
        while not done:
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            n_illegal += reward == -10
            t += 1
            done = terminated or truncated or t >= max_steps
        rewards.append(total)
        steps.append(t)
        successes.append(terminated)
        illegal.append(n_illegal)
    env.close()
    rewards, steps, successes = np.array(rewards), np.array(steps), np.array(successes)
    return {
        "episodes": episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": float(successes.mean()),
        "mean_steps": float(steps.mean()),
        "mean_steps_successful": float(steps[successes].mean()) if successes.any() else float("nan"),
        "mean_illegal_actions": float(np.mean(illegal)),
    }


def optimal_q(gamma: float = 0.99, noise_p: float = 0.0, tol: float = 1e-10) -> np.ndarray:
    """Value iteration on Taxi's known transition model - used only as a reference upper bound.

    With noise_p > 0 the model matches NoisyActionWrapper: the intended action is executed with
    probability 1 - p, and each of the other actions with probability p / (n_actions - 1).
    """
    env = gym.make(ENV_ID).unwrapped
    n_s, n_a = env.observation_space.n, env.action_space.n
    q = np.zeros((n_s, n_a))
    while True:
        v = q.max(axis=1)
        # Expected one-step backup of each *executed* action, then mix according to the noise model.
        backup = np.zeros_like(q)
        for s in range(n_s):
            for a in range(n_a):
                for prob, s2, r, term in env.P[s][a]:
                    backup[s, a] += prob * (r + (0.0 if term else gamma * v[s2]))
        if noise_p > 0:
            others = (backup.sum(axis=1, keepdims=True) - backup) / (n_a - 1)
            q_new = (1 - noise_p) * backup + noise_p * others
        else:
            q_new = backup
        if np.max(np.abs(q_new - q)) < tol:
            return q_new
        q = q_new
