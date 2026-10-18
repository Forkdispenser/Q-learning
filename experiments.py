"""Runs every experiment for the assignment (parts C-H) and writes plots + metrics to results/.

Usage:  python experiments.py            (all parts, ~a few minutes)
        python experiments.py --seeds 3  (faster)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qlearning import QConfig, evaluate, optimal_q, train

RESULTS = Path("results")
WINDOW = 50          # moving-average window for training curves
EVAL_EPISODES = 100  # greedy evaluation episodes (assignment requires >= 30)
TIE_TOL = 0.5       # eval-reward differences smaller than this are treated as ties
BASELINE = QConfig(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01,
                   epsilon_decay=0.995, episodes=2000)


def moving_average(x: np.ndarray, w: int = WINDOW) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def run_config(cfg: QConfig, seeds: int, noise_p: float = 0.0) -> dict:
    """Train over several seeds; return stacked curves, greedy-eval metrics and summary statistics."""
    curves, evals, qs = [], [], []
    for s in range(seeds):
        agent, hist = train(cfg, seed=s, noise_p=noise_p)
        curves.append(hist["rewards"])
        evals.append(evaluate(agent.q, EVAL_EPISODES, seed=1000 + 100 * s, noise_p=noise_p))
        qs.append(agent.q)
    curves = np.array(curves)
    ma = np.array([moving_average(c) for c in curves])
    mean_ma = ma.mean(axis=0)
    # Learning speed: first episode where the seed-averaged moving average reaches 0 reward.
    reach = np.flatnonzero(mean_ma >= 0)
    summary = {
        "config": cfg.to_dict(),
        "noise_p": noise_p,
        "episodes_to_ma_reward_0": int(reach[0]) + 1 if reach.size else None,
        "mean_train_reward_all": float(curves.mean()),
        "mean_train_reward_last200": float(curves[:, -200:].mean()),
        "std_train_reward_last200": float(curves[:, -200:].std()),
        "eval_mean_reward": float(np.mean([e["mean_reward"] for e in evals])),
        "eval_success_rate": float(np.mean([e["success_rate"] for e in evals])),
        "eval_mean_steps": float(np.mean([e["mean_steps"] for e in evals])),
        "eval_per_seed": evals,
    }
    return {"curves": curves, "ma": ma, "summary": summary, "qs": qs}


def plot_curves(runs: dict, title: str, fname: str, ylim=(-800, 20), zoom=(-60, 15)) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for label, run in runs.items():
        mean, std = run["ma"].mean(axis=0), run["ma"].std(axis=0)
        x = np.arange(1, len(mean) + 1)
        for ax in axes:
            ax.plot(x, mean, label=label, lw=1.6)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18)
    axes[0].set_title(f"{title} — full run")
    axes[0].set_ylim(*ylim)
    axes[1].set_title(f"{title} — zoom on converged region")
    axes[1].set_ylim(*zoom)
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.set_ylabel(f"Episode reward ({WINDOW}-ep moving avg, mean ± std over seeds)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=140)
    plt.close(fig)


def part_c(seeds: int) -> dict:
    run = run_config(BASELINE, seeds)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(1, BASELINE.episodes + 1)
    ax.plot(x, run["curves"][0], color="0.75", lw=0.6, label="raw episode reward (seed 0)")
    mean, std = run["ma"].mean(axis=0), run["ma"].std(axis=0)
    ax.plot(x, mean, color="C0", lw=2, label=f"{WINDOW}-ep moving avg (mean over {seeds} seeds)")
    ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.2)
    ax2 = ax.twinx()
    eps = np.maximum(BASELINE.epsilon_min, BASELINE.epsilon_start * BASELINE.epsilon_decay ** np.arange(BASELINE.episodes))
    ax2.plot(x, eps, color="C3", ls="--", lw=1.2, label="ε")
    ax2.set_ylabel("ε (exploration rate)", color="C3")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Baseline Q-learning on Taxi-v4 (α=0.1, γ=0.99, ε: 1.0→0.01, decay 0.995)")
    ax.set_ylim(-800, 30)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "C_baseline.png", dpi=140)
    plt.close(fig)
    return run


def sweep(param: str, values, seeds: int, label_fmt: str) -> dict:
    return {label_fmt.format(v): run_config(replace(BASELINE, **{param: v}), seeds) for v in values}


# Part H uses a smaller learning rate and longer training than the best deterministic config:
# with noisy transitions alpha=0.9 lets single unlucky samples overwrite Q-values (see report).
H_CFG = replace(BASELINE, alpha=0.1, epsilon_decay=0.999, episodes=5000)


def part_h(best_cfg: QConfig, seeds: int, noise_levels=(0.0, 0.10, 0.25)) -> dict:
    runs = {f"p = {p:.2f}": run_config(H_CFG, seeds, noise_p=p) for p in noise_levels}
    plot_curves(runs, "Action-noise level p", "H_noise_training.png", zoom=(-120, 15))

    # Robustness matrix: policy trained at p_train, evaluated (greedy) at p_eval. Averaged over seeds.
    matrix = np.zeros((len(noise_levels), len(noise_levels)))
    succ = np.zeros_like(matrix)
    steps = np.zeros_like(matrix)
    for i, p_train in enumerate(noise_levels):
        run = runs[f"p = {p_train:.2f}"]
        for j, p_eval in enumerate(noise_levels):
            ev = [evaluate(q, EVAL_EPISODES, seed=5000 + 100 * k, noise_p=p_eval) for k, q in enumerate(run["qs"])]
            matrix[i, j] = np.mean([e["mean_reward"] for e in ev])
            succ[i, j] = np.mean([e["success_rate"] for e in ev])
            steps[i, j] = np.mean([e["mean_steps"] for e in ev])

    # Reference: the true optimal policy for each noise level (value iteration on the noisy model).
    optimal = {p: evaluate(optimal_q(H_CFG.gamma, p), 5 * EVAL_EPISODES, seed=9000, noise_p=p) for p in noise_levels}

    # Sensitivity check: the best deterministic config (high alpha) trained directly under noise.
    high_alpha = {f"p = {p:.2f}": run_config(replace(best_cfg, episodes=H_CFG.episodes), seeds, noise_p=p)["summary"]
                  for p in noise_levels if p > 0}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.22
    xs = np.arange(len(noise_levels))
    for i, p_train in enumerate(noise_levels):
        ax.bar(xs + (i - 1) * width, matrix[i], width, label=f"trained at p = {p_train:.2f}")
    ax.scatter(xs, [optimal[p]["mean_reward"] for p in noise_levels], marker="_", s=900, color="k",
               zorder=3, label="optimal policy for that p (value iteration)")
    ax.set_xticks(xs, [f"evaluated at p = {p:.2f}" for p in noise_levels])
    ax.set_ylabel(f"Mean greedy reward, {EVAL_EPISODES} eps × {seeds} seeds")
    ax.set_title("Policy robustness under action noise")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "H_robustness.png", dpi=140)
    plt.close(fig)
    return {
        "config": H_CFG.to_dict(),
        "runs": {k: v["summary"] for k, v in runs.items()},
        "noise_levels": list(noise_levels),
        "cross_eval_mean_reward": matrix.tolist(),
        "cross_eval_success_rate": succ.tolist(),
        "cross_eval_mean_steps": steps.tolist(),
        "optimal_reference": {f"p = {p:.2f}": v for p, v in optimal.items()},
        "best_cfg_high_alpha_under_noise": high_alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    metrics: dict = {"seeds": args.seeds, "eval_episodes": EVAL_EPISODES, "baseline": BASELINE.to_dict()}

    print("[C] baseline")
    base = part_c(args.seeds)
    metrics["C_baseline"] = base["summary"]

    print("[D] learning rate")
    d = sweep("alpha", [0.01, 0.1, 0.5, 0.9], args.seeds, "α = {}")
    plot_curves(d, "Learning rate α", "D_alpha.png")
    metrics["D_alpha"] = {k: v["summary"] for k, v in d.items()}

    print("[E] discount factor")
    e = sweep("gamma", [0.5, 0.9, 0.99], args.seeds, "γ = {}")
    plot_curves(e, "Discount factor γ", "E_gamma.png")
    metrics["E_gamma"] = {k: v["summary"] for k, v in e.items()}

    print("[F] epsilon decay")
    f = sweep("epsilon_decay", [0.95, 0.99, 0.995, 0.999], args.seeds, "ε-decay = {}")
    plot_curves(f, "ε-decay rate", "F_epsilon_decay.png")
    metrics["F_epsilon_decay"] = {k: v["summary"] for k, v in f.items()}

    # [G] Best configuration: combine the best value found in each single-parameter sweep, then
    # rank candidates by greedy evaluation reward, tie-broken by mean training reward (learning speed).
    print("[G] best agent")
    def pick(summaries: dict) -> str:
        """Highest greedy-eval reward; anything within TIE_TOL of it counts as a tie (evaluation noise),
        and ties go to the configuration that learned fastest (highest mean training reward)."""
        top = max(s["eval_mean_reward"] for s in summaries.values())
        tied = [k for k, s in summaries.items() if s["eval_mean_reward"] >= top - TIE_TOL]
        return max(tied, key=lambda k: summaries[k]["mean_train_reward_all"])

    def best_of(sweep_runs):
        return sweep_runs[pick({k: v["summary"] for k, v in sweep_runs.items()})]["summary"]["config"]
    best_alpha = best_of(d)["alpha"]
    best_gamma = best_of(e)["gamma"]
    best_decay = best_of(f)["epsilon_decay"]
    candidates = {
        "baseline": BASELINE,
        "combined": replace(BASELINE, alpha=best_alpha, gamma=best_gamma, epsilon_decay=best_decay),
        "best α only": replace(BASELINE, alpha=best_alpha),
        "best ε-decay only": replace(BASELINE, epsilon_decay=best_decay),
    }
    cand_runs = {k: (base if k == "baseline" else run_config(c, args.seeds)) for k, c in candidates.items()}
    best_name = pick({k: v["summary"] for k, v in cand_runs.items()})
    best_cfg = candidates[best_name]
    best_run = cand_runs[best_name]
    plot_curves({f"{k}" + (" (selected)" if k == best_name else ""): v for k, v in cand_runs.items()},
                "Candidate configurations", "G_candidates.png")

    # Final evaluation of the selected agent (seed 0) and the baseline (seed 0) on the same 100 episodes.
    best_q, base_q = best_run["qs"][0], base["qs"][0]
    np.save(RESULTS / "best_q.npy", best_q)
    q_star = optimal_q(best_cfg.gamma)
    metrics["G_best"] = {
        "selected": best_name,
        "selected_config": best_cfg.to_dict(),
        "selection_rule": pick.__doc__.strip(),
        "tie_tolerance": TIE_TOL,
        "best_values_per_sweep": {"alpha": best_alpha, "gamma": best_gamma, "epsilon_decay": best_decay},
        "candidates": {k: v["summary"] for k, v in cand_runs.items()},
        "final_eval_best": evaluate(best_q, EVAL_EPISODES, seed=777),
        "final_eval_baseline": evaluate(base_q, EVAL_EPISODES, seed=777),
        "final_eval_optimal_value_iteration": evaluate(q_star, EVAL_EPISODES, seed=777),
        "policy_agreement_with_optimal": float(np.mean(
            [q_star[s, np.argmax(best_q[s])] >= q_star[s].max() - 1e-6 for s in range(best_q.shape[0])
             if np.any(best_q[s] != 0)])),
    }

    print("[H] stochastic environment")
    metrics["H_noise"] = part_h(best_cfg, args.seeds)

    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print("done -> results/metrics.json")


if __name__ == "__main__":
    main()
