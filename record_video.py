"""Render the trained (greedy) Taxi agent to an MP4.

Usage:  python record_video.py                       (uses results/best_q.npy)
        python record_video.py --episodes 5 --out results/taxi_agent.mp4
"""
from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np

from qlearning import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default="results/best_q.npy")
    parser.add_argument("--out", default="results/taxi_agent.mp4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    q = np.load(args.q)
    env = make_env(render_mode="rgb_array")
    frames = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        frames.append(env.render())
        total, done, t = 0.0, False, 0
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            frames.append(env.render())
            total += reward
            t += 1
            done = terminated or truncated
        frames.extend([frames[-1]] * args.fps)  # brief pause between episodes
        print(f"episode {ep + 1}: reward {total:.0f}, steps {t}, delivered={terminated}")
    env.close()
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
