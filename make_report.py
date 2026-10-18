"""Builds report.pdf from results/metrics.json and the plots in results/.

Usage:  python make_report.py --name "Your Name" --repo https://github.com/<user>/<repo>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RESULTS = Path("results")

# Built-in PDF fonts (Helvetica/Courier) have no Greek or math glyphs (α, γ, ε, ≥, →), so embed the
# DejaVu fonts that ship with matplotlib.
_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _name, _file in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
                     ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"), ("DejaVuMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(_name, str(_FONTS / _file)))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                   boldItalic="DejaVu-BoldOblique")


def load() -> dict:
    with open(RESULTS / "metrics.json", encoding="utf-8") as fh:
        return json.load(fh)


def build(name: str, repo: str, out: str) -> None:
    m = load()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DejaVu", fontSize=9.5, leading=13.5, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=13, spaceBefore=10, spaceAfter=4,
                        textColor=colors.HexColor("#b5401e"))
    cap = ParagraphStyle("cap", parent=body, fontSize=8.5, textColor=colors.HexColor("#555555"), alignment=1)
    mono = ParagraphStyle("mono", parent=body, fontName="DejaVuMono", fontSize=8.5, leading=11)
    story = []

    def P(text, style=body):
        story.append(Paragraph(text, style))

    def H(text):
        story.append(Paragraph(text, h1))

    def fig(fname, caption, width=6.9):
        path = RESULTS / fname
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(str(path)).getSize()
        story.append(Image(str(path), width=width * inch, height=width * inch * ih / iw))
        P(caption, cap)

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
            ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    def ep(v):
        return "never" if v is None else str(v)

    def sweep_table(section: dict, first_col: str):
        rows = [[first_col, "Episodes until\nMA reward ≥ 0", "Mean train\nreward (all eps)",
                 "Train reward\nlast 200 (± std)", "Greedy eval\nreward", "Eval success\nrate", "Eval\nsteps"]]
        for k, s in section.items():
            rows.append([k, ep(s["episodes_to_ma_reward_0"]), f'{s["mean_train_reward_all"]:.1f}',
                         f'{s["mean_train_reward_last200"]:.2f} ± {s["std_train_reward_last200"]:.2f}',
                         f'{s["eval_mean_reward"]:.2f}', f'{100 * s["eval_success_rate"]:.1f}%',
                         f'{s["eval_mean_steps"]:.1f}'])
        table(rows)

    seeds, n_eval = m["seeds"], m["eval_episodes"]
    b = m["C_baseline"]
    D, E, F, G, Hn = m["D_alpha"], m["E_gamma"], m["F_epsilon_decay"], m["G_best"], m["H_noise"]

    # ---------------------------------------------------------------- title
    P("CPSC 4420/6420: Artificial Intelligence — Assignment 2", ParagraphStyle(
        "t", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=16, textColor=colors.HexColor("#b5401e")))
    P(f"<b>Name:</b> {name} &nbsp;&nbsp;&nbsp; <b>Environment:</b> Gymnasium Taxi-v4 &nbsp;&nbsp;&nbsp; "
      f"<b>Method:</b> tabular Q-learning")
    P(f'<b>Code repository:</b> <link href="{repo}" color="blue">{repo}</link> '
      "(code, README with setup/run instructions, all plots, metrics.json and the MP4 video "
      "<font face='DejaVuMono'>results/taxi_agent.mp4</font>).")
    P(f"All learning curves are averaged over <b>{seeds} independent seeds</b> (shaded band = ±1 std across "
      f"seeds, curves smoothed with a 50-episode moving average). Every \"greedy eval\" number is the policy run "
      f"with exploration disabled (ε = 0) for <b>{n_eval} episodes</b> per seed.")

    # ---------------------------------------------------------------- A
    H("(A) Environment description")
    P("<b>Goal.</b> Taxi (Dietterich, 2000) is a 5×5 grid world with four marked locations R(0,0), G(0,4), "
      "Y(4,0) and B(4,3). A passenger waits at one of the four locations and wants to go to a different one. The "
      "taxi must drive to the passenger, pick them up, drive to the destination and drop them off, in as few "
      "steps as possible. Internal walls block some east/west moves.")
    P("<b>State / observation space.</b> Discrete(500). The state encodes (taxi row, taxi column, passenger "
      "location, destination) as ((row·5 + col)·5 + passenger)·4 + destination, where passenger ∈ {R, G, Y, B, "
      "in-taxi} and destination ∈ {R, G, Y, B}: 25 × 5 × 4 = 500 states. Episodes start uniformly in one of "
      "300 states (passenger not in the taxi and not at its destination). Because the space is already discrete "
      "and small, <b>no discretization is needed</b>. The Q-table is 500 × 6.")
    P("<b>Action space.</b> Discrete(6): 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = drop-off.")
    P("<b>Transition dynamics.</b> Deterministic under the default settings used here (<font face='DejaVuMono'>"
      "is_rainy=False, fickle_passenger=False</font>). Moves change the taxi position by one cell; moving into a "
      "wall or the border leaves the taxi in place. Pickup succeeds only when the taxi is on the passenger's "
      "cell; drop-off succeeds at the destination (dropping the passenger at another marked location just "
      "moves them there). The only randomness is the initial state.")
    P("<b>Reward structure.</b> −1 per step; +20 for delivering the passenger at the destination; −10 for an "
      "illegal pickup or drop-off. A shortest delivery takes roughly 8–16 steps, so the optimal episode reward "
      "is about +5 to +15 (≈ +8 on average).")
    P("<b>Termination.</b> The episode <i>terminates</i> when the passenger is dropped at the destination. It is "
      "<i>truncated</i> by the TimeLimit wrapper after 200 steps (so the worst possible return is about −2000 "
      "if every action were an illegal pickup, and −200 for an aimless wander). "
      "<b>Note:</b> current Gymnasium (1.3+) registers this environment as <font face='DejaVuMono'>Taxi-v4</font>; "
      "v4 only fixed the optional rainy/fickle variants and is identical to Taxi-v3 with default settings.")

    # ---------------------------------------------------------------- B
    H("(B) Q-learning implementation")
    P("The agent is in <font face='DejaVuMono'>qlearning.py</font> and is written from scratch (no RL libraries "
      "other than Gymnasium for the environment):")
    for item in [
        "<b>Hyperparameters</b> live in one dataclass, <font face='DejaVuMono'>QConfig(alpha, gamma, epsilon_start, "
        "epsilon_min, epsilon_decay, episodes, q_init, max_steps)</font>, so every experiment is a one-line "
        "<font face='DejaVuMono'>replace(BASELINE, alpha=0.5)</font>.",
        "<b>Q-value initialization:</b> a 500 × 6 NumPy array filled with <font face='DejaVuMono'>q_init</font> "
        "(0 by default).",
        "<b>ε-greedy policy:</b> with probability ε pick a uniformly random action, otherwise argmax<sub>a</sub> "
        "Q(s,a) with random tie-breaking (important early on when a whole row is 0).",
        "<b>Exploration schedule:</b> ε starts at ε<sub>start</sub> and is multiplied by ε-decay after every "
        "episode, floored at ε<sub>min</sub>.",
        "<b>Update:</b> Q(s,a) ← Q(s,a) + α [ r + γ max<sub>a'</sub> Q(s',a') − Q(s,a) ].",
        "<b>Terminal-state handling:</b> when the step <i>terminates</i> (passenger delivered) the bootstrap "
        "term is dropped, i.e. the target is just r. When the episode is only <i>truncated</i> by the 200-step "
        "limit, s' is not terminal, so we still bootstrap from max Q(s',·). Treating truncation as termination "
        "would wrongly teach the agent that some states are worth 0.",
        "<b>Training loop:</b> reset → (act, step, update) until terminated/truncated → record reward, length and "
        "success → decay ε. <b>Evaluation:</b> pure greedy policy, ε = 0, separate seeds.",
    ]:
        P("• " + item)
    P("Core update (verbatim from the code):", body)
    P("future = 0.0 if terminated else gamma * max(Q[s_next])<br/>"
      "Q[s, a] += alpha * (r + future - Q[s, a])", mono)

    # ---------------------------------------------------------------- C
    H("(C) Baseline training")
    bc = m["baseline"]
    P(f"<b>Baseline:</b> α = {bc['alpha']}, γ = {bc['gamma']}, ε<sub>start</sub> = {bc['epsilon_start']}, "
      f"ε-decay = {bc['epsilon_decay']} per episode, ε<sub>min</sub> = {bc['epsilon_min']}, "
      f"{bc['episodes']} episodes, Q initialized to 0. With this decay, ε reaches its floor of 0.01 after "
      "about 920 episodes. These are standard textbook defaults: a conservative learning rate, a "
      "far-sighted discount, and fully random exploration at first.")
    fig("C_baseline.png", "Figure 1 – Baseline training reward per episode (grey: one raw seed; blue: moving "
        "average over seeds; red dashed: ε schedule).")
    P(f"<b>Initial results.</b> The first episodes are almost random walks that hit the 200-step limit "
      "with many illegal pickups/drop-offs (reward ≈ −700 to −800). As Q-values propagate back from the +20 "
      f"delivery and ε falls, reward climbs steadily. The 50-episode moving average first reaches 0 at episode "
      f"<b>{ep(b['episodes_to_ma_reward_0'])}</b>, and the last 200 training episodes average "
      f"<b>{b['mean_train_reward_last200']:.2f}</b>, close to optimal. However, the greedy evaluation "
      f"from random start states gives only <b>{b['eval_mean_reward']:.2f}</b> mean reward with a "
      f"<b>{100 * b['eval_success_rate']:.1f}%</b> success rate. With α = 0.1, Q-values move slowly, and after "
      "2000 episodes some rarely visited state–action pairs still have values that make the greedy policy "
      "loop (for example between two cells) until the 200-step limit. So the baseline <i>looks</i> converged "
      "on the training curve but is not yet reliable. That motivates the sweeps below.")

    # ---------------------------------------------------------------- D
    H("(D) Learning-rate experiment")
    P("α ∈ {0.01, 0.1, 0.5, 0.9}; everything else fixed at the baseline.")
    fig("D_alpha.png", "Figure 2 – Effect of the learning rate α (left: full run; right: zoom near the optimum).")
    sweep_table(D, "α")
    a01, a05, a09 = D["α = 0.01"], D["α = 0.5"], D["α = 0.9"]
    P(f"<b>Learning speed.</b> Larger α learns much faster: α = 0.9 reaches a non-negative moving average after "
      f"{ep(a09['episodes_to_ma_reward_0'])} episodes and α = 0.5 after {ep(a05['episodes_to_ma_reward_0'])}, "
      f"versus {ep(b['episodes_to_ma_reward_0'])} for α = 0.1. α = 0.01 never gets there in 2000 episodes "
      f"(final training reward ≈ {a01['mean_train_reward_last200']:.0f}, greedy success "
      f"{100 * a01['eval_success_rate']:.1f}%). Each update moves Q only 1% toward the target, so "
      "information from the +20 reward barely propagates back along the ~10-step paths.")
    P(f"<b>Stability.</b> Usually a high α is unstable because it chases noisy targets. Taxi is "
      f"<i>deterministic</i>, though: every (s,a) always gives the same r and s', so the TD target has no "
      f"sampling noise, and α close to 1 just copies the (correct) target. The across-seed bands for α = 0.5 "
      f"and 0.9 are narrow, and their last-200-episode standard deviations ({a05['std_train_reward_last200']:.2f}, "
      f"{a09['std_train_reward_last200']:.2f}) are no larger than α = 0.1's. The remaining variance comes from "
      "the random start state and the 1% residual exploration, not from the learning rate. (Part H shows "
      "this reverses once the environment becomes stochastic.)")
    P(f"<b>Final performance.</b> α = 0.5 and α = 0.9 give near-optimal greedy policies "
      f"({a05['eval_mean_reward']:.2f} and {a09['eval_mean_reward']:.2f} mean reward, "
      f"{100 * a05['eval_success_rate']:.1f}% / {100 * a09['eval_success_rate']:.1f}% success, about 13 steps). "
      f"α = 0.1 is still under-trained ({b['eval_mean_reward']:.2f}), and α = 0.01 has failed.")

    # ---------------------------------------------------------------- E
    H("(E) Discount-factor experiment")
    P("γ ∈ {0.5, 0.9, 0.99}; everything else fixed at the baseline (α = 0.1).")
    fig("E_gamma.png", "Figure 3 – Effect of the discount factor γ.")
    sweep_table(E, "γ")
    g05, g09, g099 = E["γ = 0.5"], E["γ = 0.9"], E["γ = 0.99"]
    P("<b>Immediate vs. future reward.</b> The only positive reward (+20) comes at the end of a multi-step "
      "plan, while every intermediate step costs −1. With γ = 0.5, the delivery reward seen k steps ahead is "
      "worth 20·0.5<sup>k</sup>: only 0.6 at k = 5 and 0.02 at k = 10. Meanwhile the discounted cost of "
      "any infinite wander is bounded by −1/(1−γ) = −2. Far from the goal, all four movement actions therefore "
      "have almost identical Q-values (≈ −2). The agent is essentially myopic and gets no useful signal "
      f"about which way to drive. Its greedy policy delivers in only {100 * g05['eval_success_rate']:.1f}% of "
      f"evaluation episodes (mean reward {g05['eval_mean_reward']:.2f}), and its training curve never reaches 0.")
    P(f"γ = 0.9 has a 10-step effective horizon, which is about one Taxi trip, so it learns a usable policy "
      f"({100 * g09['eval_success_rate']:.1f}% success). It is still worse than γ = 0.99 "
      f"({100 * g099['eval_success_rate']:.1f}%, eval reward {g099['eval_mean_reward']:.2f} vs "
      f"{g09['eval_mean_reward']:.2f}) because value differences between good and bad routes from distant "
      "states stay small and are easily swamped by the not-yet-converged parts of the table. A discount "
      "close to 1 matches the task: the agent should care about the whole trip, and the −1 step cost already "
      "rewards short paths, so no extra discounting is needed for that.")

    # ---------------------------------------------------------------- F
    H("(F) Additional hyperparameter: ε-decay rate")
    P("ε-decay ∈ {0.95, 0.99, 0.995, 0.999} per episode (ε reaches 0.01 after about 90, 460, 920 and "
      "4600 episodes respectively), all other settings at the baseline.")
    fig("F_epsilon_decay.png", "Figure 4 – Effect of the ε-decay rate (exploration schedule).")
    sweep_table(F, "ε-decay")
    f95, f999 = F["ε-decay = 0.95"], F["ε-decay = 0.999"]
    P(f"<b>Effect on learning.</b> This is the exploration/exploitation trade-off. Fast decay (0.95) stops "
      f"exploring after about 90 episodes and has the best average training reward "
      f"({f95['mean_train_reward_all']:.1f}), since fewer random actions are wasted. But its greedy policy "
      f"varies a lot between seeds (see the per-seed values in metrics.json), because some states were "
      f"never explored enough. Slow decay (0.999) keeps ε above 0.13 for the whole 2000-episode run, so "
      f"the training reward stays low ({f999['mean_train_reward_last200']:.2f} over the last 200 episodes, "
      f"with high variance) and its moving average never reaches 0. "
      f"Its greedy policy is the best of the four, though ({f999['eval_mean_reward']:.2f}, "
      f"{100 * f999['eval_success_rate']:.1f}% success), because broad exploration visits more of the "
      "300 start states. With α = 0.1 none of the schedules is fully converged in 2000 episodes; "
      "the learning rate, not the exploration schedule, is the bottleneck for this budget.")

    # ---------------------------------------------------------------- G
    H("(G) Best agent")
    sel = G["selected_config"]
    P("<b>Selection procedure.</b> (1) From each single-parameter sweep (D, E, F) take the value with the "
      "highest greedy evaluation reward averaged over seeds. Evaluation-reward differences smaller than "
      f"{G['tie_tolerance']} are treated as ties (that is within evaluation noise), and ties go to the setting "
      "that learned fastest (highest mean training reward over all episodes). The winners were "
      f"α = {G['best_values_per_sweep']['alpha']}, γ = {G['best_values_per_sweep']['gamma']}, "
      f"ε-decay = {G['best_values_per_sweep']['epsilon_decay']}. (2) Re-train candidate combinations "
      "(baseline, all three combined, best α only, best ε-decay only) and apply the same rule.")
    fig("G_candidates.png", "Figure 5 – Candidate configurations for the best agent.")
    sweep_table(G["candidates"], "Candidate")
    P(f"\"Combined\" and \"best α only\" tie at the optimum in greedy reward, but \"combined\" inherits the slow "
      f"0.999 exploration schedule and needs about 4× as many episodes to learn. The <b>selected agent is "
      f"\"{G['selected']}\"</b>: α = {sel['alpha']}, γ = {sel['gamma']}, ε: {sel['epsilon_start']} → "
      f"{sel['epsilon_min']} with decay {sel['epsilon_decay']}, {sel['episodes']} episodes.")
    fb, fbase, fopt = G["final_eval_best"], G["final_eval_baseline"], G["final_eval_optimal_value_iteration"]
    P(f"<b>Final evaluation</b> (seed-0 agents, the same {fb['episodes']} start states for every policy, ε = 0). "
      "As a reference upper bound, the table also shows the true optimal policy computed by value iteration on "
      "the known Taxi model (not used for training).")
    rows = [["Policy", "Mean reward ± std", "Success rate", "Mean steps", "Illegal actions / ep"]]
    for label, e in [("Best Q-learning agent", fb), ("Baseline agent", fbase), ("Optimal (value iteration)", fopt)]:
        rows.append([label, f"{e['mean_reward']:.2f} ± {e['std_reward']:.2f}", f"{100 * e['success_rate']:.0f}%",
                     f"{e['mean_steps']:.2f}", f"{e['mean_illegal_actions']:.2f}"])
    table(rows)
    P(f"The best agent delivers the passenger in <b>{100 * fb['success_rate']:.0f}%</b> of episodes, with mean "
      f"reward <b>{fb['mean_reward']:.2f}</b> in {fb['mean_steps']:.2f} steps on average and zero illegal "
      f"actions. That is within {fopt['mean_reward'] - fb['mean_reward']:.2f} reward of the optimal policy on "
      f"the same start states, and its greedy action is optimal in "
      f"{100 * G['policy_agreement_with_optimal']:.1f}% of the states it visited during training. The "
      f"baseline reaches only {fbase['mean_reward']:.2f} ({100 * fbase['success_rate']:.0f}% success). Its "
      f"successful trips are just as short ({fbase['mean_steps_successful']:.1f} steps), but "
      f"{100 - 100 * fbase['success_rate']:.0f}% of starts get stuck in a loop until the 200-step limit, "
      "which drags the mean down. Raising α from 0.1 to 0.9 fixes this within the same 2000-episode budget. "
      "The video <font face='DejaVuMono'>results/taxi_agent.mp4</font> shows the best agent on 6 new episodes.")

    # ---------------------------------------------------------------- H
    H("(H) Stochastic environment (action noise)")
    hc = Hn["config"]
    P("<b>Modification.</b> Taxi is deterministic by default, so I added a controllable noise source as a "
      "Gymnasium <font face='DejaVuMono'>ActionWrapper</font> (<font face='DejaVuMono'>NoisyActionWrapper</font>). "
      "With probability p, the agent's intended action is replaced by a <i>different</i> valid action, chosen "
      "uniformly from the other 5. The agent is not told this happened. Noise levels p ∈ {0, 0.10, 0.25}. "
      f"<b>Settings:</b> α = {hc['alpha']}, γ = {hc['gamma']}, ε-decay = {hc['epsilon_decay']}, "
      f"{hc['episodes']} episodes. I deliberately did <i>not</i> reuse α = 0.9 here. With noisy transitions, "
      "a large step size lets a single unlucky sample (for example a random illegal pickup costing −10) "
      "overwrite a Q-value. A check run of the α = 0.9 configuration under noise reached only "
      + ", ".join(f"{s['eval_mean_reward']:.1f} eval reward at {k}" for k, s in
                  Hn["best_cfg_high_alpha_under_noise"].items())
      + ". A small α averages over the noise instead. This is the stability trade-off that part D could "
      "not show in the deterministic setting.")
    fig("H_noise_training.png", "Figure 6 – Training curves under action noise p (each policy is trained and its "
        "training reward measured at its own noise level).")
    sweep_table(Hn["runs"], "Noise p")
    fig("H_robustness.png", "Figure 7 – Cross-evaluation: each trained policy (bar color) evaluated greedily at "
        "every noise level (x-axis). Black marks: optimal policy for that noise level (value iteration on the "
        "noisy model), i.e. the best any policy can do.", width=5.6)
    lv = Hn["noise_levels"]
    R, S = Hn["cross_eval_mean_reward"], Hn["cross_eval_success_rate"]
    rows = [["Trained at \\ Evaluated at"] + [f"p = {p:.2f}" for p in lv]]
    for i, pt in enumerate(lv):
        rows.append([f"p = {pt:.2f}"] + [f"{R[i][j]:.2f}  ({100 * S[i][j]:.0f}%)" for j in range(len(lv))])
    rows.append(["Optimal for that p"] + [
        f"{Hn['optimal_reference'][f'p = {p:.2f}']['mean_reward']:.2f}  (100%)" for p in lv])
    table(rows)
    r = Hn["runs"]
    P(f"<b>Learning speed.</b> Noise slows learning substantially. The moving-average reward reaches 0 after "
      f"{ep(r['p = 0.00']['episodes_to_ma_reward_0'])} episodes at p = 0, after "
      f"{ep(r['p = 0.10']['episodes_to_ma_reward_0'])} at p = 0.10, and "
      f"{'never' if r['p = 0.25']['episodes_to_ma_reward_0'] is None else 'after ' + str(r['p = 0.25']['episodes_to_ma_reward_0'])} "
      "at p = 0.25. Part of this is real (it is harder to reach the goal), but the TD targets are also "
      "noisy, so with a small α many more samples are needed before Q-values settle. The variance across "
      "episodes and seeds also grows clearly with p (wider bands in Figure 6).")
    P(f"<b>Final reward.</b> Even the <i>optimal</i> achievable reward drops from "
      f"{Hn['optimal_reference']['p = 0.00']['mean_reward']:.2f} (p = 0) to "
      f"{Hn['optimal_reference']['p = 0.10']['mean_reward']:.2f} (p = 0.10) and "
      f"{Hn['optimal_reference']['p = 0.25']['mean_reward']:.2f} (p = 0.25). Random substitutions waste steps "
      f"and sometimes trigger −10 illegal pickups/drop-offs that the agent cannot avoid. The learned policies "
      f"follow the same trend ({r['p = 0.00']['eval_mean_reward']:.2f} → {r['p = 0.10']['eval_mean_reward']:.2f} → "
      f"{r['p = 0.25']['eval_mean_reward']:.2f} at their own noise level). Success stays at or near 100%, so "
      "the agent still delivers the passenger; it just takes longer and pays more penalties.")
    P(f"<b>Robustness of the original p = 0 policy.</b> The policy trained without noise remains effective: "
      f"evaluated at p = 0.10 and 0.25 it gets {R[0][1]:.2f} and {R[0][2]:.2f}, with "
      f"{100 * S[0][1]:.0f}% / {100 * S[0][2]:.0f}% success. That is close to the optimal reward for those "
      "noise levels (black marks in Figure 7). The reason is that this noise is state-independent: an action that was best "
      "without noise is still (almost always) best when every action is equally likely to be corrupted. "
      "The noise lowers the achievable return but hardly changes the optimal policy, and a greedy tabular "
      "policy simply re-plans from whatever state it ends up in. In this experiment the p = 0 policy "
      f"actually matches or beats the noise-trained ones ({R[2][2]:.2f} for the p = 0.25-trained policy at "
      f"p = 0.25), because noisy training is less converged after the same {hc['episodes']} episodes. The "
      f"p = 0.25-trained policy also drops to {R[2][0]:.2f} ({100 * S[2][0]:.0f}% success) when the noise is "
      "removed, since a few of its Q-values are still poor estimates. Overall, uniform action noise mainly "
      "costs <i>sample efficiency</i> and <i>achievable reward</i>, not policy quality. A noise source that "
      "changes which action is best (for example wind pushing in one direction) would make noise-aware "
      "training more important.")

    # ---------------------------------------------------------------- conclusion
    H("Conclusions")
    P("Tabular Q-learning solves Taxi to optimality within 2000 episodes when the hyperparameters fit the "
      "problem. (1) Because the environment is deterministic, a large learning rate (α = 0.5–0.9) is both fast "
      "and stable, while α = 0.1 looks converged on the training curve but leaves a ~15% failure rate, and "
      "α = 0.01 fails. (2) A far-sighted discount (γ = 0.99) is essential because the only positive reward "
      "comes at the end of the trip; γ = 0.5 makes the agent myopic. (3) The exploration schedule trades "
      "training reward against state coverage. (4) Once the environment becomes stochastic, the "
      "learning-rate trade-off reverses: small α is needed to average out noise, learning slows down, and "
      "the achievable reward drops, but the policy learned without noise stays near-optimal under uniform "
      "action noise.")

    SimpleDocTemplate(out, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                      topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                      title="CPSC 4420 Assignment 2 - Q-learning on Taxi").build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="____________________")
    ap.add_argument("--repo", default="https://github.com/USERNAME/REPO")
    ap.add_argument("--out", default="report.pdf")
    args = ap.parse_args()
    build(args.name, args.repo, args.out)
