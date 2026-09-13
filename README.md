# Ms. Pac-Man DQN — Class 3 (Fundamentals of Agentic AI)

This is my submission for the Class 3 assignment: train a DQN to play Ms. Pac-Man, using the
class's starter notebook, and explain what actually happened.

## How to run this

Open [pacman_dqn.ipynb](pacman_dqn.ipynb) in Google Colab (or local Jupyter/VS Code with a
Python 3.11–3.13 kernel), select a GPU under Runtime → Change runtime type if you're in Colab,
and choose Run All. Section 1 already has my three chosen values in it.

I ran this in Google Colab, on their T4 GPU when available (PyTorch 2.11.0+cu128, Python 3.13.15,
Linux).

## My three hyperparameters — and why I ended up here

My final values: **exploration = 0.20, episodes = 25, learning rate = 0.00002.**

I didn't land on these straight away — I actually went through a few rounds of testing, and the
learning rate changed along the way. Here's the reasoning:

- **Exploration (0.20):** kept the notebook's starting point the whole time. In class we talked
  about how 0% exploration locks the agent into a suboptimal path, so I wanted to leave a real
  chunk of moves exploring rather than just exploiting whatever the network had already learned.
- **Episodes (25):** enough to pass the assignment's 25-episode threshold for intermediate
  GIFs/checkpoints, without turning every test run into a 10+ minute wait.
- **Learning rate (0.00002, down from 0.0001):** I started at 0.0001, the notebook's suggested
  reference point. It didn't work well (see below), and once I noticed the problem looked like
  the "learning rate too high, overshoots and never converges" issue from lecture, I tried a
  smaller one instead. That's the version I'm submitting.

## What I expected vs. what actually happened

I expected 25 episodes at the default learning rate (0.0001) to give at least a small bump over
the untrained network. It didn't — the mean evaluation score actually dropped, from 492.0 down to
284.0. So I tried training longer (200 episodes) to see if it just needed more time. It got
*worse*, not better (mean dropped to 212.0), and the loss kept climbing the entire run instead of
leveling off — a sign the network was diverging, not learning.

That pointed at the learning rate itself as the problem, not the episode count. So I dropped it
from 0.0001 to 0.00002, kept exploration and episodes exactly the same, and reran. This time the
mean score only dropped to 446.0 (much closer to the baseline of 492.0), and the loss curve
stayed much flatter instead of climbing continuously. It's the version below.

## Evaluation results — my final run (25 episodes, lr = 0.00002)

This is the third of three experiments I actually ran — see "Other experiments I ran along the
way" further down for the first two (25 episodes and 200 episodes, both at the original
learning rate of 0.0001).

| Game | Before (untrained) | After (trained) |
|---|---|---|
| 1 | 350 | 370 |
| 2 | 500 | 810 |
| 3 | 320 | 340 |
| 4 | 800 | 500 |
| 5 | 490 | 210 |
| **Mean** | **492.0** | **446.0** |

Change in mean score: **−46.0**. Still not a genuine improvement, but a much smaller drop than
either of my earlier attempts — full numbers in [`results/comparison.json`](results/comparison.json).

### Training dashboard

![Training dashboard: score, loss, exploration](results/training_dashboard.png)

Loss rises much more gently here than in my earlier runs and starts to flatten out toward the
end, instead of climbing the whole time — that's the main evidence the lower learning rate
actually helped with the instability.

### Gameplay GIFs

| Untrained (before) | Best of 5, after training |
|---|---|
| ![Untrained gameplay](results/demos/untrained.gif) | ![Best trained gameplay](results/demos/best_trained.gif) |

Since this final run only used 25 episodes, there's no separate "intermediate" checkpoint before
the end — episode 25 is both the intermediate demo point and the final result. (My 200-episode
experiment below does have real intermediate GIFs, since it crossed that threshold several
times.)

## Actual training budget (this run)

- **Completed episodes:** 25 / 25 — not interrupted
- **Total decisions:** 15,084
- **Learning updates:** 3,522
- **Elapsed time:** ~54 seconds (training + periodic evaluation samples)
- **Hardware:** Google Colab, T4 GPU (CUDA), Python 3.13.15
- Full settings in [`config.json`](results/config.json), per-episode log in
  [`training.csv`](results/training.csv), summary in
  [`training_summary.json`](results/training_summary.json)

None of my runs were interrupted, and all of them had real learning updates — even the worst
result (200 episodes) trained for the full budget and updated the network over 30,000 times.

## In plain language

The agent doesn't see the maze as objects — it sees four stacked grayscale game screens, so it
can tell which direction things are moving. It picks one joystick move (up/down/left/right,
diagonals) at each decision point. It's rewarded by the actual points it scores in the game —
eating pellets, power pellets, and ghosts — nothing shaped or invented on top of that.

## One limitation

Even my best run (lr = 0.00002) still ended up slightly below the untrained baseline. 25
episodes and ~15,000 decisions just isn't much training for a raw-pixel Atari agent — the
original DQN paper uses tens of millions of frames. The lower learning rate clearly reduced the
instability I was seeing, but it didn't fully fix it in this short a run.

## Next experiment

I'd combine the two things I learned: keep the lower learning rate (0.00002) but let it train
for longer — 100–200 episodes instead of 25 — to see if the now-more-stable loss curve
eventually turns into an actual improvement in the evaluation score, rather than just a smaller
loss.

## Other experiments I ran along the way

I didn't just run this once — here's the full trail, in case it's useful evidence of the
process:

**25 episodes, learning rate 0.0001 (my original choice):** mean score 492.0 → 284.0
(change −208.0). Full result in [`results/experiment_25ep_lr0001/`](results/experiment_25ep_lr0001/).

**200 episodes, learning rate 0.0001 (testing "does it just need more time?"):** mean score
492.0 → 212.0 (change −280.0) — worse, not better. This run does have real intermediate GIFs and
checkpoints every 25 episodes. Full result in
[`results/experiment_200ep_lr0001/`](results/experiment_200ep_lr0001/).

| Run | Mean before | Mean after | Change | Elapsed time |
|---|---|---|---|---|
| 25 ep, lr=0.0001 | 492.0 | 284.0 | −208.0 | ~49 seconds |
| 200 ep, lr=0.0001 | 492.0 | 212.0 | −280.0 | ~7.6 minutes |
| **25 ep, lr=0.00002 (final)** | **492.0** | **446.0** | **−46.0** | **~54 seconds** |

The lower-learning-rate fix cost basically nothing extra in runtime (54s vs 49s) while cutting
the score drop by more than 4x — compared to the 200-episode run, which took 9x longer and made
things worse.

## Where everything is kept

Model checkpoints (~6-7 MB each per run) aren't in this repo — they're in my local downloaded
ZIPs from each Colab run. This repo has the notebook, the README, and the evidence files
(scores, plots, GIFs, configs) needed to follow along without rerunning anything.

## Sources

- [ALE installation](https://ale.farama.org/getting-started/)
- [Gymnasium Atari preprocessing](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing)
- [Gymnasium frame stacking](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStackObservation)
- [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
