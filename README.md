# Git & GitHub for ML: Demo Repository

A small breast-cancer classifier (scikit-learn) used to show how Git and
GitHub make ML experiments **versioned, reviewable, and reproducible**.

## What the demo does

1. Trains a baseline model (logistic regression) and commits it with its metrics.
2. Creates a branch for a new experiment (random forest) by changing only `params.json`.
3. Uses `git diff` to show exactly what changed: the parameter and the resulting metrics.
4. Switches between branches to show that `main` is untouched by the experiment.
5. Pushes the experiment branch to GitHub.

## Project structure

```
train.py                    # trains model, writes metrics.json
params.json                 # all hyperparameters (one place to change)
metrics.json                # results, committed with the code that produced them
tests/test_train.py         # accuracy guardrail (>= 90%), run with pytest
requirements.txt
```

## Setup

Requires Python 3.9+ and Git.

```bash
git clone https://github.com/<your-username>/git-ml-demo.git
cd git-ml-demo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python train.py     # trains and prints / saves metrics
python -m pytest -v          # runs the accuracy guardrail tests
```

## Try an experiment yourself

```bash
git switch -c experiment/random-forest
# edit params.json: "model": "random_forest"
python train.py
git diff                       # see params + metrics change side by side
git commit -am "Try random forest"
git push -u origin experiment/random-forest
git switch main                # main still has the baseline
```

## Note on data and models

Git is built for code and small text files. For large datasets or model
weights, use [Git LFS](https://git-lfs.com) or [DVC](https://dvc.org)
alongside Git. This demo uses a dataset bundled with scikit-learn, so
no data files are stored in the repo.
