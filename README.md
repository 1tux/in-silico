# In-Silico Entity Knowledge (Paper Repo)

This repository is the paper-ready version of the blog experiments.

## Structure
- `experiments/`: one folder per finding (F1–F6)
- `configs/`: run configs (model, data, seeds)
- `scripts/`: shared utilities (data prep, caching, eval)
- `data/`: local data artifacts (ignored from VCS)
- `results/`: metrics outputs (ignored from VCS)
- `figures/`: generated plots (ignored from VCS)
- `docs/`: experiment specs and paper notes

## Next Steps
- Define experiment specs in `docs/experiments.md`.
- Align with the existing Colab notebooks and port code into `scripts/`.
- Run F1–F3 on Qwen2.5-7B; add cross-model checks after baseline is stable.

