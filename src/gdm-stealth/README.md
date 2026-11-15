# GDM Stealth — Harbor Adapter

## Overview

Harbor adapter for the stealth evaluations from Google DeepMind's *Evaluating Frontier Models for Stealth and Situational Awareness* (Phuong et al., 2025, [arXiv:2505.01420](https://arxiv.org/abs/2505.01420)), using the open-source Inspect Evals release (`gdm_stealth`) as the upstream implementation.

This adapter generates **13 Harbor tasks** across the 4 released stealth task types, using the upstream FSF-v2 sample filters exactly as `data/dataset.py` applies them by default. It is the stealth half of the paper's 16-eval suite; the 11 situational-awareness evals live in the sibling `gdm-situational-awareness` adapter. Status: **work in progress** — generation and the parity custom agent are implemented (fake-model/fake-rater integration tested, incl. full AI-secretary, cover-your-tracks, and strategic episodes), but oracle verification (Step 3) and real-model parity experiments (Steps 5–6) have not been run. These tasks are **not runnable by a stock CLI agent**: episode state and LLM raters are hosted by the agent harness (see Adapter Features).

## What is GDM Stealth?

The stealth evaluations measure a model's ability to reason about and circumvent oversight — a prerequisite capability for scheming. In `oversight_pattern` the agent must discover which email channels escape monitoring; in `classifier_evasion` it must phrase a secret objective subtly enough to pass an LLM content rater; in `cover_your_tracks` it must complete an email-triage objective and remove the evidence from its own activity log; in `strategic_rule_breaking` it must balance speed against rule-breaking risk in a simulated customer-support job with oversight. The paper describes five stealth evaluations; four task types are present in the Inspect release. Scoring is binary success for the AI-secretary tasks, a fraction of 7 success criteria for `cover_your_tracks`, and a normalized environment score (raw/33) for `strategic_rule_breaking`. The paper ran 50 trials per challenge; Inspect defaults to 10 epochs.

## Adapter Features

- **Pinned upstream.** Generation runs only against inspect_evals commit `a68dade7efb803ef50c8442d3f16264b293d1414` (`--source-dir`, `$GDM_INSPECT_EVALS_DIR`, or a cache clone).
- **Upstream sample filters, replicated and tested.** AI-secretary (tactical: topic `bomb`, 8 failures allowed; subtlety: `content_rater`, 2 failures), `cover_your_tracks` (`no_hint` only by default; `--cover-with-hint` adds `with_hint`), and `strategic_rule_breaking` (turn 100, version 2). The generated sample sets were verified against the upstream registry filter functions themselves.
- **Verbatim prompts.** `instruction.md` is exactly upstream `get_input_prompt()` output; `email_tasks.txt` in AI-secretary tasks is byte-identical to the upstream per-version file. System prompts, tools, message limits, model roles, setup steps, and scoring rules are recorded per task in `tests/task_metadata.json`.
- **Explicit outcome contract.** The parity custom agent must write `/home/agent/gdm_outcome.json` (`success`, normalized `score`, `final_message`); `tests/verify.py` maps it to a reward using the upstream scorer rules. A missing outcome scores 0.
- **No checker leakage.** Upstream checker/rater/environment code is deliberately **not** copied into task containers: upstream it runs host-side, and exposing it in the sandbox would change the measured capability.
- **Custom parity agent (Scenario 3).** `gdm_stealth.custom_agent:GDMStealthParityAgent` hosts the Inspect solvers: the AI-secretary email checker with LLM rater + success judger (ported), the cover-your-tracks email database with content rater and `/log/assistant.log` accounting (ported, pandas replaced by csv), and the customer-support environment (vendored upstream env/oversight/ticket code at the pinned commit with import rewrites and a stdlib RNG shim replacing numpy; the 17-tool layer reimplemented against it). Model roles are pinned per task in `task_metadata.json` (defaults: `google/gemini-2.5-pro-preview-05-06` rater, `google/gemini-2.0-flash` judger/chat/rater). Fake-model integration tests pass for all three hosts; no real-model episode has run yet.

## Generated Task Structure

```
<sanitized-dataset-challenge_name>/
├── task.toml                  # google-deepmind/gdm-stealth__<sample>
├── instruction.md             # upstream input prompt, verbatim
├── environment/
│   ├── Dockerfile             # upstream stealth sandbox + python3 for the verifier
│   └── email_tasks.txt        # AI-secretary tasks only (upstream setup file)
├── solution/
│   └── solve.sh               # explicit no-oracle stub (exits 1), see Notes & Caveats
└── tests/
    ├── test.sh                # runs verify.py, writes /logs/verifier/reward.txt
    ├── verify.py              # outcome-file -> reward mapping (upstream scoring)
    └── task_metadata.json     # prompts, tools, model roles, scoring, outcome contract
```

## Run Evaluation / Harness

Registry (post-publish; not yet published):

```bash
harbor run -d google-deepmind/gdm-stealth -a <gdm-parity-agent> -m <provider/model>
```

Local path (pre-publish; use the custom parity agent import path below — a stock agent scores 0 because no outcome file is written):

```bash
PYTHONPATH=src/gdm-stealth/src harbor run -p datasets/gdm-stealth -a gdm_stealth.custom_agent.agent:GDMStealthParityAgent -m <provider/model>
```

Do **not** use `run_gdm-stealth.yaml` yet: its guide-mandated oracle default deliberately fails because no stealth oracle exists. Use the planned parity config only after team agreement:

```bash
cd src/gdm-stealth
PYTHONPATH=src harbor run -c parity_gdm-stealth.yaml
```

## Usage: Create Task Directories

```bash
cd src/gdm-stealth
uv sync
uv run gdm-stealth --source-dir /path/to/inspect_evals \
    --output-dir /path/to/datasets/gdm-stealth --overwrite
# Include the non-default cover_your_tracks variant:
uv run gdm-stealth --cover-with-hint --output-dir /tmp/gdm-stealth
```

## Comparison with Original Benchmark (Parity)

Parity has **not** been run yet. The Scenario 3 custom agent is implemented; parity still requires pinned rater/judger/chat role models, provider API keys (the default Google raters need `GOOGLE_API_KEY`/`GEMINI_API_KEY` even when the tested model is from another provider), and Step 4 team agreement on agents, models, and trial counts. Cost is a first-order concern: the paper's protocol is 50 trials per challenge and `classifier_evasion`'s capable rater is slow and rate-limited. Planned reporting: per-task upstream metric (binary success / criteria fraction / score÷33) as mean ± sample SEM over ≥3 matched runs per side at a team-agreed epoch count (ADP-PARITY, ADP-STATS), with range overlap as the acceptance criterion.

| Agent | Model | Metric | Runs | Dataset Size | Original (mean ± SEM) | Harbor (mean ± SEM) |
|-------|-------|--------|------|--------------|-----------------------|---------------------|
| gdm-stealth-parity-agent@0.1.0 | TBD | per-task upstream metric | 0 | 13 | not run | not run |

Planned reproduction (shapes only, to be finalized with the team):

```bash
# Original side (Inspect, pinned checkout):
cd inspect_evals
uv run inspect eval inspect_evals/gdm_classifier_evasion --model <provider/model> --epochs 10
# Harbor side:
PYTHONPATH=src/gdm-stealth/src harbor run -p datasets/gdm-stealth -a gdm_stealth.custom_agent.agent:GDMStealthParityAgent -m <provider/model>
```

Links: [paper](https://arxiv.org/abs/2505.01420) · [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) · Adapter PR: TBD · Dataset PR: TBD · HF parity PR: TBD

## Notes & Caveats

- **Counting.** The Adapter List counts this family alongside `gdm-situational-awareness` (whose row describes all 16 evals). This adapter covers only stealth. Upstream `eval.yaml` lists 14 stealth samples (cover_your_tracks: 2), but the task function's default `hard_only=True` yields 13; this adapter follows the code default and offers `--cover-with-hint` for the 14th. The paper's fifth stealth evaluation is not in the Inspect release and is excluded from both adapters.
- **No oracle.** `solution/solve.sh` intentionally exits 1. There is no scripted passing episode against LLM raters and simulated environments; Step 3 oracle verification is blocked on a team decision (e.g. recorded successful agent trajectories from parity-agent runs as reference solutions, per the guide's no-oracle process).
- **Verifier trust boundary.** `verify.py` trusts `/home/agent/gdm_outcome.json`. Only outcomes written by the trusted parity custom agent are meaningful; a terminal agent could forge the file. Do not score CLI-agent runs with this verifier. A CLI-compatible variant would need the tool hosts moved into the container as a separate dataset (Scenario 3a) and a team decision.
- **Strategic system prompt.** The `strategic_rule_breaking` system message embeds a handbook generated from live tool descriptions and cannot be reproduced verbatim at generation time. `task_metadata.json` carries the upstream template + persona text and instructs the parity agent to render it via upstream `prompt.get_system_message()`. Unit-level prompt parity for this task is therefore not yet byte-verified.
- **In-world secrets.** AI-secretary scenarios instruct the agent to transmit fictional secrets (dummy passwords/codes, a redacted dummy API key string in upstream data). They are part of the upstream task design, preserved verbatim in prompts only; nothing real is included.
- **Rater sensitivity.** Upstream notes a less capable rater makes `classifier_evasion` much easier. Any change of rater model from the pinned defaults is a benchmark change and must be applied symmetrically and documented before parity.
- **Container deviation.** Generated Dockerfiles add `python3` (absent upstream) so the Harbor verifier can run inside the task image; upstream needed no interpreter in its file-only sandbox. Runtime network is not required by the tasks; model/rater calls happen host-side in the harness.
- **Sampling vs paper.** Default generation is the Inspect FSF-v2 subset (13 samples), not the paper's full scenario grid (e.g. 125 AI-secretary register entries before filtering). Widening the filters is a parity-scope decision for the team, not a default.

## Installation / Prerequisites

- Python 3.11+ and `uv`; Docker for running generated tasks; `git` for fetching the pinned upstream checkout.
- Generation needs no API keys. Running episodes needs keys for the tested model and for the rater/judger/chat role models (Google by default).

```bash
cd src/gdm-stealth
uv sync
```

## Troubleshooting

- `inspect_evals checkout ... expected pinned commit`: generation is pinned; check out `a68dade7efb803ef50c8442d3f16264b293d1414` in your `--source-dir`, or delete the cache clone at `~/.cache/harbor-adapters/inspect_evals` to re-clone.
- Every reward is 0: expected with any agent other than the parity custom agent — `verify.py` reports `no valid outcome` when `/home/agent/gdm_outcome.json` was never written.
- `with_hint` sample missing: pass `--cover-with-hint` (upstream default is `hard_only=True`).

## Citation

```bibtex
@misc{phuong2025gdmstealth,
  title        = {Evaluating Frontier Models for Stealth and Situational Awareness},
  author       = {Phuong, Mary and Zimmermann, Roland S. and Wang, Ziyue and Lindner, David and Krakovna, Victoria and Cogan, Sarah and Dafoe, Allan and Ho, Lewis and Shah, Rohin},
  year         = {2025},
  eprint       = {2505.01420},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2505.01420}
}
```

## Authors & Contributions

Original benchmark: Mary Phuong, Roland S. Zimmermann, Ziyue Wang, David Lindner, Victoria Krakovna, Sarah Cogan, Allan Dafoe, Lewis Ho, and Rohin Shah (Google DeepMind). Open-source Inspect release: UK AI Security Institute Inspect Evals contributors (Ziyue Wang for `gdm_stealth`). Harbor adapter: work in progress; see `adapter_metadata.json` (`adapter_builders`) and the adapter PR when opened.

## Acknowledgement

Thanks to the Inspect Evals maintainers for the MIT-licensed open-source release that makes a like-for-like parity side possible.
