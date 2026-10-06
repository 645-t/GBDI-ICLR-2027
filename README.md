# Understanding Errors in LLM-Based Question Answering over Imperfect Tables

Anonymous reproduction code, fixed inputs, final prompts and compact results
for the paper's experiments. See [experiment settings](docs/experiments.md).

## Offline reproduction

Use Python 3.12 and run from this directory:

```bash
python -m pip install -r requirements.txt
python -m gbdi verify
python -m gbdi replay --output runs/replay
```

`verify` checks file integrity, cohorts, row orders and all 64 configurations.
`replay` recomputes metrics and paired confidence intervals from saved outputs,
writing CSV tables and `intervals.json`. Neither command calls a model API.

## New experiments

```bash
python -m gbdi list
python -m gbdi run gbdi-qwenplus-random5-skill
```

`run` displays a plan unless `--execute` is supplied. Set the required key in
your environment: `DASHSCOPE_API_KEY`, `OPENROUTER_API_KEY`, or
`DEEPSEEK_API_KEY`. No environment file is loaded automatically.

```bash
python -m gbdi run gbdi-qwenplus-random5-skill --limit 1 --output runs/smoke --execute
python -m gbdi score runs/smoke/gbdi-qwenplus-random5-skill
```

Omit `--limit` for the full cohort and use a separate output directory.
Existing records, including failures, are skipped. Discovery outputs are
cached across guidance conditions to hold the evidence fixed. Code Agent and
GBDI execute model-generated Python; use an isolated environment without
unrelated files. Fresh API outputs may differ from the saved results.

## Files

| Directory | Contents |
|---|---|
| `gbdi/` | Experiment execution, parsing and evaluation |
| `configs/` | System request parameters and experiment configurations |
| `data/` | Tables, reviewed reference, interventions, cohorts and row orders |
| `prompts/` | Final prompts and rule-deletion variants |
| `results/` | Final predictions and numerical verification targets |
| `docs/` | Experimental settings and scoring details |
| `licenses/` | Required upstream license and attribution records |

## Sources

The QA prompts, answer matcher and command interface derive from
[RADAR](https://github.com/kenqgu/RADAR), under the
[Apache 2.0 license](licenses/RADAR-Apache-2.0.txt).
[RADAR-T data](https://huggingface.co/datasets/kenqgu/RADAR) use
[CC BY 4.0](licenses/RADAR-data-CC-BY-4.0.md); the external probes use
[WikiSQL](https://github.com/salesforce/WikiSQL), with its
[BSD notice](licenses/WikiSQL-BSD-3-Clause.txt).
This package adds discovery, intervention, reviewed annotations and evaluation
code. Upstream names identify those resources, not this submission's authors.
