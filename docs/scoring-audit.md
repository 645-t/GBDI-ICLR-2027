# Scoring audit: correction to formal final submissions

This release applies one strict scoring rule to the 12,162 archived QA records
across 48 experimental cells. Code Agent and GBDI require a `done` command
accepted by the original parser within five final QA responses. Direct
requires the literal `The answer is:` prefix. Missing or invalid submissions
have blank answers and score as incorrect; Python observations are not
final answers.

The original QA statistics counted some answers taken from Python observations
after the response allowance was exhausted. Enforcing formal submissions
changes 59 correctness decisions in seven cells:

| Experimental cell | n | Originally counted correct | Strict correct |
|---|---:|---:|---:|
| DeepSeek Code baseline | 313 | 187 | 162 |
| Qwen, Discovery table, Code | 154 | 87 | 86 |
| MiniMax, Raw table, Code | 154 | 39 | 30 |
| MiniMax, Discovery table, Code | 154 | 52 | 36 |
| MiniMax, Action table, Code | 154 | 143 | 137 |
| GPT, Raw table, Code | 154 | 73 | 72 |
| GPT, Discovery table, Code | 154 | 90 | 89 |

All 59 changed decisions concern records whose original statistics used an
observation fallback. The other QA cells retain their original correct counts.
Of the changed DeepSeek records, 24 include a later `done` block after a Python
command in the same response. It was not the command accepted by the original
parser. The original prompts explicitly require exactly one command block
per response, so these later blocks do not supply valid final submissions.

The correction changes nine paired QA contrasts and 23 per-artifact QA
subcells. Code Agent Action-minus-Discovery gains are 39.6–65.6 percentage
points. DeepSeek GBDI-minus-Code is +16.0 percentage points (95% CI
[9.9, 22.3]). Across systems, overall GBDI-minus-Code gains remain
3.8–18.5 percentage points.

`results/qa.json.gz` contains only formally submitted answers and strict
correctness labels. Invalid records have a blank `answer` and `correct: false`.
The original source archives are retained separately from this upload package.
`replay` writes strict `correct` and `accuracy_pct` values to `qa.csv` and a
flat `qa_contrasts` mapping in `intervals.json`. The release exposes one
scoring profile.

This correction uses saved outputs without new API calls or experiment
reruns. Prompts, request parameters, cohorts and evaluation references are
unchanged.

The archived GPT GBDI-minus-Code source-table bootstrap interval is
approximately [-2.4491, 9.8182] percentage points; the originally printed
interval was [-2.50, 9.80]. Replay reports the values calculated from the
saved predictions and seed.
