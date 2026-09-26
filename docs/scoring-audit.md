# Scoring audit: recorded outputs versus strict final submissions

The manuscript describes Code Agent/GBDI scoring as requiring an explicit
`done` submission. The final experimental archives reveal a narrower mismatch:
the DeepSeek Code baseline and some table-state Code records include correct
answers extracted from Python observations after the response allowance was
exhausted. They have no valid final `done` submission.

| Experimental cell | n | Correct in recorded statistics | Correct with strict `done` |
|---|---:|---:|---:|
| DeepSeek Code baseline | 313 | 187 | 162 |
| Qwen, Discovery table, Code | 154 | 87 | 86 |
| MiniMax, Raw table, Code | 154 | 39 | 30 |
| MiniMax, Discovery table, Code | 154 | 52 | 36 |
| MiniMax, Action table, Code | 154 | 143 | 137 |
| GPT, Raw table, Code | 154 | 73 | 72 |
| GPT, Discovery table, Code | 154 | 90 | 89 |

All 59 changed correctness decisions have a recorded max-step fallback marker.
The other QA cells have the same recorded and strict correct counts.

`results/qa.json.gz` stores exactly one answer per final experimental record:
the answer used for the paper's recorded statistics. Its `correct` field uses
that same recorded scoring rule. `valid_submission` indicates a formal answer;
where an observation fallback was used, `answer_source` explicitly records
`python_observation_after_limit`. These fields allow `replay` to derive both
scoring profiles from the same answer, without a second answer archive or
silently declaring an observation an explicit submission. It also outputs the
IDs of every scoring disagreement. New runs use strict submissions; there is
no observation-answer fallback in the live runner.

These are scoring-rule differences, not newly collected model results. This
package does not edit the manuscript or assert that all its reported numbers
already implement the written strict rule. The recorded statistics and strict
statistics should be reconciled before a final public release.

One small printed-interval difference is also reproducible: the archived GPT
GBDI-minus-Code source-table bootstrap gives approximately
`[-2.4491, 9.8182]` pp; the current manuscript displays `[-2.50, 9.80]`.
The replay reports the values calculated from the saved predictions and seed.
