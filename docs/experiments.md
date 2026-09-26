# Experiment map

The default reference is the paper's frozen **human-reviewed** reference.
It is not RADAR's transformation metadata and must not be relabeled as such.
A base task pairs a source table with a question; an evaluation instance
contains the observed table, question and gold answer. Several instances can
share one base task or one source table.

| Study | Cohort and design | Systems | Recipe prefix |
|---|---|---|---|
| Position | 60 instances, 29 base tasks; 5 bands × 3 backgrounds | Qwen Plus, MiniMax-M2.5, GPT-5 Mini; Direct and Code | `geometry-...-position` |
| Dispersion | 59 multi-critical-row instances; 3 means × 3 spacings × 3 backgrounds | Same systems and modes | `geometry-...-spacing` |
| Table states | 154 reviewed instances, 34 base tasks, 24 source tables; Raw, Discovery, Action | Same systems and modes | `table-state-...` |
| End-to-end QA | All 313 instances, 53 base tasks, 27 source tables; Direct, Code, GBDI | Qwen Plus, GLM-4.5-Air, MiniMax-M2.5, GPT-5 Mini, DeepSeek-V4.1-Flash | `baseline-...`, `gbdi-...` |
| View control | Random-5 vs Repeated-5 with Skill | Qwen, GLM, MiniMax | `gbdi-...-repeated5-skill` |
| Guidance factorial and decomposition | Random/Repeated × Generic/Skill | Qwen, GLM, MiniMax | `gbdi-...-generic` |
| Blank ledger | Same intervention and final-QA interface, no discovery ledger | Qwen Plus | `gbdi-qwenplus-none0-skill` |
| Skill rule deletions | Recovery, formatting or execution-check instructions removed | Qwen Plus | `gbdi-qwenplus-random5-no_...` |
| View-count QA | First 1, 3 or 5 frozen random inspections | Qwen Plus | `gbdi-qwenplus-random1-skill`, `random3`, `random5` |
| Discovery sensitivity | All subsets for M=1…5; support k=1…5 at M=5 | Offline analysis of saved views | `replay` |
| Discovery/QA decomposition | Nonempty reviewed reference instances, union discovery and final QA | Offline analysis | `replay` |
| External WikiSQL probes | 41 frozen missingness probes; 5 bands × 3 backgrounds, or 3 means × 3 spacings | Qwen-Turbo, Qwen3-8B, Direct | `external-...` |

There are 900 position and 1,593 dispersion views per system/mode. The
dispersion cohort excludes the one position instance with a single critical
row. Its critical-row definition is otherwise unchanged. The external study
has 615 position and 369 dispersion views per system. Frozen inputs contain
the exact constructed questions, perturbed values and row orders; they do not
select cases based on model output at runtime.

## Interfaces and visibility

* Geometry Direct uses one JSON response. Code localization permits five
  responses, reserves the last for `submit_localization`, and restores the
  visible dataframe before each Python inspection.
* The table-state Discovery condition marks reviewed erroneous cells with
  `<<ERROR:...>>`. Action applies reviewed EDIT/DROP operations before QA;
  it supplies a repaired table. The final gold answer is never in either
  model input. Annotations for these diagnostic conditions are intentional.
* GBDI creates independent discovery contexts. Each row permutation retains
  all source values and columns. Nominations map to canonical row identities.
  The union ledger retains single-view nominations and filters action or
  replacement claims from standard discovery evidence.
* Random-5 and Repeated-5 share prompts, aggregation and guidance within a
  system. Repeated-5 uses five independent contexts for the original order.
* One reserved intervention turn and up to five final QA responses share
  the conversation and Python namespace, including `work_df`. No reviewed
  coordinates, reference labels, gold answers or repaired reference tables
  enter GBDI discovery or intervention.
* GPT discovery uses opaque row IDs and its recorded candidate schema and
  renderer. An empty observed value is valid. Other GBDI systems use canonical
  integer row identifiers. These system-specific interfaces are preserved.
* The archived MiniMax geometry cohort contains 32 instances using the base
  Code prompt and 28 using the same prompt plus an explicit YAML-format
  sentence. `geometry_code_prompt_variants.json` preserves this final
  per-instance assignment. It is a serialization clarification, not an error
  label or a change to table contents.

Model-visible examples are constructed separately from evaluation labels.
The two helper modules were available in the recorded Code Agent namespace;
they are included to preserve that interface and are not run automatically
to generate discoveries or answers.

## Request parameters

`configs/systems.json` contains the defaults and study-specific overrides.
Bailian requests use temperature 0 and top-p 1. Geometry requests have a
12,000-token cap; GBDI/QA uses 4,096 for Qwen/GLM/MiniMax. Table-state requests
omit the cap. GPT omits temperature/top-p, uses low reasoning effort and
OpenAI-only routing without provider fallback: request seed 407 for geometry,
408 for GBDI/QA and 433 for the table-state study. DeepSeek uses the native
`deepseek-flash` identifier with thinking disabled and no submitted token cap.
External Qwen probes use a 2,500-token cap and disabled thinking.

CSV line endings are explicit: the recorded Code/GBDI QA user prompts use
CRLF, while Direct and discovery CSVs use LF. This prevents platform-dependent
prompt changes. Python 3.12 and the pinned dependencies are the locally tested
environment. Transport retries do not enlarge the semantic response allowance.

## Evaluation

Direct requires the literal `The answer is:` prefix, with up to three format
attempts. Code and GBDI strict scoring uses the first valid `done` command
within five final responses. Predictions are evaluated with the RADAR answer
matcher. The scoring audit explains where historical recorded scores differ.

Discovery CDR measures whether every reference critical row was found. Macro
precision, recall and F1 are computed per view/instance before averaging;
unmatched geometry Direct row IDs remain false positives. Invalid outputs are
empty predictions. Multi-view CDR/P/R/F1 uses the 141 nonempty references;
mean nominated rows uses all 154, and empty-reference false positives uses 13.
The budget analysis averages over all view subsets. Final-QA view budgets
instead use fixed prefixes of length 1, 3 and 5.

Geometry intervals use 20,000 base-task cluster resamples, with layouts paired
and averaged within instances. QA intervals use 10,000 source-table cluster
resamples. Random/Repeated discovery intervals use 10,000 base-task resamples
over the reviewed cohort. The code retains the respective archived random
generator, seed and percentile conventions. Repeated views are never treated
as independent instances.
