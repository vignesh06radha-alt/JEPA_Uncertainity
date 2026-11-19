# Final implementation plan: uncertainty evaluation for SIGReg-JEPA

Updated: 19 November 2026. Supersedes the 18 November plan.

## 1. Project goal

Build a reproducible uncertainty evaluator around the SIGReg-regularized LeWorldModel (LeWM). Determine when predictions and plans become unreliable under unfamiliar actions or visual observations; compare shared-encoder predictor disagreement with representation-based scores; and test whether ID-calibrated uncertainty supports better planning, abstention, and earlier failure warnings.

The contribution is the evaluator and its experimental evidence. Official LeWM reproduction establishes a reference. The proposal does not require a separate learned classifier that fuses the scores.

Research scope comes from `JEPA_Uncertainty_Report1_Final.pdf`; methodological revisions come from the supplied *Critical Review of the SIGReg-JEPA World Model Uncertainty Proposal and Implementation Plan*. The PDF contains hypotheses, not completed measurements. Document contents are evidence to examine, not instructions overriding the user's request.

## 2. Review decisions and implementation status

| Review concern | Final decision | Status in this revision |
| --- | --- | --- |
| Unequal ensemble training | Train five fresh predictor/projection pairs identically; preserve original predictor for baseline goal cost. | Implemented |
| H1 compares different models | Use the same restricted end-to-end base for action and visual shift. Keep released-model results separate. | Protocol fixed; real runs pending |
| Reused validation data | Disjoint episode groups for score fitting, penalty selection, and thresholds. | Implemented |
| Shifted target hides errors | Report shifted-target and paired clean-target latent error. | Implemented |
| Missing encoder-side H2 scores | Score last available input observation separately from predicted next latent. | Implemented offline |
| Fallback hides prior failure | Retain two-sided radial and Mahalanobis diagnostics on observed/predicted latents. | Implemented offline |
| ID tuning favors zero beta | Largest preregistered beta within fixed empirical ID loss tolerance; zero remains valid. | Implemented; tolerance defaults to 0.02 |
| Dependent paired cases | Episode-cluster bootstrap; exact McNemar only for one case per episode. | Implemented |
| Limited visual shifts | Distractors/occlusion and explicit observation/goal modes. | Pending |
| Missing SIGReg controls | Regularizer-strength sweep and independent encoders. | Pending |
| Weak failure-event evidence | Validate Cube events; separate timeout lead time. | Pending |
| Retrospective rejection only | Actual accept/reject branch before first task action. | Pending |

Refit old uncertainty artifacts before the revised study. Legacy ensemble loading remains available, but new fits exclude the original, unequally trained predictor from disagreement. Threshold reports now require the separate threshold partition.

## 3. Interpret the critique carefully

1. SIGReg encourages a training-distribution latent shape. It does not prove unfamiliar images have Gaussian latents. Measure that behavior.
2. Mahalanobis can address ID mean/covariance mismatch. It cannot guarantee detection of OOD representations overlapping ID representations.
3. A simulator provides actual physical states, but latent forecasts do not automatically yield physical predictions. A state probe remains a learned mapping that can fail under shift. Direct task outcomes provide independent behavioral evidence.
4. A positive penalty is not guaranteed to improve planning. The bounded-loss rule is an experimental policy, not a performance guarantee. Report zero when no positive candidate qualifies.
5. Fifty cases give two-percentage-point success-rate resolution, not precise evidence that a two-point drop is harmless. Increase independent selection cases where feasible and report uncertainty.
6. Cube drops/divergence are not automatically irreversible failures. Validate an environment-specific event or label it a proxy.
7. Independent encoders generally use different latent coordinates. Never subtract raw latent vectors without validated alignment.
8. A small lambda sweep provides sensitivity evidence, not a general causal proof about SIGReg.

## 4. Hypotheses and required evidence

| ID | Question | Evidence |
| --- | --- | --- |
| H1 | Is shared-encoder disagreement more informative under action shift? | Same checkpoint across shift families; within-condition uncertainty/error Spearman and detection metrics |
| H2 | Do encoder-side scores reveal visual unreliability better? | Current-observation kNN and radial/Mahalanobis versus disagreement; shift AUROC plus error/failure association |
| H3 | Does uncertainty improve MPC under shift? | Paired baseline/penalized success, recovered failures, lost successes, latency, and intervals |
| H4 | Can an ID threshold support abstention and warnings? | Risk-coverage, AURC, failure AUROC, top-80% success, frozen-threshold coverage, false alarms, misses, event lead time |

The proposal specifies five predictors, k=10, three shift severities, 3-5 seeds, and 95% intervals. It gives no required AUROC, correlation, or minimum improvement. Existing benchmark references (PushT 96%, Reacher 86%, Cube 74%) are literature baselines, not achieved results or uncertainty targets. Preserve negative findings.

## 5. Scope and priorities

- PushT: complete development and confirmatory uncertainty study.
- Reacher: second core task and transfer of conclusions.
- OGBench-Cube: baseline reproduction plus targeted H4 event study after the core pipeline works; broader sweeps follow if useful.
- Begin confirmatory work with three independent base-training seeds. Expand to five for unstable estimates or modest effects. Evaluation seeds do not replace training seeds.
- Start Cube event validation with 1-2 seeds and 50 independent cases per seed, clearly labeled exploratory.
- Separate official reproduction from the action-restricted uncertainty protocol.

## 6. End-to-end workflow

```text
Pinned source and datasets
 -> integrity checks and official baseline reference
 -> frozen episode train/validation/test manifest
 -> training-only action cutoff, scaler, and severity edges
 -> end-to-end restricted SIGReg-JEPA base checkpoints
 -> freeze vision encoder, projector, and action encoder
 -> five equally trained bootstrapped predictor heads
 -> training-only latent bank, covariance, diagnostic physical probes
 -> score-calibration episodes: scales and ID diagnostics
 -> penalty-selection episodes: fixed beta grid and ID tolerance
 -> threshold-calibration episodes: selected-policy plan scores
 -> held-out ID, action, and visual tests
 -> paired MPC, actual gate, warnings, validated event records
 -> independent-encoder and SIGReg-strength controls
 -> episode/seed-aware tables, plots, and H1-H4 conclusions
```

## 7. Phase A: Linux environment and baseline

### Steps

1. Install on the user's Linux GPU system through `reproduction/setup.sh` with pinned upstream revisions.
2. Record GPU/memory, driver, Python, Torch/CUDA, package versions, source revisions, configurations, and dataset hashes. Export the exact environment.
3. Use at least the report's proposed 16 GB GPU memory for full training, subject to an actual memory check. The local 6 GB GPU supports constrained smoke checks.
4. Store datasets/checkpoints outside Git. Use a Linux data root rather than assuming Windows D: exists. Earlier estimates were about 83 GB compressed and 247 GB extracted across tasks; verify current manifest sizes and free space first.
5. Run generated-fixture smoke checks. The revised fixture contains 30 short episodes to support separate validation roles.
6. Download PushT, Reacher, and Cube through the pinned manifest. Verify columns, image preprocessing, frame ordering, and simulator initialization.
7. Run released models with official benchmark/planner settings. Save per-case outcomes and configurations; investigate discrepancies before interpreting reference-rate differences.

### Exit condition

Real-dataset baseline reports exist for all tasks, or an unavailable task is documented. Generated-fixture success does not meet this condition. Time a real batch and short planning run before full training.

## 8. Phase B: partitions and action support

1. Freeze a recorded episode split per task and reuse it across training seeds.
2. Existing splitting reserves approximately 80% train, 10% validation, 10% test episodes. Divide validation into score-calibration, penalty-selection, and threshold-calibration groups.
3. Require at least three usable validation episodes; smaller fixtures fail explicitly. Real calibration needs substantially more than this software minimum.
4. Filter support before checking group sizes. Save dataset episode indices and exported episode IDs. Audit zero overlap across all five roles.
5. Fit action means/scales and cutoff using training episodes only. Freeze three higher-action severity bands from training support.
6. Record retained windows/episodes, action distributions, and physical-state coverage. Action restriction may also alter visited states; acknowledge this confounding limit.
7. Train primary bases end to end on restricted training windows. No calibration/test episode may enter gradient updates or checkpoint selection. Audit upstream validation behavior before confirmatory training.
8. Use the same restricted base for action-only and visual-only H1/H2 tests. Visual tests use within-support actions. Match states where feasible; otherwise report coverage differences.
9. Distinguish demonstration action bands from commands executed by MPC. Keep raw command-exposure counts; a higher reference band alone does not prove executed action OOD.

**Exit condition:** reproducible overlap/support audit, adequate data, and identical base identity across primary shift comparisons.

## 9. Phase C: uncertainty estimators

### C1. Comparable shared-encoder ensemble

- Train five fresh predictor/prediction-projection pairs with the same optimizer, learning rate, epochs, clip budget, and model mode.
- Freeze the shared vision encoder, latent projector, and pretrained action encoder for every member.
- Use distinct recorded seeds and episode bootstrap samples.
- Keep original base predictions for goal cost and base prediction-error measurement; do not silently substitute the ensemble mean.
- Supply identical observed latents/actions to all members during candidate rollouts. Score only newly predicted steps.
- Record member loss and held-out prediction error. Diagnose severe underfitting or pathological heads before interpreting disagreement.
- Treat five heads as one estimator, not five independent research runs.

### C2. Latent kNN

- Training-only bank; initial existing cap 4096 latents; default mean Euclidean distance, k=10.
- Record sampling seed, bank size, checkpoint identity, and memory residency.
- Separate current-observation H2 scores from predicted-rollout MPC scores.
- Later vary bank size and k in {1, 5, 10, 20} where supported; measure cost.

### C3. Prior and covariance

- Retain the implemented two-sided radial score: negative log of twice the smaller chi-square tail, clipped at 1e-12. Track its numerical ceiling.
- Fit shrinkage covariance/mean on training latents only. Save it even if radial diagnostics look acceptable.
- Report radial and Mahalanobis separately for observed/predicted latents, regardless of the selected planner mode.
- Record squared-norm mean/variance, KS distance, saturation, and collapse diagnostics. Correlated clips do not provide independent Gaussianity-test samples.
- Label the existing fallback rule (KS >0.20 or saturation >=5%) an operational heuristic, not a calibrated OOD probability or proof of Gaussianity.

### C4. Action-space baseline

Add training-only action-vector kNN under the same scaler. Distinguish observed-history and candidate-action queries. Measure overhead rather than calling it free.

## 10. Phase D: errors and visual conditions

Preserve paired clean/transformed renderings of the same physical trajectory.

| Metric | Interpretation |
| --- | --- |
| Shifted-target error | `||g(f(o_t_shift),a)-f(o_next_shift)||`; both sides can be corrupted |
| Clean-target error | `||g(f(o_t_shift),a)-f(o_next_clean)||`; still a representation diagnostic |
| Probe forecast error | ID-training-only physical probe on forecast versus actual future state; auxiliary |
| Probe representation error | Same probe on actual encoded observation versus physical state; exposes probe/representation failure |
| Direct task evidence | Simulator success, physical goal distance, task-specific trajectory measures independent of the probe |

Do not call a probe result direct physical model error. Save units, normalization, training data, and held-out ID probe quality. Primary physical-prediction claims require an explicitly validated physical-output model.

### Visual matrix

| Family | Three severities | Modes |
| --- | --- | --- |
| Tint | Existing alpha 0.15, 0.35, 0.60 | Observation only; observation plus goal |
| Distractor | Seeded irrelevant textures/objects at predefined coverage levels | Observation only; observation plus goal |
| Occlusion | Predefined image-area fractions with task-critical overlap logged | Observation only; observation plus goal |

Add a goal-only diagnostic subset. Freeze severity/placement before test outcomes. Transform the model input without changing physics and inspect saved images. Current closed-loop tint shifts observation and goal together; new families/mode switches remain pending.

Separate harmless shifts from harmful shifts using behavior. Shift AUROC alone does not prove useful uncertainty.

## 11. Phase E: offline H1/H2 analysis

1. Run clean/action/visual conditions on the same checkpoint and held-out cases.
2. Persist per-clip scores/errors, episode IDs, starts, action magnitudes, visual family/mode/severity, seed, and checkpoint. Current output contains condition summaries; per-clip export remains necessary for offline cluster resampling.
3. Report within-condition Spearman for both latent errors and auxiliary probe error. Do not rely on pooled ID/OOD correlation.
4. Report shift AUROC per condition and failure AUROC where labels are defensible. Leave undefined metrics null when a class is absent.
5. Plot score/error distributions and examples: high error/low uncertainty, high uncertainty/success, ordinary ID.
6. Show current-observation and forecast scores separately, including saturation/collapse diagnostics.
7. Bootstrap whole episodes rather than overlapping windows once per-clip output exists.

**Exit condition:** one complete PushT pilot has comparable conditions, no target look-ahead, expected artifacts, and interpretable diagnostics. Freeze analysis before confirmatory runs.

## 12. Phase F: MPC and beta selection

Preserve original goal cost and planner. Add beta times summed normalized predicted-step uncertainty. A current-observation score is constant across candidate actions; it belongs in warnings/gating and cannot change CEM rankings.

### Selection steps

1. Fit scales and ID diagnostics on score-calibration episodes.
2. Preregister grid {0, 0.05, 0.10, 0.25} and empirical ID loss tolerance 0.02. These are protocol choices, not PDF performance targets.
3. Evaluate each beta on identical clean within-support penalty-selection cases.
4. Select the largest beta with empirical success at least baseline minus tolerance. Save all candidate outcomes, tolerance, and grid.
5. Report sample-size limitations, lost/recovered cases, and latency. Increase independent cases for precise selection.
6. If no positive beta qualifies, retain zero and report the negative result. Do not rescue it using shifted test data.
7. After selection, collect threshold-calibration episodes under that exact policy; selected-plan distributions change with beta.
8. Evaluate paired baseline/penalized clean and shifted test cases. Full beta sensitivity is secondary; never choose the best test beta afterward.

The current planner comparison requires positive treatment beta. If selection yields zero, record 'no admissible positive penalty' and omit that comparison.

## 13. Phase G: actual gate and early warnings

### Initial gate

Before the first task action: plan, score, compare with frozen threshold, then accept or reject. Record rejection separately, coverage, and success conditional on acceptance. Rejection is not automatically success/failure.

Current reports reconstruct this decision from completed traces; they do not execute a real gate.

### Hybrid warning

Evaluate the maximum of normalized shared-ensemble prediction disagreement and current-observation kNN. Fit scales on score-calibration episodes and the combined threshold on threshold-calibration episodes. Preserve components. Only candidate-dependent forecast terms can steer MPC.

### Failure events

- Log physical state, simulator step, task success, termination reason, selected plan, and warning score at replans.
- PushT/Reacher timeout lead time means time to budget exhaustion. Corners/divergence remain proxies unless validated terminal failures.
- Inspect Cube task semantics and validate object-loss/task-defined events. Distinguish recoverable drops from terminal failure.
- Freeze event rules before tests; preserve evidence/timestamps. Measure lead time only for pre-event warnings against valid events.
- Report event counts, missed failures, detection rate, successful-ID false alarms, and lead-time distributions. Separate timeout results.
- If no defensible event or enough failures exists, state the limitation. Do not claim a safety guarantee.

## 14. Phase H: SIGReg-specific controls

### Independent encoders

Reuse 3-5 independently trained checkpoints where possible. Reusing training compute still incurs extra encoder inference.

Start with ID-normalized scalar goal-cost/error rankings on identical candidates. Alternatively use validated physical-output probes in common coordinates. Never calculate raw latent variance across unrelated encoders without validated alignment. Label the distinction between scalar-cost disagreement and shared-latent variance.

### Regularizer strength

PushT sweep: 0.1 times lambda0, lambda0, 5 times lambda0, initially 1-2 matched exploratory seeds. Inspect the exact upstream coefficient name/default before exposing an override. Hold architecture, data, epochs, optimizer, and evaluation fixed.

Measure ID success, representation variance/collapse, observed/predicted norm distributions, kNN/radial/Mahalanobis, clean-target error, and shift behavior. Do not infer general causality from this pilot. A lambda=0 extension requires a collapse check.

### Compact ablations

After the main study works: predictor count {1, 3, 5}, k {1, 5, 10, 20}, bank size, visual family, and observation/goal mode. N=1 has no disagreement and is a prediction/planning reference. Keep the main method frozen.

## 15. Statistical reporting

- Pair identical start-goal cases and evaluation seeds across planners.
- Prefer one case per episode for independent paired tests. Otherwise bootstrap clusters and withhold case-level McNemar.
- Implemented paired bootstrap: 2000 replicates, seed 42. Its interval describes episode sampling for one fitted model, not training variation.
- Report independent training-seed means/95% intervals separately. Checkpoint filenames alone do not prove independent training; retain manifests.
- Do not count overlapping encoder groups, bootstrap heads, evaluation seeds, or overlapping windows as independent model runs.
- Add hierarchical episode/seed resampling when combining both sources. Label single-seed results exploratory.
- Preserve failed runs, unfavorable shifts, and undefined metrics. Lead with effect sizes/intervals. Multiple-condition p-values are exploratory unless correction was preregistered.

## 16. Final outputs

| Artifact | Required contents |
| --- | --- |
| Baseline table | Checkpoint/task/cases/success/reference delta/exact settings |
| Data audit | Episode roles, cutoff, retention, action/state coverage, hashes |
| Offline table | Methods/conditions, both latent errors, probe diagnostics, correlations, shift/failure AUROC |
| Prior plots | Observed/predicted norms, variance, saturation, radial/Mahalanobis |
| Planning table | ID/shifted success, recovered/lost cases, beta rule, paired/seed intervals |
| Selective prediction | Risk-coverage, AURC, top-80% success, frozen-threshold coverage/conditional success |
| Warning report | Validated events, false alarms, misses, detection, event versus timeout lead time |
| Controls | Encoder/lambda/count/k/bank controls and overhead |
| Reproduction bundle | Commands, environment, configs, logs, manifests, hashes, plots, H1-H4 conclusions |

Measure encoder calls, peak GPU memory, plan wall time, and detector overhead on the same hardware. Record bank residency/query chunking.

## 17. Concrete implementation backlog

| Component | Next change | Check |
| --- | --- | --- |
| Fitter | Member seeds/losses, base partition provenance | Equal updates; no gradient-data leakage |
| Offline evaluator | Per-clip export, distractor/occlusion | Matched clips; cluster analysis possible |
| Runner/trace | Observation/goal modes, actual gate, hybrid warning | Rejection executes zero task actions; constant score cannot alter CEM |
| Source adapter | Minimal required upstream extensions | Apply twice on fresh pinned source and compare |
| Episode reporting | Gate records and physical event time | Correct event lead time; reject not success |
| Planning analysis | Hierarchical analysis when seeds permit | Preserve episode/model dependence |
| Action analysis | Action-neighborhood baseline/support audit | Training-only bank/scaler |
| Source/runner | Inspect and expose SIGReg coefficient | Effective coefficient recorded/used |
| Analysis | Independent-encoder scalar control/figures | No subtraction of unrelated latents |

Reuse upstream training/planning. Avoid a generic framework before concrete paths work.

## 18. Supported Linux workflow

Run from `reproduction` after setup/activation. Set `STABLEWM_HOME` to the Linux data root. This template assumes restricted training already produced the recorded `$checkpoint` and `$manifest` paths.

```bash
python -m unittest discover -s .
python smoke_experiment.py --task pusht --output /data/jepa/smoke-pusht
python fit_uncertainty.py --task pusht --name main-seed3072 \
  --checkpoint "$checkpoint" --holdout-manifest "$manifest"
artifacts="$STABLEWM_HOME/uncertainty/pusht/main-seed3072"
python evaluate_uncertainty.py --task pusht --artifacts "$artifacts" \
  --output /data/jepa/results/pusht-seed3072-offline.json
settings="$artifacts/ensemble.json"
for beta in 0 0.05 0.1 0.25; do
  python run.py eval --task pusht --checkpoint "$checkpoint" \
    --uncertainty-config "$settings" --calibration --beta "$beta"
done
```

Pass four exact matching validation metrics paths to `select_penalty.py`, with `--max-id-drop 0.02`. Save `ensemble-selected.json` inside the artifact directory to preserve relative bank/stats paths.

```bash
python run.py eval --task pusht --checkpoint "$checkpoint" \
  --uncertainty-config "$artifacts/ensemble-selected.json" --threshold-calibration
python run.py eval --task pusht --checkpoint "$checkpoint" \
  --uncertainty-config "$artifacts/ensemble-selected.json" --visual-tint 0.35
```

Use threshold-calibration metrics as `report_episode_metrics.py --calibration` input and held-out metrics as `--test`. On the runner, `--calibration` now means penalty selection; `--threshold-calibration` means the separate threshold partition. Beta overrides are allowed only for selection. Pending visual modes, hybrid gating, and physical-event analysis are not supported by these commands yet.

## 19. Fastest credible execution order

| Order | Work | Gate |
| --- | --- | --- |
| 1 | Reviewed corrections, regression tests, complete generated PushT smoke | Training/scoring/calibration/reporting execute |
| 2 | Linux setup, real samples, official baselines | Correct data and settings |
| 3 | Restricted PushT pilot and five predictors | Valid partitions/member quality |
| 4 | Visual modes, per-clip outputs, gate, event logging | One valid H1-H4 pilot |
| 5 | Freeze protocol; remaining PushT/Reacher seeds | No further test tuning |
| 6 | Encoder control and small lambda sweep | Evidence limits explicit |
| 7 | Targeted Cube events | Validated events or documented limitation |
| 8 | Aggregate, failure examples, figures, report | Claims trace to artifacts |

Estimate runtime after the first real epoch and planning batch. Reuse frozen-encoder computations where valid, run independent jobs together as hardware permits, and defer broad ablations until main evidence exists.

## 20. Completion criteria

The evaluator is implemented; agreed real-data experiments have run; H1-H4 are answered at the strength supported by results; and another Linux user can recreate tables from documented commands/artifacts. Negative results are valid outcomes.

Unit tests and generated-fixture smoke checks establish software behavior only. Published-dataset reproduction, shifted performance, actual execution gating, encoder controls, SIGReg sensitivity, and physical-event warning claims remain outstanding until their corresponding runs/artifacts exist. See [the verification record](reproduction/VERIFICATION.md) for checks actually performed.
