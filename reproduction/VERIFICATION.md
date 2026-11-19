# Verification record

## Execution-gate implementation — 2026-11-19

- Local environment: Ubuntu/WSL, Python 3.10.20, PyTorch 2.7.1+cu128, NVIDIA RTX 4050 with 6141 MiB, driver 592.82. The user will run full experiments on a 24 GB RTX 4500 Ada.
- The 21 pre-existing regression tests passed before implementation. The expanded 32-test suite then passed, including equal bootstrap clip budgets for unequal episode lengths, fixed memberships across training seeds, policy-matched thresholds, candidate-goal broadcasting, hierarchical episode draws, and actual mixed/all-rejected gating through the upstream wait loop.
- Applied the source adapter twice against clean files from pinned LeWM revision `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; both adapted file SHA256 values were unchanged on the second application. A final repeat check is recorded below after review.
- Complete CUDA PushT smoke passed at `/var/tmp/jepa-release-check/revised-gate-smoke-v2`, using the existing 30-episode simulator-generated fixture and released checkpoint. It exercised released planning, restricted two-batch base training, five fresh equal-budget heads, offline scoring, beta selection, selected-policy threshold calibration, observation-only distractor planning, actual gating, ungated/gated episode reports, and paired planning.
- The first attempt used action quantile 0.90 and failed the support guard: the small selection partition had no eligible 25-step ID planning paths. The smoke protocol was corrected to the training-only 0.999 quantile. The failed run and its log remain preserved; this fixture choice is not a real-study recommendation.
- The fit audit records two updates for each of five heads. Offline exports contain 91 records: nine clean ID clips, nine matched clips under each of nine visual conditions, and one action-band clip. Nine clean/transformed contact sheets were written; distractor and occlusion examples were visually inspected.
- Both cases in the actual gated distractor run rejected: executed actions `[0, 0]`, outcomes `[null, null]`, physical events `[null, null]`, coverage 0, and conditional success null. No success/failure was inferred for rejection. The ungated run and paired report remain separate.
- `plot_results.py` produced diagnostic and risk-coverage figures from these artifacts.
- Two more restricted two-batch fixture bases (training seeds 3073 and 3074) and their small uncertainty fits completed. The independent-encoder control then ran on three actual distinct base checkpoints, four clips, batch size two, and four identical candidates. Its initial goal-broadcasting defect was corrected and covered by regression; the rerun wrote `encoder-control.json` and candidate records.
- These checks establish software behavior on generated data. Published-dataset baselines, useful uncertainty performance, full SIGReg sensitivity, independent-training-seed research estimates, and validated Cube physical-failure events remain unmeasured. No H1–H4 performance claim follows from these fixtures.
- Final review: a separate Sol agent reviewed the implementation and the pinned planner/simulator interfaces. Its findings about action-bound gate affinity, fit-protocol aggregation, artifact contents, and extracted revision markers were fixed and regression-tested. Terra then added integrity coverage for the offline diagnostic prior statistics.
- The final repository passed 35 tests, Python compilation, `bash -n setup.sh`, and `git diff --check`. The adapter again produced identical SHA256 values on its second application to pristine pinned source. A fresh complete run at `/var/tmp/jepa-release-check/revised-gate-smoke-reviewed` passed with hashed fit artifacts. Subsequent focused verification of Terra's metadata change is recorded below.
- After Terra's change, a fresh two-head/four-clip fit, complete nine-condition offline evaluation, and 20-replicate episode-cluster summary passed, writing `terra-offline.json` and `terra-summary.json` inside the reviewed smoke directory. This was an integration check, not a confirmatory interval estimate.

## Review revision — 2026-11-19

- 21 regression checks passed in Ubuntu/Python 3.10, including disjoint validation roles, exclusion of the original predictor from new ensemble disagreement, bounded-loss beta selection with zero fallback, and episode-dependent paired statistics.
- The revised generated PushT smoke ran end to end on CUDA using 30 short simulator episodes and the existing released checkpoint. Baseline evaluation, short training, five fresh predictor fits, offline evaluation, separate threshold calibration, tinted planning, retrospective episode reporting, and paired comparison completed.
- Saved artifacts confirm five fresh heads, disjoint score/selection/threshold/test episode IDs, current-input observed scores, both radial and Mahalanobis diagnostics, and equal shifted-target/clean-target error in the clean condition.
- An additional CUDA simulator run loaded all five fresh predictors together and completed ensemble-penalized planning. The source adapter produced identical files on its second application to pristine pinned upstream files. The penalty CLI wrote both its selected configuration and tolerance/candidate audit.
- This revision requires refitting earlier artifacts. Historical counts and protocols below describe earlier revisions and are not current recommendations.
- Real published-dataset benchmarks, meaningful uncertainty performance, execution-time abstention, physical-event warnings, independent encoders, and regularizer-strength controls remain unverified. The detailed plan distinguishes implemented code from this backlog.

## Earlier verification — 2026-11-16 onward

Passed locally:

- Four runner tests: missing checkpoints fail, each task selects the paper's CEM iteration count, smoke training uses a separate checkpoint destination, and Reacher uses its canonical HDF5 path with ten training epochs.
- Python compilation of the harness and adapted evaluation script.
- Ubuntu `bash -n` on setup.sh.
- Applying the evaluation-source adaptation twice without error (idempotence).
- Dependency resolution for Linux x86-64 and Python 3.10, with stable-pretraining 0.1.6. Its published wheel contains the referenced encoder, transforms, and training APIs.
- Published planning success rates transcribed from Figure 6 of the LeWorldModel paper.
- The first archive blocks confirm Reacher contains `reacher.h5` and Cube contains `cube_single_expert.h5`. Their tar entry sizes and the PushT Zstandard frame header give 247.1 GB total extracted size. The three decoded prefixes have the HDF5 signature. The full dataset schemas and contents still need to be checked after extraction.

A separate Ubuntu test environment installed the pinned source and dependencies with Python 3.10.20, PyTorch 2.7.1 CPU, and stable-pretraining 0.1.6. The three released checkpoints loaded with strict state-dict matching. Each scored two five-step action sequences on synthetic inputs and returned a finite cost for each sequence.

The official training and evaluation configs composed for all three tasks. The evaluation configs rendered two 224 × 224 RGB observations per task using the registered environments. MuJoCo emitted EGL cleanup warnings at interpreter exit after Cube rendered successfully.

PushT, Reacher, and OGBench-Cube each completed two closed-loop evaluation cases through the seeded evaluator on simulator-generated HDF5 fixtures. These CPU checks used four CEM candidates and two iterations. The Cube fixture's nested state groups were flattened to match the fields named in the published evaluation config. Each task also completed two training batches and one validation batch on its fixture and wrote `weights_epoch_1.pt` and the model configuration.

Pending on the target system: GPU setup, complete dataset download and extraction, training, and planning evaluation against the published success rates. The CPU checks did not use the evaluation datasets.

The first attempt at filesystem-based tests encountered restricted Windows temporary-directory access. Re-running with a temporary directory inside the workspace passed all four tests.

CUDA smoke checks completed through the packaged runner for PushT, Reacher, and Cube using the local fixtures and PyTorch 2.7.1+cu128. Both evaluation and bfloat16 training completed on each task. The packaged evaluator also ran two closed-loop PushT fixture episodes using the released checkpoint and an active prior-deviation penalty. Eleven unit tests passed, covering runner configuration, uncertainty scores, candidate ranking, risk–coverage, and warning metrics. Published-dataset validation is pending.

The PushT uncertainty fitting smoke run trained four extra predictors with the encoder frozen, saved a 12-latent kNN bank, and calibrated scores on separate validation episodes. CUDA evaluations loaded both artifacts and ran the kNN and ensemble penalties on the same two held-out fixture cases. This verifies artifact creation, loading, and execution. Full training and distribution-shift comparisons are pending.

The offline CUDA smoke run evaluated 10 held-out PushT fixture clips under clean and three matched tint conditions. It wrote latent error, physical prediction error, observed-latent probe error, Spearman correlation, and shift AUROC. Correlation is undefined for a constant score or error. The action-magnitude diagnostic uses train-set magnitude quantiles and does not establish held-out action support.

The restricted PushT CUDA workflow trained a base checkpoint from scratch, fitted the uncertainty artifacts using its episode manifest, and evaluated 13 held-out fixture clips. The observed 0.90-quantile smoke run contained 8 ID and 5 above-cutoff clips. A second run at 0.85 verified the severity-boundary fallback for tied action magnitudes. The unit suite passed 12 checks, including a check that training action statistics exclude actions outside the accepted model input windows.

The selected-plan CUDA smoke check used two clean PushT validation cases and two disjoint test cases with tint 0.35. It recorded the actual CEM output plan at steps 0 and 25 when the episode was still active. The reporting command calibrated the threshold on the clean cases and produced risk–coverage, abstention coverage, detection rate, and lead time for the tinted cases. Both tinted cases missed the goal, so failure AUROC was undefined. The full suite still passes 12 checks.

The paired planning smoke check reran the same tinted cases with beta zero. Both arms completed zero of two cases and the comparison reported a zero percentage-point difference without inferring a confidence interval. A separate test rejects mismatched start cases. The suite passes 13 checks.

Reacher and OGBench-Cube each passed a CUDA uncertainty-fitting smoke run with four additional predictors and a 12-latent bank. Each offline evaluator returned results for 10 held-out clips under clean and three tint levels. Reacher used joint positions for its physical probe, and Cube used the block position. The baseline report checks full-run status, protocol and case outcomes before comparing with Figure 6. The suite passes 14 checks.

The packaged smoke command ran end to end for PushT in a new output directory and produced `offline.json`, `episodes.json`, and `paired.json`. It generated six simulator episodes, reused the local released checkpoint, and exercised the benchmark runner, training, fitting, calibration, tinted evaluation, and paired comparison. The suite still passes 14 checks.

The closed-loop action-band CUDA smoke returned an episode report from two in-range validation episodes and two higher-band test episodes using a restricted PushT checkpoint. After bounding the validation plans, both validation episodes issued zero commands above the raw cutoff. The two higher-band episodes each issued 23 above-cutoff commands before environment clipping. The prior score was constant on this briefly trained fixture, so the smoke output does not establish useful warning performance. The suite passes 15 checks.

The prior diagnostic found 100% chi-square score saturation on clean validation latents from the restricted PushT fixture and selected the training-fitted Mahalanobis fallback. The offline GPU evaluator completed 13 clips and CEM loaded the same artifact and produced finite, nonconstant selected-plan scores over two higher-action episodes. The unit suite passes 16 checks, including a test that the fallback center comes from training latents.

The penalty selection CUDA smoke compared beta 0 and beta 0.1 on the same two clean PushT validation cases. Beta 0 completed two cases and beta 0.1 completed one. The selector emitted beta 0, and the result passed through the held-out evaluator and episode reporter. The suite passes 17 checks, including tied-penalty and mismatched-case validation.

The offline summary read a live CUDA report with 13 held-out PushT fixture clips and preserved the clean, three tint, and three action-band results. A dedicated test verifies that undefined metrics are omitted from means and that a shared checkpoint does not earn a between-checkpoint interval. The suite passes 18 checks.
