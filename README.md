# Uncertainty in SIGReg JEPA world models

Build an uncertainty evaluator for SIGReg-regularized LeWorldModel using predictor ensembles, latent kNN, and prior diagnostics under action and visual shifts. PushT and Reacher form the core study; OGBench-Cube adds targeted failure-event evaluation. Official reproduction provides the reference. See the [final implementation plan](IMPLEMENTATION_PLAN.md) for the workflow and remaining work.

## Current status

The runner and uncertainty evaluator are ready for the Linux GPU study. Local checks use small simulator-generated datasets. The published-data benchmarks, full training runs, and H1–H4 estimates are pending on the RTX 4500 Ada with 24 GB of GPU memory. See the [verification record](reproduction/VERIFICATION.md).

## Run on the Linux GPU system

Prerequisites are Git, Python 3.10 with venv support (or Python 3.11 via `PYTHON=python3.11`), a working NVIDIA driver, and EGL/OpenGL runtime libraries. The downloader keeps both the archives and the extracted HDF5 files. Allow at least 350 GB free on the data disk for all three tasks, plus room for training checkpoints and videos.

| Task | Download | Extracted HDF5 |
| --- | ---: | ---: |
| PushT | 13.1 GB | 46.3 GB |
| Reacher | 23.8 GB | 98.9 GB |
| OGBench-Cube | 46.2 GB | 101.9 GB |
| Total | 83.1 GB | 247.1 GB |

If you received the ZIP package, extract it and start in `JEPA_Uncertainity/reproduction`. Skip the first two commands below.

To verify the GPU setup before downloading the published datasets, run `bash setup.sh`, activate `.venv`, and run the self-contained smoke experiment:

```bash
python smoke_experiment.py --task pusht --output ./pusht-smoke
```

The smoke command generates 30 short episodes and runs released evaluation, restricted training, uncertainty fitting, matched visual shifts, beta selection, threshold calibration, and the execution gate. It uses two evaluation cases and a small CEM population. The fixture support quantile is 0.999 to give short random trajectories enough in-range 25-step paths. Pass `--existing-checkpoint /path/to/released/model` or `--existing-fixture /path/to/fixture.h5` to reuse local files.

```bash
git clone https://github.com/vignesh06radha-alt/JEPA_Uncertainity.git
cd JEPA_Uncertainity/reproduction
bash setup.sh
source .venv/bin/activate
export STABLEWM_HOME=/absolute/path/on/large/disk/JEPA-data

python run.py download --task pusht
python run.py eval --task pusht --smoke
python run.py eval --task pusht
```

Repeat download, smoke evaluation, and full evaluation for `reacher` and `cube`. Downloads use pinned Hugging Face revisions and validate dataset archive size and SHA256. PushT is a compressed HDF5 file, while the other two downloads are compressed tar archives. The runner handles these differently and creates canonical dataset links without duplicating the extracted data.

For training reproduction, after checkpoint evaluation succeeds:

```bash
python run.py train --task pusht --seed 3072 --smoke
python run.py train --task pusht --seed 3072
python run.py eval --task pusht --retrained --training-seed 3072 --seed 42
```

Repeat for Reacher and Cube. For three independent training runs, repeat with training seeds 3073 and 3074 and evaluate each with the same evaluation seed 42. These are our chosen training seeds, not a claim about unpublished author seeds. Changing only the evaluation seed does not constitute retraining.

Runs create separate directories beneath `$STABLEWM_HOME/results`. Each records the command, console output, and exit status. Successful evaluations save `metrics.json` with per-case success, sampled dataset rows, episode IDs, start steps, and resolved settings. Videos are saved alongside the metrics. Inspect or rename a failed run directory before retrying the same run. Preserve `environment-lock.txt` with results so the resolved dependency versions can be checked.

## Protocol and differences from upstream defaults

The paper specifies 10 training epochs, batch size 128, 224-pixel images, frame skip 5, planning horizon 5, 300 CEM samples, 30 elites, and 30 iterations for PushT versus 10 for Reacher/Cube. Evaluation uses a goal 25 steps ahead and a 50-step action budget. We explicitly override conflicting current defaults. Source: [paper, Appendices D–F](https://arxiv.org/html/2603.19312v3).

The runner uses 50 evaluation cases from the official configurations. Compare each full evaluation's `metrics.success_rate` with the corresponding percentage reported in [Figure 6](https://arxiv.org/pdf/2603.19312v3#page=7).

| Task | Published success rate |
| --- | ---: |
| PushT | 96% |
| Reacher | 86% |
| OGBench-Cube | 74% |

For PushT retraining, Table 5 reports 96.0 ± 2.83 percentage points across three training seeds on the same 50 evaluation trajectories. Use the saved episode IDs and start steps to verify that your retrained models were tested on identical cases before comparing their mean and spread. Smoke runs use two cases or two small training batches to verify that the pipeline executes.

Compare the returned full baseline results to Figure 6 with:

```bash
python benchmark_report.py /path/to/pusht/metrics.json /path/to/reacher/metrics.json \
  /path/to/cube/metrics.json --output baseline-comparison.json
```

The report checks the saved protocol, the 50 case outcomes, and successful full-run status. If you supply at least three independently trained checkpoints evaluated on identical cases, it also reports a 95% t interval across those runs.

## Pinned sources and limited adaptations

- LeWM: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- stable-worldmodel: `abdced49809d5eae38e24b27dc7b635c502c4812`, which provides the solver and simulator interfaces exercised by the fixture checks.
- stable-pretraining: `0.1.6`, a March 2026 release that provides the encoder and training interfaces required by the pinned source.
- Released models and dataset revisions: manifests in `reproduction/`.
- Training points to released HDF5 files, overriding the newer PushT Lance default and inconsistent Reacher path.
- A wrapper seeds initialization and worker randomness before executing upstream scripts.
- Baseline evaluation uses upstream sampling and success definitions and writes per-case results. The uncertainty mode adds held-out case selection, optional shifts, selected-plan traces, and a step-boundary gate around upstream simulation.
- Restricted training freezes memberships and action normalization before fitting. The archive identity, action support, episode roles, retained clip digests, and resolved training settings are saved for audit.

The pretrained checkpoint study and training-from-scratch study are separate deliverables. Return the result directories and environment lock from the GPU system so measured metrics can be compared with the paper. Only then can the baseline be declared reproduced.

## Restricted-base uncertainty study

Run from `reproduction` after completing a published-data baseline. First time a training batch and a small planning run on the 24 GB GPU. Check the restricted pilot has enough cases in each clean validation role and in the intended action bands. The example uses quantile 0.8. Record the pilot choice and keep it fixed across the primary training seeds.

```bash
task=pusht
seed=3072
quantile=0.8
name="restricted-seed$seed"
report="$STABLEWM_HOME/study/$task/$name"
mkdir -p "$report"
python run.py train --task "$task" --seed "$seed" --split-seed 42 --action-holdout "$quantile"
checkpoint="restricted/$quantile/$task/seed-$seed/weights_epoch_10.pt"
manifest="$STABLEWM_HOME/results/train-$task-released-train3072-seed$seed-action-id-$quantile/action_holdout.json"
python fit_uncertainty.py --task "$task" --name "$name" --checkpoint "$checkpoint" --holdout-manifest "$manifest"
artifacts="$STABLEWM_HOME/uncertainty/$task/$name"
python evaluate_uncertainty.py --task "$task" --artifacts "$artifacts" --output "$report/offline.json"
```

Repeat with seeds 3073 and 3074 and the same split seed. The fitter keeps the shared vision encoder, projection, and action encoder frozen. It trains five fresh bootstrapped predictor/projection pairs for equal numbers of clips and updates, retains the base goal cost, and holds out separate score-fitting, penalty-selection, threshold-calibration, and test episodes. Refit earlier uncertainty artifacts for this protocol.

The offline evaluator saves per-clip errors and scores, paired images, training physical-state coverage, and held-out probe quality. It applies the same tint, distractor, and occlusion to the history and shifted target. A clean next target is retained for a second error. The trained probe reports errors in units of training-state standard deviations. Review the paired images and annotate task-critical occlusion overlap where supported by the scene.

The original predictor provides the base forecast. The uncertainty files `ensemble.json`, `knn.json`, and `prior.json` supply candidate penalties. The prior switches to Mahalanobis when radial KS distance exceeds 0.20 or at least 5% of scores saturate. Offline reports retain radial and Mahalanobis diagnostics across both current observations and forecasts.

### Select a policy and freeze its gate

Use the clean penalty-selection episodes for all four betas. The selector chooses the largest beta within a 0.02 empirical ID success drop. If it selects zero, retain that result and omit the positive-penalty comparison.

```bash
tag=$(printf '%s' "$checkpoint" | sha256sum | cut -c1-12)
prefix="$STABLEWM_HOME/results/eval-$task-released-train3072-seed42-checkpoint-$tag"
eval_options=(eval --task "$task" --checkpoint "$checkpoint" --seed 42)
for beta in 0 0.05 0.1 0.25; do
  python run.py "${eval_options[@]}" --uncertainty-config "$artifacts/ensemble.json" --calibration --beta "$beta"
done
python select_penalty.py "$prefix"-uncertainty-ensemble-beta-*-calibration/metrics.json \
  --output "$artifacts/ensemble-selected.json"
selected="$artifacts/ensemble-selected.json"
python run.py "${eval_options[@]}" --uncertainty-config "$selected" --threshold-calibration
calibration="$prefix-uncertainty-ensemble-selected-threshold-calibration/metrics.json"
python report_episode_metrics.py --calibration "$calibration" --gate-output "$report/gate.json"
```

The threshold uses the maximum of normalized forecast ensemble disagreement and normalized kNN distance from the current observation. Only forecast disagreement contributes to the CEM penalty. The frozen threshold records the selected policy, score settings, artifact content hashes, and planner settings. Keep the calibration, selection, and test artifacts together.

### Test with and without execution gating

```bash
shift=(--visual-family distractor --visual-severity 0.15 --visual-mode observation)
python run.py "${eval_options[@]}" --uncertainty-config "$selected" "${shift[@]}"
python run.py "${eval_options[@]}" --uncertainty-config "$selected" "${shift[@]}" --gate "$report/gate.json"
condition="$prefix-uncertainty-ensemble-selected-distractor-0.15-observation-visualseed42"
python report_episode_metrics.py --calibration "$calibration" --test "$condition/metrics.json" \
  --output "$report/episodes-ungated.json"
python report_episode_metrics.py --calibration "$calibration" --test "$condition-gated-gate/metrics.json" \
  --output "$report/episodes-gated.json"
```

Repeat the ungated and gated commands on clean held-out cases. Test tint at 0.15, 0.35, and 0.60 and distractors and occlusions at 0.05, 0.15, and 0.30. Use `--visual-mode observation` and `both` for the main comparison and `goal` for the diagnostic subset. `--visual-tint 0.35` remains shorthand for the legacy observation-plus-goal tint. Use `--action-band 1`, `2`, and `3` for ungated planning comparisons on higher-action reference paths with clean visuals. Band 0 constrains raw planner commands to the training cutoff, and its frozen gate requires those same action bounds. Command exposure records raw actions actually sent to each simulator.

An actual rejection executes zero task actions and has a null outcome. The gated report gives coverage and success among accepted cases. The ungated report gives retrospective risk–coverage and failure AUROC. The traces log selected plans, physical state, and timestamps. Budget exhaustion and environment truncation are reported separately. Cube physical failure lead time requires a validated event definition and real trajectories.

For positive selected beta, save a copy of the selected JSON beside it as `ensemble-zero.json` with `beta` set to zero and run the same ungated cases and shifts. Use the planner comparison with one `--pair` per independent training seed. It checks paired case identities and reports episode-cluster intervals. The combined training-seed and episode interval requires three audited independent bases with identical test cases and matching, uncapped head fits.

```bash
python compare_planning.py --pair /path/to/beta-zero/metrics.json /path/to/positive/metrics.json \
  --output "$report/planning-comparison.json"
python plot_results.py --offline "$report/offline.json" --planning "$condition/metrics.json" \
  --episode-report "$report/episodes-ungated.json" --output "$report/figures"
```

### Research controls and aggregation

Use the same PushT training seeds, split, and action quantile for SIGReg weights 0.009, 0.09, and 0.45. The runner records the upstream coefficient and writes separate checkpoints. With `--sigreg-weight 0.009`, add `/sigreg-0.009` after `restricted/$quantile` in the checkpoint path and `-sigreg-0.009` to the training-result folder name. Refit and evaluate each model with matching settings, then compare latent coordinate variance, observed and forecast norms, errors, and ID/shifted success.

```bash
python run.py train --task pusht --seed 3072 --split-seed 42 --action-holdout "$quantile" --sigreg-weight 0.009
python summarize_uncertainty.py /path/to/seed3072/offline.json /path/to/seed3073/offline.json \
  /path/to/seed3074/offline.json --episode-bootstrap 500 --output "$STABLEWM_HOME/study/offline-summary.json"
python encoder_control.py --artifacts /path/to/seed3072/artifacts /path/to/seed3073/artifacts \
  /path/to/seed3074/artifacts --output "$STABLEWM_HOME/study/encoder-control.json"
```

The independent-encoder control uses identical offline action candidates and goal images for all models. Each model computes its scalar goal costs within its own representation and scales them using clean score-calibration cases. It reports candidate disagreement, rank agreement, and encoder cost.

Aggregate offline reports with matching prior fallback modes. Keep chi-square and Mahalanobis fallback runs in separate groups when the fitted modes differ across seeds.

For compact ablations, refit with a different `--members`, `--k`, or `--bank-size` and a distinct name. One frozen base predictor is the prediction and unpenalized planning reference. Repeat calibration for each fit before the held-out comparison.

Repeat the core protocol on Reacher. For the Cube event study, start with the published baseline and clean/shifted traces and verify the simulator event semantics before defining physical failure. Keep all timed runs, data and source manifests, environment versions, checkpoints, logs, and results for the H1–H4 analysis.
