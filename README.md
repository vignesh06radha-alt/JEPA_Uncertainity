# Uncertainty in SIGReg JEPA world models

Build an uncertainty evaluator for SIGReg-regularized LeWorldModel using predictor ensembles, latent kNN, and prior diagnostics under action and visual shifts. PushT and Reacher form the core study; OGBench-Cube adds targeted failure-event evaluation. Official reproduction provides the reference. See the [final implementation plan](IMPLEMENTATION_PLAN.md) for the workflow and remaining work.

## Current status

The Linux reproduction harness is prepared. All three tasks passed local Ubuntu CPU checks for checkpoint loading, closed-loop planning, and short training runs that saved a checkpoint. These runs used small simulator-generated datasets. The published dataset runs are pending on the target GPU system.

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

The smoke command downloads the released checkpoint and generates 30 short simulator episodes in a new output directory. This supports separate score-fitting, penalty-selection, and threshold-calibration partitions. It runs training and planning checks, fits the three uncertainty estimators, and writes offline, warning, retrospective abstention, and paired planning reports. Use `--task reacher` or `--task cube` for the other simulators. Pass `--existing-checkpoint /path/to/released/model` to reuse a downloaded checkpoint. These outputs verify execution with fixtures, not benchmark performance.

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
- stable-worldmodel: `abdced49809d5eae38e24b27dc7b635c502c4812`, selected from the same period to retain the solver interface used by LeWM. Runtime compatibility is not yet verified.
- stable-pretraining: `0.1.6`, a March 2026 release that provides the encoder and training interfaces required by the pinned source.
- Released models and dataset revisions: manifests in `reproduction/`.
- Training points to released HDF5 files, overriding the newer PushT Lance default and inconsistent Reacher path.
- A wrapper seeds initialization and worker randomness before executing upstream scripts.
- Evaluation is adapted only to isolate output folders and save machine-readable metrics. The model, loss, success definitions, and planner algorithm remain upstream implementations.
- Upstream normalization and splitting are preserved for baseline reproduction. They must be audited before the separate uncertainty experiments, which require training-only reference statistics and trajectory-disjoint validation/test splits.

The pretrained checkpoint study and training-from-scratch study are separate deliverables. Return the result directories and environment lock from the GPU system so measured metrics can be compared with the paper. Only then can the baseline be declared reproduced.

## Next stage after baseline verification

Run a prior-deviation planning smoke check with the released checkpoint:

```bash
printf '%s\n' '{"method":"prior","beta":0.1}' > prior-smoke.json
python run.py eval --task pusht --smoke --uncertainty-config prior-smoke.json
```

An uncertainty configuration accepts `method` (`prior`, `knn`, or `ensemble`), a nonnegative `beta`, and a positive `scale` fitted on score-calibration episodes. For `knn`, provide a `training_bank` tensor file and optional `k`. New ensemble fits provide five equally trained `predictor_checkpoints` with `include_base_predictor=false`; the original predictor remains the baseline goal model. All heads share the frozen vision encoder, projection, and action-encoder training policy. Legacy configurations count the base predictor by default and must be refitted for the revised study. The command stores the exact configuration beside results.

Fit the ensemble and reference bank, then evaluate on the held-out episodes:

```bash
python fit_uncertainty.py --task pusht --name run1
python run.py eval --task pusht --uncertainty-config "$STABLEWM_HOME/uncertainty/pusht/run1/knn.json"
python run.py eval --task pusht --uncertainty-config "$STABLEWM_HOME/uncertainty/pusht/run1/ensemble.json"
```

Fitting keeps the representation and action encoder frozen, bootstraps training episodes for all five fresh predictors, and saves disjoint episode roles. Score scales use only score-calibration clips. A released checkpoint may have seen data used for this downstream split; use the same separately trained restricted checkpoint for primary action/visual comparisons.

The fit saves `prior-diagnostics.json`, including norm mean/variance and saturation, and training-only shrinkage covariance in `prior-stats.pt`. Under the isotropic Gaussian assumption, the two-sided radial tail score follows an exponential distribution. The planner switches to Mahalanobis if radial KS distance exceeds 0.20 or at least 5% of scores hit the ceiling; these are fixed heuristic thresholds. Offline evaluation retains both radial and Mahalanobis scores, current-observation kNN/prior scores, and clean-target error regardless of the planner mode. Physical probe errors remain auxiliary diagnostics.

```bash
python evaluate_uncertainty.py --task pusht \
  --artifacts "$STABLEWM_HOME/uncertainty/pusht/run1" \
  --output pusht-offline.json
```

The offline evaluator applies three RGB tint strengths to the same held-out clips while preserving simulator states and action sequences. It fits the linear physical-state probe on clean training clips. It separately reports prediction error and the probe's error on observed latents, so a degraded probe can be identified.

Use `summarize_uncertainty.py` to combine matched offline reports from independent training runs. It preserves the sample count for each run and leaves undefined metrics empty. With at least three distinct checkpoints fitted without the smoke training cap, it reports a 95% t interval for each metric across the runs where that metric is defined.

```bash
python summarize_uncertainty.py seed3072-offline.json seed3073-offline.json seed3074-offline.json \
  --output offline-summary.json
```

For controlled action holdout, train the encoder and predictor from scratch with a restricted action range, then use that checkpoint and its manifest for uncertainty fitting.

```bash
python run.py train --task pusht --seed 3072 --action-holdout 0.8
checkpoint=restricted/0.8/pusht/seed-3072/weights_epoch_10.pt
manifest="$STABLEWM_HOME/results/train-pusht-released-train3072-seed3072-action-id-0.8/action_holdout.json"
python fit_uncertainty.py --task pusht --name action-run1 \
  --checkpoint "$checkpoint" --holdout-manifest "$manifest"
python evaluate_uncertainty.py --task pusht \
  --artifacts "$STABLEWM_HOME/uncertainty/pusht/action-run1" --output pusht-action.json
```

The cutoff is a training-episode quantile of the largest absolute raw action component in each predictor input window. The training and validation loaders include only windows at or below that cutoff. The test uses disjoint episodes. The three higher-action bands use the 33rd and 67th percentiles of training-partition windows excluded by the cutoff. If no training-partition windows exceed the cutoff, the two upper boundaries are 10% and 25% above it. Empty bands are reported as null. These bands use actions from the recorded trajectories, so differences in visited states can affect comparisons. The visual-only offline comparison uses held-out windows within the trained action range.

For closed-loop visual evaluation, run a clean validation pass to calibrate the abstention threshold. Then use the same estimator settings and the test split for each tint level. The command records uncertainty for the final plan returned by CEM and each replanning step.

```bash
settings="$STABLEWM_HOME/uncertainty/pusht/run1/prior.json"
python run.py eval --task pusht --uncertainty-config "$settings" --threshold-calibration
python run.py eval --task pusht --uncertainty-config "$settings" --visual-tint 0.35
base="$STABLEWM_HOME/results/eval-pusht-released-train3072-seed42-uncertainty-prior"
python report_episode_metrics.py --calibration "$base-threshold-calibration/metrics.json" \
  --test "$base-tint-0.35/metrics.json" --output pusht-tint-035-episodes.json
```

For a full study, choose the penalty on clean validation episodes before running those test commands. Specify the grid before inspecting test results:

```bash
for beta in 0 0.05 0.1 0.25; do
  python run.py eval --task pusht --uncertainty-config "$settings" --calibration --beta "$beta"
done
python select_penalty.py "$base"-beta-*-calibration/metrics.json \
  --output "$STABLEWM_HOME/uncertainty/pusht/run1/prior-selected.json"
```

The selector checks matched cases/settings and chooses the largest grid penalty within the empirical ID success-loss tolerance (`--max-id-drop 0.02` by default). Zero remains valid if no positive candidate qualifies. This is not a statistical guarantee of ID preservation. Save the selected configuration beside the original artifacts, then run it with `--threshold-calibration` on the separate threshold episodes before the held-out run. Do not use penalty-selection traces for threshold fitting. Old artifacts need refitting; at least three usable validation episodes are required. See [the revised implementation plan](IMPLEMENTATION_PLAN.md) for the complete protocol and remaining work.

The episode report replays the first-plan accept/reject decision against the recorded outcome. It also measures warnings at later replans and uses the task budget as the failure time for episodes that miss the goal. Run the clean test condition to measure ID false alarms.

To measure the effect of the planner penalty, copy the uncertainty JSON in the same directory and set `beta=0`. Run the same test condition with the same seed and checkpoint. Compare the resulting files:

```bash
python compare_planning.py --pair /path/to/zero/metrics.json /path/to/penalty/metrics.json \
  --output planning-comparison.json
```

The comparison checks that each pair used the same start cases, solver, model, estimator, and shift. Repeat `--pair` for each independent training seed. With at least three distinct checkpoints and one pair per checkpoint, it also reports a 95% t interval across checkpoint-level differences. Checkpoints supplied for this interval must come from independent training runs.

For a closed-loop action-band comparison, use uncertainty artifacts fitted from a restricted checkpoint. The validation run selects in-range recorded start-to-goal paths. Test runs can select one of three higher bands with `--action-band 1`, `2`, or `3`:

```bash
action_config="$STABLEWM_HOME/uncertainty/pusht/action-run1/prior.json"
checkpoint=restricted/0.8/pusht/seed-3072/weights_epoch_10.pt
python run.py eval --task pusht --checkpoint "$checkpoint" --uncertainty-config "$action_config" --calibration
python run.py eval --task pusht --checkpoint "$checkpoint" --uncertainty-config "$action_config" --action-band 1
```

The band labels the recorded reference path. For band 0, including threshold calibration, CEM candidates are restricted to the training action range. In the higher bands the planner chooses its own commands. Reports count raw commands sent above the cutoff, before any clipping inside the simulator. This shows whether the higher-band planner actually proposes out-of-range actions while pursuing goals from higher-action demonstrations.
