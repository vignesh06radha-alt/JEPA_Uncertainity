"""Apply documented harness fixes, leaving the model and losses unchanged."""
from pathlib import Path

root = Path(__file__).resolve().parent / 'vendor' / 'le-wm'
path = root / 'eval.py'
source = path.read_text()
old = '    results_path.mkdir(parents=True, exist_ok=True)'
new = '''    results_path = Path(os.environ['LEWM_RESULTS_DIR'])
    results_path.mkdir(parents=True, exist_ok=True)'''
if new not in source:
    assert source.count(old) == 1, 'Unexpected upstream evaluation source'
    source = source.replace(old, new)
old = '    print(metrics)'
new = '''    print(metrics)
    import json
    def json_value(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(type(value).__name__)
    (results_path / 'metrics.json').write_text(json.dumps(
        {'metrics': metrics, 'seconds': end_time - start_time,
         'dataset_rows': random_episode_indices.tolist(),
         'episode_ids': eval_episodes.tolist(),
         'start_steps': eval_start_idx.tolist(),
         'config': OmegaConf.to_container(cfg, resolve=True)},
        indent=2, default=json_value))'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '        config = swm.PlanConfig(**cfg.plan_config)'
new = '''        if os.environ.get("LEWM_UNCERTAINTY"):
            from uncertainty import load_uncertain_model
            model = load_uncertain_model(model, os.environ["LEWM_UNCERTAINTY"])
        config = swm.PlanConfig(**cfg.plan_config)'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '    policy = cfg.get("policy", "random")'
new = '''    uncertainty_settings = {}
    if os.environ.get("LEWM_UNCERTAINTY"):
        import json as uncertainty_json
        uncertainty_settings = uncertainty_json.loads(os.environ["LEWM_UNCERTAINTY"])
        if "action_mean" in uncertainty_settings:
            process["action"].mean_ = np.asarray(uncertainty_settings["action_mean"])
            process["action"].scale_ = np.asarray(uncertainty_settings["action_scale"])
        if uncertainty_settings.get("dataset_name", cfg.eval.dataset_name) != cfg.eval.dataset_name:
            raise ValueError("Uncertainty calibration was fitted to a different dataset.")
    policy = cfg.get("policy", "random")'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '        solver = hydra.utils.instantiate(cfg.solver, model=model)'
previous_solver = '''        policy_class = swm.policy.WorldModelPolicy
        policy_options = {}
        if uncertainty_settings:
            from plan_trace import TracedCEM, TracedPolicy
            solver_args = OmegaConf.to_container(cfg.solver, resolve=True)
            solver_args.pop("_target_")
            solver_args.pop("model")
            solver = TracedCEM(model=model, **solver_args)
            policy_class = TracedPolicy
            policy_options["visual_tint"] = uncertainty_settings.get("visual_tint", 0.0)
        else:
            solver = hydra.utils.instantiate(cfg.solver, model=model)'''
new = previous_solver.replace('            policy_options["visual_tint"] = uncertainty_settings.get("visual_tint", 0.0)',
    '''            visual_shift = uncertainty_settings.get("visual_shift")
            if uncertainty_settings.get("visual_tint"):
                visual_shift = {"family": "tint", "severity": uncertainty_settings["visual_tint"], "mode": "both"}
            policy_options["visual_shift"] = visual_shift
            if "gate" in uncertainty_settings:
                from report_episode_metrics import verify_frozen_threshold
                verify_frozen_threshold(uncertainty_settings, OmegaConf.to_container(cfg, resolve=True))''')
if new not in source:
    original = previous_solver if previous_solver in source else old
    assert source.count(original) == 1
    source = source.replace(original, new)
old = '''        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )'''
new = '''        policy = policy_class(
            solver=solver, config=config, process=process, transform=transform, **policy_options
        )'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
previous_filter = '''    if "test_episode_ids" in uncertainty_settings:
        valid_mask &= np.isin(dataset.get_col_data(col_name), uncertainty_settings["test_episode_ids"])
'''
source = source.replace(previous_filter, '')
source = source.replace('    elif split == "validation":', '    elif split in {"validation", "threshold"}:')
old = '    valid_indices = np.nonzero(valid_mask)[0]'
new = '''    split = uncertainty_settings.get("evaluation_split", "test")
    episode_filter = split + "_episode_ids"
    if episode_filter in uncertainty_settings:
        valid_mask &= np.isin(dataset.get_col_data(col_name), uncertainty_settings[episode_filter])
    elif split in {"validation", "threshold"}:
        raise ValueError("A fitted validation episode split is required for calibration.")
    valid_indices = np.nonzero(valid_mask)[0]'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '    end_time = time.time()'
previous_traces = '''    end_time = time.time()
    if uncertainty_settings:
        metrics["selected_plan_traces"] = [solver.traces.get(i, []) for i in range(len(eval_episodes))]
        metrics["evaluation_split"] = split
        metrics["visual_tint"] = uncertainty_settings.get("visual_tint", 0.0)'''
new = '''    end_time = time.time()
    if uncertainty_settings:
        metrics["evaluation_split"] = split
        metrics["visual_tint"] = uncertainty_settings.get("visual_tint", 0.0)'''
if new not in source:
    original = previous_traces if previous_traces in source else old
    assert source.count(original) == 1
    source = source.replace(original, new)
old = '''    metrics = world.evaluate(
        dataset=dataset,'''
new = '''    evaluate = world.evaluate
    if uncertainty_settings:
        from functools import partial
        from plan_trace import evaluate_with_trace
        evaluate = partial(evaluate_with_trace, world, policy, uncertainty_settings)
    metrics = evaluate(
        dataset=dataset,'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '    print(valid_mask.sum(), "valid starting points found for evaluation.")'
new = '''    action_band = uncertainty_settings.get("action_band")
    if action_band is not None:
        from action_holdout import reference_action_bands
        edges = uncertainty_settings["action_severity_edges"]
        bands = reference_action_bands(dataset.get_col_data("action"), valid_mask,
                                       cfg.eval.goal_offset_steps, edges)
        valid_indices = valid_indices[bands[valid_indices] == action_band]
        policy.action_cutoff = edges[0]
        if action_band == 0:
            model.bound_raw_actions(edges[0], process["action"].mean_,
                                    process["action"].scale_, cfg.plan_config.action_block)
        if len(valid_indices) < cfg.eval.num_eval:
            raise ValueError(f"Action band {action_band} has {len(valid_indices)} eligible cases; "
                             f"requested {cfg.eval.num_eval}.")
    print(len(valid_indices), "valid starting points found for evaluation.")'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '''    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])'''
previous_sampling = old.replace('len(valid_indices) - 1', 'len(valid_indices) - (action_band is None)')
new = '''    if uncertainty_settings:
        from action_holdout import sample_cases
        random_episode_indices = sample_cases(valid_indices, dataset.get_col_data(col_name), cfg.eval.num_eval, g)
    else:
        random_episode_indices = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
        random_episode_indices = np.sort(valid_indices[random_episode_indices])'''
if new not in source:
    original = previous_sampling if previous_sampling in source else old
    assert source.count(original) == 1
    source = source.replace(original, new)
old = '    print(metrics)'
previous_exposure = '''    if action_band is not None:
        metrics["action_band"] = action_band
        metrics["action_command_exposure"] = policy.action_exposure()
    print(metrics)'''
new = '''    if action_band is not None:
        metrics["action_band"] = action_band
    print(metrics)'''
if new not in source:
    original = previous_exposure if previous_exposure in source else old
    assert source.count(original) == 1
    source = source.replace(original, new)
path.write_text(source)

path = root / 'train.py'
source = path.read_text()
old = "    with open_dict(cfg):"
previous_holdout = '''    holdout = os.environ.get("LEWM_ACTION_HOLDOUT")
    if holdout:
        from action_holdout import restrict_training
        from utils import ZScoreNormalizer
        manifest = Path(os.environ["LEWM_RESULTS_DIR"]) / "action_holdout.json"
        train_set, val_set, scaler = restrict_training(
            dataset, cfg.seed, float(holdout), manifest, dataset_name, cfg.output_model_name,
        )
        transforms.append(spt.data.transforms.WrapTorchTransform(
            ZScoreNormalizer(torch.as_tensor(scaler.mean_), torch.as_tensor(scaler.scale_)),
            source="action", target="action",
        ))
    with open_dict(cfg):'''
previous_protocol = previous_holdout.replace(
    '            dataset, cfg.seed, float(holdout), manifest, dataset_name, cfg.output_model_name,',
    '''            dataset, cfg.seed, int(os.environ["LEWM_SPLIT_SEED"]), float(holdout), manifest,
            dataset_name, cfg.output_model_name, cfg.loss.sigreg.weight, os.environ["LEWM_SMOKE_TRAINING"] == "1",''')
new = previous_protocol.replace(
    '    if holdout:\n        from action_holdout', '    if holdout:\n        import json\n        from action_holdout'
).replace(
    'os.environ["LEWM_SMOKE_TRAINING"] == "1",',
    '''os.environ["LEWM_SMOKE_TRAINING"] == "1",
            json.loads(os.environ["LEWM_DATASET_PROVENANCE"]),''')
if new not in source:
    original = next((block for block in (previous_protocol, previous_holdout, old) if block in source))
    assert source.count(original) == 1
    source = source.replace(original, new)
old = '            if col.startswith("pixels"):'
new = '            if holdout or col.startswith("pixels"):'
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '''    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )'''
new = '''    if not holdout:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
    if not len(train_set) >= cfg.loader.batch_size:
        raise ValueError("Training split has fewer clips than the batch size.")'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
path.write_text(source)
