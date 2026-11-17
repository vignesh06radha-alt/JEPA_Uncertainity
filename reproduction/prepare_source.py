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
new = '''        policy_class = swm.policy.WorldModelPolicy
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
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
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
old = '    valid_indices = np.nonzero(valid_mask)[0]'
new = '''    split = uncertainty_settings.get("evaluation_split", "test")
    episode_filter = split + "_episode_ids"
    if episode_filter in uncertainty_settings:
        valid_mask &= np.isin(dataset.get_col_data(col_name), uncertainty_settings[episode_filter])
    elif split == "validation":
        raise ValueError("A fitted validation episode split is required for calibration.")
    valid_indices = np.nonzero(valid_mask)[0]'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '    end_time = time.time()'
new = '''    end_time = time.time()
    if uncertainty_settings:
        metrics["selected_plan_traces"] = [solver.traces.get(i, []) for i in range(len(eval_episodes))]
        metrics["evaluation_split"] = split
        metrics["visual_tint"] = uncertainty_settings.get("visual_tint", 0.0)'''
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
old = '        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False'
new = '        len(valid_indices) - (action_band is None), size=cfg.eval.num_eval, replace=False'
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
old = '    print(metrics)'
new = '''    if action_band is not None:
        metrics["action_band"] = action_band
        metrics["action_command_exposure"] = policy.action_exposure()
    print(metrics)'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
path.write_text(source)

path = root / 'train.py'
source = path.read_text()
old = "    with open_dict(cfg):"
new = '''    holdout = os.environ.get("LEWM_ACTION_HOLDOUT")
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
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
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
