"""Strict loading for the current LoopSight checkpoint format."""
ARCHITECTURE = "asf_task_context_cross_attention_silu_loop_vit1"
FUSION_MODE = "task_lowrank_channel"
CONTEXT = "all_memory_and_task"

def normalize_model_state(state):
    """Remove only leading DDP/compile wrappers; reject ambiguous duplicate keys."""
    normalized = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        normalized[key] = value
    return normalized

def validate_task_lowrank_model_config(args, saved, *, extra_fields=()):
    """Check shared model settings after a variant validates its own identity."""
    fields = [
        "image_size", "patch_size", "num_colors", "num_task_tokens", "embed_dim",
        "loop_core_depth", "num_heads", "mlp_dim", "dropout", "fusion_gate_rank",
        "fusion_gate_logit_bound", "loop_state_rmsnorm", "step_embedding_type",
    ]
    if args.step_embedding_type == "sinusoidal":
        fields += ["step_embedding_scale"]
    if args.step_embedding_type == "learned":
        fields += ["learned_loop_steps"]
    if args.step_embedding_type == "asymptotic":
        fields += ["asymptotic_channels", "local_clock_frequencies", "modulation_hidden_dim"]
    fields += list(extra_fields)
    mismatches = [key for key in fields if key not in saved or saved[key] != getattr(args, key)]
    if mismatches:
        raise ValueError("Checkpoint model configuration differs: " + ", ".join(mismatches))

def load_task_lowrank_state(model, checkpoint, *, skip_task_token=False):
    state = normalize_model_state(checkpoint["model_state"])
    if skip_task_token:
        state.pop("task_token_embed.weight", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {"task_token_embed.weight"} or unexpected:
            raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(state, strict=True)

def validate_task_context_checkpoint(args, checkpoint):
    saved = checkpoint.get("config", {})
    if saved.get("architecture") != ARCHITECTURE or saved.get("fusion_mode") != FUSION_MODE:
        raise ValueError("Expected a context cross-attention SiLU ASF checkpoint")
    if saved.get("fusion_gate_activation") != "silu":
        raise ValueError("Expected fusion_gate_activation=silu in the checkpoint")
    if saved.get("fusion_gate_context") != CONTEXT:
        raise ValueError(f"Expected fusion_gate_context={CONTEXT} in the checkpoint")
    if saved.get("fusion_gate_transform") != "bounded_tanh":
        raise ValueError("Expected fusion_gate_transform=bounded_tanh in the checkpoint")
    validate_task_lowrank_model_config(args, saved, extra_fields=("fusion_gate_heads",))
