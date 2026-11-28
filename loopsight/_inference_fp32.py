"""Fused CUDA FP32 kernels and prediction-memory scheduling for inference."""

import math

import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .backbone import LoopStateRMSNorm


@triton.jit
def _row_normalize(
    X, ADD, SCALE, GAMMA, BETA, Y, RESIDUAL,
    NROWS: tl.constexpr, TOKENS: tl.constexpr, WIDTH: tl.constexpr,
    BATCH_STRIDE: tl.constexpr, TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr, MODULATE: tl.constexpr, ADD_RESIDUAL: tl.constexpr,
    ROW_TILE: tl.constexpr, WIDTH_TILE: tl.constexpr,
):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    col = tl.arange(0, WIDTH_TILE)
    dense = row[:, None] * WIDTH + col[None, :]
    source = (row // TOKENS) * BATCH_STRIDE + (row % TOKENS) * TOKEN_STRIDE
    valid = (row[:, None] < NROWS) & (col[None, :] < WIDTH)
    x = tl.load(X + source[:, None] + col[None, :], valid, 0.0)
    if ADD_RESIDUAL:
        x += tl.load(ADD + dense, valid, 0.0)
        tl.store(RESIDUAL + dense, x, valid)
    inverse = tl.rsqrt(tl.sum(x * x, 1) / WIDTH + EPS)
    if MODULATE:
        scale = tl.load(SCALE + col, col < WIDTH, 0.0)
        gamma = tl.load(GAMMA + col, col < WIDTH, 0.0)
        beta = tl.load(BETA + col, col < WIDTH, 0.0)
        y = (x * scale[None, :]) * inverse[:, None]
        y = y * (1.0 + gamma[None, :]) + beta[None, :]
    else:
        y = x * inverse[:, None]
    tl.store(Y + dense, y, valid)


def _normalize(x, *, norm=None, gamma=None, beta=None, add=None):
    batch, width = x.shape[0], x.shape[-1]
    tokens = 1 if x.ndim == 2 else x.size(1)
    rows = batch * tokens
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    residual = torch.empty_like(output) if add is not None else None
    tile = 4
    _row_normalize[(triton.cdiv(rows, tile),)](
        x, add if add is not None else x, norm.scale if norm is not None else x,
        gamma if gamma is not None else x, beta if beta is not None else x,
        output, residual if residual is not None else output,
        rows, tokens, width, x.stride(0), x.stride(1) if x.ndim == 3 else 0,
        norm.eps if norm is not None else 1e-6, norm is not None, add is not None,
        tile, triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
    )
    return output, residual


@triton.jit
def _project_and_update_memory(
    FEATURES, WEIGHT, BIAS, CURRENT, PREVIOUS, OUTPUT,
    ROWS: tl.constexpr, PATCHES: tl.constexpr, WIDTH: tl.constexpr, RANK: tl.constexpr,
    CURRENT_BATCH_STRIDE: tl.constexpr, BOUND: tl.constexpr, STEP_BIAS,
    ROW_TILE: tl.constexpr, COLUMN_TILE: tl.constexpr, RANK_TILE: tl.constexpr,
):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    col = tl.program_id(1) * COLUMN_TILE + tl.arange(0, COLUMN_TILE)
    feature = tl.arange(0, RANK_TILE)
    x = tl.load(
        FEATURES + row[:, None] * RANK + feature[None, :],
        (row[:, None] < ROWS) & (feature[None, :] < RANK),
        0.0,
    )
    weight = tl.load(
        WEIGHT + col[None, :] * RANK + feature[:, None],
        (col[None, :] < WIDTH) & (feature[:, None] < RANK),
        0.0,
    )
    bias = tl.load(BIAS + col, col < WIDTH, 0.0)
    score = tl.dot(x, weight, input_precision="ieee") + bias[None, :]
    alpha = 1.0 / (1.0 + libdevice.exp(-(STEP_BIAS + BOUND * libdevice.tanh(score))))

    dense = row[:, None] * WIDTH + col[None, :]
    source = (row // PATCHES) * CURRENT_BATCH_STRIDE + (row % PATCHES) * WIDTH
    valid = (row[:, None] < ROWS) & (col[None, :] < WIDTH)
    current = tl.load(CURRENT + source[:, None] + col[None, :], valid, 0.0)
    previous = tl.load(PREVIOUS + dense, valid, 0.0)
    tl.store(OUTPUT + dense, previous + alpha * (current - previous), valid)


def _project_memory(current, previous, features, gate, step):
    batch, patches, width = current.shape
    rows = batch * patches
    row_tile, col_tile = 32, 64
    output = torch.empty_like(previous)
    _project_and_update_memory[(triton.cdiv(rows, row_tile), triton.cdiv(width, col_tile))](
        features, gate.up.weight, gate.up.bias, current, previous, output,
        rows, patches, width, gate.rank, current.stride(0), gate.logit_bound,
        -math.log(step - 1), row_tile, col_tile, max(16, triton.next_power_of_2(gate.rank)),
        num_warps=4, enable_fp_fusion=False,
    )
    return output


def _batch_modulation(model, steps):
    # Each modulation vector depends on the layer and loop, never on the input image.
    clock = model.asymptotic_clock
    decay = F.softplus(clock.decay_logits)
    step_index = torch.arange(steps, device=decay.device, dtype=decay.dtype)
    geometric = -torch.expm1(-decay) * torch.exp(-step_index[:, None] * decay)
    global_clock = clock.limit + clock.loop_projection(geometric)
    local_clock = clock.local_projection(clock.local_clock)
    clock_grid = global_clock[:, None, :] + local_clock[None, :, :]
    by_layer = (
        layer.time_modulation(clock_grid[:, layer_index]).unbind(0)
        for layer_index, layer in enumerate(model.core_layers)
    )
    return tuple(zip(*by_layer))


def _conditioned_core(layer, x, padding, modulation):
    gamma_attn, beta_attn, gamma_ffn, beta_ffn = modulation.chunk(4, dim=-1)
    pre_attn, _ = _normalize(x, norm=layer.norm1, gamma=gamma_attn, beta=beta_attn)
    attn = layer.dropout1(layer.self_attn(pre_attn, key_padding_mask=padding))
    pre_mlp, residual = _normalize(
        x, norm=layer.norm2, gamma=gamma_ffn, beta=beta_ffn, add=attn,
    )
    return residual + layer.dropout2(layer.mlp(pre_mlp, grid_size=layer.grid_size))


def _gate_features(gate, current, previous, task):
    p, _ = _normalize(current)
    f, _ = _normalize(previous)
    t, _ = _normalize(task)
    batch, patches, width = p.shape
    wp, wf, wt = gate.down.weight.split(width, dim=1)
    p_low = F.linear(p, wp)
    f_low = F.linear(f, wf)
    t_low = F.linear(t.unsqueeze(1), wt)
    local = p_low + f_low + t_low
    context, _ = _normalize(torch.cat((f_low, t_low), dim=1))
    p_normalized, _ = _normalize(p_low)
    q = gate.query(p_normalized)
    # Key and value share the same input. Use one GEMM for their projections.
    kv = F.linear(
        context, torch.cat((gate.key.weight, gate.value.weight)),
        torch.cat((gate.key.bias, gate.value.bias)),
    )
    k, v = kv.chunk(2, dim=-1)
    head_dim = gate.rank // gate.num_heads
    q = q.reshape(batch, patches, gate.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, patches + 1, gate.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, patches + 1, gate.num_heads, head_dim).transpose(1, 2)
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    attended = attended.transpose(1, 2).reshape(batch, patches, gate.rank)
    return gate.activation(local + gate.output(attended))


def _decode_pixels(model, memory):
    logits = model.head(model.norm(memory))
    batch = logits.size(0)
    grid, patch, colors = model.grid_size, model.patch_size, model.num_colors
    logits = logits.reshape(batch, grid, grid, patch, patch, colors)
    return logits.permute(0, 1, 3, 2, 4, 5).reshape(
        batch, model.image_size, model.image_size, colors,
    ).permute(0, 3, 1, 2)


def _advance_memory(model, hidden, memory, step, patch_mask):
    gate = model.fusion_gate
    current = model.fusion_projection(hidden)[:, model.num_task_tokens:]
    task = hidden[:, :model.num_task_tokens].mean(dim=1)
    if memory is None:
        # The first gate is evaluated by the reference implementation.
        return current * gate(current, None, task, step, patch_mask)
    features = _gate_features(gate, current, memory, task)
    return _project_memory(current, memory, features, gate, step)


def forward_final_fp32(model, hidden, padding, steps, *, overlap_memory):
    """Return final logits for an already encoded, CUDA FP32 inference batch."""
    modulation = _batch_modulation(model, steps)
    patch_mask = None if padding is None else padding[:, model.num_task_tokens:]
    core_stream = torch.cuda.current_stream(hidden.device) if overlap_memory else None
    memory_stream = torch.cuda.Stream(device=hidden.device) if overlap_memory else None

    memory = None
    for step, modulation_row in enumerate(modulation, start=1):
        for layer, affine in zip(model.core_layers, modulation_row):
            hidden = _conditioned_core(layer, hidden, padding, affine)
        if type(model.loop_state_norm) is LoopStateRMSNorm:
            hidden, _ = _normalize(hidden)
        else:
            hidden = model.loop_state_norm(hidden)

        if memory_stream is None:
            memory = _advance_memory(model, hidden, memory, step, patch_mask)
        else:
            # Only the memory branch consumes prior prediction memory.
            memory_stream.wait_stream(core_stream)
            with torch.cuda.stream(memory_stream):
                hidden.record_stream(memory_stream)
                if patch_mask is not None:
                    patch_mask.record_stream(memory_stream)
                memory = _advance_memory(model, hidden, memory, step, patch_mask)

    if memory_stream is not None:
        core_stream.wait_stream(memory_stream)
        memory.record_stream(core_stream)
    return _decode_pixels(model, memory)
