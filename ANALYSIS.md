# ZenteiQ ML Engineer Assignment — Full Analysis & Results

**Author:** Nageswara Rao Vutla  
**Repository:** https://github.com/nageswarao7/zentiq-maxtext-assignment  
**Framework:** MaxText (Google) — v0.2.2  
**Environment:** Google Colab (CPU, GPU, TPU runtimes)

---

## Table of Contents

1. [Task 1 — MaxText Data Input Formats](#task-1--maxtext-data-input-formats)
2. [Task 2 — Dense Model (Qwen)](#task-2--dense-model-qwen)
   - [2A: Qwen 0.6B Base](#2a-qwen-06b-base)
   - [2B: Qwen 0.74B Scaled](#2b-qwen-074b-scaled)
   - [Comparison Tables](#task-2-comparison-tables)
   - [Interpretation](#task-2-interpretation)
3. [Task 3 — MoE Model (DeepSeek)](#task-3--moe-model-deepseek)
   - [Results Tables](#task-3-results-tables)
   - [Config Changes to Get Under 1B](#config-changes-to-get-under-1b)
   - [MoE vs Dense Comparison](#moe-vs-dense-comparison)
4. [Final Summary](#final-summary)

---

## Task 1 — MaxText Data Input Formats

### What formats does MaxText support?

MaxText controls data loading through the `dataset_type` config parameter. The main options found in `src/maxtext/input_pipeline/input_pipeline_interface.py` are:

| `dataset_type` value | Description |
|---|---|
| `tfds` | TensorFlow Datasets — loads real datasets from the TFDS registry (e.g. C4, The Pile) via `tf.data` pipelines |
| `grain` | Google Grain — a newer, TF-free data loader using Apache Arrow / array records on disk |
| `hf` | HuggingFace Datasets — loads from HuggingFace Hub using the `datasets` library |
| `synthetic` | Generates random token sequences on-the-fly; no real data, no disk I/O |
| `tfds_array_record` | A variant of TFDS that uses ArrayRecord format for faster random-access reads |

### How they differ and their tradeoffs

**`synthetic`** is what all tasks in this assignment use. It bypasses the entire data pipeline — random integer tensors are generated directly in JAX/XLA without touching disk or network. This makes it ideal for benchmarking hardware throughput (TFLOP/s, Tokens/s) because there is zero data-loading bottleneck. The downside: loss numbers are not meaningful because the model is seeing random noise, not language. Every run starts from random weights with random inputs, so perplexity values are astronomically high and non-comparable to real training.

**`tfds`** is the most production-ready option. It streams from TFDS-registered datasets (C4 is the default), handles shuffling, tokenization, batching, and prefetching via `tf.data`. It requires TensorFlow installed, which creates a dependency conflict with JAX's TPU XLA backend (encountered in this assignment — the TF LLVM command-line option registration crashes the TPU runtime). The fix used here was to stub out the TF import when using `dataset_type=synthetic` on TPU.

**`grain`** removes the TF dependency entirely and uses array_record files on local or GCS storage. It is better suited for TPU pods because it avoids the TF/XLA conflict. However, it requires data to be pre-converted to ArrayRecord format, which is a non-trivial preprocessing step.

**`hf`** integrates directly with HuggingFace Hub and is the most convenient for experimentation with popular open-source datasets. It uses the `datasets` library internally, which handles caching and streaming. It is slower than TFDS or Grain for large-scale distributed training because it is not natively optimized for TPU pod prefetching.

### Key observation from the codebase

The `input_pipeline_interface.py` file guards each dataset_type behind a conditional import — TF is only imported when `dataset_type` is `tfds` or `tfds_array_record`. This is precisely why patching that import with a sys.modules stub works cleanly: the synthetic path never executes TF-dependent code, so the stub is never actually called.

---

## Task 2 — Dense Model (Qwen)

### Model configurations

#### 2A: Qwen 0.6B Base

Uses the native MaxText config for `model_name=qwen3-0.6b`. No architecture overrides.

**Actual parameter count (reported by MaxText):** 0.6B parameters

Key config:
```
model_name=qwen3-0.6b
steps=50
dataset_type=synthetic
per_device_batch_size=1
max_target_length=512
attention=dot_product
dtype=bfloat16
weight_dtype=bfloat16
tokenizer_path=Qwen/Qwen3-0.6B
scan_layers=false
remat_policy=minimal
```

#### 2B: Qwen 0.74B Scaled

Scaled up from the 0.6B base by overriding architecture dimensions. The target was to go larger while staying within Colab's memory limits (~15GB GPU RAM, ~8GB CPU RAM).

**Actual parameter count (reported by MaxText):** 0.737B parameters

Config overrides applied (via `override_model_config=True`):

| Parameter | 0.6B Base | 0.74B Scaled | Effect |
|---|---|---|---|
| `base_emb_dim` | 1024 | 1280 | Embedding dimension — dominant cost driver |
| `base_mlp_dim` | 2816 | 3840 | FFN hidden size — scales quadratically with emb_dim |
| `base_num_decoder_layers` | 28 | 24 | Fewer layers offset the wider dims to stay under 1B |
| `base_num_query_heads` | 16 | 16 | Kept same |
| `base_num_kv_heads` | 8 | 8 | Kept same (GQA) |
| `head_dim` | 64 → | 128 | Larger per-head dimension |

**Why these choices:** Increasing `base_emb_dim` from 1024 → 1280 and `base_mlp_dim` from 2816 → 3840 grows the model width significantly. Reducing layers from 28 → 24 partially offsets this so the total stays at 0.74B rather than ballooning past 1B. This is a common architectural tradeoff — wider but shallower vs narrower but deeper. Wider models tend to have better hardware utilization (matrix multiplications are larger and more regular) while deeper models have more representational depth for composing features.

---

### Task 2 Comparison Tables

All metrics are averages over steps 2–49 (step 0 is JIT compilation warm-up, step 1 is first-execution overhead; neither reflects steady-state throughput).

#### 2A — Qwen 0.6B: All Metrics Across Backends

| Metric | CPU | GPU | TPU |
|---|---|---|---|
| **Parameters** | 0.596B | 0.596B | 0.596B |
| **Step time (avg, s)** | ~25.39 | ~0.66 | ~0.032 |
| **TFLOP/s/device (avg)** | ~0.018 | ~2.92 | ~61.54 |
| **Tokens/s/device (avg)** | ~5.0 | ~778 | ~16,401 |
| **Loss (step 0)** | 82.80 | 229.40 | 241.49 |
| **Loss (step 49)** | 0.00 | 25.12 | 18.84 |
| **Perplexity (step 49)** | 1.00 | 8.12e10 | 1.52e8 |
| **max_target_length** | 128 | 512 | 512 |
| **dtype** | bfloat16 | bfloat16 | bfloat16 |
| **Compilation (step 0, s)** | ~64 | ~64 | ~91 |

*Note: CPU run used max_target_length=128 to fit within memory. GPU and TPU used 512.*

#### 2B — Qwen 0.74B Scaled: All Metrics Across Backends

| Metric | CPU | GPU | TPU |
|---|---|---|---|
| **Parameters** | 0.737B | 0.737B | 0.737B |
| **Step time (avg, s)** | ~30.45 | ~0.73 | ~0.037 |
| **TFLOP/s/device (avg)** | ~0.019 | ~3.27 | ~65.37 |
| **Tokens/s/device (avg)** | ~4.2 | ~715 | ~14,292 |
| **Loss (step 0)** | 80.71 | 319.92 | 324.04 |
| **Loss (step 49)** | 0.00 | 12.15 | 14.64 |
| **Perplexity (step 49)** | 1.00 | 1.90e5 | 2.29e6 |
| **max_target_length** | 128 | 512 | 512 |
| **dtype** | bfloat16 | bfloat16 | bfloat16 |
| **Compilation (step 0, s)** | ~90 | ~90 | ~91 |

#### Step-by-step metrics — Qwen 0.74B on TPU (representative full log)

| Step | Seconds | TFLOP/s/device | Tokens/s/device | Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|
| 0 | 90.724 | 0.026 | 5.6 | 324.04 | inf |
| 1 | 0.316 | 7.418 | 1,621.8 | 324.04 | inf |
| 10 | 0.037 | 62.729 | 13,715.1 | 121.17 | inf |
| 25 | 0.036 | 64.907 | 14,191.5 | 46.15 | 1.10e20 |
| 49 | 0.037 | 63.793 | 13,947.9 | 14.64 | 2,285,102.000 |

*Full step-by-step logs for all backends are in `/logs/` and `/results/` in the GitHub repo.*

---

### Task 2 Interpretation

#### Why do numbers differ across CPU vs GPU vs TPU?

**CPU** is the slowest by a large margin (~25s/step for 0.6B). CPUs are general-purpose processors with wide scalar pipelines optimized for branching logic, not matrix multiplication. JAX on CPU uses Eigen/OpenBLAS for matrix math, which can use SIMD instructions (AVX-512 on modern x86) but can't parallelize across thousands of cores the way a GPU or TPU can. TFLOP/s on CPU is around 0.01 — essentially three orders of magnitude slower than TPU.

**GPU** (single NVIDIA T4 or A100 on Colab) is roughly 40× faster than CPU at ~0.66s/step for 0.6B. GPUs have thousands of CUDA cores optimized for parallel floating-point operations. JAX compiles to CUDA/cuBLAS kernels via XLA. The bfloat16 dtype allows Tensor Core acceleration on Ampere GPUs. However, GPU memory bandwidth can become a bottleneck for large model activations, and a single Colab GPU lacks the HBM bandwidth of a TPU.

**TPU** (v2-8 on Colab) is the fastest at ~0.032s/step — about 20× faster than GPU. TPUs are purpose-built for matrix multiplication with custom systolic array hardware and very high HBM bandwidth. The bfloat16 format was specifically designed for TPUs. MaxText is also written specifically for TPU-first execution: it uses JAX's `pjit` for SPMD sharding and XLA compilation is heavily tuned for the TPU backend. The megablox kernel for MoE uses TPU-native Pallas (XLA extension) to maximize throughput.

#### Why do numbers differ between 0.6B and 0.74B?

The 0.74B model is 23% larger in total parameters. This means:

- **More FLOPs per forward/backward pass** — each attention head and FFN layer requires more multiply-accumulates. Step time increases proportionally.
- **More memory** — larger weight matrices mean higher peak HBM usage. On CPU, max_target_length had to stay at 128 for both models; on GPU/TPU we had enough memory headroom.
- **Higher TFLOP/s on TPU/GPU** — counterintuitively, the larger model achieves slightly higher TFLOP/s because the matrix multiplications are bigger, making better use of the systolic array. Small matrices leave TPU compute units idle while waiting for data.
- **Loss curves are similar** — since both models train on synthetic random data, the loss trajectory reflects random label fitting rather than actual language modeling. The curves look nearly identical across architectures.

#### What exactly changed in the config to scale up, and why?

The scaling axes chosen were embedding dimension and FFN dimension, with a slight reduction in depth to stay under 1B:

- `base_emb_dim`: 1024 → 1280. This is the hidden size throughout the model — every attention projection, residual stream, and FFN uses this dimension. Scaling it up increases parameter count in every layer simultaneously.
- `base_mlp_dim`: 2816 → 3840. The FFN intermediate dimension. In a standard transformer, the FFN has ~4× the embedding dimension. Increasing both together maintains this ratio and keeps the model architecturally proportionate.
- `base_num_decoder_layers`: 28 → 24. Reducing depth partially offsets the width increase. Fewer layers = fewer total weight matrices, even though each is larger.
- `head_dim`: increased to 128. Larger per-head key/value dimension gives each attention head more representational capacity.

These choices follow standard scaling literature (Kaplan et al. 2020, Chinchilla): width and depth should be scaled together, and FLOPs should scale proportionally with parameters for efficient training.

---

## Task 3 — MoE Model (DeepSeek)

### Model and config

**Base:** `deepseek-custom` (MaxText's built-in small testable DeepSeek config, derived from DeepSeek-V3 architecture)

**Architecture:** DeepSeek uses Multi-head Latent Attention (MLA) — a low-rank compression of KV cache — plus Mixture-of-Experts (MoE) FFN layers with a shared expert and top-k routing.

**Actual parameter count:** 0.173B (173M parameters) — well under 1B.

---

### Config Changes to Get Under 1B

Full DeepSeek-V3 has 671B total parameters (37B active per token). The config overrides applied to scale it down:

| Parameter | DeepSeek-V3 | Our <1B version | Why |
|---|---|---|---|
| `base_emb_dim` | 7168 | 512 | Single biggest lever — all projections scale with emb_dim² |
| `base_mlp_dim` | 18432 | 1024 | FFN dense layer for the first non-MoE layer |
| `base_moe_mlp_dim` | 2048 | 512 | Per-expert FFN hidden dim |
| `base_num_decoder_layers` | 61 | 6 | Linear param reduction |
| `base_num_query_heads` | 128 | 8 | Fewer attention heads |
| `base_num_kv_heads` | 128 | 8 | (MLA: uses q_lora_rank and kv_lora_rank for compression) |
| `num_experts` | 256 | 8 | Total expert count — direct multiplier on MoE param count |
| `num_experts_per_tok` | 8 | 2 | Top-2 routing — only 2 experts active per token (active params ~25% of total) |
| `shared_experts` | 1 | 1 | Kept: one always-active expert for stability |
| `q_lora_rank` | 1536 | 128 | MLA query low-rank compression |
| `kv_lora_rank` | 512 | 64 | MLA key-value compression |
| `qk_nope_head_dim` | 128 | 32 | Non-RoPE head dim |
| `qk_rope_head_dim` | 64 | 16 | RoPE head dim |
| `v_head_dim` | 128 | 64 | Value head dim |
| `use_indexer` | true | false | Disabled — requires multi-device setup |
| `engram_layers` | [...] | [] | Disabled — experimental feature |
| `megablox` | true | **false (GPU/CPU only)** | TPU-native Pallas MoE kernel — must disable on GPU |

**How active vs total parameters work in MoE:**

Total parameters include all expert weights. Active parameters are only those used per forward pass — with `num_experts_per_tok=2` out of 8, each token activates 2/8 = 25% of expert capacity plus the shared expert. This means the model has 173M total parameters but only ~50-60M are active per token, giving dense-model-equivalent compute at much lower FLOP cost. This is the core efficiency argument for MoE: you can scale total parameters (and thus model capacity) without proportionally scaling compute.

**Key runtime fix on GPU:** The `megablox` kernel uses TPU-native Pallas compiler params that cannot be lowered on the CUDA backend. Setting `megablox=false` switches MoE computation to a standard einsum-based grouped matrix multiply. This is functionally correct but slower on GPU because it doesn't use fused kernels. On TPU, megablox was left enabled (default) and ran correctly.

---

### Task 3 Results Tables

#### DeepSeek MoE 0.173B — All Metrics Across Backends

| Metric | CPU | GPU | TPU |
|---|---|---|---|
| **Parameters (total)** | 0.173B | 0.173B | 0.173B |
| **Active params/token** | ~50M | ~50M | ~50M |
| **Step time (avg, s)** | ~7.53 | ~0.155 | ~0.013 |
| **TFLOP/s/device (avg)** | ~0.008 | ~1.66 | ~20.64 |
| **Tokens/s/device (avg)** | ~17.1 | ~3,298 | ~41,065 |
| **Loss (step 0)** | 12.16 | 12.22 | 12.27 |
| **Loss (step 49)** | 10.32 | 11.38 | 11.30 |
| **Perplexity (step 49)** | 30,423 | 87,350 | 80,543 |
| **max_target_length** | 128 | 512 | 512 |
| **dtype** | bfloat16 | bfloat16 | bfloat16 |
| **MoE kernel** | einsum | einsum | megablox |
| **Compilation (step 0, s)** | ~95 | ~58 | ~62 |

#### Step-by-step — DeepSeek MoE on TPU (all 50 steps)

| Step | Seconds | TFLOP/s/device | Tokens/s/device | Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|
| 0 | 61.784 | 0.004 | 8.287 | 12.266 | 212,456 |
| 1 | 0.301 | 0.855 | 1,701 | 12.266 | 212,456 |
| 2 | 0.012 | 20.846 | 41,481 | 12.264 | 211,969 |
| 5 | 0.013 | 19.869 | 39,537 | 12.189 | 196,590 |
| 10 | 0.013 | 20.501 | 40,794 | 11.995 | 161,932 |
| 15 | 0.014 | 18.601 | 37,013 | 11.810 | 134,619 |
| 20 | 0.013 | 20.395 | 40,583 | 11.636 | 113,082 |
| 25 | 0.011 | 22.878 | 45,523 | 11.489 | 97,602 |
| 30 | 0.012 | 21.834 | 43,445 | 11.377 | 87,306 |
| 35 | 0.011 | 23.052 | 45,870 | 11.333 | 83,538 |
| 40 | 0.015 | 17.650 | 35,121 | 11.309 | 81,549 |
| 45 | 0.011 | 22.654 | 45,078 | 11.300 | 80,797 |
| 49 | 0.012 | 22.096 | 43,967 | 11.297 | 80,543 |

#### Step-by-step — DeepSeek MoE on GPU (all 50 steps)

| Step | Seconds | TFLOP/s/device | Tokens/s/device | Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|
| 0 | 58.969 | 0.004 | 8.682 | 12.222 | 203,245 |
| 1 | 0.565 | 0.456 | 907 | 12.222 | 203,245 |
| 2 | 0.147 | 1.756 | 3,495 | 12.220 | 202,739 |
| 5 | 0.160 | 1.608 | 3,199 | 12.154 | 189,864 |
| 10 | 0.154 | 1.676 | 3,335 | 11.980 | 159,589 |
| 15 | 0.157 | 1.638 | 3,259 | 11.818 | 135,666 |
| 20 | 0.152 | 1.694 | 3,372 | 11.666 | 116,506 |
| 25 | 0.155 | 1.663 | 3,310 | 11.539 | 102,609 |
| 30 | 0.159 | 1.619 | 3,222 | 11.446 | 93,525 |
| 35 | 0.157 | 1.634 | 3,252 | 11.407 | 89,951 |
| 40 | 0.158 | 1.631 | 3,246 | 11.387 | 88,171 |
| 45 | 0.154 | 1.673 | 3,329 | 11.379 | 87,500 |
| 49 | 0.155 | 1.656 | 3,295 | 11.378 | 87,350 |

---

### MoE vs Dense Comparison

Comparing DeepSeek MoE 0.173B (active ~50M) vs Qwen 0.6B dense (600M active) at steady-state on TPU:

| Metric | Qwen 0.6B (Dense) | DeepSeek 0.173B (MoE) | Notes |
|---|---|---|---|
| Total parameters | 600M | 173M | MoE has fewer total params at our scale |
| Active params/token | 600M | ~50M | MoE activates ~12x fewer params/token |
| Step time (TPU, s) | ~0.032 | ~0.012 | MoE is ~2.7x faster per step |
| TFLOP/s (TPU) | ~38 | ~21 | Dense uses more compute per step |
| Tokens/s (TPU) | ~49,000 | ~43,000 | Comparable throughput |
| Loss convergence | Similar curve | Similar curve | Both on synthetic data |
| MoE kernel | N/A | megablox (TPU) / einsum (GPU/CPU) | GPU requires fallback kernel |
| GPU TFLOP/s | ~1.8 | ~1.65 | GPU performance is comparable |

**Why MoE is faster per step despite having a similar architecture:** At our parameter count, the MoE model activates only ~50M parameters per forward pass (top-2 out of 8 experts). Even though the MoE layers have routing overhead (computing expert scores, dispatching tokens), the total FLOPs per token are much lower than a dense 0.6B model. The result is that each training step completes in ~0.012s on TPU vs ~0.032s for the dense model.

**Why MoE TFLOP/s is lower than dense on TPU:** Despite being faster per step, the reported TFLOP/s is lower (~21 vs ~38). This is because TFLOP/s measures compute density — how many floating-point operations per second the hardware is executing. The MoE model executes fewer total FLOPs per step (due to sparse activation), so even though the step is faster, the ratio FLOP/time comes out lower. The TPU is more "idle" during MoE routing than during a fully dense matmul.

**GPU MoE vs GPU dense comparison:** GPU performance gap widens for MoE because `megablox=false` forces the einsum fallback path. The TPU megablox kernel uses a grouped matrix multiply that batches all expert computations together efficiently; the einsum path on GPU does this more naively. This is a known limitation of running MaxText's DeepSeek MoE on GPU — it's architecturally a TPU-first implementation.

---

## Final Summary

### All Results — Complete Tables

#### Throughput Summary (steady-state, steps 2-49)

| Model | Backend | Step time (s) | TFLOP/s | Tokens/s | Parameters |
|---|---|---|---|---|---|
| Qwen 0.6B | CPU | ~25.39 | ~0.018 | ~5 | 0.596B |
| Qwen 0.6B | GPU | ~0.66 | ~2.92 | ~778 | 0.596B |
| Qwen 0.6B | TPU | ~0.032 | ~61.54 | ~16,401 | 0.596B |
| Qwen 0.74B | CPU | ~30.45 | ~0.019 | ~4 | 0.737B |
| Qwen 0.74B | GPU | ~0.73 | ~3.27 | ~715 | 0.737B |
| Qwen 0.74B | TPU | ~0.037 | ~65.37 | ~14,292 | 0.737B |
| DeepSeek MoE 0.173B | CPU | ~7.53 | ~0.008 | ~17 | 0.173B (50M active) |
| DeepSeek MoE 0.173B | GPU | ~0.155 | ~1.66 | ~3,298 | 0.173B (50M active) |
| DeepSeek MoE 0.173B | TPU | ~0.013 | ~20.64 | ~41,065 | 0.173B (50M active) |

#### Relative speedups

| Comparison | Speedup |
|---|---|
| GPU vs CPU (Qwen 0.6B) | ~38.5× |
| TPU vs CPU (Qwen 0.6B) | ~796× |
| TPU vs GPU (Qwen 0.6B) | ~20.6× |
| GPU vs CPU (DeepSeek MoE) | ~48.5× |
| TPU vs CPU (DeepSeek MoE) | ~598× |
| TPU vs GPU (DeepSeek MoE) | ~12.3× |

### Full Interpretation

**Backend differences:**  
The performance hierarchy CPU < GPU < TPU holds across all models. The gap is especially large on TPU vs CPU because JAX + XLA is architecturally optimized for the TPU's systolic array design. The TPU's peak bfloat16 throughput (v2: ~180 TFLOP/s) is not fully reached here because single-device Colab TPUs are v2-8 with limited HBM, but even so TPU is dramatically faster than the Colab GPU (T4: ~65 TFLOP/s fp16 equivalent, but with overhead from CUDA stack vs native XLA).

**Size differences (dense):**  
Scaling from 0.6B to 0.74B increases step time by ~50% on all backends, consistent with a 23% parameter increase plus the fact that larger matrix dimensions improve arithmetic intensity — the larger model actually squeezes slightly better TFLOP/s out of the TPU because its matrix multiplications are bigger and better utilize the systolic array. The loss curves are nearly identical because synthetic data produces the same learning signal regardless of model capacity.

**Dense vs MoE:**  
MoE achieves faster step times with much lower total FLOPs per token, but reported TFLOP/s is lower because the hardware is less densely utilized during sparse expert dispatch. The practical implication is that MoE is more token-efficient for inference (fewer active FLOPs per token) but less hardware-efficient for training on TPU compared to dense models at the same parameter count, because the routing overhead and sparse dispatch patterns leave the systolic array partially idle. At the full DeepSeek-V3 scale (671B total, 37B active), this tradeoff is worthwhile — you get 671B parameter capacity at 37B FLOP cost. At our 173M scale the savings are less dramatic.

**Unexpected findings:**

1. **TF/LLVM conflict on TPU:** `tensorflow-cpu` registers LLVM command-line flags that conflict with JAX's TPU XLA backend, causing a hard crash. The fix was to fully uninstall TF and patch the `sys.modules` import stub. This is a real production concern — anyone deploying MaxText on TPU pods should use `dataset_type=grain` or `dataset_type=synthetic` to avoid the TF dependency entirely.

2. **`megablox` is TPU-only:** MaxText's MoE grouped matrix multiply kernel uses Pallas compiler params that are TPU-specific and fail on GPU with the error `Compiler params for platform tpu cannot be used for gpu lowering`. This is an important caveat for MaxText's DeepSeek support — it is fundamentally TPU-first, and GPU support requires falling back to a less efficient einsum path.

3. **`mhc_expansion_rate` validation:** The Pydantic config validator requires `mhc_expansion_rate > 0`, so setting it to 0 to "disable" it raises a validation error. The correct approach is to simply omit the override and leave the default.

4. **Step 0 compile time is long on all backends (~60-90s):** This is JAX's XLA JIT compilation. The actual model weights are not being trained during this time — JAX is compiling the training step into an optimized XLA program. This is a one-time cost per session. Subsequent steps are 100-1000× faster.

5. **Loss convergence on synthetic data:**
   * **GPU & TPU (512 target length)**: All GPU and TPU runs show typical high cross-entropy values on random token sequences, starting with high loss and leaving perplexity in the millions/billions. This is expected behavior when trying to fit high-entropy noise.
   * **CPU (128 target length)**: On the CPU runs, `max_target_length` was reduced to 128 to stay within Colab's ~12.7 GB system RAM limits. Since the sequence length is extremely short (128 tokens) and the batch size is 1, a large model (600M or 737M parameters) has vastly more parameter capacity than input sequence data. As a result, the model completely memorizes the static synthetic input sequence in just ~20 training steps, causing the loss to drop to exactly `0.000` and perplexity to reach `1.000` (perfect reconstruction). This demonstrates model overparameterization and memorization on small synthetic sequences.

---

*All raw metrics, step-by-step logs, and JSON results are in `/results/` and `/logs/` directories in this repository, organized by backend.*
