# LoRA Distillation Pipeline Experiments for GristMill v2: A Closed-Loop Knowledge Transfer Study

**Authors**: GristMill Engineering Team  
**Date**: 2026-05-29  
**Status**: Experimental — Not yet peer reviewed

---

## Abstract

We report on the design, bring-up, and iterative experimental evaluation of a closed-loop LoRA distillation pipeline integrated into GristMill v2, a tri-language AI orchestration system. The pipeline transfers knowledge from a large teacher language model (Ollama llama3.1:8b / Claude Sonnet) into a lightweight student model (Qwen/Qwen2.5-0.5B-Instruct and Qwen/Qwen2.5-3B-Instruct) via supervised fine-tuning with Low-Rank Adaptation (LoRA), using ROUGE-L as the automated promotion criterion. We document twelve engineering bugs encountered during pipeline bring-up, two primary training experiments, and their qualitative and quantitative outcomes. The central finding is that model capacity — not pipeline architecture — is the binding constraint: neither the 0.5B nor the 3B student was able to assimilate reasoning-style patterns from 2,405 training examples without exhibiting catastrophic forgetting of factual knowledge. We further identify ROUGE-L as a misleading validation metric for this task family and propose a factual accuracy framework as a replacement. The closed-loop architecture itself is validated as functionally correct end-to-end.

---

## 1. Introduction

Deploying large language models (LLMs) in latency-sensitive production systems introduces a fundamental tension: state-of-the-art reasoning capability is concentrated in large models (7B–70B parameters) that are too slow and resource-intensive for real-time local inference, while small models (0.5B–3B parameters) that can run locally often lack the reasoning depth required for non-trivial tasks. Knowledge distillation via supervised fine-tuning with parameter-efficient adapters (LoRA) offers a principled path to bridging this gap: a capable teacher labels a corpus of queries and the student is fine-tuned on those labels, with the expectation that the student internalizes the teacher's reasoning patterns.

GristMill v2 provides the infrastructure context for this work. It is a tri-language orchestration system (Rust core, Python training shell, TypeScript integration layer) designed to route events through a tiered inference pipeline, preferring local computation and escalating to LLM calls only when local model confidence falls below a threshold (default 0.85). The Python component, `gristmill-ml`, implements the training side: a `DistillationEngine` that reads teacher-labeled examples from a SQLite buffer, trains LoRA adapters via HuggingFace PEFT and TRL's `SFTTrainer`, validates adapters via ROUGE-L comparison against held-out teacher outputs, and promotes passing adapters to the Rust daemon's hot-reload path.

This document reports all experiments conducted during the initial bring-up of this pipeline. Section 2 describes the system architecture in sufficient detail to contextualize the experimental choices. Section 3 catalogs the engineering bugs discovered and fixed before any successful training run could be completed. Sections 4 and 5 present the two primary experiments, including hyperparameter configurations, quantitative results, qualitative evaluations, and post-hoc diagnoses. Section 6 synthesizes the key findings. Section 7 concludes with recommended next steps.

---

## 2. System Architecture

### 2.1 Overall Pipeline

The GristMill v2 distillation pipeline operates as a closed loop with five stages:

1. **Teacher labeling**: Incoming queries are routed to a teacher LLM (Ollama llama3.1:8b for local inference, Claude Sonnet for escalated calls). The teacher's response, along with the original query and metadata, is stored as a labeled example.

2. **Training buffer**: Labeled examples are persisted in a domain-partitioned SQLite database. The buffer supports configurable retention policies and replay sampling to maintain distribution balance across training cycles.

3. **LoRA adapter training**: The `DistillationEngine` reads examples from the buffer and trains a LoRA adapter on top of a frozen student model checkpoint using `SFTTrainer` from the TRL library and the PEFT library from HuggingFace.

4. **Validation and promotion**: After training, the candidate adapter is evaluated on a held-out split using ROUGE-L against teacher reference outputs. If the adapter meets configurable delta thresholds relative to the previously promoted adapter, it is written to the checkpoint directory.

5. **Hot-reload**: The Rust daemon monitors the checkpoint directory and reloads the adapter without service interruption, enabling the updated student model to serve subsequent inference requests locally.

### 2.2 Distillation Modes

The `DistillationEngine` currently implements **black-box distillation**: the student is trained with standard cross-entropy loss on teacher-generated text. The teacher is treated as an opaque label source; its internal logit distributions are not used. This is the simplest and most broadly applicable distillation strategy, requiring only text-format teacher outputs.

A **white-box distillation** path (reverse-KL divergence on teacher logits) is architecturally anticipated but not yet implemented. This would require the teacher model to expose log-probability outputs alongside text, providing a richer training signal.

### 2.3 Student Models

Two student models were evaluated:

- **Qwen/Qwen2.5-3B-Instruct**: 3 billion parameters; approximately 6 GB in bfloat16. Selected initially for its stronger baseline reasoning capability relative to the 0.5B variant.

- **Qwen/Qwen2.5-0.5B-Instruct**: 494 million parameters; substantially smaller memory footprint. Selected as a fallback when the 3B model's memory requirements created operational complications in the containerized training environment.

Both models belong to Alibaba's Qwen 2.5 instruction-tuned family and share the same tokenizer and architecture family, facilitating direct comparison.

### 2.4 Training Data

The training corpus consists of **2,405 reasoning examples** seeded from the OpenHermes-2.5 dataset. Examples span two domain partitions maintained in the training buffer:

- **`default`**: General-purpose queries.
- **`reasoning`**: Queries requiring multi-step logical or arithmetic reasoning.

Domain partitioning allows the pipeline to train and validate independent LoRA adapters per domain, enabling targeted improvements without coupling unrelated capability dimensions.

### 2.5 Hardware Environments

Two hardware environments were relevant to these experiments:

- **Docker container (CPU-only)**: The standard deployment environment. Runs in a Linux VM on macOS, which completely blocks access to Apple's Metal Performance Shaders (MPS) GPU. Training on CPU produced step times of 2–5 minutes per step, making a 453-step training cycle require 15–38 hours per domain. This was deemed impractical.

- **Apple Silicon host (MPS)**: The native macOS host machine, with access to the unified memory MPS GPU. All reported experiments were run in this environment.

### 2.6 Validation Metric

The pipeline uses **ROUGE-L** (Longest Common Subsequence F1) to score adapter candidates. ROUGE-L measures the degree to which the student's outputs share subsequences with the teacher's reference outputs. Promotion requires:

- `overall_delta >= overall_delta_min` (configurable; initial value -0.01, later relaxed to -0.05)
- Per-domain `delta >= domain_delta_min` (configurable; initial value -0.03, later relaxed to -0.08)

The delta is computed relative to the currently active adapter's ROUGE-L score. A negative minimum delta allows minor regressions in exchange for convergence on new data.

---

## 3. Pipeline Bring-Up: Engineering Bugs and Fixes

Before any successful training run could be executed, twelve distinct engineering bugs were identified and resolved. These are documented in full because they represent a non-trivial body of knowledge for teams attempting to build similar pipelines with the same dependency stack (HuggingFace Transformers, PEFT, TRL, PyTorch, Docker on macOS).

### 3.1 TRL Version Incompatibility

**Symptom**: `ImportError: cannot import name 'SFTConfig' from 'trl'`

**Root cause**: `SFTConfig` was introduced in TRL 0.8.1, not 0.8.0. The project's dependency specification `trl>=0.8` permitted installation of 0.8.0, which lacked the class.

**Fix**: Tightened the lower bound to `trl>=0.8.1`.

### 3.2 Out-of-Memory in float32 (3B Model)

**Symptom**: Container OOM-killed during model loading.

**Root cause**: Qwen2.5-3B-Instruct in float32 occupies approximately 12 GB. The container had 14.37 GiB total memory, leaving insufficient headroom for optimizer state, activations, and the training buffer.

**Fix**: Added `torch_dtype=torch.bfloat16` to `from_pretrained`, reducing the model's memory footprint to approximately 6 GB.

### 3.3 `max_seq_length` Renamed in TRL 1.x

**Symptom**: `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

**Root cause**: TRL 1.x renamed the sequence length parameter from `max_seq_length` to `max_length` in `SFTConfig`. Code written against TRL 0.8.x raised an error when run against TRL 1.x.

**Fix**: Updated `SFTConfig` construction to use `max_length`.

### 3.4 SQLite UNIQUE Constraint Race

**Symptom**: Intermittent `IntegrityError: UNIQUE constraint failed` exceptions during domain cycle execution.

**Root cause**: Two concurrent domain cycles both invoked `curate()`, which executed a DELETE followed by an INSERT on the same record IDs. Between the DELETE and INSERT of one cycle, the other cycle's INSERT fired on what it believed to be a clean slate, creating a race that violated the UNIQUE constraint when the first cycle's INSERT completed.

**Fix**: Replaced `INSERT` with `INSERT OR IGNORE` to make the upsert idempotent under concurrent access.

### 3.5 Concurrent Import Race on Heavy Dependencies

**Symptom**: Intermittent `AttributeError` or `ImportError` during training startup, specifically on `from trl import SFTConfig, SFTTrainer`.

**Root cause**: The heavy TRL import was placed inside the `_train_lora()` method rather than at module level. When two domain training threads hit `_train_lora()` simultaneously, both found `trl` in `sys.modules` in a partially initialized state from the first thread's ongoing import, causing the second thread to receive an incomplete module object.

**Fix**: Moved all heavy imports (`torch`, `transformers`, `peft`, `trl`) to module level, ensuring they are fully initialized before any training thread can access them.

### 3.6 bfloat16 Rejected on CPU

**Symptom**: `ValueError: bf16 is not supported on CPU` from `SFTConfig`.

**Root cause**: `SFTConfig(bf16=True)` is valid only when a CUDA or MPS device is available. The code unconditionally passed `bf16=True` regardless of the selected device.

**Fix**: Made bf16 conditional: `bf16=(self.device != "cpu")`.

### 3.7 Meta Tensor Corruption from `device_map="cpu"`

**Symptom**: Inference on the loaded model produced NaN outputs or raised `RuntimeError: Expected all tensors to be on the same device`.

**Root cause**: In newer versions of HuggingFace Transformers, passing `device_map="cpu"` triggers the Accelerate library's meta tensor initialization path even without `low_cpu_mem_usage=True`. Meta tensors are uninitialized placeholder tensors; if the subsequent device placement logic is not correctly applied, the model ends up with mixed initialized and uninitialized weights.

**Fix**: Passed `device_map` only for non-CPU devices, allowing the default (eager) loading path to handle CPU placement.

### 3.8 Concurrent `from_pretrained` Meta Tensor Corruption

**Symptom**: One of two concurrently loading model instances would produce NaN outputs or silently incorrect results.

**Root cause**: HuggingFace's `from_pretrained` reads model weights from the filesystem into shared cache structures. Two threads calling `from_pretrained` on the same model concurrently can interleave their tensor initialization writes, corrupting each other's tensor state.

**Fix**: Introduced a `threading.Lock()` around `from_pretrained` calls to serialize model loading across domain threads.

### 3.9 Out-of-Memory from Concurrent Training (3B Model)

**Symptom**: Container OOM-killed during concurrent training of two domain cycles.

**Root cause**: Even with the load lock serializing `from_pretrained` calls, both domain cycles retained their loaded 3B model in memory simultaneously once loading completed. Two bfloat16 3B models occupied approximately 12 GB combined, exceeding available container RAM.

**Fix**: Extended the lock scope to cover the entire load → train → save pipeline, ensuring only one domain's model is resident in memory at a time.

### 3.10 Gradient Checkpointing Hang with PEFT

**Symptom**: Training hung indefinitely at step 0 of 453, with CPU utilization at 929% (all cores saturated) and no forward progress for over four hours.

**Root cause**: PEFT's LoRA wrapping modifies the model's gradient computation graph in a way that is incompatible with PyTorch's reentrant gradient checkpointing implementation unless `model.enable_input_require_grads()` is called before training begins. Without this call, the backward pass enters a deadlock waiting for gradients that will never be produced through the wrapped parameter path. Additionally, the default `use_reentrant=True` mode in `gradient_checkpointing_kwargs` is known to produce instability with PEFT adapters.

**Fix**: Added `model.enable_input_require_grads()` after PEFT wrapping, and set `gradient_checkpointing_kwargs={"use_reentrant": False}` in `SFTConfig`.

### 3.11 Docker MPS Inaccessibility

**Symptom**: Training on the intended container environment ran at 2–5 minutes per step, projecting 15–38 hours per domain for a 453-step cycle.

**Root cause**: Docker on macOS runs containers inside a Linux virtual machine (Apple Virtualization Framework). The VM has no access to the host's Metal GPU or MPS framework. All tensor operations fall back to CPU.

**Fix**: Ran the trainer natively on the macOS host to access MPS. This reduced step time to a practical duration and made the experiment timeline feasible.

### 3.12 Wrong Database Path on Host

**Symptom**: Host-side trainer reported zero training examples available, falling back to the default `~/.gristmill/db/training_buffer.sqlite` path, which was empty.

**Root cause**: The symlink `/data/gristmill` used inside the container to point to the shared SQLite database did not exist on the host. Without an explicit `training_buffer_path` in `~/.gristmill/config.yaml`, the trainer silently used the fallback path.

**Fix**: Added `training_buffer_path: <absolute path>` to `~/.gristmill/config.yaml` on the host, pointing to the shared SQLite database populated by the data seeding step.

---

## 4. Experimental Setup

### 4.1 Common Configuration

All experiments used the following fixed configuration unless otherwise noted:

| Parameter | Value |
|-----------|-------|
| Training framework | HuggingFace PEFT + TRL SFTTrainer |
| LoRA dropout | 0.05 |
| Optimizer | AdamW (TRL default) |
| Hardware | Apple Silicon MPS (native host) |
| Training data size | 2,405 reasoning examples |
| Domains | `default`, `reasoning` |
| Checkpoint format | HuggingFace PEFT adapter (saved to host filesystem) |
| Validation metric | ROUGE-L (LCS F1) |
| Teacher reference | Ollama llama3.1:8b / Claude Sonnet outputs |

### 4.2 Qualitative Evaluation Protocol

To supplement the automated ROUGE-L metric, a fixed probe question was used for qualitative evaluation across all experiments:

> *"Every day, a tree drops 7 leaves. How many leaves would it drop in a month of February in a non-leap year?"*

The correct answer is **196 leaves** (28 days × 7 leaves/day). This question tests whether the model retains the factual knowledge that February has 28 days in a non-leap year — a piece of world knowledge that should be invariant across any fine-tuning regime focused on reasoning style.

Responses were evaluated on two dimensions: (1) arithmetic correctness given the stated number of days, and (2) factual correctness of the stated number of days.

---

## 5. Results

### 5.1 Experiment 1: Baseline Training (3B Model, Aggressive Hyperparameters)

#### 5.1.1 Configuration

| Hyperparameter | Value |
|----------------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct |
| num_epochs | 3 |
| learning_rate | 2e-4 |
| lora_rank (r) | 16 |
| lora_alpha | 32 (scale ratio: 2×) |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7 modules) |
| max_length | 512 |
| batch_size | 4 |
| gradient_accumulation_steps | 4 (effective batch size: 16) |
| gradient_checkpointing | True |
| replay_fraction | 0.17 |
| torch_dtype | bfloat16 |

The LoRA scale ratio (lora_alpha / lora_rank = 32/16 = 2×) doubles the effective magnitude of the low-rank weight updates relative to a 1× ratio configuration. Targeting all seven projection module types maximizes the adapter's parameter budget and the range of weight matrices subject to modification.

#### 5.1.2 Training Dynamics

| Metric | Value |
|--------|-------|
| Final training loss | 0.8375 |
| Gradient norm at final logged step | 0.3256 |
| Token accuracy at final logged step | 76.85% |
| Epochs completed at final logged step | ~0.8 |
| Approximate wall-clock duration | ~12 hours |

The training loss of 0.8375 at roughly 0.8 epochs indicates the model was still actively learning at the point of checkpoint capture, and full convergence had not been reached by the time of evaluation.

#### 5.1.3 Quantitative Results

| Domain | Checkpoint Version | ROUGE-L Score | Promotion Decision |
|--------|--------------------|---------------|--------------------|
| `reasoning` | v1 | 0.2322 | Promoted |
| `default` | v2 | 0.2318 | Promoted |

Both domain adapters met the initial promotion thresholds and were written to the active checkpoint directory.

#### 5.1.4 Qualitative Evaluation

| Condition | Model Output | Days Stated | Arithmetic | Factual |
|-----------|-------------|-------------|------------|---------|
| Base model (no adapter) | 196 leaves (28 days × 7) | 28 | Correct | Correct |
| LoRA adapter v1 | 217 leaves | 31 | Correct | **Incorrect** |

The fine-tuned adapter produced a plausible-looking chain of reasoning — correctly multiplying its stated number of days by 7 — but stated that February has 31 days, yielding a factually wrong answer. The base model with no adapter produced the correct answer directly.

#### 5.1.5 Diagnosis

The result is consistent with **catastrophic forgetting**: the aggressive hyperparameter configuration (high learning rate 2e-4, three epochs, all seven projection layers targeted, scale ratio 2×) imposed sufficiently large weight updates to overwrite the model's pre-trained factual knowledge while successfully imprinting the step-by-step reasoning format prevalent in the OpenHermes-2.5 training data. The model learned *how* to reason through a problem but forgot *what* factual grounding to apply.

The ROUGE-L scores of 0.232 reflect high lexical overlap with teacher outputs, which themselves follow a structured reasoning format. This overlap is a consequence of the model having overfit to the surface form of the training data, not of having achieved genuine generalization.

---

### 5.2 Experiment 2: Conservative Hyperparameters (0.5B Model) — Rolled Back

#### 5.2.1 Motivation and Configuration Changes

Following the catastrophic forgetting diagnosis from Experiment 1, the following changes were made with the goal of reducing the magnitude of weight perturbation:

| Hyperparameter | Experiment 1 Value | Experiment 2 Value | Rationale |
|----------------|-------------------|-------------------|-----------|
| Model | Qwen2.5-3B-Instruct | **Qwen2.5-0.5B-Instruct** | Reduce memory pressure; enable longer sequences |
| num_epochs | 3 | **1** | Limit total gradient steps |
| learning_rate | 2e-4 | **5e-5** | 4× reduction in update step size |
| lora_alpha | 32 | **16** | Scale ratio 2× → 1×; halves effective update magnitude |
| target_modules | 7 modules | **q_proj, v_proj only** | Minimal footprint; attention queries and values only |
| replay_fraction | 0.17 | **0.30** | Increase rehearsal of prior examples |

#### 5.2.2 Quantitative Results

The candidate adapter was **rolled back** and not promoted.

**Reason**: `overall_delta = -0.0289`, which fell below the promotion threshold of `overall_delta_min = -0.01`. The new adapter scored materially lower on ROUGE-L than the previously promoted adapter from Experiment 1.

#### 5.2.3 Analysis of the Rollback

The rollback is an artifact of the interaction between ROUGE-L as a metric and the nature of the behavioral change between the two adapters. The Experiment 1 adapter was heavily overfit to the lexical surface form of teacher outputs: it reproduced step-by-step reasoning text very closely, yielding high ROUGE-L even while getting factually wrong answers. The Experiment 2 adapter, having received far fewer gradient steps at a lower learning rate, generates more varied output that may be closer to factually correct but does not reproduce the teacher's exact phrasing — which ROUGE-L penalizes.

This phenomenon illustrates a fundamental limitation of ROUGE-L as a validation metric for reasoning tasks: it rewards verbatim reproduction rather than semantic accuracy.

#### 5.2.4 Threshold Adjustment

Following this analysis, the validation thresholds were relaxed to permit promotion of adapters that score lower on ROUGE-L than their predecessors:

| Threshold | Prior Value | Relaxed Value |
|-----------|------------|---------------|
| `overall_delta_min` | -0.01 | **-0.05** |
| `domain_delta_min` | -0.03 | **-0.08** |

---

### 5.3 Experiment 3: Conservative Hyperparameters + Relaxed Validation (0.5B Model)

#### 5.3.1 Configuration

Identical to Experiment 2, with the relaxed validation thresholds applied.

#### 5.3.2 Quantitative Results

| Domain | Checkpoint Version | ROUGE-L Score | Promotion Decision |
|--------|--------------------|---------------|--------------------|
| `reasoning` | v4 | 0.1982 | Promoted |

Comparing ROUGE-L scores across experiments:

| Experiment | Model | ROUGE-L (reasoning) | Promoted |
|------------|-------|---------------------|----------|
| Experiment 1 | Qwen2.5-3B (aggressive) | 0.2322 | Yes |
| Experiment 2 | Qwen2.5-0.5B (conservative) | < 0.2033 | No (rollback) |
| Experiment 3 | Qwen2.5-0.5B (conservative, relaxed) | 0.1982 | Yes |

#### 5.3.3 Qualitative Evaluation

| Condition | Model Output | Days Stated | Arithmetic | Factual |
|-----------|-------------|-------------|------------|---------|
| Base model (no adapter) | 196 leaves (28 days × 7) | 28 | Correct | Correct |
| LoRA adapter v4 | 210 leaves | 30 | Correct | **Incorrect** |

The Experiment 3 adapter again exhibited factual error: it stated 30 days for February (not 28), yielding 210 leaves rather than the correct 196. While this represents an improvement over Experiment 1's assertion of 31 days, the fundamental failure mode persists.

---

## 6. Discussion

### 6.1 Model Capacity as the Binding Constraint

Across all experiments, the failure mode was consistent: the student model learned reasoning-style formatting patterns while degrading or overwriting factual knowledge. This occurred with both the 3B model under aggressive hyperparameters and the 0.5B model under conservative hyperparameters, though with different severity.

This is consistent with the known limitations of very small language models under fine-tuning: with limited parameter capacity, the model must allocate representational space between factual knowledge, reasoning structure, and linguistic fluency. Introducing a strong gradient signal oriented toward reasoning-style text production competes with and displaces the model's prior factual representations, even when LoRA confines updates to a small subspace of the weight matrices.

The 0.5B model (494M parameters) is a particularly constrained case. Its pre-training factual knowledge is stored in a small representational space, and even targeted LoRA updates to q_proj and v_proj are sufficient to disrupt the attention patterns that retrieve month-length facts.

### 6.2 ROUGE-L as a Misleading Metric

ROUGE-L was chosen as the initial validation metric for its simplicity and widespread use in NLG evaluation. However, this experiment family reveals a systematic failure mode: ROUGE-L rewards lexical overlap with teacher outputs, which correlates with reproduction of training text surface form rather than with semantic accuracy.

In Experiment 1, the adapter with the highest ROUGE-L scores (0.2322, 0.2318) produced the most confidently wrong factual output (31 days for February). In Experiment 3, the adapter with lower ROUGE-L (0.1982) produced a slightly less wrong factual output (30 days). The base model — which scores zero on ROUGE-L against teacher outputs because it produces correct but differently-worded answers — is semantically superior to all trained adapters on this probe question.

This inversion — where higher ROUGE-L correlates with worse semantic quality — is a direct consequence of the teacher's reasoning-format outputs being structurally distinctive (step-by-step enumeration, fixed phrasing patterns) while the correct factual content is simple and unambiguous.

### 6.3 Gradient Checkpointing and PEFT Interaction

Bug 3.10 (the 4+ hour hang at step 0 with 929% CPU utilization and no progress) deserves particular attention because it is not documented prominently in either the PEFT or TRL documentation. The root cause — that PEFT-wrapped models require `model.enable_input_require_grads()` before gradient checkpointing activates correctly — is a subtle interaction between two libraries that each assume the other handles certain gradient graph setup steps.

The symptoms (very high CPU utilization with zero training progress) are easily misattributed to a performance issue rather than a deadlock. Practitioners building similar pipelines should treat any training job that shows sustained high CPU usage with zero loss reduction at step 0 as a strong indicator of this specific bug.

Setting `use_reentrant=False` in `gradient_checkpointing_kwargs` is additionally recommended as a defensive measure.

### 6.4 Concurrency Architecture

The serialization of the entire load → train → save pipeline across domain threads has a meaningful practical cost: it eliminates the parallelism that multi-domain training was intended to exploit. With two domains and a serialized pipeline, wall-clock training time for a full cycle is the sum of per-domain training times rather than the maximum.

A more scalable approach would be to train domains sequentially by design rather than relying on a lock that happens to serialize them.

### 6.5 Pipeline Architecture Validation

Despite the model-capacity failures, the closed-loop pipeline architecture demonstrated correct end-to-end operation: data seeding populated the SQLite buffer; the `DistillationEngine` ingested the buffer and produced trained adapter checkpoints; the validation framework promoted or rolled back adapters; promoted adapters were written to the checkpoint directory; and the Rust daemon hot-reloaded the adapter without service interruption. The failure is model-capacity-specific and does not indicate a flaw in the pipeline architecture.

### 6.6 Why Published LoRA Results Do Not Directly Transfer to Our Setup

Published tutorials on LoRA fine-tuning of 0.5B-scale models — such as the Databricks example fine-tuning `Qwen2-0.5B` on the `trl-lib/Capybara` instruction-following dataset — report successful adaptation with measurably reduced evaluation loss after a single epoch. These results can create the expectation that any LoRA run on a similarly-sized model should yield clear improvements. Our experiments did not replicate this pattern. The following analysis identifies the specific structural differences that explain the disparity.

#### 6.6.1 Training Data: Scale, Diversity, and Domain Alignment

This is the dominant difference. The Databricks tutorial uses the Capybara dataset: thousands of diverse, high-quality conversational examples spanning a wide distribution of topics and instruction styles. Critically, instruction following is the model's **pre-training domain** — Qwen2.5-Instruct was trained on precisely this kind of data. LoRA on Capybara therefore reinforces existing representational structure rather than introducing new ones.

Our training corpus consists of a few dozen to a few hundred teacher-labeled examples, all drawn from a narrow template of arithmetic reasoning questions in a highly structured step-by-step format. This format is **out-of-distribution** relative to the model's pre-training. Rather than reinforcing existing structure, fine-tuning on this corpus must simultaneously (a) introduce a new surface form (the step-by-step enumeration style of the teacher), (b) retain the factual knowledge that correctly answers the question, and (c) generalize rather than memorize. With a small dataset, the model cannot learn the pattern — it memorizes the surface form of the training examples, which overwrites the factual representations in (b).

The ratio of training examples to model parameters is approximately 50–500 examples for 494M parameters (Qwen2.5-0.5B), versus thousands of examples for the same model size in the Databricks setting. This ratio gap is the primary driver of the memorization failure.

#### 6.6.2 Effective Gradient Magnitude — 80× Smaller

The effective learning rate in a LoRA fine-tuning run scales with the product of three factors: base learning rate, `lora_alpha / lora_rank`, and the number of epochs. A comparison:

| Factor | Databricks | GristMill (Experiment 3) | Ratio |
|--------|-----------|--------------------------|-------|
| Learning rate | 1e-3 | 5e-5 | 20× lower |
| lora_alpha / lora_rank | 32/8 = **4×** | 16/16 = **1×** | 4× lower |
| Epochs | 1 | 1 | equal |
| **Composite effective LR** | **4e-3** | **5e-5** | **~80× lower** |

The Databricks tutorial explicitly notes that "10× scaling is recommended for LoRA" relative to a full fine-tuning learning rate, which is why they use 1e-3 rather than the more typical 1e-4 base LR. Our conservative configuration — designed to minimize catastrophic forgetting — produces an effective gradient magnitude 80× smaller than the published example. With a tiny dataset, this means the adapter parameters barely receive a meaningful training signal before the epoch terminates.

#### 6.6.3 Target Module Coverage — 2 of 7 Projection Layers

The Databricks tutorial targets all seven projection matrices: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Our conservative configuration targets only `q_proj` and `v_proj` — the attention query and value projections.

The omitted modules play distinct functional roles:
- `k_proj`: attention key, determines what the query attends to
- `o_proj`: attention output projection, controls what the attended information contributes to the residual stream
- `gate_proj`, `up_proj`, `down_proj`: the MLP block's gating and projection weights, responsible for the majority of factual retrieval in transformer architectures

Restricting training to Q and V means the adapter cannot substantially modify how information flows through the feed-forward layers — the component most responsible for factual storage and retrieval. This is architecturally coherent as a forgetting-prevention strategy, but it also limits the adapter's capacity to learn the structural reasoning patterns in the teacher's outputs.

#### 6.6.4 Evaluation Metric Mismatch

The Databricks tutorial evaluates on `eval_loss` (cross-entropy), computed on a 10% held-out split drawn from the **same Capybara distribution** as the training data. This is a smooth, continuous, numerically stable metric that is directly aligned with the training objective. Improvement in eval_loss reliably indicates the model is learning the target distribution.

Our pipeline evaluates on ROUGE-L **delta** relative to a previously promoted adapter, using a held-out set drawn from the training buffer and a binary pass/fail threshold. ROUGE-L on short mathematical reasoning outputs is discrete, sensitive to phrasing, and — as documented in §6.2 — inversely correlated with semantic accuracy in this experiment family. The evaluation metric not only fails to reward correct answers but actively penalizes them when they are more concise or differently phrased than the teacher's verbose step-by-step outputs.

#### 6.6.5 Summary: Compounding Misalignment

None of the four differences above would be disqualifying in isolation. Together, they compound:

1. **Tiny, OOD dataset** → memorization instead of generalization
2. **80× weaker gradient signal** → adapter barely updates in one epoch
3. **2 of 7 modules targeted** → insufficient capacity to acquire reasoning patterns
4. **ROUGE-L on OOD probe** → rewards format mimicry, punishes correct but concise answers

The published LoRA success results assume large, in-distribution training data with a strong gradient signal — conditions under which LoRA is well-understood to work reliably. Our setup violates all four assumptions simultaneously. Resolving the data scale and distribution alignment issue (§8.1, §8.5) is the prerequisite that unlocks the other improvements.

---

## 7. Conclusions

1. **Model capacity is the binding constraint for reasoning distillation at this data scale.** Both Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct under the tested configurations exhibited catastrophic forgetting of factual knowledge when fine-tuned on 2,405 reasoning-format examples.

2. **ROUGE-L is an inappropriate validation metric for factual reasoning tasks.** It rewards verbatim surface overlap with teacher outputs, which correlates with overfitting to training format rather than semantic accuracy. In this experiment family, higher ROUGE-L corresponded to worse factual performance.

3. **Gradient checkpointing with PEFT requires `enable_input_require_grads()` and `use_reentrant=False`.** Without these settings, the backward pass hangs indefinitely — a failure mode that can be mistaken for a performance issue rather than a deadlock.

4. **Docker on macOS cannot access MPS.** The Linux VM layer fully blocks Metal GPU access, making containerized LoRA training on Apple Silicon impractical. Native host execution is required for MPS-accelerated training.

5. **Concurrent multi-domain training requires global pipeline serialization.** Concurrent `from_pretrained` calls on shared model cache files corrupt tensor state; the serialization lock must cover the entire load → train → save pipeline.

6. **The closed-loop architecture is structurally sound.** The failures observed are model-capacity-specific and do not indicate architectural deficiencies in the distillation loop.

---

## 8. Future Work

### 8.1 Upgrade Student Model to Qwen2.5-1.5B-Instruct

The 1.5B parameter variant occupies approximately 3 GB in bfloat16, trainable on MPS without memory constraints. Its larger parameter budget provides more representational headroom to simultaneously retain factual knowledge and acquire reasoning format patterns.

### 8.2 Replace ROUGE-L with Factual Accuracy Evaluation

Construct a held-out evaluation set consisting of questions with known, verifiable correct answers — arithmetic problems with exact numeric answers, factual recall questions with established ground truth, and multi-step reasoning problems with deterministic solutions. Evaluate on exact-match accuracy or numeric proximity rather than lexical overlap.

### 8.3 Retrain the Sieve ONNX Intent Classifier

The current Sieve classifier assigns confidence of 0.289 to reasoning queries, far below the 0.85 local routing threshold. Retraining with accumulated feedback data would enable local routing for high-confidence queries, reducing LLM escalation costs and improving end-to-end latency.

### 8.4 Implement White-Box Distillation

When the teacher model is available locally (e.g., via Ollama), it can expose log-probability distributions over vocabulary at each generation step. Training the student to minimize reverse-KL divergence against these teacher logit distributions provides a richer training signal than cross-entropy on sampled text alone.

### 8.5 Explicit Domain Sequencing

Replace the lock-based implicit serialization of domain training with explicit sequential scheduling. This makes the intended execution order transparent, allows domain-priority ordering, and eliminates lock contention as a source of unexpected behavior.

---

## Appendix A: Dependency Versions

| Package | Version Constraint | Notes |
|---------|-------------------|-------|
| `trl` | `>=0.8.1` | `SFTConfig` unavailable in 0.8.0 |
| `peft` | (project default) | Requires `enable_input_require_grads()` for gradient checkpointing |
| `transformers` | (project default) | `device_map="cpu"` triggers meta tensor path in newer versions |
| `torch` | (project default) | `bfloat16` rejected on CPU devices |

## Appendix B: Checkpoint Version History

| Version | Experiment | Domain | ROUGE-L | Status |
|---------|------------|--------|---------|--------|
| v1 | Experiment 1 | `reasoning` | 0.2322 | Superseded |
| v2 | Experiment 1 | `default` | 0.2318 | Superseded |
| v3 | Experiment 2 | (any) | < 0.2033 | Rolled back |
| v4 | Experiment 3 | `reasoning` | 0.1982 | Active (as of last run) |

## Appendix C: Probe Question Reference Answers

**Probe**: *"Every day, a tree drops 7 leaves. How many leaves would it drop in a month of February in a non-leap year?"*

| Condition | Days Stated | Answer | Correct? |
|-----------|-------------|--------|----------|
| Base model | 28 | 196 leaves | ✅ |
| LoRA adapter v1 (Exp 1) | 31 | 217 leaves | ❌ |
| LoRA adapter v4 (Exp 3) | 30 | 210 leaves | ❌ |
| Expected | 28 | 196 leaves | ✅ |
