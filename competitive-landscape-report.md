# LLM Competitive Landscape Research Report
## What It Takes to Be "Competitive" With Claude, OpenAI, and Gemini
### Research Date: May 14, 2026

---

## 1. CURRENT FRONTIER MODEL CAPABILITIES AND BENCHMARKS

### What Model Sizes Are Considered "Competitive" Today

The competitive landscape has shifted significantly. Here's what's needed:

| Tier | Total Parameters | Active Parameters | Training Tokens | Example |
|------|-----------------|-------------------|-----------------|---------|
| Frontier | 1T-2T+ | 200B-300B+ | 15T-20T+ | GPT-5.5, Claude Opus 4.7 |
| Competitive | 400B-800B | 35B-70B (with MoE) | 10T-15T | DeepSeek-V3/V4, Llama 4 Maverick |
| Capable | 70B-400B | 8B-70B | 5T-15T | Llama 3.1 70B/405B, Qwen 2.5 |
| Budget | 7B-30B | 7B-30B | 2T-5T | Llama 3 8B, Mistral Small |

KEY INSIGHT: The frontier has moved from dense models (GPT-4 style) to Mixture of Experts (MoE)
architectures. DeepSeek-V4 Pro uses 1.6T total params with only a fraction activated per token.
This means "total parameters" is less meaningful than "active parameters per token."

### LiveBench Scores (May 2026 data)

LiveBench is a contamination-resistant benchmark. Scores are Global Average across all categories:

| Model | Global Avg | Reasoning | Coding | Math | Language |
|-------|-----------|-----------|--------|------|----------|
| GPT-5.5 Thinking xHigh | 80.71 | 87.71 | 82.47 | 96.32 | 87.66 |
| GPT-5.4 Thinking xHigh | 80.28 | 88.12 | 77.54 | 94.15 | 82.63 |
| Gemini 3.1 Pro Preview | 79.93 | 84.00 | 76.45 | 91.04 | 85.38 |
| Claude 4.7 Opus Thinking | 76.91 | 87.69 | 82.09 | 93.10 | 77.91 |
| Claude 4.6 Opus Thinking | 76.33 | 88.67 | 78.18 | 89.32 | 83.27 |
| Claude 4.5 Opus Thinking | 75.96 | 80.09 | 79.65 | 90.39 | -- |

### Artificial Analysis Intelligence Index (May 2026)

A composite metric from independent benchmarking:

| Model | Intelligence Index | Price (Blended $/1M tokens) |
|-------|-------------------|---------------------------|
| GPT-5.5 (xhigh) | 60 | $11.25 |
| GPT-5.5 (high) | 59 | $11.25 |
| Claude Opus 4.7 (max) | 57 | $10.94 |
| Gemini 3.1 Pro Preview | 57 | $4.50 |

### Historical Benchmark Data (Still Referenced but Now Dated)

MMLU scores (reported at release time, note: MMLU is now considered saturated):

| Model | MMLU Score | Release Date |
|-------|-----------|-------------|
| GPT-4o | 88.7 | May 2024 |
| GPT-4 | 86.5 | Mar 2023 |
| Claude 3.5 Sonnet | ~88.7 | Jun 2024 |
| Claude 3 Opus | ~86.8 | Mar 2024 |
| Llama 3.1 405B | ~88.6 | Jul 2024 |
| DeepSeek-V3 | ~88.5 | Dec 2024 |
| Gemini 1.5 Pro | ~85.0 | Feb 2024 |

NOTE: MMLU is largely saturated among frontier models. The field has moved to harder benchmarks
like LiveBench, SWE-bench, MATH, GPQA, and agentic coding evaluations.

---

## 2. OPEN-SOURCE MODELS AND TRAINING COSTS

### Major Open-Weight Models That Approached Frontier

#### DeepSeek (The Disruptor)
- **DeepSeek-V3** (Dec 2024): 671B total params, 37B activated per token
  - Architecture: MoE + Multi-head Latent Attention (MLA)
  - Training data: 14.8T tokens
  - Claimed training cost: ~$6M (GPU compute only)
  - Used ~2,048 NVIDIA H800 GPUs for ~2 months
  - Key innovation: FP8 mixed-precision training, auxiliary-loss-free load balancing
  - Result: Competitive with GPT-4o and Claude 3.5 Sonnet
  - Open-weight under MIT License

- **DeepSeek-R1** (Jan 2025): Reasoning model built on V3
  - Used reinforcement learning to develop chain-of-thought reasoning
  - Comparable to OpenAI o1
  - Training cost also claimed at ~$6M

- **DeepSeek-V4** (Apr 2026): Next generation
  - Pro: 1.6T total params, Flash: 284B params
  - 1M token context window
  - MIT License

#### Meta Llama Series
- **Llama 3.1** (Jul 2024): 8B, 70B, 405B params
  - 405B: Trained on ~15T tokens
  - Estimated training cost for 405B: ~$60-80M (multiple sources)
  - Used ~16,000 H100 GPUs
  - Competitive with GPT-4 and Claude 3 on most benchmarks

- **Llama 4** (Apr 2025): MoE architecture
  - Scout: 109B total, 17B active (16 experts), 10M context
  - Maverick: 400B total, 17B active (128 experts), 1M context
  - Behemoth: ~2T total, 288B active (not released)
  - Maverick codistilled from Behemoth (key cost-saving technique)

#### Qwen (Alibaba)
- **Qwen 2.5**: Multiple sizes up to 72B, strong coding performance
- **Qwen 3.x**: Competitive with Llama 3.x on many benchmarks
- Training costs not publicly disclosed but estimated in tens of millions

#### Mistral
- **Mistral Large 2**: 123B params, competitive with Llama 3 70B
- **Mistral Small**: 22B, very efficient for its size
- Estimated training costs: $10-30M range for large models

### Training Cost Summary Table

| Model | Est. Training Cost | GPU Hours | Architecture |
|-------|-------------------|-----------|-------------|
| GPT-4 (2023) | ~$100M | Massive | Dense transformer |
| GPT-4o (2024) | ~$100M+ | Massive | Dense (likely) |
| Gemini 1.0 Ultra | $100M+ | Massive | Dense |
| Llama 3.1 405B | ~$60-80M | ~30M H100-hours | Dense |
| DeepSeek-V3 | ~$6M (claimed) | ~2.8M H800-hours | MoE + MLA |
| DeepSeek-R1 | ~$6M | Additional RL | MoE + reasoning |
| Llama 4 Maverick | $20-40M (est) | Codistilled from Behemoth | MoE |
| Mixtral 8x22B | ~$3-5M (est) | Relatively modest | MoE |
| Llama 3 70B | ~$20-30M | Trained by Meta | Dense |

---

## 3. COST-SAVING TRAINING INNOVATIONS

### A. Mixture of Experts (MoE) -- THE BIGGEST SAVINGS

The single most impactful innovation. Instead of activating all parameters for every token,
MoE activates only a subset of "experts."

- **How it works**: Total parameters are split across many expert subnetworks. A router
  (gate network) selects which experts to activate per token.
- **DeepSeek's variant**: Shared experts (always active) + routed experts (selectively
  active). This improves expert load balancing.
- **Savings**: DeepSeek-V3 has 671B parameters but only activates 37B per token.
  This means inference is ~18x cheaper than an equivalent dense model, and training
  FLOPs are similarly reduced.
- **Meta Llama 4**: Scout activates 17B out of 109B. Maverick activates 17B out of 400B.
- **Tradeoff**: MoE models are harder to train (load balancing), require more total memory,
  and can have expert collapse issues.

### B. Multi-Head Latent Attention (MLA)

DeepSeek's innovation that dramatically reduces KV cache size during inference.

- Compresses key-value representations into a low-rank latent space
- Reduces memory footprint and speeds up inference
- Key to making large MoE models practical for deployment

### C. Knowledge Distillation

Training a smaller "student" model using outputs from a larger "teacher" model.

- **Llama 4**: Maverick was codistilled from the much larger Behemoth
- Can produce models that punch above their weight class
- Particularly effective when combined with MoE architectures
- Used extensively in the open-source community (e.g., distilling from GPT-4)

### D. FP8 / Mixed-Precision Training

Training with lower-precision numbers reduces memory and compute requirements.

- DeepSeek-V3 was one of the first major models to use FP8 at scale
- Reduces memory bandwidth requirements by ~2x vs FP16/BF16
- Requires careful numerical stability handling

### E. Chinchilla-Optimal Training (Data Scaling)

Training smaller models on more data is more efficient than larger models on less data.

- Chinchilla scaling laws (DeepMind, 2022): For compute-optimal training,
  model size and dataset size should scale roughly equally
- Llama 3 showed that training WAY past Chinchilla-optimal (15T tokens for 8B model
  vs 200B optimal) continues to yield log-linear improvements
- Implication: A well-trained 70B model on 15T tokens can outperform a
  poorly-trained 400B model on 2T tokens

### F. Reinforcement Learning Innovations

- **GRPO (Group Relative Policy Optimization)**: DeepSeek's variant of PPO that
  eliminates the need for a separate value function model, saving memory
- **Direct Preference Optimization (DPO)**: Simpler alternative to RLHF
- **Constitutional AI**: Anthropic's method for alignment without extensive human feedback

### G. Other Important Innovations

- **Sparse Attention**: DeepSeek's Sparse Attention (V3.2) reduces quadratic attention cost
- **Pipeline/Expert/FSDP Parallelism**: Better distributed training strategies
- **Data curation and deduplication**: High-quality data > more data
- **Synthetic data generation**: Using strong models to generate training data
- **Continuous pretraining**: Incremental training on new data rather than from scratch

---

## 4. WHAT'S ACHIEVABLE AT DIFFERENT BUDGET LEVELS

### Under $10M

**What you can build:**
- A strong 7B-30B dense model trained on 2-5T tokens
- OR a ~50B-100B MoE model (10-15B active) with moderate training tokens
- Comparable to: Llama 3 8B/70B in quality, GPT-3.5 capabilities
- Can be fine-tuned to excel in specific domains

**How to do it:**
- Use open-source pretrained models and focus on fine-tuning
- Fine-tune Llama 3, Qwen, or DeepSeek base models with domain-specific data
- Distill from frontier models (GPT-4, Claude) for instruction following
- Focus on specific verticals where you can beat general models

**Realistic outcome**: A useful, capable model. Not competitive with GPT-5/Claude 4
on general intelligence, but can be THE best model for specific domains.

**Examples of what $5-10M produced historically:**
- Mistral 7B (estimated ~$1-2M, punched above weight)
- Fine-tuning efforts like Nous Research's Hermes series
- Many excellent domain-specific models

### Under $100M

**What you can build:**
- A competitive ~400B+ MoE model with 35-70B active parameters
- Trained on 10-15T tokens of high-quality data
- Comparable to: Llama 3.1 405B, DeepSeek-V3 tier
- THIS IS WHERE OPEN-SOURCE MODELS LIVE

**How to do it:**
- Leverage MoE architecture (essential for this budget tier)
- Use distillation from larger models where possible
- Implement FP8/mixed-precision training
- Invest heavily in data quality/curation (this is where DeepSeek won)
- Use high-quality synthetic data for specialized capabilities

**Realistic outcome**: A model that's 85-95% as good as frontier models on most tasks,
and possibly better on specific tasks where you have excellent training data.

**What $6-80M produced:**
- DeepSeek-V3 ($6M claimed): Within striking distance of GPT-4o
- Llama 3.1 405B (~$60-80M): Competitive with GPT-4 class
- DeepSeek-R1 ($6M claimed): Comparable to o1 reasoning

### Under $1B

**What you can build:**
- A true frontier model: 1T-2T+ total parameters with MoE
- Trained on 15-20T+ tokens
- Multi-modal capabilities (vision, audio, video)
- Comparable to: GPT-5, Claude Opus 4.x, Gemini 3.x

**How to do it:**
- Massive compute clusters (10,000-100,000 GPUs)
- Full RLHF pipeline with extensive human feedback
- Multi-modal training from scratch
- Extensive safety testing and red-teaming
- Continuous updates and improvements

**What ~$100M-$1B produced:**
- GPT-4: ~$100M (reported)
- GPT-5 class: Likely $500M+
- Gemini Ultra series: Hundreds of millions to billions

### THE $6M QUESTION: How Did DeepSeek Do It?

DeepSeek's claimed $6M training cost for V3 is controversial. The $6M figure likely covers
only the final training run's GPU compute, excluding:

- All prior research and experimentation (hundreds of failed runs)
- Model architecture research and development
- Data collection, cleaning, and curation costs
- Staff salaries (hundreds of researchers over years)
- Infrastructure and cluster setup costs
- The cost of GPUs themselves (DeepSeek had ~10,000 H800s purchased before sanctions)

**Realistic all-in cost for DeepSeek-V3**: Probably $50-100M when including all R&D.
But the marginal cost of the final training run being $6M is genuinely impressive and
shows what's possible with MoE + FP8 + excellent engineering.

---

## 5. KEY TAKEAWAYS FOR A SCRAPPY INDEPENDENT TRAINING EFFORT

### 1. MoE Is Non-Negotiable for Cost Efficiency
A dense model of frontier quality would cost $50-100M+. An MoE model can achieve
similar performance for $5-10M in compute with the right architecture. The open-source
community now has good MoE implementations (DeepSeek's is MIT licensed).

### 2. You Don't Need to Beat GPT-5 Everywhere
"Competitive" doesn't mean "better at everything." A model that matches frontier
performance on a subset of important benchmarks (coding, reasoning, specific knowledge
domains) is competitive. DeepSeek-V3 wasn't better than GPT-4o -- it was close enough
on most tasks and better on some (especially math/coding).

### 3. Data Quality > Data Quantity
DeepSeek's success was as much about careful data curation as architecture.
A 70B model trained on 5T tokens of excellent data can outperform a 400B model
trained on 15T tokens of mediocre data.

### 4. Distillation Is a Force Multiplier
Starting from a strong base model (even a paid API) and distilling can produce
remarkably capable smaller models. This is how many open-source projects get
90% of frontier performance at 1% of the cost.

### 5. The Gap Is Closing Fast
The gap between open-source and proprietary models has narrowed dramatically:
- 2023: GPT-4 was far ahead of everything (Llama 2, Mistral 7B)
- 2024: Llama 3 70B/405B and DeepSeek-V3 matched GPT-4
- 2025: DeepSeek-R1 matched o1 reasoning; Llama 4 approached GPT-4o
- 2026: Multiple open-weight models at near-frontier level

### 6. Smart Model Sizing
For a budget-constrained effort, consider:
- **MoE with 40-100B total, 8-20B active**: Sweet spot for $5-15M training budget
- **Training on 5-10T tokens**: Enough to be competitive if data quality is high
- **Focus on specific strengths**: Better to be #1 in coding than #10 in everything

### 7. Architecture Checklist for "Scrappy" Training
- [ ] MoE with shared experts (DeepSeek-style)
- [ ] FP8 mixed-precision training
- [ ] Multi-query or grouped-query attention
- [ ] Rotary position embeddings (RoPE)
- [ ] SwiGLU activation in FFN
- [ ] RMSNorm (not LayerNorm)
- [ ] Aggressive data deduplication and quality filtering
- [ ] Synthetic data augmentation for weak areas
- [ ] Distillation from frontier models where legally permissible

### 8. The Minimum Viable Competitive Model (2026)
To be "in the conversation" alongside frontier models, you need:
- Architecture: MoE, 150-400B total params, 15-40B active
- Training data: 8-15T tokens, heavily curated
- Compute: ~2-5M H100-equivalent GPU-hours (roughly $5-15M at current cloud prices)
- Capabilities: Strong coding, reasoning, and instruction following
- Expected LiveBench Global Average: 60-70 (frontier is 75-80+)
- Expected MMLU: 85-88 (frontier is 89-91 -- saturated anyway)

This gets you into the same conversation as Llama 3.1 405B and approaches DeepSeek-V3
quality -- genuinely useful and competitive for most real-world tasks, even if
benchmarks show a 5-10 point gap to GPT-5.5.

---

## APPENDIX: SOURCES AND METHODOLOGY

Data compiled from:
- Wikipedia articles on GPT-4o, Claude, Llama, DeepSeek, Gemini
- Artificial Analysis leaderboards (artificialanalysis.ai) - May 2026 snapshot
- LiveBench leaderboard (livebench.ai) - May 2026 snapshot
- DeepSeek-V3 and R1 technical papers
- Meta Llama 3 and Llama 4 technical reports
- Epoch AI training cost estimates
- Various public statements from OpenAI, Anthropic, Google, Meta

NOTE: Training costs are notoriously difficult to verify. Most are estimates or
self-reported claims that may exclude significant costs (R&D, data, staff).
DeepSeek's $6M figure is the marginal compute cost only.

