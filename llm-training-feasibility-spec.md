# Feasibility Spec: Training a Competitive LLM
## Data Sources: LibGen + Sci-Hub + Archive.org + Public Web Text

**Date:** May 14, 2026
**Status:** Research spec — no implementation
**Context:** Investigating whether a competitive open-weight LLM can be trained
using gray-area knowledge sources plus public web text.

---

## 1. DATA SOURCE INVENTORY

### 1a. Library Genesis (LibGen)

| Category | Count | Est. Tokens | Notes |
|----------|-------|-------------|-------|
| Non-fiction books | 2.4M | ~250B | Avg ~300 pages, ~100K tokens/book |
| Fiction books | 2.2M | ~220B | Similar density to non-fiction |
| Science journal articles | 80M | ~800B | Avg ~8K words ~10K tokens each |
| Comics | 2M | ~5B | Mostly images, low text density |
| Magazines | 0.4M | ~10B | Variable quality, lots of ads |
| **Total LibGen** | **~87M items** | **~1.3T tokens** | Heavily English + Russian |

**Quality issues:** Scanned PDFs with OCR artifacts, inconsistent formatting,
duplicate editions, some non-English content. Requires significant cleaning.

**Download infrastructure:** Torrents + IPFS available. ~33TB total for the
non-fiction collection. Sci-Hub torrents ~77TB.

**Legal:** Under active litigation. Meta and Anthropic both used LibGen for
training (revealed in court documents 2025). Domain seizures ongoing.

### 1b. Sci-Hub

| Category | Count | Est. Tokens | Notes |
|----------|-------|-------------|-------|
| Research papers | 88.3M | ~880B | Last reported July 2022; uploads frozen |

**Quality:** Mostly PDFs, scientific writing quality is high. Heavy math/STEM
content — valuable for reasoning capabilities.

**Legal:** Multiple court losses. Domain hopping. Uploads frozen since Dec 2020.
Active mirrors exist. Same grey area as LibGen — major AI labs have allegedly
used it.

### 1c. Archive.org (Internet Archive)

| Category | Count | Est. Tokens | Notes |
|----------|-------|-------------|-------|
| Text collection | 38M+ books/texts | ~3-4T | Includes Open Library, scanned texts |
| Wayback Machine | 835B+ web pages | ~100T+ raw | Massive but low signal-to-noise |
| Academic texts | Millions | ~500B | Overlaps with LibGen/Sci-Hub |
| **Relevant subset** | **~42M items** | **~4-5T** | Excluding Wayback Machine raw pages |

**Quality:** OCR quality varies (some excellent, some poor). Wayback Machine
data is extremely noisy — needs aggressive filtering. Has clean ePub/PDF
collections that are higher quality.

**Legal:** Non-profit library, mostly public domain / licensed. Wayback Machine
legal status for AI training untested. Books collection includes copyrighted
works via controlled digital lending — different legal profile from LibGen.

**Download:** Archive.org provides direct downloads, bulk APIs, and torrents.
Relatively easy to access programmatically.

### 1d. Public Web Text (FineWeb / CommonCrawl / The Pile)

| Dataset | Size | Est. Tokens | Quality |
|---------|------|-------------|---------|
| FineWeb (HuggingFace) | — | 15T | High: filtered CommonCrawl, deduped |
| FineWeb-Edu | — | 1.3T | Very high: educational subset |
| CommonCrawl (raw) | ~400TB/month | ~100T+/snapshot | Low: needs extensive filtering |
| The Pile (v1) | 825GB | ~300B | Medium: curated, academic + web + code |
| RedPajama-v2 | — | 30T | Medium: filtered CommonCrawl |
| DCLM (DataComp-LM) | — | ~4T | Very high: heavily filtered, benchmarked |
| C4 (Colossal Clean Crawled Corpus) | 750GB | ~175B | High: cleaned CommonCrawl |
| **Best combinable subset** | — | **~20-25T** | FineWeb + DCLM + curated sources |

**Recommendation:** Use FineWeb + FineWeb-Edu + DCLM as the web text base.
These are pre-processed, deduplicated, and benchmarked. Skip raw CommonCrawl
unless you have a pipeline team.

### 1e. Total Unique Data Pool

```
LibGen books + articles:     ~1.3T tokens
Sci-Hub papers:              ~0.9T tokens  
Archive.org texts:           ~4.0T tokens
Public web (FineWeb etc):   ~20.0T tokens
─────────────────────────────────────
Gross total:                 ~26T tokens
After dedup (est. 30%):     ~18T unique tokens
After quality filter:        ~12-15T high-quality tokens
```

**12-15T high-quality tokens** is sufficient to train a competitive model.
DeepSeek-V3 used 14.8T tokens. Llama 3.1 405B used ~15T tokens.

---

## 2. HARDWARE REQUIREMENTS

### 2a. Current GPU Pricing (May 2026)

From RunPod Community Cloud (spot pricing, no commitment):

| GPU | VRAM | RAM | vCPUs | $/hr |
|-----|------|-----|-------|------|
| H200 | 141GB | 276GB | 24 | $3.99 |
| H100 NVL | 94GB | 94GB | 16 | $3.07 |
| H100 SXM | 80GB | 125GB | 20 | $2.99 |
| H100 PCIe | 80GB | 188GB | 16 | $2.39 |
| A100 SXM | 80GB | 125GB | 16 | $1.49 |
| A100 PCIe | 80GB | 117GB | 8 | $1.39 |
| L40S | 48GB | 94GB | 16 | $0.86 |

Reserved/enterprise pricing: typically 30-50% discount on 1-3 year commitments.
Together AI, CoreWeave, Lambda Labs offer similar rates.

### 2b. Cluster Requirements by Model Scale

#### Budget Tier: 7-30B model ($50K-$500K compute)
```
Architecture: Dense transformer or small MoE
Parameters: 7B-30B (dense) or 40-100B MoE (8-20B active)
Training tokens: 2-5T
GPUs needed: 64-256 H100s
Training time: 2-6 weeks
GPU-hours: ~100K-500K H100-hours
Compute cost: ~$250K-$1.2M at spot pricing
```

#### Competitive Tier: DeepSeek-V3-class ($5M-$15M compute)
```
Architecture: MoE with shared experts
Parameters: 400-700B total, 35-70B active
Training tokens: 10-15T
GPUs needed: 2,000-4,000 H100s
Training time: 2-3 months
GPU-hours: ~3M-8M H100-hours
Compute cost: ~$5M-$15M at spot pricing
            ~$8M-$25M at reserved pricing
```

#### Frontier Tier: GPT-5-class ($100M-$500M+ compute)
```
Architecture: Large MoE, multimodal
Parameters: 1T-2T+ total, 200B+ active
Training tokens: 15-20T+
GPUs needed: 16,000-50,000+ H100s
Training time: 3-6 months
GPU-hours: ~30M-100M+ H100-hours
Compute cost: $100M-$500M+
```

### 2c. Infrastructure Beyond GPUs

| Component | Requirement | Est. Cost |
|-----------|-------------|-----------|
| Networking | InfiniBand or RoCE, 400Gb/s+ | Included in cluster rental |
| Storage (training data) | 50-200TB raw, 10-30TB processed | $3K-10K/mo |
| Storage (checkpoints) | 10-50TB, high IOPS | $2K-8K/mo |
| CPU nodes (data prep) | 50-200 vCPUs | $2K-5K/mo |
| Bandwidth (data download) | Multi-gigabit, 100-500TB total | $1K-5K one-time |

**Total infra for competitive tier:** ~$10K-25K/month for non-GPU resources.

### 2d. What $10K Buys (Daytona Credits)

At RunPod pricing:
- ~2,500 H100-hours (at $3.99/hr)
- OR ~4,200 H100 PCIe-hours (at $2.39/hr)
- OR ~7,200 A100-hours (at $1.39/hr)

**This is useful for:** Experimentation, small-scale training runs (100M-1B
param models), fine-tuning existing models, data preprocessing. It is NOT
nearly enough for training a competitive model from scratch. $10K is about
0.2% of what you'd need for a DeepSeek-V3-class training run.

---

## 3. DATA PIPELINE

### 3a. Pipeline Stages

```
[Download] → [Extract Text] → [Language Detect] → [Quality Filter]
    → [Deduplicate] → [PII Removal] → [Tokenize] → [Pack Sequences]
```

### 3b. Detailed Pipeline

#### Stage 1: Download & Ingest
- **LibGen:** Download via torrents (~33TB non-fiction). Resumable.
  Time: 1-2 weeks on gigabit.
- **Sci-Hub:** Torrents available (~77TB). Time: 2-4 weeks on gigabit.
  Alternatively, partner with existing Sci-Hub mirrors.
- **Archive.org:** Direct downloads via archive.org APIs. Prioritize
  text collections (not raw Wayback Machine). ~20-50TB for curated texts.
  Time: 1-3 weeks.
- **FineWeb/DCLM:** Download from HuggingFace datasets. ~2-5TB compressed.
  Time: 1-3 days on gigabit.

**Total raw download:** ~50-100TB
**Bandwidth cost:** $1K-5K (most from torrents, free)

#### Stage 2: Text Extraction
- **PDFs** (LibGen, Sci-Hub): Use marker-pdf, PyMuPDF, or Nougat for
  academic papers. Marker-pdf is state-of-the-art for converting PDF to
  clean markdown/text. ~$2K-10K in compute for 160M+ PDFs.
- **ePub/MOBI** (Archive.org, LibGen): Standard extraction libraries.
- **HTML** (Wayback Machine, web text): Trafilatura or resiliparse.
- **OCR needed:** For scanned books — Tesseract or cloud OCR API.
  This is the most expensive step if needed at scale.

**Key insight:** Much of LibGen's collection is already extracted
by previous projects (The Pile, peS2o, etc.). Check for pre-extracted
datasets before re-extracting.

#### Stage 3: Quality Filtering
- Language detection (fastText): Keep English + top 5-10 languages
- Perplexity scoring: Remove gibberish, boilerplate, SEO spam
- Length filter: Remove docs <100 chars or >100K chars
- Repetition filter: Remove docs with high n-gram repetition
- URL/email/phone stripping
- Curriculum quality classifier (used by FineWeb-Edu)

**Expected loss:** 30-50% of raw tokens removed

#### Stage 4: Deduplication
- Fuzzy dedup (MinHash): Remove near-duplicate documents
- Exact substring dedup: Remove duplicate paragraphs within/across docs
- URL-based dedup: Remove documents from same URL

**Expected loss:** 20-30% from combined sources (significant overlap
between Sci-Hub and LibGen, and between LibGen and Archive.org)

#### Stage 5: Tokenization & Packing
- Train custom tokenizer or reuse established one (Llama 3 tokenizer
  is widely compatible)
- Pack documents into 2048-8192 token sequences
- Shard into .bin files for efficient training

### 3c. Pipeline Costs & Timeline

| Stage | Compute Cost | Time (small team) |
|-------|-------------|-------------------|
| Download | $1K-5K | 3-6 weeks |
| Text extraction | $3K-15K | 2-4 weeks |
| Quality filtering | $2K-5K | 1-2 weeks |
| Deduplication | $3K-10K | 1-2 weeks |
| Tokenization | $1K-3K | 3-5 days |
| **Pipeline total** | **$10K-38K** | **2-3 months** |

Staff: 1-2 data engineers working on this full-time.

---

## 4. COMPETITIVE LANDSCAPE

### 4a. Where the Frontier Is (May 2026)

| Model | LiveBench Global | Architecture | Training Cost |
|-------|-----------------|-------------|---------------|
| GPT-5.5 Thinking xHigh | 80.71 | MoE, 1-2T+ params | $500M+ (est) |
| GPT-5.4 Thinking xHigh | 80.28 | MoE | $100-500M (est) |
| Gemini 3.1 Pro Preview | 79.93 | MoE | $100-500M (est) |
| Claude 4.7 Opus Thinking | 76.91 | MoE | $100-500M (est) |
| DeepSeek-V4 Pro | ~76 (est) | MoE, 1.6T/284B | ~$20-50M (est) |
| Llama 4 Maverick | ~70 (est) | MoE, 400B/17B | $20-40M (est) |

### 4b. What Open Source Has Achieved

| Model | Parameters | Tokens | Compute Cost | Quality |
|-------|-----------|--------|-------------|---------|
| DeepSeek-V3 | 671B/37B MoE | 14.8T | $6M marginal | Near GPT-4o |
| DeepSeek-R1 | 671B/37B MoE | RL on V3 | $6M marginal | ~OpenAI o1 |
| Llama 3.1 405B | 405B dense | 15T | $60-80M | Near GPT-4 |
| Llama 4 Maverick | 400B/17B MoE | — | $20-40M (est) | Strong |
| Qwen 2.5-72B | 72B dense | 18T | $10-20M (est) | Very good |
| Mixtral 8x22B | 141B/39B MoE | — | $3-5M (est) | Good |

### 4c. The DeepSeek Blueprint (How to Train Competitively for ~$10M)

DeepSeek-V3 proved you can approach frontier quality at ~1/10th the cost
of US labs. Key innovations:

1. **MoE with shared experts:** 671B total params, only 37B activated
   per token (~18x compute savings vs dense)
2. **Multi-head Latent Attention (MLA):** Compresses KV cache, reduces
   inference memory dramatically
3. **FP8 mixed-precision training:** Half the memory bandwidth of FP16
4. **Auxiliary-loss-free load balancing:** Solved the hardest MoE problem
5. **Multi-token prediction:** Trains model to predict multiple future
   tokens, improving data efficiency
6. **14.8T tokens of heavily curated data:** Quality over quantity

**DeepSeek-V3 and R1 are MIT licensed.** You can literally start from
their code, architecture, and innovations. This is the fastest path.

### 4d. What Different Budgets Buy

| Budget | What You Can Build | Comparable To |
|--------|-------------------|---------------|
| $50K-$500K | 7-30B dense model, 2-5T tokens | Llama 3 8B/70B quality |
| $500K-$5M | 100-300B MoE, 5-10T tokens | Mixtral, Qwen 72B |
| **$5M-$15M** | **400-700B MoE, 10-15T tokens** | **DeepSeek-V3, competitive** |
| $15M-$50M | 400B-1T MoE, 15T+ tokens | DeepSeek-V4, Llama 4 |
| $50M-$200M | 1T+ MoE, multimodal | Approaching frontier |
| $200M-$1B+ | True frontier | GPT-5, Claude Opus, Gemini 3 |

**Sweet spot for an independent effort: $5-15M compute budget.**
This targets DeepSeek-V3 quality — genuinely useful, near-frontier
performance at 1/50th the cost of a GPT-5-class training run.

---

## 5. ESTIMATED COST BREAKDOWN

### For a DeepSeek-V3-Class Training Run ($10M Budget Target)

| Category | Low Estimate | High Estimate | Notes |
|----------|-------------|---------------|-------|
| **GPU compute (training)** | $4M | $10M | 3M-8M H100-hrs at $1.50-3.00/hr |
| GPU compute (experiments) | $500K | $2M | Architecture search, hyperparams |
| **Data pipeline** | $50K | $100K | Download, extract, clean, dedup |
| **Storage (data + checkpoints)** | $20K | $50K | 100-300TB during project |
| **Bandwidth** | $5K | $15K | Download costs |
| **Staff (3-6 months)** | $300K | $600K | 3-5 ML engineers |
| **Contingency (20%)** | $1M | $2.5M | Overage, retries, infra failures |
| **──────────────────** | **──────** | **───────** | |
| **TOTAL** | **~$6M** | **~$15M** | |

### For a Smaller "Proof of Viability" Run ($500K Target)

| Category | Estimate | Notes |
|----------|---------|-------|
| GPU compute (MoE, 100-300B total) | $250K | 100K-200K H100-hrs |
| Data pipeline | $30K | Use pre-processed datasets |
| Storage | $10K | Minimal checkpointing |
| Staff | $150K | 1-2 ML engineers |
| Contingency | $50K | |
| **TOTAL** | **~$500K** | |

### What $10K Actually Buys

The dashboard comment mentions $10,000 in Daytona credits. Here's what
that realistically enables:

- ✅ **Fine-tune existing open models** (Llama 3, Qwen, DeepSeek) on
  domain-specific data — the best use of $10K
- ✅ **Small-scale experiments** — train 100M-1B parameter models to
  validate architecture ideas
- ✅ **Data preprocessing** — extract, clean, and deduplicate a
  meaningful subset of the target data
- ✅ **Benchmark and evaluate** existing models on your target use cases
- ❌ Cannot train a competitive model from scratch
- ❌ Cannot even train a 7B model from scratch (needs ~$50-100K)

**Recommendation:** Use the $10K for fine-tuning experiments and data
preparation. Use those results to raise funding for a full training run.

---

## 6. TIMELINE

### Optimistic (well-funded, experienced team)

```
Month 1:     Data download + extraction pipeline
Month 2:     Quality filtering + dedup + tokenization
Month 3:     Architecture experiments (small scale)
Month 4-5:   Scale-up experiments (1-10B params)
Month 6:     Final architecture locked
Month 7-9:   Full training run (10-15T tokens)
Month 10:    Post-training (SFT, RLHF/DPO, red-teaming)
Month 11:    Evaluation, safety testing, release prep
```

**Total: ~11 months** from start to trained model.

### Realistic (small team, funding constraints)

```
Months 1-3:   Data pipeline (download is slow, extraction is slow)
Months 3-6:   Architecture R&D and small-scale experiments
Months 6-9:   Fundraising, cluster reservation
Months 9-12:  Scale-up experiments
Months 12-15: Full training run
Months 15-18: Post-training + evaluation
```

**Total: ~18 months** from start to trained model.

### Key Bottlenecks

1. **Data download speed** — 50-100TB over residential/cloud connections
2. **PDF extraction quality** — scanned books need OCR, which is expensive
3. **Cluster availability** — large H100 clusters have wait times
4. **Training stability** — MoE models are harder to train than dense
5. **Post-training** — RLHF/DPO requires human preference data

---

## 7. COMPARISON WITH CLAUDE/OPENAI/GEMINI

### 7a. Realistic Quality Expectations

If you execute a $10M DeepSeek-V3-style training run with the data
sources listed above, you can expect:

| Benchmark | Frontier (GPT-5.5) | Your Model (est) | Gap |
|-----------|-------------------|------------------|-----|
| LiveBench Global | 80.71 | 65-72 | 9-16 pts |
| MMLU | ~90 (saturated) | 85-88 | 2-5 pts |
| HumanEval (coding) | 95+ | 75-85 | 10-20 pts |
| MATH | 96 | 80-88 | 8-16 pts |
| General usefulness | State of art | Very good, 85-90% | Noticeable gap |

**Translation:** Your model would be genuinely useful for most real-world
tasks. It would not beat GPT-5.5 on benchmarks, but would be close enough
to GPT-4o / Claude 3.5 Sonnet quality to be a serious tool. Many users
would not notice the difference on everyday tasks.

### 7b. Advantages You'd Have

1. **MIT-licensed weights** — anyone can use, fine-tune, deploy (unlike
   OpenAI/Anthropic/Google)
2. **No content restrictions** — trained on "everything," not safety-filtered
   (this is also a risk)
3. **Uncensored knowledge** — access to the full corpus of human knowledge
   without corporate filtering
4. **Specialized strengths** — LibGen + Sci-Hub give it massive academic
   knowledge; could outperform frontier models on scientific Q&A
5. **Community alignment** — open-source community can fine-tune, improve,
   and extend

### 7c. Disadvantages

1. **Legal exposure** — training on copyrighted material without license
2. **Safety concerns** — no corporate safety team, potential for misuse
3. **Distribution risk** — hosting providers may refuse to serve the model
4. **Staffing gap** — competing against labs with 500+ researchers
5. **Inference cost** — MoE models are memory-hungry; serving costs are
   higher per token than dense models

---

## 8. KEY DECISIONS & TRADE-OFFS

### 8a. Architecture Choice

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| MoE vs Dense | **MoE** | 18x compute savings, proven by DeepSeek |
| Shared experts? | **Yes** | DeepSeek-style: shared + routed experts |
| Attention type | **MLA** | DeepSeek's KV cache compression |
| Precision | **FP8 mixed** | DeepSeek-proven at scale |
| Tokenizer | **Llama 3 tokenizer** | Widely compatible, good compression |
| Base codebase | **DeepSeek-V3 (MIT)** | Proven, open-source, MIT licensed |

### 8b. Data Strategy

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Train from scratch? | **Yes, if $5M+ budget** | Otherwise fine-tune existing model |
| Include all sources? | Curate aggressively | Quality > quantity |
| Include code? | **Yes** | StarCoder, The Stack v2 — code improves reasoning |
| Synthetic data? | **Yes** | Generate from frontier models for weak areas |
| Dedup aggressiveness | **Aggressive** | 30% dedup rate is fine; duplicates hurt more than they help |

### 8c. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Legal action | High | Medium-High | Anonymize, use offshore hosting, prepare legal defense |
| Training instability | Medium | High | Extensive small-scale testing, conservative LR schedule |
| Data quality issues | High | Medium | Over-invest in filtering; budget 30-50% rejection rate |
| Cluster availability | Medium | High | Reserve early; have backup providers |
| Funding gap | High | High | Start with $500K proof-of-viability run |
| Post-training quality | Medium | Medium | Use public preference datasets + synthetic data |

---

## 9. RECOMMENDATIONS

### Immediate Next Steps ($10K Daytona Credits)

1. **Fine-tune DeepSeek-V3 or Llama 4 on a curated subset** of LibGen +
   Sci-Hub academic papers — produce a "science-focused" model as proof
   of concept. Cost: $1K-3K.
2. **Download and pre-process a 1TB pilot dataset** to validate the
   pipeline. Cost: $2K-5K.
3. **Run scaling law experiments** at 100M-1B parameter scale to estimate
   what a full training run would achieve. Cost: $2K-5K.
4. **Benchmark fine-tuned models** against GPT-4o and Claude on scientific
   QA tasks. Use results to build a fundraising deck.

### Go/No-Go Criteria for a Full Training Run

**GO if:**
- Fine-tuned model shows meaningful improvement on science/technical tasks
- You can raise $5-15M (or secure compute sponsorship)
- You can assemble a team of 3-5 experienced ML engineers
- Legal counsel gives a manageable risk assessment

**NO-GO if:**
- Fine-tuned results are not significantly better than base models
- Cannot raise enough capital
- Cannot find team members
- Legal risk is deemed unacceptable

### Alternative: The "Fine-Tuning First" Path

Instead of training from scratch, the most capital-efficient path is:

```
Phase 1 ($10K): Fine-tune DeepSeek-V3 on domain data → science specialist
Phase 2 ($100K): Full post-training pipeline (DPO, RLVR) on curated data
Phase 3 ($500K+): Continued pre-training on 500B-1T new tokens
Phase 4 ($5M+): Full training from scratch when proven
```

This de-risks each stage and produces useful models at every phase.

---

## 10. APPENDIX: SOURCE REFERENCES

### Data Source Stats
- LibGen collection (Feb 2024): Wikipedia - Library Genesis
- Sci-Hub collection (Jul 2022): 88,343,822 files - Sci-Hub website
- Archive.org: ~38M texts, 835B+ Wayback Machine pages - Internet Archive
- FineWeb: 15T tokens - https://huggingface.co/datasets/HuggingFaceFW/fineweb

### GPU Pricing
- RunPod Community Cloud pricing, May 2026 snapshot
- H100: $2.39-3.07/hr, H200: $3.99/hr, A100: $1.39-1.49/hr

### Competitive Landscape
- LiveBench leaderboard: livebench.ai (May 2026)
- Artificial Analysis: artificialanalysis.ai (May 2026)
- DeepSeek-V3 paper: arXiv 2412.19437
- Meta Llama 3/4 technical reports
- Epoch AI: epochai.org for training cost estimates

### Key Training Cost References
- DeepSeek-V3: $6M marginal compute (claimed), ~$50-100M all-in (estimated)
- Llama 3.1 405B: $60-80M estimated
- GPT-4: ~$100M estimated
- GPT-5 class: $500M+ estimated

---

*This is a living spec. All cost estimates are directional — actual costs
depend on GPU availability, negotiated rates, and engineering efficiency.
DeepSeek's $6M figure is marginal compute cost only; total investment
including R&D was likely $50-100M.*
