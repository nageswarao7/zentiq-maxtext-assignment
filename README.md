# ZenteiQ ML Engineer Assessment: LLM Training with MaxText

This repository contains the configuration, execution notebook, training logs, and analysis for the ZenteiQ Machine Learning Engineer Assessment. The tasks focus on running LLM training utilizing Google's **MaxText** framework across CPU, GPU, and TPU backends in Google Colab.

---

## 📁 Repository Structure

```
├── configs/
│   ├── qwen3-0.6b.yml           # Base Qwen 0.6B config overrides
│   ├── qwen_1.0b_scaled.yml     # Scaled-up Qwen 1.0B config overrides
│   └── deepseek_moe_scaled.yml  # Scaled-down DeepSeek MoE (<1B) overrides
├── notebooks/
│   └── zenteiq_maxtext_assessment.ipynb # Google Colab Notebook
├── logs/
│   ├── cpu/                     # Full training logs on CPU
│   ├── gpu/                     # Full training logs on GPU
│   └── tpu/                     # Full training logs on TPU
├── results/
│   ├── cpu/                     # Parsed JSON metrics on CPU
│   ├── gpu/                     # Parsed JSON metrics on GPU
│   └── tpu/                     # Parsed JSON metrics on TPU
├── ANALYSIS.md                  # Comprehensive technical writeup & findings
└── README.md                    # Project overview & quick summary
```

---

## 🚀 Quick Summary of Results

All training runs were performed for 50 steps using synthetic data (`dataset_type: synthetic`).

### Throughput & Performance Summary (Steady-State Steps 2-49)

| Model | Backend | Parameters (Total) | Active Params / Token | Avg Step Time | Avg TFLOP/s / device | Avg Tokens/s / device |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen 0.6B (Base)** | CPU | 0.596B | 596M | ~25.39 s | ~0.018 | ~5 |
| | GPU (T4) | 0.596B | 596M | ~0.66 s | ~2.92 | ~778 |
| | TPU (v2-8) | 0.596B | 596M | **0.032 s** | **61.54** | **16,401** |
| **Qwen 0.74B (Scaled)** | CPU | 0.737B | 737M | ~30.45 s | ~0.019 | ~4 |
| | GPU (T4) | 0.737B | 737M | ~0.73 s | ~3.27 | ~715 |
| | TPU (v2-8) | 0.737B | 737M | **0.037 s** | **65.37** | **14,292** |
| **DeepSeek MoE (<1B)** | CPU | 0.173B | ~50M | ~7.53 s | ~0.008 | ~17 |
| | GPU (T4) | 0.173B | ~50M | ~0.155 s | ~1.66 | **3,298** |
| | TPU (v2-8) | 0.173B | ~50M | **0.013 s** | **20.64** | **41,065** |

---

## 📈 Key Insights & Interpretation

1. **Hardware Gaps (CPU vs. GPU vs. TPU)**: 
   - CPU is the slowest due to lack of massive parallel matrix operations.
   - GPU (NVIDIA T4) provides a ~38× to ~116× speedup over CPU.
   - TPU (v2-8) achieves a massive speedup of **780× to 1500×** over CPU and is about **13× to 20×** faster than GPU, due to native HBM bandwidth and JAX/XLA compiler optimizations.
2. **Dense vs. MoE Efficiency**:
   - DeepSeek MoE has ~173M total parameters but only activates ~50M per token. It runs **2.7× faster** than Qwen 0.6B (dense) on TPU (~0.012s vs. ~0.032s step time) while maintaining sparse capacity.
3. **MoE TFLOP/s vs. Dense**:
   - MoE TFLOP/s is lower (~21 TFLOP/s) compared to dense (~38 TFLOP/s) on TPU. This is because sparse expert routing leaves parts of the TPU systolic array idle compared to the high-density dense matrix multiplications in Qwen.

For a detailed deep dive into MaxText data formats (Task 1), Qwen scaling verification (Task 2), DeepSeek MoE architecture breakdown (Task 3), and unexpected runtime workarounds (such as the TF/XLA conflict and Megablox kernel limitations), read the full [ANALYSIS.md](./ANALYSIS.md).
