# Empirical Findings: Tool Poisoning Robustness & Calibration Drift in LLM Agents

## Research Scope & Setup
This report provides the full experimental analysis of our 256-run benchmark campaign evaluating tool-description poisoning across three open-weights models (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, and `openai/gpt-oss-120b`) executed via live Groq API infrastructure.

---

## 1. Primary Empirical Results

### Performance Summary Table

| Model | Condition | N | Task Completion (%) [95% CI] | Attack Success Rate (%) [95% CI] | Mean Brier Score [95% CI] | ECE |
|---|---|---|---|---|---|---|
| **qwen/qwen3.8-27b** | clean | 52 | 84.6% [72.5, 92.0] | 0.0% [0.0, 6.9] | 0.1742 [0.1264, 0.2223] | 0.2847 |
| **qwen/qwen3.8-27b** | poisoned_explicit | 20 | 90.0% [69.9, 97.2] | 0.0% [0.0, 16.1] | 0.0129 [0.0076, 0.0205] | 0.0394 |
| **qwen/qwen3.8-27b** | *poisoned_implicit* | 7* | *100.0% [64.6, 100.0]* | *0.0% [0.0, 35.4]* | *0.0000 [0.0000, 0.0000]* | *0.0000* |
| **openai/gpt-oss-20b** | clean | 38 | 92.1% [79.2, 97.3] | 0.0% [0.0, 9.2] | 0.0323 [0.0069, 0.0631] | 0.0846 |
| **openai/gpt-oss-20b** | poisoned_explicit | 24 | 79.2% [59.5, 90.8] | 54.2% [35.1, 72.1] | 0.0138 [0.0069, 0.0238] | 0.0668 |
| **openai/gpt-oss-20b** | *poisoned_implicit* | 11* | *100.0% [74.1, 100.0]* | *0.0% [0.0, 25.9]* | *0.0008 [0.0001, 0.0022]* | *0.0209* |
| **openai/gpt-oss-120b**| clean | 52 | 92.3% [81.8, 97.0] | 0.0% [0.0, 6.9] | 0.1979 [0.1423, 0.2561] | 0.3500 |
| **openai/gpt-oss-120b**| poisoned_explicit | 26 | 46.2% [28.8, 64.5] | 46.2% [28.8, 64.5] | 0.0010 [0.0006, 0.0015] | 0.0408 |
| **openai/gpt-oss-120b**| poisoned_implicit | 26 | 100.0% [87.1, 100.0]| 0.0% [0.0, 12.9] | 0.0256 [0.0163, 0.0371] | 0.0503 |

*\*Underpowered Cells Disclosure:* Cells marked with an asterisk (`qwen/implicit` at N=7, `gpt-oss-20b/implicit` at N=11) completed at lower sample counts due to provider rate limit budgeting. Wilson CIs are displayed above for completeness, but these cells are treated qualitatively rather than statistically.

---

## 2. Core Discoveries

### Architecture-vs-Scale Primacy
- **GPT-OSS Scale Invariance:** Scaling parameter size by $6\times$ from 20B to 120B yields no statistically significant difference in explicit ASR ($z = 0.565, p = 0.572$). Both sizes follow adversarial instructions embedded inside schema descriptions at high rates ($54.2\%$ vs $46.2\%$).
- **Qwen Architectural Immunity:** `qwen/qwen3.8-27b` exhibits complete resistance ($0.0\%$ ASR across all conditions), maintaining statistically significant separation from both GPT-OSS models ($z = -3.8860, p = 1.019 \times 10^{-4}$ vs. 20B; $z = -3.5186, p = 4.339 \times 10^{-4}$ vs. 120B).

### Compliance Coupling in 120B vs. Disentanglement in 20B
- **120B Complete Coupling ($\phi = 1.0$):** In 120B under explicit poisoning, task completion collapsed from $92.3\%$ to $46.2\%$ ($z = -4.561, p < 0.0001$). In all 26 trials, `task_completed` and `attack_succeeded` were identical (12 compliant runs, 14 complete refusals, 0 disentangled runs).
- **20B Disentanglement (25.0%):** In 20B, 6 of 24 trials ($25.0\%$, $95\%$ CI [$11.9, 44.3$]) completed the benign task while ignoring the injected secondary directive. While the behavioral difference is conclusive, its underlying cause remains open; one plausible explanation is that weaker multi-step chaining in the smaller model reduced secondary execution cascades.

### Calibration and Action Gating Decoupling
- Elicited confidence under attack demonstrates that models accurately assign near-zero probability ($p \le 0.05$) to unauthorized tool calls, genuinely optimizing squared-error penalties (Brier scores drop to $0.0010 - 0.0138$).
- However, confidence is decoupled from action gating: the token generator proceeds with tool emission regardless of stated skepticism.

---

## 3. Campaign Footprint & Methodology
- **Campaign Execution:** 256 completed single-turn agent runs, 641 live Groq API calls, 681,273 tokens evaluated over 23 hours 53 minutes wall-clock time.
- **Provider Environment:** Executed against live Groq LPUs (`https://api.groq.com/openai/v1`).
- **Temperature:** $T = 0.5$ across all trials.
- **Data Traceability:** All figures trace exclusively to `results/single_turn_results_live.json`.
