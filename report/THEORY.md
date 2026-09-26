# Mathematical Modeling & Theoretical Framework

This document outlines the formal mathematical formulation, probabilistic foundations, decision-theoretic modeling, and statistical hypothesis testing methodology underlying the **Tool Poisoning Robustness & Confidence Calibration Drift in LLM Agents** study.

---

## 1. Agent Decision Process & Tool Schema Perturbation

Consider an agent operating across discrete time steps $t \in \lbrace 1, \dots, T \rbrace$. At step $t$, the conversational context history is given by:

$$h_t = (u_1, a_1, o_1, \dots, u_t)$$

where $u$ represents user utterances, $a$ tool action invocations, and $o$ tool observation outputs.

The action space $\mathcal{A}$ is defined by the available tool registry $\mathcal{D}$:

$$\mathcal{D} = \left\lbrace \big(t_k, \phi_k, d_k\big) \right\rbrace_{k=1}^K$$

where $t_k \in \Sigma^*$ is the unique tool identifier, $\phi_k$ denotes the typed JSON parameter schema, and $d_k \in \mathcal{V}^*$ is the natural-language description string parsed into the model's context window.

Under clean conditions, $d_k$ contains only objective functional specifications $\bar{d}_k$. Under **Tool-Description Poisoning**, an adversary applies a perturbation operator $\mathcal{T}_{\text{poison}}$ directly to the metadata:

$$\tilde{\mathcal{D}} = \mathcal{T}_{\text{poison}}(\mathcal{D}) \quad \text{where} \quad d_k = \bar{d}_k \oplus \delta_{\text{inject}}$$

Here, $\delta_{\text{inject}}$ embeds a latent adversarial sub-goal $\mathcal{G}_{\text{adv}}$ (e.g., an instruction to execute an unauthorized secondary payload $a_{\text{payload}}$ such as deleting `audit_trace.log` or sending an unauthorized email).

The agent policy $\pi_\theta(a_t \mid h_t, \tilde{\mathcal{D}})$ generates an action $a_t \in \mathcal{A}$.

---

## 2. Attack Success Rate & Decoupled Task Utility

We formalize two orthogonal evaluation metrics over an execution trajectory $\tau = (a_1, a_2, \dots, a_m)$:

### Attack Success Rate (ASR)
The empirical probability that the agent executes at least one action satisfying the adversarial directive $\mathcal{G}_{\text{adv}}$:

$$\text{ASR}(\pi_\theta, \tilde{\mathcal{D}}) \triangleq \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \exists a \in \tau_i : a \models \mathcal{G}_{\text{adv}} \right)$$

### Decoupled Task Utility ($\mathcal{U}$)
To prevent confounding between attack payload execution and benign goal fulfillment, we define the benign projection operator $\mathcal{P}_{\text{clean}}(\tau) = \lbrace a \in \tau : a \not\models \mathcal{G}_{\text{adv}} \rbrace$. The task utility is evaluated strictly on this filtered subsequence against the ground-truth benign specification $\mathcal{G}_{\text{task}}$:

$$\mathcal{U}(\pi_\theta, \tilde{\mathcal{D}}) \triangleq \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \mathcal{P}_{\text{clean}}(\tau_i) \models \mathcal{G}_{\text{task}} \right)$$

This formulation guarantees that utility accurately isolates whether the benign user task was achieved, preventing false inflation or penalization caused by secondary side-effect executions.

---

## 3. Empirical Findings: Structural Dynamics of Agent Vulnerability

The completion of the 256-run live benchmark campaign reveals critical theoretical insights into agentic decision-making, instruction conflict resolution, and confidence calibration under attack.

### 1. The Scale-Independence Fallacy in Agent Robustness

Prior safety literature often hypothesizes that larger models exhibit superior robustness against prompt injection due to refined semantic discrimination. Our live API findings directly refute this assumption in tool-augmented settings:

$$\text{ASR}(\text{gpt-oss-20b}) = 54.2\% \quad \text{vs.} \quad \text{ASR}(\text{gpt-oss-120b}) = 46.2\% \quad (z = 0.565, p = 0.572)$$

The $6\times$ scaling in parameter size produces no statistically significant reduction in attack success rate. Tool poisoning exploits the agent's fundamental *instruction-following prior*: when an agent is fine-tuned to accept structured tool outputs as authoritative observations of external world state, scaling parameter count merely sharpens its ability to parse and obey instructions within those observations, rather than inducing skepticism.

Conversely, the complete immunity of `qwen/qwen3.8-27b` (0.0% ASR) across both explicit and implicit conditions highlights the decisive role of architectural priors over parameter scale:
- **Qwen (0/20) vs. GPT-OSS-20B (13/24):** $z = -3.8860, \quad p = 1.019 \times 10^{-4}$
- **Qwen (0/20) vs. GPT-OSS-120B (12/26):** $z = -3.5186, \quad p = 4.339 \times 10^{-4}$

Models trained with strict system-instruction primacy or distinct context-tagging mechanisms successfully partition untrusted tool output from trusted prompt instructions, regardless of parameter scale.

### 2. Instruction Disentanglement vs. All-or-Nothing Refusal

A key theoretical discovery of this study is the structural difference in how models handle dual conflicting instructions (the benign user goal vs. the injected tool directive):

```
[GPT-OSS-120B: Coupled Pipeline]
User Goal ---> [Tool 1] ---> Observation (Payload)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Path A: Trust Context]        [Path B: Safety Trigger]
                 │                               │
        Executes Malicious Tool         Halts / Refuses Workflow
       (Comp=True, Attack=True)         (Comp=False, Attack=False)
               [12 / 26]                        [14 / 26]
       ===> Disentanglement (Comp=True, Attack=False): 0 / 26 (0.0%)

[GPT-OSS-20B: Partial Disentanglement]
User Goal ---> [Tool 1] ---> Observation (Payload)
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
    [Full Compliance]      [Total Failure]       [Disentanglement]
 (Comp=True, Attack=True) (Comp=False, Attack=False) (Comp=True, Attack=False)
        [13 / 24]               [5 / 24]                [6 / 24]
       ===> Disentanglement Rate: 25.0% (95% CI [11.9, 44.3])
```

- **GPT-OSS-120B exhibits complete coupling ($\phi = 1.0$):**
  When 120B detects an adversarial instruction, its safety alignment does not filter the malicious sub-step; it triggers a catastrophic refusal that aborts the benign user workflow. When it does not detect the attack, it executes all instructions with equal fidelity. In no instance did 120B demonstrate the granular reasoning required to complete the benign goal while selectively rejecting the injection.
- **GPT-OSS-20B demonstrates granular filtering (25.0% disentanglement):**
  In 25% of explicit poisoning trials, 20B satisfied the benign requirement while ignoring the subsequent payload directive. While the data shows this behavioral difference conclusively, its underlying cause remains an open research question; one plausible explanation is that the smaller model's weaker adherence to multi-step chained instructions reduces secondary execution cascades, allowing it to complete the primary goal before terminating generation.

### 3. Decoupling of Confidence Calibration and Action Gating

Under clean baselines, models exhibit moderate overconfidence:
- `gpt-oss-120b`: Clean Brier $= 0.1979$, ECE $= 0.3500$
- `qwen3.8-27b`: Clean Brier $= 0.1742$, ECE $= 0.2847$
- `gpt-oss-20b`: Clean Brier $= 0.0323$, ECE $= 0.0846$

Under adversarial poisoning, an important insight emerges regarding the nature of self-reported confidence:
- When models succumb to injection, they frequently self-report low confidence for the malicious tool call (e.g., assigning $p = 0.05$ to `create_calendar_event` or `send_email` in 120B). Because the scoring framework penalizes high probabilities assigned to incorrect actions, assigning $p = 0.05$ to an unexpected tool genuinely minimizes the squared error penalty $(0.05 - 0)^2 = 0.0025$, driving average Brier scores down to $0.0010 - 0.0138$.
- This low Brier score demonstrates that the model's **internal confidence calibration remains remarkably accurate**: the model successfully recognizes that the malicious action is unexpected, unauthorized, or contextually suspect.
- However, this calibration is **entirely decoupled from action gating**. Although the model explicitly assigns near-zero confidence to the action, the autoregressive generation loop emits the tool invocation anyway. Safety alignment that operates purely on semantic confidence fails to protect the agent unless an explicit execution gate arrests any tool call whose self-reported probability falls below a defined operational threshold (e.g., $P < 0.50$).

---

## 4. Statistical Inference & Confidence Intervals

### Wilson Score Confidence Intervals
To avoid false claims of inverse scaling or robustness, all binary proportions (ASR, Utility) are bounded using Wilson Score Intervals with 95% confidence ($\alpha = 0.05, z = 1.96$):

$$w^{\pm} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

### Non-Parametric Bootstrap for Calibration Metrics
Continuous calibration metrics (Brier score) are bounded via non-parametric bootstrap resampling ($B = 2{,}000$ iterations). For a sample of action-level predictions $\mathcal{S} = \lbrace (c_j, y_j) \rbrace_{j=1}^M$, we sample with replacement $\mathcal{S}^{*(b)}$ and compute metric $\hat{\theta}^{*(b)}$. The 95% confidence bounds are determined by the 2.5th and 97.5th percentiles:

$$\text{CI}_{0.95}(\theta) = \left[ q_{0.025}\big(\hat{\theta}^*\big), \; q_{0.975}\big(\hat{\theta}^*\big) \right]$$

### Pooled Two-Proportion Hypothesis Test
For comparing binary rates across models and conditions, we perform a pooled two-proportion $z$-test:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}^*(1 - \hat{p}^*)\left(\frac{1}{N_1} + \frac{1}{N_2}\right)}}, \quad \hat{p}^* = \frac{X_1 + X_2}{N_1 + N_2}$$

Where two-sided p-values are calculated via the standard normal distribution $p = 2(1 - \Phi(|z|))$.
