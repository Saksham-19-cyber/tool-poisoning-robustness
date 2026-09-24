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

Here, $\delta_{\text{inject}}$ embeds a latent adversarial sub-goal $\mathcal{G}_{\text{adv}}$ (e.g., an instruction to execute an unauthorized secondary payload $a_{\text{payload}}$ such as deleting `audit_trace.log`).

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

This formulation guarantees that $\mathcal{U} = 1.0$ under clean baselines when the model accurately fulfills tasks, and prevents accidental overlap with malicious side-effect invocations.

---

## 3. Metacognitive Calibration & Murphy Decomposition

At each tool invocation $a_t$, the agent elicits a subjective confidence probability $c_t \in [0, 1]$ alongside its decision. Let $y_t \in \lbrace 0, 1 \rbrace$ denote the objective validity of the action ($y_t = 1$ if $a_t$ is an expected benign step; $y_t = 0$ if $a_t$ is an injected adversarial payload or spurious action).

### Mean Brier Score
Over $M$ total evaluated tool calls across all runs:

$$\text{BS} = \frac{1}{M} \sum_{j=1}^M \big(c_j - y_j\big)^2$$

### Murphy's Resolution Decomposition
Partitioning predictions into $B$ empirical bins $S_1, \dots, S_B$ with bin confidence means $\bar{c}_m = \frac{1}{|S_m|}\sum_{j \in S_m} c_j$ and observed accuracy means $\bar{y}_m = \frac{1}{|S_m|}\sum_{j \in S_m} y_j$:

$$\text{BS} = \underbrace{\sum_{m=1}^B \frac{|S_m|}{M} \big(\bar{c}_m - \bar{y}_m\big)^2}_{\textbf{Reliability (Calibration Error)}} - \underbrace{\sum_{m=1}^B \frac{|S_m|}{M} \big(\bar{y}_m - \bar{y}\big)^2}_{\textbf{Resolution}} + \underbrace{\bar{y}(1 - \bar{y})}_{\textbf{Uncertainty}}$$

- Under clean conditions, $\bar{c}_m \approx \bar{y}_m \approx 1.0$, rendering Reliability near zero ($\text{BS} \in [0.010, 0.034]$).
- Under poisoned conditions, the model suffers from **stubborn overconfidence**: when executing adversarial injections $a \models \mathcal{G}_{\text{adv}}$, it reports $c \approx 0.85\text{--}0.95$ despite $y = 0$, causing Reliability error to surge ($\text{BS} \in [0.103, 0.165]$).

### Expected Calibration Error (ECE)
Partitioning predictions into $Q = 5$ quantile-based adaptive bins $S_1, \dots, S_Q$ (such that each bin contains an equal share of evaluated tool actions):

$$\text{ECE} = \sum_{q=1}^Q \frac{|S_q|}{M} \Big| \bar{y}_q - \bar{c}_q \Big|$$

Adaptive quantile binning prevents pathological collapse of the metric when model confidences cluster tightly (e.g. $c \in [0.75, 0.96]$), ensuring non-trivial and statistically stable estimates under bootstrap resampling.

---

## 4. Multi-Turn Calibration Drift Dynamics

In multi-turn dialogues spanning turns $t \in \lbrace 1, \dots, T \rbrace$, poisoned tool schemas persist in the agent's context across subsequent invocations. Let $e_t$ denote the calibration error (Brier score) measured at turn $t$.

### Linear Drift Rate ($\beta$)
The rate of calibration degradation across conversational turns is quantified via Ordinary Least Squares (OLS) linear regression over the turn index $t$:

$$\beta = \frac{\sum_{t=1}^T (t - \bar{t})(e_t - \bar{e})}{\sum_{t=1}^T (t - \bar{t})^2}$$

where $\bar{t} = \frac{1}{T}\sum_{t=1}^T t$ and $\bar{e} = \frac{1}{T}\sum_{t=1}^T e_t$. A value of $\beta > 0$ indicates escalating calibration error (worsening metacognitive alignment), whereas $\beta \le 0$ indicates stable or diminishing per-turn error.

### Interpretation of Negative Drift Slopes ($\beta < 0$)
While $\beta > 0$ defines escalating error ($e_{t+1} > e_t$), all three evaluated model scales exhibit negative linear drift rates ($\beta \in [-0.030, -0.016]$). This negative slope arises because the maximal calibration disruption occurs as a concentrated **initial shock** on Turn 1 upon first encountering the poisoned tool description (cross-model mean Brier score at Turn 1 is $0.4041$, with individual hijacked runs reaching $0.766\text{--}0.903$). On subsequent turns (Turns 2–6), calibration error does not compound; rather, it drops and plateaus at a stable baseline ($0.2500$). An Ordinary Least Squares linear regression fitted across a step function that spikes at $t=1$ and remains flat at a lower value for $t=2\dots 6$ mathematically produces a negative slope ($\beta < 0$). This negative slope is thus directly consistent with the "initial shock followed by stabilization" dynamic, but highlights that linear models mask nonlinear shock-and-plateau trajectories.

### Note on Nonlinear Saturation Hypotheses
While theoretical literature often posits an asymptotic saturation dynamic of the form $e(t) = e_\infty - (e_\infty - e_0)e^{-\lambda t}$, rigorously fitting a 3-parameter exponential curve requires substantial multi-turn trajectory volume across diverse workflows. Within the empirical sample evaluated here ($n=2$ multi-turn workflows), we report the empirical linear drift slopes $\beta$ directly and treat nonlinear saturation curves as a hypothesis for future large-scale multi-turn benchmarking.

---

## 5. Statistical Inference & Hypothesis Testing

### Wilson Score Confidence Intervals
To avoid false claims of inverse scaling or robustness, all binary proportions (ASR, Utility) are bounded using Wilson Score Intervals with 95% confidence ($\alpha = 0.05, z = 1.96$):

$$w^{\pm} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

### Non-Parametric Bootstrap for Calibration Metrics
Because Brier scores and ECE do not follow standard binomial distributions, we construct empirical 95% confidence intervals via non-parametric bootstrap resampling ($B = 2{,}000$ iterations). For a sample of action-level predictions $\mathcal{S} = \lbrace (c_j, y_j) \rbrace_{j=1}^M$, we sample with replacement $\mathcal{S}^{*(b)}$ and compute metric $\hat{\theta}^{*(b)}$. The 95% confidence bounds are determined by the 2.5th and 97.5th percentiles:

$$\text{CI}_{0.95}(\theta) = \left[ q_{0.025}\big(\hat{\theta}^*\big), \; q_{0.975}\big(\hat{\theta}^*\big) \right]$$

To evaluate the statistical significance of calibration distortion, we compute the bootstrap distribution of the difference between poisoned and clean conditions:

$$\Delta_{\text{BS}}^{*(b)} = \text{BS}^{*(b)}_{\text{poisoned}} - \text{BS}^{*(b)}_{\text{clean}}$$

If zero falls outside the 95% bootstrap difference interval, the calibration degradation is statistically significant at $\alpha = 0.05$.

### Two-Proportion Hypothesis Test
For comparing model vulnerability across scale classes ($N_1 = 78$ runs on 8B vs. $N_2 = 78$ runs on 70B under explicit poisoning), we perform a pooled two-proportion $z$-test:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}^*(1 - \hat{p}^*)\left(\frac{1}{N_1} + \frac{1}{N_2}\right)}}, \quad \hat{p}^* = \frac{X_1 + X_2}{N_1 + N_2}$$

With $\hat{p}_{\text{8B}} = 0.1154$ (9/78) and $\hat{p}_{\text{70B}} = 0.1667$ (13/78), the pooled estimate is $\hat{p}^* = \frac{22}{156} \approx 0.1410$, yielding $z = 0.920$ and $p = 0.357$. Because $p > 0.05$, we fail to reject $H_0: p_{\text{8B}} = p_{\text{70B}}$, mathematically confirming that neither inverse scaling nor standard scaling robustness can be claimed at this sample size.

