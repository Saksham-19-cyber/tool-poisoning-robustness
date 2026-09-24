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
Grouping predictions into $B=10$ equi-spaced confidence bins:

$$\text{ECE} = \sum_{m=1}^B \frac{|S_m|}{M} \Big| \bar{y}_m - \bar{c}_m \Big|$$

---

## 4. Multi-Turn Calibration Drift Dynamics

In multi-turn dialogues spanning turns $t \in \lbrace 1, \dots, T \rbrace$, poisoned tool schemas remain in the system prompt across subsequent invocations. Let $e_t$ denote the calibration error at turn $t$.

### Linear Drift Rate ($\beta$)
The linear drift rate is estimated via Ordinary Least Squares (OLS) regression over conversational turn index:

$$\beta = \frac{\sum_{t=1}^T (t - \bar{t})(e_t - \bar{e})}{\sum_{t=1}^T (t - \bar{t})^2}$$

### Saturating Context Horizon
Empirical trajectory analysis demonstrates that calibration degradation does not grow unbounded linearly. Rather, it follows an asymptotic saturation model:

$$e(t) = e_\infty - \big(e_\infty - e_0\big) e^{-\lambda t}$$

where $e_0$ is the initial baseline error, $e_\infty$ is the asymptotic steady-state error, and $\lambda$ is the context adaptation decay constant. Initial exposure (Turns 1–2) produces the largest calibration shock ($\Delta e \approx +0.12$), after which the error plateaus.

---

## 5. Statistical Inference & Hypothesis Testing

### Wilson Score Confidence Intervals
To avoid false claims of inverse scaling or robustness, all binary proportions (ASR, Utility) are bounded using Wilson Score Intervals with 95% confidence ($\alpha = 0.05, z = 1.96$):

$$w^{\pm} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

### Two-Proportion Hypothesis Test
For comparing two model scales (e.g., $N_1 = 80$ runs on 8B vs. $N_2 = 80$ runs on 70B), we perform a pooled two-proportion $z$-test:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}^*(1 - \hat{p}^*)\left(\frac{1}{N_1} + \frac{1}{N_2}\right)}}, \quad \hat{p}^* = \frac{X_1 + X_2}{N_1 + N_2}$$

With $\hat{p}_{\text{8B}} = 0.3125$ and $\hat{p}_{\text{70B}} = 0.2625$, we obtain $z = 0.710$ and $p = 0.478$. Because $p > 0.05$, we formally fail to reject $H_0: p_{\text{8B}} = p_{\text{70B}}$, mathematically refuting any statistically significant inverse-scaling claim at $N=80$.
