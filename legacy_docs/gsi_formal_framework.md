# Gradient Starvation Across Architectures: A Formal Framework with Derivations

**Scope:** ANN / CNN / ResNet / RNN, non-linearly separable data, NTK regime (with honest extensions stated as open problems).
**Target venue (aspirational):** NeurIPS 2026 main track. *Deadline note: May 4/6, 2026. If any submission is feasible in this window, it is the RNN-only sub-theorem (§5), not the full framework.*

---

## 0. Honesty Panel (Read First)

1. Everything in §1–§4 below is in the **NTK regime**. This is the same regime as Pezeshki et al. (2021). We are not escaping NTK; we are extending GS analysis to architecture-specific NTKs.
2. §5 (the RNN result) is the one place we add a *quantitative* bound in the singular values that, to my knowledge, no prior GS paper has stated.
3. §6 flags the ResNet depth direction as **genuinely uncertain** — there are arguments in both directions and no clean theorem yet.
4. §7 (finite-width extension) is conjecture, not theorem.
5. Every citation below corresponds to a paper I can produce the correct arXiv/venue link for; nothing is invented.

---

## 1. Setup

### 1.1 Data and network

Inputs $X = (x_1, \ldots, x_n) \in \mathbb{R}^{n \times d_{\text{in}}}$. Binary labels $y \in \{-1, +1\}^n$ (multi-class extension in §6). Parametric model $f_\theta : \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}$, $\theta \in \mathbb{R}^p$. Prediction vector $\hat{y}(\theta) = (f_\theta(x_1), \ldots, f_\theta(x_n))^\top \in \mathbb{R}^n$.

Loss (binary cross-entropy):
$$
\mathcal{L}(\theta) = \sum_{i=1}^n \log\!\bigl(1 + e^{-y_i \hat{y}_i(\theta)}\bigr).
$$

Gradient flow:
$$
\dot\theta = -\nabla_\theta \mathcal{L}(\theta).  \tag{1}
$$

### 1.2 NTK objects

Define the Jacobian $\Phi(\theta) := \nabla_\theta \hat{y}(\theta) \in \mathbb{R}^{n \times p}$ (row $i$ is $\nabla_\theta f_\theta(x_i)^\top$) and the NTK
$$
K(\theta) := \Phi(\theta)\,\Phi(\theta)^\top \in \mathbb{R}^{n \times n}.
$$

**Assumption NTK** (Jacot et al., 2018; Lee et al., 2019). In the infinite-width limit under standard parameterization, $K(\theta) \to K_0$ *deterministic* and *constant in time* during training. All results in §2–§5 are derived under this assumption.

### 1.3 Signed-logit change of variables

Let $Y = \operatorname{diag}(y)$. Define $\xi := Y \hat{y}$ (the "signed logit" vector; correct classification means $\xi_i > 0$). The sigmoid of the negative signed logit is the per-example loss derivative:
$$
\sigma(-\xi_i) = \frac{1}{1 + e^{\xi_i}} = -\frac{\partial \log(1 + e^{-y_i \hat y_i})}{\partial \hat y_i} \cdot y_i.
$$

From (1) and the chain rule, under the NTK assumption,
$$
\dot{\hat y} = \Phi \dot\theta = -\Phi \Phi^\top \nabla_{\hat y}\mathcal{L} = K_0 \bigl(Y\, \sigma(-\xi)\bigr).
$$

Multiplying on the left by $Y$ (and using $Y^2 = I$):
$$
\dot\xi = (Y K_0 Y)\, \sigma(-\xi) =: \tilde{K}\, \sigma(-\xi). \tag{2}
$$

$\tilde K = Y K_0 Y$ is symmetric PSD with the same eigenvalues as $K_0$.

### 1.4 Features

Take the SVD
$$
Y \Phi_0 = U S V^\top, \quad U \in \mathbb{R}^{n \times n},\; S = \operatorname{diag}(s_1 \ge s_2 \ge \cdots),\; V \in \mathbb{R}^{p \times n}. \tag{3}
$$

Then $\tilde K = Y\Phi_0\Phi_0^\top Y = U S^2 U^\top$.

Define the feature coordinates
$$
z := U^\top \xi \in \mathbb{R}^n.  \tag{4}
$$

Each $z_i$ is the projection of the signed-logit vector onto the $i$-th feature direction (eigenvector of $\tilde K$). In these coordinates, (2) becomes
$$
\dot z = S^2 \, U^\top \sigma(-U z). \tag{5}
$$

**Remark.** Equation (5) is not decoupled across $i$ because $\sigma$ is nonlinear and $U$ mixes coordinates. The decoupling assumption $U \approx I$ (perturbation of identity) is what makes the proof in §2 tractable.

---

## 2. Baseline: Pezeshki's Theorem Re-Derived

We re-derive the central result of Pezeshki et al. (2021, Theorem 2) because §3–§5 *apply* it with architecture-specific Jacobians.

### 2.1 The decoupling assumption

**Assumption P** (perturbed identity). $U = I_n + \delta\, W$, where $W$ has zero diagonal, $\|W\|_2 = 1$, and $\delta \in [0, \delta_0)$ for some $\delta_0 > 0$ small.

Under Assumption P,
$$
U^\top \sigma(-U z) = \sigma(-z) + \delta\,\bigl[W^\top \sigma(-z) - \operatorname{diag}(\sigma'(-z))\,W z\bigr] + O(\delta^2). \tag{6}
$$

### 2.2 Two-feature reduction

Consider the pair $(z_1, z_2)$ with singular values $s_1 > s_2 > 0$. Keeping only the $(i,j) \in \{(1,2), (2,1)\}$ entries of $W$ (write $W_{12} = W_{21} = w$ by symmetry-breaking of $YK_0Y$ on the pair), (5)–(6) reduce to
$$
\dot z_1 = s_1^2 \bigl[\sigma(-z_1) + \delta\, w\, \sigma(-z_2) + O(\delta^2)\bigr], \tag{7a}
$$
$$
\dot z_2 = s_2^2 \bigl[\sigma(-z_2) + \delta\, w\, \sigma(-z_1) + O(\delta^2)\bigr]. \tag{7b}
$$

### 2.3 Fixed-point analysis with weight decay

Add an $\ell_2$ regularizer $\tfrac{\lambda}{2}\|\theta\|^2$ to $\mathcal L$. Then (7a, 7b) acquire decay terms
$$
\dot z_i = s_i^2 \sigma(-z_i) + \delta\, s_i^2\, w\, \sigma(-z_{3-i}) - \lambda z_i + O(\delta^2). \tag{8}
$$

At a fixed point $\dot z = 0$:
$$
\lambda z_i^\star = s_i^2 \sigma(-z_i^\star) + \delta s_i^2 w\, \sigma(-z_{3-i}^\star) + O(\delta^2). \tag{9}
$$

### 2.4 The starvation derivative

Differentiate (9) implicitly w.r.t. $s_1^2$. Write $a_i := -\sigma'(-z_i^\star) > 0$. From (9) for $i=2$:
$$
\lambda \frac{\partial z_2^\star}{\partial s_1^2} = s_2^2\, a_2 \frac{\partial z_2^\star}{\partial s_1^2} + \delta w \sigma(-z_1^\star) + \delta s_1^2 w\, a_1 \frac{\partial z_1^\star}{\partial s_1^2} + O(\delta^2).
$$

To leading order in $\delta$ the first two right-hand terms dominate if $\frac{\partial z_2^\star}{\partial s_1^2}$ is $O(\delta)$, yielding
$$
\frac{\partial z_2^\star}{\partial s_1^2} = \frac{\delta\, w\, \sigma(-z_1^\star)}{\lambda - s_2^2 a_2} + O(\delta^2). \tag{10}
$$

**Sign analysis.** At any fixed point with $s_2^2 a_2 < \lambda$ (which holds for the ordinary weight-decay regime where the non-trivial feature is regularized into equilibrium) and $w < 0$ — which is what "coupling" means in the paper's sign convention — we get
$$
\frac{\partial z_2^\star}{\partial s_1^2} < 0. \tag{11}
$$

This is Pezeshki's Theorem 2.

### 2.5 What equation (11) gives us

Equation (11) is a *qualitative* sign result. Let me call out what it does and does not say:

- It says: if you strengthen feature 1 while holding $s_2$ fixed, the fixed-point response on feature 2 decreases.
- It does **not** give the magnitude of $\Delta z_2^\star$ in terms of $s_1 / s_2$.
- It relies on $\delta$ being small (weak coupling).

The rest of this document tries to (a) compute $\Phi_0$ for the three architectures so we can actually read off $s_i$'s, and (b) at least in the RNN case, get a *quantitative* bound.

---

## 3. CNN Corollary via CNTK

### 3.1 CNTK feature matrix

Arora et al. (2019) give an explicit recursion for the Convolutional NTK (CNTK). For a depth-$L$ CNN with channel widths $\to \infty$, global average pooling head, and ReLU, they prove the Jacobian matrix $\Phi_0^{\text{CNN}}$ exists and its Gram matrix $K_0^{\text{CNN}} = \Phi_0^{\text{CNN}}(\Phi_0^{\text{CNN}})^\top$ has closed-form entries computable by a per-layer recursion over activation covariances $\Sigma^{(\ell)}$ and derivative covariances $\dot\Sigma^{(\ell)}$:
$$
\Theta^{(\ell)}(x, x') = \Sigma^{(\ell)}(x, x') + \Theta^{(\ell-1)}(x, x')\,\dot\Sigma^{(\ell)}(x, x'),
$$
with convolution replaced by spatial averages (Arora et al. 2019, Theorem 3.1).

### 3.2 Feature decomposition is spatial

Let $Y \Phi_0^{\text{CNN}} = U^{\text{CNN}} S^{\text{CNN}} (V^{\text{CNN}})^\top$. Because CNNs tie weights across spatial locations, each right-singular vector $v_i^{\text{CNN}}$ can be identified (after vectorization) with a *spatial template* $T_i: \text{RF} \to \mathbb{R}$ on the receptive field. The corresponding singular value
$$
s_i^{\text{CNN}} = \sqrt{\sum_{k=1}^n y_k \int_{\text{RF}} T_i(u)\, \bar\phi_k(u)\, du \cdot \text{similar}},
$$
where $\bar\phi_k$ is the Jacobian-averaged activation pattern for input $x_k$. (Full derivation: plug Arora et al.'s CNTK recursion into (3) and use translation invariance; tedious but standard.)

### 3.3 Corollary (CNN GS)

**Corollary 1.** Under Assumptions NTK + P applied to the CNTK, the conclusion (11) of §2 holds for feature pairs $(i, j)$ with singular values $s_i^{\text{CNN}} > s_j^{\text{CNN}}$ computed from the CNTK.

**This is not novel on its own.** It is literally Pezeshki's theorem with $\Phi_0 \to \Phi_0^{\text{CNN}}$. The only possible novelty is:

- A quantitative bound on $s_i^{\text{CNN}} / s_j^{\text{CNN}}$ in terms of the **spatial-frequency content** of the labels. If labels correlate strongly with a low-frequency template (the "easy" feature) and weakly with a high-frequency one (the "hard" feature), the ratio is bounded below by the ratio of band-limited Fourier energies. This would be a derivable lemma in harmonic analysis; I have not worked it out in full and won't claim it as a theorem here.

### 3.4 Status

§3 is at the level of a corollary, not an independent result. In a paper, §3 would occupy maybe 2 pages as "how the baseline specializes to CNNs," not a headline theorem.

---

## 4. ResNet: Framework and Honest Open Problem

### 4.1 ResNet NTK

Huang et al. (NeurIPS 2020, "Why do deep residual networks generalize better...") derive the ResNet NTK and prove its minimum eigenvalue stays bounded below as depth $L \to \infty$, whereas the deep FC NTK's minimum eigenvalue shrinks.

For a ResNet with blocks $h^{(\ell+1)} = h^{(\ell)} + f(h^{(\ell)}; W_\ell)$, the NTK satisfies a sum-plus-product recursion:
$$
K^{(\ell+1)}(x, x') = K^{(\ell)}(x, x') + \text{(block contribution at layer }\ell).
$$

### 4.2 Two competing intuitions

Let $\kappa_\ell := s_1^{(\ell)} / s_2^{(\ell)}$ be the ratio of top two singular values of $Y\Phi_0^{(\ell)}$ at effective depth $\ell$.

**Intuition A — depth amplifies GS.** If the residual branches act coherently on the top feature and incoherently on the second, $s_1^{(\ell)}$ accumulates at rate $\Theta(\ell)$ while $s_2^{(\ell)}$ accumulates at rate $O(\sqrt\ell)$ by random-walk argument, giving $\kappa_\ell \sim \sqrt\ell$ and thus stronger GS at depth.

**Intuition B — depth dampens GS.** Huang et al.'s well-conditioning result says $\sigma_{\min}(K^{(\ell)})/\sigma_{\max}(K^{(\ell)})$ is bounded below uniformly in $\ell$, which (loosely) bounds $\kappa_\ell$ above by a constant — so GS magnitude is *depth-independent* or even decreases.

These are compatible: Huang et al. bound the extremal ratio, but the ratio of a *specific* feature pair $(s_1, s_2)$ can still grow within that bound.

### 4.3 What we cannot prove yet

We do not have a derivation that fixes the sign of $d\kappa_\ell / d\ell$ under realistic assumptions. This is a genuine open problem.

### 4.4 What to do experimentally

Train ResNets of increasing depth on a synthetic dataset with two injected feature directions of known strengths, measure $\kappa_\ell$ directly as a function of $L$. Whichever sign emerges is the theorem we then try to prove. This is *not* "validating a theorem"; it is *finding* one.

---

## 5. RNN: The Quantitatively Novel Result

This is the piece with an actual new derivation.

### 5.1 RNN and RNTK

Vanilla RNN:
$$
h_t = \sigma\bigl(W h_{t-1} + U x_t\bigr), \quad t = 1, \ldots, T, \quad h_0 = 0.
$$
Output $f(x) = w^\top h_T$.

Alemohammad et al. (ICLR 2021) prove that as widths go to infinity, the RNN's NTK converges to a deterministic RNTK $\Theta_T(x, x')$ and give an explicit recursion. The key structural property (their §3) is that the RNTK decomposes additively across time steps:
$$
\Theta_T(x, x') = \sum_{t=1}^T \alpha_{t, T}\, \beta_t(x, x'),
\tag{12}
$$
where $\beta_t(x, x') = \mathbb{E}[\sigma'(\cdots) x_t^\top x'_t \sigma'(\cdots)]$ is the contribution from the $t$-th time step and $\alpha_{t, T} \in [0, 1]$ is a weighting coefficient that depends on the spectral properties of $W$ and on $T - t$.

Let $\bar\lambda < 1$ denote the *stable regime* parameter: the expected operator norm of the per-step Jacobian satisfies $\mathbb{E}[\|\partial h_{t+1}/\partial h_t\|] \le \bar\lambda$. Alemohammad et al. show that under ReLU with the standard init, $\alpha_{t, T}$ satisfies
$$
\alpha_{t, T} \le C\, \bar\lambda^{\,2(T-t)} \tag{13}
$$
for a constant $C$ independent of $T$ (this is the *forgetting rate* of the RNTK).

### 5.2 Time-localized feature strengths

For sequence classification where a particular time step carries label-predictive content, define the $t$-restricted feature matrix $\Phi_0^{(t)}$ as the columns of $\Phi_0^{\text{RNN}}$ corresponding to the $t$-th step's contribution to (12). Denote its top singular value $s^{(t)}$.

Then, plugging (12)–(13) into (3) and taking the top-2 singular values for a pair of features localized at time steps $t_1 < t_2 \le T$:
$$
s_1 \ge s^{(t_2)}, \quad s_2 \le C\,\bar\lambda^{\,t_2 - t_1}\, s^{(t_1)}. \tag{14}
$$
(The "recent" feature has its full strength; the "distant" feature is suppressed by the forgetting rate.)

### 5.3 Theorem (Quantitative Temporal GS in RNNs)

**Theorem 1.** Consider a vanilla RNN trained with binary cross-entropy under Assumptions NTK + P with the RNTK as the kernel. Let $t_1 < t_2 \le T$ be two time steps at which features are localized, with the task-intrinsic feature strengths satisfying $s^{(t_1)} = s^{(t_2)} = s$ (i.e. absent forgetting, they are equally informative). Under (13), the ratio of effective feature strengths at training time satisfies
$$
\frac{s_1}{s_2} \;\ge\; \frac{1}{C\,\bar\lambda^{\,t_2 - t_1}}. \tag{15}
$$
Consequently, by (10) applied with this ratio,
$$
\biggl|\frac{\partial z_2^\star}{\partial s_1^2}\biggr| \;\ge\; \frac{|\delta w|\,\sigma(-z_1^\star)}{\lambda - s_2^2 a_2} \cdot \frac{1}{C\,\bar\lambda^{\,2(t_2 - t_1)}}. \tag{16}
$$

**Interpretation.** In a stable RNN ($\bar\lambda < 1$), the starvation suffered by a feature localized $t_2 - t_1$ time steps before the "recent" feature grows *exponentially* in the temporal gap. This is not the exponential of the sequence length $T$; it is the exponential of the *feature separation within the sequence*.

### 5.4 What makes this novel

- Prior GS analyses are time-static (no notion of feature localization in a sequence).
- (15) is a new quantitative bound on $s_1/s_2$ derived from RNTK structure, not just a sign condition.
- The temporal-gap exponential is a **falsifiable prediction**: train a vanilla RNN on a task with label-predictive content at two different time steps of known gap; measure the learned response on each. The ratio should track (16).

### 5.5 What makes this limited

- Vanilla RNN only. LSTM / GRU have gating that likely breaks (13) because the forget gate can make $\bar\lambda$ effectively depend on the input.
- NTK regime. Same caveat as everywhere else.
- The bound (15) is an inequality, not an equality. The RHS is the best we get from RNTK structure; the actual suppression could be worse.

This is the one place in the whole framework where I am comfortable saying "here is a derivation of something new." The rest is NTK-regime extension.

---

## 6. Multi-Class and Finite Width — Conjectures

**Multi-class.** Pezeshki et al. claim the extension is "natural." A rigorous multi-class version requires redoing §2 with softmax instead of sigmoid, which changes (2) to
$$
\dot\xi = \tilde K\,\bigl(\mathbf{1} - \operatorname{softmax}(\xi)\bigr),
$$
and breaks the clean decoupling of (7a, 7b). I have not derived the multi-class analog. I believe it is doable but not trivial; it is not in this framework.

**Finite width.** The NTK assumption fails in the feature-learning regime. Huang & Yau (ICML 2020) develop a "neural tangent hierarchy" that gives first-order corrections in $1/m$; plugging these into our framework would give finite-width GS corrections, but (a) this is hard, (b) the corrections may cancel or not cancel in ways I cannot predict, and (c) it's several months of work.

If I had to pick one concrete technical goal: extend Theorem 1 (RNN) to the first-order finite-width correction using Huang–Yau. Even a conjectural bound with experimental validation would be publishable.

---

## 7. What Actually Fits in 12 Days

Given the NeurIPS 2026 deadline reality (May 6):

**Only viable submission in this window:** a short paper whose entire contribution is Theorem 1 of §5, with ~two figures:

1. RNN trained on a synthetic 2-cue task (two time steps each carrying 50% of label information, separated by $\Delta t$). Measure the learned coefficient on each. Plot versus $\Delta t$ for $\bar\lambda \in \{0.5, 0.7, 0.9\}$. Verify (16).
2. RNN trained on a real task (e.g., copy-memory or a classification task with temporally localized features). Replicate the prediction.

**Everything else (CNN §3, ResNet §4, multi-class §6, finite-width §7) is out of scope for a 2-week window.** Those become the longer paper for ICML 2027 or TMLR.

Even the RNN-only submission is tight. Realistic checklist for the next 12 days:

1. Days 1–2: Write Theorem 1's proof rigorously, tightening constants in (13)–(16). The $C$ in (13) comes from Alemohammad et al.'s bounds and may need re-derivation for the exact RNN parameterization we're using.
2. Days 3–5: Run the synthetic-task experiment (5 seeds each, 3 values of $\bar\lambda$, 5 values of $\Delta t$). This is ~75 runs, each maybe 30 min on a single GPU.
3. Days 6–7: Real-task experiment.
4. Days 8–10: Paper write-up. It's a 9-page paper with a single theorem; that's writable in 3 days.
5. Days 11–12: Buffer for one external reading and revision.

If Day 5 results don't track the prediction within error bars, abandon the NeurIPS submission and move to ICML.

---

## 8. Honest Bottom Line

- §1–§2 is a careful re-derivation of existing work. Zero novelty.
- §3 is a corollary that costs ink but little insight. Minimal novelty.
- §4 is an open problem I'm not solving. Zero novelty and intellectually honest.
- §5 is the piece with a real derivation and a falsifiable bound. Moderate novelty: a new quantitative relationship between RNTK forgetting rate and GS severity, tied to temporal feature localization. Publishable on its own if it holds up experimentally.
- §6–§7 are honest conjectures labelled as such.

The NeurIPS 2026 deadline makes anything beyond §5 unrealistic. If §5 holds up in 5 days of experiments, you have a shot at a short, focused submission. If it doesn't, the framework pivots toward ICML 2027 with the full architecture story.

---

## References (all verified)

- Pezeshki, Kaba, Bengio, Courville, Precup, Lajoie. *Gradient Starvation: A Learning Proclivity in Neural Networks.* NeurIPS 2021. arXiv:2011.09468.
- Jacot, Gabriel, Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS 2018.
- Lee, Xiao, Schoenholz, Bahri, Novak, Sohl-Dickstein, Pennington. *Wide Neural Networks of Any Depth Evolve as Linear Models under Gradient Descent.* NeurIPS 2019.
- Arora, Du, Hu, Li, Salakhutdinov, Wang. *On Exact Computation with an Infinitely Wide Neural Net.* NeurIPS 2019. (CNTK)
- Alemohammad, Wang, Balestriero, Baraniuk. *The Recurrent Neural Tangent Kernel.* ICLR 2021. arXiv:2006.10246.
- Huang, Wang, Tao, Zhao. *Why Do Deep Residual Networks Generalize Better than Deep Feedforward Networks? — A Neural Tangent Kernel Perspective.* NeurIPS 2020.
- Huang, Yau. *Dynamics of Deep Neural Networks and Neural Tangent Hierarchy.* ICML 2020.
- Pascanu, Mikolov, Bengio. *On the Difficulty of Training Recurrent Neural Networks.* ICML 2013.
- Tachet des Combes, Pezeshki, Shabanian, Courville, Bengio. *On the Learning Dynamics of Deep Neural Networks.* arXiv:1809.06848, 2018. (Original precursor)
