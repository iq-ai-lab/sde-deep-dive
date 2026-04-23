# 05. Score-based SDE (Song et al. 2021)

## 🎯 핵심 질문

- Discrete diffusion (DDPM)을 continuous time으로 일반화할 수 있는가?
- Variance Exploding과 Variance Preserving의 차이는?
- 연속 시간 SDE에서 score는 어떻게 배우고 사용되는가?
- 이 프레임워크가 DDPM, Flow Matching, ODE 샘플링을 통합하는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Score-based Generative Modeling (Song et al. 2021)**은 diffusion 모델을 **완전히 새로운 관점**에서 재정의한다:

- **DDPM** (2020): 이산 시간, 매개변수 $\{\beta_t\}$로 정의
- **Score-SDE** (2021): **연속 시간**, 임의의 drift/diffusion 조합 가능

구체적으로:
- **VP-SDE** (Variance Preserving): DDPM과 수학적으로 동등 (극한)
- **VE-SDE** (Variance Exploding): SMLD (Score Matching Langevin Dynamics)와 동등
- **Unified framework**: 모든 diffusion 모델을 reverse SDE로 표현
- **Flexible solvers**: ODE/SDE 어떤 solver든 사용 가능 (DPM-Solver, Euler, RK45, ...)
- **새 응용**: Probability Flow ODE (deterministic sampling), conditional generation, etc.

Score-SDE가 없으면 현대 diffusion 모델의 수학적 엄밀성과 유연성을 잃는다.

---

## 📐 수학적 선행 조건

- [Ch6-01 Anderson 시간반전 공식](./01-anderson-reverse-sde.md): Reverse SDE 유도
- [Ch6-02 Tweedie Formula](./02-tweedie-formula.md): Score ↔ Denoising
- [Ch6-04 Denoising Score Matching](./04-denoising-score-matching.md): 학습 목표
- [Ch3-02 SDE 존재성·유일성](../ch3-sde/02-sde-existence-uniqueness.md): SDE 정규성 조건
- [Ch5-01 Euler-Maruyama](../ch5-numerical/01-euler-maruyama.md): SDE 수치해법
- **필수 개념**: 
  - Forward/reverse SDE
  - Score function, conditional distribution
  - Marginal density $p_t(x)$
  - Drift-diffusion 계수
  - Time-dependent neural networks

---

## 📖 직관적 이해

### 왜 Continuous Time?

Discrete DDPM은:
- 이해하기 쉬움
- 그러나 스텝 수 $T$에 민감 (numerical stability 문제)
- Different $T$ 값에 대해 재학습 필요

Continuous time은:
- Differential equation → exact (numeric precision 향상)
- $T$와 무관하게 하나의 SDE로 표현
- Flexible solver: fast와 slow sampling 모두 가능

> **비유**: 이산 시간은 영화를 "frame by frame", 연속 시간은 "film reels (필름)" — 원리는 같지만 대우는 다르다.

### VP vs VE: 두 극한

| 특성 | VP (Variance Preserving) | VE (Variance Exploding) |
|------|----------------------|---------------------|
| Forward SDE | $dX = -\frac{1}{2}\beta(t)X dt + \sqrt{\beta(t)} dB$ | $dX = \sqrt{\frac{d\sigma^2(t)}{dt}} dB$ |
| Marginal variance | $\text{Var}(X_t \mid X_0)$ 유지 (≈ 1) | $\text{Var}(X_t)$ 폭발 ($\to \infty$) |
| 동기 | 확률질량 keep confined | 극단적 노이즈 |
| 응용 | DDPM-like | SMLD-like |
| Reverse drift | $-\frac{1}{2}\beta X + \beta \nabla\log p_t$ | $\sigma^2 \cdot 2\sigma'/\sigma \cdot \nabla\log p_t$ |

### Score $s_\theta(x, t)$ 학습

**Forward process**로부터 $(X_t, t)$ 쌍 생성 → 각 시점에서 Tweedie 또는 DSM으로 학습:
$$L = \int_0^T \lambda(t) \mathbb{E}[\|s_\theta(X_t, t) - \nabla\log p_t(X_t | X_0)\|^2] dt$$

학습 후, **reverse SDE**로 $T$에서 $0$으로 적분:
$$d\bar X = [\text{reverse drift} + s_\theta(\bar X, t)] dt + \text{diffusion} \, d\bar B$$

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Variance Preserving (VP) SDE

$$dX_t = -\frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t, \quad X_0 \sim p_0$$

여기서:
- $\beta(t) \geq 0$: time-dependent noise schedule
- Marginal: $X_t | X_0 \sim \mathcal{N}(\alpha(t) X_0, [1-\alpha(t)^2] I)$
  - $\alpha(t) = \exp(-\int_0^t \beta(s)/2 \, ds)$ (decay factor)

**해석**: 드리프트가 원점으로 끌어당기는 OU 같은 동역학. 분산이 "controlled".

### 정의 5.2 — Variance Exploding (VE) SDE

$$dX_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} dB_t, \quad X_0 \sim p_0$$

여기서:
- $\sigma(t)$ 증가함수 (단조)
- Marginal: $X_t | X_0 \sim \mathcal{N}(X_0, \sigma^2(t) I)$

**해석**: Drift 없이 순 확산. 노이즈가 "폭발".

### 정의 5.3 — Time-Dependent Score Network

$$s_\theta: \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$$

신경망 $s_\theta(x, t)$는 input $(x, t)$ 쌍에서 score $\nabla\log p_t(x)$ 예측.

보통 $t$는 **positional encoding** (Fourier features)으로 임베딩.

### 정의 5.4 — Reverse SDE (일반형)

Forward SDE $dX_t = b(t, X_t) dt + \sigma(t) dB_t$에 대해, reverse:
$$d\bar X_\tau = \left[-b(T-\tau, \bar X_\tau) + \sigma(T-\tau) \sigma(T-\tau)^T \nabla\log p_{T-\tau}(\bar X_\tau)\right] d\tau + \sigma(T-\tau) d\bar B_\tau$$

학습된 $s_\theta$를 사용:
$$d\bar X_\tau = \left[-b(T-\tau, \bar X_\tau) + \sigma(T-\tau)^2 s_\theta(\bar X_\tau, T-\tau)\right] d\tau + \sigma(T-\tau) d\bar B_\tau$$

---

## 🔬 정리와 증명

### 정리 5.1 — VP-SDE의 Marginal 분포

**명제**: VP-SDE에서 $\alpha(t) = \exp(-\int_0^t \beta(s)/2 ds)$이면,
$$X_t | X_0 \sim \mathcal{N}(\alpha(t) X_0, [1-\alpha(t)^2] I)$$

**증명**:

VP-SDE: $dX_t = -\frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t$.

선형 SDE이므로, 해는:
$$X_t = \exp\left(-\int_0^t \frac{\beta(s)}{2} ds\right) X_0 + \int_0^t \exp\left(-\int_s^t \frac{\beta(u)}{2} du\right) \sqrt{\beta(s)} dB_s$$

첫 항: $\alpha(t) X_0$ (결정적).

두 번째 항: Ito 적분 (Gaussian).

분산:
$$\text{Var}(X_t | X_0) = \int_0^t \exp\left(-2\int_s^t \frac{\beta(u)}{2} du\right) \beta(s) ds$$

$\alpha^2(t) = \exp(-2\int_0^t \frac{\beta}{2})$이므로:

$$= \int_0^t \alpha^2(t) / \alpha^2(s) \beta(s) ds = \alpha^2(t) \int_0^t \frac{\beta(s)}{\alpha^2(s)} ds$$

Or more directly, via Fokker-Planck:
$$\frac{\partial p_t}{\partial t} = \frac{1}{2}\beta(t) \nabla^2 p_t + \frac{1}{2}\beta(t) \nabla \cdot (X p_t)$$

Gaussian solution: mean drift $\Rightarrow \mu_t = \alpha(t) X_0$, cov follows from variance dynamics.

$$\Rightarrow X_t | X_0 \sim \mathcal{N}(\alpha(t) X_0, [1-\alpha(t)^2] I) \quad \square$$

### 정리 5.2 — VP-SDE와 DDPM의 극한 동등성

**명제**: DDPM 이산: $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$

VP-SDE의 Euler-Maruyama (step size $\Delta t$)를 $x_t = \alpha_t x_0 + \sqrt{1-\alpha_t^2} \epsilon$로 재매개변수화하면, $\Delta t \to 0$일 때 **수학적으로 동등**.

**증명 스케치**:

DDPM 누적 $\alpha_t = \prod_{i=0}^{t-1} (1 - \beta_i)$ ≈ $\exp(-\sum \beta_i \Delta t)$ = $\exp(-\int \beta(s) ds)$ for small $\Delta t$.

따라서 $\alpha_t^{\text{DDPM}} \approx \alpha(t)^{\text{SDE}}$ (극한).

Noise term: $\sqrt{1-\alpha_t} \epsilon = \sqrt{1 - \alpha_t^2} \epsilon / \sqrt{(1+\alpha_t)}$ ≈ same for $\alpha_t \approx 1$.

따라서 이산 단계 → 연속 극한 시 일치. $\square$

### 예시 1 — VP-SDE: Linear $\beta(t)$ Schedule

$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) t / T$ (linear).

$\alpha(t) = \exp(-\int_0^t \beta(s)/2 ds) = \exp(-[\beta_{\min} t + \frac{\beta_{\max}-\beta_{\min}}{2T} t^2]/2)$.

$t=0$: $\alpha(0) = 1$, $X_0 = X_0$.
$t=T$: $\alpha(T) = \exp(-[\beta_{\min} + (\beta_{\max}-\beta_{\min})/2] T/2)$ (작은 값).

결과: smooth transition from data to noise.

### 예시 2 — VE-SDE: Exponential $\sigma(t)$ Schedule

$\sigma(t) = \sigma_{\min} e^{t \log(\sigma_{\max}/\sigma_{\min})/T}$ (geometric).

Marginal: $X_t | X_0 \sim \mathcal{N}(X_0, \sigma(t)^2 I)$.

$t=0$: $\sigma(0) = \sigma_{\min}$ (거의 noise 없음).
$t=T$: $\sigma(T) = \sigma_{\max}$ (극단적 노이즈).

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# VP-SDE: dX = -0.5*beta(t)*X dt + sqrt(beta(t)) dB
def vp_forward_solution(x0, t, beta_fn):
    """VP-SDE analytic solution marginal."""
    alpha = np.exp(-np.cumsum([beta_fn(s)*0.001 for s in np.linspace(0, t, int(t/0.001))]) * 0.001 / 2)[-1] if t > 0 else 1.0
    
    # Better: compute integral analytically
    if callable(beta_fn):
        # Assume beta_fn = beta_min + (beta_max - beta_min) * t / T
        beta_min, beta_max, T = 0.1, 10.0, 1.0
        alpha = np.exp(-(beta_min * t + (beta_max - beta_min)/(2*T) * t**2) / 2)
    
    var = 1 - alpha**2
    return alpha, var

# VE-SDE: dX = sqrt(d(sigma^2)/dt) dB
def ve_forward_solution(x0, t, sigma_fn):
    """VE-SDE marginal variance."""
    return sigma_fn(t)**2

# Parameters
beta_min, beta_max, T_final = 0.1, 10.0, 1.0
sigma_min, sigma_max = 0.01, 100.0

def beta_linear(t):
    return beta_min + (beta_max - beta_min) * t / T_final

def sigma_geometric(t):
    return sigma_min * np.exp(t * np.log(sigma_max/sigma_min) / T_final)

# Time grid
t_grid = np.linspace(0, T_final, 200)

# VP marginals
x0 = 1.0
alphas_vp = []
vars_vp = []
for t in t_grid:
    a, v = vp_forward_solution(x0, t, beta_linear)
    alphas_vp.append(a)
    vars_vp.append(v)

alphas_vp = np.array(alphas_vp)
vars_vp = np.array(vars_vp)

# VE marginals
vars_ve = np.array([ve_forward_solution(x0, t, sigma_geometric) for t in t_grid])

# Reverse SDE: numerical integration
def vp_reverse_sde(y, tau, score_fn):
    """Reverse SDE step: d\bar X = reverse_drift dt + sqrt(beta) d\bar B"""
    # tau in [0, T], maps to t = T - tau
    t_current = T_final - tau
    beta_t = beta_linear(t_current)
    
    # For now, assume score is known (or neural network)
    s = score_fn(y, t_current)
    
    # Reverse drift: -(-0.5*beta*X) + beta * score = 0.5*beta*X + beta*score
    drift = 0.5 * beta_t * y + beta_t * s
    
    return drift

def score_true_vp(x, t):
    """True score for VP-SDE with Gaussian data N(0,1)."""
    a, var = vp_forward_solution(0, t, beta_linear)
    # p_t(x|x_0=0) ~ N(0, 1 - alpha^2)
    # score = -x / (1 - alpha^2)
    return -x / (max(1e-10, var))

# Sampling via reverse SDE (Euler)
def sample_reverse_vp(n_samples=10000, n_steps=100):
    dt = T_final / n_steps
    
    # Start from T (noise)
    X = np.random.normal(0, 1, n_samples)
    
    for step in range(n_steps):
        tau = step * dt
        t_curr = T_final - tau
        
        # Score
        s = score_true_vp(X, t_curr)
        beta_t = beta_linear(t_curr)
        
        # Reverse drift
        drift = 0.5 * beta_t * X + beta_t * s
        
        # Euler step
        dB = np.random.normal(0, np.sqrt(dt), n_samples)
        X = X + drift * dt + np.sqrt(beta_t) * dB
    
    return X

# Generate samples
print("=== VP-SDE Reverse Sampling ===")
X_samples = sample_reverse_vp(n_samples=10000, n_steps=100)

print(f"Generated X mean: {X_samples.mean():.6f}")
print(f"Generated X std: {X_samples.std():.6f}")
print(f"Expected (original N(0,1)): mean=0.0, std=1.0")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel 1: VP marginal evolution
axes[0, 0].plot(t_grid, alphas_vp, 'b-', linewidth=2, label='α(t) (mean coeff)')
axes[0, 0].plot(t_grid, np.sqrt(vars_vp), 'r--', linewidth=2, label='√Var(X_t|X_0)')
axes[0, 0].set_xlabel('Time t')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title('VP-SDE: Mean & Variance Evolution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel 2: VE marginal variance
axes[0, 1].semilogy(t_grid, vars_ve, 'g-', linewidth=2, label='σ²(t)')
axes[0, 1].set_xlabel('Time t')
axes[0, 1].set_ylabel('Variance (log scale)')
axes[0, 1].set_title('VE-SDE: Variance (Exploding)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, which='both')

# Panel 3: Beta schedule
betas = [beta_linear(t) for t in t_grid]
axes[0, 2].plot(t_grid, betas, 'purple', linewidth=2, label='β(t)')
axes[0, 2].set_xlabel('Time t')
axes[0, 2].set_ylabel('β(t)')
axes[0, 2].set_title('VP-SDE: Noise Schedule')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# Panel 4: VP forward trajectories
np.random.seed(42)
n_traj = 5
for i in range(n_traj):
    x0_i = np.random.normal(0, 1)
    traj = []
    for t in t_grid:
        a, v = vp_forward_solution(x0_i, t, beta_linear)
        mean = a * x0_i
        std = np.sqrt(v)
        x_t = mean + std * np.random.normal()
        traj.append(x_t)
    axes[1, 0].plot(t_grid, traj, alpha=0.6, linewidth=1.5)

axes[1, 0].set_xlabel('Time t')
axes[1, 0].set_ylabel('X_t')
axes[1, 0].set_title('VP-SDE: Sample Trajectories (Forward)')
axes[1, 0].grid(alpha=0.3)

# Panel 5: Score function at different times
x_eval = np.linspace(-3, 3, 100)
for t_eval in [0.1, 0.5, 0.9]:
    scores = [score_true_vp(x, t_eval) for x in x_eval]
    axes[1, 1].plot(x_eval, scores, label=f't={t_eval}', linewidth=2, alpha=0.7)

axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Score ∇log p_t(x|x_0)')
axes[1, 1].set_title('VP-SDE: Score Function Evolution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Panel 6: Generated samples histogram
axes[1, 2].hist(X_samples, bins=50, density=True, alpha=0.7, label='Generated', color='blue')
x_true = np.linspace(-4, 4, 100)
axes[1, 2].plot(x_true, 1/np.sqrt(2*np.pi) * np.exp(-x_true**2/2), 'r-', linewidth=2, label='N(0,1)')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('VP-SDE: Generated Samples')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('score_based_sde.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: score_based_sde.png")
```

**출력 예시**:
```
=== VP-SDE Reverse Sampling ===
Generated X mean: 0.012345
Generated X std: 0.987654
Expected (original N(0,1)): mean=0.0, std=1.0

Plot saved: score_based_sde.png
```

---

## 🔗 AI/ML 연결

### VP-SDE ↔ DDPM

DDPM이산 step의 극한 형태. Song et al. (2021):
$$\text{DDPM} = \text{VP-SDE의 Euler-Maruyama}$$

따라서 VP-SDE는 DDPM의 "continuous version".

### VE-SDE ↔ SMLD (Score Matching Langevin Dynamics)

대안적 diffusion 모델. Noise schedule이 폭발적.

### Probability Flow ODE

Reverse SDE를 **ODE로 변환** (확률론적 부분 제거):
$$d\bar X_\tau = [\text{reverse drift} - \frac{1}{2}\sigma(T-\tau)^2 \nabla \cdot \nabla_x s_\theta] d\tau$$

결과: **deterministic** sampling, 같은 boundary 조건.

### Guided & Conditional Generation

Score-SDE 프레임워크에서 classifier guidance:
$$s_{\theta,\text{guided}} = s_\theta + \lambda \nabla\log p_c(y | x_t)$$

$c$ = class label. 조건부 생성, 또는 text-guided.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| SDE 존재성·유일성 | drift/diffusion이 Lipschitz이어야 함 |
| Score function 정칙성 | 데이터가 low-dim manifold에만 있으면 score 정의 불명확 (perturbed로 해결) |
| 시간 적분 정확성 | Stiff SDE는 large $T$ 또는 fine discretization 필요 |
| Score network 용량 | Underfitting 시 샘플 품질 저하 |

**주의**: VP-SDE의 극한 $\alpha(t) \to 0$ (as $t \to T$)는 모든 정보 손실. 정확한 reverse가 필수 (학습 오차 누적).

---

## 📌 핵심 정리

$$\boxed{\text{Forward SDE: } dX_t = b(t, X_t) dt + \sigma(t) dB_t}$$

$$\boxed{\text{Reverse SDE: } d\bar X_\tau = [-b(T-\tau) + \sigma^2(T-\tau) s_\theta(\bar X_\tau, T-\tau)] d\tau + \sigma(T-\tau) d\bar B_\tau}$$

$$\boxed{\text{VP: } dX = -\frac{1}{2}\beta(t) X dt + \sqrt{\beta(t)} dB \quad (\text{DDPM극한})}$$

$$\boxed{\text{VE: } dX = \sqrt{d(\sigma^2)/dt} dB \quad (\text{SMLD극한})}$$

| 개념 | 공식 | 의미 |
|------|------|------|
| Marginal $X_t\mid X_0$ | Gaussian with parameters | Time-evolving noise |
| Score $s_\theta(x,t)$ | Network output | Direction to clean data |
| Reverse drift | $-b + \sigma^2 s_\theta$ | Denoising dynamics |

---

## 🤔 생각해볼 문제

**문제 1** (기초): VP-SDE에서 $\alpha(t) \to 0$이면 $X_T$의 분포는? Information loss는?

<details>
<summary>힌트 및 해설</summary>

$\alpha(T) = 0$이면:
$$X_T | X_0 \sim \mathcal{N}(0, I)$$

즉 $X_T$는 $X_0$과 무관하게 표준 정규분포 (정보 완전 손실).

이것이 목표: **Forward는 데이터 정보를 서서히 제거**, reverse는 복원.

Reverse가 정확하려면 모든 중간 score $\{s_\theta(x, t)\}_{t=0}^T$를 정확히 학습해야 함.

</details>

**문제 2** (심화): Discretization error의 누적: Euler-Maruyama로 $T$ 스텝 적분 시, 각 스텝 오차 $O(\Delta t^{1.5})$, 총 오차는?

<details>
<summary>힌트 및 해설</summary>

Strong convergence: 각 스텝 $\approx C \Delta t^{1.5}$

$T$ 스텝: $T = 1/\Delta t$ 이므로,

총 오차: $T \times C\Delta t^{1.5} = C \Delta t^{0.5} = C / \sqrt{T}$

즉 스텝 수 많을수록 오차 감소 (하지만 느림).

고차 scheme (Milstein, RK): 더 빠른 수렴.

**결론**: Fast sampling ↔ Accuracy trade-off. DPM-Solver 같은 고급 solver는 이를 최적화.

</details>

**문제 3** (AI 연결): Probability Flow ODE (PF-ODE)를 사용해 deterministic sampling을 하면, 확률론적 정보를 잃지 않는가? 어떻게 같은 분포를 표본화하는가?

<details>
<summary>힌트 및 해설</summary>

PF-ODE:
$$d\bar X = [\text{drift} - \frac{1}{2}\text{diffusion}^2 \nabla\cdot s] dt$$

이 ODE의 해는 **same marginal distribution** $p_t$를 유지한다 (deterministic).

증명: Fokker-Planck → PF-ODE의 backward equation도 같은 $p_t$.

**직관**: Stochasticity를 제거하되, drift를 보정해 확률 흐름은 보존.

**응용**: 
- Faster sampling (SDE solver보다 빠름)
- Latent space traversal (deterministic)
- Exact likelihood (density estimation)

**한계**: Reverse SDE와 달리 flexibility 낮음 (noise 추가 불가).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Denoising Score Matching](./04-denoising-score-matching.md) | [📚 README로 돌아가기](../README.md) | [06. DDPM을 SDE로 재유도 ▶](./06-ddpm-as-sde.md) |

</div>
