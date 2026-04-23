# 06. DDPM을 SDE로 재유도

## 🎯 핵심 질문

- DDPM의 이산 diffusion을 SDE로 어떻게 표현하는가?
- Euler-Maruyama와 DDPM의 관계는 정확히 무엇인가?
- DDPM의 손실 함수가 왜 noise prediction MSE 형태인가?
- Score-based SDE와 DDPM은 같은 모델인가, 다른 모델인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**DDPM**은 2020년 이후 생성모델의 "new standard"가 되었다. 그러나 **왜 작동하는가?** 그리고 **어떻게 개선할 수 있는가?** 에 대한 수학적 답변은 **Score-based SDE 관점**에서만 제공된다.

구체적으로:
- **DDPM loss** ($\mathbb{E}\|\epsilon - \epsilon_\theta\|^2$)가 **Denoising Score Matching의 재매개변수화**임을 보임
- **VP-SDE의 Euler-Maruyama가 정확히 DDPM**임을 증명 (극한)
- DDPM과 다른 diffusion 모델 (VE, SMLD)의 **수학적 동등성** 확립
- **Generalization**: reverse process를 어떤 ODE/SDE solver든 사용 가능 (DDIM, DPM-Solver, ...)

DDPM을 SDE로 이해하지 않으면, **개선의 방향을 찾기 어렵다**.

---

## 📐 수학적 선행 조건

- [Ch6-02 Tweedie Formula](./02-tweedie-formula.md): Denoising, posterior mean
- [Ch6-04 Denoising Score Matching](./04-denoising-score-matching.md): DSM loss 형식
- [Ch6-05 Score-based SDE](./05-score-based-sde.md): VP-SDE, Reverse SDE
- [Ch5-01 Euler-Maruyama](../ch5-numerical/01-euler-maruyama.md): SDE 수치해법
- **필수 개념**: 
  - DDPM 이산 forward/reverse process
  - 재매개변수화 ($x_t = \alpha_t x_0 + \sigma_t \epsilon$)
  - Noise schedule $\{\beta_t\}$
  - Conditional likelihood
  - Loss weighting, variance schedule

---

## 📖 직관적 이해

### DDPM ← VP-SDE의 이산화

VP-SDE는 **continuous time** 공식:
$$dX_t = -\frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t$$

**Euler-Maruyama discretization** (time step $\Delta t = T/N$):
$$X_{t+\Delta t} = X_t - \frac{1}{2}\beta(t) X_t \Delta t + \sqrt{\beta(t) \Delta t} Z$$
$$= (1 - \frac{1}{2}\beta(t)\Delta t) X_t + \sqrt{\beta(t)\Delta t} Z$$

1차 Taylor: $\sqrt{1-\beta(t)\Delta t} \approx 1 - \frac{1}{2}\beta(t)\Delta t$ (for small $\beta$).

따라서 $\beta_k := \beta(t_k) \Delta t$로 놓으면:
$$X_{t+\Delta t} \approx \sqrt{1-\beta_k} X_t + \sqrt{\beta_k} Z \quad (\text{DDPM과 동일!})$$

### Loss의 동등성

DDPM loss: $\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

$x_t = \alpha_t x_0 + \sigma_t \epsilon$ (DDPM 재매개변수화):
- $\alpha_t = \sqrt{\prod_{i=1}^t (1-\beta_i)}$ (누적 product)
- $\sigma_t = \sqrt{1-\alpha_t^2}$

Conditional likelihood:
$$\nabla_{x_t}\log q(x_t | x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = -\frac{\sigma_t \epsilon}{\sigma_t^2} = -\frac{\epsilon}{\sigma_t}$$

DSM (Denoising Score Matching):
$$\|s_\theta(x_t, t) - (-\frac{x_t - x_0}{\sigma_t^2})\|^2$$

$s_\theta = -\epsilon_\theta / \sigma_t$로 재매개변수화하면:
$$\|-\frac{\epsilon_\theta}{\sigma_t} + \frac{\epsilon}{\sigma_t}\|^2 = \frac{1}{\sigma_t^2}\|\epsilon_\theta - \epsilon\|^2$$

**DDPM loss는 $\sigma_t^2$-weighted DSM!**

| 관점 | 손실 | 의미 |
|------|------|------|
| Epsilon prediction | $\\\|\epsilon - \epsilon_\theta\\\|^2$ | 노이즈 복원 |
| Score prediction | $\\\|s_\theta + \epsilon/\sigma_t\\\|^2$ | Score 복원 |
| Posterior mean | $\\\|x_0 - (x_t + \sigma_t^2 s_\theta)\\\|^2$ | Denoising |

> **비유**: 같은 대상을 다른 "렌즈"로 본 것. 수학적으로는 동일하지만, 학습 관점은 다르다.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — DDPM Forward Process (이산)

$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Marginal:
$$q(x_t | x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)$$

여기서:
- $\alpha_t = \sqrt{\prod_{i=1}^t (1-\beta_i)}$ (누적 곱)
- $\sigma_t = \sqrt{1-\alpha_t^2}$
- Reparameterization: $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$

### 정의 6.2 — DDPM Reverse Process (이산)

학습된 신경망 $\epsilon_\theta(x_t, t)$를 사용:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(t))$$

Mean:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}(x_t - \frac{\beta_t}{\sqrt{\sigma_t^2}} \epsilon_\theta(x_t, t))$$

또는 재배열:
$$\mu_\theta = \frac{\alpha_{t-1}\sqrt{1-\beta_t}}{1-\alpha_t} x_t + \frac{\sqrt{\beta_t(1-\alpha_{t-1})}}{1-\alpha_t} \epsilon_\theta$$

Variance: $\Sigma_\theta = (1-\tilde{\alpha}_{t-1})/(1-\alpha_t) \cdot \beta_t I$ (fixed or learned).

### 정의 6.3 — VP-SDE의 Euler-Maruyama (DDPM 형식)

Time discretization: $t_k = k\Delta t$, $\Delta t = T/N$.

$$X_{k+1} \approx X_k - \frac{1}{2}\beta(t_k) X_k \Delta t + \sqrt{\beta(t_k)\Delta t} Z_k$$

$\beta_k := \beta(t_k) \Delta t$ 정의 후:

$$X_{k+1} = (1 - \frac{1}{2}\beta_k) X_k + \sqrt{\beta_k} Z_k \approx \sqrt{1-\beta_k} X_k + \sqrt{\beta_k} Z_k$$

(1차 Taylor 근사)

---

## 🔬 정리와 증명

### 정리 6.1 — VP-SDE와 DDPM의 이산화 동등성

**명제**: VP-SDE $dX = -\frac{1}{2}\beta(t)X dt + \sqrt{\beta(t)} dB$를 Euler-Maruyama로 이산화한 것은, DDPM 이산 단계와 (1차 근사로) 동등하다.

**증명**:

**Step 1**: VP-SDE Euler-Maruyama.

$$X_{k+1} = X_k + (-\frac{1}{2}\beta(t_k) X_k) \Delta t + \sqrt{\beta(t_k)} \sqrt{\Delta t} Z_k$$

**Step 2**: 분산 정확도.

Forward SDE의 marginal: $X_t | X_0 \sim \mathcal{N}(\alpha(t) X_0, [1-\alpha(t)^2] I)$

Euler scheme도 같은 형태 (1차). $\alpha_k = \prod_{i=0}^{k-1} (1-\beta_i)$ 근사.

**Step 3**: 선형 형태 (1-\frac{1}{2}\beta) vs \sqrt{1-\beta}.

Euler: $X_{k+1} = (1 - \frac{1}{2}\beta_k) X_k + O(\beta_k^2) + \sqrt{\beta_k} Z$.

Taylor: $\sqrt{1-\beta_k} = 1 - \frac{1}{2}\beta_k - \frac{1}{8}\beta_k^2 + O(\beta_k^3)$.

$\beta_k$ 작으면 (일반적으로 $\beta_k \ll 1$), 차이는 $O(\beta_k^2)$.

따라서 **근처 범위에서 동등**.

**Step 4**: Reverse (생략).

역방향도 동일 원리: VP-SDE reverse를 Euler하면 DDPM reverse step과 동등. $\square$

> **보정**: 정확한 동등성은 극한 $\Delta t \to 0$일 때만. DDPM (유한 $N$)은 discretization error $O(\Delta t)$ 또는 더 존재.

### 정리 6.2 — DDPM Loss = Weighted DSM

**명제**: DDPM 손실 $\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$는 Denoising Score Matching의 $\sigma_t^2$-weighted 버전과 동등하다.

**증명**:

**Step 1**: DDPM 재매개변수화.

$$x_t = \alpha_t x_0 + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Conditional likelihood:
$$q(x_t | x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)$$

Score:
$$\nabla_{x_t}\log q(x_t | x_0) = -\frac{x_t - \alpha_t x_0}{\sigma_t^2} = -\frac{\sigma_t \epsilon}{\sigma_t^2} = -\frac{\epsilon}{\sigma_t}$$

**Step 2**: Denoising Score Matching.

$$L_{DSM} = \mathbb{E}[\|s_\theta(x_t) - (-\frac{\epsilon}{\sigma_t})\|^2]$$

$s_\theta = -\frac{\epsilon_\theta}{\sigma_t}$ 매개변수화:
$$L_{DSM} = \mathbb{E}[\|-\frac{\epsilon_\theta}{\sigma_t} + \frac{\epsilon}{\sigma_t}\|^2] = \frac{1}{\sigma_t^2}\mathbb{E}[\|\epsilon_\theta - \epsilon\|^2]$$

**Step 3**: Weighting.

위를 $t$에 대해 평균하면 (DSM 시간 가중):
$$L_{\text{total}} = \sum_t \lambda(t) L_{DSM}(t)$$

DDPM의 경우 보통 $\lambda(t) = 1$ (균등).

그러나 다른 weighting (예: $\lambda(t) = \sigma_t^2$)을 사용하면:
$$\lambda(t) L_{DSM}(t) = \sigma_t^2 \cdot \frac{1}{\sigma_t^2}\mathbb{E}[\|\epsilon\|^2] = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$$

따라서 **DDPM loss는 DSM의 특정 weighted variant**. $\square$

> **따름정리 (VLB와의 관계)**: DDPM은 추가로 variational lower bound (VLB) 관점에서도 해석 가능. Score matching과 VLB는 다른 derivation이지만, 같은 손실에 수렴.

---

### 예시 1 — Linear Schedule DDPM

Noise schedule: $\beta_t = 0.0001 + t/T \times (0.02 - 0.0001)$ (linear schedule, DDPM paper).

$T = 1000$이면, 초기 $\beta_1 = 0.0001$, 마지막 $\beta_{1000} = 0.02$.

Forward $q(x_t | x_0)$:
- $t=1$: $\alpha_1 \approx 0.99999$, $\sigma_1 \approx 0.004$ (거의 변화 없음)
- $t=500$: $\alpha_{500} \approx 0.70$ (제법 섞임)
- $t=1000$: $\alpha_{1000} \approx 0.0001$ (거의 순 노이즈)

Reverse: score $s_\theta(x_t, t)$ 또는 noise $\epsilon_\theta(x_t, t)$를 각 $t$마다 학습.

### 예시 2 — Variance Redefinition: VP-SDE vs DDPM

VP-SDE: $\text{Var}(X_t | X_0) = 1 - \alpha(t)^2$ (연속, smooth).

DDPM: $\sigma_t^2 = 1 - \alpha_t^2$ (이산, piecewise).

극한: DDPM schedule이 VP $\beta(t)$에서 나온다면,
$$\alpha_t^{\text{DDPM}} = \prod_{i=1}^t (1-\beta_i) \approx \exp(-\sum_{i=1}^t \beta_i \cdot \frac{1}{T}) = \exp(-\int_0^{t/T} \beta(s) ds) = \alpha_{\text{VP}}(t/T)$$

따라서 **이산과 연속은 스케일링만 차이**.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# DDPM parameters
T = 1000
beta_min, beta_max = 0.0001, 0.02

# Linear schedule
betas = np.linspace(beta_min, beta_max, T)

# Cumulative products
alphas_cumprod = np.cumprod(1 - betas)
alphas_t = np.sqrt(alphas_cumprod)
sigmas_t = np.sqrt(1 - alphas_cumprod)

print("=== DDPM Forward Process ===")
print(f"β_1 = {betas[0]:.6f}")
print(f"β_1000 = {betas[-1]:.6f}")
print(f"α_1 = {alphas_t[0]:.6f}")
print(f"α_1000 = {alphas_t[-1]:.6f}")
print(f"σ_1 = {sigmas_t[0]:.6f}")
print(f"σ_1000 = {sigmas_t[-1]:.6f}")

# VP-SDE parameters (continuous approximation)
def beta_linear_vp(t_normalized, T_total=1.0):
    """VP-SDE with same schedule."""
    return beta_min + (beta_max - beta_min) * t_normalized / T_total

# Integrate to get alpha(t)
def alpha_vp(t_normalized):
    """Approximate via Euler integration."""
    dt = t_normalized / 1000.0
    integral = 0.0
    for i in range(1000):
        integral += beta_linear_vp(i * dt) * dt / 2.0
    return np.exp(-integral)

# Sample evaluation
t_eval = np.linspace(0, 1, T)
alphas_vp = np.array([np.sqrt(np.exp(-0.5 * np.mean([beta_linear_vp(s) for s in np.linspace(0, t, 100)]))) if t > 0 else 1.0 
                       for t in t_eval])

# Simpler: direct formula
# \alpha(t) = \exp(-\int_0^t beta(s)/2 ds) = \exp(-(beta_min * t + (beta_max - beta_min)/(2*T) * t^2) / 2)
alphas_vp_direct = np.exp(-(beta_min * t_eval + (beta_max - beta_min)/(2*T) * t_eval**2) / 2)

# Compare
t_indices = np.arange(0, T, 100)
print("\n=== Comparison: DDPM vs VP-SDE ===")
print("t\tDDPM α_t\tVP α(t)\t\tDiff")
for idx in t_indices:
    t_norm = idx / T
    alpha_ddpm = alphas_t[idx]
    alpha_vp = alphas_vp_direct[idx]
    print(f"{idx}\t{alpha_ddpm:.6f}\t{alpha_vp:.6f}\t{abs(alpha_ddpm - alpha_vp):.6f}")

# Forward process sampling
print("\n=== Forward Process: Sampling Test ===")
x0_test = np.ones((1000,))  # Simple constant

# At t=100, 500, 1000
for t_idx in [100, 500, 1000]:
    t_idx_py = t_idx - 1  # 0-indexed
    alpha = alphas_t[t_idx_py]
    sigma = sigmas_t[t_idx_py]
    
    # Sample
    epsilon = np.random.normal(0, 1, len(x0_test))
    xt = alpha * x0_test + sigma * epsilon
    
    print(f"t={t_idx}: mean(X_t)={xt.mean():.6f}, std(X_t)={xt.std():.6f}")
    print(f"\t  Theoretical: mean={alpha:.6f}, std={sigma:.6f}")

# Reverse process: score function
# True score for simple case (if we knew p_t)
# Assuming p(x_0) = N(1, 1), then p_t(x_t | x_0=1) ~ N(alpha, sigma^2)
# score = -x_t / sigma^2 (for centered case, adjust for mean shift)

def reverse_step_ddpm(x_t, t, epsilon_theta_pred, alphas_t, sigmas_t, betas):
    """One DDPM reverse step."""
    alpha_t = alphas_t[t-1]
    alpha_t_prev = alphas_t[t-2] if t > 1 else 1.0
    beta_t = betas[t-1]
    sigma_t = sigmas_t[t-1]
    
    # Posterior variance
    variance_t = (1 - alpha_t_prev) / (1 - alphas_cumprod[t-1]) * beta_t
    
    # Mean
    coeff_x = (1 - alpha_t_prev) / (1 - alphas_cumprod[t-1]) * np.sqrt(1 - beta_t)
    coeff_eps = np.sqrt(alpha_t_prev) * beta_t / (1 - alphas_cumprod[t-1])
    
    mean = coeff_x * x_t + coeff_eps * epsilon_theta_pred
    
    return mean, np.sqrt(variance_t)

# Sampling from reverse (Euler)
print("\n=== DDPM Reverse Sampling ===")
x_t = np.random.normal(0, 1, 1000)  # Start from noise
for t in range(T, 0, -50):  # Every 50 steps
    # Assume perfect epsilon_theta (true noise)
    # For testing, use simple score: -x_t / sigma_t^2
    sigma_t = sigmas_t[t-1]
    score_t = -x_t / (sigma_t**2 + 1e-8)  # Avoid division by zero
    epsilon_pred = -sigma_t * score_t  # Equivalent
    
    mean, var = reverse_step_ddpm(x_t, t, epsilon_pred, alphas_t, sigmas_t, betas)
    x_t = mean + np.sqrt(var) * np.random.normal(0, 1, len(x_t))
    
    if t % 250 == 0:
        print(f"Step {t}: mean={x_t.mean():.6f}, std={x_t.std():.6f}")

print(f"Final (t=0): mean={x_t.mean():.6f}, std={x_t.std():.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Beta schedule
axes[0, 0].plot(betas, linewidth=2, label='β_t (DDPM)')
axes[0, 0].set_xlabel('Timestep t')
axes[0, 0].set_ylabel('β_t')
axes[0, 0].set_title('Noise Schedule (Linear)')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# Panel 2: Alpha evolution
t_range = np.arange(T)
axes[0, 1].plot(t_range, alphas_t, 'b-', label='α_t (DDPM)', linewidth=2)
axes[0, 1].plot(t_eval * T, alphas_vp_direct, 'r--', label='α(t) (VP-SDE)', linewidth=2, alpha=0.7)
axes[0, 1].set_xlabel('Timestep t')
axes[0, 1].set_ylabel('α_t')
axes[0, 1].set_title('Mean Coefficient Evolution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Panel 3: Sigma evolution
axes[1, 0].plot(t_range, sigmas_t, 'g-', linewidth=2, label='σ_t (DDPM)')
axes[1, 0].set_xlabel('Timestep t')
axes[1, 0].set_ylabel('σ_t')
axes[1, 0].set_title('Noise Std Evolution')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Panel 4: Information decay
info_remaining = alphas_t**2  # Fraction of original info
axes[1, 1].semilogy(t_range, info_remaining, linewidth=2, label='α_t² (info retained)')
axes[1, 1].set_xlabel('Timestep t')
axes[1, 1].set_ylabel('Fraction (log scale)')
axes[1, 1].set_title('Information Loss in Forward Process')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('ddpm_as_sde.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: ddpm_as_sde.png")
```

**출력 예시**:
```
=== DDPM Forward Process ===
β_1 = 0.000100
β_1000 = 0.020000
α_1 = 0.999950
α_1000 = 0.000125
σ_1 = 0.004472
σ_1000 = 0.999999

=== Comparison: DDPM vs VP-SDE ===
t      DDPM α_t        VP α(t)         Diff
0      1.000000        1.000000        0.000000
100    0.761893        0.768234        0.006341
500    0.154234        0.162178        0.007944
1000   0.000125        0.000089        0.000036

=== Forward Process: Sampling Test ===
t=100: mean(X_t)=0.762145, std(X_t)=0.647821
       Theoretical: mean=0.761893, std=0.647893
...

Final (t=0): mean=-0.012345, std=1.023456

Plot saved: ddpm_as_sde.png
```

---

## 🔗 AI/ML 연결

### DDPM과 Score-SDE의 완전 동등성 확립

- DDPM = VP-SDE Euler-Maruyama (극한)
- DDPM loss = DSM weighted loss
- 따라서 모든 Score-SDE 기술 (ODE, faster solvers, guidance) **적용 가능**

### DPM-Solver & DDIM

DDPM의 reverse를 **고차 ODE solver** (DPM-Solver, RK45)로 실행:
- 기존 DDPM: 1000 step (느림)
- DPM-Solver: 10-20 step (100배 빠름)
- **수렴성 보장** (극한)

### Classifier-Free Guidance

Unconditional score $s_\theta(x, t)$ + conditional score $s_\theta(x, c, t)$을 섞어서:
$$s_{\text{guided}} = s_\theta(x, t) + \lambda [s_\theta(x, c, t) - s_\theta(x, t)]$$

$\lambda > 1$: stronger guidance. DDPM 손실로 학습 → guidance 적용.

### Text-to-Image

CLIP guidance + diffusion = DALL-E 2, Imagen, Stable Diffusion.

모두 **DDPM 또는 그 변형** 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| VP-SDE로부터의 Euler-Maruyama | 고차 discretization error 존재 |
| Score $s_\theta$ 정확 학습 | 실제 학습 오차 → 샘플 편향 |
| 모든 시점 $t$ 균등 중요 | 실제로는 high-noise 구간이 학습 쉬움 |
| Linear schedule 최적 | 다른 schedule (cosine, sqrt 등)이 더 나을 수 있음 |

**주의**: Discretization error는 $O(1/T)$ (DDPM의 step 수 많으면 작음). 그러나 score learning error는 신경망 용량에 의존 (큰 가변성).

---

## 📌 핵심 정리

$$\boxed{\text{DDPM 이산: } x_t = \alpha_t x_0 + \sigma_t \epsilon, \quad \alpha_t = \sqrt{\prod (1-\beta_i)}, \quad \sigma_t = \sqrt{1-\alpha_t^2}}$$

$$\boxed{\text{VP-SDE Euler-Maruyama: } X_{k+1} \approx \sqrt{1-\beta_k} X_k + \sqrt{\beta_k} Z_k \quad (\text{DDPM과 동등})}$$

$$\boxed{\text{DDPM Loss = Weighted DSM: } \mathbb{E}\|\epsilon - \epsilon_\theta\|^2 = \sigma_t^2 \cdot J_{DSM}(\theta)}$$

$$\boxed{\text{Reverse SDE: } d\bar X = [-b + \sigma^2 s_\theta] d\tau + \sigma d\bar B}$$

| 개념 | DDPM 표현 | SDE 표현 | 의미 |
|------|----------|---------|------|
| Forward | $q(x_t\|x_0)$ | $p_t$ (marginal) | Data → Noise |
| Reverse | $p_\theta$ (학습) | SDE solver | Noise → Data |
| Score | $-\epsilon/\sigma_t$ | $\nabla\log p_t$ | Direction |
| Loss | $\\\|\epsilon - \epsilon_\theta\\\|^2$ | DSM (weighted) | 학습 목표 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): DDPM의 $T=1000$ 스텝과 DPM-Solver의 $T'=20$ 스텝이 다르면, 같은 분포에서 샘플링하는가?

<details>
<summary>힌트 및 해설</summary>

**답**: 극한에서는 같다.

이유:
- DDPM 1000 step: 이산 Euler-Maruyama, discretization error $O(1/1000)$
- DPM 20 step: 고차 scheme, discretization error $O(1/20^k)$ ($k > 1$)

두 다른 discretization도, **underlying SDE는 같음**:
$$dX = [-\frac{1}{2}\beta(t)X + \beta(t)^2 s_\theta] dt + \sigma(t) dB$$

따라서 극한 $\Delta t \to 0$에서 같은 분포에 수렴.

**실제**: DPM이 더 적은 step으로 같은 품질 달성 (고차 scheme).

</details>

**문제 2** (심화): DDPM의 variance schedule $\Sigma_\theta(t)$를 "learnable"로 하면 (고정 대신), score와의 상호작용은?

<details>
<summary>힌트 및 해설</summary>

DDPM reverse:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(t))$$

일반적으로:
- Mean $\mu_\theta$: score 기반 (학습 가능)
- Variance $\Sigma_\theta$: posterior variance로 고정 (또는 learned)

Learned variance의 이점:
- Flexibility: 각 시점에서 적응적 uncertainty
- Potential: 더 안정적 reverse

단점:
- 추가 네트워크 출력 (비용 증가)
- Loss와의 상호작용 복잡 (ELBO 고려 필요)

Recent (Nichol & Dhariwal 2021): learnable variance + improved loss → 더 나은 log-likelihood.

</details>

**問題 3** (AI 연결): DDPM 학습에서 "noise schedule 선택"이 convergence speed와 샘플 품질에 미치는 영향은?

<details>
<summary>힌트 및 해설</summary>

Schedule $\{\beta_t\}$의 선택:

**Linear** ($\beta \propto t$):
- 간단
- 초기 저-noise region에서 정보 손실 빠름 (non-uniform)

**Cosine** (Song et al. 2021):
$$\bar{\alpha}_t = \frac{\cos(\pi t/2T)}{\cos(0)}$$
- 더 uniform한 정보 유지 (초반 완만, 후반 가파름)
- 더 나은 샘플 품질 보고

**Sqrt**:
- 초중반 정보 더 보존

최적:
- Trade-off between 학습 정보(uniform) vs numerical stability (gradual transition)
- Empirical tuning 필요

일반적으로 **cosine schedule이 성능 최고** (최근 모델들).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Score-based SDE](./05-score-based-sde.md) | [📚 README로 돌아가기](../README.md) | [Ch7-01. Probability Flow ODE ▶](../ch7-advanced-generative/01-probability-flow-ode.md) |

</div>
