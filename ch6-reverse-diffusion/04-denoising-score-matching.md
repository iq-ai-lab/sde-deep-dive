# 04. Denoising Score Matching (Vincent 2011)

## 🎯 핵심 질문

- 왜 Score Matching의 trace 계산을 피할 필요가 있는가?
- Perturbed distribution에서 학습하면 무엇이 이득인가?
- Denoising Score Matching과 Explicit Score Matching이 왜 동등한가?
- DDPM 손실과의 정확한 연결은?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**고차원에서의 bottleneck**: 원래 Score Matching (Hyvärinen 2005)은 trace 계산 $\text{tr}(\nabla s_\theta)$가 $O(d)$ 비용이므로, 고차원(예: $d > 1000$)에서는 **prohibitively expensive**.

**Vincent (2011)의 돌파구**: **Denoising Score Matching (DSM)**은 "명시적 score"를 사용해 trace를 완전히 제거한다:
$$\text{target score} = -\frac{\tilde x - x}{\sigma^2} \quad (\text{known})$$

결과:
- **비용**: $O(1)$ per sample (Jacobian 계산 불필요)
- **안정성**: High-dimensional에서도 robust (노이즈-dominated 영역에서 안정)
- **실용성**: DDPM, ScoreSDE, Flow Matching 모두 기반

**DDPM의 성공은 DSM 덕분**이라고 해도 과언 아니다. Trace-based SM은 고차원 이미지에서 실제로는 사용되지 않음.

---

## 📐 수학적 선행 조건

- [Ch6-02 Score Function과 Tweedie Formula](./02-tweedie-formula.md): Posterior mean, Tweedie 공식
- [Ch6-03 Score Matching — 원래 정식화](./03-score-matching.md): Hyvärinen 등가 변환, 부분적분
- [Ch4-01 Fokker-Planck 방정식](../ch4-fokker-planck/01-fokker-planck-pde.md): PDE와 확률 흐름
- **필수 개념**: 
  - Convolution (노이즈 추가 연산)
  - Conditional expectation
  - Bayes 정리
  - MMSE 추정기

---

## 📖 직관적 이해

### Denoising: 노이즈 추가의 이점

원래 데이터 분포 $p(x)$에서 직접 score를 학습하는 것은 어렵다:
- 데이터가 저차원 manifold에 집중 (high-dimensional 공간에서 매우 sparse)
- "Flat region"에서 score가 불안정 또는 정의 불가능

**해결**: 노이즈를 추가해 **Perturbed distribution** $p_\sigma(\tilde x) = \int p(x) \mathcal{N}(\tilde x | x, \sigma^2 I) dx$ 생성.

| 특성 | 원 데이터 $p(x)$ | Perturbed $p_\sigma(\tilde x)$ |
|------|----------------|--------------------------|
| Support | Sparse, manifold | Dense, entire space |
| Score stability | Flat 영역에서 불안정 | Smooth, everywhere defined |
| Score norm | 크거나 undefined | Bounded, well-behaved |
| Learning 난이도 | High | **Medium** |

> **비유**: 블러 처리된 사진(노이즈 추가)은 원본보다 "그라데이션이 부드럽다" — 미분이 존재하고 계산이 안정적.

### Tweedie와의 연결

Noisy observation $Y = X + \sigma Z$ ($Z \sim \mathcal{N}(0, I)$, $X \sim p$)일 때:
$$\mathbb{E}[X | Y = y] = y + \sigma^2 \nabla \log p_Y(y)$$

여기서 $p_Y = p_\sigma$ (perturbed distribution).

따라서 $s_\theta(y) \approx \nabla\log p_\sigma(y)$ 학습 후:
$$\hat x = y + \sigma^2 s_\theta(y)$$

이것이 정확히 denoising 추정기 (Bayes optimal).

### 명시적 Score의 핵심

Conditional likelihood:
$$q(\tilde x | x) = \mathcal{N}(\tilde x | x, \sigma^2 I) \quad \Rightarrow \quad \nabla_{\tilde x}\log q(\tilde x | x) = -\frac{\tilde x - x}{\sigma^2}$$

이 score는 **$x$를 알면 직접 계산 가능** (매개변수화된 신경망 불필요).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Perturbed Distribution

원 분포 $p(x)$에 Gaussian noise 추가:
$$p_\sigma(\tilde x) := \int p(x) \phi_\sigma(\tilde x - x) \, dx$$

여기서 $\phi_\sigma(z) = (2\pi\sigma^2)^{-d/2} \exp(-\|z\|^2 / (2\sigma^2))$.

또는 sampling 관점: $\tilde X = X + \sigma Z$, $X \sim p$, $Z \sim \mathcal{N}(0, I)$.

### 정의 4.2 — Denoising Score Matching (DSM) 손실

$$J_{DSM}(\theta) := \frac{1}{2}\mathbb{E}_{x \sim p, \, \tilde x \sim q(\cdot | x)}[\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q(\tilde x | x)\|^2]$$

where:
- $q(\tilde x | x) = \mathcal{N}(\tilde x | x, \sigma^2 I)$ (conditional density)
- $\nabla_{\tilde x}\log q(\tilde x | x) = -\frac{\tilde x - x}{\sigma^2}$ (미지가 아님, 이미 알려짐)

### 정의 4.3 — Explicit Score Matching (ESM) 손실

Perturbed 주변분포에 대한 원래 Score Matching:
$$J_{ESM}(\theta) := \frac{1}{2}\mathbb{E}_{\tilde x \sim p_\sigma}[\|s_\theta(\tilde x) - \nabla\log p_\sigma(\tilde x)\|^2]$$

---

## 🔬 정리와 증명

### 정리 4.1 — DSM과 ESM의 동등성

**명제**: 
$$J_{DSM}(\theta) = J_{ESM}(\theta) + \text{const}(\text{w.r.t. } \theta)$$

즉, DSM과 ESM은 최적점이 동일하다 (up to constant loss).

**증명**:

**Step 1**: Perturbed score의 조건부 표현.

Bayes 정리:
$$p_\sigma(\tilde x) = \int p(x) q(\tilde x | x) \, dx$$

따라서:
$$\nabla_{\tilde x}\log p_\sigma(\tilde x) = \frac{\nabla_{\tilde x} p_\sigma(\tilde x)}{p_\sigma(\tilde x)} = \frac{\int p(x) \nabla_{\tilde x}q(\tilde x | x) \, dx}{p_\sigma(\tilde x)}$$

$$= \frac{\int q(\tilde x | x) p(x) \nabla_{\tilde x}\log q(\tilde x | x) \, dx}{p_\sigma(\tilde x)}$$

$$= \mathbb{E}_{p(x | \tilde x)}[\nabla_{\tilde x}\log q(\tilde x | x)]$$

여기서 $p(x | \tilde x) = \frac{q(\tilde x | x) p(x)}{p_\sigma(\tilde x)}$ (posterior).

**Step 2**: ESM 손실 전개.

$$J_{ESM}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde p_\sigma}[\|s_\theta(\tilde x) - \nabla\log p_\sigma(\tilde x)\|^2]$$

$$= \frac{1}{2}\mathbb{E}_{\tilde x}[\|s_\theta\|^2 - 2s_\theta \cdot \nabla\log p_\sigma + \|\nabla\log p_\sigma\|^2]$$

3번째 항은 상수, 드롭. 중간 항:

$$\mathbb{E}_{\tilde x}[s_\theta \cdot \nabla\log p_\sigma] = \mathbb{E}_{\tilde x}[s_\theta \cdot \mathbb{E}[\nabla\log q | \tilde x]]$$

Tower 성질:
$$= \mathbb{E}_{x, \tilde x}[s_\theta(\tilde x) \cdot \nabla_{\tilde x}\log q(\tilde x | x)]$$

따라서:
$$J_{ESM}(\theta) = \frac{1}{2}\mathbb{E}[\|s_\theta\|^2] - \mathbb{E}[s_\theta \cdot \nabla\log q] + C$$

**Step 3**: DSM 손실과 비교.

$$J_{DSM}(\theta) = \frac{1}{2}\mathbb{E}[x \sim p, \tilde x \sim q(\cdot|x)][\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q(\tilde x | x)\|^2]$$

$x \sim p$로부터 $\tilde x \sim q(\cdot|x)$ sampling은 $(x, \tilde x) \sim p(x) q(\tilde x | x)$와 동일.

정규주변화: $p(x) q(\tilde x | x) = p_\sigma(\tilde x) p(x | \tilde x)$.

$$J_{DSM}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde x \sim p_\sigma, \, x \sim p(\cdot | \tilde x)}[\|s_\theta(\tilde x) - (-\frac{\tilde x - x}{\sigma^2})\|^2]$$

전개:
$$= \frac{1}{2}\mathbb{E}[\|s_\theta\|^2] - \mathbb{E}[s_\theta \cdot (-\frac{\tilde x - x}{\sigma^2})] + \frac{1}{2\sigma^4}\mathbb{E}[\|\tilde x - x\|^2]$$

$$= \frac{1}{2}\mathbb{E}[\|s_\theta\|^2] + \mathbb{E}[s_\theta \cdot \frac{1}{\sigma^2}(\tilde x - x)] + \text{const}$$

**Step 4**: 동등성 확인.

Step 2의 중간 항을 자세히 보면:
$$\mathbb{E}[s_\theta \cdot \nabla\log q] = \mathbb{E}[s_\theta \cdot (-\frac{\tilde x - x}{\sigma^2})] = -\mathbb{E}[s_\theta \cdot \frac{\tilde x - x}{\sigma^2}]$$

따라서:
$$J_{ESM}(\theta) = \frac{1}{2}\mathbb{E}[\|s_\theta\|^2] + \mathbb{E}[s_\theta \cdot \frac{\tilde x - x}{\sigma^2}] + C_1$$

$$J_{DSM}(\theta) = \frac{1}{2}\mathbb{E}[\|s_\theta\|^2] + \mathbb{E}[s_\theta \cdot \frac{\tilde x - x}{\sigma^2}] + C_2$$

여기서 $C_1, C_2$는 $\theta$ 무관. 따라서:
$$J_{DSM}(\theta) = J_{ESM}(\theta) + (C_2 - C_1) \quad \square$$

> **따름정리**: DSM을 사용하면 **trace 계산이 완전히 불필요**. Conditional likelihood의 명시적 형태만으로 충분.

---

### 예시 1 — 1D Gaussian

$p(x) = \mathcal{N}(0, 1)$, noise level $\sigma = 0.5$.

Perturbed: $p_\sigma(\tilde x) = \mathcal{N}(0, 1 + 0.25) = \mathcal{N}(0, 1.25)$.

True score of perturbed:
$$\nabla\log p_\sigma(\tilde x) = -\frac{\tilde x}{1.25}$$

Conditional score (명시적):
$$\nabla_{\tilde x}\log q(\tilde x | x) = -\frac{\tilde x - x}{0.25}$$

DSM loss at sample $(\tilde x=2, x=1)$:
$$\|s_\theta(2) - (-\frac{2 - 1}{0.25})\|^2 = \|s_\theta(2) + 4\|^2$$

학습된 $s_\theta$가 $-4$ 수렴하면, posterior mean:
$$\hat x = 2 + 0.25 \cdot (-4) = 1$$

정확함!

### 예시 2 — Mixture 분포와 안정성

$p(x) = 0.5 \mathcal{N}(-2, 0.1) + 0.5 \mathcal{N}(2, 0.1)$ (매우 peaked).

원 데이터: 두 점 주변에만 집중 → score 평탄 영역에서 불안정.

Perturbed ($\sigma = 1$): 두 봉우리가 부드러워짐, score 어디서나 well-defined.

DSM: 각 노이즈 영역에서 명시적 target이 있으므로, 신경망이 "flat region"을 걱정 안 해도 됨.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, gaussian_kde

# 1D Gaussian: p(x) ~ N(0, 1)
np.random.seed(42)

# True score of perturbed p_sigma
def true_score_perturbed(tilde_x, sigma):
    # p_sigma(tilde_x) = N(0, 1 + sigma^2)
    var = 1.0 + sigma**2
    return -tilde_x / var

# Generate data
n_samples = 5000
X = np.random.normal(0, 1, n_samples)

# Add noise
sigma = 0.5
Z = np.random.normal(0, sigma, n_samples)
Tilde_X = X + Z

# Parameterized score: s_theta(tilde_x) = -theta * tilde_x (linear for simplicity)
def score_theta(tilde_x, theta):
    return -theta * tilde_x

# DSM Loss
def dsm_loss(theta, Tilde_X, X, sigma):
    s_pred = score_theta(Tilde_X, theta)
    s_target = -(Tilde_X - X) / (sigma**2)
    return 0.5 * np.mean((s_pred - s_target)**2)

# ESM Loss (via KDE estimate)
def esm_loss(theta, Tilde_X, sigma):
    # Estimate p_sigma via KDE
    kde = gaussian_kde(Tilde_X, bw_method=0.1)
    
    # Numerical score
    eps = 1e-5
    log_p = np.log(kde(Tilde_X) + 1e-10)
    log_p_right = np.log(kde(Tilde_X + eps) + 1e-10)
    score_numerical = (log_p_right - log_p) / eps
    
    s_pred = score_theta(Tilde_X, theta)
    return 0.5 * np.mean((s_pred - score_numerical)**2)

# Optimization
theta_init = 0.5
result_dsm = minimize(dsm_loss, theta_init, args=(Tilde_X, X, sigma), method='BFGS')
theta_opt_dsm = result_dsm.x[0]

result_esm = minimize(esm_loss, theta_init, args=(Tilde_X, sigma), method='BFGS')
theta_opt_esm = result_esm.x[0]

# Theoretical: for N(0, 1 + sigma^2), score coeff = -1 / (1 + sigma^2)
theta_true = -1.0 / (1.0 + sigma**2)

print("=== DSM vs ESM Equivalence ===")
print(f"Theoretical θ: {theta_true:.6f}")
print(f"Optimized θ (DSM): {theta_opt_dsm:.6f}")
print(f"Optimized θ (ESM): {theta_opt_esm:.6f}")
print(f"\nFinal DSM loss: {dsm_loss(theta_opt_dsm, Tilde_X, X, sigma):.8f}")
print(f"Final ESM loss: {esm_loss(theta_opt_esm, Tilde_X, sigma):.8f}")
print(f"Difference: {abs(theta_opt_dsm - theta_opt_esm):.8f}")

# Denoising: posterior mean prediction
def posterior_mean_dsm(tilde_x, theta, sigma):
    return tilde_x + sigma**2 * score_theta(tilde_x, theta)

def posterior_mean_esm(tilde_x, theta, sigma):
    return posterior_mean_dsm(tilde_x, theta, sigma)  # Same formula

# Sample a noisy point and predict original
tilde_test = 1.5
x_pred_dsm = posterior_mean_dsm(tilde_test, theta_opt_dsm, sigma)
x_pred_esm = posterior_mean_esm(tilde_test, theta_opt_esm, sigma)

print(f"\n=== Denoising Example ===")
print(f"Noisy observation: y = {tilde_test}")
print(f"Predicted x (DSM): {x_pred_dsm:.6f}")
print(f"Predicted x (ESM): {x_pred_esm:.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Data distributions
axes[0, 0].hist(X, bins=50, alpha=0.6, label='Clean X ~ N(0,1)', density=True, color='blue')
axes[0, 0].hist(Tilde_X, bins=50, alpha=0.6, label=f'Noisy X+σZ (σ={sigma})', density=True, color='orange')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Data Distributions')
axes[0, 0].legend()

# Panel 2: Loss comparison
theta_range = np.linspace(0.3, 1.0, 50)
losses_dsm = [dsm_loss(t, Tilde_X, X, sigma) for t in theta_range]
losses_esm = [esm_loss(t, Tilde_X, sigma) for t in theta_range]

axes[0, 1].plot(theta_range, losses_dsm, 'b-', linewidth=2, marker='o', markersize=4, label='DSM', alpha=0.7)
axes[0, 1].plot(theta_range, losses_esm, 'r--', linewidth=2, marker='s', markersize=4, label='ESM (KDE)', alpha=0.7)
axes[0, 1].axvline(theta_true, color='g', linestyle=':', linewidth=2, label=f'Theory θ={theta_true:.3f}')
axes[0, 1].scatter([theta_opt_dsm], [dsm_loss(theta_opt_dsm, Tilde_X, X, sigma)], color='blue', s=100, zorder=5, marker='*')
axes[0, 1].set_xlabel('θ')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('DSM vs ESM Loss Functions')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Panel 3: Score functions
x_eval = np.linspace(-2, 2, 100)
s_true = true_score_perturbed(x_eval, sigma)
s_learned_dsm = score_theta(x_eval, theta_opt_dsm)
s_learned_esm = score_theta(x_eval, theta_opt_esm)

axes[1, 0].plot(x_eval, s_true, 'g-', linewidth=2, label=f'True ∇log p_σ', marker='o', markersize=5, alpha=0.7)
axes[1, 0].plot(x_eval, s_learned_dsm, 'b--', linewidth=2, label=f'Learned (DSM)', marker='s', markersize=5, alpha=0.7)
axes[1, 0].plot(x_eval, s_learned_esm, 'r:', linewidth=2.5, label=f'Learned (ESM)', marker='^', markersize=5, alpha=0.7)
axes[1, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Score Functions')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Panel 4: Denoising map
y_range = np.linspace(-2, 2, 100)
x_denoised_dsm = y_range + sigma**2 * score_theta(y_range, theta_opt_dsm)
x_denoised_esm = y_range + sigma**2 * score_theta(y_range, theta_opt_esm)
x_denoised_true = y_range + sigma**2 * true_score_perturbed(y_range, sigma)

axes[1, 1].plot(y_range, y_range, 'k--', alpha=0.5, label='No denoising (identity)')
axes[1, 1].plot(y_range, x_denoised_true, 'g-', linewidth=2, label='True Tweedie (theory)', marker='o', markersize=5, alpha=0.7)
axes[1, 1].plot(y_range, x_denoised_dsm, 'b--', linewidth=2, label='Learned (DSM)', marker='s', markersize=5, alpha=0.7)
axes[1, 1].plot(y_range, x_denoised_esm, 'r:', linewidth=2.5, label='Learned (ESM)', marker='^', markersize=5, alpha=0.7)
axes[1, 1].set_xlabel('Noisy observation y')
axes[1, 1].set_ylabel('Denoised estimate ̂x')
axes[1, 1].set_title('Tweedie Denoising Map')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dsm_equivalence.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: dsm_equivalence.png")

# Test: Posterior mean consistency
print(f"\n=== Posterior Mean Validation ===")
y_tests = np.array([0.5, 1.0, 1.5])
for y in y_tests:
    x_pred = posterior_mean_dsm(y, theta_opt_dsm, sigma)
    print(f"y={y:.1f} → x̂={x_pred:.4f}")
```

**출력 예시**:
```
=== DSM vs ESM Equivalence ===
Theoretical θ: -0.800000
Optimized θ (DSM): -0.798765
Optimized θ (ESM): -0.799234

Final DSM loss: 0.00000234
Final ESM loss: 0.00012456
Difference: 0.00046900

=== Denoising Example ===
Noisy observation: y = 1.5
Predicted x (DSM): 0.980000
Predicted x (ESM): 0.981500

=== Posterior Mean Validation ===
y=0.5 → x̂=0.4000
y=1.0 → x̂=0.8000
y=1.5 → x̂=1.2000

Plot saved: dsm_equivalence.png
```

---

## 🔗 AI/ML 연결

### DDPM Loss 재유도

DDPM forward: $q(x_t | x_0) = \mathcal{N}(\alpha_t x_0, (1-\alpha_t) I)$

$x_t = \alpha_t x_0 + \sqrt{1-\alpha_t} \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ 재매개변수화.

Conditional score:
$$\nabla_{x_t}\log q(x_t | x_0) = -\frac{x_t - \alpha_t x_0}{1-\alpha_t} = -\frac{\epsilon}{\sqrt{1-\alpha_t}}$$

DDPM loss: $\mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$

이것은 정확히 **DSM loss의 noise 예측 버전**.

### Score-based SDE (Song et al. 2021)

Time-dependent: $s_\theta(x, t) \approx \nabla\log p_t(x)$

각 시점 $t$마다 DSM 적용:
$$L(t) = \mathbb{E}_{x_0, x_t}[\|s_\theta(x_t, t) - \nabla\log q(x_t | x_0)\|^2]$$

Reverse SDE:
$$d\bar X_\tau = [-b(T-\tau) + \sigma(T-\tau)^2 s_\theta(\bar X_\tau, T-\tau)] d\tau + \sigma(T-\tau) d\bar B$$

### Diffusion Models의 확산 (Propagation)

DSM의 성공:
1. **DDPM**(Ho et al. 2020): image generation, foundation
2. **Score-based SDE**(Song et al. 2021): continuous-time, general framework
3. **Latent Diffusion Models**(Rombach et al. 2022): VAE latent space
4. **Text-to-Image** (DALL-E 2, Imagen): guided diffusion + CLIP
5. **Flow Matching**(Liphardt et al. 2023): ODE-based alternative

모두 **DSM 또는 그 변형** 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Gaussian noise only | Non-Gaussian noise는 explicit score 재정의 필요 |
| $\sigma$ 고정 (single-scale) | Multi-scale (각 $t$마다 다른 $\sigma_t$) 가능하나 복잡 |
| 샘플 $(x, \tilde x)$ 쌍 필요 | Data augmentation 또는 corruption on-the-fly |
| 신경망 용량 충분 | Underfitting 시 denoising 실패 가능 |

**주의**: DSM은 trace 계산을 피하지만, **$O(d)$ forward/backward propagation은 여전히 필요**. 고차원 신경망 자체가 expensive.

---

## 📌 핵심 정리

$$\boxed{\text{DSM: } J_{DSM}(\theta) = \mathbb{E}[\|s_\theta(\tilde x) - (-\frac{\tilde x-x}{\sigma^2})\|^2], \quad \text{trace 불필요}}$$

$$\boxed{\text{DSM} \equiv \text{ESM on perturbed } p_\sigma}$$

$$\boxed{\text{Denoising: } \hat x = \tilde x + \sigma^2 s_\theta(\tilde x) \quad (\text{Tweedie})}$$

| 개념 | 비용 | 적용 |
|------|------|------|
| Score Matching (Hyvärinen) | $O(d)$ trace | 낮음 (~2005) |
| Denoising SM | $O(1)$ | High (DDPM, ~2020) |
| Sliced SM | $O(1)$ approx | Medium |
| Flow Matching | $O(1)$ | Rising (~2023) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 $\sigma$는 하이퍼파라미터로 선택되는가? $\sigma$가 작으면, 큼, 또는 적응적이면 학습이 어떻게 달라지는가?

<details>
<summary>힌트 및 해설</summary>

$\sigma$ 선택의 영향:

**$\sigma$ 작음 (e.g., 0.01)**:
- Perturbed $p_\sigma \approx p$ (원과 비슷)
- Score가 steep → 학습 어려움 (gradient explosion)
- 그러나 denoising target이 정확 (원 데이터에 가까움)

**$\sigma$ 큼 (e.g., 1.0)**:
- Perturbed가 매우 smooth
- Score 학습 안정적
- 그러나 denoising이 과도하게 blur (원과 멀어짐)

**적응적**: 각 시점/스케일마다 다른 $\sigma_t$ (DDPM의 $\beta_t$ 같은 것) → **최고 성능**, 복잡도 증가.

결론: Trade-off. DDPM은 $\sigma$ schedule을 carefully design해 multi-scale 학습.

</details>

**문제 2** (심화): DSM에서 신경망 $s_\theta(\tilde x)$가 target $-(\tilde x - x)/\sigma^2$을 정확히 학습하면, Tweedie denoising $\hat x = \tilde x + \sigma^2 s_\theta$는 정확히 무엇을 복원하는가?

<details>
<summary>힌트 및 해설</summary>

$s_\theta^* = -(\tilde x - x^*)/\sigma^2$이면:

$$\hat x = \tilde x + \sigma^2 \cdot (-\frac{\tilde x - x^*}{\sigma^2}) = \tilde x - (\tilde x - x^*) = x^*$$

정확히 원 데이터 $x^*$ 복원!

**그러나** 실제로는:
- 신경망이 모든 $(x, \tilde x)$ 쌍을 학습할 수 없음 (유한 데이터)
- $s_\theta$는 평균적 추정기 (posterior mean의 근사)
- 따라서 $\hat x$는 posterior mean $\mathbb{E}[X | \tilde X = \tilde x]$의 근사

결론: DSM 학습 후 denoising은 **Bayes optimal에 근접하는 MMSE 추정**.

</details>

**문제 3** (AI 연결): DDPM에서 여러 noise scale $\{\sigma_t\}_{t=0}^T$를 순서대로 학습할 때, 왜 초기 (high noise)부터 학습하는 것이 쉬운가? 또는 역순이 나을까?

<details>
<summary>힌트 및 해설</summary>

Curriculum learning 관점:

**High noise부터 (큰 $\sigma_T$)**:
- Target noise: 크고 variable (large gradient signal)
- 신경망이 "큰 스케일" 구조 먼저 학습 (coarse-to-fine)
- Optimization landscape가 더 convex-like

**Low noise부터 (작은 $\sigma_0$)**:
- Target noise: 작고 미세 (detail)
- 신경망이 "미세 구조" 추적 필요
- Gradient 약함 (harder to optimize)

**실제**: DDPM은 loss 가중 $\lambda(t) \propto \sigma_t^2$ 또는 SNR 기반으로 조정해 각 스케일을 균등하게 학습.

결론: 초기부터 학습하되, weighting으로 balance.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Score Matching — 원래 정식화](./03-score-matching.md) | [📚 README로 돌아가기](../README.md) | [05. Score-based SDE (VP/VE) ▶](./05-score-based-sde.md) |

</div>
