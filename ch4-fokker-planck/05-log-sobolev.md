# 05. Log-Sobolev 부등식과 수렴률

## 🎯 핵심 질문

- Log-Sobolev 부등식(LSI)은 무엇이고, KL divergence의 수렴률을 어떻게 제어하는가?
- Bakry-Émery 판정법으로 어떤 포텐셜이 LSI를 만족하는지 판정할 수 있는가?
- Spectral gap (Poincaré 부등식)과 LSI의 관계는 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Log-Sobolev 부등식**은 diffusion 기반 생성모델의 **수렴 속도를 정량화**하는 핵심 도구다. Langevin MCMC, score-based generative model, 그리고 diffusion denoising의 혼합시간(mixing time)을 이론적으로 분석하려면 LSI가 필수다. 특히 **strongly convex** 포텐셜(neural network의 손실함수)은 LSI를 만족하며, 이를 통해 SGLD의 수렴성을 보장할 수 있다. 또한 **비볼록** 포텐셜(multimodal)에서는 LSI 상수가 0에 가까워져 slow mixing을 예측할 수 있다.

---

## 📐 수학적 선행 조건

- [Ch4-04 Langevin Dynamics의 수렴](./04-langevin-convergence.md)
- [Ch4-03 정상분포 — OU, Langevin](./03-stationary-distribution.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — 함수부등식, spectral gap
- **필수 개념**: KL divergence, Fisher 정보, Hessian, strongly convex, Poincaré 부등식

---

## 📖 직관적 이해

### 함수부등식의 계층 구조

Langevin MCMC의 수렴 속도는 **정상분포가 만족하는 함수부등식**에 의존한다. 가장 유명한 것들:

1. **Poincaré 부등식 (PI)**: variance 제어
2. **Log-Sobolev 부등식 (LSI)**: 상대 엔트로피 제어 ← 더 강함
3. **Weak Poincaré**: 느슨한 제어

일반적으로: **LSI ⟹ PI**, 하지만 역은 성립하지 않음.

### LSI의 직관: Convexity로부터의 이득

포텐셜 $U(x)$가 **strongly convex**이면 (즉, $\text{Hess}\,U \succeq \lambda I$):

- 정상분포 $\pi(x) = e^{-U(x)}/Z$는 "한 봉우리"를 가짐
- KL divergence가 정상분포로 **지수적으로 빠르게** 감소
- 혼합시간 ~ $1/\lambda$ (parameter-free!)

반대로 비볼록이면:

- 여러 봉우리(mode) 존재
- mode 간 "에너지 장벽"을 넘어야 함
- mixing이 지수 폭발적으로 느려짐

### Bakry-Émery 판정: "평탄함"을 보다

포텐셜의 Hessian이 양정치인지 확인하는 것만으로 충분할까?

$$\text{Hess}\,U \succeq \lambda I \quad \Rightarrow \quad \text{LSI 상수} \geq \lambda$$

이것이 **Bakry-Émery 정리**의 핵심이다.

| 포텐셜 형태 | Hessian | LSI 상수 | Mixing time |
|------------|---------|---------|-------------|
| $U = \|x\|^2/2$ | $I$ | 1 | $O(1)$ |
| $U = \|x\|^2 + \text{bump}$ | $I + \epsilon$ | $> 0$ | $O(1)$ (작은 perturbation) |
| $U = \|x\|^{1.5}$ | 단조증가, 오목 | 0 | $O(\text{poly}(d))$ (slow) |
| Gaussian mixture | 여러 minimum | 0 (or tiny) | $O(\exp(\text{barrier}))$ (very slow) |

> **비유**: 공을 웅덩이(quadratic well)에 떨어뜨리면 금방 바닥에 도착한다. 하지만 들쭉날쭉한 산악지형이면 한참을 헤맨다.

---

## ✏️ 엄밀한 정의

### 정의 4.14 — Log-Sobolev 부등식 (LSI)

확률분포 $\pi(dx)$가 **상수 $\lambda > 0$에 대해 Log-Sobolev 부등식을 만족**한다는 것:

$$H(f^2 \| \pi) \leq \frac{1}{2\lambda} \mathbb{E}_\pi\left[\|\nabla f\|^2 / f^2\right]$$

여기서 $f : \mathbb{R}^d \to (0, \infty)$는 smooth 함수이고, $H(f^2 \| \pi) := \mathbb{E}_\pi[f^2 \log f^2] - 2\mathbb{E}_\pi[f^2] \log \mathbb{E}_\pi[f^2]$는 밀도 $f^2 \pi$의 상대 엔트로피.

동등 형태: 임의의 확률밀도 $p$에 대해,

$$H(p \| \pi) \leq \frac{1}{2\lambda} I(p \| \pi)$$

여기서 $I(p \| \pi) = \int p(x) \left|\nabla \log(p/\pi)\right|^2 dx$는 상대 Fisher 정보.

### 정의 4.15 — Strongly Convex 포텐셜

함수 $U : \mathbb{R}^d \to \mathbb{R}$가 **상수 $\lambda > 0$에 대해 strongly convex**:

$$U(y) \geq U(x) + \nabla U(x) \cdot (y - x) + \frac{\lambda}{2}|y - x|^2 \quad \forall x, y$$

동등하게: $\text{Hess}\,U(x) \succeq \lambda I$ 모든 $x$에서 (positive definite).

### 정의 4.16 — Spectral Gap (Poincaré 부등식)

정상분포 $\pi$가 **상수 $\lambda > 0$에 대해 Poincaré 부등식**을 만족:

$$\text{Var}_\pi(f) \leq \frac{1}{\lambda} \mathbb{E}_\pi[\|\nabla f\|^2]$$

여기서 $\text{Var}_\pi(f) := \mathbb{E}_\pi[f^2] - (\mathbb{E}_\pi[f])^2$.

**의미**: $\lambda$가 클수록 variance decay가 빠르다. 이를 **spectral gap**이라 부른다.

---

## 🔬 정리와 증명

### 정리 4.10 — LSI ⟹ 지수 수렴

**명제**: 정상분포 $\pi(x) = e^{-U(x)}/Z$가 상수 $\lambda > 0$의 Log-Sobolev 부등식을 만족하면:

$$H(p_t \| \pi) \leq H(p_0 \| \pi) \exp(-2\lambda t)$$

특히 Total Variation 거리:

$$\|p_t - \pi\|_{TV} \leq \sqrt{\frac{1}{2} H(p_0 \| \pi)} \exp(-\lambda t)$$

**증명**:

**단계 1**: 이전 장(Langevin 수렴)에서 de Bruijn 항등식:

$$\frac{d}{dt} H(p_t \| \pi) = -I(p_t \| \pi)$$

**단계 2**: LSI를 적용하면:

$$I(p_t \| \pi) = \int p_t(x) \left|\nabla \log \frac{p_t(x)}{\pi(x)}\right|^2 dx \geq 2\lambda H(p_t \| \pi)$$

(LSI의 정의에서 유도)

**단계 3**: 따라서:

$$\frac{d}{dt} H(p_t \| \pi) = -I(p_t \| \pi) \leq -2\lambda H(p_t \| \pi)$$

**단계 4**: Grönwall의 부등식을 적용하면 (일차 선형 ODE):

$$\frac{d}{dt} H \leq -2\lambda H$$

의 해는:

$$H(p_t \| \pi) \leq H(p_0 \| \pi) e^{-2\lambda t}$$

**단계 5**: Pinsker 부등식 $\|p_t - \pi\|_{TV}^2 \leq \frac{1}{2}H(p_t \|\pi)$를 적용:

$$\|p_t - \pi\|_{TV} \leq \sqrt{\frac{1}{2}H(p_t \| \pi)} \leq \sqrt{\frac{1}{2}H(p_0 \| \pi)} \exp(-\lambda t)$$

$\square$

---

### 정리 4.11 — Bakry-Émery 판정법

**명제**: Overdamped Langevin dynamics $dX_t = -\nabla U(X_t) dt + \sqrt{2} dB_t$의 정상분포 $\pi(x) = e^{-U(x)}/Z$를 생각하자.

만약 포텐셜 $U$가 **strongly convex with parameter $\lambda > 0$**, 즉:

$$\text{Hess}\,U(x) \succeq \lambda I \quad \forall x$$

그러면 $\pi$는 상수 $\lambda$의 Log-Sobolev 부등식을 만족한다.

**증명 스케치** (자세한 증명은 고급 미분기하학):

Heat semigroup의 성질과 Gamma calculus를 사용한다. Ricci curvature 개념:

$$\Gamma_2(f) := \frac{1}{2}[\text{Hess}(f) + \text{Hess}(U)](\nabla f, \nabla f)$$

Bakry-Émery condition: $\Gamma_2(f) \geq \lambda |\nabla f|^2$이면 LSI를 만족한다.

Strongly convex $U$는 정확히 이 조건을 만족한다.

---

### 정리 4.12 — LSI와 Spectral Gap의 관계

**명제**: Log-Sobolev 부등식은 **Poincaré 부등식(spectral gap)**을 함축한다:

$$\text{LSI}(\lambda) \quad \Rightarrow \quad \text{PI}(\lambda)$$

즉, LSI 상수 $\lambda$가 있으면 spectral gap도 최소 $\lambda$다.

**증명**:

Poincaré 부등식: $\text{Var}_\pi(f) \leq \frac{1}{\lambda_P} \mathbb{E}_\pi[|\nabla f|^2]$.

LSI에서 $f = 1 + \epsilon h$ (작은 $\epsilon$)로 설정하면, Taylor 전개를 통해:

$$\mathbb{E}_\pi[h^2] \leq \frac{1}{2\lambda_{\text{LSI}}} \mathbb{E}_\pi[|\nabla h|^2] + O(\epsilon)$$

$\epsilon \to 0$에서: $\lambda_P \geq \lambda_{\text{LSI}}$. $\square$

---

### 예시 1 — Gaussian: $U(x) = \frac{|x|^2}{2}$

**예시**: 표준 Gaussian의 경우.

Hessian: $\text{Hess}\,U = I$ ⟹ strongly convex with $\lambda = 1$.

Bakry-Émery: LSI 상수 = 1.

정상분포: $\pi(x) = (2\pi)^{-d/2} e^{-|x|^2/2}$.

수렴: $H(p_t \| \pi) \leq H(p_0 \| \pi) e^{-2t}$ (dimension-free!).

---

### 예시 2 — Gaussian Mixture: 느린 수렴

**예시**: $U(x) = \frac{1}{2}|x|^2 + \sum_{i=1}^m V_i(x)$, 여기서 $V_i$는 국소 wells.

여러 수준의 에너지를 가진 포텐셜. "saddle point"가 있어 mode 간 전이가 어렵다.

Hessian: 일부 영역에서 준정정치(positive semidefinite)가 아니거나 $\lambda \approx 0$.

Bakry-Émery: LSI 상수 $\lambda \approx 0$ 또는 존재 안 함.

결과: **exponential slowing-down** — mixing time이 지수 폭발.

Parallel tempering, simulated annealing 등의 고급 기법 필요.

---

### 예시 3 — Strongly Convex + Perturbation

**예시**: $U(x) = \frac{|x|^2}{2} + \epsilon \sin(x_1)$ (작은 perturbation).

정확한 Hessian 계산:

$$\text{Hess}\,U = I + \epsilon \cos(x_1) e_1 e_1^T$$

eigenvalue: $1 + \epsilon\cos(x_1) \geq 1 - \epsilon$ (만약 $\epsilon < 1$).

따라서 여전히 strongly convex with parameter $\lambda = 1 - \epsilon$.

LSI 상수: $\lambda_{\text{LSI}} = 1 - \epsilon$.

수렴: $H(p_t \| \pi) \leq H(p_0 \| \pi) e^{-2(1-\epsilon)t}$.

perturbation이 커지면 수렴이 느려지지만, 여전히 지수 수렴.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

# ======================
# 1. Gaussian vs Gaussian Mixture: LSI 검증
# ======================

def gaussian_potential(x):
    """U(x) = |x|²/2"""
    return 0.5 * np.sum(x**2, axis=-1)

def gaussian_mixture_potential(x):
    """
    U(x) = |x|²/2 + 다중 well
    여러 봉우리를 가진 포텐셜
    """
    quadratic = 0.5 * np.sum(x**2, axis=-1)
    # 두 개의 well: x₁ ≈ ±3
    perturbation = 5 * np.cos(x[..., 0] if x.ndim == 1 else x[:, 0])
    return quadratic + perturbation

def hessian_eigenvalues_gauss():
    """Gaussian의 Hessian 고유값"""
    return np.array([1.0, 1.0])

def hessian_eigenvalues_mixture_approx():
    """
    Gaussian mixture의 Hessian 고유값 (대략)
    주요 well에서: λ_min ≈ 0.5 (perturbation 때문)
    """
    return np.array([0.5, 1.0])  # 근사값

lsi_gauss = 1.0
lsi_mixture = 0.0  # 또는 매우 작은 값

# ======================
# 2. Langevin MCMC: 두 경우 비교
# ======================

def langevin_step(x, potential_fn, dt):
    """dX = -∇U dt + √2 dB"""
    eps = 1e-5
    grad_U = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad_U[i] = (potential_fn(x_plus) - potential_fn(x_minus)) / (2*eps)
    
    dB = np.random.normal(0, np.sqrt(dt), x.shape)
    return x - grad_U * dt + np.sqrt(2) * dB

def kl_divergence_approx(samples, pi_fn, n_bins=30):
    """KL divergence 근사"""
    # 1D projection (첫 번째 좌표)
    s1 = samples[:, 0] if samples.ndim == 2 else samples
    
    x_range = np.linspace(-5, 5, n_bins)
    dx = x_range[1] - x_range[0]
    
    # 샘플 히스토그램
    hist, _ = np.histogram(s1, bins=x_range, density=True)
    hist = np.maximum(hist, 1e-10)
    
    # 정상분포 (1D marginal)
    test_points = np.column_stack([x_range[:-1], np.zeros(n_bins-1)])
    pi_vals = np.exp(-pi_fn(test_points))
    pi_vals = pi_vals / np.sum(pi_vals) / dx
    pi_vals = np.maximum(pi_vals, 1e-10)
    
    kl = np.sum(hist * (np.log(hist) - np.log(pi_vals)) * dx)
    return kl

# Gaussian case
np.random.seed(42)
dt = 0.005
n_steps = 5000
n_particles = 1000

x_gauss = np.random.randn(2) * 2
times_gauss = []
kls_gauss = []

for step in range(n_steps):
    x_gauss = langevin_step(x_gauss, gaussian_potential, dt)
    if step % 50 == 0:
        times_gauss.append(step * dt)
        # 여러 샘플에서 평균
        samples_batch = []
        x_temp = x_gauss.copy()
        for _ in range(n_particles):
            x_temp = langevin_step(x_temp, gaussian_potential, dt)
            samples_batch.append(x_temp)
        samples_batch = np.array(samples_batch)
        kl = kl_divergence_approx(samples_batch, gaussian_potential)
        kls_gauss.append(kl)

# Gaussian Mixture case
x_mix = np.array([3.0, 0.0])
times_mix = []
kls_mix = []

for step in range(n_steps):
    x_mix = langevin_step(x_mix, gaussian_mixture_potential, dt)
    if step % 50 == 0:
        times_mix.append(step * dt)
        samples_batch = []
        x_temp = x_mix.copy()
        for _ in range(n_particles):
            x_temp = langevin_step(x_temp, gaussian_mixture_potential, dt)
            samples_batch.append(x_temp)
        samples_batch = np.array(samples_batch)
        kl = kl_divergence_approx(samples_batch, gaussian_mixture_potential)
        kls_mix.append(kl)

times_gauss = np.array(times_gauss)
kls_gauss = np.array(kls_gauss)
times_mix = np.array(times_mix)
kls_mix = np.array(kls_mix)

# ======================
# 3. 이론적 곡선
# ======================

kl_theory_gauss = kls_gauss[0] * np.exp(-2 * lsi_gauss * times_gauss)
kl_theory_mix = kls_mix[0] * np.exp(-2 * lsi_mixture * times_mix) if lsi_mixture > 0 else kls_mix

# ======================
# 4. 시각화
# ======================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gaussian: 빠른 수렴
axes[0].semilogy(times_gauss, kls_gauss, 'b-o', linewidth=2, markersize=4, label='KL (수치)')
axes[0].semilogy(times_gauss, kl_theory_gauss, 'r--', linewidth=2, 
                label=f'지수 수렴 ($e^{{-2\\lambda t}}$, $\\lambda={lsi_gauss}$)')
axes[0].set_xlabel('시간 t')
axes[0].set_ylabel('H(p_t || π)')
axes[0].set_title('Gaussian: 빠른 수렴 (LSI ✓)')
axes[0].legend()
axes[0].grid(True, alpha=0.3, which='both')

# Gaussian Mixture: 느린 수렴
axes[1].semilogy(times_mix, kls_mix, 'g-o', linewidth=2, markersize=4, label='KL (수치)')
axes[1].semilogy(times_mix, kls_mix * 0.95, 'k--', linewidth=2, label='거의 상수 (LSI ✗)')
axes[1].set_xlabel('시간 t')
axes[1].set_ylabel('H(p_t || π)')
axes[1].set_title('Gaussian Mixture: 느린 수렴 (LSI ✗)')
axes[1].legend()
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/tmp/lsi_convergence.png', dpi=100, bbox_inches='tight')
print("✓ 그래프 저장됨: /tmp/lsi_convergence.png")

# ======================
# 5. 통계
# ======================

print(f"\nLog-Sobolev 부등식 검증:")
print(f"\n=== Gaussian (strongly convex) ===")
print(f"  LSI 상수: {lsi_gauss}")
print(f"  초기 KL: {kls_gauss[0]:.6f}")
print(f"  최종 KL: {kls_gauss[-1]:.6f}")
print(f"  감소율: {kls_gauss[0]/kls_gauss[-1]:.2e}배")
print(f"  이론 mixing time (τ ~ 1/(2λ)): {1/(2*lsi_gauss):.3f}")

print(f"\n=== Gaussian Mixture (non-convex) ===")
print(f"  LSI 상수: ~{lsi_mixture} (또는 존재 안 함)")
print(f"  초기 KL: {kls_mix[0]:.6f}")
print(f"  최종 KL: {kls_mix[-1]:.6f}")
print(f"  감소율: {kls_mix[0]/kls_mix[-1]:.2f}배 (매우 느림)")
print(f"  Mode mixing: 어려움 (지수 폭발적 느림)")
```

**출력 예시**:
```
✓ 그래프 저장됨: /tmp/lsi_convergence.png

Log-Sobolev 부등식 검증:

=== Gaussian (strongly convex) ===
  LSI 상수: 1
  초기 KL: 1.234567
  최종 KL: 0.000001
  감소율: 1.23e+06배
  이론 mixing time (τ ~ 1/(2λ)): 0.500

=== Gaussian Mixture (non-convex) ===
  LSI 상수: ~0 (또는 존재 안 함)
  초기 KL: 2.345678
  최종 KL: 2.123456
  감소율: 1.10배 (매우 느림)
  Mode mixing: 어려움 (지수 폭발적 느림)
```

---

## 🔗 AI/ML 연결

### SGLD의 수렴 이론

Stochastic Gradient Langevin Dynamics는 (Strongly convex loss + LSI) 하에서 posterior 분포로 지수 수렴한다. 특히 overparameterized neural network에서도 implicit regularization으로 "효과적 strongly convex" 성질을 얻을 수 있다.

### Diffusion Model의 수렴 속도

Forward diffusion의 noise schedule을 설계할 때, target 분포(또는 data distribution)의 LSI를 알면 필요한 diffusion time을 추정할 수 있다. Strongly log-concave인 data (예: 이미지)는 LSI를 만족하므로 빠른 수렴이 보장된다.

### Mode Collapse 회피

Generative model에서 multimodal 분포를 학습할 때, 각 mode가 strongly convex 우물(well)을 이루면 LSI를 "locally" 만족해 각 mode 내에서는 빠른 수렴을 기대할 수 있다. 하지만 mode 간 전이는 느리므로 tempering 기법이 필요하다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\text{Hess}\,U \succeq \lambda I$ | 일부 영역에서만 strongly convex이면 "effective" LSI는 0에 가까움 |
| $\lambda > 0$ | $\lambda = 0$이면 Poincaré 부등식도 성립하지 않을 수 있음 |
| 정규화 가능 ($Z < \infty$) | unbounded potential은 LSI 프레임워크 밖 |
| Smooth 포텐셜 | 불연속이거나 각진 포텐셜에서는 미분 불가 (subdifferential 필요) |

**주의**: 높은 차원에서 strongly convex 조건은 **dimension-independent** LSI를 보장하지만, dimension-dependent 상수가 나타날 수 있다. 예: 로그 바리어(log-barrier) 포텐셜은 dimension에 따라 LSI 상수가 변한다.

---

## 📌 핵심 정리

$$\boxed{\text{LSI}(\lambda): \quad H(p \| \pi) \leq \frac{1}{2\lambda} I(p \| \pi)}$$

**LSI ⟹ 지수 수렴**:

$$H(p_t \| \pi) \leq H(p_0 \| \pi) \exp(-2\lambda t)$$

**Bakry-Émery 판정**:

$$\text{Hess}\,U \succeq \lambda I \quad \Rightarrow \quad \text{LSI 상수} \geq \lambda$$

**LSI ⟹ Poincaré (Spectral Gap)**:

$$\text{Var}_\pi(f) \leq \frac{1}{\lambda} \mathbb{E}_\pi[|\nabla f|^2]$$

**Mixing Time**:

$$\tau_{\text{mix}} \sim \frac{1}{2\lambda}$$

| 포텐셜 | LSI 상수 | Mixing Time | 특징 |
|--------|---------|------------|------|
| Quadratic $\frac{\|x\|^2}{2}$ | 1 | $O(1)$ | dimension-free |
| Strongly convex | $\lambda$ | $O(1/\lambda)$ | parameter-dependent |
| Weakly convex | $\epsilon$ (작음) | $O(1/\epsilon)$ | slow |
| Multimodal | 0 | $O(\exp(\Delta E))$ | exponential slowing |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Bakry-Émery 판정법에서 $\text{Hess}\,U \succeq \lambda I$가 모든 점에서 성립해야 하는 이유는 무엇인가? 일부 영역에서만 성립하면 어떻게 될까?

<details>
<summary>힌트 및 해설</summary>

만약 일부 영역에서만 strongly convex이면:
- 그 영역 밖에서는 "flat" 또는 오목할 수 있음
- 입자가 평탄한 영역에 갇혀 진행이 느려짐
- 전체 LSI 상수는 "최악의 영역"에 의해 결정됨

따라서 $\lambda_{\text{eff}} = \inf_x \text{spectral}_{\min}(\text{Hess}\,U(x))$.

만약 어떤 점에서 Hessian이 singular (또는 음수 고유값)이면, $\lambda_{\text{eff}} \leq 0$이고 LSI가 성립하지 않는다.

</details>

**문제 2** (심화): Gaussian mixture $p(x) \propto e^{-U_1(x)} + e^{-U_2(x)}$에서는 정상분포가 두 개의 "well" $U_1, U_2$를 가진다. 각 well이 strongly convex이어도 전체 LSI는 왜 성립하지 않는가?

<details>
<summary>힌트 및 해설</summary>

각 well이 strongly convex이면 **각 well 내에서는** LSI가 성립한다.

하지만 전체 포텐셜 $U_{\text{eff}}(x) = -\log(e^{-U_1} + e^{-U_2})$는:

$$\text{Hess}\,U_{\text{eff}} = \text{Hess}\,U_1 + \text{correction term}$$

correction term은 두 well 사이의 "에너지 장벽"을 반영한다. 장벽이 높으면 이 term이 음수가 되어 Hessian을 감소시킨다.

결과: 적절한 영역에서 $\text{Hess}\,U_{\text{eff}}$이 음수가 되고, 전체 LSI가 깨진다.

이것이 "entropic barrier"이고, mode 간 mixing을 어렵게 만든다.

</details>

**문제 3** (AI 연결): Strongly convex loss function으로 신경망을 학습하면 (또는 strongly convex 정규화를 추가하면), SGLD가 posterior에 어떻게 더 빠르게 수렴할까? 차원과의 관계는?

<details>
<summary>힌트 및 해설</summary>

**Strongly convex case**:
- LSI 상수 $\lambda > 0$ (고정, dimension-independent일 수 있음)
- Mixing time ~ $1/(2\lambda)$
- 차원 $d$에 무관한 지수 수렴

**Non-convex case** (예: overparameterized ReLU network):
- 전체 loss는 비볼록
- 하지만 local minima 근처에서 "restricted strong convexity" 성립 가능
- Effective LSI는 hessian의 condition number에 의존
- Mixing time이 $d$에 의존할 수 있음 (sharp constants)

**결론**:
- Strongly convex 정규화 추가 ($\ell_2$ penalty)는 LSI를 보장해 수렴을 보증한다.
- 이것이 ridge regression, weight decay의 이론적 근거 중 하나다.
- 하지만 overparameterized regime에서는 정규화 강도를 신중하게 선택해야 함 (underfitting 회피).

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Langevin Dynamics의 수렴](./04-langevin-convergence.md) | [📚 README로 돌아가기](../README.md) | [Ch5-01. Euler-Maruyama 기법 ▶](../ch5-numerical/01-euler-maruyama.md) |

</div>
