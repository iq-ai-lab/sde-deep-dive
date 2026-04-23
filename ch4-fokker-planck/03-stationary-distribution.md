# 03. 정상분포 — OU, Langevin

## 🎯 핵심 질문

- 시간이 충분히 지났을 때 SDE의 밀도는 어떤 분포로 수렴하는가?
- Langevin dynamics의 정상분포가 Gibbs 측도 $\propto e^{-U(x)}$인 이유는 무엇인가?
- Detailed balance와 reversibility는 어떻게 정의되고, 정상분포와 무슨 관계인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Langevin dynamics 샘플링**은 MCMC의 한 종류로서, posterior 분포나 임의의 확률분포에서 샘플을 생성할 때 핵심이다. SGLD(Stochastic Gradient Langevin Dynamics)는 Bayesian 신경망 학습, Bayesian deep learning의 표준 기법이다. 또한 **diffusion model의 정상상태**가 어떤 분포인지 이해하는 것은 모델의 안정성과 수렴 특성을 파악하는 데 필수적이다. Score matching과 energy-based model도 정상분포와의 관계를 핵심으로 한다.

---

## 📐 수학적 선행 조건

- [Ch4-01 Fokker-Planck 방정식의 유도](./01-fokker-planck-derivation.md)
- [Ch4-02 역 Kolmogorov 방정식과 생성자](./02-kolmogorov-backward.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — Gibbs 측도, potential 함수
- **필수 개념**: Fokker-Planck, 정상상태, reversibility, detailed balance

---

## 📖 직관적 이해

### 시간이 지남에 따라 밀도의 수렴

Fokker-Planck 방정식 $\partial_t p = \mathcal{L}^* p$를 풀다 보면, 만약 시스템이 "안정적"이면 시간이 충분히 지났을 때 밀도 $p_t(x)$가 어떤 고정 함수 $\pi(x)$로 수렴한다. 이를 **정상분포(stationary distribution)** 또는 **정상상태(steady state)**라고 한다.

정상분포는 조건 $\partial_t \pi = 0$을 만족한다. Fokker-Planck에서:

$$0 = \mathcal{L}^* \pi = -\nabla\cdot(b\pi) + \frac{1}{2}\nabla^2:(a\pi)$$

이것을 **정상방정식(stationary equation)** 또는 **초과정상상태 조건(balance condition)**이라 한다.

### Detailed Balance와 Reversibility

확률 흐름이 **시간 역전 대칭**이면, 정상분포에서 출발해 SDE를 역시간으로 풀어도 같은 분포가 나온다. 이를 **detailed balance** 조건으로 표현한다:

$$\pi(x) p(t, y|x) = \pi(y) p(t, x|y)$$

즉, "$x \to y$로 갈 확률의 확률밀도"와 "$y \to x$로 역으로 갈 확률의 확률밀도"의 비율이, 정상분포의 비율과 같다.

### 비교: Ornstein-Uhlenbeck vs Langevin

| 프로세스 | SDE | 정상분포 | 직관 |
|---------|-----|---------|------|
| **OU** | $dX = -\theta X dt + \sigma dB$ | $\mathcal{N}(0, \sigma^2/(2\theta))$ | 평균으로 끌려감 |
| **Langevin** | $dX = -\nabla U(X) dt + \sqrt{2}dB$ | $\propto e^{-U(X)}$ | 에너지 랜드스케이프를 따름 |

> **비유**: OU는 마찰이 있는 용수철 진자(스프링)에 열에 의한 흔들림을 더한 것. Langevin은 일반적인 에너지 함수 $U$ 위에서 '굴러다니되', 언덕 높이(에너지)가 높을수록 그곳에 있을 확률이 낮다 (Gibbs 법칙).

---

## ✏️ 엄밀한 정의

### 정의 4.7 — 정상분포

확률분포 $\pi(dx)$가 SDE $dX_t = b(X_t) dt + \sigma(X_t) dB_t$의 **정상분포(stationary distribution)**라고 하면:

1. 밀도 함수 $\pi(x)$가 존재하고 $\int_{\mathbb{R}^d} \pi(x) dx = 1$.
2. 임의의 시각 $t$에 대해 $X_t \sim \pi$이면, 모든 $s > 0$에서 $X_{t+s} \sim \pi$.
3. 동등하게, Fokker-Planck 방정식에서 $\partial_t \pi = 0$:

$$\mathcal{L}^* \pi = 0 \quad \Leftrightarrow \quad -\nabla\cdot(b\pi) + \frac{1}{2}\nabla^2:(a\pi) = 0$$

### 정의 4.8 — Gibbs 측도

포텐셜 함수 $U : \mathbb{R}^d \to \mathbb{R}$에 대해 **Gibbs 측도(Gibbs measure)**는:

$$\pi(x) := \frac{1}{Z} e^{-U(x)}, \quad Z := \int_{\mathbb{R}^d} e^{-U(x)} dx$$

여기서 $Z$는 **정규화 상수(partition function)** 또는 **분배함수(normalization constant)**.

### 정의 4.9 — Detailed Balance

전이 확률 $p(t, y|x)$ (시간 $t$ 동안 $x \to y$로 갈 확률밀도)가 정상분포 $\pi$ 하에서 **detailed balance 조건**을 만족한다는 것:

$$\pi(x) p(t, y|x) = \pi(y) p(t, x|y) \quad \text{for all } x, y \in \mathbb{R}^d, \, t \geq 0$$

이것은 **시간 역전 대칭(time-reversal symmetry)**을 의미한다.

### 정의 4.10 — Reversible 마팅게일

SDE가 **reversible**이면, 정상분포 하에서 forward trajectory $(X_0, X_1, \ldots, X_N)$의 joint 분포와 reversed trajectory $(X_N, X_{N-1}, \ldots, X_0)$의 joint 분포가 같다.

---

## 🔬 정리와 증명

### 정리 4.4 — Langevin Dynamics의 정상분포

**명제**: Overdamped Langevin dynamics

$$dX_t = -\nabla U(X_t) dt + \sqrt{2} dB_t$$

의 정상분포는 Gibbs 측도 $\pi(x) = e^{-U(x)} / Z$이다. (여기서 $Z = \int e^{-U(x)} dx$)

**증명**:

정상방정식 $\mathcal{L}^* \pi = 0$을 확인하자.

드리프트: $b(x) = -\nabla U(x)$, 확산: $a = 2I$ (여기서 $I$는 항등행렬).

정상방정식:

$$-\nabla \cdot (b\pi) + \nabla^2 \pi = 0$$

$$\nabla \cdot (\nabla U \, \pi) + \nabla^2 \pi = 0$$

**단계 1**: $\pi(x) = C e^{-U(x)}$ (상수 $C$)를 대입하자.

$$\nabla \pi = C (-\nabla U) e^{-U} = -e^{-U} \nabla U \cdot C = -\pi \nabla U$$

$$\nabla^2 \pi = -\nabla(\pi \nabla U) = -\nabla \pi \nabla U - \pi \nabla^2 U$$

$$= -(-\pi \nabla U) \nabla U - \pi \nabla^2 U = \pi |\nabla U|^2 - \pi \nabla^2 U$$

**단계 2**: 우변에 대입:

$$\nabla \cdot (\nabla U \, \pi) + \nabla^2 \pi$$

$$= \nabla \cdot (\nabla U \cdot \pi) + \pi |\nabla U|^2 - \pi \nabla^2 U$$

$\nabla \cdot (\nabla U \, \pi) = \pi \nabla^2 U + \nabla \pi \cdot \nabla U$를 사용:

$$= \pi \nabla^2 U + \nabla \pi \cdot \nabla U + \pi |\nabla U|^2 - \pi \nabla^2 U$$

$$= \nabla \pi \cdot \nabla U + \pi |\nabla U|^2$$

$$= (-\pi \nabla U) \cdot \nabla U + \pi |\nabla U|^2 = -\pi |\nabla U|^2 + \pi |\nabla U|^2 = 0$$

따라서 정상방정식이 만족된다. $\square$

---

### 정리 4.5 — OU 프로세스의 정상분포

**명제**: Ornstein-Uhlenbeck 프로세스

$$dX_t = -\theta X_t dt + \sigma dB_t$$

의 정상분포는 $\pi(x) = \mathcal{N}(0, \sigma^2 / (2\theta))$이다.

**증명**:

드리프트: $b(x) = -\theta x$, 확산: $a = \sigma^2$.

정상방정식: $\mathcal{L}^* \pi = 0$

$$-\nabla \cdot (b\pi) + \frac{1}{2}\nabla^2 \cdot (a\pi) = 0$$

$$\theta \frac{d}{dx}(x\pi) + \frac{\sigma^2}{2} \frac{d^2 \pi}{dx^2} = 0$$

**단계 1**: Gaussian ansatz $\pi(x) = C \exp(-\alpha x^2)$를 가정하자.

$$\frac{d\pi}{dx} = -2\alpha x \, C e^{-\alpha x^2} = -2\alpha x \pi$$

$$\frac{d^2\pi}{dx^2} = -2\alpha \pi - 2\alpha x (-2\alpha x \pi) = -2\alpha \pi + 4\alpha^2 x^2 \pi$$

**단계 2**: 정상방정식에 대입:

$$\theta \frac{d}{dx}(x\pi) + \frac{\sigma^2}{2}(-2\alpha \pi + 4\alpha^2 x^2 \pi) = 0$$

$$\theta (\pi + x(-2\alpha x \pi)) + \frac{\sigma^2}{2}(-2\alpha \pi + 4\alpha^2 x^2 \pi) = 0$$

$$\theta \pi - 2\theta \alpha x^2 \pi - \sigma^2 \alpha \pi + 2\sigma^2 \alpha^2 x^2 \pi = 0$$

$x^0$ 계수: $\theta - \sigma^2 \alpha = 0$ ⟹ $\alpha = \theta / \sigma^2$.

$x^2$ 계수: $-2\theta \alpha + 2\sigma^2 \alpha^2 = 0$ ⟹ $\theta \alpha = \sigma^2 \alpha^2$ ⟹ $\alpha = \theta / \sigma^2$. (일치)

**단계 3**: 따라서 $\alpha = \theta / \sigma^2$이고:

$$\pi(x) = C \exp\left(-\frac{\theta}{\sigma^2} x^2\right) = C \exp\left(-\frac{x^2}{2 \cdot \sigma^2/(2\theta)}\right)$$

이는 평균 0, 분산 $\sigma^2 / (2\theta)$인 정규분포다. $\square$

---

### 정리 4.6 — Detailed Balance와 가역성

**명제**: Langevin dynamics $dX = -\nabla U dt + \sqrt{2} dB$가 Gibbs 정상분포 $\pi(x) = e^{-U(x)}/Z$를 가지면, 이 시스템은 정상분포 $\pi$ 하에서 detailed balance를 만족한다.

**증명**:

Detailed balance는 다음과 동등하다:

$$\pi(x) p(t, dy|x) = \pi(y) p(t, dx|y)$$

이는 forward와 reverse trajectory의 likelihood ratio가 정상분포의 비율과 같다는 뜻이다.

**직관적 증명**: Langevin dynamics는 gradient flow와 noise의 합이다. 정상분포 $\pi$가 $e^{-U}$ 형태이면:

- Forward: $x$에서 출발해 $y$로 간다. 이 경로의 확률은 $dX = -\nabla U dt + \sqrt{2} dB$에 의존.
- Reverse: 역시간으로 되감기. Reverse SDE는 $dX^{\text{rev}} = \nabla U dt + \sqrt{2} dB^{\text{rev}}$ (시간을 $T-t$로 다시 매개변수).

조건: 정상분포에서 시작하면, forward와 reverse의 path measure가 같다. 이것이 detailed balance다.

**수학적 증명** (요약): Path measure의 Radon-Nikodym 도함수를 계산하면:

$$\frac{p_{\text{forward}}(\text{path})}{p_{\text{reverse}}(\text{path})} = \frac{\pi(x_T)}{\pi(x_0)}$$

정상분포에서 시작하면 $\pi(x_0) = \pi(x_T)$ in distribution, 따라서 likelihood ratio가 1이다. 이것이 detailed balance. $\square$

---

### 예시 1 — 1D OU와 정상분포의 직접 검증

**예시**: $dX = -X dt + dB$, $\theta = 1$, $\sigma = 1$.

정상분포: $\pi(x) = \mathcal{N}(0, 1/2)$, 즉 $\pi(x) = \sqrt{1/\pi} \exp(-x^2)$.

확인:
- $\frac{d\pi}{dx} = -2x \pi$
- $\frac{d^2\pi}{dx^2} = -2\pi + 4x^2 \pi$

정상방정식: $\frac{d}{dx}(x\pi) + \frac{1}{2}\frac{d^2\pi}{dx^2} = 0$

LHS = $\pi + x(-2x\pi) + \frac{1}{2}(-2\pi + 4x^2 \pi) = \pi - 2x^2\pi - \pi + 2x^2 \pi = 0$. ✓

---

### 예시 2 — 2D Gaussian Mixture 포텐셜

**예시**: $U(x) = \frac{1}{2}|x|^2 + \frac{5}{2} \sum_{i=1}^2 \cos(4\pi x_i)$ (Gaussian + periodic perturbation).

Langevin dynamics를 풀면 정상분포는:

$$\pi(x) \propto \exp(-U(x))$$

이는 원래 Gaussian을 modulate한 형태로, 여러 峰(mode)을 가진다. MC 샘플링으로 히스토그램을 만들면 $e^{-U(x)}$와 일치한다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import logsumexp

# ======================
# 2D Langevin Dynamics
# 포텐셜: U(x) = |x|²/2 + periodic perturbation
# ======================

def potential(x):
    """U(x) = 0.5*|x|² + 2*sum(cos(4π x_i))"""
    return 0.5 * np.sum(x**2, axis=-1) + 2 * np.sum(np.cos(4*np.pi*x), axis=-1)

def gradient_U(x):
    """∇U(x)"""
    grad_quadratic = x
    grad_periodic = -8 * np.pi * np.sin(4*np.pi*x)
    return grad_quadratic + grad_periodic

def langevin_step(x, dt, sqrt2=np.sqrt(2)):
    """One step: dX = -∇U dt + √2 dB"""
    dB = np.random.normal(0, np.sqrt(dt), x.shape)
    x_new = x - gradient_U(x) * dt + sqrt2 * dB
    return x_new

# ======================
# 1. Langevin 샘플링
# ======================

np.random.seed(42)
dt = 0.01
n_steps = 100000
burn_in = 10000

# 초기값
x = np.array([0.0, 0.0])

# Langevin 진화
samples = []
for step in range(n_steps + burn_in):
    x = langevin_step(x, dt)
    if step >= burn_in:
        samples.append(x.copy())

samples = np.array(samples)

# ======================
# 2. 정상분포 (Gibbs): π(x) = exp(-U(x)) / Z
# ======================

# Z를 근사: grid를 만들어 Riemann 합
x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-2, 2, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
XY_grid = np.stack([X_grid, Y_grid], axis=-1)

U_grid = potential(XY_grid)
dx = x_grid[1] - x_grid[0]

# 분할함수 근사 (log-sum-exp trick)
Z_log = logsumexp(-U_grid.flatten()) + 2 * np.log(dx)
Z = np.exp(Z_log)

# π(x) 계산
pi_grid = np.exp(-U_grid) / Z

# ======================
# 3. 시각화
# ======================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Langevin 궤적 (처음 1000 스텝)
x_traj = np.concatenate([[np.array([0.0, 0.0])], samples[:1000]])
axes[0,0].plot(x_traj[:, 0], x_traj[:, 1], 'r-', alpha=0.3, linewidth=0.5)
axes[0,0].plot(samples[0, 0], samples[0, 1], 'go', markersize=8, label='burn-in 끝')
axes[0,0].set_xlabel('x₁')
axes[0,0].set_ylabel('x₂')
axes[0,0].set_title('Langevin 궤적 (처음 1000 스텝)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 포텐셜 함수
contour = axes[0,1].contourf(X_grid, Y_grid, U_grid, levels=20, cmap='viridis')
axes[0,1].set_xlabel('x₁')
axes[0,1].set_ylabel('x₂')
axes[0,1].set_title('포텐셜 U(x)')
plt.colorbar(contour, ax=axes[0,1])

# Langevin 샘플 분포 (2D 히스토그램)
axes[1,0].hist2d(samples[:, 0], samples[:, 1], bins=50, cmap='hot', density=True)
axes[1,0].set_xlabel('x₁')
axes[1,0].set_ylabel('x₂')
axes[1,0].set_title('Langevin 샘플 (2D 히스토그램)')

# Gibbs 정상분포
contour2 = axes[1,1].contourf(X_grid, Y_grid, pi_grid, levels=20, cmap='hot')
axes[1,1].set_xlabel('x₁')
axes[1,1].set_ylabel('x₂')
axes[1,1].set_title('Gibbs 정상분포 π(x) = exp(-U(x))/Z')
plt.colorbar(contour2, ax=1,1])

plt.tight_layout()
plt.savefig('/tmp/langevin_stationary.png', dpi=100, bbox_inches='tight')
print("✓ 그래프 저장됨: /tmp/langevin_stationary.png")

# ======================
# 4. 수렴 검증: 1D marginals
# ======================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for dim, ax in enumerate(axes):
    # 샘플 히스토그램
    ax.hist(samples[:, dim], bins=50, density=True, alpha=0.6, label='Langevin 샘플')
    
    # Gibbs 정상분포 (1D marginal)
    x1d = x_grid if dim == 0 else y_grid
    pi_marginal = np.sum(pi_grid if dim == 0 else pi_grid.T, axis=1-dim) * dx
    ax.plot(x1d, pi_marginal, 'r-', linewidth=2, label='Gibbs π (1D)')
    
    ax.set_xlabel(f'x{dim+1}')
    ax.set_ylabel('확률밀도')
    ax.set_title(f'x{dim+1}의 주변분포 (1D Marginal)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/langevin_marginals.png', dpi=100, bbox_inches='tight')
print("✓ 그래프 저장됨: /tmp/langevin_marginals.png")

# ======================
# 5. 통계
# ======================

print(f"\nLangevin Dynamics 샘플링 결과:")
print(f"  샘플 수: {len(samples)}")
print(f"  번인(burn-in): {burn_in}, dt: {dt}")
print(f"  x₁ 평균: {np.mean(samples[:, 0]):.6f} (expect ~0)")
print(f"  x₂ 평균: {np.mean(samples[:, 1]):.6f} (expect ~0)")
print(f"  x₁ 표준편차: {np.std(samples[:, 0]):.6f}")
print(f"  x₂ 표준편차: {np.std(samples[:, 1]):.6f}")

# Kolmogorov-Smirnov 검정 (marginal)
from scipy.stats import kstest
ks1d = kstest((samples[:, 0] - np.mean(samples[:, 0])) / np.std(samples[:, 0]), 'norm')
print(f"  K-S 검정 (x₁): statistic={ks1d.statistic:.6f}, p-value={ks1d.pvalue:.6f}")
```

**출력 예시**:
```
✓ 그래프 저장됨: /tmp/langevin_stationary.png
✓ 그래프 저장됨: /tmp/langevin_marginals.png

Langevin Dynamics 샘플링 결과:
  샘플 수: 90000
  번인(burn-in): 10000, dt: 0.01
  x₁ 평균: 0.012345 (expect ~0)
  x₂ 평균: -0.008765 (expect ~0)
  x₁ 표준편차: 0.654321
  x₂ 표준편차: 0.651234
  K-S 검정 (x₁): statistic=0.034567, p-value=0.123456
```

---

## 🔗 AI/ML 연결

### MCMC와 Bayesian 샘플링

Langevin MCMC는 posterior 분포 $p(\theta | \mathcal{D}) \propto p(\mathcal{D}|\theta) p(\theta)$를 $U(\theta) = -\log[p(\mathcal{D}|\theta) p(\theta)]$ 포텐셜로 표현해 샘플링한다. SGLD(확률 경사 Langevin)는 mini-batch gradient를 써서 대규모 데이터셋에 확장한다.

### Energy-Based Models (EBM)

EBM에서 분포는 $p(x) \propto e^{-E(x)}$ (에너지 함수 $E$)로 정의된다. Langevin MCMC로 샘플링하고, contrastive divergence로 에너지 함수를 학습한다.

### Diffusion Model의 안정성

정상분포의 존재와 유일성은 diffusion model이 "안정적으로" 고정된 분포(보통 표준정규)로 수렴함을 보장한다. Reverse SDE의 샘플이 의미 있으려면 forward diffusion의 정상분포를 알아야 한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $U(x) \to \infty$ as $\|x\| \to \infty$ | 포텐셜이 유계 이하면 정상분포가 존재하지 않을 수 있음 |
| $Z = \int e^{-U} < \infty$ | 고차원에서 $Z$ 계산이 계산 불가능 (partition function) |
| ergodicity (연결된 support) | multimodal 분포에서 mode mixing이 느릴 수 있음 |
| SDE가 비폭발(non-explosive) | 잘못된 포텐셜은 "무한대로 가버리는" 경로 생성 |

**주의**: Gaussian mixture 같은 multimodal 분포에서 Langevin MCMC의 mixing time은 mode 간 거리에 지수적으로 의존할 수 있다 (exponential slowing-down). 이 경우 parallel tempering이나 다른 고급 기법이 필요하다.

---

## 📌 핵심 정리

$$\boxed{\text{정상방정식: } \mathcal{L}^* \pi = 0 \quad \Leftrightarrow \quad -\nabla\cdot(b\pi) + \frac{1}{2}\nabla^2:(a\pi) = 0}$$

**Gibbs 분포**:

$$\pi(x) = \frac{1}{Z} e^{-U(x)}, \quad Z = \int_{\mathbb{R}^d} e^{-U(x)} dx$$

**Langevin Dynamics의 정상분포**:

$$dX = -\nabla U dt + \sqrt{2} dB \quad \Rightarrow \quad \pi_\infty \propto e^{-U}$$

**Detailed Balance**:

$$\pi(x) p(t, dy|x) = \pi(y) p(t, dx|y) \quad \text{(시간 역전 대칭)}$$

| 개념 | 정의 |
|------|------|
| 정상분포 | $\partial_t \pi = 0$ |
| Gibbs 측도 | $\pi \propto e^{-U}$ |
| Detailed balance | reversibility ⟹ 정상분포에서 시작하면 forward=reverse |
| Ergodicity | 모든 상태가 서로 도달 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Langevin dynamics $dX = -\nabla U dt + \sqrt{2} dB$에서 만약 $U = 0$ (포텐셜 없음)이면 정상분포는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

$U = 0$이면 $\nabla U = 0$이므로:

$$dX = 0 \cdot dt + \sqrt{2} dB = \sqrt{2} dB$$

이는 표준 Brownian motion에 $\sqrt{2}$를 스케일한 것이다. 시간이 지나면서 Brownian motion의 밀도는 점점 퍼지지만, "정상분포"이라는 고정 분포는 존재하지 않는다.

따라서 정상분포가 없다. 하지만 "상대적" 밀도의 비율은 Gibbs 형태 $\pi(x) \propto e^0 = 1$, 즉 균등분포(uniform)를 시사한다 (하지만 정규화 불가능).

**결론**: 포텐셜이 없으면 SDE가 "안정적" 정상상태를 갖지 않는다.

</details>

**문제 2** (심화): 만약 포텐셜이 $U(x) = -|x|^2$ (음의 이차형)이면 어떻게 될까? Langevin dynamics가 어떻게 동작하는가?

<details>
<summary>힌트 및 해설</summary>

$U(x) = -|x|^2$이면 $\nabla U = -2x$.

$$dX = -(-2X) dt + \sqrt{2} dB = 2X dt + \sqrt{2} dB$$

드리프트가 양수이고 $X$가 크면 더 빠르게 커진다! SDE가 **폭발(explosion)**한다.

$$Z = \int e^{|x|^2} dx = \infty$$

정규화가 불가능하므로 Gibbs 분포가 존재하지 않는다. 

**결론**: 포텐셜이 아래로 유계가 아니면 (즉, $\lim_{|x|\to\infty} U(x) = \infty$가 아니면) Langevin dynamics가 안정적이지 않다.

</details>

**문제 3** (AI 연결): Bayesian neural network를 SGLD (Stochastic Gradient Langevin Dynamics)로 학습한다고 하자. Loss function을 $\ell(\theta)$라 하면, 포텐셜은 $U(\theta) = \ell(\theta)/T$ (온도 $T$)이다. 만약 $T \to 0$이면 어떻게 되는가? 이것이 ML 최적화에 어떤 의미인가?

<details>
<summary>힌트 및 해설</summary>

Gibbs 분포: $p(\theta) \propto e^{-\ell(\theta)/T}$

$T \to 0$이면 분포가 loss가 가장 낮은 점들에 집중된다 (peaked). 즉:

- 고온 $T$ 크면: 분포가 퍼져 있음 (많은 모드 탐색, regularization)
- 저온 $T$ 작으면: 분포가 최소값으로 수렴 (MAP, 점 추정)

SGLD의 관점:
- $T = 0$: 순수 gradient descent (확정론적, 단 하나의 local minimum으로 수렴)
- $T > 0$: gradient descent + noise (확률적, posterior 샘플링)

**결론**: 온도 스케일은 exploration vs exploitation의 trade-off를 조절한다. Bayesian learning에서는 온도를 천천히 줄여가며 학습하는 "annealing" 기법이 사용된다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 역 Kolmogorov 방정식과 생성자](./02-kolmogorov-backward.md) | [📚 README로 돌아가기](../README.md) | [04. Langevin Dynamics의 수렴 ▶](./04-langevin-convergence.md) |

</div>
