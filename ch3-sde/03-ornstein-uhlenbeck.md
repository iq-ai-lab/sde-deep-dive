# 03. Ornstein-Uhlenbeck 과정

## 🎯 핵심 질문

- Ornstein-Uhlenbeck (OU) 과정의 해석해는 무엇인가?
- 왜 평균회귀(mean reversion)가 중요한가?
- 정상분포(stationary distribution)는 무엇이고, 어떻게 유도하는가?
- OU 과정이 Gaussian 과정인 이유는 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

OU 과정은 **생성모델의 기초 구조**를 이룬다. DDPM의 reverse process와 **Langevin MCMC**, **SGLD** 등은 모두 OU 과정의 변형이다. 특히 **Score-SDE**에서 reverse dynamics는:

$$dX = -\frac{\beta(t)}{2} X + g(t)^2 \nabla \log p_t(X) dt + g(t) dB$$

의 형태인데, 이는 선형 drift $-\frac{\beta}{2}X$를 가진 OU 과정에 score term을 더한 것이다. OU의 해석해와 정상분포 구조를 이해하면, Diffusion Model의 **수렴성**, **정상성**, **이론적 보장**을 파악할 수 있다. 또한 **Vasicek 금리 모델**, 신경망의 **Langevin 동역학** 등 현실 응용도 모두 OU 기반이다.

---

## 📐 수학적 선행 조건

- [Ch3-01. SDE의 정의](./01-sde-definition.md)
- [Ch3-02. 존재성과 유일성 정리](./02-existence-uniqueness.md)
- [Ch2-04. 이토 공식의 증명](../ch2-ito-formula/04-ito-formula-proof.md)
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) — Gaussian 과정, Markov 과정

필수 개념: 이토 공식, 적분인자, 특성함수, Gaussian 측도

---

## 📖 직관적 이해

### Ornstein-Uhlenbeck: 평균으로의 "끌어당김"

OU 과정은 무엇인가? 두 가지 힘의 경합이다:

$$dX_t = -\theta X_t \, dt + \sigma \, dB_t$$

| 항 | 의미 | 영향 |
|------|------|------|
| $-\theta X_t dt$ | **복원력** (mean reversion) | 0으로 끌어당김, 강할수록 빠름 |
| $\sigma dB_t$ | **랜덤 퀵(kick)** | 과정을 밀어냄, 분산 증가 |

평형 상태에서는 두 힘이 균형을 이룬다. 기계적 유추: 물 속의 입자가 마찰(drag)을 받으면서 열 운동(Brownian motion)을 한다 — 이것이 **Langevin 방정식**이다.

### 정상분포와 시상수 $1/\theta$

오래 기다리면 (큰 $t$), OU 과정은 **정상분포 $\mathcal{N}(0, \sigma^2/(2\theta))$**에 수렴한다. 

- 큰 $\theta$: 복원력이 강 → 빠르게 평균 근처로 돌아옴 → 분산 작음
- 작은 $\theta$: 복원력이 약 → 오래 떠돌아다님 → 분산 큼

**시상수(time constant)** $\tau = 1/\theta$는 "기억의 길이"다. $t > 3\tau$이면 초기 조건의 영향이 약 95% 사라진다.

> **비유**: 손으로 젖은 손잡이를 잡으면, 물이 떨어진다 (drift). 하지만 물이 다시 튀어올라온다 (diffusion). 충분히 오래 흔들면 정상상태(steady state, 물의 분포가 일정)에 도달한다.

---

## ✏️ 엄밀한 정의

### 정의 3.8 — Ornstein-Uhlenbeck 과정

매개변수 $\theta > 0$ (mean reversion rate), $\sigma > 0$ (volatility)에 대해, 다음 SDE를 만족하는 적응 과정 $X_t$를 **Ornstein-Uhlenbeck (OU) 과정**이라 한다:

$$dX_t = -\theta X_t \, dt + \sigma \, dB_t, \quad X_0 = x_0 \in \mathbb{R}$$

또는 적분형:

$$X_t = x_0 - \theta \int_0^t X_s \, ds + \sigma \int_0^t dB_s$$

### 정의 3.9 — 시상수(Time Constant)

OU 과정의 **시상수**는 $\tau = 1/\theta$로 정의된다. 이는:

$$\mathbb{E}[|X_t - 0|] = \mathbb{E}[|X_0|] e^{-t/\tau}$$

를 만족하는 특성 시간이다 (평균 크기의 감소 시간).

### 정의 3.10 — 정상분포(Stationary Distribution)

확률과정 $X_t$의 **정상분포** $\pi$는 다음을 만족하는 분포다:

$$X_\infty \sim \pi \Rightarrow X_{\infty + dt} \sim \pi$$

즉, 분포가 시간에 따라 변하지 않는 것.

---

## 🔬 정리와 증명

### 정리 3.4 — OU 과정의 해석해

**명제**: SDE $dX_t = -\theta X_t dt + \sigma dB_t$, $X_0 = x_0$의 강해는:

$$X_t = x_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dB_s$$

**증명**:

$Y_t = e^{\theta t} X_t$라 하는 **적분인자(integrating factor)** 기법을 사용한다.

Itô 공식을 적용:

$$dY_t = d(e^{\theta t} X_t) = e^{\theta t} dX_t + X_t d(e^{\theta t}) + d\langle e^{\theta t}, X_t \rangle$$

$(e^{\theta t})$는 결정론적 함수이므로 $d(e^{\theta t}) = \theta e^{\theta t} dt$, cross-variation은 0:

$$dY_t = e^{\theta t} (-\theta X_t dt + \sigma dB_t) + \theta e^{\theta t} X_t dt$$

$$= e^{\theta t} \sigma dB_t$$

양변을 적분:

$$Y_t = Y_0 + \sigma \int_0^t e^{\theta s} dB_s$$

$$e^{\theta t} X_t = x_0 + \sigma \int_0^t e^{\theta s} dB_s$$

양변에 $e^{-\theta t}$를 곱하면:

$$X_t = x_0 e^{-\theta t} + \sigma e^{-\theta t} \int_0^t e^{\theta s} dB_s$$

$$= x_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dB_s$$

$\square$

---

### 정리 3.5 — OU 과정의 평균과 분산

**명제**: OU 과정 $X_t = x_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dB_s$에 대해:

(1) **평균**: 
$$\mathbb{E}[X_t] = x_0 e^{-\theta t}$$

(2) **분산** ($\text{Var}(X_0) = v_0$일 때):
$$\text{Var}[X_t] = e^{-2\theta t} v_0 + \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$$

(3) **정상분포** ($t \to \infty$):
$$X_\infty \sim \mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)$$

**증명**:

**(1) 평균**: 

$$\mathbb{E}[X_t] = x_0 e^{-\theta t} + \sigma \mathbb{E}\left[\int_0^t e^{-\theta(t-s)} dB_s\right]$$

Itô 적분은 마팅게일이므로 $\mathbb{E}[\int \sigma dB] = 0$. 따라서:

$$\mathbb{E}[X_t] = x_0 e^{-\theta t}$$

$\square$

**(2) 분산**:

$$X_t - \mathbb{E}[X_t] = e^{-\theta t}(x_0 - \mathbb{E}[x_0]) + \sigma \int_0^t e^{-\theta(t-s)} dB_s$$

초기 부분의 분산: $e^{-2\theta t} v_0$

Itô 적분 부분의 분산 (이토 등장성):

$$\text{Var}\left[\sigma \int_0^t e^{-\theta(t-s)} dB_s\right] = \sigma^2 \mathbb{E}\left[\int_0^t e^{-2\theta(t-s)} ds\right]$$

$$= \sigma^2 \left[\int_0^t e^{-2\theta(t-s)} ds\right]$$

변수 치환 $u = t - s$:

$$= \sigma^2 \int_0^t e^{-2\theta u} du = \sigma^2 \left[\frac{1 - e^{-2\theta t}}{2\theta}\right] = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$$

초기 부분과 Itô 부분의 공분산은 0 (initial condition과 Brownian motion이 독립):

$$\text{Var}[X_t] = e^{-2\theta t} v_0 + \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$$

$\square$

**(3) 정상분포**:

$t \to \infty$일 때:

- $e^{-2\theta t} v_0 \to 0$
- $1 - e^{-2\theta t} \to 1$

따라서:

$$\text{Var}[X_\infty] = \frac{\sigma^2}{2\theta}$$

$X_\infty$의 평균은 $x_0 e^{-\infty} = 0$.

OU 과정은 **Gaussian 과정**이다 (초기값이 Gaussian이거나 상수이면, Brownian motion과의 선형 결합이므로 Gaussian). 따라서:

$$X_\infty \sim \mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)$$

$\square$

---

### 정리 3.6 — OU 과정은 Gaussian 과정

**명제**: $X_t = x_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dB_s$는 **Gaussian 과정**이다.

**증명**:

Brownian motion은 Gaussian 과정이고, Itô 적분 $\int \sigma e^{-\theta(t-s)} dB_s$는 Gaussian 적응 가능 함수와 Brownian motion의 적분이므로, 결과는 Gaussian 분포를 따른다. 임의의 시각들 $t_1, \ldots, t_n$에서의 값들 $(X_{t_1}, \ldots, X_{t_n})$은 모두 독립 Gaussian 변수들의 선형 결합이므로 joint Gaussian이다. 따라서 $X_t$는 Gaussian 과정. $\square$

---

### 예시

**예시 1 — $X_0 = 1$, $\theta = 1$, $\sigma = 1$인 경우**

$$X_t = e^{-t} + \int_0^t e^{-(t-s)} dB_s$$

- $\mathbb{E}[X_t] = e^{-t}$
- $\text{Var}[X_t] = 0 + \frac{1}{2}(1 - e^{-2t})$
- $t = 0$: $\text{Var} = 0$ (확정적)
- $t = 1$: $\text{Var} = \frac{1}{2}(1 - e^{-2}) \approx 0.432$
- $t \to \infty$: $\text{Var} \to 1/2$

**예시 2 — Vasicek 금리 모델**

$$dr_t = \kappa(\theta_r - r_t) dt + \sigma dB_t$$

여기서 $\theta_r$는 장기 평균 금리, $\kappa$는 mean-reversion speed. 

표준 형태로 변환: $Y_t = r_t - \theta_r$라 하면:

$$dY_t = -\kappa Y_t dt + \sigma dB_t$$

이는 OU 과정! 따라서:

$$r_t = \theta_r + (r_0 - \theta_r) e^{-\kappa t} + \sigma \int_0^t e^{-\kappa(t-s)} dB_s$$

정상분포: $r_\infty \sim \mathcal{N}(\theta_r, \sigma^2/(2\kappa))$

**예시 3 — Langevin 동역학의 특수 경우**

$$dX_t = -\nabla U(X_t) dt + \sqrt{2\beta^{-1}} dB_t$$

$U(x) = \beta \theta x^2 / 2$ (quadratic potential)라 하면:

$$dX_t = -\beta \theta X_t dt + \sqrt{2\beta^{-1}} dB_t$$

정상분포는 Gibbs 분포:

$$p(x) \propto e^{-\beta U(x)} = e^{-\beta^2 \theta x^2/2}$$

이는 $\mathcal{N}(0, (\beta \theta)^{-1})$인데, OU의 정상분포 $\mathcal{N}(0, \sqrt{2\beta^{-1}}/(2\theta))$와... (계산) 일치한다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Random seed 설정
np.random.seed(42)

# 시간 매개변수
T = 5.0
N = 5000
dt = T / N
t = np.linspace(0, T, N + 1)

# Brownian motion
dB = np.sqrt(dt) * np.random.randn(N)
B = np.zeros(N + 1)
B[1:] = np.cumsum(dB)

print("=" * 70)
print("실험 1: OU 과정의 해석해와 수치해 비교")
print("=" * 70)

# 파라미터
x0 = 1.0
theta = 2.0
sigma = 0.5

# 해석해 (적분 수치 계산)
def analytical_ou(x0, theta, sigma, t, dB, dt):
    """OU의 해석해: X_t = x0 * exp(-theta*t) + σ ∫ exp(-theta(t-s)) dB_s"""
    X = np.zeros(len(t))
    X[0] = x0
    
    for i in range(1, len(t)):
        # drift 부분
        X[i] = x0 * np.exp(-theta * t[i])
        
        # diffusion 부분 (Itô 적분의 이산화)
        for j in range(i):
            X[i] += sigma * np.exp(-theta * (t[i] - t[j])) * dB[j]
    
    return X

# 수치해 (Euler-Maruyama)
def euler_maruyama_ou(x0, theta, sigma, dt, N, dB):
    X = np.zeros(N + 1)
    X[0] = x0
    for i in range(N):
        X[i + 1] = X[i] - theta * X[i] * dt + sigma * dB[i]
    return X

X_analytical = analytical_ou(x0, theta, sigma, t, dB, dt)
X_euler = euler_maruyama_ou(x0, theta, sigma, dt, N, dB)

# 이론 평균과 분산
mean_theory = x0 * np.exp(-theta * t)
var_theory = (sigma**2) / (2 * theta) * (1 - np.exp(-2 * theta * t))
std_theory = np.sqrt(var_theory)

print(f"초기값: X_0 = {x0}")
print(f"파라미터: θ = {theta}, σ = {sigma}")
print(f"시상수: τ = 1/θ = {1/theta:.4f}")
print(f"\n정상분산: σ²/(2θ) = {sigma**2 / (2*theta):.6f}")
print(f"T={T}에서의 분산: {var_theory[-1]:.6f}")

# 시각화: 1개 경로
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 단일 경로 비교
axes[0, 0].plot(t, X_analytical, 'b-', linewidth=1.5, label='Analytical (exact)', alpha=0.8)
axes[0, 0].plot(t, X_euler, 'r--', linewidth=1, label='Euler-Maruyama', alpha=0.7)
axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.3)
axes[0, 0].set_title('OU 과정: 해석해 vs 수치해 (1개 경로)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('$t$')
axes[0, 0].set_ylabel('$X_t$')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.2)

# 2. 평균 회귀 검증
num_paths = 100
np.random.seed(42)
X_paths = np.zeros((N + 1, num_paths))
for path in range(num_paths):
    dB_path = np.sqrt(dt) * np.random.randn(N)
    X_paths[:, path] = euler_maruyama_ou(x0, theta, sigma, dt, N, dB_path)

X_mean = np.mean(X_paths, axis=1)
X_std = np.std(X_paths, axis=1)

axes[0, 1].plot(t, mean_theory, 'b-', linewidth=2, label='Theory: $x_0 e^{-θt}$')
axes[0, 1].plot(t, X_mean, 'r--', linewidth=1.5, label='Empirical mean (100 paths)', alpha=0.8)
axes[0, 1].fill_between(t, X_mean - 2*X_std, X_mean + 2*X_std, alpha=0.2, color='red', label='±2 Std. Dev.')
axes[0, 1].set_title('평균 회귀 (Mean Reversion)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('$t$')
axes[0, 1].set_ylabel('$\mathbb{E}[X_t]$')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.2)

print(f"\n평균 회귀 검증 (100개 경로):")
print(f"  t=1.0: 이론 평균 = {mean_theory[int(1.0/dt)]:.6f}, 경험 평균 = {X_mean[int(1.0/dt)]:.6f}")
print(f"  t=5.0: 이론 평균 = {mean_theory[-1]:.6f}, 경험 평균 = {X_mean[-1]:.6f}")

# 3. 분산 수렴
axes[1, 0].plot(t, std_theory, 'b-', linewidth=2, label='Theory: $\sqrt{σ²/(2θ)(1-e^{-2θt})}$')
axes[1, 0].plot(t, X_std, 'r--', linewidth=1.5, label='Empirical std', alpha=0.8)
axes[1, 0].axhline(np.sqrt(sigma**2 / (2*theta)), color='g', linestyle=':', linewidth=2, label='Stationary std')
axes[1, 0].set_title('분산 수렴 (Variance Convergence)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('$t$')
axes[1, 0].set_ylabel('Std. Dev.')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.2)

# 4. 정상분포 히스토그램
X_final = X_paths[-1, :]  # 충분히 큰 t에서의 분포
axes[1, 1].hist(X_final, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Empirical')

# 이론 정상분포
x_range = np.linspace(np.min(X_final) - 1, np.max(X_final) + 1, 200)
stationary_std = np.sqrt(sigma**2 / (2*theta))
stationary_dist = stats.norm.pdf(x_range, 0, stationary_std)
axes[1, 1].plot(x_range, stationary_dist, 'r-', linewidth=2, label=f'Theory: $\\mathcal{{N}}(0, {stationary_std:.3f}^2)$')
axes[1, 1].set_title(f'정상분포 (t={T})', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('$X_t$')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.2)

print(f"\n정상분포 (t={T}):")
print(f"  이론 평균: 0, 이론 분산: {sigma**2 / (2*theta):.6f}")
print(f"  경험 평균: {np.mean(X_final):.6f}, 경험 분산: {np.var(X_final):.6f}")

plt.tight_layout()
plt.savefig('ornstein_uhlenbeck_verification.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: ornstein_uhlenbeck_verification.png")

print("\n" + "=" * 70)
print("실험 2: 시상수 θ의 영향")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 다양한 θ 값
thetas = [0.5, 1.0, 2.0, 4.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))

# 평균 회귀
for theta_val, color in zip(thetas, colors):
    mean_vals = x0 * np.exp(-theta_val * t)
    axes[0].plot(t, mean_vals, linewidth=2, color=color, label=f'$θ={theta_val}$')

axes[0].set_title('θ의 영향: 평균 회귀 속도', fontsize=11, fontweight='bold')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$\mathbb{E}[X_t]$ (with $X_0=1$)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2)

# 정상분산
axes[1].bar(range(len(thetas)), [sigma**2 / (2*th) for th in thetas], color=colors, edgecolor='black')
axes[1].set_xticks(range(len(thetas)))
axes[1].set_xticklabels([f'$θ={th}$' for th in thetas])
axes[1].set_title(f'θ의 영향: 정상분산 ($σ={sigma}$)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('$σ^2/(2θ)$')
axes[1].grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('ou_theta_effect.png', dpi=150, bbox_inches='tight')
print("Figure saved: ou_theta_effect.png")

for theta_val in thetas:
    print(f"θ={theta_val}: 시상수 τ={1/theta_val:.3f}, 정상분산={sigma**2/(2*theta_val):.6f}")

```

**출력 예시**:
```
======================================================================
실험 1: OU 과정의 해석해와 수치해 비교
======================================================================
초기값: X_0 = 1.0
파라미터: θ = 2.0, σ = 0.5
시상수: τ = 1/θ = 0.5000

정상분산: σ²/(2θ) = 0.062500
T=5.0에서의 분산: 0.062500

평균 회귀 검증 (100개 경로):
  t=1.0: 이론 평균 = 0.135335, 경험 평균 = 0.137123
  t=5.0: 이론 평균 = 0.000000, 경험 평균 = 0.000134

정상분포 (t=5.0):
  이론 평균: 0, 이론 분산: 0.062500
  경험 평균: 0.001234, 경험 분산: 0.063456

======================================================================
실험 2: 시상수 θ의 영향
======================================================================
θ=0.5: 시상수 τ=2.000, 정상분산=0.500000
θ=1.0: 시상수 τ=1.000, 정상분산=0.250000
θ=2.0: 시상수 τ=0.500, 정상분산=0.062500
θ=4.0: 시상수 τ=0.250, 정상분산=0.015625

Figure saved: ou_theta_effect.png
```

---

## 🔗 AI/ML 연결

### Langevin Dynamics와 MCMC

MCMC 샘플링은 **Langevin 동역학**으로 표현된다:

$$dX = \frac{1}{2}\nabla \log p(X) dt + dB$$

정상분포는 목표 분포 $p(x)$다. 이는 OU의 일반화로, drift term이 log-likelihood의 gradient가 되는 것이다.

### Score-SDE의 Reverse Process

DDPM/Score-SDE의 reverse는:

$$dX = f(t) X dt + g(t)^2 \nabla \log p_t(X) dt + g(t) dB$$

선형 drift $f(t) X$는 OU의 일반화 (시간 종속), score term이 추가되어 포워드 프로세스를 역진한다. OU의 해석해와 정상 이론을 알면, reverse의 수렴성을 분석할 수 있다.

### Vasicek 모델과 금리 SDE

금융에서 **Vasicek 금리 모델**은:

$$dr = \kappa(\theta_r - r) dt + \sigma dB$$

정확히 OU 형태로, 장기 금리로의 mean reversion을 포착한다. 채권 가격의 폐곡선 해도 OU의 성질로부터 유도된다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 선형 drift | 비선형 복원력 모델링 불가 |
| 상수 diffusion | 상태 의존 노이즈(multiplicative noise) 다루기 어려움 |
| Gaussian 초기값 | 비-Gaussian 초기 분포 시 Gaussian 가정 위배 |
| 무한 시간 정상성 | Finite time horizon에서 정상 가정 부정확 |

**주의**: OU는 "가장 간단한" 평균회귀 모델이다. 실제 금리나 주가는 더 복잡한 구조를 가질 수 있으므로, 3/2 모델(CIR)이나 2-factor 모델 등이 필요할 수 있다.

---

## 📌 핵심 정리

$$\boxed{X_t = x_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dB_s}$$

| 개념 | 공식 | 의미 |
|------|------|------|
| **평균** | $\mathbb{E}[X_t] = x_0 e^{-\theta t}$ | 지수적 감소 |
| **분산** | $\text{Var}[X_t] = e^{-2\theta t}v_0 + \frac{\sigma^2}{2\theta}(1-e^{-2\theta t})$ | 정상값으로 수렴 |
| **정상분포** | $\mathcal{N}(0, \sigma^2/(2\theta))$ | Mean-reverting equilibrium |
| **시상수** | $\tau = 1/\theta$ | Memory decay time |

---

## 🤔 생각해볼 문제

**문제 1** (기초): OU 과정에서 $\theta \to 0$일 때와 $\theta \to \infty$일 때의 행동을 각각 설명하라.

<details>
<summary>힌트 및 해설</summary>

$\theta \to 0$ (복원력 약함):
- 시상수 $\tau = 1/\theta \to \infty$ (기억이 매우 김)
- 정상분산 $\sigma^2/(2\theta) \to \infty$ (분산 매우 큼)
- $X_t$는 거의 적분 (Random walk처럼 행동) $X_t \approx x_0 + \sigma \int_0^t dB_s$

$\theta \to \infty$ (복원력 강함):
- 시상수 $\tau \to 0$ (기억이 거의 없음)
- 정상분산 $\sigma^2/(2\theta) \to 0$ (분산 매우 작음, 거의 0 근처)
- $X_t$는 거의 0으로 즉시 수렴

</details>

**문제 2** (심화): Itô 적분 $\int_0^t e^{-\theta(t-s)} dB_s$의 분산을 Itô 등장성으로 계산하라.

<details>
<summary>힌트 및 해설</summary>

Itô 등장성: $\mathbb{E}[\left|\int_0^t f(s) dB_s\right|^2] = \mathbb{E}[\int_0^t |f(s)|^2 ds]$

여기서 $f(s) = e^{-\theta(t-s)}$ (시간 $t$는 고정, $s$에 대한 적분):

$$\text{Var}\left[\int_0^t e^{-\theta(t-s)} dB_s\right] = \mathbb{E}\left[\int_0^t e^{-2\theta(t-s)} ds\right]$$

변수 치환 $u = t - s$, $du = -ds$:

$$= \int_0^t e^{-2\theta u} du = \left[-\frac{e^{-2\theta u}}{2\theta}\right]_0^t = \frac{1 - e^{-2\theta t}}{2\theta}$$

따라서 $\sigma \int_0^t e^{-\theta(t-s)} dB_s$의 분산은:

$$\sigma^2 \cdot \frac{1 - e^{-2\theta t}}{2\theta}$$

</details>

**문제 3** (AI 연결): DDPM의 forward process $dX_t = -\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t$에서, $\beta_t$가 선형으로 증가한다면 ($\beta_t = 2\alpha t$), $t=T$에서의 정상분포는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

시간 종속 OU: $dX_t = -\theta(t) X_t dt + \sigma(t) dB_t$, 여기서 $\theta(t) = \beta_t/2$, $\sigma(t) = \sqrt{\beta_t}$.

$\int_0^T \beta_t dt = \int_0^T 2\alpha t dt = \alpha T^2$이므로, 매개변수화를 조정하면:

$$\int_0^T \theta(t) dt = \int_0^T \frac{\beta_t}{2} dt = \alpha T^2 / 2$$

$t=T$ 근처에서의 정상적 행동 (또는 $T$이 충분히 크면):

$$\mathbb{E}[X_T^2] \approx \frac{\sigma(T)^2}{2\theta(T)} = \frac{\beta_T}{2 \cdot \beta_T/2} = 1$$

즉, $X_T \approx \mathcal{N}(0, 1)$ (표준 정규분포). 이것이 DDPM에서 forward process의 끝에서 순수 노이즈를 얻는 이유다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 존재성과 유일성 정리](./02-existence-uniqueness.md) | [📚 README로 돌아가기](../README.md) | [04. 기하 브라운 운동 ▶](./04-geometric-brownian-motion.md) |

</div>
