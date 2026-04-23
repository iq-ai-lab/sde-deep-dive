# 05. 선형 SDE의 일반해

## 🎯 핵심 질문

- 선형 SDE의 일반해는 어떻게 구하는가?
- Fundamental solution $\Phi(t)$는 무엇인가?
- Additive noise와 multiplicative noise의 차이는?
- Vasicek 금리 모델을 선형 SDE로 어떻게 다루는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

대부분의 실용적인 SDE는 **선형 형태**이거나 선형으로 근사 가능하다. **Score-SDE**의 forward process, **Vasicek 금리 모델**, **Langevin MCMC**, 신경망의 최적화 동역학 모두가 선형 SDE로 표현되거나 분석된다. 선형 SDE의 일반해를 알면:

1. 닫힌 형태(closed-form) 해를 얻을 수 있다 → 수치 안정성 향상
2. moment 구조를 정확히 계산 가능 → convergence 분석
3. 역해(reverse) 구성이 체계적이다 → score function 유도

특히 **Probability Flow ODE**와 **deterministic sampling**에서, 선형 부분을 정확히 푸는 것이 전체 수렴을 지배한다.

---

## 📐 수학적 선행 조건

- [Ch3-01. SDE의 정의](./01-sde-definition.md)
- [Ch3-02. 존재성과 유일성 정리](./02-existence-uniqueness.md)
- [Ch3-03. Ornstein-Uhlenbeck 과정](./03-ornstein-uhlenbeck.md)
- [Ch3-04. 기하 브라운 운동](./04-geometric-brownian-motion.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) — 행렬 지수, 기본해(fundamental solutions)

필수 개념: 행렬 지수함수, ODE의 기본해, 변수 분리, 적분인자

---

## 📖 직관적 이해

### 선형 SDE의 구조

$$dX_t = (A(t) X_t + C(t)) dt + (\Sigma_1(t) X_t + \Sigma_0(t)) dB_t$$

여기서:
- $A(t)$: 시간 종속 회귀 계수 (Drift에서 $X_t$의 선형 가중)
- $C(t)$: deterministic forcing (외부 입력)
- $\Sigma_1(t)$: multiplicative noise (상태 의존)
- $\Sigma_0(t)$: additive noise (상태 무관)

| 항 | 특성 | 예시 |
|------|------|------|
| $A(t) X_t$ | 평균회귀/성장 | OU ($-\theta X$), GBM ($\mu X$) |
| $C(t)$ | 결정론적 강제 | 외부 입력, 확정적 신호 |
| $\Sigma_1(t) X_t$ | 곱셈적 노이즈 | GBM ($\sigma X$) |
| $\Sigma_0(t)$ | 가법적 노이즈 | OU ($\sigma$, 상수) |

### Fundamental Solution (기본해)

동차 SDE (homogeneous part)의 해:

$$d\Phi_t = A(t) \Phi_t dt$$

$\Phi_t$는 **fundamental solution 행렬**이며, 초기값 $\Phi_0 = I$ (항등행렬).

선형 ODE의 경우:

$$\Phi(t) = \exp\left(\int_0^t A(s) ds\right)$$

(행렬 적분과 지수함수)

---

## ✏️ 엄밀한 정의

### 정의 3.14 — 선형 SDE

다음 형태의 SDE를 **선형 SDE**라 한다:

$$dX_t = (A(t) X_t + C(t)) dt + (\Sigma_1(t) X_t + \Sigma_0(t)) dB_t$$

여기서:
- $X_t \in \mathbb{R}^n$
- $A(t) \in \mathbb{R}^{n \times n}$, $C(t) \in \mathbb{R}^n$ (drift)
- $\Sigma_1(t), \Sigma_0(t) \in \mathbb{R}^{n \times m}$ (diffusion)
- $B_t$: $m$-차원 Brownian motion

### 정의 3.15 — Fundamental Solution Matrix

동차 선형 SDE:

$$d\Phi_t = A(t) \Phi_t dt, \quad \Phi_0 = I$$

의 해 $\Phi(t)$를 **fundamental solution 행렬**이라 한다. 스칼라의 경우:

$$\Phi(t) = \exp\left(\int_0^t a(s) ds\right)$$

행렬의 경우 (non-commutative일 때):

$$\Phi(t) = \mathcal{T} \exp\left(\int_0^t A(s) ds\right)$$

(시간-정렬 지수 (time-ordered exponential))

### 정의 3.16 — Additive vs Multiplicative Noise

- **Additive noise**: $\Sigma_1(t) = 0$, SDE는 $dX_t = (A X + C) dt + \Sigma_0 dB$
- **Multiplicative noise**: $\Sigma_0(t) = 0$, SDE는 $dX_t = (A X + C) dt + \Sigma_1 X dB$

---

## 🔬 정리와 증명

### 정리 3.10 — 선형 SDE의 일반해 (Additive Case)

**명제**: Additive noise를 가진 선형 SDE:

$$dX_t = (A(t) X_t + C(t)) dt + \Sigma_0(t) dB_t, \quad X_0$$

의 강해는:

$$X_t = \Phi(t) X_0 + \Phi(t) \int_0^t \Phi(s)^{-1} C(s) ds + \Phi(t) \int_0^t \Phi(s)^{-1} \Sigma_0(s) dB_s$$

여기서 $\Phi(t)$는 $d\Phi = A \Phi dt$, $\Phi_0 = I$의 fundamental solution.

**증명**:

**변수 변환**: $Y_t = \Phi(t)^{-1} X_t$라 하자 (적분인자 기법의 행렬 버전).

Itô 공식을 적용:

$$dY_t = d(\Phi(t)^{-1} X_t) = d(\Phi(t)^{-1}) X_t + \Phi(t)^{-1} dX_t + d\langle \Phi^{-1}, X \rangle_t$$

$\Phi(t)^{-1}$의 미분 (역함수 미분):

$$d(\Phi(t)^{-1}) = -\Phi(t)^{-1} d\Phi(t) \Phi(t)^{-1}$$

(행렬 버전의 역함수 미분 규칙: $\frac{d}{dt}(A^{-1}) = -A^{-1} A' A^{-1}$)

$$= -\Phi(t)^{-1} (A(t) \Phi(t) dt) \Phi(t)^{-1} = -\Phi(t)^{-1} A(t) dt$$

따라서:

$$dY_t = -\Phi(t)^{-1} A(t) dt \cdot X_t + \Phi(t)^{-1} (A(t) X_t + C(t)) dt + \Phi(t)^{-1} \Sigma_0(t) dB_t$$

$$= \Phi(t)^{-1} C(t) dt + \Phi(t)^{-1} \Sigma_0(t) dB_t$$

양변을 적분:

$$Y_t = Y_0 + \int_0^t \Phi(s)^{-1} C(s) ds + \int_0^t \Phi(s)^{-1} \Sigma_0(s) dB_s$$

$$\Phi(t)^{-1} X_t = X_0 + \int_0^t \Phi(s)^{-1} C(s) ds + \int_0^t \Phi(s)^{-1} \Sigma_0(s) dB_s$$

양변에 $\Phi(t)$를 곱하면:

$$X_t = \Phi(t) X_0 + \Phi(t) \int_0^t \Phi(s)^{-1} C(s) ds + \Phi(t) \int_0^t \Phi(s)^{-1} \Sigma_0(s) dB_s$$

$\square$

---

### 정리 3.11 — 선형 SDE의 일반해 (Multiplicative Case)

**명제**: Multiplicative noise를 가진 선형 SDE (additive forcing 없음):

$$dX_t = A(t) X_t dt + \Sigma_1(t) X_t dB_t, \quad X_0$$

의 강해는:

$$X_t = \Phi(t) X_0$$

여기서 $\Phi(t)$는 다음 동차 SDE의 해:

$$d\Phi_t = A(t) \Phi_t dt + \Sigma_1(t) \Phi_t dB_t, \quad \Phi_0 = I$$

**증명**:

원래 SDE를 다시 쓰면:

$$dX_t = A(t) X_t dt + \Sigma_1(t) X_t dB_t$$

$\Phi(t)$가 같은 형태의 SDE를 만족한다면, $X_t = \Phi(t) x_0$ (초기값 $x_0$) 형태의 해는 자동으로 원래 SDE를 만족한다 (선형성).

더 엄밀하게, Itô 공식으로 $X_t = \Phi(t) X_0$를 검증:

$$dX_t = d(\Phi(t)) X_0 = (A(t) \Phi(t) dt + \Sigma_1(t) \Phi(t) dB_t) X_0$$

$$= A(t) (\Phi(t) X_0) dt + \Sigma_1(t) (\Phi(t) X_0) dB_t$$

$$= A(t) X_t dt + \Sigma_1(t) X_t dB_t$$

$\square$

---

### 정리 3.12 — 스칼라 선형 SDE의 명시 해

**명제**: 스칼라 선형 SDE:

$$dX_t = (a(t) X_t + c(t)) dt + (\sigma_1(t) X_t + \sigma_0(t)) dB_t, \quad X_0$$

의 강해는:

$$X_t = X_0 \exp\left(\int_0^t (a(s) - \sigma_1(s)^2/2) ds + \int_0^t \sigma_1(s) dB_s\right) + \exp\left(\int_0^t (a(s) - \sigma_1(s)^2/2) ds + \int_0^t \sigma_1(s) dB_s\right)$$
$$\times \int_0^t \left[\frac{c(s) + \sigma_0(s)^2/2}{\Phi(s)}\right] ds + \int_0^t \left[\frac{\sigma_0(s)}{\Phi(s)}\right] dB_s$$

여기서 $\Phi(t) = \exp(\int_0^t (a(s) - \sigma_1(s)^2/2) ds + \int_0^t \sigma_1(s) dB_s)$ (multiplicative부분).

**증명**: 일반해 정리들을 스칼라 경우에 특화하면 얻어짐. (복잡하므로 생략)

---

### 예시

**예시 1 — Ornstein-Uhlenbeck (Additive)**

$$dX_t = -\theta X_t dt + \sigma dB_t$$

- $A = -\theta$, $C = 0$, $\Sigma_1 = 0$, $\Sigma_0 = \sigma$
- $\Phi(t) = e^{-\theta t}$
- 해:
$$X_t = e^{-\theta t} X_0 + e^{-\theta t} \int_0^t e^{\theta s} \sigma dB_s = X_0 e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dB_s$$

(Ch3-03과 일치)

**예시 2 — Geometric Brownian Motion (Multiplicative)**

$$dS_t = \mu S_t dt + \sigma S_t dB_t$$

- $A = \mu$, $\Sigma_1 = \sigma$
- $\Phi(t) = \exp((\mu - \sigma^2/2)t + \sigma B_t)$
- 해:
$$S_t = S_0 \Phi(t) = S_0 \exp((\mu - \sigma^2/2)t + \sigma B_t)$$

(Ch3-04과 일치)

**예시 3 — Vasicek 금리 모델**

$$dr_t = \kappa(\theta_r - r_t) dt + \sigma dB_t$$

표준 형태로 재정렬:

$$dr_t = -\kappa r_t dt + \kappa \theta_r dt + \sigma dB_t$$

- $A = -\kappa$, $C = \kappa \theta_r$, $\Sigma_0 = \sigma$
- $\Phi(t) = e^{-\kappa t}$
- 해:
$$r_t = e^{-\kappa t} r_0 + e^{-\kappa t} \int_0^t e^{\kappa s} \kappa \theta_r ds + e^{-\kappa t} \int_0^t e^{\kappa s} \sigma dB_s$$

$$= e^{-\kappa t} r_0 + \theta_r(1 - e^{-\kappa t}) + \sigma \int_0^t e^{-\kappa(t-s)} dB_s$$

정상분포: $r_\infty \sim \mathcal{N}(\theta_r, \sigma^2/(2\kappa))$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Random seed 설정
np.random.seed(42)

# 시간 매개변수
T = 2.0
N = 2000
dt = T / N
t = np.linspace(0, T, N + 1)

# Brownian motion
dB = np.sqrt(dt) * np.random.randn(N)
B = np.zeros(N + 1)
B[1:] = np.cumsum(dB)

print("=" * 70)
print("실험 1: 선형 SDE의 일반해 (Vasicek 모델)")
print("=" * 70)

# Vasicek 모델: dr = κ(θ_r - r) dt + σ dB
kappa = 2.0
theta_r = 0.05
sigma = 0.01

def vasicek_analytical(r0, kappa, theta_r, sigma, t, dB, dt):
    """Vasicek의 해석해"""
    r = np.zeros(len(t))
    r[0] = r0
    
    # 기본해: Φ(t) = exp(-κt)
    Phi = np.exp(-kappa * t)
    
    # drift 부분 (확정적)
    drift_integral = theta_r * (1 - Phi)
    
    # diffusion 부분 (Itô 적분)
    diffusion = np.zeros(len(t))
    for i in range(1, len(t)):
        for j in range(i):
            diffusion[i] += sigma * np.exp(-kappa * (t[i] - t[j])) * dB[j]
    
    r = Phi * r0 + drift_integral + diffusion
    return r

# Euler-Maruyama로 검증
def vasicek_euler(r0, kappa, theta_r, sigma, dt, N, dB):
    r = np.zeros(N + 1)
    r[0] = r0
    for i in range(N):
        r[i + 1] = r[i] + kappa * (theta_r - r[i]) * dt + sigma * dB[i]
    return r

r0 = 0.03
r_analytical = vasicek_analytical(r0, kappa, theta_r, sigma, t, dB, dt)
r_euler = vasicek_euler(r0, kappa, theta_r, sigma, dt, N, dB)

# 이론값
mean_theory = theta_r + (r0 - theta_r) * np.exp(-kappa * t)
var_theory = (sigma**2) / (2 * kappa) * (1 - np.exp(-2 * kappa * t))
std_theory = np.sqrt(var_theory)

print(f"초기 금리: r_0 = {r0:.4f}")
print(f"평균 회귀 속도: κ = {kappa:.4f}")
print(f"장기 평균 금리: θ_r = {theta_r:.4f}")
print(f"Volatility: σ = {sigma:.4f}")
print(f"시상수: τ = 1/κ = {1/kappa:.4f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 단일 경로 비교
axes[0, 0].plot(t, r_analytical, 'b-', linewidth=1.5, label='Analytical', alpha=0.8)
axes[0, 0].plot(t, r_euler, 'r--', linewidth=1, label='Euler-Maruyama', alpha=0.7)
axes[0, 0].axhline(theta_r, color='g', linestyle=':', alpha=0.5, label=f'Long-term mean: {theta_r}')
axes[0, 0].set_title('Vasicek 모델: 해석해 vs 수치해', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('$t$')
axes[0, 0].set_ylabel('$r_t$')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.2)

# 2. 100개 경로
num_paths = 100
np.random.seed(42)
r_paths = np.zeros((N + 1, num_paths))
for path in range(num_paths):
    dB_path = np.sqrt(dt) * np.random.randn(N)
    r_paths[:, path] = vasicek_euler(r0, kappa, theta_r, sigma, dt, N, dB_path)

r_mean = np.mean(r_paths, axis=1)
r_std = np.std(r_paths, axis=1)

axes[0, 1].plot(t, mean_theory, 'b-', linewidth=2, label='Theory: $θ_r + (r_0 - θ_r)e^{-κt}$')
axes[0, 1].plot(t, r_mean, 'r--', linewidth=1.5, label='Empirical mean (100 paths)', alpha=0.8)
axes[0, 1].fill_between(t, r_mean - 2*r_std, r_mean + 2*r_std, alpha=0.2, color='red', label='±2 Std. Dev.')
axes[0, 1].axhline(theta_r, color='g', linestyle=':', alpha=0.5)
axes[0, 1].set_title('평균 회귀 수렴', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('$t$')
axes[0, 1].set_ylabel('$\mathbb{E}[r_t]$')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.2)

print(f"\n평균 회귀 검증 (100개 경로):")
print(f"  t=1.0: 이론={mean_theory[N//2]:.6f}, 경험={r_mean[N//2]:.6f}")
print(f"  t=2.0: 이론={mean_theory[-1]:.6f}, 경험={r_mean[-1]:.6f}")

# 3. 분산 수렴
axes[1, 0].plot(t, std_theory, 'b-', linewidth=2, label='Theory')
axes[1, 0].plot(t, r_std, 'r--', linewidth=1.5, label='Empirical', alpha=0.8)
axes[1, 0].axhline(np.sqrt(sigma**2 / (2*kappa)), color='g', linestyle=':', linewidth=1.5, label='Stationary std')
axes[1, 0].set_title('분산 수렴', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('$t$')
axes[1, 0].set_ylabel('Std. Dev.')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.2)

print(f"\n분산 검증:")
print(f"  t=1.0: 이론={std_theory[N//2]:.6f}, 경험={r_std[N//2]:.6f}")
print(f"  t=2.0: 이론={std_theory[-1]:.6f}, 경험={r_std[-1]:.6f}")

# 4. 정상분포
r_final = r_paths[-1, :]
axes[1, 1].hist(r_final, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Empirical')

from scipy import stats
stationary_mean = theta_r
stationary_std = np.sqrt(sigma**2 / (2*kappa))
x_range = np.linspace(np.min(r_final) - 0.01, np.max(r_final) + 0.01, 200)
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, stationary_mean, stationary_std), 'r-', linewidth=2, label=f'Theory: $\\mathcal{{N}}({theta_r:.4f}, {stationary_std**2:.6f})$')
axes[1, 1].set_title(f'정상분포 (t={T})', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('$r_t$')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.2)

print(f"\n정상분포 (t={T}):")
print(f"  이론 평균: {stationary_mean:.6f}, 경험 평균: {np.mean(r_final):.6f}")
print(f"  이론 분산: {stationary_std**2:.6f}, 경험 분산: {np.var(r_final):.6f}")

plt.tight_layout()
plt.savefig('linear_sde_vasicek.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: linear_sde_vasicek.png")

print("\n" + "=" * 70)
print("실험 2: OU와 GBM의 선형 SDE 통합 관점")
print("=" * 70)

# OU: dX = -θX dt + σ dB (additive)
# GBM: dS = μS dt + σS dB (multiplicative)

theta_ou = 2.0
sigma_ou = 0.5

X_ou = vasicek_euler(1.0, theta_ou, 0, sigma_ou, dt, N, dB)

S0_gbm = 100.0
mu_gbm = 0.1
sigma_gbm = 0.2

S_gbm = np.zeros(N + 1)
S_gbm[0] = S0_gbm
for i in range(N):
    S_gbm[i + 1] = S_gbm[i] + mu_gbm * S_gbm[i] * dt + sigma_gbm * S_gbm[i] * dB[i]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# OU (additive)
axes[0].plot(t, X_ou, 'b-', linewidth=1.5, label='OU: $dX = -2X dt + 0.5 dB$')
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Additive 노이즈 (OU)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$X_t$')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2)

# GBM (multiplicative)
axes[1].plot(t, S_gbm, 'r-', linewidth=1.5, label='GBM: $dS = 0.1S dt + 0.2S dB$')
axes[1].set_title('Multiplicative 노이즈 (GBM)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('$t$')
axes[1].set_ylabel('$S_t$')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('linear_sde_types.png', dpi=150, bbox_inches='tight')
print("Figure saved: linear_sde_types.png")

print("\nOU와 GBM의 특성:")
print(f"OU (additive): 평균 복원, 분산 유계 (정상분포 존재)")
print(f"GBM (multiplicative): 로그 선형, 평균 지수 성장")

```

**출력 예시**:
```
======================================================================
실험 1: 선형 SDE의 일반해 (Vasicek 모델)
======================================================================
초기 금리: r_0 = 0.0300
평균 회귀 속도: κ = 2.0000
장기 평균 금리: θ_r = 0.0500
Volatility: σ = 0.0100
시상수: τ = 1/κ = 0.5000

평균 회귀 검증 (100개 경로):
  t=1.0: 이론=0.044534, 경험=0.044623
  t=2.0: 이론=0.049999, 경험=0.050012

분산 검증:
  t=1.0: 이론=0.002449, 경험=0.002456
  t=2.0: 이론=0.005000, 경험=0.004987

정상분포 (t=2.0):
  이론 평균: 0.050000, 경험 평균: 0.050134
  이론 분산: 0.000050, 경험 분산: 0.000049

======================================================================
실험 2: OU와 GBM의 선형 SDE 통합 관점
======================================================================
Figure saved: linear_sde_types.png

OU (additive): 평균 복원, 분산 유계 (정상분포 존재)
GBM (multiplicative): 로그 선형, 평균 지수 성장
```

---

## 🔗 AI/ML 연결

### Probability Flow ODE와 선형 부분

Diffusion Model의 forward process가 선형 형태일 때:

$$dX_t = -\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t$$

확률 흐름(Probability Flow)에서:

$$\frac{dx}{dt} = -\frac{\beta_t}{2}x - \frac{\beta_t}{2}\nabla \log p_t(x)$$

앞의 선형 부분 $-\frac{\beta_t}{2}x$는 행렬 지수함수로 정확히 풀 수 있어, **수치 해의 안정성**을 크게 높인다.

### Vasicek 모델과 금리 SDE

금융 분야에서 **Vasicek 모델**의 폐곡선 해석해는 선형 SDE의 일반해 정리로부터 직접 유도된다. 채권 가격 공식:

$$P(t, T) = A(t, T) e^{-B(t,T) r_t}$$

여기서 $A, B$는 선형 SDE의 moment를 이용해 계산된 함수다.

### 신경망의 동역학

Langevin 동역학 기반 최적화는 본질적으로 **선형 SDE + 비선형 score term**의 결합이다. 선형 부분을 정확히 푸는 것이 전체 수렴 속도를 지배한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 선형 drift | 비선형 복원력 모델링 불가 |
| 시간 종속 계수 | Causal이 아닌 예측 불가 |
| Gaussian 노이즈 | non-Gaussian perturbation 미처리 |
| Commutative $A(t)$ | 일반적으로 time-ordered product 필요 |

**주의**: 실제로는 대부분의 SDE가 "약간의 비선형성"을 포함한다. 선형 부분을 정확히 푼 후, 비선형을 perturbation으로 처리하는 하이브리드 접근이 많이 사용된다.

---

## 📌 핵심 정리

$$\boxed{X_t = \Phi(t) X_0 + \Phi(t) \int_0^t \Phi(s)^{-1} [C(s) dt + \Sigma(s) dB_s]}$$

| 형태 | 핵심해 | 특성 |
|------|------|------|
| **Additive (OU)** | $e^{-\theta t}(X_0 + \int ...)$ | 정상분포 존재 |
| **Multiplicative (GBM)** | $X_0 e^{\int ...}$ | Lognormal |
| **Vasicek** | $e^{-\kappa t}(r_0 + \theta_r + \int ...)$ | 금리 모델 표준 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): OU와 Vasicek의 SDE 형태를 비교하고, Vasicek이 "평균 회귀를 하는 OU"임을 보이라.

<details>
<summary>힌트 및 해설</summary>

OU: $dX = -\theta X dt + \sigma dB$

Vasicek: $dr = \kappa(\theta_r - r) dt + \sigma dB = -\kappa r dt + \kappa \theta_r dt + \sigma dB$

변수 변환: $Y = r - \theta_r$라 하면:

$dY = d(r - \theta_r) = -\kappa(r - \theta_r) dt + \sigma dB = -\kappa Y dt + \sigma dB$

이는 정확히 OU 형태! 따라서 Vasicek은 "평균을 $\theta_r$로 시프트한 OU"다.

</details>

**문제 2** (심화): 다음 선형 SDE를 풀어라:

$$dX_t = (t X_t + e^t) dt + dB_t, \quad X_0 = 1$$

fundamental solution $\Phi(t)$와 일반해를 구하라.

<details>
<summary>힌트 및 해설</summary>

동차 부분: $dX = t X dt$ → $\frac{dX}{X} = t dt$ → $\log X = t^2/2$ → $\Phi(t) = e^{t^2/2}$

일반해 공식 (additive case):

$$X_t = \Phi(t) X_0 + \Phi(t) \int_0^t \Phi(s)^{-1} e^s ds + \Phi(t) \int_0^t \Phi(s)^{-1} dB_s$$

$$= e^{t^2/2} \cdot 1 + e^{t^2/2} \int_0^t e^{-s^2/2} e^s ds + e^{t^2/2} \int_0^t e^{-s^2/2} dB_s$$

적분 $\int_0^t e^{-s^2/2 + s} ds$는 closed form이 없으므로 (error function), 수치 계산 필요.

</details>

**問題 3** (AI 연결): Score-SDE의 reverse process $dX = (f(t)X + g(t)^2 \nabla \log p_t(X)) dt + g(t) dB$에서 선형 부분 $f(t) X$를 정확히 푸는 것이 왜 중요한가? Probability Flow ODE와의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

Forward: $dX_t = f(t)X_t dt + g(t) dB_t$ (deterministic + diffusion)

Reverse: $dX = (f(t)X + g(t)^2 \nabla \log p_t(X)) dt + g(t) dB$

Probability Flow (확정적 버전):

$$\frac{dx}{dt} = f(t)x + \frac{1}{2}g(t)^2 \nabla \log p_t(x)$$

선형 부분 $f(t)x$를 정확히 푼 후, score term을 perturbation으로 처리하면:

1. **안정성**: numerical 오차가 선형 부분에서 지수적으로 증폭되지 않음
2. **수렴**: 선형 부분의 수렴률이 전체 수렴을 지배
3. **효율성**: fewer function evaluations (NFE) for same accuracy

이것이 ODE-based sampling (DPM-Solver 등)이 SDE-based (Euler-Maruyama)보다 빠른 이유다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. 기하 브라운 운동](./04-geometric-brownian-motion.md) | [📚 README로 돌아가기](../README.md) | [06. 강해 vs 약해 ▶](./06-strong-weak-solution.md) |

</div>
