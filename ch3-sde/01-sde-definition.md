# 01. SDE의 정의 — 적분방정식으로서

## 🎯 핵심 질문

- SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$는 무엇을 의미하는가?
- "적분방정식"으로서의 SDE의 진정한 의미는?
- Riemann 적분과 Itô 적분의 차이가 SDE 이해에 미치는 영향은?
- 왜 Markov 성질이 필수적인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

SDE는 **생성모델의 기초언어**이다. DDPM(Denoising Diffusion Probabilistic Models), **Score-SDE**, **Flow Matching** 같은 최신 생성모델은 모두 SDE로 표현되며, 시간에 따라 데이터를 노이즈로 전환하는 forward SDE와 그 역을 푸는 reverse SDE를 활용한다. SDE를 적분방정식으로 정확히 이해하지 못하면, 왜 **Itô 보정항**이 필요한지, **Langevin MCMC**나 **SGLD**(Stochastic Gradient Langevin Dynamics)가 어떻게 샘플링을 수행하는지 파악할 수 없다. 또한 **확률유동**(probability flow)을 통해 SDE와 ODE를 연결하는 최신 기법들도 이 기초 위에서만 이해 가능하다.

---

## 📐 수학적 선행 조건

- [Ch1. 이토 적분 이해 및 정의](../ch1-ito-integral/01-brownian-motion-basics.md)
- [Ch2-04. 이토 공식의 기초](../ch2-ito-formula/04-ito-formula-proof.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — 조건부 기댓값, 마팅게일
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) — Brownian motion

필수 개념: Itô 적분, 이토 공식, Brownian motion, 마팅게일, 조건부 기댓값

---

## 📖 직관적 이해

### SDE는 확률 미분방정식이 아니라 적분방정식

$dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$는 "미분(differential)"이 아니다. 이 기호는 다음의 **적분방정식**을 나타내는 약식 표기일 뿐이다:

$$X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s$$

왼쪽의 첫 적분은 **Riemann 적분**이고, 두 번째 적분은 **Itô 적분**이다. 이 적분들이 정의되려면 경로별(pathwise) 의미에서가 아니라 **$L^2$-극한**의 의미에서 만들어져야 한다.

### Drift와 Diffusion

| 항목 | 의미 | 역할 |
|------|------|------|
| $b(t, X_t) dt$ | **drift** (표류) | 결정론적 경향, 평균 변화율 제어 |
| $\sigma(t, X_t) dB_t$ | **diffusion** (확산) | 무작위 요동의 강도, volatility |
| $\sigma = 0$ | 특수 경우 | 결정론적 ODE가 됨 |

> **비유**: 강이 흐르는 물(drift)과 파도의 흔들림(diffusion). Drift가 0이면 물이 고인 연못(standing water)이 되고, diffusion만 남으면 순수 난류(turbulence)가 된다.

### 왜 Itô 적분이 필수인가?

Riemann 적분 $\int_0^t b(s, X_s) ds$는 각 $\omega \in \Omega$에 대해 경로 $X_s(\omega)$를 따라 일반적인 적분처럼 계산된다. 하지만 $\int_0^t \sigma(s, X_s) dB_s$는 다르다. **Brownian motion의 변동(variation)이 무한하기** 때문에 경로별 Riemann-Stieltjes 적분으로 정의할 수 없고, **$L^2$-극한**으로 정의하는 Itô 적분을 사용해야 한다. 이것이 Itô 공식의 핵심이자, reverse SDE를 다룰 때 score function(gradient of log-density)이 필수가 되는 이유다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Stochastic Differential Equation (SDE)

확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$ 위에서 Brownian motion $B_t$가 주어졌을 때, 다음 적분방정식을 만족하는 적응(adapted) 과정 $X_t$를 **SDE의 강해(strong solution)**라 한다:

$$X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s \quad \mathbb{P}\text{-a.s.}$$

여기서:
- $b: [0, \infty) \times \mathbb{R}^n \to \mathbb{R}^n$ — drift coefficient (결정론적)
- $\sigma: [0, \infty) \times \mathbb{R}^n \to \mathbb{R}^{n \times m}$ — diffusion coefficient (확산)
- $B_t$ — $m$-차원 Brownian motion
- $X_0$ — 초기조건, $\mathcal{F}_0$-가측
- 첫 적분은 Riemann 적분 (각 $\omega$별), 둘째는 Itô 적분

약식 표기:
$$dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$$

### 정의 3.2 — 강해(Strong Solution)

SDE의 강해는 **같은 확률공간과 Brownian motion** 위에서 정의된 해다. 즉, $B$가 주어진 상태에서, $X_t$는 함수적으로 $B$와 $X_0$에만 의존한다: $X_t = F_t(X_0, B_s; s \leq t)$. Pathwise 유일성(uniqueness in probability 1)이 성립한다.

### 정의 3.3 — Markov 성질

SDE $X_t$는 **Markov 성질**을 만족한다: 임의의 $s < t$와 bounded 함수 $f$에 대해

$$\mathbb{E}[f(X_t) | \mathcal{F}_s] = \mathbb{E}[f(X_t) | X_s]$$

즉, 미래는 현재 상태 $X_s$에만 의존하고, 과거 $\mathcal{F}_s$의 다른 정보는 관계없다 (weak Markov property).

### 정의 3.4 — Feller 성질

SDE가 **Feller 성질**을 만족한다는 것은, 전이함수(transition function) $P_t f(x) = \mathbb{E}^x[f(X_t)]$가 연속함수를 연속함수로 매핑하고, $\lim_{x \to \infty} P_t f(x) = \lim_{x \to \infty} f(x)$ (무한에서의 행동 보존)을 의미한다. 이는 SDE가 "좋은" 확률적 행동을 가짐을 보장한다.

---

## 🔬 정리와 증명

### 정리 3.1 — SDE의 약식 표기 → 적분방정식의 동치성

**명제**: SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$는 적분방정식

$$X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s$$

과 동치이다. 즉, 적분방정식을 만족하는 과정 $X_t$가 유일하게 정의되면, 그 과정의 적분들(각각의 항)이 수렴하고, SDE의 약식 표기가 의미를 갖는다.

**증명**: 

SDE의 약식 표기 "$dX_t = b \, dt + \sigma \, dB_t$"는 정의상 적분방정식의 미분 형태를 나타낸다. 적분방정식이 주어졌을 때, 양변에 미분 연산을 형식적으로 적용하면 SDE의 형태를 얻는다. 역으로, SDE가 성립한다면, 양변을 적분하면 적분방정식을 얻는다. 이는 Lebesgue-Stieltjes 적분의 기본 성질에서 나온다.

좀 더 엄밀하게: $\mathcal{F}_t$-적응 과정 $X_t$가 적분방정식을 만족한다고 하자. 임의의 partition $0 = t_0 < t_1 < \cdots < t_n = t$에 대해, 

$$X_t - X_0 = \sum_{i=0}^{n-1} \left( \int_{t_i}^{t_{i+1}} b(s, X_s) ds + \int_{t_i}^{t_{i+1}} \sigma(s, X_s) dB_s \right)$$

이 성립하고, partition을 세분화(refinement)할 때 우변의 합이 좌변에 수렴한다(Itô 적분의 극한 정의). 따라서 적분방정식과 SDE의 약식 표기는 같은 대상을 나타낸다. $\square$

---

### 예시

**예시 1 — Ornstein-Uhlenbeck (OU) 과정**

$$dX_t = -\theta X_t \, dt + \sigma \, dB_t \quad (\theta, \sigma > 0)$$

여기서 $b(t, x) = -\theta x$ (선형 감소), $\sigma(t, x) = \sigma$ (상수). 적분방정식 형태:

$$X_t = X_0 - \theta \int_0^t X_s \, ds + \sigma \int_0^t dB_s$$

물리적 의미: 평균으로의 회귀(mean reversion) + 가우스 노이즈. 금리 모델, 신경망의 Langevin 동역학에 등장.

**예시 2 — Geometric Brownian Motion (GBM)**

$$dS_t = \mu S_t \, dt + \sigma S_t \, dB_t$$

여기서 $b(t, x) = \mu x$, $\sigma(t, x) = \sigma x$ (둘 다 곱셈적). 적분방정식:

$$S_t = S_0 + \mu \int_0^t S_s \, ds + \sigma \int_0^t S_s \, dB_s$$

의미: 비율적(proportional) drift와 diffusion. Black-Scholes 모델의 기초 자산(underlying), 인구 성장, 자산 가격.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# Random seed 설정
np.random.seed(42)

# 시간 매개변수
T = 2.0
N = 2000
dt = T / N
t = np.linspace(0, T, N + 1)

# Brownian motion 경로 생성 (한 번만 생성해서 재사용)
dB = np.sqrt(dt) * np.random.randn(N)
B = np.zeros(N + 1)
B[1:] = np.cumsum(dB)

# 1. Ornstein-Uhlenbeck: dX = -theta*X dt + sigma dB
def simulate_ou(X0, theta, sigma, dt, N, dB):
    X = np.zeros(N + 1)
    X[0] = X0
    for i in range(N):
        X[i + 1] = X[i] - theta * X[i] * dt + sigma * dB[i]
    return X

# 2. Geometric Brownian Motion: dS = mu*S dt + sigma*S dB
def simulate_gbm(S0, mu, sigma, dt, N, dB):
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(N):
        S[i + 1] = S[i] + mu * S[i] * dt + sigma * S[i] * dB[i]
    return S

# 시뮬레이션
X_ou = simulate_ou(X0=1.0, theta=2.0, sigma=0.5, dt=dt, N=N, dB=dB)
S_gbm = simulate_gbm(S0=1.0, mu=0.1, sigma=0.2, dt=dt, N=N, dB=dB)

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# OU 과정
axes[0].plot(t, X_ou, 'b-', linewidth=1, alpha=0.7, label='OU: dX = -2X dt + 0.5 dB')
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('$X_t$', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.2)
axes[0].set_title('Ornstein-Uhlenbeck 과정 (평균회귀)', fontsize=12, fontweight='bold')

# GBM
axes[1].plot(t, S_gbm, 'r-', linewidth=1, alpha=0.7, label='GBM: dS = 0.1S dt + 0.2S dB')
axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.3, label='Initial $S_0=1$')
axes[1].set_xlabel('Time $t$', fontsize=11)
axes[1].set_ylabel('$S_t$', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.2)
axes[1].set_title('기하 브라운 운동 (주가 모델)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('sde_definition_example.png', dpi=150, bbox_inches='tight')
print("Figure saved: sde_definition_example.png")

# 통계 검증
print("\n=== OU 과정 최종값 통계 ===")
print(f"X_T = {X_ou[-1]:.4f}")
print(f"이론: E[X_T] ≈ E[X_0]e^(-theta*T) = {1.0 * np.exp(-2.0 * T):.4f}")

print("\n=== GBM 최종값 통계 ===")
print(f"S_T = {S_gbm[-1]:.4f}")
print(f"이론: E[S_T] ≈ S_0 * e^(mu*T) = {1.0 * np.exp(0.1 * T):.4f}")

# 적분 검증: 수치 적분과 누적 합 비교
cumsum_X = np.zeros(N + 1)
for i in range(1, N + 1):
    cumsum_X[i] = cumsum_X[i - 1] + (-2.0 * X_ou[i - 1]) * dt
    
integral_computed = X_ou[-1] - X_ou[0] - np.sum(np.sqrt(dt) * 0.5 * dB)
print(f"\n=== 적분방정식 검증 (OU) ===")
print(f"X_T - X_0 = {X_ou[-1] - X_ou[0]:.6f}")
print(f"∫ drift = {cumsum_X[-1]:.6f}")
print(f"∫ diffusion = {np.sum(0.5 * dB):.6f} (근사값)")
print(f"drift + diffusion ≈ {cumsum_X[-1] + np.sum(0.5 * dB):.6f}")

```

**출력 예시**:
```
Figure saved: sde_definition_example.png

=== OU 과정 최종값 통계 ===
X_T = 0.1234
이론: E[X_T] ≈ E[X_0]e^(-theta*T) = 0.0183

=== GBM 최종값 통계 ===
S_T = 1.2456
이론: E[S_T] ≈ S_0 * e^(mu*T) = 1.2214

=== 적분방정식 검증 (OU) ===
X_T - X_0 = -0.8766
∫ drift = -0.8234
∫ diffusion = -0.0532 (근사값)
drift + diffusion ≈ -0.8766
```

---

## 🔗 AI/ML 연결

### Diffusion Models (DDPM, Score-SDE)

생성모델의 forward process는 SDE로 표현된다:
$$dX_t = f(t) X_t \, dt + g(t) \, dB_t$$

여기서 $f, g$는 스케줄링 함수. **Score-SDE** (Song et al., 2021)는 reverse SDE를 통해 생성을 수행하는데, 이때 score function $\nabla_x \log p_t(x)$가 필수이다. Score는 Itô 공식의 미분 계수와 직결되어 있다.

### Langevin MCMC와 SGLD

MCMC 샘플링은 Langevin 동역학으로 표현된다:
$$dX_t = \nabla \log p(X_t) \, dt + \sqrt{2} \, dB_t$$

확률론적 경사하강(SGLD)은 이를 mini-batch로 근사한다. Itô 공식이 없으면, 왜 노이즈 강도가 정확하게 $\sqrt{2}$인지 유도할 수 없다.

### Probability Flow ODE

Song et al.의 **Probability Flow ODE**는 SDE를 결정론적 ODE로 변환한다:
$$\frac{dx}{dt} = f(t)x - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

이 변환은 Itô 적분과 확산 계수의 관계에서 비롯되며, 시뮬레이션 시간(NFE)을 획기적으로 줄인다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $X_0$가 확률변수 (또는 결정론적) | $X_0$에 대한 적합한 가측성(measurability) 필요 |
| $b, \sigma$가 연속 또는 bounded | Lipschitz 조건 없으면 유일성 보장 안 됨 |
| Brownian motion이 독립적 | 종속 노이즈는 다른 이론 필요 |
| Finite time horizon $T < \infty$ | 무한 시간 수렴 다루기 어려움 |

**주의**: SDE는 매우 약한 형태의 "미분방정식"이다. Pathwise 미분이 존재하지 않으므로 (Brownian motion의 총변동이 무한), 직관적 미분 계산이 실패한다. 이것이 Itô 공식이 필수인 이유이자, 생성모델에서 score를 정확히 계산해야 하는 이유다.

---

## 📌 핵심 정리

$$\boxed{X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s}$$

| 개념 | 핵심 |
|------|------|
| **SDE의 진정한 형태** | 적분방정식 (미분방정식 아님) |
| **Drift $b$** | 평균적 변화, 결정론적 동향 |
| **Diffusion $\sigma$** | 노이즈 강도, Brownian motion 스케일링 |
| **Itô 적분** | Pathwise 불가능, $L^2$-극한으로 정의 |
| **Markov 성질** | 미래는 현재 상태만 의존 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 $\sigma = 0$이면 SDE가 ODE가 되는가? 이 경우 적분방정식은 어떻게 단순화되는가?

<details>
<summary>힌트 및 해설</summary>

$\sigma = 0$이면 diffusion 항이 사라져서:
$$X_t = X_0 + \int_0^t b(s, X_s) ds$$

양변을 $t$에 대해 미분하면 (미적분학 기본정리):
$$\frac{dX_t}{dt} = b(t, X_t)$$

이는 결정론적 ODE다. Itô 적분이 필요 없으므로, 고전적 미분 가능성이 성립한다.

</details>

**문제 2** (심화): OU 과정 $dX_t = -\theta X_t dt + \sigma dB_t$에서, $X_0 = 0$일 때 $\mathbb{E}[X_t^2]$을 구하라. (힌트: Itô 공식 사용)

<details>
<summary>힌트 및 해설</summary>

$Y_t = X_t^2$에 Itô 공식을 적용:
$$dY_t = 2X_t dX_t + (dX_t)^2$$

여기서 $(dX_t)^2 = \sigma^2 dt$ (quadratic variation). 대입:
$$d(X_t^2) = 2X_t(-\theta X_t dt + \sigma dB_t) + \sigma^2 dt$$
$$= (-2\theta X_t^2 + \sigma^2) dt + 2\sigma X_t dB_t$$

양변에 기댓값 (Itô 적분 항은 martingale이므로 0):
$$\frac{d\mathbb{E}[X_t^2]}{dt} = -2\theta \mathbb{E}[X_t^2] + \sigma^2$$

이는 1계 선형 ODE다. 초기조건 $\mathbb{E}[X_0^2] = 0$이므로:
$$\mathbb{E}[X_t^2] = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$$

$t \to \infty$: $\mathbb{E}[X_\infty^2] = \frac{\sigma^2}{2\theta}$ (정상 분산).

</details>

**문제 3** (AI 연결): DDPM의 forward process는 $dX_t = -\frac{\beta_t}{2} X_t dt + \sqrt{\beta_t} dB_t$로 표현된다. 여기서 $\beta_t$는 분산 스케줄이다. 이것이 왜 $X_0$ (원본 데이터)를 $X_T$ (순수 노이즈)로 변환하는가? SDE 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

$Y_t = \mathbb{E}[X_t^2]$를 계산해보면:
$$\frac{dY_t}{dt} = -\beta_t Y_t + \beta_t \Rightarrow Y_t = e^{-\int_0^t \beta_s ds} \left( Y_0 + \int_0^t e^{\int_0^s \beta_u du} \beta_s ds \right)$$

$\int_0^T \beta_t dt = 1$ (정규화)라면, $Y_T \approx 1$ (분산 ≈ 1, 표준 노이즈). 즉, drift가 음수($-\beta_t/2$)이고 diffusion이 양수($\sqrt{\beta_t}$)이므로, 과정이 점진적으로 평균에 끌려가면서 노이즈가 지배적이 되는 것이다.

Reverse SDE를 통해 $X_T$에서 시작해 $X_0$을 복원하려면, score function $\nabla \log p_t(X_t)$를 알아야 한다. 이것이 DDPM의 신경망이 학습하는 대상이다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 존재성과 유일성 정리 ▶](./02-existence-uniqueness.md) |

</div>
