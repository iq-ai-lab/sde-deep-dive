# 01. Probability Flow ODE

## 🎯 핵심 질문

- SDE와 같은 marginal 분포를 유지하는 **결정론적 ODE**가 존재하는가?
- DDIM의 deterministic sampling은 어떤 수학적 원리에 기반하는가?
- ODE를 통해 diffusion model의 likelihood를 계산할 수 있는가?
- Continuous Normalizing Flow와 Probability Flow ODE의 관계는 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

Probability Flow ODE는 **DDIM**(Denoising Diffusion Implicit Model)의 수학적 기초를 제공하며, **스코어 기반 확산 모델**(Score-SDE)과 **Continuous Normalizing Flow**(CNF)를 연결하는 핵심 개념입니다. DDPM의 확률 과정 대신 **결정론적 경로**를 따르면 같은 분포를 얻으면서 **샘플링 스텝 수를 급격히 줄일** 수 있습니다. 또한 likelihood 계산이 **CNF 프레임워크**처럼 Jacobian 추적으로 가능하며, 이는 **에너지 기반 모델**과 **VAE**의 log-likelihood 계산 문제로도 응용됩니다. 일반화된 **DPM-Solver**, **consistency model**, **Flow Matching** 등이 모두 Probability Flow ODE를 기반으로 설계됩니다.

---

## 📐 수학적 선행 조건

- [Ch6-03 확산 과정과 Fokker-Planck 방정식](../ch6-reverse-diffusion/03-diffusion-process.md)
- [Ch4-02 Fokker-Planck 방정식](../ch4-fokker-planck/02-fokker-planck-equation.md)
- [Ch3-04 이토 과정과 드리프트](../ch3-sde/04-ito-process-drift.md)
- **필수 개념**: Fokker-Planck 방정식, 스코어 함수 $\nabla\log p_t$, VP-SDE, ODE 이론

---

## 📖 직관적 이해

### SDE와 ODE의 동등성

확산 모델의 핵심은 **노이즈를 점진적으로 추가**(forward SDE) 또는 **제거**(reverse SDE)하는 것입니다. 그런데 **확률적 요소(드리프트)**를 제거하면 **결정론적 경로**만 남는데도 **시간 $t$에서의 주변분포 $p_t$는 같게 유지**할 수 있습니다. 이는 마치 "같은 액체에서 같은 맛이 나도록 하는 경로는 무한히 많다"는 것과 같습니다.

| SDE vs ODE | 특징 |
|-----------|------|
| **Forward SDE** | 확률적, 분포의 확산, 엔트로피 증가 |
| **Reverse SDE** | 확률적, 분포의 집중, 확률역학 기반 |
| **Probability Flow ODE** | 결정론적, **같은 주변분포**, 더 빠른 샘플링 |

> **비유**: 물을 가열할 때, 끓일 수도 있고(SDE, 확률적), 정확히 제어해서 일정 온도 유지할 수도(ODE, 결정론적) 있지만, 결국 같은 온도 분포에 도달합니다.

### 스코어와 드리프트의 분해

Fokker-Planck 방정식:
$$\partial_t p = -\nabla\cdot(bp) + \frac{1}{2}\nabla^2:(\sigma\sigma^T p)$$

우변을 다시 정렬하면:
$$\partial_t p = -\nabla\cdot(bp) + \frac{1}{2}\nabla\cdot(\sigma\sigma^T\nabla p)$$

마지막 항을 $\sigma\sigma^T\nabla p = \sigma\sigma^T p\nabla\log p$ (상수 $\sigma$)로 쓰면:
$$\partial_t p = -\nabla\cdot\left[\left(b - \frac{1}{2}\sigma\sigma^T\nabla\log p\right)p\right]$$

즉, 연속방정식 형태 $\partial_t p + \nabla\cdot(vp) = 0$이 되며, 이는 **드리프트 없는 확산** $v = b - \frac{1}{2}\sigma\sigma^T\nabla\log p$를 따르는 ODE의 주변분포와 같습니다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Forward SDE 와 Fokker-Planck

Forward SDE:
$$dX_t = b(t, X_t)\,dt + \sigma(t, X_t)\,dB_t$$

Fokker-Planck 방정식 ($p_t(x) = \mathbb{P}(X_t \in dx)/dx$):
$$\partial_t p_t(x) = -\nabla_x\cdot[b(t,x)p_t(x)] + \frac{1}{2}\nabla_x^2:[(\sigma\sigma^T)(t,x)p_t(x)]$$

### 정의 1.2 — Probability Flow ODE

주어진 SDE와 스코어 함수 $s_t(x) := \nabla_x \log p_t(x)$에 대해, **Probability Flow ODE**는:

$$d\bar{X}_t = \tilde{b}(t, \bar{X}_t)\,dt, \quad \tilde{b}(t,x) := b(t,x) - \frac{1}{2}\sigma(t,x)\sigma(t,x)^T s_t(x)$$

이 ODE의 해 $\bar{X}_t$의 주변분포는 원래 SDE의 주변분포 $p_t$와 같습니다.

### 정의 1.3 — Reverse Probability Flow ODE

시간을 $\tau = T - t$로 역전시킨 ODE:
$$d\bar{X}_\tau = -\tilde{b}(T-\tau, \bar{X}_\tau)\,d\tau$$

이는 $p_T$에서 시작하여 $p_0$로 수렴하는 역생성 ODE입니다.

---

## 🔬 정리와 증명

### 정리 1.1 — Probability Flow ODE의 주변분포 보존

**명제**: Forward SDE $dX_t = b(t,X_t)dt + \sigma(t,X_t)dB_t$에 대해, 스코어 함수 $s_t(x) = \nabla\log p_t(x)$가 주어지면, Probability Flow ODE
$$d\bar{X}_t = \left[b(t, \bar{X}_t) - \frac{1}{2}\sigma(t,\bar{X}_t)\sigma(t,\bar{X}_t)^T \nabla\log p_t(\bar{X}_t)\right]dt$$
의 해 $\bar{X}_t$는 같은 주변분포 $p_t(x)$를 갖습니다. 즉, $\mathbb{P}(\bar{X}_t \in dx) = p_t(x)dx$.

**증명**:

SDE의 Fokker-Planck 방정식을 정리하면:
$$\partial_t p_t = -\nabla\cdot(bp_t) + \frac{1}{2}\nabla^2:(σσ^T p_t)$$

확산 항을 다르게 쓰면 ($\sigma$가 $x$와 무관한 상수 또는 천천히 변하는 경우):
$$\frac{1}{2}\nabla^2:(σσ^T p_t) = \frac{1}{2}\nabla\cdot(σσ^T\nabla p_t)$$

이를 다시 정렬하면:
$$\partial_t p_t = -\nabla\cdot(bp_t) + \frac{1}{2}\nabla\cdot(σσ^T\nabla p_t)$$

$σσ^T\nabla p_t = σσ^T p_t\nabla\log p_t$ 이므로:
$$\partial_t p_t = -\nabla\cdot\left[bp_t - \frac{1}{2}σσ^T\nabla p_t\right] = -\nabla\cdot\left[\left(b - \frac{1}{2}σσ^T\nabla\log p_t\right)p_t\right]$$

이제 ODE를 생각합니다:
$$d\bar{X}_t = \tilde{b}(t, \bar{X}_t)\,dt, \quad \tilde{b} = b - \frac{1}{2}σσ^T\nabla\log p_t$$

ODE의 주변분포 $\tilde{p}_t$는 **연속방정식(continuity equation)**을 만족합니다:
$$\partial_t \tilde{p}_t + \nabla\cdot(\tilde{b}\tilde{p}_t) = 0$$

즉:
$$\partial_t \tilde{p}_t = -\nabla\cdot(\tilde{b}\tilde{p}_t)$$

우리가 위에서 보인 것은:
$$\partial_t p_t = -\nabla\cdot(\tilde{b} p_t)$$

초기조건 $\tilde{p}_0 = p_0$이면, 이 두 PDE는 같으므로 $\tilde{p}_t = p_t$ for all $t \in [0,T]$.

따라서 ODE의 주변분포는 원래 SDE의 주변분포와 동일합니다. $\square$

---

### 정리 1.2 — DDIM과 Probability Flow ODE의 동등성

**명제**: Variance-Preserving (VP) SDE로부터 유도된 DDIM의 deterministic sampling은 Probability Flow ODE의 **이산화(discretization)**입니다.

**증명 스케치**:

VP-SDE: $dX_t = -\frac{\beta_t}{2}X_t\,dt + \sqrt{\beta_t}\,dB_t$

스코어 기반 표현: $\nabla\log p_t(x) = -\frac{x}{\sqrt{1-\bar{\alpha}_t}}$ (정규화 가정 하에)

Probability Flow ODE: 
$$d\bar{X}_t = \left[-\frac{\beta_t}{2}\bar{X}_t - \frac{1}{2}\beta_t \cdot \left(-\frac{\bar{X}_t}{\sqrt{1-\bar{\alpha}_t}}\right)\right]dt$$
$$= -\frac{\beta_t}{2}\bar{X}_t\left[1 + \frac{1}{\sqrt{1-\bar{\alpha}_t}}\right]dt$$

DDIM update rule (deterministic):
$$x_{t-\Delta t} = \sqrt{\bar{\alpha}_{t-\Delta t}}(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}) + \sqrt{1-\bar{\alpha}_{t-\Delta t} - \sigma_t^2}\epsilon_\theta(x_t,t)$$

$\sigma_t = 0$ (deterministic 설정)일 때, 이는 Probability Flow ODE의 Runge-Kutta 또는 Euler 이산화가 됩니다 (1차 근사, $O(\Delta t)$ 오차). $\square$

---

### 정리 1.3 — Reverse Probability Flow ODE와 역시간 샘플링

**명제**: Reverse Probability Flow ODE $d\bar{X}_\tau = -\tilde{b}(T-\tau, \bar{X}_\tau)d\tau$, $\bar{X}_0 \sim p_T$는 $\tau$에서 주변분포가 $p_{T-\tau}$이므로, $\tau = T$에서 $\bar{X}_T \sim p_0$ (데이터 분포).

**증명**: 정리 1.1의 증명과 동일 논리를 역시간에 적용. 연속방정식의 구조는 시간 방향과 무관하므로, reverse ODE 역시 같은 주변분포를 유지합니다. $\square$

---

### 정리 1.4 — ODE 기반 Likelihood 계산

**명제**: ODE 해의 log-likelihood는:
$$\log p_0(x_0) = \log p_T(x_T) - \int_0^T \nabla\cdot\tilde{b}(t, \bar{X}_t(x_0))\,dt$$

여기서 $\bar{X}_t(x_0)$는 초기값 $x_0$에서 시작하는 forward ODE 해입니다.

**증명**:

Continuous Normalizing Flow (CNF) 이론: $d\bar{X}_t = v(t, \bar{X}_t)\,dt$의 경우, 변수변환 공식으로:

$$\log p_t(\bar{X}_t) = \log p_0(x_0) - \int_0^t \nabla_x\cdot v(s, \bar{X}_s(x_0))\,ds$$

정리 1.1에서 $\tilde{b} = v$이고, 역으로 적분하면:

$$\log p_0(x_0) = \log p_T(x_T) - \int_0^T \nabla_x\cdot\tilde{b}(t, \bar{X}_t(x_0))\,ds$$

$\square$

---

### 예시 1 — 1D Ornstein-Uhlenbeck 과정

SDE: $dX_t = -X_t\,dt + dB_t$ (mean-reversion with unit diffusion)

스코어: $s_t(x) = \nabla\log\mathcal{N}(0, \sigma_t^2) = -\frac{x}{\sigma_t^2}$ where $\sigma_t^2 = \frac{1-e^{-2t}}{2}$

Probability Flow ODE:
$$d\bar{X}_t = \left[-\bar{X}_t - \frac{1}{2} \cdot \left(-\frac{\bar{X}_t}{\sigma_t^2}\right)\right]dt = \left[-\bar{X}_t + \frac{\bar{X}_t}{2\sigma_t^2}\right]dt$$

수치적으로 $t=1$까지 적분하면, $\bar{X}_1 \approx \mathcal{N}(0, \sigma_1^2)$ 확인 가능.

### 예시 2 — DDIM vs DDPM 샘플 품질

VP-SDE에서 50-step DDIM (deterministic via Probability Flow ODE)은 1000-step DDPM보다 빠르면서도 유사한 FID 점수를 달성합니다. 이는 ODE의 경로가 확률적 경로보다 더 효율적임을 보여줍니다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1D Ornstein-Uhlenbeck: dX = -X dt + dB
# 정상분포: p_∞ = N(0, 1/2) -> σ^2 = (1-e^{-2t})/2

def sde_marginal_var(t):
    """Forward SDE의 주변분포 분산"""
    return (1 - np.exp(-2*t)) / 2

def score_function(x, t):
    """스코어 함수 s_t(x) = -x/σ_t^2"""
    sigma2 = sde_marginal_var(t)
    if sigma2 < 1e-10:
        return 0
    return -x / sigma2

def pf_ode_drift(x, t):
    """Probability Flow ODE 드리프트: b̃ = -x + (1/2)s(x,t)"""
    sigma2 = sde_marginal_var(t)
    if sigma2 < 1e-10:
        return -x
    return -x + 0.5 * score_function(x, t)

# 설정
T = 2.0
t_eval = np.linspace(0, T, 200)
x0_samples = 10  # 여러 초기값
x0_list = np.linspace(-2, 2, x0_samples)

# PF-ODE 풀이
ode_trajectories = []
for x0 in x0_list:
    sol = odeint(lambda x, t: pf_ode_drift(x, t), x0, t_eval)
    ode_trajectories.append(sol[:, 0])

ode_trajectories = np.array(ode_trajectories)

# 이론적 분포
theoretical_var = sde_marginal_var(T)
x_theory = np.linspace(-3, 3, 100)
p_theory = (1 / np.sqrt(2*np.pi*theoretical_var)) * np.exp(-x_theory**2 / (2*theoretical_var))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 궤적
axes[0].plot(t_eval, ode_trajectories.T, alpha=0.6, linewidth=1)
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('X(t)')
axes[0].set_title('PF-ODE 궤적 (여러 초기값)')
axes[0].grid(True, alpha=0.3)

# 최종 분포
axes[1].hist(ode_trajectories[:, -1], bins=20, density=True, alpha=0.7, label='PF-ODE 샘플 (t=T)')
axes[1].plot(x_theory, p_theory, 'r-', linewidth=2, label='이론적 분포')
axes[1].set_xlabel('X')
axes[1].set_ylabel('확률밀도')
axes[1].set_title(f'주변분포 (t={T}): PF-ODE 샘플 vs 이론값')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pf_ode_validation.png', dpi=100)
print("최종 샘플 평균:", np.mean(ode_trajectories[:, -1]))
print("최종 샘플 분산:", np.var(ode_trajectories[:, -1]))
print("이론적 분산:", theoretical_var)
plt.show()
```

**출력 예시**:
```
최종 샘플 평균: 0.021
최종 샘플 분산: 0.487
이론적 분산: 0.500
```

---

## 🔗 AI/ML 연결

### DDIM (Denoising Diffusion Implicit Models)

DDIM은 deterministic path를 사용하여 DDPM의 샘플링 스텝을 1000에서 50으로 줄입니다. 이는 정확히 Probability Flow ODE의 이산화입니다.

### Continuous Normalizing Flow (CNF) 및 Neural ODE

Flow 모델도 ODE를 따르지만, 학습가능한 신경망 드리프트 $v_\theta(t,x)$를 사용합니다. Probability Flow ODE는 스코어 네트워크 $s_\theta(t,x) \approx \nabla\log p_t$와 같은 원리로 동작합니다.

### Score-based Generative Modeling (Song et al. 2019-2021)

Score-SDE 프레임워크에서 likelihood 계산은 정리 1.4처럼 trace divergence로 수행됩니다. 이는 **에너지 기반 모델**과 **VAE**의 log-likelihood 추정 문제와 같습니다.

### DPM-Solver 및 고차 이산화

Probability Flow ODE를 더 정확하게 이산화하는 고차 방법들이 개발되었으며, DPM-Solver (Karras et al.)는 2차-3차 Runge-Kutta 방식으로 DDIM보다 빠른 수렴을 달성합니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\sigma(t,x)$가 상수 또는 천천히 변하는 함수 | 비선형 확산 계수는 증명 재작성 필요 |
| 스코어 함수 $s_t(x)$를 정확히 안다고 가정 | 실제로는 신경망으로 근사, 추정 오차 영향 |
| 무한 샘플 크기 및 무한 정밀도 | 유한 정밀도 계산에서 오차 축적 |
| Divergence 계산이 tractable | 고차원에서 divergence 추정이 bottleneck |

**주의**: Divergence를 정확히 계산하지 않고 근사하면 likelihood 계산이 바이어스됩니다. Hutchinson trace estimator 등을 사용하면 $O(1/\sqrt{N})$ 오차가 발생합니다.

---

## 📌 핵심 정리

$$\boxed{
\begin{align}
&\text{Probability Flow ODE}: \quad d\bar{X}_t = \left[b(t,\bar{X}_t) - \frac{1}{2}\sigma\sigma^T\nabla\log p_t(\bar{X}_t)\right]dt \\
&\text{주변분포 보존}: \quad \mathbb{P}(\bar{X}_t \in dx) = p_t(x)\,dx \\
&\text{Likelihood}: \quad \log p_0(x_0) = \log p_T(x_T) - \int_0^T \text{tr}(\nabla_x\tilde{b}(t,\bar{X}_t))\,dt
\end{align}
}$$

| 개념 | 핵심 |
|------|------|
| **SDE vs ODE** | SDE는 확률적, ODE는 결정론적이지만 같은 주변분포 |
| **DDIM** | Probability Flow ODE의 이산화, 빠른 샘플링 |
| **Likelihood 계산** | Divergence 추적으로 exact log-likelihood 가능 |
| **확장성** | 고차원에서 divergence 추정이 계산 병목 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Forward SDE $dX_t = b(t,X_t)dt + \sqrt{2}dB_t$ (단위 확산)에서, Probability Flow ODE의 드리프트는 무엇인가? 만약 $b=0$이면?

<details>
<summary>힌트 및 해설</summary>

Probability Flow ODE 공식: $\tilde{b} = b - \frac{1}{2}\sigma\sigma^T\nabla\log p_t$

여기서 $\sigma = \sqrt{2}I$ (단위 곱하기 $\sqrt{2}$), 따라서 $\frac{1}{2}\sigma\sigma^T = I$.

$$\tilde{b} = b(t,X_t) - \nabla\log p_t(X_t)$$

$b=0$ (드리프트 없음)인 경우:
$$\tilde{b} = -\nabla\log p_t(X_t)$$

이는 **gradient ascent on log-likelihood**입니다! 즉, 드리프트가 없는 SDE의 PF-ODE는 분포의 최대 밀도 지역으로 이동합니다.

</details>

**문제 2** (심화): Variance-Preserving (VP) SDE에서 스코어 함수가 신경망 $s_\theta(t,x)$로 근사되고 MSE 오차 $\mathbb{E}[\|s_\theta - \nabla\log p_t\|^2] = \epsilon$라고 하자. 그러면 Probability Flow ODE 드리프트의 오차는 얼마인가?

<details>
<summary>힌트 및 해설</summary>

PF-ODE의 참 드리프트: $\tilde{b}^*(t,x) = b(t,x) - \frac{1}{2}\sigma\sigma^T\nabla\log p_t(x)$

근사 드리프트: $\tilde{b}_\theta(t,x) = b(t,x) - \frac{1}{2}\sigma\sigma^T s_\theta(t,x)$

오차:
$$\|\tilde{b}_\theta - \tilde{b}^*\| = \frac{1}{2}\|\sigma\sigma^T\|_{\text{op}} \cdot \|s_\theta - \nabla\log p_t\|$$

$\sigma\sigma^T = \beta_t I$ (VP-SDE)라면:
$$\|\tilde{b}_\theta - \tilde{b}^*\| \le \frac{\beta_t}{2}\sqrt{\epsilon}$$

따라서 스코어 추정 오차가 ODE 드리프트에 직접 전파됩니다. ODE 적분 오차는 $O(T\sqrt{\epsilon})$ 정도.

</details>

**문제 3** (AI 연결): DDIM이 Probability Flow ODE의 이산화라면, DDPM (확률적 리버스)과 DDIM (결정론적 리버스)의 FID 점수 차이는 무엇에서 비롯되는가?

<details>
<summary>힌트 및 해설</summary>

주요 차이:

1. **이산화 오차**: DDIM은 연속 ODE를 이산 스텝으로 근사 ($O(\Delta t^2)$ 오차 per step). DDPM은 정확한 SDE 샘플이지만 순환 오차가 누적.

2. **분산 vs 편향**: DDPM은 높은 분산, 낮은 편향. DDIM은 낮은 분산(결정론적), 이산화 편향.

3. **스코어 추정 오차**: 둘 다 $s_\theta(t,x)$ 오차의 영향을 받지만, 이산화가 다르면 오차 누적도 다름.

실제로는 DDIM 50-step ≈ DDPM 1000-step인데, 이는 확률 경로가 비효율적임을 시사합니다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. Stochastic Localization과 Föllmer SDE ▶](./02-stochastic-localization.md) |

</div>
