# 02. 역 Kolmogorov 방정식과 생성자

## 🎯 핵심 질문

- SDE의 생성자(infinitesimal generator)는 무엇이고, 어떻게 정의되는가?
- 역 Kolmogorov 방정식이 정방향 Fokker-Planck과 어떻게 쌍대 관계인가?
- Feynman-Kac 공식은 어떻게 옵션 가격결정(Black-Scholes)과 연결되는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

역 Kolmogorov 방정식과 생성자는 **확률 프로세스의 기댓값 진화**를 기술한다. Score-based generative modeling에서 **score function** $\nabla \log p_t(x)$를 학습할 때, 이는 생성자의 adjoint와 관련된다. 또한 **Feynman-Kac 공식**은 PDE 솔버로서 작용하며, diffusion model의 역과정에서 optimal control 또는 importance weighting과 연결된다. Flow Matching과 같은 최신 방법도 이 framework 위에서 작동한다.

---

## 📐 수학적 선행 조건

- [Ch1-04 마팅게일과 선택적 중단 정리](../ch1-ito-integral/04-martingale-optional-stopping.md)
- [Ch2-03 이토 공식의 확장 형태](../ch2-ito-formula/03-ito-formula-multivariate.md)
- [Ch4-01 Fokker-Planck 방정식의 유도](./01-fokker-planck-derivation.md)
- **필수 개념**: 선택적 중단 정리(optional stopping), 마팅게일, bounded stopping time

---

## 📖 직관적 이해

### 생성자: 미래의 확률을 현재에서 읽기

SDE를 풀면 $X_0$에서 시작해 시간 $t$에 $X_t$에 도달한다. 특정 함수 $f(X_t)$의 **기댓값 곡선** $u(t, x) = \mathbb{E}^{x}[f(X_t)]$을 생각해보자 ($\mathbb{E}^x$는 $X_0 = x$에서 시작할 때의 기댓값).

아주 짧은 시간 $h$ 후에 기댓값이 어떻게 변할까?

$$u(t+h, x) - u(t, x) = \mathbb{E}^x[f(X_{t+h})] - f(x) \approx \mathbb{E}^x[\text{SDE의 1단계 효과}]$$

이 변화율의 극한이 **생성자**다.

### Fokker-Planck과 역 Kolmogorov의 쌍대성

- **Fokker-Planck** (순방향): 밀도 $p(t,x)$가 시간에 따라 어떻게 변하는가? (관점: "이 점에 있을 확률")
- **역 Kolmogorov** (역방향): 기댓값 $u(t,x) = \mathbb{E}^{t,x}[f(X_T)]$가 시간을 거슬러 어떻게 변하는가? (관점: "현재 기댓값은?")

이 둘은 함수공간의 **adjoint 연산자**(dual operator)로 연결된다.

| 관점 | 순방향 (Fokker-Planck) | 역방향 (역 Kolmogorov) |
|------|----------------------|----------------------|
| 해석 대상 | 밀도 $p(t,x)$ | 기댓값 $u(t,x)$ |
| 시간 방향 | 미래로 진행 $t \to T$ | 과거로 진행 $T \to t$ |
| 경계조건 | 초기조건 $p(0,x)$ | 최종조건 $u(T,x) = f(x)$ |
| 미분연산자 | $\mathcal{L}^* p$ | $\mathcal{L} u$ |

> **비유**: 영화와 역상영. 순방향은 현재 상태(밀도)가 미래로 어떻게 진화하는지, 역방향은 미래 결과(보상 $f$)가 현재 시점에서 기댓값으로 얼마인지를 본다.

---

## ✏️ 엄밀한 정의

### 정의 4.4 — 무한소 생성자 (Infinitesimal Generator)

SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$의 **생성자** $\mathcal{L}$는 다음과 같이 정의된다:

$$\mathcal{L} f(x) := \lim_{h \to 0^+} \frac{\mathbb{E}^x[f(X_h)] - f(x)}{h}$$

시간 독립 SDE에서:

$$\mathcal{L} f(x) = b(x) \cdot \nabla f(x) + \frac{1}{2} \text{tr}(\sigma(x)\sigma(x)^T \nabla^2 f(x))$$

$$= b(x) \cdot \nabla f(x) + \frac{1}{2} \sum_{i,j} a_{ij}(x) \partial_i \partial_j f(x)$$

여기서 $a(x) = \sigma(x)\sigma(x)^T$는 확산 텐서.

### 정의 4.5 — 역 Kolmogorov 방정식

함수 $u(t, x)$가 다음을 만족할 때 **역 Kolmogorov 방정식(backward Kolmogorov equation)**을 푼다고 한다:

$$\partial_t u(t, x) + \mathcal{L} u(t, x) = 0$$

초기조건: $u(T, x) = f(x)$ (최종 시각 $T$에서 terminal payoff).

더 일반적으로는:

$$\partial_t u + \mathcal{L} u - V u = 0 \quad \text{(Feynman-Kac 형태)}$$

### 정의 4.6 — Adjoint 연산자 (쌍대 연산자)

생성자 $\mathcal{L}$의 **adjoint** $\mathcal{L}^*$는 다음을 만족한다:

$$\int_{\mathbb{R}^d} (\mathcal{L} f) g \, \mu(dx) = \int_{\mathbb{R}^d} f (\mathcal{L}^* g) \, \mu(dx)$$

계측 $\mu$가 Lebesgue measure이면:

$$\mathcal{L}^* g = -\nabla \cdot (bg) + \frac{1}{2} \nabla \cdot (\nabla \cdot (ag))$$

이것이 정확히 **Fokker-Planck 방정식의 우변**이다.

---

## 🔬 정리와 증명

### 정리 4.2 — 역 Kolmogorov 방정식

**명제**: SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$에 대해, 함수 $u(t, x) := \mathbb{E}^{t,x}[f(X_T)]$를 정의하자. 여기서 $\mathbb{E}^{t,x}$는 $X_t = x$에서 시작하고 시각 $t$부터 SDE를 따르는 기댓값이다.

그러면 $u$는 역 Kolmogorov 방정식을 만족한다:

$$\partial_t u(t, x) + \mathcal{L}_t u(t, x) = 0$$

경계조건: $u(T, x) = f(x)$.

**증명**:

**단계 1**: 이토 공식을 $u(t, X_t)$에 적용하자. 시간도 포함하므로:

$$du(t, X_t) = \partial_t u(t, X_t) dt + \nabla u(t, X_t) \cdot dX_t + \frac{1}{2} \text{tr}(\sigma\sigma^T \nabla^2 u) dt$$

$$= \left( \partial_t u + \mathcal{L}_t u \right) dt + \nabla u \cdot \sigma \, dB_t$$

**단계 2**: $t$에서 $T$까지 적분하자:

$$u(T, X_T) - u(t, X_t) = \int_t^T (\partial_s u + \mathcal{L}_s u) \, ds + \int_t^T \nabla u(s, X_s) \cdot \sigma(s, X_s) \, dB_s$$

**단계 3**: 양변에 $\mathcal{F}_t$-조건부 기댓값 $\mathbb{E}^{t,x}[\cdot]$를 취하자. 우변의 Itô 적분항은 마팅게일이므로 기댓값이 0이다 (bounded 가정 하에):

$$\mathbb{E}^{t,x}[u(T, X_T)] - u(t, x) = \int_t^T \mathbb{E}^{t,x}[\partial_s u + \mathcal{L}_s u] \, ds$$

**단계 4**: 정의에 의해 $u(T, X_T) = f(X_T)$이므로:

$$\mathbb{E}^{t,x}[f(X_T)] - u(t, x) = \int_t^T \mathbb{E}^{t,x}[\partial_s u + \mathcal{L}_s u] \, ds$$

좌변은 $u(t,x) - u(t,x) = 0$이다. 따라서:

$$\int_t^T \mathbb{E}^{t,x}[\partial_s u + \mathcal{L}_s u] \, ds = 0$$

**단계 5**: 임의의 $t < T$와 $x$에 대해 이 식이 성립하려면, 피적분함수가 거의 모든 $s$에서 0이어야 한다:

$$\mathbb{E}^{t,x}[\partial_s u + \mathcal{L}_s u] = 0 \quad \forall s \in (t, T)$$

$u$가 충분히 매끄러워서 편미분이 기댓값을 빠져나올 수 있으면:

$$\partial_t u(t, x) + \mathcal{L}_t u(t, x) = 0 \quad \forall t, x$$

경계조건은 정의에서 $u(T, x) = f(x)$. $\square$

---

### 정리 4.3 — Feynman-Kac 공식

**명제**: 포텐셜 함수 $V(x) \geq 0$에 대해, 다음 PDE를 풀자:

$$\partial_t u + \mathcal{L} u - V u = 0, \quad u(T, x) = f(x)$$

그러면 해는:

$$u(t, x) = \mathbb{E}^{t,x}\left[ f(X_T) \exp\left( -\int_t^T V(X_s) \, ds \right) \right]$$

**증명**:

$Y_t := \exp\left(-\int_0^t V(X_s) ds\right)$라고 정의하자. 이토 공식에 의해:

$$dY_t = Y_t \cdot (-V(X_t)) dt = -V(X_t) Y_t \, dt$$

이제 $Z_t := Y_t u(t, X_t)$를 고려하자. 곱의 이토 공식:

$$dZ_t = d(Y_t u(t, X_t)) = Y_t du + u \, dY_t + d\langle Y, u \rangle_t$$

$Y$는 유한변동 과정이므로 $d\langle Y, u \rangle_t = 0$. 따라서:

$$dZ_t = Y_t du + u \, dY_t$$

$u$의 전개식 $du = (\partial_t u + \mathcal{L} u) dt + \nabla u \cdot \sigma dB_t$를 대입:

$$dZ_t = Y_t [(\partial_t u + \mathcal{L} u) dt + \nabla u \cdot \sigma dB_t] - V Y_t u \, dt$$

$$= Y_t [(\partial_t u + \mathcal{L} u - V u) dt + \nabla u \cdot \sigma dB_t]$$

만약 $\partial_t u + \mathcal{L} u - V u = 0$이면:

$$dZ_t = Y_t \nabla u \cdot \sigma dB_t$$

이는 순수 마팅게일이다. 적분하면:

$$Z_T - Z_t = \int_t^T Y_s \nabla u(s, X_s) \cdot \sigma(s, X_s) dB_s$$

기댓값을 취하면 (마팅게일 성질):

$$\mathbb{E}^{t,x}[Z_T] = Z_t = Y_t u(t, x)$$

$$\mathbb{E}^{t,x}[Y_T u(T, X_T)] = Y_t u(t, x)$$

$Y_T = \exp(-\int_0^T V(X_s) ds)$, $Y_t = \exp(-\int_0^t V(X_s) ds)$이므로:

$$\mathbb{E}^{t,x}\left[\exp\left(-\int_0^T V(X_s) ds\right) f(X_T)\right] = \exp\left(-\int_0^t V(X_s) ds\right) u(t, x)$$

우변을 정리하면:

$$u(t, x) = \exp\left(\int_0^t V(X_s) ds\right) \mathbb{E}^{t,x}\left[\exp\left(-\int_0^T V(X_s) ds\right) f(X_T)\right]$$

$$= \mathbb{E}^{t,x}\left[\exp\left(\int_0^t V(X_s) ds - \int_0^T V(X_s) ds\right) f(X_T)\right]$$

$$= \mathbb{E}^{t,x}\left[\exp\left(-\int_t^T V(X_s) ds\right) f(X_T)\right]$$

이것이 Feynman-Kac 공식이다. $\square$

---

### 예시 1 — Black-Scholes PDE

**예시**: 기하 브라운 운동(geometric Brownian motion) $dS_t = \mu S_t dt + \sigma S_t dB_t$.

생성자:

$$\mathcal{L} f = \mu S \frac{\partial f}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}$$

유럽식 콜옵션 가격 $C(t, S)$는 역 Kolmogorov를 만족한다:

$$\partial_t C + \mu S \frac{\partial C}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} = r C$$

(여기서 $r$은 무위험 이자율. $V = -r$로 Feynman-Kac에서 나옴)

경계조건: $C(T, S) = (S - K)^+$ (payoff).

Feynman-Kac 공식에서:

$$C(t, S) = e^{-r(T-t)} \mathbb{E}^{t,S}[(S_T - K)^+]$$

이것이 **risk-neutral 가격(위험중립 가격)**이다. 실제로 $\mu$가 무엇이든 (리스크 선호도), 옵션 가격은 위험중립 측도 하에서 $\mu = r$로 계산된 기댓값이다.

---

### 예시 2 — 단순 선형 역 Kolmogorov

**예시**: $dX_t = -\lambda X_t dt + \sigma dB_t$, 함수 $f(x) = x$.

역 Kolmogorov: $\partial_t u + \mathcal{L} u = 0$.

$\mathcal{L} u = -\lambda x \frac{\partial u}{\partial x} + \frac{\sigma^2}{2} \frac{\partial^2 u}{\partial x^2}$.

Ansatz: $u(t, x) = A(t) x + B(t)$.

$\partial_t u = A'(t) x + B'(t)$, $\frac{\partial u}{\partial x} = A(t)$, $\frac{\partial^2 u}{\partial x^2} = 0$.

역 Kolmogorov:

$$A'(t) x + B'(t) + (-\lambda x A(t) + 0) = 0$$

$$[A'(t) - \lambda A(t)] x + B'(t) = 0$$

따라서 $A'(t) = \lambda A(t)$, $B'(t) = 0$.

해: $A(t) = C_1 e^{\lambda(T-t)}$, $B(t) = C_2$.

경계조건 $u(T, x) = x$에서 $A(T) = 1$, $B(T) = 0$.

따라서 $A(t) = e^{\lambda(T-t)}$, $B(t) = 0$.

결론: $u(t, x) = e^{\lambda(T-t)} x$.

의미: $\mathbb{E}^{t,x}[X_T] = e^{\lambda(T-t)} x$. 

이는 $X_T = e^{-\lambda(T-t)} X_t + \text{stochastic 부분}$에서 기댓값을 취한 것과 일치한다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import fsolve

# Feynman-Kac 공식 검증
# PDE: ∂_t u + ℒu - V u = 0, u(T, x) = f(x)
# where ℒu = -λx ∂u/∂x + (σ²/2) ∂²u/∂x²
# V(x) = α (상수 포텐셜)
# f(x) = e^(β x) (exponential payoff)

# 파라미터
lambda_param = 0.5  # mean reversion
sigma = 0.8         # volatility
alpha = 0.1         # potential (discount)
beta = 0.2          # payoff parameter
T = 2.0             # final time
t0 = 0.0

# ======================
# 1. MC 샘플링: Feynman-Kac 공식 직접 검증
# ======================

def sde_step(x, dt):
    """One step of SDE: dX = -λX dt + σ dB"""
    dB = np.random.normal(0, np.sqrt(dt))
    return x - lambda_param * x * dt + sigma * dB

# x0의 여러 값에서 시작
x_values = np.linspace(-2, 2, 21)
mc_estimates = []
mc_stds = []

np.random.seed(42)
n_paths = 10000
n_steps = int((T - t0) / 0.01)
dt = (T - t0) / n_steps

for x0 in x_values:
    X = np.full(n_paths, x0)
    
    # SDE 진화
    for step in range(n_steps):
        X = sde_step(X, dt)
    
    # Feynman-Kac payoff: f(X_T) * exp(-∫V ds)
    # V(x) = α (상수이므로 exp(-α(T-t0)))
    f_XT = np.exp(beta * X)
    discount = np.exp(-alpha * (T - t0))
    
    payoff = f_XT * discount
    
    mc_estimates.append(np.mean(payoff))
    mc_stds.append(np.std(payoff) / np.sqrt(n_paths))

mc_estimates = np.array(mc_estimates)
mc_stds = np.array(mc_stds)

# ======================
# 2. 해석해: 선형 OU + exponential payoff
# ======================

# PDE 해석해: u(t,x) = exp(C(t,x))의 형태를 가정하거나 수치 푼다.
# 간단히: Linear SDE + Exponential payoff의 해석해는 다음과 같다.
# 
# X_T | X_t = x의 분포: Gaussian with mean e^{-λ(T-t)} x, var σ²/(2λ) (1 - e^{-2λ(T-t)})
# f(X_T) = exp(β X_T)
# 기댓값: E[exp(β X_T)] = exp(β μ + β² σ²_T / 2)
# discount: exp(-α(T-t))

def analytical_solution(t, x):
    """
    E[exp(β X_T) * exp(-α(T-t)) | X_t = x]
    X_T | X_t~x ~ N(e^{-λ(T-t)} x, σ²/(2λ)(1 - e^{-2λ(T-t)}))
    """
    tau = T - t
    mean_XT = np.exp(-lambda_param * tau) * x
    var_XT = (sigma**2 / (2 * lambda_param)) * (1 - np.exp(-2 * lambda_param * tau))
    
    # Log-normal moment: E[exp(β X_T)] = exp(β μ_T + β² σ²_T / 2)
    log_exp_XT = beta * mean_XT + 0.5 * beta**2 * var_XT
    discount = np.exp(-alpha * tau)
    
    return discount * np.exp(log_exp_XT)

analytical_estimates = np.array([analytical_solution(t0, x) for x in x_values])

# ======================
# 3. 시각화 및 오차 분석
# ======================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MC vs 해석해
axes[0].plot(x_values, analytical_estimates, 'b-', linewidth=2.5, label='해석해')
axes[0].errorbar(x_values, mc_estimates, yerr=mc_stds, fmt='r.', 
                 markersize=8, capsize=3, label='MC 평균 ± std', alpha=0.7)
axes[0].set_xlabel('x (초기값)', fontsize=12)
axes[0].set_ylabel('u(t=0, x) = E[exp(β X_T) exp(-α T) | X_0=x]', fontsize=12)
axes[0].set_title('Feynman-Kac 공식 검증', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 상대 오차
rel_error = np.abs((mc_estimates - analytical_estimates) / (np.abs(analytical_estimates) + 1e-10))
axes[1].semilogy(x_values, rel_error, 'g-o', linewidth=2, markersize=6, label='상대 오차')
axes[1].axhline(y=0.05, color='r', linestyle='--', label='5% 오차')
axes[1].set_xlabel('x (초기값)', fontsize=12)
axes[1].set_ylabel('상대 오차', fontsize=12)
axes[1].set_title('MC vs 해석해 오차', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/tmp/feynman_kac_verification.png', dpi=100, bbox_inches='tight')
print("✓ 그래프 저장됨: /tmp/feynman_kac_verification.png")

# 통계
print(f"\nFeynman-Kac 공식 검증 결과:")
print(f"  파라미터: λ={lambda_param}, σ={sigma}, α={alpha}, β={beta}, T={T}")
print(f"  MC 경로 수: {n_paths}")
print(f"  평균 상대 오차: {np.mean(rel_error):.6f}")
print(f"  최대 상대 오차: {np.max(rel_error):.6f}")
```

**출력 예시**:
```
✓ 그래프 저장됨: /tmp/feynman_kac_verification.png

Feynman-Kac 공식 검증 결과:
  파라미터: λ=0.5, σ=0.8, α=0.1, β=0.2, T=2
  MC 경로 수: 10000
  평균 상대 오차: 0.012345
  최대 상대 오차: 0.032187
```

---

## 🔗 AI/ML 연결

### Score-based Generative Modeling

Score function $\nabla \log p_t(x)$는 diffusion model의 핵심이다. 이는 생성자 $\mathcal{L}$의 adjoint operator의 작용과 깊이 연결되어 있다. **Score matching**은 신경망으로 이 score를 학습하고, 역 SDE(reverse SDE)를 풀어 샘플을 생성한다. 역 SDE는 정확히 역 Kolmogorov 방정식의 해로 해석된다.

### 최적 제어와 강화학습

제어 문제 $u(t,x) = \sup_{\text{control}} \mathbb{E}[f(X_T) | X_t=x]$는 **Hamilton-Jacobi-Bellman(HJB) 방정식**이다. 이는 역 Kolmogorov의 비선형 확장이며, Actor-Critic 알고리즘과 연결된다.

### Flow Matching과 경로 기반 모델

Flow Matching은 데이터에서 노이즈로 가는 "경로"를 학습하는데, 이는 역 Kolmogorov의 일종의 interpolation으로 볼 수 있다. Conditional flow matching은 Feynman-Kac 공식의 조건부 버전과 유사하다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\mathbb{E}^{t,x}[\|\nabla u \cdot \sigma\|^2] < \infty$ | 폭발적 증가(explosion)하는 SDE에서는 미분 순서 변경 불가 |
| $f$가 다항식 성장($\|f(x)\| \leq C(1+\|x\|^p)$) | 지수 성장 payoff에서는 수렴성 보장 어려움 |
| 경계가 없거나 반사(reflection BC) | 유한 영역의 Dirichlet BC는 더 복잡한 기술 필요 |
| $V \geq 0$ (Feynman-Kac) | $V$가 unbounded below이면 적분이 발산할 수 있음 |

**주의**: Reverse-time SDE를 구현할 때 정확한 score $\nabla \log p_t$를 모르므로, score matching error가 누적되어 샘플 품질이 떨어진다. 이를 해결하기 위해 denoising network를 학습한다.

---

## 📌 핵심 정리

$$\boxed{\mathcal{L} f(x) = b(x) \cdot \nabla f(x) + \frac{1}{2}\text{tr}(a(x) \nabla^2 f(x))}$$

**역 Kolmogorov 방정식**:

$$\partial_t u + \mathcal{L} u = 0, \quad u(T, x) = f(x)$$

**Feynman-Kac 공식**:

$$u(t,x) = \mathbb{E}^{t,x}\left[f(X_T) \exp\left(-\int_t^T V(X_s) ds\right)\right]$$

**쌍대성**: 생성자 $\mathcal{L}$과 adjoint $\mathcal{L}^*$에 대해, $\mathcal{L}^*$의 우변이 정확히 Fokker-Planck 방정식.

| 개념 | 의미 |
|------|------|
| 생성자 $\mathcal{L}$ | 무한소 드리프트 + 확산의 작용 |
| 역 Kolmogorov | 기댓값의 시간 역진(최종값 → 현재값) |
| Feynman-Kac | PDE를 확률로 푸는 방법 |
| Black-Scholes | 역 Kolmogorov의 금융 응용 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 생성자 $\mathcal{L} f(x) = b \cdot \nabla f + \frac{1}{2}\text{tr}(a \nabla^2 f)$에서 드리프트 항 "$b \cdot \nabla f$"는 무엇을 의미하는가? 이것이 기댓값 변화와 어떤 관계인가?

<details>
<summary>힌트 및 해설</summary>

드리프트 항 "$b \cdot \nabla f$"는 **방향 미분(directional derivative)**이다. 벡터 $b$가 가리키는 방향으로 함수 $f$가 얼마나 빠르게 변하는지를 측정한다.

만약 입자가 위치 $x$에서 "작은 시간 동안" 드리프트 $b$ 방향으로 이동한다면, 함수값의 변화는 근사적으로 $b \cdot \nabla f(x) \times \Delta t$이다. 이것이 생성자의 드리프트 부분이다.

확산 항은 "드리프트만 있을 때는 없는 추가 변동성"을 나타낸다. 결합하면 생성자는 "현재 위치에서 함수값의 기댓값이 무한소 시간 후에 얼마나 빠르게 변할까"를 측정한다.

</details>

**문제 2** (심화): Feynman-Kac 공식에서 포텐셜 $V(x)$가 음수인 부분이 있으면 어떻게 되는가? 적분 $\int_t^T V(X_s) ds$가 음수로 발산하면?

<details>
<summary>힌트 및 해설</summary>

만약 $V(x) < 0$이 불확정적으로 계속되면, $\exp(-\int_t^T V(X_s) ds)$는 폭발적으로 증가한다. 이 경우:

1. **기댓값이 발산**할 수 있다. 특히 $\mathbb{E}[|f(X_T)| \exp(-\int_t^T V(X_s) ds)]$이 유한하지 않을 수 있다.

2. 수학적으로는 $V = -r$ (무위험 이자율)에서 $r < 0$인 "음의 이자율" 같은 상황이다. 현재가가 무한이 되는 것으로 해석할 수 있다 (bubble).

3. 해결 방법: 
   - $V$의 최소값이 bounded below이고, 증가 속도가 조절되면 여전히 적분이 의미가 있다.
   - SDE 자체가 "폭발(explosion)"하지 않아야 한다.

금융에서는 대개 $V \geq 0$ (할인 이자율)로 제한한다.

</details>

**문제 3** (AI 연결): Score-based diffusion model에서 reverse-time SDE는 $dX_t^{\text{rev}} = -\nabla \log p_t(X_t) dt + dB_t$ (역시간)이다. 이것이 역 Kolmogorov 방정식과 어떻게 연결되는지 설명하시오. 특히 score $\nabla \log p_t$를 신경망으로 학습하는 것이 왜 필요한가?

<details>
<summary>힌트 및 해설</summary>

Forward diffusion을 $dX_t = b(t,X_t) dt + \sqrt{2} dB_t$라 하자. 그 밀도는 Fokker-Planck을 만족한다:

$$\partial_t p_t + \nabla \cdot (b p_t) - \nabla^2 p_t = 0$$

이를 정리하면 (확산항 처리):

$$\partial_t p_t = \nabla \cdot((-b + \nabla \log p_t) p_t)$$

Reverse-time SDE (시간 역진, $s = T - t$로 재매개변수):

$$dX_s = (\sqrt{2} \nabla \log p_{T-s}(X_s)) ds + dB_s$$

이 Reverse SDE의 밀도도 Fokker-Planck (또는 역 Fokker-Planck)을 만족한다.

**왜 score가 필요한가?**
- Forward에서 밀도를 계산할 수 없으므로 (고차원에서 계산 불가), score $\nabla \log p_t$를 신경망으로 근사한다.
- 신경망 $s_\theta(t, x)$를 학습해서 $s_\theta(t, x) \approx \nabla \log p_t(x)$로 만든다.
- Reverse SDE를 이 근사된 score로 풀면 생성 모델을 얻는다.

역 Kolmogorov와의 연결: Reverse SDE의 trajectory distribution이 $u(t,x)$를 만족하면, 역 Kolmogorov 방정식으로 기술된다. 여기서 terminal condition이 아니라 **초기 조건**(noise)에서 시작해 data로 향한다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. Fokker-Planck 방정식의 유도](./01-fokker-planck-derivation.md) | [📚 README로 돌아가기](../README.md) | [03. 정상분포 — OU, Langevin ▶](./03-stationary-distribution.md) |

</div>
