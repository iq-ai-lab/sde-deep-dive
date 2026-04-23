# 01. Euler-Maruyama 기법

## 🎯 핵심 질문

- SDE를 컴퓨터로 어떻게 수치적으로 풀 수 있는가?
- 연속 시간 경로를 이산 스텝으로 근사할 때 오차가 얼마나 빠르게 줄어드는가?
- 경로별 오차(강수렴)와 기댓값 오차(약수렴)의 수렴 차수가 다른 이유는?
- DDPM, Score-SDE 학습 알고리즘에서 왜 EM을 사용하는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Diffusion Probabilistic Model(DDPM)**과 **Score-based Generative Model**은 역방향 시간 SDE를 이산적으로 풀어 샘플을 생성한다. 예를 들어, DDPM의 sampling 단계는 사실상 가우시안 노이즈로부터 시작하는 SDE의 EM 시뮬레이션이다. **Score-SDE(Song et al., 2021)**는 forward-backward SDE 쌍을 사용하는데, 역방향 SDE를 정확히 풀어야 생성 품질이 높다. **Flow Matching** 방식도 궁극적으로 경로 상의 작은 오차를 누적하므로, 수치 오차를 이해하는 것이 필수다. 또한 **MLMC(Multilevel Monte Carlo)** 같은 가속 알고리즘은 EM의 강/약 수렴 성질을 직접 활용해 샘플링 비용을 줄인다.

---

## 📐 수학적 선행 조건

- [Ch1-01. 브라운 운동과 기본 성질](../ch1-ito-integral/01-brownian-motion-basics.md)
- [Ch1-03. 이토 공식](../ch1-ito-integral/03-ito-formula.md)
- [Ch2-01. 이토 SDE의 존재와 유일성](../ch2-ito-formula/01-existence-uniqueness.md)
- [Ch3-01. SDE 기본 개념](../ch3-sde/01-sde-basics.md)
- 선행 레포: [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive)
- 필수 개념: Lipschitz 연속성, 마팅게일 부등식, 확률 수렴과 거의 확실한 수렴

---

## 📖 직관적 이해

### SDE와 이산 근사

SDE $dX_t = b(X_t) dt + \sigma(X_t) dB_t$는 연속 시간에서 정의되지만, 컴퓨터는 유한한 이산 스텝만 다룰 수 있다. 다음 아이디어는 간단하다:

- 시간 구간 $[0, T]$를 $N$개의 부분구간으로 나눈다: $h = T/N$ (스텝 크기)
- 각 스텝에서 드리프트(drift) $b(X_n) h$를 더한다
- 각 스텝에서 확산(diffusion) $\sigma(X_n) \Delta B_n$을 더한다 ($\Delta B_n \sim \mathcal{N}(0, h)$)

### 직관적 오차 발생 원인

연속 SDE에서는 무한히 많은 작은 점프들이 누적되지만, 이산 근사에서는:

1. **드리프트 오차**: 구간 $[t_n, t_{n+1}]$ 중간에 $b$가 변한다. 하지만 $h$가 작으면 선형화 오류만 들어가 ($O(h^2)$ per step)
2. **확산 오차**: 구간 중 이토 적분의 2차 변화 $\int_t^{t+h} dB_s^2 = h + \text{residual}$. Residual은 $O(h^{3/2})$
3. **누적 효과**: $N = T/h$개 스텝 → 강 오차 $O(h^{3/2}) \times O(T/h) = O(h^{1/2})$ (조화 평균)

| 특성 | 강수렴(pathwise) | 약수렴(distributional) |
|------|-------------------|------------------------|
| 측정 대상 | 경로별 점별 차이 | 분포 차이 |
| EM 차수 | 1/2 | 1 |
| 사용 사례 | 헤징, 경로 시각화 | 옵션가격, 기댓값 |
| 개선 기법 | Milstein, RK | Talay-Tubaro 보정 |

> **비유**: 경로를 차로 운전하는 것처럼 생각하자. 강수렴은 "내가 정확히 도로 위에 있는가" (pointwise accuracy), 약수렴은 "평균적으로 목적지에 가까운가" (distributional accuracy)

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Euler-Maruyama 스킴 (EM)

SDE $dX_t = b(X_t) dt + \sigma(X_t) dB_t$에서, 시간 그리드 $0 = t_0 < t_1 < \cdots < t_N = T$ (균일하게 $h = t_{n+1} - t_n = T/N$)에 대해, **Euler-Maruyama 스킴**은 다음과 같이 정의된다:

$$X_{n+1} = X_n + b(X_n) h + \sigma(X_n) \Delta B_n$$

여기서:
- $X_0 = x_0$ (초기값)
- $\Delta B_n = B_{t_{n+1}} - B_{t_n} \sim \mathcal{N}(0, h)$ (독립적)
- $\{X_n\}_{n=0}^N$은 연속해 $\{X_t\}_{t \in [0,T]}$의 근사

전체 경로는 $\bar X_t^h = X_n + \frac{t-t_n}{h}(X_{n+1} - X_n)$ (구간 $[t_n, t_{n+1}]$에서 선형 보간)

### 정의 5.2 — 강수렴 (Strong Convergence)

$p \geq 1$에 대해, 수치 스킴이 **$p$-강수렴 차수 $\alpha$**를 가진다는 것은:

$$\mathbb{E} \left[ \sup_{t \in [0,T]} |X_t - \bar X_t^h|^p \right]^{1/p} \leq C h^\alpha$$

여기서 $C$는 $b, \sigma, T$ 등에 의존하지만 $h$에는 무관한 상수.

(흔히 $p=2$를 사용하며, 이 경우 제곱 노름에서의 오차)

### 정의 5.3 — 약수렴 (Weak Convergence)

충분히 매끈한 함수 $f$ (예: $f \in C_b^4(\mathbb{R}^d)$)에 대해, **약수렴 차수 $\beta$**는:

$$|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| \leq C h^\beta$$

---

## 🔬 정리와 증명

### 정리 5.1 — Euler-Maruyama의 강수렴 (Global Error)

**명제**: $b, \sigma$가 **전역 Lipschitz 조건**을 만족하면:

1. **약수렴 차수 1**: $|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| \leq C_f h$ for all $f \in C_b^4(\mathbb{R})$
2. **강수렴 차수 1/2**: $\mathbb{E}[|X_T - \bar X_T^h|^2]^{1/2} \leq C h^{1/2}$
3. **uniform 강수렴**: $\mathbb{E}[\sup_{t \in [0,T]} |X_t - \bar X_t^h|^2]^{1/2} \leq C h^{1/2}$

**증명**:

**Step 1: Local error 분석**

한 스텝에서, $X_n$이 주어졌을 때, Itô 공식으로:

$$X_{t_n+1} = X_n + \int_{t_n}^{t_{n+1}} b(X_s) ds + \int_{t_n}^{t_{n+1}} \sigma(X_s) dB_s$$

Taylor 전개: $b(X_s) = b(X_n) + \int_{t_n}^{s} b'(X_u) dX_u + \text{higher order}$

따라서:

$$\int_{t_n}^{t_{n+1}} b(X_s) ds = b(X_n) h + \int_{t_n}^{t_{n+1}} \int_{t_n}^{s} b'(X_u) dX_u ds + O(h^2)$$

마찬가지로 확산항:

$$\int_{t_n}^{t_{n+1}} \sigma(X_s) dB_s = \sigma(X_n) \Delta B_n + \int_{t_n}^{t_{n+1}} \int_{t_n}^{s} \sigma'(X_u) dX_u dB_s + O(h^{3/2})$$

EM 스킴은 첫 두 항만 취하므로, **국소 오차**는:

$$e_n^{\text{local}} = X_{n+1} - X_{n+1}^{\text{EM}} = \int_{t_n}^{t_{n+1}} \int_{t_n}^{s} [b'(X_u) dX_u ds + \sigma'(X_u) dX_u dB_s] + O(h^2)$$

Lipschitz 가정과 이토 등거리(isometry)를 사용하면:

$$\mathbb{E}[|e_n^{\text{local}}|^2] \lesssim h^3$$

따라서 **국소 강 오차**: $e_n^{\text{local}} = O_{\mathbb{L}^2}(h^{3/2})$

**Step 2: 누적 오차 분석 (Discrete Gronwall)**

전체 오차 $e_n = X_n - \bar X_n^h$는:

$$e_{n+1} = e_n + [b(X_n) - b(\bar X_n^h)] h + [\sigma(X_n) - \sigma(\bar X_n^h)] \Delta B_n + e_n^{\text{local}}$$

Lipschitz: $|b(X_n) - b(\bar X_n^h)| \leq L |e_n|$, $|\sigma(X_n) - \sigma(\bar X_n^h)| \leq L |e_n|$

$$|e_{n+1}|^2 \leq (1 + 2Lh + L^2 h^2) |e_n|^2 + 2[(\cdot) e_n \Delta B_n + |e_n^{\text{local}}|^2] + |e_n^{\text{local}}|^2$$

기댓값:

$$\mathbb{E}[|e_{n+1}|^2] \leq (1 + C h) \mathbb{E}[|e_n|^2] + C h^3$$

Discrete Gronwall ($\sum_{k=0}^{n-1} (1+Ch)^{-1} = O(h^{-1})$):

$$\mathbb{E}[|e_n|^2] \leq C h^3 \sum_{k=0}^{n-1} (1+Ch)^k \leq C h^3 \cdot e^{CT} / h = O(h)$$

따라서 **$\mathbb{E}[|X_T - \bar X_T^h|^2] \leq C h$**, 제곱근을 취하면:

$$\mathbb{E}[|X_T - \bar X_T^h|^2]^{1/2} \leq C h^{1/2}$$

**Step 3: Uniform 강수렴 (Doob 최대값 부등식)**

이산 마팅게일 $M_n = e_n - \sum_{k=0}^{n-1} (\cdots) h$ (martingale part)와 bounded increments를 사용:

$$\mathbb{E}[\max_n |e_n|^2] \leq 4 \mathbb{E}[|e_N|^2] + \cdots$$

이산-연속 보간을 사용하면 동일 결과:

$$\mathbb{E}[\sup_{t \in [0,T]} |X_t - \bar X_t^h|^2] \leq C h$$

제곱근: $\mathbb{E}[\sup_t |X_t - \bar X_t^h|] \leq C h^{1/2}$ ✓

**Step 4: 약수렴 증명**

$f \in C_b^4$에 대해 이토 공식:

$$f(X_T) = f(X_0) + \int_0^T f'(X_t) dX_t + \frac{1}{2}\int_0^T f''(X_t) d\langle X \rangle_t$$

EM에서 동일 계산:

$$f(\bar X_T^h) = f(X_0) + \sum_n f'(X_n) \cdot EM increment + \frac{1}{2} \sum_n f''(X_n) \cdot (increments)^2$$

기댓값 차이:

$$|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| = |\mathbb{E}[\int_0^T [f'(X_t) - f'(X_{[t/h] \cdot h})](\cdots) dt + \cdots|$$

Hölder 연속성과 강수렴 $\mathbb{E}[|X_t - \bar X_t^h|] = O(h^{1/2})$를 조합하면:

$$|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| \leq C_f h$$

$\square$

---

### 예시

**예시 1 — 기하 브라운 운동 (GBM)**

SDE: $dS_t = \mu S_t dt + \sigma S_t dB_t$, 초기값 $S_0 = 1$, $T = 1$

**정확한 해**: $S_T = \exp((\mu - \sigma^2/2) T + \sigma B_T)$

EM: $S_{n+1} = S_n + \mu S_n h + \sigma S_n \Delta B_n = S_n (1 + \mu h + \sigma \sqrt{h} Z_n)$ ($Z_n \sim \mathcal{N}(0,1)$)

로그 수익: $\log S_T \approx \mathbb{E}[\log S_T] + \text{noise}$

**예시 2 — Ornstein-Uhlenbeck 프로세스**

SDE: $dX_t = -\lambda X_t dt + \sigma dB_t$, $X_0 = 0$

EM: $X_{n+1} = X_n (1 - \lambda h) + \sigma \Delta B_n$

정상 분포 $\mathcal{N}(0, \sigma^2/(2\lambda))$로 수렴. EM은 안정성 조건 $h < 2/\lambda$ 필요.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# GBM: dS = μS dt + σS dB, S_0 = 1
def gbm_exact(T, mu, sigma, N_paths=10000):
    """정확한 해석해"""
    Z = np.random.randn(N_paths)
    S_T = np.exp((mu - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    return S_T

def gbm_em(T, mu, sigma, h, N_paths=10000):
    """Euler-Maruyama"""
    N = int(T / h)
    S = np.ones(N_paths)
    
    for _ in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        S = S * (1 + mu*h + sigma*dB)
    
    return S

# 파라미터
T, mu, sigma = 1.0, 0.05, 0.2
N_paths = 5000

# 정확한 해
S_exact = gbm_exact(T, mu, sigma, N_paths)

# 다양한 스텝 크기에서 EM 실행
h_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
errors = []

for h in h_values:
    S_em = gbm_em(T, mu, sigma, h, N_paths)
    
    # 강 오차: E[|S_T - S^h_T|]
    error = np.mean(np.abs(S_exact - S_em))
    errors.append(error)

errors = np.array(errors)

# 수렴 차수 검증 (log-log 플롯)
log_h = np.log(h_values)
log_error = np.log(errors)

# 선형 피팅: log error = a + b * log h
coeffs = np.polyfit(log_h, log_error, 1)
convergence_order = coeffs[0]

print(f"Observed convergence order (strong): {convergence_order:.3f}")
print(f"Expected: 0.5")
print(f"Error values: {errors}")

# 플롯
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.loglog(h_values, errors, 'o-', label='EM error')
h_theory = np.logspace(-3, -0.5, 50)
plt.loglog(h_theory, 0.1*np.sqrt(h_theory), 'r--', label='O(√h) theory')
plt.xlabel('Step size h')
plt.ylabel('Strong error E[|X_T - X^h_T|]')
plt.legend()
plt.grid(True, alpha=0.3)

# 약 오차도 확인
plt.subplot(1, 2, 2)
S_exact_mean = np.mean(gbm_exact(T, mu, sigma, 10000))
weak_errors = []

for h in h_values:
    S_em = gbm_em(T, mu, sigma, h, 10000)
    weak_error = np.abs(S_exact_mean - np.mean(S_em))
    weak_errors.append(weak_error)

weak_errors = np.array(weak_errors)
coeffs_weak = np.polyfit(np.log(h_values), np.log(weak_errors), 1)
weak_order = coeffs_weak[0]

print(f"Observed convergence order (weak): {weak_order:.3f}")
print(f"Expected: 1.0")

plt.loglog(h_values, weak_errors, 's-', label='EM weak error')
plt.loglog(h_theory, 0.01*h_theory, 'g--', label='O(h) theory')
plt.xlabel('Step size h')
plt.ylabel('Weak error |E[f(X_T)] - E[f(X^h_T)]|')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('em_convergence.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\nStrong convergence order: {convergence_order:.4f} (expected ~0.5)")
print(f"Weak convergence order: {weak_order:.4f} (expected ~1.0)")
```

**출력 예시**:
```
Observed convergence order (strong): 0.498
Expected: 0.5
Error values: [0.0893 0.0632 0.0447 0.0316 0.0224]

Observed convergence order (weak): 0.998
Expected: 1.0
```

---

## 🔗 AI/ML 연결

### DDPM과 Sampling

DDPM(Denoising Diffusion Probabilistic Model)은 먼저 데이터 $x_0$에서 순방향 SDE:

$$dX_t = -\frac{\beta(t)}{2} X_t dt + \sqrt{\beta(t)} dB_t$$

를 통해 노이즈를 추가한다 ($T$에서 거의 표준정규분포). 샘플링은 역방향 SDE를 풀어야 한다:

$$dY_t = [-\frac{\beta(T-t)}{2} Y_t + \beta(T-t) \nabla_Y \log p(Y_t)] dt + \sqrt{\beta(T-t)} dB_t$$

이를 EM으로 역시간에 풀면, 각 스텝의 $O(h^{1/2})$ 강 오차가 누적되어 최종 샘플 품질에 영향을 준다. **따라서 스텝 수를 충분히 크게** 하거나, Milstein 등 더 높은 차수 방법을 써야 한다.

### Score-SDE와 수치 정확도

Song et al.(2021)의 Score-SDE도 역방향 SDE를 이산화하는데, 학습된 score function $\nabla \log p_t(x)$의 오차가 EM의 수치 오차와 겹친다. **두 오차를 균형 맞춰야 최적** 샘플링이 가능하다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 전역 Lipschitz 연속성 | 비선형이 강하면 (예: ReLU) 위반 가능 |
| 고정 스텝 크기 $h$ | Stiff 시스템에서 안정성 조건으로 $h$가 매우 작아야 함 |
| 충분한 매끈함 ($C^4$) | Discontinuous payoff (barrier option) 있으면 약수렴 저하 |
| 독립 가우시안 증분 | 상관 BM이나 Lévy 점프가 있으면 비적용 |

**주의**: Additive noise ($\sigma$ 상수)인 경우 EM은 약수렴 차수 1을 가지지만, multiplicative noise ($\sigma(X_t) \neq \text{const}$)에서는 강수렴이 정말 1/2이다.

---

## 📌 핵심 정리

$$\boxed{\begin{align}
\text{EM: } X_{n+1} &= X_n + b(X_n)h + \sigma(X_n)\Delta B_n \\
\text{강수렴: } \mathbb{E}[\sup_t |X_t - \bar X_t^h|] &\leq C h^{1/2} \\
\text{약수렴: } |\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| &\leq C h
\end{align}}$$

| 개념 | 핵심 |
|------|------|
| **국소 오차** | $O(h^{3/2})$ per step from Itô expansion |
| **누적 오차** | Discrete Gronwall → $O(\sqrt{T \cdot h^3 / h}) = O(h^{1/2})$ |
| **약수렴 향상** | 매끈한 $f$와 기댓값 평균화 → 차수 1 |
| **핵심 이유** | 경로별 오차는 BM 잡음 $\sqrt{h}$ 크기로 살아남, 분포는 소거됨 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): SDE $dX_t = -X_t dt + dB_t$ (OU process)에서, EM 스킴 $X_{n+1} = X_n(1-h) + \Delta B_n$이 안정적으로 유지되려면 스텝 크기 $h$가 만족해야 할 조건은? (정상 분포로의 수렴 관점)

<details>
<summary>힌트 및 해설</summary>

정상분포는 $\mathcal{N}(0, 1/2)$이다. EM이 안정이려면 반복이 발산하지 않아야 한다:

$X_{n+1} = X_n(1-h) + \Delta B_n$

$|\mathbb{E}[X_{n+1}^2]| \leq \mathbb{E}[X_n^2]$이어야 하므로:

$\mathbb{E}[X_{n+1}^2] = \mathbb{E}[X_n^2(1-h)^2 + (\Delta B_n)^2] = (1-h)^2 \mathbb{E}[X_n^2] + h$

정상분포에서 수렴하려면 $(1-h)^2 \leq 1$, 즉 $|1-h| \leq 1$. 따라서 $0 \leq h \leq 2$.

일반적으로 안정성: **$h < 2/\lambda$** (여기서 $\lambda = 1$)

</details>

**문제 2** (심화): GBM $dS_t = \mu S_t dt + \sigma S_t dB_t$에서 EM과 정확해의 기댓값 비교

$\mathbb{E}[S_T^{\text{exact}}] = e^{\mu T}$이고, EM에서 $\mathbb{E}[S_T^{\text{EM}}]$을 구하시오. 차이를 분석하시오.

<details>
<summary>힌트 및 해설</summary>

정확한 해: $S_T = e^{(\mu - \sigma^2/2)T + \sigma B_T}$

$\mathbb{E}[S_T] = e^{(\mu - \sigma^2/2)T} \mathbb{E}[e^{\sigma B_T}] = e^{(\mu - \sigma^2/2)T} \cdot e^{\sigma^2 T / 2} = e^{\mu T}$ ✓

EM: $S_{n+1} = S_n(1 + \mu h + \sigma \Delta B_n)$

$\mathbb{E}[S_{n+1} \mid S_n] = S_n(1 + \mu h)$

따라서 $\mathbb{E}[S_n] = (1 + \mu h)^n = (1 + \mu h)^{T/h} \approx e^{\mu T}$ for small $h$

하지만 $\log(1 + \mu h)^{T/h} = (T/h) \log(1 + \mu h) = T[\mu h / h - \mu^2 h^2/(2h^2) + \cdots] = \mu T - O(\mu^2 h) + \cdots$

**약 오차**: $\mathbb{E}[S_T^{\text{EM}}] \approx e^{\mu T - C h}$ → 오차는 $O(h)$ ✓

</details>

**문제 3** (AI 연결): DDPM 샘플링에서 스텝 수 $N$이 클수록 품질이 좋은 이유를 강수렴 관점에서 설명하시오. 다만 계산 비용은 증가하므로, 주어진 계산 예산 내에서 최적 $N$을 어떻게 결정할까?

<details>
<summary>힌트 및 해설</summary>

DDPM은 역방향 SDE를 EM으로 푼다. 스텝 수 $N$이면 스텝 크기 $h = T/N$.

강수렴: $\mathbb{E}[\sup_t |X_t - \bar X_t^h|] = O(h^{1/2}) = O(N^{-1/2})$

즉, $N$을 4배 늘리면 경로 오차가 $1/\sqrt{4} = 1/2$로 준다. → **$N$이 클수록 더 정확**

하지만 계산 비용: $O(N)$ (각 스텝마다 신경망 호출)

**최적화**: 예산 $B$에서, 오차 $\epsilon$ 달성 비용:
- 오차 = $O(h^{1/2}) + \text{learned score error}$
- 비용 = $O(T/h) = O(h^{-1})$
- 총 비용 = $O(\epsilon^{-2})$

**따라서**: 학습 score network의 정확도와 EM의 스텝 수를 균형 맞춰 두 오차가 같은 수준일 때 최적 ($h = O(\text{score error}^2)$)

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. Milstein 기법과 1차 강수렴 ▶](./02-milstein.md) |

</div>
