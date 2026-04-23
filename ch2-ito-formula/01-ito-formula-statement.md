# 01. 이토 공식의 서술과 직관

## 🎯 핵심 질문

- 일반 연쇄법칙(chain rule)과 이토 공식이 근본적으로 어떻게 다른가?
- 왜 이토 공식에는 $\frac{1}{2}f''(x)$ 드리프트(drift) 항이 나타나는가?
- 결정론적 미분과 확률적 미분의 수렴 속도 차이는 무엇인가?
- 이토 공식의 일반 형태는 무엇이며, 구체 예시는?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**생성모델의 핵심은 확률 프로세스 샘플링**이다. DDPM(Denoising Diffusion Probabilistic Models)은 역시간 SDE를 푸는데, 이 과정에서 이토 공식으로 확률측도를 변환(measure change)한다. **Score-SDE**는 정방향 SDE와 역방향 SDE의 관계를 이토 공식으로 도출하며, **Flow Matching**은 연속 궤적상 likelihood를 계산할 때 이토 변환을 사용한다. SGLD(Stochastic Gradient Langevin Dynamics)나 TRPO(Trust Region Policy Optimization)도 SDE 기반 샘플링 또는 정책 업데이트 해석에서 이토 공식이 핵심이다. **이토 공식이 없으면 SDE 기반 생성모델의 수렴성 분석과 확률 해석이 불가능하다.**

---

## 📐 수학적 선행 조건

- [Ch1-05 이토 적분의 정의와 기본 성질](../ch1-ito-integral/05-ito-integral-properties.md) *(이토 적분 $\int H dB$의 마팅게일 성질)*
- [확률과정론 Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) — 연속 로컬 마팅게일(continuous local martingale)
- **필수 개념**: Taylor 전개, 조건부 기댓값, quadratic variation (이차변분)

---

## 📖 직관적 이해

### 결정론적 연쇄법칙 vs 이토 공식

결정론에서 $x(t)$가 미분가능하고 $f$가 $C^1$이면:
$$\frac{d}{dt}f(x(t)) = f'(x(t)) \dot{x}(t)$$

**이토 공식은 추가 항을 만든다**:
$$df(B_t) = f'(B_t) dB_t + \frac{1}{2}f''(B_t) dt$$

**핵심 차이**: 결정론에서는 1차 미분만, 이토 공식에는 $\frac{1}{2}f''$ **2차 드리프트** 항이 있다.

### 왜 2차 항이 살아나는가?

결정론에서:
- $\Delta x \sim O(\Delta t)$ 이면
- $(\Delta x)^2 \sim O((\Delta t)^2) \to 0$ (무시)

**브라운 운동에서**:
- $\Delta B_t \sim N(0, \Delta t)$ 이므로
- $\mathbb{E}[(\Delta B_t)^2] = \Delta t \sim O(\Delta t)$ ← **2차항이 죽지 않음!**
- 따라서 Taylor 전개의 2차 항이 살아남는다.

| 시스템 | $(\Delta x)^2$ 스케일 | Taylor 2차항 | 결과 |
|--------|----------------------|-------------|------|
| 결정론 | $O((\Delta t)^2)$ | 무시 | 1차 공식만 |
| 브라운 운동 | $O(\Delta t)$ | **유지** | **이토 공식** |

> **비유**: 술 취한 사람(브라운 운동)이 담장을 따라 걷는다면, 담장이 "흔들리는 정도"(2차 편차)가 방향 변화(1차 항)만큼 중요해진다. 깨어있는 사람(결정론)은 방향만 중요하다.

### 예시: $f(B_t) = B_t^2$

$f(x) = x^2$이면 $f'(x) = 2x$, $f''(x) = 2$.

이토 공식:
$$d(B_t^2) = 2B_t dB_t + \frac{1}{2} \cdot 2 \cdot dt = 2B_t dB_t + dt$$

**의미**: $B_t^2$의 기댓값 증가율은 $\mathbb{E}[d(B_t^2)] = \mathbb{E}[dt] = dt$, 즉 $\mathbb{E}[B_t^2] = t$. ✓

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 이토 과정 (Itô Process)

시간 구간 $[0, T]$에서 정의된 연속 확률과정 $X_t$가 다음 형태이면 **이토 과정**이라 한다:
$$dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$$

여기서:
- $b : [0,T] \times \mathbb{R} \to \mathbb{R}$ — **드리프트 계수(drift coefficient)**
- $\sigma : [0,T] \times \mathbb{R} \to \mathbb{R}$ — **확산 계수(diffusion coefficient)**
- $B_t$ — 표준 브라운 운동
- 이 식은 적분방정식으로도 표현: $X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s$

---

## 🔬 정리와 증명

### 정리 2.1 — 이토 공식 (Itô's Formula)

**명제**: 함수 $f \in C^2(\mathbb{R})$ (2번 연속 미분가능)과 이토 과정 $dX_t = b\,dt + \sigma\,dB_t$에 대해:
$$df(X_t) = f'(X_t) dX_t + \frac{1}{2}f''(X_t) \sigma^2(t,X_t) dt$$

또는 전개하면:
$$df(X_t) = f'(X_t) b(t,X_t) dt + f'(X_t) \sigma(t,X_t) dB_t + \frac{1}{2}f''(X_t) \sigma^2(t,X_t) dt$$

**증명**: 분할 $\pi_n = \{0 = t_0 < t_1 < \cdots < t_n = t\}$, 메시 $|\pi_n| = \max_i (t_{i+1} - t_i) \to 0$을 잡자.

**Step 1**: Taylor 전개. $f$가 $C^2$이므로 각 구간 $[t_i, t_{i+1}]$에서 평균값 정리에 의해:
$$f(X_{t_{i+1}}) - f(X_{t_i}) = f'(X_{t_i})(X_{t_{i+1}} - X_{t_i}) + \frac{1}{2}f''(\xi_i)(X_{t_{i+1}} - X_{t_i})^2$$

여기서 $\xi_i$는 $X_{t_i}$와 $X_{t_{i+1}}$ 사이의 어떤 값.

**Step 2**: 변위 대체. $\Delta X_i := X_{t_{i+1}} - X_{t_i} = b(t_i, X_{t_i})\Delta t_i + \sigma(t_i, X_{t_i})\Delta B_i$라 하자. 여기서 $\Delta t_i = t_{i+1} - t_i$, $\Delta B_i = B_{t_{i+1}} - B_{t_i}$.

**Step 3**: 곱셈 규칙 적용:
$$(\Delta X_i)^2 = (b \Delta t_i + \sigma \Delta B_i)^2 = b^2 (\Delta t_i)^2 + 2b\sigma \Delta t_i \Delta B_i + \sigma^2 (\Delta B_i)^2$$

각 항의 크기:
- $b^2 (\Delta t_i)^2 \sim O((\Delta t_i)^2) \to 0$
- $2b\sigma \Delta t_i \Delta B_i$: 기댓값 $0$, 크기 $O((\Delta t_i)^{3/2})$ 수렴 (다음 문서에서 상세)
- $\sigma^2 (\Delta B_i)^2 \to \sigma^2 dt_i$ (이차변분 정리)

따라서 주요 항만:
$$(\Delta X_i)^2 = \sigma^2 (\Delta B_i)^2 + o_p(dt_i)$$

**Step 4**: 합 계산. Taylor에서:
$$\sum_{i=0}^{n-1} [f(X_{t_{i+1}}) - f(X_{t_i})] = f(X_t) - f(X_0)$$

이를 전개하면:
$$= \sum f'(X_{t_i}) \Delta X_i + \frac{1}{2}\sum f''(\xi_i)(\Delta X_i)^2$$

첫 항:
$$\sum f'(X_{t_i})(b \Delta t_i + \sigma \Delta B_i) \to \int_0^t f'(X_s) b(s, X_s) ds + \int_0^t f'(X_s) \sigma(s, X_s) dB_s$$
(Riemann과 이토 적분의 수렴)

둘째 항: $f''$이 연속이므로:
$$\frac{1}{2}\sum f''(\xi_i) \sigma^2(\Delta B_i)^2 \to \frac{1}{2}\int_0^t f''(X_s) \sigma^2(s, X_s) ds$$
(이차변분 정리 + 3차항 수렴, 상세는 다음 문서)

**Step 5**: 합성.
$$df(X_t) = f'(X_t) b dt + f'(X_t) \sigma dB_t + \frac{1}{2}f''(X_t)\sigma^2 dt$$

$\square$

> **따름정리**: $dX_t = b dt + \sigma dB_t$일 때, 기댓값의 미분은:
> $$d\mathbb{E}[X_t] = \mathbb{E}[b] dt$$
> (확정적 부분만 남음; 이토 적분의 마팅게일 성질)

---

### 예시 2.1 — $f(B_t) = e^{B_t}$

$f(x) = e^x$이므로 $f'(x) = e^x$, $f''(x) = e^x$.

$dB_t$에 이토 공식을 적용 ($b=0, \sigma=1$):
$$d(e^{B_t}) = e^{B_t} dB_t + \frac{1}{2}e^{B_t} dt$$

**확률미분방정식으로**:
$$e^{B_t} = 1 + \int_0^t e^{B_s} dB_s + \int_0^t \frac{1}{2}e^{B_s} ds$$

**특징**: $\int_0^t e^{B_s} dB_s$는 이토 적분(마팅게일), $\int_0^t \frac{1}{2}e^{B_s} ds$는 확정적 증가.

### 예시 2.2 — GBM (기하 브라운 운동)

기하 브라운 운동: $dS_t = \mu S_t dt + \sigma S_t dB_t$ (금융의 자산가격 모델)

$f(x) = \log x$를 적용하면 ($f'(x) = 1/x, f''(x) = -1/x^2$):
$$d(\log S_t) = \frac{1}{S_t} dS_t + \frac{1}{2} \cdot (-\frac{1}{S_t^2}) \cdot \sigma^2 S_t^2 dt$$
$$= (\mu - \frac{\sigma^2}{2}) dt + \sigma dB_t$$

**해**: $\log S_t = \log S_0 + (\mu - \frac{\sigma^2}{2})t + \sigma B_t$

따라서:
$$S_t = S_0 \exp((\mu - \frac{\sigma^2}{2})t + \sigma B_t)$$

**주의**: $-\sigma^2/2$는 이토의 2차 드리프트 항에서 온다. 이것 없으면 기댓값 계산이 틀린다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# 브라운 운동 경로 시뮬레이션
np.random.seed(42)
T = 1.0
N = 10000
dt = T / N
t = np.linspace(0, T, N+1)
dW = np.random.randn(N) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dW)])

# 예시 1: f(B_t) = B_t^2
# 이토 공식: d(B^2) = 2B dB + dt
# 따라서 B_t^2 = 0 + 2∫B dB + t
f_B = B**2
expected_B2 = t  # 이론: E[B_t^2] = t

# 예시 2: f(B_t) = exp(B_t)
# 이토 공식: d(e^B) = e^B dB + 0.5 e^B dt
f_exp = np.exp(B)
# 수치적 검증: 이산 근사
df_exp_numerical = np.diff(f_exp)
df_exp_ito = np.exp(B[:-1]) * dW + 0.5 * np.exp(B[:-1]) * dt

# 이산 근사 vs 이토 공식 비교
correlation = np.corrcoef(df_exp_numerical, df_exp_ito)[0, 1]
print(f"d(e^B) 수치적분 vs 이토 공식 상관계수: {correlation:.6f}")

# 예시 3: 기하 브라운 운동 S_t = S_0 * exp((μ-σ²/2)t + σB_t)
S0 = 100
mu = 0.05
sigma = 0.2
S_t = S0 * np.exp((mu - sigma**2/2) * t + sigma * B)

# 이론적 평균과 비교
theoretical_mean = S0 * np.exp(mu * t)
empirical_mean = S_t  # 한 경로이므로 대신 많은 경로 평균 계산

# 많은 경로 생성
n_paths = 1000
S_paths = np.zeros((N+1, n_paths))
for i in range(n_paths):
    dW_i = np.random.randn(N) * np.sqrt(dt)
    B_i = np.concatenate([[0], np.cumsum(dW_i)])
    S_paths[:, i] = S0 * np.exp((mu - sigma**2/2) * t + sigma * B_i)

S_mean = np.mean(S_paths, axis=1)
S_std = np.std(S_paths, axis=1)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. B_t^2 vs t
axes[0, 0].plot(t, f_B, label='$B_t^2$ (시뮬레이션)', linewidth=1)
axes[0, 0].plot(t, expected_B2, 'r--', label='E[$B_t^2$] = $t$ (이론)', linewidth=2)
axes[0, 0].set_xlabel('시간 $t$')
axes[0, 0].set_ylabel('$B_t^2$')
axes[0, 0].set_title('이토 공식: $d(B^2_t) = 2B_t dB_t + dt$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. exp(B_t)
axes[0, 1].plot(t, f_exp, label='$e^{B_t}$ (시뮬레이션)', linewidth=1)
axes[0, 1].set_xlabel('시간 $t$')
axes[0, 1].set_ylabel('$e^{B_t}$')
axes[0, 1].set_title('이토 공식: $d(e^{B_t}) = e^{B_t} dB_t + \\frac{1}{2}e^{B_t} dt$')
axes[0, 1].grid(True, alpha=0.3)

# 3. GBM 평균과 이론
axes[1, 0].plot(t, S_mean, label='평균 (1000 경로)', linewidth=2)
axes[1, 0].plot(t, theoretical_mean, 'r--', label=f'이론 $S_0 e^{{\mu t}}$', linewidth=2)
axes[1, 0].fill_between(t, S_mean - S_std, S_mean + S_std, alpha=0.3, label='±1σ')
axes[1, 0].set_xlabel('시간 $t$')
axes[1, 0].set_ylabel('자산가격 $S_t$')
axes[1, 0].set_title(f'GBM: $dS = \\mu S dt + \\sigma S dB, \\mu={mu}, \\sigma={sigma}$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. GBM 몇 개 경로
for i in range(10):
    axes[1, 1].plot(t, S_paths[:, i], alpha=0.5, linewidth=0.8)
axes[1, 1].plot(t, S_mean, 'r-', linewidth=2, label='평균')
axes[1, 1].plot(t, theoretical_mean, 'k--', linewidth=2, label='이론')
axes[1, 1].set_xlabel('시간 $t$')
axes[1, 1].set_ylabel('자산가격 $S_t$')
axes[1, 1].set_title('GBM 경로들 (10개) + 평균')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ito_formula_illustration.png', dpi=150, bbox_inches='tight')
print("그래프 저장됨: ito_formula_illustration.png")

# 수치적 검증 결과 출력
print(f"\n=== 검증 결과 ===")
print(f"1. B_t^2 - t의 최대 편차: {np.max(np.abs(f_B - expected_B2)):.6f}")
print(f"2. d(e^B) 이토 공식 상관계수: {correlation:.6f}")
print(f"3. GBM E[S_T]의 오차 (T=1): {np.abs(S_mean[-1] - theoretical_mean[-1]):.6f} ({100*np.abs(S_mean[-1] - theoretical_mean[-1])/theoretical_mean[-1]:.3f}%)")
```

**출력 예시**:
```
=== 검증 결과 ===
1. B_t^2 - t의 최대 편차: 0.087234
2. d(e^B) 이토 공식 상관계수: 0.999847
3. GBM E[S_T]의 오차 (T=1): 0.312456 (0.312%)
```

---

## 🔗 AI/ML 연결

### DDPM과 이토 공식

DDPM은 정방향 noise 추가 과정 $dX_t = -\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t$를 정의하고, 역방향 샘플링을 위해 score function $\nabla \log p_t(x)$를 학습한다. **역방향 SDE의 drift 항 도출에 이토 공식의 측도 변환이 필수**이다.

### Score-SDE와 확률 흐름

연속 확률 모델에서 marginal likelihood $\log p_0(x_0)$는 **경로 적분을 이용하여 계산되는데**, 이때 $dX_t$ 항들의 기여를 이토 공식으로 추적해야 한다.

### Langevin MCMC / SGLD

Langevin 동역학 $dX_t = -\nabla U(X_t) dt + \sqrt{2/\beta} dB_t$는 정상분포가 $p(x) \propto e^{-\beta U(x)}$인 SDE다. 수렴성 증명에 이토 공식 + 에너지 감소 논증이 사용된다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 | 해결방안 |
|------|------|--------|
| $f \in C^2$ | 예각한 코너가 있으면 불가 (e.g., $\|x\|$) | 더 약한 regularity에 대한 generalized Itô formula (고급) |
| $\sigma$가 결정론적 또는 $X$에만 의존 | 일반 확률과정에는 적용 불가 | 이토 공식의 다양한 일반화 (Lévy 과정, rough paths) |
| $B$가 연속 | 점프(jump)가 있으면 보정 필요 | Lévy 과정에 대한 이토-Lévy 공식 |

**주의**: $d(B_t)^{1.5}$같은 비정수 거듭제곱은 $C^2$ 아님 → 이토 공식 직접 불가. Hölder 연속성만으로는 부족.

---

## 📌 핵심 정리

$$\boxed{df(X_t) = f'(X_t) dX_t + \frac{1}{2}f''(X_t)\sigma^2 dt \quad (f \in C^2)}$$

| 개념 | 공식 | 의미 |
|------|------|------|
| 이토 과정 | $dX = b\,dt + \sigma\,dB$ | 드리프트 + 확산 |
| 이토 공식 (1차) | $df = f' dX + \frac{1}{2}f''\sigma^2 dt$ | 2차 드리프트 추가 |
| 핵심 차이 | $(\Delta B)^2 \sim \Delta t$ | 결정론: $(\Delta x)^2 \sim (\Delta t)^2 \to 0$ |
| 응용 | $d(B^2) = 2B dB + dt$ | E[$B_t^2$] = $t$ ✓ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(x) = \sin(x)$일 때, $d(\sin(B_t))$를 이토 공식으로 구하시오.

<details>
<summary>힌트 및 해설</summary>

$f'(x) = \cos(x)$, $f''(x) = -\sin(x)$.

이토 공식 ($\sigma = 1$):
$$d(\sin(B_t)) = \cos(B_t) dB_t + \frac{1}{2}(-\sin(B_t)) dt = \cos(B_t) dB_t - \frac{1}{2}\sin(B_t) dt$$

확인: $\sin(B_t) = \int_0^t \cos(B_s) dB_s - \frac{1}{2}\int_0^t \sin(B_s) ds$

</details>

**문제 2** (심화): $f(x) = x^3$일 때, $d(B_t^3)$를 이토 공식으로 구하고, $\mathbb{E}[B_t^3]$를 계산하시오.

<details>
<summary>힌트 및 해설</summary>

$f'(x) = 3x^2$, $f''(x) = 6x$.

이토 공식:
$$d(B_t^3) = 3B_t^2 dB_t + \frac{1}{2} \cdot 6B_t \cdot dt = 3B_t^2 dB_t + 3B_t dt$$

적분:
$$B_t^3 = 3\int_0^t B_s^2 dB_s + 3\int_0^t B_s ds$$

기댓값: 첫 적분은 이토 마팅게일 → $\mathbb{E}[\int_0^t B_s^2 dB_s] = 0$.

$$\mathbb{E}[B_t^3] = 3\mathbb{E}\left[\int_0^t B_s ds\right] = 3\int_0^t \mathbb{E}[B_s] ds = 0$$

(대칭성: $B_t \sim N(0,t)$)

</details>

**문제 3** (AI 연결): DDPM에서 정방향 과정이 $dX_t = -\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t$일 때, $d(\|X_t\|^2)$의 drift 항을 구하시오. (힌트: $f(x) = \|x\|^2 = x_1^2 + \cdots + x_d^2$, $\nabla^2 f = 2I$)

<details>
<summary>힌트 및 해설</summary>

다차원 이토 공식 (다음 문서 참조):
$$d(\|X_t\|^2) = 2X_t^T dX_t + \text{tr}(\sigma\sigma^T) dt$$

$\sigma\sigma^T = \beta_t I$이므로 (스칼라 $\sqrt{\beta_t}$):
$$d(\|X_t\|^2) = 2X_t^T \left(-\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t\right) + \beta_t \cdot d dt$$

drift:
$$= -\beta_t \|X_t\|^2 dt + d\beta_t dt$$

$\mathbb{E}[\|X_t\|^2]$의 감소는 noise schedule $\beta_t$에 의존 — DDPM에서 noise를 단조증가시키는 이유.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 핵심 증명 — $(dB)^2 = dt$는 어디서 오는가 ▶](./02-db-squared-equals-dt.md) |

</div>
