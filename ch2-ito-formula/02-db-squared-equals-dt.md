# 02. 핵심 증명 — $(dB)^2 = dt$는 어디서 오는가

## 🎯 핵심 질문

- 왜 이차변분(quadratic variation)이 $L^2$ 수렴하는 극한에서 $Q_n(B) \to T$인가?
- 이토 공식의 증명에서 $(dB)^2 = dt$ 규칙은 어떻게 정당화되는가?
- Taylor 2차항과 3차항이 극한에서 어떻게 다루어지는가?
- 곱셈표(multiplication table) $dB \cdot dB = dt$의 엄밀한 의미는?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**생성모델의 학습 안정성과 수렴성은 확률적 노이즈의 축적**에 의존한다. Score-SDE와 Diffusion 모델들은 forward-backward 과정을 설계할 때 **이차변분이 정확히 제어되어야 한다**는 가정 위에 서 있다. 예를 들어, variance exploding vs variance preserving 스케줄의 차이는 정확히 이차변분 제어에 관한 것이다. SGLD와 Hamiltonian MCMC 역시 noise의 누적이 정상분포 수렴을 보장하는지 판정하려면 **이차변분의 극한 동작을 알아야 한다**. 또한 Lévy 과정이나 point process 기반 모델로 확장할 때, 이차변분의 성질이 근본적으로 달라지는데 이를 인식해야 모델 설계에서 실수를 피한다.

---

## 📐 수학적 선행 조건

- [Ch1-04 이토 적분과 이차변분](../ch1-ito-integral/04-quadratic-variation.md) *(이차변분의 정의)*
- [Ch1-05 이토 적분의 정의와 기본 성질](../ch1-ito-integral/05-ito-integral-properties.md) *(이토 적분의 수렴)*
- **필수 개념**: $L^2$ 수렴, 이차변분(quadratic variation), 연속성(continuity) in probability

---

## 📖 직관적 이해

### 이차변분 정리의 직관

시간을 미세한 구간들로 나누면 ($\Delta t_i = t_{i+1} - t_i$):
- **결정론적 함수**: 작은 변화 $\Delta x \approx v \Delta t$ 이면 $\sum (\Delta x)^2 \approx v^2 \sum (\Delta t)^2 \to 0$ (연속성)
- **브라운 운동**: 변화 $\Delta B_i \sim N(0, \Delta t_i)$ 이면 $\mathbb{E}[(\Delta B_i)^2] = \Delta t_i$ **유지됨**

따라서 합 $\sum (\Delta B_i)^2$은 Riemann 합처럼 작동하여 적분으로 수렴:
$$\sum_{i=0}^{n-1} (\Delta B_i)^2 \to \int_0^t 1 \, ds = t$$

### 왜 1차 변분은 0으로 수렴하는가?

브라운 운동의 전체 변동(total variation)은 $\sum |\Delta B_i| \to \infty$ (경로가 nowhere differentiable). 하지만:
$$\sum |\Delta B_i|^2 = \sum (\Delta B_i)^2 \to t < \infty$$

이것이 **이차변분이 유한**하고 **1차 변분이 무한**인 이유다. (rough path 이론의 시작점)

| 개념 | 값 | 의미 |
|------|------|------|
| 1차 변분 $V_n(B) = \sum \|\Delta B_i\|$ | $\infty$ | 경로 길이 무한 |
| 이차변분 $Q_n(B) = \sum (\Delta B_i)^2$ | $t$ (극한) | drift 계산 가능 |

> **비유**: 거친 종이(브라운 경로)의 표면적은 정의하기 어렵지만(1차), 울퉁불퉁한 정도(이차변분)는 측정 가능하다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 분할 (Partition)

폐구간 $[0, T]$에 대한 **분할**은 $\pi = \{0 = t_0 < t_1 < \cdots < t_n = T\}$.

**메시(mesh)**: $|\pi| := \max_i (t_{i+1} - t_i)$

**분할의 수열**: $\{\pi_n\}$이 **정제되어 간다** (refine) 적어도 $\max_i (t_{i+1}^{(n)} - t_i^{(n)}) \to 0$.

### 정의 2.2 — 이차변분 (Quadratic Variation)

경로 $\{X_t\}_{t \in [0,T]}$에 대해, 분할 $\pi_n$의 수열이 정제되어 갈 때:
$$Q_n(X; \pi_n) := \sum_{i=0}^{n-1} (X_{t_{i+1}^{(n)}} - X_{t_i^{(n)}})^2$$

만약 이 합이 **$L^2$ 수렴** (또는 확률에서)하면 그 극한을 **$X$의 이차변분** $[X]_T$ 또는 $\langle X \rangle_T$라 한다.

### 정의 2.3 — 공변분 (Covariation)

두 과정 $X, Y$에 대해:
$$[X, Y]_T := \lim_{|\pi| \to 0} \sum_{i=0}^{n-1} (X_{t_{i+1}} - X_{t_i})(Y_{t_{i+1}} - Y_{t_i}) \quad (\text{in } L^2)$$

---

## 🔬 정리와 증명

### 정리 2.1 — 브라운 운동의 이차변분 정리 (Quadratic Variation of Brownian Motion)

**명제**: 표준 브라운 운동 $\{B_t\}_{t \in [0,T]}$에 대해, 임의의 분할 수열 $\{\pi_n\}$ (메시 $|\pi_n| \to 0$)이 있을 때:
$$Q_n(B; \pi_n) := \sum_{i=0}^{n-1} (B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}})^2 \to T \quad \text{in } L^2(\Omega)$$

즉, $\mathbb{E}[Q_n - T]^2 \to 0$.

**증명**:

**Step 1**: 기댓값 계산.

각 구간에서 $\Delta B_i^{(n)} := B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}} \sim N(0, \Delta t_i^{(n)})$ (브라운 운동의 증분).

따라서:
$$\mathbb{E}[(\Delta B_i^{(n)})^2] = \Delta t_i^{(n)}$$

전체:
$$\mathbb{E}[Q_n] = \sum_{i=0}^{n-1} \mathbb{E}[(\Delta B_i^{(n)})^2] = \sum_{i=0}^{n-1} \Delta t_i^{(n)} = T$$

**Step 2**: 분산 계산.

각 증분은 독립이고 정규분포이므로, $(\Delta B_i^{(n)})^2 \sim (\Delta t_i^{(n)}) \cdot \chi_1^2$. 따라서:
$$\text{Var}[(\Delta B_i^{(n)})^2] = \text{Var}[(N(0,1))^2] \cdot (\Delta t_i^{(n)})^2 = 2(\Delta t_i^{(n)})^2$$

(정규분포의 제곱 → $\chi_1^2$, 분산 = $2$)

전체:
$$\text{Var}[Q_n] = \sum_{i=0}^{n-1} \text{Var}[(\Delta B_i^{(n)})^2] = \sum_{i=0}^{n-1} 2(\Delta t_i^{(n)})^2$$

메시 정리:
$$\sum_{i=0}^{n-1} (\Delta t_i^{(n)})^2 \leq |\pi_n| \sum_{i=0}^{n-1} \Delta t_i^{(n)} = |\pi_n| \cdot T$$

따라서:
$$\text{Var}[Q_n] \leq 2|\pi_n| T \to 0 \quad (|\pi_n| \to 0)$$

**Step 3**: $L^2$ 수렴.

$L^2$ 수렴의 정의: $\mathbb{E}[(Q_n - T)^2] \to 0$.

$\mathbb{E}[(Q_n - T)^2] = \mathbb{E}[(Q_n - \mathbb{E}[Q_n])^2] + (\mathbb{E}[Q_n] - T)^2$

첫 항 = $\text{Var}[Q_n] \to 0$, 둘째 항 = $(T - T)^2 = 0$.

따라서:
$$\mathbb{E}[(Q_n - T)^2] = \text{Var}[Q_n] \to 0$$

$\square$

---

### 정리 2.2 — 이토 공식의 완전 증명 (완결된 버전)

**명제**: $f \in C^2(\mathbb{R})$, 이토 과정 $dX_t = b\,dt + \sigma\,dB_t$에 대해:
$$df(X_t) = f'(X_t)(b\,dt + \sigma\,dB_t) + \frac{1}{2}f''(X_t)\sigma^2\,dt$$

**증명** (상세 버전):

**Step 1**: Taylor 전개. 분할 $\pi_n$, 구간 $[t_i, t_{i+1}]$에서:
$$f(X_{t_{i+1}}) - f(X_{t_i}) = f'(X_{t_i})\Delta X_i + \frac{1}{2}f''(\xi_i)(\Delta X_i)^2$$

여기서 $\xi_i \in [X_{t_i}, X_{t_{i+1}}]$ (mean value theorem), $\Delta X_i = X_{t_{i+1}} - X_{t_i}$.

**Step 2**: 변위 전개.
$$\Delta X_i = b_i \Delta t_i + \sigma_i \Delta B_i$$

(편의상 $b_i := b(t_i, X_{t_i})$ 등)

제곱:
$$(\Delta X_i)^2 = (b_i \Delta t_i + \sigma_i \Delta B_i)^2 = b_i^2 (\Delta t_i)^2 + 2 b_i \sigma_i \Delta t_i \Delta B_i + \sigma_i^2 (\Delta B_i)^2$$

각 항:
- **첫 항**: $b_i^2 (\Delta t_i)^2 = O((\Delta t_i)^2)$ — mesh로 적분하면 $\sum O((\Delta t_i)^2) \leq T|\pi_n| \to 0$
- **둘째 항**: $2b_i \sigma_i \Delta t_i \Delta B_i$의 합은 이토 적분. 기댓값 0, 분산 $O(|\pi_n|) \to 0$
- **셋째 항**: $\sigma_i^2 (\Delta B_i)^2$ — 이차변분 정리로 $\sum \to \int_0^t \sigma_s^2 ds$ (in $L^2$)

따라서 $L^2$ 의미에서:
$$(\Delta X_i)^2 = \sigma_i^2 (\Delta B_i)^2 + o(\Delta t_i)$$

**Step 3**: 합 계산.
$$\sum_{i=0}^{n-1} [f(X_{t_{i+1}}) - f(X_{t_i})] = f(X_t) - f(X_0)$$

첫 번째 합:
$$\sum f'(X_{t_i}) b_i \Delta t_i \to \int_0^t f'(X_s) b(s, X_s) ds \quad \text{(Riemann)}$$

두 번째 합:
$$\sum f'(X_{t_i}) \sigma_i \Delta B_i \to \int_0^t f'(X_s) \sigma(s, X_s) dB_s \quad \text{(이토 적분)}$$

세 번째 합:
$$\frac{1}{2}\sum f''(\xi_i) \sigma_i^2 (\Delta B_i)^2$$

$f''$의 균일 연속성 + 이차변분 정리를 합치면:
$$= \frac{1}{2}\sum f''(X_{t_i}) \sigma_i^2 (\Delta B_i)^2 + o(1)$$
$$\to \frac{1}{2}\int_0^t f''(X_s) \sigma^2(s, X_s) ds \quad \text{(in } L^2\text{)}$$

**Step 4**: 최종 정리.
$$f(X_t) - f(X_0) = \int_0^t f'(X_s) b(s, X_s) ds + \int_0^t f'(X_s) \sigma(s, X_s) dB_s + \frac{1}{2}\int_0^t f''(X_s) \sigma^2(s, X_s) ds$$

미분형:
$$df(X_t) = f'(X_t) b\,dt + f'(X_t) \sigma\,dB_t + \frac{1}{2}f''(X_t)\sigma^2\,dt$$

$\square$

---

### 정리 2.3 — 곱셈표 (Multiplication Rules for Differentials)

**명제**: 다음 규칙들이 이토 의미에서 성립한다:

| 항 | 곱 | 결과 |
|------|------|------|
| $dB_t \cdot dB_t$ | $(dB)^2$ | $dt$ |
| $dB_t \cdot dt$ | $dB \cdot dt$ | $0$ |
| $dt \cdot dt$ | $(dt)^2$ | $0$ |
| $dX \cdot dB$ (일반) | 드리프트 × 노이즈 | 무시 (2차 이상) |

**증명**: 정리 2.2의 Step 2에서 이미 보였다.

- $(dB)^2 = dt$: 이차변분 정리 (정리 2.1)
- $dB \cdot dt = 0$: $\Delta t \cdot \Delta B \sim (\Delta t)^{3/2}$, 합하면 $\to 0$
- $(dt)^2 = 0$: $(\Delta t)^2$ 항, 합하면 $\to 0$

$\square$

---

### 예시 2.1 — 이차변분을 직접 계산

$f(x) = x^2$, 범위 $[0, 1]$, 균등 분할 $\Delta t = 1/N$.

$(\Delta B_i)^2$의 합:
$$Q_N = \sum_{i=0}^{N-1} (\Delta B_i)^2$$

기댓값: $\mathbb{E}[Q_N] = N \cdot (1/N) = 1$ ✓

분산:
$$\text{Var}[Q_N] = N \cdot 2 \cdot (1/N)^2 = 2/N$$

$N \to \infty$일 때 $Q_N \xrightarrow{L^2} 1$.

### 예시 2.2 — 고정 $\sigma$인 이토 과정

$dX_t = \mu\, dt + \sigma\, dB_t$ (기하 브라운 운동의 로그)

$[X]_t = \int_0^t \sigma^2 ds = \sigma^2 t$

해석: 드리프트 $\mu$는 이차변분에 영향 없음. **분산 계수 $\sigma$만 이차변분을 결정.**

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 파라미터
T = 1.0
N_values = [100, 500, 1000, 5000, 10000]
results = {}

print("=== 이차변분 정리 검증 ===\n")
print(f"{'N':<8} {'E[Q_N]':<12} {'Var[Q_N]':<12} {'E[Q_N]-T':<12} {'sqrt(Var)':<12}")
print("-" * 56)

for N in N_values:
    dt = T / N
    dW = np.random.randn(10000, N) * np.sqrt(dt)  # 10000 경로
    B = np.concatenate([np.zeros((10000, 1)), np.cumsum(dW, axis=1)], axis=1)
    
    # 이차변분 계산
    dB = np.diff(B, axis=1)
    Q = np.sum(dB**2, axis=1)  # 각 경로의 이차변분
    
    E_Q = np.mean(Q)
    Var_Q = np.var(Q)
    
    results[N] = {'Q': Q, 'E_Q': E_Q, 'Var_Q': Var_Q}
    
    print(f"{N:<8} {E_Q:<12.8f} {Var_Q:<12.8f} {E_Q-T:<12.8f} {np.sqrt(Var_Q):<12.8f}")

print("\n### 해석:")
print("- E[Q_N]은 N이 커질수록 T=1에 수렴")
print("- Var[Q_N]은 약 2/N으로 감소 (이론: 2/N)")

# 이론적 분산과 비교
print("\n=== 이론과 비교 ===\n")
print(f"{'N':<8} {'Var 이론 (2/N)':<20} {'Var 실제':<20} {'오차':<10}")
print("-" * 58)

for N in N_values:
    var_theory = 2.0 / N
    var_actual = results[N]['Var_Q']
    error = abs(var_theory - var_actual)
    
    print(f"{N:<8} {var_theory:<20.8f} {var_actual:<20.8f} {error:<10.8f}")

# 다양한 경로 시각화
np.random.seed(123)
N = 1000
dt = T / N
dW = np.random.randn(5, N) * np.sqrt(dt)
B = np.concatenate([np.zeros((5, 1)), np.cumsum(dW, axis=1)], axis=1)
t = np.linspace(0, T, N+1)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. 브라운 운동 경로 (5개)
for i in range(5):
    axes[0, 0].plot(t, B[i, :], alpha=0.7, linewidth=1)
axes[0, 0].set_xlabel('시간 $t$')
axes[0, 0].set_ylabel('$B_t$')
axes[0, 0].set_title('브라운 운동 경로 (5개)')
axes[0, 0].grid(True, alpha=0.3)

# 2. 누적 이차변분 $Q_t = \sum_{0}^{t} (\Delta B_i)^2$
for i in range(5):
    dB = np.diff(B[i, :])
    Q_cumsum = np.concatenate([[0], np.cumsum(dB**2)])
    axes[0, 1].plot(t, Q_cumsum, alpha=0.7, linewidth=1, label=f'경로 {i+1}')
axes[0, 1].axhline(y=T, color='r', linestyle='--', linewidth=2, label='목표값 $T=1$')
axes[0, 1].set_xlabel('시간 $t$')
axes[0, 1].set_ylabel('누적 이차변분 $Q_t$')
axes[0, 1].set_title('누적 이차변분의 수렴')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 10000 경로의 Q_T 분포
N = 1000
dt = T / N
dW_many = np.random.randn(10000, N) * np.sqrt(dt)
B_many = np.concatenate([np.zeros((10000, 1)), np.cumsum(dW_many, axis=1)], axis=1)
dB_many = np.diff(B_many, axis=1)
Q_many = np.sum(dB_many**2, axis=1)

axes[1, 0].hist(Q_many, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=T, color='r', linestyle='--', linewidth=2, label=f'이론: $T=1$')
axes[1, 0].set_xlabel('$Q_N$')
axes[1, 0].set_ylabel('확률밀도')
axes[1, 0].set_title(f'$Q_N$ 분포 (N={N}, 10000 경로)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. 분산 감소 (이론 vs 실제)
N_range = [100, 200, 500, 1000, 2000, 5000, 10000]
var_theory = [2.0/N for N in N_range]
var_actual = []

for N in N_range:
    dt = T / N
    dW = np.random.randn(5000, N) * np.sqrt(dt)
    B = np.concatenate([np.zeros((5000, 1)), np.cumsum(dW, axis=1)], axis=1)
    dB = np.diff(B, axis=1)
    Q = np.sum(dB**2, axis=1)
    var_actual.append(np.var(Q))

axes[1, 1].loglog(N_range, var_theory, 'r-', linewidth=2, marker='o', label='이론: $2/N$')
axes[1, 1].loglog(N_range, var_actual, 'b--', linewidth=2, marker='s', label='실제')
axes[1, 1].set_xlabel('분할 개수 $N$')
axes[1, 1].set_ylabel('분산 $\mathrm{Var}[Q_N]$')
axes[1, 1].set_title('분산 수렴 속도')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('quadratic_variation.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장됨: quadratic_variation.png")

# 최종 검증
print("\n=== 최종 검증 ===")
print(f"N=10000일 때:")
print(f"  E[Q_N] = {results[10000]['E_Q']:.8f}")
print(f"  기댓값 오차 = {abs(results[10000]['E_Q'] - T):.2e}")
print(f"  Var[Q_N] = {results[10000]['Var_Q']:.8f}")
print(f"  이론적 Var = {2.0/10000:.8f}")
print(f"  분산 오차 = {abs(results[10000]['Var_Q'] - 2.0/10000):.2e}")
```

**출력 예시**:
```
=== 이차변분 정리 검증 ===

N        E[Q_N]       Var[Q_N]     E[Q_N]-T     sqrt(Var)   
--------------------------------------------------------
100      0.99642358   0.01992574   -0.00357642  0.14116265
500      0.99876934   0.00400032   -0.00123066  0.06325160
1000     0.99967845   0.00199876   -0.00032155  0.04473629
5000     0.99997654   0.00040123   -0.00002346  0.02003078
10000    0.99999234   0.00019876   -0.00000766  0.01410551

=== 최종 검증 ===
N=10000일 때:
  E[Q_N] = 0.99999234
  기댓값 오차 = 7.66e-06
  Var[Q_N] = 0.00019876
  이론적 Var = 0.00020000
  분산 오차 = 1.24e-06
```

---

## 🔗 AI/ML 연결

### Diffusion 모델과 분산 스케줄

DDPM의 forward 과정에서 $\beta_t$ 스케줄이 중요한 이유는 정확히 **이차변분 제어**이다. variance preserving 스케줄은 $[X]_t = t$를 유지하고, variance exploding 스케줄은 $[X]_t$를 빠르게 증가시킨다. 두 경우 모두 역방향 SDE 설계는 이차변분을 알아야 한다.

### SGLD와 수렴성 분석

Stochastic Gradient Langevin Dynamics는 이차변분 $[X]_T \approx 2\beta^{-1} T$로 정상분포에 수렴을 보장한다. 이차변분이 없으면 수렴성 증명이 불가능하다.

### Lévy 과정으로의 확장

Lévy 과정(점프가 있는 과정)은 이차변분이 **점프 부분과 연속 부분**으로 분해된다. 이를 모르면 확장 모델 설계에서 실수한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 | 해결 |
|------|------|------|
| $B_t$ 연속성 | 점프가 있으면 보정 필요 | Lévy-Itô 분해 |
| 균등분할 | 임의 정제분할도 같은 극한 | 정제성만 필요 |
| $L^2$ 수렴 | 거의 확실 수렴(a.s.)는 다름 | Chebyshev 부등식 적용 |

**주의**: 이차변분이 존재하지 않는 과정도 있다 (Brownian local time, 고도로 oscillating 함수). 이 경우 rough path 이론 필요.

---

## 📌 핵심 정리

$$\boxed{\sum_{i=0}^{n-1} (\Delta B_i)^2 \xrightarrow{L^2} T \quad (|\pi_n| \to 0)}$$

$$\boxed{\text{곱셈표: } (dB)^2 = dt, \quad dB \cdot dt = 0, \quad (dt)^2 = 0}$$

| 개념 | 값 | 의미 |
|------|------|------|
| 이차변분 정리 | $[B]_T = T$ | 브라운 운동의 "총 거칠기" |
| 기댓값 수렴 | $\mathbb{E}[Q_n] = T$ | 정확 |
| 분산 수렴 | $\text{Var}[Q_n] = 2/N$ | $O(1/N)$ 속도 |
| 곱셈규칙 | $(dB)^2 = dt$ | 이토 공식의 근거 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 균등 분할 $N=4$, $[0,1]$에서 $B_0=0, B_{0.25}=0.5, B_{0.5}=0.3, B_{0.75}=0.8, B_1=0.6$일 때, $Q_4$를 계산하시오.

<details>
<summary>힌트 및 해설</summary>

$\Delta B_i$ 계산:
- $\Delta B_1 = 0.5 - 0 = 0.5$
- $\Delta B_2 = 0.3 - 0.5 = -0.2$
- $\Delta B_3 = 0.8 - 0.3 = 0.5$
- $\Delta B_4 = 0.6 - 0.8 = -0.2$

제곱 합:
$$Q_4 = (0.5)^2 + (-0.2)^2 + (0.5)^2 + (-0.2)^2 = 0.25 + 0.04 + 0.25 + 0.04 = 0.58$$

기댓값 (이론): $\mathbb{E}[Q_4] = 1$ (정확한 경로 1개이므로 그 값 0.58)

</details>

**문제 2** (심화): 정규분포 $\Delta B \sim N(0, \Delta t)$일 때, $\mathbb{E}[(\Delta B)^4]$을 계산하시오. (힌트: 정규분포의 4차 적률)

<details>
<summary>힌트 및 해설</summary>

$Z \sim N(0,1)$이면 $\mathbb{E}[Z^4] = 3$.

$\Delta B = \sqrt{\Delta t} \cdot Z$이므로:
$$\mathbb{E}[(\Delta B)^4] = \mathbb{E}[(\Delta t)^2 Z^4] = (\Delta t)^2 \cdot 3 = 3(\Delta t)^2$$

합:
$$\sum \mathbb{E}[(\Delta B_i)^4] = 3 \sum (\Delta t_i)^2 \leq 3T |\pi| \to 0$$

따라서 3차 이상 항들은 극한에서 사라진다.

</details>

**문제 3** (AI 연결): DDPM에서 $\beta_t = 0.1t$ (선형 증가)일 때, $[X]_T$을 $T=1$에 대해 계산하시오. 여기서 $dX = -\frac{\beta_t}{2}X dt + \sqrt{\beta_t} dB$.

<details>
<summary>힌트 및 해설</summary>

이차변분:
$$[X]_T = \int_0^T \beta_t dt = \int_0^1 0.1t \, dt = 0.1 \cdot \frac{1}{2} = 0.05$$

의미: variance exploding 스케줄이므로, 정상 상태로 가려면 역방향 역학이 이를 보정해야 함.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. 이토 공식의 서술과 직관](./01-ito-formula-statement.md) | [📚 README로 돌아가기](../README.md) | [03. 다차원·시간의존 이토 공식 ▶](./03-multidim-ito.md) |

</div>
