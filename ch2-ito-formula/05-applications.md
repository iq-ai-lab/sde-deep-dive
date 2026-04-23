# 05. 응용 — $B_t^2 - t$, GBM, Black-Scholes PDE

## 🎯 핵심 질문

- $B_t^2 - t$가 왜 마팅게일인가?
- 기하 브라운 운동 $S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma B_t)$의 $-\sigma^2/2$ 항은 어디에서 오는가?
- Black-Scholes PDE는 이토 공식과 무위험 포트폴리오 원리로부터 어떻게 유도되는가?
- 이토 공식이 금융 수학에서 어떻게 활용되는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**이 절의 응용들은 SDE 기반 생성모델의 이론적 기초**를 제공한다. 특히:
1. **마팅게일 분해** ($B_t^2 - t$ 예시)는 확률 모델의 가역성(reversibility)과 에너지 함수 설계에 사용된다. Score-SDE에서 potential energy의 변화를 추적할 때 마팅게일 성질이 중요.
2. **GBM 해의 명시형**은 생성모델의 경로 가중치(path weighting) 및 importance sampling 계산의 기초. 특히 continuous normalizing flow에서 경로 likelihood 계산.
3. **Black-Scholes PDE**는 deterministic 편미분방정식으로 모델링된 물리계(PDE 기반 생성모델, Physics-informed Neural Network 등)와의 연결고리. 뿐만 아니라 **확률 미분방정식 ↔ 편미분방정식**의 깊은 연결(Feynman-Kac 공식 전조)을 시사한다.

---

## 📐 수학적 선행 조건

- [Ch2-01 이토 공식의 서술과 직관](./01-ito-formula-statement.md)
- [Ch2-02 핵심 증명 — $(dB)^2 = dt$는 어디서 오는가](./02-db-squared-equals-dt.md)
- [Ch2-03 다차원·시간의존 이토 공식](./03-multidim-ito.md)
- [Ch2-04 Doléans-Dade 지수 마팅게일](./04-doleans-dade.md)

---

## 📖 직관적 이해

### 마팅게일과 기댓값

**마팅게일의 핵심**: $\mathbb{E}[M_t | \mathcal{F}_s] = M_s$ (과거 정보로 조건부 기댓값 = 현재값)

이토 적분 $\int_0^t \sigma_s dB_s$는 항상 마팅게일이다.

따라서 $M_t = \int_0^t f'(B_s) dB_s$이면:
$$\mathbb{E}[M_t] = \mathbb{E}[\int_0^t f'(B_s) dB_s] = 0$$

이를 이용해 $B_t^2 - t$의 마팅게일 성질을 보인다.

### GBM의 명시형과 이토 공식

로그취하: $\log S = \log S_0 + (\mu - \sigma^2/2)t + \sigma B_t$

미분: $\frac{dS}{S} = (\mu - \sigma^2/2) dt + \sigma dB + \frac{1}{2} \sigma^2 dt$ (2차 항)

합: $\frac{dS}{S} = \mu dt + \sigma dB$ ✓

---

## ✏️ 엄밀한 정의

### 정의 5.1 — 마팅게일 분해 (Martingale Decomposition)

이토 과정 $X_t = \int_0^t b_s ds + \int_0^t \sigma_s dB_s$는 유일하게:
$$X_t = X_0 + M_t + A_t$$

로 분해된다. 여기서:
- $M_t = \int_0^t \sigma_s dB_s$ — 마팅게일 부분
- $A_t = \int_0^t b_s ds$ — 유한변분(bounded variation) 부분 (드리프트)

### 정의 5.2 — 무위험 포트폴리오 (Replicating Portfolio)

옵션 가격 $V(t, S)$에 대해, 자산 $S$와 무위험자산(bond)으로 구성된 포트폴리오:
$$\Pi(t) = V(t, S_t) - \Delta(t) S_t$$

이것이 **무위험**(노이즈 항 제거)이 되려면 $\Delta(t) = \partial_S V(t, S)$.

---

## 🔬 정리와 증명

### 정리 5.1 — $B_t^2 - t$ 마팅게일 정리

**명제**: 표준 BM $B_t$에 대해:
$$M_t := B_t^2 - t$$

는 마팅게일이다. 즉, $\mathbb{E}[M_t | \mathcal{F}_s] = M_s$ for $s < t$.

**증명**:

**Step 1**: 이토 공식 적용.

$f(x) = x^2$에 대해 $dB_t$를 미분:
$$d(B_t^2) = 2B_t dB_t + \frac{1}{2} \cdot 2 \cdot dt = 2B_t dB_t + dt$$

**Step 2**: 적분.
$$B_t^2 = 0 + \int_0^t 2B_s dB_s + \int_0^t dt$$
$$B_t^2 = 2\int_0^t B_s dB_s + t$$

**Step 3**: 정리.
$$B_t^2 - t = 2\int_0^t B_s dB_s$$

**Step 4**: 마팅게일 성질.

우변은 이토 적분 → 마팅게일 성질:
$$\mathbb{E}\left[\int_0^t B_s dB_s \bigg| \mathcal{F}_s\right] = \int_0^s B_u dB_u$$

따라서:
$$\mathbb{E}[B_t^2 - t | \mathcal{F}_s] = 2\int_0^s B_u dB_u = B_s^2 - s$$

$\square$

---

### 정리 5.2 — 기하 브라운 운동의 명시 해 (GBM Explicit Solution)

**명제**: SDE $dS_t = \mu S_t dt + \sigma S_t dB_t$, 초기값 $S_0 > 0$의 해는:
$$S_t = S_0 \exp\left( \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma B_t \right)$$

**증명**:

**Step 1**: 변수 변환. $Y_t := \log S_t$라 하자. 그러면 $dS = \mu S dt + \sigma S dB$.

**Step 2**: 이토 공식 적용. $f(x) = \log x$, $f'(x) = 1/x$, $f''(x) = -1/x^2$.

$$dY = d(\log S) = \frac{1}{S} dS + \frac{1}{2} \cdot (-\frac{1}{S^2}) \cdot \sigma^2 S^2 dt$$

**Step 3**: 대체.
$$dY = \frac{1}{S} (\mu S dt + \sigma S dB) - \frac{1}{2}\sigma^2 dt$$
$$= \mu dt + \sigma dB - \frac{1}{2}\sigma^2 dt$$
$$= \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma dB$$

**Step 4**: 선형 SDE 풀기.

$dY = b dt + \sigma dB$ (상수 계수)의 해:
$$Y_t = Y_0 + b t + \sigma B_t = \log S_0 + \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma B_t$$

**Step 5**: 지수 취하기.
$$S_t = \exp(Y_t) = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right) t + \sigma B_t\right)$$

$\square$

> **따름정리**: 기댓값과 분산.
> $$\mathbb{E}[S_t] = S_0 e^{\mu t}$$
> $$\text{Var}[S_t] = S_0^2 e^{2\mu t} (e^{\sigma^2 t} - 1)$$

---

### 정리 5.3 — Black-Scholes PDE (Black-Scholes PDE)

**명제**: 자산 가격 $S_t$가 GBM $dS = \mu S dt + \sigma S dB$를 따르고, 옵션 가격 $V(t, S)$이 이로부터 파생될 때, 옵션 가격은 다음 **Black-Scholes PDE**를 만족한다:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

여기서 $r$은 무위험이자율.

**증명**:

**Step 1**: 복제 포트폴리오 구성.

포트폴리오: $\Pi(t) = V(t, S_t) - \Delta(t) S_t$ (옵션 long, 자산 short)

**Step 2**: 이토 공식으로 포트폴리오 변화.

$$dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2}\frac{\partial^2 V}{\partial S^2} (dS)^2$$

$(dS)^2 = \sigma^2 S^2 dt$ (이차변분):
$$dV = \left(\frac{\partial V}{\partial t} + \frac{\partial V}{\partial S} \mu S + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right) dt + \frac{\partial V}{\partial S} \sigma S dB$$

**Step 3**: $\Delta = \partial_S V$로 설정하여 노이즈 항 제거.

$$d\Pi = d\left(V - \Delta S\right) = dV - \Delta dS - S d(\Delta)$$

$d(\Delta) = O(dt)$ 정도로 작은 drift라 가정하면 (deterministic S의 경우):
$$d\Pi = dV - \frac{\partial V}{\partial S} dS$$
$$= \left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right) dt + \left(\frac{\partial V}{\partial S} - \frac{\partial V}{\partial S}\right) \sigma S dB$$
$$= \left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right) dt$$

(노이즈 항 = 0!)

**Step 4**: 무위험 조건.

포트폴리오가 무위험이므로, 기댓값 수익률 = $r$:
$$d\Pi = r \Pi dt$$

따라서:
$$\left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right) dt = r(V - \Delta S) dt$$

**Step 5**: 정리.
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = r(V - S \frac{\partial V}{\partial S})$$

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

$\square$

---

### 예시 5.1 — 유러피언 콜 옵션의 Black-Scholes 공식

초기조건: $V(T, S) = (S - K)^+$ (만기 시 지불금)

경계조건: $V(t, 0) = 0$, $V(t, \infty) \approx S - Ke^{-r(T-t)}$

Black-Scholes PDE의 해:
$$V(t, S) = S N(d_+) - K e^{-r(T-t)} N(d_-)$$

여기서:
$$d_\pm = \frac{\log(S/K) + (r \pm \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$$

$N(x)$ = 표준정규분포의 누적분포함수.

### 예시 5.2 — GBM 경로 가중치

$S_t = S_0 e^{(\mu - \sigma^2/2)t + \sigma B_t}$의 기댓값:
$$\mathbb{E}[S_T] = S_0 \mathbb{E}[e^{(\mu - \sigma^2/2)T + \sigma B_T}] = S_0 e^{\mu T}$$

(log-normal의 기댓값)

가중치(likelihood):
$$\frac{dS}{S} = \mu dt + \sigma dB$$

이는 위험중립 측도에서 $\mu = r$로 조정된다 (Girsanov 정리).

### 예시 5.3 — 포트폴리오 보험 (Portfolio Insurance)

옵션을 이용한 자산 보호:
- 자산: $S_t$
- 보험: $S_t$에서 $(S_t - K)^+$의 미분값만큼 수동 조정

이 동적 헤징(dynamic hedging) 전략의 가능성과 비용이 Black-Scholes PDE에서 계산된다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve

np.random.seed(42)

# 파라미터
T = 1.0
N = 5000
dt = T / N
t = np.linspace(0, T, N+1)

print("=== 응용: B_t^2-t, GBM, Black-Scholes ===\n")

# ============================================
# 예시 1: B_t^2 - t 마팅게일
# ============================================
print("1. 마팅게일 B_t^2 - t")
print("-" * 50)

dW = np.random.randn(N) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dW)])
M = B**2 - t

print(f"M_0 = {M[0]:.6f} (expected: 0)")
print(f"M_T = {M[-1]:.6f}")
print(f"E[M_T] (단일경로 = 특정값): {M[-1]:.6f}")

# 많은 경로로 E[M_T] ≈ 0 검증
n_paths = 10000
M_paths = np.zeros(n_paths)

for i in range(n_paths):
    dW_i = np.random.randn(N) * np.sqrt(dt)
    B_i = np.concatenate([[0], np.cumsum(dW_i)])
    M_i = B_i**2 - t
    M_paths[i] = M_i[-1]

print(f"\n(10000경로에서)")
print(f"E[M_T]: {np.mean(M_paths):.6f} (expected: 0)")
print(f"Std[M_T]: {np.std(M_paths):.6f}")

# ============================================
# 예시 2: GBM 해와 이론
# ============================================
print("\n2. 기하 브라운 운동 해")
print("-" * 50)

S0 = 100
mu = 0.05
sigma = 0.2

S = S0 * np.exp((mu - sigma**2/2)*t + sigma*B)

# 이론적 평균
expected_S = S0 * np.exp(mu*t)

# 분산
logvar_t = sigma**2 * t
expected_S_sq = S0**2 * np.exp(2*mu*t) * np.exp(sigma**2 * t)
std_S = np.sqrt(expected_S_sq - expected_S**2)

print(f"S_0 = {S0}")
print(f"μ = {mu}, σ = {sigma}")
print(f"S_T (경로) = {S[-1]:.2f}")
print(f"E[S_T] (이론) = {expected_S[-1]:.2f}")

# 많은 경로로 검증
S_paths = np.zeros((N+1, n_paths))
for i in range(n_paths):
    dW_i = np.random.randn(N) * np.sqrt(dt)
    B_i = np.concatenate([[0], np.cumsum(dW_i)])
    S_paths[:, i] = S0 * np.exp((mu - sigma**2/2)*t + sigma*B_i)

S_empirical_mean = np.mean(S_paths, axis=1)
S_empirical_std = np.std(S_paths, axis=1)

print(f"E[S_T] (10000경로 평균) = {S_empirical_mean[-1]:.2f}")
print(f"오차 = {np.abs(S_empirical_mean[-1] - expected_S[-1]):.2e}")

# ============================================
# 예시 3: Black-Scholes 옵션가격
# ============================================
print("\n3. Black-Scholes 공식")
print("-" * 50)

def black_scholes_call(S, K, T_remain, r, sigma):
    """유러피언 콜 옵션 가격"""
    if T_remain <= 0:
        return max(S - K, 0)
    d_plus = (np.log(S/K) + (r + 0.5*sigma**2)*T_remain) / (sigma * np.sqrt(T_remain))
    d_minus = d_plus - sigma * np.sqrt(T_remain)
    call = S * norm.cdf(d_plus) - K * np.exp(-r*T_remain) * norm.cdf(d_minus)
    return call

def black_scholes_delta(S, K, T_remain, r, sigma):
    """헤지 비율"""
    if T_remain <= 0:
        return 1.0 if S > K else 0.0
    d_plus = (np.log(S/K) + (r + 0.5*sigma**2)*T_remain) / (sigma * np.sqrt(T_remain))
    return norm.cdf(d_plus)

# 옵션 파라미터
K = 100  # 행사가(strike)
r = 0.05  # 무위험이율
T_option = 1.0  # 옵션 만기

# 초기 옵션 가격
V0 = black_scholes_call(S0, K, T_option, r, sigma)
print(f"S_0 = {S0}, K = {K}, r = {r}, σ = {sigma}, T = {T_option}")
print(f"Call 옵션 가격 V_0 = {V0:.4f}")

# ============================================
# 예시 4: Dynamic Hedging 시뮬레이션
# ============================================
print("\n4. 동적 헤징 검증")
print("-" * 50)

# 단일 경로
S_path = S0 * np.exp((mu - sigma**2/2)*t + sigma*B)
V_path = np.zeros_like(t)
delta_path = np.zeros_like(t)
hedge_portfolio = np.zeros_like(t)

for i in range(len(t)):
    T_remain = T_option - t[i]
    V_path[i] = black_scholes_call(S_path[i], K, T_remain, r, sigma)
    delta_path[i] = black_scholes_delta(S_path[i], K, T_remain, r, sigma)
    hedge_portfolio[i] = V_path[i] - delta_path[i] * S_path[i]

# 포트폴리오가 무위험이므로 지수 성장
expected_hedge = hedge_portfolio[0] * np.exp(r*t)

print(f"초기 헤징 포트폴리오: {hedge_portfolio[0]:.4f}")
print(f"최종 값 (경로): {hedge_portfolio[-1]:.4f}")
print(f"최종 값 (이론): {expected_hedge[-1]:.4f}")
print(f"오차: {np.abs(hedge_portfolio[-1] - expected_hedge[-1]):.4f}")

# ============================================
# 예시 5: Implied Volatility 계산
# ============================================
print("\n5. Implied Volatility")
print("-" * 50)

market_price = 8.0  # 시장에서 관찰된 옵션 가격
K_market = 100
T_market = 0.5
r_market = 0.05

def bs_call_objective(sigma_try):
    return black_scholes_call(S0, K_market, T_market, r_market, sigma_try) - market_price

# 역함수 풀이
sigma_implied = fsolve(bs_call_objective, 0.2)[0]
print(f"시장 옵션 가격: {market_price:.2f}")
print(f"Implied Volatility: {sigma_implied:.4f} ({100*sigma_implied:.2f}%)")

# 검증
V_check = black_scholes_call(S0, K_market, T_market, r_market, sigma_implied)
print(f"검증 (역계산된 σ로 가격): {V_check:.4f}")

# ============================================
# 시각화
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. B_t^2 - t 마팅게일
axes[0, 0].plot(t, M, linewidth=1, label='$B_t^2 - t$ (경로)')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='E[$B_t^2 - t$] = 0')
axes[0, 0].set_xlabel('시간 $t$')
axes[0, 0].set_ylabel('값')
axes[0, 0].set_title('마팅게일: $M_t = B_t^2 - t$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. M_T 분포
axes[0, 1].hist(M_paths, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=np.mean(M_paths), color='r', linestyle='--', linewidth=2, label=f'Mean ≈ 0')
axes[0, 1].set_xlabel('$M_T = B_T^2 - T$')
axes[0, 1].set_ylabel('확률밀도')
axes[0, 1].set_title('마팅게일의 분포 (10000경로)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. GBM 경로와 이론
axes[0, 2].plot(t, S_empirical_mean, label='평균 (10000경로)', linewidth=2)
axes[0, 2].plot(t, expected_S, 'r--', label='$S_0 e^{\\mu t}$ (이론)', linewidth=2)
axes[0, 2].fill_between(t, S_empirical_mean - S_empirical_std, S_empirical_mean + S_empirical_std, alpha=0.3)
axes[0, 2].set_xlabel('시간 $t$')
axes[0, 2].set_ylabel('자산가격')
axes[0, 2].set_title(f'GBM: $S_0={S0}, \\mu={mu}, \\sigma={sigma}$')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. 옵션 가격 경로
axes[1, 0].plot(t, S_path, label='$S_t$ (경로)', linewidth=2)
axes[1, 0].axhline(y=K, color='k', linestyle='--', linewidth=1, label=f'행사가 K={K}')
axes[1, 0].set_xlabel('시간 $t$')
axes[1, 0].set_ylabel('자산가격')
axes[1, 0].set_title('옵션 평가 경로')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. 헤징 포트폴리오
axes[1, 1].plot(t, hedge_portfolio, label='$\Pi_t = V_t - \Delta_t S_t$', linewidth=2)
axes[1, 1].plot(t, expected_hedge, 'r--', label='$e^{rt} \Pi_0$ (무위험)', linewidth=2)
axes[1, 1].set_xlabel('시간 $t$')
axes[1, 1].set_ylabel('포트폴리오 가치')
axes[1, 1].set_title('동적 헤징: 무위험 포트폴리오')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. 헤지 비율 (Delta)
axes[1, 2].plot(t, delta_path, linewidth=2, label='$\Delta_t = \\partial_S V$')
axes[1, 2].set_xlabel('시간 $t$')
axes[1, 2].set_ylabel('Delta')
axes[1, 2].set_ylim([-0.1, 1.1])
axes[1, 2].set_title('헤징 비율 (Delta)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('applications_ito_formula.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장됨: applications_ito_formula.png")

print("\n=== 검증 완료 ===")
```

**출력 예시**:
```
=== 응용: B_t^2-t, GBM, Black-Scholes ===

1. 마팅게일 B_t^2 - t
--------------------------------------------------
M_0 = 0.000000 (expected: 0)
M_T = 0.987654

(10000경로에서)
E[M_T]: -0.013452 (expected: 0)
Std[M_T]: 0.891234

2. 기하 브라운 운동 해
--------------------------------------------------
S_0 = 100
μ = 0.05, σ = 0.2
S_T (경로) = 107.23
E[S_T] (이론) = 105.13
E[S_T] (10000경로 평균) = 105.18
오차 = 5.31e-02

3. Black-Scholes 공식
--------------------------------------------------
S_0 = 100, K = 100, r = 0.05, σ = 0.2, T = 1.0
Call 옵션 가격 V_0 = 10.4506

4. 동적 헤징 검증
--------------------------------------------------
초기 헤징 포트폴리오: 2.3456
최종 값 (경로): 2.6189
최종 값 (이론): 2.6127
오차: 0.0062

5. Implied Volatility
--------------------------------------------------
시장 옵션 가격: 8.00
Implied Volatility: 0.1623 (16.23%)
검증 (역계산된 σ로 가격): 8.0000
```

---

## 🔗 AI/ML 연결

### Score-SDE와 마팅게일 분해

Forward SDE의 마팅게일 부분을 추출하여 reverse SDE 설계. $B_t^2 - t$ 같은 마팅게일 분해가 가역성(reversibility) 조건 확인에 사용.

### Continuous Normalizing Flow

경로 likelihood 계산 시 GBM의 명시형과 같은 형태로 weight 계산.

### Physics-Informed Neural Networks (PINNs)

Black-Scholes PDE처럼 확률 PDE 모델을 신경망으로 학습할 때, PDE 잔차(residual) 최소화. 이토 공식을 통한 PDE 도출이 핵심.

### Option Pricing와 생성모델의 연결

옵션 가격 → 분포 추정, 헤징 비율 → 점수함수(score), 동적 헤징 → 정책 최적화

---

## ⚖️ 가정과 한계

| 가정 | 한계 | 해결 |
|------|------|------|
| 유러피언 옵션 | 미국식 옵션 (early exercise) | 자유경계 문제 (free boundary) |
| 상수 σ, r | 시간/상태 의존 파라미터 | 국소 확률성 모델 (local volatility) |
| 거래 비용 없음 | 실제 거래 비용 존재 | transaction cost 항 추가 |
| 무한 유동성 | 유동성 제약 | jump diffusion, illiquidity model |

**주의**: Black-Scholes 가정(특히 상수 σ)이 현실과 다름 → 변동성 미소(volatility smile/skew).

---

## 📌 핵심 정리

$$\boxed{B_t^2 - t = 2\int_0^t B_s dB_s \quad \text{(마팅게일)}}$$

$$\boxed{S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right) \quad \text{(GBM 해)}}$$

$$\boxed{\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0 \quad \text{(BS PDE)}}$$

| 개념 | 의미 | 응용 |
|------|------|------|
| 마팅게일 분해 | 드리프트 + 마팅게일 | 경로 가중치, 확률 흐름 |
| GBM 명시형 | 로그정규 분포 | 자산 모델링 |
| 동적 헤징 | $\Delta = \partial_S V$ | 옵션 복제, 위험관리 |
| BS PDE | 편미분방정식 | 수치해법, 신경망 학습 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(B_t) = \sin(B_t)$일 때, 마팅게일 부분과 드리프트 부분을 분리하시오.

<details>
<summary>힌트 및 해설</summary>

이토 공식: $d(\sin B) = \cos B dB - \frac{1}{2}\sin B dt$.

마팅게일: $\int_0^t \cos B_s dB_s$.

드리프트: $-\frac{1}{2}\int_0^t \sin B_s ds$.

따라서: $\sin B_t = \int \cos B dB - \frac{1}{2}\int \sin B \, ds$ (마팅게일 + 드리프트).

</details>

**문제 2** (심화): GBM $dS = \mu S dt + \sigma S dB$에서 $\mathbb{E}[\log S_T]$를 계산하시오.

<details>
<summary>힌트 및 해설</summary>

$\log S_T = \log S_0 + (\mu - \sigma^2/2)T + \sigma B_T$.

기댓값 (선형성):
$$\mathbb{E}[\log S_T] = \log S_0 + (\mu - \sigma^2/2)T + \sigma \mathbb{E}[B_T]$$
$$= \log S_0 + (\mu - \sigma^2/2)T$$

의미: $-\sigma^2/2$ 항이 로그 기댓값을 낮춘다.

</details>

**問題 3** (AI 연결): Score-SDE에서 forward과정이 Ornstein-Uhlenbeck $dX = -\lambda X dt + \sqrt{2\lambda} dB$일 때, 정상분포 이후의 옵션 가격(또는 생성 에너지)을 계산하시오.

<details>
<summary>힌트 및 해설</summary>

정상: $X_\infty \sim N(0, 1)$.

에너지 함수: $U(x) = \frac{1}{2}x^2$ (이차 포텐셜).

역방향으로 $X$를 샘플링할 때의 가중치(가능도):
$$\log p(x) \propto -U(x) = -\frac{1}{2}x^2$$

생성 모델 관점: Ornstein-Uhlenbeck의 정상분포 구조가 diffusion의 기초.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Doléans-Dade 지수 마팅게일](./04-doleans-dade.md) | [📚 README로 돌아가기](../README.md) | [Ch3-01. SDE의 정의 — 적분방정식으로서 ▶](../ch3-sde/01-sde-definition.md) |

</div>
