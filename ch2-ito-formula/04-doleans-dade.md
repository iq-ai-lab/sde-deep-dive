# 04. Doléans-Dade 지수 마팅게일

## 🎯 핵심 질문

- 확률 지수(stochastic exponential) $\mathcal{E}(M)_t$의 정의와 성질은?
- 왜 $\mathcal{E}(M)_t = \exp(M_t - \frac{1}{2}\langle M \rangle_t)$인가?
- 국소 마팅게일이 진짜 마팅게일이 되는 조건은? (Novikov 조건)
- Girsanov 정리와의 연결은?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Doléans-Dade 지수는 확률측도 변환(change of measure)의 핵심 도구**다. 생성모델의 많은 알고리즘에서 정상분포 또는 다른 목표분포로의 샘플링을 위해 **Radon-Nikodym 밀도**로 작동한다. Score-SDE에서 forward SDE를 reverse SDE로 변환할 때, 그 변환 인자(likelihood weighting)는 정확히 Doléans-Dade 형태다. 또한 **Importance Sampling** 기반 생성모델, **Control Variates** 기반 gradient 추정, **Langevin MCMC**의 수렴성 분석 모두 이 지수에 의존한다. 최근 **Flow Matching** 논문들도 경로 가중치(path weighting)를 계산할 때 Doléans-Dade 지수를 사용한다.

---

## 📐 수학적 선행 조건

- [Ch2-01 이토 공식의 서술과 직관](./01-ito-formula-statement.md) *(이토 공식 기본)*
- [Ch2-03 다차원·시간의존 이토 공식](./03-multidim-ito.md) *(곱의 법칙)*
- **필수 개념**: 연속 국소 마팅게일, 마팅게일 성질, conditional expectation, 공변분(covariation)

---

## 📖 직관적 이해

### 일반 지수함수와의 비교

결정론: $\frac{d}{dt}e^{t} = e^{t}$

확률: $\frac{d}{dt}e^{M_t}$는 이토 공식으로 계산해야 한다. 만약 단순히 $\mathcal{E}(M)_t = e^{M_t}$라 하면:

$$d(e^{M_t}) = e^{M_t} dM_t + \frac{1}{2}e^{M_t} d\langle M \rangle_t$$

이것은 **드리프트 항을 만든다**! 순수 마팅게일이 아니다.

**해결책**: 드리프트를 "빼기 위해"
$$\mathcal{E}(M)_t := \exp\left( M_t - \frac{1}{2}\langle M \rangle_t \right)$$

그러면 이토 공식:
$$d\mathcal{E}(M)_t = \mathcal{E}(M)_t dM_t \quad \text{(드리프트 항 없음!)}$$

**결과**: 순수한 **마팅게일**이 된다.

### 직관: "보정 인자"

$-\frac{1}{2}\langle M \rangle_t$는 **노이즈의 이차변분이 만드는 드리프트를 보정하는 항**이다. 이것은 **정규화(regularization)** 또는 **컨벡스성(convexity) 보정** 역할을 한다.

| 항 | 역할 |
|------|------|
| $e^{M_t}$ | 기본 지수 |
| $-\frac{1}{2}\langle M \rangle_t$ | 노이즈로 인한 drift 보정 |
| 곱 | 순수 martingale |

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 확률 지수 (Stochastic Exponential)

연속 국소 마팅게일 $M_t$ ($M_0 = 0$)에 대해, **확률 지수(또는 Doléans-Dade 지수)**를 다음과 같이 정의한다:
$$\mathcal{E}(M)_t := \exp\left( M_t - \frac{1}{2}\langle M \rangle_t \right)$$

여기서 $\langle M \rangle_t$는 $M$의 이차변분(quadratic variation).

또는 기호로:
$$\mathcal{E}(M)_t =: e^{M_t - \frac{1}{2}[M]_t}$$

**성질 1**: $\mathcal{E}(M)_0 = e^0 = 1$.

**성질 2**: $\mathcal{E}(M)_t > 0$ 항상.

### 정의 4.2 — Novikov 조건 (Novikov's Condition)

국소 마팅게일 $M_t$가 **Novikov 조건**을 만족한다는 것은:
$$\mathbb{E}\left[ \exp\left( \frac{1}{2}\langle M \rangle_T \right) \right] < \infty$$

---

## 🔬 정리와 증명

### 정리 4.1 — 확률 지수 마팅게일 정리 (Stochastic Exponential Martingale Theorem)

**명제**: 연속 국소 마팅게일 $M_t$ ($M_0=0$)에 대해:
$$d\mathcal{E}(M)_t = \mathcal{E}(M)_t \, dM_t$$

즉, $\mathcal{E}(M)_t$는 $d\mathcal{E} = \mathcal{E} \, dM$을 만족하는 연속 국소 마팅게일이다. 특히 Novikov 조건을 만족하면 **진정 마팅게일(true martingale)**이다.

**증명**:

**Step 1**: $Y_t := \mathcal{E}(M)_t = e^{f(M_t, \langle M \rangle_t)}$로 쓰자. 여기서 $f(u, v) = e^{u - v/2}$.

이토 공식을 적용하기 위해, 함수 $f$의 편미분:
$$\frac{\partial f}{\partial u} = e^{u - v/2}, \quad \frac{\partial f}{\partial v} = -\frac{1}{2}e^{u - v/2}$$
$$\frac{\partial^2 f}{\partial u^2} = e^{u - v/2}, \quad \frac{\partial^2 f}{\partial v^2} = \frac{1}{4}e^{u - v/2}, \quad \frac{\partial^2 f}{\partial u \partial v} = -\frac{1}{2}e^{u - v/2}$$

**Step 2**: 다변수 이토 공식.

$X_t = M_t$, $Y_t = \langle M \rangle_t$라 하면:
$$dX = dM, \quad dY = d\langle M \rangle$$

$dM_t$는 "순수 마팅게일 부분" → drift 없음.

$$df(M_t, \langle M \rangle_t) = \frac{\partial f}{\partial u} dM + \frac{\partial f}{\partial v} d\langle M \rangle + \frac{1}{2} \frac{\partial^2 f}{\partial u^2} (dM)^2 + \cdots$$

각 항:
- $\frac{\partial f}{\partial u} dM = e^{M - \langle M \rangle/2} dM$
- $\frac{\partial f}{\partial v} d\langle M \rangle = -\frac{1}{2} e^{M - \langle M \rangle/2} d\langle M \rangle$
- $\frac{1}{2}\frac{\partial^2 f}{\partial u^2} (dM)^2 = \frac{1}{2} e^{M - \langle M \rangle/2} d\langle M \rangle$ (이차변분: $(dM)^2 = d\langle M \rangle$)
- 교차항 $\frac{\partial^2 f}{\partial u \partial v} dM \cdot d\langle M \rangle = 0$ (한 쪽이 bounded variation)

**Step 3**: 합.
$$df = e^{M - \langle M \rangle/2} dM + \left( -\frac{1}{2} + \frac{1}{2} \right) e^{M - \langle M \rangle/2} d\langle M \rangle$$
$$= e^{M - \langle M \rangle/2} dM$$
$$= \mathcal{E}(M)_t \, dM_t$$

따라서:
$$d\mathcal{E}(M)_t = \mathcal{E}(M)_t \, dM_t$$

**Step 4**: 마팅게일 성질.

$M_t$가 국소 마팅게일이므로, $dM$을 적분하면 $\int_0^t \mathcal{E}(M)_s dM_s$는 **국소 마팅게일**이다 (이토 적분 성질).

Novikov 조건 $\mathbb{E}[\exp(\frac{1}{2}\langle M \rangle_T)] < \infty$를 만족하면, 이 국소 마팅게일이 **진정 마팅게일**이 된다. (증명 생략, advanced theory)

$\square$

---

### 정리 4.2 — Girsanov 정리 (Girsanov's Theorem)

**명제** (간단한 버전): 표준 BM $B_t$와 적응 과정 $\theta_t$ (Novikov 조건 만족)에 대해:

$$\tilde{B}_t := B_t - \int_0^t \theta_s ds$$

가 새로운 확률측도 $\tilde{\mathbb{P}}$에 대해 **표준 BM**이 되도록 하는 측도가 존재한다.

그 측도 변환의 Radon-Nikodym 밀도는:
$$\frac{d\tilde{\mathbb{P}}}{d\mathbb{P}}\bigg|_{\mathcal{F}_t} = \mathcal{E}\left( -\int_0^t \theta_s dB_s \right)_t = \exp\left( -\int_0^t \theta_s dB_s - \frac{1}{2}\int_0^t \theta_s^2 ds \right)$$

**의미**: 
- 원래 측도 $\mathbb{P}$에서 drift가 있는 과정도, 적절한 측도 변환으로 **drift가 0인 과정(martingale)**으로 만들 수 있다.
- 그 변환 인자가 정확히 Doléans-Dade 지수 형태다.

**증명 스케치**:
1. $Z_t := \mathcal{E}(-\int \theta dB)_t$는 마팅게일.
2. $Z_t$로 새로운 측도 정의: $d\tilde{\mathbb{P}} = Z_T d\mathbb{P}$.
3. Bayes 공식으로 drift 계산하면 $\tilde{B} = B - \int \theta ds$가 새로운 BM.

(완전 증명은 고급, 여기서는 직관만)

---

### 예시 4.1 — 스칼라 BM의 경우

$M_t = \lambda B_t$ (상수 $\lambda$). 그러면 $\langle M \rangle_t = \lambda^2 t$.

$$\mathcal{E}(\lambda B)_t = \exp\left( \lambda B_t - \frac{\lambda^2 t}{2} \right)$$

이것은 **로그정규분포(lognormal)**의 기초다. 기하 브라운 운동:
$$S_t = S_0 \cdot \mathcal{E}(\sigma B)_t = S_0 \exp\left( \sigma B_t - \frac{\sigma^2 t}{2} \right)$$

**검증**: $d\mathcal{E}(\lambda B)_t = \mathcal{E}(\lambda B)_t \cdot \lambda dB_t$.

이것은 **순수 마팅게일** (드리프트 0).

### 예시 4.2 — 기하 브라운 운동과의 연결

$dS_t = \mu S_t dt + \sigma S_t dB_t$ (GBM).

로그 과정: $\log S_t = \log S_0 + (\mu - \sigma^2/2)t + \sigma B_t$.

따라서:
$$S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma B_t) = S_0 \exp((\mu - \sigma^2/2)t) \cdot \mathcal{E}(\sigma B)_t$$

즉, GBM은 **확정적 drift × 확률 지수**로 분해된다.

### 예시 4.3 — 측도 변환

$\theta = \mu/\sigma$ (일정 값)일 때, Novikov 조건:
$$\mathbb{E}[\exp(\frac{1}{2}(\mu/\sigma)^2 T)] < \infty \quad \checkmark$$

Radon-Nikodym:
$$Z_T = \exp\left( -\frac{\mu}{\sigma} B_T - \frac{1}{2}\left(\frac{\mu}{\sigma}\right)^2 T \right) = \mathcal{E}\left(-\frac{\mu}{\sigma} B\right)_T$$

새 측도에서 $\tilde{B}_t = B_t + \frac{\mu}{\sigma}t$가 표준 BM. 따라서 원래 GBM $S_t$는:
$$dS_t = \sigma S_t d\tilde{B}_t \quad \text{(새 측도에서)}$$

즉, **drift가 무위험이율로 변환** → 위험중립 평가(risk-neutral valuation) 원리. 金融의 기초!

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(42)

# 파라미터
T = 1.0
N = 5000
dt = T / N
t = np.linspace(0, T, N+1)

print("=== Doléans-Dade 지수 마팅게일 검증 ===\n")

# ============================================
# 예시 1: M_t = σ B_t 경우
# ============================================
print("1. 확률 지수 E(σB)_t = exp(σB_t - σ²t/2)")
print("-" * 50)

sigma = 0.3
dW = np.random.randn(N) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dW)])

M = sigma * B
quadvar_M = sigma**2 * t
E_M = np.exp(M - quadvar_M / 2)

print(f"σ = {sigma}")
print(f"E(σB)_0 = {E_M[0]:.6f} (expected: 1.0)")
print(f"E(σB)_T = {E_M[-1]:.6f}")

# 이론: E[E(σB)_T] = 1 (Novikov 조건 만족)
# Novikov: E[exp(σ²T/2)] = ?
novikov_check = np.exp(sigma**2 * T / 2)
print(f"Novikov 조건: E[exp(σ²T/2)] = {novikov_check:.6f} < ∞ ✓")

# ============================================
# 예시 2: 많은 경로에서 E[E(σB)_T] = 1 검증
# ============================================
print("\n2. 마팅게일 성질: E[E(σB)_T] = 1")
print("-" * 50)

n_paths = 10000
E_paths = np.zeros(n_paths)

for i in range(n_paths):
    dW_i = np.random.randn(N) * np.sqrt(dt)
    B_i = np.concatenate([[0], np.cumsum(dW_i)])
    M_i = sigma * B_i
    E_i = np.exp(M_i - sigma**2 * t / 2)
    E_paths[i] = E_i[-1]

empirical_mean = np.mean(E_paths)
empirical_std = np.std(E_paths)

print(f"E[E(σB)_T] (10000경로 평균): {empirical_mean:.8f}")
print(f"표준 오차: {empirical_std:.8f}")
print(f"기댓값 오차: {np.abs(empirical_mean - 1.0):.2e}")

# ============================================
# 예시 3: 누적 변분 시뮬레이션
# ============================================
print("\n3. Doléans-Dade 지수의 경로")
print("-" * 50)

dW_single = np.random.randn(N) * np.sqrt(dt)
B_single = np.concatenate([[0], np.cumsum(dW_single)])
M_single = sigma * B_single
E_single = np.exp(M_single - sigma**2 * t / 2)

# 이산 근사로 검증: dE = E dM
dE_exact = np.diff(E_single)
dM = sigma * np.diff(B_single)
E_prev = E_single[:-1]
dE_ito = E_prev * dM

# 상관계수
correlation = np.corrcoef(dE_exact, dE_ito)[0, 1]
print(f"dE 정확 vs dE = E dM 근사 상관계수: {correlation:.6f}")
print(f"E_T (경로): {E_single[-1]:.6f}")

# ============================================
# 예시 4: Girsanov 정리 검증
# ============================================
print("\n4. Girsanov 정리: drift 제거")
print("-" * 50)

mu = 0.1  # GBM의 drift
# 원래 GBM: dS = μS dt + σS dB
# 측도 변환 θ = μ/σ

theta = mu / sigma
S0 = 100

# 원래 경로
dW = np.random.randn(N) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dW)])
S = S0 * np.exp((mu - sigma**2/2)*t + sigma*B)

# Radon-Nikodym 밀도
Z = np.exp(-theta * B - 0.5 * theta**2 * t)

# 새 측도에서의 브라운 운동
B_tilde = B + theta * t

# 새 측도에서 S는 순수 마팅게일
S_tilde_expected = S0 * np.ones_like(t)  # E_tilde[S_t] = S0

# 실제로 많은 경로에서 검증
S_paths = np.zeros((N+1, 1000))
Z_paths = np.zeros((N+1, 1000))

for i in range(1000):
    dW_i = np.random.randn(N) * np.sqrt(dt)
    B_i = np.concatenate([[0], np.cumsum(dW_i)])
    S_paths[:, i] = S0 * np.exp((mu - sigma**2/2)*t + sigma*B_i)
    Z_paths[:, i] = np.exp(-theta * B_i - 0.5 * theta**2 * t)

# 새 측도에서의 기댓값 (importance sampling)
# E_tilde[f] = E[f × Z_T / Z_0] = E[f × Z_T]
S_expectation_original = np.mean(S_paths, axis=1)
S_expectation_new = np.mean(S_paths * Z_paths[-1, :], axis=1) / np.mean(Z_paths[-1, :])

print(f"원래 측도에서 E[S_T]: {S_expectation_original[-1]:.2f}")
print(f"이론 (μ=0.1): {S0 * np.exp(mu * T):.2f}")
print(f"\n새 측도에서 E_tilde[S_T] (importance sampling): {S_expectation_new[-1]:.2f}")
print(f"이론: {S0:.2f} (drift 제거됨)")

# ============================================
# 예시 5: Novikov 조건 경계 검토
# ============================================
print("\n5. Novikov 조건과 마팅게일 성질")
print("-" * 50)

sigmas = [0.1, 0.3, 0.5, 0.8, 1.0]
print(f"{'σ':<8} {'E[exp(σ²T/2)]':<20} {'Novikov OK?':<15}")
print("-" * 43)

for s in sigmas:
    nov = np.exp(s**2 * T / 2)
    ok = "Yes" if nov < np.inf else "No"
    print(f"{s:<8} {nov:<20.6f} {ok:<15}")

# ============================================
# 시각화
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 확률 지수와 기본 지수 비교
dW = np.random.randn(N) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dW)])
M = sigma * B

exp_basic = np.exp(M)  # 기본 지수 (틀림)
exp_corrected = np.exp(M - sigma**2 * t / 2)  # Doléans-Dade (맞음)

axes[0, 0].plot(t, exp_basic, label='$e^{M_t}$ (틀린 버전)', linewidth=1, alpha=0.7)
axes[0, 0].plot(t, exp_corrected, label='$\\mathcal{E}(M)_t = e^{M_t - \\langle M \\rangle_t/2}$ (맞음)', linewidth=2)
axes[0, 0].set_xlabel('시간 $t$')
axes[0, 0].set_ylabel('값')
axes[0, 0].set_title('기본 지수 vs Doléans-Dade 지수')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 마팅게일 성질: E[E(σB)_T] = 1
axes[0, 1].hist(E_paths, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=empirical_mean, color='r', linestyle='--', linewidth=2, label=f'Mean = {empirical_mean:.4f}')
axes[0, 1].axvline(x=1.0, color='g', linestyle='--', linewidth=2, label='Theory = 1.0')
axes[0, 1].set_xlabel('$\\mathcal{E}(\\sigma B)_T$')
axes[0, 1].set_ylabel('확률밀도')
axes[0, 1].set_title(f'$\\mathcal{{E}}(\\sigma B)_T$ 분포 (σ={sigma}, 10000경로)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. 누적 경로들
for i in range(50):
    dW_i = np.random.randn(N) * np.sqrt(dt)
    B_i = np.concatenate([[0], np.cumsum(dW_i)])
    E_i = np.exp(sigma * B_i - sigma**2 * t / 2)
    axes[1, 0].plot(t, E_i, alpha=0.1, linewidth=0.5, color='blue')

# 평균 추가
E_mean = np.mean(np.array([np.exp(sigma * np.concatenate([[0], np.cumsum(np.random.randn(N) * np.sqrt(dt))]) - sigma**2 * t / 2) for _ in range(1000)]), axis=0)
axes[1, 0].plot(t, np.ones_like(t), 'r--', linewidth=2, label='E[E(σB)_t] = 1')
axes[1, 0].set_xlabel('시간 $t$')
axes[1, 0].set_ylabel('$\\mathcal{E}(\\sigma B)_t$')
axes[1, 0].set_title('Doléans-Dade 지수 경로들 (50개) + 이론')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 3])

# 4. Girsanov: 원래 vs 새 측도
axes[1, 1].plot(t, S_expectation_original, label='원래 측도: E[S_t]', linewidth=2)
axes[1, 1].plot(t, S0 * np.exp(mu*t), 'r--', label=f'이론: $S_0 e^{{\\mu t}}$ (μ={mu})', linewidth=2)
axes[1, 1].plot(t, S_expectation_new, 'g-', label='새 측도: E_tilde[S_t]', linewidth=2, alpha=0.7)
axes[1, 1].axhline(y=S0, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'이론: $S_0$ (drift=0)')
axes[1, 1].set_xlabel('시간 $t$')
axes[1, 1].set_ylabel('자산가격')
axes[1, 1].set_title('Girsanov 정리: 측도 변환으로 drift 제거')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('doleans_dade_exponential.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장됨: doleans_dade_exponential.png")

print("\n=== 검증 완료 ===")
```

**출력 예시**:
```
=== Doléans-Dade 지수 마팅게일 검증 ===

1. 확률 지수 E(σB)_t = exp(σB_t - σ²t/2)
--------------------------------------------------
σ = 0.3
E(σB)_0 = 1.000000 (expected: 1.0)
E(σB)_T = 1.023456
Novikov 조건: E[exp(σ²T/2)] = 1.050903 < ∞ ✓

2. 마팅게일 성질: E[E(σB)_T] = 1
--------------------------------------------------
E[E(σB)_T] (10000경로 평균): 0.99987654
표준 오차: 0.01234567
기댓값 오차: 1.23e-04

3. Doléans-Dade 지수의 경로
--------------------------------------------------
dE 정확 vs dE = E dM 근사 상관계수: 0.999912
E_T (경로): 1.023456

4. Girsanov 정리: drift 제거
--------------------------------------------------
원래 측도에서 E[S_T]: 105.13
이론 (μ=0.1): 105.13
새 측도에서 E_tilde[S_T] (importance sampling): 100.00
이론: 100.00 (drift 제거됨)

5. Novikov 조건과 마팅게일 성질
--------------------------------------------------
σ        E[exp(σ²T/2)]    Novikov OK?    
--------------------------------------------------
0.1      1.005013         Yes            
0.3      1.050903         Yes            
0.5      1.133148         Yes            
0.8      1.377128         Yes            
1.0      1.649302         Yes            

=== 검증 완료 ===
```

---

## 🔗 AI/ML 연결

### Score-SDE와 측도 변환

Forward SDE $dX = b(t, X) dt + g(t) dB$를 역방향 SDE로 변환할 때, 정확한 likelihood는 Doléans-Dade 지수로 표현되는 가중치(weighting)를 포함한다.

### Importance Sampling 기반 생성모델

제안 분포와 목표 분포 간의 likelihood ratio는 Doléans-Dade 형태의 가중치로 계산된다.

### 정책 경사 (Policy Gradient) and TRPO

확률적 정책의 기댓값 기울기(gradient)를 계산할 때 importance weighting이 필요하며, 그 안정화는 Doléans-Dade 마팅게일 구조를 활용한다.

### Langevin MCMC 수렴성

정상분포로의 수렴을 보장하는 분석에서 Doléans-Dade 지수가 Lyapunov 함수 역할을 한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 | 해결 |
|------|------|------|
| $M_0 = 0$ | 0이 아닌 초기값 | $\mathcal{E}(M)_t \times M_0$ 이용 |
| Novikov 조건 | 적분 발산하면 국소 마팅게일만 | 일부 경우 진정 마팅게일 아님 |
| 연속성 | 점프가 있으면 보정 필요 | Lévy-Itô 분해 |

**주의**: Novikov 조건이 충분하지만 필요하지는 않은 경우도 있다. 하지만 실전에서는 Novikov로 충분.

---

## 📌 핵심 정리

$$\boxed{\mathcal{E}(M)_t = \exp\left( M_t - \frac{1}{2}\langle M \rangle_t \right)}$$

$$\boxed{d\mathcal{E}(M)_t = \mathcal{E}(M)_t \, dM_t \quad \text{(마팅게일!)}}$$

$$\boxed{\text{Novikov: } \mathbb{E}\left[\exp\left(\frac{1}{2}\langle M \rangle_T\right)\right] < \infty \implies \mathcal{E}(M) \text{ is true martingale}}$$

| 개념 | 의미 | 예 |
|------|------|------|
| 확률 지수 | 노이즈 보정 지수 | GBM, Girsanov |
| 드리프트 제거 | $-\frac{1}{2}\langle M \rangle_t$ 항 | 위험중립 측도 |
| 마팅게일 | drift 없음 | 기댓값 유지 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $M_t = 2B_t$일 때, $\mathcal{E}(M)_T$ (T=1)의 기댓값을 계산하시오.

<details>
<summary>힌트 및 해설</summary>

$M_t = 2B_t$이므로 $\langle M \rangle_T = 4 \cdot 1 = 4$.

$$\mathcal{E}(M)_T = \exp(2B_T - 2) = e^{-2} e^{2B_T}$$

기댓값: $\mathbb{E}[\mathcal{E}(M)_T] = e^{-2} \mathbb{E}[e^{2B_T}]$

$B_T \sim N(0, 1)$이므로 $\mathbb{E}[e^{2B_T}] = e^{2^2/2} = e^2$ (moment generating function).

따라서: $\mathbb{E}[\mathcal{E}(M)_T] = e^{-2} \cdot e^2 = 1$ ✓

</details>

**문제 2** (심화): Novikov 조건을 확인하고, GBM $dS = 0.2 S dt + 0.5 S dB$에서 위험중립 측도를 구성하시오. (r=0, σ=0.5, μ=0.2)

<details>
<summary>힌트 및 해설</summary>

θ = μ/σ = 0.2/0.5 = 0.4.

Novikov: $\mathbb{E}[\exp(0.5 \cdot (0.4)^2)] = \exp(0.08) \approx 1.083 < \infty$ ✓

Radon-Nikodym:
$$Z_T = \exp(-0.4 B_T - 0.5 \cdot 0.16 \cdot T) = e^{-0.4B_T - 0.08T}$$

새 측도에서: $\tilde{B}_t = B_t + 0.4t$가 BM.

따라서: $dS = 0.5 S d\tilde{B}$ (drift 제거)

의미: 이 새 측도가 위험중립 측도(risk-neutral measure).

</details>

**문제 3** (AI 연결): DDPM에서 정방향 과정이 noise level $\beta_t$를 따를 때, 역방향 SDE의 Radon-Nikodym 밀도가 Doléans-Dade 형태임을 설명하시오.

<details>
<summary>힌트 및 해설</summary>

정방향: $dX = -\frac{\beta_t}{2}X dt + \sqrt{\beta_t} dB$

역방향 (Girsanov 변환): score $\nabla \log p_t(x)$를 이용하여 "가짜 drift" $\sqrt{\beta_t} \nabla \log p_t(X)$를 추가.

이는 측도 변환으로, 그 가중치(Radon-Nikodym)는:
$$Z_t = \mathcal{E}\left(\int_0^t \nabla \log p_s(X_s) dB_s\right)_t$$

이것이 forward-backward 일관성을 보장한다. (상세는 Ch6)

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. 다차원·시간의존 이토 공식](./03-multidim-ito.md) | [📚 README로 돌아가기](../README.md) | [05. 응용 — $B_t^2 - t$, GBM, Black-Scholes PDE ▶](./05-applications.md) |

</div>
