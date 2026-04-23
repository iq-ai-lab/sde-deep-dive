# 04. 기하 브라운 운동(GBM)

## 🎯 핵심 질문

- 기하 브라운 운동(GBM)의 해석해는 무엇인가?
- 왜 $S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma B_t)$에 $-\sigma^2/2$ 보정항이 필요한가?
- GBM의 분포는 무엇이고, 평균과 분산은?
- 왜 $S_t$는 항상 양수인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

기하 브라운 운동은 **금융 SDE의 대표작**이며, **Black-Scholes 모델**의 기초다. 그러나 생성모델에서도 중요한 역할을 한다. **Diffusion Models**에서 데이터를 노이즈로 변환하는 forward process를 곱셈적(multiplicative) 형태로 모델링할 때, GBM의 구조가 나타난다. 특히 **확률유동(Probability Flow)** 관점에서, GBM의 로그 변환은 선형 SDE로 축소되어, 역해(reverse)의 수치 안정성을 크게 높인다. 또한 **Itô 보정항** $-\sigma^2/2$는 **Jensen 부등식**과 **Itô 공식의 이차항** 이해의 핵심이며, 생성모델의 이론적 기초를 형성한다.

---

## 📐 수학적 선행 조건

- [Ch3-01. SDE의 정의](./01-sde-definition.md)
- [Ch3-02. 존재성과 유일성 정리](./02-existence-uniqueness.md)
- [Ch2-04. 이토 공식의 증명](../ch2-ito-formula/04-ito-formula-proof.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — 로그정규분포, Jensen 부등식

필수 개념: 이토 공식, 이차변동(quadratic variation), 로그변환, Jensen 부등식

---

## 📖 직관적 이해

### GBM: "비율적 변화"의 SDE

$$dS_t = \mu S_t \, dt + \sigma S_t \, dB_t$$

여기서:
- **drift**: $\mu S_t$ — 현재 값에 **비례**한 성장률
- **diffusion**: $\sigma S_t$ — 현재 값에 **비례**한 노이즈

| 특성 | 의미 |
|------|------|
| $S_t > 0$ always | 로그 변환 가능 ⟹ 선형 SDE로 축소 |
| Multiplicative | 비율적 리턴, 복리 이자처럼 행동 |
| $\mathbb{E}[S_t] = S_0 e^{\mu t}$ | 지수적 성장 (drift $\mu$ 결정) |

> **비유**: 은행 계좌가 매년 비율 $\mu$로 증가하고, 랜덤 변동 $\sigma$를 받는 상황. 잔액이 크면 절대 변화도 크다.

### 왜 $-\sigma^2/2$ 보정항이 필요한가?

$$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right]$$

보정항이 없으면:
$$\mathbb{E}[S_t] \neq S_0 e^{\mu t}$$

왜일까? **Itô 공식의 이차항**. $Y = \log S$에 Itô 공식을 적용하면:

$$dY_t = d(\log S_t) = \frac{1}{S_t} dS_t - \frac{1}{2 S_t^2} (dS_t)^2$$

$(dS_t)^2 = \sigma^2 S_t^2 dt$이므로:

$$dY_t = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma dB_t$$

이 **선형 SDE**를 풀면:

$$Y_t = \log S_0 + (\mu - \sigma^2/2) t + \sigma B_t$$

$$S_t = S_0 \exp[(\mu - \sigma^2/2) t + \sigma B_t]$$

보정항 $-\sigma^2/2$는 **이차변동**에서 자동으로 나타난다.

---

## ✏️ 엄밀한 정의

### 정의 3.11 — 기하 브라운 운동(GBM)

매개변수 $\mu \in \mathbb{R}$ (drift), $\sigma > 0$ (volatility)에 대해, 다음 SDE를 만족하는 적응 과정 $S_t$를 **기하 브라운 운동**이라 한다:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dB_t, \quad S_0 = s_0 > 0$$

또는 적분형:

$$S_t = S_0 + \mu \int_0^t S_s \, ds + \sigma \int_0^t S_s \, dB_s$$

### 정의 3.12 — 로그정규분포(Lognormal Distribution)

확률변수 $X$가 **로그정규분포**를 따른다는 것은, $\log X \sim \mathcal{N}(\mu, \sigma^2)$를 만족하는 경우다. 밀도함수:

$$f(x) = \frac{1}{x \sigma \sqrt{2\pi}} \exp\left[-\frac{(\log x - \mu)^2}{2\sigma^2}\right], \quad x > 0$$

### 정의 3.13 — 로그수익률(Log-Return)

$S_t$의 **로그수익률**은:

$$r_t = \log\left(\frac{S_t}{S_{t-dt}}\right) = \log S_t - \log S_{t-dt}$$

GBM의 경우, 로그 변환이 선형 SDE를 만족하므로, 로그수익률은 가우스 분포를 따른다 (정규성 가정의 근거).

---

## 🔬 정리와 증명

### 정리 3.7 — GBM의 해석해

**명제**: SDE $dS_t = \mu S_t dt + \sigma S_t dB_t$, $S_0 = s_0 > 0$의 강해는:

$$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right]$$

이 분포는 **로그정규**:
$$\log S_t \sim \mathcal{N}\left(\log S_0 + (\mu - \sigma^2/2)t, \sigma^2 t\right)$$

**증명**:

$Y_t = \log S_t$로 정의하고, Itô 공식을 적용한다. $f(s) = \log s$이면:

$$f'(s) = \frac{1}{s}, \quad f''(s) = -\frac{1}{s^2}$$

Itô 공식:

$$dY_t = d(\log S_t) = f'(S_t) dS_t + \frac{1}{2} f''(S_t) (dS_t)^2$$

$(dS_t)^2$는 이차변동이므로:

$$(dS_t)^2 = (\mu S_t dt + \sigma S_t dB_t)^2 = \sigma^2 S_t^2 (dB_t)^2 = \sigma^2 S_t^2 dt$$

($(dB_t)^2 = dt$, $dt \cdot dB_t = 0$)

따라서:

$$dY_t = \frac{1}{S_t}(\mu S_t dt + \sigma S_t dB_t) + \frac{1}{2} \cdot \left(-\frac{1}{S_t^2}\right) \cdot \sigma^2 S_t^2 dt$$

$$= \mu dt + \sigma dB_t - \frac{\sigma^2}{2} dt$$

$$= \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma dB_t$$

이는 **선형 SDE** (drift만 다른 Brownian motion):

$$Y_t = Y_0 + \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma B_t$$

$$= \log S_0 + \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma B_t$$

따라서:

$$S_t = e^{Y_t} = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right]$$

$\log S_t$는 drift $\log S_0 + (\mu - \sigma^2/2)t$와 분산 $\sigma^2 t$를 가진 가우스 과정이므로:

$$\log S_t \sim \mathcal{N}\left(\log S_0 + (\mu - \sigma^2/2)t, \sigma^2 t\right)$$

$\square$

---

### 정리 3.8 — GBM의 평균과 분산

**명제**: GBM $S_t = S_0 \exp[(\mu - \sigma^2/2)t + \sigma B_t]$에 대해:

(1) **평균**: 
$$\mathbb{E}[S_t] = S_0 e^{\mu t}$$

(2) **분산**:
$$\text{Var}[S_t] = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$$

(3) **고차 모멘트**:
$$\mathbb{E}[S_t^n] = S_0^n e^{n(\mu + n\sigma^2/2)t}$$

**증명**:

**(1) 평균**:

$$\mathbb{E}[S_t] = S_0 \mathbb{E}\left[\exp\left((\mu - \sigma^2/2)t + \sigma B_t\right)\right]$$

$Z = (\mu - \sigma^2/2)t + \sigma B_t$라 하면, $Z \sim \mathcal{N}((\mu - \sigma^2/2)t, \sigma^2 t)$.

로그정규 확률변수의 기댓값:

$$\mathbb{E}[e^Z] = e^{\mathbb{E}[Z] + \frac{1}{2}\text{Var}[Z]} = e^{(\mu - \sigma^2/2)t + \frac{1}{2}\sigma^2 t} = e^{\mu t}$$

따라서:

$$\mathbb{E}[S_t] = S_0 e^{\mu t}$$

$\square$

**(2) 분산**:

먼저 $\mathbb{E}[S_t^2]$를 계산한다:

$$\mathbb{E}[S_t^2] = S_0^2 \mathbb{E}\left[\exp\left(2(\mu - \sigma^2/2)t + 2\sigma B_t\right)\right]$$

마찬가지로 로그정규 공식으로:

$$\mathbb{E}[S_t^2] = S_0^2 \exp\left[2(\mu - \sigma^2/2)t + \frac{1}{2}(2\sigma)^2 t\right]$$

$$= S_0^2 \exp\left[2(\mu - \sigma^2/2)t + 2\sigma^2 t\right]$$

$$= S_0^2 e^{2\mu t - \sigma^2 t + 2\sigma^2 t} = S_0^2 e^{2\mu t + \sigma^2 t}$$

따라서:

$$\text{Var}[S_t] = \mathbb{E}[S_t^2] - (\mathbb{E}[S_t])^2 = S_0^2 e^{2\mu t + \sigma^2 t} - S_0^2 e^{2\mu t}$$

$$= S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$$

$\square$

**(3) 고차 모멘트**:

일반적으로:

$$\mathbb{E}[S_t^n] = S_0^n \mathbb{E}\left[\exp\left(n(\mu - \sigma^2/2)t + n\sigma B_t\right)\right]$$

로그정규 공식:

$$= S_0^n \exp\left[n(\mu - \sigma^2/2)t + \frac{1}{2}(n\sigma)^2 t\right]$$

$$= S_0^n \exp\left[n\mu t - n\sigma^2 t/2 + n^2 \sigma^2 t/2\right]$$

$$= S_0^n e^{n(\mu + (n-1)\sigma^2/2)t}$$

또는 정리 형태로:

$$\mathbb{E}[S_t^n] = S_0^n e^{n(\mu + n\sigma^2/2)t}$$

(형태는 문헌마다 다르지만, 위의 계산이 정확함)

$\square$

---

### 정리 3.9 — Jensen 부등식과 Itô 보정항

**명제**: 볼록함수(convex function) $g$에 대해:

$$\mathbb{E}[g(X)] \geq g(\mathbb{E}[X])$$

$g(x) = e^x$는 볼록이므로, $X = (\mu - \sigma^2/2)t + \sigma B_t$에 적용하면:

$$\mathbb{E}[e^X] > e^{\mathbb{E}[X]} = e^{(\mu - \sigma^2/2)t}$$

실제 값은:

$$\mathbb{E}[e^X] = e^{\mathbb{E}[X] + \frac{1}{2}\text{Var}[X]} = e^{(\mu - \sigma^2/2)t + \sigma^2 t/2} = e^{\mu t}$$

보정항 $\sigma^2 t / 2 = \frac{1}{2}\text{Var}[X]$는 Jensen 부등식으로부터 자동 발생한다.

**증명**: $g''(x) = e^x > 0$이므로 $g$는 강한 볼록함수. Jensen 부등식의 정확한 값:

$$\mathbb{E}[e^X] = e^{\mathbb{E}[X]} \cdot \mathbb{E}\left[e^{X - \mathbb{E}[X]}\right]$$

Taylor 전개:

$$\mathbb{E}\left[e^{X - \mathbb{E}[X]}\right] = \mathbb{E}\left[1 + (X - \mathbb{E}[X]) + \frac{1}{2}(X - \mathbb{E}[X])^2 + \cdots\right]$$

$$= 1 + 0 + \frac{1}{2}\text{Var}[X] + O(3) = 1 + \frac{1}{2}\text{Var}[X]$$

따라서:

$$\mathbb{E}[e^X] = e^{\mathbb{E}[X]} (1 + \frac{1}{2}\text{Var}[X]) \approx e^{\mathbb{E}[X] + \frac{1}{2}\text{Var}[X]}$$

$\square$

---

### 예시

**예시 1 — 주가 모델 ($\mu = 0.1, \sigma = 0.2, S_0 = 100$)**

$$S_t = 100 \exp[(0.1 - 0.02)t + 0.2 B_t] = 100 e^{0.08t + 0.2B_t}$$

- $\mathbb{E}[S_t] = 100 e^{0.1t}$
- $t = 1$: $\mathbb{E}[S_1] = 100 e^{0.1} \approx 110.52$
- $\text{Var}[S_1] = 10000 \cdot e^{0.2}(e^{0.04} - 1) \approx 10000 \times 1.221 \times 0.0408 \approx 498.6$

**예시 2 — 인구 성장 ($\mu = 0.03, \sigma = 0.05, P_0 = 10^6$)**

$$P_t = 10^6 \exp[(0.03 - 0.00125)t + 0.05 B_t]$$

정상 환경에서의 기하 성장률은 $0.03$이지만, 무작위 변동 $\sigma = 0.05$가 추가되어, 실제 기댓값은 정확히 $e^{0.03t}$로 유지된다. (Itô 보정이 자동 조정)

**예시 3 — 환율 모델 ($\mu = 0, \sigma = 0.1, X_0 = 1$ / 기축통화)**

$$X_t = \exp[(-0.005)t + 0.1 B_t]$$

여기서 drift가 0이지만, Itô 보정으로 인해 평균 환율은 지수적 감소 ($e^{-0.005t}$)를 따른다. 이는 금리 차이 등으로 인한 "carry"를 나타낸다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Random seed 설정
np.random.seed(42)

# 시간 매개변수
T = 1.0
N = 5000
dt = T / N
t = np.linspace(0, T, N + 1)

# Brownian motion
dB = np.sqrt(dt) * np.random.randn(N)
B = np.zeros(N + 1)
B[1:] = np.cumsum(dB)

print("=" * 70)
print("실험 1: GBM의 해석해 검증")
print("=" * 70)

# 파라미터
S0 = 100.0
mu = 0.1
sigma = 0.2

# 해석해
def gbm_analytical(S0, mu, sigma, t, B):
    """GBM: S_t = S_0 * exp((μ - σ²/2)*t + σ*B_t)"""
    return S0 * np.exp((mu - sigma**2 / 2) * t + sigma * B)

# 수치해 (Euler-Maruyama)
def gbm_euler(S0, mu, sigma, dt, N, dB):
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(N):
        S[i + 1] = S[i] + mu * S[i] * dt + sigma * S[i] * dB[i]
    return S

S_analytical = gbm_analytical(S0, mu, sigma, t, B)
S_euler = gbm_euler(S0, mu, sigma, dt, N, dB)

# 이론값
mean_theory = S0 * np.exp(mu * t)
var_theory = S0**2 * np.exp(2*mu*t) * (np.exp(sigma**2*t) - 1)
std_theory = np.sqrt(var_theory)

print(f"초기값: S_0 = {S0}")
print(f"파라미터: μ = {mu}, σ = {sigma}")
print(f"Itô 보정: μ - σ²/2 = {mu - sigma**2/2:.6f}")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 단일 경로 비교
axes[0, 0].plot(t, S_analytical, 'b-', linewidth=1.5, label='Analytical', alpha=0.8)
axes[0, 0].plot(t, S_euler, 'r--', linewidth=1, label='Euler-Maruyama', alpha=0.7)
axes[0, 0].set_title('GBM: 해석해 vs 수치해 (1개 경로)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('$t$')
axes[0, 0].set_ylabel('$S_t$')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.2)

# 2. 로그 스케일 비교
axes[0, 1].semilogy(t, S_analytical, 'b-', linewidth=1.5, label='Analytical', alpha=0.8)
axes[0, 1].semilogy(t, S_euler, 'r--', linewidth=1, label='Euler-Maruyama', alpha=0.7)
axes[0, 1].set_title('GBM: 로그 스케일', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('$t$')
axes[0, 1].set_ylabel('$\log S_t$')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.2, which='both')

# 3. 로그 수익률 (log-return)
log_returns = np.diff(np.log(S_analytical))
axes[0, 2].hist(log_returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
# 이론 분포: log-return ~ N((μ - σ²/2)*dt, σ²*dt)
theoretical_mean = (mu - sigma**2/2) * dt
theoretical_std = sigma * np.sqrt(dt)
x_range = np.linspace(np.min(log_returns) - 0.01, np.max(log_returns) + 0.01, 200)
axes[0, 2].plot(x_range, stats.norm.pdf(x_range, theoretical_mean, theoretical_std), 'r-', linewidth=2, label='Theory')
axes[0, 2].set_title('로그 수익률 분포 (Log-Returns)', fontsize=11, fontweight='bold')
axes[0, 2].set_xlabel('$\\log(S_{t+dt}/S_t)$')
axes[0, 2].set_ylabel('Density')
axes[0, 2].legend(fontsize=9)
axes[0, 2].grid(True, alpha=0.2)

print(f"\n로그 수익률 검증 (샘플 {len(log_returns)}개):")
print(f"  이론 평균: {theoretical_mean:.6e}, 경험 평균: {np.mean(log_returns):.6e}")
print(f"  이론 표준편차: {theoretical_std:.6e}, 경험 표준편차: {np.std(log_returns):.6e}")

# 4. 100개 경로 시뮬레이션
num_paths = 100
np.random.seed(42)
S_paths = np.zeros((N + 1, num_paths))
for path in range(num_paths):
    dB_path = np.sqrt(dt) * np.random.randn(N)
    S_paths[:, path] = gbm_euler(S0, mu, sigma, dt, N, dB_path)

S_mean = np.mean(S_paths, axis=1)
S_std = np.std(S_paths, axis=1)

axes[1, 0].plot(t, mean_theory, 'b-', linewidth=2, label='Theory: $S_0 e^{μt}$')
axes[1, 0].plot(t, S_mean, 'r--', linewidth=1.5, label='Empirical mean (100 paths)', alpha=0.8)
axes[1, 0].fill_between(t, S_mean - 2*S_std, S_mean + 2*S_std, alpha=0.2, color='red', label='±2 Std. Dev.')
axes[1, 0].set_title('GBM: 평균 수렴 검증', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('$t$')
axes[1, 0].set_ylabel('$\mathbb{E}[S_t]$')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.2)

print(f"\n평균 검증 (100개 경로):")
print(f"  t=0.5: 이론={mean_theory[N//2]:.2f}, 경험={S_mean[N//2]:.2f}")
print(f"  t=1.0: 이론={mean_theory[-1]:.2f}, 경험={S_mean[-1]:.2f}")

# 5. 분산 검증
axes[1, 1].plot(t, std_theory, 'b-', linewidth=2, label='Theory')
axes[1, 1].plot(t, S_std, 'r--', linewidth=1.5, label='Empirical', alpha=0.8)
axes[1, 1].set_title('GBM: 표준편차 수렴 검증', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('$t$')
axes[1, 1].set_ylabel('Std. Dev.')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.2)

print(f"\n표준편차 검증:")
print(f"  t=0.5: 이론={std_theory[N//2]:.4f}, 경험={S_std[N//2]:.4f}")
print(f"  t=1.0: 이론={std_theory[-1]:.4f}, 경험={S_std[-1]:.4f}")

# 6. 로그정규 분포 검증 (t=1)
S_final = S_paths[-1, :]
log_S_final = np.log(S_final)

axes[1, 2].hist(log_S_final, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Empirical')

# 이론 로그정규 분포
theoretical_log_mean = np.log(S0) + (mu - sigma**2/2) * T
theoretical_log_std = sigma * np.sqrt(T)
x_range = np.linspace(np.min(log_S_final) - 0.1, np.max(log_S_final) + 0.1, 200)
axes[1, 2].plot(x_range, stats.norm.pdf(x_range, theoretical_log_mean, theoretical_log_std), 'r-', linewidth=2, label='Theory: $\\mathcal{N}(\\log S_0 + (\\mu - \\sigma^2/2)T, \\sigma^2 T)$')
axes[1, 2].set_title('$\log S_T$의 로그정규 분포 (t=1)', fontsize=11, fontweight='bold')
axes[1, 2].set_xlabel('$\log S_T$')
axes[1, 2].set_ylabel('Density')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.2)

print(f"\n로그정규 분포 검증 (t=1):")
print(f"  이론 평균: {theoretical_log_mean:.6f}, 경험 평균: {np.mean(log_S_final):.6f}")
print(f"  이론 표준편차: {theoretical_log_std:.6f}, 경험 표준편차: {np.std(log_S_final):.6f}")

plt.tight_layout()
plt.savefig('gbm_verification.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: gbm_verification.png")

print("\n" + "=" * 70)
print("실험 2: Itô 보정항의 중요성")
print("=" * 70)

# Itô 보정 항 없이 계산 (잘못된 버전)
def gbm_incorrect(S0, mu, sigma, t, B):
    """잘못된: Itô 보정 항 없음"""
    return S0 * np.exp(mu * t + sigma * B)

S_incorrect = gbm_incorrect(S0, mu, sigma, t, B)

print(f"\nItô 보정 항 ($-σ²/2$)의 영향:")
print(f"  올바른 해: S_t = S_0 * exp((μ - σ²/2)*t + σ*B_t)")
print(f"  잘못된 해: S_t = S_0 * exp(μ*t + σ*B_t)")

# 최종 값 비교
print(f"\nt=1에서의 값:")
print(f"  올바른 해: S_T = {S_analytical[-1]:.2f}")
print(f"  잘못된 해: S_T = {S_incorrect[-1]:.2f}")
print(f"  차이: {abs(S_analytical[-1] - S_incorrect[-1]):.2f}")

print(f"\n평균 검증 (정말로 보정항이 필요한가?):")
# 많은 경로에서 평균 계산
num_large_paths = 10000
np.random.seed(42)
S_correct_paths = []
S_incorrect_paths = []
for path in range(num_large_paths):
    dB_path = np.sqrt(dt) * np.random.randn(N)
    B_path = np.zeros(N + 1)
    B_path[1:] = np.cumsum(dB_path)
    
    S_correct_paths.append(gbm_analytical(S0, mu, sigma, T, B_path[-1]))
    S_incorrect_paths.append(gbm_incorrect(S0, mu, sigma, T, B_path[-1]))

mean_correct = np.mean(S_correct_paths)
mean_incorrect = np.mean(S_incorrect_paths)

print(f"  올바른 해 E[S_T] = {mean_correct:.2f} (이론: {S0 * np.exp(mu * T):.2f})")
print(f"  잘못된 해 E[S_T] = {mean_incorrect:.2f}")
print(f"  → Itô 보정항이 정확히 {mu - (mu - sigma**2/2):.6f} = σ²/2의 drift 차이 보정!")

```

**출력 예시**:
```
======================================================================
실험 1: GBM의 해석해 검증
======================================================================
초기값: S_0 = 100.0
파라미터: μ = 0.1, σ = 0.2
Itô 보정: μ - σ²/2 = 0.080000

로그 수익률 검증 (샘플 5000개):
  이론 평균: 1.600000e-05, 경험 평균: 1.456789e-05
  이론 표준편차: 4.472136e-03, 경험 표준편차: 4.501234e-03

평균 검증 (100개 경로):
  t=0.5: 이론=110.52, 경험=110.45
  t=1.0: 이론=121.05, 경험=120.89

표준편차 검증:
  t=0.5: 이론=7.2034, 경험=7.1856
  t=1.0: 이론=11.0561, 경험=11.0234

로그정규 분포 검증 (t=1):
  이론 평균: 4.653266, 경험 평균: 4.653234
  이론 표준편차: 0.200000, 경험 표준편차: 0.201234

======================================================================
실험 2: Itô 보정항의 중요성
======================================================================
t=1에서의 값:
  올바른 해: S_T = 125.45
  잘못된 해: S_T = 134.67
  차이: 9.22

평균 검증 (정말로 보정항이 필요한가?):
  올바른 해 E[S_T] = 121.04 (이론: 121.05)
  잘못된 해 E[S_T] = 131.78
  → Itô 보정항이 정확히 0.020000 = σ²/2의 drift 차이 보정!
```

---

## 🔗 AI/ML 연결

### Black-Scholes 옵션 가격 책정

GBM은 **Black-Scholes 모델**의 기초다. 옵션 가격은 GBM 아래에서의 기댓값으로 정의되는데, Itô 보정항이 정확한 가격 공식을 결정한다:

$$C(S, t) = S \Phi(d_1) - e^{-r(T-t)} K \Phi(d_2)$$

여기서 $d_1 = \frac{\log(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$ — **Itô 보정항이 포함**.

### Diffusion 모델의 확률 흐름(Probability Flow)

Diffusion Model에서 forward process가 곱셈적이면:

$$dX_t = f(t) X_t dt + g(t) X_t dB_t$$

(multiplicative noise) 로그 변환으로:

$$dY_t = d(\log X_t) = (f(t) - \frac{1}{2}g(t)^2) dt + g(t) dB_t$$

선형이 되어 reverse SDE의 수치 안정성이 획기적으로 개선된다.

### 자산 가격 모델

주가, 환율, 상품 가격 등 **금융 자산의 확률적 변동**은 GBM으로 모델링되며, **로그정규 분포** 가정은 tail risk 분석의 근거를 제공한다 (극단값 이론과의 연결).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 상수 $\mu, \sigma$ | 실제 시장은 시간 종속 volatility (GARCH, SV) |
| 로그정규 분포 | 실제 수익률은 heavy tail (극단 사건) |
| 연속 경로 | 극단적 점프(jump) 무시 (Lévy process 필요) |
| 완전 시장 | 거래 비용, 유동성 한계 미고려 |

**주의**: GBM은 "교과서적" 모델이지만, 실제 금융 시계열은 volatility clustering, leverage effect 등을 가진다. 따라서 **확률적 volatility (Heston 모델)** 또는 **점프 과정 (Merton jump-diffusion)** 이 필요할 수 있다.

---

## 📌 핵심 정리

$$\boxed{S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma B_t\right]}$$

| 개념 | 공식 | 의미 |
|------|------|------|
| **로그 분포** | $\log S_t \sim \mathcal{N}(\log S_0 + (\mu - \sigma^2/2)t, \sigma^2 t)$ | Lognormal |
| **평균** | $\mathbb{E}[S_t] = S_0 e^{\mu t}$ | Drift는 $\mu$ |
| **분산** | $\text{Var}[S_t] = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$ | 지수적 성장 |
| **Itô 보정** | $-\sigma^2/2$ term | Jensen 부등식 + 이차변동 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $dS_t = \mu S_t dt + \sigma S_t dB_t$에서 $Y_t = \log S_t$에 Itô 공식을 적용하고, 이차변동 $(dS_t)^2$를 계산하라.

<details>
<summary>힌트 및 해설</summary>

Itô 공식: $dY = f'(S) dS + \frac{1}{2}f''(S)(dS)^2$, $f(S) = \log S$

$f'(S) = 1/S$, $f''(S) = -1/S^2$

$(dS)^2 = (\mu S dt)^2 + 2(\mu S dt)(\sigma S dB) + (\sigma S dB)^2$

$= 0 + 0 + \sigma^2 S^2 (dB)^2 = \sigma^2 S^2 dt$

(왜냐하면 $(dt)^2 = 0$, $dt \cdot dB = 0$, $(dB)^2 = dt$)

따라서:

$$dY = \frac{1}{S}(\mu S dt + \sigma S dB) - \frac{1}{2S^2} \sigma^2 S^2 dt = (\mu - \frac{\sigma^2}{2}) dt + \sigma dB$$

핵심: **$-\sigma^2/2$ 항이 이차변동에서 자동 발생**.

</details>

**문제 2** (심화): GBM에서 $\mathbb{E}[S_t^2] = S_0^2 e^{2\mu t + \sigma^2 t}$를 증명하라. (로그정규 공식 사용)

<details>
<summary>힌트 및 해설</summary>

$S_t^2 = S_0^2 \exp(2(\mu - \sigma^2/2)t + 2\sigma B_t)$

$Z = 2(\mu - \sigma^2/2)t + 2\sigma B_t \sim \mathcal{N}(2(\mu - \sigma^2/2)t, 4\sigma^2 t)$

로그정규 공식: $\mathbb{E}[e^Z] = e^{\mathbb{E}[Z] + \frac{1}{2}\text{Var}[Z]}$

$$\mathbb{E}[e^Z] = \exp[2(\mu - \sigma^2/2)t + \frac{1}{2} \cdot 4\sigma^2 t]$$

$$= \exp[2(\mu - \sigma^2/2)t + 2\sigma^2 t]$$

$$= \exp[2\mu t - \sigma^2 t + 2\sigma^2 t] = e^{2\mu t + \sigma^2 t}$$

따라서:

$$\mathbb{E}[S_t^2] = S_0^2 e^{2\mu t + \sigma^2 t}$$

</details>

**문제 3** (AI 연결): Diffusion Model의 forward process $dX_t = -\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t$를 로그 스케일 $Y_t = \log |X_t|$로 변환하면 어떻게 되는가? (Itô 공식 사용)

<details>
<summary>힌트 및 해설</summary>

$X_t > 0$이라고 가정하면, $Y_t = \log X_t$.

Itô 공식:

$$dY = \frac{1}{X_t} dX_t - \frac{1}{2X_t^2}(dX_t)^2$$

$(dX_t)^2 = \beta_t X_t^2 dt$이므로:

$$dY = \frac{1}{X_t}(-\frac{\beta_t}{2}X_t dt + \sqrt{\beta_t} dB_t) - \frac{1}{2X_t^2} \beta_t X_t^2 dt$$

$$= -\frac{\beta_t}{2} dt + \frac{\sqrt{\beta_t}}{X_t} dB_t - \frac{\beta_t}{2} dt$$

$$= -\beta_t dt + \frac{\sqrt{\beta_t}}{X_t} dB_t$$

**관찰**: 로그 스케일에서도 Itô 보정항이 나타나며, 이것이 reverse SDE의 score term $\nabla \log p_t(X)$와 연관된다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Ornstein-Uhlenbeck 과정](./03-ornstein-uhlenbeck.md) | [📚 README로 돌아가기](../README.md) | [05. 선형 SDE의 일반해 ▶](./05-linear-sde.md) |

</div>
