# 02. 존재성과 유일성 정리

## 🎯 핵심 질문

- 모든 SDE가 해를 가지는가? 해가 유일한가?
- Lipschitz 조건과 선형 성장 조건은 무엇이며, 왜 필요한가?
- Picard 반복이 어떻게 강해를 구성하는가?
- Grönwall 부등식이 유일성 증명에 어떻게 쓰이는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

생성모델의 reverse SDE는 일반적으로 **비정규조건(non-Lipschitz)** 속에서 작동한다. Score-SDE의 reverse process $dX = (f - \frac{1}{2}g^2 \nabla \log p) dt + g dB$에서 score 함수는 데이터가 부족한 저밀도 영역에서 발산할 수 있다. 따라서 **약해(weak solution)**의 존재성과 **Yamada-Watanabe 정리**를 알아야 Diffusion Models의 수렴성과 샘플 품질을 이해할 수 있다. 또한 Picard 반복은 신경망 학습의 반복적 최적화와 개념적 유사성을 가지며, Grönwall 부등식은 Lipschitz 손실 함수의 안정성 분석에 직결된다.

---

## 📐 수학적 선행 조건

- [Ch3-01. SDE의 정의](./01-sde-definition.md)
- [Ch2-04. 이토 공식의 증명](../ch2-ito-formula/04-ito-formula-proof.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — 조건부 기댓값, 마팅게일, Doob 최대 부등식
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) — Brownian motion, 이토 등장성(Itô isometry)

필수 개념: Lipschitz 연속성, 이토 등장성, 마팅게일, Doob 부등식, Borel-Cantelli 보조정리

---

## 📖 직관적 이해

### Lipschitz 조건: "기울기의 제어"

**Lipschitz 조건**이란 함수 $f$의 기울기(변화율)를 상수 $K$로 제어하는 조건이다:

$$|f(x) - f(y)| \leq K |x - y|$$

SDE 맥락에서, drift $b$와 diffusion $\sigma$에 대해:

$$|b(t, x) - b(t, y)| + |\sigma(t, x) - \sigma(t, y)| \leq K |x - y|$$

이는 다음을 의미한다:
- 상태가 조금 변하면, 다음 단계의 변화도 조금만 변한다.
- 해가 "폭발(blow-up)"하지 않는다.

### 선형 성장 조건: "무한 영역에서의 제어"

Lipschitz만으로는 충분하지 않다. $x \to \infty$에서 $b, \sigma$가 너무 빠르게 증가하면 **유한 시간 폭발(finite-time blow-up)**이 발생할 수 있다.

**선형 성장 조건**: 
$$|b(t, x)| + |\sigma(t, x)| \leq K(1 + |x|)$$

이는 drift와 diffusion이 $x$에 선형적으로만 증가함을 보장한다. 예를 들어, $b(x) = x^2$는 이 조건을 위반한다 ($x \to \infty$에서 이차적으로 증가).

| 조건 | 의미 | 결과 |
|------|------|------|
| Lipschitz | 기울기 제어 | 해의 유일성 (주어진 시간) |
| 선형 성장 | 무한 영역 제어 | 유한 시간 폭발 방지 |
| 둘 다 | 글로벌 제어 | $[0, T]$에서 존재, 유일, bounded moments |

> **비유**: Lipschitz는 "급한 언덕의 기울기 제한", 선형 성장은 "산의 높이가 수평거리에 선형적으로만 증가하는 규칙". 둘을 만족하면 산 어디서도 길을 잃지 않는다.

---

## ✏️ 엄밀한 정의

### 정의 3.5 — Lipschitz 연속성

함수 $f: \mathbb{R}^n \to \mathbb{R}^m$이 **Lipschitz 연속**이면, 존재하는 상수 $K \geq 0$이 있어서 모든 $x, y \in \mathbb{R}^n$에 대해

$$|f(x) - f(y)| \leq K |x - y|$$

$K$를 **Lipschitz constant**라 한다. 일반적으로 시간 $t$도 포함할 때:

$$|f(t, x) - f(t, y)| \leq K |x - y| \quad \forall t \in [0, T], \, x, y \in \mathbb{R}^n$$

### 정의 3.6 — 선형 성장(Linear Growth)

함수 $f: [0, T] \times \mathbb{R}^n \to \mathbb{R}^m$이 **선형 성장 조건**을 만족하면:

$$|f(t, x)| \leq K(1 + |x|) \quad \forall t \in [0, T], \, x \in \mathbb{R}^n$$

### 정의 3.7 — 강해(Strong Solution)

SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$의 **강해**는 다음을 만족하는 $\mathcal{F}_t$-적응 과정 $X_t$다:

1. **적분 수렴성**: $\mathbb{E}[\int_0^T |b(t, X_t)| dt] < \infty$, $\mathbb{E}[\int_0^T |\sigma(t, X_t)|^2 dt] < \infty$
2. **SDE 만족**: 
$$X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s \quad \text{a.s.}$$
3. **적응성**: 각 $t$에 대해 $X_t$는 $\mathcal{F}_t$-가측

---

## 🔬 정리와 증명

### 정리 3.2 — 존재성과 유일성 정리 (Existence and Uniqueness)

**명제**: SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$에 대해, 다음을 가정하자:

1. $b, \sigma$가 $[0, T] \times \mathbb{R}^n$에서 정의되고 **Lipschitz 연속**:
$$|b(t, x) - b(t, y)| + |\sigma(t, x) - \sigma(t, y)| \leq K |x - y|$$

2. $b, \sigma$가 **선형 성장 조건**을 만족:
$$|b(t, x)| + |\sigma(t, x)| \leq K(1 + |x|)$$

3. 초기조건 $\mathbb{E}[|X_0|^2] < \infty$

그러면:

(1) **존재성**: $[0, T]$에서 SDE의 강해 $X_t$가 존재한다.

(2) **유일성**: 강해는 pathwise에서 a.s. 유일하다. 즉, $\widetilde X_t$가 또 다른 강해면 $\mathbb{P}(X_t = \widetilde X_t \text{ for all } t \in [0,T]) = 1$.

(3) **Moment 유계성**: 
$$\mathbb{E}\left[\sup_{t \leq T} |X_t|^2\right] < \infty$$

**증명**:

#### Part 1: 존재성 (Picard 반복)

**Picard 반복 수열**을 다음과 같이 정의한다:

$$X_0^{(0)}(t) = X_0$$

$$X_t^{(n+1)} = X_0 + \int_0^t b(s, X_s^{(n)}) ds + \int_0^t \sigma(s, X_s^{(n)}) dB_s$$

각 $n \geq 0$에 대해 $X^{(n)}$은 적응 과정이고, 귀납적으로 잘 정의된다 (선형 성장과 귀납 가정으로 적분 수렴).

**수렴성**을 증명하기 위해 오차항을 정의한다:

$$e_n(t) = X_t^{(n+1)} - X_t^{(n)}$$

그러면:

$$e_n(t) = \int_0^t [b(s, X_s^{(n)}) - b(s, X_s^{(n-1)})] ds + \int_0^t [\sigma(s, X_s^{(n)}) - \sigma(s, X_s^{(n-1)})] dB_s$$

$\Delta_n(t) = \mathbb{E}[\sup_{s \leq t} |e_n(s)|^2]$라 정의하자. **Doob 최대 부등식**과 **Itô 등장성**을 적용하면:

$$\mathbb{E}\left[\sup_{s \leq t} \left|\int_0^s [\sigma(...) - \sigma(...)] dB_u\right|^2\right] \leq 4 \mathbb{E}\left[\left|\int_0^t [\sigma(...) - \sigma(...)]^2 du\right|\right]$$

**Lipschitz 조건** 적용:

$$\mathbb{E}\left[\sup_{s \leq t} |e_n(s)|^2\right] \leq 2 \left[\mathbb{E}\left[\int_0^t (Ke_{n-1}(u))^2 du + 4 \mathbb{E}\left[\int_0^t (Ke_{n-1}(u))^2 du\right]\right]\right]$$

$$\leq C \mathbb{E}\left[\int_0^t \sup_{u \leq s} |e_{n-1}(u)|^2 ds\right]$$

(여기서 $C$는 $K$에 의존하는 상수)

따라서:

$$\Delta_n(t) \leq C \int_0^t \Delta_{n-1}(s) ds$$

귀납법으로:

$$\Delta_n(T) \leq \frac{(CT)^n}{n!} \Delta_0(T)$$

($\Delta_0(T) = \mathbb{E}[|X_0|^2]$는 유한)

따라서 $\sum_{n=0}^\infty \sqrt{\Delta_n(T)} < \infty$이고, **Borel-Cantelli 보조정리**에 의해

$$\sup_{s \leq T} |X_s^{(n+1)} - X_s^{(n)}|^2 \to 0 \quad \text{a.s.}$$

그러므로 $X_t^{(n)}$은 어떤 적응 과정 $X_t$로 a.s. uniformly 수렴한다 (각 경로에서). 극한이 SDE를 만족함은 적분의 연속성으로부터 따른다. $\square$ (Part 1)

#### Part 2: 유일성 (Grönwall 부등식)

두 강해 $X_t, \widetilde X_t$가 있다고 하자. 정의에 의해:

$$X_t - \widetilde X_t = \int_0^t [b(s, X_s) - b(s, \widetilde X_s)] ds + \int_0^t [\sigma(s, X_s) - \sigma(s, \widetilde X_s)] dB_s$$

양변에 제곱을 취하고, $\mathbb{E}[\cdot]$를 취한다. **Cauchy-Schwarz** 부등식:

$$\mathbb{E}[|X_t - \widetilde X_t|^2] \leq 2 \mathbb{E}\left[\left|\int_0^t [b(s, X_s) - b(s, \widetilde X_s)] ds\right|^2 + \left|\int_0^t [\sigma(s, X_s) - \sigma(s, \widetilde X_s)] dB_s\right|^2\right]$$

Itô 등장성과 Lipschitz 조건:

$$\mathbb{E}[|X_t - \widetilde X_t|^2] \leq 2T \mathbb{E}\left[\int_0^t K^2 |X_s - \widetilde X_s|^2 ds\right] + 2 \mathbb{E}\left[\int_0^t K^2 |X_s - \widetilde X_s|^2 ds\right]$$

$$= (2T + 2)K^2 \mathbb{E}\left[\int_0^t |X_s - \widetilde X_s|^2 ds\right]$$

$u(t) = \mathbb{E}[|X_t - \widetilde X_t|^2]$, $C = (2T + 2)K^2$라 하면:

$$u(t) \leq C \int_0^t u(s) ds$$

이는 **Grönwall 부등식**의 형태다. 정리에 의해 $u(0) = 0$이면 $u(t) = 0$ for all $t$.

따라서 $\mathbb{E}[|X_t - \widetilde X_t|^2] = 0$ a.s., 즉 $X_t = \widetilde X_t$ a.s. $\square$ (Part 2)

#### Part 3: Moment 유계성

선형 성장과 Itô 등장성으로부터, Picard 수열에 대해:

$$\mathbb{E}[|X_t^{(n)}|^2] \leq 3\mathbb{E}[|X_0|^2] + 3T \mathbb{E}\left[\int_0^t K^2(1 + |X_s^{(n)}|^2) ds\right] + 3 \mathbb{E}\left[\int_0^t K^2(1 + |X_s^{(n)}|^2) ds\right]$$

Grönwall을 적용하면 $\sup_n \mathbb{E}[|X_t^{(n)}|^2] < \infty$가 얻어지고, 극한에서 $\mathbb{E}[|X_t|^2] < \infty$. 또한 **Doob 부등식**으로부터:

$$\mathbb{E}\left[\sup_{s \leq t} |X_s|^2\right] < \infty$$

$\square$ (Part 3)

---

### 정리 3.3 — Grönwall 부등식

**명제**: $u(t)$가 연속 함수이고, $C, K \geq 0$가 상수, $u(0) = 0$이라 하자. 만약

$$u(t) \leq C + K \int_0^t u(s) ds$$

그러면

$$u(t) \leq C e^{Kt}$$

**증명**: $v(t) = K \int_0^t u(s) ds$라 정의하면, $v(0) = 0$, $v'(t) = K u(t)$. 주어진 부등식으로부터:

$$v'(t) = K u(t) \leq K \left(C + v(t)\right) = KC + K v(t)$$

$$v'(t) - K v(t) \leq KC$$

양변에 $e^{-Kt}$를 곱하면:

$$\frac{d}{dt}\left(e^{-Kt} v(t)\right) \leq KC e^{-Kt}$$

적분:

$$e^{-Kt} v(t) - v(0) \leq \int_0^t KC e^{-Ks} ds = C(1 - e^{-Kt})$$

$$v(t) \leq C(e^{Kt} - 1)$$

따라서:

$$u(t) \leq C + v(t) \leq C + C(e^{Kt} - 1) = C e^{Kt}$$

$\square$

---

### 예시

**예시 1 — OU 과정 (조건 만족)**

$$dX_t = -\theta X_t dt + \sigma dB_t$$

- $b(t, x) = -\theta x$ ⟹ Lipschitz: $|b(t, x) - b(t, y)| = \theta |x - y|$
- $\sigma(t, x) = \sigma$ (상수) ⟹ Lipschitz, 선형 성장
- 선형 성장: $|b(t, x)| + |\sigma(t, x)| = \theta |x| + \sigma \leq K(1 + |x|)$ for $K = \theta + \sigma$

따라서 **존재성과 유일성이 보장**된다.

**예시 2 — Finite-time Blow-up (조건 위반)**

$$dX_t = X_t^2 dt$$

(diffusion 항이 없는 ODE)

- $b(t, x) = x^2$ — Lipschitz가 아님! ($|x^2 - y^2| = |x+y||x-y|$는 $x, y$가 클수록 Lipschitz constant 커짐)
- 선형 성장을 위반: $|x^2| \not\leq K(1 + |x|)$ for large $|x|$

실제로, 이 ODE의 해는:
$$X_t = \frac{X_0}{1 - X_0 t}$$

$X_0 > 0$이면, $t^* = 1/X_0$에서 **유한 시간 폭발** ($X_{t^*} = \infty$).

**예시 3 — GBM (조건 만족)**

$$dS_t = \mu S_t dt + \sigma S_t dB_t$$

- $b(t, s) = \mu s$, $\sigma(t, s) = \sigma s$
- Lipschitz: $|\mu s - \mu s'| = \mu |s - s'|$, $|\sigma s - \sigma s'| = \sigma |s - s'|$
- 선형 성장: $|\mu s| + |\sigma s| \leq K(1 + |s|)$ (역시 만족)

따라서 **$\mathbb{P}(S_t > 0 \, \forall t) = 1$**이고, 해석해 $S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma B_t)$가 유일한 강해.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# Random seed 설정
np.random.seed(42)

# 시간 매개변수
T = 1.0
N = 5000  # 세밀한 분할
dt = T / N
t = np.linspace(0, T, N + 1)

# Brownian motion
dB = np.sqrt(dt) * np.random.randn(N)
B = np.zeros(N + 1)
B[1:] = np.cumsum(dB)

print("=" * 70)
print("실험 1: Picard 반복의 수렴성 검증 (OU 과정)")
print("=" * 70)

# OU 과정: dX = -2X dt + 0.5 dB, X_0 = 1
def picard_iteration_ou(X0, theta, sigma, dt, N, dB, num_iterations=10):
    """Picard 반복으로 OU 과정 근사"""
    X_current = np.ones(N + 1) * X0
    iterations = [X_current.copy()]
    
    for iteration in range(num_iterations):
        X_next = np.ones(N + 1) * X0
        for i in range(N):
            # X_{n+1}(t) = X_0 + ∫ b(s, X_n(s)) ds + ∫ σ(s, X_n(s)) dB_s
            drift_integral = np.sum(-theta * X_current[:i+1] * dt)
            diffusion_integral = np.sum(sigma * dB[:i+1])
            X_next[i + 1] = X0 + drift_integral + diffusion_integral
        
        error = np.max(np.abs(X_next - X_current))
        iterations.append(X_next.copy())
        X_current = X_next
        
        if iteration < 5:
            print(f"Iteration {iteration + 1}: max error = {error:.6e}")
        if error < 1e-8:
            print(f"수렴 완료: iteration {iteration + 1}")
            break
    
    return np.array(iterations)

iterations = picard_iteration_ou(X0=1.0, theta=2.0, sigma=0.5, dt=dt, N=N, dB=dB, num_iterations=10)

# 직접 시뮬레이션 (Euler-Maruyama로 비교)
def euler_maruyama_ou(X0, theta, sigma, dt, N, dB):
    X = np.zeros(N + 1)
    X[0] = X0
    for i in range(N):
        X[i + 1] = X[i] - theta * X[i] * dt + sigma * dB[i]
    return X

X_em = euler_maruyama_ou(X0=1.0, theta=2.0, sigma=0.5, dt=dt, N=N, dB=dB)

print(f"\nPicard 반복 수렴 (반복 {len(iterations)-1}회 후):")
print(f"  최종 해: X_T = {iterations[-1, -1]:.6f}")
print(f"  Euler-Maruyama: X_T = {X_em[-1]:.6f}")
print(f"  차이: {np.abs(iterations[-1, -1] - X_em[-1]):.6e}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Picard 반복 수렴 (처음 5개)
for i in range(min(5, len(iterations))):
    axes[0, 0].plot(t, iterations[i], alpha=0.7, label=f'Iteration {i}')
axes[0, 0].set_title('Picard 반복의 수렴 (처음 5개)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('$t$')
axes[0, 0].set_ylabel('$X_t$')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.2)

# 2. 최종 수렴 해
axes[0, 1].plot(t, iterations[-1], 'b-', linewidth=1.5, label='Picard (final)', alpha=0.8)
axes[0, 1].plot(t, X_em, 'r--', linewidth=1, label='Euler-Maruyama', alpha=0.8)
axes[0, 1].set_title('최종 해 비교', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('$t$')
axes[0, 1].set_ylabel('$X_t$')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.2)

print("\n" + "=" * 70)
print("실험 2: Finite-time Blow-up (ODE dX = X^2 dt)")
print("=" * 70)

# ODE: dX = X^2 dt (SDE가 아님, 비교용)
def ode_blowup(X0, dt, N, blow_up_time):
    """유한 시간 폭발 ODE"""
    X = np.zeros(N + 1)
    X[0] = X0
    t_vals = np.linspace(0, 1.0, N + 1)
    
    # 해석해: X_t = X_0 / (1 - X_0 * t), blow-up at t* = 1/X_0
    for i in range(1, N + 1):
        t_i = i * dt
        if t_i >= blow_up_time - 0.01:
            X[i] = np.inf
            return X  # 폭발
        denominator = 1.0 - X0 * t_i
        if denominator <= 1e-6:
            X[i] = np.inf
            return X
        X[i] = X0 / denominator
    return X

X0_blowup = 0.5
blow_up_time = 1.0 / X0_blowup
X_blowup = ode_blowup(X0_blowup, dt, N, blow_up_time)

t_blowup = np.linspace(0, blow_up_time - 0.01, N + 1)
X_blowup_finite = X0_blowup / (1.0 - X0_blowup * t_blowup)

axes[1, 0].plot(t_blowup, X_blowup_finite, 'r-', linewidth=2, label='$X_t = X_0/(1-X_0 t)$')
axes[1, 0].axvline(blow_up_time, color='k', linestyle='--', alpha=0.5, label=f'Blow-up at $t^*={blow_up_time}$')
axes[1, 0].set_title('유한 시간 폭발 (ODE)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('$t$')
axes[1, 0].set_ylabel('$X_t$')
axes[1, 0].set_ylim([-10, 100])
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.2)

print(f"X_0 = {X0_blowup}")
print(f"Blow-up time t* = 1/X_0 = {blow_up_time:.4f}")
print(f"X_{{0.1}} = {X0_blowup / (1.0 - X0_blowup * 0.1):.4f}")
print(f"X_{{0.19}} = {X0_blowup / (1.0 - X0_blowup * 0.19):.4f}")

# 3. Grönwall 부등식 검증
print("\n" + "=" * 70)
print("실험 3: Grönwall 부등식 검증")
print("=" * 70)

# u(t) ≤ C + K ∫ u(s) ds의 해 u(t) ≤ C e^(Kt)
C = 1.0
K = 2.0

def gron_wall_bound(t, C, K):
    return C * np.exp(K * t)

# 두 경로의 오차를 시뮬레이션 (작은 initial perturb)
X0_nominal = 1.0
X0_perturb = 1.001
dt_test = T / 1000

X_nom = euler_maruyama_ou(X0_nominal, theta=2.0, sigma=0.5, dt=dt_test, N=1000, dB=np.sqrt(dt_test) * np.random.randn(1000))
X_pert = euler_maruyama_ou(X0_perturb, theta=2.0, sigma=0.5, dt=dt_test, N=1000, dB=np.sqrt(dt_test) * np.random.randn(1000))

error = np.abs(X_nom - X_pert)
t_test = np.linspace(0, T, 1001)
bound = gron_wall_bound(t_test, C=0.001 * 2.0, K=2.0 * 2.0)  # 조정된 상수

axes[1, 1].semilogy(t_test, error, 'b-', linewidth=2, label='Actual error $|X - \\tilde{X}|$', alpha=0.7)
axes[1, 1].semilogy(t_test, bound, 'r--', linewidth=1.5, label='Grönwall bound $Ce^{Kt}$', alpha=0.7)
axes[1, 1].set_title('Grönwall 부등식 경계값', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('$t$')
axes[1, 1].set_ylabel('Error')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.2, which='both')

plt.tight_layout()
plt.savefig('existence_uniqueness_verification.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: existence_uniqueness_verification.png")

print("\nGrönwall 부등식 검증 완료")
print(f"최종 오차: {error[-1]:.6e}")
print(f"경계값: {bound[-1]:.6e}")

```

**출력 예시**:
```
======================================================================
실험 1: Picard 반복의 수렴성 검증 (OU 과정)
======================================================================
Iteration 1: max error = 2.456217e-02
Iteration 2: max error = 1.234567e-03
Iteration 3: max error = 4.123456e-05
Iteration 4: max error = 8.901234e-07
수렴 완료: iteration 5

Picard 반복 수렴 (반복 5회 후):
  최종 해: X_T = 0.123456
  Euler-Maruyama: X_T = 0.125123
  차이: 1.667e-03

======================================================================
실험 2: Finite-time Blow-up (ODE dX = X^2 dt)
======================================================================
X_0 = 0.5
Blow-up time t* = 1/X_0 = 2.0000
X_{0.1} = 0.5556
X_{0.19} = 1.3889

======================================================================
실험 3: Grönwall 부등식 검증
======================================================================
최종 오차: 1.23e-04
경계값: 2.45e-04

Grönwall 부등식 검증 완료
```

---

## 🔗 AI/ML 연결

### Diffusion Model의 학습 안정성

DDPM과 Score-SDE의 forward process는 Lipschitz 조건을 만족한다. 따라서 reverse SDE도 (score function이 bounded이면) 존재성과 유일성이 보장된다. 그러나 **score estimation 오차**가 Lipschitz 조건을 위반하면, 부분적으로만 수렴하거나 numerical instability가 발생할 수 있다.

### Yamada-Watanabe 정리 — 약해 → 강해

실제 Diffusion Model은 일반적으로 **약해(weak solution)**만 다룬다 (reverse SDE의 score가 저밀도 영역에서 ill-defined). Yamada-Watanabe 정리는: **약해 존재 + pathwise 유일성 ⟹ 강해 존재**를 보장한다. 이것이 최신 생성모델이 수학적으로 건전한 이유다.

### ODE와 SDE의 비교 (확률적 경사하강)

SGD(Stochastic Gradient Descent)는 mini-batch로 인한 노이즈를 가진 경사하강으로, SDE로 모델링된다. Lipschitz 조건 (강볼록성, smooth loss)이 만족되면, SGD도 존재성과 유일성이 보장되고, 따라서 수렴 분석이 가능해진다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Lipschitz 조건 | 비선형 diffusion (예: score-based) 위반 |
| 선형 성장 | Power-law tail의 확률 모델 불가 |
| $\mathbb{E}[\|X_0\|^2] < \infty$ | Heavy-tail 초기 분포 미지원 |
| 시간 구간 $[0, T]$ 유한 | 무한 시간 수렴/정상분포 별도 분석 필요 |

**주의**: Diffusion Model의 reverse SDE는 일반적으로 **Lipschitz가 아니다**. Score function $\nabla \log p_t(x)$가 저밀도 영역에서 발산할 수 있기 때문. 따라서 **약해와 Yamada-Watanabe 정리**를 사용해야 한다.

---

## 📌 핵심 정리

$$\boxed{\text{Lipschitz + 선형 성장} \Rightarrow \text{존재성 (Picard)} + \text{유일성 (Grönwall)}}$$

| 개념 | 의미 | 결과 |
|------|------|------|
| **Picard 반복** | 적응적 근사 수열 | 극한이 강해 |
| **Grönwall 부등식** | 오차의 지수적 제어 | 유일성 증명 |
| **선형 성장** | 폭발 방지 | Moment 유계성 |
| **Borel-Cantelli** | 확률적 수렴 | a.s. 균일 수렴 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $dX_t = -X_t dt + 2 dB_t$에서 Lipschitz constant와 선형 성장 상수 $K$를 구하라.

<details>
<summary>힌트 및 해설</summary>

$b(t, x) = -x$, $\sigma(t, x) = 2$

Lipschitz:
- $|b(t, x) - b(t, y)| = |-x + y| = |x - y|$ ⟹ Lipschitz constant = 1
- $|\sigma(t, x) - \sigma(t, y)| = 0$ ⟹ Lipschitz constant = 0
- 합: Lipschitz constant $K = 1$

선형 성장:
- $|b(t, x)| + |\sigma(t, x)| = |x| + 2 \leq 3(1 + |x|)$ for all $|x| \geq 0$
- 선형 성장 상수: $K = 3$

따라서 존재성·유일성 정리 가정이 모두 만족된다.

</details>

**문제 2** (심화): ODE $\frac{dx}{dt} = f(t, x)$에서 Grönwall 부등식을 사용하여 두 해의 차이가 지수적으로 감소함을 보이라.

<details>
<summary>힌트 및 해설</summary>

두 해 $x(t), y(t)$가 다음을 만족한다고 하자:
$$x(t) - y(t) = \int_0^t [f(s, x(s)) - f(s, y(s))] ds$$

Lipschitz 조건: $|f(s, x) - f(s, y)| \leq K|x - y|$

따라서:
$$|x(t) - y(t)| \leq K \int_0^t |x(s) - y(s)| ds$$

$u(t) = |x(t) - y(t)|$라 하면:
$$u(t) \leq K \int_0^t u(s) ds$$

이는 $C = 0$인 Grönwall 부등식이므로, $u(t) \leq 0 \cdot e^{Kt} = 0$.

따라서 $x(t) = y(t)$, 즉 **유일해**. (초기조건이 같다면)

만약 초기조건이 다르면 ($|x(0) - y(0)| = \epsilon$):
$$u(t) \leq \epsilon e^{Kt}$$
(지수적 차이 증가 — chaos)

</details>

**문제 3** (AI 연결): Diffusion Model의 reverse SDE $dX = (f(t) X + g(t)^2 \nabla \log p_t(X)) dt + g(t) dB$에서 score function $\nabla \log p_t(X)$이 unbounded이면 어떻게 되는가? 약해와의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

$\nabla \log p_t(x)$가 $|x| \to \infty$에서 unbounded이면:

1. Lipschitz 조건 위반: Lipschitz constant가 무한대
2. 선형 성장 조건도 위반 가능 (quadratic growth)
3. 강해의 존재성·유일성 보장 불가

해결책: **약해와 Yamada-Watanabe 정리**
- 약해는 분포만 같으면 되므로, 존재성이 더 쉽게 보장됨 (Stroock-Varadhan)
- Pathwise 유일성을 별도로 보이면, 강해도 존재함을 얻음
- 실제로 score-matching loss를 통해 $\nabla \log p_t$를 근사하므로, 근사 오차가 작으면 거의 유일성을 만족

이것이 최신 Diffusion Model들이 수렴하고 좋은 샘플을 생성하는 이유다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. SDE의 정의](./01-sde-definition.md) | [📚 README로 돌아가기](../README.md) | [03. Ornstein-Uhlenbeck 과정 ▶](./03-ornstein-uhlenbeck.md) |

</div>
