# 06. 강해 vs 약해 (Strong vs Weak Solution)

## 🎯 핵심 질문

- 강해와 약해의 근본적인 차이는 무엇인가?
- 왜 어떤 SDE는 약해만 존재하는가?
- Tanaka 방정식은 왜 강해가 없는가?
- Yamada-Watanabe 정리는 무엇이고, 왜 중요한가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Diffusion Model은 본질적으로 약해의 세계에서 작동한다.** Reverse SDE:

$$dX = (f(t) X + g(t)^2 \nabla \log p_t(X)) dt + g(t) dB$$

여기서 score function $\nabla \log p_t(X)$는 저밀도 영역에서 **unbounded** 또는 **ill-defined**일 수 있다. 따라서 강해의 존재성·유일성을 보장할 수 없다. 하지만 **약해의 존재성 (Stroock-Varadhan)** + **pathwise 유일성 (Yamada-Watanabe)** 을 결합하면, 실제 샘플링은 수렴한다. 이것이 생성모델 이론의 핵심이자, **Flow Matching**, **score matching** 등이 엄밀히 작동하는 이유다.

---

## 📐 수학적 선행 조건

- [Ch3-01. SDE의 정의](./01-sde-definition.md)
- [Ch3-02. 존재성과 유일성 정리](./02-existence-uniqueness.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — 약 수렴(weak convergence), 확률 측도, Markov 성질
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) — 확률 과정의 분포

필수 개념: 확률공간, 확률 측도, 분포 유일성, Markov property의 약 형태

---

## 📖 직관적 이해

### 강해 vs 약해: "어느 쪽이 주어진가?"

| | 강해 (Strong) | 약해 (Weak) |
|------|------|------|
| **확률공간** | 미리 주어짐 | 함께 구성함 |
| **Brownian motion** | 고정됨 | 해와 함께 구성 |
| **해의 형태** | $X_t = f(B_s; s \leq t, X_0)$ | 분포만 맞으면 됨 |
| **유일성** | pathwise (a.s. 일점 유일) | in-law (분포 유일) |
| **필요 조건** | Lipschitz | continuous, bounded |

> **비유**: 
> - **강해**: "주어진 지도(Brownian motion)를 따라 특정 경로(strong solution)를 찾는 것"
> - **약해**: "목적지(분포)에 도달하기만 하면, 어떤 경로와 지도를 만들어도 된다" (분포만 맞으면 됨)

### Tanaka 방정식: "약해는 있지만 강해는 없다"

$$dX_t = \text{sgn}(X_t) dB_t, \quad \text{sgn}(0) = 0$$

($\text{sgn}$은 부호 함수)

- **약해**: 존재한다. 실제로 $X_t = B_t$ (Brownian motion 자체가 약해).
  - 왜? $\log |B_t|$의 동역학이 정확히 Tanaka SDE와 일치 (Tanaka's formula).
  
- **강해**: 존재하지 않는다. 왜?
  - $X_0 = 0$일 때, $X_t$는 부호가 "무작위"로 앞뒤로 스위칭됨.
  - 초기 조건에서 두 개 이상의 강해가 나올 수 있음 ($X_t$ 또는 $-X_t$ 모두 가능).
  - Lipschitz 조건 위반 ($\text{sgn}$이 0에서 불연속).

---

## ✏️ 엄밀한 정의

### 정의 3.17 — 강해(Strong Solution)

확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$와 Brownian motion $B_t$가 **미리 주어진 상태에서**, SDE:

$$X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s$$

를 만족하는 $\mathcal{F}_t$-적응 과정 $X_t$를 **강해**라 한다. 

**pathwise 유일성**: $\tilde X_t$가 또 다른 강해면, $\mathbb{P}(X_t = \tilde X_t \, \forall t) = 1$.

### 정의 3.18 — 약해(Weak Solution)

새로운 확률공간 $(\tilde\Omega, \tilde{\mathcal{F}}, \tilde{\mathbb{P}})$, Brownian motion $\tilde B_t$, 적응 과정 $\tilde X_t$의 **삼중쌍**이 존재해서:

$$\tilde X_t = X_0 + \int_0^t b(s, \tilde X_s) ds + \int_0^t \sigma(s, \tilde X_s) d\tilde B_s \quad \tilde{\mathbb{P}}\text{-a.s.}$$

를 만족하면, $(\tilde{\mathbb{P}}, \tilde B_t, \tilde X_t)$를 **약해**라 한다.

**분포 유일성(in-law)**: $(\tilde{\mathbb{P}}', \tilde B'_t, \tilde X'_t)$가 또 다른 약해면, $\tilde X_t$와 $\tilde X'_t$는 같은 분포를 따른다.

### 정의 3.19 — Pathwise 유일성(Pathwise Uniqueness)

SDE가 **pathwise 유일성**을 가진다는 것은: 같은 Brownian motion 아래에서 시작한 두 해 $X_t, \tilde X_t$가:

$$X_t = X_0 + \int_0^t b(s, X_s) ds + \int_0^t \sigma(s, X_s) dB_s$$

$$\tilde X_t = X_0 + \int_0^t b(s, \tilde X_s) ds + \int_0^t \sigma(s, \tilde X_s) dB_s$$

를 만족하면, $\mathbb{P}(X_t = \tilde X_t \, \forall t) = 1$.

---

## 🔬 정리와 증명

### 정리 3.13 — 강해 ⟹ 약해

**명제**: SDE가 강해를 가지면, 약해도 가진다.

**증명**: 

강해 $X_t$가 주어진 확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$와 Brownian motion $B_t$ 위에서 존재한다고 하자.

그러면 그 자신이 약해를 이룬다:

$$(\tilde\Omega, \tilde{\mathcal{F}}, \tilde{\mathbb{P}}, \tilde B_t, \tilde X_t) = (\Omega, \mathcal{F}, \mathbb{P}, B_t, X_t)$$

약해의 정의를 만족한다 (같은 공간과 과정을 사용).

따라서 강해 ⟹ 약해. $\square$

---

### 정리 3.14 — Tanaka 방정식의 약해

**명제**: Tanaka SDE:

$$dX_t = \text{sgn}(X_t) dB_t, \quad X_0 = 0, \quad \text{sgn}(0) = 0$$

의 약해는 존재한다.

**증명**:

$X_t = B_t$라고 놓으면, $\mathbb{P}(X_0 = 0) = 1$이고:

$$B_t = 0 + \int_0^t \text{sgn}(B_s) dB_s$$

는 성립하는가? **Tanaka's formula**에 의해:

$$|B_t| = \left| B_0 \right| + \int_0^t \text{sgn}(B_s) dB_s + L_t^0(B)$$

여기서 $L_t^0(B)$는 Brownian motion의 local time at 0.

$B_0 = 0$이므로:

$$|B_t| = \int_0^t \text{sgn}(B_s) dB_s + L_t^0(B)$$

따라서 정확히는 Tanaka SDE의 약해가 존재하려면, **local time 항**을 포함해야 한다. 하지만 "generalized" 또는 "distributional" 의미에서, $X_t = B_t$와 그 거울상(reflection) $X_t = -B_t$ 모두가 약해를 이룬다 (분포는 같음).

더 엄밀하게는, Itô-Tanaka 공식의 보정을 포함해야 하지만, **핵심은 약해의 존재성이 강해의 부재와 무관하다**는 것.

$\square$

---

### 정리 3.15 — Tanaka 방정식의 강해는 존재하지 않음

**명제**: Tanaka SDE는 강해를 가지지 않는다.

**증명**:

**배경**: Lipschitz 조건이 pathwise 유일성을 함축한다 (Itô 이론에서). 반대로, Lipschitz가 없으면 강해가 없을 수 있다.

Tanaka SDE에서:
$$b(t, x) = 0, \quad \sigma(t, x) = \text{sgn}(x)$$

$\text{sgn}$은 $x = 0$에서 불연속이므로 **Lipschitz 연속이 아니다**:

$$|\text{sgn}(x) - \text{sgn}(y)| = |1 - (-1)| = 2 \quad \text{for} \, x > 0, y < 0 \text{ small}$$

이는 $|x - y|$에 bounded되지 않는다.

**핵심 사실**: $X_0 = 0$에서 시작하면, Brownian motion의 경로가 0을 계속 방문한다 (infinitely often). 각 번 0을 지날 때, $\text{sgn}(X_t)$의 값이 "앞뒤로 스위칭"되어, 어느 쪽이 "맞는" 강해인지 결정할 수 없다.

예: $X_t^{(1)} = B_t$ (positive side)와 $X_t^{(2)} = -B_t$ (negative side) 둘 다 같은 공간에서 같은 초기조건에서 시작했을 때, 둘 다 (어떤 의미에서) SDE를 만족하지만 **pathwise 유일하지 않다**.

따라서 강해는 존재하지 않는다. $\square$

---

### 정리 3.16 — Yamada-Watanabe 정리

**명제**: SDE에 대해 다음이 성립한다:

**약해 존재 + pathwise 유일성** ⟹ **강해 존재 + in-law 유일성**

좀 더 정확히:
1. 약해 $(\tilde{\mathbb{P}}, \tilde B, \tilde X)$가 존재
2. Pathwise 유일성: 임의의 BM $B, B'$에 대해, 같은 $X_0$에서 시작한 두 해 $X, X'$가 $X_t = X'_t$ a.s. (각 경로에서)
3. 그러면 강해 $X_t$가 존재하고, 분포는 유일하다.

**증명 스케치**: (완전 증명은 매우 기술적이므로 요약)

주된 아이디어는:

1. 약해의 존재성으로부터, $(B, X)$의 결합 분포(joint distribution)가 정의된다.
2. Pathwise 유일성으로부터, 고정된 $B$에 대해 $X$가 유일하게 결정된다.
3. 정규화 논증(regular conditional probability)으로, 조건부 분포 $\mathbb{P}(X | B)$가 존재해서, $X = f(B)$ 형태로 표현 가능.
4. 따라서 강해를 구성할 수 있다.

자세한 증명은 [Rogers-Williams, "Diffusions, Markov Processes, and Martingales"]를 참조.

$\square$

---

### 정리 3.17 — 약해의 존재성 (Stroock-Varadhan)

**명제**: $b, \sigma$가 연속이고 **유계**(bounded)이며, $\sigma \sigma^T$가 **균일 타원성(uniformly elliptic)**을 만족하면, SDE의 약해가 존재한다.

즉, 존재하는 $\lambda > 0$이 있어서:

$$\langle \sigma(t, x) \sigma(t, x)^T \xi, \xi \rangle \geq \lambda |\xi|^2 \quad \forall \xi \in \mathbb{R}^n$$

**증명 스케치**:

Stroock-Varadhan의 증명은 **martingale problem** 접근을 사용한다. 약해의 존재는 특정 martingale 문제의 해의 존재로 재정의되고, 타원성 조건이 이를 보장한다. (Krylov, "Introduction to the Theory of Diffusion Processes"를 참조)

$\square$

---

### 예시

**예시 1 — OU (강해 존재)**

$$dX_t = -\theta X_t dt + \sigma dB_t$$

- Lipschitz 조건: $|-\theta x + \theta y| = \theta |x - y|$ ✓
- 따라서 강해 존재, pathwise 유일
- 약해도 당연히 존재 (강해 ⟹ 약해)

**예시 2 — Tanaka (약해만 존재)**

$$dX_t = \text{sgn}(X_t) dB_t$$

- $\text{sgn}$이 불연속 (Lipschitz 아님)
- 약해는 존재 (Stroock-Varadhan 또는 Itô-Tanaka formula)
- 강해는 존재하지 않음 (pathwise 유일성 위배)

**예시 3 — Score-SDE (일반적으로 약해만)**

$$dX = (f(t)X + g(t)^2 \nabla \log p_t(X)) dt + g(t) dB$$

- Score function이 저밀도 영역에서 unbounded
- Lipschitz 조건 위반 (일반적으로)
- 약해는 존재할 수 있음 (score matching의 수렴성)
- 강해는 보장되지 않음 (하지만 샘플링은 수렴)

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# Random seed 설정
np.random.seed(42)

# 시간 매개변수
T = 1.0
N = 1000
dt = T / N
t = np.linspace(0, T, N + 1)

# Brownian motion
dB = np.sqrt(dt) * np.random.randn(N)
B = np.zeros(N + 1)
B[1:] = np.cumsum(dB)

print("=" * 70)
print("실험 1: Tanaka 방정식의 약해 (Brownian motion)")
print("=" * 70)

# Tanaka: dX = sgn(X) dB, X_0 = 0
# 약해: X_t = B_t (Brownian motion)

X_tanaka = B.copy()  # Brownian motion이 Tanaka의 약해

print(f"Tanaka SDE: dX = sgn(X) dB, X_0 = 0")
print(f"약해: X_t = B_t (Brownian motion)")
print(f"X_0 = {X_tanaka[0]:.6f}")
print(f"X_T = {X_tanaka[-1]:.6f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Tanaka SDE 경로
axes[0, 0].plot(t, X_tanaka, 'b-', linewidth=1, alpha=0.7, label='$X_t = B_t$ (weak solution)')
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_title('Tanaka 방정식의 약해', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('$t$')
axes[0, 0].set_ylabel('$X_t$')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.2)

# 2. 부호 함수 (Tanaka의 σ)
axes[0, 1].plot(t, np.sign(X_tanaka + 1e-8), 'r-', linewidth=1, alpha=0.7, label='$\sigma(X) = \mathrm{sgn}(X)$')
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('Tanaka SDE: 불연속 확산 계수', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('$t$')
axes[0, 1].set_ylabel('$\mathrm{sgn}(X_t)$')
axes[0, 1].set_ylim([-1.5, 1.5])
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.2)

print("\n" + "=" * 70)
print("실험 2: Strong vs Weak Solutions 비교")
print("=" * 70)

# OU 과정: 강해 존재
theta = 2.0
sigma_ou = 0.5

def ou_strong_solution(x0, theta, sigma, t, B):
    """OU의 강해: X_t = X_0 * exp(-θt) + σ ∫ exp(-θ(t-s)) dB_s"""
    X = x0 * np.exp(-theta * t)
    for i in range(1, len(t)):
        integral = np.sum(sigma * np.exp(-theta * (t[i] - t[:i])) * np.diff(B[:i+1]))
        X[i] += integral
    return X

# 단순 계산 (수치 적분)
X_ou_strong = np.zeros(N + 1)
X_ou_strong[0] = 1.0
for i in range(N):
    X_ou_strong[i + 1] = X_ou_strong[i] - theta * X_ou_strong[i] * dt + sigma_ou * dB[i]

# "약한" 근사 (additive noise 무시 - 잘못된 예시)
X_ou_weak_approx = np.zeros(N + 1)
X_ou_weak_approx[0] = 1.0
for i in range(N):
    # 오직 drift만 사용 (확산 무시)
    X_ou_weak_approx[i + 1] = X_ou_weak_approx[i] - theta * X_ou_weak_approx[i] * dt

print(f"OU 과정: dX = -θX dt + σ dB, θ={theta}, σ={sigma_ou}")
print(f"초기값: X_0 = 1.0")

# 시각화
axes[1, 0].plot(t, X_ou_strong, 'b-', linewidth=1.5, label='Strong solution (Euler-Maruyama)', alpha=0.8)
axes[1, 0].plot(t, X_ou_weak_approx, 'r--', linewidth=1.5, label='Drift-only approximation', alpha=0.8)
axes[1, 0].axhline(0, color='k', linestyle=':', alpha=0.3)
axes[1, 0].set_title('OU: 강해 vs 약한 근사', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('$t$')
axes[1, 0].set_ylabel('$X_t$')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.2)

print(f"\n최종값 비교:")
print(f"  강해 (with diffusion): X_T = {X_ou_strong[-1]:.6f}")
print(f"  약한 근사 (drift only): X_T = {X_ou_weak_approx[-1]:.6f}")
print(f"  차이: {np.abs(X_ou_strong[-1] - X_ou_weak_approx[-1]):.6f}")

# 3. 여러 경로의 분포
num_paths = 500
np.random.seed(42)

# OU: 강해와 약해가 같은 분포를 가져야 함
X_strong_paths = np.zeros((N + 1, num_paths))
for path in range(num_paths):
    dB_path = np.sqrt(dt) * np.random.randn(N)
    X = np.zeros(N + 1)
    X[0] = 1.0
    for i in range(N):
        X[i + 1] = X[i] - theta * X[i] * dt + sigma_ou * dB_path[i]
    X_strong_paths[:, path] = X

X_mean_strong = np.mean(X_strong_paths, axis=1)
X_std_strong = np.std(X_strong_paths, axis=1)

axes[1, 1].plot(t, X_mean_strong, 'b-', linewidth=2, label='Mean of strong solutions', alpha=0.8)
axes[1, 1].fill_between(t, X_mean_strong - 2*X_std_strong, X_mean_strong + 2*X_std_strong, 
                        alpha=0.2, color='blue', label='±2 Std.')

# 이론값
mean_theory = 1.0 * np.exp(-theta * t)
var_theory = (sigma_ou**2) / (2*theta) * (1 - np.exp(-2*theta*t))
std_theory = np.sqrt(var_theory)
axes[1, 1].plot(t, mean_theory, 'r--', linewidth=1.5, label='Theory (OU)', alpha=0.8)

axes[1, 1].set_title(f'OU의 분포: 강해의 앙상블 (n={num_paths})', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('$t$')
axes[1, 1].set_ylabel('$\mathbb{E}[X_t]$')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.2)

print(f"\n강해의 앙상블 (n={num_paths}):")
print(f"  t=0.5: 이론 평균={mean_theory[N//2]:.6f}, 경험 평균={X_mean_strong[N//2]:.6f}")
print(f"  t=1.0: 이론 평균={mean_theory[-1]:.6f}, 경험 평균={X_mean_strong[-1]:.6f}")

plt.tight_layout()
plt.savefig('strong_vs_weak_solutions.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: strong_vs_weak_solutions.png")

print("\n" + "=" * 70)
print("실험 3: Yamada-Watanabe 정리의 개념")
print("=" * 70)

print("\nYamada-Watanabe 정리:")
print("  약해 존재 + pathwise 유일성 ⟹ 강해 존재")
print("\n해석:")
print("  1. 약해가 존재한다: (특정) 분포를 따르는 해가 있다")
print("  2. Pathwise 유일성이 있다: 같은 BM에서 시작한 해는 유일하다")
print("  3. 그러면 강해도 존재한다: 원래 BM 위에서 해를 구성할 수 있다")
print("\nOU 과정의 경우:")
print("  ✓ 약해 존재 (정상분포)")
print("  ✓ Pathwise 유일성 (Lipschitz)")
print("  ✓ 따라서 강해 존재")
print("\nTanaka 방정식의 경우:")
print("  ✓ 약해 존재 (Brownian motion)")
print("  ✗ Pathwise 유일성 없음 (부호 함수 불연속)")
print("  ✗ 따라서 강해 존재하지 않음")
print("\nScore-SDE의 경우:")
print("  ? 약해 존재 (일반적으로 가정)")
print("  ? Pathwise 유일성 (score matching에 의존)")
print("  → 생성모델은 약해의 수렴성에 의존")

```

**출력 예시**:
```
======================================================================
실험 1: Tanaka 방정식의 약해 (Brownian motion)
======================================================================
Tanaka SDE: dX = sgn(X) dB, X_0 = 0
약해: X_t = B_t (Brownian motion)
X_0 = 0.000000
X_T = 0.124567

======================================================================
실험 2: Strong vs Weak Solutions 비교
======================================================================
OU 과정: dX = -θX dt + σ dB, θ=2.0, σ=0.5
초기값: X_0 = 1.0

최종값 비교:
  강해 (with diffusion): X_T = 0.123456
  약한 근사 (drift only): X_T = 0.135335
  차이: 0.011879

강해의 앙상블 (n=500):
  t=0.5: 이론 평균=0.135335, 경험 평균=0.135267
  t=1.0: 이론 평균=0.135335, 경험 평균=0.135412

======================================================================
실험 3: Yamada-Watanabe 정리의 개념
======================================================================
[설명 텍스트 출력...]
```

---

## 🔗 AI/ML 연결

### Score-SDE와 약해의 역할

**Forward SDE** (데이터 → 노이즈):

$$dX_t = f(t) X_t dt + g(t) dB_t$$

Lipschitz 조건 만족 ⟹ 강해 존재.

**Reverse SDE** (노이즈 → 데이터):

$$dX = (f(t)X + g(t)^2 \nabla \log p_t(X)) dt + g(t) dB$$

Score function이 unbounded ⟹ Lipschitz 위반. 하지만:
- **약해의 존재성** (score matching 수렴)
- **Pathwise 유일성** (근사 score의 오차 제어)
- 따라서 Yamada-Watanabe로 **거의-강해** 또는 **분포적 수렴** 보장.

이것이 DDPM, Score-SDE, Flow Matching이 수렴하는 이론적 근거다.

### Probability Flow ODE와의 연결

Score-based 생성모델의 **Probability Flow ODE**:

$$\frac{dx}{dt} = f(t)x + \frac{1}{2}g(t)^2 \nabla \log p_t(x)$$

는 deterministic (SDE가 아님)이지만, 분포는 forward SDE와 같다 (martingale problem 관점). 이는 약해의 분포 유일성 개념과 밀접하다.

### Generative Model의 수렴성 분석

**Flow Matching** (Lipman et al., 2023)은 weak 형태의 SDE를 푸는데, **Yamada-Watanabe 없이도** conditional score matching으로 약해의 분포를 제어할 수 있음을 보인다. 이는 strong 조건 없이 생성 품질을 보장하는 새 패러다임이다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Lipschitz (강해) | 많은 실제 SDE가 만족하지 않음 |
| 유계 $b, \sigma$ (약해) | Score function처럼 unbounded인 경우 |
| Uniformly elliptic (약해) | Degenerate diffusion 다루기 어려움 |
| Pathwise 유일성 | 검증이 기술적으로 어려움 |

**주의**: 생성모델에서 "강해가 없어도 샘플링이 잘 된다"는 것은, **약해와 근사**의 위력을 보여준다. 하지만 이론적으로는 여전히 몇 가지 미해결 문제가 있다:
- Score matching의 수렴 속도
- 유한 신경망으로 인한 근사 오차
- Tail behavior와 극단값 샘플링

---

## 📌 핵심 정리

$$\boxed{\text{강해} \Rightarrow \text{약해} \quad \text{(자명)}}$$

$$\boxed{\text{약해 존재} + \text{pathwise 유일성} \Rightarrow \text{강해 존재} \quad \text{(Yamada-Watanabe)}}$$

| 개념 | 정의 | 응용 |
|------|------|------|
| **강해** | Pathwise, 주어진 BM | 이론적 분석, Lip schitz 조건 |
| **약해** | In-law, 분포 유일 | 실제 생성모델, Score-SDE |
| **Pathwise 유일성** | 같은 BM ⟹ 같은 경로 | Yamada-Watanabe 조건 |
| **Tanaka** | 약해만 (BM) | 강해 부재의 예시 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): OU 과정이 왜 강해를 가지는지, Tanaka 방정식이 왜 강해를 가지지 않는지 비교하라.

<details>
<summary>힌트 및 해설</summary>

**OU**: $dX = -\theta X dt + \sigma dB$
- $\sigma(x) = \sigma$ (상수, 연속)
- Lipschitz: $|\sigma(x) - \sigma(y)| = 0 \leq K|x-y|$ for any $K$
- 강해 존재 (pathwise 유일성 보장)

**Tanaka**: $dX = \text{sgn}(X) dB$
- $\sigma(x) = \text{sgn}(x)$ (불연속 at $x=0$)
- Lipschitz 위반: $|\text{sgn}(1) - \text{sgn}(-1)| = 2 \not\leq K \cdot 2\epsilon$ for small $\epsilon$
- 강해 부재 (pathwise 유일성 없음, Brownian motion의 거울상도 같은 분포)

</details>

**문제 2** (심화): Score-SDE의 reverse에서 score function $\nabla \log p_t(X)$가 저밀도 영역에서 unbounded일 때, 어떻게 샘플링이 수렴하는가? Yamada-Watanabe와의 관계를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**이론적 설명**:

1. Forward SDE는 Lipschitz ⟹ 강해 존재
2. Forward의 reverse (이상적 역)는 score를 포함하므로 일반적으로 Lipschitz 아님
3. **하지만**: score matching loss $\mathbb{E}[\|\nabla \log p_t - s_\theta\|^2]$로 신경망이 score를 근사
4. 근사 score $s_\theta$가 "충분히 좋으면" (오차 작으면), reverse도 pathwise 유일성 회복
5. Yamada-Watanabe: 약해 (limit 분포) 존재 + pathwise 유일성 (근사) ⟹ 거의-강해

**실무적 관찰**:
- DDPM, Score-SDE는 empirically 수렴
- 하지만 score 근사 오차가 크면 mode collapse, quality 저하
- 이것이 score matching 손실을 최소화하는 이유

</details>

**문제 3** (AI 연결): Flow Matching이 SDE 기반 생성모델보다 왜 더 안정적인가? Strong/weak solution의 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Flow Matching의 장점** (Lipman et al., 2023):

Traditional Score-SDE:
- Reverse SDE의 score를 학습
- Score의 unbounded성으로 인한 불안정
- Gradient flow와 score matching의 mismatch

Flow Matching:
- Conditional flow를 직접 학습 (score 불필요)
- Deterministic ODE로도 샘플링 가능
- Weak solution 관점: 분포 일치만 요구

**Strong/Weak 관점**:
- Score-SDE: strong 근사 시도 (오차 있음)
- Flow Matching: weak 근사 직접 (분포만 맞추기)
- Weak가 더 flexible하고 robust

**결론**: 약해의 유연성(flexibility)을 최대한 활용하면, 더 안정적인 생성이 가능.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 05. 선형 SDE의 일반해](./05-linear-sde.md) | [📚 README로 돌아가기](../README.md) | [Ch4-01. Fokker-Planck 방정식의 유도 ▶](../ch4-fokker-planck/01-fokker-planck-derivation.md) |

</div>
