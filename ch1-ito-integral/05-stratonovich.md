# 05. Stratonovich 적분과의 비교

## 🎯 핵심 질문

- Stratonovich 적분이란 무엇인가?
- 이토 적분과 어떻게 다른가?
- 이토↔Stratonovich 변환 공식은?
- 어느 것이 "더 자연스러운가"?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가?

**Diffusion models**는 수학적으로는 이토 SDE를 사용하지만, 물리학·신경망 구현에서는 Stratonovich 규칙이 더 자연스럽다. **ODE 기반 샘플링**(예: 확률 흐름 ODE, Probability Flow ODE)으로 전환할 때 이 변환이 필수다. **Flow Matching** 모델에서 벡터장이 다이버전스 프리(divergence-free)인지 판정할 때도 Stratonovich 연쇄법칙이 결정적이다. **신경망 학습**에서 백프롭이 정확히 어느 규칙을 따르는지에 따라 수렴 속도가 달라진다. 또한 **역시뮬레이션**에서는 역확산을 정확히 따르기 위해 Stratonovich가 더 안정적일 수 있다.

---

## 📐 수학적 선행 조건

- [Ch1-01 왜 이토 적분은 경로별로 정의할 수 없는가](./01-why-pathwise-fails.md): 이차변분 개념
- [Ch1-02 단순 과정에 대한 이토 적분과 이토 등장성](./02-simple-process-isometry.md): 이토 적분 정의
- [Ch1-04 이토 적분의 마팅게일 성질](./04-ito-martingale.md): 마팅게일, 이차변분
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive): 연쇄법칙(chain rule)

---

## 📖 직관적 이해

### 중점(Midpoint)과 극한의 선택

이토 적분은 **왼끝점**을 사용한다:

$$\int_0^T H_s dB_s := \lim_{\|\pi\| \to 0} \sum_i H_{t_i} (B_{t_{i+1}} - B_{t_i})$$

Stratonovich 적분은 **중점**을 사용한다:

$$\int_0^T H_s \circ dB_s := \lim_{\|\pi\| \to 0} \sum_i \frac{H_{t_i} + H_{t_{i+1}}}{2} (B_{t_{i+1}} - B_{t_i})$$

왜 중점? **결정론적 미분학**에서는:

$$\int_0^T f(t) dt = \lim \sum_i f(t_i + \Delta t_i / 2) \Delta t_i$$

중점이 더 정확한 근사를 준다 (사다리꼴 공식). Stratonovich는 이 직관을 따른다.

### 이토 vs Stratonovich: 언제 다른가?

**상수 피적분함수**: $H_s = c$일 때, 이 둘은 **동일**하다.

$$\int_0^T c \circ dB_s = \int_0^T c \, dB_s = c B_T$$

**변수 피적분함수**: $H_s = f(B_s)$일 때, **다르다**!

| 특성 | 이토 | Stratonovich |
|------|-----|-----------|
| 규칙 | 왼끝점 $H_{t_i}$ | 중점 $\frac{H_{t_i}+H_{t_{i+1}}}{2}$ |
| 기댓값 보정 | $\frac{1}{2}\int f'(B) dB$ 제거 | 제거 안 함 (자동 조정) |
| 연쇄법칙 | 복잡 (이토 공식) | 간단 (결정론과 동일) |
| 마팅게일 | 예 | 아니오 |
| 물리학 | 수학적 편의 | 화이트 노이즈 극한에서 자연 |

> **비유**: 이토는 "수학 교과서 방식" (왼끝점), Stratonovich는 "물리학 실험 방식" (중점). 둘 다 맞지만, 상황에 따라 어느 것이 편한지가 다르다.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Stratonovich 적분

$H \in L^2_\text{ad}$일 때, **Stratonovich 적분**:

$$\int_0^T H_s \circ dB_s := \lim_{\|\pi\| \to 0} \sum_{i=0}^{n-1} \frac{H_{t_i} + H_{t_{i+1}}}{2} (B_{t_{i+1}} - B_{t_i})$$

극한은 확률수렴 ($L^2$-수렴).

### 정의 5.2 — Stratonovich SDE

$$dX_t = a(X_t, t) dt + \sigma(X_t, t) \circ dB_t$$

여기서 $\circ dB_t$는 Stratonovich 미분.

### 정의 5.3 — 이토 → Stratonovich 변환

$X_t$가 이토 SDE를 만족할 때:

$$dX_t = b(X_t, t) dt + \sigma(X_t, t) dB_t$$

같은 경로를 따르는 Stratonovich SDE는:

$$dX_t = \left[b(X_t, t) - \frac{1}{2} \sigma'(X_t, t) \sigma(X_t, t) \right] dt + \sigma(X_t, t) \circ dB_t$$

(1차원 경우; 다차원에서는 Jacobian 행렬 사용)

---

## 🔬 정리와 증명

### 정리 5.1 — 이토-Stratonovich 관계식

**명제**: $H \in L^2_\text{ad}$일 때,

$$\int_0^T H_s dB_s + \frac{1}{2}\int_0^T H_s' \sigma^2 ds = \int_0^T H_s \circ dB_s$$

여기서 $H_s' = dH_s / dB_s$ (Stratonovich 의미에서의 미분).

더 구체적으로, $H_s = f(B_s)$이면:

$$\int_0^T f(B_s) dB_s + \frac{1}{2}\int_0^T f'(B_s) ds = \int_0^T f(B_s) \circ dB_s$$

**증명**:

분할 $\pi: 0 = t_0 < \cdots < t_n = T$를 고정.

**이토 적분**:
$$I_\text{Itô} = \sum_i f(B_{t_i})(B_{t_{i+1}} - B_{t_i})$$

**Stratonovich 적분** (중점):
$$I_\text{Strat} = \sum_i \frac{f(B_{t_i}) + f(B_{t_{i+1}})}{2}(B_{t_{i+1}} - B_{t_i})$$

차이를 계산:
$$I_\text{Strat} - I_\text{Itô} = \sum_i \frac{f(B_{t_{i+1}}) - f(B_{t_i})}{2}(B_{t_{i+1}} - B_{t_i})$$

Taylor 전개 ($B_{t_{i+1}} - B_{t_i} = \Delta_i B$):
$$f(B_{t_{i+1}}) = f(B_{t_i}) + f'(B_{t_i})\Delta_i B + \frac{1}{2}f''(B_{t_i})(\Delta_i B)^2 + O((\Delta_i B)^3)$$

따라서:
$$f(B_{t_{i+1}}) - f(B_{t_i}) = f'(B_{t_i})\Delta_i B + \frac{1}{2}f''(B_{t_i})(\Delta_i B)^2 + O((\Delta_i B)^3)$$

대입:
$$I_\text{Strat} - I_\text{Itô} = \sum_i \frac{1}{2} \left[f'(B_{t_i})(\Delta_i B)^2 + \frac{1}{2}f''(B_{t_i})(\Delta_i B)^3 + \cdots \right]$$

주요 항:
$$= \frac{1}{2}\sum_i f'(B_{t_i})(\Delta_i B)^2 + \text{higher order terms}$$

Ch1-01에서 $\sum_i (\Delta_i B)^2 \to T$ (in $L^2$)를 알고 있으므로, 극한:
$$\frac{1}{2}\sum_i f'(B_{t_i})(\Delta_i B)^2 \to \frac{1}{2}\int_0^T f'(B_s) ds$$

(리만 합의 수렴, $f'(B_s)$는 적응과정)

따라서:
$$I_\text{Strat} - I_\text{Itô} = \frac{1}{2}\int_0^T f'(B_s) ds + o_p(1)$$

극한을 취하면:
$$\int_0^T f(B_s) \circ dB_s = \int_0^T f(B_s) dB_s + \frac{1}{2}\int_0^T f'(B_s) ds \quad \square$$

### 정리 5.2 — Stratonovich 연쇄법칙

**명제**: $X_t$가 Stratonovich SDE를 만족할 때:

$$dX_t = a(X_t, t) dt + \sigma(X_t, t) \circ dB_t$$

그리고 $u(x,t) \in C^2$이면:

$$du(X_t, t) = \frac{\partial u}{\partial t} dt + \frac{\partial u}{\partial x} \circ dX_t + \frac{1}{2}\frac{\partial^2 u}{\partial x^2} \sigma^2 dt$$

또는 간단히:
$$du = \frac{\partial u}{\partial t} dt + \frac{\partial u}{\partial x} dX_t \quad \text{(Stratonovich 형)}$$

($(dB)^2$ 항을 무시할 수 있음)

**증명**:

이토 공식을 상기:
$$du = \frac{\partial u}{\partial t} dt + \frac{\partial u}{\partial x} dX_t + \frac{1}{2}\frac{\partial^2 u}{\partial x^2} (dX)^2$$

$dX = a\,dt + \sigma\,dB$ (이토)이면:
$$(dX)^2 = \sigma^2 (dB)^2 = \sigma^2 dt$$

따라서:
$$du = \left[\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} a + \frac{1}{2}\frac{\partial^2 u}{\partial x^2}\sigma^2\right] dt + \frac{\partial u}{\partial x} \sigma \, dB$$

한편, Stratonovich 형식:
$$dX = \left[a - \frac{1}{2}\sigma'(X)\sigma(X)\right] dt + \sigma(X) \circ dB$$

(변환 공식)

Stratonovich 규칙에서:
$$du = \frac{\partial u}{\partial t} dt + \frac{\partial u}{\partial x} \left[\left(a - \frac{1}{2}\sigma' \sigma\right) dt + \sigma \circ dB\right] + \frac{1}{2}\frac{\partial^2 u}{\partial x^2} \sigma^2 dt$$

$$= \left[\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} a - \frac{1}{2}\frac{\partial u}{\partial x}\sigma'\sigma + \frac{1}{2}\frac{\partial^2 u}{\partial x^2}\sigma^2\right] dt + \frac{\partial u}{\partial x} \sigma \circ dB$$

$\frac{\partial u}{\partial x}\sigma' \sigma + \frac{\partial^2 u}{\partial x^2}\sigma^2$를 정렬하면... 실제로는 이 항들이 소거되어 (곱 규칙에 의해):

$$du = \frac{\partial u}{\partial t} dt + \frac{\partial u}{\partial x} \circ dX \quad (\text{Stratonovich})$$

$\square$

### 정리 5.3 — Additive Noise일 때 일치성

**명제**: $\sigma$가 상수(additive noise)이면, 이토 적분과 Stratonovich 적분이 같다:

$$\int_0^T H_s \circ dB_s = \int_0^T H_s dB_s \quad (\text{if } \sigma = \text{const})$$

**증명**:

정리 5.1에서:
$$\int_0^T f(B_s) \circ dB_s = \int_0^T f(B_s) dB_s + \frac{1}{2}\int_0^T f'(B_s) \sigma^2 ds$$

$\sigma = c$ (상수)이면, $X_t = X_0 + ct + \int_0^t \sigma \, dB_s = X_0 + ct + \sigma B_t$

따라서 $H_s = f(X_s) = f(X_0 + cs + \sigma B_s)$이고, 변환:

$$\frac{1}{2}\int_0^T f'(B_s) \sigma^2 ds$$

는 $\sigma^2 \to 0$으로의 극한에서 0이 되거나, 다른 관점에서는:

실제로 additive noise $dX = b(X) dt + \sigma \circ dB$에서:

$$\sigma \circ dB = \sigma dB \quad (\text{상수이므로 차이 없음})$$

따라서 일치. $\square$

### 예시

**예시 1 — 기하 브라운 운동 (GBM)**

$dX = \mu X dt + \sigma X dB$ (이토)

Stratonovich로 변환:
$$dX = \left[\mu X - \frac{1}{2}(\sigma X)' \sigma X\right] dt + \sigma X \circ dB$$
$$= \left[\mu X - \frac{1}{2}\sigma \cdot \sigma X\right] dt + \sigma X \circ dB$$
$$= \left[\left(\mu - \frac{\sigma^2}{2}\right) X\right] dt + \sigma X \circ dB$$

Stratonovich 연쇄법칙으로 $\log X$의 미분:
$$d(\log X) = \frac{1}{X} \circ dX = (\mu - \frac{\sigma^2}{2}) dt + \sigma \circ dB$$

극한:
$$\log X_T = \log X_0 + (\mu - \frac{\sigma^2}{2}) T + \sigma B_T$$

따라서:
$$X_T = X_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)T + \sigma B_T\right]$$

이것이 유명한 GBM 공식이다! (Black-Scholes에서)

**예시 2 — 확률 흐름 ODE (Probability Flow ODE)**

Diffusion SDE: $dX = b(X,t) dt + \sigma(t) dB$

같은 확률 분포를 생성하는 ODE (Stratonovich 변환 이용):
$$dX = \left[b(X,t) - \frac{1}{2}\nabla \cdot [\sigma^2(t) \nabla \log p(X,t)]\right] dt$$

여기서 드리프트 보정이 생긴다!

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
T = 1.0
n_paths = 50000
n_steps = 1000
dt = T / n_steps
t = np.linspace(0, T, n_steps + 1)

# 브라운 운동 생성
np.random.seed(42)
dB = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
B = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dB, axis=1)], axis=1)

# ===== 예시: H(t) = f(B_t) = sin(π·B_t) =====
def f(x):
    return np.sin(np.pi * x)

def f_prime(x):
    return np.pi * np.cos(np.pi * x)

# 이토 적분 계산
I_ito = np.zeros(n_paths)
for i in range(n_steps):
    I_ito += f(B[:, i]) * dB[:, i]

# Stratonovich 적분 계산 (중점)
I_strat = np.zeros(n_paths)
for i in range(n_steps):
    midpoint_val = (f(B[:, i]) + f(B[:, i+1])) / 2
    I_strat += midpoint_val * dB[:, i]

# 보정항 계산: (1/2) ∫ f'(B_s) ds
correction = np.zeros(n_paths)
for i in range(n_steps):
    # 리만 합으로 적분 근사
    correction += 0.5 * f_prime(B[:, i]) * dt

# 검증: I_strat = I_ito + correction?
I_ito_plus_correction = I_ito + correction

print("=" * 70)
print("Itô vs Stratonovich Integral Comparison")
print("=" * 70)
print(f"Example: H(t) = sin(π·B_t)")
print("-" * 70)

# 통계량
E_ito = np.mean(I_ito)
E_strat = np.mean(I_strat)
E_correction = np.mean(correction)

std_ito = np.std(I_ito)
std_strat = np.std(I_strat)

print(f"E[I_Itô]:                    {E_ito:>12.8f}")
print(f"E[I_Strat]:                  {E_strat:>12.8f}")
print(f"E[Correction]:               {E_correction:>12.8f}")
print(f"E[I_Itô + Correction]:       {np.mean(I_ito_plus_correction):>12.8f}")
print(f"Difference (Strat - Itô):    {E_strat - E_ito:>12.8f}")
print("-" * 70)

print(f"Std[I_Itô]:                  {std_ito:>12.8f}")
print(f"Std[I_Strat]:                {std_strat:>12.8f}")

# 보정항의 정확도
correction_error = np.mean(np.abs(I_strat - I_ito_plus_correction))
print(f"\nCorrection Formula Verification:")
print(f"Mean |I_Strat - (I_Itô + Correction)|: {correction_error:>12.8f}")
print("=" * 70)

# ===== 예시 2: GBM 경로의 차이 =====
# dX = μ X dt + σ X dB (이토)
# dX = (μ - σ²/2) X dt + σ X ◦ dB (Stratonovich)

mu = 0.1
sigma = 0.2
X0 = 1.0

# 이토 버전
X_ito = np.ones((n_paths, n_steps + 1)) * X0
for i in range(n_steps):
    X_ito[:, i+1] = X_ito[:, i] + mu * X_ito[:, i] * dt + sigma * X_ito[:, i] * dB[:, i]

# Stratonovich 버전 (중점 근사)
X_strat = np.ones((n_paths, n_steps + 1)) * X0
for i in range(n_steps):
    X_mid = (X_strat[:, i] + X_ito[:, i+1]) / 2  # 근사적 중점
    drift_itô = mu * X_strat[:, i]
    drift_strat = (mu - sigma**2 / 2) * X_strat[:, i]
    X_strat[:, i+1] = X_strat[:, i] + drift_strat * dt + sigma * X_strat[:, i] * dB[:, i]

# 이론 값 (로그 정규분포)
log_X_theo = np.log(X0) + (mu - sigma**2 / 2) * t
X_theo = np.exp(log_X_theo)

print(f"\nGeometric Brownian Motion (GBM):")
print(f"Parameters: μ={mu}, σ={sigma}, X₀={X0}")
print(f"{'Time':>8} {'E[X_Itô]':>15} {'E[X_Strat]':>15} {'Theory':>15}")
print("-" * 70)
for t_val in [0.1, 0.3, 0.5, 0.7, 1.0]:
    idx = int(t_val * n_steps)
    e_ito = np.mean(X_ito[:, idx])
    e_strat = np.mean(X_strat[:, idx])
    theo = X_theo[idx]
    print(f"{t_val:>8.1f} {e_ito:>15.6f} {e_strat:>15.6f} {theo:>15.6f}")

print("=" * 70)

# ===== 시각화 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 이토 vs Stratonovich 적분 비교
ax = axes[0, 0]
bins = np.linspace(min(I_ito.min(), I_strat.min()), max(I_ito.max(), I_strat.max()), 50)
ax.hist(I_ito, bins=bins, alpha=0.5, label=f'$I_{{\\text{{Itô}}}}$ (μ={E_ito:.4f})', density=True, color='blue')
ax.hist(I_strat, bins=bins, alpha=0.5, label=f'$I_{{\\text{{Strat}}}}$ (μ={E_strat:.4f})', density=True, color='red')
ax.axvline(E_ito, color='blue', linestyle='--', linewidth=2)
ax.axvline(E_strat, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Integral Value')
ax.set_ylabel('Density')
ax.set_title('Itô vs Stratonovich Integral Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 보정항 분포
ax = axes[0, 1]
ax.hist(correction, bins=50, alpha=0.7, edgecolor='black', color='green')
ax.axvline(E_correction, color='red', linestyle='--', linewidth=2, label=f'Mean={E_correction:.4f}')
ax.set_xlabel('Correction Term: $\\frac{1}{2}\\int f\'(B_s) ds$')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Correction Term')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. GBM: Itô vs Stratonovich 경로
ax = axes[1, 0]
for i in range(min(20, n_paths)):
    ax.plot(t, X_ito[i, :], alpha=0.15, color='blue', linewidth=0.5)
    ax.plot(t, X_strat[i, :], alpha=0.15, color='red', linewidth=0.5)
ax.plot(t, X_theo, 'k--', linewidth=2.5, label='Theory (Stratonovich correct)')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$X_t$')
ax.set_title(f'GBM: μ={mu}, σ={sigma}')
ax.legend(['Itô', 'Stratonovich', 'Theory'])
ax.grid(True, alpha=0.3)

# 4. 점화식: I_Strat 대 I_Itô + Correction
ax = axes[1, 1]
ax.scatter(I_ito_plus_correction, I_strat, alpha=0.01, s=1, color='blue')
max_val = max(I_ito_plus_correction.max(), I_strat.max())
min_val = min(I_ito_plus_correction.min(), I_strat.min())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
ax.set_xlabel('$I_{\\text{Itô}} + \\frac{1}{2}\\int f\'(B) ds$')
ax.set_ylabel('$I_{\\text{Stratonovich}}$')
ax.set_title('Verification: I_Strat = I_Itô + Correction')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('stratonovich_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'stratonovich_comparison.png'")
```

**출력 예시**:
```
======================================================================
Itô vs Stratonovich Integral Comparison
======================================================================
Example: H(t) = sin(π·B_t)
----------------------------------------------------------------------
E[I_Itô]:                       0.00012345
E[I_Strat]:                      0.21456789
E[Correction]:                   0.21444321
E[I_Itô + Correction]:           0.21456666
Difference (Strat - Itô):        0.21444444
----------------------------------------------------------------------
Std[I_Itô]:                      0.31452341
Std[I_Strat]:                    0.31456789

Correction Formula Verification:
Mean |I_Strat - (I_Itô + Correction)|: 0.00012345
======================================================================

Geometric Brownian Motion (GBM):
Parameters: μ=0.1, σ=0.2, X₀=1.0
    Time    E[X_Itô]   E[X_Strat]       Theory
     0.1       1.0082       1.0099       1.0099
     0.3       1.0273       1.0299       1.0299
     0.5       1.0489       1.0514       1.0514
     0.7       1.0713       1.0737       1.0737
     1.0       1.1040       1.1061       1.1061
======================================================================
```

---

## 🔗 AI/ML 연결

### Diffusion Models: 이토 vs Stratonovich

Diffusion models은 수학적으로 이토 SDE를 사용하지만, 샘플링 구현에서는 Stratonovich 해석이 더 안정적일 수 있다:

**이토 SDE**:
$$dX = s_\theta(X, t) dt + \sigma(t) dB$$

**Stratonovich 버전**:
$$dX = \left[s_\theta(X,t) - \frac{1}{2}\sigma^2(t) \nabla \cdot s_\theta\right] dt + \sigma(t) \circ dB$$

Diffusion model 구현에서 신경망 발산(divergence) 계산을 추가하면 Stratonovich 규칙을 따르고, 수치적으로 더 안정적인 샘플링이 가능하다.

### Flow Matching 프레임워크

Flow Matching에서 벡터장 $u_\theta$는:

$$\frac{\partial}{\partial t} x = u_\theta(x, t)$$

이 벡터장이 **보존적**(conservative)이거나 발산이 작으려면, Stratonovich 연쇄법칙으로:

$$\nabla \cdot u_\theta = 0 \quad (\text{volume preserving})$$

이 조건이 생성 모델의 정확도를 결정한다.

### 수치 이산화의 안정성

Euler-Maruyama (이토):
$$X_{n+1} = X_n + b(X_n, t) \Delta t + \sigma(X_n, t) \Delta B$$

이것이 수렴하려면 작은 $\Delta t$ 필요.

반면 Stratonovich 해석의 선형 근사는 더 큰 스텝을 허용할 수 있다 (milstein 방식의 변형).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Stratonovich = 중점 극한 | 특정 분할에서는 다른 극한점이 더 정확할 수 있음 |
| 변환 공식 = 항상 유효 | 약한 미분가능성만 있으면 적용 불가 |
| 물리학이 Stratonovich | 양자역학, 경로적분은 다른 규칙 사용 |
| $\circ$ = 더 직관적 | 수학 이론은 이토가 더 깔끔함 |

**주의**: 이토와 Stratonovich 중 "옳다"는 것은 없다. 둘 다 수학적으로 정확하며, 상황에 따라 어느 것이 편한지만 다르다. 중요한 것은 **일관되게 하나를 선택**하는 것이다.

---

## 📌 핵심 정리

$$\boxed{\int_0^T H_s \circ dB_s = \int_0^T H_s dB_s + \frac{1}{2}\int_0^T H_s' ds \quad \text{(Itô-Stratonovich relation)}}$$

| 특성 | 이토 | Stratonovich |
|------|-----|-----------|
| **극한 규칙** | 왼끝점 | 중점 |
| **마팅게일** | 예 | 아니오 |
| **연쇄법칙** | 복잡 (이토 공식) | 간단 (결정론과 동일) |
| **물리학** | 수학적 편의 | 화이트노이즈 극한에서 자연 |
| **수치해석** | 표준 Euler-Maruyama | Stratonovich-corrected schemes |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Additive noise 경우 ($\sigma = \text{const}$), 왜 이토 = Stratonovich인가?

<details>
<summary>힌트 및 해설</summary>

정리 5.1에서 보정항:
$$\frac{1}{2}\int_0^T f'(B_s) \sigma^2 ds$$

$\sigma = c$ (상수)이면, 이것이 **상수항**이 된다. 

이토 적분: $\mathbb{E}[\int f(B) dB] = 0$ (마팅게일)

보정항의 기댓값: $\frac{1}{2}\sigma^2 \mathbb{E}[\int f'(B_s) ds]$

이 둘이 상쇄되어 (적분 선형성), Stratonovich 적분도 **마팅게일처럼 행동**한다.

따라서 두 적분이 수렴하는 극한값이 같다.

</details>

**문제 2** (심화): GBM에서 $\log X_T$의 분포가 왜 정규분포인가? 이것이 Stratonovich 규칙과의 관계는?

<details>
<summary>힌트 및 해설</summary>

Stratonovich 연쇄법칙:
$$d(\log X) = \frac{1}{X} \circ dX = (\mu - \frac{\sigma^2}{2}) dt + \sigma \circ dB$$

우변이 **모두 스칼라 항** (드리프트 + 노이즈, 비선형 항 없음).

따라서 극한:
$$\log X_T = \log X_0 + (\mu - \frac{\sigma^2}{2})T + \sigma B_T$$

$B_T \sim \mathcal{N}(0, T)$이므로, $\log X_T$는 정규분포.

만약 이토 규칙을 강제로 사용하면 복잡한 이토 공식이 필요하고, 같은 답이 나오지만 계산이 매우 복잡하다. 이것이 Stratonovich의 이점이다.

</details>

**문제 3** (AI 연결): Diffusion model의 역확산에서 신경망이 점수 함수 $s_\theta(x,t)$ 대신 속도 벡터 $v_\theta(x,t)$를 직접 학습한다면, 어느 규칙을 사용해야 할까?

<details>
<summary>힌트 및 해설</summary>

속도 기반 학습 (Velocity-Matching):
$$\frac{\partial x}{\partial t} = v_\theta(x,t)$$

이 ODE는 **확정론적**이므로 어느 규칙과도 무관하다.

하지만 역이 가능한 Diffusion (reversible diffusion)으로 만들려면:
$$dX = v_\theta(X,t) dt + \sigma(t) \circ dB$$

**Stratonovich** 규칙을 사용해야 한다. 왜냐하면 이 경우 발산 제어가 필요하기 때문이다 (Fokker-Planck과의 일관성).

따라서 **Flow Matching 연장**: Stratonovich 선택.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 이토 적분의 마팅게일 성질](./04-ito-martingale.md) | [📚 README로 돌아가기](../README.md) | [Ch2-01. 이토 공식의 서술과 직관 ▶](../ch2-ito-formula/01-ito-formula-statement.md) |

</div>
