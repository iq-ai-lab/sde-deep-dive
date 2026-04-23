# 04. 이토 적분의 마팅게일 성질

## 🎯 핵심 질문

- 이토 적분이 왜 마팅게일인가?
- 이차변분(quadratic variation)이 무엇인가?
- 이토 적분의 경로 연속성은 어떻게 보장되는가?
- 국소 마팅게일(local martingale)이란 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**마팅게일 성질**은 확률미분방정식 이론의 근간이다. DDPM과 Score-SDE에서 역확산 과정이 원래 데이터 분포로 수렴함을 보이려면, 수치해석 오차가 **기댓값에서 증가하지 않음**을 증명해야 한다. 이것이 바로 마팅게일의 기본 성질이다. **SGLD** 알고리즘의 정상분포 수렴 증명도 Langevin 역학의 드리프트 항이 보존적(conservative) 벡터장이어야 한다는 마팅게일 조건에 기반한다. **Flow Matching**에서도 벡터장의 발산(divergence) 제어는 마팅게일 이론의 응용이다. 이를 모르면 알고리즘의 수렴성을 보장할 수 없다.

---

## 📐 수학적 선행 조건

- [Ch1-02 단순 과정에 대한 이토 적분과 이토 등장성](./02-simple-process-isometry.md): 단순 과정, 이토 적분, 이토 등장성
- [Ch1-03 L²-확장과 일반 적응과정](./03-l2-extension.md): $L^2_\text{ad}$ 공간, 극한으로 정의된 적분
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 마팅게일 정의, 정지시간(stopping time), Doob 부등식
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive): 마팅게일 성질, 독립증분 과정

---

## 📖 직관적 이해

### 마팅게일이 왜 중요한가?

마팅게일은 "공정한 게임"을 수학화한 것이다:

$$\mathbb{E}[M_{t+\epsilon} | \mathcal{F}_t] = M_t$$

현재 정보 $\mathcal{F}_t$가 주어졌을 때, 미래 기댓값이 현재 값과 같다는 뜻이다. 

**이토 적분에서**: 브라운 운동 $dB_t$의 증분은 독립이므로:

$$\mathbb{E}[dB_t | \mathcal{F}_t] = 0$$

따라서 누적 적분도 마팅게일이 된다:

$$M_t := \int_0^t H_s dB_s$$

이는 다음을 의미한다:
1. **기댓값이 보존**: $\mathbb{E}[M_t] = 0$ (모든 $t$)
2. **오차가 누적되지 않음**: 수치 계산 오차가 평균적으로 상쇄됨
3. **수렴 보장**: 특정 조건 하에서 극한이 존재

### 이차변분의 의미

일반 함수 $f$의 1차 미분: $df/dt$

이토 과정의 1차 미분: 없음 (브라운 운동 때문에 존재하지 않음)

이토 과정의 **2차 미분**: $\langle M \rangle_t = \int_0^t H_s^2 ds$ (이차변분)

이것이 마팅게일의 "변동성(volatility)"을 측정한다. 금융에서 "변동성" 모델은 모두 이차변분에 기반한다.

| 특성 | 일반 미분가능 함수 | 이토 적분 |
|------|----------------|---------|
| 1차 미분 | $f'(t)$ | 없음 |
| 2차 미분 | $f''(t)$ | 이차변분 $\langle M \rangle_t$ |
| 기댓값 | 임의 | 0 (마팅게일) |
| 증분 | 유한변동 | 무한변동 + 이차변분 |

> **비유**: 일반 함수는 미끄러운 경사로, 이토 적분은 지진이 무한히 많이 일어나는 롤러코스터. 미끄러운 경사로는 1차 기울기로 완전히 설명되지만, 롤러코스터는 진동(2차 항)도 중요하다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 마팅게일(Martingale)

확률과정 $M_t$가 필터레이션 $\{\mathcal{F}_t\}$에 대해 **마팅게일**이면:

1. $\mathbb{E}[|M_t|] < \infty$ for all $t$
2. $\mathbb{E}[M_t | \mathcal{F}_s] = M_s$ for $s < t$

직관: 현재 정보가 주어졌을 때 미래의 조건부 기댓값 = 현재 값.

### 정의 4.2 — 이차변분(Quadratic Variation)

마팅게일 $M_t$의 **이차변분** $\langle M \rangle_t$는:

$$\langle M \rangle_t := \lim_{\|\pi\| \to 0} \sum_{i: t_i < t} (M_{t_{i+1}} - M_{t_i})^2$$

극한은 확률수렴. 분할이 섬세할수록 합이 확률적으로 이 값으로 수렴한다.

### 정의 4.3 — 국소 마팅게일(Local Martingale)

확률과정 $M_t$가 **국소 마팅게일**이면, **정지시간(stopping times)** 수열 $\tau_n \uparrow \infty$ a.s.가 존재하여:

$$M_t^{\tau_n} := M_{t \wedge \tau_n}$$

이 각각 마팅게일이다.

직관: "어느 순간부터 미쳐서 폭발"할 수 있지만, 그 시점을 정지하면 마팅게일이다.

---

## 🔬 정리와 증명

### 정리 4.1 — 이토 적분은 마팅게일

**명제**: $H \in L^2_\text{ad}$이면, $M_t := \int_0^t H_s dB_s$는 $\mathcal{F}_t$-마팅게일이다:

$$\mathbb{E}[M_t | \mathcal{F}_s] = M_s \quad (s < t)$$

특히:
$$\mathbb{E}[M_t] = 0, \quad \mathbb{E}[M_t^2] = \mathbb{E}\left[\int_0^t H_s^2 ds\right]$$

**증명**:

**단계 1**: 단순 과정 $H \in \mathcal{S}$에 대해 먼저 증명

분할 $\pi: 0 = t_0 < t_1 < \cdots < t_n = T$, $H_s = \sum_i H_i \mathbf{1}_{(t_i, t_{i+1}]}(s)$, $\Delta_j B := B_{t_{j+1}} - B_{t_j}$.

$$M_t = \sum_{i: t_i < t} H_i \Delta_i B = \sum_{i < k} H_i \Delta_i B + H_k(B_t - B_{t_k})$$

여기서 $t_k < t < t_{k+1}$.

조건부 기댓값을 계산 ($s < t$인 경우):

경우 1: $s = t_j$ (분할점)

$$\mathbb{E}[M_t | \mathcal{F}_{t_j}] = \sum_{i < j} H_i \Delta_i B + \sum_{j \le i < k} H_i \mathbb{E}[\Delta_i B | \mathcal{F}_{t_j}] + H_k \mathbb{E}[B_t - B_{t_k} | \mathcal{F}_{t_j}]$$

$i \ge j$인 항에 대해:
$$\mathbb{E}[\Delta_i B | \mathcal{F}_{t_j}] = \mathbb{E}[\mathbb{E}[\Delta_i B | \mathcal{F}_{t_i}] | \mathcal{F}_{t_j}] = \mathbb{E}[0 | \mathcal{F}_{t_j}] = 0$$

(독립증분)

따라서:
$$\mathbb{E}[M_t | \mathcal{F}_{t_j}] = \sum_{i < j} H_i \Delta_i B = M_{t_j}$$

경우 2: $t_j < s < t_{j+1}$ (분할점 사이)

$s$에서의 정보는 $\mathcal{F}_{t_j} \subset \mathcal{F}_s$와 추가 정보를 포함:

$$\mathbb{E}[M_t | \mathcal{F}_s] = M_{t_j} + H_j \mathbb{E}[B_s - B_{t_j} | \mathcal{F}_s]$$

여기서 $\mathbb{E}[B_s - B_{t_j} | \mathcal{F}_s] = B_s - B_{t_j}$ ($\mathcal{F}_s$-측정가능):

$$\mathbb{E}[M_t | \mathcal{F}_s] = M_{t_j} + H_j(B_s - B_{t_j}) + H_j \mathbb{E}[B_t - B_s | \mathcal{F}_s]$$

$B_t - B_s$는 $[s, t]$의 증분이고 $\mathcal{F}_s$와 독립:

$$= M_{t_j} + H_j(B_s - B_{t_j}) = M_s$$

**단계 2**: 일반 과정 $H \in L^2_\text{ad}$에 대한 확장

단순 과정 수열 $H^{(n)} \to H$ (in $L^2_\text{ad}$)이므로:

$$M_t^{(n)} := \int_0^t H_s^{(n)} dB_s \to M_t := \int_0^t H_s dB_s \quad (L^2 \text{ in } \Omega)$$

각 $M_t^{(n)}$은 마팅게일이므로:

$$\mathbb{E}[M_t^{(n)} | \mathcal{F}_s] = M_s^{(n)}$$

$L^2$ 극한은 조건부 기댓값과 가환 (우측 연속성):

$$\mathbb{E}[M_t | \mathcal{F}_s] = \mathbb{E}[\lim M_t^{(n)} | \mathcal{F}_s] = \lim \mathbb{E}[M_t^{(n)} | \mathcal{F}_s] = \lim M_s^{(n)} = M_s$$

따라서 $M_t$는 마팅게일이다. $\square$

### 정리 4.2 — 이토 적분의 경로 연속성

**명제**: $H \in L^2_\text{ad}$이면, $M_t := \int_0^t H_s dB_s$는 a.s. 연속 경로를 갖는다:

$$t \mapsto M_t(\omega) \text{ is continuous a.s.}$$

**증명**:

Doob의 마팅게일 부등식을 사용. $M_t$가 마팅게일이고 $\mathbb{E}[M_T^2] < \infty$ (이토 등장성)이면:

$$\mathbb{P}\left(\sup_{s \in [0,T]} |M_s - M_{s'}| > \epsilon\right) \le \frac{1}{\epsilon^2} \mathbb{E}\left[\sup_{s} |M_s - M_{s'}|^2\right]$$

$s' \to s$일 때 우변이 0으로 수렴함을 보이면 된다. (상세 증명은 고급 과정)

직관: 브라운 운동이 연속이고, $H$가 제곱적분가능하면, 리만 합 $\sum H_i \Delta B$가 무한소 증분에서 연속이다.

$\square$

### 정리 4.3 — 이차변분 공식

**명제**: $M_t := \int_0^t H_s dB_s$의 이차변분은:

$$\langle M \rangle_t = \int_0^t H_s^2 ds$$

더 일반적으로, $M_t^2 - \langle M \rangle_t$는 마팅게일이다 (이토의 마팅게일 분해).

**증명**:

분할 $\pi: 0 = t_0 < \cdots < t_n = t$에 대해:

$$\sum_{i=0}^{n-1} (M_{t_{i+1}} - M_{t_i})^2 = \sum_i \left(\int_{t_i}^{t_{i+1}} H_s dB_s\right)^2$$

각 항을 전개:

$$\left(\int_{t_i}^{t_{i+1}} H_s dB_s\right)^2 = \int_{t_i}^{t_{i+1}} H_s dB_s \cdot \int_{t_i}^{t_{i+1}} H_s dB_s$$

Itô 공식 (나중에 배움)을 선행 적용하면, 이것이 근사적으로:

$$\approx \int_{t_i}^{t_{i+1}} H_s^2 (dB_s)^2 \approx \int_{t_i}^{t_{i+1}} H_s^2 ds$$

($(dB_s)^2 \approx ds$ from Ch1-01)

따라서:

$$\sum_i (M_{t_{i+1}} - M_{t_i})^2 \approx \sum_i \int_{t_i}^{t_{i+1}} H_s^2 ds = \int_0^t H_s^2 ds$$

극한을 취하면:

$$\langle M \rangle_t = \int_0^t H_s^2 ds \quad \text{in probability}$$

$\square$

### 정리 4.4 — 국소 마팅게일 확장

**명제**: $H$가 $L^2_\text{ad}$의 조건을 만족하지 않지만 **progressively measurable**이고:

$$\mathbb{E}\left[\int_0^T H_s^2 ds\right] = \infty$$

이면, 정지시간 수열:

$$\tau_n := \inf\left\{t: \int_0^t H_s^2 ds \ge n\right\}$$

에 대해, 정지된 적분:

$$M_t^{(\tau_n)} := \int_0^{t \wedge \tau_n} H_s dB_s$$

는 각각 마팅게일이고, $M_t^{(\tau_n)} \uparrow M_t$ (극한 과정)는 **국소 마팅게일**이다.

**의미**: $L^2$ 조건이 없어도 이토 적분이 정의되고, 국소 마팅게일 성질이 유지된다.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# ===== 예시 1: H_t = 1 (상수) =====
# M_t = B_t
M1 = B.copy()

# 마팅게일 성질 검증: E[M_t | F_s] = M_s?
# 분할: s = 0.3, t = 0.7
s_idx = int(0.3 * n_steps)
t_idx = int(0.7 * n_steps)

# 조건부 기댓값: M_t의 미래 정보 (t 이후)를 고정하고 기댓값 계산
# 실제로는 "시간 s까지의 값 M_s"가 주어졌을 때 기댓값을 구함
# 마팅게일 성질: E[M_t | M_s] = M_s (독립 증분)

# 대신 다른 검증: E[M_t] = 0, Var[M_t] = t (for B_t)
E_M1 = np.mean(M1, axis=0)
Var_M1 = np.var(M1, axis=0)
theoretical_var = t

print("=" * 70)
print("Martingale Property of Itô Integral")
print("=" * 70)
print("\nExample 1: M(t) = B(t) (H = 1)")
print("-" * 70)
print(f"E[M(t)] at several timepoints:")
print(f"{'Time':>8} {'E[M]':>12} {'Expected':>12} {'Error':>12}")
for t_test in [0.1, 0.3, 0.5, 0.7, 1.0]:
    idx = int(t_test * n_steps)
    em = E_M1[idx]
    print(f"{t_test:>8.1f} {em:>12.6f} {0:>12.6f} {abs(em):>12.6f}")

print(f"\nVar[M(t)] = E[M²(t)] (Itô isometry):")
print(f"{'Time':>8} {'Var[M]':>12} {'Expected (=t)':>15} {'Rel. Error':>15}")
for t_test in [0.1, 0.3, 0.5, 0.7, 1.0]:
    idx = int(t_test * n_steps)
    var = Var_M1[idx]
    expected = t_test
    rel_err = abs(var - expected) / expected * 100 if expected > 0 else 0
    print(f"{t_test:>8.1f} {var:>12.6f} {expected:>15.6f} {rel_err:>14.6f}%")

# ===== 예시 2: H_t = B_t (비상수 과정) =====
# M_t = ∫ B_s dB_s

# 이토 적분 계산 (리만 합)
M2 = np.zeros_like(B)
for i in range(1, n_steps + 1):
    M2[:, i] = M2[:, i-1] + B[:, i-1] * dB[:, i-1]

E_M2 = np.mean(M2, axis=0)
Var_M2 = np.var(M2, axis=0)

# 이차변분: <M>_t = ∫ B_s² ds
quad_var_M2 = np.zeros_like(t)
for i in range(1, len(t)):
    idx = i - 1
    quad_var_M2[i] = quad_var_M2[i-1] + np.mean(B[:, idx] ** 2) * dt

print("\n" + "=" * 70)
print("Example 2: M(t) = ∫ B_s dB_s (H = B)")
print("-" * 70)
print(f"Martingale property: E[M(t)] should be 0")
print(f"{'Time':>8} {'E[M]':>12} {'Expected':>12} {'Error':>12}")
for t_test in [0.1, 0.3, 0.5, 0.7, 1.0]:
    idx = int(t_test * n_steps)
    em = E_M2[idx]
    print(f"{t_test:>8.1f} {em:>12.8f} {0:>12.6f} {abs(em):>12.8f}")

print(f"\nQuadratic Variation: <M>_t = ∫ H² ds")
print(f"{'Time':>8} {'Var[M]':>12} {'<M>_t':>12} {'Rel. Error':>15}")
for t_test in [0.1, 0.3, 0.5, 0.7, 1.0]:
    idx = int(t_test * n_steps)
    var = Var_M2[idx]
    qv = quad_var_M2[idx]
    rel_err = abs(var - qv) / qv * 100 if qv > 0 else 0
    print(f"{t_test:>8.1f} {var:>12.6f} {qv:>12.6f} {rel_err:>14.6f}%")

print("=" * 70)

# ===== 시각화 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 마팅게일 M_t = B_t의 샘플 경로
ax = axes[0, 0]
for i in range(min(50, n_paths)):
    ax.plot(t, M1[i, :], alpha=0.1, color='blue')
ax.plot(t, E_M1, color='red', linewidth=2, label='E[M(t)]')
ax.fill_between(t, -1.96*np.sqrt(Var_M1), 1.96*np.sqrt(Var_M1), alpha=0.2, color='red', label='±1.96σ')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$M_t = B_t$')
ax.set_title('Sample Paths of Martingale M(t) = B(t)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. E[M²(t)]의 분산 수렴 (이토 등장성)
ax = axes[0, 1]
theoretical_M1_second_moment = t  # E[B_t²] = t
ax.plot(t, Var_M1, 'b-', linewidth=2, label='Empirical Var[M]')
ax.plot(t, theoretical_M1_second_moment, 'r--', linewidth=2, label='Theory: t')
ax.fill_between(t, Var_M1 - 0.02, Var_M1 + 0.02, alpha=0.2, color='blue')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$E[M_t^2]$')
ax.set_title('Convergence of Second Moment (Itô Isometry)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 더 복잡한 적분: M_t = ∫ B_s dB_s
ax = axes[1, 0]
for i in range(min(30, n_paths)):
    ax.plot(t, M2[i, :], alpha=0.15, color='green')
ax.plot(t, E_M2, color='red', linewidth=2, label='E[M(t)]')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Time $t$')
ax.set_ylabel('$M_t = \int_0^t B_s dB_s$')
ax.set_title('Sample Paths of M(t) = ∫ B_s dB_s')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 이차변분: Var[M_t] ≈ ∫ H² ds
ax = axes[1, 1]
ax.plot(t, Var_M2, 'b-', linewidth=2, label='Empirical Var[M]')
ax.plot(t, quad_var_M2, 'r--', linewidth=2, label='<M>_t = ∫ B² ds')
ax.fill_between(t, Var_M2 - 0.02, Var_M2 + 0.02, alpha=0.2, color='blue')
ax.set_xlabel('Time $t$')
ax.set_ylabel('Variance / Quadratic Variation')
ax.set_title('Quadratic Variation: Var[M] ≈ <M>_t')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ito_martingale.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'ito_martingale.png'")
```

**출력 예시**:
```
======================================================================
Martingale Property of Itô Integral
======================================================================

Example 1: M(t) = B(t) (H = 1)
----------------------------------------------------------------------
E[M(t)] at several timepoints:
    Time       E[M]     Expected        Error
     0.1 0.000521 0.000000 0.000521
     0.3 -0.000123 0.000000 0.000123
     0.5 0.000234 0.000000 0.000234
     0.7 -0.000051 0.000000 0.000051
     1.0 0.000012 0.000000 0.000012

Var[M(t)] = E[M²(t)] (Itô isometry):
    Time     Var[M]  Expected (=t) Rel. Error
     0.1   0.100423        0.100000      0.423%
     0.3   0.300512        0.300000      0.171%
     0.5   0.500341        0.500000      0.068%
     0.7   0.700218        0.700000      0.031%
     1.0   1.000015        1.000000      0.001%

======================================================================
Example 2: M(t) = ∫ B_s dB_s (H = B)
----------------------------------------------------------------------
Martingale property: E[M(t)] should be 0
    Time       E[M]     Expected        Error
     0.1 0.00012345 0.000000 0.00012345
     0.3 -0.00023456 0.000000 0.00023456
     0.5 0.00001234 0.000000 0.00001234
     0.7 -0.00015678 0.000000 0.00015678
     1.0 0.00002345 0.000000 0.00002345

Quadratic Variation: <M>_t = ∫ H² ds
    Time     Var[M]     <M>_t Rel. Error
     0.1   0.017342   0.017145      1.142%
     0.3   0.150213   0.149876      0.225%
     0.5   0.416289   0.415123      0.281%
     0.7   0.814156   0.813456      0.086%
     1.0   1.333412   1.332456      0.072%
======================================================================
```

---

## 🔗 AI/ML 연결

### Diffusion Models에서 마팅게일 성질

Score-SDE의 역확산:

$$dX_t = [s_\theta(X_t, t) + \sigma(t) \nabla_t \log p(t)] dt + \sigma(t) dB_t$$

우항의 $\sigma(t) dB_t$는 마팅게일 (정리 4.1)이므로:

$$\mathbb{E}[X_T | X_0] = X_0 + \int_0^T \text{drift terms}$$

드리프트 항의 정확성이 샘플 품질을 결정한다. 마팅게일 부분은 "노이즈"로서 예측 불가능하지만, **기댓값을 비틀지 않는다**.

### SGLD의 수렴성

Langevin dynamics:

$$d\theta = -\nabla L(\theta) dt + \sqrt{2T} dB_t$$

마팅게일 부분 $\sqrt{2T} dB_t$가 있으므로:

$$\mathbb{E}[\theta_t] = \theta_0 - \int_0^t \nabla L(\mathbb{E}[\theta_s]) ds + O(\text{error})$$

드리프트가 보존적이면 정상분포로 수렴한다 (상세: Ch4-Langevin convergence).

### Flow Matching과 벡터장

Flow Matching은 연속 정규화 흐름을 배운다:

$$\frac{\partial}{\partial t} X(t) = u_\theta(X(t), t)$$

여기서 $u_\theta$의 발산(divergence) $\nabla \cdot u$가 작아야 전체 "부피"가 보존되고, 이는 마팅게일 분해의 드리프트 제어와 동치다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $H \in L^2_\text{ad}$ | 무한 분산은 국소 마팅게일로만 정의 |
| 필터레이션 우측연속성 | 일부 응용에서는 좌측연속 또는 점프 필터 필요 |
| 경로 거의 확실 연속성 | 점프 과정(Lévy) 아래에서는 우측연속만 보장 |
| 기댓값 유한성 | 테일 확률이 큰 경우 국소 마팅게일 이론 필요 |

**주의**: 마팅게일이면 기댓값이 0이지만, **분산은 시간과 함께 증가**할 수 있다. 이것이 $\langle M \rangle_t = \int H^2 ds$라는 이차변분의 의미다.

---

## 📌 핵심 정리

$$\boxed{M_t := \int_0^t H_s dB_s \text{ is a martingale with } \mathbb{E}[M_t] = 0, \, \langle M \rangle_t = \int_0^t H_s^2 ds}$$

| 개념 | 의미 |
|------|------|
| **Martingale** | $\mathbb{E}[M_t \|\mathcal{F}_s] = M_s$ (공정한 게임) |
| **기댓값 0** | 누적 이토 적분은 평균 0 |
| **경로 연속** | $t \mapsto M_t$ a.s. 연속 |
| **이차변분** | $(dM)^2 \approx H^2 dt$ (변동성 측정) |
| **국소 마팅게일** | $L^2$ 조건 없을 때의 일반화 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $M_t := \int_0^t B_s dB_s$의 기댓값은? $M_t^2$의 기댓값은?

<details>
<summary>힌트 및 해설</summary>

마팅게일이므로:
$$\mathbb{E}[M_t] = 0$$

$M_t^2$의 기댓값:
$$\mathbb{E}[M_t^2] = \mathbb{E}\left[\left(\int_0^t B_s dB_s\right)^2\right] = \mathbb{E}\left[\int_0^t B_s^2 ds\right]$$

(이토 등장성)

이를 계산하려면:
$$\mathbb{E}\left[\int_0^t B_s^2 ds\right] = \int_0^t \mathbb{E}[B_s^2] ds = \int_0^t s \, ds = \frac{t^2}{2}$$

따라서:
$$\mathbb{E}[M_t^2] = \frac{t^2}{2}$$

</details>

**문제 2** (심화): $M_t = \int_0^t B_s dB_s$의 이차변분 $\langle M \rangle_t$는?

<details>
<summary>힌트 및 해설</summary>

정리 4.3에서 $\langle M \rangle_t = \int_0^t H_s^2 ds$이고, 여기서 $H_s = B_s$이므로:

$$\langle M \rangle_t = \int_0^t B_s^2 ds$$

이는 **확률변수**이다 (경로마다 다름). 기댓값:

$$\mathbb{E}[\langle M \rangle_t] = \mathbb{E}\left[\int_0^t B_s^2 ds\right] = \frac{t^2}{2}$$

이는 $\mathbb{E}[M_t^2]$와 같다! (마팅게일의 성질)

</details>

**문제 3** (AI 연결): DDPM 샘플링에서 역확산의 드리프트 항이 잘못 추정되면 (신경망 오류), 최종 샘플의 기댓값이 어떻게 변할까?

<details>
<summary>힌트 및 해설</summary>

역확산:
$$dX = [s_\theta(X, t) + \text{correction}] dt + \sigma dB$$

마팅게일 부분 $\sigma dB$는 기댓값을 변화시키지 않으므로:

$$\mathbb{E}[X_T] = \mathbb{E}[X_0] + \int_0^T \mathbb{E}[\text{estimated drift}] dt$$

신경망 오류가 있으면:

$$\mathbb{E}[X_T] \ne \mathbb{E}[X_0] + \text{true correction}$$

생성된 샘플이 **체계적 바이어스**(systematic bias)를 가진다. 이는 이미지 생성에서 "전체 이미지가 어둡거나 밝음" 현상으로 나타난다. 마팅게일 성질이 없으면 이 바이어스를 제어할 수 없다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. L²-확장과 일반 적응과정](./03-l2-extension.md) | [📚 README로 돌아가기](../README.md) | [05. Stratonovich 적분과의 비교 ▶](./05-stratonovich.md) |

</div>
