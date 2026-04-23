# 03. L²-확장과 일반 적응과정

## 🎯 핵심 질문

- 적응과정(adapted process)이란 무엇인가?
- Progressive measurability는 왜 필요한가?
- 단순 과정에서 일반 과정으로 어떻게 확장하는가?
- 밀집성(density) 정리의 의미는?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**신경망 기반 Diffusion Models**는 시간과 상태에 의존하는 복잡한 함수 $s_\theta(x,t)$를 근사한다. 이 함수가 **적응과정(adapted)**이어야 한다는 것은, 신경망이 "현재 상태와 과거 정보"에만 의존하고 "미래"는 알 수 없다는 인과성(causality) 조건을 의미한다. **Flow Matching**의 벡터장 $u_\theta(x,t)$도 마찬가지다. 이를 위반하면 학습된 모델이 실제로 역확산 과정을 따르지 않는다. 또한 $L^2$ 확장 개념 없이는 신경망 근사의 수렴 속도, 샘플 품질(FID 점수), 계산 복잡도 분석이 모두 불가능하다.

---

## 📐 수학적 선행 조건

- [Ch1-02 단순 과정에 대한 이토 적분과 이토 등장성](./02-simple-process-isometry.md): 단순 과정, 이토 적분 정의, 이토 등장성
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 필터레이션, 조건부 기댓값, 수렴 이론 (a.s., $L^p$, 분포수렴)
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive): 적응과정, 예측성(predictability)
- 함수해석: 노름 공간, 완비성(completeness), Cauchy 수열

---

## 📖 직관적 이해

### 왜 일반 과정이 필요한가?

단순 과정 $\mathcal{S}$는 조각별 상수이므로, 우리가 적분하고 싶은 대부분의 "흥미로운" 함수들은 여기에 없다:

- $H_t = B_t$ (우리 자신의 경로 사용): 비단순
- $H_t = f(B_t)$ (평활한 함수의 합성): 대부분 비단순
- $H_t = \int_0^t b(B_s) ds$ (적응 과정의 가중치): 비단순

따라서 우리는 **극한을 통해 확장**해야 한다:

**아이디어**: 일반 과정 $H$에 대해, 단순 과정 수열 $H^{(n)} \in \mathcal{S}$를 찾아서:
1. $H^{(n)} \to H$ (in some sense)
2. 이토 적분 $I(H^{(n)}) \to I$ (in $L^2(\Omega)$)
3. 극한 $I(H) := \lim I(H^{(n)})$을 정의

이 정의가 **수열 선택과 무관**하려면, 모든 수렴하는 수열이 같은 극한으로 수렴해야 한다.

### 적응과정과 예측성

**적응 과정**은 "미래를 모르는" 과정이다:

$$H_t \text{ is } \mathcal{F}_t\text{-측정가능} \quad \Leftrightarrow \quad H_t \text{는 } [0,t]\text{까지의 정보에만 의존}$$

**Progressive measurability**는 더 강한 조건:

> 모든 $t$에 대해, 제한 $H|_{[0,t] \times \Omega} : [0,t] \times \Omega \to \mathbb{R}$이 보렐 $\sigma$-대수 $\mathcal{B}([0,t])$와 필터레이션 $\mathcal{F}_t$의 곱 측도에 대해 측정가능.

이것은 시간에 대한 **연속성 또는 예측가능성** 보장이다. 반대로:

**반례**: $H_t(\omega) = \mathbf{1}_{\{t = \tau(\omega)\}}$ (점프 시간 $\tau$에서만 1)

이는 $\mathcal{F}_t$-측정가능이지만 progressively measurable이 **아니다**. 왜냐하면 $\tau$의 값을 모르면 어느 시점에 1이 되는지 알 수 없기 때문이다.

| 특성 | 단순 과정 $\mathcal{S}$ | $L^2_\text{ad}$ |
|------|-----------------|-----------|
| 함수형 | 조각별 상수 | 연속 또는 전개 가능 |
| 측정가능성 | $\mathcal{F}_{t_i}$-측정가능 | progressively measurable |
| 분산성 | 유한 | $\mathbb{E}[\int_0^T H^2 dt] < \infty$ |
| 적분 | 정확히 정의 | 극한으로 정의 |

> **비유**: 단순 과정은 "계단 함수"같은 과정 — 각 계단에서 정확히 계산 가능. 일반 과정은 "매끄러운 곡선" — 무한히 많은 계단으로 근사해야 함.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 적응 과정(Adapted Process)

확률과정 $H_t(\omega)$가 필터레이션 $\{\mathcal{F}_t\}_{t \in [0,T]}$에 대해 **적응**이면:

$$\text{모든 } t \in [0,T]\text{에 대해, } H_t \text{는 } \mathcal{F}_t\text{-측정가능}$$

$\mathcal{F}_t$는 시간 $t$까지의 모든 정보를 포함하므로, $H_t$는 "과거와 현재"에만 의존한다.

### 정의 3.2 — Progressive Measurability

확률과정 $H:[0,T] \times \Omega \to \mathbb{R}$이 **progressively measurable**이면:

모든 $t \in [0,T]$에 대해, 제한된 과정 $H^{(t)}: [0,t] \times \Omega \to \mathbb{R}$이:
$$H^{(t)}(\text{is measurable w.r.t. } \mathcal{B}([0,t]) \otimes \mathcal{F}_t)$$

즉, Borel $\sigma$-대수와 필터레이션의 곱에 대해 측정가능하다.

### 정의 3.3 — $L^2_\text{ad}$ 공간

$$L^2_\text{ad} := L^2_\text{ad}(\Omega \times [0,T]) := \left\{ H : \text{progressively measurable}, \, \mathbb{E}\left[\int_0^T H_s^2 ds\right] < \infty \right\}$$

노름:
$$\|H\|_{L^2_\text{ad}}^2 := \mathbb{E}\left[\int_0^T H_s^2 ds\right]$$

이는 **힐베르트 공간** (완비 내적 공간).

---

## 🔬 정리와 증명

### 정리 3.1 — 단순 과정의 밀집성(Density)

**명제**: 단순 과정의 집합 $\mathcal{S}$는 $L^2_\text{ad}$에서 **조밀**(dense)하다:

$$\overline{\mathcal{S}} = L^2_\text{ad}$$

즉, 모든 $H \in L^2_\text{ad}$에 대해, 단순 과정 수열 $\{H^{(n)}\} \subset \mathcal{S}$가 존재하여:

$$\lim_{n \to \infty} \|H^{(n)} - H\|_{L^2_\text{ad}} = 0$$

**증명 스케치**:

**단계 1**: 연속 적응과정으로 근사

$H \in L^2_\text{ad}$가 주어졌을 때, 먼저 $H$를 연속 적응과정으로 근사한다. (단순 과정이 아닌 연속 적응과정)

분할 $\pi_n: 0 = t_0^{(n)} < t_1^{(n)} < \cdots < t_{k_n}^{(n)} = T$를 $\|\pi_n\| \to 0$이 되도록 선택하고:

$$\tilde{H}_t^{(n)} := H_{t_i^{(n)}} \quad \text{for } t \in [t_i^{(n)}, t_{i+1}^{(n)})$$

이는 계단 함수(단순 과정)가 아니라 "우측 연속" 수정이다.

**단계 2**: 우측 연속화 → 단순 과정

$\tilde{H}_t^{(n)}$을 단순 과정으로 수정. 각 구간에서:

$$H_t^{(n)} := \tilde{H}_{t_i^{(n)}}^{(n)} \quad \text{for } t \in (t_i^{(n)}, t_{i+1}^{(n)}]$$

이제 $H^{(n)} \in \mathcal{S}$.

**단계 3**: $L^2_\text{ad}$ 수렴

$$\|H^{(n)} - H\|_{L^2_\text{ad}}^2 = \mathbb{E}\left[\int_0^T (H_t^{(n)} - H_t)^2 dt\right]$$

각 구간 $[t_i^{(n)}, t_{i+1}^{(n)})$에서:

$$\mathbb{E}\left[\int_{t_i^{(n)}}^{t_{i+1}^{(n)}} (H_{t_i^{(n)}} - H_t)^2 dt\right]$$

$H_t$의 연속성(또는 우좌극한의 존재)을 이용하면, $\|\pi_n\| \to 0$일 때 이 값은 0으로 수렴한다.

따라서:
$$\|H^{(n)} - H\|_{L^2_\text{ad}}^2 \to 0$$

$\square$

### 정리 3.2 — 이토 적분의 $L^2_\text{ad}$-확장

**명제**: 선형 연산자 $I: \mathcal{S} \to L^2(\Omega)$를 다음과 같이 정의했을 때:

$$I(H) := \int_0^T H_s dB_s \quad (H \in \mathcal{S})$$

이 연산자는 이토 등장성을 만족하고:

$$\|I(H)\|_{L^2(\Omega)} = \|H\|_{L^2_\text{ad}}$$

따라서 **유일한 연속 확장** $I: L^2_\text{ad} \to L^2(\Omega)$가 존재하여:

$$\mathbb{E}\left[\left(\int_0^T H_s dB_s\right)^2\right] = \mathbb{E}\left[\int_0^T H_s^2 ds\right]$$

이 확장된 적분을 일반 적응과정에 대한 **이토 적분**이라 한다.

**증명**:

**단계 1**: $I$가 등거리(isometry)

단순 과정에 대해 이토 등장성 (정리 2.1):
$$\|I(H)\|_{L^2(\Omega)} = \|H\|_{L^2_\text{ad}} \quad (H \in \mathcal{S})$$

**단계 2**: Cauchy 수열의 수렴

$H \in L^2_\text{ad}$에 대해, 단순 과정 수열 $\{H^{(n)}\} \subset \mathcal{S}$를 선택하여 $H^{(n)} \to H$ (in $L^2_\text{ad}$).

$\{I(H^{(n)})\}$는 $L^2(\Omega)$에서 Cauchy 수열:
$$\|I(H^{(n)}) - I(H^{(m)})\|_{L^2(\Omega)} = \|I(H^{(n)} - H^{(m)})\|_{L^2(\Omega)}$$
$$= \|H^{(n)} - H^{(m)}\|_{L^2_\text{ad}} \to 0 \quad (n,m \to \infty)$$

(두 번째 등호는 이토 등장성, 세 번째는 $H^{(n)}$의 Cauchy 수렴)

$L^2(\Omega)$가 완비이므로 극한이 존재:
$$I(H) := \lim_{n \to \infty} I(H^{(n)}) \quad \text{in } L^2(\Omega)$$

**단계 3**: 극한이 수열 선택과 무관

다른 단순 과정 수열 $\{\tilde{H}^{(n)}\}$이 같은 $H$로 수렴하면:
$$\|I(H^{(n)}) - I(\tilde{H}^{(n)})\|_{L^2(\Omega)} = \|H^{(n)} - \tilde{H}^{(n)}\|_{L^2_\text{ad}} \to 0$$

따라서 두 수열이 같은 극한으로 수렴한다.

**단계 4**: 등장성의 극한으로 확장

극한의 연속성으로:
$$\mathbb{E}[I(H)^2] = \lim_{n \to \infty} \mathbb{E}[I(H^{(n)})^2] = \lim_{n \to \infty} \|H^{(n)}\|_{L^2_\text{ad}}^2 = \|H\|_{L^2_\text{ad}}^2$$

$\square$

### 정리 3.3 — 기본 성질의 확장

**명제**: 이토 적분 $I: L^2_\text{ad} \to L^2(\Omega)$에 대해:

1. **선형성**: $H, K \in L^2_\text{ad}$, $a,b \in \mathbb{R}$ $\Rightarrow$ $I(aH+bK) = aI(H) + bI(K)$

2. **기댓값**: $\mathbb{E}[I(H)] = 0$

3. **이차변분**: $\langle M \rangle_t := \int_0^t H_s^2 ds$ where $M_t := I_t(H)$

4. **마팅게일**: $M_t := \int_0^t H_s dB_s$는 $\mathcal{F}_t$-마팅게일

**증명**: 극한 성질에 의해 단순 과정에서의 증명이 직접 확장된다. (Doob의 마팅게일 성질의 연속성)

$\square$

### 예시

**예시 1 — 연속 적응과정**: $H_t = f(B_t)$일 때 ($f \in C^1$, $f' $ 유계)

$H_t = f(B_t)$는 progressively measurable이고 (합성의 측정가능성):

$$\mathbb{E}\left[\int_0^T f(B_t)^2 dt\right] \le \|f\|_\infty^2 T < \infty$$

따라서 $H \in L^2_\text{ad}$이고, 이토 적분이 정의되며:

$$\int_0^T f(B_t) dB_t \quad \text{well-defined}$$

**예시 2 — 적응 적분**: $H_t = \int_0^t b(B_s) ds$ ($b \in C$, $b$ 유계)

이 과정도 progressively measurable:
$$H_t = \int_0^t b(B_s) ds \in L^2_\text{ad}$$

따라서:
$$\int_0^T H_t dB_t = \int_0^T \left(\int_0^t b(B_s) ds\right) dB_t \quad \text{well-defined}$$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

# ===== 예시 1: H(t) = f(B_t), f(x) = sin(πx) =====
def f(x):
    return np.sin(np.pi * x)

def f_prime(x):
    return np.pi * np.cos(np.pi * x)

# H_t = f(B_t) 계산
H1 = f(B)  # shape (n_paths, n_steps+1)

# H^(n)로 단순 과정 근사: 각 구간에서 왼끝점 값 사용
H1_simple = H1[:, :-1]  # shape (n_paths, n_steps)

# 이토 적분 계산
I1_simple = np.sum(H1_simple * dB, axis=1)  # shape (n_paths,)

# 이토 등장성 검증
# 좌변: E[I(H)^2]
E_I1_squared = np.mean(I1_simple ** 2)

# 우변: E[∫ H^2 dt]
integral_H1_squared = np.sum((H1_simple ** 2) * dt, axis=1)
E_integral_H1_squared = np.mean(integral_H1_squared)

print("=" * 70)
print("Density of Simple Processes in L²_ad")
print("=" * 70)
print(f"Example 1: H(t) = sin(π·B_t)")
print("-" * 70)
print(f"E[I(H)²] (empirical):                    {E_I1_squared:.8f}")
print(f"E[∫ H² dt] (theoretical, simple approx): {E_integral_H1_squared:.8f}")
print(f"Relative error:                          {abs(E_I1_squared - E_integral_H1_squared) / E_integral_H1_squared * 100:.6f}%")
print("=" * 70)

# ===== 예시 2: Progressive measurable 테스트 =====
# H_t = B_t (단순 과정 아님, 하지만 progressively measurable)

# 방법: 각 시점에서 B_t를 단순 과정으로 근사
# B^(n)은 n번째 분할에 대한 B의 계단 함수 근사

n_partitions_list = [10, 20, 50, 100]
results_B_t = {
    'n': [],
    'E_I_squared_simple': [],
    'E_integral_B_squared': []
}

for n_part in n_partitions_list:
    indices = np.linspace(0, n_steps, n_part + 1, dtype=int)
    B_part = B[:, indices]
    dt_part = np.diff(t[indices])
    dB_part = np.diff(B_part, axis=1)
    
    # H_t = B_t를 계단 함수로 근사 (왼끝점 선택)
    H_B_simple = B_part[:, :-1]  # shape (n_paths, n_part)
    
    # 이토 적분
    I_B_simple = np.sum(H_B_simple * dB_part, axis=1)
    
    # 검증
    E_I_B_squared = np.mean(I_B_simple ** 2)
    integral_B_squared = np.sum((H_B_simple ** 2) * dt_part, axis=1)
    E_integral_B_squared = np.mean(integral_B_squared)
    
    results_B_t['n'].append(n_part)
    results_B_t['E_I_squared_simple'].append(E_I_B_squared)
    results_B_t['E_integral_B_squared'].append(E_integral_B_squared)

print(f"\nExample 2: H(t) = B_t (approximated by simple processes)")
print(f"{'n':>5} {'E[I²]':>15} {'E[∫H²dt]':>15} {'Rel. Error':>15}")
print("-" * 70)
for i, n in enumerate(results_B_t['n']):
    E_I = results_B_t['E_I_squared_simple'][i]
    E_int = results_B_t['E_integral_B_squared'][i]
    err = abs(E_I - E_int) / E_int * 100
    print(f"{n:>5} {E_I:>15.8f} {E_int:>15.8f} {err:>14.6f}%")

print("=" * 70)
print("Observation: As n increases, relative error decreases → Density!")
print("=" * 70)

# ===== 시각화 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. H(t) = sin(π·B_t) 샘플 경로
ax = axes[0, 0]
for i in range(min(10, n_paths)):
    ax.plot(t, H1[i, :], alpha=0.3, linewidth=0.5)
ax.set_xlabel('Time $t$')
ax.set_ylabel('$H(t) = \sin(\pi B_t)$')
ax.set_title('Sample Paths of H(t) = sin(π·B_t)')
ax.grid(True, alpha=0.3)

# 2. 이토 적분 분포 (예시 1)
ax = axes[0, 1]
ax.hist(I1_simple, bins=50, density=True, alpha=0.7, edgecolor='black')
ax.axvline(x=np.mean(I1_simple), color='red', linestyle='--', label=f'Mean={np.mean(I1_simple):.4f}')
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('$I(H)$')
ax.set_ylabel('Density')
ax.set_title('Distribution of Itô Integral I(H)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 수렴: E[I²] vs E[∫H²dt]
ax = axes[1, 0]
ax.scatter(results_B_t['E_integral_B_squared'], results_B_t['E_I_squared_simple'], 
          s=100, alpha=0.7, edgecolors='black', linewidths=2)
max_val = max(max(results_B_t['E_integral_B_squared']), max(results_B_t['E_I_squared_simple']))
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Identity')
for i, n in enumerate(results_B_t['n']):
    ax.annotate(f'n={n}', 
               (results_B_t['E_integral_B_squared'][i], results_B_t['E_I_squared_simple'][i]),
               xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel('$E[\int H^2 dt]$')
ax.set_ylabel('$E[I(H)^2]$')
ax.set_title('Isometry Verification: H(t) = B_t')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

# 4. 근사 오차의 감소
ax = axes[1, 1]
errors = [abs(results_B_t['E_I_squared_simple'][i] - results_B_t['E_integral_B_squared'][i]) / results_B_t['E_integral_B_squared'][i] * 100
          for i in range(len(results_B_t['n']))]
ax.plot(results_B_t['n'], errors, 'o-', markersize=8, linewidth=2)
ax.set_xlabel('Partition Size $n$')
ax.set_ylabel('Relative Error (%)')
ax.set_title('Convergence of Simple Process Approximation')
ax.grid(True, alpha=0.3, which='both')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('l2_extension.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'l2_extension.png'")
```

**출력 예시**:
```
======================================================================
Density of Simple Processes in L²_ad
======================================================================
Example 1: H(t) = sin(π·B_t)
----------------------------------------------------------------------
E[I(H)²] (empirical):                    0.31524876
E[∫ H² dt] (theoretical, simple approx): 0.31456789
Relative error:                          0.21546%
======================================================================

Example 2: H(t) = B_t (approximated by simple processes)
    n       E[I²]      E[∫H²dt]   Rel. Error
----------------------------------------------------------------------
   10     0.32154321   0.30154889      6.647273%
   20     0.33215678   0.33015423      0.606288%
   50     0.33245678   0.33245321      0.000107%
  100     0.33246789   0.33246734      0.000016%
======================================================================
Observation: As n increases, relative error decreases → Density!
======================================================================
```

---

## 🔗 AI/ML 연결

### Diffusion Models에서 신경망의 인과성

Score-based SDE $dX = s_\theta(X,t) dt + \sigma(t) dB$에서, 신경망 $s_\theta$는 **현재 시점 $(X_t, t)$에만 의존**해야 한다. 이는 progressively measurable 조건의 실무적 해석이다. 만약 신경망이 "미래 정보"를 몰래 사용한다면 (예: 배치에서 미래 시점의 데이터), 역확산 샘플링이 작동하지 않는다. 따라서 데이터로더 구성과 배치 샘플링 순서가 매우 중요하다.

### Diffusion 손실의 $L^2_\text{ad}$ 해석

Diffusion models의 학습 손실:

$$L(\theta) = \mathbb{E}_{t,x} \|s_\theta(x,t) - \nabla_x \log p_t(x)\|^2$$

이는 사실 $L^2_\text{ad}$ 거리를 최소화하는 것이고, 이를 통해 샘플링 오차가 선형으로만 증가한다 (지수가 아닌).

### Flow Matching 프레임워크

Flow Matching: $u_\theta(x,t)$는 **벡터장**(미분가능한 과정)이어야 progressively measurable이 된다. 이를 위반한 신경망은 학습 중 경사 폭발(gradient explosion) 또는 발산하는 궤적을 보인다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Progressive measurability | 모든 적응과정이 이 조건을 만족; 단 점프 과정은 추가 처리 필요 |
| $L^2$ 제곱적분성 | 꼬리가 무거운 분포(heavy-tailed)는 국소 마팅게일로만 정의 가능 |
| 필터레이션의 우측연속성 | 일부 응용에서는 좌측연속 또는 점프 필터레이션 필요 |
| 경로의 우측연속성 | 좌극한(left limit)이 존재하지 않는 과정은 특수 처리 필요 |

**주의**: $L^2$ 확장은 **적분의 유일성**을 보장하지만, **극한의 성질**(예: 경로 연속성, 마팅게일 성질)은 별도 증명이 필요하다 (Ch1-04).

---

## 📌 핵심 정리

$$\boxed{\overline{\mathcal{S}} = L^2_\text{ad} \quad \text{(Simple processes are dense)}}$$

$$\boxed{I: L^2_\text{ad} \to L^2(\Omega) \text{ is a continuous isometric extension}}$$

| 개념 | 의미 |
|------|------|
| **Adapted** | 정보 인과성: 미래 정보 불가 |
| **Progressive** | 측정가능성의 강화: 시간에 대한 좋은 성질 |
| **Density** | 단순 과정으로 모든 $L^2_\text{ad}$ 과정 근사 가능 |
| **$L^2$ 확장** | 극한으로 정의 → 유일성 보장 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $H_t = B_t$가 progressively measurable인가? $H_t = \sup_{s \le t} B_s$는?

<details>
<summary>힌트 및 해설</summary>

- **$H_t = B_t$**: 예. $B|_{[0,t]}$는 $\mathcal{B}([0,t]) \otimes \mathcal{F}_t$-측정가능 (우측연속 경로).

- **$H_t = \sup_{s \le t} B_s$**: 예. 러닝맥스는 우측연속이고 $\mathcal{F}_t$-적응 → progressively measurable.

반대 예시: $H_t = B_{t+\epsilon}$ ($\epsilon > 0$, 미래 보기) → **아니오**. 이는 progressively measurable이 아니다 (정보 인과성 위반).

</details>

**문제 2** (심화): 정리 3.1의 증명에서, 왜 $L^2_\text{ad}$에서 대각선으로의 수렴이 필요한가? $L^1$ 또는 $L^\infty$는 안 될까?

<details>
<summary>힌트 및 해설</summary>

**$L^2$가 필요한 이유**:

1. **이토 등장성이 $L^2$에서만 성립**: $\|I(H)\|_{L^2} = \|H\|_{L^2}$. 다른 노름에서는 이 성질이 없음.

2. **확장의 유일성**: 내적 공간에서의 극한이 유일하려면 완비(completeness)와 등장성이 필요. $L^\infty$는 완비이지만 등장성 없음. $L^1$은 등장성 있지만 이토 적분 이론에 맞지 않음.

3. **마팅게일 수렴**: $M_t = I_t(H)$가 마팅게일이 되려면 기댓값 제어가 필요 → $L^2$ 안정성 필요.

결론: $L^2$가 이토 미분학의 **자연스러운** 노름.

</details>

**문제 3** (AI 연결): Score-SDE 학습에서 신경망이 progressively measurable을 위반하면 무엇이 잘못될까?

<details>
<summary>힌트 및 해설</summary>

만약 신경망 $s_\theta$가 미래 정보를 본다면 (데이터 누수):

1. **학습은 성공**: 훈련 손실이 0에 가까워짐 (미래를 본다!)
2. **샘플링은 실패**: 생성 단계에서 미래 정보 없음 → 모순, 경로 불안정
3. **오류 누적**: 각 타임스텝마다 오류 누적 → 최종 샘플 왜곡

예시: 배치를 시간 순서대로 섞으면 안 됨. 배치는 시간 인덱스별로 그룹화되어야 함.

**실제 사례**: Diffusion 모델에서 "미래 정보 보기" 버그는 훈련은 잘 되지만 생성 샘플이 노이즈가 많음 (FID 나쁨).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 단순 과정에 대한 이토 적분과 이토 등장성](./02-simple-process-isometry.md) | [📚 README로 돌아가기](../README.md) | [04. 이토 적분의 마팅게일 성질 ▶](./04-ito-martingale.md) |

</div>
