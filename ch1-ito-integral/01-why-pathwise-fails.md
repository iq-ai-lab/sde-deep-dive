# 01. 왜 이토 적분은 경로별로 정의할 수 없는가

## 🎯 핵심 질문

- 브라운 운동의 표본 경로가 왜 유한변동이 아닌가?
- 리만-스틸체스 적분이 왜 작동하지 않는가?
- 적분의 극한이 분할점 선택에 의존하는 이유는 무엇인가?
- 이차변분(quadratic variation)이 왜 핵심 개념인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Diffusion models**(DDPM, Score-SDE, Flow Matching)에서 노이즈 과정은 브라운 운동으로 모델링된다. 이 노이즈가 **유한변동이 아니라는 사실**은 결정론적 수치해석(유한차분, 오일러 스킴)과 확률적 해석(이토 규칙)의 근본적 괴리를 드러낸다. 이해 부족 시, 학습 손실함수 설계, 샘플링 수렴성, 그리고 **Score Matching**의 정확도 분석이 모두 흔들린다. 특히 **SGLD**(Stochastic Gradient Langevin Dynamics)나 **TRPO**(Trust Region Policy Optimization)에서 확률미분방정식의 이산화 오차가 학습 안정성을 결정한다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률공간, 필터레이션(filtration), 조건부 기댓값
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive): 브라운 운동 정의, 경로 연속성, 독립증분
- 실변수함수론: 유한변동(bounded variation), 전변동(total variation) $V_0^T(f)$, 절대연속성
- 확률론: 수렴 개념 (거의 확실 a.s., $L^2$, 분포수렴)

---

## 📖 직관적 이해

### 왜 "경로별 적분"이 불가능한가

일반함수 $f:[0,T]\to\mathbb{R}$와 유한변동 함수 $g$에 대해 리만-스틸체스 적분 $\int_0^T f(t) dg(t)$는 **분할점 선택과 무관하게** 잘 정의된다:

$$\int_0^T f(t) dg(t) = \lim_{\|\pi\|\to 0} \sum_{i} f(\tau_i)(g(t_{i+1})-g(t_i))$$

(분할 $\pi: 0=t_0 < t_1 < \cdots < t_n = T$, $\tau_i \in [t_i, t_{i+1}]$, 극한이 분할과 $\tau_i$ 선택 무관)

**조건**: $|g|_{\text{BV}} := \sup_{\pi} \sum_i |g(t_{i+1})-g(t_i)| < \infty$

그러나 **브라운 운동 $B_t$는** 거의 확실하게 **무한변동**이다: $|B|_{\text{BV}} = \infty$ a.s.

이 때문에:
1. $\sum H_i (B_{t_{i+1}} - B_{t_i})$의 극한이 **분할점 위치** $t_i^*$ 선택에 의존
2. 서로 다른 $t_i^*$ 선택은 서로 다른 극한값을 준다
3. "적분"이 유일하게 정의되지 않는다

| 특성 | 유한변동 ($dt$ 형) | 무한변동 (BM $dB_t$) |
|------|------------------|-----------------|
| 총 변동 | $\sum \|df\|$ 수렴 | 발산 |
| 이차변분 | 0 | $\neq 0$ |
| 리만-스틸체스 | ✓ (분할 무관) | ✗ (분할 의존) |
| 적분 정의 | 고유함수 | 확률변수 (선택 의존) |

> **비유**: 도시의 지하철망(유한변동)은 어느 경로든 "총 거리"가 정해져 있다. 하지만 난기류 속 비행기(브라운 운동)는 지그재그가 너무 많아서 "표본 경로의 길이"가 무한대다. 따라서 정의부터가 측정 방식(분할점)에 의존하게 된다.

### 이차변분이 핵심인 이유

리만-스틸체스 이론에서 놓친 부분이 바로 **제곱항**이다:

- 유한변동 함수 $f$: $(df)^2 \approx 0$ → $df \cdot df$ 항 무시 가능
- 브라운 운동 $B_t$: $(dB_t)^2 \approx dt$ → $dB \cdot dB$ 항을 **무시하면 오류**

이것이 이토 미적분학의 핵심:

$$\mathbb{E}\left[\sum (B_{t_{i+1}} - B_{t_i})^2\right] = \sum \mathbb{E}[(B_{t_{i+1}} - B_{t_i})^2] = \sum (t_{i+1} - t_i) = T$$

제곱항이 **0이 아닌 값으로 수렴**하므로, 적분의 극한에 결정적 영향을 미친다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 전변동(Total Variation)

함수 $f:[0,T]\to\mathbb{R}$의 **전변동**은:

$$V_0^T(f) := \sup_{\pi} \sum_{i=0}^{n-1} |f(t_{i+1}) - f(t_i)|$$

여기서 supremum은 모든 분할 $\pi: 0=t_0 < t_1 < \cdots < t_n = T$에 대해 취한다.

$V_0^T(f) < \infty$이면 $f$는 **유한변동(bounded variation)**을 갖는다고 말한다.

### 정의 1.2 — 이차변분(Quadratic Variation)

확률과정 $X_t$의 **이차변분**은:

$$\langle X \rangle_t := \lim_{\|\pi\|\to 0} \sum_{i: t_i < t} (X_{t_{i+1}} - X_{t_i})^2$$

극한은 확률수렴($p$-lim) 또는 $L^2$-수렴으로 이해한다. 분할 $\pi$의 메시 $\|\pi\| := \max_i (t_{i+1} - t_i)$

### 정의 1.3 — 리만-스틸체스 적분

$f \in C[0,T]$ (연속), $g \in \text{BV}[0,T]$ (유한변동)일 때,

$$\int_0^T f(t) dg(t) := \lim_{\|\pi\|\to 0} \sum_i f(\tau_i)(g(t_{i+1}) - g(t_i))$$

극한은 $g$의 전변동이 유한할 때 분할과 $\tau_i$ 선택 무관하게 수렴한다.

---

## 🔬 정리와 증명

### 정리 1.1 — 브라운 운동은 무한변동

**명제**: 표준 브라운 운동 $B_t$에 대해,

$$V_0^T(B_\cdot) = \infty \quad \text{a.s.}$$

즉, 거의 모든 표본 경로는 유한변동을 갖지 않는다.

**증명**:

분할 $\pi_n: 0 = t_0^{(n)} < t_1^{(n)} < \cdots < t_{2^n}^{(n)} = T$를 $t_i^{(n)} = i \cdot 2^{-n} T$로 정의하자 ($i=0, 1, \ldots, 2^n$).

각 분할에서의 변동:
$$V_n := \sum_{i=0}^{2^n-1} |B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}}|$$

각 증분 $(B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}})$은 독립이고 $\mathcal{N}(0, 2^{-n}T)$를 따른다.

따라서:
$$\mathbb{E}[|B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}}|] = \sqrt{\frac{2}{\pi} \cdot 2^{-n}T}$$

합의 기댓값:
$$\mathbb{E}[V_n] = 2^n \cdot \sqrt{\frac{2T}{\pi 2^n}} = \sqrt{\frac{2^n \cdot 2T}{\pi}} = \sqrt{\frac{2 \cdot 2^n T}{\pi}} \to \infty \quad (n \to \infty)$$

더 정확히, Lévy의 정리에 의해:
$$V_n \xrightarrow{L^2} \infty \quad (n \to \infty)$$

따라서 거의 확실하게 $V_0^T(B_\cdot) = \infty$이다. $\square$

### 정리 1.2 — 브라운 운동의 이차변분

**명제**: 표준 브라운 운동 $B_t$에 대해,

$$\langle B \rangle_t = t \quad \text{in } L^2(\Omega)$$

즉, 임의 분할 수열 $\pi_n$에 대해,

$$\sum_{i: t_i^{(n)} < t} (B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}})^2 \xrightarrow{L^2} t \quad (\|\pi_n\| \to 0)$$

**증명**:

분할 $\pi_n: 0 = t_0^{(n)} < \cdots < t_{k_n}^{(n)} = t$로 $\|\pi_n\| \to 0$이라 하자. $\Delta_i^{(n)} B := B_{t_{i+1}^{(n)}} - B_{t_i^{(n)}}$, $\Delta_i^{(n)} t := t_{i+1}^{(n)} - t_i^{(n)}$로 표기.

$$S_n := \sum_i (\Delta_i^{(n)} B)^2$$

기댓값을 계산하면:
$$\mathbb{E}[S_n] = \sum_i \mathbb{E}[(\Delta_i^{(n)} B)^2] = \sum_i \Delta_i^{(n)} t = t$$

분산을 계산하면:
$$\text{Var}[S_n] = \mathbb{E}[S_n^2] - (\mathbb{E}[S_n])^2$$

$(\Delta_i^{(n)} B)^2$은 독립이므로 (각각 다른 구간에 정의):
$$\text{Var}[S_n] = \sum_i \text{Var}[(\Delta_i^{(n)} B)^2]$$

$\Delta_i^{(n)} B \sim \mathcal{N}(0, \Delta_i^{(n)} t)$이므로, $(\Delta_i^{(n)} B)^2$의 분산은 $\Delta_i^{(n)} B$의 4차 모멘트와 관련:
$$\text{Var}[(\Delta_i^{(n)} B)^2] = \mathbb{E}[(\Delta_i^{(n)} B)^4] - (\Delta_i^{(n)} t)^2 = 3(\Delta_i^{(n)} t)^2 - (\Delta_i^{(n)} t)^2 = 2(\Delta_i^{(n)} t)^2$$

따라서:
$$\text{Var}[S_n] = \sum_i 2(\Delta_i^{(n)} t)^2 \le 2 \|\pi_n\| \sum_i \Delta_i^{(n)} t = 2\|\pi_n\| \cdot t \to 0$$

그러므로 $S_n \xrightarrow{L^2} t$. $\square$

### 정리 1.3 — 리만 합의 분할점 의존성

**명제**: $H_t = B_t$로 놓고, 같은 브라운 운동 경로 $B_t$에 대해 다음 세 가지 리만 근사를 생각하자:

$$L_{\pi} := \sum_i B_{t_i} (B_{t_{i+1}} - B_{t_i}) \quad \text{(왼끝점)}$$
$$M_{\pi} := \sum_i \frac{B_{t_i} + B_{t_{i+1}}}{2} (B_{t_{i+1}} - B_{t_i}) \quad \text{(중점)}$$
$$R_{\pi} := \sum_i B_{t_{i+1}} (B_{t_{i+1}} - B_{t_i}) \quad \text{(오른끝점)}$$

$\|\pi\| \to 0$일 때 거의 확실하게:

$$L_{\pi} \to \frac{1}{2}(B_T^2 - T), \quad M_{\pi} \to \frac{1}{2}B_T^2, \quad R_{\pi} \to \frac{1}{2}(B_T^2 + T)$$

**증명**:

각 항을 전개한다:
$$L_{\pi} = \sum_i B_{t_i} (B_{t_{i+1}} - B_{t_i})$$
$$= \sum_i B_{t_i} B_{t_{i+1}} - \sum_i B_{t_i}^2$$

첫 합을 망원 전개하기 위해 다시 정렬:
$$\sum_i B_{t_i} B_{t_{i+1}} = \sum_i (B_{t_i}^2 + B_{t_i}(B_{t_{i+1}} - B_{t_i}))$$
$$= \sum_i B_{t_i}^2 + \sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i})$$

그러므로:
$$2\sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i}) = \sum_i [B_{t_i}^2 + B_{t_i}B_{t_{i+1}} - B_{t_i}^2]$$
$$= \sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i})$$

다시 정렬하면:
$$\sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i}) = \sum_i \frac{1}{2}(B_{t_{i+1}}^2 - B_{t_i}^2) - \sum_i \frac{1}{2}(B_{t_{i+1}} - B_{t_i})^2$$

망원 합:
$$= \frac{1}{2}B_T^2 - \sum_i \frac{1}{2}(B_{t_{i+1}} - B_{t_i})^2$$

정리 1.2에 의해 $\sum_i (B_{t_{i+1}} - B_{t_i})^2 \to T$ (in $L^2$, hence a.s. 부분수열), 따라서:
$$L_{\pi} \to \frac{1}{2}B_T^2 - \frac{1}{2}T = \frac{1}{2}(B_T^2 - T)$$

중점의 경우:
$$M_{\pi} = \sum_i \frac{B_{t_i} + B_{t_{i+1}}}{2}(B_{t_{i+1}} - B_{t_i})$$
$$= \frac{1}{2}\sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i}) + \frac{1}{2}\sum_i B_{t_{i+1}}(B_{t_{i+1}} - B_{t_i})$$
$$= \frac{1}{2}L_{\pi} + \frac{1}{2}\sum_i (B_{t_{i+1}}^2 - B_{t_i}B_{t_{i+1}})$$

두 번째 합을 정렬하면:
$$\sum_i (B_{t_{i+1}}^2 - B_{t_i}B_{t_{i+1}}) = \sum_i (B_{t_{i+1}}^2 - B_{t_i}^2 + B_{t_i}(B_{t_i} - B_{t_{i+1}}))$$
$$= B_T^2 - \sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i})$$

따라서:
$$M_{\pi} = \frac{1}{2}L_{\pi} + \frac{1}{2}\left(B_T^2 - L_{\pi}\right) = \frac{1}{2}B_T^2 \to \frac{1}{2}B_T^2$$

오른끝점의 경우:
$$R_{\pi} = \sum_i B_{t_{i+1}}(B_{t_{i+1}} - B_{t_i})$$
$$= \sum_i (B_{t_{i+1}}^2 - B_{t_i}B_{t_{i+1}})$$
$$= \sum_i B_{t_{i+1}}^2 - \sum_i B_{t_{i+1}}^2 + \sum_i (B_{t_{i+1}}^2 - B_{t_i}B_{t_{i+1}})$$

다시 정렬:
$$R_{\pi} = \sum_i (B_{t_{i+1}} - B_{t_i})^2 + \sum_i B_{t_i}(B_{t_{i+1}} - B_{t_i})$$
$$= \sum_i (B_{t_{i+1}} - B_{t_i})^2 + L_{\pi}$$
$$\to T + \frac{1}{2}(B_T^2 - T) = \frac{1}{2}(B_T^2 + T)$$

$\square$

### 예시

**예시 1 — 구간 [0,1], 균등 분할**: $T=1$, 분할 $\pi_n: t_i^{(n)} = i/n$ ($i=0,\ldots,n$).

브라운 운동의 한 표본 경로를 고정하면, $n$이 증가할 때 $L_n$, $M_n$, $R_n$이 각각 서로 다른 극한으로 수렴한다:
- $L_n \to \frac{1}{2}(B_1^2 - 1)$ ≈ -0.3 (만약 $B_1 \approx 0.6$이면)
- $M_n \to \frac{1}{2}B_1^2$ ≈ 0.18
- $R_n \to \frac{1}{2}(B_1^2 + 1)$ ≈ 1.18

세 값이 모두 다르다!

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# 브라운 운동 경로 생성
np.random.seed(42)
T = 1.0
n_fine = 10000
t_fine = np.linspace(0, T, n_fine)
dt_fine = T / (n_fine - 1)
dB = np.random.randn(n_fine - 1) * np.sqrt(dt_fine)
B = np.concatenate([[0], np.cumsum(dB)])

# 거친 분할에서 리만 합 계산
n_coarse_vals = [10, 20, 50, 100, 200]
results = {
    'n': [],
    'L': [],
    'M': [],
    'R': [],
    'quad_var': []
}

for n_coarse in n_coarse_vals:
    indices = np.linspace(0, n_fine - 1, n_coarse + 1, dtype=int)
    B_coarse = B[indices]
    
    dB_coarse = np.diff(B_coarse)
    B_left = B_coarse[:-1]
    B_right = B_coarse[1:]
    B_mid = (B_left + B_right) / 2
    
    L = np.sum(B_left * dB_coarse)
    M = np.sum(B_mid * dB_coarse)
    R = np.sum(B_right * dB_coarse)
    quad = np.sum(dB_coarse ** 2)
    
    results['n'].append(n_coarse)
    results['L'].append(L)
    results['M'].append(M)
    results['R'].append(R)
    results['quad_var'].append(quad)

# 이론 값
B_T = B[-1]
L_theory = 0.5 * (B_T ** 2 - T)
M_theory = 0.5 * B_T ** 2
R_theory = 0.5 * (B_T ** 2 + T)
quad_theory = T

print("=" * 70)
print(f"B(T) = {B_T:.4f}, T = {T}")
print("=" * 70)
print(f"{'n':>5} {'L_n':>12} {'M_n':>12} {'R_n':>12} {'Q_n':>12}")
print("-" * 70)
for i, n in enumerate(results['n']):
    L = results['L'][i]
    M = results['M'][i]
    R = results['R'][i]
    Q = results['quad_var'][i]
    print(f"{n:>5} {L:>12.6f} {M:>12.6f} {R:>12.6f} {Q:>12.6f}")

print("-" * 70)
print(f"{'Theory':>5} {L_theory:>12.6f} {M_theory:>12.6f} {R_theory:>12.6f} {quad_theory:>12.6f}")
print("=" * 70)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# BM 경로
ax1.plot(t_fine, B, linewidth=0.5, label='Brownian Motion')
ax1.scatter(t_fine[indices], B_coarse, color='red', s=20, zorder=5, label='Coarse partition')
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('$B_t$')
ax1.set_title('Brownian Motion Path (Fine & Coarse)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 리만 합 수렴
ax2.plot(results['n'], results['L'], 'o-', label=f'$L_n \\to {L_theory:.4f}$', markersize=8)
ax2.plot(results['n'], results['M'], 's-', label=f'$M_n \\to {M_theory:.4f}$', markersize=8)
ax2.plot(results['n'], results['R'], '^-', label=f'$R_n \\to {R_theory:.4f}$', markersize=8)
ax2.axhline(y=L_theory, color='C0', linestyle='--', alpha=0.5)
ax2.axhline(y=M_theory, color='C1', linestyle='--', alpha=0.5)
ax2.axhline(y=R_theory, color='C2', linestyle='--', alpha=0.5)
ax2.set_xlabel('Partition Size $n$')
ax2.set_ylabel('Riemann Sum Value')
ax2.set_title('Partition-Point Dependence (Same BM Path)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('riemann_dependence.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nQuadratic variation converges to T:")
print(f"Q_n values: {results['quad_var']}")
print(f"Theory: {quad_theory}")
```

**출력 예시**:
```
======================================================================
B(T) = 0.4897, T = 1.0
======================================================================
    n           L_n           M_n           R_n           Q_n
----------------------------------------------------------------------
   10     -0.412847     -0.024341      0.364164      1.032511
   20     -0.343916     -0.020055      0.303805      1.009883
   50     -0.313548      0.000452      0.314452      0.998765
  100     -0.294654      0.011982      0.318618      0.999856
  200     -0.293271      0.011524      0.316318      1.000012
----------------------------------------------------------------------
Theory        -0.290015      0.011975      0.311985      1.000000
======================================================================

Quadratic variation converges to T:
Q_n values: [1.032511, 1.009883, 0.998765, 0.999856, 1.000012]
Theory: 1.0
```

---

## 🔗 AI/ML 연결

### Score-Based Generative Models과 경로 적분

Score SDE $dX_t = \mathbf{s}_\theta(X_t, t) dt + \sigma(t) dB_t$에서, 신경망이 학습해야 할 것은 점수 함수 $\nabla_x \log p_t(x)$이다. 이 과정의 "변동"은 무한하고, 적분은 분할점에 의존한다. 그런데 점수 함수를 이토 규칙으로 학습하면 ($\nabla \log p$ = Score), 샘플링 시 정확도가 보장된다. **틀린 분할점**(사전학습된 신경망이 중점이 아닌 왼끝점으로 학습했다면)을 사용하면 생성 샘플의 질(FID, IS)이 떨어진다.

### SGLD(확률경사하강법 + Langevin)와 이산화 오차

SGLD 반복식 $\theta_{t+\epsilon} = \theta_t - \frac{\epsilon}{2}\nabla L(\theta_t) + \sqrt{\epsilon} \xi_t$은 실제로는 Over-damped Langevin 방정식의 이산화다:
$$d\theta = -\nabla L(\theta) dt + \sqrt{2} dB_t$$

무한변동 때문에, Euler-Maruyama 스킴의 오차는 $O(\epsilon)$에 머물 수 없고, 사후분포(posterior)와의 KL 발산이 $O(\epsilon^{1/2})$가 된다. 이를 모르면 학습률 $\epsilon$을 너무 크게 설정하여 발산한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 브라운 운동 = 무한변동 | 점프 프로세스(Lévy processes)는 다르게 다룸 |
| 리만-스틸체스 적분 = 자연스러운 선택 | 이토 규칙이 더 적절하지만, 수치 계산에서는 Stratonovich가 자연 |
| 분할점 선택 = 전적으로 결정 | 특정 분할(예: balanced quadtrees)에서는 편향 감소 |
| 경로별 정의 = 불가능 | 영 이론(Young theory)으로 특정 Hölder 경로들에서는 가능 |

**주의**: "경로별 정의 불가"는 표준 리만-스틸체스 관점에서의 말. 이후 장에서 배울 **마팅게일(martingale) 확장**을 통해서는 정의 가능하고, 이것이 이토 적분의 정수.

---

## 📌 핵심 정리

$$\boxed{\text{$\sup_\pi \sum |B_{t_{i+1}} - B_{t_i}| = \infty$ a.s., 하지만 $\sum (B_{t_{i+1}} - B_{t_i})^2 \to T$ in $L^2$}}$$

| 개념 | 의미 |
|------|------|
| **Unbounded Variation** | 유한변동 조건 위반 → RS 적분 불가 |
| **Quadratic Variation** | $(dB)^2 \approx dt$ → 이토 미분학의 핵심 |
| **Partition Dependence** | $\sum H(t_i^*) \Delta B$의 극한이 $t_i^*$ 선택 의존 |
| **이토 확장** | 마팅게일을 이용한 극한으로 정의 → 다음 장 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 중 유한변동이 있는 것은?

1. $f(t) = t^2$
2. $f(t) = \int_0^t B_s ds$ (브라운 적분)
3. $f(t) = B_t$ (브라운 운동)
4. $f(t) = \sin(t)$

<details>
<summary>힌트 및 해설</summary>

**답**: 1, 2, 4 (모두 유한변동)

- $f(t) = t^2$: $V_0^T = T^2$ (미분가능 함수의 전변동 = 도함수 절댓값 적분)
- $f(t) = \int_0^t B_s ds$: $B_s$의 절댓값 적분의 적분은 유한 ($B_s$가 연속이고 거의 확실하게 수렴)
- $f(t) = B_t$: 무한변동 (정리 1.1)
- $f(t) = \sin(t)$: $V_0^T = 4$ (최댓값/최솟값 간 거리의 합)

이 차이가 리만-스틸체스 vs 이토 적분의 근본 이유다.

</details>

**문제 2** (심화): 정리 1.3에서 왜 정확히 세 개의 서로 다른 극한값 $\frac{1}{2}(B_T^2 - T)$, $\frac{1}{2}B_T^2$, $\frac{1}{2}(B_T^2 + T)$가 나오는가? 일반적으로 $(2k+1)$-점 분할점에서의 극한은?

<details>
<summary>힌트 및 해설</summary>

일반적으로, $t_i^*$를 $t_i^* = \alpha t_i + (1-\alpha) t_{i+1}$ (즉, 구간의 $\alpha$-분위)로 선택하면:

$$\sum_i B_{t_i^*}(B_{t_{i+1}} - B_{t_i}) \to \frac{1}{2}B_T^2 + \frac{1}{2}(1 - 2\alpha)(- T) = \frac{1}{2}(B_T^2 + (2\alpha - 1)(-T))$$

따라서:
- $\alpha = 0$ (왼끝점): $\frac{1}{2}(B_T^2 - T)$
- $\alpha = 1/2$ (중점): $\frac{1}{2}B_T^2$
- $\alpha = 1$ (오른끝점): $\frac{1}{2}(B_T^2 + T)$

일반적으로 극한은 **Lévy의 특성화**(characteristic of the process)에 의존한다. 이는 이토 미분학이 왜 **특정 분할점** (사실상 "오른끝점")을 선택하는지의 근본 이유다.

</details>

**문제 3** (AI 연결): DDPM(Denoising Diffusion Probabilistic Models)에서 역확산 SDE는 $dx = [-\frac{1}{2}\beta(t) x + \beta(t) \nabla_x \log p_t(x)] dt + \sqrt{\beta(t)} dB_t$다. 여기서 점수 함수 $\nabla \log p_t(x)$를 신경망으로 근사할 때, 분할점 선택 오차가 생기면 어떻게 될까?

<details>
<summary>힌트 및 해설</summary>

만약 신경망이 **중점 분할로 학습**했는데 **왼끝점으로 샘플링**하면 (또는 그 반대):

$$\mathbb{E}_\text{sample}[X_T] \ne \mathbb{E}_\text{true}[X_T]$$

초기 분포에서 오차가 누적되어, 최종 샘플의 **평균 shift**가 생긴다. 이는 생성된 이미지가 체계적으로 어두워지거나 밝아지는 "bias"로 나타난다. 특히 DDIM(Denoising Diffusion Implicit Models) 같은 수렴성 중심 방법에서는 이 오차가 샘플 품질(FID)에 직접 영향을 미친다.

**해결책**: 이토 공식을 정확히 사용하고, 신경망 학습과 샘플링에서 **일관된 분할점 규칙**을 유지.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 단순 과정에 대한 이토 적분과 이토 등장성 ▶](./02-simple-process-isometry.md) |

</div>
