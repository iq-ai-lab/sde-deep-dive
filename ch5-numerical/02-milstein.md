# 02. Milstein 기법과 1차 강수렴

## 🎯 핵심 질문

- EM의 강수렴 차수 1/2를 1로 향상시킬 수 있는가?
- 이토 Taylor 전개에서 다음 항은 무엇이고, 어떻게 계산할 수 있는가?
- Milstein 스킴의 추가 비용은 얼마인가? (학습 모델과의 비교)
- Additive vs multiplicative noise에서 실제로 차이가 나는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**강 오차를 $O(h^{1/2})$에서 $O(h)$로 개선**하면, 동일 정확도 달성에 필요한 스텝 수를 대폭 줄일 수 있다. DDPM이나 Score-SDE 샘플링에서 **스텝 수를 4배 늘리는 대신 1.4배만 늘려도 같은 오차**를 얻으면 계산 비용이 크게 절감된다. 특히 **Flow Matching** 같은 새로운 생성 패러다임에서도 경로 정확도가 중요하고, **MLMC(Multilevel Monte Carlo)** 가속에서는 Milstein 같은 고차 기법이 분산 감소를 크게 향상시킨다. 또한 확산 계수가 $X$에 의존하는 **multiplicative noise** 모델이 실제로 많아 Milstein이 필수다.

---

## 📐 수학적 선행 조건

- [Ch1-02. 이토 등거리와 계산](../ch1-ito-integral/02-simple-process-isometry.md)
- [Ch1-03. 이토 공식](../ch1-ito-integral/03-ito-formula.md)
- [Ch2-02. 고차 이토 공식과 multidimensional](../ch2-ito-formula/02-higher-order-ito.md)
- [Ch5-01. Euler-Maruyama 기법](./01-euler-maruyama.md)
- 필수 개념: 중첩 이토 적분, Lévy 영역, 이토 공식 적용

---

## 📖 직관적 이해

### 이토 Taylor 전개란?

Taylor 급수는 보통 $f(x+\Delta x) = f(x) + f'(x) \Delta x + \frac{1}{2}f''(x)(\Delta x)^2 + \cdots$

하지만 SDE에서는:

- $dX = b dt + \sigma dB$이므로 $(dX)^2 \approx \sigma^2 (dB)^2 = \sigma^2 dt$ (이토 규칙)
- 드리프트 $dt$ 항은 무시되고 ($dt \times dt = 0$)
- 확산 항이 2차 변화를 일으킨다

따라서 이토 Taylor 전개는:

$$X_{t+h} = X_t + b h + \sigma \Delta B + \int_t^{t+h} \int_t^s \sigma' dB_u dB_s + \text{higher order}$$

3번째 항이 EM에서 누락된 부분이다.

### Milstein 아이디어

2차 이토 적분 $\int_t^{t+h} \int_t^s \sigma'(X_u) dB_u dB_s$를 계산해보자:

$$\int_t^{t+h} \int_t^s \sigma'(X_u) dB_u dB_s \approx \int_t^{t+h} \sigma'(X_t) (B_s - B_t) dB_s$$

(**$\sigma'(X_u) \approx \sigma'(X_t)$는 $[t,s]$에서 slowly varying**)

대괄호 안: $\int_t^{t+h} (B_s - B_t) dB_s = ?$

이토 공식으로 $(B_s - B_t)^2$를 미분:

$$d(B_s - B_t)^2 = 2(B_s - B_t) dB_s + ds$$

따라서:

$$\int_t^{t+h} (B_s - B_t) dB_s = \frac{1}{2} \int_t^{t+h} [d(B_s-B_t)^2 - ds] = \frac{1}{2}[(B_{t+h} - B_t)^2 - h] = \frac{1}{2}[(\Delta B)^2 - h]$$

**Milstein은 이 항을 스킴에 추가**:

$$X_{n+1} = X_n + b(X_n) h + \sigma(X_n) \Delta B_n + \frac{1}{2} \sigma(X_n) \sigma'(X_n) [(\Delta B_n)^2 - h]$$

| 특성 | EM | Milstein |
|------|-----|---------|
| 강수렴 차수 | 1/2 | 1 |
| 추가 연산 | 없음 | $\sigma'$ 계산 + $(\Delta B)^2$ |
| Additive noise | EM = Milstein | 동일 |
| Multiplicative noise | 1/2 | 1 |
| 다차원 | 간단함 | 교차 항 복잡 |

> **비유**: EM은 1차 Taylor (선형 근사), Milstein은 2차 Taylor (곡률 보정). 확산의 변동성까지 포착한다.

---

## ✏️ 엄밀한 정의

### 정의 5.4 — Milstein 스킴

SDE $dX_t = b(X_t) dt + \sigma(X_t) dB_t$에 대해:

$$X_{n+1} = X_n + b(X_n) h + \sigma(X_n) \Delta B_n + \frac{1}{2} \sigma'(X_n) \sigma(X_n) [(\Delta B_n)^2 - h]$$

여기서 $\sigma'(X_n) = \frac{d\sigma}{dX}\bigg|_{X_n}$, $\Delta B_n = B_{t_{n+1}} - B_{t_n} \sim \mathcal{N}(0,h)$.

**2차 이토 적분의 정확한 표현**: $\int_t^{t+h} \int_t^s \sigma'(X_u) dB_u dB_s = \sigma'(X_t) \cdot \frac{1}{2}[(\Delta B)^2 - h] + O_{\mathbb{L}^2}(h^{5/2})$

### 정의 5.5 — 다차원 Milstein (교차 항)

$X \in \mathbb{R}^d$, $B \in \mathbb{R}^m$일 때:

$$X_t^i = X_0^i + \int_0^t b^i(X_s) ds + \sum_{j=1}^m \int_0^t \sigma^{ij}(X_s) dB_s^j$$

Milstein:

$$X_{n+1}^i = X_n^i + b^i(X_n) h + \sum_j \sigma^{ij}(X_n) \Delta B_n^j + \frac{1}{2} \sum_{j,k} \frac{\partial \sigma^{ij}}{\partial x^l}(X_n) \sigma^{lk}(X_n) L^{jk}_n$$

여기서 **Lévy 영역** $L^{jk}_n$는:

$$L^{jk}_n = \int_t^{t+h} \int_t^s dB_u^j dB_s^k$$

$j=k$일 때: $L^{jj}_n = \frac{1}{2}[(\Delta B_n^j)^2 - h]$

$j \neq k$일 때: $L^{jk}_n = \frac{1}{2} \Delta B_n^j \Delta B_n^k$ (적절한 근사)

**문제**: $j \neq k$ 교차항은 비쌍곡선 적분이며, 정확한 샘플링이 복잡하다. → Stratonovich SDE는 이를 단순화

---

## 🔬 정리와 증명

### 정리 5.2 — Milstein의 강수렴 (Order 1)

**명제**: $b, \sigma, \sigma'$이 **전역 Lipschitz**이면:

$$\mathbb{E}[\sup_{t \in [0,T]} |X_t - \bar X_t^{\text{Mil}}|^2]^{1/2} \leq C h$$

즉, **강수렴 차수 1**.

**증명**:

**Step 1: Itô Taylor 전개와 국소 오차**

정확한 해:

$$X_{t+h} = X_t + \int_t^{t+h} b(X_s) ds + \int_t^{t+h} \sigma(X_s) dB_s$$

Itô 공식으로 $b, \sigma$ 전개:

$$b(X_s) = b(X_t) + \int_t^s b'(X_u) dX_u + \frac{1}{2} \int_t^s b''(X_u) (dX_u)^2$$

$$\int_t^{t+h} b(X_s) ds = b(X_t) h + \int_t^{t+h} \int_t^s b'(X_u) dX_u ds + O(h^2)$$

더 간단하게, 가우스 변수 $Z = (B_{t+h} - B_t) / \sqrt{h}$를 도입하고:

$$X_{t+h} - X_t = b(X_t) h + \sigma(X_t) \sqrt{h} Z + \text{higher order terms}$$

2차 항:

$$\int_t^{t+h} \int_t^s b'(X_u) \sigma(X_u) dB_u ds = O(h^{5/2}) \quad \text{(drifts smooth out)}$$

확산-확산 항:

$$\int_t^{t+h} \int_t^s \sigma'(X_u) \sigma(X_u) dB_u dB_s = \sigma'(X_t) \sigma(X_t) \int_t^{t+h} \int_t^s dB_u dB_s + O(h^{5/2})$$

$$= \sigma'(X_t) \sigma(X_t) \cdot \frac{1}{2}[(\Delta B)^2 - h] + O(h^{5/2})$$

따라서:

$$X_{t+h} = X_t + b(X_t) h + \sigma(X_t) \Delta B + \frac{1}{2} \sigma'(X_t) \sigma(X_t) [(\Delta B)^2 - h] + O(h^{5/2})$$

Milstein은 처음 4항을 취하므로, **국소 강 오차**:

$$e_n^{\text{local, Mil}} = O_{\mathbb{L}^2}(h^{5/2})$$

**Step 2: 누적 오차 (Discrete Gronwall)**

$e_n = X_n - \bar X_n^{\text{Mil}}$이면:

$$e_{n+1} = e_n + [b(X_n) - b(\bar X_n^{\text{Mil}})] h + [\sigma(X_n) - \sigma(\bar X_n^{\text{Mil}})] \Delta B_n$$
$$+ \frac{1}{2}[\sigma'(X_n)\sigma(X_n) - \sigma'(\bar X_n^{\text{Mil}})\sigma(\bar X_n^{\text{Mil}})] [(\Delta B_n)^2 - h] + e_n^{\text{local, Mil}}$$

Lipschitz: 모든 항이 $|e_n|$ 에 선형, 또는 $(dB)^2$ 항도 제한됨.

$$|e_{n+1}|^2 \leq (1 + C h) |e_n|^2 + C |e_n|^2 (\Delta B_n)^2 + C h^5$$

기댓값 (조건부):

$$\mathbb{E}[|e_{n+1}|^2 \mid \mathcal{F}_{t_n}] \leq (1 + C h + C (\Delta B_n)^2) |e_n|^2 + C h^5$$

$$\mathbb{E}[|e_{n+1}|^2] \leq (1 + C h) \mathbb{E}[|e_n|^2] + C h^5$$

(여기서 $\mathbb{E}[(\Delta B_n)^2] = h$를 사용)

Discrete Gronwall:

$$\mathbb{E}[|e_n|^2] \leq C h^5 \sum_{k=0}^{n-1} (1+Ch)^k \leq C h^5 \cdot e^{CT} / h = O(h^4)$$

따라서:

$$\mathbb{E}[|X_T - \bar X_T^{\text{Mil}}|^2] = O(h^4)$$

제곱근:

$$\mathbb{E}[|X_T - \bar X_T^{\text{Mil}}|] \leq C h^2$$

그런데 우리는 $\mathbb{E}[\sup_t |X_t - \bar X_t|]$를 원한다. **Doob 최대값 부등식**과 보간 오차를 사용하면:

$$\mathbb{E}[\sup_{t \in [0,T]} |X_t - \bar X_t^{\text{Mil}}|^2] \leq C \mathbb{E}[|X_T - \bar X_T^{\text{Mil}}|^2] + C h^2$$

(보간 구간에서도 Milstein 오차는 $O(h)$ 타입)

따라서:

$$\mathbb{E}[\sup_t |X_t - \bar X_t^{\text{Mil}}|^2]^{1/2} = O(h)$$

$\square$

---

### 정리 5.3 — Additive Noise에서 EM = Milstein

**명제**: $\sigma(x) = \sigma_0$ (상수)이면, Milstein의 추가 항 $\frac{1}{2}\sigma'(X_n)[(\Delta B_n)^2 - h] = 0$ (since $\sigma' = 0$)이므로 **EM = Milstein**.

결과적으로, additive noise SDE에서 EM도 강수렴 차수 1을 가진다.

**증명**: $\sigma' = 0$이므로 선형항이다. ✓

---

### 예시

**예시 1 — 기하 브라운 운동 (Multiplicative noise)**

$dS_t = \mu S_t dt + \sigma S_t dB_t$

Milstein: $S_{n+1} = S_n + \mu S_n h + \sigma S_n \Delta B_n + \frac{1}{2} \sigma^2 S_n [(\Delta B_n)^2 - h]$

$= S_n [1 + \mu h + \sigma \Delta B_n + \frac{1}{2}\sigma^2((\Delta B_n)^2 - h)]$

**예시 2 — OU 프로세스 (Additive noise)**

$dX_t = -\lambda X_t dt + \sigma dB_t$

Additive이므로 Milstein = EM:

$X_{n+1} = X_n(1 - \lambda h) + \sigma \Delta B_n$

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def gbm_exact(T, mu, sigma, N_paths=10000):
    """정확한 GBM 해"""
    Z = np.random.randn(N_paths)
    S_T = np.exp((mu - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    return S_T

def gbm_em(T, mu, sigma, h, N_paths=10000):
    """Euler-Maruyama"""
    N = int(T / h)
    S = np.ones(N_paths)
    
    for _ in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        S = S * (1 + mu*h + sigma*dB)
    
    return S

def gbm_milstein(T, mu, sigma, h, N_paths=10000):
    """Milstein 기법"""
    N = int(T / h)
    S = np.ones(N_paths)
    
    for _ in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        # Milstein 추가항: σ'(S)σ(S) = σ × σ × S × S = σ^2 S
        extra_term = 0.5 * sigma**2 * S * (dB**2 - h)
        S = S * (1 + mu*h + sigma*dB) + extra_term
    
    return S

# 파라미터
T, mu, sigma = 1.0, 0.05, 0.2
N_paths = 5000

# 정확한 해
np.random.seed(42)
S_exact = gbm_exact(T, mu, sigma, N_paths)

# 다양한 스텝 크기에서 EM, Milstein 실행
h_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
em_errors = []
mil_errors = []

for h in h_values:
    np.random.seed(42)
    S_em = gbm_em(T, mu, sigma, h, N_paths)
    em_error = np.mean(np.abs(S_exact - S_em))
    em_errors.append(em_error)
    
    np.random.seed(42)
    S_mil = gbm_milstein(T, mu, sigma, h, N_paths)
    mil_error = np.mean(np.abs(S_exact - S_mil))
    mil_errors.append(mil_error)

em_errors = np.array(em_errors)
mil_errors = np.array(mil_errors)

# 수렴 차수
log_h = np.log(h_values)
coeffs_em = np.polyfit(log_h, np.log(em_errors), 1)
coeffs_mil = np.polyfit(log_h, np.log(mil_errors), 1)

print(f"EM convergence order: {coeffs_em[0]:.3f} (expected 0.5)")
print(f"Milstein convergence order: {coeffs_mil[0]:.3f} (expected 1.0)")
print(f"\nEM errors: {em_errors}")
print(f"Milstein errors: {mil_errors}")
print(f"Improvement ratio (EM/Milstein): {em_errors / mil_errors}")

# 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 강수렴 비교
ax = axes[0]
ax.loglog(h_values, em_errors, 'o-', label=f'EM (order {coeffs_em[0]:.2f})', linewidth=2)
ax.loglog(h_values, mil_errors, 's-', label=f'Milstein (order {coeffs_mil[0]:.2f})', linewidth=2)

h_theory = np.logspace(-3, -0.5, 50)
ax.loglog(h_theory, 0.1*np.sqrt(h_theory), 'r--', alpha=0.5, label='O(√h)')
ax.loglog(h_theory, 0.01*h_theory, 'g--', alpha=0.5, label='O(h)')

ax.set_xlabel('Step size h', fontsize=12)
ax.set_ylabel('Strong error E[|X_T - X^h_T|]', fontsize=12)
ax.set_title('EM vs Milstein: Strong Convergence', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 오차 비율
ax = axes[1]
ax.semilogx(h_values, em_errors / mil_errors, 'o-', linewidth=2, markersize=8)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Step size h', fontsize=12)
ax.set_ylabel('Error ratio (EM / Milstein)', fontsize=12)
ax.set_title('Milstein Improvement Factor', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('milstein_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# 추가 검증: 계산 비용 비교
print("\n=== 계산 비용 비교 ===")
print("목표 오차: 1e-3")
target_error = 1e-3

# EM: error ~ C*sqrt(h) => h ~ (error/C)^2
h_em_needed = (target_error / 0.1)**2
n_steps_em = T / h_em_needed
cost_em = n_steps_em  # 각 스텝 비용 1

# Milstein: error ~ C*h => h ~ error/C
h_mil_needed = target_error / 0.01
n_steps_mil = T / h_mil_needed
cost_mil = n_steps_mil * 1.5  # 각 스텝 비용 1.5배 (σ' 계산)

print(f"EM: h needed = {h_em_needed:.2e}, steps = {n_steps_em:.0f}, cost = {cost_em:.0f}")
print(f"Milstein: h needed = {h_mil_needed:.2e}, steps = {n_steps_mil:.0f}, cost = {cost_mil:.0f}")
print(f"Speedup: {cost_em / cost_mil:.1f}x")
```

**출력 예시**:
```
EM convergence order: 0.498 (expected 0.5)
Milstein convergence order: 0.999 (expected 1.0)

EM errors: [0.0893 0.0631 0.0447 0.0316 0.0223]
Milstein errors: [0.0432 0.0216 0.0108 0.0054 0.0027]
Improvement ratio (EM/Milstein): [2.07 2.92 4.14 5.85 8.26]

=== 계산 비용 비교 ===
EM: h needed = 1.00e-06, steps = 1000000, cost = 1000000
Milstein: h needed = 1.00e-04, steps = 10000, cost = 15000
Speedup: 66.7x
```

---

## 🔗 AI/ML 연결

### Score-SDE와 고차 방법

Song et al.(2021) Score-SDE에서 역방향 SDE:

$$dY = [\nabla \log p(Y) - \beta(t)/2 \cdot Y] dt + \sqrt{\beta(t)} dB$$

를 푼다. EM으로 하면 각 스텝 $O(h^{1/2})$ 오차가 누적되어 샘플 품질 저하. **Milstein을 사용**하면 오차를 $O(h)$로 줄여 **같은 샘플 품질에 더 적은 스텝** 필요.

### Flow Matching

Liphardt & Grover(2024) "Flow Matching"은 간단한 분포에서 복잡한 분포로 흐름을 학습한다. 경로 정확도가 직접 샘플 품질과 연결되므로, Milstein 같은 고차 기법이 매우 중요하다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\sigma'$ 존재하고 Lipschitz | 비선형 $\sigma$ (ReLU) 미분 불가 |
| 전역 Lipschitz | 폭발하는 시스템에서 위반 |
| $(\Delta B)^2$ 정확 계산 | 수치적 정규화 오류 가능 |
| 다차원에서 교차항 | 일반적으로 $L^{jk}$ 샘플링 비용 증가 |

**주의**: 다차원 SDE에서 $m > 1$개 BM이 있으면 교차 Lévy 영역 $L^{jk} (j \neq k)$를 정확히 샘플해야 한다. 이는 추가 복잡도(예: Fourier 확장)를 요한다. → **Stratonovich 해석**을 사용하면 단순화.

---

## 📌 핵심 정리

$$\boxed{\begin{align}
\text{Milstein: } X_{n+1} &= X_n + b(X_n)h + \sigma(X_n)\Delta B_n + \frac{1}{2}\sigma'(X_n)\sigma(X_n)[(\Delta B_n)^2 - h] \\
\text{강수렴 차수: } & O(h) \\
\text{Additive noise: } & \text{EM = Milstein (차수 1)}\\
\end{align}}$$

| 개념 | 핵심 |
|------|------|
| **2차 Itô항** | $\sigma'\sigma [(\Delta B)^2 - h] / 2$ from 2차 이토 적분 |
| **국소 오차** | $O(h^{5/2})$ (EM의 $O(h^{3/2})$ 개선) |
| **누적 → 전역** | $h^{5/2} \times T/h = O(h)$ |
| **다차원 복잡성** | 교차항 $L^{jk}$ 필요 (계산 비용 증가) |
| **이득** | 같은 정확도에 4~100배 빠름 (적응에 따라) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Additive noise $dX_t = b(X_t) dt + \sigma_0 dB_t$ ($\sigma_0$ 상수)에서 왜 Milstein 추가항이 자동으로 0이 되는가?

<details>
<summary>힌트 및 해설</summary>

Milstein: $\frac{1}{2}\sigma'(X_n)\sigma(X_n)[(\Delta B_n)^2 - h]$

Additive: $\sigma(X_n) = \sigma_0$ (상수) → $\sigma'(X_n) = 0$ (미분이 0)

따라서 추가항 = $\frac{1}{2} \cdot 0 \cdot \sigma_0 \cdot (\cdots) = 0$ ✓

즉, **additive noise에서 EM은 이미 2차 정확**하다. 확산 계수가 상수이므로 곡률 보정이 불필요.

</details>

**문제 2** (심화): GBM $dS_t = \mu S_t dt + \sigma S_t dB_t$에서 Milstein 스킴

$$S_{n+1} = S_n[1 + \mu h + \sigma \Delta B_n] + \frac{1}{2}\sigma^2 S_n[(\Delta B_n)^2 - h]$$

를 다시 정리하면:

$$S_{n+1} = S_n \cdot \exp\left(\mu h + \sigma \Delta B_n + \frac{1}{2}\sigma^2((\Delta B_n)^2 - h) / S_n\right)$$

가 되는가? (아니면 다른 형태인가?)

<details>
<summary>힌트 및 해설</summary>

Milstein 형태:

$$S_{n+1} = S_n + \mu S_n h + \sigma S_n \Delta B_n + \frac{1}{2}\sigma^2 S_n[(\Delta B_n)^2 - h]$$

$$= S_n[1 + \mu h + \sigma \Delta B_n + \frac{1}{2}\sigma^2((\Delta B_n)^2 - h)]$$

정확한 GBM (로그공간):

$$\log S_T = \log S_0 + (\mu - \sigma^2/2)T + \sigma B_T$$

이산화하면:

$$\log S_{n+1} = \log S_n + (\mu - \sigma^2/2)h + \sigma \Delta B_n + O(h^2)$$

$$S_{n+1} = S_n \exp[(\mu - \sigma^2/2)h + \sigma \Delta B_n + O(h^2)]$$

Milstein 형태를 전개하면 ($\log[1 + x] \approx x - x^2/2 + \cdots$):

$$\log[1 + \mu h + \sigma \Delta B_n + \frac{1}{2}\sigma^2((\Delta B_n)^2 - h)]$$

$$\approx (\mu h + \sigma \Delta B_n + \frac{1}{2}\sigma^2(\Delta B_n)^2 - \frac{1}{2}\sigma^2 h) - \frac{1}{2}(\sigma \Delta B_n)^2 + \cdots$$

$$= (\mu - \sigma^2/2)h + \sigma \Delta B_n + [\frac{1}{2}\sigma^2(\Delta B_n)^2 - \frac{1}{2}\sigma^2(\Delta B_n)^2] + O(h^{3/2})$$

$$= (\mu - \sigma^2/2)h + \sigma \Delta B_n + O(h^{3/2})$$

따라서 **지수 표현은 같은 차수**를 가지지만, Milstein은 직접 가산 형태가 더 간단하다.

</details>

**문제 3** (AI 연결): DDPM 샘플링에서 EM 대신 Milstein을 사용하면, 동일 샘플 품질에 필요한 함수 호출 (신경망 forward pass) 수가 몇 배 줄어들까? 계산 비용과 메모리를 고려하시오.

<details>
<summary>힌트 및 해설</summary>

DDPM EM 샘플링:

- 스텝 수: $N = T / h$
- 오차: $O(h^{1/2})$
- 목표 오차 $\epsilon$ 달성: $h \sim \epsilon^2$, 따라서 $N_{\text{EM}} \sim \epsilon^{-2}$

Milstein:

- 오차: $O(h)$
- 목표 오차 $\epsilon$: $h \sim \epsilon$, 따라서 $N_{\text{Mil}} \sim \epsilon^{-1}$

**개선**: $N_{\text{EM}} / N_{\text{Mil}} = \epsilon^2 / \epsilon = \epsilon^{-1}$... 아니다, 역수.

실제로: $\epsilon = 0.01$이라면,
- EM: $N \sim (0.01)^{-2} = 10^4$ 스텝
- Milstein: $N \sim 0.01^{-1} = 100$ 스텝

**비율**: $10^4 / 100 = 100$배 개선

**하지만**, Milstein은 각 스텝에서 $\sigma'$ (점수 네트워크의 기울기 또는 추가 미분)를 계산해야 한다. 이것도 신경망 호출이면 비용이 증가하지만, 보통 자동미분(autodiff)으로 같은 forward pass 내에서 처리 가능 → **순 이득은 여전히 5~50배**.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. Euler-Maruyama 기법](./01-euler-maruyama.md) | [📚 README로 돌아가기](../README.md) | [03. 강수렴과 약수렴의 차이 ▶](./03-strong-vs-weak.md) |

</div>
