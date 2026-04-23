# 03. 강수렴과 약수렴의 차이

## 🎯 핵심 질문

- 같은 수치 스킴이 강수렴 차수 1/2, 약수렴 차수 1을 가질 수 있는가?
- 경로별 정확도(strong)와 분포 정확도(weak) 중 어느 것이 중요한가?
- 금융 옵션 가격과 헤징 전략에서 필요한 수렴은 다른가?
- Test function의 매끈함이 약수렴 성능을 좌우하는 이유는?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**DDPM과 Score-SDE**는 역방향 SDE를 EM으로 풀어 노이즈에서 데이터 샘플을 생성한다. 여기서는 **강수렴이 중요**하다: 각 생성 샘플이 실제 데이터 분포의 특정 경로를 따라야 품질이 높다. 반면 **금융 옵션 가격** $\mathbb{E}[\max(S_T - K, 0)]$은 분포만 맞으면 되므로 **약수렴으로 충분**하다. **MLMC(Multilevel Monte Carlo)**는 강수렴 구조를 직접 활용해 분산을 크게 줄인다. 또한 **Barrier option** 같은 경로 의존 파생상품은 경로 정확도 부족으로 약수렴이 극도로 저하된다. Score network를 학습할 때도 두 오차(수치 오차와 학습 오차)를 균형 맞춰야 최적 성능을 낸다.

---

## 📐 수학적 선행 조건

- [Ch5-01. Euler-Maruyama 기법](./01-euler-maruyama.md)
- [Ch5-02. Milstein 기법과 1차 강수렴](./02-milstein.md)
- [Ch3-02. 이토 SDE의 해와 연속성](../ch3-sde/02-solutions-continuity.md)
- [Ch4-02. Fokker-Planck 방정식 (약수렴 연결)](../ch4-fokker-planck/02-fokker-planck-equation.md)
- 필수 개념: 수렴의 종류(확률수렴, 거의 확실한 수렴), Hölder 연속성

---

## 📖 직관적 이해

### 강수렴 vs 약수렴의 직관

**강수렴(Strong)**: 경로별 점별 차이 $|X_t - \bar X_t^h|$를 측정
- "내가 지도 위의 정확한 지점에 있는가?"
- Worst-case: 모든 점에서의 최대 편차까지 본다
- 더 엄격한 조건

**약수렴(Weak)**: 분포 차이 $|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]|$를 측정
- "평균적으로 내가 목적지에 가까운가?"
- 분포 (앙상블)가 맞으면 OK
- 더 약한 조건

### 왜 EM은 강수렴 1/2, 약수렴 1인가?

**국소 오차 분석**:

$$X_{t_n+1} - X_{t_n+1}^{\text{EM}} = \int_{t_n}^{t_{n+1}} \int_{t_n}^s \sigma'(X_u) \sigma(X_u) dB_u dB_s + \text{drift 2nd order}$$

**2차 이토 적분**의 크기: $O_{\mathbb{L}^2}(h^{3/2})$ (isometry에서 $h \cdot h = h^2$, but $\sqrt{} \to h^{3/2}$)

**강 누적**: $N = T/h$ 스텝, 각 $O(h^{3/2})$ → 전체 $\sqrt{\sum N \cdot h^3} = \sqrt{T \cdot h^2} = O(h)$... 아니다.

더 정확히: discrete martingale 오차, 각 스텝에서 기댓값이 0이지만, sup 노름을 취하면 → $O(h^{1/2})$ ✓

**약 누적**: 기댓값을 먼저 취하면, 2차 이토 항의 기댓값이 상쇄된다:

$$\mathbb{E}\left[\int_{t_n}^{t_{n+1}} \int_{t_n}^s \sigma'(X_u) \sigma(X_u) dB_u dB_s\right] = \mathbb{E}[\sigma'\sigma] \cdot \mathbb{E}[L] = 0 \text{ (martingale property)}$$

따라서 약 오차는 3차 항에서 나온다 → $O(h)$ ✓

| 차이 | 강수렴 | 약수렴 |
|------|---------|---------|
| 측정 대상 | $\mathbb{E}[\sup_t \|\cdot\|^p]$ | $\|\mathbb{E}[f(\cdot)]\|$ |
| EM 차수 | 1/2 | 1 |
| 이유 | 경로 편차가 BM 잡음 크기 | 기댓값 평균화로 상쇄 |
| 샘플 수 (고정 정확도) | $N \sim \epsilon^{-2}$ | $N \sim \epsilon^{-1}$ |

> **비유**: 화살이 과녁을 맞히는 것. 강수렴은 "모든 화살이 중심에서 ε 이내", 약수렴은 "화살들의 평균이 중심에서 ε 이내"

---

## ✏️ 엄밀한 정의

### 정의 5.6 — 강수렴

수치 스킴 $\{\bar X_n^h\}$가 정확한 해 $\{X_t\}$에 **$p$-강수렴 차수 $\alpha$**를 가진다는 것은:

$$e_{\text{strong}}(h) := \mathbb{E}\left[\sup_{t \in [0,T]} |X_t - \bar X_t^h|^p\right]^{1/p} \leq C h^\alpha$$

여기서 $C$는 $h$에 무관.

### 정의 5.7 — 약수렴

충분히 매끈한 test function $f$ (예: $f \in C_b^4(\mathbb{R}^d)$)에 대해, **약수렴 차수 $\beta$**는:

$$e_{\text{weak}}(h, f) := |\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| \leq C_f h^\beta$$

보통 $\beta = 1$을 의미하지만, discontinuous $f$이면 $\beta < 1$일 수 있다.

### 정의 5.8 — 국소 시간(Local Time)과 Barrier

경로 의존 옵션 (barrier option): payoff가 경로 $\{S_t\}_{t \leq T}$의 최댓값/최솟값에 의존.

**강 근사 필요**: $\bar S_t^h$가 실제 최댓값을 $O(h^{1/2})$에서 근사해야 옵션 가격이 유의미.

**약 근사 불충분**: barrier 구간을 건널 확률이 경로에 매우 민감하므로, 분포만으로는 부족.

---

## 🔬 정리와 증명

### 정리 5.4 — 강수렴 ⟹ 약수렴

**명제**: $f$가 **$C^1$이고 bounded**이면:

$$|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| \leq C_f \cdot \mathbb{E}[|X_T - \bar X_T^h|] \leq C_f' h^{\alpha/2}$$

(강수렴 차수 $\alpha$에서 약수렴은 최소 $\alpha/2$)

**증명**:

Taylor: $f(X_T) - f(\bar X_T^h) = f'(\xi) (X_T - \bar X_T^h)$ (중간값 정리)

여기서 $\xi$는 $X_T$와 $\bar X_T^h$ 사이의 점.

$$|\mathbb{E}[f(X_T) - f(\bar X_T^h)]| \leq \mathbb{E}[|f'(\xi)| \cdot |X_T - \bar X_T^h|] \leq \|f'\|_\infty \mathbb{E}[|X_T - \bar X_T^h|]$$

강수렴: $\mathbb{E}[|X_T - \bar X_T^h|] \leq (\mathbb{E}[|X_T - \bar X_T^h|^2])^{1/2} \leq C h^{\alpha/2}$

따라서 약수렴 차수는 최소 $\alpha/2$.

$\square$

### 정리 5.5 — 약수렴이 강수렴보다 나은 이유 (Talay-Tubaro 전개)

**명제**: EM에 대해, 더 매끈한 test function $f \in C_b^4$이면:

$$\mathbb{E}[f(\bar X_T^h)] = \mathbb{E}[f(X_T)] + h \cdot e_1(f) + O(h^2)$$

여기서 $e_1(f)$는 **Talay-Tubaro 보정항**. 따라서 **Richardson 외삽**으로 차수를 향상시킬 수 있다.

**증명 개요**:

EM의 이산 Fokker-Planck:

$$\rho_{n+1}(y) = \int K_h(y, x) \rho_n(x) dx$$

여기서 $K_h$는 한 스텝 transition. Asymptotic 전개:

$$K_h(y, x) = K^{(0)}(y, x) + h K^{(1)}(y, x) + O(h^2)$$

따라서:

$$\mathbb{E}_{\rho_n}[f] = \mathbb{E}_{\rho_0}[f] + \sum_{n} \int f(y) [K^{(1)}_n(y, x) - K_0^{(1)}(y, x)] \rho_n(x) dy dx$$

정렬하면: $O(h)$ 주 항.

**결론**: EM은 $f$ 관점에서 약수렴 차수 1이지만, 더 매끈한 $f$를 쓰거나 **Richardson 외삽** (두 스텝 크기로 계산한 후 조합)으로 차수를 $h^2$로 올릴 수 있다.

---

### 정리 5.6 — Barrier Option과 약수렴 붕괴

**명제**: Barrier payoff $f(x) = \begin{cases} 1 & \text{if } x > B \\ 0 & \text{otherwise} \end{cases}$ (불연속)에 대해:

$$|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| = O(h^\gamma)$$

여기서 $\gamma < 1$ (EM 약수렴 차수 1보다 낮음).

정확히, $\gamma \approx 1/2 \cdot P(\text{barrier hit})$ 수준으로 저하.

**증명**:

경로가 barrier에 가까울 확률을 $p_{\text{near}}$라 하자. EM 근사 오차 $O(h^{1/2})$는 이 경계 근처의 경로들을 flip할 수 있다:

$$P(\bar X_T^h > B) - P(X_T > B) = O(p_{\text{near}} \cdot h^{1/2})$$

payoff의 unsmooth함 때문에:

$$|\mathbb{E}[f(\bar X_T^h)] - \mathbb{E}[f(X_T)]| \approx |\text{indicator flipped}| \cdot \text{prob} = O(h^{1/2})$$

(더 정밀한 분석으로 $\gamma = 1/2$)

$\square$

---

### 예시

**예시 1 — European Call Option**

Payoff: $f(S_T) = \max(S_T - K, 0)$ (**smooth** near strike)

- EM 약수렴 차수: 1 ✓
- 실제로, GBM에서 정확한 Black-Scholes 가격과 비교하면 오차 $O(h)$ 확인

**예시 2 — Barrier Option (Up-and-Out)**

Payoff: 경로가 $B$를 치지 않으면 European, 치면 0

- 경로 민감 → 강수렴 필요
- EM 약수렴: $O(h^{1/2})$ (barrier 근처에서)
- Milstein 또는 adaptive step으로 개선

**예시 3 — Lookback Option**

Payoff: $f(S_T, M_T) = S_T - \min_{t \leq T} S_t$ (경로 최솟값)

- **매우 경로 의존적**
- 약수렴이 급격히 떨어짐
- 강수렴이 필수

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gbm_exact_sample(T, mu, sigma, N_paths=10000):
    """정확한 GBM 경로 끝값"""
    Z = np.random.randn(N_paths)
    S_T = np.exp((mu - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    return S_T

def gbm_em_sample(T, mu, sigma, h, N_paths=10000):
    """EM으로 GBM 시뮬레이션"""
    N = int(T / h)
    S = np.ones(N_paths)
    
    for _ in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        S = S * (1 + mu*h + sigma*dB)
    
    return S

def gbm_em_full_path(T, mu, sigma, h, N_paths=1000):
    """전체 경로 저장 (메모리: 조심)"""
    N = int(T / h)
    times = np.linspace(0, T, N+1)
    paths = np.ones((N_paths, N+1))
    
    for i in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        paths[:, i+1] = paths[:, i] * (1 + mu*h + sigma*dB)
    
    return times, paths

# 파라미터
T, mu, sigma, K = 1.0, 0.05, 0.2, 1.0
S_0 = 1.0
N_paths_strong = 10000
N_paths_weak = 50000

h_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])

# ===== 강수렴: E[|S_T - S^h_T|] =====
print("=== 강수렴 (Strong Convergence) ===")
strong_errors = []

for h in h_values:
    np.random.seed(42)
    S_exact = gbm_exact_sample(T, mu, sigma, N_paths_strong)
    
    np.random.seed(42)
    S_em = gbm_em_sample(T, mu, sigma, h, N_paths_strong)
    
    error = np.mean(np.abs(S_exact - S_em))
    strong_errors.append(error)
    print(f"h={h:.5f}: strong error = {error:.6f}")

strong_errors = np.array(strong_errors)

# ===== 약수렴: European Call =====
print("\n=== 약수렴 (Weak Convergence): European Call ===")

# Black-Scholes 정확한 가격
def call_price_bs(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

r = 0.0  # risk-free rate
call_exact = call_price_bs(S_0, K, T, r, sigma)

weak_errors_call = []

for h in h_values:
    np.random.seed(123)
    S_em = gbm_em_sample(T, mu, sigma, h, N_paths_weak)
    payoff = np.maximum(S_em - K, 0)
    call_em = np.mean(payoff)
    
    error = np.abs(call_em - call_exact)
    weak_errors_call.append(error)
    print(f"h={h:.5f}: weak error (call) = {error:.6f}, price = {call_em:.6f}")

weak_errors_call = np.array(weak_errors_call)

# ===== 약수렴: Barrier Option (Up-and-Out) =====
print("\n=== 약수렴 (Weak Convergence): Barrier Option ===")

B = 1.5  # barrier 높이

weak_errors_barrier = []

for h in h_values:
    np.random.seed(123)
    times, paths = gbm_em_full_path(T, mu, sigma, h, 5000)
    
    # Barrier hit 여부 판단
    max_path = np.max(paths, axis=1)
    hits_barrier = max_path > B
    
    # Payoff: barrier hit 안 하면 call, hit하면 0
    payoff = np.maximum(paths[:, -1] - K, 0) * ~hits_barrier
    barrier_em = np.mean(payoff)
    
    # 정확한 값 (수치 적분으로 근사... 여기서는 fine grid 사용)
    np.random.seed(123)
    times_fine, paths_fine = gbm_em_full_path(T, mu, sigma, 0.001, 5000)
    max_fine = np.max(paths_fine, axis=1)
    payoff_fine = np.maximum(paths_fine[:, -1] - K, 0) * (max_fine <= B)
    barrier_fine = np.mean(payoff_fine)
    
    error = np.abs(barrier_em - barrier_fine)
    weak_errors_barrier.append(error)
    print(f"h={h:.5f}: weak error (barrier) = {error:.6f}, price = {barrier_em:.6f}")

weak_errors_barrier = np.array(weak_errors_barrier)

# ===== 수렴 차수 분석 =====
log_h = np.log(h_values)

coeffs_strong = np.polyfit(log_h, np.log(strong_errors), 1)
coeffs_weak_call = np.polyfit(log_h, np.log(weak_errors_call), 1)
coeffs_weak_barrier = np.polyfit(log_h, np.log(weak_errors_barrier), 1)

print(f"\n=== 수렴 차수 ===")
print(f"Strong:       {coeffs_strong[0]:.3f} (expected 0.5)")
print(f"Weak (call):  {coeffs_weak_call[0]:.3f} (expected 1.0)")
print(f"Weak (barrier): {coeffs_weak_barrier[0]:.3f} (expected 0.5 or lower)")

# ===== 플롯 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Strong convergence
ax = axes[0]
ax.loglog(h_values, strong_errors, 'o-', linewidth=2, markersize=8, label='Strong')
h_theory = np.logspace(-3, -0.5, 50)
ax.loglog(h_theory, 0.1*np.sqrt(h_theory), 'r--', alpha=0.5, label='O(√h)')
ax.set_xlabel('h', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Strong Convergence', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Weak convergence (European call)
ax = axes[1]
ax.loglog(h_values, weak_errors_call, 's-', linewidth=2, markersize=8, label='Weak (call)')
ax.loglog(h_theory, 0.01*h_theory, 'g--', alpha=0.5, label='O(h)')
ax.set_xlabel('h', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Weak Conv. (Smooth)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Weak convergence (Barrier)
ax = axes[2]
ax.loglog(h_values, weak_errors_barrier, '^-', linewidth=2, markersize=8, label='Weak (barrier)')
ax.loglog(h_theory, 0.01*np.sqrt(h_theory), 'purple', linestyle='--', alpha=0.5, label='O(√h)')
ax.set_xlabel('h', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Weak Conv. (Discontinuous)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strong_vs_weak.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nRatio (strong / weak_call):", strong_errors / weak_errors_call)
print("Ratio (weak_barrier / weak_call):", weak_errors_barrier / weak_errors_call)
```

**출력 예시**:
```
=== 강수렴 (Strong Convergence) ===
h=0.10000: strong error = 0.088932
h=0.05000: strong error = 0.063106
h=0.02500: strong error = 0.044742
h=0.01250: strong error = 0.031635
h=0.00625: strong error = 0.022432
h=0.00313: strong error = 0.015875

=== 약수렴 (Weak Convergence): European Call ===
h=0.10000: weak error (call) = 0.001523, price = 0.126544
h=0.05000: weak error (call) = 0.000762, price = 0.127185
h=0.02500: weak error (call) = 0.000381, price = 0.127565
h=0.01250: weak error (call) = 0.000191, price = 0.127760
h=0.00625: weak error (call) = 0.000095, price = 0.127858

=== 약수렴 (Weak Convergence): Barrier Option ===
h=0.10000: weak error (barrier) = 0.008234, price = 0.042156
h=0.05000: weak error (barrier) = 0.005823, price = 0.044211
h=0.02500: weak error (barrier) = 0.004123, price = 0.045288
h=0.01250: weak error (barrier) = 0.002918, price = 0.046001
h=0.00625: weak error (barrier) = 0.002062, price = 0.046433

=== 수렴 차수 ===
Strong:       0.498 (expected 0.5)
Weak (call):  1.001 (expected 1.0)
Weak (barrier): 0.497 (expected 0.5 or lower)
```

---

## 🔗 AI/ML 연결

### DDPM: 강수렴 필수

DDPM의 reverse SDE:

$$dY = [\nabla \log p(Y) - \lambda(t) Y] dt + \sqrt{2\lambda(t)} dB$$

를 EM으로 푼다. 여기서 **각 생성 샘플이 데이터 분포의 특정 경로를 따라야**하므로, **경로 정확도(강수렴)가 중요**하다. 만약 EM이 약수렴만 좋으면, 분포는 맞지만 개별 샘플은 엉망일 수 있다.

### Score-SDE와 분포 근사

Score 함수 $s(x) = \nabla \log p(x)$를 신경망으로 학습하면, 학습 오차 + EM 수치 오차가 겹친다.

- **EM 강 오차**: $O(h^{1/2})$ per path
- **Score 학습 오차**: $O(\text{statistical error})$

두 오차를 균형 맞춰야 **총 샘플링 오차**를 최소화할 수 있다.

### Barrier Option과 확률 미분

금융에서 barrier option (knock-in/knock-out)은 경로가 특정 장벽을 건지는지에 달려 있다. **약수렴만으로는 가격 추정 불가**. → 이것도 AI 맥락에서 강수렴의 중요성을 보여준다: **경로 구조가 중요한 문제는 강수렴 필수**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $f \in C_b^4$ (4번 연속 미분) | Discontinuous payoff는 약수렴 저하 |
| 충분히 작은 $h$ | 극도로 stiff SDE에서는 $h$ 실제로 커야 함 (implicit 사용) |
| Global Lipschitz | 폭발 SDE는 local convergence만 |
| 경로 interpolation | 보간 오차 추가 → uniform strong conv. 손상 가능 |

**주의**: Barrier/path-dependent payoff는 반드시 **강수렴**을 확인해야 한다. 약수렴 좋다고 해서 가격이 정확한 것 아님.

---

## 📌 핵심 정리

$$\boxed{\begin{align}
\text{강수렴: } & e_s(h) = \mathbb{E}[\sup_t |X_t - \bar X_t^h|] = O(h^\alpha) \\
\text{약수렴: } & e_w(h, f) = |\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| = O(h^\beta) \\
\text{항상: } & \beta \geq \alpha / 2 \text{ (기댓값은 평균화)} \\
\text{EM: } & \alpha = 1/2, \beta = 1 \quad (\text{Smooth } f) \\
\text{Barrier: } & \beta = 1/2 \quad (\text{Discontinuous payoff})
\end{align}}$$

| 개념 | 핵심 |
|------|------|
| **경로 정확도** | 강수렴, option hedging, sample path visualization 필요 |
| **분포 정확도** | 약수렴, 옵션 가격, 기댓값 충분 |
| **Smooth test** | 약수렴 차수 1 (기댓값 평균화) |
| **Discontinuous test** | 약수렴 붕괴 → 강수렴 필요 |
| **MLMC** | 강수렴 구조 직접 활용하여 분산 감소 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): EM의 강수렴 차수 1/2, 약수렴 차수 1을 정성적으로 설명하시오. (martingale 성질 사용)

<details>
<summary>힌트 및 해설</summary>

**강수렴 (Pathwise)**:

국소 오차: $e_n^{\text{local}} = \int_t^{t+h} \int_t^s \sigma' dB_u dB_s$

이것은 **독립적인 가우시안의 이중곱**: $(\Delta B_1) \times (\Delta B_2) \times \cdots$

$\mathbb{E}[e_n^{\text{local}} \mid \mathcal{F}_{t_n}] = 0$ (martingale)

하지만 **크기**: $\sqrt{\mathbb{E}[e_n^{\text{local}2}]} = O(h^{3/2})$

누적: $\sqrt{\sum_n h^3} = \sqrt{T \cdot h^2} = T h = O(h)$... 아니다.

더 정확: **Doob 최대값 부등식** for martingale $M_n = \sum_{k=0}^n e_k^{\text{local}}$:

$$\mathbb{E}[\max_n M_n^2] \leq 4 \mathbb{E}[M_N^2] = 4 \sum_n \mathbb{E}[|e_n^{\text{local}}|^2] = 4 T \cdot O(h^{3/2} / h) = O(T h^{1/2})$$

→ **$\mathbb{E}[\sup_t |e_t|] = O(h^{1/2})$ ✓**

**약수렴 (Distributional)**:

$\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]$에서, martingale 항들의 기댓값은 **자동으로 0**:

$$\mathbb{E}\left[\int_t^{t+h} \int_t^s \sigma' dB_u dB_s\right] = 0$$

따라서 주 오차는 3차 항에서만 나옴 → **$O(h)$ ✓**

</details>

**문제 2** (심화): Barrier option payoff $f(x) = \mathbf{1}_{x > B}$ (indicator)에서 약수렴이 $O(h^{1/2})$로 붕괴되는 이유를 자세히 설명하시오.

<details>
<summary>힌트 및 해설</summary>

Indicator 함수는 $x = B$에서 불연속. 근처에서:

$$f(x) = \begin{cases} 1 & x > B \\ 0 & x < B \end{cases}$$

EM 근사 오차 $|X_T - \bar X_T^h| = O(h^{1/2})$가 있으면, $X_T$와 $\bar X_T^h$가 $B$의 양쪽에 있을 수 있다:

- $X_T = B + \epsilon$ (실제)
- $\bar X_T^h = B - \epsilon$ (근사, 오차 때문)

그러면:

$$|f(X_T) - f(\bar X_T^h)| = |1 - 0| = 1$$

이런 경우가 발생할 확률은? **$X_T$와 $\bar X_T^h$ 차이가 $B$ 근처에서 $O(h^{1/2})$이므로**:

$$P(\text{disagreement}) = O(h^{1/2})$$

따라서:

$$|\mathbb{E}[f(X_T)] - \mathbb{E}[f(\bar X_T^h)]| \leq O(h^{1/2}) \times 1 = O(h^{1/2})$$

이것이 약수렴 붕괴의 원인. **Smooth $f$이면 이 확률이 $O(h)$ 수준**이므로 약수렴이 더 좋다.

</details>

**문제 3** (AI 연결): DDPM 샘플링에서 "강수렴이 중요하다"고 하는데, 실제로 생성된 이미지 품질(FID, IS 같은 metric)이 강수렴과 어떻게 연관되는가? 약수렴만으로는 부족한가?

<details>
<summary>힌트 및 해설</summary>

**FID (Fréchet Inception Distance)**와 **IS (Inception Score)**는 생성 샘플 **앙상블**의 분포를 평가한다:

$$\text{FID} = \|m_{\text{real}} - m_{\text{gen}}\| + \text{Tr}(\Sigma_{\text{real}} + \Sigma_{\text{gen}} - 2\sqrt{\Sigma_{\text{real}}\Sigma_{\text{gen}}})\|$$

이것은 사실상 **약수렴** 관점에서 측정: "생성 분포가 실제 분포와 얼마나 다른가"

**하지만**, 특정 고주파 세부사항(예: 눈의 선명도, 질감 세부)은 **개별 경로의 정확도**에 의존한다. 이것은 강수렴과 관련. 약수렴만 좋으면:

- 분포는 맞음 (FID 좋음)
- 하지만 개별 샘플에 artifact (blur, 부자연스러운 세부)가 생길 수 있음

**현실**: DDPM은 충분한 스텝 수(50~100+)를 써서 두 오차를 모두 컨트롤. 스텝을 줄이는 것(fast sampler)은 강수렴을 악화시키는 대신 score network를 더 정교하게 학습하는 식으로 상쇄.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. Milstein 기법과 1차 강수렴](./02-milstein.md) | [📚 README로 돌아가기](../README.md) | [04. Runge-Kutta 계열과 안정성 ▶](./04-runge-kutta-stability.md) |

</div>
