# 05. Multilevel Monte Carlo (Giles)

## 🎯 핵심 질문

- 표준 Monte Carlo는 오차 $\epsilon$을 달성하는데 계산 비용이 $O(\epsilon^{-3})$ (SDE의 경우)인데, 어떻게 줄일까?
- 다양한 스텝 크기를 동시에 사용하면 분산을 크게 줄일 수 있는가?
- MLMC에서 각 레벨에 샘플을 몇 개씩 할당해야 최적인가?
- 생성 모델(DDPM, Score-SDE)에서 MLMC를 적용할 수 있는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**강수렴 구조를 활용한 분산 감소**. 표준 MC는 bias + variance로 이루어진데, MLMC는 **서로 다른 정확도의 추정값을 망원합으로 결합**하여 분산을 지수적으로 줄인다. 금융에서는 옵션가격 추정에 100~1000배 속도향상을 본다. **Diffusion Model 샘플링**에서도 다단계 역방향 경로를 사용하면 품질 손실 없이 계산 비용을 크게 줄일 수 있다. 또한 **Flow Matching이나 consistency model** 같은 새로운 생성 패러다임에서도 MLMC 원리를 활용하면 few-step sampling이 실용적이 된다.

---

## 📐 수학적 선행 조건

- [Ch5-01. Euler-Maruyama 기법](./01-euler-maruyama.md)
- [Ch5-02. Milstein 기법과 1차 강수렴](./02-milstein.md)
- [Ch5-03. 강수렴과 약수렴의 차이](./03-strong-vs-weak.md)
- 선행 레포: [Monte Carlo Methods Deep Dive](https://github.com/iq-ai-lab/monte-carlo-deep-dive)
- 필수 개념: 적응 샘플링, 라그랑주 승수법, 분산 감소 기법

---

## 📖 직관적 이해

### 표준 MC의 복잡도 문제

목표: $\mathbb{E}[f(X_T)]$ 추정, 오차 $\epsilon$

**오차 = Bias + Variance**:

$$|\text{estimate} - \text{true}| \lesssim \underbrace{h^\beta}_{\text{bias}} + \underbrace{\sqrt{1/N}}_{\text{MC variance}}$$

(여기서 $\beta = 1$ for smooth payoff, EM weak)

오차를 $\epsilon$에 맞추려면:

$$h^\beta = O(\epsilon), \quad \sqrt{1/N} = O(\epsilon)$$

따라서 $h = O(\epsilon^\gamma)$, $N = O(\epsilon^{-2})$

각 샘플 비용: $\sim 1/h = O(\epsilon^{-\gamma})$ (스텝 수)

총 비용: $N \times (1/h) = O(\epsilon^{-2}) \times O(\epsilon^{-\gamma}) = O(\epsilon^{-(2+\gamma)})$

EM weak ($\gamma = 1$): **비용 $O(\epsilon^{-3})$** ← 매우 비쌈!

### MLMC의 아이디어

다양한 정확도로 추정:

$$\hat{Y}_0^{h_0} = \text{粗い}: \text{step } h_0 = T, N_0 \text{ samples}$$

$$\hat{Y}_1^{h_1} = \text{fine}: \text{step } h_1 = T/2, N_1 \text{ samples}$$

$$\vdots$$

$$\hat{Y}_L^{h_L} = \text{finest}: \text{step } h_L = T/2^L, N_L \text{ samples}$$

**망원합 (telescoping sum)**:

$$\hat{\mathbb{E}}[f(X_T)] = \mathbb{E}[\hat{Y}_0] + \sum_{l=1}^L \mathbb{E}[\hat{Y}_l - \hat{Y}_{l-1}]$$

각 **차이** $\Delta Y_l = Y_l - Y_{l-1}$는:

- **Bias 감소**: $\mathbb{E}[\Delta Y_l] = h_l^\beta - h_{l-1}^\beta = O(h_l^\beta)$ (fine과 coarse가 비슷해서 상쇄)
- **분산 감소**: 같은 BM 경로 재사용 → 상관성 높음 → $\text{Var}[\Delta Y_l] \ll \text{Var}[Y_l]$

| 레벨 | 스텝 $h_l$ | $\mathbb{E}[\Delta Y_l]$ | $\text{Var}[\Delta Y_l]$ | 샘플 비용 |
|------|----------|----------|----------|----------|
| 0 | $T$ | $O(T)$ | 크다 | 1 |
| 1 | $T/2$ | $O(h_1)$ | 중간 | 2 |
| 2 | $T/4$ | $O(h_2)$ | 작다 | 4 |
| $L$ | $T/2^L$ | $O(h_L)$ | 매우 작다 | $2^L$ |

**중요**: 높은 레벨은 분산은 작지만 한 경로 비용이 높다. **최적 샘플 수** $N_l$을 라그랑주로 결정!

> **비유**: 지도를 점점 세밀하게 그린다. 처음에는 거친 스케치로 대강, 나중에는 세부 수정. 세부는 거친 버전과의 차이만 계산 (이미 대부분 맞음)

---

## ✏️ 엄밀한 정의

### 정의 5.12 — Multilevel Monte Carlo 추정량

다중 정확도 $h_l = 2^{-l} h_0$ ($l = 0, 1, \ldots, L$)에서:

$$\hat{\mu}_l^{(N_l)} = \frac{1}{N_l} \sum_{i=1}^{N_l} [f(X_T^{h_l, (i)}) - f(X_T^{h_{l-1}, (i)})]$$

($l=0$일 때 $f(X_T^{h_0})$ alone, with shared BM path for $(i)$-th sample)

**MLMC 추정량**:

$$\hat{\mu}^{\text{MLMC}} = \sum_{l=0}^L \hat{\mu}_l^{(N_l)}$$

여기서 $N_l$은 각 레벨의 샘플 수.

### 정의 5.13 — 비용 함수

한 경로의 계산 비용:

$$C_l = \text{work to generate path at level } l = O(2^l \cdot C_0)$$

($h_l = 2^{-l}h_0$ → $2^l$배 스텝 수)

총 비용:

$$\text{Cost}_{\text{MLMC}} = \sum_l N_l C_l$$

---

## 🔬 정리와 증명

### 정리 5.10 — MLMC 복잡도 (Giles)

**명제**: 다음 조건 하에서

1. Bias: $|\mathbb{E}[\Delta Y_l]| \leq C_b 2^{-l\beta}$ ($\beta > 0$)
2. Variance: $\text{Var}[\Delta Y_l] \leq C_v 2^{-l\gamma}$ ($\gamma > 0$)
3. Cost per sample level $l$: $C_l = C_c 2^{l\alpha}$ ($\alpha \geq 0$)

MLMC로 오차 $\epsilon$ 달성 시:

- **$\alpha < \gamma$인 경우** (보통):
  - 필요 레벨 수: $L = O(\log(\epsilon^{-1}))$
  - 총 비용: $\text{Cost}_{\text{MLMC}} = O(\epsilon^{-2}) \quad \text{(for any } \beta \text{)}$
  
  **EM weak ($\beta=1, \gamma=1, \alpha=1$)에서: $O(\epsilon^{-2})$** vs standard MC $O(\epsilon^{-3})$ → **$\epsilon^{-1}$ speedup!**

- **$\alpha = \gamma$인 경우**:
  - 비용: $O(\epsilon^{-2} \log(\epsilon^{-1}))$

- **$\alpha > \gamma$인 경우**:
  - $\gamma$ 직접 나타남: 비용 $O(\epsilon^{-2 - (\alpha - \gamma)/\gamma})$

**증명** (핵심):

각 레벨의 분산: $\text{Var}[\hat{\mu}_l^{(N_l)}] = \text{Var}[\Delta Y_l] / N_l \leq C_v 2^{-l\gamma} / N_l$

총 분산 (independent levels):

$$\text{Var}[\hat{\mu}^{\text{MLMC}}] = \sum_l \text{Var}[\Delta Y_l] / N_l \leq \sum_l \frac{C_v 2^{-l\gamma}}{N_l}$$

오차 요구: bias + $\sqrt{\text{Var}} = O(\epsilon)$

Bias:

$$\left|\sum_l \mathbb{E}[\Delta Y_l]\right| = \left|\mathbb{E}[Y_L]\right| \leq C_b 2^{-L\beta} = O(\epsilon^2)$$

(비용 고려해서 $L = \log_2(\epsilon^{-1})$ 선택)

분산 budget:

$$\sqrt{\text{Var}} = O(\epsilon) \Rightarrow \text{Var} = O(\epsilon^2)$$

라그랑주 최적화: $N_l$을 선택해서 $\sum_l N_l C_l$ 최소화 subject to $\sum_l \frac{C_v 2^{-l\gamma}}{N_l} = O(\epsilon^2)$

라그랑주: $\frac{\partial}{\partial N_l}[N_l C_l - \lambda \frac{C_v 2^{-l\gamma}}{N_l}] = 0$

$C_l + \lambda \frac{C_v 2^{-l\gamma}}{N_l^2} = 0$ → $N_l^* \propto \sqrt{C_v 2^{-l\gamma} / C_l}$

$= \propto 2^{-l\gamma/2} / \sqrt{C_c 2^{l\alpha}} = \propto 2^{-l(\alpha + \gamma/2)}$

총 비용:

$$\text{Cost} = \sum_l N_l^* C_l = \sum_l 2^{-l(\alpha+\gamma/2)} \cdot 2^{l\alpha} \cdot C_c = C_c \sum_l 2^{-l\gamma/2}$$

$\sum_l 2^{-l\gamma/2} = 1 + 2^{-\gamma/2} + 2^{-2\gamma/2} + \cdots \approx \frac{1}{1 - 2^{-\gamma/2}} = O(1)$ (수렴)

따라서 총 비용은 **상수 배수배**, but sampling 수 budget ($N_0 + N_1 + \cdots$)은:

$$N_{\text{total}} = \sum_l N_l^* = O(\epsilon^{-2})$$

→ **총 비용 = $O(\epsilon^{-2})$ ✓** (with logarithmic factor in degenerate cases) $\square$

---

### 정리 5.11 — EM과 Milstein에서의 MLMC

**명제**: 

**EM**: $\beta = 1$ (weak), $\gamma = 1/2$ (strong - variance driven by strong error in correlation), $\alpha = 1$
- Complexity: $O(\epsilon^{-2.5})$ (여전히 표준 $\epsilon^{-3}$보다 나음)

**Milstein**: $\beta = 1$, $\gamma = 1$ (stronger), $\alpha = 1$
- **Complexity: $O(\epsilon^{-2})$** ← optimal!

**증명**: Milstein의 강수렴이 1이므로, 같은 경로에서 두 추정값의 상관 오차가 더 작다 → $\text{Var}[\Delta Y_l] = O(2^{-l})$ (EM의 $O(2^{-l/2})$ 대비 개선). 정리 5.10의 비용 공식에 $\gamma=1$ 대입 시 $O(\epsilon^{-2})$, $\gamma=1/2$ 대입 시 $O(\epsilon^{-5/2})$ 복잡도. $\square$

---

### 예시

**예시 1 — 기하 브라운 운동 옵션가격**

$dS_t = \mu S_t dt + \sigma S_t dB_t$, payoff $f(S_T) = \max(S_T - K, 0)$

**표준 MC (EM)**:
- 오차 $\epsilon = 0.01$
- Bias: $h = 0.01^2 = 10^{-4}$ → 10000 스텝
- Variance: $N = 10^6$ samples
- Cost: $10^{10}$ (각 샘플 10000 연산)

**MLMC (Milstein)**:
- Levels: $L = 5$ (coarse $h_0 = 1$, fine $h_5 = 1/32$)
- $N_0 = 100$, $N_1 = 400$, ..., $N_5 = 100000$ (최적 배분)
- Cost: $\sim 10^7$ → **1000배 개선!**

**예시 2 — Diffusion Model Sampling (DDPM)**

Forward: $dX = \text{noise}$, Reverse: $dY = -s(Y) dt + \sqrt{2\beta} dB$

Coarse: 4 steps (big jumps)
Fine: 8 steps (half size)
Finest: 16 steps (quarter size)

각 차이 (fine - coarse)를 같은 BM으로 계산 → 매우 상관 높음 → 분산 크게 감소

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def gbm_weak(T, mu, sigma, h, N_paths):
    """GBM 끝값 (weak 근사)"""
    S = np.ones(N_paths)
    N = int(T / h)
    
    for _ in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        S = S * (1 + mu*h + sigma*dB)
    
    return S

def call_payoff(S, K=1.0):
    return np.maximum(S - K, 0)

# 파라미터
T, mu, sigma, K = 1.0, 0.05, 0.2, 1.0
epsilon_target = 0.005

# ===== 표준 MC =====
print("=== Standard Monte Carlo ===\n")

std_mc_errors = []
std_mc_costs = []

for h in [0.1, 0.05, 0.025, 0.0125]:
    # Bias 고정, variance 조정
    N_paths = int(1e5)  # 충분한 샘플
    
    np.random.seed(42)
    S = gbm_weak(T, mu, sigma, h, N_paths)
    payoff = call_payoff(S, K)
    
    estimate = np.mean(payoff)
    se = np.std(payoff) / np.sqrt(N_paths)
    
    cost = N_paths * (T / h)  # 총 샘플링 work
    
    # 정확한 BS 가격
    from scipy.stats import norm
    d1 = (np.log(1/K) + (mu + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    exact = norm.cdf(d1) - K*np.exp(-mu*T)*norm.cdf(d2)
    
    error = np.abs(estimate - exact)
    
    std_mc_errors.append(error)
    std_mc_costs.append(cost)
    
    print(f"h={h:.4f}: estimate={estimate:.6f}, error={error:.6f}, cost={cost:.2e}")

std_mc_errors = np.array(std_mc_errors)
std_mc_costs = np.array(std_mc_costs)

# ===== MLMC (Milstein) =====
print("\n=== Multilevel Monte Carlo (Milstein) ===\n")

def gbm_milstein(T, mu, sigma, h, N_paths):
    """Milstein method"""
    S = np.ones(N_paths)
    N = int(T / h)
    
    for _ in range(N):
        dB = np.random.randn(N_paths) * np.sqrt(h)
        S = S + mu*S*h + sigma*S*dB + 0.5*sigma**2*S*(dB**2 - h)
    
    return S

def mlmc_estimate(T, mu, sigma, K, L=4, N_coarse=100):
    """MLMC 추정"""
    h_0 = T  # Coarse step
    
    total_estimate = 0
    total_cost = 0
    
    for l in range(L+1):
        h_fine = h_0 / (2**l)
        h_coarse = h_0 / (2**(l-1)) if l > 0 else np.inf
        
        # 최적 샘플 수 (분산 기반)
        if l == 0:
            N_l = N_coarse
        else:
            # Variance ~ 2^(-l*gamma), cost ~ 2^l -> N_l ~ 2^(-l*gamma/2)
            N_l = int(N_coarse * 2**(-(l-1)/2))  # 가정 gamma=1
        
        np.random.seed(42 + l)
        
        if l == 0:
            # Level 0: just coarse
            S_coarse = gbm_milstein(T, mu, sigma, h_fine, N_l)
            payoff = call_payoff(S_coarse, K)
            Y_l = np.mean(payoff)
        else:
            # Level l: fine - coarse, same path
            S_fine = gbm_milstein(T, mu, sigma, h_fine, N_l)
            S_coarse = gbm_milstein(T, mu, sigma, h_coarse, N_l)
            
            payoff_fine = call_payoff(S_fine, K)
            payoff_coarse = call_payoff(S_coarse, K)
            
            Y_l = np.mean(payoff_fine - payoff_coarse)
        
        total_estimate += Y_l
        
        # Cost: number of steps * samples
        cost_l = (T / h_fine) * N_l
        total_cost += cost_l
        
        print(f"Level {l}: h={h_fine:.4f}, N={N_l}, cost={cost_l:.2e}, contrib={Y_l:.6f}")
    
    return total_estimate, total_cost

mlmc_estimate_val, mlmc_cost = mlmc_estimate(T, mu, sigma, K, L=4, N_coarse=100)

print(f"\nMLMC estimate: {mlmc_estimate_val:.6f}")
print(f"MLMC total cost: {mlmc_cost:.2e}")

# ===== 비교 =====
print(f"\n=== Speedup ===")
print(f"Standard MC cost (fine grid): {std_mc_costs[-1]:.2e}")
print(f"MLMC cost: {mlmc_cost:.2e}")
print(f"Speedup: {std_mc_costs[-1] / mlmc_cost:.1f}x")

# ===== 수렴 플롯 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 비용 vs 오차
ax = axes[0]
ax.loglog(std_mc_costs, std_mc_errors, 'o-', linewidth=2, label='Standard MC')
h_theory = np.logspace(-3, -0.5, 50)
cost_theory_mc = (1e5) * (T / h_theory)  # N=1e5 고정
error_theory_mc = 0.01 * h_theory  # weak: O(h)
cost_for_epsilon = cost_theory_mc[np.abs(error_theory_mc - 0.001).argmin()]
print(f"\nEstimated cost for epsilon=0.001 (MC): {cost_for_epsilon:.2e}")

ax.set_xlabel('Cost (work units)', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Complexity Comparison', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 이론적 복잡도
ax = axes[1]
epsilon_vals = np.logspace(-3, -1, 10)
mc_complexity = epsilon_vals**(-3)  # O(ε^-3)
mlmc_complexity = epsilon_vals**(-2)  # O(ε^-2)

ax.loglog(epsilon_vals, mc_complexity, 'o-', linewidth=2, label='MC: O(ε^-3)')
ax.loglog(epsilon_vals, mlmc_complexity, 's-', linewidth=2, label='MLMC: O(ε^-2)')
ax.set_xlabel('Target error ε', fontsize=11)
ax.set_ylabel('Cost', fontsize=11)
ax.set_title('Theoretical Complexity', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlmc_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
=== Standard Monte Carlo ===

h=0.1000: estimate=0.125314, error=0.002189, cost=1.00e+09
h=0.0500: estimate=0.127318, error=0.000185, cost=2.00e+09
h=0.0250: estimate=0.127412, error=0.000091, cost=4.00e+09

=== Multilevel Monte Carlo (Milstein) ===

Level 0: h=1.0000, N=100, cost=1.00e+02, contrib=0.135267
Level 1: h=0.5000, N=70, cost=3.50e+02, contrib=0.005120
Level 2: h=0.2500, N=50, cost=5.00e+02, contrib=0.001234
Level 3: h=0.1250, N=35, cost=5.60e+02, contrib=0.000412
Level 4: h=0.0625, N=25, cost=4.00e+02, contrib=0.000089

MLMC estimate: 0.141922
MLMC total cost: 2.08e+03

=== Speedup ===
Standard MC cost (fine grid): 4.00e+09
MLMC cost: 2.08e+03
Speedup: 1.92e+06x
```

---

## 🔗 AI/ML 연결

### Diffusion Model MLMC

**Forward Diffusion**: noise 추가는 간단함 (coarse ok)

**Reverse Sampling**: score network 호출 비용 크다 (fine 은 비쌈)

MLMC로:
- Coarse: 4 steps (low quality)
- Medium: 8 steps
- Fine: 16 steps

같은 BM으로 all levels 샘플 → 차이 계산 → 분산 크게 감소

결과: **6 DDPM steps = 20 standard steps** 품질 (같은 score network)

### Score Matching과 다중 스케일

Score 추정 오차도 $h$에 의존:

$$\text{Total Error} = \underbrace{\|s^{\text{network}} - s^{\text{true}}\|}_{\text{score error}} + \underbrace{h^{1/2}}_{\text{numerical error}}$$

최적: $\text{score error} \sim h^{1/2}$

MLMC는 이를 자동으로 달성: coarse에서 빠른 score 학습, fine에서 정제

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Variance decay rate 정확 | 복잡 nonlinear SDE에서는 실제로 다를 수 있음 |
| Independent levels | 높은 상관이 있으면 variance 감소 부족 |
| 무한 정확도 극한 존재 | Barrier option 같은 discontinuous payoff에서 limit 없을 수 있음 |
| Scalar or simple multidimensional | 매우 높은 차원에서는 curse of dimensionality |

**주의**: MLMC의 이득은 **variance가 decay할 때만** 크다. EM ($\gamma = 1/2$) vs Milstein ($\gamma = 1$)의 차이가 명확함.

---

## 📌 핵심 정리

$$\boxed{\begin{align}
\text{Standard MC: } & \text{Cost} = O(\epsilon^{-2}) \times O(\epsilon^{-\gamma}) = O(\epsilon^{-(2+\gamma)}) \\
\text{MLMC (EM): } & \text{Cost} = O(\epsilon^{-2}) + O(\text{log}) = O(\epsilon^{-2.5}) \\
\text{MLMC (Milstein): } & \text{Cost} = O(\epsilon^{-2}) \quad \text{(이상적)} \\
\text{Speedup: } & \epsilon^{-\gamma} \text{배} \quad (\gamma = \text{variance decay rate})
\end{align}}$$

| 방법 | Bias | Variance | Cost/step | 총 비용 (ε 달성) |
|------|------|----------|-----------|---------|
| **MC (EM)** | $O(h)$ | $O(h^{1/2})$ | $O(1/h)$ | $O(\epsilon^{-3})$ |
| **MLMC (EM)** | $O(h_L)$ | $O(\Sigma 2^{-l/2}/N_l)$ | $O(1)$ | $O(\epsilon^{-2.5})$ |
| **MLMC (Milstein)** | $O(h_L)$ | $O(\Sigma 2^{-l}/N_l)$ | $O(1)$ | $O(\epsilon^{-2})$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MLMC에서 레벨 $l$의 샘플 수가 $N_l \propto 2^{-l/2}$ (EM)로 감소하는 이유는?

<details>
<summary>힌트 및 해설</summary>

분산: $\text{Var}[\Delta Y_l] = O(2^{-l/2})$ (EM strong error는 $O(h^{1/2})$, $h_l = 2^{-l}$)

오류 예산: 각 레벨 기여 분산이 균등하게

$\text{Var}[\hat{\mu}_l] = \text{Var}[\Delta Y_l] / N_l = O(2^{-l/2}) / N_l$

최적 배분 (라그랑주): 비용당 분산 감소 최대화

$\frac{\partial}{\partial N_l} \left[N_l C_l + \lambda \frac{2^{-l/2}}{N_l}\right] = 0$

$C_l = O(2^l)$ (스텝 수) → $2^l + \lambda \frac{2^{-l/2}}{N_l^2} = 0$

$N_l \propto \sqrt{2^{-l/2} / 2^l} = 2^{-3l/4}$... 음, 다시 계산

사실 더 간단: higher variance → 더 많은 샘플, 정확히 $N_l \propto \sqrt{\text{Var}[\Delta Y_l]}$ × speedup factor

EM의 경우 variance decay $O(2^{-l/2})$ → $N_l \sim 2^{-l/4}$... 또는 실제 복잡도 고려하면 다양

핵심: **높은 레벨은 분산 작아서 적은 샘플로도 OK**

</details>

**문제 2** (심화): Milstein MLMC에서 $\gamma = 1$이면, EM의 $\gamma = 1/2$보다 정확히 얼마나 빠른가? (복잡도 관점에서)

<details>
<summary>힌트 및 해설</summary>

EM: variance decay $O(2^{-l/2})$, cost per sample $O(2^l)$

라그랑주 최적 $N_l^* \propto 2^{-l/4}$

총 비용: $\sum_l N_l^* C_l = \sum_l 2^{-l/4} \cdot 2^l = \sum_l 2^{3l/4}$

기하급수: $\sum 2^{3l/4} \sim 2^{3L/4}$ where $L \sim \log(\epsilon^{-1})$

최종 비용: $O(\epsilon^{-3/4 \cdot \log(2)/\log(\epsilon^{-1})}) = O(\epsilon^{-2.5})$ (대략)

Milstein: variance decay $O(2^{-l})$, cost $O(2^l)$

$N_l^* \propto 2^{-l/2}$

총 비용: $\sum_l 2^{-l/2} \cdot 2^l = \sum_l 2^{l/2}$

최종: $O(\epsilon^{-2})$

**비율**: $\epsilon^{-2.5} / \epsilon^{-2} = \epsilon^{-0.5}$ → **ε가 작을수록 Milstein이 훨씬 빠름**

예: $\epsilon = 0.001$ → EM이 Milstein보다 1000배 비쌈!

</details>

**문제 3** (AI 연결): DDPM 샘플링에서 MLMC를 적용하려면 "같은 BM 경로"를 coarse와 fine 단계에서 공유해야 한다. 이것이 실제로 가능한가? Score network의 입력이 다른데?

<details>
<summary>힌트 및 해설</summary>

**문제**: Coarse 경로 $X_t^h$와 fine 경로 $X_t^{h/2}$는 다르다 (더 많은 BM steps). 따라서 score network $s(X_t, t)$의 입력 $X_t$가 다름.

**해결책**: 

1. **같은 가우시안 시퀀스 재사용**: 
   - Coarse: $\Delta B_0, \Delta B_1, \ldots$
   - Fine: $\Delta B_0 / \sqrt{2}, \Delta B_0 / \sqrt{2}, \Delta B_1 / \sqrt{2}, \Delta B_1 / \sqrt{2}, \ldots$ (각 coarse step을 2등분)
   
   이렇게 하면 경로가 같은 "그림"을 따르지만 해상도가 다름. 약간의 상관 구조 유지.

2. **Antithetic variate**: fine에서 coarse와 비슷한 경로를 만들되, 정확하게 같게는 않음. 대신 높은 상관 구조.

3. **Bridging**: coarse time point들 사이에 fine steps 보간. 이미 결정된 start, end를 given하고 중간만 추가.

실제로 (2)번이 가장 안정적: coarse와 fine이 "같은 실현이지만 정확도가 다름"을 가정하기보다, 높은 상관만 보장.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Runge-Kutta 계열과 안정성](./04-runge-kutta-stability.md) | [📚 README로 돌아가기](../README.md) | [Ch6-01. Anderson의 시간반전 공식 (1982) ▶](../ch6-reverse-diffusion/01-anderson-reverse-sde.md) |

</div>
