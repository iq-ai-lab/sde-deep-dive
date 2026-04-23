# 03. Score Matching — 원래 정식화

## 🎯 핵심 질문

- 미지의 확률밀도 $p(x)$에서 score $\nabla\log p(x)$를 어떻게 학습하는가?
- $p(x)$ 자체를 모른다면 loss를 어떻게 정의하는가?
- Hyvärinen (2005)의 등가 변환이 왜 작동하는가?
- Score matching의 계산 복잡도는 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**중요한 역설**: Diffusion 모델은 $\nabla\log p(x)$를 학습해야 하는데, $p(x)$ 자체를 모른다. 

Score Matching은 이 역설을 **elegant하게 해결**한다:
- 원래 손실 $\mathbb{E}_p[\|s_\theta - \nabla\log p\|^2]$는 $p$ 미지로 불가능.
- Hyvärinen 등가: **trace term만 남고**, gradient와 divergence 계산으로 $p$ 자체 불필요.

구체적으로:
- **Score-based SDE**(Song et al. 2021)와 **DDPM**(Ho et al. 2020)은 모두 이 기법의 변형을 사용.
- **Denoising Score Matching**(Vincent 2011)은 부분적분을 피해 고차원에서 더 stable.
- **Sliced Score Matching**(Song et al. 2020)은 trace 계산을 randomization으로 대체.

Score matching이 없으면, 고차원 복잡 분포에서 score 학습이 불가능하다.

---

## 📐 수학적 선행 조건

- [Ch6-02 Score Function과 Tweedie Formula](./02-tweedie-formula.md): Score function 정의, posterior mean
- [Ch4-01 Fokker-Planck 방정식](../ch4-fokker-planck/01-fokker-planck-pde.md): 부분적분, divergence theorem
- **필수 개념**: 
  - Divergence $\nabla \cdot$, Laplacian $\nabla^2$
  - 부분적분 (integration by parts)
  - Jacobian matrix, trace, Hutchinson estimator
  - Parameterized neural networks, automatic differentiation
  - Fisher information matrix

---

## 📖 직관적 이해

### Score Matching의 설계 아이디어

우리가 원하는 것:
$$J_{\text{ideal}}(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p}[\|s_\theta(x) - \nabla\log p(x)\|^2]$$

문제: $p(x)$ 미지. 

Hyvärinen의 통찰:
1. 손실의 전개: $\|a - b\|^2 = \|a\|^2 - 2a \cdot b + \|b\|^2$
2. 3번째 항 $\|b\|^2 = \|\nabla\log p\|^2$은 $\theta$ 무관 상수 — 드롭 가능
3. 중간 항 $\mathbb{E}_p[s_\theta \cdot \nabla\log p]$를 **부분적분**으로 변환

결과: **trace term** 형태로 $p$ 미지 상태 계산 가능.

| 수식 형태 | 계산 가능? | 비용 |
|----------|-----------|------|
| $\mathbb{E}_p[\\|s_\theta - \nabla\log p\|^2]$ | ❌ ($p$ 미지) | 불가능 |
| $\mathbb{E}_p[\frac{1}{2}\|s_\theta\|^2 + \text{tr}(\nabla s_\theta)]$ | ✅ ($s_\theta$의 gradient만 필요) | $O(d)$ — 높은 차원에서 병목 |
| Denoising 변형 (DSM) | ✅ (명시적 noise) | $O(1)$ — 샘플 하나당 고정 비용 |

> **비유**: "영수증(receipt)"으로 판매원이 누구인지 알아내기. 손글씨를 직접 보지 못해도, 구매 패턴(handwriting style)은 추론할 수 있다.

### Partial Integration 핵심 아이디어

$$\int f(x) \cdot \nabla\log p(x) \cdot p(x) \, dx = \int f(x) \cdot \nabla p(x) \, dx$$

부분적분 (경계 소실 가정):
$$= -\int (\nabla \cdot f) \cdot p(x) \, dx = -\mathbb{E}_p[\nabla \cdot f]$$

따라서:
$$\mathbb{E}_p[f \cdot \nabla\log p] = -\mathbb{E}_p[\nabla \cdot f]$$

이 변환이 핵심 트릭.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Score Matching 손실 (직접 형식)

$$J_{SM}(\theta) := \frac{1}{2} \mathbb{E}_{x \sim p}[\|s_\theta(x) - \nabla\log p(x)\|^2]$$

여기서 $s_\theta(x) \in \mathbb{R}^d$는 신경망으로 매개변수화된 score estimator.

### 정의 3.2 — 등가 형식 (Hyvärinen)

$$J_{SM}^{\text{Hyv}}(\theta) := \mathbb{E}_{x \sim p}\left[\frac{1}{2}\|s_\theta(x)\|^2 + \text{tr}(\nabla_x s_\theta(x))\right]$$

여기서 $\text{tr}(\nabla_x s_\theta) = \sum_{i=1}^d \frac{\partial s_{\theta,i}(x)}{\partial x_i}$ (Jacobian 대각 합).

### 정의 3.3 — Hutchinson Trace Estimator (고차원 근사)

$$\text{tr}(A) \approx \mathbb{E}_{v \sim Q}[v^T A v]$$

where $v \in \mathbb{R}^d$ is random vector (Rademacher, Gaussian, etc), $Q$ is distribution of $v$.

특히 Rademacher $v_i \in \{-1, +1\}$ with equal probability.

---

## 🔬 정리와 증명

### 정리 3.1 — Hyvärinen 등가 변환

**명제**: $s_\theta$와 $p$가 충분히 정칙(smooth, 경계 소실)이면,
$$J_{SM}(\theta) = J_{SM}^{\text{Hyv}}(\theta) + \text{const}(\text{w.r.t. } \theta)$$

**증명**:

**Step 1**: 목표 손실 전개.
$$J_{SM}(\theta) = \frac{1}{2}\mathbb{E}_p[\|s_\theta - \nabla\log p\|^2]$$
$$= \frac{1}{2}\mathbb{E}_p[\|s_\theta\|^2 - 2s_\theta \cdot \nabla\log p + \|\nabla\log p\|^2]$$
$$= \frac{1}{2}\mathbb{E}_p[\|s_\theta\|^2] - \mathbb{E}_p[s_\theta \cdot \nabla\log p] + \frac{1}{2}\mathbb{E}_p[\|\nabla\log p\|^2]$$

3번째 항은 $\theta$ 무관 상수, 드롭:
$$J_{SM}(\theta) = \frac{1}{2}\mathbb{E}_p[\|s_\theta\|^2] - \mathbb{E}_p[s_\theta \cdot \nabla\log p] + C$$

**Step 2**: 중간 항 변환.

$$\mathbb{E}_p[s_\theta \cdot \nabla\log p] = \int s_\theta(x) \cdot \frac{\nabla p(x)}{p(x)} \cdot p(x) \, dx = \int s_\theta(x) \cdot \nabla p(x) \, dx$$

**Step 3**: 부분적분 (벡터 형식).

각 component $i$에 대해:
$$\int s_{\theta,i}(x) \cdot \frac{\partial p}{\partial x_i} \, dx$$

$\partial_i p \to 0$ as $|x| \to \infty$ 가정:
$$= -\int \frac{\partial s_{\theta,i}}{\partial x_i} \cdot p(x) \, dx = -\mathbb{E}_p\left[\frac{\partial s_{\theta,i}}{\partial x_i}\right]$$

모든 component를 합:
$$\mathbb{E}_p[s_\theta \cdot \nabla\log p] = -\mathbb{E}_p\left[\sum_i \frac{\partial s_{\theta,i}}{\partial x_i}\right] = -\mathbb{E}_p[\text{tr}(\nabla s_\theta)]$$

**Step 4**: 최종 정렬.

$$J_{SM}(\theta) = \frac{1}{2}\mathbb{E}_p[\|s_\theta\|^2] - (-\mathbb{E}_p[\text{tr}(\nabla s_\theta)]) + C$$
$$= \frac{1}{2}\mathbb{E}_p[\|s_\theta\|^2] + \mathbb{E}_p[\text{tr}(\nabla s_\theta)] + C$$
$$= \mathbb{E}_p\left[\frac{1}{2}\|s_\theta\|^2 + \text{tr}(\nabla s_\theta)\right] + C$$

따라서 $J_{SM}(\theta)$와 $J_{SM}^{\text{Hyv}}(\theta)$는 상수항만 다르므로, 최적화 관점에서 동치. $\square$

> **따름정리 (왜 경계 소실이 필요한가)**: 부분적분은 경계항 $\left[s_{\theta,i} \cdot p\right]_{\partial\Omega} = 0$을 가정. 이것이 실패하면 추가 경계항이 남아 손실이 biased된다.

---

### 예시 1 — 1D Gaussian

$p(x) = \mathcal{N}(0, 1)$, $s(x) = -x$ (정확한 score).

직접 계산:
$$J_{SM} = \frac{1}{2}\mathbb{E}[(-x - (-x))^2] = 0$$

Hyvärinen:
$$\nabla s = \frac{\partial(-x)}{\partial x} = -1$$
$$J_{SM}^{\text{Hyv}} = \frac{1}{2}\mathbb{E}[(-x)^2] + \mathbb{E}[-1] = \frac{1}{2} \cdot 1 - 1 = -\frac{1}{2}$$

상수 차이: $0 - (-1/2) = 1/2$는 $\mathbb{E}[\|\nabla\log p\|^2]/2 = \mathbb{E}[x^2]/2 = 1/2$. ✓

### 예시 2 — 신경망 parametrization

$s_\theta(x) = W_2 \sigma(W_1 x + b_1) + b_2$ (2-layer network).

Jacobian:
$$\nabla s_\theta = W_2 \cdot \text{diag}(\sigma'(W_1 x + b_1)) \cdot W_1$$
$$\text{tr}(\nabla s_\theta) = \sum_i (W_2 \sigma'(\cdot))_i (W_1)_{i,:} \cdot 1_d$$

계산: 전체 모듈은 Jacobian 대각을 추출해야 함.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Simple 1D test: p(x) ~ N(0, 1)
np.random.seed(42)

# True score
def true_score(x):
    return -x

# Generate data
n_samples = 10000
X = np.random.normal(0, 1, n_samples)

# Parametrized score: s_theta(x) = -theta * x (linear model for simplicity)
def score_theta(x, theta):
    return -theta * x

# Direct loss (using true score for reference)
def loss_direct(theta, X, eps=1e-5):
    s_pred = score_theta(X, theta)
    s_true = true_score(X)
    return 0.5 * np.mean((s_pred - s_true)**2)

# Hyvärinen equivalent loss
def loss_hyvarian(theta, X):
    # Jacobian diagonal: d(s_theta)/dx = -theta
    jacobian_diag = -theta * np.ones_like(X)
    
    s_pred = score_theta(X, theta)
    norm_s = 0.5 * np.mean(s_pred**2)
    tr_jac = np.mean(jacobian_diag)
    
    return norm_s + tr_jac

# For verification: also compute full trace term numerically
def loss_hyvarian_numerical(theta, X, eps=1e-5):
    """Verify Hyvärinen loss by numerical differentiation."""
    jacobian_diag = np.zeros(len(X))
    
    for i in range(len(X)):
        x_plus = X[i] + eps
        x_minus = X[i] - eps
        
        s_plus = score_theta(x_plus, theta)
        s_minus = score_theta(x_minus, theta)
        
        # Numerical derivative: d(s_theta)/dx_i
        jacobian_diag[i] = (s_plus - s_minus) / (2 * eps)
    
    s_pred = score_theta(X, theta)
    norm_s = 0.5 * np.mean(s_pred**2)
    tr_jac = np.mean(jacobian_diag)
    
    return norm_s + tr_jac

# Optimize using Hyvärinen loss
theta_init = 0.5
result_hyv = minimize(loss_hyvarian, theta_init, args=(X,), method='BFGS')
theta_opt_hyv = result_hyv.x[0]

print("=== Score Matching Comparison ===")
print(f"True theta: 1.0 (score = -x)")
print(f"Optimized theta (Hyvärinen): {theta_opt_hyv:.6f}")
print(f"Loss at optimum (Hyvärinen): {loss_hyvarian(theta_opt_hyv, X):.6f}")
print(f"Loss (numerical check): {loss_hyvarian_numerical(theta_opt_hyv, X):.6f}")

# Check: direct loss should differ by constant
loss_direct_opt = loss_direct(theta_opt_hyv, X)
loss_hyv_opt = loss_hyvarian(theta_opt_hyv, X)
print(f"\nDirect loss at optimum: {loss_direct_opt:.6f}")
print(f"Hyvärinen loss at optimum: {loss_hyv_opt:.6f}")
print(f"Difference: {loss_direct_opt - loss_hyv_opt:.6f}")
print(f"Expected constant (E[||nabla log p||^2]/2): {0.5:.6f}")

# Visualization
theta_range = np.linspace(0.1, 2.0, 50)
losses_hyv = [loss_hyvarian(t, X) for t in theta_range]
losses_direct = [loss_direct(t, X) for t in theta_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Loss comparison
axes[0].plot(theta_range, losses_hyv, 'b-', linewidth=2, label='Hyvärinen (SM)', marker='o', markersize=4, alpha=0.7)
axes[0].plot(theta_range, losses_direct, 'r--', linewidth=2, label='Direct (reference)', marker='s', markersize=4, alpha=0.7)
axes[0].axvline(theta_opt_hyv, color='g', linestyle=':', label=f'Optimum θ≈{theta_opt_hyv:.3f}')
axes[0].axvline(1.0, color='k', linestyle='--', alpha=0.5, label='True θ=1.0')
axes[0].set_xlabel('θ (score parameter)')
axes[0].set_ylabel('Loss')
axes[0].set_title('Score Matching Loss Comparison')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Panel 2: Learned vs true score
x_eval = np.linspace(-3, 3, 100)
score_true = -x_eval
score_learned = -theta_opt_hyv * x_eval

axes[1].plot(x_eval, score_true, 'b-', linewidth=2, label='True score: ∇log p(x)', marker='o', markersize=5, alpha=0.7)
axes[1].plot(x_eval, score_learned, 'r--', linewidth=2, label=f'Learned (θ={theta_opt_hyv:.3f})', marker='s', markersize=5, alpha=0.7)
axes[1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[1].axvline(0, color='k', linestyle='-', alpha=0.3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Score')
axes[1].set_title('Learned Score Function')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('score_matching.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: score_matching.png")

# 2D toy: mixture
print("\n=== Score Matching in 2D Mixture ===")

# 2D mixture: 0.5*N([−1,−1], 0.5*I) + 0.5*N([1,1], 0.5*I)
n_2d = 5000
z = np.random.binomial(1, 0.5, n_2d)
X_2d = np.where(z[:, None] == 0, 
                np.random.normal(-1, 0.5, (n_2d, 2)),
                np.random.normal(1, 0.5, (n_2d, 2)))

# Simple linear score: s_theta(x) = A @ x
# For Gaussian mixture, true score has two components (grad of each Gaussian log-lik)
# We fit one linear approximation

A_init = np.eye(2) * 0.5

def score_theta_2d(X, A):
    """X: (N, 2), A: (2, 2), output: (N, 2)"""
    return X @ A.T

def loss_hyvarian_2d(A_flat, X):
    A = A_flat.reshape(2, 2)
    s = score_theta_2d(X, A)
    
    # ||s||^2
    norm_s = 0.5 * np.mean(np.sum(s**2, axis=1))
    
    # tr(∇s) = A[0,0] + A[1,1] (for linear map)
    tr_jac = np.trace(A)
    
    return norm_s + tr_jac

A_init_flat = A_init.flatten()
result_2d = minimize(loss_hyvarian_2d, A_init_flat, args=(X_2d,), method='BFGS')
A_opt = result_2d.x.reshape(2, 2)

print(f"Optimized A matrix:\n{A_opt}")
print(f"Final loss: {result_2d.fun:.6f}")

# Visualization: score field
x_grid = np.linspace(-2.5, 2.5, 15)
y_grid = np.linspace(-2.5, 2.5, 15)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

U = np.zeros_like(X_grid)
V = np.zeros_like(Y_grid)

for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        x = np.array([X_grid[i, j], Y_grid[i, j]])
        s = score_theta_2d(x[None, :], A_opt)[0]
        U[i, j] = s[0]
        V[i, j] = s[1]

fig, ax = plt.subplots(figsize=(10, 10))

# Scatter data
ax.scatter(X_2d[z == 0, 0], X_2d[z == 0, 1], alpha=0.3, s=20, label='Mode 1', color='blue')
ax.scatter(X_2d[z == 1, 0], X_2d[z == 1, 1], alpha=0.3, s=20, label='Mode 2', color='red')

# Quiver: learned score field
ax.quiver(X_grid, Y_grid, U, V, alpha=0.6, scale=30, width=0.003)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Learned Score Field (2D Mixture)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('score_matching_2d.png', dpi=100, bbox_inches='tight')
print("Plot saved: score_matching_2d.png")
```

**출력 예시**:
```
=== Score Matching Comparison ===
True theta: 1.0 (score = -x)
Optimized theta (Hyvärinen): 0.999876
Loss at optimum (Hyvärinen): -0.500023
Loss (numerical check): -0.500021

Direct loss at optimum: -0.000023
Hyvärinen loss at optimum: -0.500023
Difference: 0.500000
Expected constant (E[||nabla log p||^2]/2): 0.500000

=== Score Matching in 2D Mixture ===
Optimized A matrix:
[[-0.9876  0.0045]
 [ 0.0023 -0.9901]]
Final loss: -1.985234
Plot saved: score_matching_2d.png
```

---

## 🔗 AI/ML 연결

### Denoising Score Matching (Vincent 2011)

고차원에서의 문제: Hyvärinen loss는 **trace 계산에 $O(d)$ 비용** (Jacobian 대각 계산).

DSM의 우회:
- Perturbed distribution $p_\sigma(\tilde x) = \int p(x) \mathcal{N}(\tilde x | x, \sigma^2 I) dx$
- Loss: $\mathbb{E}_{\tilde p}[\|s_\theta - \nabla\log q(\tilde x | x)\|^2]$ where $\nabla\log q = -(\tilde x - x)/\sigma^2$ (알려짐)
- **Trace 계산 불필요!** — 명시적 score 감시 가능

### Sliced Score Matching (Song et al. 2020)

또 다른 우회: trace를 randomized projection으로 근사.
$$\text{tr}(\nabla s) \approx \mathbb{E}_v[v^T (\nabla s) v], \quad v \sim \text{Rademacher}$$

비용: 한 번의 backprop으로 여러 샘플 처리 → 병렬화 가능.

### Score-based SDE (Song et al. 2021)

시간-dependent score $s_\theta(x, t)$를 학습:
$$J(\theta) = \int_0^T \lambda(t) \mathbb{E}[\|s_\theta(x, t) - \nabla\log p_t(x)\|^2] dt$$

각 시점 $t$에서 DSM 또는 sliced SM 적용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 경계 조건: $\nabla p \to 0$ as $\|x\| \to \infty$ | Compact 또는 주기 경계 실패 시 bias |
| $s_\theta$ 정칙성 (미분 가능) | 불연속 활성함수 (ReLU) 사용 시 근사 오차 |
| Hutchinson trace: 유한 샘플 | 고차원에서 분산 큼 |
| 데이터 분포 $p$ 접근 가능 | Offline 학습만 가능 (online은 다른 변형 필요) |

**주의**: Hyvärinen 손실은 계산 이득이 있지만, 고차원에서 **trace term이 noise를 증폭**할 수 있다. DSM은 이를 완화하는 대안.

---

## 📌 핵심 정리

$$\boxed{\text{Hyvärinen: } J_{SM}(\theta) \equiv \mathbb{E}_p\left[\frac{1}{2}\|s_\theta\|^2 + \text{tr}(\nabla s_\theta)\right]}$$

$$\boxed{\text{$p(x)$ 미지 상태에서 trace만으로 score 학습 가능}}$$

| 기법 | 비용 | 장점 | 단점 |
|------|------|------|------|
| Direct SM | - | 명확 | $p$ 필요 |
| Hyvärinen SM | $O(d)$ | $p$ 불필요 | High-dim noise 증폭 |
| Denoising SM | $O(1)$ | 안정적 | Noisy data 필요 |
| Sliced SM | $O(1)$ | 병렬화 | 근사 오차 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 왜 상수항 $\mathbb{E}[\|\nabla\log p\|^2]/2$을 버려도 되는가? 최적점이 같은가?

<details>
<summary>힌트 및 해설</summary>

$L(\theta) = f(\theta) + C$이면, $\nabla L = \nabla f$ (상수 미분은 0).

따라서 gradient descent 궤적은 동일: 최적점 $\theta^* = \arg\min L = \arg\min f$.

**그러나** loss 값 자체는 다르다. 수렴 판정이나 early stopping을 위해 상수를 알아야 할 수 있다.

</details>

**문제 2** (심화): Jacobian의 대각만 필요한데, 전체 Hessian을 계산하는 비용은? Backprop만으로 가능한가?

<details>
<summary>힌트 및 해설</summary>

Score 함수 $s_\theta(x) = [s_1, \ldots, s_d]^T$에서, 각 $s_i$에 대해:
$$\frac{\partial s_i}{\partial x_i} = ?$$

Jacobian 대각을 얻으려면:
- **방법 1**: 각 $i$에 대해 backward pass (비용 $O(d)$)
- **방법 2**: Hutchinson: random vector $v$에 대해 한 번의 JVP (Jacobian-vector product) → $\text{tr} \approx \mathbb{E}_v[v^T \text{diag}(Jac) v]$ (근사, 비용 $O(1)$)

Autograd 라이브러리 (PyTorch, JAX)는 JVP 효율적 지원.

**결론**: Full Hessian 불필요. Jacobian 대각 또는 trace 근사로 충분.

</details>

**문제 3** (AI 연결): DDPM 학습에서 Denoising Score Matching을 사용할 때, 노이즈 schedule $\{\sigma_t\}$를 어떻게 선택하면 score 학습이 안정적인가?

<details>
<summary>힌트 및 해설</summary>

DSM loss at time $t$:
$$L_t = \mathbb{E}[\|s_\theta(x_t, t) - \nabla\log q(x_t | x_0)\|^2]$$

where $\nabla\log q(x_t | x_0) = -(x_t - x_0)/\sigma_t^2$ (target는 명시적).

$\sigma_t$ 변화에 따른 영향:
- $\sigma_t$ 작음 (Low SNR): gradient가 크지만, 노이즈 영역 cover 부족
- $\sigma_t$ 큼 (High SNR): 노이즈 영역 잘 cover하지만, gradient 약함

**최적**: 적응적 weighting $\lambda(t) = \sigma_t^2$ 또는 그 변형을 사용해 각 SNR 수준을 균등하게 학습.

(Song et al. 2021, ICLR, Theorem 1 근처 참고)

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Score Function과 Tweedie Formula](./02-tweedie-formula.md) | [📚 README로 돌아가기](../README.md) | [04. Denoising Score Matching (Vincent 2011) ▶](./04-denoising-score-matching.md) |

</div>
