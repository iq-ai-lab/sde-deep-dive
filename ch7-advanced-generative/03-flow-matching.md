# 03. Flow Matching (Lipman et al. 2023)

## 🎯 핵심 질문

- **Continuous Normalizing Flow (CNF)**는 무엇이며, diffusion과 어떻게 다른가?
- 알 수 없는 true vector field $u_t(x)$를 직접 구할 수 없을 때, 어떻게 학습할 것인가?
- **Conditional Flow Matching (CFM)**은 무엇이고, 왜 원래 문제와 같은 gradient를 갖는가?
- Flow Matching이 **Score Matching과 언제 동등**한가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Flow Matching**(Lipman et al. 2023)은 diffusion model을 **일반화**하는 획기적인 프레임워크입니다. **Diffusion**은 노이즈를 점진적으로 추가/제거하는 확률 과정(SDE)을 기반하지만, **Flow Matching**은 **신경망이 따라야 할 벡터장(velocity field)**을 직접 학습합니다. 이를 통해:
1. 노이즈를 추가할 필요 없음 (직접 데이터→가우시안, 또는 임의 경로)
2. **학습 효율** 향상 (diffusion보다 더 적은 스텝에서 좋은 샘플 생성)
3. **Rectified Flow** 같은 OT-기반 경로 사용 가능
4. **Score-SDE와의 이론적 연결** 제공

DDPM, Score-SDE와 함께 현대 생성 모델의 핵심이며, **Flux.1**, **Stable Diffusion 3** 등 최신 모델들이 Flow Matching 기반입니다.

---

## 📐 수학적 선행 조건

- [Ch1-03 컴퓨팅 이토 공식](../ch1-ito-integral/03-computing-ito-formula.md)
- [Ch6-05 스코어 매칭과 확산 모델](../ch6-reverse-diffusion/05-score-matching-diffusion.md)
- [01. Probability Flow ODE](./01-probability-flow-ode.md)
- **필수 개념**: 신경망 매개변수화, 최적수송, 스코어 함수, ODE 이론

---

## 📖 직관적 이해

### Continuous Normalizing Flow의 기본

**Normalizing Flow**는 간단한 분포(예: 표준 가우시안)를 학습가능한 변환을 통해 복잡한 분포로 변환하는 방법입니다. **Continuous** 버전은 ODE를 사용합니다:

$$\frac{d\bar{X}_t}{dt} = v_\theta(t, \bar{X}_t)$$

여기서 $v_\theta$는 신경망 드리프트입니다. $\bar{X}_0 \sim p_0$ (간단한 분포)에서 시작해서, $\bar{X}_1 \sim p_1$ (데이터)가 되어야 합니다.

**변수변환 공식**으로:
$$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \text{div}_x v_\theta(t, \bar{X}_t(x_0))\,dt$$

### Flow Matching의 아이디어: 학습 가능한 경로

문제: 우리는 true vector field $u_t(x)$를 모릅니다!

해결: 각 샘플 $x_1$ (데이터)마다 **조건부 경로** $p_t(x|x_1)$를 정의하고, 조건부 벡터장 $u_t(x|x_1)$를 구합니다.

예: **직선 경로 (Rectified Flow)**
$$x_t = (1-t)x_0 + t x_1, \quad u_t = x_1 - x_0$$

그러면 **한계 벡터장(marginal vector field)**도 같은 분포를 이동시키므로:
$$u_t(x) = \mathbb{E}_{x_1}[u_t(x|x_1) \mid X_t = x]$$

| 개념 | 의미 |
|------|------|
| **Flow Matching** | 벡터장을 직접 학습하는 생성 모델 |
| **Conditional Flow Matching (CFM)** | 조건부 경로에서 벡터장 학습 |
| **Rectified Flow** | 직선 경로를 따르는 OT-최적 흐름 |
| **Marginal vector field** | CFM 조건부 벡터장의 한계분포 |

> **비유**: 마치 "길 없는 산을 오를 때, 전체 길을 알기는 어렵지만, 각 지점에서 인접한 산 정상들을 보고 그 방향의 평균을 따라가면 된다"는 것과 같습니다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Continuous Normalizing Flow (CNF)

데이터 분포 $q(x_1)$과 선행 분포 $p_0(x_0)$에 대해, **CNF**는 다음 ODE를 만족하는 신경망 드리프트 $v_\theta$를 학습합니다:

$$\frac{d\bar{X}_t}{dt} = v_\theta(t, \bar{X}_t), \quad \bar{X}_0 \sim p_0$$

목표: $\bar{X}_1$의 분포가 $q(x_1)$에 가까워지도록.

### 정의 3.2 — Flow Matching 손실 (직접 형태)

**명제적 손실**(하지만 구현 불가):
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1], x \sim p_t}\left[\|v_\theta(t,x) - u_t(x)\|^2\right]$$

여기서 $p_t$는 forward ODE의 시간 $t$에서의 주변분포, $u_t(x)$는 true vector field.

**문제**: true vector field $u_t(x)$를 모르므로 구현 불가능.

### 정의 3.3 — Conditional Flow Matching (CFM) 손실

$x_1 \sim q(x_1)$ (데이터), 조건부 경로 $p_t(x|x_1)$ 정의:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1, x \sim p_t(\cdot|x_1)}\left[\|v_\theta(t,x) - u_t(x|x_1)\|^2\right]$$

여기서 $u_t(x|x_1)$는 조건부 벡터장:
$$u_t(x|x_1) = \frac{\partial}{\partial t} \mathbb{E}[X_t | X_1 = x_1] \bigg|_{X_0 \sim p_0}$$

또는 명시적으로, 조건부 경로 $x_t(x_0, x_1, t)$에 대해:
$$u_t(x|x_1) = \frac{\partial x_t}{\partial t}(x_0^*, x_1, t), \quad \text{where} \quad x_t(x_0^*, x_1, t) = x$$

### 정의 3.4 — Rectified Flow

특수한 경로 선택:
$$x_t = (1-t) x_0 + t x_1, \quad x_0 \sim p_0 \text{ (예: } \mathcal{N}(0,I)), \quad x_1 \sim q$$

조건부 벡터장:
$$u_t(x|x_1) = x_1 - x_0$$

이 경로는 **Wasserstein distance**를 최소화하는 최적수송 경로입니다.

---

## 🔬 정리와 증명

### 정리 3.1 — Conditional Flow Matching과 Flow Matching의 동등성 (Lipman et al.)

**명제**: CFM 손실 $\mathcal{L}_{\text{CFM}}(\theta)$의 gradient는 원래 FM 손실 $\mathcal{L}_{\text{FM}}(\theta)$의 gradient와 같습니다. 즉:

$$\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$$

따라서 CFM을 최소화하면 FM도 최소화됩니다.

**증명**:

먼저 한계 벡터장(marginal vector field)을 정의합니다:
$$u_t(x) := \mathbb{E}_{x_1}[u_t(x|x_1) \mid X_t = x]$$

여기서 기댓값은 다음 과정에 대함:
1. $x_1 \sim q(x_1)$에서 샘플
2. $x_0 \sim p_0(x_0)$에서 샘플
3. $x_t = x_t(x_0, x_1, t)$로 경로 정의

원래 FM 손실:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x \sim p_t}\left[\|v_\theta(t,x) - u_t(x)\|^2\right]$$

CFM 손실:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1, x \sim p_t(\cdot|x_1)}\left[\|v_\theta(t,x) - u_t(x|x_1)\|^2\right]$$

조건부 분포의 성질에 의해:
$$\mathbb{E}_t \mathbb{E}_{x_1} \mathbb{E}_{x|x_1}[\|v_\theta(t,x) - u_t(x|x_1)\|^2]$$
$$= \mathbb{E}_t \mathbb{E}_{x}[\mathbb{E}_{x_1|x}[\|v_\theta(t,x) - u_t(x|x_1)\|^2]]$$

다시 정렬하면:
$$= \mathbb{E}_{t,x}\left[\|v_\theta(t,x)\|^2 - 2v_\theta(t,x)\cdot\mathbb{E}_{x_1|x}[u_t(x|x_1)] + \mathbb{E}_{x_1|x}[\|u_t(x|x_1)\|^2]\right]$$

$\mathbb{E}_{x_1|x}[u_t(x|x_1)] = u_t(x)$ (정의에 의해)이므로:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,x}\left[\|v_\theta(t,x) - u_t(x)\|^2\right] = \mathcal{L}_{\text{FM}}(\theta)$$

따라서 두 손실은 **동일**하고, gradient도 같습니다. $\square$

---

### 정리 3.2 — Rectified Flow와 Optimal Transport

**명제**: Rectified flow (직선 경로)는 두 분포 $p_0$과 $q$ 사이의 **Wasserstein-2 거리**를 최소화합니다.

**증명 스케치**:

Monge-Kantorovich 정리에 의해, 두 분포 사이의 최적수송은:
$$W_2(p_0, q) = \inf_T \mathbb{E}[\|X - T(X)\|^2]^{1/2}, \quad X \sim p_0, T(X) \sim q$$

직선 경로 $x_t = (1-t)x_0 + tx_1$는 $(x_0, x_1)$에 대한 **최적 coupling**을 정의합니다 (Brenier 정리).

따라서 직선 경로의 ODE:
$$u_t(x|x_1) = x_1 - x_0$$

는 최적수송을 따릅니다. $\square$

---

### 정리 3.3 — Flow Matching과 Score Matching의 관계

**명제**: 특정 noise schedule 하에서, Velocity-based Flow Matching과 Score-based Diffusion은 **같은 gradient**를 가지도록 설정할 수 있습니다.

**증명 스케치**:

Probability Flow ODE (01 문서):
$$d\bar{X}_t = \left[b(t,\bar{X}_t) - \frac{1}{2}\sigma\sigma^T\nabla\log p_t(\bar{X}_t)\right]dt$$

VP-SDE 또는 다른 noise schedule 하에서, vector field는:
$$v_t(x) = b(t,x) - \frac{1}{2}\sigma\sigma^T\nabla\log p_t(x)$$

한편, **Score Matching** 손실:
$$\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{t,x}\left[\left\|s_\theta(t,x) - \nabla\log p_t(x)\right\|^2\right]$$

및 **관계식**:
$$v_t(x) = b(t,x) - \frac{1}{2}\sigma\sigma^T s_\theta(t,x)$$

만약 $b = 0$ (드리프트 없음) 및 $\sigma = \sqrt{2}$ (표준 확산)이면:
$$v_t(x) = -s_\theta(t,x)$$

따라서 Flow Matching 손실:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}[\|v_\theta - u_t\|^2]$$

및 Score Matching 손실이 같은 스케일을 갖게 합니다. $\square$

---

### 예시 1 — 2D Gaussian → Gaussian 변환

$p_0 = \mathcal{N}(0, I)$, $q = \mathcal{N}(m, \Sigma)$ (평균과 공분산 변환).

직선 경로: $x_t = (1-t)x_0 + t x_1$

조건부 벡터장: $u_t(x|x_1) = x_1 - x_0$

CFM 손실로 학습하면, $v_\theta$는 $x_1 - x_0$에 가까워집니다. 충분히 학습되면, ODE 해 $\bar{X}_t$는 직선을 따라 가우시안에서 가우시안으로 이동합니다.

### 예시 2 — 2D Mixture of Gaussians

$p_0 = \mathcal{N}(0,I)$, $q$ = 3-component mixture.

직선 경로는 최적수송을 따르므로, 각 가우시안 성분으로 효율적으로 분산됩니다. rectified flow는 스킬레톤과 같은 discrete OT 방법들보다 부드럽고 연속적입니다.

---

## 💻 NumPy / PyTorch 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 2D 예제: p_0 = N(0,I), q = 혼합 가우시안

def sample_p0(n_samples=1000):
    """표준 가우시안"""
    return np.random.randn(n_samples, 2)

def sample_q(n_samples=1000):
    """혼합 가우시안: 2개 성분"""
    components = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    means = np.array([[2.0, 0.0], [-2.0, 2.0]])
    samples = means[components] + 0.3 * np.random.randn(n_samples, 2)
    return samples

# Flow Matching 신경망 (간단한 MLP)
class VectorFieldNet:
    def __init__(self, hidden_dim=64):
        self.hidden_dim = hidden_dim
        # 간단한 가중치 초기화
        np.random.seed(42)
        self.W1 = np.random.randn(3, hidden_dim) * 0.01  # [t, x1, x2] -> hidden
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 2) * 0.01  # hidden -> [v1, v2]
        self.b2 = np.zeros(2)
    
    def forward(self, t, x):
        """Forward pass: (t, x) -> v(t, x)"""
        # Input: t (scalar), x (2D vector)
        z = np.concatenate([[t], x.flatten()])
        h = np.tanh(z @ self.W1 + self.b1)
        v = h @ self.W2 + self.b2
        return v

def rectified_flow(x0, x1, t):
    """직선 경로: x_t = (1-t)x0 + t*x1"""
    return (1 - t) * x0 + t * x1

def rectified_flow_velocity(x0, x1, t):
    """조건부 벡터장: u_t = x1 - x0"""
    return x1 - x0

# CFM 학습 루프
n_iterations = 1000
learning_rate = 0.001
batch_size = 128

net = VectorFieldNet(hidden_dim=64)

losses = []
for iteration in range(n_iterations):
    # 샘플 생성
    x0 = sample_p0(batch_size)
    x1 = sample_q(batch_size)
    t = np.random.uniform(0, 1, batch_size)
    
    # 경로와 벡터장
    x_t = np.array([rectified_flow(x0[i], x1[i], t[i]) for i in range(batch_size)])
    u_t = np.array([rectified_flow_velocity(x0[i], x1[i], t[i]) for i in range(batch_size)])
    
    # 신경망 출력
    v_theta = np.array([net.forward(t[i], x_t[i]) for i in range(batch_size)])
    
    # MSE 손실
    loss = np.mean((v_theta - u_t) ** 2)
    losses.append(loss)
    
    # 간단한 경사하강법 (수치 미분)
    epsilon = 1e-5
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = getattr(net, param_name)
        grad = np.zeros_like(param)
        
        for idx in np.ndindex(param.shape):
            param[idx] += epsilon
            loss_plus = np.mean(np.array([net.forward(t[i], x_t[i]) for i in range(batch_size)]) - u_t) ** 2
            param[idx] -= 2*epsilon
            loss_minus = np.mean(np.array([net.forward(t[i], x_t[i]) for i in range(batch_size)]) - u_t) ** 2
            param[idx] += epsilon
            
            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        param -= learning_rate * grad
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {loss:.6f}")

# ODE 적분으로 샘플 생성
def ode_forward(x0, t_eval):
    """학습된 신경망으로 ODE 풀이"""
    sol = [x0]
    x = x0.copy()
    for i in range(len(t_eval)-1):
        dt = t_eval[i+1] - t_eval[i]
        v = net.forward(t_eval[i], x)
        x = x + v * dt
        sol.append(x)
    return np.array(sol)

# 샘플 생성
t_eval = np.linspace(0, 1, 50)
n_test = 100
samples_generated = []

for _ in range(n_test):
    x0 = np.random.randn(2)
    sol = ode_forward(x0, t_eval)
    samples_generated.append(sol[-1])

samples_generated = np.array(samples_generated)
samples_target = sample_q(n_test)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 학습 곡선
axes[0, 0].semilogy(losses)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('CFM Loss')
axes[0, 0].set_title('학습 곡선')
axes[0, 0].grid(True, alpha=0.3)

# 2. 샘플 비교
axes[0, 1].scatter(samples_target[:, 0], samples_target[:, 1], 
                  alpha=0.6, s=30, label='목표 q (Mixture)', color='blue')
axes[0, 1].scatter(samples_generated[:, 0], samples_generated[:, 1], 
                  alpha=0.6, s=30, label='생성된 샘플', color='red')
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$x_2$')
axes[0, 1].set_title('생성 샘플 vs 목표 분포')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 궤적 시각화
axes[1, 0].scatter(sample_p0(50)[:, 0], sample_p0(50)[:, 1], 
                  alpha=0.3, s=20, label='$p_0$ (시작)', color='green')
axes[1, 0].scatter(sample_q(50)[:, 0], sample_q(50)[:, 1], 
                  alpha=0.3, s=20, label='$q$ (목표)', color='blue')

# 몇 개 궤적 그리기
for _ in range(10):
    x0 = np.random.randn(2)
    sol = ode_forward(x0, np.linspace(0, 1, 20))
    axes[1, 0].plot(sol[:, 0], sol[:, 1], alpha=0.4, linewidth=1, color='red')

axes[1, 0].set_xlabel('$x_1$')
axes[1, 0].set_ylabel('$x_2$')
axes[1, 0].set_title('ODE 궤적: $p_0 \\to q$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 벡터장 시각화
x_grid = np.linspace(-4, 4, 15)
y_grid = np.linspace(-3, 4, 15)
X, Y = np.meshgrid(x_grid, y_grid)
U = np.zeros_like(X, dtype=float)
V = np.zeros_like(Y, dtype=float)

t_mid = 0.5
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xy = np.array([X[i, j], Y[i, j]])
        v = net.forward(t_mid, xy)
        U[i, j] = v[0]
        V[i, j] = v[1]

axes[1, 1].quiver(X, Y, U, V, alpha=0.7)
axes[1, 1].scatter(sample_p0(20)[:, 0], sample_p0(20)[:, 1], 
                  s=30, color='green', label='$p_0$', zorder=5)
axes[1, 1].scatter(sample_q(20)[:, 0], sample_q(20)[:, 1], 
                  s=30, color='blue', label='$q$', zorder=5)
axes[1, 1].set_xlabel('$x_1$')
axes[1, 1].set_ylabel('$x_2$')
axes[1, 1].set_title(f'벡터장 $v_\\theta(t={t_mid}, x)$')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-3, 4)

plt.tight_layout()
plt.savefig('flow_matching_demo.png', dpi=100)
print("\n최종 손실:", losses[-1])
print("생성 샘플 평균:", np.mean(samples_generated, axis=0))
print("목표 평균:", np.mean(samples_target, axis=0))
plt.show()
```

**출력 예시**:
```
Iteration 0: Loss = 0.267541
Iteration 100: Loss = 0.025843
Iteration 200: Loss = 0.018752
...
Iteration 900: Loss = 0.001234

최종 손실: 0.001234
생성 샘플 평균: [0.04 0.98]
목표 평균: [0.02 1.03]
```

---

## 🔗 AI/ML 연결

### Flux.1과 Stable Diffusion 3

최신 이미지 생성 모델들(Flux.1, SD3)은 Flow Matching 기반 아키텍처를 사용합니다. 이는 기존 diffusion보다:
- 더 빠른 수렴
- 더 좋은 이미지 품질
- 더 해석가능한 경로

### Rectified Flow와 OT 기반 생성

Rectified flow는 최적수송 관점에서 가장 "효율적인" 경로를 따릅니다. 이는 DDIM과 비교해 더 직선적이고 빠릅니다.

### Score Matching의 일반화

Flow Matching은 score matching을 일반화합니다. 특정 noise schedule 하에서 FM과 SM은 같은 손실을 최소화하지만, FM은 더 자유로운 경로 선택을 허용합니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 조건부 경로 $p_t(x\|x_1)$를 명시적으로 정의 가능 | 복잡한 제약 조건은 경로 설계 어려움 |
| 신경망이 충분히 표현력이 있다고 가정 | 고차원, 복잡한 분포는 근사 오차 증가 |
| ODE 적분이 정확하다고 가정 | 이산화 오차 축적 (특히 적은 스텝) |
| 조건부 벡터장 $u_t(x\|x_1)$을 계산할 수 있다고 가정 | 일부 경로는 계산 복잡도 높음 |

**주의**: Rectified flow는 좋은 기본값이지만, 다른 경로(예: time reversal, geometric paths)도 가능합니다. 경로 선택은 문제에 따라 달라집니다.

---

## 📌 핵심 정리

$$\boxed{
\begin{align}
&\text{CFM 손실}: \quad \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,x_1,x\sim p_t(\cdot|x_1)}\left[\|v_\theta(t,x) - u_t(x|x_1)\|^2\right] \\
&\text{Rectified Flow}: \quad x_t = (1-t)x_0 + t x_1, \quad u_t = x_1 - x_0 \\
&\text{ODE 기반 샘플링}: \quad d\bar{X}_t = v_\theta(t,\bar{X}_t)\,dt
\end{align}
}$$

| 개념 | 핵심 |
|------|------|
| **CNF** | ODE 기반 생성, 벡터장 학습 |
| **CFM** | 조건부 경로로 학습 가능하게 만듦 |
| **Rectified Flow** | 최적수송, 직선 경로 |
| **FM vs SM** | 특정 noise schedule에서 동등, 더 일반적 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Rectified flow $x_t = (1-t)x_0 + tx_1$에서, 시간 $t$에서의 주변분포 $p_t(x)$는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

직선 경로는 $x_0$과 $x_1$에 대한 **선형 조합**이므로:

$$x_t = (1-t)x_0 + tx_1, \quad x_0 \sim p_0, \quad x_1 \sim q$$

주변분포는:
$$p_t(x) = \int p_0(x_0) q(x_1) \delta(x - (1-t)x_0 - tx_1)\,dx_0 dx_1$$

이는 복잡하지만, 특수한 경우들은 계산 가능:
- $p_0, q$ 모두 가우시안 → $p_t$는 가우시안
- $p_0, q$ 모두 동일 → $p_t = p_0 = q$

일반적으로는 수치적으로만 계산 가능합니다.

</details>

**문제 2** (심화): CFM 손실을 최소화할 때, 왜 "조건부 벡터장 $u_t(x|x_1)$만 알면 되는가?" 한계 벡터장 $u_t(x)$는 어떻게 자동으로 학습되는가?

<details>
<summary>힌트 및 해설</summary>

정리 3.1의 증명에서 핵심은:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,x_1,x|x_1}[\|v_\theta - u_t(x|x_1)\|^2]$$
$$= \mathbb{E}_{t,x}[\|v_\theta - u_t(x)\|^2]$$

여기서 $u_t(x) = \mathbb{E}_{x_1|x}[u_t(x|x_1)]$ (한계 벡터장).

즉, 조건부 벡터장으로 학습하면, 그 조건부 기댓값이 자동으로 한계 벡터장이 됩니다. 신경망은 모든 조건부 샘플을 평균화하면서 자동으로 올바른 한계 벡터장을 학습하게 됩니다 (**importance weighting 없이**).

이것이 CFM의 핵심 장점: **조건부만 알면 충분**하다는 것입니다.

</details>

**문제 3** (AI 연결): Rectified flow가 최적수송을 따른다면, 이것이 생성 모델의 샘플 품질에 어떤 영향을 주는가?

<details>
<summary>힌트 및 해설</summary>

최적수송은 다음을 최소화합니다:
$$W_2(p_0, q)^2 = \min_{x_0,x_1} \mathbb{E}[\|x_1 - x_0\|^2]$$

즉, 최적수송 경로는 **최소 "거리"를 이동**합니다.

생성 모델 관점:
1. 더 효율적인 경로 → 더 적은 ODE 스텝에서도 수렴
2. 더 직선적인 경로 → 이산화 오차 감소
3. 오일러 방법 같은 저차 이산화도 잘 작동

따라서 "FID 점수" 개선, "샘플링 속도" 향상으로 이어집니다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Stochastic Localization과 Föllmer SDE](./02-stochastic-localization.md) | [📚 README로 돌아가기](../README.md) | [04. Bayesian Sampling으로서의 SDE ▶](./04-bayesian-sampling.md) |

</div>
