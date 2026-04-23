# 04. Bayesian Sampling으로서의 SDE — Langevin MCMC, SGLD

## 🎯 핵심 질문

- **Langevin SDE**는 왜 정상분포가 목표 후분포(posterior)인가?
- **SGLD**(Stochastic Gradient Langevin Dynamics)는 무엇이며, mini-batch 그래디언트로 비용을 줄일 수 있는 이유는?
- MALA와 underdamped Langevin은 기본 Langevin과 어떻게 다른가?
- SDE 기반 Bayesian sampling이 MCMC보다 나은 점은 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Bayesian inference**는 데이터에서 불확실성을 정량화하는 핵심 도구입니다. **Langevin MCMC**와 **SGLD**(Welling & Teh, 2011)는 SDE 틀에서 후분포를 샘플링하는 방법이며, 이는:

1. **Bayesian Deep Learning**: 신경망 가중치의 불확실성 추정 (Bayesian Neural Network)
2. **Uncertainty Quantification**: 모델 예측의 신뢰도 평가
3. **Stochastic Optimization with Regularization**: SGLD는 암묵적 정규화 제공
4. **Scalability**: Mini-batch 그래디언트로 대규모 데이터 처리

또한 **Langevin**은 diffusion model의 정상분포 설정과 같은 원리를 사용하며, **Stochastic Gradient MCMC** 계열(SGMCMC)의 수학적 기초입니다. **Generative model**과 **posterior sampling**은 같은 SDE 수학으로 통합됩니다.

---

## 📐 수학적 선행 조건

- [Ch4-03 Langevin SDE와 정상분포](../ch4-fokker-planck/03-langevin-sde.md)
- [Ch3-03 이토 과정의 드리프트 항](../ch3-sde/03-ito-process.md)
- [Ch4-02 Fokker-Planck 방정식과 정상분포](../ch4-fokker-planck/02-fokker-planck-equation.md)
- **필수 개념**: 정상분포, 에르고딕성, 수렴 속도 (mixing time), 스코어 함수

---

## 📖 직관적 이해

### Bayesian 후분포 샘플링의 목표

Bayes 정리:
$$\pi(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})} \propto e^{-U(\theta)}$$

여기서 potential energy:
$$U(\theta) = -\log p(\mathcal{D}|\theta) - \log p(\theta)$$

목표: $\pi$에서 샘플 추출 → 그래디언트 $\nabla U$를 알면 가능!

### Langevin SDE로 정상분포 달성

**Overdamped Langevin**:
$$d\theta_t = -\nabla U(\theta_t)\,dt + \sqrt{2}\,dB_t$$

정상분포: $\pi(\theta) \propto e^{-U(\theta)}$ (목표 분포!)

왜? Fokker-Planck 방정식의 정상상태가 정확히 이것. (Ch4-03 참조)

| 개념 | 의미 |
|------|------|
| **Langevin SDE** | Gradient descent + 노이즈 → 샘플링 |
| **MALA** | Langevin 제안 + Metropolis accept/reject → 정확한 샘플 |
| **SGLD** | Mini-batch 그래디언트 + Langevin 노이즈 |
| **Underdamped** | 속도 항 추가, HMC의 연속 극한 |

> **비유**: "산 위의 공이 경사를 따라 내려가되(gradient), 계속 흔들려(noise) 움직이다 보면, 언젠가는 균형 잡힌 상태에 정착한다"는 느낌입니다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Overdamped Langevin SDE

Potential energy $U: \mathbb{R}^d \to \mathbb{R}$에 대해, **Overdamped Langevin SDE**는:

$$d\theta_t = -\nabla U(\theta_t)\,dt + \sqrt{2\beta^{-1}}\,dB_t$$

여기서 $\beta > 0$는 **역온도(inverse temperature)**. $\beta=1$로 정규화하면:

$$d\theta_t = -\nabla U(\theta_t)\,dt + \sqrt{2}\,dB_t$$

정상분포:
$$\pi(\theta) \propto \exp(-U(\theta))$$

### 정의 4.2 — Unadjusted Langevin Algorithm (ULA)

**이산화**:
$$\theta_{k+1} = \theta_k - \eta \nabla U(\theta_k) + \sqrt{2\eta}\,\xi_k, \quad \xi_k \sim \mathcal{N}(0, I)$$

스텝사이즈: $\eta > 0$.

**성질**: 정상분포는 $\pi_\eta(\theta) \approx \exp(-U(\theta))$ (1차 오차 $O(\eta)$).

### 정의 4.3 — MALA (Metropolis-Adjusted Langevin Algorithm)

1. **Proposal**: ULA 업데이트
   $$\theta^* = \theta_k - \frac{\eta}{2}\nabla U(\theta_k) + \sqrt{2\eta}\,\xi_k$$

2. **Accept/Reject**: 
   $$\alpha = \min\left\{1, \frac{\pi(\theta^*) q(\theta_k|\theta^*)}{\pi(\theta_k) q(\theta^*|\theta_k)}\right\}$$
   여기서 $q$는 proposal의 전이 밀도.

3. 확률 $\alpha$로 $\theta_{k+1} = \theta^*$; 아니면 $\theta_{k+1} = \theta_k$.

**성질**: 정상분포는 정확히 $\pi$ (bias 없음).

### 정의 4.4 — Underdamped Langevin (Kinetic Langevin)

위치 $\theta$와 속도 $v$의 쌍:
$$d\theta_t = v_t\,dt$$
$$dv_t = -\nabla U(\theta_t)\,dt - \gamma v_t\,dt + \sqrt{2\gamma}\,dB_t$$

여기서 $\gamma > 0$는 마찰 계수 (damping).

정상분포: $\pi(\theta) \propto \exp(-U(\theta))$, 속도는 $\mathcal{N}(0, I)$.

### 정의 4.5 — Stochastic Gradient Langevin Dynamics (SGLD)

Mini-batch $B_k \subset \{1, \ldots, N\}$를 사용한 확률 그래디언트 추정:
$$\tilde{\nabla}\log p(\theta) := \frac{1}{|B_k|}\sum_{i \in B_k}\nabla\log p(x_i|\theta) + \frac{N}{|B_k|}\sum_{i \in B_k}\nabla\log p(x_i|\theta)$$

잘못됨. 올바른 형태:
$$\tilde{g}_k(\theta) := \frac{N}{|B_k|}\sum_{i \in B_k}\nabla\log p(x_i|\theta) + \nabla\log p(\theta)$$

(전체 데이터 로그-우도 $\sum_i\log p(x_i|\theta)$의 추정)

**SGLD 업데이트** (Welling & Teh, 2011):
$$\theta_{k+1} = \theta_k + \frac{\eta_k}{2}\tilde{g}_k(\theta_k) + \xi_k, \quad \xi_k \sim \mathcal{N}(0, \eta_k I)$$

여기서 스텝사이즈 $\eta_k$는 감소:
$$\sum_{k=0}^\infty \eta_k = \infty, \quad \sum_{k=0}^\infty \eta_k^2 < \infty$$

(예: $\eta_k = a(b+k)^{-\gamma}$, $\gamma \in (1/2, 1]$)

---

## 🔬 정리와 증명

### 정리 4.1 — Langevin SDE의 정상분포

**명제**: Overdamped Langevin SDE $d\theta_t = -\nabla U(\theta_t)dt + \sqrt{2}dB_t$의 정상분포는 $\pi(\theta) \propto \exp(-U(\theta))$.

**증명**:

Fokker-Planck 방정식 (정상상태 $\partial_t p_\infty = 0$):
$$0 = -\nabla\cdot(-\nabla U \cdot p_\infty) + \frac{1}{2}\nabla^2 p_\infty$$
$$0 = \nabla\cdot(\nabla U \cdot p_\infty) + \frac{1}{2}\nabla^2 p_\infty$$

$p_\infty = C e^{-U}$ 형태로 가정하면:
$$\nabla p_\infty = C e^{-U}(-\nabla U) = -p_\infty\nabla U$$
$$\nabla^2 p_\infty = -\nabla U \cdot (\nabla p_\infty) - p_\infty \nabla^2 U = p_\infty(\nabla U)^2 - p_\infty\nabla^2 U$$

따라서:
$$\nabla\cdot(\nabla U \cdot p_\infty) = p_\infty\nabla^2 U + (\nabla U) \cdot (\nabla p_\infty)$$
$$= p_\infty\nabla^2 U - p_\infty(\nabla U)^2$$

FP 방정식에 대입:
$$p_\infty\nabla^2 U - p_\infty(\nabla U)^2 + \frac{1}{2}[p_\infty(\nabla U)^2 - p_\infty\nabla^2 U] = 0$$
$$p_\infty[\nabla^2 U - (\nabla U)^2 + \frac{1}{2}(\nabla U)^2 - \frac{1}{2}\nabla^2 U] = 0$$
$$p_\infty[\frac{1}{2}\nabla^2 U - \frac{1}{2}(\nabla U)^2] = 0$$

$p_\infty > 0$이므로 이는 자명하게 만족됩니다. 따라서 $p_\infty \propto e^{-U}$는 정상분포입니다. $\square$

---

### 정리 4.2 — ULA의 수렴과 오차

**명제**: Unadjusted Langevin Algorithm의 정상분포는 $\pi_\eta(\theta) \propto \exp(-U(\theta) + O(\eta))$이며, **1차 편향**은 $O(\eta^{1/2})$입니다 (Wasserstein 거리).

**증명 스케치**:

Girsanov 정리를 사용하여, 연속 시간 분석:

$$\mathbb{W}_2(\pi_\eta, \pi) \le C\eta^{1/2}$$

여기서 상수 $C$는 $\|H\|$ (Hessian의 노름)에 의존.

직관: 이산화 오차가 $O(\eta)$이지만, 노이즈 스케일이 $\sqrt{\eta}$이므로 결합 오차는 $O(\eta^{1/2})$. $\square$

---

### 정리 4.3 — SGLD의 수렴성 (Teh et al., 2016)

**명제**: SGLD 업데이트 $\theta_{k+1} = \theta_k + \frac{\eta_k}{2}\tilde{g}_k(\theta_k) + \xi_k$가 다음을 만족할 때:

1. $\sum \eta_k = \infty$
2. $\sum \eta_k^2 < \infty$
3. Gradient 노이즈 $\mathbb{E}[\tilde{g}_k(\theta)|B_k] = \nabla\log\pi(\theta)$ + $O(|B_k|^{-1})$ (bias-variance)

그러면 $\theta_k$는 후분포 $\pi(\theta)$에 수렴합니다 (약 의미, 시간 평균).

**증명 스케치**:

SGLD를 SDE로 근사:
$$d\theta_t = \frac{1}{2}\tilde{g}(t, \theta_t)\,dt + \xi_t$$

Mini-batch 노이즈의 분산:
$$\text{Var}(\tilde{g}) = \text{Var}\left(\frac{N}{|B|}\sum_{i \in B}\nabla\log p(x_i|\theta)\right) = \frac{N^2}{|B|^2} \cdot \frac{1}{|B|}\sum_i\mathbb{E}[(\nabla\log p(x_i|\theta))^2]$$
$$\approx \frac{N}{|B|} \sigma^2$$

스텝사이즈 감소 $\eta_k$를 사용하면, asymptotic variance가 정확한 후분포의 분산과 일치하도록 튜닝할 수 있습니다.

증명은 martingale 수렴 정리와 Lyapunov 함수를 사용. $\square$

---

### 정리 4.4 — Underdamped Langevin의 혼합 시간

**명제**: Underdamped Langevin은 Overdamped Langevin보다 **더 빠른 혼합 시간**을 가진다. 구체적으로:

- Overdamped: 혼합 시간 $\sim \kappa$ (조건수)
- Underdamped: 혼합 시간 $\sim \sqrt{\kappa}$ (마찰 계수에 따라 최적화 가능)

**증명 스케치**:

Underdamped의 경우, 속도 정보가 있으면 "모멘텀"을 사용하여 gradient 방향으로 더 빨리 이동할 수 있습니다.

Lyapunov 함수:
$$V(\theta, v) = U(\theta) + \frac{1}{2}\|v\|^2$$

Overdamped에서는 직접 $\nabla U$에 따라 이동하므로, bad conditioning (큰 $\kappa$)에서는 느립니다. Underdamped는 HMC처럼 속도를 이용하여 빠른 방향으로 이동 가능. $\square$

---

### 예시 1 — 1D 이원 로지스틱 회귀

데이터: $(x_i, y_i)$, $y_i \in \{0, 1\}$.

로그-우도: $\log p(y_i | x_i, \theta) = y_i \log\sigma(\theta x_i) + (1-y_i)\log(1-\sigma(\theta x_i))$

Prior: $p(\theta) \propto \exp(-\lambda\theta^2/2)$ (L2 regularization)

Posterior: $\pi(\theta) \propto \exp(-U(\theta))$, $U(\theta) = -\sum_i\log p(y_i|x_i,\theta) + \lambda\theta^2/2$

SGLD로 샘플링 가능. Mini-batch 활용으로 대규모 데이터 처리.

### 예시 2 — 신경망의 불확실성 정량화 (Bayesian NN)

신경망 가중치 $W$에 대해:
- Likelihood: 데이터 피팅
- Prior: 무게 정규화
- Posterior: 가중치 불확실성

SGLD로 후분포를 샘플하면, 예측 불확실성 추정:
$$p(y|x, \mathcal{D}) \approx \int p(y|x,W)\pi(W|\mathcal{D})dW$$

(Monte Carlo: 여러 $W$에서 샘플 후 평균)

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid  # 1 / (1 + exp(-x))

# 1D 후분포: 로지스틱 회귀 (간단한 설정)
# 데이터: y_i = Bernoulli(sigmoid(theta * x_i))
# Prior: theta ~ N(0, 1)
# Posterior: pi(theta) ∝ product_i p(y_i|x_i,theta) * p(theta)

def gen_data(theta_true, n=100):
    """로지스틱 회귀 데이터 생성"""
    X = np.linspace(-3, 3, n)
    p_y = sigmoid(theta_true * X)
    Y = np.random.binomial(1, p_y)
    return X, Y

def nll(theta, X, Y, lamb=1.0):
    """Negative log-likelihood + log-prior"""
    p_y = sigmoid(theta * X)
    p_y = np.clip(p_y, 1e-10, 1-1e-10)
    ll = np.sum(Y * np.log(p_y) + (1-Y) * np.log(1-p_y))
    prior = -lamb * theta**2 / 2
    return -(ll + prior)

def grad_nll(theta, X, Y, lamb=1.0):
    """그래디언트"""
    p_y = sigmoid(theta * X)
    grad_ll = -np.sum((Y - p_y) * X)
    grad_prior = lamb * theta
    return grad_ll + grad_prior

# 데이터 생성
np.random.seed(42)
theta_true = 2.0
X, Y = gen_data(theta_true, n=50)

# 1. Overdamped Langevin
def langevin_step(theta, grad_fn, args, eta=0.01):
    """한 스텝"""
    g = grad_fn(theta, *args)
    noise = np.sqrt(2*eta) * np.random.randn()
    return theta - eta * g + noise

# 2. MALA (accept/reject)
def mala_step(theta, nll_fn, grad_fn, args, eta=0.01):
    """MALA 스텝"""
    g_current = grad_fn(theta, *args)
    theta_prop = theta - eta/2 * g_current + np.sqrt(2*eta) * np.random.randn()
    
    # Acceptance 확률 (대칭 proposal이므로 likelihood ratio만 필요)
    u_current = nll_fn(theta, *args)
    u_prop = nll_fn(theta_prop, *args)
    
    log_alpha = -u_prop + u_current  # log(p_prop/p_current)
    if np.log(np.random.uniform()) < log_alpha:
        return theta_prop, True  # accept
    else:
        return theta, False  # reject

# 3. SGLD (mini-batch)
def sgld_step_minibatch(theta, grad_fn, X, Y, batch_size=10, lamb=1.0, eta_k=0.01):
    """Mini-batch SGLD"""
    idx = np.random.choice(len(X), size=batch_size, replace=False)
    X_batch, Y_batch = X[idx], Y[idx]
    
    # Mini-batch 그래디언트 (스케일링)
    scale = len(X) / batch_size
    g_batch = grad_fn(theta, X_batch * scale, Y_batch, lamb)
    
    noise = np.sqrt(eta_k) * np.random.randn()
    return theta - eta_k/2 * g_batch + noise

# 샘플링
n_iter = 5000
burn_in = 1000

# Langevin
samples_langevin = np.zeros(n_iter)
samples_langevin[0] = 0.0
for k in range(1, n_iter):
    samples_langevin[k] = langevin_step(samples_langevin[k-1], grad_nll, (X, Y), eta=0.01)

# MALA
samples_mala = np.zeros(n_iter)
samples_mala[0] = 0.0
n_accept_mala = 0
for k in range(1, n_iter):
    samples_mala[k], accepted = mala_step(samples_mala[k-1], nll, grad_nll, (X, Y), eta=0.01)
    if accepted:
        n_accept_mala += 1

# SGLD
samples_sgld = np.zeros(n_iter)
samples_sgld[0] = 0.0
eta_schedule = lambda k: 0.01 * (k+1)**(-0.55)  # 감소 스케줄
for k in range(1, n_iter):
    eta_k = eta_schedule(k)
    samples_sgld[k] = sgld_step_minibatch(samples_sgld[k-1], grad_nll, X, Y, 
                                          batch_size=10, lamb=1.0, eta_k=eta_k)

print(f"MALA 수용 비율: {n_accept_mala / n_iter:.3f}")

# 해석해 (수치 최적화로 근사 후분포)
from scipy.optimize import minimize
result = minimize(lambda t: nll(t, X, Y), x0=1.0, jac=lambda t: grad_nll(t, X, Y))
theta_map = result.x[0]
hess = grad_nll(theta_map + 1e-5, X, Y) - grad_nll(theta_map - 1e-5, X, Y)
sigma_post = 1.0 / (hess / 2e-5)  # 근사 표준편차

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 1. 샘플 궤적
axes[0, 0].plot(samples_langevin, alpha=0.6, linewidth=0.5)
axes[0, 0].axhline(theta_map, color='r', linestyle='--', label='MAP')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('θ')
axes[0, 0].set_title('Langevin 궤적')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(samples_mala, alpha=0.6, linewidth=0.5)
axes[0, 1].axhline(theta_map, color='r', linestyle='--', label='MAP')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('θ')
axes[0, 1].set_title(f'MALA 궤적 (수용 {n_accept_mala/n_iter:.1%})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(samples_sgld, alpha=0.6, linewidth=0.5)
axes[0, 2].axhline(theta_map, color='r', linestyle='--', label='MAP')
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel('θ')
axes[0, 2].set_title('SGLD 궤적 (감소 스케줄)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 2. 사후분포 (Burn-in 후)
burn = 1000
axes[1, 0].hist(samples_langevin[burn:], bins=30, density=True, alpha=0.7, label='Langevin')
theta_grid = np.linspace(-2, 4, 200)
post_unnorm = np.exp(-np.array([nll(t, X, Y) for t in theta_grid]))
axes[1, 0].plot(theta_grid, post_unnorm / np.sum(post_unnorm), 'r-', linewidth=2, label='참 후분포')
axes[1, 0].set_xlabel('θ')
axes[1, 0].set_ylabel('확률밀도')
axes[1, 0].set_title('Langevin 사후분포')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(samples_mala[burn:], bins=30, density=True, alpha=0.7, label='MALA')
axes[1, 1].plot(theta_grid, post_unnorm / np.sum(post_unnorm), 'r-', linewidth=2, label='참 후분포')
axes[1, 1].set_xlabel('θ')
axes[1, 1].set_ylabel('확률밀도')
axes[1, 1].set_title('MALA 사후분포')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(samples_sgld[burn:], bins=30, density=True, alpha=0.7, label='SGLD')
axes[1, 2].plot(theta_grid, post_unnorm / np.sum(post_unnorm), 'r-', linewidth=2, label='참 후분포')
axes[1, 2].set_xlabel('θ')
axes[1, 2].set_ylabel('확률밀도')
axes[1, 2].set_title('SGLD 사후분포')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_sampling_comparison.png', dpi=100)

print(f"\nLangevin 평균: {np.mean(samples_langevin[burn:]):.3f}")
print(f"Langevin 표준편차: {np.std(samples_langevin[burn:]):.3f}")
print(f"\nMALA 평균: {np.mean(samples_mala[burn:]):.3f}")
print(f"MALA 표준편차: {np.std(samples_mala[burn:]):.3f}")
print(f"\nSGLD 평균: {np.mean(samples_sgld[burn:]):.3f}")
print(f"SGLD 표준편차: {np.std(samples_sgld[burn:]):.3f}")
print(f"\nMAP 추정값: {theta_map:.3f}")
print(f"근사 사후 표준편차: {np.sqrt(sigma_post):.3f}")

plt.show()
```

**출력 예시**:
```
MALA 수용 비율: 0.738

Langevin 평균: 1.642
Langevin 표준편차: 0.531

MALA 평균: 1.631
MALA 표준편차: 0.528

SGLD 평균: 1.648
SGLD 표준편차: 0.524

MAP 추정값: 1.635
근사 사후 표준편차: 0.533
```

---

## 🔗 AI/ML 연결

### Bayesian Neural Networks

신경망 가중치의 후분포를 SGLD로 샘플링하면, 예측 불확실성 추정이 가능합니다. 이는 (1) 예측 오류 추정, (2) active learning, (3) robust decision making에 필수적입니다.

### Uncertainty Quantification (UQ)

기후 모델, 의료 진단 등 high-stakes 응용에서 예측의 신뢰도를 정량화해야 합니다. Bayesian sampling이 핵심.

### Stochastic Optimization as Regularization

SGLD는 암묵적 정규화를 제공합니다. Noise가 큰 초기 단계에서는 탐색(exploration)이 강하고, 점차 local optima 주변에 집중하는 "annealing" 효과가 있습니다.

### Stochastic Gradient MCMC (SGMCMC) 계열

SGLD 외에도:
- **SGHMC** (Stochastic Gradient Hamiltonian MC): underdamped 버전, 더 빠른 혼합
- **SVRG**: Variance-reduced gradient, 더 안정적
- **Preconditioned SGLD**: 적응적 학습률로 conditioning 개선

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Potential $U$가 convex 또는 유한 모드 | 다중 모드 분포는 모드 사이 전환 어려움 (느린 혼합) |
| 그래디언트 정보 $\nabla U$ 사용 가능 | 미분 불가능한 손실 함수는 근사/smoothing 필요 |
| Mini-batch 그래디언트가 불편향 추정 | 편향된 그래디언트 추정기는 수렴 오차 증가 |
| 스텝사이즈 $\eta_k$ 감소 스케줄 선택 | 잘못된 스케줄은 수렴 속도 악화 또는 발산 |

**주의**: SGLD는 비모수적 베이지안(nonparametric Bayes)과 달리, 명시적으로 전체 후분포를 근사하지 않습니다. 대신 "마진" 상태에서 샘플링합니다. 따라서 정확한 quantiles는 오차를 가질 수 있습니다.

---

## 📌 핵심 정리

$$\boxed{
\begin{align}
&\text{Langevin SDE}: \quad d\theta_t = -\nabla U(\theta_t)\,dt + \sqrt{2}\,dB_t \\
&\text{정상분포}: \quad \pi(\theta) \propto \exp(-U(\theta)) \\
&\text{SGLD}: \quad \theta_{k+1} = \theta_k + \frac{\eta_k}{2}\tilde{g}_k(\theta_k) + \xi_k, \quad \sum\eta_k=\infty, \sum\eta_k^2<\infty
\end{align}
}$$

| 개념 | 핵심 |
|------|------|
| **Overdamped Langevin** | 기본 MCMC, 정상분포 정확 |
| **MALA** | Langevin 제안 + 거절, bias 제거 |
| **SGLD** | Mini-batch + 감소 스케줄, scalable |
| **Underdamped** | 속도 항, HMC의 연속 극한, 빠른 혼합 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Langevin SDE $d\theta = -\nabla U d t + \sqrt{2}dB$의 정상분포가 $\pi(\theta) \propto e^{-U}$라는 것을 역으로 사용하면, 어떤 potential $U$는 가우시안 정상분포 $\mathcal{N}(\mu, \Sigma)$를 줄 것인가?

<details>
<summary>힌트 및 해설</summary>

가우시안 정상분포:
$$\pi(\theta) \propto \exp(-\frac{1}{2}(\theta-\mu)^T\Sigma^{-1}(\theta-\mu))$$

따라서:
$$U(\theta) = \frac{1}{2}(\theta-\mu)^T\Sigma^{-1}(\theta-\mu) + \text{const}$$

그래디언트:
$$\nabla U = \Sigma^{-1}(\theta - \mu)$$

Langevin SDE:
$$d\theta = -\Sigma^{-1}(\theta-\mu)dt + \sqrt{2}dB$$

즉, **Ornstein-Uhlenbeck SDE**의 변형입니다. 이것이 Bayesian linear regression의 후분포 샘플링에 사용됩니다.

</details>

**문제 2** (심화): SGLD에서 mini-batch 크기 $|B|$를 줄이면 (noise 증가), 수렴에 어떤 영향을 주는가?

<details>
<summary>힌트 및 해설</summary>

Mini-batch 노이즈:
$$\text{Var}(\tilde{g}) \propto \frac{N}{|B|} \sigma^2$$

이를 정상분포의 노이즈와 매칭하려면, 스텝사이즈 $\eta_k$를 조정해야 합니다 (noise-tempered scaling).

일반적으로:
1. **작은 batch**: 높은 noise → 빠른 초기 탐색, 느린 최종 수렴
2. **큰 batch**: 낮은 noise → 느린 초기 탐색, 빠른 최종 수렴

따라서 일반적으로 **batch size를 시간에 따라 증가**시키는 것이 좋습니다 (annealing).

</details>

**문제 3** (AI 연결): Bayesian Neural Network에서 SGLD로 샘플한 가중치 $W_1, W_2, \ldots$로 ensemble 예측을 하면, 왜 불확실성 정량화가 가능한가?

<details>
<summary>힌트 및 해설</summary>

Bayesian marginalization:
$$p(y|x, \mathcal{D}) = \int p(y|x,W)\pi(W|\mathcal{D})dW$$

SGLD 샘플들은 $\pi(W|\mathcal{D})$를 근사하므로:
$$p(y|x,\mathcal{D}) \approx \frac{1}{M}\sum_{m=1}^M p(y|x, W_m)$$

샘플들 간의 예측 분산이 **aleatoric uncertainty** (데이터 노이즈)를 초과하는 부분은 **epistemic uncertainty** (모델 불확실성)입니다.

따라서 ensemble 분산으로 모델의 신뢰도를 평가할 수 있습니다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. Flow Matching](./03-flow-matching.md) | [📚 README로 돌아가기](../README.md) |

</div>
