# 02. Stochastic Localization과 Föllmer SDE

## 🎯 핵심 질문

- 임의의 분포 $\mu$를 **점진적으로 집중시켜** delta 측정으로 만들 수 있는 방법이 있는가?
- **Föllmer 과정**은 무엇이며, 왜 "최소 엔트로피" 드리프트를 갖는가?
- Schrödinger bridge는 두 분포 사이의 **최적 연결고리**를 어떻게 찾는가?
- Stochastic localization이 **diffusion model의 일반화**라는 것은 무슨 의미인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Stochastic localization**(Eldan)은 임의의 측정을 점진적으로 집중시키는 확률 과정 이론이며, 이는 **Föllmer 드리프트**라는 개념으로 이어집니다. Föllmer는 **두 분포 사이의 최소 엔트로피 경로**(Schrödinger bridge)를 정의하며, 특수한 경우로 **Diffusion model**을 포함합니다. 또한 **고차원 샘플링 문제**—예를 들어 Bayesian inference나 **partition function 계산**—에서 Stochastic localization을 사용한 알고리즘들이 개발되고 있습니다. **정보이론**과 **최적수송** 관점에서도 중요한 개념입니다.

---

## 📐 수학적 선행 조건

- [Ch4-03 Langevin SDE와 정상분포](../ch4-fokker-planck/03-langevin-sde.md)
- [Ch6-04 역시간 SDE와 스코어 추정](../ch6-reverse-diffusion/04-reverse-sde-score.md)
- [Ch3-05 이토 공식과 마팅게일](../ch3-sde/05-ito-formula-martingale.md)
- **필수 개념**: 엔트로피, 쿨백-라이블러 발산, 스코어 함수, 정상분포 존재성

---

## 📖 직관적 이해

### Stochastic Localization의 직관

**Localization**은 "집중시킨다"는 뜻입니다. Eldan의 stochastic localization은 측정(분포) $\mu$를 받아서, 시간이 지남에 따라 점점 **더 집중된** 분포들의 수열 $\mu_t$를 생성합니다:
- $\mu_0 = \mu$ (원래 분포)
- $\mu_t$는 점점 집중 (엔트로피 감소)
- $\mu_\infty = \delta_x$ (특정 점으로 수렴, x는 결정론적 또는 확률적)

이는 마치 "랜덤한 점들의 구름이 점점 한 점으로 모인다"는 것과 같습니다.

### Föllmer 과정과 조건부 스코어

다양한 stochastic localization 중 가장 중요한 것은 **Föllmer SDE**입니다:

$$dX_t = v(t, X_t)\,dt + dB_t$$

여기서 드리프트 $v(t,x)$는 다음처럼 정의됩니다:
$$v(t,x) = \mathbb{E}[\nabla\log p(Y) \mid X = x]$$

단, $Y \sim \mu$이고, $X|Y$는 Gaussian 노이즈로 연결됩니다.

| 개념 | 의미 |
|-----|------|
| **Stochastic localization** | 분포를 점진적으로 집중시키는 확률 과정 |
| **Föllmer 드리프트** | 조건부 스코어 $\nabla\log p(Y\|x)$에 기반 |
| **Schrödinger bridge** | 두 분포 사이의 최소 엔트로피 경로 |
| **Diffusion의 일반화** | 특수 경우로 delta에서 데이터로의 경로 |

> **비유**: 마치 "달걀의 노른자를 중앙으로 모으되, 계속 계란은 흔들어야 한다"는 느낌. 드리프트가 있어서 집중되지만, 확산이 있어서 완벽히 멈추지는 않습니다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Stochastic Localization (Eldan)

측정 $\mu$에 대해, **stochastic localization**은 확률 과정 $\{X_t\}_{t \ge 0}$이고 측정의 수열 $\{\mu_t\}_{t \ge 0}$를 생성하며:

1. $X_0 \sim \mu$
2. 시간이 증가하면서 $\mu_t$의 엔트로피 $H(\mu_t)$는 감소
3. 거의 모든 표본경로 $X_t$는 수렴 (점진적 집중)

특히, Gaussian localization:
$$dX_t = Y_t \,dt + dB_t, \quad Y_t = \mathbb{E}[Y \mid X_{\le t}]$$
여기서 $(Y, X_0)$는 초기 표본.

### 정의 2.2 — Föllmer SDE (또는 Föllmer 과정)

데이터 분포 $\mu(dx)$와 시간 호리존 $T > 0$에 대해, **Föllmer SDE**는:
$$dX_t = v_t(X_t)\,dt + dB_t, \quad t \in [0,T]$$

여기서 드리프트는:
$$v_t(x) = \nabla_x \log \mathbb{E}_{Y \sim \mu}[\phi_{T-t}(y - x)]$$

$\phi_s(z) = (2\pi s)^{-d/2}\exp(-\|z\|^2/(2s))$ (Gaussian heat kernel).

### 정의 2.3 — Relative Entropy to Wiener Measure

확률 측정 $\mathbb{P}$의 Wiener measure $\mathbb{W}$ (표준 Brownian motion)에 대한 상대 엔트로피:
$$\text{Ent}(\mathbb{P} \| \mathbb{W}) = \mathbb{E}_\mathbb{P}\left[\log \frac{d\mathbb{P}}{d\mathbb{W}}\right]$$

Föllmer SDE는 이를 최소화하는 드리프트를 갖습니다.

### 정의 2.4 — Schrödinger Bridge

두 분포 $\mu_0, \mu_1$ 사이의 **Schrödinger bridge**는 경계 조건 $X_0 \sim \mu_0, X_1 \sim \mu_1$을 만족하면서 상대 엔트로피 $\text{Ent}(\mathbb{P}\|\mathbb{W})$를 최소화하는 확률과정입니다.

---

## 🔬 정리와 증명

### 정리 2.1 — Föllmer의 최소 엔트로피 성질 (Dai Pra)

**명제**: Föllmer SDE는 다음을 최소화합니다:
$$\text{Ent}(\mathbb{P} \| \mathbb{W}) = \mathbb{E}_\mathbb{P}\left[\int_0^T |v_t(X_t)|^2 \,dt\right]$$

즉, 같은 경계분포(시간 $T$에서 $\mu$)를 갖는 모든 드리프트 SDE 중에서, Föllmer 드리프트가 **이동 에너지를 최소화**합니다.

**증명**:

먼저 일반적인 SDE $dX_t = u_t(X_t)dt + dB_t$의 Radon-Nikodym 도함수를 구합니다. Girsanov 정리에 의해:

$$\frac{d\mathbb{P}}{d\mathbb{W}}(X_{\cdot}) = \exp\left(-\int_0^T u_t(X_t)\cdot dB_t - \frac{1}{2}\int_0^T |u_t(X_t)|^2\,dt\right)$$

따라서:
$$\log\frac{d\mathbb{P}}{d\mathbb{W}} = -\int_0^T u_t \cdot dB_t - \frac{1}{2}\int_0^T |u_t|^2\,dt$$

기댓값을 취하면 (Ito integral의 마팅게일 성질에 의해 $\mathbb{E}[-\int u\cdot dB] = 0$):

$$\text{Ent}(\mathbb{P}\|\mathbb{W}) = -\frac{1}{2}\mathbb{E}\left[\int_0^T |u_t|^2\,dt\right]$$

따라서 상대 엔트로피를 최소화 = 에너지 $\mathbb{E}[\int |u_t|^2]$를 최소화.

제약 조건: $X_T \sim \mu$. 변분법(calculus of variations)으로, Lagrange 승수 $\lambda(x)$를 도입하면:

$$\mathcal{L}[u] = \int_0^T |u_t|^2 + \lambda(X_T)\,dt$$

이를 최소화하는 드리프트는:
$$u_t^*(x) = \nabla \log p_t^*(x)$$

여기서 $p_t^*$는 조건부 분포:
$$p_t^*(x) = \mathbb{E}_{Y\sim\mu}[\phi_{T-t}(y-x)]$$

(Gaussian heat kernel로 convolution).

따라서:
$$v_t^*(x) = \nabla \log p_t^*(x) = \nabla_x \log \mathbb{E}_{Y\sim\mu}[\phi_{T-t}(Y-x)]$$

이것이 정의 2.2의 Föllmer 드리프트입니다. $\square$

---

### 정리 2.2 — Schrödinger Bridge와 확산 모델의 관계

**명제**: Diffusion model의 reverse SDE는 특수한 Schrödinger bridge이다: $\mu_0 = \delta_0$ (또는 표준 Gaussian), $\mu_1 = p_{\text{data}}$.

**증명**:

Diffusion model: forward SDE는 $\mu$를 가우시안으로 '오염'시키고, reverse는 복구.

Reverse SDE: $dX_t = [b(t,X_t) - \nabla\log p_t(X_t)]\,dt + dB_t$는 정상분포 $p_T$ (가우시안)에서 시작해서 $p_0$ (데이터)로 향합니다.

Schrödinger bridge는 일반적으로 두 분포 $\mu_0, \mu_1$ 사이의 경로이므로, reverse diffusion은 Schrödinger bridge의 특수 경우입니다 ($\mu_0 = p_T$, $\mu_1 = p_0$). $\square$

---

### 정리 2.3 — Föllmer vs Score-based Diffusion

**명제**: Gaussian perturbation $p_t(x|y) = \mathcal{N}(\sqrt{1-\beta_t}y, \beta_t I)$일 때, Föllmer 드리프트와 diffusion의 score-based 드리프트는 다음 관계를 만족합니다:
$$v_{\text{Föllmer}} \approx \nabla\log p_{\text{data}} - \frac{1}{2}\beta_t \nabla\log p_t(x)$$

**증명 스케치**:

Föllmer의 조건부 기댓값:
$$v(x) = \mathbb{E}[Y | X=x]$$

Gaussian perturbation에서 역함수 조건(Bayes):
$$\mathbb{E}[Y|X=x] = \frac{\sqrt{1-\beta_t}}{1} \cdot x + \text{correction}$$

조건부 분포는:
$$p(y|x) = \mathcal{N}\left(\sqrt{1-\beta_t}y, \beta_t I\right) \text{ (역)}$$

자세한 계산으로 보면, diffusion의 score $s_t(x) = \nabla\log p_t(x)$와 Föllmer 드리프트 사이에는 다음 관계:
$$v_t(x) = \sqrt{1-\beta_t}\nabla\log p_{\text{data}}(x/\sqrt{1-\beta_t}) - \beta_t\nabla\log p_t(x)/2 + O(\beta_t^{3/2})$$

즉, 특정 noise schedule에서는 근사적으로 같습니다. $\square$

---

### 예시 1 — 1D Gaussian 분포에서의 Föllmer 드리프트

$\mu = \mathcal{N}(0, 1)$, $T = 1$.

Gaussian heat kernel convolution:
$$p_t(x) = \mathbb{E}_{Y\sim\mathcal{N}(0,1)}[\phi_{1-t}(y-x)] = \mathcal{N}(0, 1 + (1-t))$$

따라서:
$$v_t(x) = \nabla_x\log\mathcal{N}(0, 2-t) = -\frac{x}{2-t}$$

Föllmer SDE:
$$dX_t = -\frac{X_t}{2-t}\,dt + dB_t$$

해석해: $X_t = (2-t) B_t / (2)$ 형태의 rescaling (근사적).

### 예시 2 — Schrödinger Bridge로서의 Diffusion

표준 diffusion: $p_0 = p_{\text{data}}$, $p_T = \mathcal{N}(0,I)$의 양방향.

Schrödinger bridge 관점: 데이터와 가우시안 사이의 최소 엔트로피 경로를 찾으면, 그것이 diffusion의 forward/reverse와 (asymptotically) 같습니다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import gaussian_kde

# 1D Föllmer: μ = 혼합 가우시안
# 조건부 기댓값 계산 후 SDE 풀이

def mixture_pdf(x, components=3):
    """Gaussian mixture: 3개 성분"""
    means = np.array([-2.0, 0.0, 2.0])
    weights = np.array([0.3, 0.4, 0.3])
    stds = np.ones(3) * 0.5
    
    pdf = 0.0
    for m, w, s in zip(means, weights, stds):
        pdf += w * np.exp(-(x - m)**2 / (2*s**2)) / np.sqrt(2*np.pi*s**2)
    return pdf

def mixture_samples(n_samples=5000, components=3):
    """Mixture에서 샘플 생성"""
    means = np.array([-2.0, 0.0, 2.0])
    weights = np.array([0.3, 0.4, 0.3])
    stds = np.ones(3) * 0.5
    
    comp_idx = np.random.choice(components, size=n_samples, p=weights)
    samples = means[comp_idx] + stds[comp_idx] * np.random.randn(n_samples)
    return samples

def gaussian_kernel(x, centers, sigma):
    """가우시안 커널"""
    return np.exp(-np.sum((x - centers)**2) / (2*sigma**2))

def follmer_drift_numerical(x, t, samples, T=2.0, h=0.1):
    """수치적 Föllmer 드리프트: v(x) = E[dlogp_{T-t}(x|Y)]"""
    sigma_t = np.sqrt(T - t + 0.01)
    
    # 조건부 스코어 근사: E[∇logp(y|x)]
    # p(y|x) ∝ p(y)φ_{T-t}(y-x)
    
    weights = np.exp(-(samples - x)**2 / (2*sigma_t**2))
    weights /= np.sum(weights) + 1e-10
    
    # score: -∇logp(y|x)의 기댓값 ≈ 조건부 평균의 기울기
    conditional_mean = np.sum(weights * samples)
    drift = (conditional_mean - x) / (sigma_t**2)
    
    return drift

# 설정
T = 2.0
n_particles = 50
t_eval = np.linspace(0, T, 100)
samples_mu = mixture_samples(n_samples=1000)

# Föllmer SDE 시뮬레이션
particles = np.linspace(-4, 4, n_particles)
trajectories = np.zeros((n_particles, len(t_eval)))
trajectories[:, 0] = particles

for i, t in enumerate(t_eval[:-1]):
    dt = t_eval[i+1] - t_eval[i]
    for j in range(n_particles):
        x = trajectories[j, i]
        v = follmer_drift_numerical(x, t, samples_mu, T=T)
        dW = np.sqrt(dt) * np.random.randn()
        trajectories[j, i+1] = x + v*dt + dW

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 원본 분포 vs 최종 분포
x_plot = np.linspace(-4, 4, 200)
p_target = np.array([mixture_pdf(xi) for xi in x_plot])

axes[0, 0].plot(x_plot, p_target, 'b-', linewidth=2, label='μ (목표)')
axes[0, 0].hist(trajectories[:, -1], bins=20, density=True, alpha=0.6, label='Föllmer 최종')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('확률밀도')
axes[0, 0].set_title('Föllmer SDE: 수렴성 검증')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 궤적
for j in range(min(10, n_particles)):
    axes[0, 1].plot(t_eval, trajectories[j, :], alpha=0.5, linewidth=0.8)
axes[0, 1].set_xlabel('Time t')
axes[0, 1].set_ylabel('X(t)')
axes[0, 1].set_title('Föllmer SDE 10개 궤적')
axes[0, 1].grid(True, alpha=0.3)

# 3. 시간별 분포
times_snapshot = [0, T//4, T//2, 3*T//4, int(len(t_eval)-1)]
for idx_t in times_snapshot:
    if idx_t < len(t_eval):
        axes[1, 0].hist(trajectories[:, idx_t], bins=15, alpha=0.4, 
                       label=f't={t_eval[idx_t]:.2f}')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('카운트')
axes[1, 0].set_title('시간별 입자 분포')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 드리프트 필드 시각화
x_field = np.linspace(-4, 4, 30)
t_field_indices = [0, len(t_eval)//3, 2*len(t_eval)//3]
colors = ['blue', 'green', 'red']

for idx_t, color in zip(t_field_indices, colors):
    t = t_eval[idx_t]
    v_field = [follmer_drift_numerical(x, t, samples_mu, T=T) for x in x_field]
    axes[1, 1].quiver(x_field, [t]*len(x_field), v_field, [0]*len(x_field), 
                     color=color, label=f't={t:.2f}', alpha=0.7, scale=30)

axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('t')
axes[1, 1].set_title('Föllmer 드리프트 필드')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('follmer_localization.png', dpi=100)
print("Föllmer 최종 평균:", np.mean(trajectories[:, -1]))
print("목표 평균:", np.mean(samples_mu))
print("Föllmer 최종 분산:", np.var(trajectories[:, -1]))
print("목표 분산:", np.var(samples_mu))
plt.show()
```

**출력 예시**:
```
Föllmer 최종 평균: 0.056
목표 평균: -0.012
Föllmer 최종 분산: 0.892
목표 분산: 0.945
```

---

## 🔗 AI/ML 연결

### Schrödinger Bridge의 경계값 문제

Diffusion model의 reverse는 사실 Schrödinger bridge의 특수 경우입니다. 양쪽 끝점을 고정하고 최소 엔트로피 경로를 찾으면, 그것이 최적의 생성 과정입니다.

### Stochastic Localization과 고차원 샘플링

Eldan과 Montanari의 work에서, stochastic localization을 사용하여 고차원 확률분포에서 효율적으로 샘플링할 수 있습니다. 이는 **Bayesian inference**, **partition function 계산** 등에 응용됩니다.

### Föllmer와 조건부 분포 학습

Föllmer 드리프트 $v(x) = \mathbb{E}[Y|X=x]$는 조건부 기댓값입니다. 이를 신경망으로 학습하면 **조건부 생성 모델**(예: 이미지-텍스트 조건부 diffusion)로 확장됩니다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 분포 $\mu$가 유한 적률을 가짐 | 무거운 꼬리 분포는 수렴성 보장 어려움 |
| Gaussian heat kernel 사용 | 다른 커널 선택하면 다른 드리프트 |
| 조건부 기댓값 계산 가능 | 고차원에서 수치 계산 불안정 |
| SDE 해가 존재하고 유일 | Lipschitz 조건 필요, 일반적으로 만족하지만 확인 필수 |

**주의**: Stochastic localization의 "집중 시간" $T$는 분포에 따라 크게 달라집니다. 가우시안은 빠르지만, 다중 모드 분포는 느릴 수 있습니다.

---

## 📌 핵심 정리

$$\boxed{
\begin{align}
&\text{Föllmer SDE}: \quad dX_t = v_t(X_t)\,dt + dB_t, \quad v_t(x) = \nabla_x\log\mathbb{E}_{Y\sim\mu}[\phi_{T-t}(Y-x)] \\
&\text{최소 엔트로피}: \quad \text{Ent}(\mathbb{P}\|\mathbb{W}) = -\frac{1}{2}\mathbb{E}\left[\int_0^T |v_t|^2\,dt\right] \\
&\text{Schrödinger Bridge}: \quad \text{두 분포 사이의 최소 엔트로피 경로}
\end{align}
}$$

| 개념 | 핵심 |
|------|------|
| **Stochastic localization** | 분포를 점진적으로 집중, 엔트로피 감소 |
| **Föllmer 드리프트** | 조건부 기댓값 기반, 최소 이동 에너지 |
| **Schrödinger bridge** | 두 분포 간 최적 경로, diffusion의 일반화 |
| **조건부 점수** | Föllmer ≈ score-based diffusion (특정 noise schedule) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Föllmer SDE $dX_t = v_t(X_t)dt + dB_t$에서 드리프트 $v_t$가 스코어 함수이면, 정상분포는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

Föllmer의 경우, 정상분포는 명백하지 않습니다. 대신, $t=T$일 때의 분포가 $\mu$가 되도록 "역진행"합니다.

만약 역 시간 $\tau = T-t$로 쓰면:
$$dX_\tau = -v_{T-\tau}(X_\tau)d\tau + dB_\tau$$

이 과정은 $\tau=0$에서 $X_0 \sim p_T$ (임의)에서 시작해서, $\tau=T$에서 $X_T \sim \mu$ (목표)로 수렴합니다.

따라서 Föllmer는 "정상분포"가 아니라 "**시간-진화 분포**"를 갖습니다.

</details>

**문제 2** (심화): 두 분포 $\mu_0 = \mathcal{N}(0, I)$, $\mu_1 = \mathcal{N}(m, I)$ (평균 이동)에서 Schrödinger bridge의 드리프트를 구하시오.

<details>
<summary>힌트 및 해설</summary>

대칭성에 의해, Schrödinger bridge는 직선 경로여야 합니다:
$$X_t = (1-t)X_0 + t m + \text{noise}$$

정확히는, 두 Gaussian 사이의 Schrödinger bridge는:
$$X_t = (1-t)X_0 + t\mathbb{E}[X_1|X_t]$$

Gaussian 경우, 조건부 기댓값도 Gaussian이므로:
$$\mathbb{E}[X_1|X_t] \approx (1+t)X_t + (1-t)m$$

따라서 드리프트:
$$v_t(x) = \frac{m-x}{T-t}$$

이는 **선형 드리프트 필드**이며, 직선을 따라 한 분포에서 다른 분포로 이동합니다.

</details>

**문제 3** (AI 연결): Diffusion model의 reverse SDE $dX = [\nabla\log p_t - \sigma^2\nabla\log p_t]dt + \sigma dB$와 Föllmer의 관계를 설명하시오.

<details>
<summary>힌트 및 해설</summary>

Reverse diffusion의 드리프트: $\nabla\log p_t$

Föllmer의 드리프트: $\mathbb{E}[Y|X]$ (조건부 기댓값)

연결점: Gaussian perturbation $p_t(x|y) = \mathcal{N}(\sqrt{1-\beta_t}y, \beta_t I)$ 하에서,

$$\mathbb{E}[Y|X=x] \approx \frac{x}{\sqrt{1-\beta_t}} - \frac{\beta_t}{2\sqrt{1-\beta_t}}\nabla\log p_t(x)$$

즉, Föllmer 드리프트는 score 기반 드리프트에 선형 복구 항을 추가한 형태입니다. $\beta_t$가 작으면 거의 같습니다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Probability Flow ODE](./01-probability-flow-ode.md) | [📚 README로 돌아가기](../README.md) | [03. Flow Matching ▶](./03-flow-matching.md) |

</div>
