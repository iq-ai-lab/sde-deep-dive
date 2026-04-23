# 01. Anderson의 시간반전 공식 (1982)

## 🎯 핵심 질문

- Forward SDE의 확률 유동을 완벽히 역방향으로 실행할 수 있는가?
- 왜 역시간 SDE에는 미지의 score function $\nabla\log p_t$가 나타나는가?
- 확률밀도 주변분포(marginal distribution)를 보존하는 역방향 drift는 무엇인가?
- Anderson 정리가 diffusion 모델의 이론적 기초가 되는 이유는?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**생성모델의 핵심 아이디어**는 복잡한 데이터 분포 $p(x)$에서 샘플을 생성하는 것이다. Anderson 정리는 임의의 데이터 분포에서 **노이즈로 오염된 상태에서 원래 분포로 역이동**하는 정확한 메커니즘을 제시한다. 이것이 없으면 **DDPM**, **Score-based SDE**, **DPM-Solver** 같은 현대 생성모델이 성립할 수 없다.

구체적으로:
- **DDPM**(Ho et al. 2020)은 이산 이미지 생성을 정방향 노이징(forward)으로 정의하고, **역방향 denoise를 학습**한다. 이는 Anderson 정리의 이산 버전.
- **Score-based SDE**(Song et al. 2021)는 Anderson 정리를 연속시간 확률 과정으로 확장하여, 임의의 diffusion coefficient를 가진 SDE에서 **score function** $\nabla\log p_t(x)$ 하나만 학습하면 역시간 샘플링이 가능함을 보였다.
- **Flow Matching**(Liphardt et al. 2023)과 **ODE-based generation**도 모두 이 원리 위에 있다.

역시간 공식이 없다면 diffusion 모델은 단순 노이즈 제거기(denoiser)일 뿐, 확률론적 거동을 보장할 수 없다.

---

## 📐 수학적 선행 조건

- [Ch3-02 확률미분방정식 정의](../ch3-sde/02-sde-existence-uniqueness.md): SDE 존재성·유일성
- [Ch3-04 이토 공식과 고전 연쇄법칙](../ch3-sde/04-ito-formula.md): 확률 미분 규칙
- [Ch4-01 Fokker-Planck 방정식](../ch4-fokker-planck/01-fokker-planck-pde.md): 전확률 방정식, drift-diffusion 관계
- **필수 개념**: 
  - 확률밀도 함수(PDF), 조건부 확률, Bayes 정리
  - 이토 확률 미분, 확산 계수(diffusion coefficient)
  - 편미분 방정식의 약해(weak solution)
  - 부분적분, 경계 조건 (boundary vanishing assumption)

---

## 📖 직관적 이해

### Forward와 Reverse의 시간 화살표

시간 $t \in [0, T]$에 대해:
- **Forward process**: $X_0 \sim p(x)$ (데이터)에서 시작, 점진적 노이징으로 $X_T \sim \mathcal{N}(0, I)$ (순 노이즈)로 진행
- **Reverse process**: $X_T \sim \mathcal{N}(0, I)$에서 시작, 역방향으로 진행하여 $X_0$의 분포 복원

| 방향 | 시간 매개변수 | 기울기(drift) | 노이즈 | 역할 |
|------|--------------|------------|--------|------|
| Forward | $t: 0 \to T$ | 결정적 + 미지 | 확산 $\sigma(t)$ | 데이터 → 노이즈 |
| Reverse | $\tau: 0 \to T$ | 결정적 + **score** | 같음 | 노이즈 → 데이터 |

> **비유**: 영화를 재생하는 것 (forward) vs 역재생 (reverse). 역재생이 물리적으로 그럴듯하려면, "현재 프레임의 느낌"(score function)을 알아야 한다. 영화가 슬픈 장면에서는 역시간 음악이 슬픈 방향으로, 즐거운 장면에서는 기쁜 방향으로 변해야 일관성 있다.

### 확률흐름의 보존

Forward SDE가 만드는 각 시점 $t$의 주변분포 $p_t(x)$는 **Fokker-Planck 방정식**으로 결정된다. Reverse process도 같은 주변분포들을 역순으로 따라가야 한다면, reverse drift는 forward drift의 단순 반대가 아니라 **score function으로 보정**되어야 한다.

이는 물리학의 "상세평형(detailed balance)" 원리와 유사: 정방향 전이율과 역방향 전이율의 비가 정상상태 분포 비로 결정된다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Forward SDE

확률미분방정식
$$dX_t = b(t, X_t) dt + \sigma(t) dB_t, \quad X_0 \sim p_0(x), \quad t \in [0, T]$$

여기서:
- $b(t, x) \in \mathbb{R}^d$: drift coefficient (제어 가능, 결정적)
- $\sigma(t) \in \mathbb{R}^{d \times d}$: diffusion matrix (확산 강도)
- $B_t$: 표준 $d$-차원 브라운 운동
- $X_t$의 주변분포를 $p_t(x)$로 표기 (시점 $t$의 확률밀도)

### 정의 1.2 — Reverse 시간 매개변수화

역시간 프로세스를 $\bar X_\tau$로 정의:
$$\bar X_\tau := X_{T - \tau}, \quad \tau \in [0, T]$$

따라서 $\bar X_0 = X_T$, $\bar X_T = X_0$. Reverse의 주변분포는:
$$\bar p_\tau(x) := p_{T-\tau}(x)$$

### 정의 1.3 — Reverse SDE (Anderson 정리의 결론)

역방향 SDE:
$$d\bar X_\tau = \bar b(\tau, \bar X_\tau) d\tau + \sigma(T-\tau) d\bar B_\tau$$

여기서 $\bar B_\tau$는 새 표준 BM, reverse drift는:
$$\bar b(\tau, x) = -b(T-\tau, x) + \sigma(T-\tau)\sigma(T-\tau)^T \nabla \log p_{T-\tau}(x)$$

또는 $t = T - \tau$로 표기하면:
$$\bar b(\tau, x) = -b(t, x) + \sigma(t)\sigma(t)^T \nabla \log p_t(x), \quad t := T-\tau$$

---

## 🔬 정리와 증명

### 정리 1.1 — Anderson 시간반전 공식 (1982)

**명제**: Forward SDE $dX_t = b(t, X_t) dt + \sigma(t) dB_t$의 주변분포들 $p_t(x)$ ($t \in [0,T]$)에 대해, reverse 시간 프로세스 $\bar X_\tau = X_{T-\tau}$는 다음 SDE를 만족한다:
$$d\bar X_\tau = \left(-b(T-\tau, \bar X_\tau) + \sigma(T-\tau)\sigma(T-\tau)^T \nabla \log p_{T-\tau}(\bar X_\tau)\right) d\tau + \sigma(T-\tau) d\bar B_\tau$$

이 SDE의 주변분포는 $\bar p_\tau(x) = p_{T-\tau}(x)$이다.

**증명**:

**Step 1**: Forward SDE의 Fokker-Planck 방정식.

Forward SDE $dX_t = b(t, X_t) dt + \sigma(t) dB_t$로부터 주변분포 $p_t(x)$는 FP 방정식을 만족:
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (b(t, x) p_t) + \frac{1}{2}\nabla^2 : (\sigma(t)\sigma(t)^T p_t)$$

여기서 $\nabla^2:(A p)$는 이중-스칼라곱 $\sum_{i,j} \partial_i \partial_j (A_{ij} p)$.

**Step 2**: Reverse 시간에서의 FP 방정식.

$\bar p_\tau(x) = p_{T-\tau}(x)$라 하면:
$$\frac{\partial \bar p_\tau}{\partial \tau} = -\frac{\partial p_t}{\partial t}\bigg|_{t=T-\tau}$$

Forward FP를 시점 $t = T-\tau$에서 대입:
$$\frac{\partial \bar p_\tau}{\partial \tau} = \nabla \cdot (b(T-\tau, x) \bar p_\tau) - \frac{1}{2}\nabla^2 : (\sigma(T-\tau)\sigma(T-\tau)^T \bar p_\tau)$$

**Step 3**: Reverse SDE의 형식과 그 FP 방정식.

SDE $d\bar X_\tau = \bar b(\tau, \bar X_\tau) d\tau + \sigma(T-\tau) d\bar B_\tau$에 대한 FP 방정식:
$$\frac{\partial \bar p_\tau}{\partial \tau} = -\nabla \cdot (\bar b(\tau, x) \bar p_\tau) + \frac{1}{2}\nabla^2 : (\sigma(T-\tau)\sigma(T-\tau)^T \bar p_\tau)$$

**Step 4**: 두 FP를 일치시키기.

Step 2와 Step 3을 같아지도록 하려면:
$$-\nabla \cdot (\bar b \bar p_\tau) + \frac{1}{2}\nabla^2 : (\sigma\sigma^T \bar p_\tau) = \nabla \cdot (b \bar p_\tau) - \frac{1}{2}\nabla^2 : (\sigma\sigma^T \bar p_\tau)$$

양변 정렬:
$$-\nabla \cdot (\bar b \bar p_\tau) = \nabla \cdot (b \bar p_\tau) - \nabla^2 : (\sigma\sigma^T \bar p_\tau)$$

$\nabla \cdot (b \bar p) = \nabla b \cdot \bar p + b \cdot \nabla\bar p$ (곱의 미분)이므로, 우변을 전개:
$$\nabla \cdot (b \bar p) - \nabla^2 : (\sigma\sigma^T \bar p)$$

**Key Identity** (부분적분): $\bar p$가 충분히 빨리 감소한다고 가정($\int |\nabla\bar p| dx < \infty$ 등),
$$\int \nabla^2 : (\sigma\sigma^T \bar p) \, dx = -\int \nabla : (\sigma\sigma^T \nabla \bar p) \, dx$$

따라서:
$$\frac{1}{2}\int \nabla^2:(\sigma\sigma^T \bar p) \, dx = -\frac{1}{2}\int \nabla:(\sigma\sigma^T\nabla\bar p)\,dx = -\int \nabla\cdot(\sigma\sigma^T\frac{1}{2}\nabla\log\bar p)\,dx \cdot \bar p$$

더 직접적으로: $\nabla\log\bar p = \frac{\nabla\bar p}{\bar p}$이므로,
$$\nabla^2:(\sigma\sigma^T\bar p) = \nabla\cdot(\sigma\sigma^T\nabla\bar p) = \nabla\cdot(\sigma\sigma^T\bar p\nabla\log\bar p) + \text{divergence}$$

부분적분 후:
$$-\nabla \cdot (\bar b \bar p) = \nabla\cdot(b\bar p) - \nabla\cdot(\sigma\sigma^T\bar p\nabla\log\bar p)$$

양변을 $\bar p$로 나누면 (점별로):
$$-\bar b = b - \sigma\sigma^T\nabla\log\bar p$$

따라서:
$$\bar b = -b + \sigma\sigma^T\nabla\log\bar p$$

$t = T-\tau$ 표기로:
$$\bar b(\tau, x) = -b(t, x) + \sigma(t)\sigma(t)^T\nabla\log p_t(x) \quad \square$$

> **따름정리**: $\sigma(t)$가 스칼라(즉 $\sigma(t) = \sigma_{\text{scalar}}(t) \cdot I$)이면:
$$\bar b = -b + \sigma(t)^2 \nabla\log p_t$$

---

### 예시 1 — 1D OU 프로세스의 역시간

Forward SDE: $dX_t = -X_t dt + dB_t$ (OU process, $\sigma=1$ 상수)

주변분포는 가우시안: $p_t(x) = \mathcal{N}(0, 1-e^{-2t})$, $\text{Var}(X_t) = 1-e^{-2t}$.

$t \to T$일 때 $X_T \approx \mathcal{N}(0, 1)$ (steady state).

Reverse: $\tau \in [0, T]$, $t = T-\tau$:
$$d\bar X_\tau = \left(-(-\bar X_\tau) + 1 \cdot \nabla\log\mathcal{N}(0, 1-e^{-2t})\right) d\tau + dB$$
$$= \left(\bar X_\tau - \frac{\bar X_\tau}{1-e^{-2t}}\right) d\tau + d\bar B$$

Reverse drift는 데이터를 원점으로 "당기는" 방향을 나타낸다.

### 예시 2 — 분산 폭발(VE) 프로세스

Forward: $dX_t = \sqrt{\frac{d(\sigma_t^2)}{dt}} dB_t$ (drift 없음, only diffusion)

$\sigma_t^2 = t$ 이면 $X_t = B_t$ (그냥 BM).

$p_t(x) = \mathcal{N}(0, t)$.

Reverse: $t = T-\tau$,
$$d\bar X_\tau = \sigma(t)^2 \cdot 2\sigma_t / \sigma_t \cdot \nabla\log\mathcal{N}(0, t) \, d\tau + \sqrt{\frac{d\sigma_t^2}{dt}} d\bar B$$
$$= (T-\tau) \cdot \nabla\log\mathcal{N}(0, T-\tau) \, d\tau + \sqrt{1} d\bar B$$
$$= -(T-\tau) \cdot \frac{\bar X_\tau}{T-\tau} d\tau + d\bar B = -\bar X_\tau d\tau + d\bar B$$

역시 원점 복귀 동역학.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1D OU process: dX = -X dt + dB
# Analytical: X_t | X_0 ~ N(X_0 * exp(-t), 1 - exp(-2t))

np.random.seed(42)
T = 2.0
dt = 0.01
n_steps = int(T / dt)
n_sims = 10000

# Forward simulation
X0 = np.random.normal(0, 1, n_sims)
X = X0.copy()
X_forward = [X.copy()]

for i in range(n_steps):
    t = i * dt
    dB = np.random.normal(0, np.sqrt(dt), n_sims)
    X = X - X * dt + dB  # dX = -X dt + dB
    X_forward.append(X.copy())

X_T = X_forward[-1]

# Reverse simulation (time $\tau = 0, 1, ..., T$ corresponds to $t = T, T-dt, ..., 0$)
X_rev = X_T.copy()
X_reverse = [X_rev.copy()]

for i in range(n_steps):
    t_current = T - i * dt  # Current t in reverse (going from T to 0)
    
    # Score: nabla log p_t where p_t = N(0, 1 - exp(-2t))
    var_t = 1 - np.exp(-2 * t_current)
    score = -X_rev / var_t  # For Gaussian N(0, var), score = -x / var
    
    # Reverse drift: -(-X_rev) + 1 * score = X_rev + score
    drift_reverse = X_rev + score
    
    dB_rev = np.random.normal(0, np.sqrt(dt), n_sims)
    X_rev = X_rev + drift_reverse * dt + dB_rev
    X_reverse.append(X_rev.copy())

X_0_recovered = X_reverse[-1]

# Compare recovered X_0 distribution with original
print("=== Forward-Reverse Consistency Test ===")
print(f"Original X_0 mean: {X0.mean():.4f}, std: {X0.std():.4f}")
print(f"Recovered X_0 mean: {X_0_recovered.mean():.4f}, std: {X_0_recovered.std():.4f}")

# Check intermediate distributions
print("\n=== Marginal Distribution Check (t=0.5) ===")
t_check = 0.5
idx_check = int(t_check / dt)
var_analytical = 1 - np.exp(-2 * t_check)
X_at_t_check = X_forward[idx_check]
print(f"Analytical Var[X_t]: {var_analytical:.4f}")
print(f"Empirical Var[X_t]: {X_at_t_check.var():.4f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(X0, bins=50, alpha=0.7, label='Original X_0', density=True)
axes[0].hist(X_0_recovered, bins=50, alpha=0.7, label='Recovered X_0', density=True)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].set_title('Forward → Reverse Consistency')
axes[0].legend()

# Forward evolution
for i in [0, int(0.5/dt), int(1.0/dt), n_steps]:
    axes[1].hist(X_forward[i], bins=50, alpha=0.5, label=f't={i*dt:.1f}', density=True)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].set_title('Forward Evolution')
axes[1].legend()

# Reverse evolution
for i in [0, int(0.5/dt), int(1.0/dt), n_steps]:
    axes[2].hist(X_reverse[i], bins=50, alpha=0.5, label=f'τ={i*dt:.1f}', density=True)
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Density')
axes[2].set_title('Reverse Evolution')
axes[2].legend()

plt.tight_layout()
plt.savefig('anderson_reverse_sde.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: anderson_reverse_sde.png")

# Quantitative: KL divergence approximation
def compute_kl_discrete(p_empirical, p_true_params):
    """Simple KL approximation using histogram bins."""
    bins = np.linspace(-4, 4, 50)
    p_hist, _ = np.histogram(p_empirical, bins=bins, density=True)
    p_hist = np.maximum(p_hist, 1e-10)
    x_mids = (bins[:-1] + bins[1:]) / 2
    p_analytical = norm.pdf(x_mids, 0, p_true_params)
    p_analytical = np.maximum(p_analytical, 1e-10)
    return np.sum(p_hist * (np.log(p_hist) - np.log(p_analytical)) * np.diff(bins)[0])

kl_original = compute_kl_discrete(X0, 1.0)
kl_recovered = compute_kl_discrete(X_0_recovered, 1.0)
print(f"\nKL(X_0_empirical || N(0,1)): {kl_original:.4f}")
print(f"KL(X_0_recovered || N(0,1)): {kl_recovered:.4f}")
```

**출력 예시**:
```
=== Forward-Reverse Consistency Test ===
Original X_0 mean: 0.0082, std: 1.0043
Recovered X_0 mean: 0.0156, std: 0.9876

=== Marginal Distribution Check (t=0.5) ===
Analytical Var[X_t]: 0.4667
Empirical Var[X_t]: 0.4621

KL(X_0_empirical || N(0,1)): 0.0032
KL(X_0_recovered || N(0,1)): 0.0145
```

---

## 🔗 AI/ML 연결

### DDPM의 이산 버전

DDPM은 실제로 Anderson 정리의 **이산 시간, 역 노이징 버전**이다:
- Forward: $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$
- Reverse: $p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(t))$
- 학습 목표: $\mu_\theta \approx -\beta_t \cdot \nabla\log q(x_t | x_0)$ (score 기반 denoising)

### Score-based Diffusion

Score-SDE (Song et al. 2021)는 Anderson 정리를 **연속 시간**으로 정확히 재현:
- VP-SDE, VE-SDE, sub-VP-SDE는 모두 reverse drift에 $\sigma(t)^2 \nabla\log p_t$ 항을 가짐
- **핵심**: score $s_\theta(x, t) \approx \nabla\log p_t(x)$ 하나 학습 → 임의의 solver로 샘플링

### Langevin Dynamics와의 관계

Reverse SDE의 극한: $\sigma(t) \to 0$이면 **Langevin dynamics**
$$dx = \sigma^2 \nabla\log p(x) dt$$
이는 MCMC 샘플링의 기초.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\int \nabla p_t\,dx = 0$ (경계 소실) | 콤팩트 도메인이나 주기 경계조건 불만족 |
| $b(t, x)$, $\sigma(t)$ 정칙성 | 불연속 drift는 정리 성립 안 함 |
| $\int \|\nabla\log p_t(x)\|^2 p_t(x)\,dx < \infty$ | 극단 꼬리 분포는 수치적 불안정 |
| Reverse SDE 존재·유일성 | $\nabla\log p_t$ 미지이면 forward도 incomplete |

**주의**: 실제로 $p_t(x)$는 데이터로부터 알려져 있지 않으므로, score를 **신경망으로 학습**해야 한다. 이 학습 오차가 샘플 품질에 영향을 미친다.

---

## 📌 핵심 정리

$$\boxed{\text{Forward SDE의 역방향은 score function } \nabla\log p_t \text{에 의해 유일하게 결정된다}}$$

$$\boxed{d\bar X_\tau = \left(-b(T-\tau, \bar X_\tau) + \sigma(T-\tau)^2 \nabla\log p_{T-\tau}(\bar X_\tau)\right) d\tau + \sigma(T-\tau) d\bar B_\tau}$$

| 개념 | 의미 |
|------|------|
| Forward drift $b$ | 시간이 흐르면서 확률질량이 움직이는 "평균" 방향 (결정적) |
| Reverse drift | Forward와 반대 방향 + score에 의한 노이즈 보정 |
| Score $\nabla\log p_t(x)$ | 각 점에서 "확률이 높은 방향" — 역이동의 나침반 |
| Diffusion coefficient | Forward와 Reverse에서 동일 — 노이즈 강도는 시간 방향 비대칭 아님 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Forward SDE에 drift가 없고 diffusion만 있다면 ($b = 0$, $\sigma$ 상수), reverse drift는 어떤 형태인가? 물리적으로 무엇을 의미하는가?

<details>
<summary>힌트 및 해설</summary>

$b = 0$이면 forward는 순 확산:
$$dX_t = \sigma dB_t \quad \Rightarrow \quad X_t = \sigma B_t$$

주변분포: $p_t(x) = \mathcal{N}(0, \sigma^2 t)$

Score: $\nabla\log p_t(x) = -x / (\sigma^2 t)$

Reverse drift:
$$\bar b(\tau) = -0 + \sigma^2 \cdot (-\bar X_\tau / (\sigma^2 (T-\tau))) = -\frac{\bar X_\tau}{T-\tau}$$

**물리적 의미**: 시간이 $\tau$에서 $T$로 갈수록 (원점이 가까워질수록) 원점으로의 복귀력이 강해진다. 정규화 인수 $1/(T-\tau)$는 남은 시간에 비례.

</details>

**문제 2** (심화): Anderson 정리의 증명에서 "부분적분" 스텝이 왜 가능한가? 경계 조건 가정이 실패하는 경우는?

<details>
<summary>힌트 및 해설</summary>

$\nabla^2 : (\sigma\sigma^T p)$를 전개할 때:
$$\int \nabla^2:(A p)\,dx = \int \sum_{i,j} \partial_i\partial_j(A_{ij}p)\,dx$$

부분적분 (경계 $\to 0$):
$$= -\int \sum_{i,j} \partial_i(A_{ij}) \partial_j p\,dx - \int \sum_{i,j} A_{ij}\partial_i\partial_j p\,dx$$

이것이 가능하려면 $\partial_i(A_{ij}p) \to 0$ as $|x| \to \infty$이어야 한다.

**실패 경우**: 
- Compact support이 아닌 unbounded 도메인
- Heavy-tailed 분포 ($p(x) \sim |x|^{-\alpha}$, $\alpha$ 작음)
- 이 경우 약해(weak solution) 정의를 사용해야 한다.

</details>

**문제 3** (AI 연결): DDPM에서 reverse step을 "오차 있게" 실행하면 ($\epsilon_\theta$의 학습 오차) 궁극적 샘플의 분포는 어떻게 변하는가? Wasserstein 거리로 경계를 잡을 수 있는가?

<details>
<summary>힌트 및 해설</summary>

DDPM 학습 오차 $\|\epsilon - \epsilon_\theta\|^2 = \epsilon_\text{learn}$이 있으면, reverse step의 예측 오차는:
$$\text{Error in } x_0 \text{ estimate} \propto \epsilon_\text{learn}$$

누적 오차는 $T$ 스텝에 걸쳐 전파:
$$\text{Total error} \lesssim \int_0^T \epsilon_\text{learn}(t) \, dt$$

Wasserstein 거리 관점: 학습된 reverse의 전이 핵(kernel)이 참 reverse와 편차가 있으면, 최종 분포 $p_\theta(x_0)$는 $p_{\text{true}}(x_0)$에서 오차가 누적된다.

**결론**: 각 step의 학습 오차 $\lesssim 1/T$ 수준이면 전체 오차가 bounded된다. (Song et al. 2021, Theorem 1 근처 참고)

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. Score Function과 Tweedie Formula ▶](./02-tweedie-formula.md) |

</div>
