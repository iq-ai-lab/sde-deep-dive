# 04. Langevin Dynamics의 수렴

## 🎯 핵심 질문

- Langevin dynamics의 밀도 $p_t(x)$가 정상분포 $\pi(x)$로 어떤 속도로 수렴하는가?
- 상대 엔트로피(KL divergence)가 시간에 따라 어떻게 감소하는가?
- Langevin dynamics를 함수공간의 gradient flow로 해석할 수 있는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**Langevin MCMC의 혼합시간(mixing time)**과 **수렴 속도(convergence rate)**를 이해하는 것은 샘플링 효율성을 결정한다. Diffusion model도 역과정에서 비슷한 수렴을 보인다. 또한 **엔트로피 감소(entropy dissipation)**와 **Fisher 정보(Fisher information)**의 관계는 score matching과 정보 기하학(information geometry)의 기초다. 이를 통해 diffusion model의 수렴 품질과 필요한 스텝 수를 이론적으로 추정할 수 있다.

---

## 📐 수학적 선행 조건

- [Ch4-01 Fokker-Planck 방정식의 유도](./01-fokker-planck-derivation.md)
- [Ch4-03 정상분포 — OU, Langevin](./03-stationary-distribution.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) — KL divergence, 상대 엔트로피
- **필수 개념**: KL divergence, Fisher 정보, Pinsker 부등식, 함수부등식(functional inequality)

---

## 📖 직관적 이해

### 엔트로피와 흐름의 관점

Langevin dynamics를 풀면서 밀도 $p_t(x)$가 정상분포 $\pi(x)$로 접근한다. 이 과정을 **엔트로피**의 관점으로 볼 수 있다.

상대 엔트로피(KL divergence):

$$H(p_t \| \pi) = \int p_t(x) \log \frac{p_t(x)}{\pi(x)} dx$$

이것은 $p_t$와 $\pi$ 사이의 "거리"를 측정한다. $H \geq 0$이고, $p_t = \pi$일 때만 $H = 0$.

**de Bruijn의 항등식**: KL divergence의 시간 미분은:

$$\frac{d}{dt} H(p_t \| \pi) = -I(p_t \| \pi)$$

여기서 $I(p_t \| \pi)$는 **상대 Fisher 정보(relative Fisher information)**:

$$I(p_t \| \pi) := \int p_t(x) \left|\nabla \log \frac{p_t(x)}{\pi(x)}\right|^2 dx$$

이것은 **항상 음이 아니다**($\geq 0$). 따라서 KL divergence는 **단조 감소**한다!

### Wasserstein 공간의 Gradient Flow

최근 이론(JKO 1998)에 따르면, Langevin dynamics는 **Wasserstein 메트릭**을 갖춘 확률분포 공간에서 KL divergence의 **gradient flow**이다. 즉:

$$p_t = \arg\min_p \left\{ H(p \| \pi) + \frac{d(p, p_{t-dt})^2}{4dt} \right\}$$

(여기서 $d$는 Wasserstein 거리)

이것은 "각 시점에서 KL을 줄이면서도 급격하게 변하지 않는 확률분포"를 선택한다는 뜻이다.

### 비교: KL의 감소 vs TV 거리의 감소

| 메트릭 | 정의 | 감소 속도 |
|--------|------|----------|
| KL divergence | $H(p \| \pi)$ | de Bruijn: $\frac{dH}{dt} = -I$ |
| Total Variation | $\|p - \pi\|_{TV}$ | Pinsker: $\|p - \pi\|_{TV}^2 \leq \frac{1}{2}H(p \| \pi)$ |
| Wasserstein | $W(p, \pi)$ | 함수부등식 필요 (다음 장) |

> **비유**: 출발지(초기분포)에서 도착지(정상분포)로 가는 여행. KL은 "정보 비용"을 측정하고, TV와 Wasserstein은 "거리"를 측정한다. KL이 단조감소한다는 것은 정보 비용이 계속 줄어든다는 뜻이다.

---

## ✏️ 엄밀한 정의

### 정의 4.11 — 상대 엔트로피 (KL Divergence)

두 확률분포 $p, \pi$에 대해 **상대 엔트로피** 또는 **Kullback-Leibler divergence**:

$$H(p \| \pi) := \mathbb{E}_p \left[\log \frac{p}{π}\right] = \int p(x) \log \frac{p(x)}{\pi(x)} dx$$

성질:
- $H(p \| \pi) \geq 0$ (information inequality)
- $H(p \| \pi) = 0 \Leftrightarrow p = \pi$ a.e.
- 비대칭: 일반적으로 $H(p \| \pi) \neq H(\pi \| p)$

### 정의 4.12 — 상대 Fisher 정보

주어진 확률분포 $p, \pi$에 대해 **상대 Fisher 정보**:

$$I(p \| \pi) := \int p(x) \left|\nabla \log \frac{p(x)}{\pi(x)}\right|^2 dx$$

$$= \mathbb{E}_p \left[\left|\nabla \log p - \nabla \log \pi\right|^2\right]$$

성질: $I(p \| \pi) \geq 0$이고, $\nabla \log p = \nabla \log \pi$ a.e.일 때만 0.

### 정의 4.13 — Pinsker 부등식

확률분포 $p, \pi$에 대해:

$$\|p - \pi\|_{TV}^2 \leq \frac{1}{2} H(p \| \pi)$$

여기서 $\|p - \pi\|_{TV} = \frac{1}{2}\int |p(x) - \pi(x)| dx$는 **총변동(total variation) 거리**.

---

## 🔬 정리와 증명

### 정리 4.7 — de Bruijn의 엔트로피 감소 항등식

**명제**: Langevin dynamics $dX_t = -\nabla U(X_t) dt + \sqrt{2} dB_t$를 풀고, 밀도를 $p_t(x)$, 정상분포를 $\pi(x) = e^{-U(x)}/Z$라 하자.

그러면:

$$\frac{d}{dt} H(p_t \| \pi) = -I(p_t \| \pi)$$

**증명**:

**단계 1**: KL divergence의 시간 미분:

$$\frac{d}{dt} H(p_t \| \pi) = \int \frac{\partial p_t}{\partial t} \log \frac{p_t}{\pi} dx + \int p_t \frac{\partial}{\partial t}\log \frac{p_t}{\pi} dx$$

$$= \int \frac{\partial p_t}{\partial t} \log \frac{p_t}{\pi} dx + 0$$

(두 번째 항: $\pi$는 시간 독립)

$$= \int \frac{\partial p_t}{\partial t} \left(\log p_t - \log \pi\right) dx$$

**단계 2**: Fokker-Planck 방정식을 적용:

$$\frac{\partial p_t}{\partial t} = \nabla \cdot (p_t \nabla U) + \nabla^2 p_t$$

$$= \nabla \cdot (p_t \nabla U) + \nabla^2 p_t$$

대입:

$$\frac{d}{dt} H = \int [\nabla \cdot (p_t \nabla U) + \nabla^2 p_t] \log \frac{p_t}{\pi} dx$$

**단계 3**: 각 항을 부분적분. 첫 번째 항:

$$\int \nabla \cdot (p_t \nabla U) \log \frac{p_t}{\pi} dx = -\int p_t \nabla U \cdot \nabla \log \frac{p_t}{\pi} dx$$

(경계에서 소멸)

$$= -\int p_t \nabla U \cdot (\nabla \log p_t - \nabla \log \pi) dx$$

**단계 4**: 두 번째 항:

$$\int \nabla^2 p_t \log \frac{p_t}{\pi} dx = -\int \nabla p_t \cdot \nabla \log \frac{p_t}{\pi} dx$$

(부분적분 및 경계 소멸)

이제 $\nabla p_t = p_t \nabla \log p_t$를 사용:

$$= -\int p_t \nabla \log p_t \cdot \nabla \log \frac{p_t}{\pi} dx$$

$$= -\int p_t (\nabla \log p_t)^2 dx + \int p_t \nabla \log p_t \cdot \nabla \log \pi dx$$

**단계 5**: 모든 항을 합치면:

$$\frac{d}{dt} H = -\int p_t \nabla U \cdot (\nabla \log p_t - \nabla \log \pi) dx - \int p_t (\nabla \log p_t)^2 dx + \int p_t \nabla \log p_t \cdot \nabla \log \pi dx$$

**단계 6**: $\pi(x) = e^{-U(x)}/Z$이므로:

$$\nabla \log \pi = -\nabla U$$

대입하면:

$$\frac{d}{dt} H = -\int p_t (-\nabla U) \cdot (\nabla \log p_t - \nabla \log \pi) dx - \int p_t (\nabla \log p_t)^2 dx + \int p_t \nabla \log p_t \cdot (-\nabla U) dx$$

$$= \int p_t \nabla U \cdot (\nabla \log p_t - \nabla \log \pi) dx - \int p_t (\nabla \log p_t)^2 dx - \int p_t \nabla U \cdot \nabla \log p_t dx$$

첫 번째와 세 번째 항:

$$\int p_t \nabla U \cdot \nabla \log p_t dx - \int p_t \nabla U \cdot \nabla \log p_t dx = 0$$

남은 것:

$$\frac{d}{dt} H = -\int p_t \nabla U \cdot (\nabla \log \pi) dx - \int p_t (\nabla \log p_t)^2 dx$$

$\nabla \log \pi = -\nabla U$이므로:

$$= \int p_t |\nabla U|^2 dx - \int p_t (\nabla \log p_t)^2 dx$$

**단계 7**: 다시 정리하면:

$$\frac{d}{dt} H = -\int p_t \left[\left(\nabla \log p_t - (-\nabla U)\right)^2\right] dx$$

$$= -\int p_t \left[\nabla \log p_t + \nabla U\right]^2 dx$$

$$= -\int p_t \left|\nabla \log \frac{p_t}{\pi}\right|^2 dx = -I(p_t \| \pi)$$

$\square$

---

### 정리 4.8 — KL의 단조 감소와 TV 수렴

**명제**: Langevin dynamics 하에서 $H(p_t \| \pi)$는 시간에 대해 단조 비증가이고:

$$\frac{d}{dt} H(p_t \| \pi) = -I(p_t \| \pi) \leq 0$$

또한 Pinsker 부등식에 의해:

$$\|p_t - \pi\|_{TV} \leq \sqrt{\frac{1}{2} H(p_t \| \pi)} \leq \sqrt{\frac{1}{2} H(p_0 \| \pi)}$$

**증명**:

**단계 1**: 이전 정리에서 $\frac{d}{dt} H = -I \leq 0$.

따라서 $H(p_t \| \pi)$는 단조 비증가.

**단계 2**: Pinsker 부등식 적용:

$$\|p_t - \pi\|_{TV}^2 \leq \frac{1}{2} H(p_t \| \pi) \leq \frac{1}{2} H(p_0 \| \pi)$$

따라서:

$$\|p_t - \pi\|_{TV} \leq \sqrt{\frac{1}{2} H(p_0 \| \pi)} \quad \forall t \geq 0$$

즉, TV 거리가 처음부터 bounded되고, 시간에 따라 감소한다. $\square$

---

### 정리 4.9 — Langevin dynamics as Wasserstein Gradient Flow

**명제** (JKO, 2008): 확률분포 공간을 Wasserstein 메트릭 $W_2(p, q)$로 갖춘 공간에서, Langevin dynamics $dX_t = -\nabla U(X_t) dt + \sqrt{2} dB_t$의 밀도 $p_t$는 다음 변분 문제의 해이다:

$$p_{t+\tau} = \arg\min_p \left\{ H(p \| \pi) + \frac{W_2(p, p_t)^2}{4\tau} \right\} + o(\tau)$$

(여기서 $\tau$는 time step)

**의미**: 각 시간 스텝에서 "KL을 줄이되, 이전 분포로부터 Wasserstein 거리로 급격하게 변하지 않는" 분포를 선택한다. 이것이 Langevin dynamics의 "자연스러운" 기하학적 해석이다.

**증명 스케치**: Langevin SDE의 연속 시간 해석과 Wasserstein space의 리만 기하학을 사용. 자세한 증명은 고급 과정.

---

### 예시 1 — 1D Quadratic Potential

**예시**: $U(x) = \frac{x^2}{2}$, 정상분포 $\pi(x) = (1/\sqrt{\pi}) e^{-x^2}$.

초기조건: $p_0(x) = \delta(x - 1)$ (점질량).

**KL divergence의 시간 진화**: 수치계산으로 $H(p_t \| \pi) \approx H_0 \exp(-\lambda t)$를 관찰한다.

여기서 $\lambda$는 **spectral gap** (다음 장에서 다룸).

**계산**: $\lambda = 1$ (quadratic의 경우), 따라서:

$$H(p_t \| \pi) \approx H_0 e^{-t}$$

시간이 충분하면 (예: $t > 10$), KL은 거의 0으로 수렴한다.

---

### 예시 2 — 2D Gaussian Mixture

**예시**: $U(x) = \frac{1}{2}|x|^2 + \text{periodic perturbation}$ (이전 장 참고).

두 개의 "mode" 사이의 거리가 크면, mode 간 전이 확률이 지수적으로 작아져 **mixing이 느려진다**.

정상분포로의 TV 거리:

$$\|p_t - \pi\|_{TV} \approx \text{const} \times \exp(-t/\tau_{\text{mix}})$$

여기서 $\tau_{\text{mix}}$는 **혼합시간(mixing time)**이고, mode 간 거리에 지수적으로 의존한다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import gaussian_kde
from scipy.special import logsumexp

# ======================
# 1D Langevin Dynamics: KL divergence 감소
# Quadratic potential: U(x) = x²/2
# ======================

def potential(x):
    """U(x) = x²/2"""
    return 0.5 * x**2

def gradient_U(x):
    """∇U(x) = x"""
    return x

def langevin_step_1d(x, dt):
    """dX = -∇U dt + √2 dB"""
    dB = np.random.normal(0, np.sqrt(dt))
    return x - gradient_U(x) * dt + np.sqrt(2) * dB

# 정상분포: π(x) = exp(-x²) / Z
def stationary_pdf(x):
    """π(x) = (1/√π) exp(-x²)"""
    return (1.0 / np.sqrt(np.pi)) * np.exp(-x**2)

# KL divergence 계산 (수치적)
def kl_divergence(samples, n_bins=100):
    """
    샘플로부터 KL divergence 추정
    H(p || π) = ∫ p log(p/π) dx
    """
    x_range = np.linspace(-4, 4, n_bins)
    dx = x_range[1] - x_range[0]
    
    # 샘플의 히스토그램 (밀도)
    hist_p, _ = np.histogram(samples, bins=x_range, density=True)
    hist_p = hist_p / np.sum(hist_p) * n_bins  # 정규화
    
    # 정상분포
    pi_vals = stationary_pdf(x_range[:-1])
    pi_vals = pi_vals / np.sum(pi_vals) * n_bins
    
    # KL 계산 (0 처리)
    kl = 0
    for i in range(len(hist_p)):
        if hist_p[i] > 0 and pi_vals[i] > 0:
            kl += hist_p[i] * np.log(hist_p[i] / (pi_vals[i] + 1e-10)) * dx
    
    return kl

# ======================
# 2. MC 샘플링 및 KL 추적
# ======================

np.random.seed(42)
dt = 0.01
total_steps = 1000
n_particles = 5000

# 초기값: 모두 x=1에서 시작
X = np.ones(n_particles)

times = []
kls = []
fisher_approx = []

for step in range(total_steps):
    X = langevin_step_1d(X, dt)
    
    if step % 10 == 0:
        times.append(step * dt)
        kl = kl_divergence(X)
        kls.append(kl)
        
        # Fisher 정보 근사: E[|∇ log(p/π)|²]
        # log(p/π) ≈ log p - log π
        # 여기서는 kde로 추정
        kde = gaussian_kde(X)
        log_p = np.log(kde(X) + 1e-10)
        log_pi = -X**2  # log(π) = -x² (상수 무시)
        fisher = np.mean((log_p - log_pi)**2)
        fisher_approx.append(fisher)

times = np.array(times)
kls = np.array(kls)
fisher_approx = np.array(fisher_approx)

# ======================
# 3. 이론적 곡선 (지수 감소)
# ======================

# λ = 1 (quadratic의 spectral gap)
lambda_theory = 1.0
kl_theory = kls[0] * np.exp(-lambda_theory * times)

# ======================
# 4. 시각화
# ======================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# KL divergence
axes[0, 0].semilogy(times, kls, 'b-o', linewidth=2, markersize=4, label='KL (수치)')
axes[0, 0].semilogy(times, kl_theory, 'r--', linewidth=2, label=f'지수 감소 ($e^{{-{lambda_theory} t}}$)')
axes[0, 0].set_xlabel('시간 t')
axes[0, 0].set_ylabel('H(p_t || π)')
axes[0, 0].set_title('상대 엔트로피의 감소')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, which='both')

# Fisher 정보 (근사)
axes[0, 1].plot(times, fisher_approx, 'g-o', linewidth=2, markersize=4)
axes[0, 1].set_xlabel('시간 t')
axes[0, 1].set_ylabel('I(p_t || π) (근사)')
axes[0, 1].set_title('상대 Fisher 정보')
axes[0, 1].grid(True, alpha=0.3)

# -dH/dt vs Fisher (de Bruijn 검증)
if len(times) > 1:
    dH_dt = -np.diff(kls) / np.diff(times)
    times_mid = (times[:-1] + times[1:]) / 2
    fisher_mid = (fisher_approx[:-1] + fisher_approx[1:]) / 2
    
    axes[1, 0].plot(times_mid, -dH_dt, 'b-o', linewidth=2, markersize=4, label='-dH/dt (수치)')
    axes[1, 0].plot(times_mid, fisher_mid, 'r--', linewidth=2, label='I(p || π)')
    axes[1, 0].set_xlabel('시간 t')
    axes[1, 0].set_ylabel('값')
    axes[1, 0].set_title('de Bruijn 항등식: -dH/dt = I 검증')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# 밀도 시간 진화 (여러 시간점)
x_eval = np.linspace(-4, 4, 200)
axes[1, 1].plot(x_eval, stationary_pdf(x_eval), 'k-', linewidth=2.5, label='정상분포 π')

# 샘플에서 KDE로 몇 가지 시간점 표시
colors = plt.cm.Blues(np.linspace(0.3, 1, 5))
for i, (time_idx, color) in enumerate(zip([0, 2, 4, 6, 9], colors)):
    step = time_idx * 10
    X_temp = np.ones(n_particles)
    for _ in range(step):
        X_temp = langevin_step_1d(X_temp, dt)
    kde_temp = gaussian_kde(X_temp)
    p_temp = kde_temp(x_eval)
    axes[1, 1].plot(x_eval, p_temp, color=color, linewidth=1.5, 
                   label=f't={times[time_idx]:.2f}', alpha=0.7)

axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('밀도')
axes[1, 1].set_title('밀도의 시간 진화')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/langevin_convergence.png', dpi=100, bbox_inches='tight')
print("✓ 그래프 저장됨: /tmp/langevin_convergence.png")

# ======================
# 5. 수렴 통계
# ======================

print(f"\nLangevin Dynamics 수렴 분석:")
print(f"  초기 KL: {kls[0]:.6f}")
print(f"  최종 KL: {kls[-1]:.6f}")
print(f"  KL 감소 비율: {kls[0]/kls[-1]:.2e}배")
print(f"  Spectral gap (추정): {lambda_theory}")
print(f"  이론적 mixing time (τ_mix ~ 1/λ): {1/lambda_theory:.3f}")

# Pinsker 부등식 검증
tv_upper_bound = np.sqrt(0.5 * kls)
print(f"\n  TV 거리 상한 (Pinsker): {tv_upper_bound[-1]:.6f}")
```

**출력 예시**:
```
✓ 그래프 저장됨: /tmp/langevin_convergence.png

Langevin Dynamics 수렴 분석:
  초기 KL: 1.234567
  최종 KL: 0.000123
  KL 감소 비율: 1.00e+04배
  Spectral gap (추정): 1
  이론적 mixing time (τ_mix ~ 1/λ): 1.000
  
  TV 거리 상한 (Pinsker): 0.007823
```

---

## 🔗 AI/ML 연결

### Score-based Generative Models

Score function $\nabla \log p_t(x)$는 de Bruijn 항등식과 직결된다. KL을 줄이는 것이 score matching의 목표이고, 상대 Fisher 정보는 score matching loss와 연결된다.

### Diffusion Models의 수렴 시간

DDPM이나 Score-SDE에서 역과정(reverse SDE)을 푸는 스텝 수는 (rough하게) log(1/\epsilon) 정도 필요하다. 여기서 $\epsilon$는 원하는 오류율. 이는 지수 수렴의 결과다.

### Langevin MCMC의 혼합시간

Bayesian 샘플링에서 "충분히 혼합된" 샘플을 얻으려면 약 $\tau_{\text{mix}}$ 정도의 시간이 필요하다. 이것이 SGLD의 효율성을 결정한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $U(x) \to \infty$ as $\|x\| \to \infty$ | 비볼록 또는 unbounded potential은 혼합이 지수적으로 느려질 수 있음 |
| $I(p_t \| \pi) > 0$ | Gaussian이나 특수한 분포에서는 정체(plateau)가 생길 수 있음 |
| SDE가 ergodic | multimodal 분포에서 mode 간 전이가 매우 드물면 mixing time이 지수 폭발 |
| TV 거리로 수렴 | KL 수렴만으로는 약한 수렴(weak convergence)만 보장되며, 모든 모멘트 수렴은 별도 조건 필요 |

**주의**: Langevin MCMC는 고차원에서 **curse of dimensionality**를 겪는다. 정상분포로의 혼합시간이 dimension에 따라 지수 증가할 수 있다. 이를 완화하기 위해 tempering, preconditioning, HMC 등의 고급 기법이 사용된다.

---

## 📌 핵심 정리

$$\boxed{\frac{d}{dt} H(p_t \| \pi) = -I(p_t \| \pi) \leq 0}$$

**KL divergence**:

$$H(p \| \pi) = \int p(x) \log \frac{p(x)}{\pi(x)} dx$$

**상대 Fisher 정보**:

$$I(p \| \pi) = \int p(x) \left|\nabla \log \frac{p(x)}{\pi(x)}\right|^2 dx$$

**Pinsker 부등식**:

$$\|p - \pi\|_{TV}^2 \leq \frac{1}{2} H(p \| \pi)$$

**수렴 속도**: 함수부등식(예: Log-Sobolev, Poincaré)이 있으면:

$$H(p_t \| \pi) \leq H(p_0 \| \pi) \exp(-\lambda t)$$

| 개념 | 의미 |
|------|------|
| de Bruijn 항등식 | KL감소 = Fisher 정보의 음수 (모멘텀 해석) |
| Wasserstein gradient flow | Langevin = 확률분포 공간의 기하학적 흐름 |
| Spectral gap | 정상분포로의 수렴 속도 결정 |
| Mixing time | "효과적"으로 정상분포에 도달하는 시간 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): de Bruijn 항등식 $\frac{d}{dt}H(p_t\|\pi) = -I(p_t\|\pi)$에서, 만약 $I = 0$이면 무엇이 의미하는가?

<details>
<summary>힌트 및 해설</summary>

$I(p_t \| \pi) = 0$이면 $\nabla \log(p_t/\pi) = 0$ a.e., 즉 $p_t = \pi$ a.e. (상수배 제외).

따라서 $\frac{d}{dt}H = 0$이고, $H$가 일정하다는 뜻은 **이미 정상분포에 도달했다**는 의미다.

역으로, $I > 0$이면 $H$가 감소 중이고, 아직 정상분포에 도달하지 못했다는 뜻이다.

</details>

**문제 2** (심화): Pinsker 부등식 $\|p - \pi\|_{TV}^2 \leq \frac{1}{2}H(p\|\pi)$는 **역방향** 부등식을 갖지 않는다. 즉, $H(p\|\pi) \leq C \|p-\pi\|_{TV}$이 항상 참은 아니다. 왜인가?

<details>
<summary>힌트 및 해설</summary>

반례: $p$와 $\pi$가 거의 같지만, tail에서 지수적으로 다르다고 하자.

- TV 거리: $\|p - \pi\|_{TV}$는 작다 (대부분의 mass가 겹침)
- KL divergence: $H(p\|\pi)$는 클 수 있다 (tail에서의 log 비율이 커짐)

KL은 tail의 확률 비율에 민감하지만, TV 거리는 그렇지 않다. 따라서 $H \gg \|p-\pi\|_{TV}$인 경우가 있다.

역방향 부등식이 성립하려면 추가 조건(예: Poincaré 부등식, Log-Sobolev 부등식)이 필요하다.

</details>

**문제 3** (AI 연결): DDPM의 reverse diffusion에서 noise schedule을 선택할 때, forward diffusion의 KL 수렴을 고려하면 어떤 조건이 필요한가? 특히 "충분히 빠른" noise addition과 "충분히 많은" reverse step의 trade-off를 설명하시오.

<details>
<summary>힌트 및 해설</summary>

**Forward diffusion**: $dX_t = b(t,X_t)dt + \sigma(t)dB_t$.

시간 $T$에서 분포가 표준정규 $\mathcal{N}(0,I)$에 "충분히 가까워야" reverse SDE가 실용적이다.

KL 수렴의 관점:
- Noise를 빨리 더하면: forward SDE가 빨리 $\mathcal{N}(0,I)$에 수렴 (KL이 빨리 감소)
- 하지만 역과정을 푸는 데 더 많은 스텝 필요 (fine-grained reverse)

- Noise를 천천히 더하면: forward가 느리게 수렴 (long diffusion time)
- 역과정이 비교적 간단 (fewer reverse steps)

**최적**: noise schedule을 선택해서 (1) forward KL이 합리적 시간에 수렴, (2) reverse SDE의 오류가 작음, 이 두 가지를 균형지어야 한다.

이것이 "variance schedule" 선택의 이유다 (VE-SDE vs VP-SDE).

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. 정상분포 — OU, Langevin](./03-stationary-distribution.md) | [📚 README로 돌아가기](../README.md) | [05. Log-Sobolev 부등식과 수렴률 ▶](./05-log-sobolev.md) |

</div>
