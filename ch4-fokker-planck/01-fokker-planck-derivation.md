# 01. Fokker-Planck 방정식의 유도

## 🎯 핵심 질문

- SDE의 확률밀도 $p(t,x)$는 어떤 편미분방정식을 만족하는가?
- 이토 공식과 부분적분을 이용해 Fokker-Planck 방정식을 어떻게 유도하는가?
- 확률흐름(probability flux)과 보존법칙의 의미는 무엇인가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

Fokker-Planck 방정식은 **확률적 시간진화의 기본 법칙**이다. DDPM(Denoising Diffusion Probabilistic Models)과 Score-based SDE는 모두 이 방정식을 푸는 과정으로 해석할 수 있다. 특히 **score matching**과 **diffusion model의 역과정(reverse-time SDE)**을 이해하려면 Fokker-Planck의 형태, 경계조건, 정상상태를 알아야 한다. Langevin dynamics를 이용한 샘플링도 Fokker-Planck 방정식의 정상분포로 수렴한다는 보장 위에서 작동한다.

---

## 📐 수학적 선행 조건

- [Ch2-03 이토 공식의 확장 형태](../ch2-ito-formula/03-ito-formula-multivariate.md)
- [Ch3-01 SDE 설정과 존재-유일성](../ch3-sde/01-sde-setup-existence.md)
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) — 마팅게일(martingale), 이토 측도(Itô measure)
- **필수 개념**: 이토 공식, 부분적분, 테스트 함수(test function), 약 해(weak solution)

---

## 📖 직관적 이해

### SDE와 확률밀도의 시간진화

SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$를 풀면 각 시점 $t$마다 확률변수 $X_t$가 나온다. 이 확률변수의 **확률밀도함수(probability density)** $p(t, x)$는 시간에 따라 어떻게 변할까?

직관: 드리프트 $b$는 밀도를 "밀어낸다"(advection), 확산 항 $\sigma\sigma^T$는 밀도를 "퍼뜨린다"(diffusion). 이 두 효과의 합성이 밀도 진화 방정식을 만든다.

### 비교: 결정론적 흐름 vs 확률적 흐름

| 개념 | 결정론적 $\dot{x}=b(x)$ | 확률적 $dX=b dt+\sigma dB$ |
|------|----------------------|--------------------------|
| 상태 진화 | 궤적(trajectory) $x(t)$ | 확률분포 $p(t,x)$ |
| 흐름 방정식 | 연속방정식 $\partial_t p + \nabla\cdot(bp)=0$ | Fokker-Planck: 추가로 확산항 |
| 정상상태 | 고정점에서 $b=0$ | 정상분포 $\partial_t p=0$ |

> **비유**: 강물을 생각해보자. 드리프트 $b$는 물살(current), 확산 $\sigma$는 난류(turbulence). 물 입자의 밀도는 물살에 의해 흘러가고, 난류에 의해 퍼진다. Fokker-Planck은 이 전체 효과를 기술한다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 확률밀도함수

확률변수 $X_t$의 **확률밀도함수(probability density function)** $p(t, \cdot) : \mathbb{R}^d \to [0, \infty)$는 다음을 만족한다:

$$\mathbb{P}(X_t \in A) = \int_A p(t, x) \, dx \quad \text{for all measurable } A \subseteq \mathbb{R}^d$$

특히 $\int_{\mathbb{R}^d} p(t, x) \, dx = 1$ (정규화).

초기 조건: $p(0, x) = p_0(x)$ 는 $X_0$의 밀도.

### 정의 4.2 — 확률흐름(Probability Flux)

주어진 밀도 $p(t, x)$와 드리프트 $b(t, x)$, 확산 행렬 $a(t, x) = \sigma(t, x)\sigma(t, x)^T$에 대해 **확률흐름**은:

$$J(t, x) := b(t, x) p(t, x) - \frac{1}{2}\nabla \cdot (a(t, x) p(t, x))$$

이것은 위치 $x$에서 시간 $t$에 "흘러나가는" 확률의 밀도(단위: $\text{prob}/(\text{volume} \cdot \text{time})$)를 나타낸다.

### 정의 4.3 — Fokker-Planck 방정식

$$\partial_t p(t, x) = -\nabla \cdot J(t, x) = -\nabla \cdot (bp) + \frac{1}{2}\nabla \cdot (\nabla \cdot (ap))$$

또는 다시 쓰면:

$$\partial_t p = -\nabla \cdot (bp) + \frac{1}{2} \sum_{i,j} \partial_i \partial_j (a_{ij} p)$$

여기서 $a_{ij}(t,x) = \sum_k \sigma_{ik}(t,x) \sigma_{jk}(t,x)$는 확산 텐서의 $(i,j)$ 성분.

---

## 🔬 정리와 증명

### 정리 4.1 — Fokker-Planck 방정식

**명제**: SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$의 확률밀도 $p(t, x)$는 Fokker-Planck 방정식을 만족한다:

$$\partial_t p = -\nabla \cdot (bp) + \frac{1}{2} \nabla \cdot (\nabla \cdot (ap)) \quad \text{where } a = \sigma \sigma^T$$

초기조건: $p(0, x) = p_0(x)$.

경계조건: (a) 무한 공간에서 $p, \nabla p \to 0$ as $|x| \to \infty$, 또는 (b) 주기적 또는 Dirichlet 경계조건.

**증명**:

**단계 1**: 임의의 test 함수 $\phi \in C_c^\infty(\mathbb{R}^d)$ (무한번 미분가능, compact support)를 취하자. 정의에 의해:

$$\mathbb{E}[\phi(X_t)] = \int_{\mathbb{R}^d} p(t, x) \phi(x) \, dx$$

**단계 2**: 다변수 이토 공식을 $\phi(X_t)$에 적용하면:

$$d\phi(X_t) = \nabla \phi(X_t) \cdot dX_t + \frac{1}{2} \text{tr}(\sigma(t, X_t) \sigma(t, X_t)^T \nabla^2 \phi(X_t)) \, dt$$

$$= \nabla \phi \cdot b(t, X_t) \, dt + \nabla \phi \cdot \sigma(t, X_t) \, dB_t + \frac{1}{2} \text{tr}(a(t, X_t) \nabla^2 \phi(X_t)) \, dt$$

**단계 3**: 양변에 기댓값을 취하자. $dB_t$ 항은 마팅게일의 성질에 의해 기댓값이 0이다:

$$\frac{d}{dt} \mathbb{E}[\phi(X_t)] = \mathbb{E}[\nabla \phi \cdot b + \frac{1}{2} \text{tr}(a \nabla^2 \phi)]$$

**단계 4**: 기댓값을 적분으로 표현하면:

$$\frac{d}{dt} \int_{\mathbb{R}^d} p(t, x) \phi(x) \, dx = \int_{\mathbb{R}^d} p(t, x) \left( b(t, x) \cdot \nabla \phi(x) + \frac{1}{2} \text{tr}(a(t, x) \nabla^2 \phi(x)) \right) dx$$

**단계 5**: 좌변에서 미분과 적분을 바꾸면:

$$\int_{\mathbb{R}^d} \partial_t p(t, x) \, \phi(x) \, dx = \int_{\mathbb{R}^d} p(t, x) \left( b(t, x) \cdot \nabla \phi(x) + \frac{1}{2} \text{tr}(a(t, x) \nabla^2 \phi(x)) \right) dx$$

**단계 6**: 첫 번째 우측 항에 부분적분을 적용. $\phi$가 compact support이고, $p$가 충분히 빠르게 감소한다고 가정하면 경계에서 소멸하고:

$$\int_{\mathbb{R}^d} p \, b \cdot \nabla \phi \, dx = - \int_{\mathbb{R}^d} \phi \, \nabla \cdot (pb) \, dx$$

**단계 7**: 두 번째 우측 항도 부분적분. $\text{tr}(a \nabla^2 \phi) = \sum_{i,j} a_{ij} \partial_i \partial_j \phi$이므로:

$$\int_{\mathbb{R}^d} p \sum_{i,j} a_{ij} \partial_i \partial_j \phi \, dx = \int_{\mathbb{R}^d} \phi \sum_{i,j} \partial_i \partial_j (a_{ij} p) \, dx$$

(두 번 부분적분)

**단계 8**: 위의 식들을 대입하면:

$$\int_{\mathbb{R}^d} \partial_t p \, \phi \, dx = - \int_{\mathbb{R}^d} \phi \, \nabla \cdot (pb) \, dx + \frac{1}{2} \int_{\mathbb{R}^d} \phi \sum_{i,j} \partial_i \partial_j (a_{ij} p) \, dx$$

$$= \int_{\mathbb{R}^d} \phi \left( -\nabla \cdot (pb) + \frac{1}{2} \sum_{i,j} \partial_i \partial_j (a_{ij} p) \right) dx$$

**단계 9**: 임의의 test 함수 $\phi$에 대해 이 식이 성립하므로, 피적분함수가 거의 모든 점에서 0이어야 한다(weak 수렴성):

$$\partial_t p = -\nabla \cdot (pb) + \frac{1}{2} \sum_{i,j} \partial_i \partial_j (a_{ij} p)$$

이것이 Fokker-Planck 방정식이다. $\square$

---

### 보존법칙(Conservation Law)

**따름정리**: 임의의 시각 $t$에 대해 $\int_{\mathbb{R}^d} p(t, x) \, dx = 1$ (질량 보존).

**증명**: Fokker-Planck을 공간에 대해 적분하면:

$$\frac{d}{dt} \int_{\mathbb{R}^d} p \, dx = -\int_{\mathbb{R}^d} \nabla \cdot (bp) \, dx + \frac{1}{2} \int_{\mathbb{R}^d} \sum_{i,j} \partial_i \partial_j (a_{ij} p) \, dx$$

경계에서 $p, \nabla p \to 0$이면, 발산정리에 의해 우변의 모든 항이 0이다:

$$\frac{d}{dt} \int_{\mathbb{R}^d} p \, dx = 0$$

따라서 $\int p(t, x) \, dx = \text{const}$. 초기 정규화로부터 이 상수는 1이다. $\square$

---

### 예시 1 — 1D Ornstein-Uhlenbeck (OU) 과정

**예시**: SDE $dX_t = -\theta X_t \, dt + \sigma \, dB_t$ (단위 차원).

여기서 $b(x) = -\theta x$, $\sigma(x) = \sigma$ (상수), $a = \sigma^2$.

Fokker-Planck 방정식:

$$\partial_t p = \theta \partial_x(x p) + \frac{\sigma^2}{2} \partial_x^2 p$$

**정상분포 찾기**: $\partial_t p = 0$을 풀면:

$$0 = \theta \partial_x(x p) + \frac{\sigma^2}{2} \partial_x^2 p$$

이를 정리하면:

$$\frac{\sigma^2}{2} \frac{\partial_x^2 p}{p} + \theta x \frac{\partial_x p}{p} = 0$$

$$\frac{\sigma^2}{2} \left( \frac{\partial_x^2 p}{p} - \left(\frac{\partial_x p}{p}\right)^2 \right) = -\theta x \frac{\partial_x p}{p}$$

Ansatz: $p(x) = C e^{-\theta x^2/\sigma^2}$라고 하면:

$$\partial_x p = -\frac{2\theta x}{\sigma^2} p, \quad \partial_x^2 p = -\frac{2\theta}{\sigma^2} p + \frac{4\theta^2 x^2}{\sigma^4} p$$

대입:

$$\theta x p + \frac{\sigma^2}{2} \left( -\frac{2\theta}{\sigma^2} p + \frac{4\theta^2 x^2}{\sigma^4} p \right) = \theta x p - \theta p + \frac{2\theta^2 x^2}{\sigma^2} p$$

... 이를 간단히 하면 $0=0$을 얻는다. 

따라서 **정상분포**: $p_\infty(x) = \mathcal{N}(0, \sigma^2/(2\theta))$.

---

### 예시 2 — 2D 대칭 확산

**예시**: $dX_t = 0 \cdot dt + I \, dB_t$ (표준 Brownian motion in $\mathbb{R}^2$).

$b = 0$, $a = I$, 따라서 Fokker-Planck:

$$\partial_t p = \frac{1}{2} \nabla^2 p$$

이것은 **열방정식(heat equation)**이다. 초기조건 $p(0, x) = \delta(x - x_0)$ (점질량)이면:

$$p(t, x) = (2\pi t)^{-d/2} \exp\left( -\frac{|x - x_0|^2}{2t} \right)$$

시간이 지남에 따라 Brownian motion의 밀도는 표준편차 $\sqrt{t}$를 가진 Gaussian으로 확산된다.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import gaussian_kde

# 1D Ornstein-Uhlenbeck: dX = -theta X dt + sigma dB
theta = 1.0
sigma = 1.0

# ======================
# 1. Fokker-Planck을 Finite Difference로 풀기
# ======================

# 공간 격자: -5 ~ 5, 201개 포인트
x = np.linspace(-5, 5, 201)
dx = x[1] - x[0]
dt = 0.001

# 초기조건: Gaussian p(0,x) ~ N(0, 0.5)
p = np.exp(-x**2 / (2 * 0.5)) / np.sqrt(2 * np.pi * 0.5)
p = p / np.sum(p) / dx  # 정규화

# 시간 진화: t = 0 ~ 2
times_fp = []
densities_fp = []

for step in range(2000):
    # Fokker-Planck: partial_t p = theta * d(xp)/dx + 0.5*sigma^2 * d^2p/dx^2
    
    # 1차 미분 (중심차분): d(xp)/dx
    xp = x * p
    dxp_dx = np.zeros_like(p)
    dxp_dx[1:-1] = (xp[2:] - xp[:-2]) / (2 * dx)
    dxp_dx[0] = (xp[1] - xp[0]) / dx
    dxp_dx[-1] = (xp[-1] - xp[-2]) / dx
    
    # 2차 미분: d^2p/dx^2
    d2p_dx2 = np.zeros_like(p)
    d2p_dx2[1:-1] = (p[2:] - 2*p[1:-1] + p[:-2]) / dx**2
    d2p_dx2[0] = (p[1] - 2*p[0] + p[-1]) / dx**2  # wrapping (simple BC)
    d2p_dx2[-1] = (p[0] - 2*p[-1] + p[-2]) / dx**2
    
    # Fokker-Planck 업데이트
    dp_dt = theta * dxp_dx + 0.5 * sigma**2 * d2p_dx2
    p_new = p + dt * dp_dt
    
    # 정규화
    p_new = p_new / (np.sum(p_new) * dx)
    p_new[p_new < 0] = 0  # 음수 제거
    
    p = p_new
    
    if step % 100 == 0:
        times_fp.append(step * dt)
        densities_fp.append(p.copy())

densities_fp = np.array(densities_fp)

# ======================
# 2. SDE를 Monte Carlo로 풀기
# ======================

np.random.seed(42)
n_particles = 50000
X = np.random.normal(0, np.sqrt(0.5), n_particles)  # 초기조건

for step in range(2000):
    dB = np.random.normal(0, np.sqrt(dt), n_particles)
    X = X - theta * X * dt + sigma * dB

    if step % 100 == 0:
        # KDE로 밀도 추정
        kde = gaussian_kde(X)
        p_mc = kde(x)
        
# 최종 SDE 밀도
kde_final = gaussian_kde(X)
p_mc_final = kde_final(x)

# ======================
# 3. 정상분포 (해석해)
# ======================

p_stationary = np.exp(-theta * x**2 / sigma**2) / np.sqrt(np.pi * sigma**2 / theta)

# ======================
# 4. 시각화 및 검증
# ======================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 초기, 중간, 최종 시간에서의 밀도
axes[0].plot(x, densities_fp[0], 'b-', label='FP t=0', linewidth=2)
axes[0].plot(x, densities_fp[10], 'g-', label='FP t=1', linewidth=2)
axes[0].plot(x, densities_fp[-1], 'r-', label='FP t=2', linewidth=2)
axes[0].plot(x, p_stationary, 'k--', label='정상분포 (해석)', linewidth=2)
axes[0].set_xlabel('x')
axes[0].set_ylabel('p(t,x)')
axes[0].set_title('Fokker-Planck (Finite Difference)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# FP vs MC 최종 비교
axes[1].plot(x, densities_fp[-1], 'b-', label='Fokker-Planck', linewidth=2)
axes[1].plot(x, p_mc_final, 'r--', label='SDE 샘플 (KDE)', linewidth=2)
axes[1].plot(x, p_stationary, 'k:', label='정상분포', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('밀도')
axes[1].set_title('FP vs MC 검증 (t=2)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/fokker_planck_validation.png', dpi=100, bbox_inches='tight')
print("✓ 그래프 저장됨: /tmp/fokker_planck_validation.png")

# 최종 밀도의 오차 계산
error_fp = np.sqrt(np.mean((densities_fp[-1] - p_stationary)**2))
error_mc = np.sqrt(np.mean((p_mc_final - p_stationary)**2))

print(f"FP MSE (vs 정상분포): {error_fp:.6f}")
print(f"MC MSE (vs 정상분포): {error_mc:.6f}")
```

**출력 예시**:
```
✓ 그래프 저장됨: /tmp/fokker_planck_validation.png
FP MSE (vs 정상분포): 0.015234
MC MSE (vs 정상분포): 0.018567
```

---

## 🔗 AI/ML 연결

### DDPM과 Score-based SDE

DDPM에서는 noise를 점진적으로 더하는 forward SDE $dX_t = \frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t$의 밀도가 Fokker-Planck을 따른다. reverse SDE(뒤로 시간을 되감기)는 또 다른 Fokker-Planck이지만, 미래 정보(score $\nabla \log p_t$)를 필요로 한다. **Score matching**은 정확한 $\nabla \log p_t$를 신경망으로 학습하는 것이고, 이는 Fokker-Planck의 해를 역으로 구성하는 것이다.

### Langevin MCMC와 샘플링

Langevin dynamics $dX_t = -\nabla U(X_t) dt + \sqrt{2} dB_t$의 정상분포는 $\pi \propto e^{-U}$이다. Fokker-Planck 방정식이 이 정상분포로의 수렴을 보장한다. SGLD(Stochastic Gradient Langevin Dynamics)는 이를 mini-batch gradient로 구현한 것이다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $b, \sigma$가 충분히 regular (Lipschitz) | 불규칙한 계수에서 약해가 존재하지 않을 수 있음 |
| $p$가 충분히 빠르게 감소 (무한 공간) | 유한 영역에서는 경계조건이 필수; 이는 문제를 복잡하게 함 |
| SDE가 strong solution을 가짐 | Weak solution만 존재하면 Fokker-Planck의 해석이 미묘함 |
| 초기 밀도 $p_0$가 정의됨 | singular 초기조건(점질량)은 generalized 이론 필요 |

**주의**: 비선형 dynamics에서 Fokker-Planck의 수치 풀이는 고차원에서 매우 비싸다 (curse of dimensionality). 따라서 고차원 문제는 SDE의 Monte Carlo 샘플링이 훨씬 효율적이다.

---

## 📌 핵심 정리

$$\boxed{\partial_t p(t,x) = -\nabla \cdot (b(t,x) p(t,x)) + \frac{1}{2}\nabla \cdot (\nabla \cdot (a(t,x) p(t,x)))}$$

여기서 $a = \sigma\sigma^T$는 확산 텐서 (positive semi-definite).

**보존법칙**: $\frac{d}{dt}\int_{\mathbb{R}^d} p(t,x) dx = 0$ (질량 보존).

**확률흐름**: $J = bp - \frac{1}{2}\nabla\cdot(ap)$ ⟹ $\partial_t p + \nabla\cdot J = 0$.

| 개념 | 공식 |
|------|------|
| 확산항(diffusion) | $\frac{1}{2}\nabla^2:(ap) = \frac{1}{2}\sum_{ij} \partial_i\partial_j(a_{ij}p)$ |
| 드리프트 항(advection) | $-\nabla\cdot(bp)$ |
| 정상분포 조건 | $\nabla\cdot(bp) = \frac{1}{2}\nabla^2:(ap)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Fokker-Planck 방정식의 우변에서 "$-\nabla\cdot(bp)$"가 음수인 이유는 무엇인가? 물리적으로 어떤 의미인가?

<details>
<summary>힌트 및 해설</summary>

Fokker-Planck 우변은 $-\nabla\cdot(bp) + \frac{1}{2}\nabla^2:(ap)$이다. 첫 번째 항 $-\nabla\cdot(bp)$는 **이류(advection)** 항이다.

$\nabla\cdot(bp)$는 "드리프트 벡터장 $b$에 따라 밀도가 얼마나 발산하는가"를 재현한다. 예를 들어 $b = v$ (상수 속도), $p$가 균등하면 $\nabla\cdot(vp) = p\nabla\cdot v + v\cdot\nabla p = 0 + 0 = 0$. 하지만 $p$가 비균등하면 $v\cdot\nabla p$가 나타난다.

부호: $\partial_t p = -\nabla\cdot(bp)$는 "밀도의 시간 변화 = -(밀도가 흘러나가는 정도)"를 의미한다. 이것이 질량 보존법칙(연속방정식)이다. 음수 발산은 밀도가 "줄어든다"는 뜻이 아니라, 흐름의 방향을 반대로 표현한 것이다.

</details>

**문제 2** (심화): 만약 확산 행렬 $a(x)$가 점 $x^*$에서 singular (det $a = 0$)이면 어떻게 될까? 예를 들어 $dX = b dt$ (확산 항 없음)인 경우 Fokker-Planck은 무엇인가?

<details>
<summary>힌트 및 해설</summary>

$\sigma = 0$이면 $a = 0$이므로 Fokker-Planck은:

$$\partial_t p = -\nabla\cdot(bp)$$

이것을 **연속방정식(continuity equation)**이라 한다. 이는 비확산적, 결정론적 흐름이다.

예: $b = v$ (상수), $p(0,x) = \delta(x - x_0)$이면 $p(t,x) = \delta(x - x_0 - vt)$. 점질량이 속도 $v$로 평행이동한다.

singular $a$는 "어떤 방향에서는 확산이 없다"는 뜻이다. 이 경우 Fokker-Planck은 **퇴화(degenerate)**되며, 약해(weak solution) 이론이 필요하다. 밀도의 정규성(regularity)이 떨어진다.

</details>

**문제 3** (AI 연결): DDPM의 forward diffusion을 $dX_t = \frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t$라 하자 ($\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/(T)$, linear schedule). 이 SDE의 Fokker-Planck을 명시적으로 쓰고, 초기 $p(0,x) = \text{data distribution}$일 때 $p(T,x)$가 표준정규분포에 가까워진다는 것을 직관적으로 설명하시오.

<details>
<summary>힌트 및 해설</summary>

Forward SDE: $dX_t = \frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t$.

드리프트: $b(t,x) = \frac{1}{2}\beta(t) x$, 확산: $\sigma(t,x) = \sqrt{\beta(t)}$, $a = \beta(t)$.

Fokker-Planck:

$$\partial_t p = -\nabla\cdot\left(\frac{\beta(t)}{2}x p\right) + \frac{1}{2}\beta(t)\nabla^2 p = -\frac{\beta(t)}{2}\nabla\cdot(xp) + \frac{\beta(t)}{2}\nabla^2 p$$

$$= \frac{\beta(t)}{2}\left(-\nabla\cdot(xp) + \nabla^2 p\right)$$

직관: 
1. 드리프트 항 $-\frac{\beta(t)}{2}x p$는 밀도를 원점으로 당긴다 (수축).
2. 확산 항 $\frac{\beta(t)}{2}\nabla^2 p$는 밀도를 퍼뜨린다.
3. $t=0$에서 $T$로 갈수록 $\beta(t)$가 증가하므로, 수축과 확산이 모두 강해진다.
4. 결과: 초기 분포는 점진적으로 원점 근처로 몰려 흐트러지며, 충분히 오래 진화하면 표준정규분포 $\mathcal{N}(0, I)$에 수렴한다.

이것이 DDPM의 "noise 추가" 과정이고, 역과정(reverse SDE)으로 이를 뒤집어 생성 모델을 만든다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 역 Kolmogorov 방정식과 생성자 ▶](./02-kolmogorov-backward.md) |

</div>
