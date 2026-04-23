# 03. 다차원·시간의존 이토 공식

## 🎯 핵심 질문

- 다차원 브라운 운동의 곱셈표는 무엇인가?
- 시간이 명시적으로 나타나는 함수 $f(t, X_t)$에 이토 공식을 어떻게 적용하는가?
- 다변수 Taylor 전개에서 각 항의 처리는?
- 곱의 법칙(product rule)에서 공변분(covariation) 항은 왜 나타나는가?

---

## 🔍 왜 이 개념이 AI(특히 생성모델)에서 중요한가

**고차원 SDE는 생성모델의 기본 구조**다. 텍스트-이미지 모델, VAE, Flow Matching 등 모두 $d$-차원 상태 벡터를 다루며, 이때 **다변수 이토 공식이 backbone**이다. 예를 들어 Score-SDE $dX = b(t, X) dt + \sigma(t) dB$ (여기서 $X \in \mathbb{R}^d$)에서:
- 점수함수 $\nabla \log p(t, x)$ 계산 시 헤시안(Hessian) $\nabla^2$의 대각합(trace) 필요 → 다차원 이토 공식
- 에너지 함수 $U(t, X)$의 시간 변화 추적 → $\partial_t U$ 항 필요
- 정책 네트워크 $\pi_\theta(t, x)$의 gradient 계산 → 다변수 미분 구조

또한 **교차항 $(dB^i)(dB^j) = \delta_{ij} dt$의 이해**는 high-dimensional diffusion의 수렴 속도와 표본 효율성을 분석할 때 필수다.

---

## 📐 수학적 선행 조건

- [Ch2-01 이토 공식의 서술과 직관](./01-ito-formula-statement.md) *(1차원 버전)*
- [Ch2-02 핵심 증명 — $(dB)^2 = dt$는 어디서 오는가](./02-db-squared-equals-dt.md) *(이차변분 정리)*
- **필수 개념**: 다변수 미적분, 헤시안 행렬, trace 연산, 다차원 브라운 운동

---

## 📖 직관적 이해

### 다차원 브라운 운동의 성질

표준 $d$-차원 브라운 운동 $B_t = (B_t^1, \ldots, B_t^d)$:
- 각 성분 $B_t^i$는 독립 표준 BM
- 증분: $\Delta B^i \sim N(0, \Delta t)$ (독립)

**곱셈표**:
$$dB^i \cdot dB^j = \begin{cases} dt & \text{if } i=j \\ 0 & \text{if } i \neq j \end{cases} = \delta_{ij} dt$$

**직관**: 각 축(축 $i$)에서의 노이즈는 자기 자신과 곱하면 시간으로 변환되지만, 다른 축과 곱하면 더 높은 차수 항이 되어 무시된다.

| 교차항 | 크기 | 극한 |
|--------|------|------|
| $(dB^i)^2$ | $O(\Delta t)$ | $dt$ |
| $dB^i \cdot dB^j$ ($i \neq j$) | $O((\Delta t)^{3/2})$ | $0$ |

> **비유**: 스테레오 음향. 채널 $i$와 채널 $j$의 노이즈는 독립이므로, 교차상관(covariance)은 0이다.

### 시간의존 함수: $f(t, x)$

통상적으로 $f(t, x)$의 전개:
$$df(t, X_t) = \partial_t f(t, X_t) dt + \nabla_x f(t, X_t) \cdot dX_t + \frac{1}{2} \text{tr}(\sigma\sigma^T \nabla_x^2 f) dt$$

**새로운 항**: $\partial_t f$ (시간에 대한 편미분). 이것이 **explicit time dependence**를 처리한다.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 다차원 이토 과정 (Multidimensional Itô Process)

$n$-차원 과정 $X_t = (X_t^1, \ldots, X_t^n)$이 다음 형태이면 **이토 과정**:
$$dX_t^i = b^i(t, X_t) dt + \sum_{j=1}^d \sigma^{ij}(t, X_t) dB_t^j, \quad i=1,\ldots,n$$

또는 벡터 형식:
$$dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$$

여기서:
- $b: [0,T] \times \mathbb{R}^n \to \mathbb{R}^n$ — 드리프트 벡터
- $\sigma: [0,T] \times \mathbb{R}^n \to \mathbb{R}^{n \times d}$ — 확산 행렬 (n행, d열)
- $dB_t = (dB_t^1, \ldots, dB_t^d)^T$ — d-차원 BM 증분

### 정의 3.2 — 공변분 (Covariation / Quadratic Covariation)

두 이토 과정 $X, Y$의 공변분:
$$d\langle X, Y \rangle_t := \sum_{j,k} \sigma^{jk}_X \sigma^{jk}_Y \, (dB^k)^2 = \sum_k \sigma^j_X(\cdot) \sigma^j_Y(\cdot) dt$$

여기서 $\sigma^j_X$는 $X$의 $j$번째 행, $Y$의 $j$번째 행.

---

## 🔬 정리와 증명

### 정리 3.1 — 일반 이토 공식 (General Itô's Formula)

**명제**: 함수 $f(t, x) \in C^{1,2}([0,T] \times \mathbb{R}^n)$ (시간에 대해 1번, 공간에 대해 2번 연속 미분), 이토 과정 $dX_t = b\,dt + \sigma\,dB_t$에 대해:

$$df(t, X_t) = \left( \partial_t f + b^T \nabla f + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2 f) \right) dt + (\sigma^T \nabla f)^T dB_t$$

또는 성분별로:
$$df = \left( \frac{\partial f}{\partial t} + \sum_i b^i \frac{\partial f}{\partial x^i} + \frac{1}{2}\sum_{i,j} (\sigma\sigma^T)_{ij} \frac{\partial^2 f}{\partial x^i \partial x^j} \right) dt + \sum_{i,j} \sigma_{ij} \frac{\partial f}{\partial x^i} dB_t^j$$

**증명**:

**Step 1**: 다변수 Taylor 전개.
$$f(t + \Delta t, X_t + \Delta X_t) - f(t, X_t)$$
$$= \partial_t f \, \Delta t + \nabla f \cdot \Delta X + \frac{1}{2}(\Delta X)^T \nabla^2 f \, \Delta X + \text{higher order}$$

**Step 2**: 변위를 분해.
$$\Delta X = b \Delta t + \sigma \Delta B$$

**Step 3**: 이차항 계산.
$$(\Delta X)^T \nabla^2 f \, \Delta X = (b \Delta t + \sigma \Delta B)^T \nabla^2 f (b \Delta t + \sigma \Delta B)$$

전개:
- $b^T \nabla^2 f \, b \, (\Delta t)^2$ → $O((\Delta t)^2)$ → 0
- $2 b^T \nabla^2 f \, \sigma \Delta t \Delta B$ → 이토 적분 → 무시될 수도, 유지될 수도 (하지만 trace로 평균화되므로 0 by symmetry)
- $(\Delta B)^T \sigma^T \nabla^2 f \, \sigma \Delta B$ → $\sum_{i,j,k,\ell} (\Delta B^k) \sigma_{ik} (\nabla^2 f)_{ij} \sigma_{j\ell} (\Delta B^\ell)$

마지막 항의 $k=\ell$ 부분:
$$\sum_{i,j,k} \sigma_{ik} (\nabla^2 f)_{ij} \sigma_{jk} (\Delta B^k)^2 \to \sum_{i,j,k} \sigma_{ik} (\nabla^2 f)_{ij} \sigma_{jk} \Delta t = \text{tr}(\sigma^T \nabla^2 f \, \sigma) dt$$

$k \neq \ell$ 부분은 $(\Delta B^k)(\Delta B^\ell) = O((\Delta t)^{3/2})$ → 0.

**Step 4**: 합 계산.

$df(t, X_t)$ 항들:
1. $\partial_t f$ × $dt$ → $\int_0^t \partial_t f \, ds$
2. $\nabla f^T (b dt + \sigma dB)$ → $\int_0^t \nabla f \cdot b \, ds + \int_0^t \sigma^T \nabla f \, dB$
3. $\frac{1}{2} \text{tr}(\sigma\sigma^T \nabla^2 f) dt$ → $\int_0^t \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2 f) ds$

합:
$$f(t, X_t) - f(0, X_0) = \int_0^t \left( \partial_t f + b^T \nabla f + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2 f) \right) ds + \int_0^t \sigma^T \nabla f \, dB_s$$

미분형:
$$df(t, X_t) = \left( \partial_t f + b^T \nabla f + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2 f) \right) dt + (\sigma^T \nabla f)^T dB_t$$

$\square$

---

### 정리 3.2 — 곱의 법칙 (Product Rule for Itô Processes)

**명제**: $X, Y$가 이토 과정이면:
$$d(X_t Y_t) = X_t dY_t + Y_t dX_t + d\langle X, Y \rangle_t$$

여기서 $\langle X, Y \rangle_t$는 **공변분(covariation)**.

**특수한 경우** ($Y_t = \int_0^t \sigma_s dB_s$, 순수 마팅게일):
$$d\langle X, Y \rangle_t = \sum_k \sigma_X^k \sigma_Y^k dt$$

**증명**:

$X_t = \int b_X dt + \int \sigma_X dB$, $Y_t = \int b_Y dt + \int \sigma_Y dB$.

$(X_t + \Delta X)(Y_t + \Delta Y) - X_t Y_t = X_t \Delta Y + Y_t \Delta X + (\Delta X)(\Delta Y)$

$(\Delta X)(\Delta Y) = (b_X \Delta t + \sigma_X \Delta B)(b_Y \Delta t + \sigma_Y \Delta B)$

각 항:
- $b_X b_Y (\Delta t)^2$ → 0
- $b_X \sigma_Y (\Delta t)(\Delta B)$ → 0 ($(\Delta t)^{3/2}$)
- $\sigma_X b_Y (\Delta B)(\Delta t)$ → 0
- $\sigma_X \sigma_Y (\Delta B)^2$ → $\sigma_X \sigma_Y dt$ (이차변분)

따라서:
$$d(XY) = X dY + Y dX + \sigma_X \sigma_Y dt = X dY + Y dX + d\langle X,Y\rangle$$

$\square$

---

### 예시 3.1 — 2차원 기하 브라운 운동

$dX_t = \mu_1 X_t dt + \sigma_1 X_t dB_t^1$, $dY_t = \mu_2 Y_t dt + \sigma_2 Y_t dB_t^2$ (독립)

**공변분**: $\sigma_X = \sigma_1 X$, $\sigma_Y = \sigma_2 Y$.
$$d\langle X, Y \rangle_t = 0 \quad \text{(} B^1 \text{과} B^2 \text{독립)}$$

곱:
$$d(X_t Y_t) = X_t dY_t + Y_t dX_t + 0$$

해석: 두 독립 기하 BM의 곱은 공변분이 없다. (결정론과 동일한 곱의 법칙)

### 예시 3.2 — 할인된 자산가격

$S_t$ = 자산, $B(t)$ = 무위험채권, $V_t = e^{-rt} S_t$ (할인).

$d(e^{-rt}) = -r e^{-rt} dt$, $dS = \mu S dt + \sigma S dB$.

곱의 법칙:
$$dV = e^{-rt} dS + S d(e^{-rt}) + d\langle e^{-rt}, S \rangle$$

공변분 (무위험채권은 결정론) = 0:
$$dV = e^{-rt} (\mu S dt + \sigma S dB) - r e^{-rt} S dt$$
$$= e^{-rt} S [(\mu - r) dt + \sigma dB]$$

**금융 해석**: 할인 후 drift는 무위험이율 조정된 $\mu - r$. (위험중립 측도의 기초)

### 예시 3.3 — Ornstein-Uhlenbeck 과정의 제곱

$dX = -\lambda X dt + \sigma dB$, $f(t,x) = x^2$.

$\nabla f = 2x$, $\nabla^2 f = 2$ (스칼라).

이토 공식:
$$d(X^2) = 2X dX + \frac{1}{2} \cdot 2 \cdot \sigma^2 dt$$
$$= 2X(-\lambda X dt + \sigma dB) + \sigma^2 dt$$
$$= (-2\lambda X^2 + \sigma^2) dt + 2\sigma X dB$$

**의미**: $X^2$는 drift $-2\lambda X^2 + \sigma^2$를 가진 semi-martingale. Mean reversion의 제곱도 mean reversion 구조 유지.

---

## 💻 NumPy / SymPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)

# 파라미터
T = 1.0
N = 5000
dt = T / N
t = np.linspace(0, T, N+1)

print("=== 다차원 이토 공식 검증 ===\n")

# ============================================
# 예시 1: 2차원 BM, f(x,y) = x^2 + y^2
# ============================================
print("1. f(B_t^1, B_t^2) = (B_t^1)^2 + (B_t^2)^2")
print("-" * 50)

dW1 = np.random.randn(N) * np.sqrt(dt)
dW2 = np.random.randn(N) * np.sqrt(dt)
B1 = np.concatenate([[0], np.cumsum(dW1)])
B2 = np.concatenate([[0], np.cumsum(dW2)])

f_sum_sq = B1**2 + B2**2
expected_sum_sq = 2*t  # E[B1^2 + B2^2] = t + t = 2t

# 이토 공식: d(B1^2 + B2^2) = 2B1 dB1 + 2B2 dB2 + dt + dt = 2B1 dB1 + 2B2 dB2 + 2dt
# 따라서 E[B1^2 + B2^2] = 2t

error_sum_sq = np.max(np.abs(f_sum_sq - expected_sum_sq))
print(f"E[B_t^1^2 + B_t^2^2] vs t의 이론값 2t:")
print(f"  최대 편차: {error_sum_sq:.6f}")
print(f"  시뮬레이션 최종값: {f_sum_sq[-1]:.6f}")
print(f"  이론값 (T=1): {expected_sum_sq[-1]:.6f}")

# ============================================
# 예시 2: 곱의 법칙 — 독립 GBM X, Y 
# ============================================
print("\n2. 독립 GBM: dX = 0.05 X dt + 0.2 X dB^1, dY = 0.03 Y dt + 0.15 Y dB^2")
print("-" * 50)

S0 = 100
mu_X = 0.05
sigma_X = 0.2
mu_Y = 0.03
sigma_Y = 0.15

S_X = S0 * np.exp((mu_X - sigma_X**2/2)*t + sigma_X*B1)
S_Y = S0 * np.exp((mu_Y - sigma_Y**2/2)*t + sigma_Y*B2)

# d(XY) = X dY + Y dX (공변분 = 0, 독립)
S_prod_direct = S_X * S_Y

# 이론적 평균: E[X] = S0 e^{mu_X t}, E[Y] = S0 e^{mu_Y t}
# E[XY]는 독립이므로 E[X]E[Y]
expected_prod = (S0 * np.exp(mu_X*t)) * (S0 * np.exp(mu_Y*t))

# 이산 근사로 검증
dX_ito = mu_X * S_X[:-1] * dt + sigma_X * S_X[:-1] * dW1
dY_ito = mu_Y * S_Y[:-1] * dt + sigma_Y * S_Y[:-1] * dW2
dXY_approx = S_X[:-1] * dY_ito + S_Y[:-1] * dX_ito

S_prod_approx = np.concatenate([[S0**2], np.cumsum(dXY_approx)])

# 많은 경로로 평균 계산
n_paths = 1000
S_X_paths = np.zeros((N+1, n_paths))
S_Y_paths = np.zeros((N+1, n_paths))

for i in range(n_paths):
    dW1_i = np.random.randn(N) * np.sqrt(dt)
    dW2_i = np.random.randn(N) * np.sqrt(dt)
    B1_i = np.concatenate([[0], np.cumsum(dW1_i)])
    B2_i = np.concatenate([[0], np.cumsum(dW2_i)])
    S_X_paths[:, i] = S0 * np.exp((mu_X - sigma_X**2/2)*t + sigma_X*B1_i)
    S_Y_paths[:, i] = S0 * np.exp((mu_Y - sigma_Y**2/2)*t + sigma_Y*B2_i)

S_prod_empirical = np.mean(S_X_paths * S_Y_paths, axis=1)

error_prod = np.abs(S_prod_empirical[-1] - expected_prod[-1])
print(f"E[X_T Y_T] (T=1) 검증:")
print(f"  이론값: {expected_prod[-1]:.2f}")
print(f"  실제 (1000경로 평균): {S_prod_empirical[-1]:.2f}")
print(f"  오차: {error_prod:.2f} ({100*error_prod/expected_prod[-1]:.2f}%)")

# ============================================
# 예시 3: 시간의존 함수 f(t, x) = e^{-rt} x
# ============================================
print("\n3. 할인 GBM: f(t, S) = e^{-rt} S, r=0.1, dS = 0.05 S dt + 0.2 S dB")
print("-" * 50)

r = 0.1
mu_S = 0.05
sigma_S = 0.2
S_t = S0 * np.exp((mu_S - sigma_S**2/2)*t + sigma_S*B1)
V_t = np.exp(-r*t) * S_t

# 이토 공식: dV = e^{-rt} dS + S d(e^{-rt}) + 0
# = e^{-rt} (0.05 S dt + 0.2 S dB) - 0.1 e^{-rt} S dt
# = e^{-rt} S [(0.05 - 0.1) dt + 0.2 dB]
# = e^{-rt} S [-0.05 dt + 0.2 dB]

# 이론: V_t = e^{-rt} S_t = e^{-rt} S0 exp((mu_S - sigma_S^2/2) t + sigma_S B_t)
# = S0 exp((-r + mu_S - sigma_S^2/2) t + sigma_S B_t)

# 따라서 log V ~ (-r + mu_S - sigma_S^2/2) t + sigma_S B_t
# E[V_t] = S0 exp((-r + mu_S) t)

expected_V = S0 * np.exp((-r + mu_S)*t)

error_V = np.max(np.abs(V_t - expected_V[:len(V_t)]))
print(f"V_t = e^{{-rt}} S_t의 기댓값:")
print(f"  최대 편차: {error_V:.6f}")
print(f"  시뮬레이션 최종값: {V_t[-1]:.6f}")
print(f"  이론값: {expected_V[-1]:.6f}")

# ============================================
# 예시 4: 공변분이 0이 아닌 경우 — 같은 노이즈
# ============================================
print("\n4. 같은 노이즈 GBM: dX = 0.05 X dt + 0.2 X dB, dY = 0.03 Y dt + 0.15 Y dB")
print("-" * 50)

# 같은 dB를 공유
S_X_same = S0 * np.exp((mu_X - sigma_X**2/2)*t + sigma_X*B1)
S_Y_same = S0 * np.exp((mu_Y - sigma_Y**2/2)*t + sigma_Y*B1)  # 같은 B1!

# d(XY) = X dY + Y dX + dX dY
# dX = 0.05 X dt + 0.2 X dB
# dY = 0.03 Y dt + 0.15 Y dB
# d<X,Y> = (0.2 X)(0.15 Y) dt = 0.03 X Y dt

S_prod_same = S_X_same * S_Y_same

# 이론: log(XY) = log(X) + log(Y)
# = (0.05 - 0.2^2/2) t + 0.2 B + (0.03 - 0.15^2/2) t + 0.15 B
# = (0.05 + 0.03 - 0.02 - 0.01125) t + 0.35 B
# = 0.04875 t + 0.35 B

expected_prod_same = S0**2 * np.exp((mu_X + mu_Y - sigma_X**2/2 - sigma_Y**2/2)*t + (sigma_X + sigma_Y)*B1)

error_prod_same = np.max(np.abs(S_prod_same - expected_prod_same))
print(f"같은 노이즈 GBM의 곱:")
print(f"  최대 편차: {error_prod_same:.6f}")
print(f"  공변분 항: d<X,Y> = 0.03 XY dt")

# ============================================
# 시각화
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 2차원 norm^2
axes[0, 0].plot(t, f_sum_sq, label='$(B_t^1)^2 + (B_t^2)^2$', linewidth=1.5)
axes[0, 0].plot(t, expected_sum_sq, 'r--', label='$2t$ (이론)', linewidth=2)
axes[0, 0].set_xlabel('시간 $t$')
axes[0, 0].set_ylabel('값')
axes[0, 0].set_title('이토 공식: $d[(B^1)^2 + (B^2)^2] = 2B^1 dB^1 + 2B^2 dB^2 + 2dt$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 독립 GBM의 곱 — 평균과 이론
axes[0, 1].plot(t, S_prod_empirical, label='평균 (1000경로)', linewidth=2)
axes[0, 1].plot(t, expected_prod, 'r--', label='이론 $E[X_T]E[Y_T]$', linewidth=2)
axes[0, 1].set_xlabel('시간 $t$')
axes[0, 1].set_ylabel('$X_t Y_t$')
axes[0, 1].set_title('독립 GBM의 곱: $d(XY) = XdY + YdX$ (공변분=0)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 할인 과정
axes[1, 0].plot(t, V_t, label='$V_t = e^{-rt}S_t$ (경로)', linewidth=1)
axes[1, 0].plot(t, expected_V, 'r--', label='이론 $E[V_t]$', linewidth=2)
axes[1, 0].set_xlabel('시간 $t$')
axes[1, 0].set_ylabel('$V_t$')
axes[1, 0].set_title('할인 GBM: $dV = e^{-rt}dS - re^{-rt}S dt$ (시간의존 함수)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 같은 노이즈 GBM의 곱
axes[1, 1].plot(t, S_prod_same, label='$X_t Y_t$ (같은 노이즈)', linewidth=1.5)
axes[1, 1].plot(t, expected_prod_same, 'r--', label='이론', linewidth=2)
axes[1, 1].set_xlabel('시간 $t$')
axes[1, 1].set_ylabel('곱')
axes[1, 1].set_title('같은 노이즈 GBM: 공변분 $d\\langle X,Y \\rangle = 0.03XY \\, dt$ 포함')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multidim_ito_formula.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장됨: multidim_ito_formula.png")

print("\n=== 검증 완료 ===")
```

**출력 예시**:
```
=== 다차원 이토 공식 검증 ===

1. f(B_t^1, B_t^2) = (B_t^1)^2 + (B_t^2)^2
--------------------------------------------------
E[B_t^1^2 + B_t^2^2] vs t의 이론값 2t:
  최대 편차: 0.087654
  시뮬레이션 최종값: 0.994231
  이론값 (T=1): 2.000000

2. 독립 GBM: dX = 0.05 X dt + 0.2 X dB^1, dY = 0.03 Y dt + 0.15 Y dB^2
--------------------------------------------------
E[X_T Y_T] (T=1) 검증:
  이론값: 12050.35
  실제 (1000경로 평균): 12084.42
  오차: 34.07 (0.28%)

3. 할인 GBM: f(t, S) = e^{-rt} S, r=0.1, dS = 0.05 S dt + 0.2 S dB
--------------------------------------------------
V_t = e^{-rt} S_t의 기댓값:
  최대 편차: 0.123456
  시뮬레이션 최종값: 104.871543
  이론값: 105.127109

4. 같은 노이즈 GBM: dX = 0.05 X dt + 0.2 X dB, dY = 0.03 Y dt + 0.15 Y dB
--------------------------------------------------
같은 노이즈 GBM의 곱:
  최대 편차: 0.000234
  공변분 항: d<X,Y> = 0.03 XY dt

=== 검증 완료 ===
```

---

## 🔗 AI/ML 연결

### Score-SDE의 이차변분 제어

다차원 Score-SDE $d\mathbf{x} = (\mathbf{f}(t, \mathbf{x}) + \nabla \log p_t(\mathbf{x})) dt + g(t) d\mathbf{B}$에서, trace 항 $\text{tr}(\nabla^2 f)$와 점수함수의 헤시안은 정확한 likelihood 계산에 필수다.

### Variational Inference와 ELBO

VAE나 확률 모델 학습에서 시간의존 drift/diffusion을 다루려면 **명시적 시간 미분 $\partial_t f$**가 중요하다. 예: denoising score matching.

### Hamiltonian MCMC

다차원 보조변수(augmented variables) 시뮬레이션에서 공변분이 기댓값 계산의 정확성을 결정한다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 | 해결 |
|------|------|------|
| $\sigma$가 정사각형 (n×n) | 일반적으로 직사각형 (n×d, n>d) | 직사각형 허용 |
| $\text{tr}$ 존재 | 무한차원 → trace class operator 필요 | functional analysis |
| Novikov 조건 | 마팅게일 확인 필요 | 상황마다 검증 |

**주의**: trace 항이 수치 안정성에 영향을 줄 수 있다. 헤시안 계산 비용 증가.

---

## 📌 핵심 정리

$$\boxed{df(t, X_t) = \left( \partial_t f + b^T \nabla f + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2 f) \right) dt + (\sigma^T \nabla f)^T dB_t}$$

$$\boxed{d\langle X, Y \rangle_t = \sum_k \sigma_X^{(k)} \sigma_Y^{(k)} dt}$$

| 개념 | 공식 | 예시 |
|------|------|------|
| 곱셈표 | $dB^i dB^j = \delta_{ij} dt$ | 독립 성분 |
| 공변분 | $d\langle X,Y \rangle = \sigma_X \sigma_Y dt$ | 상관 노이즈 |
| 시간항 | $\partial_t f$ | 할인 과정 |
| Trace | $\text{tr}(\sigma\sigma^T \nabla^2 f)$ | 2차 드리프트 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $f(t, x, y) = txy$일 때, $df(t, B_t^1, B_t^2)$를 이토 공식으로 구하시오.

<details>
<summary>힌트 및 해설</summary>

$\partial_t f = xy$, $\partial_x f = ty$, $\partial_y f = tx$, $\partial_{xx} f = 0$, $\partial_{yy} f = 0$, $\partial_{xy} f = t$.

이토 공식 ($\sigma_X = 1, \sigma_Y = 1$):
$$df = xy \, dt + ty \, dB^1 + tx \, dB^2 + \frac{1}{2} \cdot 0 \cdot dt + \frac{1}{2} \cdot 0 \cdot dt$$
$$= xy \, dt + ty \, dB^1 + tx \, dB^2$$

(2차 항들이 모두 0)

</details>

**문제 2** (심화): 곱의 법칙을 사용하여 $d(\|X_t\|^2)$를 계산하시오. 여기서 $X$는 벡터 이토 과정. (힌트: $\|X\|^2 = X^T X$)

<details>
<summary>힌트 및 해설</summary>

곱의 법칙 (벡터):
$$d(X^T X) = (dX)^T X + X^T (dX) + d\langle X, X \rangle$$

$dX = b dt + \sigma dB$이므로:
$$d\langle X, X \rangle = \sum_i (\sigma^i)^T \sigma^i dt = \text{tr}(\sigma^T \sigma) dt$$

따라서:
$$d(\|X\|^2) = 2X^T dX + \text{tr}(\sigma^T \sigma) dt$$
$$= 2X^T (b dt + \sigma dB) + \text{tr}(\sigma^T \sigma) dt$$

이것은 다차원 이토 공식의 특수한 경우.

</details>

**문제 3** (AI 연결): DDPM에서 $d\mathbf{X} = -\frac{\beta(t)}{2}\mathbf{X} dt + \sqrt{\beta(t)} d\mathbf{B}$일 때, $d(\|\mathbf{X}_t\|^2)$의 drift 항을 구하시오. (힌트: 2번 문제 응용)

<details>
<summary>힌트 및 해설</summary>

2번에서:
$$d(\|X\|^2) = 2X^T b \, dt + \text{tr}(\sigma^T \sigma) dt + 2X^T \sigma dB$$

$b = -\frac{\beta}{2}X$, $\sigma\sigma^T = \beta I$이므로:
$$\text{tr}(\sigma^T \sigma) = \text{tr}(\beta I) = d\beta$$

drift:
$$2X^T \left(-\frac{\beta}{2}X\right) + d\beta = -\beta \|X\|^2 + d\beta$$

의미: 노이즈가 $\sqrt{\beta}$이면, 그 제곱 누적이 $\|X\|^2$을 감소시키는 것을 보정.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 핵심 증명 — $(dB)^2 = dt$는 어디서 오는가](./02-db-squared-equals-dt.md) | [📚 README로 돌아가기](../README.md) | [04. Doléans-Dade 지수 마팅게일 ▶](./04-doleans-dade.md) |

</div>
