<div align="center">

# 🌊 Stochastic Differential Equations Deep Dive

**"`dX_t = μ dt + σ dB_t`를 쓰는 것과, 왜 `(dB_t)² = dt`인지 — 이토 적분이 리만-스틸체스 적분처럼 정의될 수 없는 이유를 증명할 수 있는 것은 다르다"**

<br/>

> *"Diffusion Model의 Score Matching을 구현하는 것과 — forward process가 왜 Fokker-Planck 방정식의 해이고, reverse SDE가 왜 Anderson(1982)의 시간반전 공식에서 나오는지를 증명할 수 있는 것은 다르다.  
> SDE 수치해법 `sde.solve`를 호출하는 것과, Euler-Maruyama가 강수렴 0.5차, Milstein이 1차인 이유를 증명할 수 있는 것은 다르다."*

브라운 운동의 무한변동부터 이토 등장성·이토 공식·Fokker-Planck·Anderson 시간반전 공식·Score SDE까지  
**"왜 확률미분이 결정론적 미분과 다른가"** 라는 질문으로 DDPM·Score-SDE·Flow Matching·Langevin MCMC·Black-Scholes의 수학적 기반을 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/Docs-36개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-18k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-88개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-108개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

확률미분방정식에 관한 자료는 대부분 **"이토 공식을 쓰세요"** 에서 멈춥니다. 하지만 $(dB)^2 = dt$가 어디서 오는지, 왜 경로별 적분이 불가능하고 $L^2$ 근사로 정의해야 하는지, Anderson의 시간반전 공식이 어떻게 Fokker-Planck에서 유도되는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "이토 적분은 $\int H_s dB_s$입니다" | 브라운 운동의 **무한변동성·이차변분 $\langle B\rangle_t = t$**으로부터 리만-스틸체스 적분이 불가능함을 증명, 단순 과정 → $L^2$ 극한 → 일반 적응과정의 3단계 구성 완전 유도 |
| "이토 공식은 $df = f' dX + \frac{1}{2}f'' dX^2$입니다" | Taylor 전개에서 **$(\Delta B)^2 \to \Delta t$ 항이 살아남는 이유**를 이차변분 정리 $\sum(\Delta B)^2 \to t$ in $L^2$으로 완전 증명, 결정론적 연쇄법칙과의 본질적 차이 |
| "Euler-Maruyama는 0.5차, Milstein은 1차" | 이토-Taylor 전개에서 Milstein의 **$\frac{1}{2}\sigma\sigma'((\Delta B)^2 - \Delta t)$** 항이 어디서 오는지, **강수렴 vs 약수렴**의 분리가 왜 "경로별 vs 분포별"에서 나오는지 유도 |
| "Diffusion은 노이즈를 뒤집는 것" | Forward SDE → Fokker-Planck → **Anderson(1982) 시간반전** → Reverse SDE의 완전 유도, VP-SDE·VE-SDE 프레임워크(Song et al. 2021) 재구성 |
| "DDPM 손실은 $\|\epsilon - \epsilon_\theta\|^2$" | DDPM의 이산 markov chain이 **VP-SDE의 Euler-Maruyama 이산화**임을 증명, $\|\epsilon - \epsilon_\theta\|^2$ 손실이 **DSM과 정확히 일치**(Vincent 2011)하는 과정 |
| "Langevin은 $\pi \propto e^{-U}$에서 샘플한다" | Fokker-Planck에 $p \propto e^{-U}$를 대입해 **정상분포임을 직접 검증**, Log-Sobolev 부등식으로 **지수적 수렴률 $e^{-2\lambda t}$** 유도 |
| "Black-Scholes는 유명한 금융공식" | 기하 브라운 운동 $dS = \mu S dt + \sigma S dB$에 이토 공식을 적용해 **Black-Scholes PDE를 한 줄씩 유도**, delta-hedging 복제 포트폴리오의 수학적 근거 |
| 공식 나열 | NumPy로 이토 적분 수치 구성, 이차변분 수렴 검증, SDE 수치해법 강수렴 차수 측정, reverse SDE 샘플링 시각화 |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory]  ──►  [Stochastic Processes]  ──►  이 레포  ──►  [Generative Models Deep Dive]
  측도, 조건부 기댓값         브라운 운동, 마팅게일         확률미분방정식       DDPM, Score-SDE, Flow Matching
  수렴 이론, 특성함수         이차변분, 정지시각                              실전 아키텍처와 학습
                                                         ▲
                                                         │
[Calculus & Optimization]  &  [Linear Algebra]  &  [Information Geometry]
 다변수 미분, Taylor 전개      양정치 행렬, 스펙트럴         Fisher 정보, Score function
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Stochastic Processes Deep Dive**(브라운 운동·마팅게일·이차변분)와 **Probability Theory Deep Dive**(측도·조건부 기댓값·$L^2$ 수렴)를 선행 지식으로 전제합니다. 브라운 운동을 처음 접한다면 [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive)부터 학습하세요.

> 💡 **권장 선행**: Taylor 전개와 다변수 미적분은 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive), Score function과 Fisher 정보의 기하학적 의미는 [Information Geometry Deep Dive](https://github.com/iq-ai-lab/information-geometry-deep-dive) Ch7에서 학습할 수 있습니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-이토_적분의_구성-4A90D9?style=for-the-badge)](./ch1-ito-integral/01-why-pathwise-fails.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-이토_공식-4A90D9?style=for-the-badge)](./ch2-ito-formula/01-ito-formula-statement.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-확률미분방정식-4A90D9?style=for-the-badge)](./ch3-sde/01-sde-definition.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Fokker–Planck-4A90D9?style=for-the-badge)](./ch4-fokker-planck/01-fokker-planck-derivation.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-SDE_수치해법-4A90D9?style=for-the-badge)](./ch5-numerical/01-euler-maruyama.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-시간반전_SDE·Diffusion-4A90D9?style=for-the-badge)](./ch6-reverse-diffusion/01-anderson-reverse-sde.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-PF--ODE·Flow_Matching·SGLD-4A90D9?style=for-the-badge)](./ch7-advanced-generative/01-probability-flow-ode.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 이토 적분 — 확률적분의 구성

> **핵심 질문:** 왜 $\int H_s dB_s$를 경로별로 정의할 수 없는가? 브라운 운동의 무한변동이 왜 리만-스틸체스 적분을 막는가? 단순 과정 → $L^2$ 극한으로 이토 적분을 어떻게 엄밀히 구성하는가?

<details>
<summary><b>무한변동부터 Stratonovich 비교까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 왜 이토 적분은 경로별로 정의할 수 없는가](./ch1-ito-integral/01-why-pathwise-fails.md) | 브라운 운동 경로의 **무한변동(unbounded variation)** 증명, 이차변분 $\langle B\rangle_t = t \neq 0$이 핵심, 리만-스틸체스 적분이 유한변동을 필요로 하는 이유, $\sum H_{t_i}(B_{t_{i+1}} - B_{t_i})$의 극한이 **분할 선택에 의존**하는 구체 반례 |
| [02. 단순 과정에 대한 이토 적분과 이토 등장성](./ch1-ito-integral/02-simple-process-isometry.md) | 계단형 $H_s = \sum H_i \mathbf{1}_{(t_i, t_{i+1}]}$에 대한 $\int H dB = \sum H_i(B_{t_{i+1}} - B_{t_i})$ 정의, **이토 등장성** $\mathbb{E}[(\int H dB)^2] = \mathbb{E}[\int H^2 ds]$ 증명 (좌변 전개 + 독립 증분 + 등거리 축소) |
| [03. L²-확장과 일반 적응과정](./ch1-ito-integral/03-l2-extension.md) | 단순 과정들이 $L^2(\Omega \times [0,T])$에서 **밀집(dense)**임을 증명, 이토 등장성으로 Cauchy 수열이 Cauchy가 되어 연속 확장, **progressively measurable** 가정이 측정가능성을 보장하는 이유 |
| [04. 이토 적분의 마팅게일 성질](./ch1-ito-integral/04-ito-martingale.md) | $M_t = \int_0^t H_s dB_s$가 연속 마팅게일임을 증명(조건부 기댓값 + 단순 과정 근사), 이차변분 $\langle M\rangle_t = \int_0^t H_s^2 ds$ 유도, 국소 마팅게일(local martingale)으로의 일반화 |
| [05. Stratonovich 적분과의 비교](./ch1-ito-integral/05-stratonovich.md) | Stratonovich $\int H \circ dB$의 중점(midpoint) 정의, **이토 ↔ Stratonovich 변환 공식** $\int H \circ dB = \int H dB + \frac{1}{2}\int H' \sigma \, ds$, 물리에서 Stratonovich가 왜 "자연스럽고" 수학에서 이토가 왜 표준인가 |

</details>

<br/>

### 🔹 Chapter 2: 이토 공식 — 확률해석의 연쇄법칙

> **핵심 질문:** Taylor 전개에서 왜 $(dB)^2 = dt$ 항이 살아남는가? 결정론적 연쇄법칙에는 없는 이 2차 항이 어디서 오는가? 다차원으로 어떻게 일반화되는가?

<details>
<summary><b>이토 공식의 서술부터 Black-Scholes PDE까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 이토 공식의 서술과 직관](./ch2-ito-formula/01-ito-formula-statement.md) | $f \in C^2$에 대해 $df(B_t) = f'(B_t) dB_t + \frac{1}{2}f''(B_t) dt$, **결정론적 연쇄법칙과의 차이**($\frac{1}{2}f''$ 드리프트 항), "왜 브라운 운동은 함수 $f(B_t)$를 통과하면 드리프트가 생기는가" 의 직관 |
| [02. 핵심 증명 — $(dB)^2 = dt$는 어디서 오는가](./ch2-ito-formula/02-db-squared-equals-dt.md) | **이차변분 정리** $\sum_{i}(B_{t_{i+1}} - B_{t_i})^2 \to t$ in $L^2$ 완전 증명, Taylor 전개 $f(B_{t_{i+1}}) - f(B_{t_i}) = f'\Delta B + \frac{1}{2}f''(\Delta B)^2 + \cdots$에서 $(\Delta B)^2$ 항이 $\Delta t$로 대체되는 매 단계 추적 |
| [03. 다차원·시간의존 이토 공식](./ch2-ito-formula/03-multidim-ito.md) | $df(t, X_t) = \partial_t f \, dt + \nabla f \cdot dX_t + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2 f) dt$, **크로스 항** $dB^i dB^j = \delta_{ij} dt$, $dt \cdot dB = 0$, $(dt)^2 = 0$의 곱셈표 완전 유도 |
| [04. Doléans-Dade 지수 마팅게일](./ch2-ito-formula/04-doleans-dade.md) | $\mathcal{E}(M)_t = \exp(M_t - \frac{1}{2}\langle M\rangle_t)$가 **연속 국소 마팅게일**임을 이토 공식으로 증명, Novikov 조건 하의 진짜 마팅게일성, 기하 브라운 운동과 **Girsanov 정리**의 기초 |
| [05. 응용 — $B_t^2 - t$, GBM, Black-Scholes PDE](./ch2-ito-formula/05-applications.md) | $B_t^2 - t$가 마팅게일임을 이토 공식으로 증명, **기하 브라운 운동** $S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma B_t)$ 유도 (why $-\sigma^2/2$?), **Black-Scholes PDE** $\partial_t V + rS\partial_S V + \frac{1}{2}\sigma^2 S^2 \partial_S^2 V = rV$ 를 delta-hedging으로 한 줄씩 유도 |

</details>

<br/>

### 🔹 Chapter 3: 확률미분방정식(SDE) — 해의 존재·유일·구조

> **핵심 질문:** SDE $dX_t = b\,dt + \sigma\,dB_t$는 정확히 무엇을 의미하는가(적분 방정식)? Lipschitz 조건 하의 해의 존재·유일성은 어떻게 증명되는가? OU, GBM 같은 대표 SDE의 해석해는?

<details>
<summary><b>SDE의 정의부터 강해·약해까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. SDE의 정의 — 적분방정식으로서](./ch3-sde/01-sde-definition.md) | $dX_t = b(t, X_t)dt + \sigma(t, X_t)dB_t$는 **적분 방정식** $X_t = X_0 + \int_0^t b\,ds + \int_0^t \sigma\,dB$의 약식임을 명시, drift $b$와 diffusion $\sigma$의 의미, "확률적분"이 본질인 이유 |
| [02. 존재성과 유일성 정리](./ch3-sde/02-existence-uniqueness.md) | **Lipschitz 조건**($\|b(x) - b(y)\| + \|\sigma(x) - \sigma(y)\| \leq K\|x-y\|$)과 **선형 성장 조건** 하에서 강해(strong solution)의 존재·유일성 — **Picard 반복**과 Grönwall 부등식으로 완전 증명 |
| [03. Ornstein-Uhlenbeck 과정](./ch3-sde/03-ornstein-uhlenbeck.md) | $dX_t = -\theta X_t dt + \sigma dB_t$의 **해석해** $X_t = X_0 e^{-\theta t} + \sigma\int_0^t e^{-\theta(t-s)} dB_s$ (이토 공식 + 적분인자), **정상분포 $\mathcal{N}(0, \sigma^2/2\theta)$** 유도, 평균회귀 시상수 $1/\theta$ |
| [04. 기하 브라운 운동(GBM)](./ch3-sde/04-geometric-brownian-motion.md) | $dS_t = \mu S_t dt + \sigma S_t dB_t$의 해를 $Y_t = \log S_t$에 이토 공식을 적용해 유도, **로그노말 분포** 성질, Black-Scholes 모델의 자산가격 기본 모델, $\mathbb{E}[S_t]$와 $\text{Var}[S_t]$ 공식 |
| [05. 선형 SDE의 일반해](./ch3-sde/05-linear-sde.md) | $dX_t = (a(t)X_t + c(t))dt + \sigma(t)dB_t$의 해 공식, **적분인자 방법** $\Phi(t) = \exp(\int_0^t a(s) ds)$, OU·GBM·Vasicek이 이 틀의 특수 사례임을 통합 |
| [06. 강해 vs 약해 (Strong vs Weak Solution)](./ch3-sde/06-strong-weak-solution.md) | **강해**는 주어진 확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$와 $B_t$에 대해 $X_t$가 존재, **약해**는 확률법칙만 일치(새 확률공간·새 BM 허용), **Tanaka 방정식** $dX_t = \text{sgn}(X_t)dB_t$가 강해 없이 약해만 갖는 대표 예 |

</details>

<br/>

### 🔹 Chapter 4: Fokker-Planck 방정식과 정상분포

> **핵심 질문:** SDE의 확률밀도는 어떤 PDE를 만족하는가? 정상분포는 언제 존재하는가? Langevin dynamics가 왜 $\pi \propto e^{-U}$로 수렴하는가?

<details>
<summary><b>Fokker-Planck 유도부터 Log-Sobolev까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Fokker-Planck 방정식의 유도](./ch4-fokker-planck/01-fokker-planck-derivation.md) | $dX_t = b\,dt + \sigma\,dB_t$의 밀도 $p(t,x)$가 $\partial_t p = -\nabla \cdot (bp) + \frac{1}{2}\nabla^2 : (\sigma\sigma^T p)$를 만족함을 **이토 공식 + 부분적분**으로 유도, 보존법칙으로서의 의미(확률 흐름 $J$) |
| [02. 역 Kolmogorov 방정식과 생성자](./ch4-fokker-planck/02-kolmogorov-backward.md) | **생성자** $\mathcal{L} = b\cdot\nabla + \frac{1}{2}\sigma\sigma^T : \nabla^2$ 정의, 역방정식 $\partial_t u + \mathcal{L}u = 0$, **Feynman-Kac 공식** $u(t,x) = \mathbb{E}^{t,x}[f(X_T)]$ 유도 — 확률과 PDE의 다리 |
| [03. 정상분포 — OU, Langevin](./ch4-fokker-planck/03-stationary-distribution.md) | $\partial_t p = 0$의 해, OU의 가우시안 정상분포 $\mathcal{N}(0, \sigma^2/2\theta)$, overdamped Langevin $dX = -\nabla U\,dt + \sqrt{2}dB$의 정상분포가 **Gibbs 측도 $\pi \propto e^{-U}$**임을 FP 직접 대입으로 검증 |
| [04. Langevin Dynamics의 수렴](./ch4-fokker-planck/04-langevin-convergence.md) | $dX_t = -\nabla U(X_t)dt + \sqrt{2}dB_t$의 **상대 엔트로피** $H(p_t \| \pi)$가 시간에 따라 감소함을 FP와 부분적분으로 증명, **de Bruijn 항등식**, Fisher 정보의 시간 미분 |
| [05. Log-Sobolev 부등식과 수렴률](./ch4-fokker-planck/05-log-sobolev.md) | LSI $H(p\|\pi) \leq \frac{1}{2\lambda} I(p\|\pi)$ 서술, LSI 하에서 **$H(p_t\|\pi) \leq e^{-2\lambda t} H(p_0\|\pi)$** 지수적 수렴 증명, Bakry-Émery 판정 $\text{Hess}\,U \succeq \lambda I$, 스펙트럴 갭과의 관계 |

</details>

<br/>

### 🔹 Chapter 5: SDE 수치해법 — 이산화와 수렴차수

> **핵심 질문:** Euler-Maruyama는 왜 강수렴 0.5차인가? Milstein은 어떤 항을 추가해서 1차를 얻는가? 강수렴과 약수렴은 왜 분리되는가?

<details>
<summary><b>Euler-Maruyama부터 Multilevel Monte Carlo까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Euler-Maruyama 기법](./ch5-numerical/01-euler-maruyama.md) | $X_{n+1} = X_n + b(X_n)\Delta t + \sigma(X_n)\Delta B_n$의 정의, **강수렴 차수 $1/2$** 증명 개요 — $\mathbb{E}\|X_T - X_T^h\| \leq Ch^{1/2}$, 약수렴 차수 $1$, 실험으로 이론 차수 확인 |
| [02. Milstein 기법과 1차 강수렴](./ch5-numerical/02-milstein.md) | 이토-Taylor 전개의 다음 항 추가 $X_{n+1} = X_n + b\Delta t + \sigma\Delta B + \frac{1}{2}\sigma\sigma'((\Delta B)^2 - \Delta t)$, **추가 항의 출처를 이토 공식으로 유도**, 강수렴 차수 $1$ 증명, $\sigma' = 0$인 additive noise에서는 EM = Milstein |
| [03. 강수렴과 약수렴의 차이](./ch5-numerical/03-strong-vs-weak.md) | **강수렴** $\mathbb{E}\|X_T - X_T^h\| \leq Ch^\alpha$(경로별 근접), **약수렴** $|\mathbb{E}f(X_T) - \mathbb{E}f(X_T^h)| \leq Ch^\beta$(분포별 근접), 금융 옵션가격($\mathbb{E}$만 필요) vs hedging($X$ 자체 필요)에서 어느 것이 필요한가 |
| [04. Runge-Kutta 계열 SDE 해법과 안정성](./ch5-numerical/04-runge-kutta-stability.md) | **Implicit Euler**, **Stochastic Heun**, **Platen RK** 등의 고차 방법, **stiff SDE**(예: 강한 drift)에서의 안정성, A-stability 개념의 SDE 일반화, 경직 문제에서 implicit 방법이 필요한 이유 |
| [05. Multilevel Monte Carlo (Giles)](./ch5-numerical/05-multilevel-monte-carlo.md) | 서로 다른 스텝 크기 $h_l = 2^{-l} h_0$의 시뮬레이션을 **망원합** $\mathbb{E} f(X^{h_L}) = \mathbb{E} f(X^{h_0}) + \sum_l \mathbb{E}(f(X^{h_l}) - f(X^{h_{l-1}}))$으로 결합, 분산 감소로 **복잡도 $O(\epsilon^{-2})$** 달성 |

</details>

<br/>

### 🔹 Chapter 6: 시간반전 SDE와 Diffusion Model

> **핵심 질문:** Forward SDE의 시간반전은 어떤 SDE인가? Anderson(1982) 공식의 score function 항은 어디서 오는가? DDPM이 VP-SDE의 이산화임을 어떻게 증명하는가?

<details>
<summary><b>Anderson 시간반전부터 DDPM 재유도까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Anderson의 시간반전 공식 (1982)](./ch6-reverse-diffusion/01-anderson-reverse-sde.md) | Forward $dX_t = b\,dt + \sigma dB_t$에 대해, **Reverse $d\bar{X}_t = (b - \sigma\sigma^T \nabla\log p_t)dt + \sigma d\bar{B}_t$** — Fokker-Planck의 시간반전 + 이토 공식으로 유도, **score $\nabla\log p_t$가 튀어나오는 수학적 필연성** |
| [02. Score Function과 Tweedie Formula](./ch6-reverse-diffusion/02-tweedie-formula.md) | $\nabla\log p_t(x)$의 기하학적 의미(확률밀도가 "올라가는" 방향), **Tweedie 공식** $\mathbb{E}[X_0 \mid X_t = x] = x + \sigma_t^2 \nabla\log p_t(x)$ (Gaussian noise 가정), posterior mean으로서의 denoising 해석 |
| [03. Score Matching — 원래 정식화](./ch6-reverse-diffusion/03-score-matching.md) | 목적함수 $\mathbb{E}[\|s_\theta(x) - \nabla\log p(x)\|^2]$에서 $\nabla\log p$가 미지인 문제, **Hyvärinen(2005)** 등가 변환 $= \mathbb{E}[\frac{1}{2}\|s_\theta\|^2 + \text{tr}(\nabla s_\theta)] + \text{const}$ 증명 (부분적분), 고차원에서 trace의 계산 비용 |
| [04. Denoising Score Matching (Vincent 2011)](./ch6-reverse-diffusion/04-denoising-score-matching.md) | $\mathbb{E}_{x,\tilde{x}}[\|s_\theta(\tilde{x}) - \nabla\log p(\tilde{x}\|x)\|^2]$가 **원래 SM과 동치**임을 증명, Gaussian perturbation에서 $\nabla\log p(\tilde{x}\|x) = -(\tilde{x}-x)/\sigma^2$, 학습 가능한 형태로의 전환 |
| [05. Score-based SDE (Song et al. 2021)](./ch6-reverse-diffusion/05-score-based-sde.md) | **VE-SDE** (Variance Exploding, SMLD) $dX = \sqrt{d\sigma^2/dt}\,dB$와 **VP-SDE** (Variance Preserving, DDPM) $dX = -\frac{1}{2}\beta(t)X\,dt + \sqrt{\beta(t)}\,dB$의 정의, Anderson 공식으로 reverse SDE 생성, 통합 프레임워크 |
| [06. DDPM을 SDE로 재유도](./ch6-reverse-diffusion/06-ddpm-as-sde.md) | DDPM의 이산 markov chain $q(x_t\|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$이 **VP-SDE의 Euler-Maruyama 이산화**임을 증명, DDPM 손실 $\|\epsilon - \epsilon_\theta\|^2$가 **DSM과 스케일만 다른 동일 목적**임을 유도 |

</details>

<br/>

### 🔹 Chapter 7: SDE 기반 생성모델 심화

> **핵심 질문:** Reverse SDE와 같은 marginal을 갖는 deterministic ODE는? Flow Matching은 SM과 어떻게 다른가? SGLD는 왜 Bayesian posterior에서 샘플링하는가?

<details>
<summary><b>Probability Flow ODE부터 SGLD까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Probability Flow ODE](./ch7-advanced-generative/01-probability-flow-ode.md) | Reverse SDE와 **동일 marginal $p_t$** 를 갖는 deterministic ODE $d\bar{X}_t = (b - \frac{1}{2}\sigma\sigma^T\nabla\log p_t)dt$ 유도 (FP가 같도록 drift를 조정), **DDIM과의 수학적 동치**, likelihood 계산 가능성(변수변환 공식) |
| [02. Stochastic Localization과 Föllmer SDE](./ch7-advanced-generative/02-stochastic-localization.md) | 조건부 분포의 확률적 경로로서의 Föllmer SDE, **Dai Pra의 entropy minimization** 해석, **Schrödinger bridge**와의 연결, 최근 연구(Eldan, Montanari)에서의 응용 |
| [03. Flow Matching (Lipman et al. 2023)](./ch7-advanced-generative/03-flow-matching.md) | Continuous Normalizing Flow를 OT 경로로 학습, **Conditional Flow Matching** 목적함수 $\mathbb{E}[\|v_\theta(t, x) - u_t(x\|x_1)\|^2]$, SM과의 비교(score ≈ 속도장), rectified flow |
| [04. Bayesian Sampling으로서의 SDE — Langevin MCMC, SGLD](./ch7-advanced-generative/04-bayesian-sampling.md) | **Langevin MCMC**의 이산화 $x_{k+1} = x_k - \eta\nabla U(x_k) + \sqrt{2\eta}\xi_k$, **Underdamped Langevin**(HMC의 연속극한), **SGLD**(Welling & Teh 2011) — posterior $\propto \text{prior} \times \text{likelihood}$에서 샘플하는 Bayesian deep learning |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 88개 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **브라운 운동의 무한변동** | $\sup_\pi \sum \|B_{t_{i+1}} - B_{t_i}\| = \infty$ a.s. — 경로별 이토 적분 불가능성의 근거 | [Ch1-01](./ch1-ito-integral/01-why-pathwise-fails.md) |
| **이토 등장성(Itô isometry)** | $\mathbb{E}[(\int H\,dB)^2] = \mathbb{E}[\int H^2\,ds]$ — $L^2$ 확장의 수학적 기반 | [Ch1-02](./ch1-ito-integral/02-simple-process-isometry.md) |
| **이차변분 $L^2$ 수렴** | $\sum (B_{t_{i+1}} - B_{t_i})^2 \to t$ in $L^2$ — $(dB)^2 = dt$의 원천 | [Ch2-02](./ch2-ito-formula/02-db-squared-equals-dt.md) |
| **이토 공식** | $df(X_t) = f'(X_t)\,dX_t + \frac{1}{2}f''(X_t)\,d\langle X\rangle_t$ — 확률해석의 연쇄법칙 | [Ch2-01](./ch2-ito-formula/01-ito-formula-statement.md) |
| **SDE 존재·유일성** | Lipschitz + 선형성장 하에서 강해 존재·유일 (Picard 반복 + Grönwall) | [Ch3-02](./ch3-sde/02-existence-uniqueness.md) |
| **Fokker-Planck 방정식** | $\partial_t p = -\nabla\cdot(bp) + \frac{1}{2}\nabla^2:(\sigma\sigma^T p)$ — 이토 공식 + 부분적분 유도 | [Ch4-01](./ch4-fokker-planck/01-fokker-planck-derivation.md) |
| **Langevin의 Gibbs 정상분포** | $dX = -\nabla U\,dt + \sqrt{2}\,dB$ ⇒ $\pi \propto e^{-U}$ — FP 직접 검증 | [Ch4-03](./ch4-fokker-planck/03-stationary-distribution.md) |
| **Log-Sobolev ⇒ 지수 수렴** | LSI($\lambda$) ⇒ $H(p_t\|\pi) \leq e^{-2\lambda t} H(p_0\|\pi)$ | [Ch4-05](./ch4-fokker-planck/05-log-sobolev.md) |
| **Euler-Maruyama 강수렴 1/2차** | $\mathbb{E}\|X_T - \bar X_T^h\| \leq C h^{1/2}$ | [Ch5-01](./ch5-numerical/01-euler-maruyama.md) |
| **Milstein 강수렴 1차** | 이토-Taylor 전개의 추가 항 $\frac{1}{2}\sigma\sigma'((\Delta B)^2 - h)$로 차수 향상 | [Ch5-02](./ch5-numerical/02-milstein.md) |
| **Anderson 시간반전 공식** | $d\bar X_\tau = (-b + \sigma\sigma^T \nabla\log p_{T-\tau})d\tau + \sigma\,d\bar B_\tau$ — FP 시간반전 유도 | [Ch6-01](./ch6-reverse-diffusion/01-anderson-reverse-sde.md) |
| **Tweedie 공식** | $\mathbb{E}[X\mid Y=y] = y + \sigma^2\nabla\log p_Y(y)$ — score ↔ posterior mean | [Ch6-02](./ch6-reverse-diffusion/02-tweedie-formula.md) |
| **Hyvärinen SM 등가 변환** | $\mathbb{E}\|s_\theta - \nabla\log p\|^2 = \mathbb{E}[\frac{1}{2}\|s_\theta\|^2 + \text{tr}(\nabla s_\theta)] + \text{const}$ | [Ch6-03](./ch6-reverse-diffusion/03-score-matching.md) |
| **DSM ⇔ SM 등가성** | Vincent(2011) — Gaussian 섭동 하의 perturbed SM이 원래 SM과 $\theta$ 관점에서 동치 | [Ch6-04](./ch6-reverse-diffusion/04-denoising-score-matching.md) |
| **DDPM = VP-SDE의 EM 이산화** | DDPM 이산 markov chain이 VP-SDE의 Euler-Maruyama 이산화이고 손실이 $\sigma_t^2$-가중 DSM과 일치 | [Ch6-06](./ch6-reverse-diffusion/06-ddpm-as-sde.md) |
| **PF-ODE drift 공식** | Reverse SDE와 같은 marginal을 갖는 ODE의 drift $b - \frac{1}{2}\sigma\sigma^T\nabla\log p_t$ | [Ch7-01](./ch7-advanced-generative/01-probability-flow-ode.md) |
| **CFM ⇔ FM 등가성** | Lipman(2023) — Conditional Flow Matching gradient가 Flow Matching gradient와 일치 | [Ch7-03](./ch7-advanced-generative/03-flow-matching.md) |

> 💡 **챕터별 총 정리 수**: Ch1(15) · Ch2(11) · Ch3(17) · Ch4(12) · Ch5(11) · Ch6(8) · Ch7(14) — 합계 **88개 정리 + 증명**, 약 **18,000+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
tqdm==4.66.0
sdeint==0.3.0         # SDE 수치적분 비교 레퍼런스
torch==2.1.0          # Score network 학습 (Ch6~7에서 최소한)
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            tqdm==4.66.0 sdeint==0.3.0 torch==2.1.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — OU 과정의 Euler-Maruyama + 정상분포 검증 + 이차변분 확인
import numpy as np
import matplotlib.pyplot as plt

# dX = -θX dt + σ dB,  정상분포 N(0, σ²/2θ)
theta, sigma = 2.0, 1.0
T, dt = 10.0, 0.01
N = int(T / dt)
n_paths = 10_000

# ─────────────────────────────────────────────
# 1. Euler-Maruyama 시뮬레이션 — 정상분포 수렴
# ─────────────────────────────────────────────
rng = np.random.default_rng(0)
X = np.zeros((n_paths, N + 1))
X[:, 0] = rng.standard_normal(n_paths) * 3  # 비정상 초기분포
for k in range(N):
    dB = rng.standard_normal(n_paths) * np.sqrt(dt)
    X[:, k + 1] = X[:, k] + (-theta * X[:, k]) * dt + sigma * dB

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(np.linspace(0, T, N + 1), X[:50].T, alpha=0.3)
axes[0].set_title('OU 샘플 경로 50개'); axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$X_t$')
axes[0].grid(True, alpha=0.3)

# t=0, 1, 3, 10에서의 marginal과 이론 정상분포 비교
for t_idx, t in [(0, 0), (100, 1), (300, 3), (1000, 10)]:
    axes[1].hist(X[:, t_idx], bins=50, density=True, alpha=0.3, label=f't={t}')
stationary_std = sigma / np.sqrt(2 * theta)
x_grid = np.linspace(-3, 3, 200)
axes[1].plot(x_grid,
             np.exp(-x_grid**2 / (2 * stationary_std**2)) / (stationary_std * np.sqrt(2 * np.pi)),
             'k--', label=r'정상분포 $\mathcal{N}(0, \sigma^2/2\theta)$')
axes[1].legend(); axes[1].set_title('시간별 marginal 분포')
axes[1].set_xlabel('x'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 2. 이차변분 확인 — Σ(ΔB)² → t in L²
# ─────────────────────────────────────────────
B = np.cumsum(rng.standard_normal(N) * np.sqrt(dt))
quadratic_var = np.cumsum(np.diff(B, prepend=0)**2)
print(f'이차변분 이론값 (t=T)  : {T}')
print(f'이차변분 측정값        : {quadratic_var[-1]:.4f}')
# → T에 수렴. 이것이 이토 공식의 (dB)² = dt 항의 수치적 근거.

# ─────────────────────────────────────────────
# 3. 강수렴 차수 검증 (Euler-Maruyama vs Milstein)
# ─────────────────────────────────────────────
# GBM: dS = μ S dt + σ S dB, 해석해 S_T = S_0 exp((μ - σ²/2)T + σ B_T)로 비교
# h를 줄여가며 경로별 오차 E|S_T - S_T^h| 측정 → EM은 기울기 0.5, Milstein은 1.0
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 이론이 AI(특히 생성모델)에서 중요한가** | DDPM, Score-SDE, Flow Matching, Langevin MCMC와의 연결점 |
| 3 | 📐 **수학적 선행 조건** | Stochastic Processes, Probability Theory 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | 브라운 운동의 무한변동, "노이즈를 더하고 뒤집기"의 직관 |
| 5 | ✏️ **엄밀한 정의** | 이토 적분·SDE·Fokker-Planck의 측도론적 정의 |
| 6 | 🔬 **정리와 증명** | 이토 공식, Anderson 시간반전, DSM 동치성 — "자명하다" 없이 |
| 7 | 💻 **NumPy 구현 검증** | SDE 수치적분, score 추정, reverse SDE 샘플링 |
| 8 | 🔗 **AI/ML 연결** | DDPM, VP/VE-SDE, Flow Matching, SGLD |
| 9 | ⚖️ **가정과 한계** | Lipschitz 깨지면? Score 추정 오차는? |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현 문제 |

> 📚 **연습문제 총 108개**: 36문서 × 문서당 3문제(기초/심화/AI 연결), 모든 문제에 `<details>` 펼침 해설 포함. 손 계산 재현부터 DDPM/TRPO 연결까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되므로 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 523줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 36문서는 약 **45~55시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Diffusion Model을 구현하지만 왜 reverse SDE인지 모른다" — Diffusion 집중 (5일, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch1-01  브라운 운동의 무한변동 → 왜 (dB)²가 dt인지
       Ch1-02  이토 등장성
Day 2  Ch2-01~02  이토 공식의 서술과 (dB)² = dt 증명
Day 3  Ch3-01, Ch4-01  SDE 정의와 Fokker-Planck 유도
Day 4  Ch6-01  Anderson 시간반전 공식
       Ch6-02  Tweedie / Score function 기초
Day 5  Ch6-03~04  Score Matching과 DSM 동치성
       Ch6-05~06  VP-SDE와 DDPM 재유도
```

</details>

<details>
<summary><b>🟡 "SDE 수치해법을 쓰지만 수렴차수 논리를 모른다" — 수치해법 집중 (1주, 약 14~18시간)</b></summary>

<br/>

```
Day 1  Ch1-01~02  이토 적분 기초
Day 2  Ch2-01~02  이토 공식, (dB)² = dt
Day 3  Ch3-01~02  SDE 정의와 존재·유일성
Day 4  Ch3-03~04  OU와 GBM — 해석해로 수치 비교 대상 확보
Day 5  Ch5-01  Euler-Maruyama와 강수렴 1/2차
Day 6  Ch5-02~03  Milstein(1차)과 강/약 수렴 분리
Day 7  Ch5-04~05  Implicit 방법과 MLMC
```

</details>

<details>
<summary><b>🔴 "확률해석과 Diffusion의 수학적 기반을 완전 정복한다" — 전체 정복 (8주, 약 45~55시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 이토 적분의 엄밀한 구성
        → 무한변동 경로별 적분 불가능성 직접 체감
        → 이토 등장성 증명 숙지

2주차  Chapter 2 전체 — 이토 공식
        → (dB)² = dt의 이차변분 증명 재구성
        → Black-Scholes PDE를 이토 공식 한 줄씩 유도

3주차  Chapter 3 전체 — SDE
        → Picard 반복으로 존재·유일성 증명
        → OU·GBM의 해석해 손 계산

4주차  Chapter 4 전체 — Fokker-Planck
        → 부분적분으로 FP 유도
        → Langevin의 Gibbs 정상분포 직접 검증
        → LSI로 지수 수렴 증명

5주차  Chapter 5 전체 — 수치해법
        → NumPy로 EM/Milstein 강수렴 차수 실측
        → MLMC 복잡도 $O(\epsilon^{-2})$ 구현

6주차  Chapter 6 (1~3) — 시간반전과 Score Matching
        → Anderson 공식을 FP에서 한 줄씩 유도
        → Hyvärinen SM 등가 변환 증명 재구성

7주차  Chapter 6 (4~6) — DSM과 Score-SDE
        → Vincent 2011 DSM 동치성 증명
        → DDPM = VP-SDE Euler-Maruyama 이산화 증명
        → 실제 DSM 학습 미니 실험 (MNIST 수준)

8주차  Chapter 7 전체 — 심화
        → PF-ODE로 likelihood 계산
        → Flow Matching 목적함수 유도
        → SGLD로 베이지안 posterior 샘플
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 측도, 조건부 기댓값, $L^2$ 수렴, 특성함수 | Ch1 전체(이토 적분의 $L^2$ 구성), Ch4-02(Feynman-Kac) |
| [stochastic-processes-deep-dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) | 브라운 운동, 마팅게일, 이차변분, 정지시각 | Ch1-01(무한변동), Ch1-04(마팅게일 성질), Ch2-02(이차변분 정리) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 다변수 미분, Taylor 전개, 편미분방정식 기초 | Ch2 전체(Taylor → 이토), Ch4-01(FP의 부분적분) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 양정치 행렬, 스펙트럴 분해 | Ch2-03(다차원 이토의 $\sigma\sigma^T$), Ch4-05(LSI 스펙트럴 갭) |
| [information-geometry-deep-dive](https://github.com/iq-ai-lab/information-geometry-deep-dive) | Fisher 정보, Score function, KL divergence | Ch6-02(Score function), Ch6-03(SM의 Fisher divergence 해석), Ch4-04(KL 감소) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | 볼록성, Legendre 변환, Bregman divergence | Ch4-04(Langevin과 경사흐름), Ch7-03(Flow Matching과 OT) |

> 💡 이 레포는 **확률미분과 생성모델의 수학적 기반**에 집중합니다. Stochastic Processes에서 브라운 운동의 측도론적 성질을, Probability에서 $L^2$ 수렴을 학습한 후 오면 Ch1의 이토 적분 구성이 훨씬 자연스럽습니다. Ch6~7(Diffusion Model)는 딥러닝 실전 경험(MNIST/CIFAR 수준의 학습 경험)이 있을 때 최대의 효과를 냅니다.

---

## 📖 Reference

### 🏛️ 확률해석 바이블·표준 교재
- **Brownian Motion and Stochastic Calculus** (Karatzas & Shreve, 1991) — 확률해석 표준 교재, 이토 적분의 엄밀한 구성
- **Stochastic Differential Equations: An Introduction with Applications** (Øksendal, 2003) — SDE 입문의 표준
- **Continuous Martingales and Brownian Motion** (Revuz & Yor, 1999) — 마팅게일 이론 심화
- **Stochastic Calculus for Finance II: Continuous-Time Models** (Shreve, 2004) — Black-Scholes, 금융 응용
- **Diffusions, Markov Processes, and Martingales** (Rogers & Williams, 2000) — Markov 과정과 SDE

### 🔢 SDE 수치해법
- **Numerical Solution of Stochastic Differential Equations** (Kloeden & Platen, 1992) — 수치해법의 바이블
- **Multilevel Monte Carlo Methods** (Giles, 2015) — MLMC 종합
- **An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations** (Higham, 2001) — 입문자 친화적 리뷰

### 🌀 Fokker-Planck · Langevin · Log-Sobolev
- **Reverse-Time Diffusion Equation Models** (Anderson, 1982) — **시간반전 SDE 원전**
- **Analysis and Geometry of Markov Diffusion Operators** (Bakry, Gentil & Ledoux, 2014) — LSI와 Markov 반군 이론
- **Stochastic Modelling and Applied Probability** (Pavliotis, 2014) — Fokker-Planck 교재
- **The Variational Formulation of the Fokker-Planck Equation** (Jordan, Kinderlehrer & Otto, 1998) — JKO, Wasserstein gradient flow

### 🎨 Score-based Generative Model · Diffusion
- **Estimation of Non-Normalized Statistical Models by Score Matching** (Hyvärinen, 2005) — **SM 원전**
- **A Connection Between Score Matching and Denoising Autoencoders** (Vincent, 2011) — **DSM 원전**
- **Generative Modeling by Estimating Gradients of the Data Distribution** (Song & Ermon, 2019) — SMLD (VE-SDE의 이산판)
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) — **DDPM 원전**
- **Score-Based Generative Modeling through Stochastic Differential Equations** (Song et al., 2021) — **Score SDE / VP-VE 통합 프레임워크**
- **Denoising Diffusion Implicit Models** (Song et al., 2020) — DDIM, PF-ODE
- **Elucidating the Design Space of Diffusion-Based Generative Models** (Karras et al., 2022) — EDM

### 🌊 Flow Matching · Schrödinger Bridge · 후속 연구
- **Flow Matching for Generative Modeling** (Lipman et al., 2023) — **Flow Matching 원전**
- **Rectified Flow** (Liu et al., 2023) — 직선화된 경로
- **Stochastic Interpolants** (Albergo, Boffi, Vanden-Eijnden, 2023) — 보간자 기반 통합
- **Schrödinger Bridge Sampler** (De Bortoli et al., 2021) — SB와 확산모델

### 🎲 MCMC · Bayesian Inference
- **Bayesian Learning via Stochastic Gradient Langevin Dynamics** (Welling & Teh, 2011) — **SGLD 원전**
- **Hamiltonian Monte Carlo** (Neal, 2011) — HMC
- **A Complete Recipe for Stochastic Gradient MCMC** (Ma, Chen, Fox, 2015) — SG-MCMC 통합

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"브라운 운동 위에서 미적분하는 것과, 왜 $(dB)^2 = dt$이며 이토 적분이 $L^2$ 극한으로만 정의될 수 있는지를 증명할 수 있는 것은 다르다"*

</div>
