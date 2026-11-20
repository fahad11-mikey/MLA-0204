# MLA02 – Fundamentals of Machine Learning
## Assignment V – Unit V: Advanced Learning

**Course:** Fundamentals of Machine Learning  
**Course Code:** MLA02  
**Total Marks:** 50  
**CO5:** Create graphical models to analyse and infer patterns in complex probabilistic systems for structured decision-making.

---

## Task 1 – Monte Carlo Estimation of π (5 Marks)

### (a) Points inside the quarter circle
A point is inside when `u² + v² < 1`.

| Point | u | v | u² + v² | Result |
|---|---:|---:|---:|---|
| 1 | 0.10 | 0.20 | 0.05 | Inside |
| 2 | 0.90 | 0.80 | 1.45 | Outside |
| 3 | 0.50 | 0.50 | 0.50 | Inside |
| 4 | 0.30 | 0.95 | 0.9925 | Inside |
| 5 | 0.70 | 0.60 | 0.85 | Inside |
| 6 | 0.85 | 0.40 | 0.8825 | Inside |
| 7 | 0.20 | 0.10 | 0.05 | Inside |
| 8 | 0.95 | 0.95 | 1.805 | Outside |
| 9 | 0.60 | 0.30 | 0.45 | Inside |
| 10 | 0.40 | 0.90 | 0.97 | Inside |

There are 8 inside points out of 10.

`π̂ = 4 × (8/10) = 3.2`


### (b) Standard error
Using the observed proportion `p̂ = 0.8`:

`SE(π̂) = 4 √[p̂(1 − p̂)/n]`

`= 4 √[(0.8 × 0.2)/10] ≈ 0.506`

### (c) Samples for standard error 0.01
Using `p = π/4 ≈ 0.7854` and requiring the standard error of the π estimate to be 0.01:

`4 √[p(1 − p)/n] = 0.01`

`n = 16p(1 − p)/(0.01)² ≈ 31,360` samples (rounded up).

Monte Carlo standard error decreases at the rate `O(1/√n)`. Therefore, reducing error by half requires approximately four times as many samples.

---

## Task 2 – Inverse Transform Sampling (2 Marks)

For an exponential distribution, `F(x) = 1 − e^(−λx)`.

Set `u = F(x)`:

`u = 1 − e^(−λx)`  
`e^(−λx) = 1 − u`  
`x = −ln(1 − u)/λ`

Given `λ = 0.5` and `u = 0.6`:

`x = −ln(0.4)/0.5 = −2 ln(0.4) ≈ 1.8326`

Verification:

`F(1.8326) = 1 − e^(−0.5 × 1.8326) ≈ 1 − 0.4 = 0.6`

The generated sample is **1.8326**.

---

## Task 3 – Rejection Sampling and Importance Sampling (6 Marks)

### (a) Rejection sampling (3 Marks)
Target density: `p(x) = 6x(1 − x)` and proposal `q(x) = 1`. Bound: `M = 1.5`.

Acceptance probability for a candidate `x`:

`p(x)/(M q(x)) = 6x(1 − x)/1.5 = 4x(1 − x)`

Overall acceptance rate is `1/M = 2/3 ≈ 66.67%`.

- Candidate `(x = 0.3, v = 0.7)`: acceptance probability `4(0.3)(0.7) = 0.84`. Since `0.7 < 0.84`, **accept**.
- Candidate `(x = 0.9, v = 0.5)`: acceptance probability `4(0.9)(0.1) = 0.36`. Since `0.5 > 0.36`, **reject**.

### (b) Importance sampling (3 Marks)
For `p = N(0,1)` and `q = N(1,1)`,

`w(x) = p(x)/q(x) = exp(0.5 − x)`

| x | Weight w(x) |
|---:|---:|
| 0.5 | 1.0000 |
| 1.2 | 0.4966 |
| 2.0 | 0.2231 |

Self-normalised estimate:

`Ê[X²] = Σwᵢxᵢ² / Σwᵢ ≈ 1.0802`

Effective sample size:

`ESS = (Σw)²/Σw² ≈ 2.2813`

The estimate is close to the true value 1, but only three samples were used and the ESS is low. The estimate is therefore not highly reliable; more samples are needed.

---

## Task 4 – Three-Armed Bandit (4 Marks)

### (a) Sample-average estimates

| Arm | Reward history | Q(a) |
|---|---|---:|
| 1 | 1, 0, 1, 1 | 0.75 |
| 2 | 1, 0 | 0.50 |
| 3 | 0 | 0.00 |

### (b) ε-greedy selection (ε = 0.2)
Assuming exploration selects uniformly from all three arms, the greedy arm is Arm 1.

- Arm 1: `1 − ε + ε/3 = 0.8667`
- Arm 2: `ε/3 = 0.0667`
- Arm 3: `ε/3 = 0.0667`

### (c) Update Arm 2
Old estimate `Q₂ = 0.5`, old count `n = 2`, reward `R = 1`:

`Qnew = 0.5 + (1/3)(1 − 0.5) = 0.6667`

### (d) Exploration–exploitation
Arm 1 currently has the highest average reward, but the estimates are based on limited observations. Exploiting Arm 1 gives a good immediate reward estimate, while exploring Arm 2 or Arm 3 may reveal better rewards. The agent must balance choosing the best-known arm and collecting information.

---

## Task 5 – Upper Confidence Bound (UCB1) (3 Marks)

Given `t = 32`, `Q = (0.6, 0.5, 0.3)`, `N = (20, 10, 2)`, and `c = 1`:

`UCB(a) = Q(a) + √(ln 32/N(a))`

| Arm | UCB Index (approx.) |
|---|---:|
| 1 | 1.0163 |
| 2 | 1.0887 |
| 3 | 1.6164 |

**UCB1 selects Arm 3**, because its uncertainty bonus is high due to fewer pulls.

- Greedy selects Arm 1 (highest Q).
- ε-greedy (ε = 0.1) usually selects Arm 1, with a small probability of exploration.
- UCB1 selects Arm 3 to explore its uncertain reward estimate.

---

## Task 6 – Discounted Returns (3 Marks)

Rewards: `2, 0, 4, 1`, discount `γ = 0.9`.

Working backward:

- `G₃ = 1`
- `G₂ = 4 + 0.9(1) = 4.9`
- `G₁ = 0 + 0.9(4.9) = 4.41`
- `G₀ = 2 + 0.9(4.41) = 5.969`

| Return | Value |
|---|---:|
| G₀ | 5.969 |
| G₁ | 4.410 |
| G₂ | 4.900 |
| G₃ | 1.000 |

Undiscounted total: `2 + 0 + 4 + 1 = 7`.

With `γ = 0.5`, returns are `G₀ = 3.125`, `G₁ = 2.25`, `G₂ = 4.5`, `G₃ = 1`. A lower discount factor gives less importance to future rewards.

---

## Task 7 – Value Iteration (4 Marks)

Given `γ = 0.9`, `V₀(A) = V₀(B) = 0`, and `V(G) = 0`.

Action equations:

- `A, Right = 0.9V(B) + 0.1V(A)`
- `A, Stay = 1 + 0.9V(A)`
- `B, Right = 8 + 0.18V(A)`
- `B, Left = 0.9V(A)`

| Sweep | V(A) | V(B) |
|---|---:|---:|
| 1 | 1.000 | 8.000 |
| 2 | 7.300 | 8.180 |
| 3 | 8.092 | 9.314 |

Using the values from sweep 2, the greedy actions for sweep 3 are:

- A → Right (value 8.092)
- B → Right (value 9.314)

Largest change in sweep 3:

`max(|8.092 − 7.3|, |9.314 − 8.18|) = 1.134`

The values have not converged because the change is still significant.

---

## Task 8 – Policy Iteration (4 Marks)

Initial policy: `A → Stay`, `B → Right`.

### (a) Policy evaluation
For A:

`V(A) = 1 + 0.9V(A)`  
`V(A) = 10`

For B:

`V(B) = 8 + 0.18V(A) = 8 + 1.8 = 9.8`

### (b) Policy improvement
- At A: Stay gives `10`; Right gives `0.9(9.8) + 0.1(10) = 9.82`. Stay remains best.
- At B: Right gives `9.8`; Left gives `0.9(10) = 9`. Right remains best.

The policy is **stable**: A → Stay, B → Right.

### (c) Value iteration after three sweeps
After three sweeps: `V₃(A) = 8.092`, `V₃(B) = 9.314`, with greedy actions A → Right and B → Right.

### (d) Why they appear to disagree
Policy iteration evaluates its chosen policy to convergence before improving it. Value iteration here has only completed three sweeps from zero, so its values are still estimates and can produce a temporary different action. More value-iteration sweeps are needed for convergence.

---

## Task 9 – Temporal-Difference Learning (TD(0)) (3 Marks)

Given `α = 0.2`, `γ = 0.9`, `V(S₁) = 2`, `V(S₂) = 3`, and `V(T) = 0`.

### Transition 1: S₁ → S₂, reward 1

`δ₁ = r + γV(S₂) − V(S₁) = 1 + 0.9(3) − 2 = 1.7`

`Vnew(S₁) = 2 + 0.2(1.7) = 2.34`

### Transition 2: S₂ → T, reward 4

`δ₂ = 4 + 0.9(0) − 3 = 1`

`Vnew(S₂) = 3 + 0.2(1) = 3.2`

The TD target for S₁ is `1 + 0.9(3) = 3.7`, using the current estimate of S₂. The Monte Carlo return is `1 + 0.9(4) = 4.6`. TD uses the next state's estimate, while Monte Carlo uses the observed full return.


---

## Task 10 – Q-Learning, SARSA and Expected SARSA (4 Marks)

Given `Q(s,a) = 2`, `r = 1`, `γ = 0.9`, `α = 0.5`, `Q(s′,a₁) = 3`, `Q(s′,a₂) = 5`, and ε = 0.2. The greedy next action is a₂, while the behaviour selected exploratory action a₁.

### (i) Q-learning (off-policy)

`Qnew = 2 + 0.5[1 + 0.9(5) − 2] = 3.75`

### (ii) SARSA (on-policy)

`Qnew = 2 + 0.5[1 + 0.9(3) − 2] = 2.85`

### (iii) Expected SARSA (on-policy expected update)
Assuming ε-greedy exploration chooses uniformly between the two actions, probabilities are `P(a₁) = 0.1`, `P(a₂) = 0.9`.

`E[Q(s′,a′)] = 0.1(3) + 0.9(5) = 4.8`

`Qnew = 2 + 0.5[1 + 0.9(4.8) − 2] = 3.66`

| Method | Updated Q | Policy type |
|---|---:|---|
| Q-learning | 3.75 | Off-policy; uses maximum next Q |
| SARSA | 2.85 | On-policy; uses action actually selected |
| Expected SARSA | 3.66 | On-policy; uses expected next Q |

The exploratory action affects SARSA directly. Q-learning uses the greedy maximum regardless of the exploratory action. Expected SARSA accounts for the behaviour policy's probabilities.

---

## Task 11 – Sample Complexity, Halving and Occam Learning (4 Marks)

### (a) Consistent learner
`|H| = 3¹⁰ = 59,049`, `ε = 0.1`, `δ = 0.05`.

Using the PAC bound `m ≥ (1/ε)[ln|H| + ln(1/δ)]`:

`m ≥ 10[ln(59049) + ln(20)] ≈ 139.82`

Therefore, **140 training examples** are sufficient under this bound.

### (b) Halving mistake bound
`log₂(59,049) ≈ 15.85`

Maximum mistakes: **16**.

### (c) Occam learner
For a hypothesis described by at most 20 bits, `|H| ≤ 2²⁰`.

`m ≥ (1/ε)[20 ln 2 + ln(1/δ)]`

`m ≥ 10[20 ln 2 + ln 20] ≈ 168.5`

Therefore, **169 examples** are sufficient using this bound. Occam learning favours simpler hypotheses because shorter descriptions correspond to smaller hypothesis spaces and can require fewer samples to generalise.

---

## Task 12 – VC Dimension and Boosting (8 Marks)

### (a) Axis-aligned rectangles: VC(Hᵣ) = 4
Choose four points at the corners of a rectangle. An axis-aligned rectangle can include any selected subset of these points while excluding the others, so the four points can be shattered. A set of five points cannot be shattered by axis-aligned rectangles; therefore, VC dimension is 4.

### (b) Linear separators: VC(Hₗ) = 3
Three non-collinear points can be shattered by a linear separator in two dimensions. Four points cannot be shattered because a configuration with one point inside the triangle formed by the other three cannot have that inner point separated from all three outer points by a line. Therefore, VC dimension is 3.

### (c) Sample-complexity bound
Given:

`m ≥ (1/ε)[4log₂(2/δ) + 8VC(H)log₂(13/ε)]`

with `ε = 0.1`, `δ = 0.05`.

- Rectangles, VC = 4: `m ≥ 2460.04`, so **2,461 examples**.
- Linear separators, VC = 3: `m ≥ 1898.25`, so **1,899 examples**.

The linear-separator class needs fewer examples according to this bound because its VC dimension is lower.

### (d) Repeated runs and model choice
Success probability per run = 0.5. Probability that at least one run succeeds after n independent runs:

`1 − (1 − 0.5)ⁿ ≥ 0.99`

`1 − 0.5ⁿ ≥ 0.99`  
`0.5ⁿ ≤ 0.01`

`n ≥ log₂(100) ≈ 6.64`

Therefore, **7 independent runs** are required. Evaluate each run on a validation set and select the hypothesis with the lowest validation error (with appropriate safeguards against overfitting). For a data-poor project, linear separators are recommended because they have a smaller VC dimension and a lower sample-complexity bound, assuming the data relationship is reasonably linear.

---

## Final Conclusion
Sampling methods estimate quantities using random draws, while reinforcement-learning techniques improve decisions through rewards and feedback. Computational learning theory explains how the hypothesis space, sample size, VC dimension and confidence affect generalisation. Monte Carlo, bandit strategies, value-based learning and sample-complexity bounds provide practical tools for making data-driven decisions under uncertainty.

---

## GitHub Submission Checklist
- [ ] Upload this file as `MLA02_Assignment5_GitHub.md`.
- [ ] Add your name and register number to your repository README.
- [ ] If required by your faculty, add calculations, handwritten workings or screenshots separately.
- [ ] Review the answers and make sure you can explain each method during verification.
