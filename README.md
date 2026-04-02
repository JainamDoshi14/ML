# Gradient Descent from Scratch

> A pure Python + NumPy implementation of linear regression with Batch, Mini-batch, and Stochastic Gradient Descent — no sklearn, no autograd, no black boxes.  
> Built to deeply understand the math behind how machine learning models actually learn.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [The Math](#the-math)
   - [The Model](#the-model)
   - [The Cost Function — MSE](#the-cost-function--mse)
   - [Why Partial Derivatives?](#why-partial-derivatives)
   - [Deriving the Gradients](#deriving-the-gradients)
   - [Why We Subtract While Updating](#why-we-subtract-while-updating)
   - [Why the Gradient Always Points Uphill](#why-the-gradient-always-points-uphill)
   - [Why MSE Always Has One Minimum](#why-mse-always-has-one-minimum)
5. [ML Concepts](#ml-concepts)
   - [Epochs and Learning Rate](#epochs-and-learning-rate)
   - [Weight and Bias](#weight-and-bias)
   - [Types of Gradient Descent](#types-of-gradient-descent)
6. [Code Walkthrough](#code-walkthrough)
7. [Output and Visualization](#output-and-visualization)
8. [Hyperparameters](#hyperparameters)
9. [Observations](#observations)
10. [Possible Extensions](#possible-extensions)

---

## What This Project Does

Trains a linear regression model using **gradient descent** — the same core algorithm that powers neural networks — implemented entirely from scratch using only NumPy.

Given a dataset of `(x, y)` pairs, the model learns the best-fit line:

```
ŷ = w·x + b
```

by starting with `w = 0, b = 0` and iteratively improving them over many epochs until error is minimized. Three optimization variants are implemented and compared:

| Variant | Data Per Update | Speed | Stability |
|---|---|---|---|
| Batch GD | Entire dataset | Slow | Very stable |
| Mini-batch GD | Small subset (e.g. 32) | Balanced | Mildly noisy |
| SGD | One sample | Fast | Highly noisy |

---

## Project Structure

```
├── gradient_descent.py      # Core implementation (training + plotting)
├── gradient_descent.ipynb   # Interactive notebook for experimentation
├── data.csv                 # Input dataset (x, y pairs)
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install numpy matplotlib

# Run
python gradient_descent.py
```

**Expected output:**
```
starting gradient descent at b = 0, w = 0, error = XXXX.XX
ending gradient descent at b = X.XX, w = X.XX, error = XX.XX
```

A matplotlib window will appear showing the **Error vs Epoch** graph for all three variants. Close it to let the program finish.

> **Note:** `plt.show()` is blocking — the program pauses until you close the plot. Use `plt.show(block=False)` if you want the program to continue without waiting.

**Or explore interactively:**
```bash
jupyter notebook gradient_descent.ipynb
```

---

## The Math

### The Model

The model fits a straight line through the data:

$$\hat{y} = wx + b$$

| Symbol | Name | Role |
|---|---|---|
| $x$ | Input feature | The data point we give the model |
| $y$ | Target output | The true value we want to predict |
| $w$ | Weight (slope) | How much $x$ influences $y$ |
| $b$ | Bias (intercept) | Shifts the line up or down |

At the start, both `w = 0` and `b = 0` — the model predicts zero for everything. Gradient descent fixes this.

---

### The Cost Function — MSE

We measure how wrong the model is using **Mean Squared Error**:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - (wx_i + b))^2$$

- Squaring makes all errors positive
- Larger errors are penalized disproportionately
- The function is smooth and differentiable — easy to optimize

**The goal:** find `w` and `b` that minimize this.

---

### Why Partial Derivatives?

The error surface `E(b, w)` is a 3D bowl — we start somewhere on the rim (`b=0, w=0`) and need to slide to the bottom (minimum error).

```
        Error (height)
          │     * * *
          │   *       *
          │  *         *   ← bowl shape
          │   *       *
          │     * * *
          └──────────────── b and w (ground plane)
                 ↑
           bottom = minimum error = best b and w
```

Since error depends on **two variables**, we use partial derivatives — one per variable:

- $\partial E / \partial b$ → *"If I nudge only `b`, how does error change?"*
- $\partial E / \partial w$ → *"If I nudge only `w`, how does error change?"*

Together, they form the **gradient** — a vector pointing in the direction of steepest increase. We move in the **opposite** direction to descend.

---

### Deriving the Gradients

#### With respect to `b`

Let $u = y_i - (wx_i + b)$, so $E = \frac{1}{N} \sum u^2$.

By the chain rule: $\frac{d(u^2)}{db} = 2u \cdot \frac{du}{db}$, and $\frac{du}{db} = -1$

$$\frac{\partial E}{\partial b} = -\frac{2}{N} \sum (y_i - (wx_i + b))$$

This is exactly what the code computes:
```python
b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
```

#### With respect to `w`

Same process, but $\frac{du}{dw} = -x$ (since `w` appears as $-wx$ inside $u$):

$$\frac{\partial E}{\partial w} = -\frac{2}{N} \sum x_i (y_i - (wx_i + b))$$

```python
w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
```

The extra $x$ in the weight gradient comes directly from the chain rule — because `w` is multiplied by `x` in the model.

---

### Why We Subtract While Updating

$$w \leftarrow w - \eta \cdot \frac{\partial E}{\partial w}, \qquad b \leftarrow b - \eta \cdot \frac{\partial E}{\partial b}$$

The gradient points **uphill**. We want to go **downhill**. So we subtract:

| Situation | Gradient sign | Effect |
|---|---|---|
| Error increases as $b$ increases | $\partial E/\partial b > 0$ | $b - \text{positive}$ → $b$ decreases ✅ |
| Error decreases as $b$ increases | $\partial E/\partial b < 0$ | $b - \text{negative}$ → $b$ increases ✅ |

Subtracting always moves toward lower error, regardless of which side of the bowl we're on.

---

### Why the Gradient Always Points Uphill

This follows directly from the definition of the derivative as rise over run:

- If $f'(x) > 0$ → function rises in the $+x$ direction → that's uphill
- If $f'(x) < 0$ → function falls in the $+x$ direction → uphill is the other way

Quick proof with $f(x) = x^2$:

| Position | Derivative | Meaning |
|---|---|---|
| $x = 3$ (right of minimum) | $f'(3) = +6$ | Uphill is rightward |
| $x = -3$ (left of minimum) | $f'(-3) = -6$ | Uphill is leftward |
| $x = 0$ (the minimum) | $f'(0) = 0$ | No uphill direction |

Subtracting the gradient always moves toward $x = 0$, the minimum.

---

### Why MSE Always Has One Minimum

MSE for linear regression is a **convex function** — a perfect bowl with exactly one global minimum.

```
MSE — Convex ✅              Non-convex (Neural Nets) ❌

     │   U shape               │  bumpy surface
     │  *       *              │ *  * *   * *
     │ *         *             │*    *     *  *
     │─────────────            │─────────────────
       one bottom                multiple valleys
```

Because MSE is convex, the gradient **always** points away from the single true minimum. Gradient descent is therefore **guaranteed to converge** for linear regression.

In neural networks, the loss surface is non-convex — gradient descent can get stuck in local minima, which is why training deep networks is fundamentally harder.

---

## ML Concepts

### Epochs and Learning Rate

**Epoch** — one full pass over the entire dataset. The model runs for `N` epochs, improving `w` and `b` slightly each time.

**Learning rate** (`η`) — controls the step size per update:

| Value | Effect |
|---|---|
| Too high (e.g. `0.1`) | Overshoots the minimum → error explodes |
| Too low (e.g. `0.0000001`) | Tiny steps → very slow convergence |
| Just right (e.g. `0.0001`) | Steadily slides down to the minimum |

---

### Weight and Bias

**Weight (`w`)** — the slope. Controls how strongly the input $x$ influences the output $y$.

**Bias (`b`)** — shifts the entire line up or down. Without it, the line is forced through the origin $(0, 0)$, severely limiting what the model can fit.

Both start at `0` and are learned entirely from data.

---

### Types of Gradient Descent

| Type | Data Per Update | Pros | Cons |
|---|---|---|---|
| **Batch GD** | All $N$ points | Stable, accurate gradients | Slow on large datasets |
| **Mini-batch GD** | Small batch (e.g. 32) | Balance of speed and stability | Slight noise |
| **Stochastic GD (SGD)** | 1 random point | Very fast, can escape local minima | Noisy, unstable |

Mini-batch GD is the default choice in modern deep learning — it's what PyTorch's `DataLoader` implements under the hood.

---

## Code Walkthrough

```
compute_error_for_line_given_points(b, w, points)
  └── Computes MSE over all data points
  └── Used to track error before and after training

step_gradient(b_current, w_current, points, learning_rate)
  └── Core of gradient descent
  └── Computes ∂E/∂b and ∂E/∂w analytically for every point
  └── Updates and returns new b and w

gradient_descent_runner(points, starting_b, starting_w, learning_rate, epochs, batch_size)
  └── Runs step_gradient for `epochs` iterations
  └── batch_size=None  → Batch GD
  └── batch_size=32    → Mini-batch GD
  └── batch_size=1     → SGD
  └── Logs and plots Error vs Epoch for all variants

run()
  └── Loads data.csv
  └── Sets hyperparameters (learning_rate, epochs)
  └── Calls gradient_descent_runner for all three variants
  └── Prints starting and ending error + final w and b
```

---

## Output and Visualization

After training, a matplotlib graph shows MSE dropping over epochs for all three variants:

```
MSE
 │ *                  ← Batch GD (smooth)
 │  *·                ← Mini-batch (mild noise)
 │   **~              ← SGD (noisy)
 │     ****
 │          **********────
 └──────────────────────── Epoch
   0                    N
```

- **Batch GD** → smooth, steady descent
- **Mini-batch GD** → slight oscillations, fast convergence
- **SGD** → rapid early progress, oscillates near the minimum

---

## Hyperparameters

| Parameter | Description |
|---|---|
| `learning_rate` | Step size for each parameter update |
| `epochs` | Number of full passes through the training data |
| `batch_size` | Controls GD variant: `None` = Batch, `1` = SGD, `n` = Mini-batch |

---

## Observations

- A **high learning rate** causes loss to diverge
- A **low learning rate** converges reliably but slowly
- **SGD** converges fast early but oscillates around the minimum
- **Batch GD** is stable but memory-inefficient on large datasets
- **Mini-batch GD** is the practical standard in real-world ML pipelines

---

## Possible Extensions

- Add **Momentum** or **Adam** optimizer
- Extend to **multivariate regression** (multiple input features)
- Visualize the **loss surface in 3D**
- Add **data normalization** as a preprocessing step
- Implement **learning rate scheduling** (decay over epochs)
- Save trained weights with `numpy.save`

---

## Why Build This From Scratch?

Most ML tutorials call `sklearn.fit()` in one line. This project exists to answer: **what is that line actually doing?**

Understanding gradient descent at this level — the partial derivatives, the convexity, the update rule — is the prerequisite for understanding neural networks, where backpropagation runs the same algorithm across millions of parameters simultaneously. Every deep learning framework (PyTorch, TensorFlow, JAX) is doing exactly this math, just faster and at scale.
