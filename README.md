# Linear Regression from Scratch using Gradient Descent

> A pure Python + NumPy implementation of linear regression — no sklearn, no autograd, no black boxes.  
> Built to deeply understand the math behind how machine learning models actually learn.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [The Math — From Scratch](#the-math--from-scratch)
   - [The Model](#the-model)
   - [The Cost Function — MSE](#the-cost-function--mse)
   - [Why Partial Derivatives?](#why-partial-derivatives)
   - [Deriving the Partial Derivatives](#deriving-the-partial-derivatives)
   - [Why We Subtract While Updating](#why-we-subtract-while-updating)
   - [Why the Gradient Always Points Uphill](#why-the-gradient-always-points-uphill)
   - [Why MSE Always Has One Minimum](#why-mse-always-has-one-minimum)
5. [ML Concepts Explained](#ml-concepts-explained)
   - [What is an Epoch?](#what-is-an-epoch)
   - [What is a Learning Rate?](#what-is-a-learning-rate)
   - [Bias and Weight](#bias-and-weight)
   - [Types of Gradient Descent](#types-of-gradient-descent)
6. [Code Walkthrough](#code-walkthrough)
7. [Output and Visualization](#output-and-visualization)

---

## What This Project Does

Trains a linear regression model using **gradient descent** — the same core algorithm that powers neural networks — implemented entirely from scratch using only NumPy.

Given a dataset of `(x, y)` pairs, the model learns the best-fit line:

```
y = w·x + b
```

by starting with `w = 0` and `b = 0` and iteratively improving them over many epochs until the error is minimized.

---

## Project Structure

```
├── gradient_descent.py   # Full implementation
├── data.csv              # Dataset (x, y pairs)
├── README.md             # This file
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
A matplotlib window will appear showing the **Error vs Epoch** graph.  
Close the window to let the program finish printing the final values.

> **Note:** `plt.show()` is blocking — the program pauses until you close the plot window. This is normal matplotlib behavior.

---

## The Math — From Scratch

### The Model

The model fits a straight line through the data:

```
y = w·x + b
```

| Symbol | Name | Role |
|--------|------|------|
| `x` | Input feature | The data point we give the model |
| `y` | Target output | The true value we want to predict |
| `w` | Weight (slope) | How much x influences y |
| `b` | Bias (intercept) | Shifts the line up or down |

At the start, both `w` and `b` are 0, meaning the model predicts `y = 0` for everything — a flat line on the x-axis. Gradient descent fixes this.

---

### The Cost Function — MSE

We need to measure how wrong the model is. We use **Mean Squared Error (MSE)**:

```
E(b, w) = (1/N) · Σ (yᵢ - (w·xᵢ + b))²
```

- `yᵢ` is the true value
- `(w·xᵢ + b)` is our prediction
- `(yᵢ - prediction)` is the error for one point
- Squaring it makes all errors positive and penalizes large errors more
- We average over all N points

**The goal:** find values of `w` and `b` that make `E(b, w)` as small as possible.

---

### Why Partial Derivatives?

The error function `E(b, w)` depends on **two variables** — `b` and `w`. Imagine it as a 3D bowl-shaped surface:

```
        Error (height)
          │     * * *
          │   *       *
          │  *         *   ← bowl shape
          │   *       *
          │     * * *
          └──────────────── b and w (ground plane)
                 ↑
           bottom of bowl = minimum error = best b and w
```

We start somewhere on the rim of this bowl (b=0, w=0) and need to slide down to the bottom.

To do that, we need to know: **which direction is downhill?**

- A regular derivative works for 1 variable
- We have 2 variables (`b` and `w`), so we use **partial derivatives** — one for each

**∂E/∂b** answers: *"If I nudge only `b`, how does error change?"*  
**∂E/∂w** answers: *"If I nudge only `w`, how does error change?"*

Together they form the **gradient** — a vector pointing in the direction of steepest increase of error (uphill). We move in the **opposite direction** to reduce error.

---

### Deriving the Partial Derivatives

#### Partial derivative with respect to `b`

Start with the MSE:
```
E = (1/N) · Σ (yᵢ - (w·xᵢ + b))²
```

Let `u = yᵢ - (w·xᵢ + b)`, so `E = (1/N) · Σ u²`

Apply the **chain rule**: `d(u²)/db = 2u · (du/db)`

`du/db = -1` (since `b` appears as `-b` inside `u`)

So:
```
∂E/∂b = (1/N) · Σ 2(yᵢ - (w·xᵢ + b)) · (-1)
       = -(2/N) · Σ (yᵢ - (w·xᵢ + b))
```

This is exactly what the code computes:
```python
b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
```

#### Partial derivative with respect to `w`

Same process, but `du/dw = -x` (since `w` appears as `-w·x` inside `u`)

```
∂E/∂w = (1/N) · Σ 2(yᵢ - (w·xᵢ + b)) · (-xᵢ)
       = -(2/N) · Σ xᵢ · (yᵢ - (w·xᵢ + b))
```

This is exactly what the code computes:
```python
w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
```

The extra `x` in the weight gradient comes directly from the chain rule — because `w` is multiplied by `x` in our model.

---

### Why We Subtract While Updating

The update rule is:
```
b := b - α · (∂E/∂b)
w := w - α · (∂E/∂w)
```

The gradient points **uphill** (toward increasing error). We want to go **downhill**. So we move in the **opposite direction** by subtracting:

| Situation | Gradient sign | Effect of subtracting |
|-----------|--------------|----------------------|
| Error increases as b increases | ∂E/∂b is **positive** | `b - positive` → b gets smaller ✅ |
| Error decreases as b increases | ∂E/∂b is **negative** | `b - negative` → b gets larger ✅ |

Subtracting always moves us toward lower error, regardless of which side of the bowl we're on.

**Yes, the initial values are 0 — but they don't stay 0.** Each iteration, the gradients are non-zero (because the error is non-zero), so the update moves `b` and `w` away from 0 toward better values. Over 1000 epochs, they accumulate into the correct answer.

---

### Why the Gradient Always Points Uphill

This is not an assumption — it comes directly from the definition of a derivative.

The derivative is defined as:
```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

This is simply **rise over run** — the slope of the function at a point.

- If `f'(x) > 0` → the function is rising in the +x direction → that direction is **uphill**
- If `f'(x) < 0` → the function is falling in the +x direction → that direction is **downhill**

The gradient in multiple dimensions is just this idea extended. It is mathematically defined as the direction of **steepest ascent**. This is not a rule someone invented — it falls out of calculus automatically.

Quick 1D proof with `f(x) = x²`:

| Position | Derivative | Meaning |
|----------|-----------|---------|
| x = 3 (right of minimum) | f'(3) = +6 | Function rising to the right → gradient points right (uphill) |
| x = -3 (left of minimum) | f'(-3) = -6 | Function falling to the right → gradient points left (uphill) |
| x = 0 (the minimum) | f'(0) = 0 | At the bottom → no uphill direction |

Subtracting the gradient always moves you toward x = 0, the minimum.

---

### Why MSE Always Has One Minimum

MSE for linear regression is a **convex function** — it has exactly one bowl shape with one global minimum.

```
MSE — Convex ✅             Non-convex (Neural Nets) ❌
                            
     │   U shape               │  bumpy surface
     │  *       *              │ *  * *   * *
     │ *         *             │*    *     *  *
     │─────────────            │─────────────────
       one bottom                multiple valleys
```

Because MSE is convex, the gradient **always points away from the single true minimum**. Subtracting it always moves you closer to the answer. This is why gradient descent is guaranteed to work for linear regression.

In neural networks with non-convex loss surfaces, gradient descent can get stuck in local minima — which is why training deep networks is much harder.

---

## ML Concepts Explained

### What is an Epoch?

```python
epochs = 1000
```

One **epoch** = one full pass over the entire dataset.

In this project, since all data is loaded at once and used in every iteration of `gradient_descent_runner`, **one iteration = one epoch**.

The model does 1000 epochs = 1000 full passes over the data, improving `b` and `w` slightly each time.

---

### What is a Learning Rate?

```python
learning_rate = 0.0001
```

The learning rate `α` controls **how big each step is** when updating `b` and `w`.

| Learning Rate | Effect |
|--------------|--------|
| Too high (e.g. 0.1) | Steps are too large → overshoots the minimum → error explodes |
| Too low (e.g. 0.0000001) | Steps are tiny → takes forever to converge |
| Just right (e.g. 0.0001) | Steadily slides down to the minimum |

`0.0001` is a conservative, safe choice for this problem.

---

### Bias and Weight

**Weight (`w`)** — the slope of the line. Controls how much the input `x` influences the output `y`.  
**Bias (`b`)** — shifts the entire line up or down. Without it, the line is forced through the origin (0, 0), limiting what the model can fit.

Both start at 0 and are learned from data through gradient descent.

---

### Types of Gradient Descent

This project uses **Batch Gradient Descent** — every weight update uses all N data points at once.

| Type | Data used per update | Pros | Cons |
|------|---------------------|------|------|
| **Batch GD** ← this project | All N points | Stable, accurate gradients | Slow for large datasets |
| **Stochastic GD (SGD)** | 1 random point | Very fast | Noisy updates, unstable |
| **Mini-batch GD** | Small batch (e.g. 32) | Balance of both | Most common in deep learning |

---

## Code Walkthrough

```
compute_error_for_line_given_points(b, w, points)
  └── Computes MSE over all data points
  └── Used to track error before and after training

step_gradient(b_current, w_current, points, learning_rate)
  └── Core of gradient descent
  └── Computes ∂E/∂b and ∂E/∂w for every point
  └── Updates and returns new b and w

gradient_descent_runner(points, starting_b, starting_w, learning_rate, epochs)
  └── Runs step_gradient for `epochs` iterations
  └── Logs error each epoch into error_history
  └── Plots Error vs Epoch graph using matplotlib
  └── Returns final [b, w]

run()
  └── Loads data.csv
  └── Sets hyperparameters (learning_rate, epochs)
  └── Calls gradient_descent_runner
  └── Prints starting and ending error
```

---

## Output and Visualization

After training, a **matplotlib graph** appears showing MSE error dropping over epochs:

```
MSE
 │ *
 │  *
 │   **
 │     ****
 │          **********─────
 └───────────────────────── Epoch
   0                    1000
```

The sharp drop at the start shows the model learning quickly, then flattening as it approaches the minimum.

> **Important:** The plot window is blocking. The program pauses until you close it, then prints the final `b` and `w` values. This is normal `plt.show()` behavior. Use `plt.show(block=False)` if you want the program to continue without waiting.

---

## Why Build This From Scratch?

Most ML tutorials use `sklearn.fit()` in one line. This project exists to answer: **what is that one line actually doing?**

Understanding gradient descent at this level — the partial derivatives, the convexity, the update rule — is essential before moving to neural networks, where the same algorithm (backpropagation) runs on millions of parameters simultaneously. Every deep learning framework (PyTorch, TensorFlow) is doing exactly this math, just faster and at scale.
