# Too Good To Go - Agent-Based Microeconomic Model

A microeconomic agent-based simulation to investigate the effects of food waste reduction services (like **Too Good To Go**) on total production and waste in a bakery setting.

## 📋 Overview

This project models a bakery that can sell leftover products at a discount through a "Too Good To Go" style service. The goal is to understand how such services impact:
- **Total production** of goods
- **Total food waste**

The model uses **evolutionary algorithms** to optimize the baker's production strategy based on a risk-adjusted utility function.

---

## 🎯 Model Description

### Daily Simulation Flow

1. **Production Decision**: Each day, the baker decides:
   - How many units of each good type to produce ($\mathbf{q}_t$)
   - What share of production to reserve for "Too Good To Go" bags ($b_t$)

2. **Consumer Behavior**: Each consumer:
   - Has fixed personal preferences (ranking of goods)
   - Independently decides whether to visit the store (probability $\alpha$)
   - If visiting, attempts to buy their most preferred available good
   - If preferred good is unavailable, may walk out (probability $r$) or try next preference

3. **Too Good To Go Mechanism**: 
   - Bags are reserved **before** observing demand
   - These bags sell for certain but at a lower price ($\tau < \chi$)
   - Once remaining inventory equals reserved bags, the shop closes for regular sales

---

## 🔧 Parameters

### Agent Parameters (Trainable)

| Parameter | Domain | Description |
|-----------|--------|-------------|
| $\mathbf{q}_t$ | $\mathbb{N}^L$ | Vector of quantities produced for each good at epoch $t$ |
| $b_t$ | $[0, 1]$ | Share of production reserved as "Too Good To Go" bags |

### Agent Parameters (Fixed)

| Parameter | Domain | Description |
|-----------|--------|-------------|
| $\gamma$ | $[-1, 2]$ | Risk aversion factor |

### Environment Parameters

| Parameter | Domain | Description |
|-----------|--------|-------------|
| $N$ | $\mathbb{N}$ | Number of consumers |
| $F$ | $\mathbb{R}^{N \times L}$ | Consumer preference matrix (rows = consumers, columns = goods by preference order) |
| $r$ | $[0, 1]$ | Walk-out probability when preferred good is unavailable |
| $\chi$ | $\mathbb{R}^+$ | Production cost per unit (same for all goods) |
| $\rho$ | $\mathbb{R}^+$ | Regular sale price per unit ($\rho > \chi$) |
| $\tau$ | $\mathbb{R}^+$ | Too Good To Go bag price per unit ($\tau < \chi$) |
| $L$ | $\mathbb{N}$ | Number of different goods |
| $\alpha$ | $[0, 1]$ | Daily probability of each consumer visiting the store |

### Evolution Hyperparameters

| Parameter | Domain | Description |
|-----------|--------|-------------|
| $\mu$ | $[0, 1]$ | Mutation probability per parameter |
| $\sigma_q$ | $\mathbb{R}^+$ | Standard deviation for quantity mutation noise |
| $\sigma_b$ | $\mathbb{R}^+$ | Standard deviation for bag share mutation noise |
| $D$ | $\mathbb{N}$ | Number of days per epoch |
| $E$ | $\mathbb{N}$ | Number of generations to run |

---

## 📊 Fitness Function

The agent's fitness is evaluated using a **risk-adjusted reward**:

$$R_\gamma = \pi - \frac{\gamma}{2} \cdot \sigma^2$$

Where:
- $\pi = \rho \sum_{j=1}^{D} \left( s_j + D \|\mathbf{q}\|_1 (\tau b - \chi) \right)$ — total profit
- $s_j$ — units sold on day $j$
- $\sigma^2 = \text{Var}(\pi)$ — variance of profit

The baker's utility increases with profit and decreases with profit variance, scaled by the risk aversion factor $\gamma$.

---

## 🧬 Evolutionary Algorithm

### Selection
- At the end of each epoch, agents are ranked by fitness
- **Top 10%** are selected as parents

### Mutation
New agents are created by cloning parents and applying mutations:

$$\mathbf{q}_{t+1, i} = \mathbf{q}_{t, i} + \epsilon_q \quad \text{where } \epsilon_q \sim \mathcal{N}(0, \sigma_q)$$

$$b_{t+1} = b_t + \epsilon_b \quad \text{where } \epsilon_b \sim \mathcal{N}(0, \sigma_b)$$

Mutations occur with probability $\mu$.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd tgtg

# Install dependencies
pip install -r requirements.txt
```

### Usage
Open and run the Jupyter notebook:
```bash
jupyter notebook tgtg.ipynb
```

---

## 📁 Project Structure

```
tgtg/
├── README.md          # Project documentation
├── prompt.tex         # LaTeX specification of the model
├── tgtg.ipynb         # Main simulation notebook
└── requirements.txt   # Python dependencies (to be added)
```

---

## 🎯 Research Questions

1. How does the availability of a "Too Good To Go" option affect optimal production levels?
2. Does offering discounted leftover bags reduce or increase total food waste?
3. How does consumer walk-out probability influence the effectiveness of the service?
4. What is the relationship between risk aversion and adoption of the TGTG mechanism?

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
