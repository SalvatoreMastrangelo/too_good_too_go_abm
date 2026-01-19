# Implementation TODO

A step-by-step guide to implement the Too Good To Go agent-based microeconomic model.

---

## Phase 1: Core Data Structures

### 1.1 Define Environment Configuration Class
Create a dataclass or configuration object to hold all environment parameters:
- `N`: number of consumers
- `L`: number of different goods
- `r`: walk-out probability
- `chi` (χ): production cost per unit
- `rho` (ρ): regular sale price per unit
- `tau` (τ): TGTG bag price per unit
- `alpha` (α): daily store visit probability

Include validation to ensure constraints are met (e.g., ρ > χ > τ > 0).

### 1.2 Define Consumer Class
Create a consumer entity with:
- `preferences`: ordered list of goods (length L), representing personal preference ranking
- Method to decide whether to visit the store on a given day (Bernoulli draw with probability α)
- Method to attempt purchase: iterate through preferences, check availability, and handle walk-out logic

### 1.3 Define Agent (Baker) Class
Create the baker agent with:
- **Trainable parameters**:
  - `q`: numpy array of shape (L,) representing quantities to produce for each good
  - `b`: float in [0,1] representing share reserved for TGTG bags
- **Fixed parameter**:
  - `gamma` (γ): risk aversion factor in [-1, 2]
- Methods for cloning and mutation (to be implemented in Phase 4)

### 1.4 Define Preference Matrix Generator
Implement function to generate the consumer preference matrix F:
- Shape: (N, L) where each row is a permutation of goods [0, 1, ..., L-1]
- Option 1: Random permutation for each consumer (uniform preferences)
- Option 2: Sample from a distribution to create correlated preferences (e.g., some goods are generally more popular)
- Store as environment parameter or generate fresh each simulation run

---

## Phase 2: Daily Simulation Logic

### 2.1 Implement Inventory Management
Create an inventory system for a single day:
- Initialize inventory from production vector `q`
- Calculate `reserved_units = floor(b * sum(q))` for TGTG bags
- Track `available_for_sale = sum(inventory) - reserved_units`
- Method to check if a specific good is available (considering reservation threshold)
- Method to sell a unit (decrement inventory, check if shop must close)

### 2.2 Implement Consumer Purchase Logic
For each consumer who decides to visit:
1. Iterate through their preference list in order
2. For each preferred good:
   - If available AND shop still open: purchase it, return success
   - If unavailable: draw Bernoulli(r) to decide walk-out
     - If walk-out: return failure (no purchase)
     - If stay: continue to next preference
3. If all preferences exhausted without purchase: return failure

### 2.3 Implement Single Day Simulation
Create function `simulate_day(agent, environment, consumers)`:
1. Initialize inventory from agent's production vector `q`
2. Determine which consumers visit (Bernoulli draws with probability α)
3. Shuffle visiting consumers (randomize arrival order)
4. For each visiting consumer:
   - Execute purchase logic
   - Track number of successful sales
5. After all consumers processed:
   - Remaining inventory (minus reserved) becomes waste
   - Reserved units are sold as TGTG bags
6. Return: `sales_count`, `tgtg_sales`, `waste_count`

### 2.4 Implement Daily Profit Calculation
Calculate profit for a single day:
```
daily_profit = (regular_sales * ρ) + (tgtg_sales * τ) - (total_production * χ)
```
Where:
- `regular_sales`: units sold at full price
- `tgtg_sales`: units sold as TGTG bags (= reserved_units, always sold)
- `total_production`: ||q||_1 = sum of all produced units

---

## Phase 3: Epoch Simulation and Fitness

### 3.1 Implement Epoch Simulation
Create function `simulate_epoch(agent, environment, consumers, D)`:
1. Run `D` independent day simulations with the same agent and environment
2. Collect daily profits into an array
3. Note: Each day has different demand realizations due to stochastic consumer behavior
4. Return: array of daily profits, total sales, total waste

### 3.2 Implement Fitness Calculation
Calculate the risk-adjusted reward for an agent after an epoch:
```
π = sum of daily profits
σ² = variance of daily profits
R_γ = π - (γ/2) * σ²
```
This captures the trade-off between expected profit and profit stability, weighted by risk aversion.

### 3.3 Implement Metrics Tracking
Track and store for analysis:
- Total production per epoch: `D * ||q||_1`
- Total waste per epoch: sum of daily waste
- Total regular sales per epoch
- Total TGTG sales per epoch
- Profit mean and variance
- Fitness score

---

## Phase 4: Evolutionary Algorithm

### 4.1 Implement Population Initialization
Create initial population of agents:
- Population size should be configurable (e.g., 100 agents)
- Initialize `q` randomly (e.g., uniform in reasonable range based on N and α)
- Initialize `b` randomly in [0, 1]
- Assign `gamma` values (either fixed for all, or sampled from distribution)

### 4.2 Implement Selection Mechanism
After evaluating all agents:
1. Sort agents by fitness (descending)
2. Select top 10% as parents (elite selection)
3. Store parent indices for reproduction

### 4.3 Implement Mutation Operators
For each trainable parameter:

**Quantity mutation:**
```python
if random() < μ:
    q_new[i] = q[i] + normal(0, σ_q)
    q_new[i] = max(0, round(q_new[i]))  # Ensure non-negative integer
```

**Bag share mutation:**
```python
if random() < μ:
    b_new = b + normal(0, σ_b)
    b_new = clip(b_new, 0, 1)  # Ensure valid range
```

### 4.4 Implement Reproduction
Create new generation:
1. Keep elite parents (optional: elitism to preserve best solutions)
2. For remaining population slots:
   - Randomly select a parent from the elite pool
   - Clone parent's parameters
   - Apply mutation to clone
3. Replace old population with new generation

### 4.5 Implement Evolution Loop
Main training loop:
```python
for generation in range(E):
    for agent in population:
        fitness = simulate_epoch(agent, env, consumers, D)
        agent.fitness = fitness
    
    parents = select_top_10_percent(population)
    population = reproduce(parents, population_size, μ, σ_q, σ_b)
    
    log_generation_stats(generation, population)
```

---

## Phase 5: Experiment Infrastructure

### 5.1 Implement Environment Sampling
Allow environment parameters to be sampled from distributions:
- `N ~ Poisson(λ_N)` or fixed
- `α ~ Beta(a, b)` or fixed
- `r ~ Uniform(0, r_max)` or fixed
- This creates variability across different simulation runs

### 5.2 Implement Logging and Checkpointing
- Log fitness statistics per generation (mean, max, min, std)
- Log best agent's parameters over time
- Save population state periodically for resuming experiments
- Export training curves to CSV/JSON

### 5.3 Implement Visualization
Create plotting functions for:
- Fitness evolution over generations
- Production quantities evolution (per good type)
- TGTG share (b) evolution
- Distribution of final population parameters
- Production vs. waste trade-off curves

### 5.4 Implement Parameter Sweep Framework
Create infrastructure to run multiple experiments:
- Vary γ (risk aversion) across runs
- Vary τ (TGTG price) to study adoption incentives
- Vary r (walk-out probability) to study consumer patience impact
- Collect and compare final outcomes

---

## Phase 6: Analysis and Validation

### 6.1 Implement Baseline Comparisons
Create baseline agents for comparison:
- **No TGTG baseline**: Agent with `b = 0` (no bags reserved)
- **Full TGTG baseline**: Agent with `b = 1` (all production as TGTG)
- **Random agent**: Random production and bag share
- Compare evolved agents against baselines

### 6.2 Implement Statistical Analysis
For research questions:
1. Compare total waste with/without TGTG option
2. Analyze how optimal `b` varies with γ (risk aversion)
3. Study relationship between `r` (walk-out) and optimal strategy
4. Test statistical significance of differences

### 6.3 Implement Sensitivity Analysis
Test model robustness:
- Vary mutation rates and observe convergence
- Test different selection pressures (top 5%, 20%, etc.)
- Vary number of days per epoch (D)
- Check if results are stable across random seeds

### 6.4 Document Results and Insights
- Summarize findings in notebook with clear visualizations
- Document any unexpected emergent behaviors
- Propose extensions or model refinements based on observations

---

## Phase 7: Code Quality and Documentation

### 7.1 Add Type Hints
Add comprehensive type annotations to all functions and classes for better maintainability and IDE support.

### 7.2 Write Unit Tests
Test critical components:
- Consumer purchase logic (edge cases: empty inventory, all goods unavailable)
- Inventory management (reservation thresholds)
- Mutation operators (bounds checking)
- Fitness calculation

### 7.3 Create Requirements File
Document all dependencies:
- numpy (array operations)
- matplotlib (visualization)
- pandas (data handling, optional)
- tqdm (progress bars, optional)
- scipy (statistical tests, optional)

### 7.4 Write Usage Examples
Add example notebooks or scripts showing:
- Basic simulation run
- Full evolutionary training
- Parameter sweep experiment
- Results visualization

---

## Milestones Checklist

- [ ] **M1**: Core classes implemented and tested (Phases 1-2)
- [ ] **M2**: Single epoch simulation working (Phase 3)
- [ ] **M3**: Evolution algorithm complete (Phase 4)
- [ ] **M4**: First successful training run
- [ ] **M5**: Visualization and analysis tools ready (Phases 5-6)
- [ ] **M6**: Full experiment suite completed
- [ ] **M7**: Documentation and code cleanup (Phase 7)
