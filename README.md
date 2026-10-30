# Physics-Constrained Hybrid Model for PV Performance Ratio

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a **Physics-Constrained Hybrid Neural Network (PCHNN)** designed to predict daily Photovoltaic (PV) Performance Ratio ($\text{PR} = \text{Energy} / [\text{Capacity} \times \text{POA Irradiation}]$). The project tests a central hypothesis: **Does incorporating explicit algebraic solar physics into a neural network enable robust extrapolation to older panels where purely data-driven machine learning models fail?**

---

## 🏗️ 1. Mathematical Formulation & Physics Backbone

Rather than treating PR prediction as an unconstrained black-box regression task, the model embeds domain-expert algebraic relationships for solar performance, cell thermal dynamics, soiling, and spectral degradation:

$$\text{PR} = \eta_{\text{system}} \cdot \left[1 + \gamma \cdot (T_{\text{cell}} - 25)\right] \cdot (1 - d_{\text{tech}} \cdot \text{age}) \cdot (1 - \text{soiling}) \cdot \text{low}_{\text{light}} \cdot \left[1 + 0.25 \cdot \tanh(\text{NN}(x))\right]$$

### Sub-Component Equations

#### 1. Per-System Baseline Efficiency ($\eta_{\text{system}}$)

$$\eta_{\text{system}} = 0.3 + \text{sigmoid}(\eta_{\text{raw}})$$

Bounds system efficiency between $0.3$ and $1.3$. Around 60% of Performance Ratio (PR) variance sits between systems, making this per-system parameter essential.

#### 2. Per-Technology Degradation Rate ($d_{\text{tech}}$)

$$d_{\text{tech}} = 0.03 \cdot \text{sigmoid}(d_{\text{raw}})$$

Enforces degradation bounds between $0\%$ and $3\%/\text{year}$ per panel technology.

#### 3. Cell Temperature Model ($T_{\text{cell}}$)

$$T_{\text{cell}} = T_{\text{mean}} + \beta \cdot (T_{\text{max}} - T_{\text{mean}}) + k \cdot G$$

* $\beta = 1.5 \cdot \text{sigmoid}(\beta_{\text{raw}})$ blends mean and maximum ambient temperatures.
* $k = 0.2 \cdot \text{sigmoid}(k_{\text{raw}})$ models irradiance-driven heating ($\sim 0.04 \ ^\circ\text{C}$ per $\text{W/m}^2$).

#### 4. Temperature Correction Factor ($\text{temp}$)

$$\text{temp} = 1 + \text{softplus}(\gamma_{\text{raw}}) \cdot \gamma_0 \cdot (T_{\text{cell}} - 25.0)$$

Scales the temperature coefficient relative to Standard Test Conditions ($25 \ ^\circ\text{C}$).

#### 5. Soiling Accumulation Model ($\text{soil}$)

$$\text{soil} = a_0 \cdot \left(\frac{\text{PM}_{10}}{50}\right) \cdot \min\left(\frac{\text{days}_{\text{since\_rain}}}{30}, 1.0\right) + a_1 \cdot \min\left(\frac{\text{days}_{\text{since\_cleaning}}}{365}, 1.0\right)$$

Captures particulate pollution accumulation and washout dynamics.

#### 6. Low-Light Attenuation ($\text{low}_{\text{light}}$)

$$\text{low}_{\text{light}} = 1 - 0.5 \cdot \text{sigmoid}(ll_{\text{raw}}) \cdot \exp\left(-\frac{G}{150}\right)$$

Models efficiency loss under low solar irradiance conditions.

#### 7. Bounded Neural Residual Correction

$$\text{residual} = 1 + 0.25 \cdot \tanh(\text{NN}(x))$$

The neural network operates as a small residual multiplier bounded to $\pm 25\%$ via $0.25 \cdot \tanh(\text{NN}(x))$, ensuring it refines physics predictions without overriding foundational physical constraints.

---

## 🛡️ 2. Monotonicity Regularization

Pure machine learning models often learn spurious correlations, such as predicting that older solar panels perform *better* over time due to data artifacts. 
- **Soft Constraint Penalty:** During training, autograd computes the partial derivative of predictions with respect to system age ($\frac{d\text{PR}}{d\text{age}}$). 
- A penalty loss term $\lambda_{\text{mono}} \cdot \text{ReLU}\left(\frac{d\text{PR}}{d\text{age}}\right)$ is added to the MSE loss.
- Setting $\lambda_{\text{mono}} = 1.0$ suppresses physical monotonicity violations from $\approx 12\% - 18\%$ down to **$< 0.4\%$**.

---

## 📂 3. Repository Structure

```text
.
├── README.md               # Detailed project documentation
├── requirements.txt        # Python package dependencies
├── main.py                 # Main CLI pipeline entrypoint
├── src/
│   ├── __init__.py         # Package module exports
│   ├── data.py             # Data loading, cleaning, feature encoding, & splits
│   ├── models.py           # PyTorch PCHNN, MLP baseline, and normalization modules
│   ├── training.py         # Training loops with autograd penalties & evaluation metrics
│   └── visualization.py    # Publication-quality figure generation functions
└── output/                 # Generated figures and evaluation plots (.png)
```

---

## 🚀 4. How to Run the Code

### Step 1: Environment Setup
Ensure Python 3.8+ is installed. Clone the repository and install all required dependencies:

```bash
git clone https://github.com/your-username/solar-pv-hybrid-model.git
cd solar-pv-hybrid-model
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
Ensure your dataset file (`solar_panel_degradation.csv`) is placed inside a `data/` directory relative to `main.py`:

```text
data/solar_panel_degradation.csv
```

### Step 3: Run the Complete Training & Evaluation Pipeline
Execute the main entrypoint script from the root directory:

```bash
# Standard run using default settings
python main.py
```

#### Advanced Execution Examples:

```bash
# Run with customized data path and output folder
python main.py --data_path data/solar_panel_degradation.csv --output_dir my_figures

# Run with a single random seed for quick testing
python main.py --seeds 42

# Display nested training epoch progress bars using tqdm
python main.py --show_epoch_progress
```

### CLI Command Line Arguments:
- `--data_path`: Path to the raw solar dataset CSV file (default: `data/solar_panel_degradation.csv`).
- `--output_dir`: Directory where all generated diagnostic plots will be saved (default: `output`).
- `--seeds`: Space-separated list of integer random seeds for cross-seed evaluation (default: `0 1 2 3 4`).
- `--show_epoch_progress`: Flag to enable epoch-level nested progress bars during training.

---

## 📊 5. Empirical Results & Visualizations

The pipeline automatically evaluates models across strict train/test splits (never random rows) and generates 5 publication-quality diagnostic plots inside the specified `--output_dir` (e.g., `output/`):

1. **Age Extrapolation (`vis_1_age_extrapolation.png`)**  
   - *Finding:* Physics-constrained models successfully extrapolate across the 8–10 year missing gap to older panels (10–18 years). Data-driven baselines (`HistGradientBoosting`, `MLP`) drift and diverge outside their training envelope.
2. **Monotonicity Violation Rates (`vis_2_monotonicity_violations.png`)**  
   - *Finding:* Unconstrained MLPs exhibit 11.9%–18.7% violation rates ($\frac{d\text{PR}}{d\text{age}} > 0$). Applying the autograd penalty ($\lambda_{\text{mono}}=1$) drops violations to $<0.4\%$.
3. **Residual Distribution (`vis_3_residual_distribution.png`)**  
   - *Finding:* Confirms that the neural network correction factor remains tightly bound between $\pm25\%$, verifying that the physics backbone carries the primary predictive load.
4. **Outage Detection ROC & PR Curves (`vis_4_anomaly_detection_roc_pr.png`)**  
   - *Finding:* Evaluates model prediction residuals on operational outage days, achieving an **ROC-AUC of 1.0000** and **PR-AUCs $>0.997$** across all model variants.
5. **Learned Technology Degradation (`vis_5_learned_degradation_rates.png`)**  
   - *Finding:* Showcases effective annual degradation rates learned by the physics module across panel technologies (CdTe, TOPCon, mono-PERC, mono-Si, and poly-Si) against literature expectations ($0.3\% - 0.7\%$/year).

---

## 📄 6. License

Distributed under the MIT License. See `LICENSE` for more information.