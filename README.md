# Physics-Constrained Hybrid Model for PV Performance Ratio Extrapolation

A hybrid deep learning and physics-constrained modeling framework designed to predict daily Photovoltaic (PV) Performance Ratio ($PR$) and reliably extrapolate degradation across aging solar panel infrastructure.

## Project Overview

Standard data-driven models (e.g., Gradient Boosted Trees, MLPs) often fail to extrapolate when predicting performance decay for solar panels older than those seen in the training distribution. This repository implements a **Physics-Constrained Hybrid Neural Network (PCHNN)** that combines an algebraic physics backbone with a bounded neural network residual correction and an autograd monotonicity penalty ($\frac{dPR}{d\text{age}} \le 0$).

### Key Equation

$$\text{PR} = \eta_{\text{system}} \cdot \left[1 + \gamma (T_{\text{cell}} - 25)\right] \cdot (1 - d_{\text{tech}} \cdot \text{age}) \cdot (1 - \text{soiling}) \cdot \text{low\_light} \cdot \left(1 + 0.25 \cdot \tanh(\text{NN}(x))\right)$$

Where physical parameters ($\eta_{\text{system}}$, $d_{\text{tech}}$, $\gamma$, $T_{\text{cell}}$, soiling, low-light attenuation) are learnable parameters bounded by domain physics.

## Experimental Results

Evaluated across two non-random split protocols (systems retained across train/test splits):
1. **Temporal Split:** Train on data $< 2023$, Test on data $\ge 2023$.
2. **Age Extrapolation Split:** Train on systems $< 8$ years old, Test on systems $\ge 10$ years old.

| Split Experiment | Model | $R^2$ | MAE | Monotonicity Violations |
| :--- | :--- | :--- | :--- | :--- |
| **Time Split** | HistGradientBoosting | $0.9157 \pm 0.0016$ | $0.0173 \pm 0.0002$ | N/A |
| | MLP (Unconstrained) | $0.8876 \pm 0.0043$ | $0.0200 \pm 0.0004$ | $11.9\% \pm 1.6\%$ |
| | Physics Only | $0.8641 \pm 0.0016$ | $0.0219 \pm 0.0001$ | $0.0\% \pm 0.0\%$ |
| | **Hybrid ($\lambda_{\text{mono}}=1$)** | **$0.9416 \pm 0.0016$** | **$0.0144 \pm 0.0002$** | **$0.4\% \pm 0.2\%$** |
| **Age Split** | HistGradientBoosting | $0.7964 \pm 0.0101$ | $0.0271 \pm 0.0010$ | N/A |
| | MLP (Unconstrained) | $0.7159 \pm 0.1508$ | $0.0311 \pm 0.0075$ | $18.7\% \pm 22.7\%$ |
| | Physics Only | $0.8220 \pm 0.0020$ | $0.0252 \pm 0.0002$ | $0.0\% \pm 0.0\%$ |
| | **Hybrid ($\lambda_{\text{mono}}=1$)** | **$0.9042 \pm 0.0037$** | **$0.0185 \pm 0.0004$** | **$0.2\% \pm 0.2\%$** |

## Key Findings

- **Extrapolation Superiority:** On older panel extrapolation ($\ge 10$ years), the Hybrid model achieves $R^2 = 0.9042$, significantly outperforming HistGradientBoosting ($R^2 = 0.7964$) and standard MLPs ($R^2 = 0.7159$).
- **Monotonicity Enforcement:** Applying an autograd loss penalty for non-physical aging ($\frac{dPR}{d\text{age}} > 0$) reduces physical violations from $\sim 18.7\%$ down to $0.2\%$ without sacrificing predictive accuracy.
- **Anomaly Detection:** Residual analysis ($[PR_{\text{pred}} - PR_{\text{obs}}] / PR_{\text{pred}}$) effectively flags full or partial string/inverter outages, reaching a ROC-AUC $> 0.99$.

## Installation

```bash
git clone [https://github.com/your-username/pv-performance-hybrid.git](https://github.com/your-username/pv-performance-hybrid.git)
cd pv-performance-hybrid
pip install -r requirements.txt