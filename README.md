# CHAMP-PV: Constrained Hybrid Approach for Modeling Photovoltaics

A physics-constrained hybrid neural network framework for predicting daily solar Photovoltaic (PV) Performance Ratio ($\text{PR}$). This project evaluates whether embedding explicit physical algebraic equations into a neural network architecture enables better extrapolation to older, unseen panels compared to purely data-driven models.

---

## 1. Project Overview & Motivation

* **Goal**: Predict daily Performance Ratio ($\text{PR} = \frac{\text{Energy AC}}{\text{Capacity} \times \text{POA Irradiance}}$) and evaluate whether a physics-structured model extrapolates to older panels without physical violations ($d\text{PR}/d\text{age} > 0$).
* **Model Approach**: Combines learnable physical equations with a bounded, zero-initialized residual neural network ($1 + 0.25 \cdot \tanh(\text{NN}(x))$) and an autograd monotonic penalty constraint.
* **Evaluation Strategy**: Out-of-sample evaluation using non-random splits (Split by time: $<2023$ vs. $\ge 2023$; Split by panel age: $<8$ years vs. $\ge 10$ years).

---

## 2. Data Source & Synthetic Nature

* **Source**: The dataset used in this repository is sourced from Kaggle: [Solar Panel Degradation Output Prediction](https://www.kaggle.com/datasets/vinay24baghira/solar-panel-degradation-output-prediction).
* **Dataset Characteristics**: 
  * The dataset contains daily synthetic operational PV metrics (111k rows across 48 distinct systems).
  * Feature variables include system age, POA irradiance ($G$), mean/max ambient temperatures, wind speed, particulate pollution ($\text{PM}_{10}$), humidity, days since rain, and days since cleaning.
  * Around **60% of Performance Ratio variance** exists between systems, making per-system baseline efficiency modeling essential.

---

## 3. Physics Formulation & Architecture

The model expresses the Performance Ratio through an algebraic system of physics relations combined with a neural correction factor:

$$\text{PR} = \eta_{\text{system}} \cdot \left[1 + \gamma (T_{\text{cell}} - 25)\right] \cdot (1 - d_{\text{tech}} \cdot \text{age}) \cdot (1 - \text{soil}) \cdot \text{low}_{\text{light}} \cdot \left(1 + 0.25 \cdot \tanh(\text{NN}(x))\right)$$

### Sub-Component Equations

#### 1. Per-System Baseline Efficiency ($\eta_{\text{system}}$)

$$\eta_{\text{system}} = 0.3 + \text{sigmoid}(\eta_{\text{raw}})$$

Bounds system efficiency between $0.3$ and $1.3$.

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

$$\text{soil} = a_0 \cdot \left(\frac{\text{PM}_{10}}{50}\right) \cdot \min\left(\frac{\text{days-since-rain}}{30}, 1.0\right) + a_1 \cdot \min\left(\frac{\text{days-since-cleaning}}{365}, 1.0\right)$$

Captures particulate pollution accumulation and washout dynamics.

#### 6. Low-Light Attenuation ($\text{low}_{\text{light}}$)

$$\text{low}_{\text{light}} = 1 - 0.5 \cdot \text{sigmoid}(ll_{\text{raw}}) \cdot \exp\left(-\frac{G}{150}\right)$$

Models efficiency loss under low solar irradiance conditions.

#### 7. Bounded Neural Residual Correction

$$\text{residual} = 1 + 0.25 \cdot \tanh(\text{NN}(x))$$

The neural network operates as a small residual multiplier bounded to $\pm 25\%$ via $0.25 \cdot \tanh(\text{NN}(x))$.

---

## 4. How to Run the Code

### Step 1: Clone Repository & Install Dependencies
```bash
git clone [https://github.com/your-username/champ-pv.git](https://github.com/your-username/champ-pv.git)
cd champ-pv
pip install -r requirements.txt