# Latent Time-to-Collision (TTC) Evaluation with 20-Frame Context & Multi-Step Lookahead

**TEA Lab Technical Summary** | **Kinemamba Project** | _September 2026_

---

## 1. Executive Summary

This report summarizes the experimental evaluation of the **DIAMOND Latent TTC Predictor** across 10 evaluation episodes. The model was evaluated using:

- **Context Length**: **20 ground-truth historical frames** ($2.0\text{ s}$ at $10\text{ Hz}$) to prime the temporal LSTM hidden states ($T = -19 \dots 0$).
- **Lookahead Horizon**: **30 future dream rollout steps** ($+0.1\text{ s}$ to $+3.0\text{ s}$) generated autoregressively by DIAMOND's diffusion world model without new ground-truth observation frames.

### Overall Performance Overview

| Metric                        | Overall Value (All 10 Episodes)       | Active Collision Episodes (Eps 2, 8, 9) | Clear Road Episodes (Eps 1, 3–7, 10) |
| :---------------------------- | :------------------------------------ | :-------------------------------------- | :----------------------------------- |
| **Context Window**            | **20 frames** ($2.0\text{ s}$)        | 20 frames                               | 20 frames                            |
| **Lookahead Horizon**         | **+1 to +30 steps** ($+3.0\text{ s}$) | +1 to +30 steps                         | +1 to +30 steps                      |
| **Overall MAE**               | **$4.90\text{ s}$**                   | **$0.78\text{ s}$**                     | **$6.67\text{ s}$**                  |
| **Overall MSE**               | **$41.32\text{ s}^2$**                | **$1.24\text{ s}^2$**                   | **$58.50\text{ s}^2$**               |
| **Overall RMSE**              | **$6.43\text{ s}$**                   | **$1.11\text{ s}$**                     | **$7.65\text{ s}$**                  |
| **Pearson Correlation ($r$)** | **0.685**                             | **0.716**                               | N/A (Fixed $15.0\text{ s}$ cap)      |

---

## 2. Step-by-Step Lookahead Horizon Breakdown

The lookahead horizon shows three operational regimes:

1. **Ultra-High Precision Window ($+1$ to $+3$ steps / $0.1\text{s} - 0.3\text{s}$)**: $\text{MSE} \le 0.10\text{ s}^2$, $\text{MAE} \le 0.23\text{ s}$. Kinematic momentum is preserved.
2. **Tactical Planning Window ($+4$ to $+6$ steps / $0.4\text{s} - 0.6\text{s}$)**: $\text{MAE} = 0.35\text{ s} - 1.45\text{ s}$. Matches training validation baseline ($\approx 0.77\text{ s}$).
3. **Deep World-Model Dreaming ($+7$ to $+30$ steps / $0.7\text{s} - 3.0\text{s}$)**: Error accumulation from unconditioned diffusion dreaming.

| Lookahead Step | Horizon ($\Delta t$) | Mean MSE ($\text{s}^2$) | $\pm 1$ SEM ($\text{s}^2$) | Mean MAE ($\text{s}$) | Mean RMSE ($\text{s}$) |
| :------------: | :------------------: | :---------------------: | :------------------------: | :-------------------: | :--------------------: |
|     **+1**     |      **+0.1 s**      |       **0.0255**        |        $\pm 0.0115$        |      **0.1289**       |       **0.1596**       |
|     **+2**     |      **+0.2 s**      |       **0.0351**        |        $\pm 0.0184$        |      **0.1315**       |       **0.1873**       |
|     **+3**     |      **+0.3 s**      |       **0.1021**        |        $\pm 0.0520$        |      **0.2336**       |       **0.3196**       |
|     **+4**     |      **+0.4 s**      |       **0.6458**        |        $\pm 0.5926$        |      **0.3564**       |       **0.8036**       |
|     **+5**     |      **+0.5 s**      |       **3.1401**        |        $\pm 2.9059$        |      **0.7402**       |       **1.7720**       |
|     **+6**     |      **+0.6 s**      |       **5.5546**        |        $\pm 3.3014$        |      **1.4476**       |       **2.3568**       |
|     **+8**     |      **+0.8 s**      |       **17.4017**       |        $\pm 5.9320$        |      **3.1525**       |       **4.1715**       |
|    **+10**     |      **+1.0 s**      |       **30.0245**       |        $\pm 5.3093$        |      **5.0762**       |       **5.4795**       |
|    **+15**     |      **+1.5 s**      |       **40.4090**       |        $\pm 8.2253$        |      **5.5653**       |       **6.3568**       |
|    **+20**     |      **+2.0 s**      |       **54.3917**       |       $\pm 11.5736$        |      **6.2892**       |       **7.3751**       |
|    **+30**     |      **+3.0 s**      |       **88.1336**       |       $\pm 19.0118$        |      **8.1393**       |       **9.3880**       |

---

## 3. Dynamic Driving vs. Clear Highway Behavior

- **Active Interaction Episodes (Episodes 2, 8, 9)**:
  - In dynamic traffic, the model anticipates close encounters with sub-second accuracy ($\text{MAE} \approx 0.60\text{ s} - 1.05\text{ s}$, $r > 0.71$).
  - _Example_: In Episode 2, at lookahead step +5, ground truth is $0.20\text{s}$ and prediction is $0.21\text{s}$. At step +27, ground truth is $0.40\text{s}$ and prediction is $0.39\text{s}$.
- **Clear Highway Episodes (Episodes 1, 3–7, 10)**:
  - In open highway scenarios, ground truth remains capped at $15.0\text{s}$.
  - In steps $+1$ to $+5$, predictions are tightly aligned ($\approx 14.9\text{s}$).
  - In deeper rollouts ($>6$ steps), the diffusion model dreams phantom visual clutter ahead, driving TTC predictions down to $2\text{s} - 8\text{s}$ and creating artificial squared error penalties against the fixed $15.0\text{s}$ ceiling.

---

## 4. Cold Start vs. 20-Frame Primed Context

- **Training Validation ($\text{Val MAE} \approx 0.77\text{ s}$)**: Averages all 20 frames from a cold start ($h_0 = 0$), including early ramp-up frames before context is formed.
- **Primed Lookahead Step 1 ($\text{MAE} \approx 0.13\text{ s}$)**: Occurs _after_ 20 full frames of warmup, allowing the saturated LSTM to predict 100ms ahead with minimal error.
- **Code Update**: `scripts/train_latent_ttc.py` has been updated to track and save best checkpoints explicitly using the **20-frame context validation metric** ($\text{targets}_{t=20}$ vs $\widehat{\text{TTC}}_{t=20}$).

---

## 5. Artifacts and Generated Files

- **LaTeX Document**: `ttc_evaluation_report.tex`
- **Compiled PDF Report**: `ttc_evaluation_report.pdf`
- **Primary Plots**:
  - `visualizations/latent_ttc_eval/avg_mse_over_lookahead.png`
  - `visualizations/latent_ttc_eval/latent_ttc_horizon_metrics.png`
  - `visualizations/latent_ttc_eval/episode_1_ttc_trajectory.png` through `episode_10_ttc_trajectory.png`
- **Raw Metric Tables**:
  - `visualizations/latent_ttc_eval/eval_horizon_metrics.csv`
  - `visualizations/latent_ttc_eval/eval_summary.json`
  - `visualizations/latent_ttc_eval/eval_ttc_predictions.csv`
