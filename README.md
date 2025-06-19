# Machine-learning-and-programming

# 🔋 LSTM-Based Energy Consumption Prediction for Machine Tools

This project is a time-series deep learning model using an LSTM (Long Short-Term Memory) architecture to predict power consumption (`pwr`) based on sensor data from machine tools.

## 📁 Project Folder Structure
```
project_root/
├── data/
│   ├── train_file_1.csv
│   ├── train_file_2.csv
│   ├── train_file_3.csv
│   └── test_file.csv
│
├── results/
│   ├── models/
│   ├── scalers/
│   └── predictions/
│
├── lstm_train_predict.py
├── README.md
└── requirements.txt
```

---

### ⏳ Time Series Model Structure

- This LSTM model is based on time series prediction, configured with `SEQ_LEN=20` and `LAG=10`.
- `SEQ_LEN=20` means the input sequence consists of 20 consecutive time steps of feature data.
- `LAG=10` means the target prediction point is 10 time steps ahead of the input sequence.

- This LAG setting is not a simple delay, but reflects the **actual delay in sensor data acquisition from machine tools**,  
  helping the model learn the time lag between the moment power is consumed and when it is recorded.

- Consequently, an input sequence of `[t ~ t+19]` is used to predict power at time `t+30`.

### ⚠️ Idle Power Handling

- While the model includes idle (non-cutting) sections in its prediction,  
  it does **not directly use idle power as a target**, but uses a **pre-adjusted value**.

- In reality, idle power consumption exists — approximately **0.4 kW for KITECH machine tools**.

- Thus, power values in non-cutting sections are **not zero**, but constant around 0.4 kW.  
  Including them would distort error calculations, so **idle power (0.4 kW) was subtracted**, and only net cutting power is used as the prediction target.

- A binary feature `is_cut_active` was added to help the model distinguish clearly between active and idle cutting phases.

---

## ⚙️ How to Run

### 1. Set Up the Environment

- Install Python 3.8 (Anaconda environment recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Code Explanation

- Modify path settings at the top of the code to match your local environment.
- Set training parameters:
  1. input features
  2. target
  3. sequence length (`SEQ_LEN`)
  4. time delay (`LAG`)
  5. number of seeds (`NUM_SEEDS`)
  6. group size (`GROUP_SIZE`)

### 3. Code Flow Summary

1. **Load Data**
   - Load three training CSV files and one test CSV file.

2. **Feature Engineering & Preprocessing**
   - Generate derived variables such as `force_norm`, `feed_norm`, and `energy_proxy` from cutting force vectors (clx/clz).
   - Remove outliers using the IQR method to improve training stability.

3. **Scaling**
   - Normalize features and targets to a 0–1 range using `MinMaxScaler`.
   - Save the scaler for inverse transformation after prediction.

4. **Sequence Generation**
   - Convert time-series data into sequences (`SEQ_LEN`) using a sliding window.
   - Set the target as the value `LAG` steps after each sequence.

5. **Model Training (Repetitions)**
   - Train models with different seeds (1–10).
   - Architecture: 2-layer LSTM + Dense, with EarlyStopping.
   - Save models in `.h5` format.

6. **Prediction & Save Results**
   - Predict using trained models.
   - Inversely transform predictions to original scale.
   - Save predictions and actual values to `.csv`.

7. **Visualization**
   - Plot predicted vs actual values.
   - Two line plots, each showing predictions from 5 seed models.

### 4. Model Selection

Due to the stochastic nature of LSTM training, even with identical parameters, results may vary.  
Model selection is done either visually or using evaluation metrics.

---

## 📈 Prediction Summary

- Trained on time-series data from machining processes (slotting, drilling, face milling) on **SM45C** and **AL6061**.
- Target: real-time power consumption during cutting (unit: kW).

- Average prediction error is **below 0.3 kW**.
- The model accurately captures **power peaks**, unlike previous RNN models.

- Key improvements were achieved by adding:
  - `is_cut_active`
  - `energy_proxy`
  - cutting ratio

- Total energy accuracy may be lower, but **pattern tracking (R²)** is visually reliable.

- Visualization:
  - Black line = actual power
  - Colored lines = predicted values (per seed)
  - Two grouped comparison plots provided

- Prediction targets exclude idle power (~0.4 kW), using **only net cutting power**.

### 🔢 Model Performance Summary (By Tool)

| Tool         | RMSE (kW) | MAE (kW) | R²     |
|--------------|-----------|----------|--------|
| Facemill     | 0.1941    | 0.1325   | 0.7039 |
| 10pi Flat    | 0.9721    | 0.2331   | -1.0278 |
| 16pi Flat    | 0.3098    | 0.1456   | 0.3619 |
| 10pi Drill   | 0.4519    | 0.2058   | 0.6558 |
| 16pi Drill   | 0.4922    | 0.2677   | 0.8410 |

- Some tools (e.g., 10pi Flat) showed negative R², indicating predictions worse than a mean estimate.  
  These require feature engineering or model tuning.

- The 16pi Drill achieved the highest R² (0.84), accurately reproducing the actual power curve.

> ⚠️ Note: R² < 0 means the prediction is worse than the mean. Such cases require further model improvement.
