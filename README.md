# 🏦 Indian State-wise ATM Cash Forecasting Dashboard

A data-driven Python application that predicts and visualizes ATM cash requirements across Indian states using machine learning, synthetic data generation, and interactive charts.

---

## 📌 Features

- ✅ **Predict ATM Cash Requirements** using a pre-trained ML model (`RandomForestRegressor`)
- 📊 **Visualize Predictions** with real-time bar charts, pie charts, and histograms (`matplotlib`)
- 🏙️ Analyze across 10+ Indian states and 3 location types (Urban, Suburban, Rural)
- 💡 Adjustable inputs: ATM count, day of the week, holiday flag, and location type
- 📁 **Export Results to CSV** for further analysis
- 🖱️ User-friendly scrollable interface built with `tkinter`

---

## 🚀 Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core language |
| tkinter | GUI framework |
| matplotlib | Data visualization |
| pandas | Data handling & CSV export |
| NumPy | Synthetic data simulation |
| joblib | Model loading and integration |

---

## 🧠 How It Works

1. **Input Parameters**: Select states, number of ATMs, day of the week, holiday, and location type.
2. **Data Simulation**: ATM features are randomly generated using statistical distributions.
3. **Prediction**: A machine learning model (loaded via `joblib`) predicts cash needs.
4. **Visualization**: Charts update live to reflect total cash, distribution by location, and prediction spread.
5. **Export**: Save results as a `.csv` file for reporting or analysis.

---

## 📸 Screenshots

*(You can add actual screenshots here after running the app)*

---
