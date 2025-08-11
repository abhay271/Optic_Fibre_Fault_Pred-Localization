# 🔍 OTDR Fiber Fault Detection & Localization Dashboard

Interactive Streamlit dashboard for detecting, localizing, and characterizing fiber faults from OTDR traces using pre-trained ML models. Designed with a modern dark theme, professional analysis views, and one-click exports.

## ✨ Highlights
- Auto-loads all required models and scaler from a local `models/` folder
- Multiple data input modes: CSV upload, single-sample form, or auto-generated sample
- Binary fault detection + detailed analysis (class, position, reflectance, loss)
- Interactive Plotly charts, including predicted fault highlight and optional 3D view
- Research-style report with actionable insights and recommended steps
- Quick CSV export of predictions and results

## 📦 Repository Contents
- `otdr_dashboardv2.py` – Streamlit app
- `models/` – Place your pre-trained models here (see below)
- `README.md` – You are here

## 🧠 Required Models (Auto-loaded)
Place the following files in a local `models/` directory (relative to the project root):

| Purpose | Filename | Type |
|---|---|---|
| Feature Scaler | `scaler.pkl` | Pickle (sklearn StandardScaler) |
| Binary Classification | `binary_model.h5` | Keras/TensorFlow |
| Fault Class Detection | `multiclass_model.h5` | Keras/TensorFlow |
| Position Localization | `position_model.pkl` | Pickle |
| Reflectance Analysis | `reflectance_model.pkl` | Pickle |
| Loss Analysis | `loss_model.pkl` | Pickle |

If any file is missing, the app will show a clear sidebar error (it won’t crash). You can still use parts of the pipeline that have their models available.

## ✅ Prerequisites
- Python 3.9–3.11
- Recommended: TensorFlow 2.15+ (for `.h5` model compatibility)

Install the required packages:

```bash
pip install streamlit pandas numpy plotly joblib scikit-learn tensorflow
```

If you prefer a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate   # on macOS/Linux
.venv\Scripts\activate # on Windows
pip install streamlit pandas numpy plotly joblib scikit-learn tensorflow
```

## 🚀 Run the App
Windows PowerShell:

```powershell
cd "C:\Users\abhay\OneDrive\Desktop\IBMfinal"
py -m streamlit run otdr_dashboardv2.py
```

Or generic Python:

```bash
streamlit run otdr_dashboardv2.py
```

## 📥 Data Format (CSV)
Required columns:
- `SNR`
- `P1` … `P30` (30 OTDR trace points)

Optional columns (if available in your dataset):
- `Class` (0–7)
- `Position` (0.00–0.30 normalized)
- `Reflectance`, `Loss` (normalized)

Example (header):

```
SNR,P1,P2,...,P30,Class,Position,Reflectance,Loss
```

## 🖥️ Using the Dashboard
1. Models are auto-loaded from `models/` at startup.
2. Choose an input method:
   - Upload CSV dataset
   - Enter a single sample via form
   - Generate a sample OTDR trace
3. Run Binary Fault Detection.
4. If a fault is detected, run Detailed Analysis to get:
   - Fault class, predicted position, reflectance, loss
   - Interactive trace highlighting the predicted fault location
   - Summary cards and research-style report
5. Export results as CSV via quick-download or the Export tab.

## 📊 Visualizations
- OTDR Trace with Predicted Fault Position (interactive line+markers)
- Residual/Error Distribution (histogram)
- Model Performance (accuracy, precision, recall, F1) as metrics + bar chart
- Optional 3D OTDR trace (position vs value vs time index)

Prediction-to-position mapping:
- Model output for position is normalized (0.00–0.30 in 0.01 steps)
- Mapped as: `pX` where `X = int(prediction * 100)`
  - Example: `0.06 → p6`

## 🧪 Notes on Models & Compatibility
- Keras `.h5` models are loaded with `compile=False`; ensure TensorFlow 2.15+ if possible.
- If you trained with normalization, ensure your `scaler.pkl` is the same one used during training.
- If loading fails, the app will display model-specific error messages in the sidebar.

## 🧭 Project Structure
```
IBMfinal/
├─ otdr_dashboardv2.py
├─ models/
│  ├─ scaler.pkl
│  ├─ binary_model.h5
│  ├─ multiclass_model.h5
│  ├─ position_model.pkl
│  ├─ reflectance_model.pkl
│  └─ loss_model.pkl
└─ README.md
```

## 🆘 Troubleshooting
- Error: “Required model file '…' not found”
  - Ensure the file exists in `models/` with the exact filename.
- TensorFlow/Keras load errors
  - Re-save the model with your current TF/Keras version.
  - Consider exporting in SavedModel format if you have access to training code.
- Poor predictions
  - Verify dataset column names and scales (SNR, P1–P30).
  - Ensure the same scaler used in training is provided (`scaler.pkl`).

## 🤝 Contributing
Issues and PRs are welcome! If you add support for new model formats or better visualizations, please include clear instructions and sample files.

## 📄 License
MIT License (or your preferred license). Update this section accordingly.

---
Built with Streamlit, Plotly, TensorFlow, and scikit-learn. Designed for practical OTDR fault analysis with clean, modern UX.


