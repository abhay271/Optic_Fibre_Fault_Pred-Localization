## OTDR Fiber Fault Detection & Localization Dashboard

An interactive Streamlit dashboard for OTDR-based fiber fault detection, classification, localization, and characterization (reflectance and loss). It loads your trained ML models and analyzes data with SNR and 30 OTDR trace points.

---

### Features

- Binary fault detection (Normal vs Fault)
- Multi-class fault classification (0–7)
- Fault position localization (normalized 0–1; optional km view)
- Reflectance and loss estimation
- Robust CSV ingestion with multiple fallback parsers
- Debug mode and OTDR trace visualization
- CSV export of analysis results

---

### Project Structure

```
OTDR_IBM/
  ├─ Optic_Fibre_Fault_Pred-Localization/
  │   └─ otdr_dashboardv2.py
  └─ README.md
```

---

### Requirements and Versions

- Preferred runtime: Python 3.11 for this project
- Key libraries: TensorFlow 2.15.x, NumPy < 2.0, scikit-learn, Streamlit, Pandas, Plotly, Joblib, h5py, protobuf

Important: TensorFlow 2.15.x does not support Python 3.13 yet. If you install Python 3.13 for other work, keep a separate Python 3.11 virtual environment for this app.

---

### Quickstart (Windows, PowerShell)

1. Install Python

- Latest (typically 3.13.x):

```powershell
winget install -e --id Python.Python.3 --source winget --accept-package-agreements --accept-source-agreements --silent
```

- Python 3.11 for this project:

```powershell
winget install -e --id Python.Python.3.11 --source winget --accept-package-agreements --accept-source-agreements --silent
```

2. Create and activate a virtual environment (3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

3. Install dependencies

```powershell
python -m pip install --only-binary=:all: "numpy<2" "tensorflow==2.15.1" "h5py>=3.9,<4" "scikit-learn==1.6.1" "protobuf>=3.20.3,<5" streamlit pandas plotly joblib
```

If your saved `StandardScaler` (or other sklearn model) was created with a different scikit-learn version, pin to that exact version to avoid version warnings.

4. Run the app

```powershell
streamlit run Optic_Fibre_Fault_Pred-Localization/otdr_dashboardv2.py
```

---

### Using the Dashboard

1. Launch the app (above).
2. In the sidebar, upload:
   - Feature scaler (`.pkl` or `.joblib`) used during training (recommended)
   - Models: Binary (required), Class/Position/Reflectance/Loss (optional)
3. Provide data (left column): upload CSV, enter a single sample, or generate a sample.
4. Run analysis (right column): Binary first; if fault is detected, run Detailed Analysis.
5. Export results as CSV.

---

### Data Format (CSV)

- Required columns:
  - `SNR`
  - `P1` … `P30` (30 OTDR trace points)
- Optional columns (for comparison and visuals):
  - `Class` (0–7)
  - `Position` (0–1 normalized)
  - `Reflectance`, `Loss`

The app includes multiple parsing strategies and a simple cleaning flow for problematic rows.

---

### Models

- Supported: `.pkl`, `.joblib` (scikit-learn), `.h5` (Keras/TensorFlow)
- Binary model input: 31 features (1 `SNR` + 30 `P` points) or 30 features (only `P` points)
- Keras `.h5` loader includes compatibility fallbacks for older models (e.g., `InputLayer` `batch_shape`).

Tip: For best compatibility, re-save Keras models with your current TensorFlow/Keras and `compile=False` for inference.

---

### Troubleshooting

- Excess TensorFlow logs: The app suppresses TF INFO/WARNING logs and disables oneDNN notices.
- scikit-learn version warning: Install the exact sklearn version used to pickle your scaler/model.
- `.h5` model fails to load: The app auto-applies fixes; if it still fails, re-save with current TF/Keras or export weights.
- CPU-only note: Windows wheel is CPU-optimized. GPU requires a CUDA/CuDNN-compatible TF; this project assumes CPU.
- Python 3.13: Use a separate 3.11 venv for this project until TF supports 3.13.

---

### Useful Commands

Check Python installations:

```powershell
py -0p
py -3.11 --version
py -3.13 --version
```

Create venvs:

```powershell
py -3.11 -m venv .venv
py -3.13 -m venv .venv313
```

Activate on PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Deactivate:

```powershell
deactivate
```

---

### License

No license provided. Add one if you plan to distribute.

---

### Acknowledgements

Built with Streamlit, TensorFlow/Keras, scikit-learn, Pandas, Plotly, and Joblib.
