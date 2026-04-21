# Project: Analysis & Prediction for Energy Consumption of Data Centers

## Project Objective
Build a modular, web-based localhost Streamlit application that:
- Loads and cleans raw energy consumption datasets
- Performs exploratory data analysis (EDA) with rich interactive visualizations
- Builds and evaluates three regression models
- Provides two independent prediction interfaces for future energy forecasting

---

## Tech Stack (always use these, never suggest alternatives)
- Language: Python 3
- Data: pandas, numpy
- Visualization (static): matplotlib, seaborn
- Visualization (interactive): plotly
- Web Interface: streamlit (run with: streamlit run app.py)
- ML: scikit-learn
  - Models: LinearRegression, RandomForestRegressor, GradientBoostingRegressor
  - Preprocessing: StandardScaler, OneHotEncoder, ColumnTransformer, LabelEncoder
  - Pipeline: Pipeline
  - Evaluation: mean_absolute_error, mean_squared_error, r2_score
  - Tuning: GridSearchCV or RandomizedSearchCV (5-fold cross-validation)
- Model Saving: joblib (primary), pickle (fallback)
- Memory: streamlit session_state, st.cache_data, st.cache_resource

---

## Project Structure (always use this exactly, never deviate)
```
dataset-project/
├── app.py                     ← Streamlit entry point, navigation/routing
├── data_processing.py         ← raw data ingestion, cleaning, storage
├── preprocessing.py           ← feature engineering, encoding, scaling
├── visualization.py           ← all reusable chart/graph functions
├── model_training.py          ← train, evaluate, compare, save models
├── prediction_models.py       ← load models, run inference, risk logic
├── pages/
│   ├── 1_overview.py          ← Overview dashboard (static graphs, summary cards)
│   ├── 2_insights.py          ← Interactive EDA, trends, "Most Common Sources"
│   └── 3_prediction.py        ← Model 1 (forecast) + Model 2 (new data center)
├── datasets/
│   ├── raw/                   ← original unmodified datasets go here
│   └── processed/             ← cleaned datasets saved here
├── models/                    ← saved .pkl / .joblib model files and scalers
└── graphs/                    ← saved static .png plot images
```

---

## Key Dataset Columns (always treat these as primary features)
- `country`
- `total_data_centers`
- `colocation_data_centers`
- `power_capacity_MW`
- `Average_renewable_energy`
- `Growth_rate_of_data_centers`
- `Climate_Type`

---

## Phase-by-Phase Workflow (follow this order strictly)

### Phase 1 — Data Loading & Cleaning (data_processing.py)
- Load all raw CSV/Excel files from `datasets/raw/`
- On load, print: shape, column names, dtypes, null counts, duplicate counts
- Drop unnamed/index columns automatically (e.g., columns named `Unnamed: 0`)
- Remove duplicate rows
- Impute missing values: numeric → median, categorical → mode
- Fix incorrect data types: parse dates, cast numeric strings to float/int
- Handle outliers using IQR method on numeric columns
- Aggregate data at country level where required
- Save cleaned output to `datasets/processed/cleaned_data.csv`
- Return cleaned DataFrame via a function for use across modules

### Phase 2 — Feature Engineering & Preprocessing (preprocessing.py)
- Engineer new features:
  - `Renewable_Capacity_MW` = power_capacity_MW × (Average_renewable_energy / 100)
  - `DC_Density` = total_data_centers per unit (if population/area available)
- Apply One-Hot Encoding to `Climate_Type`
- Apply Label Encoding to `country` for model input
- Apply StandardScaler to all continuous numeric features
- Return processed feature matrix X and target vector y
- Save fitted scaler and encoder to `models/` using joblib

### Phase 3 — Visualizations (visualization.py)
All graph functions must be reusable and accept a DataFrame as input.
Create the following chart types:
- **Bar Charts:** total data centers by country; power capacity by country
- **Pie Charts:** distribution of climate types; renewable vs non-renewable share
- **Line Graphs:** growth rate trends over time; power capacity trends
- **Histograms:** distribution of power_capacity_MW; renewable energy %
- **Box Plots:** power capacity by climate type; outlier inspection
- **Heatmap:** correlation matrix of all numeric features (seaborn annotated)
- **Correlation Matrix:** interactive version using plotly heatmap
- **Country Comparison:** grouped bar chart comparing key metrics across countries
- **Trend Analysis:** rolling average of data center growth
- **Renewable Energy Distribution:** stacked bar or area chart by country/year
- **"Most Common Sources" Section:** summary bar chart of dominant patterns

Rules:
- Use matplotlib/seaborn for static plots → save as `.png` to `graphs/`
- Use plotly for all interactive charts displayed in Streamlit
- Every plot must have: title, axis labels, legend (where applicable), tight_layout

### Phase 4 — Model Training (model_training.py)
- Load cleaned data from `datasets/processed/cleaned_data.csv`
- Call preprocessing.py to prepare features
- Train all three models: LinearRegression, RandomForestRegressor, GradientBoostingRegressor
- Train/Test split: 80/20, random_state=42
- Use sklearn Pipeline: preprocessing + model
- Print training progress for each model (e.g., "Training Random Forest...")
- Evaluate all three: MAE, MSE, RMSE, R² Score → print and store results
- Auto-select best model by lowest RMSE
- Tune best model using RandomizedSearchCV (5-fold CV)
- Save best model → `models/best_model.pkl`
- Save model metadata (name, metrics) → `models/model_info.json`

### Phase 5 — Prediction Logic (prediction_models.py)

#### Model 1 — Future Energy Consumption Forecast
- Input: selected `country` from dropdown
- Filter historical data for that country
- Generate future year range (e.g., next 5–10 years)
- Extrapolate using best trained regression model
- Output: DataFrame of year-wise predicted values + interactive Plotly line graph
- NO risk indicator in this model

#### Model 2 — New Data Center Energy Prediction
- Input: `country`, `Power Capacity (MW)`, `Renewable Percentage (%)`
- Auto-determine `Climate_Type` from country lookup table
- Run inference using best trained regression model
- Output: predicted energy consumption value
- Calculate Risk Level:
  - **Low Risk:** renewable % ≥ 60 AND power_capacity_MW < 50
  - **Medium Risk:** renewable % between 30–59 OR power_capacity_MW between 50–150
  - **High Risk:** renewable % < 30 OR power_capacity_MW > 150 OR Climate_Type is hot/arid
- Display risk badge with color (green / orange / red)

### Phase 6 — Streamlit Pages

#### pages/1_overview.py — Overview Dashboard
- Display: project title, objective, and key summary cards (total countries, avg power capacity, etc.)
- Load and display all pre-generated static graphs from `graphs/`
- Arrange in clean grid layout (2-column or 3-column st.columns)
- Add short explanatory text below each graph
- Acts as the professional introduction to the system

#### pages/2_insights.py — Insights & EDA
- Display: cleaned dataset summary (shape, dtypes, null counts before/after)
- Show df.describe() in an expandable section
- Display interactive Plotly charts (all chart types from Phase 3)
- Include dynamic filters (country selector, climate type selector)
- Include "Most Common Sources" dedicated section at the bottom
- Show feature correlation heatmap

#### pages/3_prediction.py — Prediction Interface
- Use st.tabs() to separate Model 1 and Model 2
- **Tab 1 — Future Forecast:**
  - Country dropdown → "Generate Forecast" button
  - Show year-wise prediction table + interactive line chart
- **Tab 2 — New Data Center:**
  - Country dropdown, Power Capacity slider, Renewable % slider
  - "Predict Energy Consumption" button
  - Show: predicted value, risk badge, explanation text

### Phase 7 — Memory & Caching
- Use `@st.cache_data` on all data loading and preprocessing functions
- Use `@st.cache_resource` on model loading functions
- Use `st.session_state` to persist: cleaned DataFrame, trained model, predictions
- Graphs must not recompute on page switch

---

## Streamlit Navigation (app.py)
- Use `st.navigation()` or sidebar `st.page_link()` for multi-page routing
- Sidebar must show: project title, brief description, navigation links
- Pages order: Overview → Insights → Prediction
- Keep sidebar clean and minimal

---

## Code Rules (follow always, no exceptions)
- Write clean, modular Python — one responsibility per function
- Every function must have a docstring
- Add inline comments for non-obvious logic
- Use try/except blocks for: file I/O, model training, inference calls
- Use relative paths only — never hardcode absolute paths
- Print progress messages during training
- All cached functions decorated before first call

---

## Output Rules
- Every static plot: saved to `graphs/` as `.png` with descriptive filename
- Every model artifact: saved to `models/` using joblib
- Cleaned data: saved to `datasets/processed/cleaned_data.csv`
- Terminal must show evaluation metrics for all 3 models during training
- Final app must contain: at least 8 chart types across pages
- Prediction page must show both Model 1 and Model 2 in separate tabs

---

## How to Run
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn plotly scikit-learn streamlit joblib

# Run the application
streamlit run app.py
```

---

## requirements.txt (generate this file at end of development)
```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
streamlit
joblib
```
