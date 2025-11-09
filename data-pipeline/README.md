# Lab Lens - Healthcare Intelligence Pipeline

Multi-modal MLOps pipeline for medical report summarization and bias detection using MIMIC-III clinical data.

---

## Overview

Lab Lens processes clinical discharge summaries through a complete MLOps pipeline:
1. Data acquisition from MIMIC-III via BigQuery
2. Preprocessing with duplicate removal and demographic standardization
3. Data validation with quality scoring
4. Feature engineering (87 advanced features)
5. Bias detection across demographic groups
6. Automated bias mitigation

**Pipeline Duration:** ~1 minute for 7,000+ records  
**Data Quality:** 82% validation score  
**Features:** 87 engineered clinical and demographic features

---

## Prerequisites

### Required Access
1. **PhysioNet MIMIC-III access**
   - Complete CITI training at https://physionet.org/
   - Request MIMIC-III dataset access
   - Link Google account at https://physionet.org/settings/cloud/
   - Wait for approval (1-3 business days)

2. **Google Cloud Platform**
   - GCP account with active project
   - BigQuery API enabled
   - OAuth 2.0 Desktop credentials

### Required Software
- Python 3.9+
- Git
- Jupyter Notebook

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/asad-waraich/lab-lens.git
cd lab-lens
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r data-pipeline/requirements.txt
```

---

## Configuration Setup

### Step 1: Create .env File
Create `.env` in project root:
```bash
MIMIC_PROJECT_ID="your-gcp-project-id"
LOG_LEVEL="INFO"
```

### Step 2: Set Up OAuth Credentials
```bash
mkdir -p credentials
```

Get OAuth credentials from https://console.cloud.google.com/apis/credentials:
- Create Credentials → OAuth 2.0 Client ID → Desktop app
- Download JSON file

Create `credentials/oauth_client.json`:
```json
{
  "installed": {
    "client_id": "your-client-id-here",
    "client_secret": "your-client-secret-here",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "redirect_uris": ["http://localhost"]
  }
}
```

### Step 3: Update .gitignore
```bash
echo -e "credentials/\n.env\ntoken.json\ntoken.pickle" >> .gitignore
```

### Step 4: Configure Pipeline Settings
The pipeline configuration is in `data-pipeline/configs/pipeline_config.json`:

**Key settings (already configured):**
```json
{
  "pipeline_config": {
    "input_path": "data-pipeline/data/raw",
    "output_path": "data-pipeline/data/processed",
    "logs_path": "data-pipeline/logs"
  },
  "validation_config": {
    "text_length_min": 100,
    "text_length_max": 100000,
    "age_min": 0,
    "age_max": 120,
    "validation_score_threshold": 80
  },
  "bias_detection_config": {
    "alert_thresholds": {
      "gender_cv_max": 5.0,
      "ethnicity_cv_max": 10.0,
      "age_cv_max": 8.0,
      "overall_bias_score_max": 10.0
    }
  }
}
```

**No changes needed** - paths are already set correctly.

---

## Pipeline Execution

### Option 1: Run Complete Pipeline (Recommended)
Runs all steps automatically in sequence:
```bash
python data-pipeline/scripts/main_pipeline.py
```

**What it does:**
- Preprocessing → Validation → Feature Engineering → Bias Detection → Bias Mitigation
- Duration: ~1 minute
- Outputs: All reports and processed datasets

### Option 2: Run Individual Steps
For debugging or selective execution:

#### Step 1: Data Acquisition
```bash
# Open Jupyter notebook
jupyter notebook data-pipeline/notebooks/data_acquisition.ipynb

# Update Cell 2 with your OAuth credentials (client_id, client_secret, project_id)
# Run all cells in sequence
# Output: data-pipeline/data/raw/mimic_discharge_labs.csv
```

#### Step 2: Preprocessing
```bash
python data-pipeline/scripts/preprocessing.py
# Output: data-pipeline/data/processed/processed_discharge_summaries.csv
```

#### Step 3: Validation
```bash
python data-pipeline/scripts/validation.py
# Output: data-pipeline/logs/validation_report.json
```

#### Step 4: Feature Engineering
```bash
python data-pipeline/scripts/feature_engineering.py
# Output: data-pipeline/data/processed/mimic_features.csv (87 features)
```

#### Step 5: Bias Detection
```bash
python data-pipeline/scripts/bias_detection.py
# Output: data-pipeline/logs/bias_report.json + visualizations
```

#### Step 6: Bias Mitigation
```bash
python data-pipeline/scripts/automated_bias_handler.py
# Output: data-pipeline/data/processed/mimic_features_mitigated.csv
```

---

## Expected Results

### Data Flow
```
Raw Data: 9,715 records (15 columns)
    ↓ Preprocessing (removes 2,580 duplicates)
Processed: 7,135 records (43 columns)
    ↓ Feature Engineering
Features: 7,135 records (87 columns)
    ↓ Bias Mitigation
Mitigated: 10,665 records (87 columns)
```

### Quality Metrics
- **Validation Score:** 82% (PASS - above 80% threshold)
- **Duplicates Removed:** ~2,580 records
- **Invalid Ages Removed:** All filtered
- **Features Created:** 87 (from 15 original)

### Bias Detection Results
- **Overall Bias Score:** 18.5%
- **Primary Bias:** Age-based (clinical complexity)
- **Mitigation Applied:** Oversampling minority age groups
- **Post-Mitigation:** 18.46% (minimal change - clinically justified)

---

## Output Files

### Data Files
```
data-pipeline/data/raw/
├── mimic_discharge_labs.csv              # Raw data from BigQuery

data-pipeline/data/processed/
├── processed_discharge_summaries.csv     # After preprocessing
├── mimic_features.csv                    # After feature engineering (87 features)
└── mimic_features_mitigated.csv          # After bias mitigation
```

### Reports
```
data-pipeline/logs/
├── validation_report.json                # Data quality metrics
├── validation_summary.csv                # Validation summary table
├── bias_report.json                      # Bias detection results
├── bias_summary.csv                      # Bias summary table
├── bias_mitigation_report.json           # Mitigation results
├── pipeline_results_latest.json          # Complete pipeline summary
└── bias_plots/                           # Visualization PNG files
    ├── text_length_by_gender.png
    ├── text_length_by_ethnicity.png
    ├── treatment_intensity_by_age.png
    └── completeness_by_ethnicity.png
```

---

## Advanced Usage

### Skip Specific Steps
```bash
# Skip preprocessing (if already done)
python data-pipeline/scripts/main_pipeline.py --skip-preprocessing

# Skip validation and bias detection
python data-pipeline/scripts/main_pipeline.py --skip-validation --skip-bias-detection
```

### Custom Paths
```bash
# Feature engineering with custom paths
python data-pipeline/scripts/feature_engineering.py \
  --input path/to/input.csv \
  --output path/to/output.csv
```

### Enable Section Features
```bash
# Add section-level features and negation density
python data-pipeline/scripts/feature_engineering.py --with_sections
```

---

## Troubleshooting

### Error: FileNotFoundError for config
**Solution:** Ensure you're running from project root and `data-pipeline/configs/pipeline_config.json` exists

### Error: 403 Forbidden - Access Denied
**Solution:** 
- Verify Google account linked at https://physionet.org/settings/cloud/
- Ensure OAuth credentials match your GCP project
- Check MIMIC-III access approval status

### Error: Module import errors
**Solution:** Ensure virtual environment is activated and dependencies installed:
```bash
source .venv/bin/activate
pip install -r data-pipeline/requirements.txt
```

### Validation Score Below 80%
**Expected:** Real healthcare data has quality issues. 82% is good for MIMIC-III data.

### High Bias Score (18%)
**Expected:** Age-based bias reflects clinical complexity (older patients = longer notes). This is clinically justified, not a pipeline failure.

---

## Project Structure

```
lab-lens/
├── data-pipeline/
│   ├── configs/
│   │   └── pipeline_config.json          # Main configuration
│   ├── data/
│   │   ├── raw/                           # Raw data from BigQuery
│   │   └── processed/                     # Processed and features
│   ├── logs/                              # Reports and visualizations
│   ├── notebooks/
│   │   └── data_acquisition.ipynb        # Step 1: Data acquisition
│   └── scripts/
│       ├── preprocessing.py              # Step 2: Cleaning and standardization
│       ├── validation.py                 # Step 3: Quality checks
│       ├── feature_engineering.py        # Step 4: Feature creation
│       ├── bias_detection.py             # Step 5: Bias analysis
│       ├── automated_bias_handler.py     # Step 6: Bias mitigation
│       └── main_pipeline.py              # Orchestrator (runs all steps)
├── credentials/                           # OAuth credentials (not in git)
├── .env                                   # Environment variables (not in git)
└── .gitignore                            # Excludes sensitive files
```

---

## Security Notes

**Never commit to GitHub:**
- `credentials/` folder (OAuth secrets)
- `.env` file (project IDs)
- `token.json` or `token.pickle` (authentication tokens)

**Always verify before pushing:**
```bash
git status
# Should NOT show credentials/ or .env
```

---

## Team

- **Asad Ullah Waraich** - Project Lead
- **Shahid Kamal** - Technical Lead
- **Shruthi Kashetty** - Data Science Lead
- **Dhruv Rameshbhai Gajera** - Infrastructure Lead
- **Mihir Harishankar Parab** - Team Member
- **Sri Lakshmi Swetha Jalluri** - Team Member

---

## License

- Project Code: MIT License
- MIMIC-III Data: PhysioNet Credentialed Health Data License 1.5.0
- MedMNIST: Apache License 2.0

---

## Contact

For questions or issues, please open a GitHub issue or contact the team.