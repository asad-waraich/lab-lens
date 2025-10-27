# MIMIC-III Medical Data Pipeline

## Project Overview
A comprehensive data pipeline for processing and analyzing MIMIC-III medical discharge summaries, implementing data acquisition, preprocessing, validation, and bias detection for healthcare documentation analysis.

## Team Members
- **Asad Ullah Waraich**: Data Acquisition


## Dataset
**MIMIC-III (Medical Information Mart for Intensive Care)**: De-identified health data from ~60,000 ICU patients at Beth Israel Deaconess Medical Center (2001-2012).

### Data Statistics
- **Discharge Summaries**: 5,000 records (initial processing)
- **Demographics Dataset**: 9,996 records (with patient demographics)
- **Average Text Length**: 9,558 characters per summary
- **Lab Data Coverage**: 99.2% of records have associated lab results

## Project Structure
```
Data-Pipeline/
├── data/
│   ├── raw/                             # Raw data from BigQuery
│   │   ├── mimic_discharge_labs.csv     # 5,000 discharge summaries with labs
│   │   └── mimic_complete_with_demographics.csv  # 9,996 records with demographics
│   └── processed/                       # Processed data
│       ├── processed_discharge_summaries.csv
│       └── preprocessing_report.csv
├── scripts/
│   ├── preprocessing.py                 # Text cleaning and feature extraction
│   ├── validation.py                    # Data quality validation
│   └── bias_detection.py                # Demographic bias analysis
├── logs/
│   ├── validation_report.json           # Data quality metrics
│   ├── validation_summary.csv           
│   ├── bias_report.json                 # Bias analysis results
│   ├── bias_summary.csv
│   └── bias_plots/                      # Visualization outputs
│       ├── text_length_by_gender.png
│       ├── text_length_by_ethnicity.png
│       └── abnormal_labs_by_age.png
├── notebooks/
│   ├── data_acquisition.ipynb           # BigQuery data extraction
│   └── preprocessing.ipynb              # Preprocessing development
├── requirements.txt                     # Python dependencies
└── README.md                            
```

## Setup Instructions

### Prerequisites
- Python 3.12+ (Note: Python 3.13 has compatibility issues with some packages)
- Google Cloud Platform account with BigQuery access
- PhysioNet credentialing for MIMIC-III access

### Installation
```bash
# Clone repository
git clone 
cd Data-Pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux

# Install dependencies
.venv/bin/pip install -r requirements.txt
```

### GCP Authentication
```bash
# Install gcloud CLI (if needed)
brew install --cask google-cloud-sdk  # Mac

# Authenticate
gcloud auth application-default login
gcloud config set project 
```

## Pipeline Components

### 1. Data Acquisition (BigQuery)
**Script**: `notebooks/data_acquisition.ipynb`

Extracts MIMIC-III data from Google BigQuery:
- Discharge summaries from `physionet-data.mimiciii_notes.noteevents`
- Lab results from `physionet-data.mimiciii_clinical.labevents`
- Patient demographics from `physionet-data.mimiciii_clinical.patients`
- Diagnosis codes from `physionet-data.mimiciii_clinical.diagnoses_icd`

**Key Features**:
- Efficient cloud-based querying (no local download of 7GB dataset)
- Automatic de-identification marker removal
- Lab result aggregation with abnormal flags
- Demographic data joining for bias analysis

### 2. Data Preprocessing
**Script**: `scripts/preprocessing.py`

Processes raw discharge summaries:
- **Section Extraction**: Identifies and extracts 9 key medical sections
- **Abbreviation Expansion**: Expands 20+ common medical abbreviations
- **Text Cleaning**: Removes extra whitespace, fixes formatting
- **Feature Engineering**: Creates 26 new features including:
  - Word count, sentence count, average word length
  - Medical term frequency
  - Section completeness scores
  - Urgency indicators
  - Complexity metrics

**Output**: Structured dataset with extracted features ready for ML modeling

### 3. Data Validation
**Script**: `scripts/validation.py`

Comprehensive quality checks:
- **Schema Validation**: Verifies required columns exist
- **Completeness Check**: Identifies missing values (0% missing critical fields)
- **Data Quality**: 
  - Text length validation (100-100,000 characters)
  - Duplicate detection (0 duplicates found)
  - Outlier identification using IQR method
- **Identifier Validation**: Checks patient/admission ID integrity
- **Overall Score**: 95/100 (excellent data quality)

### 4. Bias Detection
**Script**: `scripts/bias_detection.py`

Analyzes potential biases in medical documentation:

**Demographic Analysis**:
- Gender distribution: 55.7% Male, 44.3% Female
- Ethnicity: Analyzed across 5 major groups
- Age: 5 age brackets (18-35, 36-50, 51-65, 66-80, 80+)
- Insurance type: Private, Medicare, Medicaid, Self-pay

**Key Findings**:
- **Overall Bias Score**: 5.88 (low bias - good!)
- **Gender CV**: 2.81% (minimal variation between genders)
- **Documentation Length**: No significant difference by gender (p=0.21)
- **Lab Testing**: Consistent across demographic groups

**Visualizations Generated**:
- Box plots for text length by gender/ethnicity
- Bar charts for abnormal labs by age group

## Pipeline Execution

### Running Individual Components
```bash
# Run preprocessing
.venv/bin/python scripts/preprocessing.py

# Run validation
.venv/bin/python scripts/validation.py

# Run bias detection
.venv/bin/python scripts/bias_detection.py
```

### Complete Pipeline Execution Order
1. Data Acquisition (Jupyter notebook)
2. Preprocessing → Validation → Bias Detection (Python scripts)

## Results Summary

### Data Quality Metrics
- **Validation Score**: 95/100
- **Schema Valid**: ✅ All required columns present
- **Missing Text**: 0 records
- **Duplicate Records**: 0
- **Average Text Length**: 9,558 characters

### Bias Analysis Results
- **Gender Bias**: Not statistically significant (p=0.21)
- **Ethnicity Variation**: 8.4% coefficient of variation
- **Age Group Variation**: 7.2% coefficient of variation
- **Overall Bias Score**: 5.88 (scale 0-100, lower is better)

### Key Achievements
✅ Successfully processed 5,000 discharge summaries in main pipeline
✅ Analyzed 9,996 records with demographics for bias detection  
✅ Extracted structured information from unstructured text  
✅ Validated data quality with 95% score  
✅ Identified minimal demographic bias in documentation  
✅ Created reusable, modular pipeline components 

## Technical Specifications

### Dependencies
- **Data Processing**: pandas, numpy
- **Cloud Services**: google-cloud-bigquery, google-cloud-storage
- **Visualization**: matplotlib, seaborn
- **Statistics**: scipy
- **Authentication**: google-auth, pydata-google-auth

### Performance
- **Query Time**: ~30 seconds for 5,000 records
- **Processing Time**: ~45 seconds for full preprocessing
- **Total Pipeline Runtime**: ~3 minutes end-to-end

### Scalability
- Current: 10,000 records
- Tested up to: 30,000 records
- Maximum available: 59,652 discharge summaries
- Cloud-based architecture supports full dataset processing

## Future Improvements
1. **Airflow Integration**: Automated scheduling and orchestration
2. **DVC Implementation**: Data versioning for reproducibility
3. **Real-time Processing**: Stream processing for new admissions
4. **ML Model Integration**: BioBERT for medical entity extraction
5. **Dashboard Creation**: Interactive bias monitoring dashboard

## Ethical Considerations
- All data is de-identified per HIPAA guidelines
- Bias detection ensures equitable healthcare documentation
- No patient privacy violations possible with cleaned data
- Analysis promotes fairness in medical AI systems

## License
This project uses MIMIC-III data under PhysioNet Credentialed Health Data License 1.5.0

## Contact
For questions about this pipeline, please contact the team lead or refer to the individual component documentation in each script file.

---
*Developed as part of MLOps Course Project - Fall 2025*