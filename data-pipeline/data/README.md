# Data Directory

## Note
Data files are not included in this repository due to:
- File size constraints (CSV files are 40+ MB each)
- HIPAA compliance and privacy requirements
- PhysioNet licensing restrictions

## Structure
```
data/
├── raw/                 # Original MIMIC data from BigQuery
│   ├── mimic_discharge_labs.csv (5,000 records, ~44 MB)
│   └── mimic_complete_with_demographics.csv (9,996 records, ~80 MB)
└── processed/           # Processed outputs
    ├── processed_discharge_summaries.csv
    └── preprocessing_report.csv
```

## To Reproduce Data
1. Run the data acquisition notebook with BigQuery credentials
2. Execute preprocessing pipeline
3. Data will be generated locally