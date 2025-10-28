# Lab Lens: AI-Powered Healthcare Intelligence Platform

Multi-Modal Healthcare Intelligence Platform for Medical Report Summarization and Diagnostic Image Analysis

---

## 🏥 Project Overview

Lab Lens is an end-to-end MLOps pipeline for healthcare that combines medical report summarization with diagnostic image analysis. The platform processes MIMIC-III clinical data to provide intelligent insights while ensuring fairness and bias mitigation in healthcare AI systems.

### 🎯 Key Features

- **Medical Report Summarization**: Simplifying discharge summaries using MIMIC-III clinical notes
- **Chest X-ray Classification**: Pathology detection using MedMNIST-ChestMNIST (CPU-optimized)
- **Automated Bias Detection**: Comprehensive bias analysis and mitigation
- **Data Quality Validation**: Robust data validation and quality assurance
- **Production-Ready Pipeline**: Complete MLOps pipeline with error handling and logging

### 👥 Team Members

- **Asad Ullah Waraich** - Data Acquisition & Pipeline Architecture
- **Shruthi Kashetty** - Model Development & Training
- **Mihir Harishankar Parab** - Data Processing & Validation
- **Sri Lakshmi Swetha Jalluri** - Bias Detection & Fairness
- **Dhruv Rameshbhai Gajera** - Infrastructure & Deployment
- **Shahid Kamal** - System Integration & Quality Assurance

## 📊 Project Components

### 1. Medical Report Summarization
- **Dataset**: MIMIC-III v1.4 clinical notes and discharge summaries
- **Processing**: Text cleaning, section extraction, abbreviation expansion
- **Features**: 26+ engineered features including complexity metrics
- **Quality**: 95/100 validation score with comprehensive bias analysis

### 2. Chest X-ray Classification
- **Dataset**: MedMNIST-ChestMNIST pre-processed 28×28 chest X-rays
- **Optimization**: CPU-optimized for efficient processing
- **Coverage**: Multiple pathology detection capabilities

### 3. Automated Bias Detection & Mitigation
- **Demographic Analysis**: Gender, ethnicity, age, insurance type
- **Statistical Testing**: T-tests, ANOVA for significance detection
- **Automated Mitigation**: Stratified sampling, oversampling, weighting
- **Compliance Monitoring**: Real-time bias score tracking

### 4. Data Quality Assurance
- **Validation Pipeline**: Schema validation, completeness checks
- **Error Handling**: Comprehensive error management and recovery
- **Logging System**: Centralized logging with performance metrics
- **Monitoring**: Continuous data quality monitoring

## 🗂️ Project Structure

```
lab-lens/
├── 📁 docs/                    # Documentation and project materials
│   └── LabLens_ AI-Powered Health Report Simplification .pdf
├── 📁 data-pipeline/           # Main data processing pipeline
│   ├── 📁 configs/            # Configuration files
│   │   └── pipeline_config.json
│   ├── 📁 data/               # Data storage
│   │   ├── 📁 raw/            # Raw data from BigQuery
│   │   └── 📁 processed/      # Processed data
│   ├── 📁 logs/               # Logs and reports
│   │   ├── 📁 bias_plots/     # Bias visualization outputs
│   │   ├── validation_report.json
│   │   └── bias_report.json
│   ├── 📁 notebooks/          # Jupyter notebooks for exploration
│   │   └── data_acquisition.ipynb
│   ├── 📁 scripts/            # Processing scripts
│   │   ├── main_pipeline.py   # Main orchestration script
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── validation.py      # Data validation
│   │   ├── bias_detection.py  # Bias analysis
│   │   └── automated_bias_handler.py  # Automated mitigation
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Pipeline documentation
├── 📁 src/                   # Source code modules
│   ├── 📁 utils/             # Utility modules
│   │   ├── logging_config.py # Centralized logging
│   │   └── error_handling.py # Error management
│   ├── 📁 data/              # Data processing modules
│   ├── 📁 training/          # Model training modules
│   └── 📁 utils/             # General utilities
├── 📁 configs/               # Global configuration files
├── 📁 notebooks/             # Global notebooks
├── 📁 scripts/               # Global scripts
├── 📁 tests/                 # Unit tests
├── LICENSE                   # Project license
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12+ (Note: Python 3.13 has compatibility issues)
- **Google Cloud Platform**: Account with BigQuery access
- **PhysioNet Credentialing**: Required for MIMIC-III access
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: 10GB+ free space for data processing

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/lab-lens.git
cd lab-lens

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r data-pipeline/requirements.txt
```

### GCP Authentication

```bash
# Install gcloud CLI (if needed)
brew install --cask google-cloud-sdk  # Mac
# or download from https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Running the Pipeline

```bash
# Run complete pipeline
python data-pipeline/scripts/main_pipeline.py

# Run with custom configuration
python data-pipeline/scripts/main_pipeline.py --config data-pipeline/configs/pipeline_config.json

# Run specific steps only
python data-pipeline/scripts/main_pipeline.py --skip-preprocessing --skip-validation

# Run with custom paths
python data-pipeline/scripts/main_pipeline.py \
  --input-path /path/to/raw/data \
  --output-path /path/to/processed/data \
  --logs-path /path/to/logs
```

## 📈 Pipeline Components

### 1. Data Acquisition
- **Source**: Google BigQuery MIMIC-III tables
- **Method**: Cloud-based querying (no local download)
- **Output**: 5,000+ discharge summaries with demographics
- **Performance**: ~30 seconds for 5,000 records

### 2. Data Preprocessing
- **Text Processing**: Section extraction, abbreviation expansion
- **Feature Engineering**: 26+ new features created
- **Quality Control**: Comprehensive text cleaning and validation
- **Output**: Structured dataset ready for ML modeling

### 3. Data Validation
- **Schema Validation**: Required columns verification
- **Quality Metrics**: Completeness, duplicates, outliers
- **Score**: 95/100 overall validation score
- **Coverage**: 0% missing critical fields

### 4. Bias Detection
- **Demographic Analysis**: Gender, ethnicity, age, insurance
- **Statistical Testing**: Significance tests for differences
- **Visualization**: Automated bias plot generation
- **Score**: 5.88 overall bias score (lower is better)

### 5. Automated Bias Mitigation
- **Detection**: Automatic bias threshold monitoring
- **Strategies**: Stratified sampling, oversampling, weighting
- **Compliance**: Real-time bias score tracking
- **Reporting**: Comprehensive mitigation reports

## 🔧 Configuration

### Pipeline Configuration

The pipeline can be configured using JSON configuration files:

```json
{
  "pipeline_config": {
    "input_path": "data/raw",
    "output_path": "data/processed",
    "logs_path": "logs",
    "enable_preprocessing": true,
    "enable_validation": true,
    "enable_bias_detection": true,
    "enable_automated_bias_handling": true
  },
  "bias_detection_config": {
    "alert_thresholds": {
      "gender_cv_max": 5.0,
      "ethnicity_cv_max": 10.0,
      "overall_bias_score_max": 10.0
    }
  }
}
```

### Environment Variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export MIMIC_PROJECT_ID="your-gcp-project-id"
export LOG_LEVEL="INFO"
```

## 📊 Results & Metrics

### Data Quality Metrics
- **Validation Score**: 95/100 (Excellent)
- **Schema Valid**: ✅ All required columns present
- **Missing Text**: 0 records
- **Duplicate Records**: 0
- **Average Text Length**: 9,558 characters

### Bias Analysis Results
- **Gender Bias**: Not statistically significant (p=0.21)
- **Ethnicity Variation**: 8.4% coefficient of variation
- **Age Group Variation**: 7.2% coefficient of variation
- **Overall Bias Score**: 5.88 (scale 0-100, lower is better)

### Performance Metrics
- **Query Time**: ~30 seconds for 5,000 records
- **Processing Time**: ~45 seconds for full preprocessing
- **Total Pipeline Runtime**: ~3 minutes end-to-end
- **Memory Usage**: <2GB peak usage

## 🛡️ Error Handling & Logging

### Comprehensive Error Management
- **Custom Exceptions**: Specialized error types for different operations
- **Error Recovery**: Automatic retry mechanisms and fallback strategies
- **Context Preservation**: Detailed error context and stack traces
- **Graceful Degradation**: Pipeline continues despite non-critical errors

### Centralized Logging System
- **Structured Logging**: JSON-formatted logs with metadata
- **Performance Metrics**: Automatic timing and resource usage tracking
- **Data Metrics**: Record counts and processing statistics
- **Log Rotation**: Automatic log file rotation and archival

### Log Levels
- **DEBUG**: Detailed execution information
- **INFO**: General pipeline progress and metrics
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Processing errors with recovery attempts
- **CRITICAL**: System failures requiring intervention

## 🔍 Monitoring & Alerting

### Real-time Monitoring
- **Pipeline Status**: Live pipeline execution monitoring
- **Data Quality**: Continuous data quality assessment
- **Bias Metrics**: Real-time bias score tracking
- **Performance**: Resource usage and timing metrics

### Automated Alerts
- **Bias Thresholds**: Alerts when bias scores exceed limits
- **Data Quality**: Notifications for validation failures
- **Performance**: Alerts for slow processing or resource issues
- **System Health**: Overall system status monitoring

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_validation.py
python -m pytest tests/test_bias_detection.py
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/

# Test complete pipeline
python tests/test_pipeline_integration.py
```

## 📚 Documentation

### API Documentation
- **Preprocessing API**: Complete preprocessing function documentation
- **Validation API**: Data validation and quality checking APIs
- **Bias Detection API**: Bias analysis and mitigation APIs
- **Pipeline API**: Main pipeline orchestration documentation

### User Guides
- **Getting Started**: Quick start guide for new users
- **Configuration Guide**: Detailed configuration options
- **Troubleshooting**: Common issues and solutions
- **Best Practices**: Recommended usage patterns

## 🔮 Future Roadmap

### Phase 1: Enhanced ML Integration
- [ ] BioBERT integration for medical entity extraction
- [ ] Advanced NLP models for text summarization
- [ ] Multi-modal model training pipeline
- [ ] Model versioning and experiment tracking

### Phase 2: Production Deployment
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline integration
- [ ] Production monitoring dashboard

### Phase 3: Advanced Features
- [ ] Real-time data processing
- [ ] Stream processing for new admissions
- [ ] Interactive bias monitoring dashboard
- [ ] Automated model retraining

### Phase 4: Scalability & Performance
- [ ] Distributed processing with Dask/Ray
- [ ] GPU acceleration for model training
- [ ] Cloud-native deployment options
- [ ] Auto-scaling based on workload

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ data-pipeline/scripts/
isort src/ data-pipeline/scripts/
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Data Usage
- **MIMIC-III**: Licensed under PhysioNet Credentialed Health Data License 1.5.0
- **MedMNIST**: Licensed under Apache License 2.0
- **Project Code**: Licensed under MIT License

## 📞 Support & Contact

### Getting Help
- **Documentation**: Check the comprehensive documentation in `/docs`
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Email**: Contact the team at lab-lens-support@example.com

### Team Contacts
- **Project Lead**: Asad Ullah Waraich
- **Technical Lead**: Shahid Kamal
- **Data Science Lead**: Shruthi Kashetty
- **Infrastructure Lead**: Dhruv Rameshbhai Gajera

---

## 🏆 Key Achievements

✅ **Successfully processed 5,000+ discharge summaries**  
✅ **Achieved 95/100 data validation score**  
✅ **Implemented comprehensive bias detection**  
✅ **Automated bias mitigation strategies**  
✅ **Built production-ready MLOps pipeline**  
✅ **Comprehensive error handling and logging**  
✅ **Real-time monitoring and alerting**  
✅ **Scalable and maintainable architecture**  

*Developed as part of MLOps Course Project - Fall 2025*
