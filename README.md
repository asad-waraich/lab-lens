# Lab Lens:  Healthcare Intelligence Platform

Multi-Modal Healthcare Intelligence Platform for Medical Report Summarization and Diagnostic Image Analysis

---


### Project Overview
An end-to-end MLOps pipeline for healthcare that combines medical report summarization with diagnostic image analysis.

### Team Members
- Asad Ullah Waraich
- Shruthi Kashetty
- Mihir Harishankar Parab
- Sri Lakshmi Swetha Jalluri
- Dhruv Rameshbhai Gajera
- Shahid Kamal

## Project Components
1. **Medical Report Summarization**: Simplifying discharge summaries using MIMIC-III clinical notes
2. **Chest X-ray Classification**: Pathology detection using MedMNIST-ChestMNIST (CPU-optimized)
3. **Department Routing**: (Future implementation)
4. **OTC Recommendations**: (Future implementation)

## Datasets
- **MIMIC-III v1.4**: Clinical notes and discharge summaries (requires PhysioNet credentialing)
- **MedMNIST-ChestMNIST**: Pre-processed 28×28 chest X-rays (immediate access)

## Project Structure
```
mlops-group14/
├── docs/           # Documentation and scoping documents
├── notebooks/      # Data exploration and experiments
├── src/            # Source code
│   ├── data/       # Data loading and preprocessing
│   ├── models/     # Model implementations
│   └── training/   # Training pipelines
├── configs/        # Configuration files
├── scripts/        # Setup and utility scripts
├── tests/          # Unit tests
└── data/           # Data storage (not tracked in git)
```

## Status
- [x] Introduction
- [x] Project scoping
- [ ] Data
- [ ] Modeling
- [ ] Deployment
- [ ] Monitoring# Renamed repository to lab-lens
