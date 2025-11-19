## Model Architecture & Training

### Base Model
- **Model**: Meta's LLaMA 3.2-3B-Instruct
- **Source**: `unsloth/llama-3.2-3b-instruct`
- **Parameters**: 3 billion
- **Architecture**: Decoder-only transformer

### Fine-tuning Approach
- **Method**: Low-Rank Adaptation (LoRA)
- **Framework**: Unsloth (2x faster training, 60% less memory)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Quantization**: 4-bit quantization for memory efficiency

### Training Specifications
- **Dataset**: MIMIC-III discharge summaries (10,000 samples)
- **Train/Val/Test Split**: 70%/15%/15%
- **Batch Size**: 1 (gradient accumulation: 1)
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW (8-bit)
- **Epochs**: 1
- **Training Duration**: 95.68 minutes on Google Colab (T4 GPU)
- **Max Sequence Length**: 2048 tokens

## Data Pipeline

### Data Processing
1. **Source**: MIMIC-III Clinical Database
2. **Preprocessing**:
   - Extracted discharge summaries from NOTEEVENTS table
   - Cleaned demographic data (gender, age groups, ethnicity)
   - Removed PHI and standardized formatting
   - Created instruction-tuned format for LLaMA

### Data Format
```
System: You are a medical assistant. Explain this hospital discharge summary to the patient in simple, everyday language. Avoid medical jargon.
User: [Medical discharge summary]
Assistant: [Simplified explanation]
```

## Model Performance

### Validation Metrics (ROUGE Scores)
- **ROUGE-1**: 0.356 (35.6% unigram overlap)
- **ROUGE-2**: 0.266 (26.6% bigram overlap)
- **ROUGE-L**: 0.283 (28.3% longest common subsequence)

### Training Metrics
- **Final Training Loss**: 0.614
- **Final Validation Loss**: 0.629
- **Loss Reduction**: 51.3% (from 1.26 to 0.614)

## MLOps Pipeline Components

### 1. Experiment Tracking
- **Tool**: MLflow 3.6.0
- **Tracking Server**: Local file store
- **Logged Artifacts**:
  - Model parameters and hyperparameters
  - Training metrics and validation scores
  - Bias analysis reports (JSON)
  - Sensitivity analysis results

### 2. Bias Detection
- **Analysis Dimensions**:
  - Gender (M/F)
  - Age groups (<18, 18-35, 35-50, 50-65, 65+)
  - Ethnicity (top 5 groups)
- **Result**: No significant bias detected (all disparities <10% threshold)
- **Method**: ROUGE score comparison across demographic slices

### 3. Sensitivity Analysis
- **Most Sensitive Feature**: Medications (0.068 impact score)
- **Least Sensitive Feature**: History (0.000 impact score)
- **Other Features**:
  - Diagnosis: 0.034
  - Instructions: 0.026

### 4. Quality Gates
- ✅ ROUGE-1 score > 0.3 threshold
- ✅ No significant demographic bias
- ✅ Model approved for deployment

## Reproducibility

### Environment Setup
```bash
# Python version: 3.11
# CUDA: 11.8 (for GPU training)
# Key dependencies:
pip install transformers==4.37.2
pip install unsloth[colab-new]
pip install peft==0.7.1
pip install datasets==2.16.1
pip install rouge-score
pip install mlflow==3.6.0
```

### Model Artifacts
- **Adapter Weights**: `saved_models/content/saved_model/`
- **Tokenizer Config**: `saved_models/content/saved_model/tokenizer_config.json`
- **Training Config**: Saved in MLflow experiments

### Running the Pipeline
```bash
# 1. Data Processing
python data-pipeline/scripts/process_mimic.py

# 2. Model Training (in Google Colab)
# Run: Llama3_2_(1B_and_3B)_Conversational.ipynb

# 3. Model Validation
python model-development/scripts/validate_model.py

# 4. Bias Detection
python model-development/scripts/bias_detection.py

# 5. Sensitivity Analysis
python model-development/scripts/sensitivity_analysis.py

# 6. MLOps Pipeline Orchestration
python model-development/scripts/train_model.py
```

## CI/CD Integration
- **GitHub Actions**: Automated testing and validation on push
- **Model Registry**: Ready for GCP Artifact Registry deployment
- **Monitoring**: MLflow tracking for all experiments

## Project Structure
```
lab-lens/
├── data-pipeline/          # MIMIC data processing
├── model-development/      # Training and evaluation scripts
│   ├── scripts/           # Python scripts
│   ├── saved_models/      # LoRA adapter weights
│   ├── results/           # Validation results
│   └── mlruns/           # MLflow experiments
├── documentation/         # Project documentation
└── requirements.txt      # Dependencies
```

## Key Achievements
1. Successfully fine-tuned a 3B parameter LLM for medical text simplification
2. Implemented comprehensive bias detection across demographics
3. Achieved 35.6% ROUGE-1 score on test set
4. Built complete MLOps pipeline with experiment tracking
5. Model passes all quality gates and is deployment-ready


## References
- LLaMA 3.2: Meta's Latest Language Model
- Unsloth: https://github.com/unslothai/unsloth
- MIMIC-III Database: https://mimic.physionet.org/
- LoRA Paper: https://arxiv.org/abs/2106.09685
