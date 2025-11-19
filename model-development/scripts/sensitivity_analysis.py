import torch
from validate_model import load_model, format_prompt
import numpy as np
import json
import os

def analyze_feature_importance(sample_text=None):
    """Analyze which input features most affect output"""
    model, tokenizer = load_model()
    
    # Use a default sample if none provided
    if sample_text is None:
        sample_text = """Discharge Summary
        
Medications:
1. Metoprolol 50mg twice daily
2. Aspirin 81mg daily
3. Lisinopril 10mg daily

Diagnosis:
Acute myocardial infarction

History:
Patient presented with chest pain and shortness of breath.

Discharge Instructions:
Follow up with cardiologist in 1 week.
Take all medications as prescribed.
"""
    
    print("Analyzing feature sensitivity...")
    
    # Define sections to test
    sections = {
        'Medications': ['medication', 'mg', 'daily'],
        'Diagnosis': ['diagnosis', 'acute', 'infarction'],
        'History': ['history', 'presented', 'pain'],
        'Instructions': ['follow', 'prescribed', 'week']
    }
    
    sensitivity_scores = {}
    
    # Get baseline output
    baseline_prompt = format_prompt(sample_text)
    baseline_inputs = tokenizer(baseline_prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        baseline_outputs = model.generate(
            baseline_inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_text = tokenizer.decode(baseline_outputs[0][len(baseline_inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Test each section
    for section_name, keywords in sections.items():
        # Remove section from text
        modified_text = sample_text
        for keyword in keywords:
            modified_text = modified_text.replace(keyword, "[REMOVED]")
        
        # Generate with modified text
        modified_prompt = format_prompt(modified_text)
        modified_inputs = tokenizer(modified_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            modified_outputs = model.generate(
                modified_inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        modified_text_output = tokenizer.decode(modified_outputs[0][len(modified_inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Calculate difference (simplified - using length difference as proxy)
        impact_score = abs(len(baseline_text) - len(modified_text_output)) / max(len(baseline_text), 1)
        sensitivity_scores[section_name] = float(impact_score)
        
        print(f"  {section_name}: {impact_score:.3f}")
    
    # Identify most important features
    most_important = max(sensitivity_scores.items(), key=lambda x: x[1])
    least_important = min(sensitivity_scores.items(), key=lambda x: x[1])
    
    analysis = {
        'sensitivity_scores': sensitivity_scores,
        'most_important_section': most_important[0],
        'least_important_section': least_important[0],
        'baseline_length': len(baseline_text),
        'analysis_summary': f"The model is most sensitive to {most_important[0]} (score: {most_important[1]:.3f}) and least sensitive to {least_important[0]} (score: {least_important[1]:.3f})"
    }
    
    return analysis

if __name__ == "__main__":
    # Create results directory if needed
    os.makedirs('../results', exist_ok=True)
    
    print("Running sensitivity analysis...")
    analysis = analyze_feature_importance()
    
    # Save results
    with open('../results/sensitivity_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "="*50)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*50)
    print(analysis['analysis_summary'])
    print(f"\nDetailed results saved to ../results/sensitivity_analysis.json")
