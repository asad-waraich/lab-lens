# Replace the validate_model.py with this corrected version
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
import numpy as np
import json
import os

def load_model():
    """Load your fine-tuned model"""
    adapter_path = "../saved_models/content/saved_model"
    base_model = "unsloth/llama-3.2-3b-instruct"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def format_prompt(input_text):
    """Format input text with the training template"""
    instruction = "You are a medical assistant. Explain this hospital discharge summary to the patient in simple, everyday language. Avoid medical jargon."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def validate_on_test_set(test_csv_path="../data/test_set.csv"):
    """Validate model on test set"""
    
    # Check if test file exists, if not use sample data
    if not os.path.exists(test_csv_path):
        print(f"Test file not found at {test_csv_path}")
        print("Using sample data instead...")
        return validate_with_samples()
    
    model, tokenizer = load_model()
    test_df = pd.read_csv(test_csv_path)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = []
    
    # Sample 5 examples for quick validation
    num_samples = min(5, len(test_df))
    print(f"Validating on {num_samples} samples...")
    
    for idx in range(num_samples):
        row = test_df.iloc[idx]
        
        # Format prompt
        prompt = format_prompt(row['cleaned_text'] if 'cleaned_text' in row else row.get('input_text', str(row)))
        
        # Generate summary
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Get reference summary
        reference = row.get('simple_summary', row.get('target_summary', 'No reference available'))
        
        # Calculate ROUGE score
        score = scorer.score(reference, generated)
        scores.append(score)
        
        print(f"\nSample {idx+1}:")
        print(f"Generated (first 100 chars): {generated[:100]}...")
        print(f"ROUGE-1: {score['rouge1'].fmeasure:.3f}")
    
    return scores

def validate_with_samples():
    """Validate with sample discharge summaries when no test file exists"""
    model, tokenizer = load_model()
    
    # Sample test data
    test_samples = [
        {
            "input_text": "Patient admitted with chest pain and shortness of breath. Diagnosis: acute myocardial infarction. Treatment: cardiac catheterization with stent placement. Medications: aspirin, metoprolol, lisinopril.",
            "expected_summary": "You had a heart attack. We opened your blocked heart artery and placed a stent. You'll take heart medications."
        },
        {
            "input_text": "Admission for pneumonia. Chest x-ray showed infiltrates. Treated with IV antibiotics. Discharged home stable on oral antibiotics.",
            "expected_summary": "You had a lung infection. We gave you antibiotics through IV and now pills to take at home."
        }
    ]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = []
    
    for i, sample in enumerate(test_samples):
        prompt = format_prompt(sample['input_text'])
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        score = scorer.score(sample['expected_summary'], generated)
        scores.append(score)
        
        print(f"\nSample {i+1}:")
        print(f"Input: {sample['input_text'][:100]}...")
        print(f"Generated: {generated[:200]}...")
        print(f"ROUGE-1: {score['rouge1'].fmeasure:.3f}")
    
    return scores

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    print("Starting validation...")
    scores = validate_on_test_set()
    
    # Calculate averages
    avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in scores])
    avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in scores])
    avg_rougeL = np.mean([s['rougeL'].fmeasure for s in scores])
    
    print(f"\n{'='*50}")
    print("VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Average ROUGE-1: {avg_rouge1:.3f}")
    print(f"Average ROUGE-2: {avg_rouge2:.3f}")
    print(f"Average ROUGE-L: {avg_rougeL:.3f}")
    
    # Save results
    results = {
        'avg_rouge1': float(avg_rouge1),
        'avg_rouge2': float(avg_rouge2),
        'avg_rougeL': float(avg_rougeL),
        'num_samples': len(scores)
    }
    
    with open('../results/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ../results/validation_results.json")
