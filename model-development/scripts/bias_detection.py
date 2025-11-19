import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from validate_model import load_model, format_prompt
from rouge_score import rouge_scorer

def detect_bias(test_csv_path="../data/test_set.csv"):
    """Detect bias across demographic groups"""
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    model, tokenizer = load_model()
    
    print("Analyzing model bias across demographic groups...")
    
    # Initialize results
    bias_results = {
        'gender': {},
        'age_group': {},
        'ethnicity_clean': {}
    }
    
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    
    # Analyze by gender
    if 'gender' in test_df.columns:
        print("\nAnalyzing gender bias...")
        for gender in test_df['gender'].dropna().unique():
            subset = test_df[test_df['gender'] == gender]
            scores = evaluate_subset(model, tokenizer, subset, scorer)
            bias_results['gender'][str(gender)] = {
                'avg_score': float(np.mean(scores)),
                'count': len(subset)
            }
            print(f"  {gender}: {np.mean(scores):.3f} (n={len(subset)})")
    
    # Analyze by age group
    if 'age_group' in test_df.columns:
        print("\nAnalyzing age bias...")
        for age in test_df['age_group'].dropna().unique():
            subset = test_df[test_df['age_group'] == age]
            if len(subset) > 0:
                scores = evaluate_subset(model, tokenizer, subset, scorer)
                bias_results['age_group'][str(age)] = {
                    'avg_score': float(np.mean(scores)),
                    'count': len(subset)
                }
                print(f"  {age}: {np.mean(scores):.3f} (n={len(subset)})")
    
    # Analyze by ethnicity
    if 'ethnicity_clean' in test_df.columns:
        print("\nAnalyzing ethnicity bias...")
        for eth in test_df['ethnicity_clean'].dropna().unique()[:5]:  # Top 5 groups
            subset = test_df[test_df['ethnicity_clean'] == eth]
            if len(subset) > 0:
                scores = evaluate_subset(model, tokenizer, subset, scorer)
                bias_results['ethnicity_clean'][str(eth)] = {
                    'avg_score': float(np.mean(scores)),
                    'count': len(subset)
                }
                print(f"  {eth}: {np.mean(scores):.3f} (n={len(subset)})")
    
    # Check for significant disparities
    bias_report = generate_bias_report(bias_results)
    
    # Create visualization
    plot_bias_results(bias_results)
    
    return bias_results, bias_report

def evaluate_subset(model, tokenizer, subset_df, scorer, max_samples=3):
    """Evaluate model on a subset of data"""
    scores = []
    
    for idx in range(min(max_samples, len(subset_df))):
        row = subset_df.iloc[idx]
        
        # Generate summary
        prompt = format_prompt(row.get('cleaned_text', row.get('input_text', '')))
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        reference = row.get('simple_summary', row.get('target_summary', 'No reference'))
        
        score = scorer.score(reference, generated)
        scores.append(score['rouge1'].fmeasure)
    
    return scores

def generate_bias_report(bias_results):
    """Generate report on bias findings"""
    report = {
        'significant_disparities': [],
        'recommendations': []
    }
    
    # Check each demographic category
    for category, results in bias_results.items():
        if len(results) > 1:
            scores = [r['avg_score'] for r in results.values()]
            disparity = max(scores) - min(scores)
            
            if disparity > 0.1:  # 10% threshold
                report['significant_disparities'].append({
                    'category': category,
                    'disparity': float(disparity),
                    'max_score': float(max(scores)),
                    'min_score': float(min(scores))
                })
                
                report['recommendations'].append(
                    f"Address {category} bias: {disparity:.1%} performance gap detected"
                )
    
    if not report['significant_disparities']:
        report['summary'] = "No significant bias detected (threshold: 10%)"
    else:
        report['summary'] = f"Bias detected in {len(report['significant_disparities'])} categories"
    
    return report

def plot_bias_results(bias_results):
    """Create visualization of bias analysis"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (category, results) in enumerate(bias_results.items()):
        if results and idx < 3:
            groups = list(results.keys())
            scores = [results[g]['avg_score'] for g in groups]
            
            axes[idx].bar(groups, scores)
            axes[idx].set_title(f'Performance by {category}')
            axes[idx].set_ylabel('ROUGE-1 Score')
            axes[idx].set_ylim(0, 1)
            axes[idx].axhline(y=np.mean(scores), color='r', linestyle='--', label='Mean')
            axes[idx].legend()
            
            # Rotate x labels if needed
            if len(groups) > 3:
                axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../results/bias_analysis.png')
    print("\nBias visualization saved to ../results/bias_analysis.png")

if __name__ == "__main__":
    import torch
    
    # Create results directory if needed
    os.makedirs('../results', exist_ok=True)
    
    # Run bias detection
    bias_results, bias_report = detect_bias()
    
    # Save results
    with open('../results/bias_report.json', 'w') as f:
        json.dump({
            'results': bias_results,
            'report': bias_report
        }, f, indent=2)
    
    print("\n" + "="*50)
    print("BIAS DETECTION SUMMARY")
    print("="*50)
    print(bias_report['summary'])
    
    if bias_report['significant_disparities']:
        print("\nSignificant disparities found:")
        for disp in bias_report['significant_disparities']:
            print(f"  - {disp['category']}: {disp['disparity']:.1%} gap")
    
    if bias_report['recommendations']:
        print("\nRecommendations:")
        for rec in bias_report['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nDetailed report saved to ../results/bias_report.json")