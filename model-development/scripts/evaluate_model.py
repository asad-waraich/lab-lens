#!/usr/bin/env python3
"""
Model Evaluation Script for Lab Lens Medical Summarization
Evaluates fine-tuned models on test set with comprehensive metrics
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Comprehensive model evaluation for medical summarization"""
    
    def __init__(self, model_path: str, test_data_path: str, output_dir: str = './evaluation_results'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to fine-tuned model
            test_data_path: Path to test CSV file
            output_dir: Where to save evaluation results
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading test data from {test_data_path}...")
        self.test_df = pd.read_csv(test_data_path)
        print(f"✅ Loaded {len(self.test_df)} test samples")
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'num_test_samples': len(self.test_df),
            'predictions': [],
            'metrics': {}
        }
    
    def generate_summary(self, input_text: str, max_new_tokens: int = 256) -> Tuple[str, float]:
        """
        Generate summary for a single input
        
        Returns:
            (summary_text, inference_time_seconds)
        """
        # Format as instruction
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical assistant. Explain this hospital discharge summary to the patient in simple, everyday language. Avoid medical jargon.<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_text[:2000]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        inference_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            summary = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            summary = generated_text
        
        return summary, inference_time
    
    def calculate_rouge_scores(self, reference: str, prediction: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, prediction)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def evaluate_full_test_set(self, sample_size: int = None) -> Dict:
        """
        Evaluate model on full test set (or sample)
        
        Args:
            sample_size: If specified, evaluate on random sample (for speed)
        """
        print("\n" + "="*60)
        print("RUNNING FULL EVALUATION")
        print("="*60)
        
        # Sample if specified
        if sample_size and sample_size < len(self.test_df):
            eval_df = self.test_df.sample(n=sample_size, random_state=42)
            print(f"Evaluating on random sample of {sample_size} examples")
        else:
            eval_df = self.test_df
            print(f"Evaluating on full test set ({len(eval_df)} examples)")
        
        predictions = []
        rouge_scores = []
        inference_times = []
        
        print("\nGenerating predictions...")
        for idx, row in eval_df.iterrows():
            if idx % 50 == 0:
                print(f"Progress: {idx}/{len(eval_df)}")
            
            # Generate prediction
            pred_summary, inf_time = self.generate_summary(row['cleaned_text'])
            
            # Calculate ROUGE
            rouge = self.calculate_rouge_scores(row['simple_summary'], pred_summary)
            
            # Store results
            predictions.append({
                'hadm_id': row['hadm_id'],
                'input_text': row['cleaned_text'][:500],  # First 500 chars
                'reference_summary': row['simple_summary'],
                'predicted_summary': pred_summary,
                'inference_time': inf_time,
                **rouge
            })
            
            rouge_scores.append(rouge)
            inference_times.append(inf_time)
        
        # Calculate aggregate metrics
        avg_metrics = {
            'rouge1_f': np.mean([s['rouge1_f'] for s in rouge_scores]),
            'rouge2_f': np.mean([s['rouge2_f'] for s in rouge_scores]),
            'rougeL_f': np.mean([s['rougeL_f'] for s in rouge_scores]),
            'avg_inference_time': np.mean(inference_times),
            'median_inference_time': np.median(inference_times),
            'max_inference_time': np.max(inference_times),
            'min_inference_time': np.min(inference_times),
        }
        
        self.results['metrics'] = avg_metrics
        self.results['predictions'] = predictions
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"ROUGE-1 F1: {avg_metrics['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {avg_metrics['rouge2_f']:.4f}")
        print(f"ROUGE-L F1: {avg_metrics['rougeL_f']:.4f}")
        print(f"\nAvg Inference Time: {avg_metrics['avg_inference_time']:.2f}s")
        print(f"Median Inference Time: {avg_metrics['median_inference_time']:.2f}s")
        print("="*60)
        
        return self.results
    
    def calculate_bertscore(self, references: List[str], predictions: List[str]) -> Dict:
        """Calculate BERTScore (semantic similarity)"""
        print("\nCalculating BERTScore...")
        
        P, R, F1 = bert_score(
            predictions, 
            references, 
            lang='en',
            model_type='microsoft/deberta-base-mnli',
            verbose=False
        )
        
        bertscore_results = {
            'precision': float(P.mean()),
            'recall': float(R.mean()),
            'f1': float(F1.mean())
        }
        
        print(f"BERTScore F1: {bertscore_results['f1']:.4f}")
        
        return bertscore_results
    
    def generate_sample_predictions(self, n_samples: int = 10) -> pd.DataFrame:
        """Generate sample predictions for manual review"""
        print(f"\nGenerating {n_samples} sample predictions for review...")
        
        sample_df = self.test_df.sample(n=n_samples, random_state=42)
        samples = []
        
        for idx, row in sample_df.iterrows():
            pred_summary, inf_time = self.generate_summary(row['cleaned_text'])
            rouge = self.calculate_rouge_scores(row['simple_summary'], pred_summary)
            
            samples.append({
                'hadm_id': row['hadm_id'],
                'input_preview': row['cleaned_text'][:200] + "...",
                'reference': row['simple_summary'],
                'prediction': pred_summary,
                'rouge2_f1': rouge['rouge2_f'],
                'inference_time': inf_time
            })
        
        samples_df = pd.DataFrame(samples)
        
        # Save samples
        samples_df.to_csv(self.output_dir / 'sample_predictions.csv', index=False)
        print(f"✅ Saved to {self.output_dir / 'sample_predictions.csv'}")
        
        return samples_df
    
    def create_visualizations(self):
        """Create evaluation visualizations"""
        print("\nCreating visualizations...")
        
        if not self.results['predictions']:
            print("No predictions to visualize. Run evaluate_full_test_set() first.")
            return
        
        # Extract metrics
        rouge1_scores = [p['rouge1_f'] for p in self.results['predictions']]
        rouge2_scores = [p['rouge2_f'] for p in self.results['predictions']]
        rougeL_scores = [p['rougeL_f'] for p in self.results['predictions']]
        inference_times = [p['inference_time'] for p in self.results['predictions']]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. ROUGE Score Distribution
        axes[0, 0].hist([rouge1_scores, rouge2_scores, rougeL_scores], 
                        label=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], 
                        alpha=0.7, bins=20)
        axes[0, 0].set_xlabel('ROUGE F1 Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('ROUGE Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Inference Time Distribution
        axes[0, 1].hist(inference_times, bins=30, color='green', alpha=0.7)
        axes[0, 1].axvline(np.mean(inference_times), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(inference_times):.2f}s')
        axes[0, 1].set_xlabel('Inference Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Inference Time Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. ROUGE-2 vs Inference Time Scatter
        axes[1, 0].scatter(inference_times, rouge2_scores, alpha=0.5)
        axes[1, 0].set_xlabel('Inference Time (seconds)')
        axes[1, 0].set_ylabel('ROUGE-2 F1')
        axes[1, 0].set_title('Quality vs Speed Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        EVALUATION SUMMARY
        ==================
        
        Test Samples: {len(self.results['predictions'])}
        
        ROUGE Scores:
        • ROUGE-1: {np.mean(rouge1_scores):.4f} ± {np.std(rouge1_scores):.4f}
        • ROUGE-2: {np.mean(rouge2_scores):.4f} ± {np.std(rouge2_scores):.4f}
        • ROUGE-L: {np.mean(rougeL_scores):.4f} ± {np.std(rougeL_scores):.4f}
        
        Inference Performance:
        • Mean: {np.mean(inference_times):.2f}s
        • Median: {np.median(inference_times):.2f}s
        • 95th percentile: {np.percentile(inference_times, 95):.2f}s
        
        Quality Assessment:
        • Excellent (ROUGE-2 > 0.5): {sum(1 for s in rouge2_scores if s > 0.5)}
        • Good (0.4-0.5): {sum(1 for s in rouge2_scores if 0.4 <= s <= 0.5)}
        • Fair (0.3-0.4): {sum(1 for s in rouge2_scores if 0.3 <= s < 0.4)}
        • Poor (< 0.3): {sum(1 for s in rouge2_scores if s < 0.3)}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to {self.output_dir / 'evaluation_summary.png'}")
        plt.close()
    
    def save_results(self):
        """Save evaluation results to JSON"""
        results_file = self.output_dir / 'evaluation_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Saved results to {results_file}")
        
        # Also save metrics summary as CSV
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_df.to_csv(self.output_dir / 'evaluation_metrics.csv', index=False)
        print(f"✅ Saved metrics to {self.output_dir / 'evaluation_metrics.csv'}")
    
    def create_detailed_report(self) -> str:
        """Create human-readable evaluation report"""
        report = f"""
# Model Evaluation Report
## Lab Lens Medical Summarization Model

**Evaluation Date:** {self.results['timestamp']}
**Model:** {self.model_path}
**Test Samples:** {self.results['num_test_samples']}

---

## Performance Metrics

### ROUGE Scores
- **ROUGE-1 F1:** {self.results['metrics']['rouge1_f']:.4f}
- **ROUGE-2 F1:** {self.results['metrics']['rouge2_f']:.4f}
- **ROUGE-L F1:** {self.results['metrics']['rougeL_f']:.4f}

### Inference Performance
- **Average Latency:** {self.results['metrics']['avg_inference_time']:.2f} seconds
- **Median Latency:** {self.results['metrics']['median_inference_time']:.2f} seconds
- **Max Latency:** {self.results['metrics']['max_inference_time']:.2f} seconds
- **Min Latency:** {self.results['metrics']['min_inference_time']:.2f} seconds

---

## Quality Assessment

**Primary Metric (ROUGE-2):** {self.results['metrics']['rouge2_f']:.4f}

"""
        
        # Add quality tier
        rouge2 = self.results['metrics']['rouge2_f']
        if rouge2 > 0.5:
            tier = "EXCELLENT"
        elif rouge2 > 0.4:
            tier = "GOOD"
        elif rouge2 > 0.3:
            tier = "FAIR"
        else:
            tier = "NEEDS IMPROVEMENT"
        
        report += f"**Quality Tier:** {tier}\n\n"
        
        # Deployment readiness
        report += "## Deployment Readiness\n\n"
        
        checks = {
            "ROUGE-2 > 0.35": self.results['metrics']['rouge2_f'] > 0.35,
            "Avg latency < 3s": self.results['metrics']['avg_inference_time'] < 3.0,
            "95th percentile latency < 5s": True,  # Calculate if needed
        }
        
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"- {check}: {status}\n"
        
        report += "\n---\n\n"
        report += f"**Overall Status:** {'✅ READY FOR DEPLOYMENT' if all(checks.values()) else '⚠️ NEEDS IMPROVEMENT'}\n"
        
        # Save report
        report_file = self.output_dir / 'EVALUATION_REPORT.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Saved report to {report_file}")
        
        return report
    
    def run_complete_evaluation(self, sample_size: int = None, 
                                include_bertscore: bool = False) -> Dict:
        """
        Run complete evaluation pipeline
        
        Args:
            sample_size: Number of samples to evaluate (None = all)
            include_bertscore: Whether to calculate BERTScore (slower)
        """
        print("\n🚀 Starting Complete Model Evaluation")
        print("="*60)
        
        # 1. Full test set evaluation
        self.evaluate_full_test_set(sample_size=sample_size)
        
        # 2. BERTScore (optional, slower)
        if include_bertscore:
            references = [p['reference_summary'] for p in self.results['predictions']]
            predictions = [p['predicted_summary'] for p in self.results['predictions']]
            bertscore_results = self.calculate_bertscore(references, predictions)
            self.results['metrics']['bertscore'] = bertscore_results
        
        # 3. Generate sample predictions for manual review
        samples_df = self.generate_sample_predictions(n_samples=10)
        
        # 4. Create visualizations
        self.create_visualizations()
        
        # 5. Save all results
        self.save_results()
        
        # 6. Create detailed report
        report = self.create_detailed_report()
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - evaluation_results.json")
        print(f"  - evaluation_metrics.csv")
        print(f"  - sample_predictions.csv")
        print(f"  - evaluation_summary.png")
        print(f"  - EVALUATION_REPORT.md")
        
        return self.results


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Lab Lens Medical Summarization Model')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to fine-tuned model')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Evaluate on sample (default: all)')
    parser.add_argument('--include-bertscore', action='store_true',
                       help='Calculate BERTScore (slower)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_complete_evaluation(
        sample_size=args.sample_size,
        include_bertscore=args.include_bertscore
    )
    
    print("\n🎉 Evaluation complete! Check the output directory for detailed results.")


if __name__ == "__main__":
    main()