from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model():
    # Path to your adapter
    adapter_path = "../saved_models/content/saved_model"
    
    # Base model name (from your adapter_config.json)
    base_model = "unsloth/llama-3.2-3b-instruct"
    
    print("Loading base model...")
    # Load base model - using smaller precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print("Loading LoRA adapter...")
    # Load your fine-tuned adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def test_inference():
    model, tokenizer = load_model()
    
    # Test prompt - medical discharge summary
    test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical assistant. Explain this hospital discharge summary to the patient in simple, everyday language. Avoid medical jargon.<|eot_id|><|start_header_id|>user<|end_header_id|>

The patient was admitted with chest pain and shortness of breath. Diagnosis: acute myocardial infarction. Treatment: cardiac catheterization with stent placement. Medications: aspirin, metoprolol, lisinopril.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    print("\nGenerating response...")
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    test_inference()