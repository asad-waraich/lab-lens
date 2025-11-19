import json
import os

# Check if files exist
adapter_path = "../saved_models/content/saved_model"
files_to_check = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json"
]

print("Checking model files...")
for file in files_to_check:
    path = os.path.join(adapter_path, file)
    if os.path.exists(path):
        print(f"✓ {file} found")
    else:
        print(f"✗ {file} missing")

# Load and display adapter config
config_path = os.path.join(adapter_path, "adapter_config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("\n" + "="*50)
    print("MODEL CONFIGURATION:")
    print("="*50)
    print(f"Base model: {config['base_model_name_or_path']}")
    print(f"LoRA rank: {config['r']}")
    print(f"Target modules: {', '.join(config['target_modules'])}")
    print("\nYour fine-tuned model is ready to use!")
