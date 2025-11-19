from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import warnings
warnings.filterwarnings('ignore')

def load_model():
    adapter_path = "../saved_models/content/saved_model"
    base_model = "unsloth/llama-3.2-3b-instruct"
    
    print("Loading base model (this may take a few minutes on first run)...")
    try:
        # Try loading with less memory
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU to avoid MPS issues on Mac
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def simple_test():
    """Test with a simple medical prompt"""
    model, tokenizer = load_model()
    
    if model is None:
        print("Failed to load model")
        return
    
    # Simple medical text - MATCHING TRAINING FORMAT
    discharge_text = """Discharge Medications:
1. Insulin Regular Human 100 unit/mL Solution Sig: Per sliding
scale units Injection ASDIR (AS DIRECTED).
2. Chlorhexidine Gluconate 0.12 % Mouthwash Sig: One (1) ML
Mucous membrane  (2 times a day).
3. Docusate Sodium 50 mg/5 mL Liquid Sig: One (1)  PO BID (2
times a day).
4. Heparin (Porcine) 5,000 unit/mL Solution Sig: One (1)
Injection TID (3 times a day).
5. Bisacodyl 5 mg Tablet, Delayed Release (E.C.) Sig: Two (2)
Tablet, Delayed Release (E.C.) PO DAILY (Daily) as needed for
constipation.
6. Metoprolol Tartrate 50 mg Tablet Sig: 1.5 Tablets PO TID (3
times a day).
7. Famotidine 20 mg Tablet Sig: One (1) Tablet PO BID (2 times a
day).
8. Bacitracin-Polymyxin B 500-10,000 unit/g Ointment Sig: One
(1) Appl Ophthalmic Q6H (every 6 hours): last dose 9/25.
9. Acetaminophen 325 mg Tablet Sig: Two (2) Tablet PO Q4H (every
4 hours) as needed for fever or pain.
10. Polyvinyl Alcohol-Povidone 1.4-0.6 % Dropperette Sig: 
Drops Ophthalmic Q4H (every 4 hours).
11. Miconazole Nitrate 2 % Powder Sig: One (1) Appl Topical TID
(3 times a day) as needed for bilat axilla.
12. Vancomycin 500 mg Recon Soln Sig: One (1) Drop Intravenous
Q2H (every 2 hours): last dose to be on  or as directed by
ophthalmologist.
13. Tobramycin Sulfate 0.3 % Drops Sig: One (1) Drop Ophthalmic
Q2H (every 2 hours): last dose to be on  or as directed by
ophthalmologist.     .
14. Lactulose 10 gram/15 mL Syrup Sig: Thirty (30) ML PO MWF
(Monday-Wednesday-Friday) as needed for constipation.
15. Vancomycin 500 mg Recon Soln Sig: 1.5 Recon Solns
Intravenous Q 24H (Every 24 Hours): For Staph bacteremia.  Last
dose to be on .
16. Piperacillin-Tazobactam-Dextrs 4.5 gram/100 mL Piggyback
Sig: One (1)  Intravenous Q8H (every 8 hours): For Pseudomonas
pneumonia.  Last dose to be .
17. Heparin, Porcine (PF) 10 unit/mL Syringe Sig: One (1) ML
Intravenous PRN (as needed) as needed for line flush.


Discharge Disposition:
Extended Care

Facility:
 Northeast - 

Discharge Diagnosis:
brainstem hemorrhage involving R midbrain, bilateral pons, and
4th ventricle, likely hypertensive etiology.

Discharge Condition:
The patient was hemodynamically stable and afebrile.  His
neurolgic exam was notable for the following:
Responds to voice and follows simple commands.  Pupils equal but
right eye with minimal reaction.  Able to move eyes in the
vertical plane with down-gaze better than upgaze; occasional
lateral eye movement; minimal adduction of the eyes.
Can move right fingers, wrist and occasionally his arm, right
toes. Myoclonic movements of the mouth and right shoulder.
Occasional movement of the fingers of the left hand an toes.
Reflexes are brisk throughout.  Appropriate responses to noxious
stimuli in all extremities.


Discharge Instructions:
You were admitted after being found unresponsive.  You have
suffered a major hemorrhage in your brain which has resulted in
your inability to move, eat or speak.  You are being discharged
to a long term care facility for continued treatment and
therapy.

Followup Instructions:
None


"""
    
    # FORMAT THE PROMPT EXACTLY LIKE TRAINING
    instruction = "You are a medical assistant. Explain this hospital discharge summary to the patient in simple, everyday language. Avoid medical jargon."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{discharge_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    print("\nGenerating response...")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("\n" + "="*50)
    print("Generated Summary:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    print("Testing your fine-tuned medical summarization model...")
    simple_test()