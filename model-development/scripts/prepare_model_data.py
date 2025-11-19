"""
Medical Summary Template and Data Preparation for Model Training
Author: Lab Lens Team
Description: Prepares MIMIC-III discharge summaries for BioBART fine-tuning
             Creates structured summaries + detailed clinical narratives using Gemini API
"""

import pandas as pd
import numpy as np
import os
import json
import time
from typing import Dict
from sklearn.model_selection import train_test_split

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Using rule-based summaries.")

# Maximum summary length constraints
MAX_SUMMARY_LENGTH = 1500  # Increased for detailed narrative
MAX_BRIEF_SUMMARY_LENGTH = 600  # For detailed clinical narrative

# Enhanced template with CLINICAL SUMMARY section
SUMMARY_TEMPLATE = """
PATIENT: {age}-year-old {gender}

DATES: Admitted {admission_date}, Discharged {discharge_date}

ADMISSION: {chief_complaint}

HISTORY: {medical_history}

DIAGNOSIS: {diagnosis}

HOSPITAL COURSE: {hospital_course}

LABS: {lab_results}

MEDICATIONS: {medications}

FOLLOW-UP: {follow_up}

CLINICAL SUMMARY: {detailed_narrative}
"""

def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text preserving sentence boundaries"""
    if not text or pd.isna(text):
        return "Not documented"
    
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    
    if last_period > max_chars * 0.7:
        return truncated[:last_period + 1]
    return truncated.rstrip() + '...'

def format_date(date_value) -> str:
    """Format date to YYYY-MM-DD"""
    if pd.isna(date_value):
        return 'Not documented'
    try:
        return pd.to_datetime(date_value).strftime('%Y-%m-%d')
    except:
        date_str = str(date_value)
        return date_str[:10] if len(date_str) >= 10 else 'Not documented'

def generate_detailed_narrative_gemini(row: pd.Series, model) -> str:
    """Generate detailed clinical narrative using Gemini API"""
    
    # Extract clinical information
    age = row.get('age_at_admission', 'Unknown')
    gender_code = str(row.get('gender', '')).upper()
    gender = 'male' if gender_code == 'M' else 'female' if gender_code == 'F' else 'patient'
    
    chief_complaint = row.get('chief_complaint', '')
    history = row.get('past_medical_history', '')
    diagnosis = row.get('discharge_diagnosis', '')
    hospital_course = row.get('hospital_course', '')
    medications = row.get('discharge_medications', '')
    follow_up = row.get('follow_up', '')
    
    # Build clinical context (truncate to avoid token limits)
    clinical_info = f"""Patient: {age}-year-old {gender}
Chief Complaint: {str(chief_complaint)[:200] if chief_complaint else 'Not stated'}
Past Medical History: {str(history)[:200] if history else 'None documented'}
Discharge Diagnosis: {str(diagnosis)[:300] if diagnosis else 'Not documented'}
Hospital Course: {str(hospital_course)[:500] if hospital_course else 'Not documented'}
Discharge Medications: {str(medications)[:200] if medications else 'None'}
Follow-up: {str(follow_up)[:100] if follow_up else 'Not documented'}"""
    
    # Prompt for Gemini
    prompt = f"""You are a medical documentation specialist. Create a concise clinical narrative summary (2-3 sentences, 400-600 characters) from this discharge information.

{clinical_info}

REQUIREMENTS:
- Write in third-person clinical narrative style
- Start with: "A [age]-year-old [gender] was admitted with..."
- Include: chief complaint, key diagnoses, major treatments/procedures, clinical course highlights, disposition
- Use professional medical terminology
- Be factual - ONLY use information explicitly stated above
- If information is missing, omit that detail (do NOT fabricate)
- Maximum 600 characters
- Focus on clinically significant events and interventions
- End with disposition (discharged home, transferred, etc.)

CRITICAL: Do NOT hallucinate. Only summarize what is documented.

Clinical Narrative:"""

    try:
        # Call Gemini API with configuration
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,  # Lower temperature = more factual
            )
        )
        
        narrative = response.text.strip()
        
        # Ensure length constraint
        if len(narrative) > MAX_BRIEF_SUMMARY_LENGTH:
            truncated = narrative[:MAX_BRIEF_SUMMARY_LENGTH]
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_period > MAX_BRIEF_SUMMARY_LENGTH * 0.7:
                narrative = truncated[:last_period + 1]
            else:
                narrative = truncated.rstrip() + '...'
        
        return narrative
        
    except Exception as e:
        # Fallback to rule-based
        return generate_enhanced_fallback(row)

def generate_enhanced_fallback(row: pd.Series) -> str:
    """Enhanced rule-based narrative (no API needed)"""
    age = row.get('age_at_admission', 'Unknown')
    gender_code = str(row.get('gender', '')).upper()
    gender = 'male' if gender_code == 'M' else 'female' if gender_code == 'F' else 'patient'
    
    # Get chief complaint
    chief = row.get('chief_complaint', '')
    if pd.notna(chief) and chief:
        presenting = str(chief).split('\n')[0].strip()
        if len(presenting) > 100:
            presenting = presenting[:100]
    else:
        presenting = "acute medical condition"
    
    # Get diagnosis
    diagnosis = row.get('discharge_diagnosis', '')
    if pd.notna(diagnosis) and diagnosis:
        dx_text = str(diagnosis).replace('\n', '; ')
        if len(dx_text) > 150:
            dx_text = dx_text[:150]
    else:
        dx_text = "clinical condition"
    
    # Get hospital course details (extract first few meaningful sentences)
    hospital_course = row.get('hospital_course', '')
    course_detail = ""
    
    if pd.notna(hospital_course) and hospital_course and len(str(hospital_course)) > 50:
        course_text = str(hospital_course)
        
        # Find first few sentences
        sentences = course_text.replace('!', '.').replace('?', '.').split('.')
        course_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        if course_sentences:
            course_detail = '. '.join(course_sentences) + '.'
            if len(course_detail) > 250:
                course_detail = course_detail[:250]
    
    # Get disposition
    follow_up = row.get('follow_up', '')
    if pd.notna(follow_up) and follow_up:
        follow_lower = str(follow_up).lower()
        if 'home' in follow_lower:
            disposition = "discharged home with outpatient follow-up"
        elif 'rehab' in follow_lower or 'facility' in follow_lower or 'extended care' in follow_lower:
            disposition = "transferred to extended care facility"
        elif 'died' in follow_lower or 'expired' in follow_lower:
            disposition = "expired during hospitalization"
        else:
            disposition = "discharged with appropriate follow-up"
    else:
        disposition = "discharged from hospital"
    
    # Construct detailed narrative
    if course_detail:
        narrative = f"A {age}-year-old {gender} was admitted with {presenting}. Primary diagnoses included {dx_text}. {course_detail} Patient was {disposition}."
    else:
        narrative = f"A {age}-year-old {gender} was admitted with {presenting}. Primary diagnoses included {dx_text}. Clinical treatment was provided and patient was {disposition}."
    
    # Ensure length limit
    if len(narrative) > MAX_BRIEF_SUMMARY_LENGTH:
        narrative = narrative[:MAX_BRIEF_SUMMARY_LENGTH-3] + '...'
    
    return narrative

def create_structured_summary(row: pd.Series, gemini_model=None) -> str:
    """Create complete structured summary with detailed narrative"""
    age = row.get('age_at_admission', 'Unknown')
    gender = row.get('gender', 'Unknown')
    admission_date = format_date(row.get('admittime'))
    discharge_date = format_date(row.get('dischtime'))
    
    chief_complaint = row.get('chief_complaint', 'Not documented')
    medical_history = row.get('past_medical_history', 'Not documented')
    diagnosis = row.get('discharge_diagnosis', 'Not documented')
    hospital_course = row.get('hospital_course', 'Not documented')
    medications = row.get('discharge_medications', 'Not documented')
    follow_up = row.get('follow_up', 'Not documented')
    
    lab_results = row.get('lab_summary', 'Not available')
    if pd.notna(lab_results) and lab_results != 'Not available':
        lab_parts = str(lab_results).split(';')[:3]
        lab_results = '; '.join(lab_parts).strip()
    
    # Generate detailed narrative
    if gemini_model:
        detailed_narrative = generate_detailed_narrative_gemini(row, gemini_model)
    else:
        detailed_narrative = generate_enhanced_fallback(row)
    
    summary = SUMMARY_TEMPLATE.format(
        age=age,
        gender=gender,
        admission_date=admission_date,
        discharge_date=discharge_date,
        chief_complaint=truncate_text(chief_complaint, 80),
        medical_history=truncate_text(medical_history, 100),
        diagnosis=truncate_text(diagnosis, 100),
        hospital_course=truncate_text(hospital_course, 150),
        lab_results=truncate_text(lab_results, 80),
        medications=truncate_text(medications, 150),
        follow_up=truncate_text(follow_up, 80),
        detailed_narrative=detailed_narrative
    )
    
    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH-3] + '...'
    
    return summary

def prepare_summarization_dataset(input_file: str, output_dir: str, use_gemini: bool = True) -> Dict[str, str]:
    """Prepare train/val/test datasets with enhanced summaries"""
    
    print("="*60)
    print("DATA PREPARATION FOR BIOBART MODEL TRAINING")
    if use_gemini and GEMINI_AVAILABLE:
        print("Mode: Enhanced with Gemini API")
    else:
        print("Mode: Rule-based (enhanced)")
    print("="*60)
    print(f"Loading data from: {input_file}")
    
    # Initialize Gemini if available and requested
    gemini_model = None
    if use_gemini and GEMINI_AVAILABLE:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use gemini-pro instead of gemini-1.5-flash (stable API)
                gemini_model = genai.GenerativeModel('gemini-pro')
                print("✅ Gemini API initialized (gemini-pro)")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                print("Using rule-based fallback")
        else:
            print("Warning: GOOGLE_API_KEY not set. Using rule-based fallback.")
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} total records")
    print(f"Input has {len(df.columns)} columns")
    
    # Filter records
    print("\nFiltering records...")
    print("Criteria: text > 100 chars AND any clinical section present")
    
    has_diagnosis = (df['discharge_diagnosis'].notna()) & (df['discharge_diagnosis'] != '')
    has_medications = (df['discharge_medications'].notna()) & (df['discharge_medications'] != '')
    has_followup = (df['follow_up'].notna()) & (df['follow_up'] != '')
    has_chief = (df['chief_complaint'].notna()) & (df['chief_complaint'] != '')
    has_history = (df['past_medical_history'].notna()) & (df['past_medical_history'] != '')
    has_course = (df['hospital_course'].notna()) & (df['hospital_course'] != '')
    
    df_valid = df[
        (df['cleaned_text'].notna()) & 
        (df['cleaned_text'].str.len() > 100) &
        (has_diagnosis | has_medications | has_followup | has_chief | has_history | has_course)
    ].copy()
    
    print(f"Filtered to {len(df_valid)} records ({len(df_valid)/len(df)*100:.1f}%)")
    print(f"Removed {len(df) - len(df_valid)} incomplete records")
    
    # Create summaries with progress tracking
    print("\nCreating enhanced summaries...")
    if gemini_model:
        print("Using Gemini API (gemini-pro) for detailed clinical narratives")
        print(f"Estimated time: {len(df_valid) * 1.5 / 60:.1f} minutes for {len(df_valid)} records")
        print("Rate: ~1 request/second (to avoid rate limits)")
    else:
        print("Using enhanced rule-based extraction")
        print("Estimated time: 1-2 minutes")
    
    summaries = []
    api_errors = 0
    api_success = 0
    
    for idx in range(len(df_valid)):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(df_valid)} ({idx/len(df_valid)*100:.0f}%) | Success: {api_success} | Fallback: {api_errors}")
        
        row = df_valid.iloc[idx]
        
        try:
            summary = create_structured_summary(row, gemini_model)
            summaries.append(summary)
            
            # Track if Gemini was used successfully
            if gemini_model and 'A ' in summary and len(summary) > 800:
                api_success += 1
            
            # Rate limiting for Gemini (60 req/min = 1 per second to be safe)
            if gemini_model:
                time.sleep(1.0)  # 1 request per second
                
        except Exception as e:
            # Use fallback
            summary = create_structured_summary(row, gemini_model=None)
            summaries.append(summary)
            api_errors += 1
            if api_errors <= 5:  # Only print first 5 errors
                print(f"    Warning: Error on record {idx}, using fallback")
    
    df_valid['input_text'] = df_valid['cleaned_text']
    df_valid['target_summary'] = summaries
    
    df_valid['input_length'] = df_valid['input_text'].str.len()
    df_valid['summary_length'] = df_valid['target_summary'].str.len()
    df_valid['compression_ratio'] = df_valid['input_length'] / df_valid['summary_length']
    
    print(f"\n✅ Created {len(df_valid)} summaries")
    if gemini_model:
        print(f"   Gemini API successful: {api_success}")
        print(f"   Rule-based fallback: {api_errors}")
    print(f"   Avg input: {df_valid['input_length'].mean():.0f} chars")
    print(f"   Avg summary: {df_valid['summary_length'].mean():.0f} chars")
    print(f"   Compression: {df_valid['compression_ratio'].mean():.1f}x")
    
    # Split datasets
    print("\nSplitting into train/val/test (70/15/15)...")
    train_val, test_df = train_test_split(df_valid, test_size=0.15, random_state=42,
        stratify=df_valid['age_group'] if 'age_group' in df_valid.columns else None)
    train_df, val_df = train_test_split(train_val, test_size=0.176, random_state=42,
        stratify=train_val['age_group'] if 'age_group' in train_val.columns else None)
    
    print(f"Train: {len(train_df)} ({len(train_df)/len(df_valid)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df_valid)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df_valid)*100:.1f}%)")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'validation.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nSaved datasets to {output_dir}/")
    
    # Save sample summaries for review
    samples = []
    for idx in range(min(5, len(df_valid))):
        row = df_valid.iloc[idx]
        samples.append({
            'record_id': int(row['hadm_id']),
            'input_length': int(row['input_length']),
            'summary_length': int(row['summary_length']),
            'generated_summary': str(row['target_summary'])
        })
    
    with open(os.path.join(output_dir, 'sample_summaries.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    
    print("Saved sample summaries")
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return {'train': train_file, 'validation': val_file, 'test': test_file}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare data with enhanced clinical narratives')
    parser.add_argument('--input', type=str, 
        default='data-pipeline/data/processed/processed_discharge_summaries.csv',
        help='Input CSV file')
    parser.add_argument('--output', type=str, 
        default='model-development/data/model_ready',
        help='Output directory')
    parser.add_argument('--use-gemini', action='store_true',
        help='Use Gemini API for narratives (requires GOOGLE_API_KEY env var)')
    parser.add_argument('--no-gemini', action='store_true',
        help='Force rule-based narratives (no API)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input not found: {args.input}")
        exit(1)
    
    # Determine whether to use Gemini
    use_gemini_api = args.use_gemini and not args.no_gemini and GEMINI_AVAILABLE
    
    if use_gemini_api and not os.environ.get('GOOGLE_API_KEY'):
        print("Warning: --use-gemini specified but GOOGLE_API_KEY not set")
        print("Get API key from: https://aistudio.google.com/app/apikey")
        print("Falling back to rule-based summaries")
        use_gemini_api = False
    
    prepare_summarization_dataset(args.input, args.output, use_gemini=use_gemini_api)
    print("\n✅ Ready for training!")