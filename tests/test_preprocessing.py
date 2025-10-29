import sys
import os
import pandas as pd

# Add project root (Data-Pipeline) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocessing import basic_clean

def test_raw_file_exists():
    assert os.path.exists("data/raw/mimic_complete_with_demographics.csv"), "❌ Raw data file missing!"

def test_basic_clean_structure():
    df = pd.read_csv("data/raw/mimic_complete_with_demographics.csv")
    cleaned = basic_clean(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.shape[0] > 0, "❌ Cleaned dataframe is empty!"
    assert "age_at_admission" in cleaned.columns, "❌ Missing key column: age_at_admission"
    assert cleaned["age_at_admission"].between(18,120).all(), "❌ Invalid age values found!"