import sys
import os
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_validation import validate_data

def test_validation_creates_report(tmp_path):
    report_path = tmp_path / "validation_report.csv"
    validate_data("data/processed/mimic_clean.csv", report_path)
    assert report_path.exists(), "❌ Validation report not created!"
    df = pd.read_csv(report_path)
    assert not df.empty, "❌ Validation report is empty!"
    assert "missing_values" in df.columns or "mean" in df.columns, "❌ Expected columns missing in validation report!"