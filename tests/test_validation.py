import pandas as pd

from scripts.data_validation import validate_data


def test_validation_creates_report(tmp_path):
    report_path = tmp_path / "validation_report.csv"
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "age_at_admission": [45, 67, None],
            "score": [1.0, 2.5, 3.0],
            "text": ["a", None, "c"],
        }
    ).to_csv(input_path, index=False)

    validate_data(input_path, report_path)
    assert report_path.exists(), " Validation report not created!"
    df = pd.read_csv(report_path)
    assert not df.empty, " Validation report is empty!"
    assert "missing_values" in df.columns or "mean" in df.columns, " Expected columns missing in validation report!"
