import pandas as pd
from scripts.preprocessing import basic_clean


def test_basic_clean_structure():
    df = pd.DataFrame(
        {
            "age_at_admission": [45, 67, 150, 35, -5, None],
            "note": ["a", "b", "c", "d", "e", "f"],
        }
    )
    cleaned = basic_clean(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.shape[0] > 0, " Cleaned dataframe is empty!"
    assert "age_at_admission" in cleaned.columns, " Missing key column: age_at_admission"
    assert cleaned["age_at_admission"].between(18, 120).all(), " Invalid age values found!"
