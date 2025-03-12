from bootcamp_ml_framework.data_processing import column_mean, multiply_dataframe, min_max_scale
import pandas as pd
from pandas import testing as pdt
import pytest


def test_success_column_mean():
    df = pd.DataFrame(
        {
            "A": [10, 20, 30, 40, 50],
            "B": [1, 2, 3, 4, 5]
        }
    )
    assert column_mean(df, "A") == 30
    assert column_mean(df, "B") == 3
    

def test_empty_column():
    df = pd.DataFrame({"A": []})

    assert pd.isna(column_mean(df, "A"))


def test_invalid_column():
    df = pd.DataFrame({"A": [1,2,3]})
    with pytest.raises(ValueError):
        column_mean(df, 1)


def test_missing_column():
    df = pd.DataFrame(
        {
            "A": [10, 20, 30, 40, 50],
            "B": [1, 2, 3, 4, 5]
        }
    )

    with pytest.raises(KeyError):
        column_mean(df, "C")
 


def test_multiply_dataframe():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    
    multiplier = 3
    expected_df = pd.DataFrame({
        "A": [3, 6, 9],
        "B": [12, 15, 18]
    })

    result_df = multiply_dataframe(df, multiplier)

    # Assert DataFrame equality
    pdt.assert_frame_equal(result_df, expected_df, check_dtype=False)

def test_multiply_dataframe_with_zero():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    expected_df = pd.DataFrame({
        "A": [0, 0, 0],
        "B": [0, 0, 0]
    })

    result_df = multiply_dataframe(df, 0)
    pdt.assert_frame_equal(result_df, expected_df, check_dtype=False)

def test_multiply_dataframe_with_negative():
    df = pd.DataFrame({
        "A": [1, -2, 3],
        "B": [-4, 5, -6]
    })

    expected_df = pd.DataFrame({
        "A": [-10, 20, -30],
        "B": [40, -50, 60]
    })

    result_df = multiply_dataframe(df, -10)
    pdt.assert_frame_equal(result_df, expected_df, check_dtype=False)

def test_multiply_dataframe_empty():
    df = pd.DataFrame()
    expected_df = pd.DataFrame()

    result_df = multiply_dataframe(df, 5)
    pdt.assert_frame_equal(result_df, expected_df)

def test_multiply_dataframe_invalid_multiplier():
    df = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        multiply_dataframe(df, "invalid") 


def test_min_max_scale():
    series = pd.Series([10, 20, 30, 40, 50])
    
    # Expected values after min-max scaling
    expected_series = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    
    result_series = min_max_scale(series)

    # Using pandas testing to assert Series equality
    pdt.assert_series_equal(result_series, expected_series, check_dtype=False, atol=1e-6)

def test_min_max_scale_empty_series():
    series = pd.Series([])
    result_series = min_max_scale(series)
    expected_series = pd.Series([])

    pdt.assert_series_equal(result_series, expected_series)  # Empty Series should remain unchanged

def test_min_max_scale_single_value():
    series = pd.Series([42])  # Single value should return 1.0
    result_series = min_max_scale(series)

    expected_series = pd.Series([1.0])
    pdt.assert_series_equal(result_series, expected_series, check_dtype=False)

def test_min_max_scale_identical_values():
    series = pd.Series([5, 5, 5, 5])  # All identical values
    result_series = min_max_scale(series)

    expected_series = pd.Series([1.0, 1.0, 1.0, 1.0])  # Should return all 1.0
    pdt.assert_series_equal(result_series, expected_series, check_dtype=False)