import pandas as pd
from typing import Union

def column_mean(df: pd.DataFrame, column_name: str) -> float:
    """

    Args:
        df (pd.DataFrame): _description_
        column_name (_type_): _description_

    Returns:
        float: _description_
    """
    if type(column_name) is not str:
        raise ValueError(f"Column name should be a string! but it is : {type(column_name)}")
    
    if column_name not in df.columns:
        raise KeyError(f"{column_name} does not exist in the input dataframe!")
    
    return df[column_name].mean()


def multiply_dataframe(df: pd.DataFrame, multiplier: Union[float, int]) -> pd.DataFrame:
    """Multiplies all numerical values in a DataFrame by a given integer."""
    if not isinstance(multiplier, (int, float)):
        raise ValueError("Multiplier must be a number.")
    return df * multiplier


def min_max_scale(series: pd.Series) -> pd.Series:
    """Scales a Pandas Series between 0 and 1 using min-max scaling."""
    if series.empty:
        return series  # Return an empty series unchanged
    min_val, max_val = series.min(), series.max()
    if min_val == max_val:
        return pd.Series([1.0] * len(series), index=series.index)  # Avoid division by zero
    return (series - min_val) / (max_val - min_val)