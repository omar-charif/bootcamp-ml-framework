from typing import List, Dict, Union

import numpy as np
import pandas as pd
from numpy import log, log10, log2

from multiprocessing import Pool
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler, LabelEncoder

from bootcamp_ml_framework.utils.logging_utils import get_logger

ROLLING_COLUMN_NAME_MODIFIER = "roll"
MISSING_DATA_CODE = -999.25

logger = get_logger(name="Enverus.DS.MLFramework.preprocessing")


def integer_encode(values: pd.Series):
    """
    takes a list-like object of categorical data and encodes it with integers

    Parameters
    ----------
    values: list-like object of categorical data

    Returns
    -------
    integer-encoded list
    """
    le = LabelEncoder()
    le.fit(values)

    return le.transform(values), le


transforms_dict = {
    "ln": log,
    "log10": log10,
    "log2": log2,
    "integer_encode": integer_encode,
}

stats_dict = {
    "median": np.nanmedian,
    "mean": np.nanmean,
    "std": np.nanstd,
    "variance": np.nanvar,
    "max": np.max,
    "min": np.min,
}


def transform_variable(
    input_data: pd.DataFrame, transform_namelist: List, transform_variable_list: List
):
    """
    Applies mathematical transforms to a variable (single column)

    inputs
        input_data: input df with variables you want transformed
        transfrom_name_list: list of strings with the transform keyword ;
        e.g. ["ln", "log10", "log2"]
        variable_list: list of variables you want transformed


    outputs
    input_data: df now with additional columns of transformed data
    """
    # TODO: consiliodate the transform variables function to check if the variables to transform
    transformation_dict = {}
    transformation_objects = {}
    # TODO: instead of zip, change config to be a dictionary of method: column
    method_variable_zip = zip(transform_namelist, transform_variable_list)
    for transform_name, variable_name in method_variable_zip:
        if transform_name in transforms_dict.keys():
            transformed_variable_name = variable_name + "_" + transform_name
            transformation_dict[variable_name] = transformed_variable_name
            transformed_values = transforms_dict[transform_name](
                input_data[variable_name]
            )
            if transform_name == "integer_encode":
                if transform_name not in transformation_objects.keys():
                    transformation_objects[transform_name] = {}
                # integer_encode returns 2 objects in a tuple: the transformed values and the transformation object
                input_data[transformed_variable_name] = transformed_values[0]
                transformation_objects[transform_name][
                    variable_name
                ] = transformed_values[1]
            else:
                input_data[transformed_variable_name] = transformed_values

        else:
            print("Transformation is not supported")
    return input_data, transformation_dict, transformation_objects


def check_contains(big_list: list, small_list: list) -> bool:
    """
    checks if small_list is in big_list
    Parameters
    ----------
    big_list : large list of item to examine
    small_list : small list of item to check if they are in the large one

    Returns
    -------
    true if small_list is in big_list and flase if not
    """
    big_set = set(big_list)
    small_set = set(small_list)

    return small_set.issubset(big_set)


def join_tables(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    parent_primary_key: List[str],
    child_foreign_key: List[str],
    parent_column_list: List[str],
) -> (pd.DataFrame, int):
    """
    joins two dataframes using parent child relationship based on primary and foreign keys
    Parameters
    ----------
    parent_df : parent dataframe that have 1 cardinality in 1 to many relationship
    child_df: child dataframe that have many cardinality in 1 to many relationship
    parent_primary_key : list primary key of the parent dataframe
    child_foreign_key : list foreign key of the child dataframe
    parent_column_list: parent column list to add to child columns

    Returns
    -------
    pandas dataframe resulting from join parent and child dfs
    """
    if check_contains(parent_primary_key, list(parent_df.columns)):
        return pd.DataFrame(), 1

    if check_contains(child_foreign_key, list(child_df.columns)):
        return pd.DataFrame(), 2

    parent_primary_key_sorted = sorted(parent_primary_key)
    child_foreign_key_sorted = sorted(child_foreign_key)
    if parent_primary_key_sorted != child_foreign_key_sorted:
        return pd.DataFrame(), 3

    updated_parent_column_list = parent_column_list + [
        x for x in parent_primary_key if x not in parent_column_list
    ]
    joined_df = parent_df[updated_parent_column_list].merge(
        child_df, left_on=parent_primary_key, right_on=child_foreign_key
    )

    return joined_df, 0


def hide_data(input_data, hide_dict: Dict[str, str]):
    """
     Hides input data based on keys and values in hide_dict

    inputs
        input_data: input df in which we want to hide some values
        hide_dict: a dictionary where the keys are columns with data
        to hide and the values are a flag, where values <0 indicate
        data should be hidden
    outputs
        input_data: the original input data with hidden values replaced
        with a missing value code
    """
    cleaned_input_data = input_data.copy(deep=True)
    for key in hide_dict.keys():
        if (hide_dict[key] in cleaned_input_data.columns) and (
            key in cleaned_input_data.columns
        ):
            cleaned_input_data.loc[
                cleaned_input_data[hide_dict[key]] < 0, key
            ] = MISSING_DATA_CODE
    return cleaned_input_data


def normalize_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    normalizes input data using mean and standard deviation

    inputs
        data: input df to normalize
    outputs
        normalized_data: normalized dataframe
    """
    normalized_data = StandardScaler().fit_transform(input_df)
    normalized_data_df = pd.DataFrame(normalized_data, columns=list(input_df.columns))
    return normalized_data_df


def normalize_data_variables(
    input_df: pd.DataFrame, variable_list: List[str]
) -> pd.DataFrame:
    """
    normlizes selected columns of the inputs data set

    inputs
        data: input dataframe to normalize
        variables_to_normalize: list of variables to normalize
    outputs
        normalized_data: partially normalized dataframe
    """
    normalized_data_df = input_df.copy(deep=True)
    normalized_data_df[variable_list] = (
        normalized_data_df[variable_list]
        - normalized_data_df[variable_list].mean(axis=0)
    ) / normalized_data_df[variable_list].std(axis=0)

    return normalized_data_df


def range_scaling_data_variables(
    input_df: pd.DataFrame, variables_to_normalize: List[str]
):
    """
    scales the a inputs variables by min and max
    Parameters
    ----------
    input_df : input data to to scale
    variables_to_normalize : list of variables to scale

    Returns
    -------
    scaled pandas dataframe
    """
    scaled_data_df = input_df.copy(deep=True)
    scaled_data_df[variables_to_normalize] = (
        scaled_data_df[variables_to_normalize]
        - scaled_data_df[variables_to_normalize].min(axis=0)
    ) / (
        scaled_data_df[variables_to_normalize].max(axis=0)
        - scaled_data_df[variables_to_normalize].min(axis=0)
    )

    return scaled_data_df


def remove_nan_data(
    input_df: pd.DataFrame, variable_list: List[str]
) -> (pd.DataFrame, pd.DataFrame):
    """
    removes nan from deleting rows with nan in columns of interest and returns one cleaned dataframe without nans and
    another without.
    Parameters
    ----------
    input_df : input dataframe to clean
    variable_list : list of columns to check for nans

    Returns
    -------
    cleaned dataframe and a dataframe with only rows with nan in columns of interest
    """
    data_with_nan = input_df[variable_list].isnull().any(axis=1)
    cleaned_df = input_df[~data_with_nan].reset_index(drop=True)
    data_with_nan = input_df[data_with_nan].reset_index(drop=True)
    return cleaned_df, data_with_nan


def split_integer_float_data(
    input_df: pd.DataFrame, variable_name: str
) -> (pd.DataFrame, pd.DataFrame):
    """
    splits input pandas dataframe based on the type of a variable
    Parameters
    ----------
    input_df : input dataframe
    variable_name : variable name to split data based on

    Returns
    -------
    two split dataframe
    """
    float_data_flag = input_df[variable_name] != input_df[variable_name].astype(int)

    float_data_df, int_data_df = split_data_bool_flag(
        input_df=input_df, indexed_bool_flag=float_data_flag
    )

    return int_data_df, float_data_df


def remove_bad_data(
    input_df: pd.DataFrame, flags_column_list: List[str], bad_data_keys: list = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    partition the input data set
    Parameters
    ----------
    input_df : input data to partition into good and bad data
    flags_column_list : columns list of the flags specify good and bad data
    bad_data_keys : a list of bad data keys

    Returns
    -------
    cleaned dataframe and a dataframe with only rows with nan in columns of interest
    """
    is_bad_data = input_df[flags_column_list].isin(bad_data_keys).any(axis=1)

    data_with_bad_data, cleaned_df = split_data_bool_flag(
        input_df=input_df, indexed_bool_flag=is_bad_data
    )

    return cleaned_df, data_with_bad_data


def split_data_bool_flag(
    input_df: pd.DataFrame, indexed_bool_flag: pd.Series
) -> (pd.DataFrame, pd.DataFrame):
    """
    extract datasets based on an indexed boolean flag
    Parameters
    ----------
    input_df : input  dataframe to divide into two partitions
    indexed_bool_flag : indexed boolean flag column

    Returns
    -------
    two pandas dataframes partitioned based on indexed_bool_flag column
    """
    true_data_df = input_df[indexed_bool_flag].reset_index(drop=True)
    false_data_df = input_df[~indexed_bool_flag].reset_index(drop=True)

    return true_data_df, false_data_df


def null_bad_data(
    input_df: pd.DataFrame,
    bad_data_column_flags: Dict[str, str],
    variable_list: List[str],
    bad_data_keys: list = None,
):
    """
    nulls data based on bad data flags and return them in the same dataframe
    Parameters
    ----------
    input_df : input dataframe to clean
    bad_data_column_flags : dict of flag and columns
    variable_list : list of variable to clean
    bad_data_keys : keys to identify bad data

    Returns
    -------
    cleaned pandas dataframe
    """
    input_data = input_df.copy(deep=True)
    for variable in variable_list:
        if variable in bad_data_column_flags:
            input_data.loc[
                input_data[bad_data_column_flags[variable]].isin(bad_data_keys),
                variable,
            ] = np.nan
        else:
            print("variable does not exist in the bad data column flag dict!")

    return input_data


def data_variables_group_by(
    input_df: pd.DataFrame, group_by_variable_list: List[str]
) -> object:
    """
    group by input data based on the selected gorup_by_column_list
    Parameters
    ----------
    input_df : input data set to group by
    group_by_variable_list : list of variables to group by the input data over

    Returns
    -------
    DataFrameGroupBy object
    """
    try:
        grouped_by_data = input_df.groupby(by=group_by_variable_list)

    except KeyError:
        error_out = "column does not exist"
        return error_out

    return grouped_by_data


def calculate_moving_window_data_stat(
    input_df: pd.DataFrame,
    variables: List[str],
    window: int = 1,
    window_type: str = None,
    stat_method: List[str] = "mean",
    quantile_per: float = 0.50,
) -> pd.DataFrame:
    """
    calculates moving stat for a set of features and adds the moving stat as new columns to the original dataframe
    Parameters
    ----------
    input_df : input data to calculate moving stat for
    variables : columns to calculate moving stat for
    window_type : weighting strategy of the moving window e.g. equally weighted sample or normally weighted samples
    window : size of the moving window
    stat_method : stat method to use e.g. sum, mean, median and std
    quantile_per : quantile percentage needed if quantile stat is slected
    Returns
    -------
    pandas dataframe with rooling statistics as new columns

    """
    # loop over all features
    for a_variable in variables:
        postfix = f"{ROLLING_COLUMN_NAME_MODIFIER}_{stat_method}"
        new_column_name = f"{a_variable}_{postfix}"
        if stat_method == "mean":
            input_df[new_column_name] = (
                input_df[a_variable]
                .rolling(window=window, win_type=window_type, min_periods=1)
                .mean()
            )

        elif stat_method == "median":
            input_df[new_column_name] = (
                input_df[a_variable]
                .rolling(window=window, win_type=window_type, min_periods=1)
                .median()
            )
        elif stat_method == "std":
            input_df[new_column_name] = (
                input_df[a_variable]
                .rolling(window=window, win_type=window_type, min_periods=1)
                .std()
            )
        elif stat_method == "sum":
            input_df[new_column_name] = (
                input_df[a_variable]
                .rolling(window=window, win_type=window_type, min_periods=1)
                .sum()
            )
        elif stat_method == "quantile":
            input_df[new_column_name] = (
                input_df[a_variable]
                .rolling(window=window, win_type=window_type, min_periods=1)
                .quantile(quantile=quantile_per, interpolation="nearest")
            )
    return input_df


def calculate_moving_stat_full_data_set(
    input_df: pd.DataFrame,
    variables: List[str],
    grouping_column: str,
    sorting_column: str,
    window: int = 1,
    window_type: str = None,
    stat_method: str = "mean",
    quantile_per: float = 0.75,
) -> (pd.DataFrame, dict):
    """
    calculates moving window for a dataset groupedby and sorted by a couple of columns
    Parameters
    ----------
    input_df : input data to calculate moving for
    variables : columns to calculate moving stat for
    grouping_column : column to be used to group by the input data
    sorting_column : column to be used to sort the group by data
    window : size of moving window
    window_type : weighting strategy of the moving window e.g. equally weighted sample or normally weighted samples
    stat_method : stat method to use e.g. sum, mean, median and std
    quantile_per : quantile percentage needed if quantile stat is slected

    Returns
    -------
    pandas dataframe with rooling statistics as new columns
    """
    grouped_df = input_df.sort_values([grouping_column, sorting_column]).groupby(
        grouping_column
    )

    data_with_moving_stat = grouped_df.apply(
        calculate_moving_window_data_stat,
        variables=variables,
        window=window,
        window_type=window_type,
        stat_method=stat_method,
        quantile_per=quantile_per,
    )
    var_dict = {}
    for var in variables:
        postfix = f"{ROLLING_COLUMN_NAME_MODIFIER}_{stat_method}"
        var_dict[var] = f"{var}_{postfix}"

    return data_with_moving_stat, var_dict


def trim_data_to_limits(
    input_data: pd.DataFrame, features_dict: Dict, variable_list: List[str]
) -> (pd.DataFrame, pd.DataFrame):
    """
    trims input data to limits set in the config, setting data outside the given ranges to the limit
    Parameters
    ----------
    input_data : sampled log data
    variable_list: list of variables to trim the data upon
    features_dict : from config file, these are the column names and upper/lower limits to restrict data

    Returns
    -------
    pd.Dataframe of input data, with data outside the given ranges set to the limits
    """
    original_input_data = pd.DataFrame(input_data)
    for variable in variable_list:
        if variable in features_dict:
            lower_limit = features_dict[variable]["min_value"]
            upper_limit = features_dict[variable]["max_value"]
            input_data.loc[input_data[variable] < lower_limit, variable] = lower_limit
            input_data.loc[input_data[variable] > upper_limit, variable] = upper_limit
        else:
            print(f"Missing {variable} in values range config dictionary!")
    return input_data, original_input_data


def extract_out_of_range_values(
    input_df: pd.DataFrame, features_dict: Dict[str, dict], variable_list: List[str],
) -> (pd.DataFrame, pd.DataFrame):
    """
    fills features out of range values with np.nan
    Parameters
    ----------
    input_df : input dataframe to transform
    variable_list: list of variable to clean the data upon
    features_dict : dictionary of features and min and max values used for clipping data

    Returns
    -------
    clipped pandas dataframe
    """
    # fill values of features with nan outside the min-max values
    out_of_range_data = pd.DataFrame()
    for variable in variable_list:
        if variable in features_dict:
            # where keeps values satisfying the specified conditions and set the remaining as nan
            min_value = features_dict[variable]["min_value"]
            max_value = features_dict[variable]["max_value"]

            # between function would compare np.nan to numbers. work around to ignore it is to fill nan with
            # mean values of bounds
            out_range_data = (
                ~input_df[variable]
                .fillna((min_value + max_value) / 2)
                .between(
                    features_dict[variable]["min_value"],
                    features_dict[variable]["max_value"],
                )
            )
            out_of_range_data = pd.concat(
                [out_of_range_data, input_df[out_range_data]], axis=0
            )
            input_df = input_df[~out_range_data]
        else:
            print(f"Missing {variable} in values range config dictionary!")

    input_df.reset_index(drop=True)
    out_of_range_data = out_of_range_data.reset_index(drop=True)

    return input_df, out_of_range_data


def null_out_of_range_values(
    input_df: pd.DataFrame,
    features_dict: dict = None,
    variable_list: List[str] = None,
    nulled_df_columns: List[str] = None,
) -> (pd.DataFrame, pd.DataFrame):
    """
    nulls the values in input_df
    Parameters
    ----------
    input_df : input data set to null value in.
    features_dict : features dict containing varaible ranges
    variable_list : list of variables to filter data upon
    nulled_df_columns : columns to return from input_df for nulled rows

    Returns
    -------
    Dataframe with np.nan for out of range values
    """
    input_data = input_df.copy(deep=True)
    full_out_of_range_data = pd.DataFrame(columns=nulled_df_columns)
    for variable in variable_list:
        if variable in features_dict:
            # where keeps values satisfying the specified conditions and set the remaining as nan
            min_value = features_dict[variable]["min_value"]
            max_value = features_dict[variable]["max_value"]

            out_range_data = (
                ~input_data[variable]
                .fillna((min_value + max_value) / 2)
                .between(
                    features_dict[variable]["min_value"],
                    features_dict[variable]["max_value"],
                )
            )

            if nulled_df_columns is not None:
                full_out_of_range_data = pd.concat(
                    [
                        full_out_of_range_data,
                        input_df.loc[out_range_data, nulled_df_columns],
                    ],
                    axis=0,
                )

            input_data.loc[out_range_data, variable] = np.nan

    return input_data, full_out_of_range_data


def handle_out_of_range_values(
    input_df: pd.DataFrame,
    features_dict: Dict[str, dict],
    variable_list: List[str],
    set_to_limits: bool = True,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Wrapper function for handling data outside of ranges set in config_rt
    trim_data_to_limits will replace the out-of-range values with the max or min value
    features_filler will replace out-of-range values with np.nan
    Parameters
    ----------
    input_df : input dataframe to transform
    features_dict : dictionary of features and min and max values used for clipping data
    variable_list : list of variable for out of range handling
    set_to_limits : bool to determine which process to follow. 'False' sets out-of-range values to np.nan
    Returns

    -------
    trimmed pandas dataframe
    """

    if set_to_limits is True:
        trimmed_data, additional_out_df = trim_data_to_limits(
            input_df, features_dict, variable_list=variable_list
        )
    else:
        trimmed_data, additional_out_df = extract_out_of_range_values(
            input_df, features_dict, variable_list=variable_list
        )
    return trimmed_data, additional_out_df


def get_stats_array(input_data: np.ndarray, stat_measures: List[str]):
    """
    calculates a set of stats measure for numpy array
    Parameters
    ----------
    input_data : input numpy array
    stat_measures : list of stats measures to return

    Returns
    -------
    dict of stat measures and values
    """
    results_dict = {}
    for stat in stat_measures:
        if stat in stats_dict:
            results_dict[stat] = stats_dict[stat](input_data)
        else:
            print(f"{stat} is not a supported statistic measure!")
    return results_dict


def get_stats(df: pd.DataFrame, varaible_list: List[str], percentile_list: List[str]):
    """
    Ported from qc_utils.py 20181205
    Get basic Descriptive Statistics on a dataframe for a chosen
    set of variables

    FUTURE -- support custom calculations of descriptive stats

    Arguments:
        df - a data frame
        varaible_list - list of variable names as strings
        percentile_list - list of percentiles to calculate

    EXAMPLE:
        In [1]: datafilename
        Out[1]: 'Appalachia_LowerMarcellus_DensityPorosity.csv.dat'

        In [2]: datadf.head()
        Out[2]:
            long        lat  variable
            0 -83.236854  37.885122  0.099942
            1 -83.122744  37.890799  0.100085
            2 -83.008615  37.896357  0.100300
            3 -82.894468  37.901795  0.100634
            4 -82.780304  37.907115  0.101146

        In [3]: datadf.tail()
        Out[3]:
            long        lat  variable
            5 -82.666121  37.912315  0.101855
            6 -82.551922  37.917396  0.102680
            7 -82.437706  37.922358  0.103359
            8 -82.323474  37.927201  0.103357
            9 -82.209225  37.931924  0.101998

        In [4]: stats = stats = qc_utils.get_stats(df=datadf, columnslist=['long', 'lat','variable'],
        percentileslist=[0.05,0.1,0.25,0.5,0.75,0.9,0.95])
        In [5]: stats
        Out[5]:
                        long        lat   variable
                count  10.000000  10.000000  10.000000
                mean  -82.723143  37.909238   0.101536
                std     0.345701   0.015751   0.001308
                min   -83.236854  37.885122   0.099942
                5%    -83.185505  37.887677   0.100006
                10%   -83.134155  37.890231   0.100071
                25%   -82.980078  37.897717   0.100384
                50%   -82.723212  37.909715   0.101500
                75%   -82.466260  37.921118   0.102510
                90%   -82.312049  37.927673   0.103357
                95%   -82.260637  37.929799   0.103358
                max   -82.209225  37.931924   0.103359

        In [6]: allstats.query("index in ['min','max','mean', 'std']")
        Out[6]:
                        long        lat  variable
               mean -82.723143  37.909238  0.101536
               std    0.345701   0.015751  0.001308
               min  -83.236854  37.885122  0.099942
               max  -82.209225  37.931924  0.103359


    """
    statsdf = df[varaible_list]
    stats = statsdf.describe(percentiles=percentile_list, include="all")
    return stats


def create_repeating_groups(df, mnemonics_rsf_dict, tolerance_dict):
    """
    taken from laskit. creates groups of repeating values where consecutive values do not meet a given tolerance

    Parameters
    ----------
    df
    mnemonics_rsf_dict: {"RHOB_E": "RHOB_F", ...}
    tolerance_dict: dict of diff() thresholds per mnemonic to treat consecutive rows as repeats

    Returns
    -------
    df with mnemonic repeat groups added as columns
    """
    for mnemonic, _ in mnemonics_rsf_dict.items():
        if mnemonic in tolerance_dict.keys():
            tolerance = tolerance_dict[mnemonic]
        else:
            tolerance = 0
        # take the difference of each column element by element and then shift column up
        if mnemonic in df.columns:
            # this flags nan's as repeats. this is handled later in flag_repeating_groups
            df[f"{mnemonic}_group"] = (
                np.abs(df[mnemonic] - df[mnemonic].shift()) > tolerance
            ).cumsum()
    return df


def flag_repeating_groups(df, mnemonics_rsf_dict, repeat_flag_id, repeat_count):
    """
    modified from laskit. looks at the size of repeating groups and
    flags as repeated data if repeat_count threshold is met

    Parameters
    ----------
    df
    mnemonics_rsf_dict: {"RHOB_E": "RHOB_F", ...}
    repeat_flag_id: typically 2, used to indicate a repeated mnemonic
    repeat_count: number of rows required to be a repeat group

    Returns
    -------
    df with modified _F (or new) columns. 0's and 2's in existing columns are replaced with
    new 2's where the size of the repeating group exeeds the repeat_count
    """
    for mnemonic, mnemonic_flag in mnemonics_rsf_dict.items():
        # take the difference of each column element by element and then shift column up
        if mnemonic in df.columns:
            # ok bear with me on this one. this counts the number of rows per group of values within 'tolerance'
            # specified in the repeat_flag process
            # it then converts False to 0 and True to the repeat_flag_id (typically 2)
            group_counts = (
                (df.groupby(f"{mnemonic}_group")[mnemonic].agg("count") >= repeat_count)
                .replace(True, repeat_flag_id)
                .astype(int)
                .reset_index()
            )
            group_counts = group_counts.rename(columns={mnemonic: mnemonic_flag})

            # ignore nan's (these aren't repeating groups)
            group_counts.loc[group_counts[mnemonic_flag].isnull(), mnemonic_flag] = 0

            # if the _F column already exists, replace the 0's in the _F column
            if mnemonic_flag in df.columns:
                # merge the modified repeat stats,
                mnemonic_flag_repeat_modified = f"{mnemonic_flag}_repeat_modified"
                group_counts = group_counts.rename(
                    columns={mnemonic_flag: mnemonic_flag_repeat_modified}
                )
                df = df.merge(group_counts, how="left", on=f"{mnemonic}_group")

                # we don't want to replace the 1's in the current mnemonic_flag column
                df.loc[df[mnemonic_flag].isin([0, 2]), mnemonic_flag] = df.loc[
                    df[mnemonic_flag].isin([0, 2]), mnemonic_flag_repeat_modified
                ]
                del df[mnemonic_flag_repeat_modified]

            else:  # this is usually just for GR, which won't already have an _F column
                # merge the F flag
                df = df.merge(group_counts, how="left", on=f"{mnemonic}_group")

            del df[f"{mnemonic}_group"]

    return df


def overwrite_repeat_flag(
    df, bad_data_dict, overwrite_repeat_flag_count, repeat_flag_tolerance_dict
) -> pd.DataFrame:
    """

    Parameters
    ----------
    df
    bad_data_dict: {"RHOB_E": "RHOB_F", ...}
    overwrite_repeat_flag_count: number of rows to treat as a repeat outlier
    repeat_flag_tolerance_dict: dict of diff() thresholds per mnemonic to treat consecutive rows as repeats

    Returns
    -------

    """
    df_repeating_groups = create_repeating_groups(
        df=df,
        mnemonics_rsf_dict=bad_data_dict,
        tolerance_dict=repeat_flag_tolerance_dict,
    )
    df = flag_repeating_groups(
        df=df_repeating_groups,
        mnemonics_rsf_dict=bad_data_dict,
        repeat_flag_id=2,
        repeat_count=overwrite_repeat_flag_count,
    )

    return df


def get_columns_bins(
    input_df: pd.DataFrame,
    variable: str,
    values_cuts: Union[List[float], List[int], int],
    labels: List[str],
    binning_method: str = "quantile",
) -> pd.DataFrame:
    """

    Parameters
    ----------
    input_df : input_df with the columns to be binned
    variable : list of columns to be binned
    values_cuts : int or list of percentiles or quantiles
    labels : labels to give for bins
    binning_method : method of variable binning to use (use quantile or custom cut

    Returns
    -------
    pandas dataframe with binned columns
    """
    if values_cuts == 0 and len(values_cuts) == 0:
        return pd.DataFrame()
    else:
        if binning_method == "quantile":
            results = pd.qcut(input_df[variable], q=values_cuts, labels=labels)
        else:
            results = pd.cut(input_df[variable], bins=values_cuts, labels=labels)

    return results


def get_bins_variable(
    input_df: pd.DataFrame,
    variable_list: List[str],
    values_cuts: Union[List[float], List[int], int],
    use_quantiles: bool = True,
) -> (pd.DataFrame, Dict[str, str]):
    """
    creates a binned variable
    Parameters
    ----------
    input_df : input_df with the columns to be binned
    variable_list : list of columns to be binned
    values_cuts : int or list of percentiles or quantiles
    use_quantiles : enable/disable using quantile to calculate column bins

    Returns
    -------
    pandas dataframe with binned variables
    """
    input_df = input_df.copy(deep=True)
    if len(input_df) == 0:
        return (pd.DataFrame(), {})

    if use_quantiles is True:
        binning_method = "quantile"
    else:
        binning_method = "custom"

    variable_tranformation_mapping = {}
    for variable in variable_list:

        if type(values_cuts) is int:
            labels = list(range(1, values_cuts + 1, 1))
        else:
            labels = list(range(1, len(values_cuts), 1))

        input_df[f"{variable}_binned"] = get_columns_bins(
            input_df=input_df,
            variable=variable,
            values_cuts=values_cuts,
            labels=labels,
            binning_method=binning_method,
        )

        variable_tranformation_mapping[variable] = f"{variable}_binned"

    return input_df, variable_tranformation_mapping


def binary_encoder(input_df: pd.DataFrame, variable_list: List[str]) -> pd.DataFrame:
    """
    encodes a set of variables to a set of binary columns
    Parameters
    ----------
    input_df : input pandas dataframe
    variable_list : list of variables to binary encode (one hot encode)

    Returns
    -------
    pandas dataframe with encoded variables
    """
    if len(input_df) == 0:
        return pd.DataFrame()

    input_df = input_df.copy(deep=True)

    column_name_dict = {}
    # take of copy of the column
    for column in variable_list:
        new_column_name = f"backup_{column}"
        input_df[new_column_name] = input_df[column]
        column_name_dict[new_column_name] = column

    updated_df = pd.get_dummies(data=input_df, columns=variable_list)
    updated_df.rename(columns=column_name_dict, inplace=True)
    return updated_df


def run_function_parallel(
    function: callable,
    params_sets: List[tuple],
    multi_processing: bool = False,
    number_of_cpus: int = None,
) -> list:
    """Reads multiple file defined by paths into one DataFrame

    Parameters
    ----------
    function : object, required
        function to run in parallel
    params_sets : list[tuple], required
        list of the function argument set. Each item of this list is a tuple of arguments for 'function'.
        The order of the arguments should be kept intact.
    multi_processing : bool, optional
        disable/enable multiprocessing
    number_of_cpus : int, optional
        number of cpu to use to run input function in parallel. When set to -1, all available cpus will be used

    Returns
    -------
    list of results from the input function on the params set
    """
    print(f"number of cpus in multi_processing function:{number_of_cpus}")
    if multi_processing is True:
        with Pool(processes=number_of_cpus) as pool:
            results = pool.starmap(function, params_sets)
    else:
        results = [function(*params_set) for params_set in params_sets]

    return results


def get_ks_stat_one_id(
    df,
    id_col,
    one_id,
    col,
    ks_reject_level,
    df_compare=None,
    sample_df_compare=True,
    n_samples=10000,
):
    """

    Parameters
    ----------
    df: needs `id_col` and `col` as columns
    id_col: column containing identifiers such as well apis
    one_id: one identifier (such as a well's api)
    col: column containing values whose distribution you want to assess
    ks_reject_level: float, [0-1]. values above this number indicate the distribution are too far from the global
    distribution and will likely be removed/handled in other functions
    df_compare: needs `col` as a column, if not None

    Returns
    -------
    If ks stat is > ks reject level, tuple of (one_id, col, ks statistic), else None.
    """

    df_one_api = df.loc[
        (df[id_col] == one_id) & (df[col] != -999.25) & ~(df[col].isnull()), col,
    ].values
    if df_compare is None:
        df_compare = df.loc[
            (df[id_col] != one_id) & (df[col] != -999.25) & ~(df[col].isnull()), col,
        ].values
    else:
        df_compare = df_compare.loc[
            (df_compare[col] != -999.25) & ~(df_compare[col].isnull()), col,
        ].values
    if sample_df_compare is True and len(df_compare) > n_samples:
        df_compare = np.random.choice(df_compare, n_samples)

    if len(df_one_api) > 0:
        ks = ks_2samp(df_one_api, df_compare)

        if ks.statistic > ks_reject_level:
            return (one_id, col, ks.statistic)

    return None


def get_ks_stat_rejects(
    df, id_col, cols_to_analyze, ks_reject_level, num_processes, df_compare=None
):
    """

    Parameters
    ----------
    df: dataframe
    id_col: column name containing IDs. each unique value in this column will be treated as one distribution when
     evaluated for ks stat
    cols_to_analyze: columns to evaluate KS stat for
    ks_reject_level: float, [0-1]. values above this number indicate the distribution are too far from the global
    distribution and will likely be removed/handled in other functions
    num_processes: number of CPUs to use

    Returns
    -------
    list of tuples where a one_id/col pair was above ks_reject_level; each tuple contains (one_id, col, ks statistic)

    """
    df_data = df[[id_col] + cols_to_analyze].dropna()

    # compile list of parameters to pass to multiprocessing
    ks_search_params = []
    if len(df_data) > 0:
        for col in cols_to_analyze:
            for one_id in df_data[id_col].unique():
                ks_search_params.append(
                    tuple([df_data, id_col, one_id, col, ks_reject_level, df_compare])
                )

    print("Calculating KS values...")
    # compare each curve for each api against other curves
    ks_results = run_function_parallel(
        function=get_ks_stat_one_id,
        params_sets=ks_search_params,
        multi_processing=True,
        number_of_cpus=num_processes,
    )
    ks_results = [x for x in ks_results if x is not None]

    # convert to dict
    ks_results_dict = {}
    for result in ks_results:
        col = result[1]
        one_id = result[0]
        if col not in ks_results_dict:
            ks_results_dict[col] = []
        ks_results_dict[col].append(one_id)

    return ks_results_dict


# TODO: move to outliers detection module
def null_bad_ks_stat(
    data: pd.DataFrame,
    id_col: str,
    cols_to_analyze: list,
    ks_reject_level: float,
    num_processes: int,
    depth_col: str = None,
    df_compare: pd.DataFrame = None,
):
    """
    nulls outliers based on comparing log curves distributions among wells
    Parameters
    ----------
    data : pandas df containing wells log data
    cols_to_analyze : column to assess outliers based
    ks_reject_level : float, [0-1]. values above this number indicate the distribution are too far from the global
    distribution and will be removed
    num_processes: number of cpus to use during multiprocessing
    depth_col: name of the depth column
    df_compare: if using comparison data in another dataframe, add it here. must have cols_to_analyze

    Returns
    -------
    pandas dataframe with outliers well values nulled.
    """
    df = data.copy(deep=True)

    # Check K-S value to see if the curve matches any existing curve
    ks_results = get_ks_stat_rejects(
        df, id_col, cols_to_analyze, ks_reject_level, num_processes, df_compare
    )

    bad_api_list = []
    for x in ks_results.values():
        bad_api_list.extend(x)
    bad_api_list = list(set(bad_api_list))

    # null api/column pairs with high KS values (entire distribution of data is likely untrustworthy)
    for col, api_list in ks_results.items():
        logger.info("KS stat check: removing %f ids in %s", int(len(api_list)), col)
        df.loc[df[id_col].isin(api_list), col] = np.nan

    reported_null_cols = [id_col]
    if depth_col is not None:
        reported_null_cols.append(depth_col)

    # record which apis were nulled (and the associated depths)
    nulled_apis_df = df.loc[
        df[id_col].isin(bad_api_list), reported_null_cols
    ].drop_duplicates()

    return df, nulled_apis_df


def null_bad_preds_with_ks(
    df: pd.DataFrame,
    id_col: str,
    model_specific_cols: list,
    y_variable: str,
    ks_null_pred_external: float,
    num_processes: int = 1,
):
    """

    Parameters
    ----------
    df: input data
    model_specific_cols: predicted and residual curves that model produced
    y_variable: target variable
    ks_null_pred_external: ks statistic threshold (above this value gets thrown out)

    Returns
    -------
    df, with nulls for predicted curve & residuals where the distribution didn't match
    """

    #

    pred_curve_name = model_specific_cols[0]

    df_compare = df.loc[df["data_class"].isin([0, 1]), y_variable].to_frame()
    df_compare = df_compare.rename(columns={y_variable: pred_curve_name})

    df, nulled_apis = null_bad_ks_stat(
        data=df,
        id_col=id_col,
        cols_to_analyze=[pred_curve_name],
        ks_reject_level=ks_null_pred_external,
        num_processes=num_processes,
        df_compare=df_compare,
    )

    # null the prediction and residual and change data_is_nulled to 1
    df.loc[df[id_col].isin(nulled_apis[id_col]), model_specific_cols] = -999.25
    df.loc[df[id_col].isin(nulled_apis[id_col]), "data_is_nulled"] = 1

    return df


def main():
    path = r""
    pd.read_csv(path)


if __name__ == "__main__":
    main()
