from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from bootcamp_ml_framework.preprocessing import preprocessing_utils as pre_utils

dict_stats = {
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
}


def get_silhouette_score(
    data: np.ndarray, clustering_results: np.ndarray, random_state: int
) -> float:
    """
    calculate the silhouette score for kmeans clustering predict results

    inputs
        data: input data from clustering algo
        clustering_results: results of the kmeans predict clustering algo
        random_state: random seed for repeatability
    """
    s_score = silhouette_score(
        X=data, labels=clustering_results, random_state=random_state
    )
    return s_score


def get_silhouette_samples_score(
    data: np.ndarray, clustering_results: np.ndarray
) -> np.ndarray:
    """
    calculate the silhouette score for each sample for kmeans clustering predict results

    inputs
        data: input data from clustering algo
        clustering_results: results of the kmeans predict clustering algo
        random_state: random seed for repeatability
    """
    s_scores = silhouette_samples(X=data, labels=clustering_results)
    return s_scores


def get_silhouette_samples_score_stats(s_scores: np.ndarray, stat_name) -> float:
    """
    calculate stat measures for the samples silhouette scores

    inputs
        s_scores: array of silhouette samples scores
        stat_name: name of stat to calculate
    """
    stat = dict_stats[stat_name](s_scores)
    return stat


def get_bic_scores_gmm_model(data: np.ndarray, gmm_model: GaussianMixture) -> float:
    """
    calculate bic scores using the fitted gmm model

    inputs
        data: input data from gmm clustering algo
        gmm_model: fitted gaussian mixture object
    outputs
        bic_scores: 1 dimensional ndarray of bic for the input samples
    """
    bic_score = gmm_model.bic(data)
    return bic_score


def get_mic_scores(
    data: pd.DataFrame, features: List[str], results_column_name: str
) -> Dict[str, float]:
    """
    calculates the mic scores for features specified in the features set

    inputs
        data: pandas dataframe with features used for the clustering
        features: list of features to calculate mic scores for
        result_column_name: column name of the clustering in the 'data' pandas dataframe
    outputs
        mic_scores: dict of mic scores for all features
    """
    mic_scores_dict = {}
    for feature in features:
        model = OLS.from_formula(
            str(feature) + "~" + str(results_column_name), data=data
        ).fit()

        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_model_ss = anova_table["sum_sq"][0]
        anova_model_residual = anova_table["sum_sq"][1]
        anova_total = anova_model_ss + anova_model_residual
        anova_eta_sq = anova_model_ss / anova_total

        mic_scores_dict[feature] = anova_eta_sq

    return mic_scores_dict


def get_r_square(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    calculate and return the r-squared correlation metric
    Parameters
    ----------
    y_true : true target values
    y_pred  : predicted target values

    Returns
    -------
     r_square score as a float
    """
    return metrics.r2_score(y_true=y_true, y_pred=y_pred)


def get_mean_squared_error(y_true, y_pred) -> float:
    """
    calculate and return the mean square error
    Parameters
    ----------
    y_true : true target values
    y_pred  : predicted target values

    Returns
    -------
    mean squared error as a float
    """
    return metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)


def get_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, use_absolute: bool = False
) -> np.ndarray:
    """
    calculates residuals from prediction processing
    Parameters
    ----------
    y_true : actual target values
    y_pred : predicted values
    use_absolute : if false return calculated difference and if true return absolute values

    Returns
    -------
    numpy matrix with the residuals, normalized, and standardized residuals of a prediction process
    """

    residuals = (y_true - y_pred) if use_absolute is False else np.abs(y_true - y_pred)
    stats = pre_utils.get_stats_array(
        input_data=residuals, stat_measures=["mean", "std"]
    )
    normalized_residuals = (residuals - stats["mean"]) / stats["std"]
    standardized_residuals = residuals / stats["std"]
    output_residuals = np.column_stack(
        (residuals, normalized_residuals, standardized_residuals)
    )

    return output_residuals


def get_counts_of_standardized_residuals(residuals: np.ndarray, num_std: int = 2):
    """
    Produces the cutoffs for the standardized residuals

    Parameters
    ----------
    residuals : array of residuals
    num_std : number of standard deviations to count past

    Returns
    -------
    percentage of residuals above an absolute cutoff
    """
    return np.sum(np.abs(residuals) > num_std) / len(residuals) * 100


def get_counts(input_data: pd.DataFrame, variable_list: List[str]) -> pd.DataFrame:
    """
    counts missing and filled data in the selected columns from the input data frame
    Parameters
    ----------
    input_data : input data frame
    variable_list : selected columns

    Returns
    -------
    counts_data_dfs : dataframe with two columns for each variable in the column list (filled, empty)
    """
    total_counts = len(input_data)
    counts_data_dfs = pd.DataFrame()

    for column in variable_list:
        missing_counts = [input_data[column].isna().sum()]
        counts_data_df = pd.DataFrame({f"{column}_missing": missing_counts})
        counts_data_df[f"{column}_filled"] = (
            total_counts - counts_data_df[f"{column}_missing"]
        )
        counts_data_dfs = pd.concat([counts_data_dfs, counts_data_df], axis=1)

    return counts_data_dfs


def get_residuals_maxmin(input_data: np.array) -> pd.DataFrame:
    """
    per model, per curve, max and min values
    Parameters
    ----------
    input_data : input array for residuals

    Returns
    -------
    two columns pandas dataframe with max and min values per model
    """
    get_residual_maxmin = pre_utils.get_stats_array(
        input_data=input_data, stat_measures=["max", "min"]
    )
    residual_maxmin_df = pd.DataFrame([get_residual_maxmin]).reset_index(drop=True)

    return residual_maxmin_df


def get_confusion_matrix(y_true: np.array, y_pred: np.array) -> np.ndarray:
    """
    calculates confusion matrix of predictions results
    Parameters
    ----------
    y_true : true values as numpy array
    y_pred : predicted values as numpy array

    Returns
    -------
    confusion matrix as numpy array
    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix


def get_classification_accuracy(y_true: np.array, y_pred: np.array) -> float:
    """
    calculates classification accuracy
    Parameters
    ----------
    y_true : true values as numpy array
    y_pred : preeicted values as numpy array

    Returns
    -------
    accuracy metric as float
    """
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    return accuracy


def get_f_score(y_true: np.array, y_pred: np.array, average: str = "macro") -> float:
    """
    calculates classification accuracy
    Parameters
    ----------
    y_true : true values as numpy array
    y_pred : true values as nump array
    average : specify the method used to average f1-score in case of multiclass/multilabel classification problem

    Returns
    -------
    f-score metric as a float
    """
    f_score = metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)
    return f_score


def get_precision(y_true: np.array, y_pred: np.array, average: str = "macro") -> float:
    """
    calculate classification precision
    Parameters
    ----------
    y_true : true values as numpy array
    y_pred : true values as nump array
    average : specify the method used to average f1-score in case of multiclass/multilabel classification problem

    Returns
    -------
    precision metric as float
    """
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, average=average)
    return precision
