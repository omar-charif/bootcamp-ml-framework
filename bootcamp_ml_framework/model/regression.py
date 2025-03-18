import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor

from bootcamp_ml_framework.postprocessing import metrics

Model = namedtuple("Model", "name model_family model_type predict_type")

logger = logging.getLogger(name="Prediction module")

metrics_processes_dict = {
    "r_squared": metrics.get_r_square,
    "mean_squared_error": metrics.get_mean_squared_error,
    "confusion_matrix": metrics.get_confusion_matrix,
    "f_score": metrics.get_f_score,
    "accuracy": metrics.get_classification_accuracy,
    "precision": metrics.get_precision,
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb


class Regressor(ABC):
    def __init__(
        self,
        name: str,
        model_family: str,
        random_state: int,
        max_iteration: int = 1,
        is_single_predict_model: bool = True,
        n_jobs=1,
        **model_configs,
    ):
        """
        creates a prediction object
        Parameters
        ----------
        random_state : seed used when performing any random operation
        name : prediction model name
        model_family : family of the prediction model (e.g. neural network, trees)
        is_regression : specify if the model is a regression or classification model
        max_iteration : maximum number of iteration for training
        n_jobs : number of parallel jobs that the model can use
        """
        self.random_state = random_state


        model_type = "regression"


        if is_single_predict_model is True:
            predict_type = "single"
        else:
            predict_type = "multi"

        self.model_details = Model(
            name=name,
            model_family=model_family,
            model_type=model_type,
            predict_type=predict_type,
        )
        self.max_iteration = max_iteration
        self.model_additional_configs = model_configs
        self.n_jobs = n_jobs
       
        self.available_metrics_list = [
            "r_squared",
            "mean_squared_error",
            "residuals",
        ]

        self.model = None
        self.fitted_model = None
        self.metrics_dict = {}
        self.training_set_true = None
        self.training_set_pred = None
        self.early_stopping_enabled = False
        self.early_stopping_params = {}
        self.grid_search_enabled = False
        self.model_params = {}

    @abstractmethod
    def get_features_ranking(self) -> dict:
        """
        return features ranking
        Returns
        -------
        dict of features as keys and ranking as values
        """

    def get_performance_metrics(self, metrics_list: List[str]):
        if len(metrics_list) == 0:
            return self.metrics_dict
        else:
            for a_metric in metrics_list:
                if a_metric not in self.available_metrics_list:
                    logging.exception("metric unavailable for this model!")
                else:
                    self.metrics_dict[a_metric] = metrics_processes_dict[a_metric](
                        y_true=self.training_set_true, y_pred=self.training_set_pred
                    )

        return self.metrics_dict

    @classmethod
    def train_test_split_by_unique_id(
        cls, input_data, train_size, random_state, id_column
    ):
        """
        splits dataset by api_col - selects a 'train_size' fraction of unique apis for the training set
        this is desirable when you're dealing with groups of data that could leak information to the test set

        Parameters
        ----------
        input_data : input dataframe to split
        train_size : percentage of unique id's to use as training set
        random_state : random seed
        id_column : identifier column to split groups of associated data (such as API/UWI)

        Returns
        -------

        """
        nprs = np.random.RandomState(
            random_state
        )  # preferred to np.random.seed() by np's creator

        # TODO: make y optional

        # get count of number of wells
        num_wells = input_data[id_column].nunique()

        # select apis for the training set
        train_apis = nprs.choice(
            input_data[id_column].unique(), int(num_wells * train_size)
        )

        train_df = input_data[input_data[id_column].isin(train_apis)]
        test_df = input_data[~input_data[id_column].isin(train_apis)]

        return train_df, test_df

    @classmethod
    def get_train_test_data(
        cls,
        input_data: pd.DataFrame,
        train_percentage: float = 0.75,
        random_state: int = 1,
        method: str = "standard",
        id_column: str = "",
    ):
        """
        split the input data into train and test parts
        Parameters
        ----------
        input_data : input_dataframe to split
        train_percentage : percentage to use for train
        random_state : seed to use for running the model
        method : 'standard' (random row-wise split) or 'id_split' (splits by id_column groups)
        id_column : if `method` is `id_split`, splits data based on unique id's in this col

        Returns
        -------
        training and test dataframes
        """
        if method == "standard":
            return train_test_split(
                input_data, train_size=train_percentage, random_state=random_state
            )
        elif method == "id_split":
            if id_column != "":
                return cls.train_test_split_by_unique_id(
                    input_data,
                    train_size=train_percentage,
                    random_state=random_state,
                    id_column=id_column,
                )
            else:
                print("'id_split' method selected but 'id_column' not defined")

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, **fit_configs: dict) -> Model:
        """
        fits a machine learning model a created a fitted model
        Parameters
        ----------
        x_data :  numpy array of input X_data for the model fitting
        y_data : numpy array of target data
        fit_configs : dict of additional parameters for fitting the model
        Returns
        -------
        return a fitted model
        """

        self.fitted_model = self.model.fit(x_data, y_data, **fit_configs)
        self.training_set_true = y_data
        self.training_set_pred = self.model.predict(x_data)
        return self.fitted_model

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        res_array = self.fitted_model.predict(x_data)
        return res_array

    def optimize_hyperparameters(
        self,
        x_data,
        y_data,
        params_dict,
        sample=True,
        cv_count=3,
        score_function="neg_mean_absolute_error",
    ):
        if sample is True:
            selected_indices = random.sample(
                range(0, len(x_data)), int(len(x_data) * 0.2)
            )
            x_data_final = x_data[selected_indices]
            y_data_final = y_data[selected_indices]
        else:
            x_data_final = x_data
            y_data_final = y_data

        gs = GridSearchCV(self.model, params_dict, cv=cv_count, scoring=score_function)

        gs.fit(x_data_final, y_data_final)

        print(f"\nBest Params --> {gs.best_params_}")
        print(f"Best MAE --> {-gs.best_score_: .3f}")
        print(
            f"Best r_squared --> {metrics.get_r_square(y_true=y_data, y_pred=gs.predict(x_data)): .3f}"
        )

        return gs, gs.best_params_

    @abstractmethod
    def create_object_model(self):
        """
        """

    def update_model_params(self, new_params: dict) -> dict:
        """
        """
        for param in new_params:
            self.model_params[param] = new_params[param]
        self.create_object_model()

    def process_early_stopping(
        self,
        test_df: pd.DataFrame = None,
        explanatory_variables: list = None,
        target_variable: list = None,
    ):
        # explanatory  and target variables are printed just to deal with the linting error.
        print(
            f"process_early_stopping is not available for this model!{test_df.shape}{explanatory_variables}"
            f"{target_variable}"
        )
        return None


class LinearRegressor(Regressor):
    def __init__(self, random_state: int = 1, n_jobs: int = 1):
        """
        create a linear regression model
        Parameters
        ----------
        random_state : seed for random model
        """
        super().__init__(
            name="Linear_Regression",
            model_family="Regression",
            random_state=random_state,
            max_iteration=1,
            n_jobs=n_jobs,
        )

        self.model_params["n_jobs"] = n_jobs
        self.create_object_model()

    def create_object_model(self):
        self.model = LinearRegression(self.model_params)

    def get_features_ranking(self) -> dict:
        # the code of features ranking is to be determined
        pass


class ElasticNetRegressor(Regressor):
    def __init__(
        self, random_state, n_jobs: int = 1, alpha: int = 0, l1_ratio: float = 0.5
    ):
        """
        create a linear regression model
        Parameters
        ----------
        random_state : seed for random model
        """
        super().__init__(
            name="ElasticNet_Regression",
            model_family="Regression",
            random_state=random_state,
            is_regression=True,
            max_iteration=1,
            n_jobs=n_jobs,
        )

        self.model_params["alpha"] = alpha
        self.model_params["random_state"] = random_state
        self.model_params["l1_ratio"] = l1_ratio
        self.create_object_model()

    def create_object_model(self):
        if self.model_params["alpha"] == 0:

            self.model = LinearRegressor(
                n_jobs=self.n_jobs, random_state=self.model_params["random_state"]
            )
        else:
            self.model = ElasticNet(
                alpha=self.model_params["alpha"],
                l1_ratio=self.model.model_params["l1_ratio"],
                random_state=self.model_params["random_state"],
            )

    def get_features_ranking(self) -> dict:
        # the code of features ranking is to be determined
        pass


class RandomForestRegression(Regressor):
    def __init__(
        self,
        n_estimators: int = 10,
        random_state: int = 1,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features=3,
        n_jobs: int = 1,
        early_stopping_enabled: bool = False,
        grid_search_enabled: bool = False,
        grid_search_params: dict = None,
        **model_configs,
    ):
        """
        creates a random forest object and fill in all attributes
        Parameters
        ----------
        nb_estimators : nunber of tree to use in the random forest
        random_state : seed to use for drawing random numbers
        max_depth : max depth of a tree in the forest
        min_samples_split : minimum number of leaf needed for split a node
        """
        super().__init__(
            name="Random_Forest",
            model_family="tree_regressor",
            random_state=random_state,
            n_jobs=n_jobs,
            **model_configs,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.early_stopping_enabled = early_stopping_enabled
        self.grid_search_enabled = grid_search_enabled
        self.grid_search_params = grid_search_params
        self.model_params["n_estimators"] = n_estimators
        self.model_params["max_depth"] = max_depth
        self.model_params["min_samples_split"] = min_samples_split
        self.model_params["max_features"] = max_features
        self.model_params["random_state"] = random_state
        self.model_params["n_jobs"] = n_jobs
        self.create_object_model()

    def create_object_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.model_params["n_estimators"],
            max_depth=self.model_params["max_depth"],
            min_samples_split=self.model_params["min_samples_split"],
            max_features=self.model_params["max_features"],
            n_jobs=self.model_params["n_jobs"],
            random_state=self.model_params["random_state"],
            oob_score=True,
            **self.model_additional_configs,
        )

    def get_features_ranking(self):
        # the code of features ranking is to be determined
        pass


class NNRegressor(Regressor):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        learning_rate="adaptive",
        learning_rate_init=0.1,
        max_iter=200,
        n_iter_no_change=10,
        random_state=1,
        n_jobs: int = 1,
        early_stopping_enabled=False,
        early_stopping_params: dict = None,
        grid_search_enabled=False,
        grid_search_params: dict = None,
        **model_configs,
    ):
        """
        creates a simple neural networkobject and fill in all attributes
        Parameters
        ----------
        nb_estimators : nunber of tree to use in the random forest
        random_state : seed to use for drawing random numbers
        max_depth : max depth of a tree in the forest
        min_samples_split : minimum number of leaf needed for split a node
        """
        super().__init__(
            name="Neural_Network_Regressor",
            model_family="neural_network",
            random_state=random_state,
            is_regression=True,
            n_jobs=n_jobs,
            **model_configs,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_params = early_stopping_params
        self.grid_search_enabled = grid_search_enabled
        self.grid_search_params = grid_search_params

        self.model_params["hidden_layer_sizes"] = hidden_layer_sizes
        self.model_params["learning_rate"] = learning_rate
        self.model_params["learning_rate_init"] = learning_rate_init
        self.model_params["max_iter"] = max_iter
        self.model_params["random_state"] = random_state
        self.model_params["n_iter_no_change"] = n_iter_no_change
        self.model_params["early_stopping"] = self.early_stopping_enabled
        self.create_object_model()

    def get_features_ranking(self):
        # the code of features ranking is to be determined
        pass

    def create_object_model(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=self.model_params["hidden_layer_sizes"],
            learning_rate=self.model_params["learning_rate"],
            learning_rate_init=self.model_params["learning_rate_init"],
            max_iter=self.model_params["max_iter"],
            random_state=self.model_params["random_state"],
            n_iter_no_change=self.model_params["n_iter_no_change"],
            early_stopping=self.model_params["early_stopping"],
            **self.model_additional_configs,
        )

    def process_early_stopping(
        self,
        test_df: pd.DataFrame = None,
        explanatory_variables: list = None,
        target_variable: list = None,
    ):
        # print exist to avoid linting error
        # TODO: change it to not have the useless print
        print(f"Ignore this: {test_df.shape}{explanatory_variables}{target_variable}")
        return self.early_stopping_params


class XGBoostRegressor(Regressor):
    def __init__(
        self,
        nb_estimators: int = 100,
        max_depth: int = 10,
        eta: int = 0.01,
        gamma: int = 0,
        random_state: int = 1,
        n_jobs: int = 1,
        objective: str = "reg:squarederror",
        early_stopping_enabled: bool = True,
        early_stopping_params: dict = None,
        grid_search_enabled=False,
        grid_search_params: dict = None,
        **model_configs,
    ):
        """

        Parameters
        ----------
        nb_estimators : number of tree  or boosting steps
        max_depth : max depth of a tree in the tree
        learning_rate : eta or boosting learning rate
        gamma : Minimum loss reduction required to make a further partition on a leaf
        random_state : seed to draw random numbers
        """
        super().__init__(
            name="XGBoostTree",
            model_family="tree_regressor",
            random_state=random_state,
            is_regression=True,
            n_jobs=n_jobs,
            **model_configs,
        )
        self.nb_estimators = nb_estimators
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.objective = objective
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_params = early_stopping_params
        self.grid_search_enabled = grid_search_enabled
        self.grid_search_params = grid_search_params

        self.model_params["n_estimators"] = nb_estimators
        self.model_params["max_depth"] = max_depth
        self.model_params["eta"] = eta
        self.model_params["gamma"] = gamma
        self.model_params["random_state"] = random_state
        self.model_params["n_jobs"] = n_jobs
        self.model_params["objective"] = objective
        self.create_object_model()

    def get_features_ranking(self):
        # the code of features ranking is to be determined
        pass

    def create_object_model(self):
        self.model = XGBRegressor(
            n_estimators=self.model_params["n_estimators"],
            max_depth=self.model_params["max_depth"],
            eta=self.model_params["eta"],
            gamma=self.model_params["gamma"],
            random_state=self.model_params["random_state"],
            n_jobs=self.model_params["n_jobs"],
            objective=self.model_params["objective"],
            **self.model_additional_configs,
        )

    def process_early_stopping(
        self,
        test_df: pd.DataFrame = None,
        explanatory_variables: list = None,
        target_variable: list = None,
    ):
        eval_set = [
            (test_df[explanatory_variables].values, test_df[target_variable].values)
        ]
        self.early_stopping_params["eval_set"] = eval_set

        return self.early_stopping_params


class LightGBMRegressor(Regressor):
    def __init__(
        self,
        objective="regression",
        metric=None,
        subsample=0.8,
        subsample_freq=5,
        boosting_type="gbdt",
        colsample_bytree=0.9,
        learning_rate=0.01,
        max_bin=255,
        num_leaves=128,
        reg_lambda=0.3,
        num_iterations=500,
        random_state: int = 1,
        n_jobs: int = 1,
        early_stopping_enabled=True,
        early_stopping_params: dict = None,
        grid_search_enabled=False,
        grid_search_params: dict = None,
        **model_configs,
    ):
        """

        Parameters
        ----------
        nb_estimators : number of tree  or boosting steps
        max_depth : max depth of a tree in the tree
        learning_rate : eta or boosting learning rate
        gamma : Minimum loss reduction required to make a further partition on a leaf
        random_state : seed to draw random numbers
        """
        super().__init__(
            name="LightGBMTree",
            model_family="tree_regressor",
            random_state=random_state,
            is_regression=True,
            n_jobs=n_jobs,
            **model_configs,
        )
        if metric is None:
            metric = ["l2", "l1"]
        self.objective = objective
        self.metric = metric
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.boosting_type = boosting_type
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.num_leaves = num_leaves
        self.reg_lambda = reg_lambda
        self.num_iterations = num_iterations
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_params = early_stopping_params
        self.grid_search_enabled = grid_search_enabled
        self.grid_search_params = grid_search_params

        self.model_params["objective"] = objective
        self.model_params["metric"] = metric
        self.model_params["subsample"] = subsample
        self.model_params["subsample_freq"] = subsample_freq
        self.model_params["boosting_type"] = boosting_type
        self.model_params["colsample_bytree"] = colsample_bytree
        self.model_params["learning_rate"] = learning_rate
        self.model_params["max_bin"] = max_bin
        self.model_params["num_leaves"] = num_leaves
        self.model_params["reg_lambda"] = reg_lambda
        self.model_params["num_iterations"] = num_iterations
        self.create_object_model()

    def create_object_model(self):
        self.model = lgb.LGBMRegressor(
            objective=self.model_params["objective"],
            metric=self.model_params["metric"],
            subsample=self.model_params["subsample"],
            subsample_freq=self.model_params["subsample_freq"],
            boosting_type=self.model_params["boosting_type"],
            colsample_bytree=self.model_params["colsample_bytree"],
            learning_rate=self.model_params["learning_rate"],
            max_bin=self.model_params["max_bin"],
            num_leaves=self.model_params["num_leaves"],
            reg_lambda=self.model_params["reg_lambda"],
            num_iterations=self.model_params["num_iterations"],
            histogram_pool_size=5000,
            silent=True,
            **self.model_additional_configs,
        )

    def get_features_ranking(self):
        # the code of features ranking is to be determined
        pass

    def process_early_stopping(
        self,
        test_df: pd.DataFrame = None,
        explanatory_variables: list = None,
        target_variable: list = None,
    ):
        eval_set = [
            (test_df[explanatory_variables].values, test_df[target_variable].values)
        ]
        self.early_stopping_params["eval_set"] = eval_set

        return self.early_stopping_params

