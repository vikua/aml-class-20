import logging
import os
from collections import namedtuple
from datetime import datetime
from functools import reduce
from itertools import combinations
from typing import Optional, Sequence

import pandas as pd
import yaml
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score, max_error, mean_absolute_error,
    mean_squared_error, median_absolute_error,
)
from sklearn.model_selection import (
    RandomizedSearchCV, train_test_split
)
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .tfidf import TfIdf
from .utils import least_populated_class, to_text

log = logging.getLogger('Pipeline')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - [%(name)s] -  %(message)s"
)


FEATURE_TYPES = ['Numeric', 'Text', 'Categorical']
FType = namedtuple('FType', FEATURE_TYPES)(*FEATURE_TYPES)
METRICS = [
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
]
RANDOM_STATE = 42


class AMLPipeline:
    def __init__(
            self,
            target: str,
            data_path: str,
            scoring: str = 'neg_mean_absolute_error',
            model_type: str = 'rf'  # one of {'lr', 'rf', 'gb'}
    ):
        self.target: str = target
        self.data_path: str = data_path
        self.scoring: str = scoring
        self.model_type: str = model_type
        self.dataset: Optional[pd.DataFrame] = None
        self.ds_train: Optional[pd.DataFrame] = None
        self.ds_test: Optional[pd.DataFrame] = None
        self.feature_mapping: Optional[dict] = None
        self.input_features: Optional[Sequence[str]] = None
        self.estimator: Optional[BaseEstimator] = None
        self.hyperparams_search: Optional[BaseSearchCV] = None

    def run(self):
        log.info('Step 1: Read data...')
        self.read_data()
        log.info('Step 2: EDA...')
        self.eda()
        log.info('Step 3: Feature engineering...')
        self.feature_engineering()
        log.info('Step 4: CV partitioning...')
        self.cv_partitioning()
        log.info('Step 5: Build train pipeline...')
        self.build_train_pipeline()
        log.info('Step 6: Optimize hyperparameters...')
        self.optimize_hyperparameters()
        log.info('Step 7: Report scores...')
        self.report_scores()
        log.info('Pipeline completed!')

    def read_data(self):
        """ Task 1
        Reads provided data.
        """
        log.info(f'Reading data from {self.data_path}')
        self.dataset = pd.read_csv(self.data_path)
        # self.dataset.to_pickle('dataset.pkl')
        # self.dataset = pd.read_pickle('dataset.pkl')

    def eda(self):
        """ Task 2
        Does exploratory data analysis.
        """
        with open(os.path.join('pipeline', 'feature_type_mapping.yml')) as f:
            self.feature_mapping = yaml.load(f, Loader=yaml.FullLoader)

    def feature_engineering(self):
        """ Task 3 (Optional)
        Does feature engineering.
        """
        log.info('Skipping feature engineering')

    def cv_partitioning(self):
        """ Task 4
        Does CV partitioning
        """
        cat_features = [k for k, v in self.feature_mapping.items()
                        if v == FType.Categorical]
        stratification_data = self.dataset[cat_features].fillna('nan')
        columns = [stratification_data[c] for c in stratification_data.columns]

        max_subset = 0
        for i in range(len(columns)):
            if not any(least_populated_class(subset) >= 2
                       for subset in combinations(columns, i + 1)):
                max_subset = i
                break
            max_subset = i + 1

        stratification_data = None

        if max_subset != 0:
            for subset in combinations(columns, max_subset):
                if least_populated_class(subset) >= 2:
                    stratification_data = reduce(lambda x, y: x + '\n\n' + y, subset)
                    break

        self.ds_train, self.ds_test = train_test_split(
            self.dataset,
            test_size=.2,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=stratification_data
        )

    def build_train_pipeline(self):
        """ Task 5
        Builds a pipeline which can train model.
        """
        # Categorical
        categorical_features = [k for k, v in self.feature_mapping.items()
                                if v == FType.Categorical and k != self.target]
        categorical_pipeline = Pipeline(steps=[
            ('cat_imputer', SimpleImputer(
                strategy='constant',
                fill_value='MissingValue'
            )),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ])

        # Numeric
        numeric_features = [k for k, v in self.feature_mapping.items()
                            if v == FType.Numeric and k != self.target]
        numeric_pipeline = [
            ('num_imputer', SimpleImputer())
        ]
        if self.model_type == 'lr':
            numeric_pipeline.append(
                ('standardization', StandardScaler())
            )
        numeric_pipeline = Pipeline(steps=numeric_pipeline)

        # Text
        text_features = [k for k, v in self.feature_mapping.items()
                         if v == FType.Text and k != self.target]
        text_pipeline = Pipeline(steps=[
            ('tfidf', TfIdf())
        ])

        # Append estimator
        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_pipeline, categorical_features),
            ('numeric', numeric_pipeline, numeric_features),
            ('text', text_pipeline, text_features),
        ])
        pipeline = [('preprocessor', preprocessor)]
        if self.model_type == 'lr':
            pipeline.append(('model', LinearRegression(n_jobs=-1)))
        elif self.model_type == 'rf':
            pipeline.append(('model',
                             RandomForestRegressor(
                                 n_jobs=-1,
                                 random_state=RANDOM_STATE
                             )))
        elif self.model_type == 'gb':
            pipeline.append(('model',
                             GradientBoostingRegressor(
                                 random_state=RANDOM_STATE
                             )))
        else:
            raise ValueError(f'model_type can not be {self.model_type}')

        self.estimator = Pipeline(steps=pipeline)

    def optimize_hyperparameters(self):
        """ Task 6
        Optimizes hyperparameters of models.
        """

        X, y = self.get_Xy(self.ds_train)

        param_distribution = {
            'preprocessor__text__tfidf__strip_accents':
                ['unicode', 'ascii', None],
            'preprocessor__text__tfidf__stop_words': ['english', None],
            'preprocessor__text__tfidf__ngram_range':
                [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
            'preprocessor__text__tfidf__max_df': stats.uniform(.3, 1.),
            'preprocessor__text__tfidf__min_df': stats.uniform(.001, .25),
            'preprocessor__text__tfidf__max_features': range(1, 21),
            'preprocessor__text__tfidf__norm': ['l1', 'l2'],
            'preprocessor__text__tfidf__use_idf': [True, False],
            'preprocessor__text__tfidf__smooth_idf': [True, False],
            'preprocessor__text__tfidf__sublinear_tf': [True, False],
        }
        if self.model_type in {'rf', 'gb'}:
            param_distribution.update({
                'model__n_estimators': range(10, 50),
                'model__max_features': range(2, 10),
                'model__min_samples_split': range(2, 5),
                'model__min_samples_leaf': range(1, 5),
                'model__max_depth': range(2, 10),
                'model__criterion': ['friedman_mse', 'mse', 'mae'],
            })

        search = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=param_distribution,
            n_iter=10,
            scoring=self.scoring,
            cv=3,
            verbose=2,
            refit=True,
            random_state=RANDOM_STATE
        )
        search.fit(X, y)
        self.hyperparams_search = search
        self.estimator = search.best_estimator_

    def report_scores(self):
        """ Task 7
        Reports scores of models.
        """

        X, y = self.get_Xy(self.ds_test)
        y_pred = self.estimator.predict(X)

        with open('model_metrics.txt', 'w') as file:
            print(f'Trained on {datetime.now()}', file=file)

            msg = to_text(self.hyperparams_search.cv_results_)
            log.info(msg)
            print(msg, file=file)
            for metric in METRICS:
                try:
                    value = metric(y, y_pred)
                except ValueError as e:
                    log.warning(e)
                    value = None
                msg = f'{metric.__name__.upper()}: \n' \
                      f'\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t{value}'
                log.info(msg)
                print(msg, file=file)

    def get_Xy(self, ds):
        X = ds[ds.columns.difference([self.target])]
        y = ds[self.target]
        return X, y
