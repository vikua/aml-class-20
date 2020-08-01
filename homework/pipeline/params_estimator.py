import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.utils import get_columns_from_type
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
from tqdm import tqdm


class ParamsEstimator:
    def __init__(self,
                 url: str,
                 explanatory: [str],
                 target: str,
                 scoring: str,
                 types_dict: dict,
                 n_folds=10,
                 split_ratio=.2,
                 random_state=42):
        """
        :param url: Path to dataset
        :param explanatory: Array of column names of explanatory features
        :param target: Name of target variable
        :param scoring: Score name for GridSearch
        :param n_folds: Number of fold for Cross-Validation
        :param split_ratio: Split ratio of train/val separation
        """

        data = pd.read_csv(url)
        X, y = data[explanatory], data[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=split_ratio,
            random_state=random_state
        )

        self.numeric_features = get_columns_from_type(types_dict, 'numeric')
        self.categorical_features = get_columns_from_type(types_dict, 'categorical')
        self.text_features = get_columns_from_type(types_dict, 'text')

        self.scoring = scoring
        self.n_folds = n_folds

        self.total_num_of_features = sum([
            len(self.numeric_features),
            len(self.categorical_features),
            len(self.text_features)
        ])

        self.models = {
            'rf': RandomForestRegressor(),
            'lr': ElasticNet(),
            'gbr': GradientBoostingRegressor(),
            'svr': SVR()
        }

        self.models_params = {
            'rf': {
                'rf__criterion': ['mse', 'mae'],
                'rf__max_depth': range(2, 8),
                'rf__n_estimators': range(10, 200, 10),
                'rf__max_features': ['sqrt', 'log2', 'auto']
            },
            'lr': {
                'lr__alpha': np.linspace(0, 1, 10),
                'lr__selection': ['cyclic', 'random'],
                'lr__l1_ratio': np.linspace(0, 1, 10),
                'lr__positive': [True, False]
            },
            'gbr': {
                'gbr__loss': ['ls', 'lad', 'huber', 'quantile'],
                'gbr__learning_rate': np.logspace(-1, -5, 5),
                'gbr__num_estimator': range(10, 200, 10),
                'gbr__criterion': ['friedman_mse', 'mse', 'mae']
            },
            'svr': {
                'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'svr__gamma': ['scale', 'auto'],
                'svr__shrinking': [True, False]
            }
        }

    def build_pipeline(self, model):
        numeric_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=-99999)),
                ('scaler', StandardScaler())
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('fx_selection', SelectFromModel(
                    ElasticNet(alpha=.1),
                    max_features=self.total_num_of_features,
                    threshold=None))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numeric_features),
                ('cat', categorical_pipeline, self.categorical_features),
                # ('txt', text_pipeline, self.text_features)
            ]
        )

        est = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('rf', model)
            ]
        )

        return est

    def fit(self):
        scores = {}
        searchers = {}

        for model_name in tqdm(self.models.keys()):
            searchers[model_name] = RandomizedSearchCV(
                estimator=self.build_pipeline(self.models[model_name]),
                param_distributions=self.models_params[model_name],
                scoring=self.scoring,
                cv=self.n_folds,
                refit=True,
                verbose=1
            )

            scores[model_name] = cross_val_score(
                searchers[model_name],
                self.X_train,
                self.y_train,
                scoring=self.scoring,
                cv=self.n_folds
            ).mean()

        for est, score in scores.items():
            print('Model: {}, test score: {}'.format(est, score))

        best_model_name = max(scores, key=scores.get)
        est = searchers[best_model_name]
        est.fit(self.X_train, self.y_train)

        test_score = est.score(self.X_test, self.y_test)
        print('Best model: {}, train score: {}'.format(best_model_name, test_score))
        print('Best params:')

        for k, v in est.best_params_.items():
            print(f'{k}: {v}')

        return est.best_estimator_

