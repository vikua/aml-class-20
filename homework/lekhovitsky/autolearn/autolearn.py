import numpy as np
import pandas as pd
from sklearn.metrics import SCORERS
from sklearn.model_selection import (
    cross_val_score, 
    train_test_split, 
    RandomizedSearchCV
)

from .pipeline import build_pipeline, build_params_space
from .utils import Printer, validate_dtypes, infer_dtypes



def train_regressor(path_or_url, 
                    target, 
                    explanatory=None,
                    scoring='neg_mean_squared_error',
                    dtypes='infer',
                    test_size=0.2,
                    n_folds=4,
                    verbose=False,
                    random_state=42):
    """Automatically train a regression model for a given dataset.
    
    Parameters
    ----------
    path_or_url : str
        Filepath or URL to a valid CSV file.
    target : str
        Name of the column to use as target variable.
    explanatory : iterable of str, optional
        List of data frame columns to use as explanatory
        variables. 
        If None, use all variables except target.
    scoring : str, optional
        Any sklearn-compatible regression scoring.
        Default 'neg_mean_squared_error'.
    dtypes : 'infer' or dict, optional
        If 'infer', try to automatically deduce the types of
        explanatory variables based on their values.
        If dict, mapping from variable name to its type, which
        is one of 'numeric', 'categorical', 'date' and 'text'.
    test_size : float, optional
        Fraction of held-out test set.
    n_folds: int, optional
        Number of folds to use for cross-validation. Default 4.
    verbose : bool, optional
        If True, display some messages.
    random_state : int, optional
        Set RNG seed to this number.
    
    Returns
    -------
    estimator : Pipeline
    test_score : float
    cv_score : float
    """
    pprint = Printer(verbose)
    scorer = SCORERS[scoring]
    
    pprint('Loading the dataset...')
    data = pd.read_csv(path_or_url)
    pprint(f'Dataset loaded. Shape: {data.shape}\n')
    
    variables = list(data.columns)
    assert target in variables, f'{target} is not a name of any columns of the dataset'
    if explanatory is not None:
        assert len(explanatory), 'List of explanatory variables must not be empty'
        assert all([var in variables for var in explanatory])
    else:
        explanatory = list(variables)
        explanatory.remove(target)
    
    X, y = data[explanatory], data[target]
    
    if dtypes == 'infer':
        dtypes = infer_dtypes(X)
        pprint('Inferred types of variables:')
        for var, dtype in dtypes.items():
            pprint(f'    {var: dtype}')
    else:
        dtypes = validate_dtypes(X, dtypes)
    
    assert 0 < test_size < 1
    (X_train, X_test,
     y_train, y_test
    ) = train_test_split(
        X, y, test_size=test_size, 
        random_state=random_state)
    
    # nested-CV parameters search
    scores = {}
    estimators = {}
    for model in ['lr', 'svr', 'gbm', 'rf']:
        pprint(f'Fitting {model.upper()}...')
        estimators[model] = RandomizedSearchCV(
            build_pipeline(model, dtypes),
            build_params_space(model), 
            scoring=scoring,
            random_state=random_state,
            cv=n_folds)
        scores[model] = cross_val_score(
            estimators[model], 
            X_train, y_train,
            scoring=scoring,
            cv=n_folds
        ).mean()
        pprint(f'Cross-validation score: {scores[model]:.3f}\n')
    
    # final model scoring
    best = max(scores, key=scores.get)
    cv_score = max(scores)
    estimator = estimators[best]
    
    pprint(f'Fitting {best.upper()} on the full train set...')
    test_score = scorer(estimator.fit(X_train, y_train), X_test, y_test)
    pprint(f'Test-set score: {test_score:.4f}')
    pprint(f'Best params:')
    for p, v in estimator.best_params_.items():
        pprint(f'    {p}: {v}')
    
    return estimator.best_estimator_, test_score, cv_score
