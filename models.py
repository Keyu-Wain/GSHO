#  Model definitions and training
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from utils import calculate_metrics, calculate_grid_size



def train_traditional_models(X, y, config):
    # Parameter grids
    param_grids = {
        "Random Forest (RF)": {
            "model__n_estimators": [50, 100, 150, 200, 300],
            "model__max_depth": [4, 5, 6, 7, 8, 9, 10],
            "model__min_samples_split": [4, 6, 8],
            "model__min_samples_leaf": [2, 3, 4],
            "model__max_features": ["sqrt"],
            "model__bootstrap": [True],
            "model__ccp_alpha": [0.0, 0.01, 0.03],
            "model__random_state": [config["RANDOM_SEED"]]
        },
        "XGBoost": {
            "model__n_estimators": [50, 100, 150, 200, 300],
            "model__max_depth": [4, 5, 6, 7, 8, 9, 10],
            "model__eta": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            "model__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [1.0],
            "model__gamma": [0.1, 0.3, 0.5],
            "model__reg_alpha": [0.0, 0.1, 0.5],
            "model__reg_lambda": [1.0, 2.0, 3.0],
            "model__min_child_weight": [2, 3, 4],
            "model__enable_categorical": [False],
            "model__random_state": [config["RANDOM_SEED"]]
        },
        "LGBM": {
            "model__n_estimators": [50, 100, 150, 200, 300],
            "model__max_depth": [4, 5, 6, 7, 8, 9, 10],
            "model__learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            "model__num_leaves": [31, 45, 63],
            "model__min_child_samples": [6, 8, 10],
            "model__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5],
            "model__reg_lambda": [1.0, 2.0, 3.0],
            "model__bagging_fraction": [0.9, 1.0],
            "model__disable_feature_name_check": [True],
            "model__random_state": [config["RANDOM_SEED"]],
            "model__verbose": [-1]
        },
        "SVM (SVR)": {
            "model__kernel": ["rbf", "linear"],
            "model__C": [1.0, 5.0, 10.0, 20.0],
            "model__gamma": ["scale", 0.01, 0.1],
            "model__epsilon": [0.1, 0.2, 0.3],
            "model__tol": [1e-3],
            "model__cache_size": [5000],
            "model__max_iter": [-1],
            "model__verbose": [False]
        }
    }

    # Preprocessing pipeline
    full_preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Base models
    base_models = {
        "Random Forest (RF)": Pipeline(steps=[
            ("preprocessor", full_preprocessor),
            ("model", RandomForestRegressor(random_state=config["RANDOM_SEED"], n_jobs=-1))
        ]),
        "XGBoost": Pipeline(steps=[
            ("preprocessor", full_preprocessor),
            ("model", XGBRegressor(random_state=config["RANDOM_SEED"], n_jobs=-1, enable_categorical=False))
        ]),
        "LGBM": Pipeline(steps=[
            ("preprocessor", full_preprocessor),
            ("model",
             LGBMRegressor(random_state=config["RANDOM_SEED"], n_jobs=-1, verbose=-1, disable_feature_name_check=True))
        ]),
        "SVM (SVR)": Pipeline(steps=[
            ("preprocessor", full_preprocessor),
            ("model", SVR())
        ])
    }

    final_results = {}
    outer_kf = KFold(n_splits=config["CV_FOLDS"], shuffle=True, random_state=config["RANDOM_SEED"])

    # Train each model
    for model_name, base_pipeline in base_models.items():
        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grids[model_name],
            cv=config["INNER_GRID_CV"],
            scoring="r2",
            n_jobs=-1,
            verbose=config["VERBOSE_LEVEL"],
            error_score="raise"
        )

        fold_metrics = {"R²": [], "RMSE": [], "MAE": []}
        fold_best_params = []

        for fold, (train_idx, val_idx) in enumerate(outer_kf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            grid_search.fit(X_train, y_train)
            y_pred = grid_search.best_estimator_.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred)

            fold_metrics["R²"].append(metrics["R²"])
            fold_metrics["RMSE"].append(metrics["RMSE"])
            fold_metrics["MAE"].append(metrics["MAE"])
            fold_best_params.append(grid_search.best_params_)

        # Calculate average metrics
        avg_metrics = {
            "Average R²": np.mean(fold_metrics["R²"]),
            "Average RMSE": np.mean(fold_metrics["RMSE"]),
            "Average MAE": np.mean(fold_metrics["MAE"])
        }

        final_results[model_name] = {
            **avg_metrics,
            "Fold R²": fold_metrics["R²"],
            "Fold RMSE": fold_metrics["RMSE"],
            "Fold MAE": fold_metrics["MAE"],
            "Fold Best Params": fold_best_params,
            "Params Summary": pd.DataFrame(fold_best_params)
        }

    return final_results


def train_mlp_model(X, y, config):
    """Train MLP model with randomized search"""

    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            max_iter=3000,
            early_stopping=True,
            random_state=config["RANDOM_SEED"],
            verbose=False
        ))
    ])

    # MLP parameter grid
    mlp_param_grid = {
        'mlp__hidden_layer_sizes': [(16,), (32,), (64,), (128,), (256,),
                                    (64, 32), (128, 64), (256, 128),
                                    (128, 64, 32), (256, 128, 64)],
        'mlp__activation': ['relu', 'tanh', 'identity'],
        'mlp__solver': ['adam', 'sgd'],
        'mlp__learning_rate_init': [0.00001, 0.0001, 0.001, 0.01],
        'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'mlp__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'mlp__batch_size': ['auto', 16, 32, 64],
        'mlp__max_iter': [2000, 3000, 5000],
        'mlp__tol': [1e-5, 1e-4],
        'mlp__validation_fraction': [0.1, 0.15, 0.2],
        'mlp__n_iter_no_change': [10, 20, 30],
        'mlp__warm_start': [False]
    }

    total_combinations = calculate_grid_size(mlp_param_grid)
    actual_iter = min(total_combinations, config["MAX_SEARCH_ITER"])
    mlp_searcher = None
    best_mlp_model = None

    try:
        mlp_searcher = RandomizedSearchCV(
            estimator=mlp_pipeline,
            param_distributions=mlp_param_grid,
            n_iter=actual_iter,
            cv=KFold(n_splits=config["CV_FOLDS"], shuffle=True, random_state=config["RANDOM_SEED"]),
            scoring='r2',
            n_jobs=-1,
            random_state=config["RANDOM_SEED"],
            verbose=config["VERBOSE_LEVEL"],
            refit=True
        )
        mlp_searcher.fit(X, y)
        best_mlp_model = mlp_searcher.best_estimator_
    except Exception as e:
        print(f"MLP search failed: {str(e)}")
        best_mlp_model = mlp_pipeline.fit(X, y)

    # Store MLP results
    mlp_results = {}
    if mlp_searcher is not None and hasattr(mlp_searcher, 'best_score_'):
        cv_metrics = {"RMSE": [], "MAE": []}
        for train_idx, test_idx in KFold(n_splits=config["CV_FOLDS"], shuffle=True,
                                         random_state=config["RANDOM_SEED"]).split(X, y):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]

            cv_model = mlp_pipeline.set_params(**mlp_searcher.best_params_)
            cv_model.fit(X_train_cv, y_train_cv)
            y_pred_cv = cv_model.predict(X_test_cv)

            metrics = calculate_metrics(y_test_cv, y_pred_cv)
            cv_metrics["RMSE"].append(metrics["RMSE"])
            cv_metrics["MAE"].append(metrics["MAE"])

        mlp_results = {
            "Average R²": mlp_searcher.best_score_,
            "Average RMSE": np.mean(cv_metrics["RMSE"]),
            "Average MAE": np.mean(cv_metrics["MAE"]),
            "Best Params": mlp_searcher.best_params_
        }

    # Evaluation
    if len(X) >= 5 and best_mlp_model is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["TEST_SIZE"], random_state=config["RANDOM_SEED"]
        )
        best_mlp_model.fit(X_train, y_train)
        y_pred = best_mlp_model.predict(X_test)
        calculate_metrics(y_test, y_pred)

    return mlp_results, best_mlp_model