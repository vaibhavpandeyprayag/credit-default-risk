from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from src.config import RANDOM_STATE
from src.features.data_transformer import preprocessor_with_scaling, preprocessor_without_scaling
from src.features.feature_builder import add_features

def get_logistic_regression_pipeline():
    return Pipeline([
        ("feature_engineering", FunctionTransformer(add_features)),
        ("preprocessor", preprocessor_with_scaling),
        ("model", LogisticRegression(class_weight='balanced',
            max_iter=1000,
            C=0.1,
            solver='saga'))
    ])

def get_random_forest_pipeline():
    return Pipeline([
        ("feature_engineering", FunctionTransformer(add_features)),
        ("preprocessor", preprocessor_without_scaling),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE,
            class_weight='balanced',
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            max_features='sqrt'))
    ])

def get_xgboost_pipeline():
    return Pipeline([
        ("feature_engineering", FunctionTransformer(add_features)),
        ("preprocessor", preprocessor_without_scaling),
        ("model", XGBClassifier(random_state=RANDOM_STATE,
            scale_pos_weight=2.33,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='auc'))
    ])