from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def get_classifier(name, **kwargs):
    models = {
        "logistic_regression": LogisticRegression(**kwargs),
        "sgd": SGDClassifier(**kwargs),
        "knn": KNeighborsClassifier(**kwargs),
        "decision_tree": DecisionTreeClassifier(**kwargs),
        "random_forest": RandomForestClassifier(**kwargs),
        "gradient_boosting": GradientBoostingClassifier(**kwargs),
        "xgboost": xgb.XGBClassifier(**kwargs),
        "lightgbm": lgb.LGBMClassifier(**kwargs),
        "catboost": CatBoostClassifier(verbose=0, **kwargs),
        "svm": SVC(**kwargs),
        "naive_bayes": GaussianNB(**kwargs)
    }
    if name not in models:
        raise ValueError(f"Model '{name}' is not supported.")
    return models[name]