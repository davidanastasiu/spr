# Pre-Built Estimators
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class LambdaMapper:
  clf = None
  regs = None
  name = None

  def __init__(self, name=None, seed=0):
    n_regs = 4
    clf, reg = None, None
    if name == "XGBoost":
        clf = XGBClassifier(objective="multi:softmax", random_state=seed, seed=seed, seed_per_iteration=True)
        reg = XGBRegressor(objective="reg:squarederror", random_state=seed, seed=seed, seed_per_iteration=True)
    elif name == "Histogram_Boosted_Trees" or name == "HBT":
        clf = HistGradientBoostingClassifier(class_weight="balanced")
        reg = HistGradientBoostingRegressor()
    elif name == "Random_Forest" or name == "RF":
        clf = RandomForestClassifier()
        reg = RandomForestRegressor()
    elif name == "Decision_Tree" or name == "DT":
        clf = DecisionTreeClassifier()
        reg = DecisionTreeRegressor()
    elif name == "KNN":
        clf = KNeighborsClassifier()
        reg = KNeighborsRegressor()
    elif name == "Logistic_Regression" or name == "LR":
        clf = LogisticRegression()
        reg = ElasticNet()
    elif name == "SVM":
        clf = SVC()
        reg = SVR()
    elif name == "Custom":
        clf = XGBClassifier(
                tree_method='gpu_hist', booster='gbtree', grow_policy="depthwise",
                max_bin=763, eta=0.1, sampling_method='gradient_based', subsample=0.2,
                colsample_bytree=0.25, n_estimators=125,
                max_depth=4, min_child_weight=0, reg_lambda=0,
                seed=seed, seed_per_iteration=True,
                n_jobs=-1
                )
        reg = HistGradientBoostingRegressor()
    else:
      raise ValueError('Name must be one of: XGBoost, Histogram_Boosted_Trees, HBT, Random_Forest, RF, Decision_Tree, DT, KNN, Logistic_Regression, LR, SVM, or Custom.')

    self.clf = clf
    self.regs = []
    for i in range(n_regs):
      self.regs.append(deepcopy(reg))
