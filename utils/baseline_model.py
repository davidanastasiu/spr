# Generic
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator
# Pre-Built Estimators
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import precision_recall_fscore_support, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from xgboost import XGBClassifier, XGBRegressor

# KHP Specific
import utils.util as aux
from utils.data_loader import DataLoader


class BaselineModel(BaseEstimator):

  def __init__(self, name=None, data=None, etype="clf or reg"):
    self.name = name
    self.data: DataLoader = data
    self.etype = etype
    # Metrics
    self.f1 = 0.
    self.precision = 0.
    self.recall = 0.
    self.error = 0.
    # Misc
    self.isFitted = None
    self.n_features = None
    self.testDF = None
    # Hyperparams
    self.grid = None
    estimator = None
    
    if name == "XGBoost":
        estimator = XGBClassifier(objective="multi:softmax") if etype == 'clf' else XGBRegressor(max_bin=32)
        self.grid = defaultdict(
                tree_method=["gpu_hist"],
                gpu_id=['0'],
                predictor=["auto"],
                sampling_method=['gradient_based'],
                n_estimators=[10, 25, 50, 100, 150, 200],
                subsample=[0.05],
                learning_rate=[0.001, 0.01, 0.1, 0.25, 0.3, 0.5],
                max_depth=scipy.stats.randint(low=1, high=16),
                ),
    elif name == "Histogram_Boosted_Trees" or name == "HBT":
        estimator = HistGradientBoostingClassifier() if etype == 'clf' else HistGradientBoostingRegressor()
        self.grid = defaultdict(
                learning_rate=[0.001, 0.01, 0.1, 0.25, 0.5, 0.75],
                max_iter=[10, 25, 50, 75, 100, 200],
                max_depth=scipy.stats.randint(low=1, high=64),
                ),
    elif name == "Random_Forest" or name == "RF":
        estimator = RandomForestClassifier() if etype == 'clf' else RandomForestRegressor()
        self.grid = defaultdict(
                n_estimators=[10, 25, 50, 100, 150, 200],
                max_depth=[None, 2, 3, 4, 8, 16, 32],
                max_samples=[0.5, 0.75, 0.8, 0.9, 1.0],
                ),
    elif name == "Decision_Tree" or name == "DT":
        estimator = DecisionTreeClassifier() if etype == 'clf' else DecisionTreeRegressor()
        self.grid = defaultdict(
                max_depth=[None, 2, 3, 4, 8, 16, 32],
                min_samples_leaf=[0.1, 0.2, 0.3, 0.5],
                ),
    elif name == "KNN":
        estimator = KNeighborsClassifier() if etype == 'clf' else KNeighborsRegressor()
        self.grid = [
          # Grid 1
          dict(
                  n_neighbors=[1, 2, 3, 4, 5, 6, 7, 8],
                  weights=['uniform', 'distance'],
                  leaf_size=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  metric=['euclidean', 'manhattan', 'minkowski', 'cosine'],
                  ),
          ]
    elif name == "Logistic_Regression" or name == "LR":
        if etype == 'clf':
          estimator = LogisticRegression()
          self.grid = defaultdict(
                  C=scipy.stats.uniform(loc=0, scale=1),
                  l1_ratio=scipy.stats.uniform(loc=0, scale=1),
                  max_iter=[10, 25, 50, 75, 100], )

        elif etype == 'reg':
          estimator = ElasticNet()
          self.grid = defaultdict(
                  alpha=scipy.stats.uniform(loc=0, scale=2),
                  l1_ratio=scipy.stats.uniform(loc=0, scale=1),
                  max_iter=[100, 200],
                  tol=[0.0001, 0.001, 0.01, 0.1],
                  selection=['cyclic', 'random'],
                  )

    elif name == "SVM":
        estimator = SVC() if etype == 'clf' else SVR()
        self.grid = defaultdict(
                C=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                kernel=['linear', 'poly', 'rbf', 'sigmoid'],
                # degree=[1, 2, 3, 4, 5, 6, 7],
                # gamma=['scale', 'auto'],
                max_iter=[10, 50, 100, 200],  # -1 means unrestricted
                )
    else:
       raise ValueError('Name must be one of: XGBoost, Histogram_Boosted_Trees, HBT, Random_Forest, RF, Decision_Tree, DT, KNN, Logistic_Regression, LR, or SVM.')
    self.estimator = estimator

  def transform(self, X, y=None, metadata=None):
    # Add Age,Male,African as features if type is classifier
    if self.etype == 'clf':
      data = metadata[['age', 'male', 'african']]
      X = np.append(X, data.astype(int), axis=1)
    # Change Labels to concentration for regression
    elif self.etype == 'reg' and y is not None:
      y = metadata['concentration-gt']

    return pd.DataFrame(X), np.ravel(y)

  def __sklearn_is_fitted__(self):
    return self.isFitted

  def fit(self, x, y, **fit_params):
    metadata = self.data.metadata.iloc[x.index]
    # Check Correct Shape
    X, Y = check_X_y(x, y)
    # Estimated Attributes (i.e. coef_ or matrix_weight_)
    # self.coef_ = None
    # Transform X-Y
    X, Y = self.transform(X, Y, metadata)
    self.n_features = X.shape[1]
    # Store classes seen during fit
    # self.classes_ = unique_labels(y)

    self.estimator.fit(X, Y)
    self.isFitted = True
    return self

  def predict(self, x):
    predictions = []
    # Extract Metadata
    metadata = self.data.metadata.iloc[x.index]
    # Number of Input Samples
    n_samples = len(x)
    # Check if Fit was called first
    check_is_fitted(self)
    # Input validation
    X = check_array(x)
    X, _ = self.transform(X, metadata=metadata)
    # Check that the input is of the same shape as the one passed during fit.
    if X.shape[1] != self.n_features:
      raise ValueError('Shape of input is different from what was seen in `fit`')

    status_predictions, concentration_predictions = [], []
    try:
      predictions = self.estimator.predict(X)
    except Exception as e:
      print(e)
      print(e, file=open('errors.log', 'w'))
      pass

    if self.etype == 'clf':
      status_predictions = predictions
      concentration_predictions = None
    elif self.etype == 'reg':
      # Apply eGFR formula
      concentration_predictions = predictions
      age, male, african = metadata['age'], metadata['male'], metadata['african']
      status_predictions = [aux.egfr(a, b, x, y)[1] for a, b, x, y in
                            zip(concentration_predictions, age, male, african)]
      status_predictions = self.data.le.fit_transform(status_predictions)
      egfr_predictions = [aux.egfr(a, b, x, y)[0] for a, b, x, y in
                          zip(concentration_predictions, age, male, african)]

    return status_predictions, concentration_predictions

  def score(self, X, y, sample_weight=None):
    # Maximize F1 Score (Default) | Minimize RMSE on Concentration
    # Extract metadata
    metadata = self.data.metadata.iloc[X.index]
    status_true = self.data.le.fit_transform(metadata['status-gt'])
    # Inference
    status_predictions, concentration_predictions = self.predict(X)
    # Calculate Metrics
    if self.etype == 'reg':
      status_predictions = self.data.le.fit_transform(status_predictions)
    (_, _, f1, _) = precision_recall_fscore_support(
            y_true=status_true, y_pred=status_predictions,
            average='weighted'
            )
    return f1

  def set_params(self, **params):
    self.estimator.set_params(**params)
    return self

  def print_params(self):
    params = self.estimator.get_params()
    for k, v in params.items():
      print(k, v)

  def load_params(self, directory="", img_time=""):
    print("\n***************\nDefault Parameters\n***************\n")
    self.print_params()

    # Read Best Parameters from CV Training
    print(f"\n****************\n{img_time}\n****************\n")
    path = os.path.join(directory, img_time, f"reg-{self.name}.json")
    print(f"\n****************\n{path}\n****************\n")

    print("\n***************\nBest Parameters\n***************\n")
    self.print_params()
    return self

  def get_params(self, deep=True):
    out = dict()
    for key in self.estimator.get_params():
      value = getattr(self.estimator, key)
      if deep and hasattr(value, "get_params") and not isinstance(value, type):
        deep_items = value.get_params().items()
        out.update((key + "__" + k, val) for k, val in deep_items)
      out[key] = value

    params = {'name': self.name, 'etype': self.etype}
    return params

  '''
  # Get parameters
  # dict = estimator.get_params()
  '''

  def eval(self, options=None):
    # Use same samples but re-seed labels and average F1, RMSE
    X = self.data.X_test
    num_rounds = 5
    # self.data.metadata.iloc[X.index]

    Y = self.data.Y_test
    M = self.data.metadata
    tdf = M.iloc[X.index]

    # Inference
    status_predictions, concentration_predictions = self.predict(X)
    # Calculate Metrics
    concentration_true, status_true = M.iloc[X.index]['concentration-gt'], Y  # M.iloc[X.index]['status-gt']
    self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
      y_true=status_true,
      y_pred=status_predictions,
      average='weighted'
    )
    self.error = rmse = root_mean_squared_error(
      y_true=concentration_true,
      y_pred=concentration_predictions
    )

    if self.etype == "reg":
      egfr_pred = [aux.egfr(a, b, x, y)[0] for a, b, x, y in
                   zip(concentration_predictions, tdf['age'], tdf['male'], tdf['african'])]
      error_egfr = [(a - b) ** 2 for a, b in zip(tdf['egfr-gt'], egfr_pred)]
      errors_c = [(cp - ct) ** 2 for cp, ct in zip(concentration_predictions, concentration_true)]
    else:
      self.error = -1.
      egfr_pred = np.zeros(shape=(len(X),))
      error_egfr = np.zeros(shape=(len(X),))
      errors_c = np.zeros(shape=(len(X),))

    # Save Predictions
    self.testDF = pd.DataFrame()
    self.testDF['bin'] = tdf['bin']
    self.testDF['age'] = tdf['age']
    self.testDF['male'] = tdf['male']
    self.testDF['african'] = tdf['african']
    self.testDF['concentration-pred'] = concentration_predictions
    self.testDF['concentration-true'] = tdf['concentration-gt']
    self.testDF['c-error'] = errors_c
    self.testDF['status-pred'] = self.data.le.inverse_transform(status_predictions)
    self.testDF['status-gt'] = tdf['status-gt']
    self.testDF['egfr-pred'] = egfr_pred
    self.testDF['egfr-gt'] = tdf['egfr-gt']
    self.testDF['egfr-error'] = error_egfr

  def cross_validate_train(self):
    options = self.data.options
    search = None

    # Resource Intensive
    if options['gs_type'] == 'full':
        pass
    # Generic Search
    elif options['gs_type'] == 'random':
        jobs = options['cv_jobs']

        search = RandomizedSearchCV(
                estimator=self,
                param_distributions=self.grid,
                n_iter=options['gs_iter'],
                cv=options['cv_folds'],
                verbose=options['cv_verbose'],
                n_jobs=jobs,
                error_score='raise',
                ).fit(X=self.data.X_train, y=self.data.Y_train)
    # Most Appropriate for Bin Search
    elif options['gs_type'] == 'halving':
        from sklearn.experimental import enable_halving_search_cv  # noqa
        from sklearn.model_selection import HalvingGridSearchCV

        search = HalvingGridSearchCV(
                estimator=self,
                param_grid=self.grid,
                factor=64,
                resource="n_samples",
                max_resources="auto",
                min_resources="exhaust",
                aggressive_elimination=False,
                cv=options['cv_folds'],
                verbose=options['cv_verbose'],
                n_jobs=options['cv_jobs'],
                error_score='raise',
                ).fit(X=self.data.X_train, y=self.data.Y_train)
    else:
      raise ValueError('Incorrect option for gs_type. Choose one of: full, random, or halving.')
    
    return search.best_estimator_.set_params(**search.best_params_)
