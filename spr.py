# Generic
import json
import os.path
import re
import sys
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import precision_recall_fscore_support, root_mean_squared_error, accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

# KHP Specific
import utils.util as aux
from utils.gridsearch import hybrid_grids
from utils.data_loader import DataLoader
from utils.lambda_mapper import LambdaMapper
from utils.options import Options


class SPR(BaseEstimator, ClassifierMixin):

  def __init__(self, name="model_name", data=None, train=None, test=None, lm=None):
    # TODO: We can create static or dynamic lambda mapping here according to the clustering done in PreProcess
    # i.e. partition of bins according to concentration ranges
    # assignment of N regressors (we use N=4)

    if name == 'HBT':
      name = "Histogram_Boosted_Trees"
    elif name == 'RF':
      name = 'Random_Forest'
    elif name == 'DT':
      name = 'Decision_Tree'
    elif name == 'LR':
      name = 'Logistic_Regression'    

    self.name = name
    self.lm = lm if lm is not None else LambdaMapper(name=name)
    self.data: DataLoader = data
    # Metrics
    self.f1 = 0.
    self.precision = 0.
    self.recall = 0.
    self.error = 0.
    self.n_features = None
    self.testDF = None
    # Hyperparams
    self.grid = hybrid_grids[self.name]
    # Misc Metrics
    self.bin_predictions = list()
    self.bin_accuracy = 0.

  def transform(self, X):
    return X

  def fit(self, x=None, y=None, **fit_params):
    if x is None:
      x = self.data.X_train
    if y is None:
      y = self.data.Y_train
    self.n_samples = len(x)
    self.clf = self.lm.clf
    self.regs = self.lm.regs

    metadata = self.data.metadata.iloc[x.index]
    self.bins = metadata['bin']
    self.concentration = metadata['concentration-gt']

    # Check Correct Shape
    X, Y = check_X_y(x, y)
    # Store classes seen during fit
    self.classes_ = unique_labels(Y)

    X = self.transform(X)
    self.n_features = X.shape[1]

    # Fit the Bin Classifier
    self.clf.fit(X, y=self.bins)

    # Get indices mask and fit to regressors
    for bin in range(len(self.regs)):
      mask = list(self.bins == bin)
      if len(X[mask]) > 0:
        self.regs[bin].fit(X[mask], self.concentration[mask])
      else:
        print(f"[ERROR]: Tried to fit bin{bin} with 0 samples", file=sys.stderr)

    return self

  def predict(self, x):
    # Extract Metadata
    metadata = self.data.metadata.iloc[x.index]
    # Number of Input Samples
    n_samples = len(x)
    # Check if Fit was called first
    check_is_fitted(self)
    # Input validation
    X = check_array(x)
    X = self.transform(X)

    # Check that the input is of the same shape as the one passed during fit.
    if X.shape[1] != self.n_features:
      raise ValueError('Shape of input is different from what was seen in `fit`')

    '''
    Phase 1: Bin Classification
    '''
    bin_predictions = self.clf.predict(X)
    if len(self.bin_predictions) == 0:
      self.bin_predictions = bin_predictions
    else:
      # Take bin_predictions from other estimator
      bin_predictions = self.bin_predictions

    '''
    Phase 2: Concentration Regression
    '''
    concentration_predictions = np.zeros(shape=(n_samples,))
    for ix, b in enumerate(bin_predictions):
      c = self.regs[b].predict(X[ix, :].reshape(1, -1))
      concentration_predictions[ix] = c

    '''
    Phase 3: Apply eGFR and return status
    '''
    status_predictions = []
    for idx, c in enumerate(concentration_predictions):
      age, male, african = metadata.iloc[idx]['age'], metadata.iloc[idx]['male'], metadata.iloc[idx]['african']
      _, status = aux.egfr(c, age, male, african)
      status_predictions.append(status)

    return status_predictions, concentration_predictions

  def score(self, X, y, sample_weight=None):
    # Maximize F1 Score (Default) | Minimize MSE on Concentration
    options = self.data.options
    # Extract metadata
    metadata = self.data.metadata.iloc[X.index]
    # Inference
    status_predictions, concentration_predictions = self.predict(X)
    # Calculate Metrics

    if options['cv_score'] == 'f1':
      (_, _, f1, _) = precision_recall_fscore_support(y_true=metadata['status-gt'], y_pred=status_predictions, average='weighted')
      return f1
    elif options['cv_score'] == 'rmse':
      rmse = root_mean_squared_error(y_true=metadata['concentration-gt'], y_pred=concentration_predictions)
      # return 1.0/rmse
      return rmse * -1.0
    elif options['cv_score'] == 'r2':
      r2 = r2_score(y_true=metadata['concentration-gt'], y_pred=concentration_predictions)
      return r2
    else:
      (_, _, f1, _) = precision_recall_fscore_support(y_true=metadata['status-gt'], y_pred=status_predictions, average='weighted')
      return f1

  '''
  # Get parameters
  # dict = estimator.get_params()
  '''

  def get_params(self, deep=False):
    params = {'name': self.name, 'data': self.data, 'lm': self.lm}
    return params

  def log_params(self):
    params = self.lm.clf.get_params()

    for ix, r in enumerate(self.lm.regs):
      p = r.get_params()
      for k, v in p.items():
        new_k = f"reg{ix}__" + k
        params[new_k] = v

    return params

  def print_params(self):
    clf_params = self.lm.clf.get_params()
    for k, v in clf_params.items():
      print(k, v)

    regs_list = self.lm.regs
    for ix, r in enumerate(regs_list):
      print("\n***************\n", ix, "\n***************\n")
      for k, v in r.get_params().items():
        print(k, v)

  def load_params(self, directory="", img_time=""):
    print("\n***************\nDefault Parameters\n***************\n")
    self.print_params()

    # Read Best Parameters from CV Training
    print(f"\n****************\n{img_time}\n****************\n")
    path = os.path.join(directory, img_time, f"cmp-{self.name}.json")
    print(f"\n****************\n{path}\n****************\n")

    params = json.load(fp=open(path, 'r'))

    print("\n***************\nBest Parameters\n***************\n")

    best_clf_params = dict()
    best_reg_params = {0: dict(), 1: dict(), 2: dict(), 3: dict()}

    for k, v in params.items():
      if re.match("reg.__", k):
        # Regressor Parameters
        index = int(k[3])
        best_reg_params[index][k[6:]] = v
      else:
        # Classifier Parameters
        best_clf_params[k] = v

    self.lm.clf.set_params(**best_clf_params)
    for reg_id, params_dict in best_reg_params.items():
      self.lm.regs[reg_id].set_params(**params_dict)

    self.print_params()
    return self

  def set_params(self, **params):
    clf_params = dict()
    reg_params = {0: dict(), 1: dict(), 2: dict(), 3: dict()}

    # Split clf and reg parameters
    for k, v in params.items():
      if k.startswith('clf__'):
        new_k = k[5:]
        clf_params[new_k] = v
      elif k.startswith('reg'):
        index = k[3]
        if index != "_":
          index = int(index)
          temp_params = reg_params[index]
          new_k = k[6:]
          temp_params[new_k] = v
          reg_params[index] = temp_params

    # Set Classifier Parameters ONLY
    self.lm.clf.set_params(**clf_params)
    # Set Each Regressor Parameters
    for ix, r in enumerate(self.lm.regs):
      cpy_params = reg_params[ix]
      self.lm.regs[ix].set_params(**cpy_params)

    return self

  def eval(self, x=None):
    if x is None:
      x = self.data.X_test
    X = x
    M = self.data.metadata
    tdf = M.iloc[X.index]

    # Inference
    status_predictions, concentration_predictions = self.predict(X)
    # Calculate Metrics
    concentration_true, status_true = M.iloc[X.index]['concentration-gt'], M.iloc[X.index]['status-gt']
    self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
            y_true=status_true,
            y_pred=status_predictions,
            average='weighted'
            )

    self.bin_accuracy = accuracy_score(y_true=tdf["bin"], y_pred=self.bin_predictions)
    rmse = root_mean_squared_error(
      y_true=concentration_true,
      y_pred=concentration_predictions
    )
    self.error = rmse

    egfr_pred = [aux.egfr(a, b, x, y)[0] for a, b, x, y in zip(concentration_predictions, tdf['age'], tdf['male'], tdf['african'])]
    error_egfr = [(a - b) ** 2 for a, b in zip(tdf['egfr-gt'], egfr_pred)]
    errors_c = [(cp - ct) ** 2 for cp, ct in zip(concentration_predictions, concentration_true)]

    # Save Predictions
    self.testDF = pd.DataFrame()

    self.testDF['bin-true'] = tdf['bin']
    self.testDF['age'] = tdf['age']
    self.testDF['male'] = tdf['male']
    self.testDF['african'] = tdf['african']
    self.testDF['concentration-pred'] = concentration_predictions
    self.testDF['concentration-true'] = tdf['concentration-gt']
    self.testDF['c-error'] = errors_c
    self.testDF['status-pred'] = status_predictions
    self.testDF['status-gt'] = tdf['status-gt']
    self.testDF['egfr-pred'] = egfr_pred
    self.testDF['egfr-gt'] = tdf['egfr-gt']
    self.testDF['egfr-error'] = error_egfr

    self.metrics = {
      "f1": float(f"{self.f1 : .4f}"),
      "precision": float(f"{self.precision : .4f}"),
      "recall": float(f"{self.recall : .4f}"),
      "error": float(f"{self.error : .4f}"),
      "n_features": int(f"{self.n_features}"),
      "bin_accuracy": float(f"{self.bin_accuracy :.4f}")
    }

    return self.testDF, self.metrics

  def cross_validate_train(self, options=None):
    options = options or self.data.options
    search = None
    n_iter = options['gs_iter']

    if options['gs_type'] == 'full':
      # Resource Intensive
        raise ValueError('gs_type = full not currently implemented.')
      # Generic Search
    elif options['gs_type'] == 'random':
        search = RandomizedSearchCV(
                estimator=self,
                param_distributions=self.grid,
                n_iter=n_iter,
                cv=options['cv_folds'],
                verbose=options['cv_verbose'],
                n_jobs=options['cv_jobs'],
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

    return search.best_estimator_


  @staticmethod
  def initialize_models(options:Options):
    dl = DataLoader(options)

    models = {
      "XGBoost":
        SPR(name="XGBoost", data=dl, lm=LambdaMapper(name="XGBoost")),
      "Histogram_Boosted_Trees":
        SPR(name="Histogram_Boosted_Trees", data=dl, lm=LambdaMapper(name="Histogram_Boosted_Trees")),
      "Random_Forest":
        SPR(name="Random_Forest", data=dl, lm=LambdaMapper(name="Random_Forest")),
      "Decision_Tree":
        SPR(name="Decision_Tree", data=dl, lm=LambdaMapper(name="Decision_Tree")),
      "KNN":
        SPR(name="KNN", data=dl, lm=LambdaMapper(name="KNN")),
      "Logistic_Regression":
        SPR(name="Logistic_Regression", data=dl, lm=LambdaMapper(name="Logistic_Regression")),
      "SVM":
        SPR(name="SVM", data=dl, lm=LambdaMapper(name="SVM"))
      }

    return models