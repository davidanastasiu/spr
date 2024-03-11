from copy import copy

import scipy

hybrid_grids = {
  "Custom": {
    "clf__eta": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
    "clf__subsample": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    "clf_colsample_bytree": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
    },
  "XGBoost": [
    # GBTree Only
    {
      # Classifier Parameters
      "clf__n_jobs": [-1],
      "clf__tree_method": ["gpu_hist"],
      "clf__booster": ["gbtree"],
      "clf__gpu_id": ["1"],
      "clf__predictor": ["auto"],
      "clf__sampling_method": ['gradient_based'],
      "clf__subsample": [0.75, 0.9, 1.0],
      "clf__n_estimators": range(50, 400, 50),
      # "clf_max_bin": [8, 16, 32, 64, 128, 256, 512],
      "clf__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
      "clf__max_depth": scipy.stats.randint(low=1, high=10),
      # Regularization
      "clf__min_child_weight": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],
      "clf__gamma": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],
      "clf__reg_lambda": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],
      "clf__reg_alpha": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],

      # Regressor Parameters
      "reg__n_jobs": [-1],
      "reg__tree_method": ["gpu_hist"],
      "reg__gpu_id": ["1"],
      "reg__booster": ["gbtree"],
      "reg__predictor": ["auto"],
      "reg__sampling_method": ['gradient_based'],
      "reg__subsample": [0.75, 0.9, 1.0],
      "reg__n_estimators": range(50, 400, 50),
      # "reg_max_bin": [8, 16, 32, 64, 128, 256, 512],
      "reg__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
      "reg__max_depth": scipy.stats.randint(low=1, high=10),
      # Regularization
      "reg__min_child_weight": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],
      "reg__gamma": [0, 0.1, 0.25, 0.5, 1, 2, 4],
      "reg__reg_lambda": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],
      "reg__reg_alpha": [0, 0.1, 0.25, 0.5, 1, 2, 4, 6],
      },
    ],
  "Gradient_Boosted_Trees":
    {
      "clf__learning_rate": [0.001, 0.01, 0.1, 0.25, 0.5, 0.75],
      "clf__n_estimators": [10, 25, 50, 100, 150, 200],
      "clf__max_depth": [None, 4, 8, 16, 32],
      "reg__learning_rate": [0.001, 0.01, 0.1, 0.25, 0.5, 0.75],
      "reg__n_estimators": [10, 25, 50, 100, 150, 200],
      "reg__max_depth": [None, 4, 8, 16, 32],
      },
  "Histogram_Boosted_Trees":
    {
      "clf__loss": ['log_loss'],
      "clf__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
      "clf__max_iter": [25, 50, 75, 100, 150, 200],
      "clf__max_depth": [None],
      "clf__max_leaf_nodes": [None],
      # "clf__max_leaf_nodes": [2, 4, 8, 16, 32],
      # "clf__min_samples_leaf": [2, 4, 8, 16, 20, 32],
      "clf__l2_regularization": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.0, 8.0],
      "clf__class_weight": [None, 'balanced'],

      "reg__loss": ["squared_error", "absolute_error"],
      "reg__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
      "reg__max_iter": [25, 50, 75, 100, 150, 200],
      "reg__max_depth": [None],
      "reg__max_leaf_nodes": [None],
      # reg__max_leaf_nodes": [2, 4, 8, 16, 32],
      # "reg__min_samples_leaf": [2, 4, 8, 16, 20, 32],
      "reg__l2_regularization": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.0, 8.0],
      },
  "Random_Forest":
    {
      "clf__n_estimators": [10, 25, 50, 100, 150, 200],
      "clf__max_depth": [None, 2, 3, 4, 8, 16, 32],
      "clf__max_samples": [0.5, 0.75, 0.8, 0.9, 1.0],
      "reg__n_estimators": [10, 25, 50, 100, 150, 200],
      "reg__max_depth": [None, 2, 3, 4, 8, 16, 32],
      "reg__max_samples": [0.5, 0.75, 0.8, 0.9, 1.0],
      },
  "Decision_Tree":
    {
      "clf__criterion": ["gini", "entropy", "log_loss"],
      "clf_splitter": ["random", "best"],
      "clf__max_depth": [None, 2, 3, 4, 8, 16, 32, 64],
      "clf__min_samples_split": [2, 4, 8, 16, 32],
      "clf__min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 8, 16, 32],
      "clf_min_weight_fraction_leaf": scipy.stats.uniform(loc=0, scale=1),
      "clf_max_features": ["sqrt", "log2", None],
      "clf_min_impurity_decrease": scipy.stats.uniform(loc=0, scale=1),
      "reg__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
      "reg_splitter": ["random", "best"],
      "reg__max_depth": [None, 2, 3, 4, 8, 16, 32, 64],
      "reg__min_samples_split": [2, 4, 8, 16, 32],
      "reg__min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 8, 16, 32],
      "reg_min_weight_fraction_leaf": scipy.stats.uniform(loc=0, scale=1),
      "reg_max_features": ["sqrt", "log2", None],
      "reg_min_impurity_decrease": scipy.stats.uniform(loc=0, scale=1),
      },
  "KNN":
    [  # Grid 1 = Mixed
      # Metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis'],
      {
        "clf__n_neighbors": scipy.stats.randint(low=1, high=8),
        "clf__weights": ['uniform', 'distance'],
        "clf__leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "clf__metric": ['euclidean', 'manhattan', 'cosine'],
        "reg__n_neighbors": scipy.stats.randint(low=1, high=8),
        "reg__weights": ['uniform', 'distance'],
        "reg__leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "reg__metric": ['euclidean', 'manhattan', 'cosine'],
        },
      # Grid 2 = Minkowski Only
      {
        "clf__n_neighbors": scipy.stats.randint(low=1, high=8),
        "clf__weights": ['uniform', 'distance'],
        "clf__leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "clf__metric": ['minkowski'],
        "clf_p": [1, 2, 3, 4],
        "reg__n_neighbors": scipy.stats.randint(low=1, high=8),
        "reg__weights": ['uniform', 'distance'],
        "reg__leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "reg__metric": ['minkowski'],
        "reg_p": [1, 2, 3, 4],
        },
      # Exotic Metrics
      {
        "clf__n_neighbors": scipy.stats.randint(low=1, high=8),
        "clf__weights": ['uniform', 'distance'],
        "clf__leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "clf__metric": ['braycurtis', 'canberra', 'chebyshev'],
        "reg__n_neighbors": scipy.stats.randint(low=1, high=8),
        "reg__weights": ['uniform', 'distance'],
        "reg__leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "reg__metric": ['braycurtis', 'canberra', 'chebyshev'],
        },
      ],
  "Logistic_Regression":
    {
      # clf__penalty=['l1', 'l2', 'elasticnet', 'none'],
      "clf__C": scipy.stats.uniform(loc=0.1, scale=2),
      "clf__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
      "clf__max_iter": [10, 25, 50, 75, 100],
      "reg__alpha": scipy.stats.uniform(loc=0, scale=2),
      "reg__l1_ratio": scipy.stats.uniform(loc=0, scale=1),
      "reg__max_iter": [10, 25, 50, 75, 100],
      },
  "SVM":
    [  # Mixed
      {
        "clf__C": scipy.stats.uniform(loc=0.1, scale=2),
        "clf__kernel": ['poly', 'rbf', 'sigmoid'],
        "clf__degree": [1, 2, 3, 4],
        "clf__max_iter": [10, 25, 50, 75, 100, 250],  # -1 means unrestricted
        "reg__C": scipy.stats.uniform(loc=0.1, scale=2),
        "reg__kernel": ['poly', 'rbf', 'sigmoid'],
        "reg__degree": [1, 2, 3, 4],
        "reg__max_iter": [10, 25, 50, 75, 100, 250],  # -1 means unrestricted
        },
      # Poly Only
      {
        "clf__C": scipy.stats.uniform(loc=0.1, scale=2),
        "clf__kernel": ['poly'],
        "clf__degree": [1, 2, 3, 4],
        "clf__gamma": ['scale', 'auto'],
        "clf__max_iter": [10, 25, 50, 75, 100, 250],  # -1 means unrestricted
        "reg__C": scipy.stats.uniform(loc=0.1, scale=2),
        "reg__kernel": ['poly'],
        "reg__degree": [1, 2, 3, 4],
        "reg__gamma": ['scale', 'auto'],
        "reg__max_iter": [10, 25, 50, 75, 100, 250],  # -1 means unrestricted
        },

      # All others
      {
        "clf__C": scipy.stats.uniform(loc=0.1, scale=2),
        "clf__kernel": ['rbf', 'sigmoid'],
        "clf__gamma": ['scale', 'auto'],
        "clf__max_iter": [10, 25, 50, 75, 100, 250],  # -1 means unrestricted
        "reg__C": scipy.stats.uniform(loc=0.1, scale=2),
        "reg__kernel": ['rbf', 'sigmoid'],
        "reg__gamma": ['scale', 'auto'],
        "reg__max_iter": [10, 25, 50, 75, 100, 250],  # -1 means unrestricted
        },
      ]
  }

'''
Lambda Mapper Alpha
----------------------------------
We tune each regressor separately
'''


def allocate_regressors(grid):
  n_regs = 4
  grid_copy = copy(grid)
  for k, v in grid.items():
    if k.startswith('reg__'):
      for n in range(n_regs):
        new_k = k[0:3] + str(n) + k[3:]
        new_v = copy(v)
        grid_copy[new_k] = new_v
  return grid_copy


for model_name, grid in hybrid_grids.items():
  # List of Grids
  if isinstance(grid, list):
    for ix, g in enumerate(grid):
      grid[ix] = allocate_regressors(g)
    hybrid_grids[model_name] = grid
  # Single Grid
  else:
    grid = allocate_regressors(grid)
    hybrid_grids[model_name] = grid
