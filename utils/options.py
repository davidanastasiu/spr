import os
import json
import time
import shutil
import argparse
from collections import defaultdict
from copy import copy

import yaml

from utils.util import init_seeds


class Options():
  # default options
  options = {
    'data_dir' : 'dataset',
    'exp_dir' : None,
    'exp_name' : None,
    # Dataset Augmentation : ["None", "tile"]
    'ds_aug' : "tile",
    'img_w' : 64,
    'img_h' : 128,
    # Resize : center, uniform, None
    'resize_method' : "center",
    'img_time' : "22",
    # color_space : ['rgb', 'hsv', 'hls', 'lab', 'YCrCb', ]
    'color_space' : 'lab',
    'dual' : False,
    # illuminant{“A”, “B”, “C”, “D50”, “D55”, “D65”, “D75”, “E”}
    'illuminant' : 'D65',
    # observer{“2”, “10”, “R”}
    'observer' : '2',
    # Histogram Method : ['channel-wise', '2d', '3d', 'pixel']
    'hist_method' : 'channel-wise',
    'global_features' : False,
    # channel_mask : [0,1,2] -> [R|G|B]
    'channel_mask' : [0, 1, 2],
    'hist_bins' : [256, 256, 256],
    'c1_range' : [0, 256],
    'c2_range' : [0, 256],
    'c3_range' : [0, 256],
    # Regression Partitioning : [0.0, 60.0] -> Methods = ['default', 'manual', 'uniform-3', 'uniform-4', 'scaled']
    'partition_type' : 'uniform-4',
    'r1_range' : [0.0, 3.0],
    'r2_range' : [3.0, 6.0],
    'r3_range' : [6.0, 20.0],
    'r4_range' : [20.0, 60.0],
    # logical_mask : [True, False] --> Mask Image by thresholding
    'logical_mask' : False,
    # Normalization : ['z_score', 'min_max', None]
    'normalization' : None,
    # filter_type : ['mean-k', 'median-k', 'gaussian', None]
    'filter_type' : None,
    # Gridsearch Options : [full, random, halving]
    'gs_type' : 'random',
    'gs_iter' : 0,
    'cv_folds' : 5,
    'cv_verbose' : 42,
    'cv_jobs' : 10,
    # cv_score : ['f1' | 'rmse' | 'r2' ]
    'cv_score' : 'rmse',
    # Sample Splits
    'seed' : 0,
    'test_size' : 0.2,
    'validation_size' : 0.05,
    'max_samples' : 0,
    # Sub-Estimators for Baseline & Composite
    'models' : ['XGBoost', 'Histogram_Boosted_Trees', 'Random_Forest', 'Decision_Tree', 'KNN', 'Logistic_Regression', 'SVM', 'Custom'],
  }

  def __getitem__(self, key):
    '''Dictionary-like getter.'''
    return self.options[key]
  
  def __setitem__(self, key, value):
    '''Dictioanry like setter.'''
    if key in ['img_w', 'img_h', 'gs_iter', 'cv_verbose', 'cv_jobs', 'seed', 'max_samples']:
      self.options[key] = int(value)
    elif key in ['test_size', 'validation_size']:
      self.options[key] = float(value)
    else:
      self.options[key] = value

  def __init__(self, settings=None, data_dir=None, exp_dir=None, parse_cmdline=False):

    if settings:
      self.parse_settings_file(settings)
    if parse_cmdline:
      self.parse_command_line_arguments()
      
    # Experiments Vars
    if data_dir is not None:
      self.options['data_dir'] = data_dir
    self.data_dir = self.options['data_dir']
    if exp_dir is not None:
      self.options['exp_dir'] = exp_dir
    self.exp_dir = self.options['exp_dir']

    if self.data_dir is None:
      raise ValueError('Data directory must be provided.')

    init_seeds(self.options['seed'])

    # Custom Name or Increment ID
    self.exp_name = self.options['exp_name']
    if self.exp_name is None:
      self.exp_name = self.options['exp_name'] = time.strftime("%m-%d-%Y-%H-%M-%S")

    if self.exp_dir:
      self.config_path = os.path.join(self.exp_dir, self.exp_name)
      self.exp_path = os.path.join(self.exp_dir, self.exp_name)
      self.hp_path = os.path.join(self.exp_path, 'parameters')
      self.predictions_path = os.path.join(self.exp_path, 'predictions')

      if os.path.exists(self.config_path):
        print(f'Clearing previous experiment directory at {self.config_path}.')
        shutil.rmtree(self.config_path)

      try:
        os.makedirs(self.exp_path, exist_ok=True)
        os.makedirs(self.hp_path, exist_ok=True)
        os.makedirs(self.predictions_path, exist_ok=True)
      except Exception as e:
        print(e)

      yaml.safe_dump(data=self.options, stream=open(f"{self.config_path}/config.yaml", 'w'))

    # Print Options
    print('Execution Options:')
    print('\n'.join(f"{k} : {v}" for k, v in self.options.items()))
    print('CWD: ', os.getcwd())

    # Create Empty Stats Dict
    self.stats = None
    self.cmp_stats = defaultdict()
    self.dl_stats = defaultdict()
    self.clf_stats = defaultdict()
    self.reg_stats = defaultdict()


  def parse_settings_file(self, settings):
    """Parse yaml settings file."""
    if settings is None:
      return
    with open(settings, 'r') as fh:
      options = yaml.safe_load(fh)
      for k,v in options.items():
        if v is not None and repr(v) != '':
          self.__setitem__(k, v)


  def parse_command_line_arguments(self):
    """Parse command line arguments to update default options"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Dataset directory', default=None)
    parser.add_argument('--seed', help='Seed for splitting data and generating parameters', type=int, default=None)
    parser.add_argument('--exp_name', help='Experiment name', default=None)
    parser.add_argument('--exp_dir', help='Experiments directory', default=None)
    parser.add_argument('--color_space', help='Color space for samples', default=None)
    parser.add_argument('--illuminant', type=str, default=None)
    parser.add_argument('--observer', type=str, default=None)
    parser.add_argument('--img_time', help='Elapsed time after solution application', default=None)
    parser.add_argument('--hist_bins', help='Bin configuration', nargs="+", type=int, default=None)
    parser.add_argument('--hist_method', help='Bin configuration', type=str, default=None)
    parser.add_argument('--partition_type', help='Regressor splits', default=None)
    parser.add_argument('--filter_type', help='Filter method-k', default=None)
    parser.add_argument('--cv_score', help='Optimize rmse, f1, r2', default=None)
    parser.add_argument('--gs_iter', help='Gridsearch iterations', type=int, default=None)
    parser.add_argument('--r1', help='Regressor bin range', nargs="+", type=float, default=None)
    parser.add_argument('--r2', help='Regressor bin range', nargs="+", type=float, default=None)
    parser.add_argument('--r3', help='Regressor bin range', nargs="+", type=float, default=None)
    parser.add_argument('--r4', help='Regressor bin range', nargs="+", type=float, default=None)

    args, _ = parser.parse_known_args()

    def find_type(k):
      for a in parser._actions:
        if a.dest == k:
          return a.type
      return None

    for k,v in args.__dict__.items():
      if v is not None:
        t = find_type(k)
        if isinstance(t, int):
          self.options[k] = int(v)
        elif isinstance(t, float):
          self.options[k] = float(v)
        else:
          self.options[k] = v

    if args.r1 is not None and args.r2 is not None and args.r3 is not None and args.r4 is not None:
      self.options['r1_range'] = [0., args.r1]
      self.options['r2_range'] = [args.r1, args.r2]
      self.options['r3_range'] = [args.r2, args.r3]
      self.options['r4_range'] = [args.r3, args.r4]
    elif args.r1 is not None and args.r2 is not None and args.r3 is not None:
      self.options['r1_range'] = [0., args.r1]
      self.options['r2_range'] = [args.r1, args.r2]
      self.options['r3_range'] = [args.r2, args.r3]
      self.options['r4_range'] = None


  def save_model(self, model):
    self.get_model_type(model)
    self.save_params(estimator=model)
    self.save_metrics(model)
    self.save_predictions(model)


  def save_predictions(self, model):
    if self.mtype == "dl":
      pass
    with open(f"{self.predictions_path}/{self.mtype}-{model.name}", 'w') as fp:
      print(model.testDF.to_markdown(tablefmt="grid"), file=fp)


  def save_metrics(self, model):
    stats_path = os.path.join(self.exp_path, f"{self.mtype}-stats.json")

    if os.path.exists(stats_path):
      self.stats = json.load(fp=open(stats_path, 'r'))

    self.stats[model.name] = model.metrics
    json.dump(self.stats, fp=open(stats_path, 'w'), indent=2)


  def save_params(self, estimator):
    if self.mtype == 'cmp':
      params = estimator.log_params()
    elif self.mtype == 'dl':
      # params = estimator.get_config()
      # path = os.path.join(self.hp_path, f"{self.mtype}-{estimator.name}.json")
      # with open(file=path, mode='w') as fp:
      #     json.dump(params, fp, indent=2)
      return
    else:
      params = estimator.estimator.get_params()

    path = os.path.join(self.hp_path, f"{self.mtype}-{estimator.name}.json")

    with open(file=path, mode='w') as fp:
      pc = copy(params)
      for k, v in params.items():
        if not isinstance(v, (int, str, float, type(None),)):
          pc.pop(k)
      json.dump(pc, fp, indent=2)


  def get_model_type(self, model):
    from utils.baseline_model import BaselineModel
    from spr import SPR

    if isinstance(model, BaselineModel):
      self.mtype = model.etype
      if self.mtype == 'clf':
        self.stats = self.clf_stats
      elif self.mtype == 'reg':
        self.stats = self.reg_stats
    elif isinstance(model, SPR):
      self.mtype = "cmp"
      self.stats = self.cmp_stats
