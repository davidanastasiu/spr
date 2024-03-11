import os
import json
import random
import numpy as np
from collections import defaultdict

def init_seeds(seed=0):
  # Initialize random number generator (RNG) seeds
  random.seed(seed)
  np.random.seed(seed)


def get_bin(c, options):
  if options['partition_type'] == "default":
      return get_bin_default(c)
  elif options['partition_type'] == "scaled":
      return get_bin_scaled(c)
  elif options['partition_type'] == "uniform-3":
      return get_bin_uniform_3(c)
  elif options['partition_type'] == "uniform-4":
      return get_bin_uniform_4(c)
  elif options['partition_type'] == "manual":
      return get_bin_manual(c, options)
  raise ValueError('Incorrect option for partition_type. Choose one of: default, scaled, uniform-3, uniform-4, or manual.')


def get_bin_scaled(c):
  concentration = float(c)
  if 0.0 <= concentration <= 3.2:
    return 0
  elif 3.2 < concentration <= 16.0:
    return 1
  elif 16.0 < concentration <= 60.0:
    return 2


def get_bin_uniform_3(c):
  concentration = float(c)
  if 0.0 <= concentration <= 3.2:
    return 0
  elif 3.2 <= concentration <= 16:
    return 1
  elif 16 <= concentration <= 60.0:
    return 2


def get_bin_uniform_4(c):
  concentration = float(c)
  if 0.0 <= concentration <= 3.0:
    return 0
  elif 3.0 <= concentration <= 6.0:
    return 1
  elif 6.0 <= concentration <= 20:
    return 2
  elif 20 <= concentration <= 60:
    return 3


# Default Regressor Splits according to collected data granularity
def get_bin_default(c):
  concentration = float(c)
  if 0.0 <= concentration <= 3.9:
    return 0
  elif 4.0 <= concentration <= 7.5:
    return 1
  elif 8.0 <= concentration <= 19.0:
    return 2
  elif 20.0 <= concentration <= 60.0:
    return 3


def get_bin_manual(c, options):
  concentration = float(c)

  [r1_min, r1_max] = options['r1_range']
  [r2_min, r2_max] = options['r2_range']
  [r3_min, r3_max] = options['r3_range']
  [r4_min, r4_max] = options['r4_range']

  if r1_min <= concentration <= r1_max:
    return 0
  elif r2_min <= concentration <= r2_max:
    return 1
  elif r3_min <= concentration <= r3_max:
    return 2
  elif r4_min <= concentration <= r4_max:
    return 3


def egfr(c, age, male, african):
  value, status = None, None

  c = float(c)
  age = int(age)

  # Scale from [True,False] --> [+1,-1]
  african = 1 if african else -1
  male = 1 if male else -1

  # Concentration cannot be 0 : Div by 0
  concentration = c if c > 0 else 1e-5
  value = 175.0 * concentration ** (-1.154) * age ** (-0.203)

  if male == -1:
    value *= 0.742
  if african == 1:
    value *= 1.212
  if 0 <= value <= 15:
    status = "critical"
  elif 15 < value <= 60:
    status = "intermediate"
  elif value > 60:
    status = "healthy"
  else:
    status = "unknown"

  # Truncate Float *EXPERIMENTAL*
  value = float(f"{value:.2f}")

  return value, status


def distribution():
  # Distribution Percentages Taken from
  # https://www.census.gov/quickfacts/fact/table/US/LFE046219
  age, male, african = None, None, None

  p = random.uniform(0, 1)
  pm = random.uniform(0, 1)
  pa = random.uniform(0, 1)

  # Sample Age from distribution
  if 0 <= p <= 0.187:
    age = random.randint(1, 14)
  elif 0.187 < p <= 0.227:
    age = random.randint(15, 17)
  elif 0.227 < p <= 0.264:
    age = random.randint(18, 20)
  elif 0.264 < p <= 0.583:
    age = random.randint(21, 44)
  elif 0.583 < p <= 0.837:
    age = random.randint(45, 64)
  elif p > 0.837:
    age = random.randint(65, 100)

  # Sample Male/Female from distribution
  male = 1 if pm >= 0.508 else -1
  # Sample African/Other from distribution
  african = 1 if pa <= 0.134 else -1

  return age, male, african


def average_stats(root_dir="", m_type="cmp"):
  total = 0.

  avg_stats = defaultdict()

  # Initialize output dictionaries from seed == 0
  avg_stats = json.load(fp=open(f"{root_dir}/0/{m_type}-stats.json"))
  for model_name, metrics in avg_stats.items():
    for metric, score in metrics.items():
      avg_stats[model_name][metric] = 0.

  # Read all seeds and store all times [fps, training] and metrics [f1, error]
  for seed in sorted(os.listdir(root_dir)):
    seed_path = os.path.join(root_dir, seed)
    try:
      if not os.path.isdir(seed_path):
        continue
      elif not seed.isdigit():
        continue
      else:
        total += 1
    except Exception as e:
      print(e)
      raise "STOP"

    stats_path = os.path.join(root_dir, seed, f"{m_type}-stats.json")
    stats = json.load(fp=open(stats_path))
    for model_name, metrics in stats.items():
      for metric, score in metrics.items():
        avg_stats[model_name][metric] += score

  # Average by weighted seed
  for model_name, metrics in avg_stats.items():
    for metric, score in metrics.items():
      avg_stats[model_name][metric] = round(avg_stats[model_name][metric] / total, 5)

  json.dump(avg_stats, open(f"{root_dir}/average-stats.json", 'w'), indent=2)
  return avg_stats