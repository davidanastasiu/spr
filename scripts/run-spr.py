import os
import sys
import warnings
import traceback

sys.path.append(os.getcwd())
warnings.simplefilter(action='ignore', category=FutureWarning)

from spr import SPR
from utils.options import Options

def main():
  options = Options(data_dir='dataset', settings='settings.yml', parse_cmdline=True)

  models = SPR.initialize_models(options)

  for name, spr_model in models.items():
    print(f"Estimator: {name}")

    try:
      # Train
      best_spr = spr_model.fit()
      # CV Test Results
      _, metrics = best_spr.eval()
      # Save Results [f1, precision, recall, mse, params, etc]
      # logger.save_model(best_spr)
      print(name, metrics)
    except Exception as e:
      print(e)
      traceback.print_exc()
      exit()


if __name__ == "__main__": main()