import os
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils.util import init_seeds
# from utils.Logger import logger
from utils.pre_process import PreProcess
from utils.options import Options


class DataLoader(object):
  """
  This class is responsible for FileIO according to the directory structure

  0. Read config.yml and initialize proper submodules (i.e. PreProcessor)
  1. reading in data, img stored as numpy arrays, metadata as dict
  2. splitting into train/val/test

  """

  def __init__(self, options:Options=None, override_color=False):
    if options is None:
      raise ValueError("Options object is necessary to initialize a DataLoader.")
    self.options = options
    init_seeds(self.options['seed'])
    self.metadata = pd.DataFrame()
    # [0,1,2] -> [critical, healthy, intermediate]
    self.le = LabelEncoder()
    DataLoader.override_color = override_color
    # Verify Correct Directory
    self.Images = []
    print("DataLoader Initializing: ", self.options['data_dir'], self.options['img_time'])

    self.Images = []
    self.X, self.Y, self.metadata = self.read()
    self.X_train, self.X_val, self.X_test, \
      self.Y_train, self.Y_val, self.Y_test = self.split(self.X, self.Y, self.options)

  def read(self):
    options = self.options

    pp = PreProcess(options, override_color=DataLoader.override_color)

    X, Y = [], []
    img_list = sorted([f for f in os.listdir(options['data_dir']) if f.endswith('.png')])
    if options['max_samples'] > 0:
      img_list = img_list[:options['max_samples']]
    total_samples = len(img_list)

    for i, file in enumerate(tqdm(np.arange(1, total_samples + 1, 1), desc="Reading Samples", total=total_samples)):
      file = os.path.splitext(img_list[i])[0]
      img_path = f"{options['data_dir']}/{file}.png"
      meta_path = f"{options['data_dir']}/{file}.json"
      if not os.path.exists(meta_path):
        raise ValueError(f"Missing metadata for image file {img_path}.")

      metadata = self.load_json(meta_path)
      metadata['uid'] = file
      img = self.load_img(img_path)

      # Filter samples by Time
      if options['img_time'] != 'all':
        if int(options['img_time']) != metadata['time']:
          continue

      # No Sample Augmentation Use original Data
      if options['ds_aug'] == 'None':
        self.Images.append(img)
        # Transform X-Y
        img = pp.tx(img)
        metadata = pp.ty(metadata)
        X.append(img)
        Y.append(metadata)
      else:
        # Perform Sample Augmentation and re-seed metadata values
        imgs, ha, ya = pp.aug(img, metadata)
        X.extend(ha)
        Y.extend(ya)
        self.Images.extend(imgs)

    X, Y, M = self.post_process_dataset(X, Y)

    # Save Tiled Images

    # for ix, img in enumerate(self.Images):
    #     fig, ax = plt.subplots()
    #     img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    #     ax.imshow(img)
    #     fig.tight_layout()
    #     fig.savefig(f"data/post-process/{logger.options['seed']}-{ix}.png", bbox_inches="tight")
    #     plt.close()

    return X, Y, M

  def split(self, X, Y, options):
    # Create a stratified array to split according to distribution
    X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=options['test_size'],
            stratify=Y
            )
    X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train,
            test_size=options['validation_size'],
            stratify=Y_train
            )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

  def load_json(self, file):
    with open(file, 'r') as fp:
      metadata = json.load(fp)
    return metadata

  def load_img(self, file):
    img = cv2.imread(file)
    return img

  def post_process_dataset(self, x, y):
    X = pd.DataFrame(x)
    Y = pd.DataFrame(y)
    M = Y[['age', 'male', 'african', 'status-gt', 'egfr-gt', 'concentration-gt', 'bin']]
    # TODO: Label can be status or true concentration
    Y = Y[['status-gt']].squeeze()
    Y = self.le.fit_transform(Y)
    Y = np.ravel(Y)

    return X, Y, M
