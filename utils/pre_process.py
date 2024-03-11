import cv2
import numpy as np
from copy import deepcopy
import utils.util as aux


class PreProcess:
  def __init__(self, options=None, override_color=False):
    # Run Pre-Processor Return Correct Data according to options
    # Resize
    # Color Space
    # Histogram or Raw Data
    # Augmentations
    # Filters
    #
    #
    #
    # return X, Y, M
    self.id = 0
    self.total_pixels = 0
    self.options = options
    self.override_color = override_color

    self.bin_features = []


  def mask(self, img):
    options = self.options

    if options['logical_mask']:
      [c1_min, c1_max] = options['c1_range']
      [c2_min, c2_max] = options['c2_range']
      [c3_min, c3_max] = options['c3_range']

      channels = cv2.split(img)
      c1, c2, c3 = channels[0], channels[1], channels[2]

      # MASKING
      ############################################################################################################
      mask = ((c1 >= c1_min) * (c1 <= c1_max)) * \
             ((c2 >= c2_min) * (c2 <= c2_max)) * \
             ((c3 >= c3_min) * (c3 <= c3_max))

      # [False  True] [1155 7037]
      unique, counts = np.unique(mask, return_counts=True)
      print(unique, counts)
      self.total_pixels = counts[1] if len(unique) == 2 else img.shape[0] * img.shape[1]

      mask = np.reshape(mask, newshape=(img.shape[:2])).astype('uint8')
      masked = cv2.bitwise_and(img, img, mask=mask)
      ############################################################################################################

      return deepcopy(masked)
    else:
      return img

  def channel_mask(self, img):
    options = self.options

    if options['logical_mask']:
      [c1_min, c1_max] = options['c1_range']
      [c2_min, c2_max] = options['c2_range']
      [c3_min, c3_max] = options['c3_range']

      # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      channels = cv2.split(img)
      c1, c2, c3 = channels[0], channels[1], channels[2]

      # MASKING
      ############################################################################################################
      m1 = ((c1 >= c1_min) * (c1 <= c1_max))
      m2 = ((c2 >= c2_min) * (c2 <= c2_max))
      m3 = ((c3 >= c3_min) * (c3 <= c3_max))

      # mask = np.reshape(m1, newshape=(img.shape[:2])).astype('uint8')
      # masked = cv2.bitwise_and(img, img, mask=mask)
      # masked = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
      ############################################################################################################
      return m1.astype('float') * 255., m2.astype('float') * 255., m3.astype('float') * 255.
    else:
      return img

  def tx(self, x):

    X = self.color_space(x)
    X = self.resize(X)
    X = self.mask(X)
    X = self.filter(X)
    self.id += 1
    X = self.hist(X)

    return X

  def ty(self, y):
    egfr, status = aux.egfr(c=y['concentration'], age=y['age'], male=y['male'], african=y['african'])

    d = {'age': y['age'], 'male': y['male'], 'african': y['african'],
         'status-gt': status, 'egfr-gt': egfr,
         'concentration-gt': y['concentration'],
         'bin': aux.get_bin(y['concentration']), 'uid': y['uid']}
    return d

  def filter(self, X):
    # Filter Type : median-k
    opt = self.options['filter_type']

    method, k = None, None
    try:
      method, k = opt.split("-")
    except:
      return X

    if method == 'median':
      median = cv2.medianBlur(src=X, ksize=int(k))
      return deepcopy(median)
    return X

  def resize(self, img):
    opt = self.options

    if opt['resize_method'] == 'uniform':
      return cv2.resize(img, [opt['img_w'], opt['img_h']])
    elif opt['resize_method'] == 'center':
      def crop_center(img, cropx, cropy):
        y, x, _ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

      return crop_center(img, cropx=opt['img_w'], cropy=opt['img_h'])
    else:
      return img

  def color_space(self, img_in):
    options = self.options
    img_out = None

    if self.override_color:
      options['color_space'] = 'hsv'
      self.options['c1_range'] = self.options['c2_range'] = self.options['c3_range'] = [0, 256]
      self.options['hist_bins'] = [256, 256, 256]

    if options['color_space'] == 'rgb':
      # img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
      # labels = ['red', 'green', 'blue']
      img_out = img_in[..., ::-1].copy()
    elif options['color_space'] == 'hsv':
      img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV_FULL)
      # img_out = cv2.cvtColor(np.float32(img_in), cv2.COLOR_BGR2HSV_FULL)
      # img_out[..., 0] /= 360.
      # img_out[..., 1] = (img_out[..., 1].copy())/255.
      # img_out[..., 2] = (img_out[..., 2].copy())/255.
      # labels = ['hue', 'saturation', 'value']
      if self.options['luma'] == False:
        self.options['channel_mask'] = [0, 1]
    elif options['color_space'] == 'hls':
      img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2HLS)
      # labels = ['hue', 'lightness', 'saturation']
      pass
    elif options['color_space'] == 'lab':
      img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)
      # img_out = cv2.cvtColor(img_in.astype(np.float32)/255., cv2.COLOR_BGR2LAB)
      # inc = 127.
      # img_out[..., 0] /= 100.
      # img_out[..., 1] = (img_out[..., 1].copy() + inc)/254.
      # img_out[..., 2] = (img_out[..., 2].copy() + inc)/254.

      # img_rgb = img_in[..., ::-1].copy()
      # img_out = sk.color.rgb2lab(img_rgb, illuminant=options['illuminant'], observer=options['observer'])
      # img_out[..., 0] = (img_out[..., 0] / 1.)
      # img_out[..., 1] = (img_out[..., 1] + 127.) / 1.
      # img_out[..., 2] = (img_out[..., 2] + 127.) / 1.

      # img_out = np.uint8(img_out)
      # labels = ['lightness', 'a (green-red)', 'b (blue--yellow)']
    elif options['color_space'] == 'YCrCb':
      img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2YCrCb)
      # labels = ['Y – Luminance', 'Cr = R – Y', 'Cb = B – Y']
    else:
      raise ValueError("Incorrect choice for color space. Choose one of hsv, rgb, lab, hls, or YCrCb.")

    return img_out

  def hist(self, img):
    options = self.options

    # imgf = np.float32(img.copy()) / 255.
    channels = cv2.split(img)

    c1, c2, c3 = channels[0], channels[1], channels[2]
    cs = [c1, c2, c3]

    [h1, h2, h3] = options['hist_bins']
    hs = [h1, h2, h3]

    [c1_min, c1_max] = options['c1_range']
    [c2_min, c2_max] = options['c2_range']
    [c3_min, c3_max] = options['c3_range']

    output = []

    if options['hist_method'] == 'channel-wise':
      # c1 = crimmins(c1)
      # c1 = crimmins(c1)
      # c1 = crimmins(c1)

      # c1 = median(c1, star(3))
      # c2 = median(c2, square(5))
      # c3 = median(c3, square(5)) # works

      c1_hist = cv2.calcHist(images=[c1], channels=[0], mask=None, histSize=[h1], ranges=[c1_min, c1_max])
      c2_hist = cv2.calcHist(images=[c2], channels=[0], mask=None, histSize=[h2], ranges=[c2_min, c2_max])
      c3_hist = cv2.calcHist(images=[c3], channels=[0], mask=None, histSize=[h3], ranges=[c3_min, c3_max])
      cs = [c1_hist, c2_hist, c3_hist]

      for ix in options['channel_mask']:
        self.total_pixels = img.shape[0] * img.shape[1]
        x = np.array(cs[ix]) 
        output.extend(x)

    elif options['hist_method'] == '2d':
      assert len(options['channel_mask']) == 2

      cs = [cs[ix] for ix in options['channel_mask']]
      hs = [hs[ix] for ix in options['channel_mask']]
      rs = [(c1_min, c1_max), (c2_min, c2_max), (c3_min, c3_max)]
      rs = [rs[ix] for ix in options['channel_mask']]
      _2d_hist = cv2.calcHist(
              images=cs, channels=[0, 1], mask=None, histSize=[hs[0], hs[1]],
              ranges=[rs[0][0], rs[0][1], rs[1][0], rs[1][1]]
              )
      output = _2d_hist

    elif options['hist_method'] == '3d':
      assert len(options['channel_mask']) == 3
      c123_hist = cv2.calcHist(
              images=[c1, c2, c3], channels=[0, 1, 2], mask=None, histSize=[h1, h2, h3],
              ranges=[c1_min, c1_max, c2_min, c2_max, c3_min, c3_max]
              )
      output = c123_hist
    elif options['hist_method'] == 'pixel':
      # Default Case
      # Return flattened channel-wise optionally normalized raw pixel values
      norm_fn = None
      
      if options['normalization'] == 'z_score':
        def fn(x):
          from sklearn.preprocessing import StandardScaler
          z_score = StandardScaler()
          return z_score.fit_transform(x)

        norm_fn = fn
      elif options['normalization'] == 'min_max':
        def fn(x):
          from sklearn.preprocessing import MinMaxScaler
          min_max = MinMaxScaler()
          return min_max.fit_transform(x)

        norm_fn = fn
      else:
        def fn(x):
          return x

        norm_fn = fn
      
      output = [norm_fn(np.array(cs[ix])) for ix in options['channel_mask']]

    if options['global_features']:
      output.extend(self.bin_features)

    output = np.array(output).flatten()
    return output

  def reseed(self, metadata):
    c = metadata['concentration']
    uid = metadata['uid']
    age, male, african = aux.distribution()
    egfr, status = aux.egfr(c, age, male, african)

    d = {'age': age, 'male': male, 'african': african,
         'status-gt': status, 'egfr-gt': egfr,
         'concentration-gt': c,
         'bin': aux.get_bin(c, self.options), 'uid': uid}
    return d

  def aug(self, img_in, metadata):
    img = self.color_space(img_in)
    img = self.resize(img)
    # img = self.mask(img)
    img = self.filter(img)

    # TODO: Use self.tx()

    if self.options["global_features"]:
      self.bin_features = self.bin_predictor_model.predict(img)

    # Tile into 4 Partitions
    '''
    X1 | X2
    -------
    X3 | X4
    '''
    y_max, x_max, _ = img.shape
    x1 = img[0:y_max // 2, 0:x_max // 2]
    x2 = img[0:y_max // 2, x_max // 2:x_max]
    x3 = img[y_max // 2:, 0:x_max // 2]
    x4 = img[y_max // 2:, x_max // 2:x_max]

    h1 = self.hist(x1)
    h2 = self.hist(x2)
    h3 = self.hist(x3)
    h4 = self.hist(x4)

    y1 = self.reseed(metadata)
    y2 = self.reseed(metadata)
    y3 = self.reseed(metadata)
    y4 = self.reseed(metadata)

    # TODO: DNN needs separate normalization
    if self.options['normalization'] == 'z_score':
      from sklearn.preprocessing import StandardScaler
      z_score = StandardScaler()

      for ix in self.options['channel_mask']:
        # x1[:, :, ix] = z_score.fit_transform(x1[:, :, ix])
        # x2[:, :, ix] = z_score.fit_transform(x2[:, :, ix])
        # x3[:, :, ix] = z_score.fit_transform(x3[:, :, ix])
        # x4[:, :, ix] = z_score.fit_transform(x4[:, :, ix])
        x1 = x1 / 1.
        x2 = x2 / 1.
        x3 = x3 / 1.
        x4 = x4 / 1.

    return [x1, x2, x3, x4], [h1, h2, h3, h4], [y1, y2, y3, y4]