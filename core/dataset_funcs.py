import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_directory(dir_path):
  if not os.path.exists(dir_path):
    try:
        os.makedirs(dir_path)
        print('Directory {} is created.'.format(dir_path.split('/')[-2]))
    except OSError as error:
        print(error)
  else:
    print('Directory {} already exists.'.format(dir_path.split('/')[-2]))


def increase_brightness(img, value=0):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(hsv)

  lim = 255 - value
  v[v > lim] = 255
  v[v <= lim] += value

  hsv = cv2.merge((h, s, v))
  img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return img


def visualize_dataset(img_path, mask_path, num_classes=8, shape=(2,4), name='Visualizing image and mask'):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  mask = cv2.imread(mask_path)

  fig = plt.figure(figsize = (10,10))
  plt.suptitle(name, fontsize=16)
  axs = np.zeros(1+num_classes, dtype=object)
  row = 2 + shape[0]
  column = shape[1]
  gs = fig.add_gridspec(row, column*4)
  axs[0] = fig.add_subplot(gs[0:2,column:(3*column)])
  for i in range(2, row):
    for j in range(0,column):
      index = (i-2)*column+j+1
      if index > num_classes:
        break
      axs[index] = fig.add_subplot(gs[i, (j*4):((j+1)*4)])

  axs[0].imshow(img)
  axs[0].set_title('Original image')

  for i in range(num_classes):
    axs[i+1].imshow((mask == i)*1.0)
    axs[i+1].set_title('label = {}'.format(i))
  fig.tight_layout()
  fig.subplots_adjust(top=0.92)
  plt.show()