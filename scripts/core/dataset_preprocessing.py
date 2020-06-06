import os
from copy import deepcopy

import cv2
import numpy as np

from scripts.core.dataset_funcs import (create_directory, increase_brightness,
                           visualize_dataset)


class DatasetPreprocessing():
  def __init__(self, img_dir, mask_dir, output_dir, num_classes=8, class_id_reduction=((6,4),(7,4)),
               bright_value=0, dataset_split_size=(7500, 2500, 500)):
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.output_dir = output_dir
    self.num_classes = num_classes
    self.class_id = np.array(class_id_reduction)
    self.brightness_value = bright_value
    self.train_size = dataset_split_size[0]
    self.val_size = dataset_split_size[1]
    self.test_size = dataset_split_size[2]

  
  def get_number_of_clasess(self):
    return self.num_classes-self.class_id.shape[0]


  def crop_image(self, img, size=((50,550), (150,650))):
    return img[size[0][0]:size[0][1], size[1][0]:size[1][1]]


  def reduce_num_of_classes(self, mask):
    class_id = deepcopy(self.class_id)
    num_classes = self.num_classes
    for i in range(class_id.shape[0]):
      mask[mask==class_id[i][0]] = class_id[i][1]
 
    for i in range(class_id.shape[0]):
      class_id[i] -= i
      for j in range(class_id[i][0]+1, num_classes):
        mask[mask==j] = j-1
      num_classes -= 1
    return mask


  def img_process(self, path, output_dir, mask=False):
    img = cv2.imread(path)
    img = self.crop_image(img)
    if mask:
      img = self.reduce_num_of_classes(img)
    else:
      img = increase_brightness(img, value=self.brightness_value)
    cv2.imwrite(output_dir + path.split('_')[-1], img)


  def split_dataset(self, img_dir, mask_dir, size):
    name_array = np.array(os.listdir(img_dir[0]))
    rand_index = np.random.choice(name_array.shape[0], size, replace=False)
    try:
      for name in name_array[rand_index]:
        os.rename(img_dir[0] + name, img_dir[1] + name)
        os.rename(mask_dir[0] + name, mask_dir[1] + name)
    except OSError as error:
      print(error)


  def start_preprocessing_dataset(self):
    train_img_dir = self.output_dir + 'train/images/'
    train_mask_dir = self.output_dir + 'train/masks/'
    validation_img_dir = self.output_dir + 'validation/images/'
    validation_mask_dir = self.output_dir + 'validation/masks/'
    test_img_dir = self.output_dir + 'test/images/'
    test_mask_dir = self.output_dir + 'test/masks/'

    splited_data_directories = [train_img_dir, train_mask_dir,
                                validation_img_dir, validation_mask_dir,
                                test_img_dir, test_mask_dir]

    for dir_path in splited_data_directories:
      create_directory(dir_path)
    
    img_names = np.sort(np.array([os.path.join(self.img_dir, img_name) for img_name in os.listdir(self.img_dir)]))
    mask_names = np.sort(np.array([os.path.join(self.mask_dir, mask_name) for mask_name in os.listdir(self.mask_dir)]))

    visualize_dataset(img_path=img_names[0], mask_path=mask_names[0], num_classes=self.num_classes, 
                      name='Visualizing raw image and corresponding masks')

    print('Processing image files...')
    for name in img_names:
      self.img_process(path=name, mask=False, output_dir=train_img_dir)
    print('Processing image files successfully ended.')

    print('Processing mask files...')
    for name in mask_names:
      self.img_process(path=name, mask=True, output_dir=train_mask_dir)
    self.num_classes = self.get_number_of_clasess()
    print('Processing mask files successfully ended.')

    print('Started spliting dataset...')
    self.split_dataset(img_dir=(train_img_dir, validation_img_dir), 
                       mask_dir=(train_mask_dir, validation_mask_dir), size=self.val_size)
    self.split_dataset(img_dir=(train_img_dir, test_img_dir), 
                       mask_dir=(train_mask_dir, test_mask_dir), size=self.test_size)
    print('Spliting dataset successfully ended.')

    visualize_dataset(img_path=train_img_dir + np.sort(os.listdir(train_img_dir))[0],
                      mask_path=train_mask_dir + np.sort(os.listdir(train_mask_dir))[0],
                      shape=(2,3), num_classes=self.num_classes, 
                      name='Visualizing preprocessed image and corresponding masks')




if __name__ == '__main__':
  img_dir = '/home/matija/Downloads/data/data/synthetic_image_color/'
  mask_dir = '/home/matija/Downloads/data/data/synthetic_label_class_grayscale/synthetic_label_class_all_grayscale/'
  output_dir = '/home/matija/FER/Masters/DeeplabV3p/capricum_annuum_dataset/'
  data_preprocess = DatasetPreprocessing(img_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir, 
                                         num_classes=8, class_id_reduction=((6,4),(7,4)),
                                         bright_value=50, dataset_split_size=(7500, 2500, 500))
  data_preprocess.start_preprocessing_dataset()
