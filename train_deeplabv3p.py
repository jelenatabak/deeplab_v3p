import datetime
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from core.dataset_funcs import (calculate_weights_for_classes,
                                create_directory, visualize_dataset)
from core.dataset_preprocessing import DatasetPreprocessing
from core.deeplabv3p_model import DeeplabV3p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# To-do: Improve last visualization by using class for inference -> after it is created
#        Add evaluation


from_tensor_slices = tf.data.Dataset.from_tensor_slices
decode_png = tf.image.decode_png
resize = tf.image.resize
read_file = tf.io.read_file
AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_dataset(image, mask, num_classes, size=(500,500)):
  img = read_file(image)
  img = decode_png(contents=img, channels=3)  # output is RGB
  img = resize(images=img, size=size)
  img = tf.cast(x=img, dtype=tf.float32)
  imagenet_normalization = tf.constant([103.939, 116.779, 123.68]) # For normalization data around the ImageNet mean (BGR) on which the classifier was initialy trained
  img = img[:,:,::-1] - imagenet_normalization

  mask = read_file(mask)
  mask = decode_png(contents=mask, channels=1)
  mask = resize(images=mask, size=size)
  mask = tf.cast(x=mask, dtype=tf.uint8)
  mask = tf.one_hot(tf.squeeze(mask), depth=num_classes)
  return img, mask


def dataset_pipeline(images, masks, batch_size, num_classes, size=(500,500)):
  dataset = from_tensor_slices((images, masks))
  dataset = dataset.shuffle(buffer_size=200)
  dataset = dataset.map(lambda x,y: load_dataset(x, y, num_classes, size), num_parallel_calls=AUTOTUNE)
  dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
  dataset = dataset.repeat(5)
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset



flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('preprocess_dataset', False, 'Preprocess dataset before fiting model.')
flags.DEFINE_boolean('class_weights', False, 'Calculate class weights for training.')
flags.DEFINE_integer('batch_size', 16, 'Model batch size.')
flags.DEFINE_integer('brightness_value', 50, 'Increasing brightness of image in dataset preprocessing by value of.')
flags.DEFINE_integer('num_of_classes', 6, 'Number of different masks.')
flags.DEFINE_integer('epochs', 1000, 'Epochs')
flags.DEFINE_list('image_shape', '500,500,3', 'Shape of input image for model.')
flags.DEFINE_list('dataset_split_size', '7500,2500,500', 'Number of images in training, validation and test sets.')
flags.DEFINE_string('backbone_model', 'ResNet50', 'Backbone model for DeeplabV3+.')
flags.DEFINE_string('img_dir', '/home/matija/Downloads/data/data/synthetic_image_color/', 'Path to raw images.')
flags.DEFINE_string('mask_dir', '/home/matija/Downloads/data/data/synthetic_label_class_grayscale/synthetic_label_class_all_grayscale/',
                    'Path to raw masks.')


batch_size = FLAGS.batch_size
brightness_value = FLAGS.brightness_value
num_of_classes = FLAGS.num_of_classes
image_shape = [int(value) for value in FLAGS.image_shape]
image_shape = tuple(image_shape)
dataset_split_size = [int(values) for values in FLAGS.dataset_split_size]
epochs = FLAGS.epochs
class_id_reduction = ((6,4),(7,4))
mask_id_to_color = {0: (0, 0, 0),     # BGR
                    1: (245, 49, 0),
                    2: (82, 252, 255),
                    3: (30, 0, 255),
                    4: (243, 22, 255),
                    5: (76, 254, 0),
                    6: (254, 254, 254),
                    7: (255, 255, 0)}
for i in range(np.shape(class_id_reduction)[0]):
    mask_id_to_color.pop(class_id_reduction[i][0])

current_dir = os.path.abspath(os.getcwd())
data_dir = current_dir + '/capricum_annuum_dataset/'

if FLAGS.preprocess_dataset:
  data_preprocess = DatasetPreprocessing(img_dir=FLAGS.img_dir, mask_dir=FLAGS.mask_dir, output_dir=data_dir, 
                                         num_classes=num_of_classes, class_id_reduction=class_id_reduction,
                                         bright_value=brightness_value, dataset_split_size=dataset_split_size)
  num_of_classes = data_preprocess.get_number_of_clasess()


train_img_dir = data_dir + 'train/images/'
train_mask_dir = data_dir + 'train/masks/'
validation_img_dir = data_dir + 'validation/images/'
validation_mask_dir = data_dir + 'validation/masks/'
test_img_dir = data_dir + 'test/images/'
test_mask_dir = data_dir + 'test/masks/'

train_img = np.sort(np.array([os.path.join(train_img_dir, img_name) for img_name in os.listdir(train_img_dir)]))
train_mask = np.sort(np.array([os.path.join(train_mask_dir, mask_name) for mask_name in os.listdir(train_mask_dir)]))
validation_img = np.sort(np.array([os.path.join(validation_img_dir, img_name) for img_name in os.listdir(validation_img_dir)]))
validation_mask = np.sort(np.array([os.path.join(validation_mask_dir, mask_name) for mask_name in os.listdir(validation_mask_dir)]))
test_img = np.sort(np.array([os.path.join(test_img_dir, img_name) for img_name in os.listdir(test_img_dir)]))
test_mask = np.sort(np.array([os.path.join(test_mask_dir, mask_name) for mask_name in os.listdir(test_mask_dir)]))

print('Number of images in training folder: {}'.format(len(train_img)))
print('Number of images in validation folder: {}'.format(len(validation_img)))
print('Number of images in test folder: {}'.format(len(test_img)))


if FLAGS.class_weights:
  class_weights = calculate_weights_for_classes(path_to_masks=train_mask, num_classes=num_of_classes)
else:
  class_weights = [2.20770018, 1., 5.76543898, 240.53078806, 11.79186119, 9.29862377]


# Visualize first image and corresponding mask in training set
visualize_dataset(img_path=train_img[0], mask_path=train_mask[0], num_classes=num_of_classes,
                  shape=(2,3), name='Visualizing first image from train set with corresponding masks')


# Defining dataset pipeline for training and evaluating model
train_dataset = dataset_pipeline(images=train_img, masks=train_mask, batch_size=batch_size, num_classes=num_of_classes)
validation_dataset = dataset_pipeline(images=validation_img, masks=validation_mask, batch_size=batch_size, num_classes=num_of_classes)
test_dataset = dataset_pipeline(images=test_img, masks=test_mask, batch_size=batch_size, num_classes=num_of_classes)


# Create model, define model parameters and fit model
deeplab = DeeplabV3p(backbone_model_name=FLAGS.backbone_model, img_size=image_shape, num_classes=num_of_classes) 
model = deeplab.create_model()

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.9997
    elif isinstance(layer, tf.keras.layers.Conv2D):
        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
losses = weighted_categorical_crossentropy(class_weights)
metrics = [tf.keras.metrics.Accuracy(), 
           tf.keras.metrics.CategoricalAccuracy()]
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=5)

log_dir = current_dir + '/log/'
create_directory(dir_path=log_dir)
log_dir += datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                      write_images=True, update_freq='batch')
callbacks = [callbacks, tensorboard_callback]

model.compile(optimizer=optimizer,
              loss=losses,
              metrics=metrics)

model.fit(train_dataset,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=validation_dataset,
          steps_per_epoch=len(train_img)//batch_size,
          validation_steps=len(validation_img)//batch_size,
          validation_freq=1)


# Save model
print('Saving model...')
model_dir = current_dir + '/model/'
create_directory(dir_path=model_dir)
model_dir += datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
model.save(model_dir)

model.save_weights(model_dir + '_weights')
print('Model is successfully saved to {} location.'.format(model_dir))


# Testing model on two random image in test set
imagenet_normalization = [103.939, 116.779, 123.68]

for img_path in test_img[np.random.choice(len(test_img), 2, replace=False)]:
  img = cv2.imread(img_path)    # BGR
  img_process = img.copy()
  img_process[:,:,0] -= imagenet_normalization[0]
  img_process[:,:,1] -= imagenet_normalization[1]
  img_process[:,:,2] -= imagenet_normalization[2]
  img_process = np.expand_dims(img_process, axis=0)

  prediction = model.predict(img_process)               # Shape (batch, h, w, channels)
  prediction = np.squeeze(prediction)                   # Shape (h, w, channels)
  prediction = np.argmax(prediction, axis=2)            # Shape (index_of_class)

  name = test_mask_dir + img_path.split('/')[-1]
  ground_truth = cv2.imread(name)
  iou = []
  for i in range(num_of_classes):
    intersection = np.logical_and(prediction==i, ground_truth[:,:,0]==i)
    union = np.logical_or(prediction==i, ground_truth[:,:,0]==i)
    iou.append(np.sum(intersection) / np.sum(union))
  print(iou)

  mask = img.copy()
  for i in mask_id_to_color:
      mask[prediction==i] = mask_id_to_color[i]
  
  img_with_mask = img.copy()
  cv2.addWeighted(src1=img, alpha=0.5, src2=mask, beta=0.5, gamma=0, dst=img_with_mask)
  

  fig = plt.figure(figsize = (10,10))
  axs = np.zeros(3, dtype=object)
  gs = fig.add_gridspec(4, 4, height_ratios=[2,2,2,2])
  axs[0] = fig.add_subplot(gs[0:2,1:3])
  axs[1] = fig.add_subplot(gs[2:4,0:3])
  axs[2] = fig.add_subplot(gs[2:4,2:4])
  
  axs[0].imshow(img_with_mask)
  axs[0].set_title('Original image with predicted mask')
  axs[1].imshow(img)
  axs[1].set_title('Original image')
  axs[2].imshow(mask)
  axs[2].set_title('Predicted mask')
  plt.show()














# for evaluation on test dataset
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
# test_dataset = test_dataset.batch(BATCH_SIZE)
# model.evaluate(test_dataset)
