import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from core.deeplabv3p_model import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


class DeeplabInference():
  def __init__(self, model_path, ros_structure=False):
    self.model = tf.keras.models.load_model(model_path, custom_objects={"tf": tf})
    self.ros_structure = ros_structure
    self.imagenet_normalization = [103.939, 116.779, 123.68]
    self.mask_id_to_color = {0: (0, 0, 0),     # RGB
                             1: (0, 49, 245),
                             2: (255, 252, 82),
                             3: (255, 0, 30),
                             4: (255, 22, 243),
                             5: (0, 254, 76)}
    

  def predict(self, img):
    img_process = img.copy()
    img_process[:,:,0] -= self.imagenet_normalization[0]
    img_process[:,:,1] -= self.imagenet_normalization[1]
    img_process[:,:,2] -= self.imagenet_normalization[2]
    img_process = np.expand_dims(img_process, axis=0)

    prediction = self.model.predict(img_process)          # Shape (batch, h, w, channels)
    prediction = np.squeeze(prediction)                   # Shape (h, w, channels)
    prediction = np.argmax(prediction, axis=2)            # Shape (index_of_class)

    mask = img.copy()
    for i in self.mask_id_to_color:
      mask[prediction==i] = self.mask_id_to_color[i]

    if self.ros_structure:
      return mask
    else:
      self.visualize(img, mask)


  def visualize(self, img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_mask = img.copy()
    cv2.addWeighted(src1=img, alpha=0.5, src2=mask, beta=0.5, gamma=0, dst=img_with_mask)

    fig = plt.figure(figsize = (10,10))
    axs = np.zeros(3, dtype=object)
    gs = fig.add_gridspec(4, 4, wspace=0.5)
    axs[0] = fig.add_subplot(gs[0:2,1:3])
    axs[1] = fig.add_subplot(gs[2:4,0:2])
    axs[2] = fig.add_subplot(gs[2:4,2:4])
    
    axs[0].imshow(img_with_mask/255)
    axs[0].set_title('Original image with predicted mask')
    axs[1].imshow(img/255)
    axs[1].set_title('Original image')
    axs[2].imshow(mask/255)
    axs[2].set_title('Predicted mask')
    plt.show(block=False)




if __name__=='__main__':
  current_dir = os.path.abspath(os.getcwd())
  model_path = current_dir + '/tensorflow_models/model_1'
  data_dir = current_dir + '/dataset/capricum_annuum_dataset/'
  test_img_dir = data_dir + 'test/images/'
  
  test_img = np.sort(np.array([os.path.join(test_img_dir, img_name) for img_name in os.listdir(test_img_dir)]))

  deeplab_predict = DeeplabInference(model_path, ros_structure=False)

  for img_path in test_img[np.random.choice(len(test_img), 2, replace=False)]:
    img = cv2.imread(img_path).astype(np.float32)
    deeplab_predict.predict(img)

  input('Press ENTER to exit')