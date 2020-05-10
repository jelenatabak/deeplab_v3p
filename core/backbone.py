import os

import tensorflow as tf
import tensorflow.keras.applications as keras_app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# To-do: check for newer xception model -> in paper they say there is newer version that works better


class BackboneSwitch():
  def __init__(self, model_name='ResNet50', include_top=True, weights=None, input_tensor=None,
               input_shape=(299, 299, 3), pooling=None, classes=1000):
    self.model_name = model_name.lower()
    self.include_top = include_top
    self.weights = weights
    self.input_tensor = input_tensor
    self.input_shape = input_shape
    self.pooling = pooling
    self.classes = classes

  def get_model(self):
    method = getattr(self, self.model_name, lambda: 'Invalid selection of backbone model!')
    return method()
  
  def xception(self):
    return keras_app.xception.Xception(include_top=self.include_top, weights=self.weights,
                                       input_tensor=self.input_tensor, input_shape=self.input_shape,
                                       pooling=self.pooling, classes=self.classes)

  def vgg16(self):
    return keras_app.vgg16.VGG16(include_top=self.include_top, weights=self.weights,
                                 input_tensor=self.input_tensor, input_shape=self.input_shape,
                                 pooling=self.pooling, classes=self.classes)

  def vgg19(self):
    return keras_app.vgg19.VGG19(include_top=self.include_top, weights=self.weights,
                                 input_tensor=self.input_tensor, input_shape=self.input_shape,
                                 pooling=self.pooling, classes=self.classes)

  def resnet50(self):
    return keras_app.resnet.ResNet50(include_top=self.include_top, weights=self.weights,
                                     input_tensor=self.input_tensor, input_shape=self.input_shape,
                                     pooling=self.pooling, classes=self.classes)
  
  def resnet101(self):
    return keras_app.resnet.ResNet101(include_top=self.include_top, weights=self.weights,
                                      input_tensor=self.input_tensor, input_shape=self.input_shape,
                                      pooling=self.pooling, classes=self.classes)

  def resnet152(self):
    return keras_app.resnet.ResNet152(include_top=self.include_top, weights=self.weights,
                                      input_tensor=self.input_tensor, input_shape=self.input_shape,
                                      pooling=self.pooling, classes=self.classes)

  def resnet50v2(self):
    return keras_app.resnet_v2.ResNet50V2(include_top=self.include_top, weights=self.weights,
                                          input_tensor=self.input_tensor, input_shape=self.input_shape,
                                          pooling=self.pooling, classes=self.classes)

  def resnet101v2(self):
    return keras_app.resnet_v2.ResNet101V2(include_top=self.include_top, weights=self.weights,
                                           input_tensor=self.input_tensor, input_shape=self.input_shape,
                                           pooling=self.pooling, classes=self.classes)

  def resnet152v2(self):
    return keras_app.resnet_v2.ResNet152V2(include_top=self.include_top, weights=self.weights,
                                           input_tensor=self.input_tensor, input_shape=self.input_shape,
                                           pooling=self.pooling, classes=self.classes)

  def inceptionv3(self):
    return keras_app.inception_v3.InceptionV3(include_top=self.include_top, weights=self.weights,
                                              input_tensor=self.input_tensor, input_shape=self.input_shape,
                                              pooling=self.pooling, classes=self.classes)

  def inceptionresnetv2(self):
    return keras_app.inception_resnet_v2.InceptionResNetV2(include_top=self.include_top, weights=self.weights,
                                                          input_tensor=self.input_tensor, input_shape=self.input_shape,
                                                          pooling=self.pooling, classes=self.classes)

  def mobilenet(self):
    return keras_app.mobilenet.MobileNet(include_top=self.include_top, weights=self.weights,
                                         input_tensor=self.input_tensor, input_shape=self.input_shape,
                                         pooling=self.pooling, classes=self.classes)

  def mobilenetv2(self):
    return keras_app.mobilenet_v2.MobileNetV2(include_top=self.include_top, weights=self.weights,
                                              input_tensor=self.input_tensor, input_shape=self.input_shape,
                                              pooling=self.pooling, classes=self.classes)

  def densenet121(self):
    return keras_app.densenet.DenseNet121(include_top=self.include_top, weights=self.weights,
                                          input_tensor=self.input_tensor, input_shape=self.input_shape,
                                          pooling=self.pooling, classes=self.classes)

  def densenet169(self):
    return keras_app.densenet.DenseNet169(include_top=self.include_top, weights=self.weights,
                                          input_tensor=self.input_tensor, input_shape=self.input_shape,
                                          pooling=self.pooling, classes=self.classes)

  def densenet201(self):
    return keras_app.densenet.DenseNet201(include_top=self.include_top, weights=self.weights,
                                          input_tensor=self.input_tensor, input_shape=self.input_shape,
                                          pooling=self.pooling, classes=self.classes)

  def nasnetlarge(self):
    return keras_app.nasnet.NASNetLarge(include_top=self.include_top, weights=self.weights,
                                        input_tensor=self.input_tensor, input_shape=self.input_shape,
                                        pooling=self.pooling, classes=self.classes)




def select_backbone(model_name='ResNet50', include_top=True, weights=None, 
                    input_tensor=None, input_shape=(299, 299, 3), pooling=None, classes=1000):
  model = BackboneSwitch(model_name=model_name, include_top=include_top, weights=weights, input_tensor=input_tensor,
                         input_shape=input_shape, pooling=pooling, classes=classes)              
  return model.get_model()




if __name__=='__main__':
  model = select_backbone(model_name='ResNet50', input_shape=(500, 500, 3))
  print('Backbone model summary:')
  print(model.summary())
