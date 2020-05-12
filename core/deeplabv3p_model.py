import warnings

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D,
                                     DepthwiseConv2D, Lambda, ZeroPadding2D,
                                     concatenate)
from tensorflow.keras.models import Model

from DeeplabV3p.core.backbone import select_backbone


class DeeplabV3p():
  def __init__(self, backbone_model_name='ResNet50', img_size=(500,500,3), num_classes=10):
    self.backbone_model_name = backbone_model_name

    if (backbone_model_name.lower() == 'mobilenet') and (img_size[0] != 224) and (img_size[1] != 224):
      raise NameError('img_size should be (331,331,3) while using NASNetLarge pretrained weights.')
    elif (backbone_model_name.lower() == 'mobilenetv2') and (img_size[0] != 224) and (img_size[1] != 224):
      raise NameError('img_size should be (224,224,3) while using MobileNetV2 pretrained weights.')
    elif backbone_model_name.lower() == 'nasnetlarge' and (img_size[0] != 331) and (img_size[1] != 331):
      raise NameError('img_size should be (331,331,3) while using NASNetLarge pretrained weights.')

    self.h = img_size[0]
    self.w = img_size[1]
    self.channels = img_size[2]
    self.num_classes = num_classes
    self.layer_name = {'xception': ['block13_sepconv2_act', 'block3_sepconv2_act'],         # 728, 256
                       'vgg16': ['block5_conv3', 'block3_conv3'],                           # 512, 256
                       'vgg19': ['block5_conv4', 'block3_conv4'],                           # 512, 256
                       'resnet50': ['conv4_block6_2_relu', 'conv2_block3_2_relu'],          # 256, 64
                       'resnet101': ['conv4_block23_out', 'conv2_block3_out'],              # 1024, 256
                       'resnet152': ['conv4_block36_out', 'conv2_block3_out'],              # 1024, 256
                       'resnet50v2': ['conv4_block6_1_relu', 'conv2_block3_preact_relu'],   # 256, 256
                       'resnet101v2': ['conv4_block23_1_relu', 'conv2_block3_preact_relu'], # 256, 256
                       'resnet152v2': ['conv4_block36_1_relu', 'conv2_block3_preact_relu'], # 256, 256
                       'inceptionv3': ['activation_74', 'activation_4'],                    # 192, 192
                       'inceptionresnetv2': ['activation_158', 'activation_4'],             # 256, 192
                       'mobilenet': ['conv_pw_11_relu', 'conv_pw_3_relu'],                  # 512, 128
                       'mobilenetv2': ['block_13_expand_relu', 'block_3_expand_relu'],      # 576, 144
                       'densenet121': ['pool4_relu', 'pool2_relu'],                         # 1024, 256
                       'densenet169': ['pool4_relu', 'pool2_relu'],                         # 1280, 256
                       'densenet201': ['pool4_relu', 'pool2_relu'],                         # 1792, 256
                       'nasnetlarge': ['activation_180', 'activation_11']}                  # 672, 168
  

  def depthwise_separable_convolution(self, tensor, filters=256, kernel_size_depth=(3,3), kernel_size_point=(1,1),
                                      dilation_rate=1, padding_depth='same', padding_point='same', 
                                      use_bias_depth=False, use_bias_point=False, name='depthwise_separable_convolution'):

    tensor = DepthwiseConv2D(kernel_size=kernel_size_depth, dilation_rate=dilation_rate, padding=padding_depth,
                               name=name+'_depthwiseConv2d', use_bias=use_bias_depth)(tensor)
    tensor = BatchNormalization(name=name+'_batchNorm_depth')(tensor)
    tensor = Activation('elu', name=name+'_activation_depth')(tensor)
    
    tensor = Conv2D(filters=filters, kernel_size=kernel_size_point, dilation_rate=1, padding=padding_point,
                      kernel_initializer='he_normal', name=name+'_poinwiseConv2d', use_bias=use_bias_point)(tensor)
    tensor = BatchNormalization(name=name+'_batchNorm_point')(tensor)
    tensor = Activation('elu', name=name+'_activation_point')(tensor)
    return tensor


  def ASPP(self, backbone_tensor):
    tensor_1 = self.depthwise_separable_convolution(backbone_tensor, filters=256, kernel_size_depth=(1,1), 
                                                    kernel_size_point=(1,1), dilation_rate=1, padding_depth='same', 
                                                    padding_point='same', use_bias_depth=False, 
                                                    use_bias_point=False, name='ASPP_tensor_1')

    tensor_6 = self.depthwise_separable_convolution(backbone_tensor, filters=256, kernel_size_depth=(3,3), 
                                                    kernel_size_point=(1,1), dilation_rate=6, padding_depth='same', 
                                                    padding_point='same', use_bias_depth=False, 
                                                    use_bias_point=False, name='ASPP_tensor_6')

    tensor_12 = self.depthwise_separable_convolution(backbone_tensor, filters=256, kernel_size_depth=(3,3), 
                                                     kernel_size_point=(1,1), dilation_rate=12, padding_depth='same', 
                                                     padding_point='same', use_bias_depth=False, 
                                                     use_bias_point=False, name='ASPP_tensor_12')

    tensor_18 = self.depthwise_separable_convolution(backbone_tensor, filters=256, kernel_size_depth=(3,3), 
                                                     kernel_size_point=(1,1), dilation_rate=18, padding_depth='same', 
                                                     padding_point='same', use_bias_depth=False, 
                                                     use_bias_point=False, name='ASPP_tensor_18')

    dim = backend.int_shape(backbone_tensor)
    tensor_pool = AveragePooling2D(pool_size=(dim[1], dim[2]), name = 'ASPP_avgPooling_tensor_pool')(backbone_tensor)
    tensor_pool = Conv2D(filters=256, kernel_size= (1,1), dilation_rate=1, padding='same',
                         kernel_initializer='he_normal', name='ASPP_conv2d_tensor_pool', use_bias=False)(tensor_pool)
    tensor_pool = BatchNormalization(name='ASPP_batchNorm_tensor_pool')(tensor_pool)
    tensor_pool = Activation('elu', name='ASPP_activation_tensor_pool')(tensor_pool)
    tensor_pool = Lambda(lambda x: tf.image.resize(images=x, size=[dim[1], dim[2]]), output_shape=[dim[1], dim[2]], 
                         name='ASPP_bilinear_upsample')(tensor_pool)

    tensor = concatenate([tensor_1, tensor_6, tensor_12, tensor_18, tensor_pool], name='ASPP_concatenate')

    tensor = Conv2D(filters=256, kernel_size=(1,1), dilation_rate=1, padding='same',
                    kernel_initializer='he_normal', name='ASPP_conv2d_tensor', use_bias=False)(tensor)
    tensor = BatchNormalization(name='ASPP_batchNorm_tensor')(tensor)
    tensor = Activation('elu', name='ASPP_activation_tensor')(tensor)
    return tensor


  def decoder(self, aspp_tensor, backbone_tensor):
    backbone_tensor = Conv2D(filters=48, kernel_size=(1,1), dilation_rate=1, padding='same',
                             kernel_initializer='he_normal', name='decoder_conv2d_backbone', use_bias=False)(backbone_tensor)
    backbone_tensor = BatchNormalization(name='decoder_batchNorm_backbone')(backbone_tensor)
    backbone_tensor = Activation('elu', name = 'decoder_activation_backbone')(backbone_tensor)

    tensor = concatenate([aspp_tensor, backbone_tensor], name='decoder_concatenate')

    tensor = self.depthwise_separable_convolution(tensor, filters=256, kernel_size_depth=(3,3), 
                                                  kernel_size_point=(1,1), dilation_rate=1, padding_depth='same', 
                                                  padding_point='same', use_bias_depth=False, 
                                                  use_bias_point=False, name='decoder_1')

    tensor = self.depthwise_separable_convolution(tensor, filters=256, kernel_size_depth=(3,3), 
                                                  kernel_size_point=(1,1), dilation_rate=1, padding_depth='same', 
                                                  padding_point='same', use_bias_depth=False, 
                                                  use_bias_point=False, name='decoder_2')

    size = [self.h, self.w]
    tensor = Lambda(lambda x: tf.image.resize(images=x, size=size), output_shape=size, 
                    name='decoder_bilinear_upsample')(tensor)
    tensor = Conv2D(filters=self.num_classes, kernel_size=(1,1), name='output_layer')(tensor)
    return tensor


  def create_model(self):
    backbone_model = select_backbone(model_name=self.backbone_model_name, include_top=False,
                                     weights='imagenet', input_shape=(self.h, self.w, self.channels))
    backbone_model_layer_aspp = backbone_model.get_layer(self.layer_name[self.backbone_model_name.lower()][0]).output
    backbone_model_layer_decoder = backbone_model.get_layer(self.layer_name[self.backbone_model_name.lower()][1]).output

    dim_backbone_decoder = backend.int_shape(backbone_model_layer_decoder)
    if (dim_backbone_decoder[0] != self.h//4) and (dim_backbone_decoder[1] != self.w//4):
      size = [self.h//4, self.w//4]
      backbone_model_layer_decoder = Lambda(lambda x: tf.image.resize(images=x, size=size), output_shape=size, 
                                            name='backbone_model_layer_decoder_bilinear_upsample')(backbone_model_layer_decoder)

    aspp_output_layer = self.ASPP(backbone_model_layer_aspp)
    size = [self.h//4, self.w//4]
    aspp_output_layer = Lambda(lambda x: tf.image.resize(images=x, size=size), output_shape=size, 
                               name='aspp_output_layer_bilinear_upsample')(aspp_output_layer)

    output_layer = self.decoder(aspp_tensor=aspp_output_layer, backbone_tensor=backbone_model_layer_decoder)
    model = Model(inputs = backbone_model.input, outputs = output_layer, name = 'DeeplabV3p')
    return model




if __name__=='__main__':
  deeplab = DeeplabV3p(backbone_model_name='nasnetlarge', img_size=(500,500,3), num_classes=8)
  model = deeplab.create_model()
  print(model.summary())
