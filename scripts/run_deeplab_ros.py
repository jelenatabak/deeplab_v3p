#!/usr/bin/env python
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import matplotlib.pyplot as plt
import numpy as np
import roslib
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from run_deeplab import DeeplabInference


class DeeplabROS():
  def callback(self, data):
    try:
      img = self.bridge.imgmsg_to_cv2(data,'bgr8')
    except CvBridgeError as e:
      print(e)

    img = img.astype("float")
    img = img[:,80:560,:]
    #img = cv2.resize(img, (640,640))
    mask = self.deeplab_predict.predict(img)

    #plt.imshow(mask/255)
    mask = mask.astype(np.uint8)
    mask_msg = self.bridge.cv2_to_imgmsg(mask, 'rgb8')
    mask_msg.header = data.header
    self.mask_pub.publish(mask_msg)
    rospy.loginfo("Published mask")



  def __init__(self, model_path):
    self.bridge = CvBridge()
    self.deeplab_predict = DeeplabInference(model_path, ros_structure=True)
    rospy.sleep(10)
    self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback, queue_size=1)
    self.mask_pub = rospy.Publisher('/segmentation_mask', Image, queue_size=1)


  def run(self):
    while not rospy.is_shutdown():
      rospy.spin()




if __name__=='__main__':
  rospy.init_node('Deeplab', anonymous=True)
  rospack = rospkg.RosPack()
  model_path = rospack.get_path('deeplab_v3p') + '/tensorflow_models/leaves_pepper_basic'

  try:
    deeplab_ros = DeeplabROS(model_path)
    deeplab_ros.run()
  except rospy.ROSInterruptException:
    pass
