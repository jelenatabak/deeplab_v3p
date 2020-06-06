#!/usr/bin/env python
import time

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
    #plt.imshow(img)
    #plt.show()
    start = time.time()
    img = img.astype("float")
    mask = self.deeplab_predict.predict(img)
    end = time.time()
    print(end - start)
    plt.imshow(mask/255)
    plt.show()


  def __init__(self, model_path):
    self.bridge = CvBridge()
    self.deeplab_predict = DeeplabInference(model_path, ros_structure=True)
    rospy.sleep(10)
    self.sub = rospy.Subscriber("/image_raw", Image, self.callback, queue_size=1, buff_size=2**24)


  def run(self):
    while not rospy.is_shutdown():
      rospy.spin()




if __name__=='__main__':
  rospy.init_node('Deeplab', anonymous=True)
  rospack = rospkg.RosPack()
  model_path = rospack.get_path('deeplab_v3p') + '/tensorflow_models/model_2'

  try:
    deeplab_ros = DeeplabROS(model_path)
    deeplab_ros.run()
  except rospy.ROSInterruptException:
    pass
