#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import roslib
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from run_deeplab import DeeplabInference
from deeplab_v3p.srv import Inference, InferenceResponse, InferenceRequest


class InferenceServer:
    def __init__(self):
        self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()
        self.model_path = self.rospack.get_path('deeplab_v3p') + '/tensorflow_models/leaves_pepper_basic'
        self.deeplab_predict = DeeplabInference(self.model_path, ros_structure=True)
        self.mask_pub = rospy.Publisher('/segmentation_mask', Image, queue_size=1)
        self.srv = rospy.Service('inference', Inference, self.inference)
        rospy.spin()

    def inference(self, req):
        rospy.loginfo("In inference server.");

        #test_pub = rospy.Publisher('/test', Image, queue_size=1)
        #test_pub.publish(req.image)

        img = self.bridge.imgmsg_to_cv2(req.image, 'bgr8')
        rospy.loginfo("Loaded image in server")
        img = img.astype("float32")

        img = img[:,80:560,:]

        mask = self.deeplab_predict.predict(img)
        mask = mask.astype(np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask, 'rgb8')
        mask_msg.header = req.image.header

        self.mask_pub.publish(mask_msg)
        rospy.loginfo("Published mask.");
        return InferenceResponse(mask_msg)

if __name__ == "__main__":
    rospy.init_node('deeplab_server')
    inf = InferenceServer()
