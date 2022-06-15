#!/usr/bin/env python3

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from deeplab_v3p.srv import Inference
from ros_object_detector.msg import DetectedObjectArray

def callback(img_msg):
    rospy.loginfo("In cbck")
    rospy.wait_for_service("/inference")
    rospy.loginfo("Service available!")

    inf = rospy.ServiceProxy('/inference', Inference)
    res = inf(img_msg)

    mask_pub = rospy.Publisher("/mask", Image, queue_size=1)
    mask_pub.publish(res.mask)
    rospy.loginfo("Published mask")


if __name__ == "__main__":
    rospy.init_node("test")
    img_sub = rospy.Subscriber('/camera/color/image_raw', Image, callback)
    rospy.spin()
