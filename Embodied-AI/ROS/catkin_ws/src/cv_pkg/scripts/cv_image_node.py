#! /usr/bin/env python3
#coding=utf-8
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 

def Cam_Imu_Callback(msg):
    bridge=CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("格式转换错误: %s", e)
        return
    cv2.imshow("RGB", cv_image)
    cv2.waitKey(1)
    
if __name__ == '__main__':
    rospy.init_node('image_node')
    rgb_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, Cam_Imu_Callback,queue_size=10)

    rospy.spin()
