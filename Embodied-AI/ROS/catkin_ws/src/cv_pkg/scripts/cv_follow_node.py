#! /usr/bin/env python3
#coding=utf-8

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist


hue_min = 10
hue_max = 40
satu_min = 90
satu_max = 255
val_min = 1
val_max = 255

vel_cmd = Twist()
vel_pub = None

def Cam_RGB_Callback(msg):
    global hue_min, hue_max, satu_min, satu_max, val_min, val_max
    global vel_cmd, vel_pub
    bridge=CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("格式转换错误: %s", e)
        return
    
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v=cv2.equalizeHist(v)
    hsv_image = cv2.merge([h, s, v])
    th_image = cv2.inRange(hsv_image, (hue_min, satu_min, val_min), (hue_max, satu_max, val_max))

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th_image = cv2.morphologyEx(th_image, cv2.MORPH_OPEN, element)
    th_image = cv2.morphologyEx(th_image, cv2.MORPH_CLOSE, element)

    target_x, target_y, pix_count = 0, 0, 0
    image_height, image_width = th_image.shape[:2]
    for y in range(image_height):
        for x in range(image_width):
            if th_image[y, x] == 255:
                target_x += x
                target_y += y
                pix_count += 1

    if pix_count > 0:
        target_x //= pix_count
        target_y //= pix_count
        print(f"颜色质心坐标( {target_x} , {target_y} )  点数 = {pix_count}")
        cv2.line(cv_image, (target_x-10, target_y), (target_x+10, target_y), (255, 0, 0), 2)
        cv2.line(cv_image, (target_x, target_y-10), (target_x, target_y+10), (255, 0, 0), 2)
        #速度
        vel_forward = (image_height/2 - target_y) * 0.01
        vel_turn = (image_width/2 - target_x) * 0.01
        vel_cmd.linear.x = vel_forward
        vel_cmd.angular.z = vel_turn
    else:
        print("目标颜色消失...")
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0

    cv2.imshow("RGB", cv_image)
    cv2.imshow("HSV", hsv_image)
    cv2.imshow("Result", th_image)
    cv2.waitKey(5)

    vel_pub.publish(vel_cmd)
    print(f"机器人运动速度( linear.x= {vel_cmd.linear.x:.2f} , angular.z= {vel_cmd.angular.z:.2f} )")
    
def nothing(x):
    pass

if __name__ == '__main__':
    rospy.init_node('cv_hsv_node')
    rgb_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, Cam_RGB_Callback,queue_size=1)
    vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=30)
    cv2.namedWindow("Threshold")
    cv2.createTrackbar("hue_min", "Threshold", hue_min, 179, nothing)
    cv2.createTrackbar("hue_max", "Threshold", hue_max, 179, nothing)
    cv2.createTrackbar("satu_min", "Threshold", satu_min, 255, nothing)
    cv2.createTrackbar("satu_max", "Threshold", satu_max, 255, nothing)
    cv2.createTrackbar("val_min", "Threshold", val_min, 255, nothing)
    cv2.createTrackbar("val_max", "Threshold", val_max, 255, nothing)

    cv2.namedWindow("RGB")
    cv2.namedWindow("HSV")
    cv2.namedWindow("Result")

    rate=rospy.Rate(30)
    while not rospy.is_shutdown():
        #获取此时的滑杆对应的值
        hue_min = cv2.getTrackbarPos("hue_min", "Threshold")
        hue_max = cv2.getTrackbarPos("hue_max", "Threshold")
        satu_min = cv2.getTrackbarPos("satu_min", "Threshold")
        satu_max = cv2.getTrackbarPos("satu_max", "Threshold")
        val_min = cv2.getTrackbarPos("val_min", "Threshold")
        val_max = cv2.getTrackbarPos("val_max", "Threshold")
        rate.sleep()

    cv2.destroyAllWindows()

