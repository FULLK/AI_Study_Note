#! /usr/bin/env python3
#coding=utf-8

import rospy
from std_msgs.msg import String
if __name__=="__main__":
    rospy.init_node("yao_node")
    pub=rospy.Publisher("yao",String,queue_size=10)
    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        msg=String()
        msg.data="do this all day woman!!!"
        pub.publish(msg)
        rospy.loginfo("woman!!!")
        rate.sleep()
