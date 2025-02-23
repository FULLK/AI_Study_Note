#! /usr/bin/env python3
#coding=utf-8

import rospy
from std_msgs.msg import String
from qq_msgs.msg import Carry
if __name__=="__main__":
    rospy.init_node("chao_node")
    pub=rospy.Publisher("chao",Carry,queue_size=10)
    rate=rospy.Rate(10)
    
    while not rospy.is_shutdown():
        msg=Carry()
        msg.grade="man"
        msg.star=100
        msg.data="do this all day man!!!"
        pub.publish(msg)
        rospy.loginfo("man!!!")
        rate.sleep()
