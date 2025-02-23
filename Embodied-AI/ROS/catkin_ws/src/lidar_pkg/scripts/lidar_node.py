#! /usr/bin/env python3
#coding=utf-8

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
count=0
def LidarCallback(msg):
    global vel_pub
    global count
    FMidDist=msg.ranges[180]
    rospy.loginfo("前方测距 ranges[180] = %.2f 米",FMidDist)
    if count>0:
        count=count-1
        return
    vel_cmd=Twist()
    if FMidDist<1.5:
        vel_cmd.angular.z=0.5
        count=30
    else:
        vel_cmd.linear.x=0.5
    vel_pub.publish(vel_cmd)

if __name__=="__main__":
    rospy.init_node("lidar_node")
    lidar_sub=rospy.Subscriber("/scan",LaserScan,LidarCallback,queue_size=10)
    vel_pub=rospy.Publisher("/cmd_vel",Twist,queue_size=10)
    rospy.spin()
 