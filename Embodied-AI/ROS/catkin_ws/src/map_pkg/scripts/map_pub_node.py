#! /usr/bin/env python3
# coding: utf-8
import rospy
from nav_msgs.msg import OccupancyGrid

if __name__ == '__main__':
    rospy.init_node('map_pub_node')
    pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'

        msg.info.resolution = 1
        msg.info.width = 4
        msg.info.height = 2
        msg.info.origin.position.x = 0
        msg.info.origin.position.y = 0

        msg.data = [0]*4*2
        msg.data[0]=100
        msg.data[1]=90
        msg.data[2]=0
        msg.data[3]=-1

        pub.publish(msg)
        rate.sleep()
        