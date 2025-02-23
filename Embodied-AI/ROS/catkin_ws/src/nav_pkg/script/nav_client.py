#! /usr/bin/env python3
# coding: utf-8

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

if __name__ == '__main__':
    rospy.init_node('nav_client')
    ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    ac.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = -3.0
    goal.target_pose.pose.position.y = 1.0
    goal.target_pose.pose.orientation.w = 1.0

    
    ac.send_goal(goal)
    rospy.loginfo("send goal")
    ac.wait_for_result()
    
    if ac.get_state()==actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("goal reached")
    else:
        rospy.loginfo("goal failed")

