
#include<ros/ros.h>
#include<move_base_msgs/MoveBaseAction.h>
#include<actionlib/client/simple_action_client.h>

//参数需要指定动作的类型
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

int main(int argc, char** argv){
    ros::init(argc, argv, "nav_client");
    MoveBaseClient ac("move_base", true);

//   它的作用是等待动作服务器启动，并返回一个布尔值。
// 如果服务器在指定的时间内（由 ros::Duration 设置）启动，函数返回 true。
// 如果超时未启动，函数返回 false。
    while(!ac.waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the move_base action server to come up");
    }

    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.pose.position.x = -3.0;
    goal.target_pose.pose.position.y = 1.0;

    goal.target_pose.pose.orientation.w = 1;

    ROS_INFO("Sending goal");
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("complete mission");
    else
        ROS_INFO("failed mission");

}