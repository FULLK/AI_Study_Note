#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
// geometry_msgs 是 ROS 中的一个标准消息包，它定义了几何相关的消息类型，如 Point、Vector3、Twist 等。
// sensor_msgs 包依赖于 geometry_msgs，因此当你在 find_package 中指定了 sensor_msgs 时，catkin 会自动加载 geometry_msgs 作为依赖项。
ros::Publisher vel_pub;
static int nCount=0;
void LidarCallback(const sensor_msgs::LaserScan msg)
{   
    //fmiddist 通常是指 前方中间距离（Front Middle Distance）的缩写。它表示激光雷达在正前方方向（通常是 180 度）测得的距离值
    float FMidDist=msg.ranges[180];
    ROS_INFO("前方测距 ranges[180] = %f 米", FMidDist);
    //避障部分
    if(nCount>0)
    {
        nCount--;
        return;  //继续保持之前的运动
    }
    geometry_msgs::Twist twist;
    if(FMidDist<1.5) //1.5米 一个grid是1米
    {
        twist.angular.z = 0.5;
        nCount=30;
    }
    else
    {
        twist.linear.x = 0.5;  
    }
    vel_pub.publish(twist);
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL,"");
    ros::init(argc,argv,"lidar_node");
    ros::NodeHandle n;
    ros::Subscriber lidar_sub = n.subscribe("/scan",10,&LidarCallback);
    //避障部分
    vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel",10);
    //得到scan话题里的雷达数据后选择行为
    ros::spin();
    return 0;
}