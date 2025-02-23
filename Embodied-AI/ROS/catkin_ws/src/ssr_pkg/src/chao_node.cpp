#include<ros/ros.h>          // ROS的核心功能头文件
#include<std_msgs/String.h>  // 字符串消息类型的头文件
#include<qq_msgs/Carry.h>
int main(int argc, char  *argv[])
{   

    ros::init(argc,argv,"chao_node");// 初始化ROS节点，节点名为"chao_node"
    ros::NodeHandle nh;  // 创建节点句柄，用于管理节点资源
    ros::Publisher pub = nh.advertise<qq_msgs::Carry>("chao",10);

    ros::Rate loop_rate(10);// 设置循环频率为10Hz
    while(ros::ok())//使用ture时候运行节点不能响应外界信号
    {
        printf("keep\n");
        qq_msgs::Carry msg; // 创建字符串消息对象
        msg.grade="man";
        msg.star=100;
        msg.data = "do this all day man!!!"; // 设置消息内容
        pub.publish(msg);// 发布消息
        loop_rate.sleep(); // 控制循环频率
    }
    printf("hello world");
    return 0;
}
